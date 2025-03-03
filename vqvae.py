import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.functional import scaled_dot_product_attention

from quant import VectorQuantizerM
from vitamin import ViTaminDecoder, GeGluMlp


class CausalAttention(nn.Module):
    """
    因果注意力机制模块，用于处理序列数据，确保每个位置只能关注其之前的位置。

    参数:
        in_dim (int): 输入的维度。
        out_dim (int): 输出的维度。
        num_heads (int): 多头注意力机制中的头数。
    """
    def __init__(self, in_dim, out_dim, num_heads):
        super().__init__()
        if in_dim > out_dim:
            # 确保 in_dim 可以被 num_heads 整除，并且结果等于 out_dim
            # assert in_dim // num_heads == out_dim
            # 每个注意力头的维度
            self.head_dim = in_dim // num_heads
            # 查询、键和值线性变换层，不使用偏置
            self.qkv = nn.Linear(in_dim, in_dim * 3, bias=False)
            # 查询偏置参数
            self.q_bias = nn.Parameter(torch.zeros(in_dim))
            # 值偏置参数
            self.v_bias = nn.Parameter(torch.zeros(in_dim))
            # 键偏置参数，初始化为零并注册为缓冲区
            self.register_buffer('zero_k_bias', torch.zeros(in_dim))
        else:
            # 确保 out_dim 可以被 num_heads 整除，并且结果等于 in_dim
            # assert out_dim // num_heads == in_dim
            # 每个注意力头的维度
            self.head_dim = out_dim // num_heads
            # 查询、键和值线性变换层，不使用偏置
            self.qkv = nn.Linear(in_dim, out_dim * 3, bias=False)
            # 查询偏置参数
            self.q_bias = nn.Parameter(torch.zeros(out_dim))
            # 值偏置参数
            self.v_bias = nn.Parameter(torch.zeros(out_dim))
            # 键偏置参数，初始化为零并注册为缓冲区
            self.register_buffer('zero_k_bias', torch.zeros(out_dim))

        # 输入维度
        self.in_dim = in_dim
        # 输出维度
        self.out_dim = out_dim
        # 多头数
        self.num_heads = num_heads
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        # 输出投影层
        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, in_dim)。

        返回:
            torch.Tensor: 注意力机制的输出，形状为 (batch_size, sequence_length, out_dim)。
        """
        # 获取批量大小、序列长度和输入维度
        B, N, C = x.shape
        # 应用线性变换，并添加偏置
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
        # 重塑张量形状为 (batch_size, sequence_length, 3, num_heads, head_dim)
        # 调整维度顺序为 (3, batch_size, num_heads, sequence_length, head_dim)
        # 分离查询、键和值
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)

        # 计算因果缩放点积注意力
        x = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=True)

        if self.in_dim > self.out_dim:
            # 如果输入维度大于输出维度，则对注意力输出进行平均池化
            x = torch.mean(x, dim=1)
            if self.in_dim // self.num_heads != self.out_dim:
                # 如果平均池化后的维度不等于输出维度，则应用自适应平均池化
                x = nn.functional.adaptive_avg_pool1d(x, self.out_dim)
        else:
            # 如果输入维度小于或等于输出维度，则调整维度顺序并重塑张量
            x = x.transpose(1, 2).reshape(B, N, -1)

        # 应用输出投影层
        x = self.proj(x)
        # 返回输出
        return x


class AttnProjection(nn.Module):
    """
    注意力投影模块，结合了归一化、因果注意力机制和前馈神经网络。

    参数:
        in_dim (int): 输入的维度。
        out_dim (int): 输出的维度。
        num_heads (int): 多头注意力机制中的头数。
        norm_layer (nn.Module): 归一化层类型，默认为 nn.LayerNorm。
        mlp_ratio (float): MLP（多层感知机）隐藏层维度与输入维度的比率，默认为2。
    """
    def __init__(self, in_dim, out_dim, num_heads, norm_layer=nn.LayerNorm, mlp_ratio=2):
        super().__init__()
        assert out_dim % in_dim == 0 or in_dim % out_dim == 0
        # 输入维度
        self.in_dim = in_dim
        # 输出维度
        self.out_dim = out_dim
        # 第一层归一化
        self.norm1 = norm_layer(in_dim)
        # 因果注意力机制
        self.attn = CausalAttention(in_dim, out_dim, num_heads)
        # 投影层
        self.proj = nn.Linear(in_dim, out_dim)
        # 第三层归一化
        self.norm3 = norm_layer(in_dim)
        
        # 第二层归一化
        self.norm2 = norm_layer(out_dim)
        # MLP隐藏层维度
        hidden_dim = int(out_dim * mlp_ratio)
        # MLP层
        self.mlp = GeGluMlp(
            in_features=out_dim,
            hidden_features=hidden_dim
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量。

        返回:
            torch.Tensor: 注意力投影模块的输出。
        """
        # 应用投影和注意力机制
        x = self.proj(self.norm3(x)) + self.attn(self.norm1(x))
        # 应用前馈神经网络
        x = x + self.mlp(self.norm2(x))
        # 返回输出
        return x


class VQVAE(nn.Module):
    """
    VQVAE（Vector Quantized Variational AutoEncoder）模型类，用于图像的编码、量化、解码。

    参数:
        args: 模型配置参数，包含以下属性：
            model (str): 基础模型的名称，用于创建编码器和解码器。
            quant_proj (str): 量化投影的类型，可以是 'linear' 或 'attn'。
            vocab_width (int): 词汇表宽度，用于量化。
            vocab_size (int): 词汇表大小，用于量化。
            vq_beta (float): VQ 损失的 beta 参数。
            le (float): 熵损失的权重，如果大于0，则使用熵损失。
            e_temp (float): 熵温度，用于控制熵损失的平滑程度。
            num_codebooks (int): 码本数量，用于量化。
            img_size (int): 输入图像的大小。
            drop_path (float): DropPath 的概率。
            grad_ckpt (bool): 是否使用梯度检查点。
    """
    def __init__(self, args):
        super().__init__()

        # 1. 构建编码器
        self.encoder = timm.create_model(
            args.model,
            patch_size=1,
            fc_norm=True,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            class_token=False,
            mlp_layer=GeGluMlp,
            img_size=args.img_size,
            drop_path_rate=args.drop_path,
        )
        # 设置编码器的梯度检查点
        self.encoder.set_grad_checkpointing(args.grad_ckpt)

        # 2. 构建量化前的卷积层
        if args.quant_proj == 'linear':
            # 线性投影层
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.vocab_width)
        elif args.quant_proj == 'attn':
            # 注意力投影层
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, args.num_codebooks)
        else:
            raise NotImplementedError

        # 3. 构建量化器
        self.quantize = VectorQuantizerM(
            vocab_size=args.vocab_size, # 词汇表大小
            vocab_width=args.vocab_width, # 词汇表宽度
            beta=args.vq_beta, # VQ 损失的 beta 参数
            use_entropy_loss=args.le > 0, # 是否使用熵损失
            entropy_temp=args.e_temp, # 熵温度
            num_codebooks=args.num_codebooks, # 码本数量
        )

        # 4. 构建量化后的卷积层
        if args.quant_proj == 'linear':
            # 线性投影层
            self.post_quant_proj = nn.Linear(args.vocab_width, self.encoder.embed_dim)
        elif args.quant_proj == 'attn':
            # 注意力投影层
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, args.num_codebooks)
        else:
            raise NotImplementedError

        # 5. 构建解码器
        self.decoder = ViTaminDecoder(
            args.model, # 基础模型名称 
            depths=(4, 2), # 解码器的深度，假设为 (4, 2)
            img_size=args.img_size, # 输入图像大小
            drop_path=args.drop_path, # DropPath 概率
            grad_ckpt=args.grad_ckpt # 是否使用梯度检查点
        )

        # 上下文管理器，假设为无操作
        self.maybe_record_function = nullcontext

    def forward(self, img):
        """
        前向传播函数。

        参数:
            img (torch.Tensor): 输入图像张量。

        返回:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 返回重构图像、VQ 损失、熵损失和码本使用情况。
        """
        # 使用编码器对图像进行编码，并转换为浮点类型
        features = self.encoder(img).float()
        # 禁用自动混合精度
        with torch.cuda.amp.autocast(enabled=False):
            # 应用量化投影
            features = self.quant_proj(features)
            # 应用量化器
            quant_out = self.quantize(features)
            # 获取量化后的特征、VQ 损失、熵损失和码本使用情况
            features, vq_loss, entropy_loss, usages = quant_out
            # 应用后量化投影
            features = self.post_quant_proj(features)
        # 使用解码器对量化后的特征进行解码，并转换为浮点类型
        rec_img = self.decoder(features).float()
        # 返回重构图像、VQ 损失、熵损失和码本使用情况
        return rec_img, vq_loss, entropy_loss, usages

    def img_to_idx(self, img):
        """
        将输入图像转换为量化索引。

        参数:
            img (torch.Tensor): 输入图像张量。

        返回:
            torch.Tensor: 量化后的图像索引。
        """
        # 使用编码器对图像进行编码，并转换为浮点类型
        features = self.encoder(img).float()
        # 应用量化投影
        features = self.quant_proj(features)
        # 将特征转换为量化索引
        return self.quantize.f_to_idx(features)

    def idx_to_img(self, indices):
        """
        将量化索引转换回图像。

        参数:
            indices (torch.Tensor): 输入的量化索引。

        返回:
            torch.Tensor: 重构的图像张量。
        """
        # 将索引转换回特征
        features = self.quantize.idx_to_f(indices)
        # 应用后量化投影
        features = self.post_quant_proj(features)
        # 使用解码器解码特征，并限制值在 [-1, 1] 范围内
        img = self.decoder(features).clamp_(-1, 1)
        # 返回重构的图像
        return img

    def img_to_reconstructed_img(self, img) -> torch.Tensor:
        """
        将输入图像编码并解码，生成重构图像。

        参数:
            img (torch.Tensor): 输入图像张量。

        返回:
            torch.Tensor: 重构的图像张量。
        """
        # 使用编码器对图像进行编码，并转换为浮点类型
        features = self.encoder(img).float()
        # 禁用自动混合精度
        with torch.cuda.amp.autocast(enabled=False):
            # 应用量化投影
            features = self.quant_proj(features)
            # 应用量化器
            quant_out = self.quantize(features)
            # 获取量化后的特征
            features, _, _, _ = quant_out
            # 应用后量化投影
            features = self.post_quant_proj(features)
        # 使用解码器解码特征，并限制值在 [-1, 1] 范围内
        rec_img = self.decoder(features).float().clamp_(-1, 1)
        # 返回重构的图像
        return rec_img

