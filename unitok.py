import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from vitamin import GeGluMlp, ViTaminDecoder
from quant import VectorQuantizerM
from vqvae import AttnProjection


class UniTok(nn.Module):
    """
    UniTok 模型类，用于图像编码、量化和解码，以及文本编码和图像-文本对比学习。

    参数:
        args: 模型配置参数，包含以下属性：
            num_query (int): 查询数量。
            model (str): 基础模型的名称，用于创建编码器和解码器。
            quant_proj (str): 量化投影的类型，可以是 'linear' 或 'attn'。
            vocab_width (int): 词汇表宽度，用于量化。
            vocab_size (int): 词汇表大小，用于量化。
            vq_beta (float): VQ 损失的 beta 参数。
            le (float): 熵损失的权重，如果大于0，则使用熵损失。
            e_temp (float): 熵温度，用于控制熵损失的平滑程度。
            num_codebooks (int): 码本数量，用于量化。
            text_width (int): 文本编码器的宽度。
            text_heads (int): 文本编码器的多头注意力头数。
            text_layers (int): 文本编码器的层数。
            text_vocab_size (int): 文本词汇表大小。
            text_context_length (int): 文本上下文长度。
            embed_dim (int): 嵌入维度，用于对比学习。
            img_size (int): 输入图像的大小。
            drop_path (float): DropPath 的概率。
            grad_ckpt (bool): 是否使用梯度检查点。
    """
    def __init__(self, args):
        super().__init__()
        
        # 查询数量
        self.num_query = args.num_query

        # 创建编码器，使用 timm 库
        self.encoder = timm.create_model(
            args.model,
            patch_size=1,
            fc_norm=False,
            drop_rate=0.0,
            num_classes=0,
            global_pool='',
            pos_embed='none',
            class_token=False,
            mlp_layer=GeGluMlp,
            reg_tokens=args.num_query,
            img_size=args.img_size,
            drop_path_rate=args.drop_path,
        )
        # 设置位置嵌入为不可训练的参数，形状为 (1, 1, embed_dim)
        self.encoder.pos_embed = nn.Parameter(torch.zeros(1, 1, self.encoder.embed_dim), requires_grad=False)

        # 根据量化投影类型，初始化量化投影层
        if args.quant_proj == 'linear':
            # 线性投影层
            self.quant_proj = nn.Linear(self.encoder.embed_dim, args.vocab_width)
        elif args.quant_proj == 'attn':
            # 注意力投影层
            self.quant_proj = AttnProjection(self.encoder.embed_dim, args.vocab_width, self.encoder.embed_dim // args.vocab_width)
        else:
            raise NotImplementedError

        # 初始化量化器，使用 VectorQuantizerM
        self.quantizer = VectorQuantizerM(
            vocab_size=args.vocab_size, # 词汇表大小
            vocab_width=args.vocab_width, # 词汇表宽度
            beta=args.vq_beta, # VQ 损失的 beta 参数
            use_entropy_loss=args.le > 0, # 是否使用熵损失
            entropy_temp=args.e_temp, # 熵温度
            num_codebooks=args.num_codebooks, # 码本数量
        )

        # 根据量化投影类型，初始化后量化投影层
        if args.quant_proj == 'linear':
            # 线性投影层
            self.post_quant_proj = nn.Linear(args.vocab_width, self.encoder.embed_dim)
        elif args.quant_proj == 'attn':
            # 注意力投影层
            self.post_quant_proj = AttnProjection(args.vocab_width, self.encoder.embed_dim, self.encoder.embed_dim // args.vocab_width)
        else:
            raise NotImplementedError

        # 创建解码器，使用 ViTaminDecoder
        self.decoder = ViTaminDecoder(
            args.model,
            num_query=args.num_query,
            img_size=args.img_size,
            drop_path=args.drop_path,
            grad_ckpt=args.grad_ckpt,
        )

        # 定义文本编码器配置
        text_cfg = {
            "width": args.text_width, # 文本编码器宽度
            "heads": args.text_heads, # 多头注意力头数
            "layers": args.text_layers, # 层数
            "vocab_size": args.text_vocab_size, # 文本词汇表大小
            "context_length": args.text_context_length, # 上下文长度
        }

        from open_clip.model import _build_text_tower
        # 构建文本编码器
        self.text_encoder = _build_text_tower(args.embed_dim, text_cfg)

        # 定义归一化层和投影层
        # 归一化层
        self.fc_norm = nn.LayerNorm(self.encoder.embed_dim, eps=1e-6)
        # 投影层
        self.projection = nn.Linear(self.encoder.embed_dim, args.embed_dim)
        # 对比学习中的 logit 缩放参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 设置文本编码器的上下文长度、词汇表大小
        self.context_length = self.text_encoder.context_length
        self.vocab_size = self.text_encoder.vocab_size
        # 上下文管理器，假设为无操作
        self.maybe_record_function = nullcontext

        # 文本梯度标志，假设不使用
        self.text_no_grad = False
        # 设置编码器的梯度检查点
        self.encoder.set_grad_checkpointing(args.grad_ckpt)
        # 设置文本编码器的梯度检查点
        self.text_encoder.set_grad_checkpointing(args.grad_ckpt)

    def forward(self, img, vae_bs, text=None, ret_usages=False):
        """
        前向传播函数。

        参数:
            img (torch.Tensor): 输入图像张量，形状为 (batch_size, channels, height, width)。
            vae_bs (int): VAE 的批量大小。
            text (torch.Tensor, optional): 输入文本张量，形状为 (batch_size, context_length)。
            ret_usages (bool): 是否返回码本使用情况，默认为 False。

        返回:
            dict: 包含以下键的字典：
                "img_rec" (torch.Tensor): 重构图像。
                "vq_loss" (torch.Tensor): VQ 损失。
                "entropy_loss" (torch.Tensor): 熵损失。
                "codebook_usages" (torch.Tensor): 码本使用情况。
                "clip_image_features" (torch.Tensor): CLIP 图像特征。
                "clip_text_features" (torch.Tensor): CLIP 文本特征。
                "logit_scale" (torch.Tensor): logit 缩放因子。
        """
        # 对图像进行编码
        img_tokens = self.encoder(img).float()
        # 禁用自动混合精度
        with torch.cuda.amp.autocast(enabled=False):
            # 应用量化投影
            img_tokens = torch.utils.checkpoint.checkpoint(self.quant_proj, img_tokens, use_reentrant=False)
            # 应用量化器
            img_tokens, vq_loss, entropy_loss, usages = self.quantizer(img_tokens)
            # 应用后量化投影
            img_tokens = torch.utils.checkpoint.checkpoint(self.post_quant_proj, img_tokens, use_reentrant=False)
        # 对量化后的图像进行解码
        img_rec = self.decoder(img_tokens[:vae_bs]).float()

        # 计算图像的 CLIP 视觉特征
        clip_visual = img_tokens.mean(dim=1)
        # 应用投影和归一化
        clip_visual = self.projection(self.fc_norm(clip_visual))
        # 归一化
        clip_visual = F.normalize(clip_visual, dim=-1)
        if text is not None:
            # 对文本进行编码
            clip_text = self.text_encoder(text)
            # 归一化
            clip_text = F.normalize(clip_text, dim=-1)
        else:
            clip_text = None

        output_dict = {
            "img_rec": img_rec, # 重构图像
            "vq_loss": vq_loss, # VQ 损失
            "entropy_loss": entropy_loss, # 熵损失
            "codebook_usages": usages, # 码本使用情况
            "clip_image_features": clip_visual, # CLIP 图像特征
            "clip_text_features": clip_text, # CLIP 文本特征
            "logit_scale": self.logit_scale.exp() # logit 缩放因子
        }

        # 返回输出字典
        return output_dict

    def encode_image(self, image, normalize: bool = False):
        """
        对输入图像进行编码，生成特征向量。

        参数:
            image (torch.Tensor): 输入图像张量。
            normalize (bool): 是否对输出特征进行归一化，默认为 False。

        返回:
            torch.Tensor: 编码后的图像特征向量。如果 `normalize` 为 True，则返回归一化后的特征向量。
        """
        # 使用编码器对图像进行编码
        img_tokens = self.encoder(image)
        # 应用量化投影
        img_tokens = self.quant_proj(img_tokens)
        # 将特征转换为索引
        img_indices = self.quantizer.f_to_idx(img_tokens)
        # 将索引转换回特征
        img_tokens = self.quantizer.idx_to_f(img_indices)
        # 应用后量化投影
        img_tokens = self.post_quant_proj(img_tokens)
        # 对特征进行平均池化，得到图像特征向量
        features = img_tokens.mean(dim=1)
        # 应用投影和归一化层
        features = self.projection(self.fc_norm(features))
        # 如果需要，则对特征进行归一化
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        """
        对输入文本进行编码，生成特征向量。

        参数:
            text (str 或 List[str]): 输入文本。
            normalize (bool): 是否对输出特征进行归一化，默认为 False。

        返回:
            torch.Tensor: 编码后的文本特征向量。如果 `normalize` 为 True，则返回归一化后的特征向量。
        """
        # 使用文本编码器对文本进行编码
        features = self.text_encoder(text)
        # 如果需要，则对特征进行归一化
        return F.normalize(features, dim=-1) if normalize else features

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
        return self.quantizer.f_to_idx(features)

    def idx_to_img(self, indices):
        """
        将量化索引转换回图像。

        参数:
            indices (torch.Tensor): 输入的量化索引。

        返回:
            torch.Tensor: 重构的图像张量，形状为 (batch_size, channels, height, width)。
        """
        # 将索引转换回特征
        features = self.quantizer.idx_to_f(indices)
        # 应用后量化投影
        features = self.post_quant_proj(features)
        # 使用解码器解码特征，并限制值在 [-1, 1] 范围内
        img = self.decoder(features).clamp_(-1, 1)
        # 返回重构的图像
        return img

    def img_to_reconstructed_img(self, image) -> torch.Tensor:
        """
        将输入图像编码并解码，生成重构图像。

        参数:
            image (torch.Tensor): 输入图像张量。

        返回:
            torch.Tensor: 重构的图像张量。
        """
        # 使用编码器对图像进行编码
        img_tokens = self.encoder(image)
        # 应用量化投影
        img_tokens = self.quant_proj(img_tokens)
        # 应用量化器
        img_tokens, _, _, _ = self.quantizer(img_tokens)
        # 应用后量化投影
        img_tokens = self.post_quant_proj(img_tokens)
        # 使用解码器解码特征，并限制值在 [-1, 1] 范围内
        img_rec = self.decoder(img_tokens).clamp_(-1, 1)
        # 返回重构的图像
        return img_rec

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True, unlock_text_proj=False):
        """
        锁定文本编码器的大部分层，只解锁指定数量的层。

        参数:
            unlocked_layers (int): 解锁的层数，默认为 0。
            freeze_layer_norm (bool): 是否冻结层归一化层，默认为 True。
            unlock_text_proj (bool): 是否解锁文本投影层，默认为 False。
        """
        # 调用文本编码器的锁定方法
        self.text.lock(unlocked_layers, freeze_layer_norm, unlock_text_proj)
        # 设置文本梯度标志为 True
        self.text_no_grad = True


if __name__ == '__main__':

    # 创建模型实例，使用 'vitamin_base' 作为基础模型
    model = timm.create_model(
        'vitamin_base',
        patch_size=1,
        fc_norm=True,
        drop_rate=0.0,
        num_classes=0,
        global_pool='',
        pos_embed='none',
        class_token=False,
        mlp_layer=GeGluMlp,
        reg_tokens=0,
        img_size=256,
        drop_path_rate=0.1,
    )
    # 设置位置嵌入为不可训练的参数，形状为 (1, 1, embed_dim)
    model.pos_embed = nn.Parameter(torch.zeros(1, 1, model.embed_dim), requires_grad=False)

    # 获取模型的当前状态字典
    model_dict = model.state_dict()
    # 加载预训练的模型参数
    ckpt_dict = torch.load('ViTamin-B/pytorch_model.bin')
    visual_dict = dict()
    for k, v in ckpt_dict.items():
        if k.startswith('visual.'):
            if 'head' in k or 'pos_embed' in k:
                continue
            new_k = k.replace('visual.trunk.', '')
            visual_dict[new_k] = v

    # 加载视觉部分的参数到模型中
    model.load_state_dict(visual_dict, strict=False)
    # 打印模型中缺失的键和多余的键
    print(set(model_dict.keys()) - set(visual_dict.keys()))
    print(set(visual_dict.keys() - set(model_dict.keys())))

