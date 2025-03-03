import os
import torch
import argparse
from PIL import Image
from torchvision.transforms import transforms, InterpolationMode

from unitok import UniTok
from utils.config import Args
from utils.data import normalize_01_into_pm1


def save_img(img: torch.Tensor, path):
    """
    将处理后的图像张量保存为图像文件。

    参数:
        img (torch.Tensor): 输入的图像张量，形状为 (batch_size, channels, height, width)。
        path (str): 保存图像的路径。
    """
    # 将图像张量从[-1, 1]范围缩放到[0, 255]，然后四舍五入到最接近的整数
    # 将NaN值替换为128，负无穷替换为0，正无穷替换为255，并限制值在[0, 255]范围内
    img = img.add(1).mul_(0.5 * 255).round().nan_to_num_(128, 0, 255).clamp_(0, 255)
    # 将张量转换为uint8类型，并调整维度顺序为 (batch_size, height, width, channels)
    img = img.to(dtype=torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    # 从numpy数组创建PIL图像对象，假设批量大小为1
    img = Image.fromarray(img[0])
    # 保存图像到指定路径
    img.save(path)


def main(args):
    """
    主函数，用于加载模型、预处理图像、进行重建并保存结果。

    参数:
        args: 命令行参数，包含以下属性：
            ckpt_path (str): 模型检查点的路径。
            src_img (str): 源图像的路径。
            rec_img (str): 保存重建图像的路径。
    """
    # 加载模型检查点
    ckpt_path = args.ckpt_path
    # 从指定路径加载检查点，并将张量映射到CPU
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # 创建一个Args实例，用于存储配置参数
    unitok_cfg = Args()
    unitok_cfg.load_state_dict(ckpt['args'])
    # 初始化UniTok模型，传入配置参数
    unitok = UniTok(unitok_cfg)
    unitok.load_state_dict(ckpt['trainer']['unitok'])
    unitok.to('cuda')
    unitok.eval()

    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(int(unitok_cfg.img_size * unitok_cfg.resize_ratio)), # 调整图像大小，缩放因子为resize_ratio
        transforms.CenterCrop(unitok_cfg.img_size), # 中心裁剪图像到img_size大小
        transforms.ToTensor(), # 将PIL图像或numpy数组转换为张量，并归一化到[0,1]
        normalize_01_into_pm1, # 将图像归一化到[-1, 1]范围（假设normalize_01_into_pm1是一个预定义的归一化函数）
    ])
    # 打开源图像并转换为RGB
    img = Image.open(args.src_img).convert("RGB")
    # 应用预处理步骤，并将图像移动到GPU
    img = preprocess(img).unsqueeze(0).to('cuda')

    # 无梯度地进行推理
    with torch.no_grad():
        code_idx = unitok.img_to_idx(img)
        rec_img = unitok.idx_to_img(code_idx)

    # 将原始图像和重建图像在通道维度上拼接
    final_img = torch.cat((img, rec_img), dim=3)
    # 保存拼接后的图像到指定路径
    save_img(final_img, args.rec_img)

    print('The image is saved to {}. The left one is the original image after resizing and cropping. The right one is the reconstructed image.'.format(args.rec_img))


if __name__ == '__main__':

    # 设置命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--src_img', type=str, default='')
    parser.add_argument('--rec_img', type=str, default='')
    args = parser.parse_args()
    
    main(args)

