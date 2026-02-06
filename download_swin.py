#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 Swin Transformer 模型到本地目录
用于服务器环境，避免训练时访问 HuggingFace
"""

import os
from transformers import SwinForImageClassification, SwinConfig, AutoImageProcessor

def download_swin_model(model_name="microsoft/swin-base-patch4-window7-224", 
                        save_dir="/root/data/weights/swin"):
    """
    下载 Swin 模型到指定目录
    
    Args:
        model_name: HuggingFace 模型名称，默认使用 swin-base
        save_dir: 保存目录，默认 /root/data/weights/swin
    """
    print(f"开始下载 Swin 模型: {model_name}")
    print(f"保存目录: {save_dir}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 下载模型配置
        print("正在下载模型配置...")
        config = SwinConfig.from_pretrained(model_name)
        config.save_pretrained(save_dir)
        print("✅ 配置下载完成")
        
        # 下载模型权重
        print("正在下载模型权重（这可能需要几分钟）...")
        model = SwinForImageClassification.from_pretrained(model_name)
        model.save_pretrained(save_dir)
        print("✅ 模型权重下载完成")
        
        # 下载图像处理器
        print("正在下载图像处理器...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        processor.save_pretrained(save_dir)
        print("✅ 图像处理器下载完成")
        
        # 验证文件
        print("\n验证下载的文件...")
        required_files = [
            "config.json",
            "model.safetensors",  # 或 model.bin（旧版本）
            "preprocessor_config.json"
        ]
        
        for file in required_files:
            file_path = os.path.join(save_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  ✅ {file} ({size:.2f} MB)")
            else:
                # 检查是否有 model.bin（旧格式）
                if file == "model.safetensors":
                    bin_path = os.path.join(save_dir, "model.bin")
                    if os.path.exists(bin_path):
                        size = os.path.getsize(bin_path) / (1024 * 1024)
                        print(f"  ✅ model.bin ({size:.2f} MB) [旧格式]")
                    else:
                        print(f"  ⚠️  {file} 未找到")
                else:
                    print(f"  ⚠️  {file} 未找到")
        
        print(f"\n✅ Swin 模型下载完成！")
        print(f"模型已保存到: {save_dir}")
        print("\n使用方法:")
        print(f"  在 main.py 中设置: image_model_name='swin'")
        print(f"  代码会自动从 {save_dir} 加载模型")
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("1. 网络连接问题（服务器无法访问 HuggingFace）")
        print("2. 模型名称不正确")
        print("3. 磁盘空间不足")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载 Swin Transformer 模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/swin-base-patch4-window7-224",
        help="HuggingFace 模型名称（默认: microsoft/swin-base-patch4-window7-224）"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/root/data/weights/swin",
        help="保存目录（默认: /root/data/weights/swin）"
    )
    
    args = parser.parse_args()
    
    download_swin_model(model_name=args.model_name, save_dir=args.save_dir)