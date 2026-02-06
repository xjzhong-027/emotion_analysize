"""
双级解耦模块 (Dual-level Disentanglement Module, DDM)

本模块实现MDReID中的特征解耦机制，将文本和图像特征分离为：
- 模态特定特征（Modality-Specific Features）：各模态独有的信息
- 共享特征（Modality-Shared Features）：跨模态共同的语义信息

设计来源：MDReID的对抗解耦机制，适配到ICT序列标注任务

作者：AI Assistant
日期：2026-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DDM(nn.Module):
    """
    双级解耦模块 (Dual-level Disentanglement Module)
    
    功能：
    1. 将文本特征分离为文本特定特征和共享特征
    2. 将图像特征分离为图像特定特征和共享特征
    3. 通过跨模态注意力对齐共享特征到文本序列长度
    4. 可选：image_specific 按 token 对齐（query=text_specific, kv=image_specific），便于下游情感监督
    
    参数：
        embed_dim (int): 输入特征维度，默认768
        specific_dim (int): 模态特定特征维度，默认384
        shared_dim (int): 共享特征维度，默认384
        use_image_token_align (bool): 是否对 image_specific 做 token 级 cross-attention 对齐，默认 True（消融时可关）
    
    输入：
        text_features: [B, L_text, embed_dim] - 文本序列特征
        image_features: [B, L_image, embed_dim] - 图像patch特征
    
    输出：
        text_specific: [B, L_text, specific_dim]
        image_specific: [B, L_image, specific_dim]
        shared_features: [B, L_text, shared_dim]
        image_specific_aligned: [B, L_text, specific_dim]（token 级，用于 CFM/融合；关闭对齐时为池化结果）
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        specific_dim: int = 384,
        shared_dim: int = 384,
        use_image_token_align: bool = True,
    ):
        """
        初始化DDM模块
        
        Args:
            embed_dim: 基础特征维度，默认768
            specific_dim: 模态特定特征维度，默认384
            shared_dim: 共享特征维度，默认384
            use_image_token_align: 是否对 image_specific 做 token 级对齐（cross-attention），默认 True
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.specific_dim = specific_dim
        self.shared_dim = shared_dim
        self.use_image_token_align = use_image_token_align
        
        # 文本模态的特征分离投影
        self.text_specific_proj = nn.Linear(embed_dim, specific_dim)
        self.text_shared_proj = nn.Linear(embed_dim, shared_dim)
        
        # 图像模态的特征分离投影
        self.image_specific_proj = nn.Linear(embed_dim, specific_dim)
        self.image_shared_proj = nn.Linear(embed_dim, shared_dim)
        
        # 跨模态对齐层（将图像共享特征对齐到文本序列长度）
        self.cross_modal_align = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=8,
            batch_first=True
        )
        
        # 图像特定特征按 token 对齐：query=text_specific, key/value=image_specific -> [B, L_text, specific_dim]
        if use_image_token_align:
            self.image_to_token_align = nn.MultiheadAttention(
                embed_dim=specific_dim,
                num_heads=8,
                batch_first=True
            )
    
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_specific: [B, L_text, specific_dim]
            image_specific: [B, L_image, specific_dim]
            shared_features: [B, L_text, shared_dim]
            image_specific_aligned: [B, L_text, specific_dim]
        """
        # 1. 提取模态特定特征
        text_specific = self.text_specific_proj(text_features)  # [B, L_text, specific_dim]
        image_specific = self.image_specific_proj(image_features)  # [B, L_image, specific_dim]
        
        # 2. 提取共享特征
        text_shared = self.text_shared_proj(text_features)  # [B, L_text, shared_dim]
        image_shared = self.image_shared_proj(image_features)  # [B, L_image, shared_dim]
        
        # 3. 跨模态对齐（以文本共享特征为query，图像共享特征为key和value）
        shared_features, _ = self.cross_modal_align(
            query=text_shared,
            key=image_shared,
            value=image_shared
        )  # [B, L_text, shared_dim]
        
        # 4. image_specific 的 token 级对齐：用于 CFM/融合（正交损失仍用 patch 级 image_specific 在 model 内池化）
        if self.use_image_token_align:
            image_specific_aligned, _ = self.image_to_token_align(
                query=text_specific,   # [B, L_text, specific_dim]
                key=image_specific,    # [B, L_image, specific_dim]
                value=image_specific,
            )  # [B, L_text, specific_dim]
        else:
            # 消融：与原先行为一致，池化到文本长度
            L_text = text_specific.size(1)
            img_sp = image_specific.permute(0, 2, 1)  # [B, specific_dim, L_image]
            img_sp_aligned = F.adaptive_avg_pool1d(img_sp, L_text)
            image_specific_aligned = img_sp_aligned.permute(0, 2, 1)  # [B, L_text, specific_dim]
        
        return text_specific, image_specific, shared_features, image_specific_aligned


# ==================== 单元测试 ====================

if __name__ == "__main__":
    """
    单元测试：验证DDM模块的正确性
    """
    print("=" * 50)
    print("DDM模块单元测试")
    print("=" * 50)
    
    # 测试参数
    B, L_text, L_image, embed_dim = 4, 60, 197, 768
    specific_dim, shared_dim = 384, 384
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 创建DDM模块
    print("\n1. 创建DDM模块...")
    ddm = DDM(embed_dim=embed_dim, specific_dim=specific_dim, shared_dim=shared_dim).to(device)
    print("✓ DDM模块创建成功")
    
    # 2. 创建测试输入
    print("\n2. 创建测试输入...")
    text_features = torch.randn(B, L_text, embed_dim).to(device)
    image_features = torch.randn(B, L_image, embed_dim).to(device)
    print(f"✓ 输入创建成功: text={text_features.shape}, image={image_features.shape}")
    
    # 3. 前向传播（返回 4 个量）
    print("\n3. 执行前向传播...")
    text_specific, image_specific, shared_features, image_specific_aligned = ddm(text_features, image_features)
    print(f"✓ 前向传播成功")
    
    # 4. 验证输出维度
    print("\n4. 验证输出维度...")
    assert text_specific.shape == (B, L_text, specific_dim), \
        f"text_specific维度错误: 期望{(B, L_text, specific_dim)}, 得到{text_specific.shape}"
    assert image_specific.shape == (B, L_image, specific_dim), \
        f"image_specific维度错误: 期望{(B, L_image, specific_dim)}, 得到{image_specific.shape}"
    assert shared_features.shape == (B, L_text, shared_dim), \
        f"shared_features维度错误: 期望{(B, L_text, shared_dim)}, 得到{shared_features.shape}"
    assert image_specific_aligned.shape == (B, L_text, specific_dim), \
        f"image_specific_aligned维度错误: 期望{(B, L_text, specific_dim)}, 得到{image_specific_aligned.shape}"
    print(f"✓ text_specific: {text_specific.shape}")
    print(f"✓ image_specific: {image_specific.shape}")
    print(f"✓ shared_features: {shared_features.shape}")
    print(f"✓ image_specific_aligned: {image_specific_aligned.shape}")
    
    # 5. 验证梯度回传
    print("\n5. 验证梯度回传...")
    loss = text_specific.mean() + image_specific.mean() + shared_features.mean() + image_specific_aligned.mean()
    loss.backward()
    has_grad = any(p.grad is not None for p in ddm.parameters() if p.requires_grad)
    assert has_grad, "梯度回传失败"
    print("✓ 梯度回传正常")
    
    # 6. 测试 use_image_token_align=False（消融）
    print("\n6. 测试 use_image_token_align=False...")
    ddm_no_align = DDM(embed_dim=embed_dim, specific_dim=specific_dim, shared_dim=shared_dim, use_image_token_align=False).to(device)
    _, _, _, aligned_pooled = ddm_no_align(text_features, image_features)
    assert aligned_pooled.shape == (B, L_text, specific_dim)
    print("✓ 消融模式（池化）测试通过")
    
    # 7. 测试不同batch size
    print("\n7. 测试不同batch size...")
    for test_b in [1, 8, 16]:
        test_text = torch.randn(test_b, L_text, embed_dim).to(device)
        test_image = torch.randn(test_b, L_image, embed_dim).to(device)
        ts, is_, sf, al = ddm(test_text, test_image)
        assert ts.shape[0] == test_b and al.shape[0] == test_b, f"batch size {test_b} 测试失败"
    print("✓ 不同batch size测试通过")
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)
