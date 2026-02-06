"""
分类概率融合机制 (Classification-based Fusion Mechanism, CFM)

本模块在 DDM 解耦后的三路特征（text_specific / image_specific / shared_features）上，
基于特征拼接的 MLP 计算动态融合权重，加权融合后投影到 hidden_size，供后续序列标注使用。
可选：为三路特征分别做分类得到 branch_logits，用于分支一致性损失。

设计来源：阶段目标文档 2.4
作者：AI Assistant
日期：2026-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CFM(nn.Module):
    """
    分类概率融合机制 (Classification-based Fusion Mechanism)

    功能：
    1. 将 image_specific 池化到文本序列长度
    2. 根据三路特征拼接计算动态权重（softmax 归一化）
    3. 加权融合三路特征后投影到 output_dim
    4. 可选：为三路特征分别做 token 级分类，用于分支一致性损失

    参数：
        input_dim (int): 每路特征维度，应与 DDM 的 specific_dim/shared_dim 一致（如 384 或 512）
        output_dim (int): 融合后特征维度，应与 backbone hidden_size 一致（如 768 或 1024）
        num_features (int): 特征路数，默认 3（text / image / shared）
        num_classes (int): 序列标注类别数，默认 5（O, I, B-NEG, B-NEU, B-POS）
        seq_len (int): 文本序列长度，用于 image 池化，默认 60

    输入：
        text_specific: [B, L, input_dim]
        image_specific: [B, L_image, input_dim]（如 [B, 197, input_dim]）
        shared_features: [B, L, input_dim]

    输出：
        fused_features: [B, L, output_dim]
        fusion_weights: [B, L, 3]
        branch_logits: tuple of (text_logits, image_logits, shared_logits)，各 [B, L, num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_features: int = 3,
        num_classes: int = 5,
        seq_len: int = 60,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features = num_features
        self.num_classes = num_classes
        self.seq_len = seq_len

        # 图像特征池化到文本长度
        self.image_pool = nn.AdaptiveAvgPool1d(seq_len)

        # 动态权重 MLP：拼接三路 -> hidden -> 3 维 -> softmax
        self.weight_mlp = nn.Sequential(
            nn.Linear(input_dim * num_features, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_features),
        )

        # 加权融合后投影到 output_dim（加权和为 [B, L, input_dim]，再投影）
        self.fusion_proj = nn.Linear(input_dim, output_dim)

        # 三路分支分类器（用于分支一致性损失）
        self.text_clf = nn.Linear(input_dim, num_classes)
        self.image_clf = nn.Linear(input_dim, num_classes)
        self.shared_clf = nn.Linear(input_dim, num_classes)

    def forward(
        self,
        text_specific: torch.Tensor,
        image_specific: torch.Tensor,
        shared_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            text_specific: [B, L, C]
            image_specific: [B, L_image, C]
            shared_features: [B, L, C]

        Returns:
            fused_features: [B, L, output_dim]
            fusion_weights: [B, L, 3]
            branch_logits: (text_logits, image_logits, shared_logits)，各 [B, L, num_classes]
        """
        B, L, C = text_specific.size()
        # 1. 图像特征池化到 L
        # [B, L_image, C] -> [B, C, L_image]
        img = image_specific.permute(0, 2, 1)
        img = self.image_pool(img)  # [B, C, seq_len]
        image_aligned = img.permute(0, 2, 1)  # [B, L, C]

        # 2. 拼接三路 [B, L, 3*C]
        concat = torch.cat([text_specific, image_aligned, shared_features], dim=-1)

        # 3. 动态权重 [B, L, 3]，softmax 保证和为 1
        w = self.weight_mlp(concat)
        fusion_weights = F.softmax(w, dim=-1)

        # 4. 加权融合：fused = w_t*text + w_i*image + w_s*shared -> [B, L, C]
        w_t = fusion_weights[:, :, 0:1]
        w_i = fusion_weights[:, :, 1:2]
        w_s = fusion_weights[:, :, 2:3]
        fused = w_t * text_specific + w_i * image_aligned + w_s * shared_features

        # 5. 投影到 output_dim
        fused_features = self.fusion_proj(fused)  # [B, L, output_dim]

        # 6. 三路分支 logits（用于一致性损失）
        text_logits = self.text_clf(text_specific)
        image_logits = self.image_clf(image_aligned)
        shared_logits = self.shared_clf(shared_features)

        return fused_features, fusion_weights, (text_logits, image_logits, shared_logits)


# ==================== 单元测试 ====================

if __name__ == "__main__":
    B, L, L_img, C = 4, 60, 197, 512
    out_dim = 1024
    num_classes = 5
    cfm = CFM(input_dim=C, output_dim=out_dim, num_features=3, num_classes=num_classes, seq_len=L)
    t = torch.randn(B, L, C)
    i = torch.randn(B, L_img, C)
    s = torch.randn(B, L, C)
    fused, weights, (tl, il, sl) = cfm(t, i, s)
    assert fused.shape == (B, L, out_dim)
    assert weights.shape == (B, L, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, L))
    assert tl.shape == (B, L, num_classes)
    assert il.shape == (B, L, num_classes)
    assert sl.shape == (B, L, num_classes)
    print("CFM 单元测试通过")
