"""
情感相关/不相关双流模块（替代 DDM）

仅对图像解耦：情感相关流、情感不相关流。
文本 token 按位置 query 情感相关流，得到每个 token 专属的图像贡献（方式 A）。

设计来源：阶段目标文档「核心改进方案：情感相关/不相关双流（按位置 query 情感相关流）」
"""

import torch
import torch.nn as nn
from typing import Tuple


class SentimentStreamModule(nn.Module):
    """
    情感相关/不相关双流模块（只解耦图像，不解耦文本）。

    功能：
    1. 图像特征 → 两条投影 → 情感相关流 [B, L_image, stream_dim]、情感不相关流 [B, L_image, stream_dim]
    2. 按位置 query：Q=文本表示，K/V=情感相关流 → sent_relevant_aligned [B, L_text, stream_dim]
    3. 可选：正交损失在两条流（池化后）上施加

    输入：
        text_features: [B, L_text, embed_dim]
        image_features: [B, L_image, embed_dim]

    输出：
        sent_relevant_aligned: [B, L_text, stream_dim]  用于融合与情感监督
        sent_irrelevant: [B, L_image, stream_dim]      patch 级，用于可选正交损失
    """

    def __init__(
        self,
        embed_dim: int = 768,
        stream_dim: int = 384,
        num_heads: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.stream_dim = stream_dim

        # 图像 → 两条流
        self.image_sent_relevant_proj = nn.Linear(embed_dim, stream_dim)
        self.image_sent_irrelevant_proj = nn.Linear(embed_dim, stream_dim)

        # 按位置 query 情感相关流：Q 来自文本，K/V 来自情感相关流
        self.text_to_query = nn.Linear(embed_dim, stream_dim)
        self.sent_relevant_align = nn.MultiheadAttention(
            embed_dim=stream_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sent_relevant_aligned: [B, L_text, stream_dim]
            sent_irrelevant: [B, L_image, stream_dim]
        """
        B, L_text, _ = text_features.size()
        L_image = image_features.size(1)

        # 1. 图像双流
        sent_relevant = self.image_sent_relevant_proj(image_features)   # [B, L_image, stream_dim]
        sent_irrelevant = self.image_sent_irrelevant_proj(image_features)  # [B, L_image, stream_dim]

        # 2. 按位置 query 情感相关流：Q=text, K/V=sent_relevant
        q = self.text_to_query(text_features)  # [B, L_text, stream_dim]
        sent_relevant_aligned, _ = self.sent_relevant_align(
            query=q,
            key=sent_relevant,
            value=sent_relevant,
        )  # [B, L_text, stream_dim]

        return sent_relevant_aligned, sent_irrelevant
