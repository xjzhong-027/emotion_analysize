"""
模态内情感原型模块（Sentiment Per Modality，做法2：共用一套原型）

在 DDM 三支（text_specific、image_specific_aligned、shared）上共用一套情感原型，
对每支序列按位置算与 NEG/NEU/POS 原型的距离，经线性层得到 3 类 logits。
用于：模态内部的方面情感特征层级（可解释）+ 辅助损失 + 可选参与预测（方式 A）。

设计来源：阶段目标文档「核心改进方案：模态内部的方面情感特征层级」
"""

import torch
import torch.nn as nn


class SentModModule(nn.Module):
    """
    模态内情感原型模块（做法2：一套原型，对三支分别前向）。

    参数：
        branch_dim (int): 单支特征维度，与 DDM 的 specific_dim/shared_dim 一致（如 512）
        num_classes (int): 情感类别数，默认 3（NEG/NEU/POS）
        num_prototypes (int): 每类原型数量，默认 5

    输入：
        x: [B, L, branch_dim]，单支序列（text_specific / image_specific_aligned / shared 之一）

    输出：
        logits: [B, L, num_classes]
    """

    def __init__(
        self,
        branch_dim: int,
        num_classes: int = 3,
        num_prototypes: int = 5,
    ):
        super().__init__()
        self.branch_dim = branch_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # 共用一套：每类 num_prototypes 个原型
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, branch_dim) * 0.02)
        self.classifier = nn.Linear(num_classes * num_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, branch_dim]
        return: [B, L, num_classes]
        """
        B, L, C = x.size()
        x_flat = x.reshape(B * L, C)  # [B*L, branch_dim]
        device = x.device
        distances = []
        for c in range(self.num_classes):
            d = torch.cdist(x_flat, self.prototypes[c], p=2)  # [B*L, num_prototypes]
            distances.append(d)
        dist_feat = torch.cat(distances, dim=-1)  # [B*L, num_classes*num_prototypes]
        logits_flat = self.classifier(dist_feat)  # [B*L, num_classes]
        logits = logits_flat.reshape(B, L, self.num_classes)
        return logits
