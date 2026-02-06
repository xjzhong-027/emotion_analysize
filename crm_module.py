"""
原型向量分类机制 (Classification with Prototype Vector Mechanism, CRM)

在融合特征上维护 NEG/NEU/POS 三类可学习原型，将方面位置的融合特征与原型算距离后
经线性层得到 3 类情感 logits，作为辅助损失或参与预测。

设计来源：阶段目标文档「核心改进方案：CRM」、开题报告 4.4 节
"""

import torch
import torch.nn as nn


class CRM(nn.Module):
    """
    原型向量分类机制：按情感类别维护原型，特征与原型距离经线性层得到 3 类 logits。

    参数：
        hidden_dim (int): 融合特征维度，与 backbone hidden_size 一致（如 1024）
        num_classes (int): 情感类别数，默认 3（NEG/NEU/POS）
        num_prototypes (int): 每类原型数量，默认 5

    输入：
        x: [N, hidden_dim]，仅方面位置的融合特征（N 为方面 token 总数）

    输出：
        logits: [N, num_classes]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 3,
        num_prototypes: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes

        # 每类 num_prototypes 个原型向量
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, hidden_dim) * 0.02)

        # 距离特征 -> 3 类 logits
        self.classifier = nn.Linear(num_classes * num_prototypes, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, hidden_dim]
        return: [N, num_classes]
        """
        N = x.size(0)
        device = x.device
        distances = []
        for c in range(self.num_classes):
            # prototypes[c]: [num_prototypes, hidden_dim]
            # cdist(x, prototypes[c]) -> [N, num_prototypes]
            d = torch.cdist(x, self.prototypes[c], p=2)
            distances.append(d)
        # [N, num_classes * num_prototypes]
        dist_feat = torch.cat(distances, dim=-1)
        logits = self.classifier(dist_feat)
        return logits
