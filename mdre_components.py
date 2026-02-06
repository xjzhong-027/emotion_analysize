"""
MDReID关键组件提取与适配

本文件包含从MDReID（多模态行人重识别）项目中提取的关键组件，
并适配到ICT（多模态情感分类）序列标注任务。

主要组件：
1. featureL2Norm: L2归一化函数
2. orthogonal_loss: 正交损失函数（适配序列标注）
3. CoRefine: 协同精炼函数
4. EnhancedTripletLossForSequence: 增强三元组损失（适配序列标注）

作者：AI Assistant
日期：2026-01-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# ==================== 辅助函数 ====================

def featureL2Norm(feature: torch.Tensor) -> torch.Tensor:
    """
    L2归一化函数
    
    作用：将特征向量归一化到单位长度，使得内积等于余弦相似度
    
    Args:
        feature: [B, N, C] - 输入特征
            B: batch size
            N: 特征数量（如3个模态：text_specific, image_specific, shared）
            C: 特征维度
    
    Returns:
        normalized: [B, N, C] - 归一化后的特征
    
    数学公式：
        normalized = feature / ||feature||_2
    
    示例：
        >>> feat = torch.randn(4, 3, 384)
        >>> feat_norm = featureL2Norm(feat)
        >>> print(feat_norm.shape)  # [4, 3, 384]
    """
    epsilon = 1e-6
    # 计算L2范数：对最后一个维度（特征维度）求和
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 2) + epsilon, 0.5)
    # 扩展维度以匹配原始特征形状
    norm = norm.unsqueeze(2).expand_as(feature)
    return torch.div(feature, norm)


def normalize(x: torch.Tensor, axis: int = -1) -> torch.Tensor:
    """
    归一化到单位长度（沿指定维度）
    
    Args:
        x: pytorch Tensor
        axis: 归一化的维度，默认-1（最后一个维度）
    
    Returns:
        x: 归一化后的Tensor，形状与输入相同
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算欧氏距离矩阵
    
    Args:
        x: [m, d] - 第一个特征矩阵
        y: [n, d] - 第二个特征矩阵
    
    Returns:
        dist: [m, n] - 距离矩阵
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # 数值稳定性
    return dist


# ==================== 正交损失（适配序列标注） ====================

def orthogonal_loss_for_sequence(
    text_specific: torch.Tensor,
    image_specific: torch.Tensor,
    shared_features: torch.Tensor
) -> torch.Tensor:
    """
    正交损失函数（适配序列标注任务）
    
    功能：强制模态特定特征与共享特征正交，确保特征解耦
    
    数学原理：
        对于特征矩阵 W = [w1, w2, w3]，其中：
        - w1 = text_specific（文本特定特征）
        - w2 = image_specific（图像特定特征）
        - w3 = shared_features（共享特征）
        
        计算Gram矩阵：G = W^T @ W
        理想情况下，G应该是单位矩阵I（特征正交）
        损失函数：L = ||G - I||^2
    
    Args:
        text_specific: [B, L, C] - 文本特定特征
            B: batch size
            L: 序列长度（如60）
            C: 特征维度（如384）
        image_specific: [B, L, C] - 图像特定特征（已池化到序列长度）
        shared_features: [B, L, C] - 共享特征
    
    Returns:
        loss: scalar - 正交损失值
    
    示例：
        >>> text_spec = torch.randn(4, 60, 384)
        >>> image_spec = torch.randn(4, 60, 384)
        >>> shared = torch.randn(4, 60, 384)
        >>> loss = orthogonal_loss_for_sequence(text_spec, image_spec, shared)
        >>> print(loss.item())  # scalar
    """
    B, L, C = text_specific.shape
    
    # 对每个序列位置计算正交损失
    loss = 0.0
    for i in range(L):
        # 提取第i个位置的特征 [B, 3, C]
        # 顺序：text_specific, image_specific, shared
        features = torch.stack([
            text_specific[:, i, :],      # [B, C]
            image_specific[:, i, :],      # [B, C]
            shared_features[:, i, :]      # [B, C]
        ], dim=1)  # [B, 3, C]
        
        # L2归一化
        features_norm = featureL2Norm(features)  # [B, 3, C]
        
        # 计算相似度矩阵（Gram矩阵）
        # WWT = features_norm @ features_norm^T
        WWT = torch.bmm(features_norm, features_norm.transpose(1, 2))  # [B, 3, 3]
        
        # 目标矩阵：单位矩阵（理想情况下特征正交）
        # [[1, 0, 0],
        #  [0, 1, 0],
        #  [0, 0, 1]]
        target_matrix = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(features.device)
        
        # MSE损失
        loss += F.mse_loss(WWT, target_matrix)
    
    # 平均所有位置的损失
    return loss / L


# ==================== 协同精炼函数 ====================

def CoRefine(output1: torch.Tensor, output2: torch.Tensor) -> torch.Tensor:
    """
    协同精炼函数：计算两组特征的相似度分布
    
    功能：通过计算两组特征之间的相似度矩阵，实现特征协同精炼
    
    Args:
        output1: [B, L, C] - 第一组特征
        output2: [B, L, C] - 第二组特征
    
    Returns:
        kl: [B, L, L] - 相似度矩阵（KL散度形式）
    
    数学原理：
        1. 对两组特征分别进行L2归一化
        2. 计算归一化后的特征之间的内积矩阵
        3. 得到相似度分布
    
    示例：
        >>> feat1 = torch.randn(4, 60, 384)
        >>> feat2 = torch.randn(4, 60, 384)
        >>> similarity = CoRefine(feat1, feat2)
        >>> print(similarity.shape)  # [4, 60, 60]
    """
    # L2归一化
    output1_prob = output1 / (output1.norm(dim=-1, keepdim=True) + 1e-12)
    output2_prob = output2 / (output2.norm(dim=-1, keepdim=True) + 1e-12)
    
    # 计算相似度矩阵：output2 @ output1^T
    kl = torch.bmm(output2_prob, output1_prob.transpose(1, 2))
    
    return kl


# ==================== 增强三元组损失（适配序列标注） ====================

def hard_example_mining_for_sequence(
    dist_mat: torch.Tensor,
    labels: torch.Tensor,
    return_inds: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    困难样本挖掘（适配序列标注）
    
    功能：为每个anchor找到最困难的正样本和负样本
    
    Args:
        dist_mat: [N, N] - 样本对之间的距离矩阵
        labels: [N] - 样本标签（序列标注中为每个token的标签）
        return_inds: 是否返回索引
    
    Returns:
        dist_ap: [N] - anchor到正样本的距离
        dist_an: [N] - anchor到负样本的距离
        p_inds: [N] - 正样本索引（如果return_inds=True）
        n_inds: [N] - 负样本索引（如果return_inds=True）
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    
    # 初始化输出
    dist_ap = torch.zeros(N, dtype=dist_mat.dtype, device=dist_mat.device)
    dist_an = torch.zeros(N, dtype=dist_mat.dtype, device=dist_mat.device)
    p_inds = torch.zeros(N, dtype=torch.long, device=dist_mat.device) if return_inds else None
    n_inds = torch.zeros(N, dtype=torch.long, device=dist_mat.device) if return_inds else None
    
    # 对每个anchor分别处理（因为不同anchor的正负样本数量可能不同）
    for i in range(N):
        # 找到正样本（相同标签）
        pos_mask = (labels == labels[i])  # [N]
        pos_mask[i] = False  # 排除自己
        
        # 找到负样本（不同标签）
        neg_mask = (labels != labels[i])  # [N]
        
        # 处理正样本
        if pos_mask.sum() > 0:
            pos_dists = dist_mat[i, pos_mask]  # [num_pos]
            dist_ap[i], relative_p_idx = torch.max(pos_dists, dim=0)
            if return_inds:
                pos_indices = torch.where(pos_mask)[0]
                p_inds[i] = pos_indices[relative_p_idx]
        else:
            # 如果没有正样本，使用一个很小的值
            dist_ap[i] = torch.tensor(0.0, device=dist_mat.device)
            if return_inds:
                p_inds[i] = i  # 使用自己作为占位符
        
        # 处理负样本
        if neg_mask.sum() > 0:
            neg_dists = dist_mat[i, neg_mask]  # [num_neg]
            dist_an[i], relative_n_idx = torch.min(neg_dists, dim=0)
            if return_inds:
                neg_indices = torch.where(neg_mask)[0]
                n_inds[i] = neg_indices[relative_n_idx]
        else:
            # 如果没有负样本，使用一个很大的值
            dist_an[i] = torch.tensor(1e6, device=dist_mat.device)
            if return_inds:
                n_inds[i] = i  # 使用自己作为占位符
    
    return dist_ap, dist_an, p_inds, n_inds


class EnhancedTripletLossForSequence(nn.Module):
    """
    增强三元组损失（适配序列标注任务）
    
    功能：通过对抗性优化，强制融合特征（fused）在判别能力上优于
         单独的模态特定特征和共享特征
    
    数学原理：
        s_p = d_fused_max / (d_fused_max + d_text_max + d_shared_max)
        s_n = d_fused_min / (d_fused_min + d_text_min + d_shared_min)
        
        目标：
        - s_p → 0（正样本距离主要由融合特征决定）
        - s_n → 1（负样本距离主要由融合特征决定）
        
        损失：L = |s_p - 0| + |s_n - 1|
    
    Args:
        hard_factor: 困难样本因子，默认0.0
    
    使用示例：
        >>> loss_fn = EnhancedTripletLossForSequence(hard_factor=0.0)
        >>> 
        >>> # 输入特征（已池化到全局特征）
        >>> fused_feat = torch.randn(32, 768)  # 融合特征
        >>> text_feat = torch.randn(32, 768)   # 文本特定特征
        >>> shared_feat = torch.randn(32, 768) # 共享特征
        >>> labels = torch.randint(0, 5, (32,)) # 标签
        >>> 
        >>> loss = loss_fn(fused_feat, text_feat, shared_feat, labels)
        >>> print(loss.item())
    """
    
    def __init__(self, hard_factor: float = 0.0):
        super().__init__()
        self.hard_factor = hard_factor
        self.l1_loss = nn.L1Loss()
    
    def forward(
        self,
        fused_features: torch.Tensor,
        text_specific_features: torch.Tensor,
        shared_features: torch.Tensor,
        labels: torch.Tensor,
        normalize_feature: bool = False
    ) -> torch.Tensor:
        """
        计算增强三元组损失
        
        Args:
            fused_features: [N, C] - 融合后的特征（全局池化后）
            text_specific_features: [N, C] - 文本特定特征（全局池化后）
            shared_features: [N, C] - 共享特征（全局池化后）
            labels: [N] - 样本标签
            normalize_feature: 是否归一化特征
        
        Returns:
            loss: scalar - 增强损失值
        """
        # 分离text和shared特征（不参与梯度更新）
        text_specific_features = text_specific_features.detach()
        shared_features = shared_features.detach()
        
        # 特征归一化（可选）
        if normalize_feature:
            fused_features = normalize(fused_features, axis=-1)
            text_specific_features = normalize(text_specific_features, axis=-1)
            shared_features = normalize(shared_features, axis=-1)
        
        # 计算距离矩阵
        dist_fused = euclidean_dist(fused_features, fused_features)  # [N, N]
        dist_text = euclidean_dist(text_specific_features, text_specific_features)  # [N, N]
        dist_shared = euclidean_dist(shared_features, shared_features)  # [N, N]
        
        # 困难样本挖掘
        dist_ap_fused, dist_an_fused, _, _ = hard_example_mining_for_sequence(
            dist_fused, labels, return_inds=True
        )
        dist_ap_text, dist_an_text, _, _ = hard_example_mining_for_sequence(
            dist_text, labels, return_inds=True
        )
        dist_ap_shared, dist_an_shared, _, _ = hard_example_mining_for_sequence(
            dist_shared, labels, return_inds=True
        )

        # 一次性导出用于可视化分析的特征样本（避免频繁写盘）
        if not hasattr(self, "_saved_enh_feats"):
            self._saved_enh_feats = False
        if (not self._saved_enh_feats) and fused_features.size(0) >= 4:
            try:
                k = min(256, fused_features.size(0))
                feats = {
                    "fused": fused_features[:k].detach().cpu().numpy(),
                    "text": text_specific_features[:k].detach().cpu().numpy(),
                    "shared": shared_features[:k].detach().cpu().numpy(),
                    "labels": labels[:k].detach().cpu().numpy(),
                    "dist_ap_fused": dist_ap_fused[:k].detach().cpu().numpy(),
                    "dist_ap_text": dist_ap_text[:k].detach().cpu().numpy(),
                    "dist_ap_shared": dist_ap_shared[:k].detach().cpu().numpy(),
                }
                np.save("enh_triplet_feats.npy", feats)
            except Exception:
                pass
            self._saved_enh_feats = True
        
        # 应用困难因子
        dist_ap_fused = dist_ap_fused * (1.0 + self.hard_factor)
        dist_an_fused = dist_an_fused * (1.0 - self.hard_factor)
        dist_ap_text = dist_ap_text * (1.0 + self.hard_factor)
        dist_an_text = dist_an_text * (1.0 - self.hard_factor)
        dist_ap_shared = dist_ap_shared * (1.0 + self.hard_factor)
        dist_an_shared = dist_an_shared * (1.0 - self.hard_factor)
        
        # 计算比例（对抗性目标）
        # s_p应该接近0（融合特征主导正样本距离）
        s_p = dist_ap_fused.max() / (
            dist_ap_fused.max() + 
            dist_ap_text.max().detach() + 
            dist_ap_shared.max().detach() + 
            1e-12
        )
        
        # s_n应该接近1（融合特征主导负样本距离）
        s_n = dist_an_fused.min() / (
            dist_an_fused.min() + 
            dist_an_text.min().detach() + 
            dist_an_shared.min().detach() + 
            1e-12
        )
        
        # L1损失
        target_p = torch.tensor([0.0]).to(s_p.device)
        target_n = torch.tensor([1.0]).to(s_n.device)
        
        loss = self.l1_loss(s_p.unsqueeze(0), target_p) + \
               self.l1_loss(s_n.unsqueeze(0), target_n)
        
        return loss


# ==================== 单元测试（可选） ====================

if __name__ == "__main__":
    """
    单元测试：验证各个组件的正确性
    """
    print("=" * 50)
    print("MDReID组件单元测试")
    print("=" * 50)
    
    # 测试参数
    B, L, C = 4, 60, 384
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 测试featureL2Norm
    print("\n1. 测试 featureL2Norm...")
    feat = torch.randn(B, 3, C).to(device)
    feat_norm = featureL2Norm(feat)
    assert feat_norm.shape == (B, 3, C), f"形状错误: {feat_norm.shape}"
    # 验证归一化：每个向量的L2范数应该接近1
    norms = torch.norm(feat_norm, dim=2)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "归一化失败"
    print("✓ featureL2Norm 测试通过")
    
    # 2. 测试orthogonal_loss_for_sequence
    print("\n2. 测试 orthogonal_loss_for_sequence...")
    text_spec = torch.randn(B, L, C).to(device)
    image_spec = torch.randn(B, L, C).to(device)
    shared = torch.randn(B, L, C).to(device)
    loss = orthogonal_loss_for_sequence(text_spec, image_spec, shared)
    assert loss.dim() == 0, f"损失应该是标量，但得到形状: {loss.shape}"
    assert loss.item() >= 0, "损失应该非负"
    print(f"✓ orthogonal_loss_for_sequence 测试通过，损失值: {loss.item():.4f}")
    
    # 3. 测试CoRefine
    print("\n3. 测试 CoRefine...")
    output1 = torch.randn(B, L, C).to(device)
    output2 = torch.randn(B, L, C).to(device)
    similarity = CoRefine(output1, output2)
    assert similarity.shape == (B, L, L), f"形状错误: {similarity.shape}"
    print("✓ CoRefine 测试通过")
    
    # 4. 测试EnhancedTripletLossForSequence
    print("\n4. 测试 EnhancedTripletLossForSequence...")
    N = 32  # 样本数
    fused_feat = torch.randn(N, C).to(device)
    text_feat = torch.randn(N, C).to(device)
    shared_feat = torch.randn(N, C).to(device)
    labels = torch.randint(0, 5, (N,)).to(device)
    
    loss_fn = EnhancedTripletLossForSequence(hard_factor=0.0).to(device)
    loss = loss_fn(fused_feat, text_feat, shared_feat, labels, normalize_feature=True)
    assert loss.dim() == 0, f"损失应该是标量，但得到形状: {loss.shape}"
    assert loss.item() >= 0, "损失应该非负"
    print(f"✓ EnhancedTripletLossForSequence 测试通过，损失值: {loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("所有测试通过！✓")
    print("=" * 50)
