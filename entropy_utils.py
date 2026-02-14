"""
熵计算工具模块
用于计算特征熵和注意力熵，支持熵增约束
"""

import torch
import torch.nn.functional as F


def compute_entropy(features, dim=-1, eps=1e-10):
    """
    计算特征的信息熵

    参数：
        features: torch.Tensor, 特征张量，任意形状
        dim: int, 计算熵的维度，默认为最后一维
        eps: float, 数值稳定性的小常数

    返回：
        entropy: torch.Tensor, 熵值，形状为去除dim维度后的形状

    示例：
        features: [batch_size, seq_len, hidden_dim]
        entropy: [batch_size, seq_len]  # 每个token的特征熵
    """
    # 将特征转换为概率分布（使用softmax）
    probs = F.softmax(features, dim=dim)

    # 计算熵: H = -sum(p * log(p))
    log_probs = torch.log(probs + eps)
    entropy = -torch.sum(probs * log_probs, dim=dim)

    return entropy


def compute_attention_entropy(attention_weights, mask=None, eps=1e-10):
    """
    计算注意力权重的信息熵

    参数：
        attention_weights: torch.Tensor, 注意力权重
                          形状: [batch_size, num_heads, seq_len, seq_len]
        mask: torch.Tensor, 可选的掩码，形状: [batch_size, seq_len]
        eps: float, 数值稳定性的小常数

    返回：
        entropy: torch.Tensor, 注意力熵
                形状: [batch_size, num_heads, seq_len]  # 每个query位置的注意力熵

    说明：
        注意力熵衡量注意力分布的集中程度：
        - 熵值高：注意力分散，关注多个位置
        - 熵值低：注意力集中，关注少数位置
    """
    # attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]

    if mask is not None:
        # 扩展mask以匹配注意力维度
        # mask: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        mask_expanded = mask.unsqueeze(1).unsqueeze(2)

        # 将padding位置的注意力权重设为0
        attention_weights = attention_weights * mask_expanded

        # 重新归一化（确保每行和为1）
        attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + eps)

    # 计算熵: H = -sum(p * log(p))
    log_attention = torch.log(attention_weights + eps)
    entropy = -torch.sum(attention_weights * log_attention, dim=-1)

    # entropy: [batch_size, num_heads, seq_len_q]
    return entropy


def compute_cross_attention_entropy(attention_weights, token_type_ids, attention_mask, eps=1e-10):
    """
    计算交叉注意力（文本-表格）的信息熵

    参数：
        attention_weights: torch.Tensor, 注意力权重
                          形状: [batch_size, num_heads, seq_len, seq_len]
        token_type_ids: torch.Tensor, token类型标识
                       形状: [batch_size, seq_len]
                       0表示文本，1表示表格
        attention_mask: torch.Tensor, 注意力掩码
                       形状: [batch_size, seq_len]
        eps: float, 数值稳定性的小常数

    返回：
        dict: 包含以下键值对
            - 'text_to_table_entropy': 文本对表格的注意力熵
            - 'table_to_text_entropy': 表格对文本的注意力熵
            - 'mean_cross_entropy': 平均交叉注意力熵
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape

    # 创建文本和表格的掩码
    text_mask = (token_type_ids == 0) & (attention_mask == 1)  # [batch_size, seq_len]
    table_mask = (token_type_ids == 1) & (attention_mask == 1)  # [batch_size, seq_len]

    # 扩展掩码以匹配注意力维度
    text_mask_q = text_mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, seq_len, 1]
    text_mask_k = text_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    table_mask_q = table_mask.unsqueeze(1).unsqueeze(3)  # [batch_size, 1, seq_len, 1]
    table_mask_k = table_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

    # 提取文本到表格的注意力（query是文本，key是表格）
    text_to_table_attn = attention_weights * text_mask_q * table_mask_k
    # 归一化
    text_to_table_attn = text_to_table_attn / (text_to_table_attn.sum(dim=-1, keepdim=True) + eps)
    # 计算熵
    text_to_table_entropy = compute_attention_entropy(text_to_table_attn, mask=None, eps=eps)
    # 只保留文本位置的熵值
    text_to_table_entropy = text_to_table_entropy * text_mask.unsqueeze(1)  # [batch_size, num_heads, seq_len]

    # 提取表格到文本的注意力（query是表格，key是文本）
    table_to_text_attn = attention_weights * table_mask_q * text_mask_k
    # 归一化
    table_to_text_attn = table_to_text_attn / (table_to_text_attn.sum(dim=-1, keepdim=True) + eps)
    # 计算熵
    table_to_text_entropy = compute_attention_entropy(table_to_text_attn, mask=None, eps=eps)
    # 只保留表格位置的熵值
    table_to_text_entropy = table_to_text_entropy * table_mask.unsqueeze(1)  # [batch_size, num_heads, seq_len]

    # 计算平均熵（考虑有效位置）
    text_count = text_mask.sum(dim=1, keepdim=True).unsqueeze(1)  # [batch_size, 1, 1]
    table_count = table_mask.sum(dim=1, keepdim=True).unsqueeze(1)  # [batch_size, 1, 1]

    mean_text_to_table = text_to_table_entropy.sum(dim=(1, 2)) / (text_count.squeeze() * num_heads + eps)
    mean_table_to_text = table_to_text_entropy.sum(dim=(1, 2)) / (table_count.squeeze() * num_heads + eps)

    mean_cross_entropy = (mean_text_to_table + mean_table_to_text) / 2

    return {
        'text_to_table_entropy': text_to_table_entropy,
        'table_to_text_entropy': table_to_text_entropy,
        'mean_text_to_table': mean_text_to_table,
        'mean_table_to_text': mean_table_to_text,
        'mean_cross_entropy': mean_cross_entropy
    }


def compute_entropy_increase_loss(current_entropy, min_delta=0.1):
    """
    计算熵增约束损失

    参数：
        current_entropy: torch.Tensor, 当前的熵值
        min_delta: float, 期望的最小熵增量

    返回：
        loss: torch.Tensor, 熵增约束损失（标量）

    说明：
        鼓励熵值增加，如果熵值低于期望的最小值，则施加惩罚
        loss = max(0, min_delta - current_entropy)
    """
    # 计算平均熵
    mean_entropy = current_entropy.mean()

    # 如果熵值低于最小期望值，施加惩罚
    loss = F.relu(min_delta - mean_entropy)

    return loss
