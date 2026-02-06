"""
CFM融合权重可视化脚本

功能：
1. 读取 cfm_fusion_weights_sample.npy（包含融合权重和标签）
2. 分析权重分布（text/image/shared的平均权重、方差、熵）
3. 可视化权重分布（柱状图、热图、按标签分组分析）
4. 分析不同样本/位置的权重模式

使用方法：
    python visualize_cfm_weights.py

输出文件（保存在 cfm_vis/ 或 cfm_vis_<suffix>/ 目录）：
- cfm_weights_distribution.png：权重分布柱状图
- cfm_weights_heatmap.png：权重热图（样本×位置×模态）
- cfm_weights_by_label.png：按标签分组的权重分析
- cfm_weights_statistics.txt：权重统计信息

保留历史对比：加 --output_suffix <模型名或run_id> 或 --timestamp，输出到 cfm_vis_<suffix>/
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_cfm_weights(file_path="cfm_fusion_weights_sample.npy"):
    """加载CFM权重数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"权重文件不存在: {file_path}")
    
    data = np.load(file_path, allow_pickle=True).item()
    fusion_weights = data["fusion_weights"]  # [N, L, 3]
    labels = data.get("labels", None)
    
    print(f"加载权重数据: shape={fusion_weights.shape}")
    if labels is not None:
        print(f"标签数据: shape={labels.shape}")
    
    return fusion_weights, labels


def compute_weight_statistics(weights):
    """
    计算权重统计信息
    
    Args:
        weights: [N, L, 3] - 融合权重
    
    Returns:
        dict: 包含各种统计信息
    """
    N, L, _ = weights.shape
    
    # 平均权重（每个模态在所有样本和位置上的平均）
    mean_weights = weights.mean(axis=(0, 1))  # [3]
    
    # 标准差
    std_weights = weights.std(axis=(0, 1))  # [3]
    
    # 每个样本的平均权重
    sample_mean = weights.mean(axis=1)  # [N, 3]
    
    # 每个位置的平均权重
    position_mean = weights.mean(axis=0)  # [L, 3]
    
    # 权重熵（每个样本每个位置）
    entropy = -(weights * np.log(weights + 1e-8)).sum(axis=-1)  # [N, L]
    mean_entropy = entropy.mean()
    max_entropy = np.log(3)  # 均匀分布时的最大熵
    
    # 权重集中度（熵越低，权重越集中）
    concentration = 1 - (mean_entropy / max_entropy)
    
    stats = {
        "mean_weights": mean_weights,  # [3] text/image/shared
        "std_weights": std_weights,  # [3]
        "sample_mean": sample_mean,  # [N, 3]
        "position_mean": position_mean,  # [L, 3]
        "entropy": entropy,  # [N, L]
        "mean_entropy": mean_entropy,
        "max_entropy": max_entropy,
        "concentration": concentration,
    }
    
    return stats


def plot_weight_distribution(weights, stats, save_path):
    """绘制权重分布柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 平均权重柱状图
    ax1 = axes[0]
    modalities = ['Text', 'Image', 'Shared']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax1.bar(modalities, stats["mean_weights"], color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(modalities, stats["mean_weights"], yerr=stats["std_weights"], 
                 fmt='none', color='black', capsize=5, capthick=2)
    ax1.set_ylabel('Average Weight', fontsize=12)
    ax1.set_title('Average Fusion Weights by Modality', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, stats["mean_weights"], stats["std_weights"])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. 权重熵分布
    ax2 = axes[1]
    entropy_flat = stats["entropy"].flatten()
    ax2.hist(entropy_flat, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(stats["mean_entropy"], color='red', linestyle='--', linewidth=2, 
                label=f'Mean Entropy: {stats["mean_entropy"]:.3f}')
    ax2.axvline(stats["max_entropy"], color='green', linestyle='--', linewidth=2,
                label=f'Max Entropy (Uniform): {stats["max_entropy"]:.3f}')
    ax2.set_xlabel('Entropy', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Weight Entropy Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存权重分布图: {save_path}")
    plt.close()


def plot_weight_heatmap(weights, save_path, max_samples=20):
    """绘制权重热图（样本×位置×模态）"""
    N, L, _ = weights.shape
    
    # 如果样本太多，随机选择一部分
    if N > max_samples:
        indices = np.random.choice(N, max_samples, replace=False)
        weights_subset = weights[indices]
    else:
        weights_subset = weights
        indices = np.arange(N)
    
    N_subset = weights_subset.shape[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, N_subset * 0.3))
    
    for i, (modality, ax) in enumerate(zip(['Text', 'Image', 'Shared'], axes)):
        # 提取该模态的权重 [N_subset, L]
        w_mod = weights_subset[:, :, i]
        
        # 绘制热图
        im = ax.imshow(w_mod, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_title(f'{modality} Weight Heatmap', fontsize=12, fontweight='bold')
        ax.set_xlabel('Token Position', fontsize=10)
        ax.set_ylabel('Sample Index', fontsize=10)
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, label='Weight')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存权重热图: {save_path}")
    plt.close()


def plot_weights_by_label(weights, labels, save_path):
    """按标签分组分析权重"""
    if labels is None:
        print("警告: 没有标签数据，跳过按标签分析")
        return
    
    N, L = labels.shape
    
    # 标签映射
    label_names = {0: 'O', 1: 'I', 2: 'B-NEG', 3: 'B-NEU', 4: 'B-POS'}
    label_colors = {0: 'gray', 1: 'lightblue', 2: 'red', 3: 'orange', 4: 'green'}
    
    # 统计每个标签的平均权重
    label_weights = {}
    for label_id, label_name in label_names.items():
        mask = (labels == label_id)
        if mask.sum() > 0:
            # 提取该标签对应的权重 [M, 3]，M是匹配的token数量
            w_masked = weights[mask]  # [M, 3]
            label_weights[label_id] = {
                'name': label_name,
                'mean': w_masked.mean(axis=0),  # [3]
                'std': w_masked.std(axis=0),  # [3]
                'count': mask.sum(),
            }
    
    # 绘制按标签分组的权重
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(label_weights))
    width = 0.25
    
    modalities = ['Text', 'Image', 'Shared']
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    for i, (modality, color) in enumerate(zip(modalities, colors)):
        means = [label_weights[lid]['mean'][i] for lid in sorted(label_weights.keys())]
        stds = [label_weights[lid]['std'][i] for lid in sorted(label_weights.keys())]
        labels_list = [label_weights[lid]['name'] for lid in sorted(label_weights.keys())]
        
        ax.bar(x_pos + i * width, means, width, yerr=stds, label=modality, 
               color=color, alpha=0.7, edgecolor='black', capsize=3)
    
    ax.set_xlabel('Label', fontsize=12)
    ax.set_ylabel('Average Weight', fontsize=12)
    ax.set_title('Fusion Weights by Label', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(labels_list)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存按标签权重分析: {save_path}")
    plt.close()


def save_statistics(stats, save_path):
    """保存统计信息到文本文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("CFM融合权重统计信息\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("1. 平均权重（所有样本和位置）:\n")
        f.write(f"   Text:   {stats['mean_weights'][0]:.4f} ± {stats['std_weights'][0]:.4f}\n")
        f.write(f"   Image:  {stats['mean_weights'][1]:.4f} ± {stats['std_weights'][1]:.4f}\n")
        f.write(f"   Shared: {stats['mean_weights'][2]:.4f} ± {stats['std_weights'][2]:.4f}\n\n")
        
        f.write("2. 权重熵:\n")
        f.write(f"   平均熵: {stats['mean_entropy']:.4f}\n")
        f.write(f"   最大熵（均匀分布）: {stats['max_entropy']:.4f}\n")
        f.write(f"   权重集中度: {stats['concentration']:.4f} (0=均匀, 1=完全集中)\n\n")
        
        f.write("3. 权重分布分析:\n")
        f.write(f"   如果Text权重 > 0.5: 模型主要依赖文本信息\n")
        f.write(f"   如果Image权重 > 0.5: 模型主要依赖图像信息\n")
        f.write(f"   如果Shared权重 > 0.5: 模型主要依赖共享特征\n")
        f.write(f"   如果权重集中度 > 0.3: 融合策略偏向单一模态\n")
        f.write(f"   如果权重集中度 < 0.1: 融合策略较为平衡\n")
    
    print(f"已保存统计信息: {save_path}")


def main(output_dir=None):
    """主函数。output_dir 为 None 时使用默认 cfm_vis。"""
    output_dir = Path(output_dir) if output_dir is not None else Path("cfm_vis")
    output_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("CFM融合权重可视化")
    print("=" * 60)
    
    # 1. 加载数据
    try:
        weights, labels = load_cfm_weights()
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行训练，并启用 --cfm_debug 参数生成权重文件")
        return
    
    # 2. 计算统计信息
    print("\n计算统计信息...")
    stats = compute_weight_statistics(weights)
    
    # 3. 绘制可视化图表
    print("\n生成可视化图表...")
    plot_weight_distribution(weights, stats, output_dir / "cfm_weights_distribution.png")
    plot_weight_heatmap(weights, output_dir / "cfm_weights_heatmap.png")
    if labels is not None:
        plot_weights_by_label(weights, labels, output_dir / "cfm_weights_by_label.png")
    
    # 4. 保存统计信息
    save_statistics(stats, output_dir / "cfm_weights_statistics.txt")
    
    # 5. 打印关键信息
    print("\n" + "=" * 60)
    print("关键统计信息:")
    print("=" * 60)
    print(f"平均权重 - Text: {stats['mean_weights'][0]:.4f}, "
          f"Image: {stats['mean_weights'][1]:.4f}, "
          f"Shared: {stats['mean_weights'][2]:.4f}")
    print(f"权重熵: {stats['mean_entropy']:.4f} / {stats['max_entropy']:.4f} "
          f"(集中度: {stats['concentration']:.4f})")
    
    if stats['mean_weights'][0] > 0.5:
        print("⚠️ 模型主要依赖文本信息，图像信息利用不足")
    elif stats['mean_weights'][1] > 0.5:
        print("⚠️ 模型主要依赖图像信息，文本信息利用不足")
    elif stats['mean_weights'][2] > 0.5:
        print("✅ 模型主要依赖共享特征，跨模态融合良好")
    else:
        print("✅ 模型平衡使用三路特征")
    
    if stats['concentration'] > 0.3:
        print("⚠️ 权重分布较集中，可能偏向单一模态")
    else:
        print("✅ 权重分布较均匀，融合策略平衡")
    
    print(f"\n可视化文件已保存到 {output_dir}/ 目录")


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="CFM 融合权重可视化，支持输出目录带模型名/时间轴便于对比")
    parser.add_argument("--weights", default="cfm_fusion_weights_sample.npy", help="权重 npy 文件路径")
    parser.add_argument("--output_dir", default="cfm_vis", help="输出目录前缀")
    parser.add_argument("--output_suffix", default=None, help="输出目录后缀（模型名或 run_id），实际目录为 output_dir_output_suffix")
    parser.add_argument("--timestamp", action="store_true", help="使用当前时间作为 output_suffix（YYYYMMDD_HHMMSS）")
    args = parser.parse_args()
    suffix = args.output_suffix
    if args.timestamp:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{args.output_dir}_{suffix}" if suffix else args.output_dir
    main(output_dir=out)
