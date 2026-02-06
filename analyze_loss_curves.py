"""
损失曲线分析脚本

功能：
1. 解析 adv_loss_log.txt（对抗损失日志）
2. 解析 cfm_debug.log（CFM调试日志）
3. 绘制损失曲线（各项损失的收敛情况）
4. 分析损失占比和平衡性

使用方法：
    python analyze_loss_curves.py

输出文件（保存在 cfm_vis/ 或 cfm_vis_<suffix>/ 目录）：
- loss_curves.png：各项损失曲线
- loss_ratio.png：损失占比饼图
- loss_statistics.txt：损失统计信息

保留历史对比：加 --output_suffix <模型名或run_id> 或 --timestamp，输出到 cfm_vis_<suffix>/
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_adv_loss_log(file_path="adv_loss_log.txt"):
    """解析对抗损失日志"""
    if not Path(file_path).exists():
        print(f"警告: {file_path} 不存在")
        return None
    
    data = {
        'step': [],
        'cross_crf': [],
        'text': [],
        'align': [],
        'ortho': [],
        'enh': [],
        'lambda1': [],
        'lambda2': [],
        'cfm_consistency': [],
        'cfm_weight_reg': [],
        'lambda3': [],
        'lambda4': [],
        'sent_cls': [],
        'sent_aux': [],
        'lambda_sent_cls': [],
        'lambda_sent_aux': [],
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析各项损失
            step_match = re.search(r'step=(\d+)', line)
            if step_match:
                data['step'].append(int(step_match.group(1)))
            
            for key in ['cross_crf', 'text', 'align', 'ortho', 'enh', 
                       'lambda1', 'lambda2', 'cfm_consistency', 'cfm_weight_reg', 
                       'lambda3', 'lambda4', 'sent_cls', 'sent_aux', 
                       'lambda_sent_cls', 'lambda_sent_aux']:
                pattern = rf'{key}=([\d.]+)'
                match = re.search(pattern, line)
                if match:
                    data[key].append(float(match.group(1)))
                else:
                    # 如果该项不存在，填充NaN（用于CFM未启用的情况）
                    if len(data['step']) > len(data[key]):
                        data[key].append(np.nan)
    
    # 转换为numpy数组
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def parse_cfm_log(file_path="cfm_debug.log"):
    """解析CFM调试日志"""
    if not Path(file_path).exists():
        print(f"警告: {file_path} 不存在")
        return None
    
    data = {
        'step': [],
        'w_text_mean': [],
        'w_image_mean': [],
        'w_shared_mean': [],
        'w_text_std': [],
        'w_image_std': [],
        'w_shared_std': [],
        'w_entropy': [],
        'consistency_loss': [],
        'weight_reg_loss': [],
        'lambda3': [],
        'lambda4': [],
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('[CFM]'):
                continue
            
            # 解析各项数据
            step_match = re.search(r'step=(\d+)', line)
            if step_match:
                data['step'].append(int(step_match.group(1)))
            
            for key in data.keys():
                if key == 'step':
                    continue
                pattern = rf'{key}=([\d.]+)'
                match = re.search(pattern, line)
                if match:
                    data[key].append(float(match.group(1)))
                else:
                    if len(data['step']) > len(data[key]):
                        data[key].append(np.nan)
    
    # 转换为numpy数组
    for key in data:
        data[key] = np.array(data[key])
    
    return data


def plot_loss_curves(adv_data, cfm_data, save_path):
    """绘制损失曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 主任务损失（cross_crf, text, align）
    ax1 = axes[0, 0]
    if adv_data is not None and len(adv_data['step']) > 0:
        steps = adv_data['step']
        ax1.plot(steps, adv_data['cross_crf'], label='Cross CRF', linewidth=2, alpha=0.8)
        ax1.plot(steps, adv_data['text'], label='Text Loss', linewidth=2, alpha=0.8)
        ax1.plot(steps, adv_data['align'], label='Alignment Loss', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Step', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Main Task Losses', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
    
    # 2. 对抗损失（ortho, enhanced）
    ax2 = axes[0, 1]
    if adv_data is not None and len(adv_data['step']) > 0:
        steps = adv_data['step']
        ax2.plot(steps, adv_data['ortho'], label=f'Orthogonal (λ={adv_data["lambda1"][0]:.2f})', 
                linewidth=2, alpha=0.8, color='orange')
        if not np.isnan(adv_data['enh']).all():
            ax2.plot(steps, adv_data['enh'], label=f'Enhanced (λ={adv_data["lambda2"][0]:.2f})', 
                    linewidth=2, alpha=0.8, color='red')
        ax2.set_xlabel('Step', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.set_title('Adversarial Losses', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    # 3. CFM辅助损失（consistency, weight_reg）
    ax3 = axes[1, 0]
    if cfm_data is not None and len(cfm_data['step']) > 0:
        steps = cfm_data['step']
        ax3.plot(steps, cfm_data['consistency_loss'], 
                label=f'Consistency (λ={cfm_data["lambda3"][0]:.2f})', 
                linewidth=2, alpha=0.8, color='purple')
        ax3.plot(steps, cfm_data['weight_reg_loss'], 
                label=f'Weight Reg (λ={cfm_data["lambda4"][0]:.2f})', 
                linewidth=2, alpha=0.8, color='brown')
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Loss', fontsize=11)
        ax3.set_title('CFM Auxiliary Losses', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    elif adv_data is not None and not np.isnan(adv_data['cfm_consistency']).all():
        steps = adv_data['step']
        ax3.plot(steps, adv_data['cfm_consistency'], 
                label=f'Consistency (λ={adv_data["lambda3"][0]:.2f})', 
                linewidth=2, alpha=0.8, color='purple')
        ax3.plot(steps, adv_data['cfm_weight_reg'], 
                label=f'Weight Reg (λ={adv_data["lambda4"][0]:.2f})', 
                linewidth=2, alpha=0.8, color='brown')
        ax3.set_xlabel('Step', fontsize=11)
        ax3.set_ylabel('Loss', fontsize=11)
        ax3.set_title('CFM Auxiliary Losses', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # 4. 总损失（加权后）
    ax4 = axes[1, 1]
    if adv_data is not None and len(adv_data['step']) > 0:
        steps = adv_data['step']
        # 计算总损失（加权后）
        total_loss = (
            adv_data['cross_crf'] +
            adv_data['text'] * 0.635 +  # alpha
            adv_data['align'] * 0.565 +  # beta
            adv_data['ortho'] * adv_data['lambda1'] +
            adv_data['enh'] * adv_data['lambda2']
        )
        
        # 如果启用了CFM，加上CFM损失
        if not np.isnan(adv_data['cfm_consistency']).all():
            total_loss += (
                adv_data['cfm_consistency'] * adv_data['lambda3'] +
                adv_data['cfm_weight_reg'] * adv_data['lambda4']
            )
        # 如果启用了情感监督，加上情感损失
        if 'sent_cls' in adv_data and not np.isnan(adv_data.get('sent_cls', [np.nan])).all():
            total_loss += adv_data['sent_cls'] * adv_data['lambda_sent_cls']
        if 'sent_aux' in adv_data and not np.isnan(adv_data.get('sent_aux', [np.nan])).all():
            total_loss += adv_data['sent_aux'] * adv_data['lambda_sent_aux']
        
        ax4.plot(steps, total_loss, label='Total Loss (Weighted)', 
                linewidth=2, alpha=0.8, color='black')
        ax4.set_xlabel('Step', fontsize=11)
        ax4.set_ylabel('Loss', fontsize=11)
        ax4.set_title('Total Weighted Loss', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存损失曲线: {save_path}")
    plt.close()


def plot_loss_ratio(adv_data, save_path):
    """绘制损失占比饼图"""
    if adv_data is None or len(adv_data['step']) == 0:
        print("无法绘制损失占比：数据不足")
        return
    
    # 使用最后100个step的平均值
    last_n = min(100, len(adv_data['step']))
    recent_data = {k: v[-last_n:] for k, v in adv_data.items() if isinstance(v, np.ndarray)}
    
    # 计算加权后的平均损失
    avg_cross_crf = np.nanmean(recent_data['cross_crf'])
    avg_text = np.nanmean(recent_data['text']) * 0.635  # alpha
    avg_align = np.nanmean(recent_data['align']) * 0.565  # beta
    avg_ortho = np.nanmean(recent_data['ortho']) * np.nanmean(recent_data['lambda1'])
    avg_enh = np.nanmean(recent_data['enh']) * np.nanmean(recent_data['lambda2'])
    
    losses = {
        'Cross CRF': avg_cross_crf,
        'Text Loss': avg_text,
        'Alignment': avg_align,
        'Orthogonal': avg_ortho,
        'Enhanced': avg_enh,
    }
    
    # 如果启用了CFM，加上CFM损失
    if not np.isnan(recent_data.get('cfm_consistency', [np.nan])).all():
        avg_cfm_consistency = np.nanmean(recent_data['cfm_consistency']) * np.nanmean(recent_data['lambda3'])
        avg_cfm_weight_reg = np.nanmean(recent_data['cfm_weight_reg']) * np.nanmean(recent_data['lambda4'])
        losses['CFM Consistency'] = avg_cfm_consistency
        losses['CFM Weight Reg'] = avg_cfm_weight_reg
    
    # 如果启用了情感监督，加上情感损失
    if 'sent_cls' in recent_data and not np.isnan(recent_data.get('sent_cls', [np.nan])).all():
        losses['Sentiment Cls'] = np.nanmean(recent_data['sent_cls']) * np.nanmean(recent_data.get('lambda_sent_cls', [0]))
    if 'sent_aux' in recent_data and not np.isnan(recent_data.get('sent_aux', [np.nan])).all():
        losses['Sentiment Aux'] = np.nanmean(recent_data['sent_aux']) * np.nanmean(recent_data.get('lambda_sent_aux', [0]))
    
    # 过滤掉NaN和0值
    losses = {k: v for k, v in losses.items() if not np.isnan(v) and v > 0}
    
    if len(losses) == 0:
        print("无法绘制损失占比：所有损失值无效")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(losses)))
    wedges, texts, autotexts = ax.pie(losses.values(), labels=losses.keys(), autopct='%1.1f%%',
                                       colors=colors, startangle=90)
    
    # 美化文本
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    ax.set_title('Loss Ratio (Weighted Average, Last 100 Steps)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存损失占比图: {save_path}")
    plt.close()


def save_loss_statistics(adv_data, cfm_data, save_path):
    """保存损失统计信息"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("损失曲线分析统计信息\n")
        f.write("=" * 60 + "\n\n")
        
        if adv_data is not None and len(adv_data['step']) > 0:
            f.write("1. 主任务损失（最后100步平均）:\n")
            last_n = min(100, len(adv_data['step']))
            f.write(f"   Cross CRF: {np.nanmean(adv_data['cross_crf'][-last_n:]):.4f}\n")
            f.write(f"   Text Loss: {np.nanmean(adv_data['text'][-last_n:]):.4f}\n")
            f.write(f"   Alignment: {np.nanmean(adv_data['align'][-last_n:]):.4f}\n\n")
            
            f.write("2. 对抗损失（最后100步平均）:\n")
            f.write(f"   Orthogonal: {np.nanmean(adv_data['ortho'][-last_n:]):.4f} "
                   f"(λ={np.nanmean(adv_data['lambda1']):.3f})\n")
            if not np.isnan(adv_data['enh']).all():
                f.write(f"   Enhanced: {np.nanmean(adv_data['enh'][-last_n:]):.4f} "
                       f"(λ={np.nanmean(adv_data['lambda2']):.3f})\n")
            f.write("\n")
            
            if not np.isnan(adv_data.get('cfm_consistency', [np.nan])).all():
                f.write("3. CFM辅助损失（最后100步平均）:\n")
                f.write(f"   Consistency: {np.nanmean(adv_data['cfm_consistency'][-last_n:]):.4f} "
                       f"(λ={np.nanmean(adv_data['lambda3']):.3f})\n")
                f.write(f"   Weight Reg: {np.nanmean(adv_data['cfm_weight_reg'][-last_n:]):.4f} "
                       f"(λ={np.nanmean(adv_data['lambda4']):.3f})\n\n")
            if 'sent_cls' in adv_data and not np.isnan(adv_data.get('sent_cls', [np.nan])).all():
                f.write("4. 情感监督损失（最后100步平均）:\n")
                f.write(f"   Sent Cls: {np.nanmean(adv_data['sent_cls'][-last_n:]):.4f} "
                       f"(λ={np.nanmean(adv_data.get('lambda_sent_cls', [0])):.3f})\n")
                f.write(f"   Sent Aux: {np.nanmean(adv_data['sent_aux'][-last_n:]):.4f} "
                       f"(λ={np.nanmean(adv_data.get('lambda_sent_aux', [0])):.3f})\n\n")
        
        if cfm_data is not None and len(cfm_data['step']) > 0:
            f.write("4. CFM权重统计（最后100步平均）:\n")
            last_n = min(100, len(cfm_data['step']))
            f.write(f"   Text Weight: {np.nanmean(cfm_data['w_text_mean'][-last_n:]):.4f} "
                   f"± {np.nanmean(cfm_data['w_text_std'][-last_n:]):.4f}\n")
            f.write(f"   Image Weight: {np.nanmean(cfm_data['w_image_mean'][-last_n:]):.4f} "
                   f"± {np.nanmean(cfm_data['w_image_std'][-last_n:]):.4f}\n")
            f.write(f"   Shared Weight: {np.nanmean(cfm_data['w_shared_mean'][-last_n:]):.4f} "
                   f"± {np.nanmean(cfm_data['w_shared_std'][-last_n:]):.4f}\n")
            f.write(f"   Weight Entropy: {np.nanmean(cfm_data['w_entropy'][-last_n:]):.4f}\n")
    
    print(f"已保存损失统计信息: {save_path}")


def main(output_dir=None):
    """主函数。output_dir 为 None 时使用默认 cfm_vis。"""
    output_dir = Path(output_dir) if output_dir is not None else Path("cfm_vis")
    output_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("损失曲线分析")
    print("=" * 60)
    
    # 1. 解析日志文件
    print("\n解析日志文件...")
    adv_data = parse_adv_loss_log()
    cfm_data = parse_cfm_log()
    
    if adv_data is None and cfm_data is None:
        print("错误: 没有找到任何日志文件")
        print("请先运行训练，并确保生成了 adv_loss_log.txt 或 cfm_debug.log")
        return
    
    # 2. 绘制损失曲线
    print("\n生成损失曲线...")
    plot_loss_curves(adv_data, cfm_data, output_dir / "loss_curves.png")
    
    # 3. 绘制损失占比
    if adv_data is not None:
        plot_loss_ratio(adv_data, output_dir / "loss_ratio.png")
    
    # 4. 保存统计信息
    save_loss_statistics(adv_data, cfm_data, output_dir / "loss_statistics.txt")
    
    print(f"\n分析完成！可视化文件已保存到 {output_dir}/ 目录")


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="损失曲线分析，支持输出目录带模型名/时间轴便于对比")
    parser.add_argument("--output_dir", default="cfm_vis", help="输出目录前缀")
    parser.add_argument("--output_suffix", default=None, help="输出目录后缀（模型名或 run_id），实际目录为 output_dir_output_suffix")
    parser.add_argument("--timestamp", action="store_true", help="使用当前时间作为 output_suffix（YYYYMMDD_HHMMSS）")
    args = parser.parse_args()
    suffix = args.output_suffix
    if args.timestamp:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"{args.output_dir}_{suffix}" if suffix else args.output_dir
    main(output_dir=out)
