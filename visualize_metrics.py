"""
评估指标可视化脚本（单轮 multi / text：F1、Precision、Recall）

功能：
1. 从 result.txt 解析最近一轮的 multi / text 指标，或直接传入指标字典
2. 绘制 multi vs text 的 F1 / Precision / Recall 柱状图
3. 可选保存多轮结果到 CSV，便于后续对比

使用示例：
  # 解析当前目录 result.txt 最后一轮并出图
  python visualize_metrics.py

  # 指定结果文件与输出目录
  python visualize_metrics.py --result_file ./result.txt --output_dir ./metrics_vis

  # 仅对当前一轮数据出图（不读文件）
  python visualize_metrics.py --multi_f1 70.75 --multi_p 69.88 --multi_r 71.64 --text_f1 70.55 --text_p 70.35 --text_r 70.75 --output_suffix 2017_dualc
"""

import argparse
import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_last_run_from_file(result_file):
    """从 result.txt 中解析最后一轮的 参数、multi、text。"""
    path = Path(result_file)
    if not path.exists():
        return None, None, None
    text = path.read_text(encoding="utf-8")
    # 按空行分块，每块为一轮
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        return None, None, None
    last = blocks[-1]
    para, multi, text_metrics = None, None, None
    for line in last.split("\n"):
        line = line.strip()
        if line.startswith("参数:"):
            try:
                para = ast.literal_eval(line.replace("参数:", "").strip())
            except Exception:
                para = None
        elif line.startswith("multi:"):
            try:
                multi = ast.literal_eval(line.replace("multi:", "").strip())
            except Exception:
                multi = None
        elif line.startswith("text:"):
            try:
                text_metrics = ast.literal_eval(line.replace("text:", "").strip())
            except Exception:
                text_metrics = None
    return para, multi, text_metrics


def plot_multi_text_metrics(multi_metrics, text_metrics, save_path, title_suffix=""):
    """绘制 multi vs text 的 F1 / Precision / Recall 分组柱状图。"""
    if not multi_metrics or not text_metrics:
        raise ValueError("需要同时提供 multi 与 text 指标")
    metrics_names = ["F1", "Precision", "Recall"]
    keys = ["f1", "precision", "recall"]
    multi_vals = [float(multi_metrics.get(k, 0)) for k in keys]
    text_vals = [float(text_metrics.get(k, 0)) for k in keys]

    x = np.arange(len(metrics_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, multi_vals, width, label="Multi (图文)", color="#2ecc71", edgecolor="black")
    bars2 = ax.bar(x + width / 2, text_vals, width, label="Text (纯文)", color="#3498db", edgecolor="black")

    ax.set_ylabel("分数 (%)")
    ax.set_title("Multi vs Text 评估指标" + ((" " + title_suffix) if title_suffix else ""))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    def add_value_labels(bars):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    add_value_labels(bars1)
    add_value_labels(bars2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[指标可视化] 已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="评估指标可视化：multi vs text 的 F1/P/R")
    parser.add_argument("--result_file", type=str, default="./result.txt", help="结果文件路径")
    parser.add_argument("--output_dir", type=str, default="./metrics_vis", help="输出目录")
    parser.add_argument("--output_suffix", type=str, default="", help="输出子目录或文件名后缀，如 2017_dualc")
    # 直接传入当前轮指标（不读文件）
    parser.add_argument("--multi_f1", type=float, default=None)
    parser.add_argument("--multi_p", type=float, default=None)
    parser.add_argument("--multi_r", type=float, default=None)
    parser.add_argument("--text_f1", type=float, default=None)
    parser.add_argument("--text_p", type=float, default=None)
    parser.add_argument("--text_r", type=float, default=None)
    args = parser.parse_args()

    multi_metrics = None
    text_metrics = None
    title_suffix = args.output_suffix

    if args.multi_f1 is not None and args.text_f1 is not None:
        multi_metrics = {"f1": args.multi_f1, "precision": args.multi_p or 0, "recall": args.multi_r or 0}
        text_metrics = {"f1": args.text_f1, "precision": args.text_p or 0, "recall": args.text_r or 0}
    else:
        para, multi_metrics, text_metrics = parse_last_run_from_file(args.result_file)
        if para and args.output_suffix == "":
            title_suffix = f"{para.get('dataset_type', '')}_{para.get('ddm_mode', '')}"

    if not multi_metrics or not text_metrics:
        print("未解析到 multi/text 指标。请确认 result_file 格式或使用 --multi_f1/--text_f1 等参数。")
        return

    out_dir = Path(args.output_dir)
    if args.output_suffix:
        out_dir = out_dir / args.output_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "metrics_multi_vs_text.png"
    plot_multi_text_metrics(multi_metrics, text_metrics, save_path, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
