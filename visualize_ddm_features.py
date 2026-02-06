"""
DDM 特征可视化脚本
==================

功能：
- 读取训练过程中由 ICTModel 导出的 `ddm_features_sample.npy`
- 对 text_specific / image_specific / shared_features / text_fused 进行降维可视化
- 默认使用 t-SNE 将高维特征投影到 2D 平面，并保存为 PNG 图片
- 支持按情感类别（label）着色，观察同类样本的聚集情况
- 支持输出目录带模型名/时间轴后缀，便于保留历史、对比不同 run

使用方法（在服务器或本地均可，只要有 ddm_features_sample.npy 即可）：

1. 确保在训练时打开了 DDM 调试开关（ddm_debug=True），并生成了特征文件：
   - 文件路径默认为：ddm_features_sample.npy
2. 在同一目录下运行：
   - python visualize_ddm_features.py
   - 保留历史对比：加 --output_suffix <模型名或run_id> 或 --timestamp，输出到 ddm_vis_<suffix>/
   - 示例：python visualize_ddm_features.py --output_suffix sent2017
   - 示例：python visualize_ddm_features.py --timestamp
3. 生成的图片：
   - ddm_tsne_text_vs_shared.png        （text_specific vs shared_features，按模态类型着色）
   - ddm_tsne_image_vs_shared.png       （image_specific vs shared_features，按模态类型着色）
   - ddm_tsne_text_image_shared.png     （text_specific / image_specific / shared 三类对比，按模态类型着色）
   - ddm_tsne_shared_by_label.png       （shared 按情感类别着色，情感类大点便于找 neutral）
   - ddm_tsne_shared_by_label_sentiment_only.png   （仅 B-NEG/B-NEU/B-POS，便于看 shared 情感分布）
   - ddm_tsne_text_specific_by_label.png（text_specific 按情感类别着色）
   - ddm_tsne_text_specific_by_label_sentiment_only.png（仅情感）
   - ddm_tsne_image_specific_by_label.png（image_specific 按情感类别着色）
   - ddm_tsne_image_specific_by_label_sentiment_only.png（仅情感，便于在图像空间找 NEG/NEU/POS）

标签体系：
- 0: 非方面词（O）
- 1: 方面词延续（I）
- 2: 负面情感开始（B-NEG）
- 3: 中性情感开始（B-NEU）
- 4: 正面情感开始（B-POS）
"""

import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    """确保输出目录存在。"""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def visualize_ddm_features(
    feat_path: str = "ddm_features_sample.npy",
    output_dir: str = "ddm_vis",
    output_suffix: str = None
) -> None:
    """
    可视化 DDM 导出的特征。

    Args:
        feat_path: DDM 特征 numpy 文件路径，默认 'ddm_features_sample.npy'
        output_dir: 可视化结果输出目录（不含后缀时即为此名）
        output_suffix: 可选。若提供，实际输出目录为 output_dir + '_' + output_suffix，便于保留历史、对比不同模型/run
    """
    if output_suffix:
        output_dir = f"{output_dir}_{output_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"未找到特征文件: {feat_path}，请先在训练时开启 ddm_debug 并确保模型保存了该文件。"
        )

    data = np.load(feat_path, allow_pickle=True).item()
    text_specific = data.get("text_specific")
    image_specific = data.get("image_specific")
    shared_features = data.get("shared_features")
    text_fused = data.get("text_fused")
    image_specific_aligned = data.get("image_specific_aligned", None)  # 可选：DDM token 级对齐后的图像特征
    labels = data.get("labels", None)

    # 展平到 [N, C]
    def flatten(feat):
        if feat is None:
            return None
        # [B, L, C] -> [B*L, C]
        return feat.reshape(-1, feat.shape[-1])

    ts = flatten(text_specific)
    is_ = flatten(image_specific)
    sh = flatten(shared_features)
    tf = flatten(text_fused)
    is_aligned = flatten(image_specific_aligned) if image_specific_aligned is not None else None

    # 展平labels到 [B*L]，并过滤掉-100（padding标签）
    def flatten_labels(labels_data):
        if labels_data is None:
            return None
        # [B, L] -> [B*L]
        labels_flat = labels_data.reshape(-1)
        # 过滤掉-100（padding标签）
        valid_mask = (labels_flat != -100)
        return labels_flat[valid_mask], valid_mask

    labels_flat, labels_mask = flatten_labels(labels) if labels is not None else (None, None)

    _ensure_dir(output_dir)

    def tsne_plot(X_list, y_list, label_names, title, filename):
        """
        通用 t-SNE 绘图辅助函数（按模态类型着色）。

        Args:
            X_list: 特征数组列表，每个元素是 [N_i, C]
            y_list: 每组特征对应的整型标签（如 0,1,2）
            label_names: 标签名称列表
            title: 图标题
            filename: 输出文件名
        """
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        tsne = TSNE(n_components=2, random_state=42, init="random", learning_rate="auto")
        X_2d = tsne.fit_transform(X)

        plt.figure(figsize=(6, 6))
        for label_id, name in enumerate(label_names):
            mask = (y == label_id)
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=5,
                alpha=0.6,
                label=name
            )
        plt.legend()
        plt.title(title)
        plt.tight_layout()

        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[DDM VIS] 保存可视化结果到: {out_path}")

    # 情感类别标签（2=B-NEG, 3=B-NEU, 4=B-POS），用于“仅情感”视图和突出显示
    SENTIMENT_LABELS = (2, 3, 4)
    label_names = {
        0: "O (Non-aspect)",
        1: "I (Aspect continuation)",
        2: "B-NEG (Negative)",
        3: "B-NEU (Neutral)",
        4: "B-POS (Positive)"
    }
    colors = {
        0: 'gray',
        1: 'lightblue',
        2: 'red',
        3: 'orange',
        4: 'green'
    }

    def tsne_plot_by_label(X, labels_data, title, filename, max_samples=2000, sentiment_only=False, sentiment_bigger=True):
        """
        按情感类别（label）着色的 t-SNE 绘图函数。

        Args:
            X: 特征数组 [N, C]
            labels_data: 标签数组 [N]（已过滤-100）
            title: 图标题
            filename: 输出文件名
            max_samples: 最大样本数（防止t-SNE过慢）
            sentiment_only: 若为 True，只保留 B-NEG/B-NEU/B-POS，便于观察 shared 情感分布和 neutral
            sentiment_bigger: 若为 True，情感类（2,3,4）用更大点且最后绘制，便于找到 neutral
        """
        if labels_data is None or len(labels_data) == 0:
            print(f"[DDM VIS] 跳过 {filename}：无标签数据")
            return

        if sentiment_only:
            mask_sent = np.isin(labels_data, list(SENTIMENT_LABELS))
            if mask_sent.sum() == 0:
                print(f"[DDM VIS] 跳过 {filename}（sentiment_only）：无情感样本")
                return
            X_subset = X[mask_sent]
            labels_subset = labels_data[mask_sent]
            n = min(len(X_subset), max_samples)
            X_subset, labels_subset = X_subset[:n], labels_subset[:n]
        else:
            n = min(len(X), len(labels_data), max_samples)
            X_subset = X[:n]
            labels_subset = labels_data[:n]

        tsne = TSNE(n_components=2, random_state=42, init="random", learning_rate="auto")
        X_2d = tsne.fit_transform(X_subset)

        plt.figure(figsize=(10, 8))
        # 绘制顺序：先 O、I（小点），再 B-NEG、B-NEU、B-POS（大点、最后画以便在上层）
        order = [0, 1, 2, 3, 4] if not sentiment_bigger else [0, 1, 2, 3, 4]
        unique_labels = [u for u in order if u in np.unique(labels_subset)]
        for label_id in unique_labels:
            mask = (labels_subset == label_id)
            if not mask.any():
                continue
            label_name = label_names.get(label_id, f"Label {label_id}")
            color = colors.get(label_id, 'black')
            is_sentiment = label_id in SENTIMENT_LABELS
            size = 24 if (sentiment_bigger and is_sentiment) else 8
            alpha = 0.85 if is_sentiment else 0.5
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=size,
                alpha=alpha,
                c=color,
                label=label_name,
                edgecolors='white' if is_sentiment else 'none',
                linewidths=0.3
            )
        
        plt.legend(loc='best', fontsize=9)
        plt.title(title, fontsize=12)
        plt.xlabel("t-SNE Dimension 1", fontsize=10)
        plt.ylabel("t-SNE Dimension 2", fontsize=10)
        plt.tight_layout()

        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[DDM VIS] 保存可视化结果到: {out_path}")

    # 1. text_specific vs shared_features
    if ts is not None and sh is not None:
        n = min(len(ts), len(sh), 2000)  # 限制样本数，防止 t-SNE 过慢
        X_list = [ts[:n], sh[:n]]
        y_list = [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        tsne_plot(
            X_list,
            y_list,
            ["text_specific", "shared_features"],
            "t-SNE: text_specific vs shared_features",
            "ddm_tsne_text_vs_shared.png",
        )

    # 2. image_specific vs shared_features
    if is_ is not None and sh is not None:
        n = min(len(is_), len(sh), 2000)
        X_list = [is_[:n], sh[:n]]
        y_list = [np.zeros(n, dtype=int), np.ones(n, dtype=int)]
        tsne_plot(
            X_list,
            y_list,
            ["image_specific", "shared_features"],
            "t-SNE: image_specific vs shared_features",
            "ddm_tsne_image_vs_shared.png",
        )

    # 3. text_specific / image_specific / shared_features 三类一起
    if ts is not None and is_ is not None and sh is not None:
        n = min(len(ts), len(is_), len(sh), 1500)
        X_list = [ts[:n], is_[:n], sh[:n]]
        y_list = [
            np.zeros(n, dtype=int),
            np.ones(n, dtype=int),
            np.full(n, 2, dtype=int),
        ]
        tsne_plot(
            X_list,
            y_list,
            ["text_specific", "image_specific", "shared_features"],
            "t-SNE: text_specific vs image_specific vs shared",
            "ddm_tsne_text_image_shared.png",
        )

    # 4-6. 按情感类别（label）着色的可视化（全类别，情感类用大点便于找 neutral）
    if labels_flat is not None:
        # 4. shared_features 按 label 着色
        if sh is not None:
            sh_valid = sh[labels_mask] if labels_mask is not None else sh
            tsne_plot_by_label(
                sh_valid,
                labels_flat,
                "t-SNE: shared_features (colored by sentiment labels)",
                "ddm_tsne_shared_by_label.png",
                max_samples=2000,
                sentiment_only=False,
                sentiment_bigger=True
            )
            # 4b. shared 仅情感视图：只看 B-NEG/B-NEU/B-POS，便于观察“shared 情感”和 neutral
            tsne_plot_by_label(
                sh_valid,
                labels_flat,
                "t-SNE: shared_features (sentiment only: NEG/NEU/POS)",
                "ddm_tsne_shared_by_label_sentiment_only.png",
                max_samples=2000,
                sentiment_only=True,
                sentiment_bigger=False
            )

        # 5. text_specific 按 label 着色
        if ts is not None:
            ts_valid = ts[labels_mask] if labels_mask is not None else ts
            tsne_plot_by_label(
                ts_valid,
                labels_flat,
                "t-SNE: text_specific (colored by sentiment labels)",
                "ddm_tsne_text_specific_by_label.png",
                max_samples=2000,
                sentiment_only=False,
                sentiment_bigger=True
            )
            tsne_plot_by_label(
                ts_valid,
                labels_flat,
                "t-SNE: text_specific (sentiment only: NEG/NEU/POS)",
                "ddm_tsne_text_specific_by_label_sentiment_only.png",
                max_samples=2000,
                sentiment_only=True,
                sentiment_bigger=False
            )

        # 6. image_specific 按 label 着色
        if is_ is not None:
            if labels_mask is not None and len(is_) == len(labels_mask):
                is_valid = is_[labels_mask]
                labels_flat_for_image = labels_flat
            else:
                n_min = min(len(is_), len(labels_flat))
                is_valid = is_[:n_min]
                labels_flat_for_image = labels_flat[:n_min]
            
            tsne_plot_by_label(
                is_valid,
                labels_flat_for_image,
                "t-SNE: image_specific (colored by sentiment labels)",
                "ddm_tsne_image_specific_by_label.png",
                max_samples=2000,
                sentiment_only=False,
                sentiment_bigger=True
            )
            tsne_plot_by_label(
                is_valid,
                labels_flat_for_image,
                "t-SNE: image_specific (sentiment only: NEG/NEU/POS)",
                "ddm_tsne_image_specific_by_label_sentiment_only.png",
                max_samples=2000,
                sentiment_only=True,
                sentiment_bigger=False
            )

        # 7. image_specific_aligned 按 label 着色（可选：当 npy 含 image_specific_aligned 时，用于对比 token 级对齐效果）
        if is_aligned is not None and labels_mask is not None and len(is_aligned) == len(labels_mask):
            is_aligned_valid = is_aligned[labels_mask]
            tsne_plot_by_label(
                is_aligned_valid,
                labels_flat,
                "t-SNE: image_specific_aligned (colored by sentiment labels)",
                "ddm_tsne_image_specific_aligned_by_label.png",
                max_samples=2000,
                sentiment_only=False,
                sentiment_bigger=True
            )
            tsne_plot_by_label(
                is_aligned_valid,
                labels_flat,
                "t-SNE: image_specific_aligned (sentiment only: NEG/NEU/POS)",
                "ddm_tsne_image_specific_aligned_by_label_sentiment_only.png",
                max_samples=2000,
                sentiment_only=True,
                sentiment_bigger=False
            )
    else:
        print("[DDM VIS] 警告：未找到标签数据，跳过按label着色的可视化")

    print("[DDM VIS] 特征可视化完成。")


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(description="DDM 特征 t-SNE 可视化，支持输出目录带模型名/时间轴便于对比")
    parser.add_argument("--features", default="ddm_features_sample.npy", help="特征 npy 文件路径")
    parser.add_argument("--output_dir", default="ddm_vis", help="输出目录前缀，默认 ddm_vis")
    parser.add_argument("--output_suffix", default=None, help="输出目录后缀（模型名或 run_id），实际目录为 output_dir_output_suffix，便于保留历史对比")
    parser.add_argument("--timestamp", action="store_true", help="使用当前时间作为 output_suffix（格式 YYYYMMDD_HHMMSS）")
    args = parser.parse_args()
    suffix = args.output_suffix
    if args.timestamp:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualize_ddm_features(
        feat_path=args.features,
        output_dir=args.output_dir,
        output_suffix=suffix
    )

