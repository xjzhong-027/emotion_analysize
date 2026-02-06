"""
SentMod 方面位置预测可视化验证脚本

不修改 main.py / model.py，通过加载训练好的 checkpoint 做前向推理，
用 forward hook 采集 SentMod 三支的 [B,L,3] logits，再计算方面位置准确率并画热力图。

使用步骤：
1. 训练完成后，在 ICT-main 目录下执行（参数需与训练时一致）：
   python visualize_sent_mod.py \
     --checkpoint_path checkpoints/2017/best_xxx/best_model.pth \
     --dataset_type 2017 \
     --split test \
     --text_model_name roberta-large \
     --image_model_name vit-large \
     --use_ddm --use_cfm --use_crm --use_sent_mod \
     --max_batches 100

2. 输出目录默认为 sent_mod_vis/，可用 --output_suffix 指定后缀（如 sent_mod_vis_2017）。

输出文件：
- sent_mod_aspect_accuracy.txt：方面位置准确率（text/image/shared/平均）
- sent_mod_aspect_accuracy_bar.png：三支准确率柱状图
- sent_mod_logits_heatmap_sample*.png：若干样本的 [位置, 3类] logits 热力图（三支各一子图）
"""

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    CLIPModel,
    ViTForImageClassification,
    SwinForImageClassification,
    DeiTModel,
    ConvNextForImageClassification,
    ResNetModel,
    BertForTokenClassification,
    RobertaForTokenClassification,
    AlbertForTokenClassification,
    ElectraForTokenClassification,
)

from model import ICTModel
from utils.MyDataSet import MyDataSet2


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_datasets(dataset_type: str, task_name: str, split: str) -> MyDataSet2:
    data_input_file = os.path.join("datasets/finetune", task_name, dataset_type, "input.pt")
    data_inputs = torch.load(data_input_file, weights_only=False)
    ds_inputs = data_inputs[split]
    ds_inputs.pop("pairs", None)
    return MyDataSet2(inputs=ds_inputs, dataset_type=split, data=dataset_type)


def build_model(
    text_model_name: str,
    image_model_name: str,
    alpha: float,
    beta: float,
    use_ddm: bool,
    ddm_mode: str,
    use_image_token_align: bool = True,
    lambda1: float = 1.5,
    lambda2: float = 0.5,
    use_cfm: bool = True,
    lambda3: float = 0.3,
    lambda4: float = 0.1,
    cfm_debug: bool = False,
    lambda_sent_cls: float = 0.3,
    lambda_sent_aux: float = 0.15,
    use_crm: bool = True,
    lambda_crm: float = 0.1,
    crm_num_prototypes: int = 5,
    use_sent_mod: bool = True,
    lambda_sent_mod: float = 0.1,
    use_sent_mod_fuse: bool = True,
    use_sent_mod_fuse_b: bool = False,
    sent_mod_num_prototypes: int = 5,
    use_legacy_plan_b: bool = False,
) -> ICTModel:
    if text_model_name == "roberta-large":
        model_path1 = "/root/data/weights/roberta-large"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
    elif text_model_name == "roberta":
        model_path1 = "/root/data/weights/roberta"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
    elif text_model_name == "bert":
        model_path1 = "/root/data/weights/bert-large"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
    else:
        raise ValueError(f"Unsupported text_model_name: {text_model_name}")

    if image_model_name in ("vit", "vit-large"):
        model_path2 = "/root/data/weights/vit-large-patch16-224"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
    elif image_model_name == "swin":
        model_path2 = "/root/data/weights/swin"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
    else:
        raise ValueError(f"Unsupported image_model_name: {image_model_name}")

    model = ICTModel(
        config1,
        config2,
        text_num_labels=5,
        text_model_name=text_model_name,
        image_model_name=image_model_name,
        alpha=alpha,
        beta=beta,
        use_ddm=use_ddm,
        ddm_mode=ddm_mode,
        use_image_token_align=use_image_token_align,
        ddm_debug=False,
        lambda1=lambda1,
        lambda2=lambda2,
        use_cfm=use_cfm,
        lambda3=lambda3,
        lambda4=lambda4,
        cfm_debug=cfm_debug,
        lambda_sent_cls=lambda_sent_cls,
        lambda_sent_aux=lambda_sent_aux,
        use_crm=use_crm,
        lambda_crm=lambda_crm,
        crm_num_prototypes=crm_num_prototypes,
        use_sent_mod=use_sent_mod,
        lambda_sent_mod=lambda_sent_mod,
        use_sent_mod_fuse=use_sent_mod_fuse,
        use_sent_mod_fuse_b=use_sent_mod_fuse_b,
        sent_mod_num_prototypes=sent_mod_num_prototypes,
        use_legacy_plan_b=use_legacy_plan_b,
    )

    model_dict = model.state_dict()
    for k, v in image_pretrained_dict.items():
        if model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            model_dict[k] = v
    for k, v in text_pretrained_dict.items():
        if model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            model_dict[k] = v
    model.load_state_dict(model_dict)
    return model


def collect_sent_mod_outputs(
    model: ICTModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 forward hook 采集 SentMod 三支 logits 与 labels。"""
    hook_outputs: List[torch.Tensor] = []

    def hook(_module, _input, output: torch.Tensor):
        hook_outputs.append(output.detach())

    if not getattr(model, "use_sent_mod", False) or model.sent_mod is None:
        raise RuntimeError("模型未启用 SentMod，请使用 --use_sent_mod 训练的 checkpoint。")

    handle = model.sent_mod.register_forward_hook(hook)

    all_text = []
    all_image = []
    all_shared = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if bi >= max_batches:
                break
            hook_outputs.clear()
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _ = model(**batch)
            if len(hook_outputs) != 3:
                continue
            all_text.append(hook_outputs[0].cpu().numpy())
            all_image.append(hook_outputs[1].cpu().numpy())
            all_shared.append(hook_outputs[2].cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    handle.remove()

    text_logits = np.concatenate(all_text, axis=0)
    image_logits = np.concatenate(all_image, axis=0)
    shared_logits = np.concatenate(all_shared, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return text_logits, image_logits, shared_logits, labels


def compute_aspect_accuracy(
    text_logits: np.ndarray,
    image_logits: np.ndarray,
    shared_logits: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """仅对方面位置 (labels in {2,3,4}) 计算各支预测准确率。"""
    mask = (labels >= 2) & (labels <= 4)
    if mask.sum() == 0:
        return {"text": 0.0, "image": 0.0, "shared": 0.0, "avg": 0.0, "n_aspect": 0}

    gold = (labels - 2).astype(np.int64)
    pred_text = np.argmax(text_logits, axis=-1)
    pred_image = np.argmax(image_logits, axis=-1)
    pred_shared = np.argmax(shared_logits, axis=-1)

    gold_flat = gold[mask]
    acc_text = (pred_text[mask] == gold_flat).mean()
    acc_image = (pred_image[mask] == gold_flat).mean()
    acc_shared = (pred_shared[mask] == gold_flat).mean()
    acc_avg = (acc_text + acc_image + acc_shared) / 3.0

    return {
        "text": float(acc_text),
        "image": float(acc_image),
        "shared": float(acc_shared),
        "avg": float(acc_avg),
        "n_aspect": int(mask.sum()),
    }


def plot_accuracy_bar(stats: dict, save_path: str) -> None:
    import matplotlib.pyplot as plt

    names = ["Text", "Image", "Shared", "Avg"]
    vals = [stats["text"], stats["image"], stats["shared"], stats["avg"]]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals, color=colors, alpha=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_title("SentMod Aspect-Position Accuracy (labels in {2,3,4})")
    ax.set_ylim(0, 1.05)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {save_path}")


def plot_logits_heatmap(
    text_logits: np.ndarray,
    image_logits: np.ndarray,
    shared_logits: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    num_samples: int = 5,
) -> None:
    """对若干样本画 [位置, 3类] logits 热力图，标出方面位置。"""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    N, L, _ = text_logits.shape
    indices = np.random.choice(N, min(num_samples, N), replace=False) if N > num_samples else np.arange(N)

    for idx in indices:
        t_log = text_logits[idx]
        i_log = image_logits[idx]
        s_log = shared_logits[idx]
        lab = labels[idx]
        aspect_pos = np.where((lab >= 2) & (lab <= 4))[0]

        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        for ax, data, name in zip(
            axes,
            [t_log, i_log, s_log],
            ["Text", "Image", "Shared"],
        ):
            im = ax.imshow(data.T, aspect="auto", cmap="RdYlBu_r", vmin=-2, vmax=2)
            ax.set_ylabel(name)
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels(["NEG", "NEU", "POS"])
            for p in aspect_pos:
                ax.axvline(x=p, color="black", linewidth=0.8, alpha=0.7)
        axes[-1].set_xlabel("Position")
        fig.suptitle(f"Sample {idx}: SentMod logits (black line = aspect position)")
        fig.tight_layout(rect=[0, 0, 0.92, 0.96])
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(os.path.join(output_dir, f"sent_mod_logits_heatmap_sample{idx}.png"), dpi=150, bbox_inches="tight")
        plt.close()
    print(f"已保存 {len(indices)} 张热力图到 {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SentMod 方面位置可视化：加载 checkpoint，hook 采集 logits，算准确率并画图")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="best_model.pth 路径")
    parser.add_argument("--dataset_type", type=str, default="2017")
    parser.add_argument("--task_name", type=str, default="dualc")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=100, help="最多跑多少个 batch 用于统计")
    parser.add_argument("--text_model_name", type=str, default="roberta-large")
    parser.add_argument("--image_model_name", type=str, default="vit-large")
    parser.add_argument("--alpha", type=float, default=0.635)
    parser.add_argument("--beta", type=float, default=0.565)
    parser.add_argument("--use_ddm", action="store_true", default=True)
    parser.add_argument("--ddm_mode", type=str, default="text_shared")
    parser.add_argument("--no_image_token_align", action="store_true")
    parser.add_argument("--lambda1", type=float, default=1.5)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--use_cfm", action="store_true", default=True)
    parser.add_argument("--lambda3", type=float, default=0.3)
    parser.add_argument("--lambda4", type=float, default=0.1)
    parser.add_argument("--lambda_sent_cls", type=float, default=0.3)
    parser.add_argument("--lambda_sent_aux", type=float, default=0.15)
    parser.add_argument("--use_crm", action="store_true", default=True)
    parser.add_argument("--lambda_crm", type=float, default=0.1)
    parser.add_argument("--crm_num_prototypes", type=int, default=5)
    parser.add_argument("--use_sent_mod", action="store_true", default=True)
    parser.add_argument("--lambda_sent_mod", type=float, default=0.1)
    parser.add_argument("--use_sent_mod_fuse", action="store_true", default=True)
    parser.add_argument("--use_sent_mod_fuse_b", action="store_true", default=False, help="方案B: 表示层拼接情感特征，加载方案B checkpoint 时需加")
    parser.add_argument("--use_legacy_plan_b", action="store_true", default=False, help="旧版方案B: multi=cross+sent(H+3)，无 multi_fuse_proj；与阶段目标文档一致，加载旧 checkpoint 时可用")
    parser.add_argument("--sent_mod_num_prototypes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="sent_mod_vis")
    parser.add_argument("--output_suffix", type=str, default=None, help="输出目录后缀，如 2017")
    parser.add_argument("--num_heatmap_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    set_random_seed(args.seed)
    if args.output_suffix:
        args.output_dir = f"{args.output_dir}_{args.output_suffix}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_datasets(args.dataset_type, args.task_name, args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    # 根据 checkpoint 自动判断是否旧版方案B（classifier0 输入 H+3、无 multi_fuse_proj），与阶段目标文档一致
    use_legacy_plan_b = args.use_legacy_plan_b
    if not use_legacy_plan_b and args.use_sent_mod_fuse_b:
        if "multi_fuse_proj.weight" not in state_dict and "classifier0.fc1.weight" in state_dict:
            h = state_dict["classifier0.fc1.weight"].shape[1]
            hidden = 1024 if args.text_model_name == "roberta-large" else 768
            if h == hidden + 3:
                use_legacy_plan_b = True
                print("[SentMod Vis] 检测到旧版方案B checkpoint（classifier0 输入 H+3、无 multi_fuse_proj），已启用 use_legacy_plan_b")

    model = build_model(
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        alpha=args.alpha,
        beta=args.beta,
        use_ddm=args.use_ddm,
        ddm_mode=args.ddm_mode,
        use_image_token_align=not args.no_image_token_align,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        use_cfm=args.use_cfm,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        lambda_sent_cls=args.lambda_sent_cls,
        lambda_sent_aux=args.lambda_sent_aux,
        use_crm=args.use_crm,
        lambda_crm=args.lambda_crm,
        crm_num_prototypes=args.crm_num_prototypes,
        use_sent_mod=args.use_sent_mod,
        lambda_sent_mod=args.lambda_sent_mod,
        use_sent_mod_fuse=args.use_sent_mod_fuse,
        use_sent_mod_fuse_b=args.use_sent_mod_fuse_b,
        sent_mod_num_prototypes=args.sent_mod_num_prototypes,
        use_legacy_plan_b=use_legacy_plan_b,
    )
    load_ret = model.load_state_dict(state_dict, strict=False)
    if load_ret.missing_keys or load_ret.unexpected_keys:
        print(f"[SentMod Vis] 部分权重未加载: missing={load_ret.missing_keys[:5]}{'...' if len(load_ret.missing_keys) > 5 else ''}, unexpected={load_ret.unexpected_keys[:5]}{'...' if len(load_ret.unexpected_keys) > 5 else ''}")
    model.to(device)
    model.eval()

    print("[SentMod Vis] 开始采集 logits ...")
    text_logits, image_logits, shared_logits, labels = collect_sent_mod_outputs(
        model, dataloader, device, args.max_batches
    )
    print(f"[SentMod Vis] 采集到 shape: text={text_logits.shape}, labels={labels.shape}")

    stats = compute_aspect_accuracy(text_logits, image_logits, shared_logits, labels)
    txt_path = os.path.join(args.output_dir, "sent_mod_aspect_accuracy.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"方面位置数: {stats['n_aspect']}\n")
        f.write(f"Text 准确率:  {stats['text']:.4f}\n")
        f.write(f"Image 准确率: {stats['image']:.4f}\n")
        f.write(f"Shared 准确率: {stats['shared']:.4f}\n")
        f.write(f"平均准确率:   {stats['avg']:.4f}\n")
    print(f"已保存: {txt_path}")
    print(stats)

    plot_accuracy_bar(stats, os.path.join(args.output_dir, "sent_mod_aspect_accuracy_bar.png"))
    plot_logits_heatmap(
        text_logits, image_logits, shared_logits, labels,
        args.output_dir,
        num_samples=args.num_heatmap_samples,
    )
    print("[SentMod Vis] 完成。")


if __name__ == "__main__":
    main()
