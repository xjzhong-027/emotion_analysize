"""
probe_checkpoint.py

作用：
    1. 在服务器上加载训练好的 best_model.pth（state_dict）
    2. 重新构建 ICTModel，并打开 DDM / CFM 的 debug 开关
    3. 在 dev 或 test 集上跑若干 batch 的前向推理
       - DDM 会导出 ddm_features_sample.npy
       - CFM 会导出 cfm_fusion_weights_sample.npy
    4. 然后你可以把这两个 npy 文件拷回本地，用现有的
       visualize_ddm_features.py / visualize_cfm_weights.py 做可视化

使用示例（在服务器上）：

    cd ~/data/ICT-main-mdre/ICT-main
    python probe_checkpoint.py \
      --dataset_type 2017 \
      --split test \
      --checkpoint_path checkpoints/2017/best_20260129_144000/best_model.pth \
      --text_model_name roberta-large \
      --image_model_name vit-large \
      --batch_size 16 \
      --max_batches 200

注意：
    - 本脚本不会进行训练，只前向推理几百个 batch，主要目的是触发
      DDM / CFM 内部的 npy 导出逻辑。
    - 跑完后，当前目录下会出现（如果还没有的话）：
        ddm_features_sample.npy
        cfm_fusion_weights_sample.npy
"""

import argparse
import os
import random
from typing import Tuple

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
    """和 main.py 保持一致的随机种子设置。"""
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.deterministic = True


def build_datasets(
    dataset_type: str,
    task_name: str,
    split: str,
) -> Tuple[MyDataSet2, dict]:
    """
    加载预处理好的 input.pt，并构建指定 split 的数据集。

    Args:
        dataset_type: "2015" 或 "2017"
        task_name: 通常是 "dualc"
        split: "train" / "dev" / "test"
    """
    assert split in {"train", "dev", "test"}
    data_input_file = os.path.join("datasets/finetune", task_name, dataset_type, "input.pt")

    # 与 main.py 一致，显式 weights_only=False
    data_inputs = torch.load(data_input_file, weights_only=False)

    ds_inputs = data_inputs[split]
    pairs = ds_inputs["pairs"]
    ds_inputs.pop("pairs")

    dataset = MyDataSet2(inputs=ds_inputs, dataset_type=split, data=dataset_type)
    return dataset, data_inputs


def build_model(
    text_model_name: str,
    image_model_name: str,
    alpha: float,
    beta: float,
    use_ddm: bool,
    ddm_mode: str,
    use_image_token_align: bool = True,
    lambda1: float = 0.5,
    lambda2: float = 0.0,
    use_cfm: bool = True,
    lambda3: float = 0.3,
    lambda4: float = 0.05,
    lambda_sent_cls: float = 0.3,
    lambda_sent_aux: float = 0.15,
    use_crm: bool = False,
    lambda_crm: float = 0.1,
    crm_num_prototypes: int = 5,
) -> ICTModel:
    """
    根据 main.py 中的逻辑构建 ICTModel，并加载文本 / 图像预训练权重。
    """
    # 文本 encoder 配置与预训练权重
    if text_model_name == "bert":
        model_path1 = "/root/data/weights/bert-large"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
    elif text_model_name == "roberta":
        model_path1 = "/root/data/weights/roberta"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
    elif text_model_name == "roberta-large":
        model_path1 = "/root/data/weights/roberta-large"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
    elif text_model_name == "albert":
        model_path1 = "../weights/albert-base-v2"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
    elif text_model_name == "deberta" or text_model_name == "deberta-large":
        model_path1 = "/root/data/weights/deberta-large"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = AutoModel.from_pretrained(model_path1).state_dict()
    elif text_model_name == "electra":
        model_path1 = "../weights/electra-base-discriminator"
        config1 = AutoConfig.from_pretrained(model_path1)
        text_pretrained_dict = ElectraForTokenClassification.from_pretrained(model_path1).state_dict()
    else:
        raise ValueError(f"Unsupported text_model_name: {text_model_name}")

    # 图像 encoder 配置与预训练权重（与 main.py 中的映射保持一致）
    if image_model_name in ("vit", "vit-large"):
        model_path2 = "/root/data/weights/vit-large-patch16-224"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
    elif image_model_name == "swin":
        model_path2 = "/root/data/weights/swin"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
    elif image_model_name == "swin-large":
        model_path2 = "/root/data/weights/swin-large"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
    elif image_model_name == "deit":
        model_path2 = "../weights/deit-base-patch16-224"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
    elif image_model_name == "convnext":
        model_path2 = "../weights/convnext-tiny-224"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ConvNextForImageClassification.from_pretrained(model_path2).state_dict()
    elif image_model_name == "resnet":
        model_path2 = "../weights/resnet-50"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = ResNetModel.from_pretrained(model_path2).state_dict()
    elif image_model_name == "clip-large":
        model_path2 = "/root/data/weights/clip-large"
        config2 = AutoConfig.from_pretrained(model_path2)
        image_pretrained_dict = CLIPModel.from_pretrained(model_path2).state_dict()
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
        ddm_debug=True,   # 强制打开 DDM 调试
        lambda1=lambda1,
        lambda2=lambda2,
        use_cfm=use_cfm,
        lambda3=lambda3,
        lambda4=lambda4,
        cfm_debug=True,   # 强制打开 CFM 调试
        lambda_sent_cls=lambda_sent_cls,
        lambda_sent_aux=lambda_sent_aux,
        use_crm=use_crm,
        lambda_crm=lambda_crm,
        crm_num_prototypes=crm_num_prototypes,
    )

    # 将 text / image 预训练权重加载到 model（与 main.py 相同逻辑）
    model_dict = model.state_dict()
    for k, v in image_pretrained_dict.items():
        if model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            model_dict[k] = v
    for k, v in text_pretrained_dict.items():
        if model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            model_dict[k] = v
    model.load_state_dict(model_dict)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="2017", help="2015 or 2017")
    parser.add_argument("--task_name", type=str, default="dualc")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["dev", "test"],
        help="使用哪个 split 做 probe（推荐 test）",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="best_model.pth 的路径，例如 checkpoints/2017/best_xxx/best_model.pth",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=200, help="最多前向多少个 batch，用于触发 debug 保存")
    parser.add_argument("--text_model_name", type=str, default="roberta-large")
    parser.add_argument("--image_model_name", type=str, default="vit-large")
    parser.add_argument("--alpha", type=float, default=0.635)
    parser.add_argument("--beta", type=float, default=0.565)
    parser.add_argument("--lambda1", type=float, default=0.5)
    parser.add_argument("--lambda2", type=float, default=0.0)
    parser.add_argument("--lambda3", type=float, default=0.3)
    parser.add_argument("--lambda4", type=float, default=0.05)
    parser.add_argument("--lambda_sent_cls", type=float, default=0.3)
    parser.add_argument("--lambda_sent_aux", type=float, default=0.15)
    parser.add_argument("--use_crm", action="store_true", help="与训练时 --use_crm 一致时使用")
    parser.add_argument("--lambda_crm", type=float, default=0.1)
    parser.add_argument("--crm_num_prototypes", type=int, default=5)
    parser.add_argument("--no_image_token_align", action="store_true", help="与训练时 --no_image_token_align 一致时使用")
    parser.add_argument("--seed", type=int, default=2022)
    args = parser.parse_args()

    set_random_seed(args.seed)

    print(f"[Probe] dataset={args.dataset_type}, split={args.split}")
    print(f"[Probe] checkpoint={args.checkpoint_path}")

    # 1. 构建数据集
    dataset, _ = build_datasets(args.dataset_type, args.task_name, args.split)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 2. 构建模型并加载 best checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        alpha=args.alpha,
        beta=args.beta,
        use_ddm=True,
        ddm_mode="text_shared",
        use_image_token_align=not args.no_image_token_align,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        use_cfm=True,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        lambda_sent_cls=args.lambda_sent_cls,
        lambda_sent_aux=args.lambda_sent_aux,
        use_crm=args.use_crm,
        lambda_crm=args.lambda_crm,
        crm_num_prototypes=args.crm_num_prototypes,
    )

    state_dict = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print("[Probe] 模型与 checkpoint 已加载，开始前向推理以导出 DDM / CFM 特征 ...")

    batches_run = 0
    with torch.no_grad():
        for batch in dataloader:
            # 将 batch 移到同一 device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _ = model(**batch)
            batches_run += 1

            # 如果内部已经保存过样本，可以提前停止
            saved_ddm = getattr(model, "_saved_ddm_feats", False)
            saved_cfm = getattr(model, "_saved_cfm_weights", False)
            if saved_ddm and saved_cfm:
                print(f"[Probe] DDM / CFM 样本均已保存，提前结束（batches_run={batches_run}）")
                break

            if batches_run >= args.max_batches:
                print(f"[Probe] 达到 max_batches={args.max_batches} 上限，停止。")
                break

    print("[Probe] 结束。若一切正常，当前目录下应出现 / 更新：")
    print("    ddm_features_sample.npy")
    print("    cfm_fusion_weights_sample.npy")


if __name__ == "__main__":
    main()

