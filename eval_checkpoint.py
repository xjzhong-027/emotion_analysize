"""
eval_checkpoint.py：仅评估已保存的 best_model.pth，不训练。

用法示例（在 ICT-main 目录下）：
    python eval_checkpoint.py --dataset_type 2017 --checkpoint_path checkpoints/2017/best_20260130_052049/best_model.pth --use_ddm --use_crm

注意：--use_ddm / --use_crm / --use_cfm / --use_sent_mod / --use_sent_mod_fuse_b 等须与训练该 checkpoint 时一致。
"""

import argparse
import os
import random

import numpy as np
import torch
from transformers import (
    AutoConfig,
    TrainingArguments,
    Trainer,
    EvalPrediction,
    AutoModel,
    BertForTokenClassification,
    RobertaForTokenClassification,
    AlbertForTokenClassification,
    ElectraForTokenClassification,
    ViTForImageClassification,
    SwinForImageClassification,
    DeiTModel,
    ConvNextForImageClassification,
    ResNetModel,
    CLIPModel,
)

from model import ICTModel
from utils.MyDataSet import MyDataSet2
from utils.metrics import cal_f1


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_compute_metrics_fn(text_inputs, pairs):
    """仅评估用：根据 (text_logits, cross_logits) 算 multi / text F1，返回单次指标 dict。"""

    def compute_metrics_fn(p: EvalPrediction):
        if isinstance(p.predictions, dict):
            text_logits = p.predictions["logits"]
            cross_logits = p.predictions["cross_logits"]
        else:
            text_logits, cross_logits = p.predictions
        pred_labels = np.argmax(cross_logits, -1)
        text_pred_labels = np.argmax(text_logits, -1)
        precision, recall, f1 = cal_f1(pred_labels, text_inputs, pairs)
        text_precision, text_recall, text_f1 = cal_f1(text_pred_labels, text_inputs, pairs)
        return {"precision": precision, "recall": recall, "f1": f1, "text_f1": text_f1}
    return compute_metrics_fn


class ICTTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.get("loss")
        if prediction_loss_only:
            return (loss, None, None)
        if isinstance(outputs, dict) and "logits" in outputs and "cross_logits" in outputs:
            logits = (outputs["logits"], outputs["cross_logits"])
        else:
            logits = outputs.get("logits")
        return (loss, logits, inputs.get("labels"))


def main():
    parser = argparse.ArgumentParser(description="仅评估 checkpoint，不训练")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="best_model.pth 路径")
    parser.add_argument("--dataset_type", type=str, default="2017")
    parser.add_argument("--task_name", type=str, default="dualc")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--text_model_name", type=str, default="roberta")
    parser.add_argument("--image_model_name", type=str, default="vit")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--random_seed", type=int, default=2022)

    parser.add_argument("--use_ddm", action="store_true")
    parser.add_argument("--use_sentiment_stream", action="store_true", help="情感双流替代 DDM，与 --use_ddm 互斥")
    parser.add_argument("--ddm_mode", type=str, default="text_shared", choices=["text_shared", "image_shared", "all"])
    parser.add_argument("--no_image_token_align", action="store_true")
    parser.add_argument("--lambda1", type=float, default=1.5)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--use_cfm", action="store_true")
    parser.add_argument("--lambda3", type=float, default=0.3)
    parser.add_argument("--lambda4", type=float, default=0.1)
    parser.add_argument("--lambda_sent_cls", type=float, default=0.0)
    parser.add_argument("--lambda_sent_aux", type=float, default=0.0)
    parser.add_argument("--use_crm", action="store_true")
    parser.add_argument("--lambda_crm", type=float, default=0.1)
    parser.add_argument("--crm_num_prototypes", type=int, default=5)
    parser.add_argument("--use_sent_mod", action="store_true")
    parser.add_argument("--lambda_sent_mod", type=float, default=0.1)
    parser.add_argument("--use_sent_mod_fuse", action="store_true")
    parser.add_argument("--use_sent_mod_fuse_b", action="store_true")
    parser.add_argument("--use_legacy_multi", action="store_true", help="旧版 multi：无 multi_fuse_proj，multi=cross_attention 直进 classifier0（0130 等）")
    parser.add_argument("--sent_mod_num_prototypes", type=int, default=5)
    parser.add_argument("--sent_mod_fuse_text_weight", type=float, default=0.5)

    args = parser.parse_args()
    if args.use_sentiment_stream and args.use_ddm:
        raise ValueError("use_sentiment_stream 与 use_ddm 互斥，请只启用其一")
    if args.use_sent_mod_fuse and args.use_sent_mod_fuse_b:
        raise ValueError("方案A和方案B不能同时启用")
    if args.use_sent_mod_fuse_b and not args.use_sent_mod:
        raise ValueError("方案B需先启用 --use_sent_mod")

    set_random_seed(args.random_seed)

    # 若 checkpoint 无 multi_fuse_proj（如 0130），自动启用 use_legacy_multi，否则 multi 分支会经未训练投影导致异常
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    use_legacy_multi = args.use_legacy_multi or (
        args.use_ddm and "multi_fuse_proj.weight" not in ckpt
    )
    if use_legacy_multi and not args.use_legacy_multi:
        print("[eval_checkpoint] 检测到 checkpoint 无 multi_fuse_proj（如 0130），已启用 use_legacy_multi")

    # 数据（与 main.py 一致）
    data_input_file = os.path.join("datasets/finetune", args.task_name, args.dataset_type, "input.pt")
    data_inputs = torch.load(data_input_file, weights_only=False)
    for split in ("train", "dev", "test"):
        data_inputs[split].pop("pairs", None)
    train_dataset = MyDataSet2(inputs=data_inputs["train"], dataset_type="train", data=args.dataset_type)
    test_dataset = MyDataSet2(inputs=data_inputs["test"], dataset_type="test", data=args.dataset_type)
    # 恢复 test pairs 供 cal_f1
    data_inputs_full = torch.load(data_input_file, weights_only=False)
    test_pairs = data_inputs_full["test"]["pairs"]

    # 文本 / 图像预训练：本地路径加载，加 local_files_only=True 避免新版 huggingface_hub 把路径当 repo_id 校验
    _local = {"local_files_only": True}
    if args.text_model_name == "bert":
        model_path1 = "/root/data/weights/bert-large"
        config1 = AutoConfig.from_pretrained(model_path1, **_local)
        text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1, **_local).state_dict()
    elif args.text_model_name == "roberta":
        model_path1 = "/root/data/weights/roberta"
        config1 = AutoConfig.from_pretrained(model_path1, **_local)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1, **_local).state_dict()
    elif args.text_model_name == "roberta-large":
        model_path1 = "/root/data/weights/roberta-large"
        config1 = AutoConfig.from_pretrained(model_path1, **_local)
        text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1, **_local).state_dict()
    elif args.text_model_name == "albert":
        model_path1 = "../weights/albert-base-v2"
        config1 = AutoConfig.from_pretrained(model_path1, **_local)
        text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1, **_local).state_dict()
    elif args.text_model_name == "electra":
        model_path1 = "../weights/electra-base-discriminator"
        config1 = AutoConfig.from_pretrained(model_path1, **_local)
        text_pretrained_dict = ElectraForTokenClassification.from_pretrained(model_path1, **_local).state_dict()
    elif args.text_model_name in ("deberta", "deberta-large"):
        model_path1 = "/root/data/weights/deberta-large"
        config1 = AutoConfig.from_pretrained(model_path1, **_local)
        text_pretrained_dict = AutoModel.from_pretrained(model_path1, **_local).state_dict()
    else:
        raise ValueError(f"不支持的 text_model_name: {args.text_model_name}")

    if args.image_model_name in ("vit", "vit-large"):
        model_path2 = "/root/data/weights/vit-large-patch16-224"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2, **_local).state_dict()
    elif args.image_model_name == "swin":
        model_path2 = "/root/data/weights/swin"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2, **_local).state_dict()
    elif args.image_model_name == "swin-large":
        model_path2 = "/root/data/weights/swin-large"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2, **_local).state_dict()
    elif args.image_model_name == "deit":
        model_path2 = "../weights/deit-base-patch16-224"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = DeiTModel.from_pretrained(model_path2, **_local).state_dict()
    elif args.image_model_name == "convnext":
        model_path2 = "../weights/convnext-tiny-224"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = ConvNextForImageClassification.from_pretrained(model_path2, **_local).state_dict()
    elif args.image_model_name == "resnet":
        model_path2 = "../weights/resnet-50"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = ResNetModel.from_pretrained(model_path2, **_local).state_dict()
    elif args.image_model_name == "clip-large":
        model_path2 = "/root/data/weights/clip-large"
        config2 = AutoConfig.from_pretrained(model_path2, **_local)
        image_pretrained_dict = CLIPModel.from_pretrained(model_path2, **_local).state_dict()
    else:
        raise ValueError(f"不支持的 image_model_name: {args.image_model_name}")

    model = ICTModel(
        config1,
        config2,
        text_num_labels=5,
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        alpha=args.alpha,
        beta=args.beta,
        use_ddm=args.use_ddm,
        ddm_mode=args.ddm_mode,
        use_image_token_align=not args.no_image_token_align,
        ddm_debug=False,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        use_cfm=args.use_cfm,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        cfm_debug=False,
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
        sent_mod_fuse_text_weight=args.sent_mod_fuse_text_weight,
        use_legacy_multi=use_legacy_multi,
        use_sentiment_stream=args.use_sentiment_stream,
    )
    vb_model_dict = model.state_dict()
    for k, v in image_pretrained_dict.items():
        if vb_model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            vb_model_dict[k] = v
    for k, v in text_pretrained_dict.items():
        if vb_model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            vb_model_dict[k] = v
    model.load_state_dict(vb_model_dict)

    if not os.path.isfile(args.checkpoint_path):
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint_path}")
    ckpt = {k: v.to(next(model.parameters()).device) for k, v in ckpt.items()}
    load_ret = model.load_state_dict(ckpt, strict=False)
    if load_ret.missing_keys or load_ret.unexpected_keys:
        print("[eval_checkpoint] load_state_dict 非严格:", "missing:", load_ret.missing_keys, "unexpected:", load_ret.unexpected_keys)

    training_args = TrainingArguments(
        output_dir=os.path.join(".", "eval_out_temp"),
        eval_strategy="no",
        per_device_eval_batch_size=args.batch_size,
        label_names=["labels", "cross_labels"],
    )
    trainer = ICTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=build_compute_metrics_fn(data_inputs_full["test"], test_pairs),
    )

    eval_out = trainer.evaluate()
    print("Eval 结果:", eval_out)

    pred_out = trainer.predict(test_dataset)
    if isinstance(pred_out.predictions, tuple):
        # predictions = (text_logits, cross_logits)，即 (text 分支, multi 分支)
        text_prec, text_rec, text_f1 = cal_f1(np.argmax(pred_out.predictions[0], -1), data_inputs_full["test"], test_pairs)
        multi_prec, multi_rec, multi_f1 = cal_f1(np.argmax(pred_out.predictions[1], -1), data_inputs_full["test"], test_pairs)
        print("multi: precision={}, recall={}, f1={}".format(multi_prec, multi_rec, multi_f1))
        print("text:  precision={}, recall={}, f1={}".format(text_prec, text_rec, text_f1))
    else:
        print("predictions 格式:", type(pred_out.predictions))


if __name__ == "__main__":
    main()
