import os.path
import torch
import numpy as np
import os
import argparse
import random
from datetime import datetime
from model import ICTModel
from utils.MyDataSet import MyDataSet2
from utils.metrics import cal_f1
from typing import Callable, Dict
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import DebertaForTokenClassification
from transformers import SwinForImageClassification
from transformers import AutoConfig, TrainingArguments, Trainer, EvalPrediction, CLIPModel
from transformers import BertForTokenClassification, RobertaForTokenClassification, AlbertForTokenClassification, ViTForImageClassification, SwinForImageClassification, DeiTModel, ConvNextForImageClassification, ResNetModel, ElectraForTokenClassification

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default='2015', nargs='?', help='display a string')
parser.add_argument('--task_name', type=str, default='dualc', nargs='?', help='display a string')
parser.add_argument('--batch_size', type=int, default=16, nargs='?', help='display an integer')
parser.add_argument('--output_result_file', type=str, default="./result.txt", nargs='?', help='display a string')
parser.add_argument('--output_dir', type=str, default="./results", nargs='?', help='display a string')
parser.add_argument('--lr', type=float, default=2e-5, nargs='?', help='display a float')
parser.add_argument('--epochs', type=int, default=50, nargs='?', help='display an integer')
parser.add_argument('--alpha', type=float, default=0.6, nargs='?', help='display a float')
parser.add_argument('--beta', type=float, default=0.6, nargs='?', help='display a float')
parser.add_argument('--text_model_name', type=str, default="roberta", nargs='?')    # roberta
parser.add_argument('--image_model_name', type=str, default="vit", nargs='?')
parser.add_argument('--random_seed', type=int, default=2022, nargs='?')
parser.add_argument('--use_ddm', action='store_true', help='Use Dual-level Disentanglement Module')
parser.add_argument('--use_sentiment_stream', action='store_true', help='Use sentiment-relevant/irrelevant two streams (replaces DDM), per-token query image')
parser.add_argument(
    '--ddm_mode',
    type=str,
    default='text_shared',
    choices=['text_shared', 'image_shared', 'all'],
    help='DDM feature fusion mode for ablation: '
         '"text_shared" (text_specific+shared), '
         '"image_shared" (image_specific+shared), '
         '"all" (text_specific+image_specific+shared).'
)
parser.add_argument(
    '--ddm_debug',
    action='store_true',
    help='Enable DDM debug mode (log feature stats and export sample features)',
)
parser.add_argument(
    '--use_image_token_align',
    action='store_true',
    help='Enable image_specific token-level alignment in DDM (default: disabled)',
)
parser.add_argument(
    '--lambda1',
    type=float,
    default=1.5,
    help='Weight for orthogonal loss (default: 1.5)',
)
parser.add_argument(
    '--lambda2',
    type=float,
    default=0.5,
    help='Weight for enhanced triplet loss (default: 0.5)',
)
parser.add_argument('--use_cfm', action='store_true', help='Use CFM (Classification Fusion Mechanism) when use_ddm')
parser.add_argument('--lambda3', type=float, default=0.3, help='Weight for CFM branch consistency loss')
parser.add_argument('--lambda4', type=float, default=0.1, help='Weight for CFM fusion weight regularization')
parser.add_argument('--cfm_debug', action='store_true', help='Enable CFM debug mode (log weights and save samples)')
parser.add_argument('--lambda_sent_cls', type=float, default=0.0, help='Weight for sentiment classification loss on shared (core improvement 2.5)')
parser.add_argument('--lambda_sent_aux', type=float, default=0.0, help='Weight for sentiment center loss on shared (core improvement 2.5)')
parser.add_argument('--use_crm', action='store_true', help='Use CRM (prototype classification) when use_ddm, scheme A: auxiliary loss only')
parser.add_argument('--lambda_crm', type=float, default=0.1, help='Weight for CRM auxiliary loss')
parser.add_argument('--crm_num_prototypes', type=int, default=5, help='Number of prototypes per sentiment class in CRM')
parser.add_argument('--use_sent_mod', action='store_true', help='Use modality-internal sentiment prototypes (SentMod) when use_ddm')
parser.add_argument('--lambda_sent_mod', type=float, default=0.1, help='Weight for SentMod auxiliary loss')
parser.add_argument('--use_sent_mod_fuse', action='store_true', help='Fuse SentMod 3-class logits with cross_logits at aspect positions (way A)')
parser.add_argument('--use_sent_mod_fuse_b', action='store_true', help='方案B: concat sentiment features to representation layer, then classifier0')
parser.add_argument('--sent_mod_num_prototypes', type=int, default=5, help='Number of prototypes per sentiment class in SentMod')
parser.add_argument('--sent_mod_fuse_text_weight', type=float, default=0.5, help='Weight for text branch when fusing SentMod logits (default 0.5, image/shared share the rest; >0.33 emphasizes text)')

args = parser.parse_args()
if args.use_sentiment_stream and args.use_ddm:
    raise ValueError("use_sentiment_stream 与 use_ddm 互斥，请只启用其一")
if args.use_sent_mod_fuse and args.use_sent_mod_fuse_b:
    raise ValueError("方案A和方案B不能同时启用，请选择其一")
if args.use_sent_mod_fuse_b and not args.use_sent_mod:
    raise ValueError("方案B需要先启用 SentMod，请同时使用 --use_sent_mod")
dataset_type = args.dataset_type
task_name = args.task_name
alpha = args.alpha
beta = args.beta
batch_size = args.batch_size
output_dir = args.output_dir
lr = args.lr
epochs = args.epochs
text_model_name = args.text_model_name
image_model_name = args.image_model_name
output_result_file = args.output_result_file
random_seed = args.random_seed

def set_random_seed(random_seed):
    """Set random seed"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    torch.backends.cudnn.deterministic = True

def predict(p_dataset, p_inputs, p_pairs):
    outputs = trainer.predict(p_dataset)
    pred_labels = np.argmax(outputs.predictions[0], -1)
    return cal_f1(pred_labels,p_inputs,p_pairs)

def build_compute_metrics_fn(text_inputs,pairs) -> Callable[[EvalPrediction], Dict]:
    def compute_metrics_fn(p: EvalPrediction):
        # 兼容 dict（模型返回 {"logits", "cross_logits"}）或元组 (text_logits, cross_logits)
        if isinstance(p.predictions, dict):
            text_logits = p.predictions["logits"]
            cross_logits = p.predictions["cross_logits"]
        else:
            text_logits, cross_logits = p.predictions
        text_pred_labels = np.argmax(text_logits,-1)
        pred_labels = np.argmax(cross_logits,-1)
        precision, recall, f1 = cal_f1(pred_labels,text_inputs,pairs)
        text_precision, text_recall, text_f1 = cal_f1(text_pred_labels, text_inputs, pairs)
        if best_metric.get("f1") is not None:
            if f1 > best_metric["f1"]:
                best_metric["f1"] = f1
                best_metric["precision"] = precision
                best_metric["recall"] = recall
                # with open("my_model_result.txt", "w", encoding="utf-8") as f:
                #     f.write(str(pred_labels.tolist())+ '\n')
        else:
            best_metric["f1"] = f1
            best_metric["precision"] = precision
            best_metric["recall"] = recall
            # with open("my_model_result.txt", "w", encoding="utf-8") as f:
            #     f.write(str(pred_labels.tolist())+ '\n')
        if text_best_metric.get("f1") is not None:
            if text_f1 > text_best_metric["f1"]:
                text_best_metric["f1"] = text_f1
                text_best_metric["precision"] = text_precision
                text_best_metric["recall"] = text_recall
        else:
            text_best_metric["f1"] = text_f1
            text_best_metric["precision"] = text_precision
            text_best_metric["recall"] = text_recall
        return {"precision": precision,"recall":recall, "f1": f1}
    return compute_metrics_fn


# set random seed
set_random_seed(random_seed)

data_input_file = os.path.join("datasets/finetune", task_name, dataset_type, "input.pt")
if not os.path.exists(data_input_file):
    raise FileNotFoundError(f"Data file not found: {data_input_file}. Run TrainInputProcess first.")
# PyTorch 2.6 默认 weights_only=True 会导致包含 BatchEncoding 等对象的旧格式 input.pt 无法直接反序列化，
# 这里显式设置 weights_only=False，确保可以加载我们自己生成的预处理数据。
data_inputs = torch.load(data_input_file, weights_only=False)
train_word_ids = data_inputs["train"].word_ids
train_pairs = data_inputs["train"]["pairs"]
data_inputs["train"].pop("pairs")
train_dataset  = MyDataSet2(inputs=data_inputs["train"], dataset_type='train', data=args.dataset_type)

dev_word_ids = data_inputs["dev"].word_ids
dev_pairs = data_inputs["dev"]["pairs"]
data_inputs["dev"].pop("pairs")
dev_dataset  = MyDataSet2(inputs=data_inputs["dev"], dataset_type='dev', data=args.dataset_type)

test_word_ids = data_inputs["test"].word_ids
test_pairs = data_inputs["test"]["pairs"]
data_inputs["test"].pop("pairs")
test_dataset  = MyDataSet2(inputs=data_inputs["test"], dataset_type='test', data=args.dataset_type)


# text pretrained model selected
if text_model_name == 'bert':
    model_path1 = "/root/data/weights/bert-large"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = BertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'roberta':
    model_path1 = "/root/data/weights/roberta"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'roberta-large':
    model_path1 = "/root/data/weights/roberta-large"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'albert':
    model_path1 = "../weights/albert-base-v2"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'albertv3':
    model_path1 = "../weights/text_analyzer_albert-base-v3"
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AlbertForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'electra':
    model_path1 = '../weights/electra-base-discriminator'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = ElectraForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'twitter-roberta':
    model_path1 = '../weights/twitter-roberta-base-emotion'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = RobertaForTokenClassification.from_pretrained(model_path1).state_dict()
elif text_model_name == 'deberta-large':
    model_path1 = '/root/data/weights/deberta-large'
    config1 = AutoConfig.from_pretrained(model_path1)
    text_pretrained_dict = AutoModel.from_pretrained(model_path1).state_dict()
else:
    os.error("出错了")
    exit()

# image pretrained model selected
# 服务器实际只有 /root/data/weights/vit-large-patch16-224
# 无论传入 vit 还是 vit-large，都统一映射到这个本地目录，避免访问 /root/data/weights/vit[-large]
if image_model_name in ('vit', 'vit-large'):
    model_path2 = "/root/data/weights/vit-large-patch16-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ViTForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'swin':
    model_path2 = "/root/data/weights/swin"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'swin-large':
    model_path2 = "/root/data/weights/swin-large"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = SwinForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'deit':
    model_path2 = "../weights/deit-base-patch16-224"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = DeiTModel.from_pretrained(model_path2).state_dict()
elif image_model_name == 'convnext':
    model_path2 = '../weights/convnext-tiny-224'
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ConvNextForImageClassification.from_pretrained(model_path2).state_dict()
elif image_model_name == 'resnet':
    model_path2 = '../weights/resnet-50'
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = ResNetModel.from_pretrained(model_path2).state_dict()
elif image_model_name == 'clip-large':
    model_path2 = "/root/data/weights/clip-large"
    config2 = AutoConfig.from_pretrained(model_path2)
    image_pretrained_dict = CLIPModel.from_pretrained(model_path2).state_dict()
else:
    os.error("出错了")
    exit()

vb_model = ICTModel(
    config1,
    config2,
    text_num_labels=5,
    text_model_name=text_model_name,
    image_model_name=image_model_name,
    alpha=alpha,
    beta=beta,
    use_ddm=args.use_ddm,
    ddm_mode=args.ddm_mode,
    use_image_token_align=args.use_image_token_align,
    ddm_debug=args.ddm_debug,
    lambda1=args.lambda1,
    lambda2=args.lambda2,
    use_cfm=args.use_cfm,
    lambda3=args.lambda3,
    lambda4=args.lambda4,
    cfm_debug=args.cfm_debug,
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
    use_sentiment_stream=args.use_sentiment_stream,
)
vb_model_dict = vb_model.state_dict()

# load pretrained model weights
for k,v in image_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
for k,v in text_pretrained_dict.items():
    if vb_model_dict.get(k) is not None and k not in {'classifier.bias', 'classifier.weight'}:
        vb_model_dict[k] = v
vb_model.load_state_dict(vb_model_dict)

best_metric = dict()
text_best_metric = dict()

# 为本次训练创建时间戳，用于区分不同实验
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# 本次实验的输出目录（按时间轴区分）
run_output_dir = os.path.join(
    output_dir,
    f"{dataset_type}_{text_model_name}_{image_model_name}_{run_id}"
)

# 最优模型保存目录（按时间轴区分）
best_model_dir = os.path.join(
    "checkpoints",
    str(dataset_type),
    f"best_{run_id}"
)
os.makedirs(best_model_dir, exist_ok=True)

training_args = TrainingArguments(
    output_dir=run_output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_steps=10000,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    label_names=["labels","cross_labels"]
)

# 自定义 Trainer：模型返回 dict(logits, cross_logits)，prediction_step 返回 (logits, cross_logits) 供 compute_metrics 使用
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
        labels = inputs.get("labels")
        return (loss, logits, labels)

trainer = ICTTrainer(
    model=vb_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=build_compute_metrics_fn(
        text_inputs=data_inputs["test"],
        pairs=test_pairs),
)

trainer.train()

# 训练结束后，Trainer 会根据 metric_for_best_model 加载最优模型到 trainer.model
# 再将当前最优模型单独保存一份到按时间轴区分的目录
best_model_path = os.path.join(best_model_dir, "best_model.pth")
torch.save(trainer.model.state_dict(), best_model_path)
print(f"Best model saved to: {best_model_path}")

# save results（包含完整关键超参数与模块开关，便于复现实验）
with open(output_result_file, "a", encoding="utf-8") as f:
    model_para = dict()
    model_para["dataset_type"] = dataset_type
    model_para["text_model"] = text_model_name
    model_para["image_model"] = image_model_name
    model_para["batch_size"] = batch_size
    model_para["alpha"] = alpha
    model_para["beta"] = beta
    model_para["lr"] = lr
    model_para["epochs"] = epochs
    model_para["random_seed"] = random_seed
    # DDM / 对抗解耦 相关
    model_para["use_ddm"] = args.use_ddm
    model_para["ddm_mode"] = args.ddm_mode
    model_para["use_image_token_align"] = not args.no_image_token_align
    model_para["lambda1"] = args.lambda1
    model_para["lambda2"] = args.lambda2
    model_para["ddm_debug"] = args.ddm_debug
    model_para["use_cfm"] = args.use_cfm
    model_para["lambda3"] = args.lambda3
    model_para["lambda4"] = args.lambda4
    model_para["lambda_sent_cls"] = args.lambda_sent_cls
    model_para["lambda_sent_aux"] = args.lambda_sent_aux
    model_para["use_crm"] = args.use_crm
    model_para["lambda_crm"] = args.lambda_crm
    model_para["crm_num_prototypes"] = args.crm_num_prototypes
    model_para["use_sent_mod"] = args.use_sent_mod
    model_para["lambda_sent_mod"] = args.lambda_sent_mod
    model_para["use_sent_mod_fuse"] = args.use_sent_mod_fuse
    model_para["use_sent_mod_fuse_b"] = args.use_sent_mod_fuse_b
    model_para["sent_mod_num_prototypes"] = args.sent_mod_num_prototypes
    model_para["sent_mod_fuse_text_weight"] = args.sent_mod_fuse_text_weight
    model_para["use_sentiment_stream"] = args.use_sentiment_stream
    model_para["use_multi_read_fused"] = args.use_ddm or args.use_sentiment_stream

    f.write("参数: " + str(model_para) + "\n")
    f.write("multi: " + str(best_metric) + "\n")
    f.write("text: " + str(text_best_metric) + "\n")
    f.write("\n")