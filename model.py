# model.py
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
from transformers import CLIPVisionModel
from transformers import DebertaModel
from transformers import RobertaModel, BertModel, AlbertModel, ElectraModel, ViTModel, SwinModel, DeiTModel, ConvNextModel
from new_model import InterlanceDecoder
from timm.models.vision_transformer import Mlp
from ddm_module import DDM
from cfm_module import CFM
from crm_module import CRM
from sent_mod_module import SentModModule
from sentiment_stream_module import SentimentStreamModule
from mdre_components import (
    orthogonal_loss_for_sequence,
)
from entropy_utils import (
    compute_entropy,
    compute_attention_entropy,
    compute_cross_attention_entropy,
    compute_entropy_increase_loss,
)

class ICTModel(nn.Module):
    def __init__(
        self,
        config1,
        config2,
        text_num_labels,
        alpha,
        beta,
        text_model_name: str = "roberta",
        image_model_name: str = "vit",
        use_ddm: bool = False,
        use_asymmetric_dims: bool = False,
        ddm_mode: str = "text_shared",
        use_image_token_align: bool = True,
        ddm_debug: bool = False,
        ddm_log_path: str = "ddm_debug.log",
        ddm_feat_path: str = "ddm_features_sample.npy",
        lambda1: float = 1.5,
        use_cfm: bool = False,
        lambda3: float = 0.3,
        lambda4: float = 0.1,
        cfm_debug: bool = False,
        cfm_log_path: str = "cfm_debug.log",
        cfm_weights_path: str = "cfm_fusion_weights_sample.npy",
        lambda_sent_cls: float = 0.0,
        lambda_sent_aux: float = 0.0,
        use_crm: bool = False,
        lambda_crm: float = 0.1,
        crm_num_prototypes: int = 5,
        use_sent_mod: bool = False,
        lambda_sent_mod: float = 0.1,
        use_sent_mod_fuse: bool = False,
        use_sent_mod_fuse_b: bool = False,
        sent_mod_num_prototypes: int = 5,
        sent_mod_fuse_text_weight: float = 0.5,
        use_legacy_plan_b: bool = False,
        use_legacy_multi: bool = False,
        use_sentiment_stream: bool = False,
        use_contrastive_loss: bool = False,
        contrastive_temperature: float = 0.07,
        use_curriculum: bool = False,
        curriculum_start_ratio: float = 0.3,
        curriculum_end_ratio: float = 1.0,
        curriculum_warmup_epochs: int = 30,
        total_epochs: int = 60,
        use_class_weights: bool = False,
        class_weights: torch.Tensor = None,
        use_focal_loss: bool = False,
        focal_loss_alpha: torch.Tensor = None,
        focal_loss_gamma: float = 2.0,
        use_entropy_increase: bool = False,
        entropy_increase_weight: float = 0.1,
        entropy_increase_min_delta: float = 0.1,
    ):
        super().__init__()
        # 情感双流替代 DDM：二者互斥
        if use_sentiment_stream:
            use_ddm = False
        
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        # 根据模型类型设置参数
        if image_model_name == 'swin-large':
            self.hidden_size = 1536
            patch_size = 49
        elif image_model_name == 'swin':
            self.hidden_size = 1024
            patch_size = 49
        elif image_model_name == 'vit-large':
            self.hidden_size = 1024
            patch_size = 197
        elif image_model_name == 'clip-large':
            self.hidden_size = 1024
            patch_size = 257
        elif image_model_name == 'clip':
            self.hidden_size = 768
            patch_size = 50
        else:
            self.hidden_size = 768
            patch_size = 197
    
        print(f"文本模型: {text_model_name}, 图像模型: {image_model_name}, 隐藏维度: {self.hidden_size}, patch数量: {patch_size}")
        
        
        if text_model_name == 'roberta' or text_model_name == 'roberta-large':
            self.roberta = RobertaModel(config1,add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'albert':
            self.albert = AlbertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'electra':
            self.electra = ElectraModel(config1)
        elif text_model_name == 'deberta' or text_model_name == 'deberta-large':
            self.deberta = DebertaModel(config1)
        if image_model_name == 'vit' or image_model_name == "vit-large":
            self.vit = ViTModel(config2)
        elif image_model_name == 'swin' or image_model_name == 'swin-large':
            self.swin = SwinModel(config2)
        elif image_model_name == 'deit':
            self.deit = DeiTModel(config2)
        elif image_model_name == 'convnext':
            self.convnext = ConvNextModel(config2)
        elif image_model_name == 'clip':
            self.clip = CLIPVisionModel.from_pretrained("/root/data/weights/clip")
        elif image_model_name == 'clip-large':
            self.clip = CLIPVisionModel.from_pretrained("/root/data/weights/clip-large")
        
        self.alpha = alpha
        self.beta = beta
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        self.config1 = config1
        self.config2 = config2
        self.gelu0 = nn.GELU()
        self.gelu1 = nn.GELU()
        self.gelu = nn.GELU()
        self.ys_t = nn.Linear(60, 128)
        self.ys_i = nn.Linear(patch_size, 128)
        
        self.text_num_labels = text_num_labels
        self.image_text_cross = InterlanceDecoder(embed_dim=config1.hidden_size,num_classes=self.text_num_labels,num_heads=16,mlp_ratio=4.0, drop_rate=0.1,attn_drop_rate=0.1,drop_path_rate=0.0)
        self.dropout = nn.Dropout(config1.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.mlp0 = Mlp(in_features=config1.hidden_size, hidden_features=int(config1.hidden_size*2))
        
        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        # multi 分支：use_ddm 时 multi = cross_attention + text_fused，经投影层压回 H 再进 classifier0，便于复用旧 checkpoint
        # classifier0 输入维始终与原作者一致：无方案B → H，有方案B → H+3
        # use_legacy_plan_b：阶段目标文档中的旧版方案B，multi = cross_attention + fused_sent_feat（H+3），无 text_fused、无 multi_fuse_proj
        need_plan_b_classifier = use_ddm and use_sent_mod and use_sent_mod_fuse_b
        if need_plan_b_classifier:
            self.classifier0 = Mlp(
                in_features=config1.hidden_size + 3,
                out_features=self.text_num_labels,
                hidden_features=int(config1.hidden_size * 2),
            )
        else:
            self.classifier0 = Mlp(
                in_features=config1.hidden_size,
                out_features=self.text_num_labels,
                hidden_features=int(config1.hidden_size * 2),
            )
        self.use_sent_mod_fuse_b = need_plan_b_classifier
        self.use_legacy_plan_b = use_legacy_plan_b and need_plan_b_classifier
        # use_legacy_multi：仅用于加载无 multi_fuse_proj 的旧 checkpoint；默认 False，use_ddm 时用方案一 cross+detach(text_fused)→multi_fuse_proj
        self.use_legacy_multi = use_legacy_multi and use_ddm
        # use_ddm 时：方案一 multi = cross+detach(text_fused)（+ 可选 fused_sent_feat）→ multi_fuse_proj → classifier0
        if use_ddm and not self.use_legacy_plan_b and not self.use_legacy_multi:
            if need_plan_b_classifier:
                self.multi_fuse_proj = nn.Linear(config1.hidden_size * 2 + 3, config1.hidden_size + 3)
            else:
                self.multi_fuse_proj = nn.Linear(config1.hidden_size * 2, config1.hidden_size)
        else:
            self.multi_fuse_proj = None

        self.CRF = CRF(self.text_num_labels, batch_first=True)
        self.loss_dst = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.9]), reduction='mean')

        # 对抗损失权重（阶段2.3）
        self.lambda1 = lambda1  # 正交损失权重
        # enhanced loss disabled; lambda2 removed
        
        # DDM模块与调试配置（阶段2.2）
        self.use_ddm = use_ddm
        self.use_asymmetric_dims = use_asymmetric_dims
        self.use_image_token_align = use_image_token_align  # image_specific 按 token 对齐（消融可关）
        # ddm_mode 用于 Step 2.6 消融实验：
        # - "text_shared"：text_specific + shared_features
        # - "image_shared"：image_specific + shared_features
        # - "all"：text_specific + image_specific + shared_features
        self.ddm_mode = ddm_mode
        self.ddm_debug = ddm_debug
        self.ddm_log_path = ddm_log_path
        self.ddm_feat_path = ddm_feat_path
        self._ddm_step = 0               # DDM调试步数计数器
        self._saved_ddm_feats = False    # 是否已经保存过一份特征样本

        # CFM 参数（阶段2.4），仅当 use_ddm 时 CFM 才实例化
        self.use_cfm = use_cfm if use_ddm else False
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        self.cfm_debug = cfm_debug
        self.cfm_log_path = cfm_log_path
        self.cfm_weights_path = cfm_weights_path
        self._cfm_step = 0  # CFM调试步数计数器
        self._saved_cfm_weights = False  # 是否已经保存过一份权重样本

        # 核心改进方案（阶段2.5）：情感相关特征监督，仅当 use_ddm 时有效
        self.lambda_sent_cls = lambda_sent_cls
        self.lambda_sent_aux = lambda_sent_aux

        # CRM（原型向量分类机制），use_ddm 或 use_sentiment_stream 时均可用，方案 A：仅辅助损失
        self.use_crm = use_crm if (use_ddm or use_sentiment_stream) else False
        self.lambda_crm = lambda_crm
        self.crm_num_prototypes = crm_num_prototypes

        # 对比学习与课程学习参数（阶段2.6）
        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.use_curriculum = use_curriculum
        self.curriculum_start_ratio = curriculum_start_ratio
        self.curriculum_end_ratio = curriculum_end_ratio
        self.curriculum_warmup_epochs = curriculum_warmup_epochs
        self.total_epochs = total_epochs

        # 类别不平衡处理参数（阶段2.7.1）
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        if self.use_ddm:
            # 使用文本编码器的hidden_size作为DDM的embed_dim
            # specific_dim 和 shared_dim 动态设置为 hidden_size 的一半，
            # 这样 concat(text_specific, shared_features) 的维度与原 hidden_size 一致，
            # 可直接送入 mlp0，避免维度不匹配。
            specific_dim = config1.hidden_size // 2
            shared_dim = config1.hidden_size // 2
            self.ddm = DDM(
                embed_dim=config1.hidden_size,
                specific_dim=specific_dim,
                shared_dim=shared_dim,
                use_image_token_align=use_image_token_align,
                use_asymmetric_dims=use_asymmetric_dims,
            )
            # 根据是否使用非对称维度来设置 shared_dim（供情感监督头使用）
            if use_asymmetric_dims:
                self.shared_dim = 256  # 非对称模式下 shared_dim 固定为 256
            else:
                self.shared_dim = shared_dim  # 对称模式下使用计算的 shared_dim
            # 情感分类头（3 类：NEG/NEU/POS），仅对方面位置监督
            self.sentiment_cls_head = nn.Linear(shared_dim, 3)
            # 情感中心损失：3 个可学习中心
            self.sentiment_centers = nn.Parameter(torch.zeros(3, self.shared_dim))
            nn.init.xavier_uniform_(self.sentiment_centers)
            if lambda_sent_cls > 0 or lambda_sent_aux > 0:
                print(f"情感监督已启用: lambda_sent_cls={lambda_sent_cls}, lambda_sent_aux={lambda_sent_aux}")

            # 打印DDM配置信息
            if use_asymmetric_dims:
                print(f"DDM模块已启用（非对称维度）: embed_dim={config1.hidden_size}, "
                      f"text_specific=512, image_specific=256, shared=256, "
                      f"ddm_mode={self.ddm_mode}, use_image_token_align={self.use_image_token_align}, ddm_debug={self.ddm_debug}")
            else:
                print(f"DDM模块已启用（对称维度）: embed_dim={config1.hidden_size}, "
                      f"specific_dim={specific_dim}, shared_dim={shared_dim}, "
                      f"ddm_mode={self.ddm_mode}, use_image_token_align={self.use_image_token_align}, ddm_debug={self.ddm_debug}")

            # 当使用 "all" 模式时，需要一个线性层将 3 路拼接特征还原到 hidden_size
            if use_asymmetric_dims:
                # 非对称模式：需要为不同的 ddm_mode 添加投影层
                # text_shared: text_specific(512) + shared(256) = 768 -> 1024
                self.ddm_text_shared_proj = nn.Linear(512 + 256, config1.hidden_size)
                # image_shared: image_specific(256) + shared(256) = 512 -> 1024
                self.ddm_image_shared_proj = nn.Linear(256 + 256, config1.hidden_size)
                # all: text_specific(512) + image_specific(256) + shared(256) = 1024 -> 1024
                self.ddm_all_proj = nn.Linear(512 + 256 + 256, config1.hidden_size)
            else:
                # 对称模式：specific_dim + shared_dim = hidden_size，不需要投影
                self.ddm_text_shared_proj = None
                self.ddm_image_shared_proj = None
                # all 模式需要投影：specific_dim * 3 -> hidden_size
                self.ddm_all_proj = nn.Linear(specific_dim * 3, config1.hidden_size)

            # 增强三元组损失（序列级，对全局池化后的特征进行约束）
            # enhanced loss disabled; no enhanced loss function

            # CFM 分类概率融合（阶段2.4）
            if self.use_cfm:
                # CFM 的 input_dim 需要根据是否使用非对称维度来设置
                if use_asymmetric_dims:
                    # 非对称模式：需要将三路特征投影到统一维度
                    cfm_input_dim = 256  # 统一投影到 256
                    # 添加投影层：text_specific (512) -> 256
                    self.cfm_text_proj = nn.Linear(512, 256)
                    # image_specific_aligned 和 shared 已经是 256，不需要投影
                    self.cfm_image_proj = None
                    self.cfm_shared_proj = None
                else:
                    cfm_input_dim = config1.hidden_size // 2
                    self.cfm_text_proj = None
                    self.cfm_image_proj = None
                    self.cfm_shared_proj = None

                self.cfm = CFM(
                    input_dim=cfm_input_dim,
                    output_dim=config1.hidden_size,
                    num_features=3,
                    num_classes=text_num_labels,
                    seq_len=60,
                )
                print(f"CFM已启用: input_dim={cfm_input_dim}, output_dim={config1.hidden_size}, lambda3={lambda3}, lambda4={lambda4}, cfm_debug={cfm_debug}")

            # CRM 原型向量分类（仅当 use_crm 且 lambda_crm > 0 时实例化）
            if self.use_crm and self.lambda_crm > 0:
                self.crm = CRM(
                    hidden_dim=config1.hidden_size,
                    num_classes=3,
                    num_prototypes=crm_num_prototypes,
                )
                print(f"CRM已启用: hidden_dim={config1.hidden_size}, num_prototypes={crm_num_prototypes}, lambda_crm={lambda_crm}")

            # 模态内情感原型（做法2：共用一套），use_ddm 或 use_sentiment_stream 时均可启用
            self.use_sent_mod = use_sent_mod if (use_ddm or use_sentiment_stream) else False
            self.lambda_sent_mod = lambda_sent_mod
            self.use_sent_mod_fuse = use_sent_mod_fuse if use_sent_mod else False
            self.use_sent_mod_fuse_b = use_sent_mod_fuse_b if use_sent_mod else False  # 方案B：表示层拼接情感特征
            self.sent_mod_num_prototypes = sent_mod_num_prototypes
            self.sent_mod_fuse_weight = 0.5  # 方面位置 cross_logits 与模态内情感 logits 融合权重
            # 融合三支 SentMod logits 时 text 支权重（数据集上情感主要由文本决定，可设 >1/3）
            self.sent_mod_fuse_text_weight = max(0.0, min(1.0, sent_mod_fuse_text_weight))
            if self.use_sent_mod:
                branch_dim = config1.hidden_size // 2
                self.sent_mod = SentModModule(
                    branch_dim=branch_dim,
                    num_classes=3,
                    num_prototypes=sent_mod_num_prototypes,
                )
                print(f"SentMod已启用: branch_dim={branch_dim}, num_prototypes={sent_mod_num_prototypes}, "
                      f"lambda_sent_mod={lambda_sent_mod}, use_sent_mod_fuse={use_sent_mod_fuse}, "
                      f"use_sent_mod_fuse_b={use_sent_mod_fuse_b}, fuse_text_weight={self.sent_mod_fuse_text_weight:.2f}")

        # 情感相关/不相关双流（替代 DDM）：只解耦图像，按位置 query 情感相关流
        self.use_sentiment_stream = use_sentiment_stream
        if use_sentiment_stream:
            stream_dim = config1.hidden_size // 2
            self.sentiment_stream = SentimentStreamModule(
                embed_dim=config1.hidden_size,
                stream_dim=stream_dim,
            )
            self.text_stream_proj = nn.Linear(config1.hidden_size, stream_dim)
            self.stream_fusion_proj = nn.Linear(2 * stream_dim, config1.hidden_size)
            self.sentiment_cls_head_stream = nn.Linear(stream_dim, 3)
            self.sentiment_centers_stream = nn.Parameter(torch.zeros(3, stream_dim))
            nn.init.xavier_uniform_(self.sentiment_centers_stream)
            self.multi_fuse_proj = nn.Linear(config1.hidden_size * 2, config1.hidden_size)
            self.use_sent_mod = use_sent_mod
            self.lambda_sent_mod = lambda_sent_mod
            self.use_sent_mod_fuse = use_sent_mod_fuse
            self.use_sent_mod_fuse_b = use_sent_mod_fuse_b
            self.sent_mod_fuse_text_weight = max(0.0, min(1.0, sent_mod_fuse_text_weight))
            self.sent_mod_fuse_weight = 0.5
            print(f"情感双流已启用(替代DDM): stream_dim={stream_dim}, lambda_sent_cls={lambda_sent_cls}, lambda_sent_aux={lambda_sent_aux}")
            if self.use_crm and self.lambda_crm > 0:
                self.crm = CRM(
                    hidden_dim=config1.hidden_size,
                    num_classes=3,
                    num_prototypes=crm_num_prototypes,
                )
                print(f"CRM已启用(情感双流路径): lambda_crm={lambda_crm}")
            if use_sent_mod:
                self.sent_mod = SentModModule(
                    branch_dim=stream_dim,
                    num_classes=3,
                    num_prototypes=sent_mod_num_prototypes,
                )
                print(f"SentMod已启用(情感双流路径): branch_dim={stream_dim}, lambda_sent_mod={lambda_sent_mod}")

        # 熵增约束参数（阶段3.1）
        self.use_entropy_increase = use_entropy_increase
        self.entropy_increase_weight = entropy_increase_weight
        self.entropy_increase_min_delta = entropy_increase_min_delta
        if self.use_entropy_increase:
            print(f"熵增约束已启用: weight={entropy_increase_weight}, min_delta={entropy_increase_min_delta}")

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        elif self.text_model_name == 'roberta' or self.text_model_name == 'roberta-large':
            text_outputs = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'albert':
            text_outputs = self.albert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'electra':
            text_outputs = self.electra(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'deberta' or self.text_model_name == "deberta-large":
            text_outputs = self.deberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            text_outputs=None
            
        if self.image_model_name == 'clip' or self.image_model_name == 'clip-large':
            image_outputs = self.clip(pixel_values)
        else:
            if self.image_model_name == 'vit' or self.image_model_name == "vit-large":
                image_outputs = self.vit(pixel_values,head_mask=head_mask)
            elif self.image_model_name == 'swin' or self.image_model_name == "swin-large":
                image_outputs = self.swin(pixel_values,head_mask=head_mask)
            elif self.image_model_name == 'deit':
                image_outputs = self.deit(pixel_values,head_mask=head_mask)
            elif self.image_model_name == 'convnext':
                image_outputs = self.convnext(pixel_values)
            else:
                image_outputs=None
        if hasattr(image_outputs, 'last_hidden_state'):
            image_last_hidden_states = image_outputs.last_hidden_state
        else:
            image_last_hidden_states = image_outputs[0]    
            

        text_last_hidden_states = text_outputs["last_hidden_state"]            # [B, L_text, hidden_size]
        image_last_hidden_states = image_outputs["last_hidden_state"]          # [B, L_image, hidden_size]

        # 情感相关/不相关双流（替代 DDM）：只解耦图像，按位置 query 情感相关流
        if self.use_sentiment_stream:
            sent_relevant_aligned, sent_irrelevant = self.sentiment_stream(
                text_last_hidden_states, image_last_hidden_states
            )
            text_branch = self.text_stream_proj(text_last_hidden_states)
            text_fused = self.stream_fusion_proj(
                torch.cat([text_branch, sent_relevant_aligned], dim=-1)
            )
            sequence_output1 = self.dropout(self.gelu(self.mlp0(text_fused)))
            text_token_logits = self.classifier1(sequence_output1)

            image_text_cross_attention, mk, _ = self.image_text_cross(
                text_last_hidden_states, image_last_hidden_states
            )
            text_sent_logits = None
            image_sent_logits = None
            shared_sent_logits = None
            fused_sent_feat = None
            if self.use_sent_mod:
                text_sent_logits = self.sent_mod(text_branch)
                image_sent_logits = self.sent_mod(sent_relevant_aligned)
                w_t = self.sent_mod_fuse_text_weight
                fused_sent_feat = w_t * text_sent_logits + (1.0 - w_t) * image_sent_logits

            multi_input = torch.cat([image_text_cross_attention, text_fused.detach()], dim=-1)
            multi_input = self.multi_fuse_proj(multi_input)
            cross_logits = self.classifier0(multi_input)
            if self.use_sent_mod_fuse and fused_sent_feat is not None:
                cross_logits = cross_logits.clone()
                cross_logits[:, :, 2:5] = cross_logits[:, :, 2:5] + self.sent_mod_fuse_weight * fused_sent_feat

            mask = (labels != -100)
            mask[:, 0] = 1
            cross_crf_loss = -self.CRF(cross_logits, cross_labels, mask=mask) / 10

            T_e = self.gelu0(F.normalize(self.ys_t(text_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
            I_e = self.gelu1(F.normalize(self.ys_i(image_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
            logits_ie = torch.matmul(T_e.permute(0, 2, 1), I_e)
            word_region_align_loss = self.loss_dst(logits_ie, eye(logits_ie))

            text_loss = self.loss_fct(
                text_token_logits.view(-1, self.text_num_labels), labels.view(-1)
            )
            loss = (
                cross_crf_loss
                + self.beta * word_region_align_loss
                + self.alpha * text_loss
            )

            # 情感监督：施加在情感相关流（方面位置）
            sent_cls_loss_val = 0.0
            sent_aux_loss_val = 0.0
            if self.lambda_sent_cls > 0 or self.lambda_sent_aux > 0:
                sentiment_targets = torch.where(
                    (labels >= 2) & (labels <= 4),
                    labels - 2,
                    torch.full_like(labels, -100, device=labels.device, dtype=labels.dtype),
                )
                sent_logits = self.sentiment_cls_head_stream(sent_relevant_aligned)
                sent_cls_loss = F.cross_entropy(
                    sent_logits.view(-1, 3), sentiment_targets.view(-1), ignore_index=-100
                )
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    feats_valid = sent_relevant_aligned[valid_aspect]
                    labels_valid = (labels[valid_aspect] - 2).long()
                    diff = feats_valid - self.sentiment_centers_stream[labels_valid]
                    sent_center_loss = (diff ** 2).sum(dim=1).mean()
                else:
                    sent_center_loss = torch.tensor(0.0, device=sent_relevant_aligned.device)
                sent_cls_loss_val = sent_cls_loss.item()
                sent_aux_loss_val = sent_center_loss.item()
                loss = loss + self.lambda_sent_cls * sent_cls_loss + self.lambda_sent_aux * sent_center_loss

            # CRM 辅助损失（方面位置）
            crm_loss_val = 0.0
            if self.use_crm and self.lambda_crm > 0:
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    feats_aspect = sequence_output1[valid_aspect]
                    crm_logits = self.crm(feats_aspect)
                    sentiment_targets = (labels[valid_aspect] - 2).long()
                    crm_loss = F.cross_entropy(crm_logits, sentiment_targets)
                    crm_loss_val = crm_loss.item()
                    loss = loss + self.lambda_crm * crm_loss

            # SentMod 辅助损失（两支：text_branch, sent_relevant_aligned）
            sent_mod_loss_val = 0.0
            if self.use_sent_mod and self.lambda_sent_mod > 0 and text_sent_logits is not None:
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    sentiment_targets = (labels[valid_aspect] - 2).long()
                    sent_mod_loss = (
                        F.cross_entropy(text_sent_logits[valid_aspect], sentiment_targets)
                        + F.cross_entropy(image_sent_logits[valid_aspect], sentiment_targets)
                    ) / 2.0
                    sent_mod_loss_val = sent_mod_loss.item()
                    loss = loss + self.lambda_sent_mod * sent_mod_loss

            if self.training:
                if not hasattr(self, "adv_step"):
                    self.adv_step = 0
                self.adv_step += 1
                if self.adv_step % 100 == 0:
                    try:
                        with open("adv_loss_log.txt", "a", encoding="utf-8") as f:
                            log_line = (
                                f"step={self.adv_step} use_sentiment_stream=1 "
                                f"cross_crf={cross_crf_loss.item():.4f} text={text_loss.item():.4f} "
                                f"align={word_region_align_loss.item():.4f} "
                                f"sent_cls={sent_cls_loss_val:.4f} sent_aux={sent_aux_loss_val:.4f} "
                                f"crm_loss={crm_loss_val:.4f} sent_mod_loss={sent_mod_loss_val:.4f}\n"
                            )
                            f.write(log_line)
                    except Exception:
                        pass
            return {
                "loss": loss,
                "logits": text_token_logits,
                "cross_logits": cross_logits,
            }

        # DDM特征解耦（阶段2.2）+ 对抗损失（阶段2.3）+ image_specific token 级对齐（下一步核心改进）
        if self.use_ddm:
            # 使用DDM进行特征解耦；返回 4 个量，第 4 个为 token 级 image_specific_aligned（或池化，由 use_image_token_align 控制）
            text_specific, image_specific, shared_features, image_specific_aligned = self.ddm(
                text_last_hidden_states,   # [B, L_text, C/2]
                image_last_hidden_states   # [B, L_image, C/2]
            )

            # CFM 融合（阶段2.4）：使用 DDM 返回的 image_specific_aligned（token 级），替代池化版
            cfm_fusion_weights = None
            cfm_branch_logits = None
            if self.use_cfm:
                # 非对称模式：需要投影 text_specific 到统一维度
                if self.use_asymmetric_dims:
                    text_specific_cfm = self.cfm_text_proj(text_specific)  # [B, L, 512] -> [B, L, 256]
                    image_specific_cfm = image_specific_aligned  # 已经是 256
                    shared_features_cfm = shared_features  # 已经是 256
                else:
                    text_specific_cfm = text_specific
                    image_specific_cfm = image_specific_aligned
                    shared_features_cfm = shared_features

                fused_features, cfm_fusion_weights, cfm_branch_logits = self.cfm(
                    text_specific_cfm, image_specific_cfm, shared_features_cfm
                )
                text_fused = fused_features  # [B, L, hidden_size]
            else:
                # 根据 ddm_mode 选择不同的融合策略（用于 Step 2.6 消融实验）；均使用 image_specific_aligned（token 级）
                if self.ddm_mode == "text_shared":
                    concat_feats = torch.cat([text_specific, shared_features], dim=-1)  # [B, L_text, 768 或 768]
                    # 非对称模式需要投影到 hidden_size
                    if self.use_asymmetric_dims:
                        text_fused = self.ddm_text_shared_proj(concat_feats)  # [B, L_text, 1024]
                    else:
                        text_fused = concat_feats  # 对称模式：512+512=1024，不需要投影

                elif self.ddm_mode == "image_shared":
                    concat_feats = torch.cat([image_specific_aligned, shared_features], dim=-1)  # [B, L_text, 512 或 768]
                    # 非对称模式需要投影到 hidden_size
                    if self.use_asymmetric_dims:
                        text_fused = self.ddm_image_shared_proj(concat_feats)  # [B, L_text, 1024]
                    else:
                        text_fused = concat_feats  # 对称模式：512+512=1024，不需要投影

                elif self.ddm_mode == "all":
                    concat_feats = torch.cat(
                        [text_specific, image_specific_aligned, shared_features], dim=-1
                    )  # [B, L_text, 1024 或 1152]
                    text_fused = self.ddm_all_proj(concat_feats)  # [B, L_text, 1024]

                else:
                    concat_feats = torch.cat([text_specific, shared_features], dim=-1)
                    if self.use_asymmetric_dims:
                        text_fused = self.ddm_text_shared_proj(concat_feats)
                    else:
                        text_fused = concat_feats

            # ========== DDM 调试日志与特征导出（可选） ==========
            if self.ddm_debug:
                # 抽样记录特征范数，减少I/O频率（例如每100步记一次）
                if self._ddm_step % 100 == 0:
                    with open(self.ddm_log_path, "a", encoding="utf-8") as f:
                        ts_norm = text_specific.detach().norm(dim=-1).mean().item()
                        is_norm = image_specific.detach().norm(dim=-1).mean().item()
                        sh_norm = shared_features.detach().norm(dim=-1).mean().item()
                        f.write(
                            f"[DDM] step={self._ddm_step} "
                            f"ts_norm={ts_norm:.4f} "
                            f"is_norm={is_norm:.4f} "
                            f"sh_norm={sh_norm:.4f}\n"
                        )

                # 只在首次开启调试时，导出一份特征样本用于离线可视化（避免文件过大）
                if not self._saved_ddm_feats:
                    B = text_specific.size(0)
                    K = min(64, B)  # 最多保存64个样本
                    feats = {
                        "text_specific": text_specific[:K].detach().cpu().numpy(),
                        "image_specific": image_specific[:K].detach().cpu().numpy(),
                        "shared_features": shared_features[:K].detach().cpu().numpy(),
                        "image_specific_aligned": image_specific_aligned[:K].detach().cpu().numpy(),
                        "text_fused": text_fused[:K].detach().cpu().numpy(),
                        "labels": labels[:K].detach().cpu().numpy() if labels is not None else None,
                    }
                    np.save(self.ddm_feat_path, feats)
                    self._saved_ddm_feats = True

                self._ddm_step += 1
            # ========== DDM 调试日志与特征导出结束 ==========

            sequence_output1 = self.dropout(self.gelu(self.mlp0(text_fused)))
            
            # 计算text_token_logits（用于text_loss）
            text_token_logits = self.classifier1(sequence_output1)
            
            # 仍然需要计算cross_crf_loss和word_region_align_loss（使用原始特征，保持损失计算一致性）
            image_text_cross_attention, mk, attn_weights_d = self.image_text_cross(text_last_hidden_states, image_last_hidden_states)
            # 模态内情感原型：三支分别前向得到 3 类 logits；方式 A 在 logits 上融合，方式 B 在表示层拼接
            text_sent_logits = None
            image_sent_logits = None
            shared_sent_logits = None
            fused_sent_feat = None
            if self.use_sent_mod:
                text_sent_logits = self.sent_mod(text_specific)  # [B, L, 3]
                image_sent_logits = self.sent_mod(image_specific_aligned)
                shared_sent_logits = self.sent_mod(shared_features)
                w_t = self.sent_mod_fuse_text_weight
                w_i = (1.0 - w_t) * 0.5
                w_s = (1.0 - w_t) * 0.5
                fused_sent_feat = w_t * text_sent_logits + w_i * image_sent_logits + w_s * shared_sent_logits  # [B, L, 3]
            # 旧版方案B（use_legacy_plan_b）：multi = cross_attention + fused_sent_feat（H+3），无 multi_fuse_proj
            if self.use_legacy_plan_b and fused_sent_feat is not None:
                multi_input = torch.cat([image_text_cross_attention, fused_sent_feat], dim=-1)  # [B, L, H+3]
                cross_logits = self.classifier0(multi_input)
            elif self.use_legacy_multi or self.multi_fuse_proj is None:
                # 旧 checkpoint（无 multi_fuse_proj）：multi = cross_attention 直进 classifier0(H)
                cross_logits = self.classifier0(image_text_cross_attention)
            else:
                # 方案一：multi = cross_attention + detach(text_fused)，cross_crf 不反传 text_fused，避免与 CRM 冲突
                multi_input = torch.cat([image_text_cross_attention, text_fused.detach()], dim=-1)  # [B, L, 2*H]
                if self.use_sent_mod_fuse_b and fused_sent_feat is not None:
                    multi_input = torch.cat([multi_input, fused_sent_feat], dim=-1)  # [B, L, 2*H+3]
                multi_input = self.multi_fuse_proj(multi_input)  # [B, L, H] 或 [B, L, H+3]
                cross_logits = self.classifier0(multi_input)
            if self.use_sent_mod_fuse and not self.use_sent_mod_fuse_b and fused_sent_feat is not None:
                cross_logits = cross_logits.clone()
                cross_logits[:, :, 2:5] = cross_logits[:, :, 2:5] + self.sent_mod_fuse_weight * fused_sent_feat
            mask = (labels != -100)
            mask[:,0] = 1
            cross_crf_loss =  -self.CRF(cross_logits, cross_labels,mask=mask) / 10

            T_e = self.gelu0(F.normalize(self.ys_t(text_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
            I_e = self.gelu1(F.normalize(self.ys_i(image_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
            logits_ie = torch.matmul(T_e.permute(0, 2, 1), I_e)
            word_region_align_loss = self.loss_dst(logits_ie, eye(logits_ie))
        else:
            # 原始逻辑保持不变
            # cross_crf_loss
            image_text_cross_attention, mk, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states)
            cross_logits = self.classifier0(image_text_cross_attention)
            mask = (labels != -100)
            mask[:,0] = 1
            cross_crf_loss =  -self.CRF(cross_logits, cross_labels,mask=mask) / 10

            T_e = self.gelu0(F.normalize(self.ys_t(text_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
            I_e = self.gelu1(F.normalize(self.ys_i(image_last_hidden_states.permute(0, 2, 1)), p=2, dim=-1, eps=1e-7))
            logits_ie = torch.matmul(T_e.permute(0, 2, 1), I_e)
            word_region_align_loss = self.loss_dst(logits_ie, eye(logits_ie))

            # text_loss
            sequence_output1 = self.dropout(self.gelu(self.mlp0(text_last_hidden_states) + mk))
        
            text_token_logits = self.classifier1(sequence_output1)

        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))

        # ===== 阶段2.3：对抗损失（正交损失 + 增强损失） =====
        if self.use_ddm:
            # 1) 正交损失使用「池化」的 image_specific（纯图像、无 text query），保持解耦几何意义
            # 只在 lambda1 > 0 且非对称模式关闭时计算（非对称模式下维度不同，无法计算正交损失）
            if self.lambda1 > 0 and not self.use_asymmetric_dims:
                B_ddm, L_text_ddm, C_half_ddm = text_specific.size()
                img_sp = image_specific.permute(0, 2, 1)  # [B, C/2, L_image]
                img_sp_pooled = F.adaptive_avg_pool1d(img_sp, L_text_ddm)  # [B, C/2, L_text]
                image_specific_pooled = img_sp_pooled.permute(0, 2, 1)  # [B, L_text, C/2]

                ortho_loss = orthogonal_loss_for_sequence(
                    text_specific,
                    image_specific_pooled,
                    shared_features,
                )
            else:
                ortho_loss = torch.tensor(0.0, device=text_specific.device)

            # 3) 增强损失已禁用（enhanced loss disabled）
            # 注意：非对称模式下三路特征维度不同，无法直接相加
            # fused_features = text_specific + image_specific_aligned + shared_features
            # fused_global = fused_features.mean(dim=1)
            # text_global = text_specific.mean(dim=1)
            # shared_global = shared_features.mean(dim=1)

            # 4) 从 token 级标签构造句子级标签（简单策略：取每句中最大标签值，O/I 为低优先级）
            with torch.no_grad():
                # labels: [B, L]，取每个样本的最大标签作为该句情感标签
                # 0: O, 1: I, 2: B-NEG, 3: B-NEU, 4: B-POS
                seq_labels = labels.max(dim=1).values  # [B]

            loss = (
                cross_crf_loss
                + self.beta * word_region_align_loss
                + self.alpha * text_loss
                + self.lambda1 * ortho_loss
            )

            # CFM 辅助损失（阶段2.4）：分支一致性 + 权重正则
            consistency_loss = 0.0
            weight_reg_loss = 0.0
            if self.use_cfm and cfm_fusion_weights is not None and cfm_branch_logits is not None:
                # 分支一致性：三路 branch_logits 与主路 text_token_logits 的 KL 散度（仅有效 token）
                valid = (labels != -100).unsqueeze(-1).float()  # [B, L, 1]
                log_p_main = F.log_softmax(text_token_logits, dim=-1)
                p_main = F.softmax(text_token_logits, dim=-1)
                consistency_loss = 0.0
                for branch_logits in cfm_branch_logits:
                    log_p_b = F.log_softmax(branch_logits, dim=-1)
                    kl = (p_main * (log_p_main - log_p_b)).sum(dim=-1, keepdim=True)  # [B, L, 1]
                    consistency_loss = consistency_loss + (kl * valid).sum() / (valid.sum() + 1e-8)
                consistency_loss = consistency_loss / 3.0

                # 权重正则：鼓励 fusion_weights 熵更大（避免过度尖锐）
                entropy = -(cfm_fusion_weights * (cfm_fusion_weights + 1e-8).log()).sum(dim=-1).mean()
                weight_reg_loss = -entropy  # 最小化 -entropy 即最大化熵

                loss = loss + self.lambda3 * consistency_loss + self.lambda4 * weight_reg_loss

                # ========== CFM 调试日志与权重导出（可选） ==========
                if self.cfm_debug:
                    # 周期性记录融合权重统计和损失值
                    if self._cfm_step % 100 == 0:
                        try:
                            with open(self.cfm_log_path, "a", encoding="utf-8") as f:
                                # 计算平均权重（text/image/shared）
                                w_mean = cfm_fusion_weights.detach().mean(dim=(0, 1)).cpu().numpy()  # [3]
                                w_std = cfm_fusion_weights.detach().std(dim=(0, 1)).cpu().numpy()  # [3]
                                # 计算权重熵（平均）
                                w_entropy = entropy.detach().item()
                                f.write(
                                    f"[CFM] step={self._cfm_step} "
                                    f"w_text_mean={w_mean[0]:.4f} w_image_mean={w_mean[1]:.4f} w_shared_mean={w_mean[2]:.4f} "
                                    f"w_text_std={w_std[0]:.4f} w_image_std={w_std[1]:.4f} w_shared_std={w_std[2]:.4f} "
                                    f"w_entropy={w_entropy:.4f} "
                                    f"consistency_loss={consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss:.4f} "
                                    f"weight_reg_loss={weight_reg_loss.item() if isinstance(weight_reg_loss, torch.Tensor) else weight_reg_loss:.4f} "
                                    f"lambda3={self.lambda3:.3f} lambda4={self.lambda4:.3f}\n"
                                )
                        except Exception:
                            pass

                    # 只在首次开启调试时，导出一份权重样本用于离线可视化
                    if not self._saved_cfm_weights:
                        B = cfm_fusion_weights.size(0)
                        K = min(64, B)  # 最多保存64个样本
                        weights_data = {
                            "fusion_weights": cfm_fusion_weights[:K].detach().cpu().numpy(),  # [K, L, 3]
                            "labels": labels[:K].detach().cpu().numpy() if labels is not None else None,
                            "text_logits": cfm_branch_logits[0][:K].detach().cpu().numpy() if cfm_branch_logits else None,
                            "image_logits": cfm_branch_logits[1][:K].detach().cpu().numpy() if cfm_branch_logits else None,
                            "shared_logits": cfm_branch_logits[2][:K].detach().cpu().numpy() if cfm_branch_logits else None,
                        }
                        np.save(self.cfm_weights_path, weights_data)
                        self._saved_cfm_weights = True
                        print(f"[CFM] 已保存权重样本到 {self.cfm_weights_path}")

                    self._cfm_step += 1
                # ========== CFM 调试日志与权重导出结束 ==========

            # 核心改进方案（阶段2.5）：情感相关特征监督（仅对方面位置，NEG/NEU/POS）
            sent_cls_loss_val = 0.0
            sent_aux_loss_val = 0.0
            crm_loss_val = 0.0
            if self.use_ddm and (self.lambda_sent_cls > 0 or self.lambda_sent_aux > 0):
                # labels: 0=O, 1=I, 2=B-NEG, 3=B-NEU, 4=B-POS -> 情感 0,1,2
                sentiment_targets = torch.where(
                    (labels >= 2) & (labels <= 4),
                    labels - 2,
                    torch.full_like(labels, -100, device=labels.device, dtype=labels.dtype),
                )
                sent_logits = self.sentiment_cls_head(shared_features)  # [B, L, 3]
                sent_cls_loss = F.cross_entropy(
                    sent_logits.view(-1, 3),
                    sentiment_targets.view(-1),
                    ignore_index=-100,
                )
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    feats_valid = shared_features[valid_aspect]  # [N, shared_dim]
                    labels_valid = (labels[valid_aspect] - 2).long()  # [N]
                    centers = self.sentiment_centers  # [3, shared_dim]
                    diff = feats_valid - centers[labels_valid]
                    sent_center_loss = (diff ** 2).sum(dim=1).mean()
                else:
                    sent_center_loss = torch.tensor(0.0, device=shared_features.device)
                sent_cls_loss_val = sent_cls_loss.item()
                sent_aux_loss_val = sent_center_loss.item()
                loss = loss + self.lambda_sent_cls * sent_cls_loss + self.lambda_sent_aux * sent_center_loss

            # CRM 原型向量分类辅助损失（方案 A：仅对方面位置，与情感标签 CE）
            if self.use_ddm and self.use_crm and self.lambda_crm > 0:
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    feats_aspect = sequence_output1[valid_aspect]
                    crm_logits = self.crm(feats_aspect)
                    sentiment_targets = (labels[valid_aspect] - 2).long()
                    crm_loss = F.cross_entropy(crm_logits, sentiment_targets)
                    crm_loss_val = crm_loss.item()
                    loss = loss + self.lambda_crm * crm_loss

            # 模态内情感原型辅助损失（仅对方面位置，三支 CE 取平均）
            sent_mod_loss_val = 0.0
            if self.use_ddm and self.use_sent_mod and self.lambda_sent_mod > 0 and text_sent_logits is not None:
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    sentiment_targets = (labels[valid_aspect] - 2).long()
                    t_logits = text_sent_logits[valid_aspect]
                    i_logits = image_sent_logits[valid_aspect]
                    s_logits = shared_sent_logits[valid_aspect]
                    sent_mod_loss = (
                        F.cross_entropy(t_logits, sentiment_targets)
                        + F.cross_entropy(i_logits, sentiment_targets)
                        + F.cross_entropy(s_logits, sentiment_targets)
                    ) / 3.0
                    sent_mod_loss_val = sent_mod_loss.item()
                    loss = loss + self.lambda_sent_mod * sent_mod_loss

            # 熵增约束损失（阶段3.1）
            entropy_increase_loss_val = 0.0
            if self.use_entropy_increase and self.entropy_increase_weight > 0 and attn_weights_d is not None:
                # attn_weights_d 的形状应该是 [B, L_text, L_image]
                # 计算每个文本token对图像区域的注意力分布的熵
                # 熵 = -sum(p * log(p))，熵越大表示注意力越分散

                # 添加小的epsilon避免log(0)
                eps = 1e-10
                attn_probs = attn_weights_d + eps

                # 计算每个文本token的注意力熵
                # attn_probs: [B, L_text, L_image]
                entropy = -torch.sum(attn_probs * torch.log(attn_probs), dim=-1)  # [B, L_text]

                # 只对方面词位置计算熵增损失
                valid_aspect = (labels >= 2) & (labels <= 4)
                if valid_aspect.any():
                    aspect_entropy = entropy[valid_aspect]  # [N_aspect]
                    # 熵增损失：鼓励方面词的注意力更加分散（熵更大）
                    # 使用负熵作为损失，最小化负熵等价于最大化熵
                    entropy_increase_loss = -torch.mean(aspect_entropy)
                    entropy_increase_loss_val = entropy_increase_loss.item()
                    loss = loss + self.entropy_increase_weight * entropy_increase_loss

            # 训练阶段：周期性记录各项损失到日志，便于诊断对抗损失占比
            if self.training:
                if not hasattr(self, "adv_step"):
                    self.adv_step = 0
                self.adv_step += 1
                # 每 100 个 step 记录一次，避免 I/O 过于频繁
                if self.adv_step % 100 == 0:
                    try:
                        with open("adv_loss_log.txt", "a", encoding="utf-8") as f:
                            log_line = (
                                f"step={self.adv_step} "
                                f"cross_crf={cross_crf_loss.item():.4f} "
                                f"text={text_loss.item():.4f} "
                                f"align={word_region_align_loss.item():.4f} "
                                f"ortho={ortho_loss.item():.4f} "
                                f"lambda1={self.lambda1:.3f}"
                            )
                            # 如果启用了CFM，也记录CFM相关损失
                            if self.use_cfm and cfm_fusion_weights is not None and cfm_branch_logits is not None:
                                consistency_loss_val = consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss
                                weight_reg_loss_val = weight_reg_loss.item() if isinstance(weight_reg_loss, torch.Tensor) else weight_reg_loss
                                log_line += (
                                    f" cfm_consistency={consistency_loss_val:.4f} "
                                    f"cfm_weight_reg={weight_reg_loss_val:.4f} "
                                    f"lambda3={self.lambda3:.3f} "
                                    f"lambda4={self.lambda4:.3f}"
                                )
                            if self.use_ddm and (self.lambda_sent_cls > 0 or self.lambda_sent_aux > 0):
                                log_line += (
                                    f" sent_cls={sent_cls_loss_val:.4f} sent_aux={sent_aux_loss_val:.4f} "
                                    f"lambda_sent_cls={self.lambda_sent_cls:.3f} lambda_sent_aux={self.lambda_sent_aux:.3f}"
                                )
                            if self.use_ddm and self.use_crm and self.lambda_crm > 0:
                                log_line += (
                                    f" crm_loss={crm_loss_val:.4f} lambda_crm={self.lambda_crm:.3f}"
                                )
                            if self.use_ddm and self.use_sent_mod and self.lambda_sent_mod > 0:
                                log_line += (
                                    f" sent_mod_loss={sent_mod_loss_val:.4f} lambda_sent_mod={self.lambda_sent_mod:.3f}"
                                )
                            if self.use_entropy_increase and self.entropy_increase_weight > 0:
                                log_line += (
                                    f" entropy_increase_loss={entropy_increase_loss_val:.4f} entropy_weight={self.entropy_increase_weight:.3f}"
                                )
                            log_line += "\n"
                            f.write(log_line)
                    except Exception:
                        # 日志失败不影响训练
                        pass
        else:
            # 原始损失（无对抗解耦）
            loss = cross_crf_loss + self.beta * word_region_align_loss + self.alpha * text_loss

        # end train
        # 返回详细的损失分量，便于验证时分析
        loss_details = {
            "loss": loss,
            "logits": text_token_logits,
            "cross_logits": cross_logits,
            # 主要损失分量
            "cross_crf_loss": cross_crf_loss.item(),
            "text_loss": text_loss.item(),
            "word_region_align_loss": word_region_align_loss.item(),
        }

        # DDM 相关损失
        if self.use_ddm:
            loss_details["ortho_loss"] = ortho_loss.item()

            # CFM 损失（只有当真正计算了CFM损失时才添加）
            if self.use_cfm and isinstance(consistency_loss, torch.Tensor):
                loss_details["cfm_consistency_loss"] = consistency_loss.item()
                loss_details["cfm_weight_reg_loss"] = weight_reg_loss.item() if isinstance(weight_reg_loss, torch.Tensor) else weight_reg_loss

            # 情感分类损失
            if self.lambda_sent_cls > 0 or self.lambda_sent_aux > 0:
                loss_details["sent_cls_loss"] = sent_cls_loss_val
                loss_details["sent_aux_loss"] = sent_aux_loss_val

            # CRM 损失
            if self.use_crm and self.lambda_crm > 0:
                loss_details["crm_loss"] = crm_loss_val

            # SentMod 损失
            if self.use_sent_mod and self.lambda_sent_mod > 0:
                loss_details["sent_mod_loss"] = sent_mod_loss_val

            # 熵增约束损失
            if self.use_entropy_increase and self.entropy_increase_weight > 0:
                loss_details["entropy_increase_loss"] = entropy_increase_loss_val

        return loss_details

def eye(x):
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.float32, device=x.device).unsqueeze(0).expand_as(x)
    return mask

def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss

def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance
