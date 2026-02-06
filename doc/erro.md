错误记录（ICT-main / 阶段2：DDM集成）
=================================

> 说明：本文件用于记录在ICT-main项目中出现的各类错误信息，便于后续排查与回溯。

---

2026-01-26：DDM集成后训练时报维度不匹配错误
----------------------------------------

- **环境**：服务器 `/root/data/ICT-main-mdre/ICT-main`，conda 环境 `ict`
- **运行命令**：
  - `python main.py --dataset_type 2015 --text_model_name roberta-large --image_model_name vit-large --batch_size 16 --epochs 60 --alpha 0.635 --beta 0.565 --lr 2.26e-05 --use_ddm --random_seed 2022`
- **核心报错信息（截取）**：
  - `RuntimeError: mat1 and mat2 shapes cannot be multiplied (960x768 and 1024x2048)`
  - 报错位置：`model.py` 中 `sequence_output1 = self.dropout(self.gelu(self.mlp0(text_fused)))`
- **问题原因**：
  - 原始 ICT 模型中 `mlp0` 的输入维度为 `config1.hidden_size`：
    - 对 `roberta-large` 来说，`hidden_size = 1024`
  - 初版 DDM 设计中，将文本特征拆分为：
    - `text_specific: [B, 60, 384]`
    - `shared_features: [B, 60, 384]`
    - 拼接后 `text_fused: [B, 60, 768]`
  - 将 768 维特征直接送入输入维度为 1024 的 `mlp0`，导致矩阵维度不匹配，触发上述错误。
- **修复方案**：
  - 在 `ICTModel.__init__` 中初始化 DDM 时，不再写死 384 维，而是根据 `config1.hidden_size` 动态设置：
    - `specific_dim = config1.hidden_size // 2`
    - `shared_dim = config1.hidden_size // 2`
    - `self.ddm = DDM(embed_dim=config1.hidden_size, specific_dim=specific_dim, shared_dim=shared_dim)`
  - 这样：
    - `text_specific: [B, 60, hidden_size/2]`
    - `shared_features: [B, 60, hidden_size/2]`
    - `text_fused = concat(text_specific, shared_features)` → `[B, 60, hidden_size]`
    - 与 `mlp0` 的输入维度一致，错误消失。
- **状态**：
  - 已在本地更新 `model.py` 并重新上传。
  - 重新运行训练命令后，维度不匹配错误不再出现。

---

后续改进建议：增加训练步骤日志
----------------------------

- **动机**：
  - 当前错误排查主要依赖终端输出和手动回忆运行步骤，不够系统。
- **建议**：
  - 后续可以在训练脚本（如 `main.py` 或单独的工具模块）中：
    - 在每个关键阶段（数据加载、模型构建、DDM/损失集成、训练开始、eval 等）写入简要“步骤 log”到独立日志文件（例如 `logs/train_steps.log`）。
    - 日志内容只记录关键信息（时间戳 + 步骤名称 + 关键参数），便于快速还原出错时刻的操作上下文。

