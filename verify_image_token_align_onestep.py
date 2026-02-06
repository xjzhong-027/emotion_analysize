"""
image_specific token 级对齐 —— 一步训练验证脚本

用途：在不加载预训练权重、不读数据集的前提下，验证「DDM 四路输出 + 正交损失用池化 + 融合/增强用 aligned」
     的 forward/backward 无 shape 错误、梯度可回传。对应阶段目标文档 Step A.2 最后一项 Checklist。

运行：在 ICT-main 目录下执行
  python verify_image_token_align_onestep.py

成功：打印 "OK: 一步 forward/backward 通过，无 shape 错误，梯度可回传。"
失败：抛出异常或 shape 断言错误。
"""

import torch
import torch.nn.functional as F
from ddm_module import DDM
from mdre_components import orthogonal_loss_for_sequence, EnhancedTripletLossForSequence


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L_text, L_image, embed_dim = 2, 60, 197, 768
    specific_dim = shared_dim = embed_dim // 2  # 384

    # 1. DDM（与 model.py 中一致：use_image_token_align=True）
    ddm = DDM(
        embed_dim=embed_dim,
        specific_dim=specific_dim,
        shared_dim=shared_dim,
        use_image_token_align=True,
    ).to(device)

    text_feat = torch.randn(B, L_text, embed_dim, device=device, requires_grad=True)
    image_feat = torch.randn(B, L_image, embed_dim, device=device, requires_grad=True)

    # 2. DDM 返回 4 个量
    text_specific, image_specific, shared_features, image_specific_aligned = ddm(text_feat, image_feat)

    assert text_specific.shape == (B, L_text, specific_dim), text_specific.shape
    assert image_specific.shape == (B, L_image, specific_dim), image_specific.shape
    assert shared_features.shape == (B, L_text, shared_dim), shared_features.shape
    assert image_specific_aligned.shape == (B, L_text, specific_dim), image_specific_aligned.shape

    # 3. 正交损失：使用池化后的 image_specific（与 model.py 一致）
    img_sp = image_specific.permute(0, 2, 1)  # [B, C/2, L_image]
    img_sp_pooled = F.adaptive_avg_pool1d(img_sp, L_text)
    image_specific_pooled = img_sp_pooled.permute(0, 2, 1)  # [B, L_text, C/2]
    ortho_loss = orthogonal_loss_for_sequence(
        text_specific,
        image_specific_pooled,
        shared_features,
    )

    # 4. 增强损失：融合特征用 token 级 image_specific_aligned（与 model.py 一致）
    fused_features = text_specific + image_specific_aligned + shared_features  # [B, L, C/2]
    fused_global = fused_features.mean(dim=1)
    text_global = text_specific.mean(dim=1)
    shared_global = shared_features.mean(dim=1)
    # 构造假标签：每句取一个标签 0/1/2（仅用于 loss 需要）
    seq_labels = torch.randint(0, 3, (B,), device=device)
    enh_fn = EnhancedTripletLossForSequence(hard_factor=0.0)
    enhanced_loss = enh_fn(
        fused_features=fused_global,
        text_specific_features=text_global,
        shared_features=shared_global,
        labels=seq_labels,
        normalize_feature=True,
    )

    # 5. 总损失并反传
    loss = ortho_loss + enhanced_loss
    loss.backward()

    # 6. 检查 DDM 参数有梯度
    has_grad = any(p.grad is not None for p in ddm.parameters() if p.requires_grad)
    assert has_grad, "DDM 参数未收到梯度"

    print("OK: 一步 forward/backward 通过，无 shape 错误，梯度可回传。")


if __name__ == "__main__":
    main()
