# new_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.1,
        num_classes=20,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = attn.softmax(dim=-1)
        weights = attn

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.mha = nn.MultiheadAttention(out_dim, num_heads, dropout=attn_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = (
            self.q_map(q)
            .view(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_map(k)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_map(v)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        x, weights = self.mha(q, k, v)
        return x, weights


class CrossSkipClsAttention(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        num_classes=5,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.mha = nn.MultiheadAttention(out_dim, num_heads, dropout=attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = (
            self.q_map(q)
            .view(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_map(k)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_map(v)
            .view(B, NK, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        weights = attn

        # skip the attentions
        no_cls_attn = attn[..., self.num_classes :].softmax(dim=-1)
        no_cls_v = v[:, :, self.num_classes :, :]
        no_cls_attn = self.attn_drop(no_cls_attn)

        x = (no_cls_attn @ no_cls_v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # t = (no_cls_attn @ no_cls_v).transpose(1, 2).reshape(B, N, C)
        # x, _ = self.mha(q.permute(0, 2, 1, 3).reshape(B, N, C), t, t)
        
        return x, weights


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_classes=20,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_classes=num_classes,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dim_q=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim,
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, v):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1, _ = self.self_attn(norm_q)
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)

        q_2, weights = self.attn(norm_q, norm_v)
        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q, weights


class SkipClsDecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_classes,
        dim_q=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossSkipClsAttention(
            dim,
            dim,
            num_classes,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity())
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,hidden_features=int(dim * mlp_ratio),act_layer=act_layer,drop=drop)

    def forward(self, q, v):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1, _ = self.self_attn(norm_q)
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)

        q_2, weights = self.attn(norm_q, norm_v)
        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q, weights


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=2,
        num_heads=8,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        mlp_ratio=0.5,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.1,
    ):
        super(TransformerEncoder, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

    def forward(self, x, n=12):
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            # if pos:
            x, weights_i = blk(x)
            # else:
            #     x, weights_i = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights_i)

        return x, attn_weights


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        depth=2,
        num_heads=8,
        is_swap=False,
        mlp_ratio=0.5,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
    ):
        super(TransformerDecoder, self).__init__()
        self.is_swap = is_swap
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, key, q):
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            q, weights_i = blk(q, key)
            attn_weights.append(weights_i)
        return q, attn_weights


class InterlanceDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        depth=2,
        num_classes=5,
        num_heads=8,
        mlp_ratio=0.2,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super(InterlanceDecoder, self).__init__()
        self.blocks = nn.ModuleList(
            [
                SkipClsDecoderBlock(
                    dim=embed_dim,
                    num_classes=num_classes,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                )
                for _ in range(depth)
            ]
        )

    # def forward(self, voxel, view):
    #     attn_weights = []

    #     for i, blk in enumerate(self.blocks):
    #         if i % 2:  # even layer
    #             view, weights_i = blk(q=view, v=voxel)
    #         else:
    #             voxel, weights_i = blk(q=voxel, v=view)
    #         attn_weights.append(weights_i)
    #     return voxel, view, attn_weights
    def forward(self, voxel, view, eps=1e-7):
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            if i % 2:  # even layer
                view, weights_i = blk(q=view, v=voxel)
            else:
                voxel, weights_i = blk(q=voxel, v=view)
            attn_weights.append(torch.mean(weights_i, dim=1))
        c = torch.einsum('bij,bjk->bik', view.permute(0,2,1), attn_weights[1])
        d = F.normalize(attn_weights[0], p=2, dim=-1, eps=eps).matmul(F.normalize(attn_weights[1], p=2, dim=-1, eps=eps))
        return voxel, c.permute(0, 2, 1), d

if __name__ == '__main__':
    # 创建一个大小为 (16, 60, 768) 的张量
    tensor1 = torch.randn(16, 60, 768)
    
    # 创建一个大小为 (16, 197, 768) 的张量
    tensor2 = torch.randn(16, 197, 768)
    
    # 打印张量的形状
    print("tensor1 shape:", tensor1.shape)
    print("tensor2 shape:", tensor2.shape)

    model = InterlanceDecoder(embed_dim=768,
                            num_classes=20,
                            drop_rate=0.1,
                            attn_drop_rate=0.1,
                            drop_path_rate=0.0,)
    model.eval()
    view_attn_token, voxel_attn_token, _, cross_attn = model.forward(tensor1, tensor2)
    print(view_attn_token.size(), tensor1.shape)
    print(voxel_attn_token.size(), tensor2.shape)   # n*m *m*h = n*h
    print(len(cross_attn))
    print(cross_attn[0].size())
    print(cross_attn[1].size())
    # 16, 768, 197 16, 197, 60
    c = torch.einsum('bij,bjk->bik', voxel_attn_token.permute(0,2,1), cross_attn[1])
    print(c.permute(0,2,1).size())

    num_classes = 20
    cross_cls_token1 = voxel_attn_token[0, :num_classes]  # 20 x D
    cross_cls_token2 = view_attn_token[0, :num_classes]
    cross_cls_token = (cross_cls_token1 + cross_cls_token2) / 2
    cross_mct_logits = cross_cls_token.unsqueeze(0)
    print(cross_mct_logits.size())