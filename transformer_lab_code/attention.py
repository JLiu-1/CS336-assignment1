"""M2 注意力。多头重排、RoPE 缓存和广播均已提供。"""

import torch
from torch import nn

from .layers import Linear, softmax  # noqa: F401 -- T7 使用 softmax


def apply_mask(scores, mask):
    """True = 允许关注；False = 在 softmax 前替换为 -inf。"""
    return scores if mask is None else scores.masked_fill(~mask, float("-inf"))


def scaled_dot_product_attention(q, k, v, mask=None):
    """T7：softmax(Q K^T / sqrt(Dk)) V。

    q [..., Q, Dk]、k [..., K, Dk]、v [..., K, Dv] → [..., Q, Dv]。
    先缩放并调用 apply_mask，再用自己实现的 softmax；不要调用融合 attention。
    可以使用 @、transpose(-2, -1)、shape[-1]。Q 和 K 长度不一定相等。
    """
    raise NotImplementedError("TODO T7: scaled_dot_product_attention")


def rotate_pairs(even, odd, cos, sin):
    """T8：返回旋转后的两个坐标组成的 tuple。

    (a, b) → (a*cos - b*sin, a*sin + b*cos)。四个输入已经可以广播。
    只写这两条坐标公式；组合回向量的代码已提供。
    """
    raise NotImplementedError("TODO T8: rotate_pairs")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        if d_k % 2 or d_k <= 0 or theta <= 0 or max_seq_len <= 0:
            raise ValueError("RoPE 要求正偶数 head_dim、正 theta 和正 context_length")
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = torch.arange(max_seq_len, device=device, dtype=torch.float32)[:, None] * frequencies
        # buffer 会随 model.to(device) 移动；不是可学习参数，不写入权重快照。
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        # token_positions 可为 [T] 或 [B,T]；为多头输入补充 head 广播维。
        while cos.ndim < x.ndim:
            cos, sin = cos.unsqueeze(-3), sin.unsqueeze(-3)
        even, odd = rotate_pairs(x[..., 0::2], x[..., 1::2], cos, sin)
        return torch.stack((even, odd), dim=-1).flatten(-2)


def causal_mask(length, device=None):
    """T9：返回 bool [T,T]；第 i 行只允许 j <= i（含对角线）。

    可使用 torch.arange 和索引比较，或 torch.ones(..., dtype=torch.bool).tril()。
    """
    raise NotImplementedError("TODO T9: causal_mask")


def split_heads(x, num_heads):
    """[..., T, D] → [..., H, T, Dh]。"""
    return x.reshape(*x.shape[:-1], num_heads, x.shape[-1] // num_heads).transpose(-3, -2)


def merge_heads(x):
    """[..., H, T, Dh] → [..., T, H*Dh]；reshape 处理非连续内存。"""
    return x.transpose(-3, -2).reshape(*x.shape[:-3], x.shape[-2], x.shape[-3] * x.shape[-1])


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=256, theta=10000.0, use_rope=True, device=None, dtype=None):
        super().__init__()
        if num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model 必须被 num_heads 整除")
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model, device, dtype)
        self.k_proj = Linear(d_model, d_model, device, dtype)
        self.v_proj = Linear(d_model, d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device) if use_rope else None

    def forward(self, x, token_positions=None):
        q, k, v = [split_heads(proj(x), self.num_heads) for proj in (self.q_proj, self.k_proj, self.v_proj)]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        values = scaled_dot_product_attention(q, k, v, causal_mask(x.shape[-2], x.device))
        return self.output_proj(merge_heads(values))
