"""M2 基础层。框架已提供，填写 T1–T6；公式和验收见 docs/m2_transformer.md。"""

import math

import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        """T1：x [..., Din]，weight [Dout, Din] → [..., Dout]，没有 bias。"""
        raise NotImplementedError("TODO T1: Linear.forward")


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids):
        """T2：整数 IDs [...] → 向量 [..., D]。使用索引，不需要构造 one-hot。"""
        raise NotImplementedError("TODO T2: Embedding.forward")


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        original_dtype = x.dtype
        x = x.float()
        result = self.normalize(x)
        return result.to(original_dtype)

    def normalize(self, x):
        """T3：沿最后一维计算 x / sqrt(mean(x²) + eps)，再乘可学习的 weight。

        输入已转换成 float32。保留归约维度以便广播，不减均值。
        """
        raise NotImplementedError("TODO T3: RMSNorm.normalize")


def silu(x):
    """T4：SiLU(x) = x * sigmoid(x)，可以使用 torch.sigmoid。"""
    raise NotImplementedError("TODO T4: silu")


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        """T5：w2(SiLU(w1(x)) ⊙ w3(x))。⊙ 是逐元素乘法，不是矩阵乘法。"""
        raise NotImplementedError("TODO T5: SwiGLU.forward")


def softmax(x, dim=-1):
    """T6：沿 dim 减最大值、取指数、归一化。允许某些项为 -inf。

    本课程保证每行至少有一个未被屏蔽的有限值，不要求处理全 -inf 行。
    不能调用 torch.softmax / nn.functional.softmax。
    """
    raise NotImplementedError("TODO T6: softmax")
