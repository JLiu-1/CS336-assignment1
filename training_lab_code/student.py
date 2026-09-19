"""M4：三人小组填写 S1–S3。M1–M3 的实现继续由原文件提供。"""

import numpy as np  # noqa: F401 -- S1 使用
import torch

from .losses import cross_entropy
from .optimizer import clip_gradients  # noqa: F401 -- S2 使用


def shifted_windows(data, starts, context_length):
    """S1：返回 numpy 数组 (x,y)，形状均为 [B,T]。

    data 是一维 token 数组；starts 是有效起点数组，已由框架采样。
    对每个起点 s，x=data[s:s+T]，y=data[s+1:s+T+1]。
    可使用列表推导式与 np.stack，不要求自己写高效索引或 DataLoader 类。
    """
    raise NotImplementedError("TODO S1: shifted_windows")


def forward_loss(model, x, y, precision="fp32"):
    """已提供：参数保持 FP32；可选 CUDA BF16 autocast，损失在 FP32 下计算。"""
    if precision == "bf16" and x.device.type != "cuda":
        raise ValueError("课程 bf16 模式仅用于 CUDA；CPU 验收用 fp32")
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        return cross_entropy(model(x), y)


def train_step(model, optimizer, batches, max_norm=1.0, precision="fp32"):
    """S2：完成一次优化器更新；返回 Python float 的平均训练 loss。

    batches 是等大 microbatch 的 (x,y) 列表，长度即梯度累积次数。
    顺序：zero_grad → 对每个 batch 调 forward_loss → loss/len(batches) 反传
    → clip_gradients → optimizer.step。记录 loss 用 loss.detach().item()。
    检查 loss 是否有限，非有限时抛 FloatingPointError，不做参数更新。
    可用 torch.isfinite。不要在 microbatch 之间清零或更新参数。
    """
    raise NotImplementedError("TODO S2: train_step")


def weighted_validation_loss(batch_mean_losses, token_counts):
    """S3：按每批有效 token 数加权，返回整个验证集的平均 loss。

    两个等长、非空 Python 列表，token_counts 都为正。
    不要直接平均 batch 的平均值，因为最后一个 batch 可能较小。
    """
    raise NotImplementedError("TODO S3: weighted_validation_loss")
