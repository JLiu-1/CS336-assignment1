"""已提供：固定验证窗口，不消耗训练随机数，不在验证集上反向传播。"""

import numpy as np
import torch

from .data import batch_at
from .student import forward_loss, weighted_validation_loss


@torch.no_grad()
def evaluate(model, data, context_length, batch_size, max_batches, device, precision="fp32"):
    starts = np.arange(0, len(data) - context_length, context_length)[: batch_size * max_batches]
    if len(starts) == 0:
        raise ValueError("验证数据不足一个窗口")
    was_training = model.training
    model.eval()
    losses, counts = [], []
    try:
        for offset in range(0, len(starts), batch_size):
            x, y = batch_at(data, starts[offset : offset + batch_size], context_length, device)
            loss = forward_loss(model, x, y, precision)
            if not torch.isfinite(loss):
                raise FloatingPointError("验证损失非有限")
            losses.append(loss.item())
            counts.append(y.numel())
    finally:
        model.train(was_training)
    return weighted_validation_loss(losses, counts), sum(counts)
