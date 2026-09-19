"""M3：从 logits 计算稳定的交叉熵。"""

import torch  # noqa: F401 -- O1 可使用 torch 操作


def cross_entropy(logits, targets):
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets 的形状必须等于 logits 除最后一维外的形状")
    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    flat_targets = targets.reshape(-1).long()
    if not flat_targets.numel():
        raise ValueError("不能对空 batch 计算损失")
    return cross_entropy_rows(flat_logits, flat_targets)


def target_scores(scores, targets):
    """已提供：从每行取出该行目标类别的分数，返回 [N]。"""
    return scores.gather(-1, targets[:, None]).squeeze(-1)


def cross_entropy_rows(logits, targets):
    """O1：[N,V] logits、[N] targets → 平均交叉熵标量。

    每行先减最大值，再计算 log(sum(exp(shifted))) - target_score(shifted)。
    用 target_scores 取目标项；不要先 softmax 再 log，也不要调用 F.cross_entropy。
    不能 detach：M4 会通过这个标量反向传播。
    """
    raise NotImplementedError("TODO O1: cross_entropy_rows")
