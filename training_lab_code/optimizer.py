"""M3：状态与参数组管理已提供；学生只填写 AdamW 的数学更新。"""

import math  # noqa: F401 -- O3/O4 使用

import torch


def update_moments(grad, m, v, beta1, beta2):
    """O2：返回 (new_m, new_v)，分别为梯度及梯度平方的指数移动平均。

    不要求原地修改；torch.sqrt / 平方 / 逐元素运算都可用。
    """
    raise NotImplementedError("TODO O2: update_moments")


def update_parameter(parameter, m, v, t, lr, beta1, beta2, eps, weight_decay):
    """O3：m/v 已更新；t 从 1 开始，返回新参数 Tensor。

    与本作业 Algorithm 1 一致：
    alpha = lr * sqrt(1-beta2**t) / (1-beta1**t)
    先对旧 parameter 施加解耦 weight decay，再减 alpha*m/(sqrt(v)+eps)。
    eps 放在未作偏差修正的 sqrt(v) 外。不要把 weight decay 混进 grad。
    """
    raise NotImplementedError("TODO O3: update_parameter")


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("无效的 AdamW 超参数")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise ValueError("本实验仅支持 dense gradients")
                state = self.state[p]
                if not state:
                    state.update(t=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
                state["t"] += 1
                m, v = update_moments(p.grad, state["m"], state["v"], beta1, beta2)
                state["m"].copy_(m)
                state["v"].copy_(v)
                p.copy_(
                    update_parameter(
                        p,
                        state["m"],
                        state["v"],
                        state["t"],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["weight_decay"],
                    )
                )
        return loss


def cosine_schedule(step, max_lr, min_lr, warmup_steps, total_steps):
    """O4：返回标量学习率，定义见 docs/m3_optimization.md。

    step 是已完成的更新次数，从 0 开始；warmup_steps=0 时直接进入 cosine。
    step > total_steps 时返回 min_lr。用 math.cos、math.pi 即可。
    """
    if step < 0 or not 0 <= warmup_steps < total_steps or not 0 <= min_lr <= max_lr:
        raise ValueError("无效的学习率日程")
    raise NotImplementedError("TODO O4: cosine_schedule")


def clipping_scale(grads, max_norm, eps=1e-6):
    """O5：所有梯度共同组成一个长向量，返回 (global_norm, scale)。

    global_norm = sqrt(sum_i sum(g_i.float()**2))。
    未超过 max_norm 时 scale=1，否则 scale=max_norm/(global_norm+eps)。
    grads 是非空列表；不要分别裁剪每个参数。
    """
    raise NotImplementedError("TODO O5: clipping_scale")


@torch.no_grad()
def clip_gradients(parameters, max_norm):
    if max_norm <= 0:
        raise ValueError("max_norm 必须为正数")
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    norm, scale = clipping_scale(grads, max_norm)
    if not torch.isfinite(torch.as_tensor(norm)):
        raise FloatingPointError("梯度范数非有限；停止更新以便排查")
    for grad in grads:
        grad.mul_(scale)
    return norm
