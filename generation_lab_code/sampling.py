"""M4：循环、排序、采样和索引已提供，填写 G1–G3。"""

import torch

from transformer_lab_code.layers import softmax  # noqa: F401 -- G2 使用


def last_logits(logits):
    """G1：[B,T,V] → [B,V]。生成时只使用最后一个位置的预测。"""
    raise NotImplementedError("TODO G1: last_logits")


def temperature_probabilities(logits, temperature):
    """G2：temperature > 0；调用自己的 softmax，返回最后一维上的概率。

    greedy 分支由框架单独处理，不能把 temperature=0 代入除法。
    """
    raise NotImplementedError("TODO G2: temperature_probabilities")


def nucleus_keep(sorted_probs, top_p):
    """G3：返回 bool mask，保留概率降序排列后累计达到 top_p 的最短前缀。

    输入 [...,V]，各行和为 1；0 < top_p <= 1。
    必须包含首次使累计概率达到/超过阈值的 token，至少保留一个。
    提示：判断某项之前的累计概率是否小于 top_p。torch.cumsum 可用。
    """
    raise NotImplementedError("TODO G3: nucleus_keep")


def sample_next(logits, temperature=1.0, top_p=1.0, generator=None):
    if temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("temperature >= 0 且 0 < top_p <= 1")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = temperature_probabilities(logits.float(), temperature)
    sorted_probs, indices = probs.sort(dim=-1, descending=True)
    keep = nucleus_keep(sorted_probs, top_p)
    retained = sorted_probs * keep
    retained = retained / retained.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(retained, 1, generator=generator)
    return indices.gather(-1, sampled)


@torch.no_grad()
def generate(model, token_ids, max_new_tokens=128, temperature=1.0, top_p=1.0, eos_id=None, seed=0):
    """单条 prompt；超长时采用滑动窗口并重新从 0 编号，不实现 KV cache。"""
    if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
        raise ValueError("生成接口要求非空 [1,T] prompt")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负")
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=token_ids.device).manual_seed(seed)
    try:
        for _ in range(max_new_tokens):
            context = token_ids[:, -model.config.context_length :]
            token = sample_next(last_logits(model(context)), temperature, top_p, generator)
            token_ids = torch.cat((token_ids, token), dim=-1)
            if eos_id is not None and token.item() == eos_id:
                break
        return token_ids
    finally:
        model.train(was_training)
