"""M2：除原作业数值快照外，检查因果性、维度、梯度和 RoPE 不变量。"""

import torch

from transformer_lab_code.attention import RotaryPositionalEmbedding, causal_mask, scaled_dot_product_attention
from transformer_lab_code.layers import RMSNorm, softmax
from transformer_lab_code.model import ModelConfig, TransformerLM


def tiny_model():
    return TransformerLM(ModelConfig(vocab_size=32, context_length=16, d_model=16, num_layers=2, num_heads=4, d_ff=32))


def test_causality_and_gradients():
    torch.manual_seed(17)
    model = tiny_model()
    ids = torch.randint(0, 32, (2, 8))
    changed = ids.clone()
    changed[:, 5:] = (changed[:, 5:] + 1) % 32
    a, b = model(ids), model(changed)
    assert a.shape == (2, 8, 32)
    torch.testing.assert_close(a[:, :5], b[:, :5], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(a[:, :5], model(ids[:, :5]), atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_mask_direction_and_attention_values():
    assert causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    q, k = torch.zeros(1, 3, 2), torch.zeros(1, 3, 2)
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [8.0, 12.0]]])
    result = scaled_dot_product_attention(q, k, v, causal_mask(3))
    torch.testing.assert_close(result, torch.tensor([[[1.0, 2.0], [2.0, 3.0], [4.0, 6.0]]]))


def test_rope_zero_position_and_norm():
    x = torch.randn(2, 3, 5, 8)
    rope = RotaryPositionalEmbedding(10000, 8, 16)
    y = rope(x, torch.arange(5).expand(2, 5))
    torch.testing.assert_close(x[..., 0, :], y[..., 0, :])
    torch.testing.assert_close(x.square().sum(-1), y.square().sum(-1))


def test_normalization_and_softmax_axes():
    norm = RMSNorm(8)
    x = torch.full((2, 3, 8), 300.0, dtype=torch.float16)
    output = norm(x)
    assert output.dtype == torch.float16 and torch.isfinite(output).all()
    scores = torch.randn(2, 3, 4) + 10000
    for dim in (0, 1, -1):
        torch.testing.assert_close(softmax(scores, dim), torch.softmax(scores, dim))
