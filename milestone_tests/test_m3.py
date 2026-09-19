"""M3：边界和数值测试；不依赖 M4 TODO。"""

import pytest
import torch

from training_lab_code.losses import cross_entropy
from training_lab_code.optimizer import AdamW, clip_gradients, cosine_schedule, update_moments, update_parameter


def test_cross_entropy_gradient_and_batch_dims():
    x = (torch.randn(2, 3, 5) * 100).requires_grad_()
    y = torch.tensor([[0, 1, 2], [3, 4, 0]])
    actual = cross_entropy(x, y)
    expected = torch.nn.functional.cross_entropy(x.reshape(-1, 5), y.reshape(-1))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0])


def test_adamw_hand_calculation():
    grad = torch.tensor([2.0])
    m, v = update_moments(grad, torch.zeros(1), torch.zeros(1), 0.9, 0.99)
    torch.testing.assert_close(m, torch.tensor([0.2]))
    torch.testing.assert_close(v, torch.tensor([0.04]))
    # t=1, epsilon 明显非零，固定本课程的 Algorithm 1 约定。
    actual = update_parameter(torch.tensor([1.0]), m, v, 1, 0.1, 0.9, 0.99, 0.01, 0.2)
    torch.testing.assert_close(actual, torch.tensor([0.98 - 0.1 * 0.2 / 0.21]))


def test_no_gradient_is_not_decayed():
    active = torch.nn.Parameter(torch.tensor([2.0]))
    frozen = torch.nn.Parameter(torch.tensor([3.0]))
    opt = AdamW([active, frozen], weight_decay=0.5)
    active.square().sum().backward()
    opt.step()
    assert frozen.item() == 3 and frozen not in opt.state
    assert opt.state[active]["t"] == 1


def test_global_clipping():
    p, q = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    p.grad, q.grad = torch.tensor([3.0]), torch.tensor([4.0])
    norm = clip_gradients([p, q], 2.5)
    assert float(norm) == pytest.approx(5)
    assert p.grad.item() == pytest.approx(1.5, abs=1e-6)
    assert q.grad.item() == pytest.approx(2, abs=1e-6)
    before = p.grad.clone()
    clip_gradients([p, q], 10)
    torch.testing.assert_close(p.grad, before)


def test_schedule_zero_warmup_and_boundaries():
    assert cosine_schedule(0, 1.0, 0.1, 0, 10) == pytest.approx(1)
    assert cosine_schedule(0, 1.0, 0.1, 2, 10) == pytest.approx(0)
    assert cosine_schedule(2, 1.0, 0.1, 2, 10) == pytest.approx(1)
    assert cosine_schedule(10, 1.0, 0.1, 2, 10) == pytest.approx(0.1)
    assert cosine_schedule(11, 1.0, 0.1, 2, 10) == pytest.approx(0.1)
