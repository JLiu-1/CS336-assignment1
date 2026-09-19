"""这些测试不调用任何学生 TODO；用于确认框架和环境可用。"""

import io
import json

import numpy as np
import pytest
import torch

from lab_tools.check import targets
from lab_tools.configuration import TrainConfig, load_config
from lab_tools.plot import plot
from training_lab_code.checkpoint import load_checkpoint, save_checkpoint
from transformer_lab_code.attention import apply_mask, merge_heads, split_heads
from transformer_lab_code.model import ModelConfig, TransformerLM


def test_model_parameters_registered():
    config = ModelConfig(vocab_size=288, context_length=32, d_model=64, num_layers=2, num_heads=4, d_ff=192)
    model = TransformerLM(config)
    expected = 2 * 288 * 64 + 2 * (4 * 64**2 + 3 * 64 * 192 + 2 * 64) + 64
    assert sum(p.numel() for p in model.parameters()) == expected
    assert "layers.0.attn.q_proj.weight" in model.state_dict()
    assert not any("rope" in name for name in model.state_dict())
    assert len(list(model.named_buffers())) == 4


def test_heads_roundtrip_and_mask():
    x = torch.arange(2 * 3 * 16).reshape(2, 3, 16)
    assert split_heads(x, 4).shape == (2, 4, 3, 4)
    torch.testing.assert_close(merge_heads(split_heads(x, 4)), x)
    mask = torch.tensor([[True, False]])
    scores = apply_mask(torch.zeros(2, 1, 2), mask)
    assert (scores[..., 0] == 0).all() and torch.isneginf(scores[..., 1]).all()


def test_checkpoint_filelike_and_rng():
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    stream = io.BytesIO()
    save_checkpoint(model, optimizer, 7, stream)
    expected_torch, expected_numpy = torch.rand(3), np.random.rand(3)
    stream.seek(0)
    restored = torch.nn.Linear(2, 2)
    assert load_checkpoint(stream, restored, optimizer, restore_random=True) == 7
    torch.testing.assert_close(torch.rand(3), expected_torch)
    np.testing.assert_array_equal(np.random.rand(3), expected_numpy)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)


def test_stage_boundaries():
    assert all("tokenizer" in name for name in targets(1))
    assert "tests/test_optimizer.py" not in targets(2)
    assert "tests/test_data.py" not in targets(3)
    assert "milestone_tests/test_m5.py" not in targets(4)
    assert set(targets(4)) < set(targets(5))


def test_configuration_rejects_typo_and_invalid_dims(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"vocab_size": 0}, "training": {"batc_size": 4}}))
    with pytest.raises(TypeError):
        load_config(path, 288)
    with pytest.raises(ValueError):
        ModelConfig(d_model=65)
    with pytest.raises(ValueError):
        TrainConfig(gradient_accumulation=0)


def test_svg_plot(tmp_path):
    run = tmp_path / "a"
    run.mkdir()
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(dict(step=i, tokens=i * 8, wall_seconds=i, val_loss=3 - i)) for i in range(2))
    )
    plot([run], tmp_path / "curve.svg")
    assert "polyline" in (tmp_path / "curve.svg").read_text()
