"""M5：研究设施检查。研究是否完成还需提交清单与报告。"""

import json

import pytest
import torch

from lab_tools.submission import validate_run
from transformer_lab_code.model import ModelConfig, TransformerLM


@pytest.mark.parametrize(
    "options", [dict(use_rope=False), dict(use_norm=False), dict(ffn_type="silu", d_ff=64), dict(tie_embeddings=True)]
)
def test_experiment_switches_have_gradients(options):
    values = dict(vocab_size=32, context_length=8, d_model=16, num_heads=2, num_layers=1, d_ff=48)
    values.update(options)
    model = TransformerLM(ModelConfig(**values))
    model(torch.randint(0, 32, (2, 8))).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if options.get("tie_embeddings"):
        assert model.lm_head.weight is model.token_embeddings.weight


def test_incomplete_run_rejected(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps(dict(status="paused")))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "provenance.json").write_text("{}")
    (tmp_path / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="未按计划完成"):
        validate_run(tmp_path)
