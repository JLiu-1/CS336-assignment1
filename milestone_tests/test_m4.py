"""M4：训练/采样小测试，以及真正经过 M1→M4 的端到端实验。"""

import copy
import json

import numpy as np
import pytest
import torch

from generation_lab_code.sampling import generate, last_logits, nucleus_keep, sample_next
from lab_tools.generate import run as generate_run
from lab_tools.prepare import demo
from lab_tools.train import run
from lab_tools.verify_training import verify
from training_lab_code.checkpoint import read_checkpoint
from training_lab_code.optimizer import AdamW
from training_lab_code.student import shifted_windows, train_step, weighted_validation_loss
from transformer_lab_code.model import ModelConfig, TransformerLM


def test_shift_and_weighted_validation():
    x, y = shifted_windows(np.arange(10), np.array([0, 6]), 3)
    np.testing.assert_array_equal(x, [[0, 1, 2], [6, 7, 8]])
    np.testing.assert_array_equal(y, [[1, 2, 3], [7, 8, 9]])
    assert weighted_validation_loss([2.0, 8.0], [6, 2]) == pytest.approx(3.5)


def test_gradient_accumulation_matches_full_batch():
    torch.manual_seed(12)
    config = ModelConfig(vocab_size=12, context_length=4, d_model=8, num_layers=1, num_heads=2, d_ff=16)
    a, b = TransformerLM(config), TransformerLM(config)
    b.load_state_dict(a.state_dict())
    oa, ob = AdamW(a.parameters()), AdamW(b.parameters())
    x, y = torch.randint(0, 12, (4, 4)), torch.randint(0, 12, (4, 4))
    for _ in range(2):
        la = train_step(a, oa, [(x, y)])
        lb = train_step(b, ob, [(x[:2], y[:2]), (x[2:], y[2:])])
        assert la == pytest.approx(lb, abs=1e-5)
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, atol=2e-6, rtol=1e-4)


def test_sampling_boundaries():
    probs = torch.tensor([[0.6, 0.25, 0.15]])
    assert nucleus_keep(probs, 0.7).tolist() == [[True, True, False]]
    assert nucleus_keep(probs, 0.6).tolist() == [[True, False, False]]
    assert nucleus_keep(probs, 0.01).tolist() == [[True, False, False]]
    assert nucleus_keep(probs, 1.0).all()
    scores = torch.tensor([[0.0, 10.0, 1.0]])
    assert sample_next(scores, temperature=0).item() == 1
    assert sample_next(scores, temperature=1, top_p=0.01).item() == 1
    sequence = torch.arange(24).reshape(2, 3, 4)
    torch.testing.assert_close(last_logits(sequence), sequence[:, -1])


def test_generation_eos_and_context():
    class Fixed(torch.nn.Module):
        config = ModelConfig(vocab_size=3, context_length=2, d_model=4, num_heads=1)

        def forward(self, ids):
            assert ids.shape[-1] <= 2
            result = torch.zeros(*ids.shape, 3)
            result[..., 2] = 10
            return result

    model = Fixed()
    prompt = torch.tensor([[0, 1, 0]])
    assert generate(model, prompt, 9, temperature=0, eos_id=2).tolist() == [[0, 1, 0, 2]]
    assert generate(model, prompt, 0).tolist() == prompt.tolist()
    assert model.training


def test_end_to_end_and_resume(tmp_path):
    data = tmp_path / "data"
    demo(data)  # 调用自己的 A1–A4/C1，再供模型使用。
    config = {
        "model": dict(vocab_size=0, context_length=16, d_model=16, num_layers=1, num_heads=2, d_ff=32),
        "training": dict(
            batch_size=2,
            gradient_accumulation=1,
            total_steps=6,
            schedule_steps=6,
            warmup_steps=0,
            eval_every=3,
            eval_batches=1,
            eval_batch_size=2,
            eval_context_length=16,
            save_every=3,
        ),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    run(path, data, tmp_path / "full")
    first = run(path, data, tmp_path / "split", stop_after=3)
    assert first["status"] == "paused"
    second = run(path, data, tmp_path / "split", resume=tmp_path / "split/last.pt")
    assert second["status"] == "complete"
    full, resumed = read_checkpoint(tmp_path / "full/last.pt"), read_checkpoint(tmp_path / "split/last.pt")
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=1e-7, rtol=1e-6)
    sample = generate_run(tmp_path / "full/last.pt", data, "Once", tmp_path / "sample.json", max_new_tokens=4)
    assert 1 <= len(sample["continuation_ids"]) <= 4
    modified = copy.deepcopy(config)
    modified["training"]["max_lr"] = 0.1
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="恢复"):
        run(path, data, tmp_path / "split", resume=tmp_path / "split/last.pt")


def test_overfit_and_recovery_experiment(tmp_path):
    demo(tmp_path / "data")
    result = verify(tmp_path / "data", tmp_path / "verification")
    assert result["recovery_max_abs_diff"] <= 1e-6
