"""M4 实验：固定 batch 过拟合，以及 CPU 连续训练/恢复训练一致性。"""

import argparse
from pathlib import Path

import torch

from tokenizer_lab_code.tokenizer_lab.storage import write_json
from training_lab_code.checkpoint import load_checkpoint, save_checkpoint
from training_lab_code.data import Dataset, get_batch
from training_lab_code.optimizer import AdamW
from training_lab_code.student import forward_loss, train_step
from transformer_lab_code.model import ModelConfig, TransformerLM

from .train import seed_all


def verify(data_dir, out):
    torch.set_num_threads(2)
    dataset = Dataset(data_dir)
    config = ModelConfig(
        vocab_size=dataset.manifest["vocab_size"], context_length=16, d_model=32, num_layers=1, num_heads=4, d_ff=96
    )
    dataset.check_context(16)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(123)
    model = TransformerLM(config)
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0)
    x, y = get_batch(dataset.train, 2, 16)
    before = forward_loss(model, x, y).item()
    losses = [train_step(model, optimizer, [(x, y)]) for _ in range(100)]
    after = forward_loss(model, x, y).item()
    if not after < before * 0.5:
        raise AssertionError("固定 batch 的 loss 未明显下降；先检查 S2 和前面各阶段")

    seed_all(321)
    continuous = TransformerLM(config)
    continuous_opt = AdamW(continuous.parameters())
    for _ in range(3):
        train_step(continuous, continuous_opt, [get_batch(dataset.train, 2, 16)])
    checkpoint = out / "recovery_start.pt"
    save_checkpoint(continuous, continuous_opt, 3, checkpoint)
    tail_losses = [train_step(continuous, continuous_opt, [get_batch(dataset.train, 2, 16)]) for _ in range(3)]
    resumed = TransformerLM(config)
    resumed_opt = AdamW(resumed.parameters())
    load_checkpoint(checkpoint, resumed, resumed_opt, restore_random=True)
    resumed_losses = [train_step(resumed, resumed_opt, [get_batch(dataset.train, 2, 16)]) for _ in range(3)]
    differences = [
        (a - b).abs().max().item() for a, b in zip(continuous.parameters(), resumed.parameters(), strict=True)
    ]
    if max(differences) > 1e-6 or max(abs(a - b) for a, b in zip(tail_losses, resumed_losses, strict=True)) > 1e-6:
        raise AssertionError("CPU 恢复后的参数/loss 不一致")
    result = dict(
        data_identity=dataset.identity,
        overfit_initial_loss=before,
        overfit_final_loss=after,
        overfit_steps=100,
        overfit_losses=losses,
        recovery_max_abs_diff=max(differences),
        continuous_losses=tail_losses,
        resumed_losses=resumed_losses,
        device="cpu",
    )
    write_json(out / "verification.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = verify(args.data, args.out)
    print(f"overfit: {result['overfit_initial_loss']:.4f} → {result['overfit_final_loss']:.4f}; recovery passed")


if __name__ == "__main__":
    main()
