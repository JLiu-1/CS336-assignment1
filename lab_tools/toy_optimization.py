"""M3：SGD 学习率、AdamW 收敛与 warmup/cosine 曲线数据。全部在 CPU 运行。"""

import argparse
import math
from pathlib import Path

import torch

from tokenizer_lab_code.tokenizer_lab.storage import write_json
from training_lab_code.optimizer import AdamW, cosine_schedule


def run(out):
    records = {}
    for lr in (1.0, 10.0, 100.0, 1000.0):
        torch.manual_seed(0)
        weights = torch.nn.Parameter(5 * torch.randn(10, 10))
        losses = []
        for step in range(10):
            loss = weights.square().mean()
            if not torch.isfinite(loss):
                losses.append("non-finite")
                break
            losses.append(loss.item())
            loss.backward()
            with torch.no_grad():
                weights.add_(weights.grad, alpha=-lr / math.sqrt(step + 1))
                weights.grad = None
        records[str(lr)] = losses
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(8))
    opt = AdamW([parameter], lr=0.05, weight_decay=0)
    adam_losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = (parameter - 2).square().mean()
        adam_losses.append(loss.item())
        loss.backward()
        opt.step()
    schedule = [cosine_schedule(t, 0.001, 0.0001, 10, 100) for t in range(111)]
    result = dict(sgd=records, adamw=adam_losses, schedule=schedule)
    write_json(Path(out) / "toy_results.json", result)
    # 使用 SVG 直接显示日程，不增加画图库依赖。
    coords = " ".join(f"{40 + 5 * i},{240 - 200 * lr / 0.001}" for i, lr in enumerate(schedule))
    Path(out, "schedule.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="280">'
        '<rect width="100%" height="100%" fill="white"/><text x="40" y="25">Learning rate: warmup + cosine</text>'
        f'<polyline points="{coords}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        '<text x="40" y="265">step 0</text><text x="540" y="265">step 110</text></svg>',
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="var/m3")
    args = parser.parse_args()
    run(args.out)
    print(f"结果已写入 {args.out}")
