"""将 metrics.jsonl 绘制为 SVG，不需要额外画图库。"""

import argparse
import html
import json
from pathlib import Path


def plot(runs, out, x_axis="tokens"):
    if x_axis not in ("tokens", "step", "wall_seconds"):
        raise ValueError("横轴应为 tokens、step 或 wall_seconds")
    series = []
    for run in runs:
        path = Path(run)
        rows = [json.loads(s) for s in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
        points = [(r[x_axis], r["val_loss"]) for r in rows if "val_loss" in r]
        if not points:
            raise ValueError(f"{run} 没有验证记录")
        series.append((path.name, points))
    xmax = max(1, max(x for _, points in series for x, _ in points))
    ymin = min(y for _, points in series for _, y in points)
    ymax = max(y for _, points in series for _, y in points)
    margin = max((ymax - ymin) * 0.1, 0.1)
    ymin, ymax = ymin - margin, ymax + margin
    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"]
    height = 460 + 22 * len(series)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}" viewBox="0 0 900 {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="14" fill="#172033">',
        '<text x="70" y="30" font-size="21">Validation loss</text>',
    ]
    for i in range(6):
        x, y = 80 + 760 * i / 5, 380 - 320 * i / 5
        parts += [
            f'<path d="M80 {y} H840" stroke="#e2e8f0"/>',
            f'<text x="15" y="{y + 5}">{ymin + (ymax - ymin) * i / 5:.3f}</text>',
            f'<text x="{x - 20}" y="410">{xmax * i / 5:.2g}</text>',
        ]
    parts += [f'<text x="400" y="435">{x_axis}</text>']
    for i, (label, points) in enumerate(series):
        coordinates = " ".join(
            f"{80 + x / xmax * 760:.2f},{380 - (y - ymin) / (ymax - ymin) * 320:.2f}" for x, y in points
        )
        color = colors[i % len(colors)]
        parts += [
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>',
            f'<text x="80" y="{465 + i * 22}" fill="{color}">{html.escape(label)}</text>',
        ]
    parts.append("</g></svg>")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--x", choices=("tokens", "step", "wall_seconds"), default="tokens")
    args = parser.parse_args()
    plot(args.runs, args.out, args.x)


if __name__ == "__main__":
    main()
