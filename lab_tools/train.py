"""训练入口：复用 M1–M4 的学生实现，框架负责配置、记录和恢复。"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from tokenizer_lab_code.tokenizer_lab.storage import write_json
from training_lab_code.checkpoint import read_checkpoint, restore_rng, save_checkpoint
from training_lab_code.data import Dataset, get_batch
from training_lab_code.evaluation import evaluate
from training_lab_code.optimizer import AdamW, cosine_schedule
from training_lab_code.student import train_step
from transformer_lab_code.model import TransformerLM

from .configuration import load_config, resolved_config


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in (
        "tokenizer_lab_code",
        "transformer_lab_code",
        "training_lab_code",
        "generation_lab_code",
        "lab_tools",
    ):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def run(config_path, data_dir, out_dir, device="cpu", resume=None, stop_after=None, max_seconds=None):
    start = time.perf_counter()
    if stop_after is not None and stop_after <= 0 or max_seconds is not None and max_seconds <= 0:
        raise ValueError("停止步数/时间预算必须为正")
    dataset = Dataset(data_dir)
    mc, tc = load_config(config_path, dataset.manifest["vocab_size"])
    if torch.device(device).type == "cpu":
        torch.set_num_threads(tc.cpu_threads)
    dataset.check_context(mc.context_length)
    if tc.eval_context_length > mc.context_length:
        raise ValueError("eval_context_length 不能超过模型 context_length")
    config = resolved_config(mc, tc)
    if tc.precision == "bf16" and (not str(device).startswith("cuda") or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 配置需要支持 BF16 的 CUDA 平台；CPU 使用 fp32")
    seed_all(tc.seed)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    model = TransformerLM(mc, device=device)
    optimizer = AdamW(
        model.parameters(), lr=tc.max_lr, betas=(tc.beta1, tc.beta2), eps=tc.eps, weight_decay=tc.weight_decay
    )
    out = Path(out_dir)
    if not resume and out.exists() and any(out.iterdir()):
        raise FileExistsError("训练输出目录非空；恢复请用 --resume，新实验请更换 --out")
    out.mkdir(parents=True, exist_ok=True)
    history, completed, prior_seconds = [], 0, 0.0
    code_hash = source_digest()
    if resume:
        payload = read_checkpoint(resume)
        extra = payload["extra"]
        if extra["config"] != config or extra["data_identity"] != dataset.identity or extra["code_sha256"] != code_hash:
            raise ValueError("恢复时配置、数据或代码与 checkpoint 不同；新实验请重新训练")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        completed, history = payload["iteration"], extra["history"]
        prior_seconds = extra["wall_seconds"]
    write_json(out / "config.json", config)
    write_json(
        out / "provenance.json",
        dict(
            data_identity=dataset.identity,
            tokenizer_sha256=dataset.manifest["tokenizer_sha256"],
            code_sha256=code_hash,
            torch_version=str(torch.__version__),
            device=str(device),
            parameters=sum(p.numel() for p in model.parameters()),
            time_budget_seconds=max_seconds,
        ),
    )
    tokens_per_step = tc.batch_size * tc.gradient_accumulation * mc.context_length
    final_step = min(tc.total_steps, stop_after) if stop_after is not None else tc.total_steps
    if final_step < completed:
        raise ValueError("--stop-after 不能小于 checkpoint 已完成步数")

    def elapsed():
        synchronize(device)
        return prior_seconds + time.perf_counter() - start

    def save():
        save_checkpoint(
            model,
            optimizer,
            completed,
            out / "last.pt",
            config=config,
            data_identity=dataset.identity,
            code_sha256=code_hash,
            history=history,
            wall_seconds=elapsed(),
        )
        with (out / "metrics.jsonl").open("w", encoding="utf-8") as stream:
            for entry in history:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")

    if not history:
        loss, count = evaluate(
            model, dataset.valid, tc.eval_context_length, tc.eval_batch_size, tc.eval_batches, device, tc.precision
        )
        history.append(dict(step=0, tokens=0, wall_seconds=elapsed(), val_loss=loss, val_tokens=count, lr=0.0))
    status = "complete"
    model.train()
    try:
        while completed < final_step:
            if max_seconds is not None and time.perf_counter() - start >= max_seconds:
                status = "budget_exhausted"
                break
            lr = cosine_schedule(completed, tc.max_lr, tc.min_lr, tc.warmup_steps, tc.schedule_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            # 逐个构造 microbatch；课程配置只保留少量输入 ID，不保存各步计算图。
            batches = [
                get_batch(dataset.train, tc.batch_size, mc.context_length, device)
                for _ in range(tc.gradient_accumulation)
            ]
            loss = float(train_step(model, optimizer, batches, tc.max_grad_norm, tc.precision))
            if not math.isfinite(loss):
                raise FloatingPointError("训练损失非有限")
            completed += 1
            entry = dict(
                step=completed, tokens=completed * tokens_per_step, wall_seconds=elapsed(), train_loss=loss, lr=lr
            )
            if completed % tc.eval_every == 0 or completed == final_step:
                value, count = evaluate(
                    model,
                    dataset.valid,
                    tc.eval_context_length,
                    tc.eval_batch_size,
                    tc.eval_batches,
                    device,
                    tc.precision,
                )
                entry.update(val_loss=value, val_tokens=count)
                entry["wall_seconds"] = elapsed()
                print(f"step={completed} tokens={entry['tokens']} train={loss:.4f} valid={value:.4f}", flush=True)
            history.append(entry)
            if completed % tc.save_every == 0:
                save()
        if completed < tc.total_steps and status == "complete":
            status = "paused"
        # 时间预算停止可能不在评估点，仍记录当前 checkpoint 的验证结果。
        if history[-1]["step"] == completed and "val_loss" not in history[-1]:
            value, count = evaluate(
                model, dataset.valid, tc.eval_context_length, tc.eval_batch_size, tc.eval_batches, device, tc.precision
            )
            history[-1].update(val_loss=value, val_tokens=count, wall_seconds=elapsed())
        save()
    except (FloatingPointError, KeyboardInterrupt) as error:
        status = "diverged" if isinstance(error, FloatingPointError) else "interrupted"
        # 不把可能已损坏的状态覆盖到最后一个有效 checkpoint。
        write_json(out / "failure.json", dict(status=status, completed_steps=completed, reason=str(error)))
        raise
    last_val = next(row for row in reversed(history) if "val_loss" in row)
    seconds = elapsed()
    summary = dict(
        status=status,
        step=completed,
        tokens=completed * tokens_per_step,
        wall_seconds=seconds,
        val_loss=last_val["val_loss"],
        val_step=last_val["step"],
        val_tokens=last_val["val_tokens"],
        perplexity=math.exp(min(last_val["val_loss"], 80)),
        tokens_per_second=completed * tokens_per_step / max(seconds, 1e-9),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
    )
    write_json(out / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="本次停在这个全局步数；不改变学习率日程")
    parser.add_argument("--max-seconds", type=float, help="本次调用的软时间上限；当前步、验证和保存可略超时")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.config, args.data, args.out, args.device, args.resume, args.stop_after, args.max_seconds), indent=2
        )
    )


if __name__ == "__main__":
    main()
