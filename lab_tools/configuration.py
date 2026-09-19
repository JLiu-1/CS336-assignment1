"""已提供：严格读取配置，防止参数名拼错却静默使用默认值。"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transformer_lab_code.model import ModelConfig


@dataclass
class TrainConfig:
    batch_size: int = 8
    gradient_accumulation: int = 4
    total_steps: int = 5000
    schedule_steps: int = 5000
    max_lr: float = 0.001
    min_lr: float = 0.0001
    warmup_steps: int = 100
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.01
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    eval_every: int = 100
    eval_batches: int = 64
    eval_batch_size: int = 8
    eval_context_length: int = 256
    save_every: int = 100
    seed: int = 42
    precision: str = "fp32"
    cpu_threads: int = 2

    def __post_init__(self):
        for name in (
            "batch_size",
            "gradient_accumulation",
            "total_steps",
            "schedule_steps",
            "eval_every",
            "eval_batches",
            "eval_batch_size",
            "eval_context_length",
            "save_every",
            "cpu_threads",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if not 0 <= self.warmup_steps < self.schedule_steps:
            raise ValueError("要求 0 <= warmup_steps < schedule_steps")
        if self.schedule_steps < self.total_steps:
            raise ValueError("schedule_steps 不能小于 total_steps；短试跑可沿用较长日程")
        if self.precision not in ("fp32", "bf16"):
            raise ValueError("precision 只能是 fp32 或 bf16")
        if not 0 <= self.min_lr <= self.max_lr or self.max_lr <= 0 or self.max_grad_norm <= 0:
            raise ValueError("学习率/梯度阈值无效")


def load_config(path, vocab_size):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(obj) - {"model", "training", "description"}:
        raise ValueError("配置顶层仅支持 model、training、description")
    model_values = dict(obj["model"])
    if model_values.get("vocab_size") == 0:
        model_values["vocab_size"] = vocab_size
    model = ModelConfig(**model_values)
    if model.vocab_size != vocab_size:
        raise ValueError("模型 vocab_size 与数据词表不同；demo 用 0 自动读取")
    training = TrainConfig(**obj["training"])
    return model, training


def resolved_config(model, training):
    return {"model": asdict(model), "training": asdict(training)}
