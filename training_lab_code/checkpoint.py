"""已提供：只加载自己或课程可信来源的 checkpoint，文件可包含 Python 对象。"""

import os
import random
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def read_checkpoint(src):
    return torch.load(src, map_location="cpu", weights_only=False)


def save_checkpoint(model, optimizer, iteration, out, **extra):
    payload = dict(
        model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration, rng=rng_state(), extra=extra
    )
    if hasattr(out, "write"):
        torch.save(payload, out)
    else:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)


def load_checkpoint(src, model, optimizer, restore_random=False):
    payload = read_checkpoint(src)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if restore_random:
        restore_rng(payload["rng"])
    return payload["iteration"]
