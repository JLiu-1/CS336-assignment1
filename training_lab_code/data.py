"""已提供：内存映射、数据身份与 batch 搬运。S1 在 student.py。"""

import json
from pathlib import Path

import numpy as np
import torch

from tokenizer_lab_code.tokenizer_lab.storage import file_sha256

from .student import shifted_windows


def get_batch(data, batch_size, context_length, device="cpu"):
    if data.ndim != 1 or len(data) <= context_length or batch_size <= 0 or context_length <= 0:
        raise ValueError("数据长度必须大于 context_length，且 batch_size/context_length 为正")
    starts = np.random.randint(0, len(data) - context_length, size=batch_size)
    return batch_at(data, starts, context_length, device)


def batch_at(data, starts, context_length, device):
    x, y = shifted_windows(data, starts, context_length)
    return (
        torch.as_tensor(np.array(x, dtype=np.int64), device=device),
        torch.as_tensor(np.array(y, dtype=np.int64), device=device),
    )


class Dataset:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("format") != "cs336-course-data-v1":
            raise ValueError("未知数据格式，请使用 lab_tools.prepare")
        self.identity = file_sha256(self.root / "manifest.json")
        self.tokenizer_path = self.root / "tokenizer.json"
        if verify and file_sha256(self.tokenizer_path) != self.manifest["tokenizer_sha256"]:
            raise ValueError("tokenizer 与 manifest 不匹配")
        for split in ("train", "valid"):
            path = self.root / f"{split}.npy"
            info = self.manifest[split]
            if verify and file_sha256(path) != info["sha256"]:
                raise ValueError(f"{split} 数据校验和不匹配")
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.ndim != 1 or data.dtype.kind not in "ui" or len(data) != info["tokens"]:
                raise ValueError("token 数组形状、dtype 或长度不匹配")
            if len(data) < 2 or int(data.min()) < 0 or int(data.max()) >= self.manifest["vocab_size"]:
                raise ValueError("数据过短或包含超出词表的 ID")
            setattr(self, split, data)

    def check_context(self, context_length):
        if min(len(self.train), len(self.valid)) <= context_length:
            raise ValueError("train/valid 各自至少需要 context_length + 1 个 token")
