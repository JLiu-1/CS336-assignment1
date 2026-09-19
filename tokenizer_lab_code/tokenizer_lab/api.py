"""原作业 adapter 和教学 CLI 共用的入口。"""

import os
from collections import Counter

from . import incremental, naive
from .pretokenize import PATTERN, parallel_pretokenize


def train_counts(
    counts: Counter[bytes],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    *,
    variant: str = "naive",
    trace: bool = False,
    debug: bool = False,
):
    if variant == "naive":
        return naive.train(counts, vocab_size, special_tokens, trace=trace)
    if variant == "incremental":
        return incremental.train(counts, vocab_size, special_tokens, trace=trace, debug=debug)
    raise ValueError("variant 必须是 naive 或 incremental")


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    *,
    variant: str = "naive",
    num_workers: int = 1,
    pattern: str = PATTERN,
    debug: bool = False,
):
    counts = parallel_pretokenize(input_path, special_tokens, num_workers, pattern)
    result = train_counts(counts, vocab_size, special_tokens, variant=variant, debug=debug)
    return result.vocab, result.merges
