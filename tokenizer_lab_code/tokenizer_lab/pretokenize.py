"""教师提供：由原 tests/parallel_tokenizer.py 整理而来，默认单进程。

多特殊 token 时保守退回串行，避免物理切块切入另一个较长特殊 token。
"""

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from itertools import pairwise

import regex

from .common import validate_special_tokens

PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@lru_cache(maxsize=32)
def _special_pattern(tokens: tuple[str, ...]):
    return regex.compile("(" + "|".join(regex.escape(t) for t in sorted(tokens, key=len, reverse=True)) + ")")


def split_special(text: str, special_tokens: list[str]):
    """产生 (是否特殊 token, 原文片段)，保留最长匹配的特殊 token。"""
    if not special_tokens:
        if text:
            yield False, text
        return
    cursor = 0
    for match in _special_pattern(tuple(special_tokens)).finditer(text):
        if match.start() > cursor:
            yield False, text[cursor : match.start()]
        yield True, match.group()
        cursor = match.end()
    if cursor < len(text):
        yield False, text[cursor:]


def _find_chunk_boundaries(filename: str, num_chunks: int, boundary_token: bytes) -> list[int]:
    if num_chunks < 1 or not boundary_token:
        raise ValueError("分块数必须为正，边界 token 不能为空")
    size = os.path.getsize(filename)
    boundaries = [0, size]
    with open(filename, "rb") as stream:
        for i in range(1, num_chunks):
            pos = size * i // num_chunks
            while pos < size:
                stream.seek(pos)
                block = stream.read(max(4096, len(boundary_token)))
                index = block.find(boundary_token)
                if index >= 0:
                    pos += index
                    break
                if pos + len(block) >= size:
                    pos = size
                    break
                pos += max(1, len(block) - len(boundary_token) + 1)
            boundaries.append(pos)
    return sorted(set(boundaries))


def _process_chunk(args) -> Counter[bytes]:
    filename, start, end, specials, pattern = args
    with open(filename, "rb") as stream:
        stream.seek(start)
        text = stream.read(end - start).decode("utf-8")
    counts = Counter()
    compiled = regex.compile(pattern)
    for is_special, piece in split_special(text, specials):
        if not is_special:
            counts.update(match.group().encode("utf-8") for match in compiled.finditer(piece))
    return counts


def parallel_pretokenize(
    input_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
    num_workers: int = 1,
    pattern: str = PATTERN,
) -> Counter[bytes]:
    specials = validate_special_tokens(special_tokens)
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers 必须是正整数")
    regex.compile(pattern)  # 启动子进程前检查表达式。
    filename = os.fspath(input_path)
    size = os.path.getsize(filename)
    if num_workers == 1 or len(specials) != 1:
        return _process_chunk((filename, 0, size, specials, pattern))
    boundaries = _find_chunk_boundaries(filename, num_workers, specials[0].encode("utf-8"))
    tasks = [(filename, a, b, specials, pattern) for a, b in pairwise(boundaries) if a < b]
    if len(tasks) <= 1:
        return _process_chunk((filename, 0, size, specials, pattern))
    total = Counter()
    with ProcessPoolExecutor(max_workers=min(num_workers, len(tasks))) as executor:
        for counts in executor.map(_process_chunk, tasks):
            total.update(counts)
    return total
