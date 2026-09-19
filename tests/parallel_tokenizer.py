from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import os
import regex as re


# GPT-2 / CS336 使用的 pre-tokenization pattern
PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _find_chunk_boundaries(
    filename: str,
    num_chunks: int,
    boundary_token: bytes,
) -> list[int]:
    """
    将文件大致平均切成 num_chunks 份，
    但切分位置必须落在 boundary_token 的开头。

    例如：
        ... document 1 <|endoftext|> document 2 ...
                       ^
                    可以在这里切
    """
    file_size = os.path.getsize(filename)

    if num_chunks <= 1:
        return [0, file_size]

    approximate_chunk_size = file_size // num_chunks
    boundaries = [0]

    with open(filename, "rb") as f:

        for i in range(1, num_chunks):
            pos = i * approximate_chunk_size

            f.seek(pos)

            while pos < file_size:
                block = f.read(4096)

                if not block:
                    pos = file_size
                    break

                index = block.find(boundary_token)

                if index != -1:
                    pos += index
                    break

                # 保留一些 overlap，防止 boundary_token
                # 恰好横跨两个 4096-byte block
                overlap = len(boundary_token) - 1
                step = max(1, len(block) - overlap)

                pos += step
                f.seek(pos)

            boundaries.append(pos)

    boundaries.append(file_size)

    # 不同 approximate position 有可能找到同一个 boundary
    boundaries = sorted(set(boundaries))

    return boundaries


def _process_chunk(args) -> Counter:
    """
    Worker process:
    读取文件的一部分，对其中普通文本执行 pre-tokenization。
    """
    (
        filename,
        start,
        end,
        special_tokens,
        pattern,
    ) = args

    with open(filename, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    text = data.decode("utf-8")

    # ------------------------------------------------
    # special tokens 是 hard boundaries。
    #
    # hello<|endoftext|>world
    #
    # 会变成：
    #
    # hello
    # world
    #
    # special token 本身不参与 BPE statistics。
    # ------------------------------------------------

    if special_tokens:
        special_pattern = "|".join(
            re.escape(token)
            for token in sorted(
                special_tokens,
                key=len,
                reverse=True,
            )
        )

        pieces = re.split(special_pattern, text)

    else:
        pieces = [text]

    regex_pattern = re.compile(pattern)

    counts = Counter()

    for piece in pieces:

        if not piece:
            continue

        for match in regex_pattern.finditer(piece):
            pretoken = match.group()

            # BPE 最终处理的是 bytes
            counts[pretoken.encode("utf-8")] += 1

    return counts


def parallel_pretokenize(
    input_path: str,
    special_tokens: list[str] | None = None,
    num_workers: int = 8,
    pattern: str = PATTERN,
) -> Counter:
    """
    Parallel pre-tokenizer for CS336 BPE training.

    Parameters
    ----------
    input_path:
        Training corpus 文件路径。

    special_tokens:
        Special tokens，例如：
            ["<|endoftext|>"]

        它们：
        1. 不参与 pre-token statistics
        2. 作为 hard boundaries
        3. 不允许 BPE merge 跨过它们

    num_workers:
        并行进程数量。

    pattern:
        Pre-tokenization regex。
        默认使用 CS336 / GPT-2 风格规则。

    Returns
    -------
    Counter[bytes]

    Example
    -------
    Counter({
        b" the": 123456,
        b" a": 56789,
        b"hello": 1234,
        ...
    })
    """

    if special_tokens is None:
        special_tokens = []

    # ------------------------------------------------
    # 没有 special token 时，我们无法在任意位置安全切文件，
    # 因为可能切断一个 pre-token。
    #
    # CS336 的 TinyStories / OWT 通常都有 <|endoftext|>，
    # 所以实际训练一般会进入下面的并行路径。
    # ------------------------------------------------

    if not special_tokens or num_workers <= 1:

        args = (
            input_path,
            0,
            os.path.getsize(input_path),
            special_tokens,
            pattern,
        )

        return _process_chunk(args)

    # 用第一个 special token 作为物理文件切分边界
    boundary_token = special_tokens[0].encode("utf-8")

    boundaries = _find_chunk_boundaries(
        input_path,
        num_workers,
        boundary_token,
    )

    tasks = []

    for start, end in zip(
        boundaries[:-1],
        boundaries[1:],
    ):
        tasks.append(
            (
                input_path,
                start,
                end,
                special_tokens,
                pattern,
            )
        )

    total_counts = Counter()

    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:

        results = executor.map(
            _process_chunk,
            tasks,
        )

        for local_counts in results:
            total_counts.update(local_counts)

    return total_counts