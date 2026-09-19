"""C 部分：两版共用的编码器；填写 C1、C2。"""

from collections.abc import Iterable, Iterator

import regex

from .common import Pair, validate_special_tokens
from .pretokenize import PATTERN, split_special


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[Pair],
        special_tokens: list[str] | None = None,
        *,
        pattern: str = PATTERN,
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = validate_special_tokens(special_tokens)
        if any(type(i) is not int or i < 0 or not isinstance(v, bytes) or not v for i, v in self.vocab.items()):
            raise ValueError("vocab 必须是非负整数到非空 bytes 的映射")
        if len(set(self.vocab.values())) != len(self.vocab):
            raise ValueError("词表中有重复的 bytes")
        if not {bytes([i]) for i in range(256)} <= set(self.vocab.values()):
            raise ValueError("词表必须覆盖全部 256 个单字节")
        # 兼容原作业：构造编码器时可补入尚未登记的特殊 token。
        for token in self.special_tokens:
            raw = token.encode("utf-8")
            if raw not in self.vocab.values():
                self.vocab[max(self.vocab, default=-1) + 1] = raw
        self.byte_to_id = {value: token_id for token_id, value in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        if len(self.merge_ranks) != len(self.merges):
            raise ValueError("merges 中有重复 pair")
        for pair in self.merges:
            if len(pair) != 2 or any(not isinstance(v, bytes) or not v for v in pair):
                raise ValueError("每个 merge 必须由两个非空 bytes 构成")
            if any(v not in self.byte_to_id for v in (*pair, pair[0] + pair[1])):
                raise ValueError("merge 的组成项或合并结果不在词表中")
        self.pattern = regex.compile(pattern)

    def encode_pretoken(self, raw: bytes) -> list[int]:
        """TODO C1：把一个普通 pre-token 的 bytes 编码为 token IDs。

        使用 self.merge_ranks 的训练顺序（rank 越小优先），不是当前出现频次。
        self.byte_to_id 提供最终 bytes -> ID 映射，可复用 A3。
        不修改词表或 merges；空 bytes 返回空列表。
        """
        raise NotImplementedError("TODO C1: encode_pretoken")

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("encode 需要 str")
        ids = []
        for is_special, piece in split_special(text, self.special_tokens):
            if is_special:
                ids.append(self.byte_to_id[piece.encode("utf-8")])
            else:
                for match in self.pattern.finditer(piece):
                    ids.extend(self.encode_pretoken(match.group().encode("utf-8")))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """TODO C2：查词表，拼接全部 bytes 后统一解码 UTF-8（errors='replace'）。

        不能逐个 token 解码，因为一个汉字/emoji 的字节可能分布在多个 token 中。
        空输入返回空字符串；未知 ID 抛出 KeyError，不要静默丢弃。
        """
        raise NotImplementedError("TODO C2: decode")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """教师提供：逐项编码、逐项产出，不一次性读入全部输入。

        每个字符串视为完整的编码单元。不是支持任意字符切块的流式 BPE；
        encode_iterable(['hel', 'lo']) 不保证与 encode('hello') 的 IDs 相同。
        """
        for text in iterable:
            yield from self.encode(text)
