"""教师提供：两版共用的数据结构、初始化和局部计数工具。"""

from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise

Pair = tuple[bytes, bytes]
Sequence = tuple[bytes, ...]


def validate_special_tokens(tokens: list[str] | None) -> list[str]:
    result = list(tokens or [])
    if any(not isinstance(t, str) or not t for t in result):
        raise ValueError("special_tokens 必须是非空字符串")
    if len(set(result)) != len(result):
        raise ValueError("special_tokens 不能重复")
    return result


def adjacent_pairs(sequence: Sequence) -> Counter[Pair]:
    """单个序列的相邻 pair 次数；重叠位置也计数：aaa 的 aa 次数是 2。

    本函数不乘语料频次。A1、B2 负责处理频次权重。
    """
    return Counter(pairwise(sequence))


@dataclass
class Word:
    """word_id 是 words 列表下标，在整个训练中不变。

    tokens 是一个 pre-token 当前的切分；frequency 是它在语料中的次数。
    """

    tokens: Sequence
    frequency: int


@dataclass(frozen=True)
class MergeStep:
    step: int
    pair: Pair
    frequency: int


@dataclass
class TrainingResult:
    vocab: dict[int, bytes]
    merges: list[Pair]
    trace: list[MergeStep] = field(default_factory=list)
    token_to_id: dict[bytes, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        self.token_to_id = {raw: token_id for token_id, raw in self.vocab.items()}

    def register_token(self, raw: bytes) -> int:
        """教师提供：登记/复用词条，避免每次合并都线性扫描整个词表。"""
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("词条必须为非空 bytes")
        if raw not in self.token_to_id:
            token_id = len(self.vocab)
            self.vocab[token_id] = raw
            self.token_to_id[raw] = token_id
        return self.token_to_id[raw]


def initialize(
    counts: Counter[bytes], vocab_size: int, special_tokens: list[str] | None
) -> tuple[list[Word], TrainingResult]:
    """提供稳定顺序；避免两版比较受到输入字典顺序的影响。"""
    specials = validate_special_tokens(special_tokens)
    vocab = {i: bytes([i]) for i in range(256)}
    for token in specials:
        value = token.encode("utf-8")
        if value not in vocab.values():
            vocab[len(vocab)] = value
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < len(vocab):
        raise ValueError(f"vocab_size 至少为 {len(vocab)}（包括基础字节和特殊 token）")
    words = []
    for raw, frequency in counts.items():
        if not isinstance(raw, bytes) or not raw:
            raise ValueError("预分词统计的 key 必须是非空 bytes")
        if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
            raise ValueError("预分词次数必须是正整数")
    for raw in sorted(counts):
        words.append(Word(tuple(bytes([b]) for b in raw), counts[raw]))
    return words, TrainingResult(vocab, [])


def check_record(result: TrainingResult, pair: Pair, previous_merges: int) -> None:
    """尽早报告 A4 接口错误；此处不实现词表更新。"""
    if len(result.merges) != previous_merges + 1 or result.merges[-1] != pair:
        raise AssertionError("A4 必须向 merges 末尾追加当前 pair 一次")
    token_id = result.token_to_id.get(pair[0] + pair[1])
    if token_id is None or result.vocab.get(token_id) != pair[0] + pair[1]:
        raise AssertionError("A4 应使用 result.register_token 登记合并后的 bytes")
