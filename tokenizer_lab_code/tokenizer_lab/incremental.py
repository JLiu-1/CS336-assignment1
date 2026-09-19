"""B 版：以 pre-token 为单位做局部更新；复用已完成的 A 版函数。"""

from collections import Counter

from . import naive
from .agenda import PairAgenda
from .common import MergeStep, Pair, TrainingResult, Word, adjacent_pairs, check_record, initialize

PairIndex = dict[Pair, set[int]]


def build_index(words: list[Word]) -> PairIndex:
    """TODO B1：返回 pair -> 包含该 pair 的 word_id 集合。

    word_id 是 words 的下标。同一 word 中出现多次，只登记一次 ID。
    可调用 adjacent_pairs；不得修改 words。
    """
    raise NotImplementedError("TODO B1: build_index")


def update_counts(
    pair_counts: Counter[Pair],
    before: Counter[Pair],
    after: Counter[Pair],
    frequency: int,
) -> None:
    """TODO B2：原地更新全局统计，反映一个 word 合并前后的变化。

    before/after 是该 word 未乘 frequency 的局部计数。
    其他 word 的贡献必须保留；删除归零项；不能留下负数或修改 before/after。
    """
    raise NotImplementedError("TODO B2: update_counts")


def update_index(index: PairIndex, word_id: int, before: Counter[Pair], after: Counter[Pair]) -> None:
    """TODO B3：原地更新该 word 的 pair 出现索引。

    索引记录存在性，不是次数。仍然存在的 pair 保留 ID，消失的删除，新增的加入。
    保留其他 word 的 ID；删除空集合。不能修改 before/after。
    """
    raise NotImplementedError("TODO B3: update_index")


def assert_consistent(words: list[Word], counts: Counter[Pair], index: PairIndex) -> None:
    """教师提供：调试模式下用 A1 全量重算，定位局部更新偏差。"""
    assert counts == naive.count_pairs(words), "B2: 局部更新与 A1 全量统计不一致"
    assert all(c > 0 for c in counts.values()), "B2: 全局计数中存在非正数"
    for word_id, word in enumerate(words):
        for pair in adjacent_pairs(word.tokens):
            assert word_id in index.get(pair, set()), "B1/B3: 索引漏掉 word_id"
    for pair, ids in index.items():
        assert ids, "B3: 索引中存在空集合"
        for word_id in ids:
            assert 0 <= word_id < len(words), "B1/B3: word_id 越界"
            assert pair in adjacent_pairs(words[word_id].tokens), "B3: 索引保留了过期 word_id"


def train(
    counts: Counter[bytes],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    *,
    trace: bool = False,
    debug: bool = False,
) -> TrainingResult:
    words, result = initialize(counts, vocab_size, special_tokens)
    if len(result.vocab) == vocab_size:
        return result
    pair_counts = naive.count_pairs(words)
    index = build_index(words)
    agenda = PairAgenda(pair_counts)
    if debug:
        assert_consistent(words, pair_counts, index)
    while len(result.vocab) < vocab_size:
        pair = agenda.pop_best()
        if pair is None:
            break
        selected_count = pair_counts[pair]
        affected = sorted(index.get(pair, set()))  # 快照：循环中 B3 会修改 index。
        if not affected:
            raise AssertionError("B1/B3: 候选 pair 在索引中没有受影响的 word")
        changed = set()
        for word_id in affected:
            word = words[word_id]
            before = adjacent_pairs(word.tokens)
            updated = naive.merge_sequence(word.tokens, pair)
            if len(updated) >= len(word.tokens):
                raise AssertionError("A3/B3: 受影响的 word 没有缩短")
            after = adjacent_pairs(updated)
            update_counts(pair_counts, before, after, word.frequency)
            update_index(index, word_id, before, after)
            word.tokens = updated
            changed.update(before)
            changed.update(after)
        agenda.refresh(changed)
        previous = len(result.merges)
        naive.record_merge(result, pair)
        check_record(result, pair, previous)
        if trace:
            result.trace.append(MergeStep(previous, pair, selected_count))
        if debug:
            assert_consistent(words, pair_counts, index)
    return result
