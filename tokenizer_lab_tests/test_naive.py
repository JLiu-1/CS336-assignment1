"""学生 A 版测试：未填写 TODO 时应失败，不使用 xfail 掩盖未完成任务。"""

from collections import Counter

import pytest

from tokenizer_lab_code.tokenizer_lab.common import TrainingResult, Word
from tokenizer_lab_code.tokenizer_lab.naive import count_pairs, merge_sequence, record_merge, select_pair, train

pytestmark = pytest.mark.lab_a


def test_a1_weighted_overlapping_pairs():
    words = [Word((b"a", b"a", b"a"), 3), Word((b"a", b"b"), 2)]
    assert count_pairs(words) == Counter({(b"a", b"a"): 6, (b"a", b"b"): 2})
    assert words[0].tokens == (b"a", b"a", b"a")
    assert count_pairs([]) == Counter()


def test_a2_tie_is_tuple_lexicographic_not_concatenation():
    a, b = (b"a", b"zz"), (b"aa", b"a")
    for pairs in [{a: 3, b: 3}, {b: 3, a: 3}]:
        assert select_pair(Counter(pairs)) == b
    assert select_pair(Counter()) is None
    assert select_pair(Counter({a: 0, b: -1})) is None


@pytest.mark.parametrize(
    "sequence,expected",
    [
        ((), ()),
        ((b"a",), (b"a",)),
        ((b"a", b"b"), (b"a", b"b")),
        ((b"a", b"a", b"a"), (b"aa", b"a")),
        ((b"a", b"a", b"a", b"a"), (b"aa", b"aa")),
    ],
)
def test_a3_nonoverlapping(sequence, expected):
    assert merge_sequence(sequence, (b"a", b"a")) == expected


def test_a4_record_order_and_reuse():
    r = TrainingResult({i: bytes([i]) for i in range(256)}, [])
    record_merge(r, (b"a", b"b"))
    assert r.vocab[256] == b"ab" and r.merges == [(b"a", b"b")]
    r.register_token(b"abc")  # 相同 bytes 已登记时不重复添加，但仍记录当前规则。
    record_merge(r, (b"ab", b"c"))
    assert len(r.vocab) == 258
    assert r.merges[-1] == (b"ab", b"c")


def test_a_end_to_end_and_early_stop():
    counts = Counter({b"ab": 2, b"ac": 2})
    r = train(counts, 999, ["<eot>"], trace=True)
    assert r.merges == [(b"a", b"c"), (b"a", b"b")]
    assert len(r.vocab) == 259
    assert [s.frequency for s in r.trace] == [2, 2]
    assert counts == {b"ab": 2, b"ac": 2}


def test_a_empty_corpus():
    assert train(Counter(), 300).merges == []
