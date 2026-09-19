"""学生 B 版测试：先完成 A，再做局部更新和逐轮差分测试。"""

import random
from collections import Counter

import pytest

from tokenizer_lab_code.tokenizer_lab import incremental, naive
from tokenizer_lab_code.tokenizer_lab.common import Word

pytestmark = pytest.mark.lab_b


def test_b1_index_uses_presence_not_multiplicity():
    words = [Word((b"a", b"a", b"a"), 4), Word((b"a", b"b"), 2), Word((b"a",), 3)]
    assert incremental.build_index(words) == {(b"a", b"a"): {0}, (b"a", b"b"): {1}}


def test_b2_delta_preserves_other_words():
    aa, merged, xx = (b"a", b"a"), (b"aa", b"aa"), (b"x", b"x")
    global_counts = Counter({aa: 20, xx: 9})
    before, after = Counter({aa: 3}), Counter({merged: 1})
    incremental.update_counts(global_counts, before, after, 5)
    assert global_counts == {aa: 5, merged: 5, xx: 9}
    assert before == {aa: 3} and after == {merged: 1}
    incremental.update_counts(global_counts, Counter({aa: 1}), Counter(), 5)
    assert aa not in global_counts


def test_b3_index_retains_surviving_pairs_and_other_ids():
    aa, ab, bc = (b"a", b"a"), (b"a", b"b"), (b"b", b"c")
    index = {aa: {0, 1}, ab: {0}}
    incremental.update_index(index, 0, Counter({aa: 3, ab: 1}), Counter({aa: 1, bc: 1}))
    assert index == {aa: {0, 1}, bc: {0}}
    incremental.update_index(index, 0, Counter({aa: 1, bc: 1}), Counter())
    assert index == {aa: {1}}


@pytest.mark.parametrize("seed", range(5))
def test_b_matches_a_every_round(seed):
    rng = random.Random(seed)
    counts = Counter()
    for _ in range(24):
        raw = bytes(rng.choice(b"abc") for _ in range(rng.randint(1, 8)))
        counts[raw] += rng.randint(1, 6)
    a = naive.train(counts, 290, ["<eot>"], trace=True)
    b = incremental.train(counts, 290, ["<eot>"], trace=True, debug=True)
    assert a == b


def test_b_empty_corpus():
    assert incremental.train(Counter(), 300, debug=True).merges == []
