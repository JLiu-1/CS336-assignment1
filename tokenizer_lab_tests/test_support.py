"""教师设施测试：不需要学生先完成任何 TODO，也不导入 torch。"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from tokenizer_lab_code.tokenizer_lab import incremental, naive
from tokenizer_lab_code.tokenizer_lab.agenda import PairAgenda
from tokenizer_lab_code.tokenizer_lab.common import TrainingResult, adjacent_pairs, initialize
from tokenizer_lab_code.tokenizer_lab.pretokenize import _find_chunk_boundaries, parallel_pretokenize, split_special
from tokenizer_lab_code.tokenizer_lab.storage import load_cache, load_model, save_cache, save_model

pytestmark = pytest.mark.lab_support


def test_local_counter_includes_overlaps():
    assert adjacent_pairs((b"a", b"a", b"a")) == {(b"a", b"a"): 2}


def test_initialize_stable_ids_and_bytes():
    words, result = initialize(Counter({b"z": 2, b"a": 3}), 257, ["<eot>"])
    assert [w.tokens for w in words] == [(b"a",), (b"z",)]
    assert result.vocab[256] == b"<eot>"
    assert initialize(Counter(), 256, ["a"])[1].vocab[97] == b"a"


@pytest.mark.parametrize("specials", [[""], ["x", "x"], [b"x"]])
def test_invalid_specials(specials):
    with pytest.raises(ValueError):
        initialize(Counter(), 300, specials)


@pytest.mark.parametrize(
    "counts,size", [(Counter({b"a": 0}), 256), (Counter({b"": 1}), 256), (Counter({b"a": 1.2}), 256), (Counter(), 255)]
)
def test_invalid_training_input(counts, size):
    with pytest.raises(ValueError):
        initialize(counts, size, [])


def test_special_longest_match():
    assert list(split_special("aaX<a>Ybb<a>cc", ["<a>", "X<a>Y"])) == [
        (False, "aa"),
        (True, "X<a>Y"),
        (False, "bb"),
        (True, "<a>"),
        (False, "cc"),
    ]


def test_serial_special_boundary(tmp_path):
    p = tmp_path / "input.txt"
    p.write_text("aa<eot>bb<eot>aa", encoding="utf-8")
    assert parallel_pretokenize(p, ["<eot>"]) == Counter({b"aa": 2, b"bb": 1})


def test_parallel_matches_serial_unicode_and_block_crossing(tmp_path):
    p = tmp_path / "input.txt"
    # 长文档、多字节字符、连续特殊 token、重复边界。
    p.write_text(("你好🙂 " * 900 + "<eot><eot>abc\n") * 3, encoding="utf-8")
    assert parallel_pretokenize(p, ["<eot>"], 2) == parallel_pretokenize(p, ["<eot>"], 1)


def test_boundary_token_straddles_search_block(tmp_path):
    p = tmp_path / "input.txt"
    # 20000/2=10000；第一次4096字节读取结束于14096，分隔符从14094开始。
    p.write_bytes(b"x" * 14094 + b"<eot>" + b"x" * (20000 - 14099))
    assert _find_chunk_boundaries(str(p), 2, b"<eot>") == [0, 14094, 20000]


def test_overlapping_specials_fall_back_safely(tmp_path, monkeypatch):
    p = tmp_path / "input.txt"
    p.write_text("aaX<a>Ybb<a>cc", encoding="utf-8")
    import tokenizer_lab_code.tokenizer_lab.pretokenize as module

    def forbidden_pool(*args, **kwargs):
        raise AssertionError("多特殊 token 不应进入未经证明安全的物理切块路径")

    monkeypatch.setattr(module, "ProcessPoolExecutor", forbidden_pool)
    assert parallel_pretokenize(p, ["<a>", "X<a>Y"], 8) == Counter({b"aa": 1, b"bb": 1, b"cc": 1})


@pytest.mark.parametrize("content,specials", [("", ["<eot>"]), ("你好abc", []), ("abcabc", ["<eot>"])])
def test_empty_or_missing_boundaries(tmp_path, content, specials):
    p = tmp_path / "input.txt"
    p.write_text(content, encoding="utf-8")
    assert parallel_pretokenize(p, specials, 3) == parallel_pretokenize(p, specials, 1)


def test_zero_workers_rejected(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(ValueError):
        parallel_pretokenize(p, num_workers=0)


def test_agenda_tie_and_stale_entries():
    ab, ac = (b"a", b"b"), (b"a", b"c")
    counts = Counter({ab: 3, ac: 2})
    agenda = PairAgenda(counts)
    counts[ab] = 1
    agenda.refresh({ab})
    assert agenda.pop_best() == ac  # 旧的 ab=3 已失效。
    del counts[ac]
    counts[ab] = 0
    assert agenda.pop_best() is None
    agenda = PairAgenda(Counter({ab: 2, ac: 2}))
    assert agenda.pop_best() == ac


def test_agenda_compaction_bounds_stale_heap():
    pair = (b"a", b"b")
    counts = Counter({pair: 1})
    agenda = PairAgenda(counts)
    for value in range(2, 2200):
        counts[pair] = value
        agenda.refresh({pair})
    assert len(agenda.heap) <= 4 * len(counts) + 1024
    assert agenda.pop_best() == pair


def test_cache_roundtrip_and_provenance(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("你好", encoding="utf-8")
    cache = tmp_path / "counts.json"
    expected = Counter({"你好".encode(): 3, b"\n": 2})
    save_cache(cache, expected, special_tokens=["<eot>"], source=source)
    actual, meta = load_cache(cache)
    assert actual == expected
    assert len(meta["source_sha256"]) == 64


def test_model_roundtrip_without_student_code(tmp_path):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"<eot>"
    result = TrainingResult(vocab, [(b"a", b"b")])
    path = tmp_path / "model.json"
    save_model(path, result, special_tokens=["<eot>"])
    tokenizer = load_model(path)
    assert tokenizer.vocab == vocab and tokenizer.merges == result.merges


def test_reject_wrong_artifact_type(tmp_path):
    path = tmp_path / "wrong.json"
    path.write_text('{"format": "unknown"}')
    with pytest.raises(ValueError):
        load_cache(path)
    with pytest.raises(ValueError):
        load_model(path)


def test_teacher_loops_with_one_handwritten_transition(monkeypatch):
    """只用 ab×2 -> ab 的手工夹具验证调度，不提供通用算法答案。"""
    pair = (b"a", b"b")
    monkeypatch.setattr(naive, "count_pairs", lambda words: Counter({pair: 2}))
    monkeypatch.setattr(naive, "select_pair", lambda counts: pair)
    monkeypatch.setattr(naive, "merge_sequence", lambda sequence, chosen: (b"ab",))

    def record(result, chosen):
        result.register_token(b"ab")
        result.merges.append(pair)

    monkeypatch.setattr(naive, "record_merge", record)
    monkeypatch.setattr(incremental, "build_index", lambda words: {pair: {0}})
    monkeypatch.setattr(incremental, "update_counts", lambda counts, before, after, weight: counts.clear())
    monkeypatch.setattr(incremental, "update_index", lambda index, word_id, before, after: index.clear())
    a = naive.train(Counter({b"ab": 2}), 257, trace=True)
    b = incremental.train(Counter({b"ab": 2}), 257, trace=True)
    assert a == b
    assert a.trace[0].frequency == 2


def test_cli_pretokenize_and_zero_merge_training(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cache = tmp_path / "cache.json"
    prefix = [sys.executable, "-m", "tokenizer_lab_code.tokenizer_lab.cli"]
    p = subprocess.run(
        prefix + ["pretokenize", "--input", "tokenizer_lab_data/tiny.txt", "--output", str(cache)],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["unique_pretokens"] > 0
    # 最小词表无需训练即可保存；不会假装学生 TODO 已完成。
    p = subprocess.run(
        prefix + ["train", "--cache", str(cache), "--vocab-size", "257", "--output", str(tmp_path / "model.json")],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert p.returncode == 0, p.stderr
