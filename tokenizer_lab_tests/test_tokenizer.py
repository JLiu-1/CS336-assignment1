"""学生 C 部分：只测试编码解码，不依赖先训练出词表。"""

import pytest

from tokenizer_lab_code.tokenizer_lab.tokenizer import Tokenizer

pytestmark = pytest.mark.lab_c


def base_vocab():
    return {i: bytes([i]) for i in range(256)}


def test_c1_rank_not_frequency_or_longest_match():
    vocab = base_vocab() | {256: b"ab", 257: b"bc"}
    t = Tokenizer(vocab, [(b"a", b"b"), (b"b", b"c")])
    assert t.encode_pretoken(b"abc") == [256, 99]
    t = Tokenizer(vocab, [(b"b", b"c"), (b"a", b"b")])
    assert t.encode_pretoken(b"abc") == [97, 257]


def test_c1_multi_step_merge():
    t = Tokenizer(base_vocab() | {256: b"ab", 257: b"abc"}, [(b"a", b"b"), (b"ab", b"c")])
    assert t.encode_pretoken(b"abcabc") == [257, 257]


@pytest.mark.parametrize("text", ["", "hello world!", "你好🙂", "a\n\n b", "<eot>你好<eot>"])
def test_c_roundtrip(text):
    t = Tokenizer(base_vocab(), [], ["<eot>"])
    assert t.decode(t.encode(text)) == text


def test_c2_join_bytes_before_utf8_decode():
    t = Tokenizer(base_vocab(), [])
    assert t.decode(list("你🙂".encode())) == "你🙂"
    assert t.decode([255]) == "\ufffd"
    with pytest.raises(KeyError):
        t.decode([99999])


def test_special_longest_and_no_cross_boundary_merge():
    t = Tokenizer(base_vocab() | {256: b"ab"}, [(b"a", b"b")], ["<x>", "<x><x>"])
    assert t.encode("<x><x>") == [t.byte_to_id[b"<x><x>"]]
    assert t.encode("a<x>b") == [97, t.byte_to_id[b"<x>"], 98]


def test_iterable_is_lazy_and_record_based():
    t = Tokenizer(base_vocab() | {256: b"ab"}, [(b"a", b"b")])
    consumed = []

    def records():
        consumed.append(1)
        yield "a"
        consumed.append(2)
        yield "b"

    stream = t.encode_iterable(records())
    assert consumed == []
    assert next(stream) == 97 and consumed == [1]
    assert list(stream) == [98]
    assert t.encode("ab") == [256]
