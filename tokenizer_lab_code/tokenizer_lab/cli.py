"""CPU-only 教学入口：python -m tokenizer_lab_code.tokenizer_lab.cli --help。"""

import argparse
import json
import time

from .api import train_counts
from .pretokenize import PATTERN, parallel_pretokenize
from .storage import load_cache, load_model, save_cache, save_model, write_json


def _input_args(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="UTF-8 原始文本路径")
    group.add_argument("--cache", help="pretokenize 生成的缓存路径")
    parser.add_argument("--workers", type=int, default=1, help="预分词进程数；共享服务器默认 1")
    parser.add_argument(
        "--special-token",
        action="append",
        dest="special_tokens",
        help="可重复；原始文本默认 <|endoftext|>，缓存默认沿用缓存配置",
    )
    parser.add_argument("--no-special-tokens", action="store_true")


def _load_input(args):
    if args.no_special_tokens and args.special_tokens:
        raise ValueError("--no-special-tokens 与 --special-token 不能同时使用")
    specified = args.no_special_tokens or args.special_tokens is not None
    specials = [] if args.no_special_tokens else (args.special_tokens or ["<|endoftext|>"])
    start = time.perf_counter()
    if args.cache:
        counts, meta = load_cache(args.cache)
        if specified and specials != meta["special_tokens"]:
            raise ValueError("特殊 token 配置与缓存不一致，请重建缓存")
        specials, pattern = meta["special_tokens"], meta["pattern"]
    else:
        pattern = PATTERN
        counts = parallel_pretokenize(args.input, specials, args.workers, pattern)
    return counts, specials, pattern, time.perf_counter() - start


def parser():
    p = argparse.ArgumentParser(description="BPE A/B 两版教学框架（不需要 GPU）")
    sub = p.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("pretokenize", help="生成两版共用的 pre-token 缓存，无需填写 TODO")
    _input_args(pre)
    pre.add_argument("--output", required=True)
    train = sub.add_parser("train", help="填写 A/B 后训练并保存词表")
    _input_args(train)
    train.add_argument("--variant", choices=["naive", "incremental"], default="naive")
    train.add_argument("--vocab-size", type=int, default=280)
    train.add_argument("--trace", action="store_true")
    train.add_argument("--debug", action="store_true", help="B 版每轮对照 A1；不能用于公平计时")
    train.add_argument("--output", required=True)
    compare = sub.add_parser("compare", help="同一份预分词结果上比较两版，每轮核对 pair 和频次")
    _input_args(compare)
    compare.add_argument("--vocab-size", type=int, default=280)
    compare.add_argument("--debug", action="store_true")
    compare.add_argument("--repeats", type=int, default=1, help="交替运行顺序，报告每次耗时")
    compare.add_argument("--output", required=True)
    encode = sub.add_parser("encode", help="填写 C 部分后演示编码与解码")
    encode.add_argument("--model", required=True)
    encode.add_argument("--text", required=True)
    return p


def run(args):
    if args.command == "encode":
        tokenizer = load_model(args.model)
        ids = tokenizer.encode(args.text)
        print(json.dumps({"ids": ids, "decoded": tokenizer.decode(ids)}, ensure_ascii=False))
        return
    counts, specials, pattern, input_seconds = _load_input(args)
    if args.command == "pretokenize":
        save_cache(args.output, counts, special_tokens=specials, pattern=pattern, source=args.input)
        print(
            json.dumps(
                {
                    "unique_pretokens": len(counts),
                    "pretoken_occurrences": counts.total(),
                    "input_seconds": input_seconds,
                    "output": args.output,
                },
                ensure_ascii=False,
            )
        )
        return
    if args.command == "train":
        start = time.perf_counter()
        result = train_counts(
            counts, args.vocab_size, specials, variant=args.variant, trace=args.trace, debug=args.debug
        )
        report = {
            "variant": args.variant,
            "vocab_size": len(result.vocab),
            "merges": len(result.merges),
            "input_seconds": input_seconds,
            "training_seconds": time.perf_counter() - start,
            "debug": args.debug,
            "input": args.input or args.cache,
        }
        save_model(args.output, result, special_tokens=specials, pattern=pattern, metadata=report)
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.repeats < 1:
        raise ValueError("repeats 必须为正数")
    trials = []
    for repeat in range(args.repeats):
        results = {}
        timings = {}
        order = ["naive", "incremental"] if repeat % 2 == 0 else ["incremental", "naive"]
        for variant in order:
            start = time.perf_counter()
            results[variant] = train_counts(
                counts, args.vocab_size, specials, variant=variant, trace=True, debug=args.debug
            )
            timings[variant] = time.perf_counter() - start
        a, b = results["naive"], results["incremental"]
        if a.trace != b.trace:
            for i in range(max(len(a.trace), len(b.trace))):
                sa = a.trace[i] if i < len(a.trace) else None
                sb = b.trace[i] if i < len(b.trace) else None
                if sa != sb:
                    raise AssertionError(f"第 {i + 1} 轮不一致：A={sa!r}, B={sb!r}")
        if a.vocab != b.vocab or a.merges != b.merges:
            raise AssertionError("两版最终 vocab/merges 不一致")
        trials.append(
            {
                "repeat": repeat + 1,
                "order": order,
                "seconds": timings,
                "speedup": timings["naive"] / timings["incremental"],
            }
        )
    report = {
        "equal": True,
        "input_seconds": input_seconds,
        "unique_pretokens": len(counts),
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(a.vocab),
        "debug": args.debug,
        "trials": trials,
        "timing_note": "debug=true 包含全量校验成本，不可作为优化加速比"
        if args.debug
        else "两版均开启轻量 trace；输入准备时间不计入训练时间",
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False))


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (NotImplementedError, ValueError, OSError, AssertionError) as exc:
        p.exit(2, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
