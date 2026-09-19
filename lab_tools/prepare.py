"""准备小规模数据或将课程文本转为 token 数组。复用 M1 的编码器。"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tokenizer_lab_code.tokenizer_lab.api import train_counts
from tokenizer_lab_code.tokenizer_lab.pretokenize import parallel_pretokenize
from tokenizer_lab_code.tokenizer_lab.storage import file_sha256, load_model, save_model, write_json

EOT = "<|endoftext|>"


def documents(path):
    """仅在完整文档边界切分；不会把正则 pre-token 从中间切断。"""
    with open(path, encoding="utf-8") as stream:
        pending = ""
        while chunk := stream.read(1024 * 1024):
            pending += chunk
            pieces = pending.split(EOT)
            for piece in pieces[:-1]:
                yield piece + EOT
            pending = pieces[-1]
            if len(pending) > 64 * 1024 * 1024:
                raise ValueError("单个文档超过 64M 字符，请先按完整文档整理输入")
        if pending:
            yield pending


def encode_split(source, tokenizer, destination):
    dtype = np.uint16 if len(tokenizer.vocab) <= 65536 else np.uint32
    raw = destination.with_suffix(".tokens.tmp")
    count = 0
    try:
        with raw.open("wb") as stream:
            for doc in documents(source):
                ids = np.asarray(tokenizer.encode(doc), dtype=dtype)
                ids.tofile(stream)
                count += len(ids)
        if count < 2:
            raise ValueError("每个 split 至少需要两个 token")
        array = np.lib.format.open_memmap(destination, mode="w+", dtype=dtype, shape=(count,))
        with raw.open("rb") as stream:
            offset = 0
            while chunk := stream.read(2 * 1024 * 1024):
                values = np.frombuffer(chunk, dtype=dtype)
                array[offset : offset + len(values)] = values
                offset += len(values)
        array.flush()
        del array
    finally:
        raw.unlink(missing_ok=True)
    return dict(
        tokens=count, sha256=file_sha256(destination), source_sha256=file_sha256(source), dtype=np.dtype(dtype).name
    )


def prepare(tokenizer_path, train_path, valid_path, out):
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError("输出已有 manifest；请选择新的输出目录")
    if Path(train_path).resolve() == Path(valid_path).resolve() or file_sha256(train_path) == file_sha256(valid_path):
        raise ValueError("训练集和验证集不能是同一份文本")
    tokenizer = load_model(tokenizer_path)
    if set(tokenizer.vocab) != set(range(len(tokenizer.vocab))):
        raise ValueError("训练用词表 ID 必须从 0 连续编号")
    out.mkdir(parents=True, exist_ok=True)
    target = out / "tokenizer.json"
    if Path(tokenizer_path).resolve() != target.resolve():
        shutil.copyfile(tokenizer_path, target)
    manifest = dict(
        format="cs336-course-data-v1",
        vocab_size=len(tokenizer.vocab),
        tokenizer_sha256=file_sha256(target),
        eos_id=tokenizer.byte_to_id.get(EOT.encode()),
    )
    manifest["train"] = encode_split(train_path, tokenizer, out / "train.npy")
    manifest["valid"] = encode_split(valid_path, tokenizer, out / "valid.npy")
    write_json(out / "manifest.json", manifest)
    return manifest


def demo(out):
    """仅用于调试：独立的短文本，不代表 TinyStories 训练质量。"""
    out = Path(out)
    if (out / "manifest.json").exists():
        raise FileExistsError("demo 已存在，请复用或指定新的 --out")
    out.mkdir(parents=True, exist_ok=True)
    train = "\n".join(
        f"Once upon a time, {name} found a {item}. The {item} was very small. "
        f"{name} gave it to a friend. They were happy and went home. {EOT}"
        for name in ("Lily", "Ben", "Tom", "May")
        for item in ("book", "ball", "toy", "flower")
    )
    valid = "\n".join(
        f"One day, {name} saw a little bird. The bird was hungry. {name} gave it some food. "
        f"The bird sang a song and flew away. {EOT}"
        for name in ("Alice", "Sam", "Lucy", "Jack")
    )
    train_path, valid_path = out / "source_train.txt", out / "source_valid.txt"
    train_path.write_text(train, encoding="utf-8")
    valid_path.write_text(valid, encoding="utf-8")
    counts = parallel_pretokenize(train_path, [EOT], 1)
    result = train_counts(counts, 288, [EOT])
    save_model(out / "tokenizer.json", result, special_tokens=[EOT])
    return prepare(out / "tokenizer.json", train_path, valid_path, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--tokenizer")
    parser.add_argument("--train")
    parser.add_argument("--valid")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.demo:
        result = demo(args.out)
    else:
        if not all((args.tokenizer, args.train, args.valid)):
            parser.error("需要 --tokenizer、--train 和 --valid，或使用 --demo")
        result = prepare(args.tokenizer, args.train, args.valid, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
