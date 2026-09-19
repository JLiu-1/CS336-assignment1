"""验收后打包代码、说明和小型证据；大数据/checkpoint 留在课程平台。"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milestone", choices=[f"m{i}" for i in range(1, 6)])
    parser.add_argument("--submission", help="M4/M5 必需，传给阶段验收入口")
    parser.add_argument("--evidence", help="要附入的小型实验产物目录（仓库内路径）")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.milestone in ("m4", "m5") and not args.submission:
        parser.error("M4/M5 请指定 --submission 实验清单")
    command = [sys.executable, "-m", "lab_tools.check", args.milestone]
    if args.submission:
        command += ["--submission", args.submission]
    subprocess.run(command, check=True)
    root = Path.cwd().resolve()
    folders = [
        "tokenizer_lab_code",
        "transformer_lab_code",
        "training_lab_code",
        "generation_lab_code",
        "lab_tools",
        "docs",
        "configs",
        "templates",
        "milestone_tests",
        "tokenizer_lab_tests",
        "tokenizer_lab_data",
    ]
    files = [root / f for f in ("README.md", "pyproject.toml", "uv.lock", "LICENSE")]
    for folder in folders:
        files.extend((root / folder).rglob("*"))
    files += list((root / "tests").rglob("*"))
    if args.evidence:
        evidence = (root / args.evidence).resolve()
        if not evidence.is_relative_to(root):
            parser.error("--evidence 必须在仓库内")
        files.extend(evidence.rglob("*"))
    if args.submission:
        files.append((root / args.submission).resolve())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        parser.error("输出 zip 已存在，请更换文件名")
    allowed = {".py", ".md", ".json", ".jsonl", ".svg", ".csv", ".toml", ".lock", ".txt"}
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            fixture = path.is_relative_to(root / "tests" / "fixtures") or path.is_relative_to(
                root / "tests" / "_snapshots"
            )
            if (
                not path.is_file()
                or "__pycache__" in path.parts
                or path.suffix not in allowed
                and path.name != "LICENSE"
                and not fixture
            ):
                continue
            if not path.is_relative_to(root) or path.stat().st_size > 10 * 1024 * 1024:
                continue
            archive.write(path, path.relative_to(root).as_posix())
    print(f"已生成 {out}。checkpoint 未打包，请保留课程平台中的原运行目录供核验。")


if __name__ == "__main__":
    main()
