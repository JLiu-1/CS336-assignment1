#!/usr/bin/env bash
set -euo pipefail
# 兼容入口：参数与 python -m lab_tools.submit 相同。
# 示例：bash make_submission.sh m1 --out var/m1_submission.zip
uv run python -m lab_tools.submit "$@"
