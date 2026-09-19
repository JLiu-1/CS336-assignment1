"""逐阶段验收；只收集已进入的阶段，M1 不会导入 torch。"""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    1: [
        "tokenizer_lab_tests/test_support.py",
        "tokenizer_lab_tests/test_naive.py",
        "tokenizer_lab_tests/test_tokenizer.py",
    ],
    2: ["milestone_tests/test_m2.py", "tests/test_model.py", "tests/test_nn_utils.py::test_softmax_matches_pytorch"],
    3: [
        "milestone_tests/test_m3.py",
        "tests/test_optimizer.py",
        "tests/test_nn_utils.py::test_cross_entropy",
        "tests/test_nn_utils.py::test_gradient_clipping",
    ],
    4: ["milestone_tests/test_m4.py", "tests/test_data.py", "tests/test_serialization.py"],
    5: ["milestone_tests/test_m5.py"],
}


def targets(stage, only=False, support=False):
    if support:
        return ["tokenizer_lab_tests/test_support.py", "milestone_tests/test_support.py"]
    stages = [stage] if only else range(1, stage + 1)
    return [item for n in stages for item in STAGES[n]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("milestone", nargs="?", choices=[f"m{i}" for i in range(1, 6)])
    parser.add_argument("--only", action="store_true", help="仅跑当期；正式验收默认回归之前阶段")
    parser.add_argument("--support", action="store_true", help="无需填写 TODO 的设施测试；完整设施测试需要 torch")
    parser.add_argument("--submission", help="M4/M5 实验清单；同时检查实际实验产物")
    args = parser.parse_args()
    if not args.milestone and not args.support:
        parser.error("请选择 m1–m5 或 --support")
    if Path.cwd().resolve() != ROOT:
        parser.error("请从仓库根目录运行")
    stage = int(args.milestone[1]) if args.milestone else 1
    import pytest

    result = pytest.main([*targets(stage, args.only, args.support), "-q", "-o", "addopts=", "--tb=short"])
    if result:
        raise SystemExit(result)
    if args.submission:
        if stage not in (4, 5) or args.support:
            parser.error("--submission 只用于 M4/M5 正式验收")
        from lab_tools.submission import validate_submission

        validate_submission(args.submission, stage)
        print("代码测试与实验产物检查均通过；结论质量由报告及演示验收。")
    elif stage in (4, 5):
        print("代码测试通过。正式提交还需 --submission 清单；这不代表训练/研究任务已完成。")


if __name__ == "__main__":
    main()
