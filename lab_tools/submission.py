"""M4/M5 实验产物检查。验证完整性；不自动评价研究结论，也不代替代码测试。"""

import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_run(path, allow_budget_stop=False):
    path = Path(path)
    summary, config, provenance = [read(path / name) for name in ("summary.json", "config.json", "provenance.json")]
    rows = [
        json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    allowed = ("complete", "budget_exhausted") if allow_budget_stop else ("complete",)
    require(summary["status"] in allowed, f"{path}: 运行未按计划完成")
    require((path / "last.pt").is_file(), f"{path}: 缺少 checkpoint")
    require(0 < summary["step"] <= config["training"]["total_steps"], "训练步数和配置不符")
    if summary["status"] == "complete":
        require(summary["step"] == config["training"]["total_steps"], "complete 运行尚未完成配置的步数")
    expected = (
        summary["step"]
        * config["training"]["batch_size"]
        * config["training"]["gradient_accumulation"]
        * config["model"]["context_length"]
    )
    require(summary["tokens"] == expected, "token 预算记录不符")
    require(len(rows) >= 2 and rows[0]["step"] == 0 and rows[-1]["step"] == summary["step"], "曲线缺少起点或终点")
    require(
        all(a["step"] < b["step"] and a["wall_seconds"] <= b["wall_seconds"] for a, b in zip(rows, rows[1:])),
        "曲线步数/时间不递增",
    )
    require(all(math.isfinite(r[k]) for r in rows for k in ("val_loss", "train_loss") if k in r), "曲线含非有限 loss")
    require(summary["val_step"] == summary["step"] and math.isfinite(summary["val_loss"]), "缺少最终验证")
    require(abs(rows[-1]["val_loss"] - summary["val_loss"]) < 1e-6, "最终曲线与 summary 不一致")
    return summary, config, provenance


def validate_submission(path, stage):
    path = Path(path).resolve()
    obj = read(path)
    base = path.parent
    require(obj.get("milestone") == stage, "清单 milestone 不匹配")
    members = obj.get("members", [])
    require(
        len(members) == 3
        and len(set(members)) == 3
        and all(isinstance(m, str) and m.strip() and "填写" not in m for m in members),
        "填写三位不同成员的姓名/学号",
    )
    contributions = obj.get("contributions", {})
    require(
        set(contributions) == set(members)
        and all(isinstance(v, str) and len(v.strip()) >= 10 for v in contributions.values()),
        "每人填写具体贡献及复核工作",
    )
    for name in ("report", "curve"):
        require(name in obj and (base / obj[name]).is_file(), f"缺少 {name} 文件")
    require(len((base / obj["report"]).read_text(encoding="utf-8").strip()) >= 100, "报告内容过短")
    runs = obj.get("runs", [])
    require(len(runs) >= (1 if stage == 4 else 2), "运行数量不足")
    require(len({(base / record["path"]).resolve() for record in runs}) == len(runs), "不能用同一个运行目录重复充当多项实验")
    results = [
        (record, validate_run(base / record["path"], stage == 5 and obj.get("comparison") == "time")) for record in runs
    ]
    identities = {result[2]["data_identity"] for _, result in results}
    require(len(identities) == 1, "参与对照的运行必须使用相同数据和 tokenizer")
    samples = obj.get("samples", [])
    require(bool(samples), "至少提交一个生成记录")
    for sample in samples:
        record = read(base / sample)
        require(record["data_identity"] in identities and bool(record["prompt_ids"]), "生成记录的数据/prompt 不匹配")
        require(
            isinstance(record["continuation"], str) and 0 < len(record["continuation_ids"]) <= record["max_new_tokens"],
            "生成记录无有效输出",
        )
    if stage == 4:
        verification = read(base / obj["verification"])
        require(
            verification["overfit_final_loss"] < verification["overfit_initial_loss"] * 0.5, "未通过固定 batch 过拟合"
        )
        require(verification["recovery_max_abs_diff"] <= 1e-6, "未通过 CPU 恢复检查")
    else:
        require(
            any(r.get("role") == "baseline" for r in runs) and any(r.get("role") == "candidate" for r in runs),
            "M5 需要 baseline 和 candidate",
        )
        require(obj.get("comparison") in ("tokens", "time"), "comparison 应为 tokens 或 time")
        require(len(obj.get("hypothesis", "").strip()) >= 10, "请填写优化假设")
        if obj["comparison"] == "tokens":
            require(len({result[0]["tokens"] for _, result in results}) == 1, "token 对照要求相同训练 token 预算")
        else:
            require(
                len(obj.get("budget_explanation", "").strip()) >= 20, "时间对照需要说明统一预算、计时范围及超时处理"
            )
            budgets = {result[2].get("time_budget_seconds") for _, result in results}
            require(
                len(budgets) == 1 and None not in budgets,
                "时间对照运行需设置相同 --max-seconds；恢复后的累计时长还须人工核验",
            )
        require(len({result[0]["val_tokens"] for _, result in results}) == 1, "对照验证 token 数不一致")
        require(
            len(
                {
                    (
                        result[1]["training"]["eval_context_length"],
                        result[1]["training"]["eval_batch_size"],
                        result[1]["training"]["eval_batches"],
                    )
                    for _, result in results
                }
            )
            == 1,
            "对照需要相同的验证窗口设置；训练 context/batch 可以不同",
        )
    return obj
