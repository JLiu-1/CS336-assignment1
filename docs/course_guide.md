# 课程约定与使用方法

## 一条贯穿五阶段的链路

文本 → M1 分词器 → token IDs → M2 Transformer → logits → M3 损失与优化器 → M4 训练/生成 → M5 优化研究。

所有框架从开始就可见。每阶段只补当期 TODO，保留之前完成的代码；不要用别人的完整实现替换自己之前的任务。M4 组队后可以选择一位组员的已验收实现作为共同基础，记录选择理由、修复和交叉复核。代码来自哪位组员与谁最终提交是不同的信息，都应说明。

## 环境

课程 Linux 平台已配置 Python 3.12/3.13、PyTorch 和项目依赖。命令中的 `uv run python` 也可替换为已激活课程环境里的 `python`。所有命令从仓库根目录执行，不需要进入各代码目录。

```sh
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
uv run python -m lab_tools.check --support
```

如果 CUDA 显示 False，M1–M3 仍可在 CPU 上完成；GPU 训练请在已分配的课程 GPU 节点运行。不要自行在共享环境中升级 torch 或修改 CUDA。M1 的轻量环境可只运行 `uv run pytest tokenizer_lab_tests/test_support.py -q`；该测试本身不导入 torch。

## 允许使用什么

可以使用 Tensor 索引、广播、`@`、`transpose`、`reshape`、`sum/mean/max`、`exp/log/sqrt/sigmoid` 等基本操作。`nn.Module`、`nn.Parameter`、`ModuleList` 和参数初始化已在框架中使用。

M2 的必做实现不直接调用 `nn.Linear`、`nn.Embedding`、内置 RMSNorm、融合 attention 或内置 softmax；M3 不用内置交叉熵和 AdamW 代替待实现算法。反向传播使用 PyTorch autograd，不手写 backward。测试里使用内置算子作参照不代表学生 TODO 可以被替换。

M5 可在保留基线的基础上探索内置融合算子等性能优化，但先通过等价性检查，并在报告中声明改变；不能修改测试期望值来适配错误输出。

## 测试与验收

```sh
uv run python -m lab_tools.check m2         # M1 + M2 必做测试
uv run python -m lab_tools.check m2 --only  # 只定位 M2 错误
```

M1 的 B 版优化是选做；累计验收不依赖它。M2/M3/M4 会同时使用课程行为测试和相关原作业数值测试。原 `tests/adapters.py` 已接好，不需要在 adapter 中写算法。

M4/M5 还需实验验收：

```sh
uv run python -m lab_tools.check m4 --submission var/m4_submission/submission.json
uv run python -m lab_tools.check m5 --submission var/m5_submission/submission.json
```

清单中的相对路径以清单文件所在目录为起点。检查器验证文件、曲线、token 预算、数据身份等一致性，不自动判断报告质量，不保证某一 loss 就一定代表实现正确。正式实验规模以课堂发布的预算档为准；demo 通过不等于已完成正式实验。

## Git 与提交

一直使用自己的工作分支即可，每阶段完成后提交 commit，也可以打 `m1-submission`、`m2-submission` 等 tag。不要直接 checkout 干净模板覆盖前面写过的文件。M4 小组仓库应保留所选实现的历史，并在报告列出整合时的 commit。

`var/` 不会被 Git 提交。备份运行目录，尤其是 `last.pt`、`config.json`、`provenance.json`、`metrics.jsonl`、`summary.json` 和生成结果。打包工具默认不放入大数据和 checkpoint；这些文件留在课程平台供抽查。

```sh
uv run python -m lab_tools.submit m1 --evidence var/m1_submission --out var/m1_submission.zip
uv run python -m lab_tools.submit m4 --submission var/m4_submission/submission.json --evidence var --out var/m4_submission.zip
```

每人独立部分可讨论概念与排错思路，但提交自己理解的实现。小组阶段三人共同对代码和实验负责。AI 可用于解释概念/API和分析报错，不能直接生成必做 TODO 的完整答案；任何实验数据都应来自实际运行。
