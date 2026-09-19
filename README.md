# 从分词器到语言模型：五阶段实验

本仓库是基于 Stanford CS336 Assignment 1 的课程版。你将补齐关键算法，逐步得到一个可以训练、恢复和生成文本的语言模型。Python/PyTorch 的工程框架、测试接线、数据管理和日志工具已经提供。课程平台已经配置环境，无需重新安装。

**所有阶段在同一个分支、同一套代码中完成。后面的阶段直接调用你前面写的实现，没有附带前面阶段的答案。** 每完成一个阶段提交一次 Git commit；如需标记提交，可使用 `m1-submission` 等 tag，不必切换 milestone 分支。

## 从哪里开始

| 阶段 | 形式 | 实验说明 | 待填写代码 | 阶段能力 |
|---|---|---|---|---|
| M1 文本变成 token | 个人 | [M1](docs/m1_tokenizer.md) | [A版](tokenizer_lab_code/tokenizer_lab/naive.py)、[编码器](tokenizer_lab_code/tokenizer_lab/tokenizer.py) | 小语料 BPE 训练、编码解码 |
| M2 token 变成预测 | 个人 | [M2](docs/m2_transformer.md) | [基础层](transformer_lab_code/layers.py)、[Attention](transformer_lab_code/attention.py)、[模型](transformer_lab_code/model.py) | Transformer 前向、因果性与梯度 |
| M3 让参数学习 | 个人 | [M3](docs/m3_optimization.md) | [损失](training_lab_code/losses.py)、[优化器](training_lab_code/optimizer.py) | 交叉熵、AdamW、调度、裁剪 |
| M4 训练并生成 | 三人小组 | [M4](docs/m4_training.md) | [训练核心](training_lab_code/student.py)、[采样](generation_lab_code/sampling.py) | 可恢复训练、验证、文本生成 |
| M5 优化与研究 | 原三人小组 | [M5](docs/m5_experiments.md) | 自己的模型与 [配置](configs) | 基线与改动方案的受控实验 |

先读 [课程约定、环境与阶段验收](docs/course_guide.md)。M1 仅需 CPU；M2/M3 的微型测试也在 CPU 上完成。

```sh
# 从仓库根目录运行。设施检查不调用学生 TODO。
uv run python -m lab_tools.check --support

# 完成哪个阶段，运行哪个命令；默认回归此前所有必做阶段。
uv run python -m lab_tools.check m1
uv run python -m lab_tools.check m2
uv run python -m lab_tools.check m3
uv run python -m lab_tools.check m4
uv run python -m lab_tools.check m5
```

未填写的 TODO 抛出 `NotImplementedError` 是正常状态。M1 验收不会收集 M2–M5 的测试；选做的增量 BPE 不影响后续必做验收。不要通过删除测试、标记跳过或硬编码测试输出取得通过。

M4/M5 的上述命令只验证代码；正式提交还需加 `--submission` 检查实验产物，详见对应说明。`--only` 用于仅调试当前阶段，不替代正式的累计验收。

## 文件导航

- `tokenizer_lab_code/`：M1 的实现，后续继续使用。
- `transformer_lab_code/`：M2 模型，也是 M4/M5 真正训练的模型。
- `training_lab_code/`：M3 算法、M4 训练核心与已提供的数据/恢复设施。
- `generation_lab_code/`：M4 自回归采样。
- `lab_tools/`：已经接好的命令入口，不要求从零编写工具。
- `configs/`：debug、短训练、正式实验和原规模选做配置。
- `templates/`：报告和小组提交清单模板。
- `milestone_tests/`、`tokenizer_lab_tests/`：课程验收；`tests/`：原测试与已接好的 adapters。
- `var/`：运行结果、数据和 checkpoint，已被 Git 忽略。

根目录 `test.py` 和 `tests/parallel_tokenizer.py` 是保留的早期原型，不是正式入口。原始 [作业 PDF](cs336_assignment1_basics.pdf) 可作扩展阅读；本课程以五份实验说明的必做/选做边界为准。

## 提交与共享资源

每人 M1–M3 独立完成；M4 开始组队，从组员已通过验收的实现中选定共同基础，记录整合与复核工作。每组同时最多运行一个 GPU 作业。不要未经规划启动原规模训练或让预分词占满所有 CPU。

正式训练用平台发放的 tokenizer 与配套数据；本地 demo 可通过自己的 M1 实现生成。数据/tokenizer 不配套时框架会报错。规模、预算和降档规则见 [资源说明](docs/resources.md)。

`python -m lab_tools.submit` 会先验收再打包。不要直接运行全仓库 pytest 作为阶段验收，原测试还包含课程未要求的速度门槛与平台相关测试。

## 来源

基于 Stanford CS336 的 [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)。原始许可证保留于 [LICENSE](LICENSE)。课程版增加中文引导、填空框架和分阶段验收，保留核心算法的学习任务。
