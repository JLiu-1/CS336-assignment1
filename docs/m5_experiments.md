# M5：优化一个小语言模型（三人小组）

## 研究问题由你们选择

沿用 M4 小组与自己实现的模型。至少建立一条可靠基线，提出一个优化假设，完成一个改动方案与基线的受控比较。可以自由选题，也可以使用原作业中的调参或消融思路。没有强制“三个学习率＋指定消融”的组合。

评分重在问题、实现、对照和解释。优化未提升也可以是完整成果；单次短训练没有显著差异时，不要编造结论。只调生成 temperature/top-p 可作补充，不能单独代替训练或模型实验。

## 可选方向

| 方向 | 可以问什么 | 代码/配置入口 | 比较时控制什么 |
|---|---|---|---|
| 学习率/日程 | 更大 lr 是否更快，warmup 是否有帮助？ | 自己的 JSON 配置 | 模型、数据、种子、训练 token 数 |
| RoPE / NoPE | 在此小数据和长度下位置编码有何影响？ | `model.use_rope` | 除开关外配置相同 |
| RMSNorm | 去掉归一化会怎样？ | `model.use_norm` | 默认先固定 lr；另调 lr 算第二组实验 |
| SwiGLU / SiLU | 门控是否值得它的计算成本？ | `ffn_type` 与 `d_ff` | SiLU 可用 4D，SwiGLU 约 8D/3，记录实际参数量 |
| 权重共享 | 输入 embedding 与 LM head 共用参数怎样？ | `tie_embeddings` | 参数量与初始化尺度也变了，需明确讨论 |
| 宽度/深度 | 给定预算下怎样分配模型容量？ | d_model、num_layers 等 | 训练/验证 token、或总时间预算 |
| 训练效率 | BF16、batch、梯度累积如何影响吞吐？ | precision、batch_size 等 | 相同评估；记录显存、速度及质量 |
| 自主改动 | 如 post-norm、新激活、等价融合算子 | 模型代码，保留默认基线 | 先微型正确性，再增加规模 |

配置开关只是接线，不会替代你们之前的 T/O/S/G 实现。M5 中修改模型时，应保留默认基线的行为，使前四阶段验收仍通过。新架构用显式选项启用，不要把参考测试的答案或 baseline 实现改掉。

## 先写一段实验计划

运行前在报告中写下：假设、只改变什么、主指标、固定哪些条件、预计 token 数/GPU 小时、提前停止条件。三位成员轮换负责实验、复核和分析，不让同一人长期只负责报告排版。

推荐用 `configs/m5_trial.json` 做短试跑，再决定是否执行完整规模。不强制重复搜索大量超参数。短试跑与完整训练的日程是否相同必须记录；默认 trial 沿用 5000 步日程，便于观察正式日程的前缀。

## 两种公平比较方式

**等 token 预算：**适合架构/优化算法比较。两次运行处理相同训练 token 数，不一定有相同 step 数；改变 batch 时相应调整 total_steps，并把 warmup/cosine 换算到相同 token 进度。以验证 loss 为主，另报耗时和显存。

**等时间预算：**适合系统效率和综合优化。预先声明相同硬件与总预算，记录所有试跑成本、启动/验证/保存的计时范围，比较截止时的验证 loss。用 `--max-seconds` 控制；它是软上限，当前步与最后验证/保存可能超时，应报告超出的时间。恢复训练不能借机重置组内累计配额。

训练入口的 `wall_seconds` 包含本次数据读取、初始化、训练、验证和保存相关时间，并累加恢复前的记录；队列等待和进程启动前的环境加载不计入。需做严格系统计时时，额外记录平台作业总时长。

所有候选必须使用同一份数据/tokenizer。评估固定取验证集前部不重叠窗口；`eval_context_length`、`eval_batch_size`、`eval_batches` 单独配置，改变训练 batch 时不要顺手改变验证范围。改变 tokenizer 或长上下文评估协议属于进阶研究，需提供独立的可比指标（例如同一文本上的 bits/byte），当前自动清单不把不同 tokenizer 的 per-token loss 当可比结果。

## 一个可运行示例：RoPE 与 NoPE

示例不是指定必做选题。以下正式配置各处理 40,960,000 token，必须先确认课堂发布的预算档。

```sh
uv run python -m lab_tools.train --config configs/m5_baseline.json --data var/course_data --out var/m5_baseline --device cuda
uv run python -m lab_tools.train --config configs/m5_no_rope.json --data var/course_data --out var/m5_candidate --device cuda
uv run python -m lab_tools.plot var/m5_baseline var/m5_candidate --out var/m5_submission/curve.svg --x tokens
uv run python -m lab_tools.plot var/m5_baseline var/m5_candidate --out var/m5_submission/time_curve.svg --x wall_seconds
uv run python -m lab_tools.generate --checkpoint var/m5_baseline/last.pt --data var/course_data --prompt "Once upon a time" --out var/m5_baseline_sample.json --device cuda --max-new-tokens 128
uv run python -m lab_tools.generate --checkpoint var/m5_candidate/last.pt --data var/course_data --prompt "Once upon a time" --out var/m5_candidate_sample.json --device cuda --max-new-tokens 128
```

如果正式训练预算不足，统一把两个配置的 total_steps/schedule_steps 改成 1000、warmup_steps 改成 20，即每次 8,192,000 token；必要时再统一降到 500 步。改变预算必须对整个对照组同时生效，报告实际值，不能把不同长度训练的终点直接比较。

不要把 M4 的短训练直接续成一条“新 baseline”却重新设置日程；可以从头训练，或事先定义一致的长日程与前缀暂停方案。本框架会拒绝配置/代码改变后的直接恢复。

## 应提交的证据

1. 基线与至少一个改动方案的配置、代码 commit 和运行记录；保留失败实验的 failure.json 并解释。
2. 以训练 token 数和时间为横轴的验证曲线；记录验证范围与随机种子。
3. 相同 prompt、生成长度、temperature/top-p 和随机种子的文本比较，不只挑最好的样本。
4. 训练 token、耗时、吞吐、参数量、峰值 CUDA allocated memory 表格；该显存值不是整张 GPU 的总占用。
5. 分析是否支持假设、可能混杂因素、一次随机种子的局限，以及后续实验建议。

用 [M5 报告模板](../templates/m5_report.md) 和 [清单模板](../templates/m5_submission.json)。模板中的 runs 只放当前要比较的一组运行；探索性短试跑另列报告，避免把不同预算混进同一对照清单。

```sh
uv run python -m lab_tools.check m5 --submission var/m5_submission/submission.json
```

这个命令先回归 M1–M4 代码，再检查研究设施和实际证据。自动通过不等于结论得到科学证明；无收益、短预算下无法判断，同样应如实报告。

## 最后再做：长实验和进阶

第二个种子、更多消融、完整原模型规模、OWT 训练、KV cache、融合 attention 或编译，都在必做完成且仍有配额时进行。不要默认每组用满 8 卡；独立单卡实验通常更适合本课程。没有 B200 leaderboard 的提交要求，也不沿用原文 1.45 的 loss 门槛。
