# M4：训练、验证、恢复与生成（三人小组）

## 组队与验收目标

三名成员均先完成 M1–M3。选择一份已验收实现作为共同基础，记录来源 commit 和整合修改。建议分工为训练正确性、评估/恢复、生成；每人至少复核另一人的工作。所有成员都应能解释训练一步和下一 token 预测。

本阶段新增 S1–S3、G1–G3 共 6 个 TODO。其余配置、读盘、保存、绘图、命令行和恢复框架已经提供。原来写过的模型和优化器仍在原位置被调用。

完成后应能够：小数据端到端运行、固定 batch 过拟合、断点继续训练、计算验证损失、生成文本，并完成一次平台预算内的短训练。

## 第一步：理解数据和训练一步

修改 [student.py](../training_lab_code/student.py)。

- S1：给定起点 s 和长度 T，输入为 `data[s:s+T]`，目标为 `data[s+1:s+T+1]`。数据加载器已处理随机起点和设备搬运。
- S2：清零梯度 → 前向和交叉熵 → backward → 全局裁剪 → optimizer.step。训练框架传入的是一组等大小 microbatch；每个 loss 除以 microbatch 数再反传，全部累积完成后只更新一次。
- S3：验证集不同 batch 大小可能不同，按 token 数加权合并 batch 的平均 loss。

`forward_loss` 已处理可选 autocast；调用它后再进行 backward。参数仍是 FP32。`train_step` 返回未除以累积次数的 batch loss 的平均值，用于日志。非有限 loss 应停止，不继续更新。

有效 batch size = `batch_size × gradient_accumulation`；每次更新处理的 token 数再乘 `context_length`。

## 第二步：从预测分布生成

修改 [sampling.py](../generation_lab_code/sampling.py)。G1 选择 `[B,T,V]` 中最后一个位置，G2 用 temperature 缩放 logits 后计算概率，G3 保留 top-p 最短前缀。

例：排序概率 `[0.60,0.25,0.15]`，p=0.70 时保留前两项，不能丢掉首次跨越阈值的第二项。保留后框架会重新归一化再抽样。temperature=0 走已提供的 greedy 分支。

循环、EOS、随机种子和解码已提供。长 prompt 使用最近 context_length 个 token；窗口位置重新从 0 编号。这是简化的滑动窗口生成，不使用 KV cache。模型输出仍是 logits，采样阶段才使用 softmax。

## 第三步：先在 CPU 跑通完整流程

```sh
uv run python -m lab_tools.check m4
uv run python -m lab_tools.prepare --demo --out var/demo_data
uv run python -m lab_tools.verify_training --data var/demo_data --out var/m4_checks
uv run python -m lab_tools.train --config configs/debug.json --data var/demo_data --out var/m4_debug --device cpu
uv run python -m lab_tools.generate --checkpoint var/m4_debug/last.pt --data var/demo_data --prompt "Once upon a time" --out var/m4_debug_sample.json --max-new-tokens 64
uv run python -m lab_tools.plot var/m4_debug --out var/m4_debug_curve.svg
```

prepare 的 demo 会调用你们自己的 M1 A 版训练与编码器。训练和验证文本不同，但它们是人为构造的小样本，只用于连通流程，不用于研究结论。重复运行 prepare 时直接复用已有数据，或指定新的目录。

verify_training 执行 100 次固定 batch 更新，检查 loss 明显下降；另用 CPU 小模型比较连续训练与“训练三步→保存→恢复→再训练三步”。恢复包括模型、优化器和随机状态。结果为 `verification.json`。

## 第四步：亲自观察暂停与恢复

```sh
uv run python -m lab_tools.train --config configs/debug.json --data var/demo_data --out var/m4_resume --stop-after 20
uv run python -m lab_tools.train --config configs/debug.json --data var/demo_data --out var/m4_resume --resume var/m4_resume/last.pt
```

`--stop-after` 是全局步数，不改变配置的学习率日程。恢复时代码、配置和数据身份必须相同；如果修改了模型或超参数，应新建实验目录重新训练。只加载权重继续训练，和完整恢复不是同一件事。

不要共享覆盖同一个输出目录。新训练遇到非空目录会拒绝；明确指定 resume 才会从 checkpoint 继续。仅加载自己或课程可信来源的 checkpoint。

## 第五步：课程数据上的短训练

平台会提供一套目录，包含 `manifest.json`、`tokenizer.json`、`train.npy` 和 `valid.npy`。下面以 `var/course_data` 表示该目录；可替换为平台实际路径，不要重新用 demo 词表编码课程数据。

```sh
uv run python -m lab_tools.train --config configs/m4_pilot.json --data var/course_data --out var/m4_pilot --device cuda --max-seconds 5400
uv run python -m lab_tools.generate --checkpoint var/m4_pilot/last.pt --data var/course_data --prompt "Once upon a time" --out var/m4_sample.json --device cuda --max-new-tokens 128
uv run python -m lab_tools.plot var/m4_pilot --out var/m4_submission/curve.svg
```

默认规模为 500 次更新、每次 8192 token，共 4,096,000 token。时长上限只是停止保护，不是预计耗时；到达上限时状态为 `budget_exhausted`，应在配额内恢复或按课堂发布的降档配置重新运行，不把未完成状态当完整实验。

M4 不要求达到某个固定验证 loss，也不要求生成流畅故事。记录初始与最终损失、曲线、生成文本，以及一个你们实际排查过的问题。课程数据尚未发放时，先完成前四步；不要把 demo 曲线冒充正式结果。

## 提交

将 [清单模板](../templates/m4_submission.json) 复制为 `var/m4_submission/submission.json`，按 [报告模板](../templates/m4_report.md) 写 `var/m4_submission/report.md`。填写真实成员、贡献和文件路径；模板的路径已经对应上述命令。

```sh
uv run python -m lab_tools.check m4 --submission var/m4_submission/submission.json
```

自动检查不评价文本文笔，不按单次跑出的最低 loss 排名。个人理解与贡献通过代码复核记录和演示问答验收。

## 最后再做：扩展

尝试 BF16 或不同梯度累积次数，先比较正确性，再测吞吐。KV cache、多样本并行生成、编译和大模型训练都是选做。更长训练、调参和模型修改统一放到 M5。
