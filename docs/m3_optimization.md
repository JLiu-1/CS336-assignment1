# M3：让参数学起来（个人）

## 目标

M2 给出 logits；本阶段把预测误差变成标量 loss，通过 autograd 求梯度，再更新参数。只用小张量验证算法，不申请 GPU，也不开始长时间语言模型训练。

优化器的继承、参数遍历、状态字典和原地赋值已提供。你负责 O1–O5 共 5 个数学计算 TODO。

## O1：稳定交叉熵

文件：[losses.py](../training_lab_code/losses.py)。输入 logits `[...,V]` 与 targets `[...]` 已整理为 `[N,V]` 和 `[N]`。

单样本损失为 `log(sum_j exp(z_j)) - z_target`。直接计算 exp 容易溢出；设 `u=z-max(z)`，损失保持为 `log(sum_j exp(u_j)) - u_target`。用已提供的 `target_scores` 从各行取出目标分数，最后对 N 个样本求平均。不要先算 softmax 再取 log，也不要 detach。

困惑度 `perplexity=exp(mean_loss)`。它依赖 tokenizer 和评估数据，不能跨不同词表直接比较。

```sh
uv run pytest tests/test_nn_utils.py -k cross_entropy -q
```

## O2/O3：AdamW 的数学部分

文件：[optimizer.py](../training_lab_code/optimizer.py)。每个参数的状态包括步数 t、一阶矩 m 和二阶矩 v，t 从 1 开始。本课程对齐原作业 Algorithm 1：

```text
m_new = beta1 * m + (1-beta1) * gradient
v_new = beta2 * v + (1-beta2) * gradient²
alpha = lr * sqrt(1-beta2^t) / (1-beta1^t)
parameter_new = parameter_old * (1-lr*weight_decay)
                - alpha * m_new / (sqrt(v_new)+eps)
```

O2 返回更新后的 m/v，O3 返回新参数。框架在 `torch.no_grad()` 中复制回参数与状态，你不需要写 `.data` 操作。没有梯度的参数会被跳过，不能给它偷偷施加 weight decay。

注意这里 eps 的位置是本课程约定；不要从不同教程拼接多套“等价”公式，特别是较大 eps 时它们会有数值差异。

```sh
uv run pytest tests/test_optimizer.py -k adamw -q
```

## O4：warmup + cosine

t 表示已经完成的优化器更新数。第一个更新使用 t=0 的学习率。设 Tw 为 warmup 步数，Tc 为整个日程结束步数：

- `t<Tw`：`lr=max_lr*t/Tw`；
- `Tw<=t<=Tc`：`lr=min_lr+(max_lr-min_lr)*(1+cos(pi*(t-Tw)/(Tc-Tw)))/2`；
- `t>Tc`：`lr=min_lr`。

支持 `Tw=0`：直接走 cosine 分支，不除以零。框架要求 `0<=Tw<Tc`。`schedule_steps` 控制日程，`total_steps` 控制本次实验训练到哪里；短试跑可以保留较长日程。

## O5：全局梯度裁剪

将所有参数梯度视为一个向量，其范数是 `sqrt(sum_i sum(g_i²))`。范数超过 M 时，所有梯度乘同一系数 `M/(norm+1e-6)`；否则不变。框架提供梯度收集和缩放操作。计算范数用 FP32，不要逐参数独立裁剪。

## CPU 实验与正式验收

```sh
uv run python -m lab_tools.toy_optimization --out var/m3
uv run python -m lab_tools.check m3
```

`toy_results.json` 包含四种学习率的 SGD 结果、AdamW 收敛记录和学习率曲线数据；`schedule.svg` 可直接打开。SGD 示例已提供，不需要再写一个优化器类。

按 [报告模板](../templates/m3_report.md) 解释：哪些学习率稳定/发散、为什么要清零梯度、AdamW 为什么有额外内存、warmup 的作用，以及 clipping 对梯度方向和长度的影响。

## 资源理解

若模型有 P 个 FP32 参数，参数、梯度、m、v 各约 `4P` bytes，合计约 `16P` bytes。这不包括激活、临时结果、缓存与运行时开销，不能拿它当峰值显存预测。M4/M5 使用 BF16 autocast 时，框架仍保持参数和 AdamW 状态为 FP32。

## 最后再做：选做

增加随机梯度对照、绘制 loss 曲线，或估计激活显存。原作业的 H100 MFU 推算和 GPT-2 XL 复杂内存推导不属于必做；不需要运行这些大模型。
