# M2：token 变成预测（个人）

## 本阶段要完成什么

输入是整数 token IDs `[B,T]`，输出是下一 token 的 logits `[B,T,V]`。B 是 batch，T 是序列长度，V 是词表大小，D 是向量维度，H 是头数，Dh=D/H。logits 是未归一化分数；训练损失和生成采样会分别使用它，此处不在模型末尾加 softmax。

本阶段不训练语言模型，不需要 GPU。M1 的词表能提供 token IDs；模型测试直接构造整数输入，帮助你只关注模型正确性。

## 先读框架

```text
token IDs [B,T]
    ↓ Embedding 查表
x [B,T,D]
    ↓ 重复 L 次 TransformerBlock
    │  y = x + Attention(RMSNorm(x))
    │  z = y + FFN(RMSNorm(y))
    ↓ 最终 RMSNorm
    ↓ Linear(D,V)
logits [B,T,V]
```

`nn.Parameter` 注册可学习权重；`ModuleList` 注册各层，保证 `.parameters()`、`.to(device)` 和 `state_dict()` 可以找到它们。RoPE 的 sin/cos 是固定缓存，使用 `register_buffer`；它随设备移动，但不是优化器要更新的参数。本框架使用 `persistent=False`，不把可重建缓存存入权重快照。

构造函数、初始化、参数加载及多头变形都已提供。只在 TODO 中完成计算，不重新设计类接口。权重初始化使用截断正态；Linear 没有 bias。

## 第一关：基础层 T1–T6

修改 [layers.py](../transformer_lab_code/layers.py)。

| TODO | 计算 | 必须注意 |
|---|---|---|
| T1 Linear | `Y = X Wᵀ` | W 存为 `[Dout,Din]`；保留任意前导维度 |
| T2 Embedding | 按 ID 选择权重矩阵的行 | `[B,T]` 变为 `[B,T,D]`，不是 one-hot 分类输出 |
| T3 RMSNorm | `x / sqrt(mean(x²)+eps) * gain` | 只沿最后一维；不减均值；FP32 转换已提供 |
| T4 SiLU | `x * sigmoid(x)` | 可使用 torch.sigmoid |
| T5 SwiGLU | `w2(SiLU(w1(x)) * w3(x))` | 中间乘法逐元素进行；输出维度回到 D |
| T6 Softmax | `exp(x-max(x))/sum(exp(x-max(x)))` | 沿传入 dim；归约时保留维度以便广播 |

```sh
uv run pytest tests/test_model.py -k "linear or embedding or rmsnorm or silu or swiglu" -q
uv run pytest tests/test_nn_utils.py -k softmax -q
```

这些测试仍可能提示相关 TODO 未完成；先补基础层再进入后续关卡。不得用内置同名层替代本阶段实现。

## 第二关：Attention 与 RoPE T7–T9

修改 [attention.py](../transformer_lab_code/attention.py)。Attention 将 query 与各 key 的相似度变成权重，再加权求和 value：

`Attention(Q,K,V) = softmax(Q Kᵀ / sqrt(Dh)) V`

`Q:[...,Qlen,Dh]`、`K:[...,Klen,Dh]`、`V:[...,Klen,Dv]`。矩阵转置只交换最后两个维度；softmax 沿 key 维度计算，输出为 `[...,Qlen,Dv]`。

T7 要调用已提供的 `apply_mask`。本实验统一 **True 表示允许关注**。屏蔽位置在 softmax 前设为负无穷；设为零会保留非零概率。基础任务保证每行至少有一个可见位置。

T9 构造因果 mask：第 i 行允许第 0 到第 i 列。当前位置能够看到自己，不能看到未来 token。注意：给定当前位置输入，模型预测的是下一个 token，所以这不会泄漏目标。

多头拆分已提供：`[B,T,D] → [B,H,T,Dh]`。各头并行做 Attention 后恢复 `[B,T,D]`，再经过输出投影。Q/K/V 是投影后的激活，投影矩阵才是参数。

RoPE 把每对坐标 `(a,b)` 按位置旋转：`(a cosθ-b sinθ, a sinθ+b cosθ)`。T8 只补这两条公式，缓存、按位置查询、广播和重新交错已提供。旋转只用于 Q、K，不用于 V；每头使用相同位置规则，旋转维度为 Dh。

```sh
uv run pytest tests/test_model.py -k "attention or rope" -q
```

## 第三关：完整模型 T10–T11

修改 [model.py](../transformer_lab_code/model.py)。T10 写两段 pre-norm 残差；第二段处理第一段得到的 y。T11 用 embedding 开始，顺序遍历 layers，经最终 norm 和 lm_head 返回 logits。不要 detach、转成 numpy，或在 forward 中重新创建参数。

```sh
uv run python -m lab_tools.check m2
```

正式验收包含 M1 回归、所有模型数值快照，以及因果性、位置零旋转、范数保持和梯度检查。微型测试验证计算正确性，不证明大模型一定收敛。

## 报告与资源计算

使用 [M2 报告模板](../templates/m2_report.md)，说明 tensor 各维含义，并填写一个 block 的矩阵规模：Q/K/V/O 各 `D×D`，SwiGLU 三个矩阵共 `3DF` 个权重；两个 RMSNorm 共 `2D` 个参数。完整模型未共享输入输出 embedding 时：

`参数量 = 2VD + L(4D² + 3DF + 2D) + D`。

矩阵乘法 `[m,n] @ [n,p]` 约需 `2mnp` FLOPs。填写 QKV、注意力分数、加权 V、输出投影和 FFN 的计算量，解释序列长度增加时哪些项按 T² 增长。不要为计算参数量去实例化 GPT-2 XL。

## 最后再做：扩展

尝试不同 batch/序列长度，检查形状；解释 RoPE 的相对位置含义。融合 QKV、KV cache、FlashAttention 不属于本阶段必做。M5 的配置开关先不要改变，保持默认基线能通过原作业测试。
