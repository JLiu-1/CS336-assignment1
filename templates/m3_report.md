# M3 报告

- 姓名/学号、代码 commit、测试结果：
- 大 logits 下的交叉熵与梯度检查：
- AdamW 一步手算与运行值：
- SGD 不同学习率的现象，附 toy_results.json：
- AdamW toy loss 是否下降：
- warmup/cosine 边界值，附 schedule.svg：
- 裁剪前后的全局梯度范数与方向：
- 为什么 backward 前需要清零；哪些变量是优化器状态？
- P 个 FP32 参数的参数/梯度/m/v 内存，以及估算未包含什么：
