# M2 报告

- 姓名/学号、代码 commit、测试结果：

## 张量形状

| 环节 | 输入形状 | 输出形状 | 哪个维度的含义发生变化 |
|---|---|---|---|
| Embedding | | | |
| Q/K/V 投影与拆头 | | | |
| attention 分数和 softmax | | | |
| 合头、输出投影 | | | |
| FFN | | | |
| LM head | | | |

## 理解与验证

- Parameter、buffer、ModuleList 各有什么作用？
- 为什么因果 mask 包含对角线？为何 softmax 沿 key 维度？
- pre-norm 残差的两步数据流；模型为什么返回 logits？
- 修改未来输入后的因果性检查；各参数是否有有限梯度？

## 资源

填写指定配置的参数量，并列出 QKV、QKᵀ、对 V 加权、O 投影、FFN 的矩阵形状与 FLOPs。说明 T 变长时变化最大的部分。无需运行大模型。
