# M1：文本变成 token（个人）

## 目标与范围

本阶段完成 byte-level BPE：把文本转为 UTF-8 bytes，通过反复合并高频相邻项得到词表，再用已学习的合并规则编码新文本。所有实验使用 CPU。

必做为 A1–A4、C1–C2，共 6 个 TODO；B1–B3 是选做。详细概念、边界条件与代码导航见 [BPE 自包含实验说明](tokenizer_lab.md)。下面给出本课程 M1 的完成路径和提交边界。

## 步骤

1. 阅读 `Word(tokens, frequency)`：一个 pre-token 在语料中出现多少次，与其中某个 pair 出现多少次是两回事。
2. 完成 [naive.py](../tokenizer_lab_code/tokenizer_lab/naive.py) 中 A1 加权计数、A2 选 pair、A3 非重叠合并、A4 登记。
3. 完成 [tokenizer.py](../tokenizer_lab_code/tokenizer_lab/tokenizer.py) 中 C1 按 merge rank 编码、C2 拼接 bytes 后解码。
4. 在 tiny 语料训练词表，记录前几次合并；解释重复字符、频次平局、中文和特殊 token。
5. 正式验收，不要求先完成增量优化。

```sh
uv run python -m tokenizer_lab_code.tokenizer_lab.cli pretokenize --input tokenizer_lab_data/tiny.txt --output var/m1/counts.json
uv run python -m tokenizer_lab_code.tokenizer_lab.cli train --cache var/m1/counts.json --variant naive --vocab-size 280 --trace --output var/m1/tokenizer.json
uv run python -m tokenizer_lab_code.tokenizer_lab.cli encode --model var/m1/tokenizer.json --text "你好，hello world!🙂"
uv run python -m lab_tools.check m1
```

一个 `aaa` 包含两个用于统计的 `aa`，但执行一轮非重叠合并只变成 `(aa,a)`。频次相同时比较 bytes pair 元组的字典序。编码新文本时使用训练时的 merge rank，不再按当前文本的频次训练。特殊 token 是硬边界，不能跨过它合并。

## 阶段验收与提交

验收只收集 CPU 设施、A 版与编码解码测试；不收集 M2、增量 BPE 或原作业 1.5 秒速度测试。

提交自己修改的两个文件、测试结果、小词表及合并 trace，按 [报告模板](../templates/m1_report.md) 写简短报告。必须验证中文/emoji roundtrip，并解释为什么 roundtrip 正确仍不足以证明编码规则正确。

本阶段结束，你可以用自己的分词器给 M4 的 demo 生成数据。正式 TinyStories 实验会使用统一发放的 tokenizer 和配套 token 数组，以保证不同组结果可比；仍然调用你自己的编码/解码类处理 prompt 和输出。

## 最后再做：进阶与耗时任务

- B 版：补齐 [incremental.py](../tokenizer_lab_code/tokenizer_lab/incremental.py)，运行 `uv run pytest tokenizer_lab_tests/test_incremental.py -q`；在同一缓存上逐轮比较 A/B。
- 先 correctness，再关闭 debug 测性能；样本级运行三次，比较预处理和 BPE 训练的分项耗时。
- 更大样本、10K/32K 词表、全量编码均非本阶段必做。不要用朴素版直接开始全量语料训练。
