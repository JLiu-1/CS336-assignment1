# BPE 分词器实验：从全量统计到局部更新

本实验是课程教师在 CS336 Assignment 1 上添加的教学框架。环境已经配置完成；本部分只使用 CPU，不申请 GPU。必做为 A1–A4、C1–C2 共 6 个 TODO；B1–B3 是增量优化选做。课程阶段入口见 [M1](m1_tokenizer.md)。工程设施由教师提供，不要求你从空文件开始写。

**待补充代码统一位于 `tokenizer_lab_code/tokenizer_lab/`。** 从下面三个文件开始：

- [naive.py：A 版，填写 A1–A4](../tokenizer_lab_code/tokenizer_lab/naive.py)
- [incremental.py：B 版，填写 B1–B3](../tokenizer_lab_code/tokenizer_lab/incremental.py)
- [tokenizer.py：共用编码器，填写 C1–C2](../tokenizer_lab_code/tokenizer_lab/tokenizer.py)

本说明位于仓库的 `docs/tokenizer_lab.md`；测试仍在 `tokenizer_lab_tests/`，小语料仍在 `tokenizer_lab_data/`。以下命令均从仓库根目录执行，不需要进入代码子目录。

## 1. 学习目标与范围

- 理解 byte-level BPE 如何从训练文本得到词表与有序的合并规则。
- 区分相邻 pair 的计数、一次不重叠合并，以及编码时使用的 merge rank。
- 理解如何只更新受影响的 pre-token，并验证增量算法与全量算法等价。
- 用训练出的分词器编码/解码，为后续 Transformer 实验准备接口。

推荐顺序：A1 → A2 → A3 → A4 → C1 → C2 → B1 → B2 → B3。B 版复用 A 版，不是另一套必须从头重写的代码。

## 2. 文件导航

| 文件 | 用途 | 是否需要修改 |
|---|---|---|
| `tokenizer_lab_code/tokenizer_lab/naive.py` | A 版全量统计与主循环 | 只填 A1–A4 |
| `tokenizer_lab_code/tokenizer_lab/incremental.py` | B 版局部更新与主循环 | 只填 B1–B3 |
| `tokenizer_lab_code/tokenizer_lab/tokenizer.py` | 两版共用的编码与解码 | 只填 C1–C2 |
| `common.py` | 数据结构、初始化、局部 pair 计数 | 教师提供 |
| `pretokenize.py` | 特殊 token、正则预分词、CPU 并行 | 教师提供 |
| `agenda.py` | 最佳 pair 的优先队列、过期条目清理 | 教师提供 |
| `storage.py` / `cli.py` / `api.py` | 缓存、保存加载、命令行与原作业接口 | 教师提供 |
| `tokenizer_lab_tests/` | 分层测试；不依赖 PyTorch | 运行即可 |
| `tokenizer_lab_data/tiny.txt` | 首次实验用的小语料，含中文/emoji | 可以观察 |

表中的 `common.py`、`pretokenize.py`、`agenda.py`、`storage.py`、`cli.py` 和 `api.py` 也都位于 `tokenizer_lab_code/tokenizer_lab/`。

仓库根目录的 `test.py` 和 `tests/parallel_tokenizer.py` 是教师早期原型，保留供参考；正式实验使用上表入口。不要直接运行原型中的完整 TinyStories 训练来开始本实验。

## 3. 首先认识数据

教师提供的预分词器返回 `Counter[bytes]`，例如 `Counter({b'aaa': 3, b'ab': 2})`。它表示这些 pre-token 在语料中的出现次数，而不是 token ID。

框架把它转为 `Word` 列表：

| word_id | tokens | frequency |
|---|---|---:|
| 0 | `(b'a', b'a', b'a')` | 3 |
| 1 | `(b'a', b'b')` | 2 |

`word_id` 是列表下标，在整个训练中保持不变；合并只改变 `tokens`，不改变 `frequency`。这里的 “word” 指 pre-token，不一定是自然语言中的一个单词。

`common.adjacent_pairs(sequence)` 已提供，返回一个序列中所有相邻 pair 的次数；它没有乘上 frequency。

关键约定：

1. token 始终是 `bytes`，合并后可以包含多个字节；token ID 是整数。
2. 单个 `aaa` 中有两个相邻 `aa`，出现三次的 `aaa` 对 `aa` 的频次贡献为 6。
3. 执行一次合并时，从左到右不重叠合并，所以 `aaa` 变为 `(aa, a)`。
4. 不统计也不合并跨 pre-token 的 pair；特殊 token 同样是硬边界。
5. 频次平局按 **pair 的 bytes 元组**取字典序较大者，不比较拼接后的 bytes。
6. 目标词表大小包含基础 256 字节和特殊 token；没有候选时允许提前结束。

## 4. A 版：每轮重新统计

| TODO | 函数 | 你要完成的工作 |
|---|---|---|
| A1 | `count_pairs` | 汇总所有 Word 的加权 pair 频次 |
| A2 | `select_pair` | 选择最优正频次 pair；无候选返回 None |
| A3 | `merge_sequence` | 一个序列内从左到右、不重叠地合并 |
| A4 | `record_merge` | 登记合并后的词条，追加有序 merges |

A4 使用 `result.register_token(raw)` 登记新词条。该教师函数负责分配连续 ID、维护反向索引，以及复用已经存在的 bytes，避免你处理重复词条和反复扫描词表的细节。

每个 TODO 的输入、输出、边界情况都写在函数 docstring 中。A 版通过测试后，可以作为 B 版的正确性参照。无需修改教师主循环，也无需满足原作业 1.5 秒性能门槛。

## 5. B 版：只更新受影响的 pre-token

B 版教师主循环已经负责：选出 pair、取得受影响 word_id 的快照、调用 A3 合并、计算该 word 合并前后的局部计数、通知候选队列更新。

你只需要完成：

| TODO | 函数 | 正确性要求 |
|---|---|---|
| B1 | `build_index` | pair → 含有它的 word_id 集合；同一 word 只登记一次 |
| B2 | `update_counts` | 用一个 word 的前后变化维护全局频次，乘上 frequency，保留其他 word 的贡献 |
| B3 | `update_index` | 维护 pair 的存在性索引，保留其他 ID，删除空集合 |

注意 B2 处理的是**出现次数**，B3 处理的是**存在性**。一个 pair 的局部次数从 3 变为 1 时，word_id 仍须保留在索引中。

B 版是 pre-token 级局部更新：会重新扫描受影响的 pre-token，不维护每一个 token 的链表位置。堆和过期条目机制已经提供，不是本次 TODO。

`--debug` 会在每轮用 A1 全量重算并检查索引。它用于排错，不用于测量加速比；A1 本身必须先通过独立的预期值测试。

## 6. C 部分：两版共用的编码解码

| TODO | 函数 | 正确性要求 |
|---|---|---|
| C1 | `Tokenizer.encode_pretoken` | 根据训练得到的 merge rank 编码普通 pre-token |
| C2 | `Tokenizer.decode` | token ID → bytes，全部拼接后再统一做 UTF-8 解码 |

框架提供 `byte_to_id`、`merge_ranks`、特殊 token 的最长匹配、预分词和 `encode_iterable`。

训练阶段选择当前频次最大的 pair；编码阶段使用已学好的规则优先级（rank 越小越优先），不重新训练，也不简单地选择最长词条。

一个汉字或 emoji 的多个字节可能分布在多个 token 中，因此不能逐个 token 解码成字符串。C2 要使用 UTF-8 的 `errors='replace'` 处理任意不完整字节序列。

`encode_iterable` 按输入字符串逐项编码，不会将整个迭代器一次读入内存。每项应视为完整编码单元：对 `['hel', 'lo']` 的编码不保证与对 `'hello'` 的编码产生相同 IDs；这不是任意字符切块都等价的流式 BPE。

## 7. 运行顺序

以下命令都从仓库根目录运行。首次使用无需下载完整数据，也无需申请 GPU。

### 7.1 验证教师设施

```sh
uv run pytest tokenizer_lab_tests/test_support.py -q
```

此文件应在未完成任何 TODO 时通过。学生测试会因 `NotImplementedError` 失败，这是正常的未完成状态，不是环境错误。不要删测试、改断言或把 TODO 标成 xfail。

### 7.2 生成两版共用的预分词缓存

```sh
uv run python -m tokenizer_lab_code.tokenizer_lab.cli pretokenize --input tokenizer_lab_data/tiny.txt --output var/tokenizer_lab/tiny-counts.json
```

默认进程数是 1。教师允许时，可增加 `--workers 2`。多个特殊 token 或没有特殊 token 时会保守退回串行；仅有一个特殊 token 时才沿该边界并行切块。单进程/单块会将对应文本块读入内存，本框架不承诺常数内存的全量预处理。

默认特殊 token 为 `<|endoftext|>`；可重复使用 `--special-token` 指定多个特殊 token，或使用 `--no-special-tokens`。缓存保存 pattern、特殊 token 配置和原始文件 SHA-256。使用缓存意味着使用该快照；原文变化后需重新生成，框架不会自动重新检查原始文件。

### 7.3 完成 A 版

```sh
uv run pytest tokenizer_lab_tests/test_naive.py -q
uv run python -m tokenizer_lab_code.tokenizer_lab.cli train --cache var/tokenizer_lab/tiny-counts.json --variant naive --vocab-size 280 --trace --output var/tokenizer_lab/naive.json
```

如提前没有可合并的 pair，实际词表可能小于 280。请检查模型文件中十六进制表示的 vocab、merges 和 trace，解释前几轮合并。

### 7.4 完成 C 部分

```sh
uv run pytest tokenizer_lab_tests/test_tokenizer.py -q
uv run python -m tokenizer_lab_code.tokenizer_lab.cli encode --model var/tokenizer_lab/naive.json --text "你好，hello world!🙂"
```

### 7.5 完成 B 版，先核对正确性

```sh
uv run pytest tokenizer_lab_tests/test_incremental.py -q
uv run python -m tokenizer_lab_code.tokenizer_lab.cli compare --cache var/tokenizer_lab/tiny-counts.json --vocab-size 280 --debug --output var/tokenizer_lab/check.json
```

compare 对照每一轮选中的 pair、频次、最终 vocab 和 merges；不一致时报告首个不同的轮次。局部序列与计数的一致性由 B 版 debug 检查辅助定位。

### 7.6 测量性能

```sh
uv run python -m tokenizer_lab_code.tokenizer_lab.cli pretokenize --input tests/fixtures/tinystories_sample.txt --output var/tokenizer_lab/sample-counts.json
uv run python -m tokenizer_lab_code.tokenizer_lab.cli compare --cache var/tokenizer_lab/sample-counts.json --vocab-size 500 --repeats 3 --output var/tokenizer_lab/benchmark.json
```

此时不要加 `--debug`。输入准备时间单独报告；两版交替运行顺序并使用相同缓存、词表目标和轻量 trace。先用小样本完成上述流程，再由教师决定是否使用 5MB 样本或更大语料。小数据上的优化版可能更慢；不能预设必须获得某个固定加速倍数。

## 8. 与原 CS336 测试衔接

`tests/adapters.py` 的两个分词相关入口已经连接到本框架，模型、优化和训练入口也已接线；每个阶段只验收对应的学生实现。

完成 A/C 后，在课程配置的原作业环境中运行：

```sh
uv run pytest tests/test_train_bpe.py -k "not speed"
uv run pytest tests/test_tokenizer.py
```

原 `tests/test_tokenizer.py` 使用 Unix `resource`，应在课程 Linux 平台运行；新增 `tokenizer_lab_tests` 可在 Windows/Linux 运行，且不依赖 torch。运行整仓库 pytest 会包含尚未完成的 Transformer 等测试，不是本 milestone 的验收方式。

原 adapter 默认选择 A 版。完成 B 后可设置 `CS336_BPE_VARIANT=incremental` 再运行原 BPE 测试：

```sh
# Linux shell
CS336_BPE_VARIANT=incremental uv run pytest tests/test_train_bpe.py -k "not speed"
```

```powershell
# PowerShell
$env:CS336_BPE_VARIANT = "incremental"
uv run pytest tests/test_train_bpe.py -k "not speed"
Remove-Item Env:CS336_BPE_VARIANT
```

训练输出仍为 `vocab: dict[int, bytes]` 与有序 `merges`。正式 LM 实验必须将分词器文件和它编码出的 token IDs 配套使用，不可混用其他词表。

## 9. 提交内容

1. 三个学生文件中的实现；不得改教师测试来取得通过。
2. A/C 的测试结果；选择 B 版任务时附 B 测试和 compare 结果。
3. 词表文件、合并过程记录、中文/emoji/特殊 token 的编码解码例子。
4. 简短报告：解释加权计数、重叠计数与非重叠合并、tie-break、训练频次与编码 rank 的区别。
5. B 版附性能表：语料、唯一 pre-token 数、目标/实际词表大小、CPU/进程配置、分阶段耗时、重复次数，以及为什么优化在某些输入上不明显。

报告中的运行时间应来自你自己的测量。速度分不替代正确性分，也不根据其他同学占用共享 CPU 时的一次绝对耗时直接打分。
