下面是一版**实践执行版实验设计**。我按你的新定位来组织：**静态实验是核心，动态实验是静态筛选方法在合成数据迭代场景中的扩展验证**。在这版中，我加入了 **PPL-based selection baseline**，用于证明你的方法不是简单的 likelihood / fluency / difficulty heuristic。

------

# 0. 总体实验定位

你的实验主线建议写成：

\[
\textbf{我们的指标能否从 synthetic math reasoning 数据中筛选出更适合 SFT 的样本？}
\]

并进一步验证：

\[
\textbf{这种静态筛选能力能否帮助 iterative synthetic data generation？}
\]

所以实验优先级是：

| 优先级 | 实验模块 | 作用 |
| ------ | -------- | ---- |
| 最高 | 静态数据筛选实验 | 证明你的指标本身有效 |
| 高 | 静态实验后的 SVD / 奇异谱分析 | 解释为什么 Ours 比 naive SFT 更好 |
| 中 | 动态 iterative-SFT 实验 | 证明你的指标能接入合成数据闭环 |
| 中 | 7B scaling | 证明不是 3B 特例 |

加入 PPL 后，静态实验还需要额外回答一个问题：

\[
\textbf{我们的指标是否优于简单的 perplexity-based data selection？}
\]

也就是证明：

\[
\textbf{Ours 不是简单地选择低 PPL、 高 PPL，或按 PPL 分布均衡采样的数据。}
\]

------

# 1. 实验需要的数据集

## 1.1 训练数据

主训练数据建议使用：

\[
\text{OpenR1-Math-220k} \rightarrow \text{Math-Syn-CoT}
\]

你需要把数据拆成几部分：

| 数据部分 | 用途 |
| -------- | ---- |
| Candidate pool | 静态实验中用于打分、排序、筛选 |
| Static train set | 从 candidate pool 中选出的 SFT 训练数据 |
| Dynamic initial set \(D_0\) | 动态实验第一轮 SFT 的初始训练集 |
| Dynamic generated candidates \(C_t\) | 每一轮由模型或 teacher 生成的新候选数据 |
| Validation set | 选 checkpoint、调阈值、确定早停 |
| Held-out OpenR1-Math test | Target-ID 评估，不能参与筛选或训练 |

建议你一定保留一个严格的 held-out split，避免 reviewer 质疑你的方法只是在 OpenR1-Math 分布内过拟合。

------

## 1.2 评估数据

评估集分三组，不要混在一起平均。

| 评估组 | Benchmark | 作用 |
| ------ | --------- | ---- |
| Target-ID | Held-out OpenR1-Math | 是否学会当前 synthetic math 分布 |
| Target-near / math generalization | MATH500、AIME24，OlympiadBench 可选，minerva_math500可选 | 是否泛化到标准数学推理 |
| General OOD retention | MMLU、TruthfulQA、WinoGrande、IFEval、HumanEval 或 MBPP，ETHICS 可选 | 是否伤害通用能力 |

主表里建议至少保留：

\[
\text{MMLU},\ \text{TruthfulQA},\ \text{WinoGrande},\ \text{IFEval},\ \text{HumanEval/MBPP}
\]

ETHICS 可以放 appendix，除非你的论文特别强调 safety 或 value retention。

------

# 2. 实验需要的模型和其它准备

## 2.1 主模型

3B 是完整实验模型。

| 角色 | 推荐模型 |
| ---- | -------- |
| 3B 主实验 | Qwen2.5-3B 或同级别 3B math-capable base model |
| 7B 扩展验证 | Qwen2.5-7B 或同系列 7B model |

如果你希望 3B 和 7B 可比，最好使用同一个模型家族，例如：

\[
\text{Qwen2.5-3B} \rightarrow \text{Qwen2.5-7B}
\]

不要 3B 用 Qwen，7B 用 Llama，否则 scaling 结论会混入模型家族差异。

------

## 2.2 训练准备

需要提前固定这些东西：

| 项目 | 要求 |
| ---- | ---- |
| Tokenizer | 所有方法共用同一个 tokenizer |
| SFT format | 固定 prompt / response / CoT 格式 |
| Training recipe | learning rate、batch size、epoch、warmup、max length 固定 |
| Token budget | 同一组比较中总训练 token 尽量一致 |
| Checkpoint selection | 用 validation set 选，不用 test set |
| Answer extraction | 数学评估需要统一答案抽取规则 |
| Evaluation script | 所有方法共用同一套 eval pipeline |
| Seed | 建议使用 42、43、44 |

主实验建议使用 full-parameter SFT。LoRA 可以作为补充，但不要作为主结果，否则 reviewer 可能会问 full SFT 下是否仍成立。

------

## 2.3 你的指标 pipeline

你需要有一个稳定的 scoring pipeline：

\[
x_i \rightarrow s_i
\]

其中 \(x_i\) 是一条 synthetic math reasoning 数据，\(s_i\) 是你的指标分数。

需要记录：

| 内容 | 说明 |
| ---- | ---- |
| raw score | 每条数据的原始指标分数 |
| normalized score | 是否做归一化 |
| score percentile | 位于候选池的哪个分位 |
| selected flag | 是否被选入训练集 |
| correctness flag | 如果有 verifier，需要记录答案是否正确 |
| length / token count | 控制长度偏差 |
| source / generation round | 静态数据还是第几轮生成数据 |

------

## 2.4 PPL pipeline

为了加入 PPL baseline，需要对 candidate pool 中每条样本计算 response-level PPL。

建议定义：

\[
\text{PPL}(y_i \mid x_i)
=
\exp\left(
-\frac{1}{T}
\sum_{t=1}^{T}
\log p_{\theta_0}(y_t \mid x_i, y_{<t})
\right)
\]

其中：

| 符号 | 含义 |
| ---- | ---- |
| \(x_i\) | problem / prompt |
| \(y_i\) | reasoning + answer |
| \(\theta_0\) | base model |
| \(T\) | response token 数 |

建议只在 **response / CoT answer 部分**计算 PPL，而不是对 prompt + response 全部计算。原因是你要筛选的是 synthetic reasoning trace 是否适合 SFT，而不是题目本身是否常见。

新增数据字段：

| 字段 | 说明 |
| ---- | ---- |
| ppl_response | response-only length-normalized PPL |
| nll_response | response-only token 平均 NLL |
| ppl_percentile | 在 candidate pool 中的 PPL 分位 |
| ppl_bucket | low / mid / high |
| token_count_response | response token 数 |
| answer_correct | verifier 或 gold answer 判断 |
| ours_score | 你的指标分数 |
| ours_score_percentile | 你的指标分位 |
| selected_by | 被哪种方法选中 |

PPL baseline 的核心作用是排除以下质疑：

| 质疑 | PPL baseline 的作用 |
| ---- | ------------------ |
| 你的方法是不是只是在选更流畅、更自然的数据？ | 对比 PPL-low |
| 你的方法是不是只是在选更困难、更 surprise 的数据？ | 对比 PPL-high |
| 你的方法是不是只是 likelihood balancing？ | 对比 PPL-balanced |

------

# 3. 静态实验 3B 需要准备的 baseline

静态实验是你的核心实验。目标是回答：

\[
\textbf{在已有 synthetic CoT pool 中，我们的指标能不能筛出更好的 SFT 数据？}
\]

加入 PPL 后，还要回答：

\[
\textbf{我们的指标是否优于 simple PPL heuristic？}
\]

------

## 3.1 必须跑的 baseline

| Method | 是否必须 | 说明 |
| ------ | -------- | ---- |
| Base | 必须 | 未微调模型 |
| Vanilla SFT | 必须 | 标准 SFT，对照 naive synthetic SFT |
| Random-k | 必须 | 从 candidate pool 随机选 \(k\) 条 |
| Verifier-filter / correctness-filter | 建议必须 | 只用答案正确性或 verifier 筛选 |
| PPL-low | 建议 | 选择 base model PPL 最低的一部分样本 |
| PPL-high | 建议 | 选择 base model PPL 最高的一部分样本 |
| PPL-balanced | 必须 | 按 PPL bucket 均衡采样 |
| CFS | 如果是相关工作，必须 | 用于和你提到的相关方法对比 |
| Ours-low | 必须 | 选指标最低的一部分 |
| Ours-high | 必须 | 选指标最高的一部分 |
| Ours-balanced / Ours-final | 必须 | 你的最终静态筛选策略 |

这里最关键的是 **Ours-low / Ours-high / Ours-final**。它们不是普通 ablation，而是证明你的指标排序有语义。

PPL 系列 baseline 的定位是：

\[
\textbf{证明 Ours 不是 perplexity / likelihood / fluency heuristic。}
\]

------

## 3.2 PPL baseline 的具体筛选规则

假设静态训练大小是 10k。

| Method | 选择规则 |
| ------ | -------- |
| PPL-low | 选择 response-level PPL 最低的 10k |
| PPL-high | 选择 response-level PPL 最高的 10k |
| PPL-mid | 选择 PPL 位于中间区间的 10k，例如 40%–60% |
| PPL-balanced | 从 low / mid / high PPL bucket 中按比例采样 |

最推荐主表使用：

\[
\textbf{PPL-balanced}
\]

原因是 PPL-low 和 PPL-high 都比较极端，容易被认为是 weak baseline；PPL-balanced 更公平。

如果版面有限：

| 位置 | 放哪些 PPL baseline |
| ---- | ------------------ |
| 主表 | PPL-balanced |
| Appendix | PPL-low、PPL-high、PPL-mid 可选 |

------

## 3.3 推荐静态 baseline 列表

10k 主设置建议跑完整 baseline：

| Method | Train size | 作用 |
| ------ | ---------- | ---- |
| Base | 0 | 未微调模型 |
| Vanilla SFT | 10k | naive SFT |
| Random-k | 10k | 随机选择 |
| Verifier-filter | 10k | correctness heuristic |
| PPL-low | 10k | 低困惑度样本 |
| PPL-high | 10k | 高困惑度样本 |
| PPL-balanced | 10k | 困惑度均衡筛选 |
| CFS | 10k | 相关工作 baseline |
| Ours-low | 10k | 你的低分样本 |
| Ours-high | 10k | 你的高分样本 |
| Ours-balanced / Ours-final | 10k | 你的最终静态筛选策略 |

如果主表太大，可以压缩为：

| Method | Train size |
| ------ | ---------- |
| Base | 0 |
| Vanilla SFT | 10k |
| Random-k | 10k |
| Verifier-filter | 10k |
| PPL-balanced | 10k |
| CFS | 10k |
| Ours-low | 10k |
| Ours-high | 10k |
| Ours-final | 10k |

如果 CFS 是重要相关工作，就必须保留 CFS；如果不是核心对手，可以放 appendix。

------

# 4. 静态实验 3B 需要观察的数据和重复维度

## 4.1 静态实验需要重复的维度

静态实验建议按这个优先级做：

| 维度 | 推荐设置 | 说明 |
| ---- | -------- | ---- |
| Train size | 5k、10k、20k | 10k 是主设置，5k/20k 用于趋势 |
| Seed | 42、43、44 | 关键方法做 3 seeds |
| Method | 所有 baseline | 10k 跑完整 baseline |
| Model | 3B | 静态主实验只先做 3B |

加入 PPL 后，建议这样分配成本：

| Train size | 需要跑的方法 |
| ---------- | ------------ |
| 5k | Random-k、Verifier-filter、PPL-balanced、Ours-final |
| 10k | 所有 baseline，包括 PPL-low / PPL-high / PPL-balanced |
| 20k | Random-k、Vanilla SFT、PPL-balanced、Ours-final |

10k 是最完整的主表。5k 和 20k 主要用于证明趋势，不需要所有 ablation 都跑。

------

## 4.2 静态实验需要记录的指标

核心指标包括：

\[
\text{Target Gain}
=
\text{Target}_{\text{SFT}}
-
\text{Target}_{\text{Base}}
\]

\[
\text{OOD Drop}
=
\text{OOD}_{\text{Base}}
-
\text{OOD}_{\text{SFT}}
\]

\[
\text{General OOD Avg}
=
\text{Avg}(\text{MMLU},\ \text{TruthfulQA},\ \text{WinoGrande},\ \text{IFEval},\ \text{Code})
\]

还需要额外记录 PPL 相关指标：

| 指标 | 作用 |
| ---- | ---- |
| PPL mean / std | 证明 PPL baseline 确实选到了不同 likelihood 区间 |
| PPL percentile range | 显示选择区间 |
| Avg response length | 控制 PPL 是否被长度影响 |
| Correctness rate | 控制 PPL-high 是否只是选了错误 / 噪声样本 |
| Ours score mean / std | 展示不同 baseline 在你的指标空间中的分布 |

主结论不是单纯追求 math 分数最高，而是：

\[
\textbf{Target-domain gain 更高，同时 OOD drop 更小。}
\]

加入 PPL 后，静态实验最好能得到这样的结论：

| Method | 预期现象 |
| ------ | -------- |
| PPL-low | 可能 OOD drop 小，但 target gain 不够 |
| PPL-high | 可能 target gain 高一些，但不稳定或 OOD drop 大 |
| PPL-balanced | 比 low/high 稳定，但通常不如 Ours |
| Ours-final | 更好的 target/OOD trade-off |

------

## 4.3 静态实验主记录表

建议每个 run 都记录成下面这张表。

| Run ID | Model | Method | Train size | Data seed | Train seed | Train tokens | Ours score mean | Ours score std | PPL mean | PPL std | Avg response length | Correctness rate | Target-ID ↑ | MATH500 ↑ | AIME24 ↑ | MMLU ↑ | TruthfulQA ↑ | WinoGrande ↑ | IFEval ↑ | HumanEval/MBPP ↑ | General OOD Avg ↑ | Target Gain ↑ | OOD Drop ↓ |
| ------ | ----- | ------ | ---------- | --------- | ---------- | ------------ | --------------- | -------------- | -------- | ------- | ------------------- | ---------------- | ----------- | --------- | -------- | ------ | ------------ | ------------ | -------- | ---------------- | ----------------- | ------------- | ---------- |
| static-3b-001 | 3B | Base | 0 | - | 42 | 0 | - | - | - | - | - | - |  |  |  |  |  |  |  |  |  | 0 | 0 |
| static-3b-002 | 3B | Vanilla SFT | 10k | 42 | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-003 | 3B | Random-k | 10k | 42 | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-004 | 3B | Verifier-filter | 10k | 42 | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-005 | 3B | PPL-low | 10k | - | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-006 | 3B | PPL-high | 10k | - | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-007 | 3B | PPL-balanced | 10k | 42 | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-008 | 3B | CFS | 10k | 42 | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-009 | 3B | Ours-low | 10k | - | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-010 | 3B | Ours-high | 10k | - | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| static-3b-011 | 3B | Ours-final | 10k | - | 42 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

注意：

- Random-k 必须有 data seed。
- PPL-low / PPL-high 如果排序是 deterministic，可以没有 data seed。
- PPL-balanced 如果从 bucket 中采样，需要有 data seed。
- Ours 如果 ranking 是 deterministic，可以没有 data seed，但最好保留 train seed。
- 如果 Ours 的 score 或 selection 有随机性，也要记录 selection seed。
- 所有方法都要记录 train tokens，避免 reviewer 质疑训练预算不公平。

------

## 4.4 SVD / 奇异谱分析单独记录

这部分作为静态实验的附加机制分析，不要和主结果表混在一起。

### 4.4.1 SVD 分析目的

它要回答：

\[
\textbf{为什么 Ours-static 比 naive SFT 更能保持 general OOD？}
\]

你的解释是：

\[
\textbf{Ours-static 更少扰动预训练模型的主奇异谱和 principal subspace。}
\]

------

### 4.4.2 SVD 分析只做一个 data size

建议固定：

\[
\text{Train size} = 10k
\]

这是主静态表的中心设置，最合适。

------

### 4.4.3 SVD 分析比较的方法

主图建议只放：

| Method | 作用 |
| ------ | ---- |
| Base | 原始谱结构 |
| Vanilla SFT | naive SFT 对照 |
| Random-k | 随机数据对照，可选 |
| Ours-final | 你的方法 |

PPL-balanced 可以作为 appendix 中的额外对照，而不是主图必须项。

可选 appendix：

| Method | 作用 |
| ------ | ---- |
| PPL-balanced | 证明 Ours 不仅优于 Vanilla SFT，也比 PPL heuristic 更少扰动主结构 |
| Ours-low / Ours-high | 进一步展示你的指标分数和谱扰动之间的关系 |

------

### 4.4.4 分析哪一层

建议默认选择：

| 项目 | 推荐 |
| ---- | ---- |
| Layer | 中间层，比如第 \(L/2\) 层 |
| Matrix | attention \(o\_proj\) 或 MLP \(down\_proj\) |
| Rank | top-64 或 top-128 |
| Seed | 主图用一个代表性 seed，appendix 可放 3 seeds 平均 |

为了避免 cherry-pick，最好补充一句：

> We use the middle transformer block by default and observe similar trends in neighboring layers.

------

### 4.4.5 SVD 单独记录表

| Model | Method | Train size | Layer | Matrix | Seed | Top-k | Spectral shift ↓ | Mean singular ratio | Mean principal angle ↓ | Mean cos angle ↑ | Update-principal overlap ↓ | Notes |
| ----- | ------ | ---------- | ----- | ------ | ---- | ----- | ---------------- | ------------------- | ---------------------- | ---------------- | -------------------------- | ----- |
| 3B | Vanilla SFT | 10k | mid | o_proj | 42 | 64 |  |  |  |  |  |  |
| 3B | Random-k | 10k | mid | o_proj | 42 | 64 |  |  |  |  |  |  |
| 3B | PPL-balanced | 10k | mid | o_proj | 42 | 64 |  |  |  |  |  | optional |
| 3B | Ours-final | 10k | mid | o_proj | 42 | 64 |  |  |  |  |  |  |

推荐记录：

\[
\text{Spectral Shift}
=
\frac{\lVert\sigma(W_+) - \sigma(W_0)\rVert_2}{\lVert\sigma(W_0)\rVert_2}
\]

以及 principal angle：

\[
\cos \theta_i
=
\sigma_i(U_{0,k}^{\top}U_{+,k})
\]

如果 Ours 的 spectral shift 更小、principal angle 更小、cos angle 更接近 1，就能支持：

> Ours-static adapts the model while better preserving pretrained principal geometry.

------

# 5. 动态实验 3B 需要对比的 baseline

动态实验是扩展验证。它的目标不是取代静态实验，而是说明：

\[
\textbf{静态筛选指标也可以接入 iterative synthetic data generation。}
\]

PPL 在动态实验中不是主 baseline。它可以作为 appendix 中的 optional baseline，但不建议放进动态主表，以免实验膨胀。

------

## 5.1 动态流程

统一流程：

\[
D_0
\rightarrow
\text{SFT}_0
\rightarrow
\text{Generate candidates } C_1
\rightarrow
\text{Score / Select}
\rightarrow
D_1
\rightarrow
\text{SFT}_1
\]

做 2 轮即可：

\[
t = 1, 2
\]

最多 3 轮，不建议更多，否则成本高，而且实验解释会复杂。

------

## 5.2 动态实验必须公平控制的变量

所有动态方法必须保持：

| 控制项 | 要求 |
| ------ | ---- |
| 初始数据 \(D_0\) | 相同 |
| 每轮生成候选数 \(|C_t|\) | 相同 |
| 每轮加入训练数据数 \(k_t\) | 相同 |
| 总训练 token budget | 相同 |
| iteration 数 | 相同 |
| generation prompt | 相同 |
| teacher model | 相同 |
| answer verifier | 相同，除非它本身就是 baseline 差异 |
| sampling temperature | 相同 |
| max output length | 相同 |

最重要的是：

**Ours 不能因为加入更多数据、训练更多 step 或生成更多候选而占便宜。**

------

## 5.3 动态 3B baseline

建议主表包含：

| Method | Rounds | 说明 |
| ------ | ------ | ---- |
| One-shot SFT | 0 | 只用 \(D_0\)，不迭代 |
| Iterative-SFT accept-all | 2 | 每轮生成的数据全部加入 |
| Iterative-SFT random | 2 | 每轮随机选同等数量数据 |
| Iterative-SFT verifier-filter | 2 | 用 correctness / verifier 筛选 |
| Ours-guided selection | 2 | 生成相同候选，但用你的指标筛选 |
| Ours-guided generation + selection | 2 | 如果你的方法也指导生成，则作为完整方法 |
| Ours-static + dynamic | 2 | 可选：初始 \(D_0\) 也来自 Ours-static |

如果你想控制实验量，动态主表保留前五个就够：

\[
\text{One-shot},\quad
\text{Accept-all},\quad
\text{Random},\quad
\text{Verifier-filter},\quad
\text{Ours-guided}
\]

------

## 5.4 动态实验中的 PPL optional baseline

如果算力允许，可以在 appendix 加一个 PPL dynamic baseline：

| Method | Rounds | 说明 |
| ------ | ------ | ---- |
| Iterative-SFT PPL-low | 2 | 每轮选择低 PPL generated candidates |
| Iterative-SFT PPL-balanced | 2 | 每轮按 PPL bucket 均衡选择 |

如果只能选一个，建议选：

\[
\textbf{Iterative-SFT PPL-balanced}
\]

不要把 PPL-low / PPL-high 放进动态主表，因为它们比较极端，也会稀释动态实验的主线。

------

# 6. 动态实验 3B 需要观察记录的数据

动态实验除了最终分数，还要记录每一轮数据质量的变化。

------

## 6.1 动态实验需要记录的 run-level 指标

| 类别 | 指标 |
| ---- | ---- |
| 训练规模 | initial data size、generated candidates per round、selected per round、total train tokens |
| 数据质量 | ours score mean、ours score std、PPL mean、PPL std、correctness rate、avg length |
| 目标性能 | Target-ID、MATH500、AIME24 |
| OOD 保持 | MMLU、TruthfulQA、WinoGrande、IFEval、HumanEval/MBPP |
| 综合指标 | Target Gain、OOD Drop、General OOD Avg |
| 迭代变化 | 每一轮后的 eval score |

即使动态主表不加入 PPL baseline，也建议记录每轮 selected data 的 PPL。这样可以证明 Ours-guided selection 不是简单地选 low-PPL 或 high-PPL 数据。

------

## 6.2 动态实验主记录表

| Run ID | Model | Method | Rounds | Initial size | Candidates / round | Selected / round | Total train tokens | Seed | Target-ID ↑ | MATH500 ↑ | AIME24 ↑ | General OOD Avg ↑ | Target Gain ↑ | OOD Drop ↓ |
| ------ | ----- | ------ | ------ | ------------ | ------------------ | ---------------- | ------------------ | ---- | ----------- | --------- | -------- | ----------------- | ------------- | ---------- |
| dyn-3b-001 | 3B | One-shot SFT | 0 |  | - | - |  | 42 |  |  |  |  |  |  |
| dyn-3b-002 | 3B | Accept-all | 2 |  |  | all |  | 42 |  |  |  |  |  |  |
| dyn-3b-003 | 3B | Random | 2 |  |  |  |  | 42 |  |  |  |  |  |  |
| dyn-3b-004 | 3B | Verifier-filter | 2 |  |  |  |  | 42 |  |  |  |  |  |  |
| dyn-3b-005 | 3B | Ours-guided | 2 |  |  |  |  | 42 |  |  |  |  |  |  |
| dyn-3b-006 | 3B | PPL-balanced | 2 |  |  |  |  | 42 |  |  |  |  |  | optional |

------

## 6.3 动态实验每轮数据记录表

这张表很重要，因为它能证明 Ours 不是最后偶然好，而是每轮选出的数据质量不同。

| Run ID | Method | Round | Generated candidates | Selected samples | Selected tokens | Ours score mean | Ours score std | PPL mean | PPL std | Correctness rate | Avg length | Target-ID after round ↑ | General OOD Avg after round ↑ | OOD Drop after round ↓ |
| ------ | ------ | ----- | -------------------- | ---------------- | --------------- | --------------- | -------------- | -------- | ------- | ---------------- | ---------- | ----------------------- | ----------------------------- | ---------------------- |
| dyn-3b-005 | Ours-guided | 0 | - | initial |  |  |  |  |  |  |  |  |  |  |
| dyn-3b-005 | Ours-guided | 1 |  |  |  |  |  |  |  |  |  |  |  |  |
| dyn-3b-005 | Ours-guided | 2 |  |  |  |  |  |  |  |  |  |  |  |  |

动态部分最好展示一张趋势图：

\[
x = \text{round}
\]

\[
y_1 = \text{Target-ID / MATH500}
\]

\[
y_2 = \text{General OOD Avg}
\]

理想结果是：

- naive iterative-SFT 的 Target-ID 上升，但 OOD 掉得多；
- Ours-guided 的 Target-ID 也上升，但 OOD drop 更小；
- Ours-guided 每轮选出的样本不是简单低 PPL 或高 PPL，而是有不同的 score/PPL profile。

------

# 7. 3–6 点对于 7B 模型的安排

7B 不需要完整复刻 3B。它的作用是：

\[
\textbf{证明 3B 上的结论不是小模型特例。}
\]

所以 7B 要做压缩版实验。

------

## 7.1 7B 静态实验 baseline

7B 静态实验建议只跑关键方法：

| Method | 是否建议 |
| ------ | -------- |
| Base | 必须 |
| Vanilla SFT | 必须 |
| Random-k | 必须 |
| Verifier-filter | 建议 |
| PPL-balanced | 建议必须 |
| CFS | 如果相关工作重要，则必须 |
| Ours-final | 必须 |
| Ours-low / Ours-high | 可选，通常不必 |
| PPL-low / PPL-high | 可选，通常不必 |

7B 静态主表可以是：

| Method | Train size | Target-ID ↑ | MATH500 ↑ | AIME24 ↑ | General OOD Avg ↑ | Target Gain ↑ | OOD Drop ↓ |
| ------ | ---------- | ----------- | --------- | -------- | ----------------- | ------------- | ---------- |
| Base | 0 |  |  |  |  | 0 | 0 |
| Vanilla SFT | 10k / 20k |  |  |  |  |  |  |
| Random-k | 10k / 20k |  |  |  |  |  |  |
| Verifier-filter | 10k / 20k |  |  |  |  |  |  |
| PPL-balanced | 10k / 20k |  |  |  |  |  |  |
| CFS | 10k / 20k |  |  |  |  |  |  |
| Ours-final | 10k / 20k |  |  |  |  |  |  |

推荐只做一个 train size：

\[
10k \quad \text{或} \quad 20k
\]

如果 3B 主表是 10k，7B 也用 10k，便于对比。如果你更想展示明显效果，可以用 20k。

------

## 7.2 7B 静态实验重复维度

| 维度 | 建议 |
| ---- | ---- |
| Train size | 10k 或 20k，选一个 |
| Seed | 至少 1 seed；关键方法可 3 seeds |
| Baseline | 只跑关键方法 |
| SVD | 可选，不必作为主实验 |

7B 不建议跑完整 5k / 10k / 20k 曲线。数据量曲线交给 3B。

7B 中 PPL 只建议保留：

\[
\textbf{PPL-balanced}
\]

不建议再跑 PPL-low / PPL-high，除非 3B 的 PPL 结果非常关键，需要进一步验证。

------

## 7.3 7B 是否需要 SVD 分析

不强制。

如果算力和时间允许，可以做一个 appendix 级别的 7B SVD：

| Method | Train size | Layer | Matrix | Spectral shift ↓ | Principal angle ↓ |
| ------ | ---------- | ----- | ------ | ---------------- | ----------------- |
| Vanilla SFT | 10k | mid | o_proj |  |  |
| PPL-balanced | 10k | mid | o_proj |  | optional |
| Ours-final | 10k | mid | o_proj |  |  |

但主论文里，SVD 分析放 3B 就够了。因为 SVD 的作用是解释机制，不是 scaling 证明。

------

## 7.4 7B 动态实验 baseline

7B 动态实验只跑最关键对比：

| Method | 是否必须 |
| ------ | -------- |
| One-shot SFT | 必须 |
| Iterative-SFT random 或 accept-all | 必须 |
| Verifier-filter | 建议 |
| Ours-guided iterative-SFT | 必须 |
| Iterative-SFT PPL-balanced | 可选 appendix |

如果 CFS 是重要相关工作，也可以加：

| Method | 说明 |
| ------ | ---- |
| CFS iterative | 只有当它能自然适配 dynamic setting 时才加 |

7B 动态表建议是：

| Method | Rounds | Initial size | Selected / round | Total train tokens | Target-ID ↑ | MATH500 ↑ | AIME24 ↑ | General OOD Avg ↑ | Target Gain ↑ | OOD Drop ↓ |
| ------ | ------ | ------------ | ---------------- | ------------------ | ----------- | --------- | -------- | ----------------- | ------------- | ---------- |
| One-shot SFT | 0 |  | - |  |  |  |  |  |  |  |
| Iterative-SFT random | 2 |  |  |  |  |  |  |  |  |  |
| Verifier-filter | 2 |  |  |  |  |  |  |  |  |  |
| Ours-guided | 2 |  |  |  |  |  |  |  |  |  |
| PPL-balanced | 2 |  |  |  |  |  |  |  |  | optional |

7B 动态实验不需要做 3 seeds，除非主结果非常接近。如果 Ours 和 baseline 差距明显，1 seed 可以接受；如果差距小，关键对比补 3 seeds。

------

# 最终推荐执行顺序

## 第一阶段：3B 静态主实验

优先跑：

1. Base
2. Random-k 10k
3. Vanilla SFT 10k
4. Verifier-filter 10k
5. PPL-balanced 10k
6. Ours-low 10k
7. Ours-high 10k
8. Ours-final 10k

然后补：

1. PPL-low 10k
2. PPL-high 10k
3. 5k / 20k data-size curve，其中包含 PPL-balanced
4. 3 seeds
5. CFS baseline

------

## 第二阶段：3B SVD 分析

固定：

\[
\text{3B},\quad 10k,\quad \text{middle layer},\quad o\_proj \text{ or } down\_proj
\]

主图比较：

\[
\text{Vanilla SFT},\quad \text{Random-k},\quad \text{Ours-final}
\]

可选 appendix 加：

\[
\text{PPL-balanced}
\]

输出：

1. singular value ratio curve
2. spectral shift
3. principal angle curve

------

## 第三阶段：3B 动态实验

先跑：

1. One-shot SFT
2. Iterative random
3. Verifier-filter
4. Ours-guided

如果时间允许，再加：

1. Accept-all
2. Ours-static + dynamic
3. Iterative-SFT PPL-balanced，作为 appendix baseline

------

## 第四阶段：7B 缩小版验证

7B 只验证关键结论。

### 7B 静态

\[
\text{Base},\quad
\text{Vanilla},\quad
\text{Random-k},\quad
\text{Verifier-filter},\quad
\text{PPL-balanced},\quad
\text{Ours-final}
\]

如果 CFS 是重要相关工作，则加：

\[
\text{CFS}
\]

### 7B 动态

\[
\text{One-shot},\quad
\text{Iterative random},\quad
\text{Verifier-filter},\quad
\text{Ours-guided}
\]

PPL-balanced dynamic 只作为 appendix optional。

------

# 论文中最终应该形成的结果结构

| Table / Figure | 内容 | 是否包含 PPL | 作用 |
| -------------- | ---- | ------------ | ---- |
| Table 1 | 数据、模型、训练设置总览 | 说明 PPL 计算 | 说明实验配置 |
| Table 2 | 3B 静态筛选主结果 | 包含 PPL-balanced；appendix 可含 PPL-low/high | 核心结果 |
| Table 3 | 3B 静态数据量曲线 | 包含 PPL-balanced | 证明趋势稳定 |
| Figure 1 | SVD / spectral / principal-angle 分析 | 可选包含 PPL-balanced | 解释 Ours 为什么 OOD drop 小 |
| Table 4 | 3B 动态 iterative-SFT 结果 | 主表不含 PPL；appendix 可含 PPL-balanced | 证明可接入合成数据闭环 |
| Table 5 | 7B 静态 + 动态关键验证 | 7B 静态含 PPL-balanced | 证明可扩展性 |
| Table 6 | Ablation | 可包含 PPL-low / PPL-high | 证明指标、生成、筛选分别有效 |

这样组织后，你的论文主线会非常清楚：

> 静态指标筛选是核心贡献；
> 它能提升 synthetic math SFT 的 target/OOD trade-off；
> 它优于 random、correctness filter 和 PPL heuristic；
> SVD 分析说明它更少扰动预训练主结构；
> 动态实验说明它还能用于 iterative synthetic data generation；
> 7B 实验证明该趋势可扩展。



ID & OOD测试集

| 记录名                    | task/group             | metric                         |
| ------------------------- | ---------------------- | ------------------------------ |
| GSM8K                     | gsm8k                  | exact_match,flexible-extract   |
| MATH500                   | `hendrycks_math500`    | `exact_match,none`             |
| AIME24                    | `aime24`               | `exact_match,none`             |
| Minerva-MATH500 EM        | `minerva_math500`      | `exact_match,none`             |
| Minerva-MATH500 Verify    | `minerva_math500`      | `math_verify,none`             |
| MMLU                      | `mmlu`                 | `acc,none`                     |
| MMLU-STEM                 | `mmlu_stem`            | `acc,none`                     |
| MMLU-Humanities           | `mmlu_humanities`      | `acc,none`                     |
| MMLU-Social Sciences      | `mmlu_social_sciences` | `acc,none`                     |
| MMLU-Other                | `mmlu_other`           | `acc,none`                     |
| TruthfulQA-MC1            | `truthfulqa_mc1`       | `acc,none`                     |
| TruthfulQA-MC2            | `truthfulqa_mc2`       | `acc,none`                     |
| WinoGrande                | `winogrande`           | `acc,none`                     |
| IFEval prompt strict      | `ifeval`               | `prompt_level_strict_acc,none` |
| IFEval instruction strict | `ifeval`               | `inst_level_strict_acc,none`   |
| HumanEval                 | `humaneval`            | `pass@1,create_test`           |
