# QA: Cycle 06 — Experiment Design Phase

```yaml
phase: experiment_design
cycle: cycle_06_sft_feasibility_and_degradation
status: concluded_2026-06-23
created: 2026-06-23
write_scope: exp_only
internalize_to:
  - exp/current_experiment_design.md
  - exp/current_acceptance_criteria.md
  - exp/current_code_handoff.md
  - exp/experimental_methodology/  # 仅在 methodology 实际变化时
```

本 QA 文件是 Cycle 06 experiment design 阶段开始前的审查清单。
按工作区规则，本阶段只写 `exp/`（含 `code_handoffs/cycle_06/`），读取其他目录但不修改。
用户回答并讨论确认后，再写正式设计文件。

---

## 背景摘要（已确认，来自上游阶段）

**Next Cycle Seed 确认：**
- Cycle 06 是 SFT-only feasibility and degradation cycle，不训练 OPD arm。
- 候选数据规模：n=256, n=512, n=1024（默认）；n=128 可选（onset 分辨率）；n=2048 可选（需资源理由）。
- 始终包含未训练 base 作为参照行。

**Theory 确认（QA_cycle06_theory.md，讨论结论 2026-06-23）：**
- K1-base-aware 主框架保留，SFT-feasibility 是 K1 的前置条件。
- Gate A06 / B06 的解释边界已写入 theory，具体数字 threshold 由 Experiment Design 决定。
- OOD-lite = capability-preservation check，不是判别轴。
- Geometry = diagnostic-only。

**现有协议（稳定层）：**
- 当前学生模型：`/root/autodl-tmp/model/Qwen/Qwen3-1.7B`（instruct / post-trained）
- 历史 SFT format：chat messages + 非思考模式 target（无 `<think>` tag）
- eval：`apply_chat_template=True + enable_thinking=False + max_gen_toks=3072`
- Cycle 05 已验证的任务协议：NuminaMath-test、GSM8K、MATH500、MMLU（no chat）、TruthfulQA（chat）、WinoGrande（no chat）、ARC-challenge（no chat）

---

## 核心问题

### Q1：模型身份 — 保持还是切换？

这是本轮所有其他设计决策的前提。

**选项：**

- **a. 保持** `Qwen3-1.7B`（本地 instruct checkpoint）作为 base，直接从它开始 SFT，不设 theta0。
  - 已有完整协议，无需重新验证格式。
  - 风险：Cycle 04 已知此模型在 NuminaMath ID 上几乎无训练增益（base=0.5516，所有 Cycle 04 SFT arm 均未显著超越），Gate A06 通过概率有限。
  - 但 Cycle 06 从 base 直接 SFT（不经 theta0 热启动）是新的条件，headroom 情况未知。

- **b. 切换至** `Qwen3-1.7B-Base`（官方预训练 checkpoint，无 instruction tuning）。
  - 初始能力更低，预期 SFT headroom 更大，Gate A06 更容易通过。
  - 代价：需要重新验证全套 eval 协议（base 模型无 chat template 行为，MMLU/WinoGrande 协议要重新审）；当前 FINDING 文件里的数字与它不能直接比较。
  - 官方 Qwen3-1.7B-Base 报告分数：MMLU 62.63、GSM8K 75.44、MATH 43.50（官方协议）。

- **c. 两者并行**（instruct base + official base 各跑一个 SFT arm，共享同一 data scale）。
  - 可以同时回答"instruct base 是否有 headroom"和"pretraining base 是否有更大 headroom"。
  - 代价：至少翻倍的训练和 eval 成本。

- **d. 换任务**：保持 instruct 模型，但换一个 SFT 任务或数据集（例如 GSM8K direct SFT，或换更难/更有学习空间的数据集）。

**你的回答：**
c，我认为两者都要，这样能比较两者，后续扩大实验也能知道使用哪个会好一点。因为训练实际上大量开销在OPD上，现在多测试一下对总体而言是有利的。

---

### Q2：训练格式 — 是否改变 target format？

**背景：** 历史 SFT format 是 chat messages + 无 `<think>` 标签的 target（teacher step-by-step + `\boxed{}`）。这决定了 eval 必须用 `enable_thinking=False`。

如果 Q1 选 a（继续 instruct 模型）：
- **a. 保持现有 format**（chat + 非思考 target + `enable_thinking=False` eval）。无协议变更风险。
- **b. 切换到 thinking format**：target 包含 `<think>...</think>` reasoning，eval 用 `enable_thinking=True`。预期 GSM8K/MATH500 性能更高，但引入新的 train/eval 匹配验证需求。

如果 Q1 选 b（官方 Base 模型）：
- 需要决定是引入 chat SFT format，还是用 plain prompt/completion SFT。
- 不同选择对应不同的 eval 协议，且与历史数字不可直接比较。

**（Q2 答案取决于 Q1 答案，如果 Q1=a 则主要回答上方两个选项）**

**你的回答：**
我的回答中说需要使用两个模型。对于两个模型在eval阶段，我认为instruct暂时保持现在的format；base模型则使用基础模式不使用chat。但是我希望你再进一步之前再阅读一下有没有官方文件或者社区讨论使用什么格式，我需要你对于所有要测试benchmark都去联网搜索，支不支持chat格式、支不支持think。现在我们的测试还没有完全定下来。

---

### Q3：主 ID 轴是否继续用 NuminaMath-test？

**背景：** Cycle 04/05 的主 ID 轴是 NuminaMath-test（N=1024 held-out）。但在 instruct 模型 + ≤1024 NuminaMath SFT 数据的设置下，所有 Cycle 04 arm 在 NuminaMath ID 上均未超过 base（0.5516）。

**选项：**

- **a. 保持 NuminaMath-test 为主 ID 轴**。如果 Gate A06 最终不过，就根据失败信号改变模型/任务，而不是先换轴。保留比较连续性。
- **b. 换任务 / 数据源**：SFT 数据和 ID 评测都换成 GSM8K（例如 GSM8K train → 直接 SFT → GSM8K test 作为 ID 轴）。可能 headroom 更大。但 GSM8K train 有 7473 个样本，选小子集的话与之前的 NuminaMath 设置结构相同。
- **c. NuminaMath 作为主 ID 轴，同时在 design 中添加 per-source breakdown**（按 NuminaMath 子数据集细分，帮助诊断潜在的分布 shift）。

**你的回答：**
你之前说只使用1024的训练数据，我认为这是不是很不足够？请你联网搜索一下大致要多少，我现在不认为1024是足够的。

---

### Q4：SFT 数据规模 — 哪些 n 值要跑？

**背景：** next_cycle_intent 推荐默认规模 n ∈ {256, 512, 1024}；n=128 可选（onset 分辨率）；n=2048 可选（需资源理由）。

历史参照：Cycle 04 的 sft_n512（from theta0）在 NuminaMath ID 上已低于 base（0.5179），sft_n1024 严重退化（0.3890）。Cycle 06 从裸 base 开始，退化起始点可能不同。

**选项：**
- **a. 只跑 n ∈ {256, 512, 1024}**（intent 默认，3 个 arm + base 参照）。
- **b. 加 n=128**（4 个 arm），帮助观察退化起始点是否已在 n=128 出现。
- **c. 加 n=2048**（需要额外训练资源）。
- **d. 加 n=128 + n=2048**（完整 5 点曲线，资源最高）。

注意：每增加一个 arm 都需要对应的 eval（NuminaMath、GSM8K、MATH500、OOD-lite），资源线性增长。

**你的回答：**
实际上我看有些论文说SFT会出现drop-recover的现象，我认为可以增大一些，但是间隔可以大一些，512、1024、2048、4096.这样我认为是可以进行一下探索的。

---

### Q5：Geometry diagnostics — 是否运行？

**背景：**
- Theory 确认：Cycle 06 geometry = diagnostic-only，不作为主 gate，不能生成 OPD mechanism claim。
- Cycle 04 的问题：UV/sketch 未保存，导致无法做 principal-angle / subspace-overlap claim。
- 运行一次 GetSlice（8 个模型 × 3 layers × 7 modules）在已有代码基础上成本较低，但需要配置探针数据、决定是否保存 UV。

**选项：**

- **a. 运行 geometry diagnostics**，明确保存 UV/sketch artifacts（为将来 subspace-overlap 分析留后路）。需要在 design 里定义：哪些层、多少 probe samples、"PE metric" 是什么、以什么为 drift reference（base？还是 base_row？）。
- **b. 运行 minimal geometry**（只计算 effective_rank 和 spectral_drift，不保存 UV/sketch，和 Cycle 04 相同的保存策略）。
- **c. 跳过 geometry**。本轮 SFT-only，diagnostic 价值有限，先把 capability 结果做干净。

**你的回答：**
b但是我记得还有别的数据啊，同时最基础的奇异值数据不要删除。而且我希望在这里顺便比较一下我的观察方法和《the path not taken》中的差距，所以我需要测试一下主权重命中率，具体方法请你参照
/root/LLM-output-density/AnalyseMat/principalEvidence.py和联网搜索一下原文https://arxiv.org/abs/2511.08567

---

### Q6：Gate A06 的具体 threshold

**背景：** Theory 只定义了 Gate A06 的解释意义（SFT-from-base 是否产生可测量 gain），具体数字 threshold 由 Experiment Design 决定，必须预先注册，不能结果出来后临时换。

**候选 threshold 方案（主 ID 轴 NuminaMath-test）：**

- **a. >2 SE 优于 base**（约 >0.0252 超过 base 0.5516 → 即超过 0.5768）。比较严格，确保 gain 是真实的。
- **b. >1 SE 优于 base**（约 >0.0126，即超过 0.5642）。相对宽松。
- **c. >任意正值且至少有一个 SFT arm 达到**（最宽松，只要任一 arm 不低于 base 就算通过）。
- **d. 自定义**（例如相对提升 >3%，或某个任务特定的阈值）。

如果 Q3=b（换 GSM8K 为主轴），threshold 也需要对应调整。

**你的回答：**
b稍微保守一点

---

## 讨论结论（2026-06-23 确认）

用户已回答 Q1–Q6，经研究补充后，最终结论如下。已内化到正式设计文件。

| 问题 | 确认答案 | 备注 |
|---|---|---|
| Q1 模型身份 | **双线并行：Qwen3-1.7B（instruct）+ Qwen3-1.7B-Base** | c 选项；10 模型总量 |
| Q2 训练/eval 格式 | **instruct 系列：保持 Cycle 05 协议；base 系列：plain 格式（no chat），eval 用 4-shot CoT no-chat** | 见下方详述 |
| Q3 主 ID 轴 | **继续用 NuminaMath-test**，数据量问题由 Q4 解决 | 保持比较连续性 |
| Q4 数据规模 | **n ∈ {512, 1024, 2048, 4096}**，4 点曲线 | 用户提出 drop-recover 探索 |
| Q5 Geometry | **minimal（effective_rank + spectral_drift）+ 保留原始奇异值 + 加跑 principalEvidence.py** | 对照 The Path Not Taken |
| Q6 Gate A06 | **>1 SE 高于各自 base 参照行** | b 选项，稍保守 |

### Q2 详述（协议研究结论）

**instruct 系列（Qwen3-1.7B）：**
- 训练：chat messages 格式 + 非思考 target（Cycle 04/05 沿用）
- 评测 math（NuminaMath-test / GSM8K / MATH500）：`apply_chat_template=True` + `enable_thinking=False` + 3072 tokens（Cycle 05 已验证）
- 评测 OOD-lite：MMLU/WinoGrande/ARC no chat loglikelihood；TruthfulQA chat loglikelihood（Cycle 05 已验证）

**base 系列（Qwen3-1.7B-Base）：**
- 训练：plain 格式 `"Problem: {problem}\n\nSolution: {solution}"`，no chat template，no `<think>` tag
- 评测 math：**4-shot CoT，no chat template，3072 tokens**（与 Qwen3 官方 base 评测协议对齐：arXiv 2505.09388）
- 评测 OOD-lite：MMLU 5-shot no chat loglikelihood；WinoGrande no chat loglikelihood；ARC 25-shot no chat loglikelihood；TruthfulQA **no chat** loglikelihood（base 模式，与 instruct 不同）
- 不传 `enable_thinking` 参数（base 模型无此功能）

所有 base 系列 eval 均在 "base model mode"（no chat template 的 loglikelihood / plain generation）。

### Q5 详述（The Path Not Taken 对比）

arXiv 2511.08567 核心发现：RLVR 更新落在权重矩阵的"off-principal"方向（低奇异值方向，相对随机；spectral drift 小）；SFT 更新落在"on-principal"方向（高奇异值方向，集中；hit rate > random）。

`principalEvidence.py` 的 `OverlapLift` 是同一概念的 element-level 近似：
- **Principal mask** = 重构矩阵 |U0k × S0k × Vh0k|_ij 的 top_ratio 分位元素
- **Update mask** = bfloat16 精度下值发生变化的元素
- `OverlapLift` = (observed overlap) / (random expected overlap)

Cycle 06 预测：SFT 模型的 `OverlapLift >> 1`，且随 n 增大而增大。这与 The Path Not Taken 的 SFT-on-principal 结论一致，为之后加入 OPD 对比预留基准。

需要权重导出脚本（model checkpoint → per-layer .npy），格式 `model_layers_{layer}_{module}_weight.npy`；由代码对话实现。
