# 来源矩阵 — 相关工作综述

```yaml
view_type: synthesis_matrix_ch
status: active
maintenance_rule: add_new_entries_per_cycle; remove_only_if_confirmed_unused_and_note_reason
last_updated: 2026-06-27
cycle06_position: sft_feasibility_and_degradation_first
cycle07_position: base_model_sft_feasibility_generalization_conditional + theory_deepening_sft_trajectory
cycle08_position: h_opd_vs_sft_comparison + opd_recipe_tooling + matched_comparison_validity
sync_status: 与 source_matrix.md 同步至 Cycle 08（2026-06-27）
```

本文件是**综述文件**：分类介绍各引用论文之间的关系，说明本项目为什么需要这一组论文。
每篇论文的具体信息（标题、URL、内容概括）见 `related_work-ch.md`。
每轮新增/删除记录见 `incremental_related_work-ch.md`。

---

## 文献定位：为什么是这一组论文

Cycle 06 的文献定位发生了局部转向：**当前阶段不继续做 OPD-vs-SFT 对比，而是先建立干净的 SFT baseline、SFT 数据规模曲线和 Qwen3 训练/评测协议。**

这不是推翻 Cycle 03-05 的 OPD 主线，而是把研究 ladder 往前补一格：如果 plain SFT-from-base 不能在选定任务上产生可解释增益，或者训练/评测格式本身不稳定，那么后续 OPD comparison 没有可靠锚点。

当前 source list 因此分成五条脉络：

1. **OPD 机制文献**（Agarwal et al. GKD；verl OPD 文档；Rethinking OPD；OPD survey）：保留为背景。它解释为什么未来仍值得比较 OPD 和 SFT，但不是 Cycle 06 的主动实验主轴。

2. **SFT 退化 / forgetting / data-scale 文献**（Lobo et al.; Luo et al.; Kalajdzievski; Dong et al.; Ding & Wang）：Cycle 06 的核心来源。它们共同建立：SFT 不是保证提升的 baseline，数据量、更新步数、数据组成和目标能力之间可能存在 tradeoff。

3. **Qwen3 模型身份与协议来源**（Qwen3 Technical Report；Qwen3 官方博客；Qwen3-1.7B / Qwen3-1.7B-Base model cards；社区 Qwen3 GSM8K SFT model card；Unsloth Qwen3 guide）：作为工程方法学来源，帮助 Experiment Design 在训练前确认模型身份、chat/think 格式、baseline sanity check 和 scorer 风险。

4. **谱 / 几何机制文献**（LoRA vs Full Fine-tuning；OPRD）：本轮只作 diagnostic 背景，不支持机制 claim。Geometry 若运行，作用是帮助解释 SFT learning/degradation curve，而不是证明 OPD mechanism。

5. **Base/headroom/control-anchor 方法论文述（留位）**：本轮不搜索。后续一旦重新进入 OPD-vs-SFT aligned comparison，Related Work 必须提醒是否补做 baseline construction / control anchor / matched comparison validity 文献搜索。

---

## Cycle 07 文献定位更新（2026-06-24）

Cycle 07 把主实验模型从 Qwen3-1.7B instruct 转到 **Qwen3-4B-Base**（pretraining checkpoint），OPD arm 推迟到 Cycle 08。文献定位在 Cycle 06"SFT feasibility/degradation"基础上做两处推进：

**(A) 新增核心来源 Rethink SFT（Ren et al. 2026），把"SFT 是否泛化"重定为条件问题。** 该文用 base 模型 + long-CoT 数学 SFT 系统证明 cross-domain generalization 由 **optimization dynamics × training data × model capability** 三者联合决定。它一次性回答了本轮四条决策线：(1) 模型——1.7B-Base 实测训不出泛化（能力地板），4B-Base 用 long-CoT 全面正向；(2) 数据——NuminaMath-1.5 短解被点名为"广泛损害泛化"的低质量数据，应换 verified long-CoT；(3) 格式——long-CoT/procedural pattern 是迁移来源，支持本项目从 no-think 翻转到 think-format；(4) 评测——ID=MATH500+AIME24、OOD=LCB v2/GPQA-D/MMLU-Pro，并提供可核对 base 分数。详见 `source_notes/rethink_sft_experiment_tables.md`（含 App D 全表抽取与 4B 专题）。

**(B) 理论深化：把 SFT 从单点拆成轨迹（SFT_dip / SFT_recover），把 OOD-drop 重定义为 checkpoint 属性。** Rethink SFT 的 dip-and-recovery（Sec 3）+ response-length-as-diagnostic（Sec 3.2）+ shallow-imitation vs internalized-pattern（C.3/C.6）支撑升级后的研究框架：原"OPD vs SFT 的 OOD-drop/继续学习/几何指标"二元对照，升级为 **OPD vs SFT_dip vs SFT_recover** 三方对照，核心假设是"**OPD ≈ 不经过 dip 暂态、直达 recover 状态**"。

> 由此产生一个**新文献缺口并已在本轮补齐**：三方框架的 OPD 腿需要"on-policy/online 训练忘得少"文献簇——**Chen et al. 2025（Retaining by doing）、Shenfeld et al. 2026（RL's razor）、Tajwar et al. 2024**。本轮登记为 OPD 腿备料（Cycle 08 引用），不在 Cycle 07 作 OPD claim。
>
> **写作边界**：正式理论改写归 Theory 对话写 `theory/`；Related Work 只提供文献脚手架与"三方框架的文献支撑"判断。
>
> **实验含义（提醒 Experiment Design）**：SFT_dip/recover 与几何指标的关系**本轮（Cycle 07）即可测**（无需 OPD），但要求沿 SFT 轨迹**多 checkpoint 采样几何指标 + response length**，否则 Cycle 08 三方对照缺基线。OPD 腿留到 Cycle 08。

**control-anchor 留位（Cycle 06 约定）仍然有效：** 一旦 Cycle 08 进入 OPD-vs-SFT 对齐比较，Related Work 必须提醒是否补做 baseline construction / control anchor / matched comparison validity 搜索——届时与上面的 on-policy-forgetting 文献簇一并激活。

---

## Cycle 08 文献定位更新（2026-06-27）

Cycle 08 = **H-OPD vs SFT 对比**（4B-Base + Math-CoT-20k + think-LoRA，单 OPD arm 单 lambda，Cycle 07 SFT 轨迹作 matched control）。intent 未指定 **teacher** 与 **cold-start**——本轮补三类来源（研究笔记 `source_notes/cycle08_opd_teacher_coldstart_tooling.md`）：

**(A) OPD teacher 选择 + cold-start 配方（QA Q1/Q2）。** capacity-gap 定律（2311.07052）：最优 teacher 随 student 线性、非越大越好 → 4B-Base 选 **Qwen3-8B-Instruct 同家族**（用户已定），32B 仅离线数据。cold-start：Rethinking OPD 实测——纯 OPD-from-base 可能但条件差，判据是 **overlap ratio**（成功 72%→91%）；步数 scale-dependent（~3000 步 plateau 不照搬）。⚠️ base(无think)→instruct(有think)+think-format，初始 overlap 很可能低，**纯 from-base 风险高，需 overlap probe 定夺最小 cold-start**（→ Exp/Code）。

**(B) OPD 训练工具（QA Q3）。** TRL GKDTrainer 能跑但**非高速**（rollout 走 HF generate，3–8×，GKD 无一等 vLLM）；**verl async OPD 是高速专用**（vLLM rollout + async teacher + top-k forward KL；约束：推理服务器只返 top-k logprob → 必须 top-k KL）。框架决策归 Code/Exp Design。

**(C) control-anchor / matched-comparison validity（QA Q4=a，留位激活）。** Cycle 08 首次 OPD-vs-SFT 对齐比较，按 Cycle 06 约定**本轮激活**。新增 The Efficiency Misnomer（2110.12894）等。**核心方法学输入（用户 Q4 追问"为什么单点而非整条趋势"）**：单点匹配结果变量(ID gain)= Cycle 03/04"matched-GSM8K-gain 锚点退化"那类错误；与 OOD-drop 的**轨迹属性**自相矛盾。正解：在**输入轴**（budget/grid，已采用）匹配 → 比**整条 dip-recover 轨迹**为 primary；单点作 landmark；iso-ID 作二级 cross-section。正式比较设计归 Theory/Exp Design。

**(D) 能力退化 eval 广度拓展（用户追加）。** Rethink SFT 套件偏难推理 + 偏"获得"轴；本项目训练在数学 long-CoT 上，需补**非数学、base 已强**的退化轴。拍板候选（详见 `source_notes/cycle08_capability_degradation_benchmarks.md`，含 Qwen3-4B/8B-Base 官方分数）：Tier 1 cheap-MC 全 checkpoint track（MMLU/HellaSwag/WinoGrande/ARC-C/PIQA/TruthfulQA-MC1）+ Tier 2 生成式抽测（BBH/GSM8K/EvalPlus）。⚠️ 协议依赖不可混用（官方仅作 sanity-check，本项目统一协议自测）；退化轴须 base 强；commonsense 须先验 4B 判别力。**不改 H-OPD primary gate（B08/C08 仍以 MATH500+MMLU-Pro 为主）**；正式选轴/预注册归 Experiment Design。

---

## 分类综述

### 1. OPD 作为未来比较背景，而非 Cycle 06 主轴

OPD 的核心论点仍然成立：学生应该在它实际访问的 states 上学习。Agarwal et al. 的 GKD 在 student 自生成序列上用 teacher feedback 训练；verl OPD 文档强调 OPD 从 student policy 采样 states，相比 SFT/标准 KD 减少 exposure bias；Rethinking OPD 进一步把 OPD 成功条件解释为 student distribution 与 teacher-supported high-probability tokens 的重叠增长。

Cycle 06 暂时不训练新 OPD arms。OPD 文献在本轮的作用是保留后续比较的理论背景，而不是支持任何“OPD 已经更稳定”或“OPD 已经防止 forgetting”的结论。

### 2. SFT 可能带来能力 tradeoff 和 catastrophic forgetting

Lobo et al. 表明 fine-tuning 会影响 CoT reasoning 与 faithfulness；Luo et al. 在 1B-7B LLM continual instruction tuning 中观察到 catastrophic forgetting；Kalajdzievski 将 forgetting 与 fine-tuning performance、update steps 和 tuned parameter scale 联系起来；Dong et al. 直接研究 SFT data amount、data composition、model size 和 strategy 对数学、代码、通用 alignment 能力的不同影响；Ding & Wang 则把 domain-specific SFT 后的 general capability degradation 作为需要 mitigation 的问题。

实验含义：Cycle 06 必须把 SFT 当成需要测量的 live baseline。关键不是“更多 SFT 数据是否更好”，而是：

- SFT-from-base 是否先能提升目标 ID/downstream 轴；
- 数据规模增加时是否出现 ID/OOD degradation；
- 如果目标轴提升而相邻数学轴下降，应把它视为待解释的 tradeoff，而不是简单成功。

### 3. Qwen3 模型身份与格式协议是 related-work 的一部分

Qwen3-1.7B 与 Qwen3-1.7B-Base 不是同一个实验对象。Qwen3-1.7B 是 post-trained 模型，官方文档和 model card 明确有 chat template、`enable_thinking`、`</think>` parsing 等行为；Qwen3-1.7B-Base 是 pretraining-only causal LM，不应自动继承 post-trained 模型的 chat/think 假设。

Cycle 05 已经证明：格式协议错误足以制造 GSM8K、MATH500、MMLU 的假结论。Cycle 06 因此需要把模型身份、训练 target format、评测 chat/think/token settings 和 scorer/extractor audit 写成设计前置条件。详见 `source_notes/cycle06_qwen3_sft_protocol_brief-ch.md`，其中已经补入当前项目的训练格式和 GSM8K / MATH500 / MMLU / TruthfulQA / WinoGrande / ARC 评测格式表。

### 4. OOD-lite 和 Geometry 的本轮地位

Cycle 05 发现全部 8 个模型在 MMLU、TruthfulQA、WinoGrande、ARC-challenge 上聚集在 <1.5% 以内。因此，OOD-lite 在当前设置中是 capability-preservation check，不是 OPD/SFT 判别轴。

Geometry 同样降级为 diagnostic-only。LoRA-vs-full-finetuning 和 OPRD 仍然说明 spectral / representation movement 是有意义的机制背景，但 Cycle 06 的 geometry 只能帮助解释 SFT learning/degradation curve，不能生成 OPD mechanism claim。

### 5. Future Slot: Control Anchor / Matched Comparison Validity

Cycle 04-05 的一个重要教训是 base headroom 与 matched-control anchor 会改变稳定性结论。但用户已确认：本轮不搜索该方向，以免影响 SFT baseline 工作。

后续触发条件：

- Cycle 06 证明 SFT-from-base 有可解释 headroom；
- 或未来重新进入 OPD-vs-SFT aligned comparison；
- 或 paper positioning 明确把 matched-control warning 作为贡献之一。

触发后应补做 baseline construction、control selection、matched comparison validity 方向的 related-work search。

---

## 综述矩阵

### 核心论文（论文论点）

| 来源 | 核心主张 | 在项目中的角色 | 使用级别 |
|---|---|---|---|
| Agarwal et al., GKD / OPD, ICLR 2024 | Student 自生成序列 + teacher feedback 解决标准 KD 固定输出的局限 | 说明未来 OPD-like training 为什么可能不同于 SFT | 后续 OPD 背景 / 核心 |
| verl OPD 文档 | OPD 从 student policy 采样 states，减少 exposure bias，提供 dense token-level 监督 | 定义本项目 OPD 与 SFT/KD 的实际区别 | 背景 / 实现 |
| Rethinking OPD, 2026 | OPD 成功依赖 student 分布与 teacher 高概率 token 的重叠增长；cold start 可帮助 | 为未来 OPD cold-start 和 alignment 设计提供动机，不是 Cycle 06 SFT-only 训练输入 | 后续 OPD 框架 |
| OPD survey, 2026 | OPD 方法可按反馈信号、teacher 访问级别、loss 粒度分类 | 将本实验定位为白盒、token-level、output-space OPD-like | 框架定位 |
| Lobo et al., CoT fine-tuning, NAACL 2025 | Fine-tuning 可能改变 reasoning behavior 并降低 CoT faithfulness | 支持在目标增益之外测量 reasoning degradation | Cycle 06 核心 |
| Luo et al., catastrophic forgetting, 2023/2025 | Continual instruction tuning 可在 1B-7B LLM 上产生 catastrophic forgetting | 支持将 1.7B SFT 视为有 forgetting 风险的设置 | Cycle 06 核心 |
| Kalajdzievski, scaling laws for forgetting, 2024 | Forgetting 与 fine-tuning performance、update steps、tuned parameter scale 有关；LoRA 不能消除 | 支持预注册 SFT 数据量 / update exposure | Cycle 06 核心 |
| Dong et al., SFT data composition, ACL 2024 | 不同能力的 SFT scaling pattern 不同；数据量/组成可造成 conflict；sequential training 有 forgetting 风险 | 与 Cycle 06 的 SFT scale curve 和 ID/OOD 分轴最接近 | Cycle 06 核心 |
| Ding & Wang, improved SFT mitigation, 2025 | Domain-specific SFT 常导致 general capability degradation；混合/重构通用数据可缓解 forgetting | 若 Cycle 06 发现 SFT 退化，可作为未来 mitigation baseline | Mitigation / 未来 |
| LoRA vs Full Fine-tuning, 2024 | 谱结构可以揭示 fine-tuning 行为和遗忘机制 | 支持 geometry 作为 diagnostic，而不是当前机制证明 | Diagnostic |
| OPRD, 2026 | 表示层 on-policy distillation 是新兴 OPD 方向 | 未来方向；不是当前 SFT-only cycle 的一部分 | 未来 |
| **Rethink SFT (Ren et al.), 2026** | Reasoning-SFT cross-domain generalization 是**条件性**的：由 optimization（dip-and-recovery）、data quality/structure（verified long-CoT ≫ NuminaMath 短解）、model capability（1.7B 失败、4B+ 泛化）联合决定 | **Cycle 07 主来源**：决定 model(4B-base)/data(long-CoT)/format(think)/eval(MATH500+AIME24+LCB/GPQA/MMLU-Pro) 四线；并提供 SFT_dip/recover 轨迹的实证骨架 | **Cycle 07 核心** |
| Chen et al., Retaining by doing, 2025 | On-policy 数据相比 off-policy SFT 缓解 catastrophic forgetting | Cycle 08 三方框架 OPD 腿备料：支撑"OPD ≈ recover-without-dip" | 前向（Cycle 08 OPD 腿） |
| Shenfeld et al., RL's razor, ICLR 2026 | Online/on-policy RL 忘得少；mode-seeking 更新贴近 base 分布 | Cycle 08 OPD 腿备料：预测 OPD 的 drift-from-base 平于 SFT_dip 暂态 | 前向（Cycle 08 OPD 腿） |
| Tajwar et al., on-policy preference FT, ICML 2024 | On-policy 采样本身（非仅数据质量）驱动 fine-tuning 优势 | Cycle 08 OPD 腿次级支撑 | 前向（Cycle 08 支撑） |
| **Law of Capacity Gap, 2311.07052** | 最优 teacher 随 student 线性缩放，非越大越好；gap 过大 student 反而更差 | **Cycle 08 teacher 选择定律**：4B-Base → Qwen3-8B 同家族 | **Cycle 08 核心（teacher）** |
| **The Efficiency Misnomer, ICLR 2022 (2110.12894)** | 单一/单点成本指标会误导；公平对比应多维 + best-effort 多 alternative | **Cycle 08 对比有效性**：轨迹级对比为主、单点作 landmark、iso-ID 作二级 | **Cycle 08 核心（对比有效性）** |
| **TRL GKD + colocated vLLM** | GKDTrainer 能做 OPD，但 rollout 走 HF generate（3–8×），GKD 无一等 vLLM | Q3：TRL 能跑但非高速 OPD | 工程/工具（Cycle 08） |
| **verl Async OPD recipe** | vLLM rollout + async teacher + top-k forward KL；服务器只返 top-k logprob → 必须 top-k KL | Q3：高速 OPD 专用方案 | 工程/工具（Cycle 08） |

### 评测与训练协议来源（Methodology Grounding）

这类来源不是直接的论文论证，而是**评测/训练协议设计的依据**。

| 来源 | 核心主张 | 协议用途 | 使用级别 |
|---|---|---|---|
| Qwen3 Technical Report, arXiv 2505.09388 | Qwen3 报告官方 base-model benchmarks，并描述 pretraining/post-training stages | 作为 Qwen3 模型身份、官方 base 分数、MMLU base-model protocol 的最高权威 | 官方 methodology grounding |
| Qwen3 官方博客 | Qwen3 post-training 产生 hybrid thinking/non-thinking models；示例使用 `apply_chat_template(..., enable_thinking=True/False)` | 在决定 post-trained Qwen3-1.7B 的 train/eval chat 和 thinking settings 前必须查阅 | 官方工程依据 |
| Qwen/Qwen3-1.7B model card | Post-trained Qwen3-1.7B 在 chat template 中默认 thinking mode，并暴露 `enable_thinking` | 使用 post-trained Qwen3-1.7B 的 Cycle 06 arm 必读 | 官方 model-card 依据 |
| Qwen/Qwen3-1.7B-Base model card | Qwen3-1.7B-Base 是 pretraining-only，1.7B，28 层，32k context | 使用 base-model arm 必读；除非 SFT 引入，否则不能假设 chat/think 行为 | 官方 model-card 依据 |
| HuggingFaceTB/qwen3-1.7b-gsm8k-sft | 社区 GSM8K SFT 使用 conversational `<think>` reasoning，报告目标增益与相邻 MATH-500 下降 | 作为 SFT target/adjacent-axis tradeoff 的 warning light；不是官方 benchmark | 社区工程参考 |
| Unsloth Qwen3 guide | 总结 practical thinking/non-thinking mode 处理和 non-thinking 空 think framing | 作为格式不匹配检查的二级实现参考 | 社区工程参考 |
| lm-eval-harness Issue #3405 / #3576 / #3322 | 社区确认 Qwen3 在 chat-template loglikelihood MMLU 设置下可能塌陷 | 支持 Cycle 05 `mmlu_protocol_version=2`，继续作为 protocol-warning 来源 | 社区协议确认 |
| `source_notes/cycle06_qwen3_sft_protocol_brief-ch.md` | 上述协议来源的本地综合 | Experiment Design 前阅读；它本身不是外部引用 | 工作区来源笔记 |
| **Qwen3 Technical Report base 表（Table 6/7）+ benchmark 原始论文** | Qwen3-4B/8B-Base 官方 base 分数 + 各 benchmark 测什么/规模。 | **Cycle 08 能力退化 eval 候选轴**（Tier 1 cheap-MC + Tier 2 生成式）的量级 sanity-check 参照；正式选轴归 Experiment Design；不可跨源混用。 | 官方 benchmark 依据（Cycle 08） |
| `source_notes/cycle08_capability_degradation_benchmarks.md` | benchmark 目录：能力维度 × 测什么/规模/Qwen3-4B·8B-Base 分数 + 分层建议 + 协议警告。 | Experiment Design 前阅读选退化套件；非外部引用。 | 工作区来源笔记 |

---

## 本地实验证据（跨 Cycle 锚点）

以下不是外部文献，而是本项目跨 cycle 积累的局部实验基础：

| 观察结果 | Cycle 影响 |
|---|---|
| OPD(800) vs SFT-256 构成有效 GSM8K matched pair，gain gap 0.010（Cycle 03） | 历史正向信号；保留 OPD 作为未来比较背景 |
| OPD 的 OOD-lite penalty 和最差 OOD-lite 下降均低于 SFT-256（Cycle 03） | 仅作为历史动机；不是 Cycle 06 claim |
| OPD 在所有 7 个测量模块上 layer-14 谱漂移均低于 SFT（Cycle 03） | Geometry 仍值得保存，但 Cycle 06 仅作 diagnostic |
| Cycle 04 在更严格条件下没有复现 Cycle 03 的 OPD-like stability signal | 暂不声称 OPD stability advantage |
| 修正评测协议后：theta0 在 GSM8K（+0.0175）和 MATH500（+0.0120）上均领先 base；headroom 存在（Cycle 05） | base-ceiling 读数是测量 artifact；protocol validity 非常关键 |
| 全部 8 个模型在 4 个 OOD-lite 任务上聚集在 <1.5% 以内；判别力不足（Cycle 05） | OOD-lite 仅作为 capability-preservation check，不作为 OPD-vs-SFT 轴 |
| `sft_n1024` 数学轴严重退化而 OOD-lite 保持稳定（Cycle 05） | 触发 Cycle 06 SFT-from-base scale curve；不能写成已证明的过拟合机制 |
| Cycle 05 中 Qwen3 train/eval format mismatch 曾造成或可能造成重大 scoring artifacts | Cycle 06 必须在训练前预注册模型身份、chat/think/token settings 和 scorer audits |
| 本项目 1.7B 反复撞到的 ID/OOD 天花板，被 Rethink SFT Table 6 在官方 Qwen3-1.7B-Base 上独立复现（训练后 MATH500 净负） | 确认天花板是能力地板、与 instruct 无关 → Cycle 07 转 4B-base（外部证据 + 本项目历史一致） |
| 本项目当前训练数据 NuminaMath-1.5 短解被 Rethink SFT Table 2 点名为"广泛损害泛化"的低质量数据（外部证据，OOD reasoning 被打到地板） | Cycle 07 换 verified long-CoT 配方，NuminaMath 不再作训练数据 |
| Cycle 06 本地发现"no-think SFT 破坏 thinking 功能"（indicative）与 Rethink SFT"long-CoT/procedural pattern 是迁移来源"互证 | Cycle 07 翻转到 long-CoT think-format |
| Rethink SFT 的 dip-and-recovery（外部证据）：OOD-drop 是优化轨迹上的暂态、checkpoint 依赖，非 method 属性 | 升级研究框架为 OPD vs SFT_dip vs SFT_recover 三方对照；Cycle 07 沿 SFT 轨迹多 checkpoint 采样几何 + response length（为 Cycle 08 三方对照备基线） |
| **Cycle 03/04 "matched-GSM8K-gain 锚点退化"**：按结果变量单点匹配，SFT 锚点塌缩成近未训练的 sft_n128，对比不公平（本项目自身教训） | Cycle 08 control-anchor 的**内部锚点证据**：印证"单点匹配结果变量"之误 → 改用 matched grid 的**轨迹级对比**为 primary、iso-ID 作二级；paper 阶段可作方法学论据（呼应 The Efficiency Misnomer） |
| Cycle 07 已产出 4B-Base+Math-CoT 的 10 点 SFT 轨迹（step_000…624） | 作 Cycle 08 OPD 的 **cold-start 候选池 + matched control**：用 overlap ratio 选 cold-start 点（不盲猜步数），并作 OPD-vs-SFT 轨迹对照的对照臂 |
