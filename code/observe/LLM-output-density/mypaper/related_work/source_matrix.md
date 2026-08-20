# Source Matrix — Related Work Synthesis (综述)

```yaml
view_type: synthesis_matrix
status: active
maintenance_rule: add_new_entries_per_cycle; remove_only_if_confirmed_unused_and_note_reason
last_updated: 2026-07-07
cycle06_position: sft_feasibility_and_degradation_first
cycle07_position: base_model_sft_feasibility_generalization_conditional + theory_deepening_sft_trajectory
cycle08_position: h_opd_vs_sft_comparison + opd_recipe_tooling + matched_comparison_validity
cycle09_position: aaai_scoop_audit + moat_gap_map + er_construct_vs_2605.30524
```

本文件是**综述文件**：分类介绍各引用论文之间的关系，说明本项目为什么需要这一组论文。
每篇论文的具体信息（标题、URL、内容概括）见 `related_work.md`。
每轮新增/删除记录见 `incremental_related_work.md`。

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

**(B) 理论深化：把 SFT 从单点拆成轨迹（SFT_dip / SFT_recover），把 OOD-drop 重定义为 checkpoint 属性。** Rethink SFT 的 dip-and-recovery（Sec 3）+ response-length-as-diagnostic（Sec 3.2）+ shallow-imitation vs internalized-pattern（C.3/C.6）支撑一个升级后的研究框架：原"OPD vs SFT 的 OOD-drop/继续学习/几何指标"二元对照，升级为 **OPD vs SFT_dip vs SFT_recover** 三方对照，核心假设是"**OPD ≈ 不经过 dip 暂态、直达 recover 状态**"。

> 由此产生一个**新文献缺口并已在本轮补齐**：三方框架的 OPD 腿需要"on-policy/online 训练忘得少"文献簇——**Chen et al. 2025（Retaining by doing）、Shenfeld et al. 2026（RL's razor）、Tajwar et al. 2024**。本轮登记为 OPD 腿备料（Cycle 08 引用），不在 Cycle 07 作 OPD claim。
>
> **写作边界**：正式理论改写归 Theory 对话写 `theory/`；Related Work 只提供文献脚手架与"三方框架的文献支撑"判断。
>
> **实验含义（提醒 Experiment Design）**：SFT_dip/recover 与几何指标的关系**本轮（Cycle 07）即可测**（无需 OPD），但要求沿 SFT 轨迹**多 checkpoint 采样几何指标 + response length**，否则 Cycle 08 三方对照缺基线。OPD 腿留到 Cycle 08。

**control-anchor 留位（Cycle 06 约定）仍然有效：** 一旦 Cycle 08 进入 OPD-vs-SFT 对齐比较，Related Work 必须提醒是否补做 baseline construction / control anchor / matched comparison validity 搜索——届时与上面的 on-policy-forgetting 文献簇一并激活。

---

## Cycle 08 文献定位更新（2026-06-27）

Cycle 08 = **H-OPD vs SFT 对比**（4B-Base + Math-CoT-20k + think-LoRA，单 OPD arm 单 lambda，Cycle 07 SFT 轨迹作 matched control）。intent 未指定 **teacher** 与 **cold-start**——本轮 related work 补三类来源（研究笔记 `source_notes/cycle08_opd_teacher_coldstart_tooling.md`）：

**(A) OPD teacher 选择 + cold-start 配方（QA Q1/Q2）。** capacity-gap 定律（arXiv 2311.07052）：最优 teacher 随 student 线性、非越大越好 → 本项目 4B-Base student 选 **Qwen3-8B-Instruct 同家族**（用户已定），32B 仅离线数据。cold-start：Rethinking OPD（Li et al.）实测——纯 OPD-from-base 可能但条件差，判据是 **overlap ratio**（成功 72%→91%）；步数 scale-dependent（其 ~3000 步 plateau 不可照搬）。⚠️ 本项目 base(无think)→instruct(有think)+think-format 训练，初始 overlap 很可能低，**纯 from-base 风险高，需 overlap probe 定夺最小 cold-start**（→ Exp/Code）。

**(B) OPD 训练工具（QA Q3）。** TRL GKDTrainer 能跑但**非高速**（rollout 走 HF generate，3–8× 开销，GKD 无一等 vLLM）；**verl async OPD 是高速专用**（vLLM rollout + async teacher + top-k forward KL，工程约束：推理服务器只返 top-k logprob → 必须 top-k KL）。框架决策归 Code/Exp Design，本节只供工具综述。

**(C) control-anchor / matched-comparison validity（QA Q4=a，留位激活）。** Cycle 08 是首次 OPD-vs-SFT 对齐比较，按 Cycle 06 约定**本轮激活**该方向。新增 The Efficiency Misnomer（Dehghani et al., 2110.12894）等方法学锚点。**核心方法学输入（用户 Q4 追问"为什么单点而非整条趋势"）**：单点匹配一个**结果变量**（ID gain）= Cycle 03/04"matched-GSM8K-gain 锚点退化"那类错误；与本项目把 OOD-drop 重定义为**轨迹属性**自相矛盾。正解：在**输入轴**（budget / checkpoint grid，Cycle 08 已采用）匹配 → 比**整条 dip-recover 轨迹**为 primary；单点（dip 深度/时点、recover step、终点）作 **landmark**；iso-ID 作**二级 cross-section**（P1）。正式比较设计归 Theory/Exp Design，此处仅作文献+方法学输入。

**(D) 能力退化 eval 广度拓展（用户追加）。** Rethink SFT 套件偏难推理 + 偏"获得"轴；本项目训练在数学 long-CoT 上，需补**非数学、base 已强**的退化轴。拍板候选（详见 `source_notes/cycle08_capability_degradation_benchmarks.md`，含 Qwen3-4B/8B-Base 官方分数）：Tier 1 cheap-MC 全 checkpoint track（MMLU/HellaSwag/WinoGrande/ARC-C/PIQA/TruthfulQA-MC1）+ Tier 2 生成式抽测（BBH/GSM8K/EvalPlus）。⚠️ 协议依赖不可混用（官方仅作 sanity-check，本项目统一协议自测）；退化轴须 base 强；commonsense 须先验 4B 判别力。**不改 H-OPD primary gate（B08/C08 仍以 MATH500+MMLU-Pro 为主）**，退化套件是补充的能力保持证据。正式选轴/预注册归 Experiment Design。

---

## Cycle 09 文献定位更新（2026-07-07）：Scoop Audit + 护城河 Gap Map

Cycle 09 = AAAI 竞争力收尾。Related Work 任务=scoop 审计 + 新颖性缺口地图，供 Theory 选 framing。**完整转交文件：`source_notes/cycle09_scoop_audit_and_gap_map.md`（Theory 先读它，再读 claims_allowed）。**

**三篇 scoop 审计**：2606.07082 / 2606.13657（OPD **weight-space** 几何，off-principal，已抢发→framing (a) 死；但不碰 activation/OOD）；**2605.30524（头号威胁）** 已发"activation 压缩→OOD 差"，但**SFT族/LoRA/静态预测/方向相反/raw 非白化 ER**，且**不做 OPD、不比 OPD-vs-SFT、不做单run dip-recover**。

**ER 构念对照**：本项目 GetSlice ER 源自 **SVD-LLM（2403.07378）白化**（输出相关截断最优谱）+ 逐模块 + 相对参考；2605.30524 是 raw 中心化残差流协方差。→ 构念不同，"方向冲突"未必正面矛盾；建议 Exp Design **双 ER 并测**直接 engage。

**护城河交集（无人占）**：OPD-vs-SFT × SVD-LLM 白化输出相关谱 × 单run dip-recover transient × 方向对比 × 机制/过程 framing。→ **framing (a) 弃；(b) 须锐化成"OPD-vs-SFT+白化谱+轨迹+方向对比"；(c) geometry→process→result 链最稳。** 最终 framing 归 Theory。

**新创新点（用户提，Related Work 评估）**：ER-based 数据筛选让 SFT 逼近 OPD——把观测升级为**干预/因果**（同 2605.30524 mitigation 逻辑）+ actionable payoff，与全部 scoop 差异化；呼应早期 `source_notes/ood_id_positioning_with_ppl.md`。⚠️ 属新实验轴，**建议 Cycle 10 核心**（Cycle 09 至多 preliminary）；需补搜"geometry/ER-based data selection"确认未被占。

**AAAI 判断**：当前纯观测交集**偏薄**（n=2、相关非因果、三篇近邻 scoop 抬门槛）；强化路径=双 ER 直接 engage 2605.30524 + dip-recover CI（Cycle 09）+ 数据筛选干预（Cycle 10 最强）。

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

Cycle 05 已经证明：格式协议错误足以制造 GSM8K、MATH500、MMLU 的假结论。Cycle 06 因此需要把模型身份、训练 target format、评测 chat/think/token settings 和 scorer/extractor audit 写成设计前置条件。详见 `source_notes/cycle06_qwen3_sft_protocol_brief.md`，其中已经补入当前项目的训练格式和 GSM8K / MATH500 / MMLU / TruthfulQA / WinoGrande / ARC 评测格式表。

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

### 核心论文（Research Argument）

| Source | Relevant Claim | Role | Use Level |
|---|---|---|---|
| Agarwal et al., GKD / OPD, ICLR 2024 | Student self-generated sequences plus teacher feedback address the fixed-output limitation of standard KD. | Justifies why OPD-like training may differ from SFT in later comparisons. | Background / Core for future OPD |
| verl OPD documentation | OPD samples states from the student policy, reduces exposure bias, and gives dense token-level supervision. | Defines the practical OPD contrast against SFT/KD. | Background / Implementation |
| Rethinking OPD, 2026 | Successful OPD depends on growing overlap between student distribution and teacher-supported high-probability tokens; cold start can help. | Motivates future OPD cold-start and alignment design, not Cycle 06 SFT-only training. | Future OPD framing |
| OPD survey, 2026 | OPD methods can be organized by feedback signal, teacher access, and loss granularity. | Positions this project as white-box, token-level, output-space OPD-like. | Framing |
| Lobo et al., CoT fine-tuning, NAACL 2025 | Fine-tuning can alter reasoning behavior and reduce CoT faithfulness. | Supports measuring reasoning degradation alongside target gain. | Core for Cycle 06 |
| Luo et al., catastrophic forgetting, 2023/2025 | Continual instruction tuning can produce catastrophic forgetting in 1B-7B LLMs. | Supports treating 1.7B SFT as a forgetting-risk setting. | Core for Cycle 06 |
| Kalajdzievski, scaling laws for forgetting, 2024 | Forgetting relates to fine-tuning performance, update steps, and tuned parameter scale; LoRA does not remove it. | Supports pre-registering SFT data size/update exposure. | Core for Cycle 06 |
| Dong et al., SFT data composition, ACL 2024 | SFT abilities scale differently; data amount/composition can create conflicts; sequential training risks forgetting. | Closest analogue for Cycle 06's SFT scale curve and ID/OOD separation. | Core for Cycle 06 |
| Ding & Wang, improved SFT mitigation, 2025 | Domain-specific SFT often degrades general capabilities; mixed/reconstructed general data can mitigate forgetting. | Future mitigation baseline if Cycle 06 finds SFT degradation. | Mitigation / Future |
| LoRA vs Full Fine-tuning, 2024 | Spectral structure can reveal fine-tuning behavior and forgetting mechanisms. | Supports geometry as diagnostic, not current mechanism proof. | Diagnostic |
| OPRD, 2026 | Representation-level on-policy distillation is an emerging OPD axis. | Future direction; not part of current SFT-only cycle. | Future |
| **Rethink SFT (Ren et al.), 2026** | Reasoning-SFT cross-domain generalization is **conditional** on optimization (dip-and-recovery), data quality/structure (verified long-CoT ≫ NuminaMath short), and model capability (1.7B fails, 4B+ generalizes). | **Cycle 07 主来源**：决定 model(4B-base)/data(long-CoT)/format(think)/eval(MATH500+AIME24/LCB/GPQA/MMLU-Pro) 四线；并提供 SFT_dip/recover 轨迹的实证骨架。 | **Core for Cycle 07** |
| Chen et al., Retaining by doing, 2025 | On-policy data mitigates catastrophic forgetting relative to off-policy SFT. | Cycle 08 三方框架 OPD 腿备料：支撑"OPD ≈ recover-without-dip"。 | Forward (Cycle 08 OPD leg) |
| Shenfeld et al., RL's razor, ICLR 2026 | Online/on-policy RL forgets less; mode-seeking updates stay near base distribution. | Cycle 08 OPD 腿备料：预测 OPD 的 drift-from-base 应平于 SFT_dip 暂态。 | Forward (Cycle 08 OPD leg) |
| Tajwar et al., on-policy preference FT, ICML 2024 | On-policy sampling itself (not just data quality) drives the fine-tuning advantage. | Cycle 08 OPD 腿次级支撑。 | Forward (Cycle 08 support) |
| **Law of Capacity Gap, 2311.07052** | 最优 teacher 随 student 线性缩放，非越大越好；gap 过大 student 反而更差。 | **Cycle 08 teacher 选择定律**：4B-Base → Qwen3-8B 同家族（非 14B/32B）。 | **Core for Cycle 08 (teacher)** |
| **The Efficiency Misnomer, ICLR 2022 (2110.12894)** | 单一成本/单点指标会误导；公平对比应多维 + best-effort 多 alternative。 | **Cycle 08 matched-comparison validity**：支撑轨迹级对比为主、单点作 landmark、iso-ID 作二级；防御混杂质疑。 | **Core for Cycle 08 (comparison validity)** |

### OPD 工程 / 工具来源（Engineering / Tooling，Cycle 08）

回答"用什么训练框架高速跑 OPD"，归 Code/Exp Design 决策，本节只作工具综述。

| Source | Relevant Claim | Tooling Decision / Use | Use Level |
|---|---|---|---|
| TRL GKD Trainer + colocated vLLM (PR #3394, v0.18.0) | GKDTrainer 做 OPD（lmbda/beta/teacher_model）；rollout 走 HF generate（3–8× 开销），GKD 无一等 vLLM；colocate 文档化的是 GRPO。 | 本项目 Q3：TRL 能跑但**非高速** OPD。 | Engineering grounding (Cycle 08) |
| verl Async On-Policy KD recipe | vLLM rollout + async teacher（独立 pool，无 batch barrier）+ top-k forward KL（`forward_kl_topk`，topk 32）；推理服务器只返 top-k logprob → 必须 top-k KL；cold-start 不内置。 | 本项目 Q3：**高速 OPD 专用方案**；提供实现约束（top-k KL、同家族 teacher）。 | Engineering grounding (Cycle 08) |

### 评测与训练协议来源（Methodology Grounding）

这类来源不是直接的 related work 论证，而是**评测/训练协议设计的依据**。

| Source | Relevant Claim | Protocol Decision / Use | Use Level |
|---|---|---|---|
| Qwen3 Technical Report, arXiv 2505.09388 | Qwen3 reports official base-model benchmarks and describes pretraining/post-training stages. | Use as the top-level authority for Qwen3 model identity, official base scores, and MMLU base-model protocol. | Official methodology grounding |
| Qwen3 official blog | Qwen3 post-training creates hybrid thinking/non-thinking models; examples use `apply_chat_template(..., enable_thinking=True/False)`. | Use before deciding train/eval chat and thinking settings for post-trained Qwen3-1.7B. | Official engineering grounding |
| Qwen/Qwen3-1.7B model card | Post-trained Qwen3-1.7B defaults to thinking mode in chat template and exposes `enable_thinking`. | Required source for any Cycle 06 arm using post-trained Qwen3-1.7B. | Official model-card grounding |
| Qwen/Qwen3-1.7B-Base model card | Qwen3-1.7B-Base is pretraining-only, 1.7B, 28 layers, 32k context. | Required source for any Cycle 06 base-model arm; do not assume chat/think behavior unless introduced by SFT. | Official model-card grounding |
| HuggingFaceTB/qwen3-1.7b-gsm8k-sft | Community GSM8K SFT uses conversational `<think>` reasoning and reports target gain with adjacent MATH-500 drop. | Use as warning light for SFT target/adjacent-axis tradeoff; not official benchmark. | Community engineering reference |
| Unsloth Qwen3 guide | Summarizes practical thinking/non-thinking mode handling and non-thinking empty think framing. | Secondary implementation check for format mismatch. | Community engineering reference |
| lm-eval-harness Issue #3405 / #3576 / #3322 | Community confirms Qwen3 MMLU can collapse under chat-template loglikelihood settings. | Supports Cycle 05 `mmlu_protocol_version=2`; keep as protocol-warning source. | Community protocol confirmation |
| `source_notes/cycle06_qwen3_sft_protocol_brief.md` | Local synthesis of the above protocol sources. | Read before Experiment Design; not itself an external citation. | Workspace source note |
| **Qwen3 Technical Report base 分数表（Table 6/7）+ benchmark 原始论文** | Qwen3-4B/8B-Base 官方 base 分数（MMLU/BBH/GSM8K/EvalPlus/MMLU-Pro/GPQA/MATH…），各 benchmark 测什么/规模。 | **Cycle 08 能力退化 eval 候选轴**（Tier 1 cheap-MC + Tier 2 生成式）的量级 sanity-check 参照；正式选轴归 Experiment Design。仅作参照，不可跨源混用。 | Official benchmark grounding (Cycle 08) |
| `source_notes/cycle08_capability_degradation_benchmarks.md` | benchmark 目录：能力维度 × 测什么/规模/Qwen3-4B·8B-Base 分数 + 分层建议 + 协议警告。 | Read before Experiment Design 选退化 eval 套件；非外部引用。 | Workspace source note |

---

## 本地实验证据（Cross-Cycle Anchors）

以下不是外部文献，而是本项目跨 cycle 积累的局部实验基础，与 source matrix 并列记录：

| Observation | Cycle Consequence |
|---|---|
| OPD(800) vs SFT-256 was a valid GSM8K matched pair with gain gap 0.010 (Cycle 03). | Historical positive signal; keep OPD as future comparison background. |
| OPD had lower OOD-lite penalty and lower worst OOD-lite drop than SFT-256 (Cycle 03). | Historical motivation only; not a Cycle 06 claim. |
| OPD had lower layer-14 spectral drift across all 7 measured modules (Cycle 03). | Geometry remains worth saving, but Cycle 06 uses it diagnostic-only. |
| Cycle 04 did not reproduce Cycle 03's OPD-like stability signal under stricter conditions. | Do not claim OPD stability advantage yet. |
| After corrected eval protocol: theta0 leads base on GSM8K (+0.0175) and MATH500 (+0.0120); headroom exists (Cycle 05). | Base-ceiling reading was measurement artifact; protocol validity matters. |
| All 8 models cluster within <1.5% on 4 OOD-lite tasks; discriminative power insufficient (Cycle 05). | OOD-lite is capability-preservation check only, not OPD-vs-SFT axis. |
| `sft_n1024` shows severe math regression while OOD-lite remains stable (Cycle 05). | Motivates Cycle 06 SFT-from-base scale curve; do not write it as proven overfitting mechanism. |
| Qwen3 train/eval format mismatches caused or risked major scoring artifacts in Cycle 05. | Cycle 06 must pre-register model identity, chat/think/token settings, and scorer audits before training. |
| 本项目 1.7B 反复撞到的 ID/OOD 天花板，被 Rethink SFT Table 6 在官方 Qwen3-1.7B-Base 上独立复现（训练后 MATH500 净负）。| 确认天花板是能力地板、与 instruct 无关 → Cycle 07 转 4B-base（外部证据 + 本项目历史一致）。 |
| 本项目当前训练数据 NuminaMath-1.5 短解被 Rethink SFT Table 2 点名为"广泛损害泛化"的低质量数据（外部证据，OOD reasoning 被打到地板）。| Cycle 07 换 verified long-CoT 配方，NuminaMath 不再作训练数据。 |
| Cycle 06 本地发现"no-think SFT 破坏 thinking 功能"（indicative）与 Rethink SFT"long-CoT/procedural pattern 是迁移来源"互证。| Cycle 07 翻转到 long-CoT think-format。 |
| Rethink SFT 的 dip-and-recovery（外部证据）：OOD-drop 是优化轨迹上的暂态、checkpoint 依赖，非 method 属性。| 升级研究框架为 OPD vs SFT_dip vs SFT_recover 三方对照；Cycle 07 沿 SFT 轨迹多 checkpoint 采样几何 + response length（为 Cycle 08 三方对照备基线）。 |
| **Cycle 03/04 "matched-GSM8K-gain 锚点退化"**：按结果变量(GSM8K gain)单点匹配，SFT 锚点塌缩成近未训练的 sft_n128，对比不公平（本项目自身教训）。| Cycle 08 control-anchor 的**内部锚点证据**：印证"单点匹配结果变量"之误 → 改用 matched checkpoint grid 的**轨迹级对比**为 primary、iso-ID 作二级；paper 阶段可直接作方法学论据（呼应 The Efficiency Misnomer）。 |
| Cycle 07 已产出 4B-Base+Math-CoT 的 10 点 SFT 轨迹（step_000…624）。| 作 Cycle 08 OPD 的 **cold-start 候选池 + matched control**：用 overlap ratio 在其上选 cold-start 点（不盲猜步数），并作 OPD-vs-SFT 轨迹对照的对照臂。 |
