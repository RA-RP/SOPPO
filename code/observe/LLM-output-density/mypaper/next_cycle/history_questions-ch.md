# 问题历史记录

```yaml
view_type: history_questions_ch
status: active
source_file: history_questions.md
maintenance_rule: append_only_per_cycle; never_overwrite_existing_entries
last_updated: 2026-06-18
```

每轮 cycle 解决的问题和新提出的问题的逐轮记录。每轮 cycle 收尾时在末尾追加一节，不修改已有条目。

---

## Cycle 04 — 初始建立（2026-06-15）

### 新提出

初始问题集建立：

- **Base-Ceiling**（Q1–Q4）：untrained base 作为参考、headroom 阈值、任务/base 选择
- **Control Construction**（Q1–Q3）：负 SFT slope 时的正确 control、matching 标准、多锚点报告
- **ID/OOD Measurement**（Q1–Q4）：NuminaMath-test 作为 ID axis、GSM8K 作为 OOD axis、per-source buckets、extraction audit
- **Geometry And Mechanism**（Q1–Q4）：effective rank 作为信号、drift 归一化、principal-angle 必要性、artifact 充分性
- **Paper Positioning**（Q1–Q3）：方法学警示转向、较小经验贡献、K1 falsifier

### 已解决

无。

---

## Cycle 05 — 评测协议修复（2026-06-17/18 关闭）

### 新提出

| 问题 | 分类 | 新增原因 |
|---|---|---|
| OPD 是否能在等效 budget 下防止灾难性遗忘？ | 遗忘预防 Q1 | sft_n1024 出现严重 NuminaMath ID 坍缩（0.5516→0.3890）；OPD 等效 budget 对比成为自然后续 |
| 灾难性遗忘在什么训练规模开始出现？ | 遗忘预防 Q2 | n=512 稳定，n=1024 严重——发生曲线未知 |
| 在缺乏 OPD 增益证据的情况下，遗忘预防是否是更可辩护的主张？ | 遗忘预防 Q3 | 若 Gate A06 失败，遗忘预防可能是唯一有支持的框架 |
| 思考模型的 MMLU 如何评测？ | 评测协议 Q1 | lm-eval 加 chat template 产生接近随机的分数（~0.23）；查阅 Qwen3 arXiv 前正确协议不明确 |
| 三轴协议检查清单 | 评测协议 Q2 | 缺少 think-axis / chat-axis / cutoff 文档导致测量 bug（MATH500=0.0 artifact、MMLU 崩溃） |

### 已解决

| 问题 | 解决结论 | 日期 |
|---|---|---|
| **Base-Ceiling Q1**：是否必须将 untrained base 作为 reference row？ | **是。** 规则确立：所有 eval 表必须包含 untrained base 作为 mandatory reference row，无例外。 | 2026-06-17 |
| **ID/OOD Q1**：NuminaMath-test 是否应成为默认 ID axis？ | **是。** NuminaMath-test 就是默认 ID axis。OPD vs SFT 比较的 matching basis 是 NuminaMath-test ID。GSM8K 是 OOD/迁移轴。（来自 `claims_allowed.md`） | 2026-06-18 |
| **ID/OOD Q2**：GSM8K 是否应作为 OOD/transfer 报告？ | **是。** GSM8K 应被视为 OOD/迁移轴，而非主要性能轴。（来自 `claims_allowed.md`） | 2026-06-18 |
| **ID/OOD Q4**：解释任何 math score 前必须做什么 extraction audit？ | **部分解决。** 项目级规则确立：任何 math score 必须先验证 scorer 提取逻辑与模型输出格式匹配。具体操作协议写入 `exp/experimental_methodology/evaluation_protocol.md`。持续有效：对所有新任务均适用。 | 2026-06-17 |
| **评测协议 Q1**：思考模型的 MMLU 如何评测？ | 采用 Qwen3 官方评测协议（arXiv 2505.09388）：base model mode（无 chat template，5-shot，standard loglikelihood），`mmlu_protocol_version=2`。所有 8 个模型有效分数在 0.60–0.61 范围内。 | 2026-06-17 |
| **评测协议 Q2**：三轴协议检查清单 | 已写入 `exp/experimental_methodology/evaluation_protocol.md`。每个评测任务必须明确记录 think-axis / chat-axis / token-cutoff-axis 设置。 | 2026-06-17 |

---

## Cycle 06 — Seed QA：SFT-first 收窄（2026-06-18）

### 新提出

| 问题 | 分类 | 新增原因 |
|---|---|---|
| 普通 SFT 是否能提高选定的下游 / ID 目标？ | Cycle 06 SFT 可行性 Q1 | 用户要求在 OPD 比较前重新一步一步确认：先证明 SFT 能产生有意义的学习信号。 |
| 哪个模型、数据规模、任务组合和 train/eval 格式能产生可靠 SFT baseline？ | Cycle 06 SFT 可行性 Q2 | 之前 cycle 一次性绑定了太多 checkpoint 和评测轴；Cycle 06 先解决工程 / 协议基础。 |
| SFT 本身是否随 scale 增加导致 ID/OOD 退化？ | 遗忘预防 / SFT 退化 Q2 | `sft_n1024` 出现严重数学退化；在追问 OPD 是否能防止之前，需要先理解 SFT 退化曲线。 |
| 是否可以在 SFT 诊断中低成本同步收集 S/X 与 PE 类 geometry，而不让其成为主 gate？ | Geometry Q4 | 用户倾向收集成本较低的 geometry 信号，但将能力/退化保持为主决策层。PE 需在 Experiment Design 阶段先给出操作性定义。 |

### 已解决

| 问题 / 决策 | 解决结论 | 日期 |
|---|---|---|
| Cycle 06 范围 | **收窄为 SFT-only。** Cycle 06 不训练新的 OPD arm，也不运行完整 OPD-vs-SFT pipeline。 | 2026-06-18 |
| OPD 遗忘预防方向 | **后置。** 先确认 SFT 是否能提高目标，以及 SFT 退化在哪里出现。 | 2026-06-18 |
| 当前 SFT 诊断起点 | **以 SFT-from-base 为主动方向。** 这样去除之前 matched-anchor comparison 中的 `theta0` 饱和 / 混淆因素。 | 2026-06-18 |
| Cycle 06 中的 geometry 角色 | **诊断项，不是主 gate。** 若成本低，可包含 S/X 与 PE 类 probes，但 PE 需在执行前定义；主 gate 仍是 SFT 学习和 ID/OOD 退化。 | 2026-06-18 |
| 论文定位 | **暂不大转向。** 项目仍聚焦 SFT 与 OPD 在 activation-space changes 和 ID/OOD degradation 上的差异；Cycle 06 是工程/测量基础步骤。 | 2026-06-18 |

---

## Cycle 06 — Close（2026-06-24）

### 新提出

| 问题 | 分类 | 新增原因 |
|---|---|---|
| No-think concise SFT 在 instruct 模型全规模下是否都无法提升 ID 能力？ | SFT 可行性 | A06 FAIL 确认：所有 instruct SFT arm 在 n∈{512,1024,2048,4096} 下 NuminaMath 均低于 base |
| No-think SFT 是否破坏了 Qwen3 思考模型的 thinking **功能**？ | 训练格式 | Think probe（indicative，n=100）：instruct_base think 模式 +0.13，no-think SFT arm −0.02..−0.06——形式保留，功能丧失 |
| Base 模型 SFT（预训练 checkpoint，无 instruct tuning，无 think/chat 格式）是否是正确实验设计？ | 模型/格式 | 用户指出 "Rethink SFT" 相关工作使用无 think/chat 的 base model；instruct 模型"太复杂"，不适合作为干净 baseline |
| 项目是否应从 Qwen3-1.7B instruct 迁移到 Qwen3-4B base？ | 模型选择 | 用户："大概率是要换成 qwen3-4B-base"；instruct 模型留作补充实验，等核心论文写完后再用 |
| 使用正确 stop sequence 后，base 模型 SFT 可行性（A06b）如何？ | 评测伪影 | Base series numina/gsm8k 是截断伪影（无 stop seq，~66% 截断）。修复：`stop=["\n\nProblem:"]`。后置到 Cycle 07。 |

### 已解决 / 新回答

| 问题 | 解决结论 | 日期 |
|---|---|---|
| **SFT 可行性 Q1（instruct + no-think）**：SFT 能否提升 ID？ | **否（确认性）。** A06 FAIL：instruct no-think SFT 在所有测试规模下 NuminaMath 退化 −0.12 至 −0.17。 | 2026-06-24 |
| **SFT 退化 Q1（instruct + no-think）**：SFT 是否导致 ID 退化？ | **是（确认性）。** 所有 instruct SFT arm 在 NuminaMath 和 MATH500 上退化；GSM8K 大体保持。OOD-lite 在噪声范围内保持（B06 = 通用能力无退化）。 | 2026-06-24 |
| **训练数据量**：训练数据太少是否是核心制约？ | **否。** n=4096 不比 n=512 好；更多数据单调加重 ID 退化。数量不是瓶颈，格式才是。 | 2026-06-24 |
| **Base GSM8K 伪影**：base series 的 GSM8K/numina 增益是否真实？ | **否。** 无 stop sequence 时约 66% 截断；表面上的 +0.28 GSM8K 增益是差分去截断伪影。加 stop 后 base GSM8K 持平约 0.66，SFT 中性。 | 2026-06-24 |
| **Geometry copyback（C06）**：cycle06 geometry CSV 是否已分析？ | **否。** `geometry_metrics.csv` 和 `principal_evidence.csv` 未 copyback；OverlapLift 未分析。C06 PARTIAL。作为 Cycle 07 第一项任务延续。 | 2026-06-24 |

---

## Cycle 07 — Seed QA（2026-06-24）

### 关键决策

| 决策 | 内容 |
|---|---|
| **Cycle 07 模型方向** | **迁移到 Qwen3-4B base**（预训练 checkpoint，非 instruct）。在 Experiment Design 之前，需由 Related Work（Rethink SFT 等）确认。Instruct 模型后置到补充实验。 |
| **Cycle 07 训练格式** | Base 模型 SFT——无 think-format 需求，无 chat-format 需求。具体格式决策待 Related Work。 |
| **Think-format SFT（B）** | 后置；相关工作确认 base 模型协议后可能不再需要。 |
| **训练数据** | 大概率继续使用 NuminaMath；待 Related Work。无相关工作依据不更换数据源。 |
| **Cycle 07 即时任务** | ①Base series stop-fix 重新收集；②Cycle 06 geometry copyback + OverlapLift 分析。两项均开销较低，应在新训练设计前完成。 |
| **Think-probe 确认性扩展** | **后置。** 项目迁移到 base 模型（无 instruct）后，instruct think-probe 发现变为次要，无需现在扩展到 full math500 + multi-seed。 |
| **Gate 结构** | Gate A07（base SFT 可行性）+ Gate D07（A06b 重确认）为主要 gate。B07（格式对比：instruct vs no-think）和 C07（think-SFT forgetting）大概率不需要（模型已迁移）。A07 的下游任务待 Related Work 确认。 |
| **Related Work 依赖** | Cycle 07 Experiment Design 阻塞于 Related Work 阶段完成。具体需确认：模型选择、训练格式、下游任务评测轴。 |

### 新提出

| 问题 | 分类 |
|---|---|
| Qwen3-4B base 模型 SFT（base 格式，无 instruct/think）是否能产生可测量的 ID / 下游增益？ | Cycle 07 SFT 可行性（base 模型） |
| 对 base 模型，正确的下游 / ID 评测轴是什么？（NuminaMath 开放回答 vs few-shot 生成格式？） | Base 模型评测协议 |
| 相关工作中 base 模型 SFT 的训练格式（数据结构、目标、chat/no-chat）是如何确立的？ | 训练格式（base 模型） |
| "Rethink SFT" 和相关文献对 base vs instruct 模型的 SFT 对比实验有何建议？ | Related Work 依赖 |

---

## Cycle 07 — Close（2026-06-27）

### 新提出

| 问题 | 分类 | 新增原因 |
|---|---|---|
| Rethink SFT 的 dip-and-recovery 轨迹是否在 LoRA 微调下复现？ | SFT 轨迹 | B07 FULL PASS 确认：step_20 下陷（finisher acc 0.594），step_160 恢复，峰值 0.698@step_320 |
| think-format CoT 下 max_tokens=4096 的 as-run acc 是否 cap-robust？ | 评测协议 | Cap-pilot 确认：0.620@4096 ≈ 0.617@24576；截断在计算上合理，acc 测量有效 |
| step_20 三重共定位（argmin acc = argmax rank = argmin xs_gap）的机制意义是什么？ | Geometry | 三重共定位鲁棒；concurrent-not-causal 解读；效应量小（~1.8% rank span） |
| LoRA r=32 下如何归一化 OverlapLift 以给出 on/off-principal 结论？ | OverlapLift / 方法论 | OverlapLift < 1 全程（单调上升 0.48→0.74）；random baseline 在 LoRA 下不归一化低秩结构；解读延后至 Cycle 08 |
| Think-format 数学 SFT 是否导致 OOD 退化？ | OOD 泛化 | GPQA-D 平稳；MMLU-Pro 轻微真实退化（final −2.2 SE，非单调）；LoRA 变体下 SFT 增益局限于数学领域 |

### 已解决 / 新回答

| 问题 | 解决结论 | 日期 |
|---|---|---|
| **A07：Base SFT 可行性（Qwen3-4B-Base + Math-CoT-20k）** | **PASS。** Final MATH500 0.680 > base+1SE 0.658。Final step finisher acc 0.98。Think-format base SFT 可行。 | 2026-06-27 |
| **B07：LoRA 下的 dip-and-recovery 轨迹** | **FULL PASS。** Step_20 下陷（finisher acc 0.594，非终止率 6.8%——真实能力下降）。Step_160 恢复至 base 以上。峰值 0.698@step_320。 | 2026-06-27 |
| **C07：OOD-lite 迁移** | **无迁移 + MMLU-Pro 轻微退化。** GPQA-Diamond 平稳（|Δ|<0.75 SE）。MMLU-Pro final −2.2 SE（轻微，真实，非单调）。 | 2026-06-27 |
| **模型选择** | **Qwen3-4B-Base 确认。** SFT 可行性已在此 checkpoint 上建立。 | 2026-06-27 |
| **训练格式** | **Math-CoT-20k 上的 think-format LoRA 确认。** Dip-and-recovery 与 Rethink SFT LoRA 变体动态吻合。 | 2026-06-27 |
| **Base 模型评测协议** | **已确立：** MATH500 生成式，无 chat template，think-format，max_tokens=4096，as-run acc（cap-robust）。 | 2026-06-27 |
| **As-run acc 与 finisher acc** | **As-run acc 为主要指标。** Cap-pilot 已证明 cap-robust（0.620@4096 ≈ 0.617@24576）。非终止是该制度的已知属性，计算上已接受。Finisher acc 可用于分解分析。 | 2026-06-27 |
| **Drift 参考 checkpoint** | **以 base checkpoint（step_000）为 drift 参考**，而非 theta0。Cycle 07 已确认 drift_from_base 单调 = 负对照。 | 2026-06-27 |
| **C06 PARTIAL（geometry copyback）** | **已关闭。** 决策：跳过；模型已迁移至 4B+Math-CoT，不再相关。 | 2026-06-27 |
| **A06b UNDETERMINED（base stop-fix）** | **已关闭。** 决策：跳过；1.7B base 实验不再是主线。 | 2026-06-27 |

---

## Cycle 08 — Seed QA（2026-06-27）

### 关键决策

| 决策 | 内容 |
|---|---|
| **Cycle 08 主要目标** | 在同样的 Qwen3-4B-Base + Math-CoT-20k + LoRA 设置下训练 OPD arm。对比 OverlapLift、geometry 和能力/OOD-lite 与 Cycle 07 SFT 轨迹。 |
| **OPD arm 设计** | 单个 arm，一个 best-guess lambda（不做 sweep）。OPD 开销大；Experiment Design 为 4B+Math-CoT 制度选择最合适的 lambda。 |
| **非终止处理** | 接受 max_tokens=4096（cap-robust 已确认）。As-run acc 为主要指标。分解 finisher acc + 非终止率用于对比分析。Cycle 08 **不更改** max_tokens。 |
| **OOD-lite 覆盖** | 全轨迹跟踪：OPD arm 全部 10 个 checkpoint，GPQA-D + MMLU-Pro。 |
| **C06 + A06b** | 两者均已关闭，不再追踪。 |
| **Gate 优先级** | B08（能力保留）> C08（MMLU-Pro 保留）> D08（geometry 模式）> A08（OverlapLift 轨迹）。四个 gate 全部活跃；B08 为主要阻塞性 gate。 |

### 新提出

| 问题 | 分类 |
|---|---|
| OPD arm final MATH500 是否匹配 SFT（非劣效性），OPD 是否避免 step_20 级别的下陷？ | Gate B08：能力保留 |
| OPD arm 在 final checkpoint 及完整轨迹上 MMLU-Pro 退化是否小于 SFT arm？ | Gate C08：MMLU-Pro 保留 |
| OPD arm 在其能力低点的 geometry 是否不同于 SFT step_20 三重共定位？ | Gate D08：Geometry 模式 |
| OPD OverlapLift 轨迹是否与 SFT（单调上升 0.48→0.74）系统性地不同？ | Gate A08：OverlapLift 比较 |
| Qwen3-4B-Base + Math-CoT 制度下最适合的 OPD lambda 是什么？ | OPD 设计 |
| LoRA-aware OverlapLift 归一化参考基准是什么，使 Cycle 08 能给出 on/off-principal 结论？ | OverlapLift 方法论 |
| OPD arm 是否展示与 SFT 相近的非终止率（~50%），还是 OPD 会影响输出控制？ | 输出控制 |

---

## Cycle 08 — Close（2026-07-07）

### 新提出（来自 Cycle 08 Seed QA）→ 已解决

| 问题 | 分类 | 结论 |
|---|---|---|
| OPD arm final MATH500 是否匹配 SFT（非劣效性），OPD 是否避免 step_20 级别的下陷？ | Gate B08：能力保留 | **STRONG PASS**：OPD final 0.848 vs SFT 0.752（+0.096），unified cap 16384；每个 step OPD 均优于 SFT（+0.10 至 +0.24）。 |
| OPD arm 在 final checkpoint 及轨迹上 MMLU-Pro 退化是否小于 SFT arm？ | Gate C08：MMLU-Pro 保留 | **PASS**：OPD net +0.016 vs SFT net −0.029；delta margin +0.045。轨迹非单调（step_40/80 大跌，step_160 恢复）。 |
| OPD arm 在其能力低点的 geometry 是否不同于 SFT step_20 三重共定位？ | Gate D08：Geometry 模式 | **正面**：SFT 出现 L18 step_20 rank bump；OPD ER 单调收缩无 bump。ER 是唯一清晰区分两个 arm 并追踪 OOD 结果的指标。 |
| OPD OverlapLift 轨迹是否与 SFT 系统性地不同？ | Gate A08：权重空间方向 | **不确定**：OverlapLift retracted（≈ random-LoRA null，量级/bf16 主导）。ρ 读 OPD on-principal / SFT 早期 off-principal；与外部全参数文献矛盾（2606.13657/07082 说 OPD off-principal）。全参数/fp32 测量推迟。 |
| Qwen3-4B-Base + Math-CoT 制度下最适合的 OPD lambda？ | OPD 设计 | **已解决**：使用 lambda=1.0（verl supervised forward_kl_topk，topk=32）。 |
| LoRA-aware OverlapLift 归一化参考基准是什么？ | OverlapLift 方法论 | **已收回**：random-LoRA null 证明 OverlapLift 无意义。ρ 已引入但与外部文献矛盾→尚未解决。 |
| OPD arm 是否展示与 SFT 相近的非终止率（~50%）？ | 输出控制 | **新发现**：OPD response length 单调增加（626→3974 token）。在 4096 上晚期截断严重→需 unified cap 16384；B08 使用修正后数字。OPD 终止模式与 SFT 不同。 |

### 新发现（Cycle 08 Seed 中未预注册）

| 发现 | 意义 |
|---|---|
| **OPD 早期下陷 step_5**（MATH500 −0.100；numina −0.070） | 比 SFT step_20 下陷更早、更深（−0.064）。OPD 到 step_20 恢复；SFT 到 step_160。两者均与 ER uptick 共定位。支持"更尖锐更短暂瞬态"机制故事。 |
| **LoRA ρ 与全参数文献矛盾** | LoRA ρ 读 OPD on-principal；外部全参数文献说 OPD off-principal。LoRA 结构可能改变了全参数下的主方向几何规律→新贡献角度。 |
| **OPD 需要 eval-cap 修正** | OPD 长蒸馏 CoT 在 4096 下系统性截断→晚期 MATH500 偏低。已应用 unified cap 16384。"峰值后下降"是截断伪影，非真实能力下降。 |

---

## Cycle 09 — Seed QA（2026-07-07）

### 关键决策

| 决策 | 内容 |
|---|---|
| **Cycle 09 首要任务（Q1=c，扩展范围）** | 首先：明确并收窄论文的创新贡献（鉴于已有外部全参数文献）。然后：跑 fp32/全参数 retrain + P0（统计 CI + geometry + 输出控制）+ P2（理论 + 论文改写）。P1 推迟至 Cycle 10。 |
| **权重空间方向（Q2=a，LoRA 补充 + 新观测空间愿景）** | 跑全参数/fp32 retrain 以解决 ρ 矛盾。三部分贡献框架：(1) LoRA 改变了主方向几何规律 vs 全参数→LoRA 补充贡献；(2) activation-ER 作为具有区分力的新观测空间；(3) 几何-过程-结果链：ER 状态→dip-recovery 过程→OOD 结果。可能需要全参数+多任务验证。 |
| **OPD 早期下陷作为核心贡献（Q3=a）** | OPD step_5"更尖锐更短暂瞬态"vs SFT step_20"更浅更缓慢瞬态"是核心机制贡献之一。需与 CI 和最终结果结合。 |
| **P1 推迟至 Cycle 10（Q4=d）** | off-policy KD 控制（Gate D09）和 robustness 轴（Gate E09）推迟。主要贡献尚未确立；次要验证在核心主张固化后再做。 |
| **Cycle 09 范围** | P0 扩展（贡献定位 + fp32 测量 + 统计 CI + geometry + 输出控制）+ P2（理论 + 论文改写）。P1（控制 + robustness + teacher baselines）→ Cycle 10。 |

### 新提出

| 问题 | 分类 |
|---|---|
| 鉴于外部全参数文献（OPD 在权重空间 off-principal）已发表，我们的精确创新贡献是什么？ | 贡献定位（紧急） |
| 全参数/fp32 retrain 是否确认 LoRA ρ 读数（OPD on-principal）还是符合外部文献（OPD off-principal）？ | Gate A09：权重空间方向 |
| LoRA vs 全参数几何差异能否本身成为一项贡献（"LoRA 改变主方向"）？ | 贡献定位 |
| Paired bootstrap CI 能否证明 OPD 早期下陷深度/恢复速度差异在统计上显著？ | OPD Dip 机制主张门控 |
| 能否在 OOD 领域输入（MMLU-Pro prompts）上测量 ER，以强化压缩→OOD 因果链？ | 压缩→OOD 因果链 |
| 几何-过程-结果链（ER 状态→dip-recovery→OOD 结果）能否在论文中作为主叙事？ | 理论 / 论文框架 |
| OPD 输出控制（response length、非终止率）是否能解释与 SFT 的准确率差异，还是纯能力差距？ | 输出控制分析（C09） |
| 所有 main-text OPD-vs-SFT claim 是否均附有 paired bootstrap 95% CI？ | 统计可靠性（A09） |
