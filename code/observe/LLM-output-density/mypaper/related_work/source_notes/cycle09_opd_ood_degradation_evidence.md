# Cycle 09 证据卷宗：OPD 的 OOD/遗忘衰退（供 Theory 精读 + 批判）

```yaml
view_type: extended_source_note
cycle: cycle_09_aaai_competitiveness_completion
purpose: 回答 Theory 五问（OPD OOD 衰退程度 / OPD vs SFT / vs off-policy KD / vs RL / 知识 vs 格式）；逐篇详列实验设置+结果+问题所在
status: reference
created: 2026-07-07
caveat: 含两篇博客（非同行评审）；多为 7-8B instruct，非本项目 4B-base；数字为抽取，精读以原文为准。
```

**一句话结论**：衰退**SFT ≈ off-policy KD ＞ OPD ＞ RL**（RL 最小、OPD 贴近 RL、SFT 最大）；破坏的**主要是格式/行为/分布（instruction-following、reasoning depth、mode），知识大体保住**。⚠️ 但"OPD 比 SFT 保 OOD 好"**已被广泛确立** → 本项目不能当新颖 claim，新颖须落在**机制/几何**。

---

## 1. Thinking Machines,《On-Policy Distillation》(blog, 2025)

**设置**：起点 Qwen3-8B（已 post-train，有 IF 能力）。
- *个性化/遗忘实验*：在内部公司文档 + chat 数据（不同配比）上 midtrain，再用 OPD 恢复。评测：IF-eval（行为/格式）、内部 QA（知识）。
- *数学能力实验（另一组）*：AIME'24；off-policy distill / RL / OPD 叠加对比。

**结果**：
| | IF-eval（行为） | 内部 QA（知识） |
|---|---|---|
| baseline | 85% | — |
| SFT 70%文档+30%chat | 79% | — |
| SFT 100%文档 | **45%** | — |
| **OPD 恢复** | **83%** | **41%（保住）** |
- "no weighting maintains original IF-eval via SFT alone"（配比调不回来）。
- 数学：off-policy distill **60%** → RL **68%**（高成本）→ **OPD 74.4%（RL 的 1/10 成本）**。

**知识 vs 格式**：**明确区分**——midtraining 损坏"post-training **behaviors**（IF style）"，与领域知识分离；OPD"recovers behavior **without losing any knowledge**"。

**问题所在（对我们）**：blog、非评审；Qwen3-8B-instruct（非 4B-base）；是"个性化后恢复行为"设定，非我们的 math-CoT OPD-vs-SFT；IF-eval=格式、内部 QA=知识，分解干净但任务不同。→ 支持"OPD 保行为 >> SFT""衰退=行为非知识"，但不能直接搬数字。

---

## 2. nrehiew,《SFT, RL, and OPD Through a Distributional Lens》(blog, 2025)

**设置**：对比 SFT / RL / OPD；teacher 变体（SFT teacher 在 code 上退化、RL teacher 不退化）。任务=minimal editing；遗忘在 **LiveCodeBench**（通用 code）上测。

**结果**（LiveCodeBench，越高=忘得越少）：
| 方法 | LiveCodeBench | minimal-edit pass |
|---|---|---|
| **SFT** | **0.286**（退化） | 0.775 |
| **OPD** | 0.297–0.314（slight） | — |
| **RL** | **0.320**（no meaningful forgetting） | 0.792 |
- **排序 SFT ＜ OPD ＜ RL**。关键：**OPD 即使从退化的 SFT teacher 蒸馏也几乎不忘 → "数据来源（on-policy 采样）很重要，teacher 次要"**。

**机制**：**SFT=Forward KL（mode-covering，牺牲旧模式）；RL=Reverse KL（mode-seeking）；OPD 居中**。衰退=**distribution narrowing**，非知识丢失。

**问题所在**：blog；delta 小（0.286 vs 0.314）；单一 code-edit 设定。→ 但 **KL-方向机制 + "数据源 > teacher"** 对我们理论极有用（可接几何/压缩叙事）。

---

## 3. Retaining by Doing (arXiv 2510.18874, 2025) — 最干净的 on/off-policy 消融

**设置**：Llama-3.1-8B、Qwen2.5-7B（instruct）。新域=MATH。遗忘 benchmark：**MMLU（知识）、AlpacaEval（IF 质量）、WildGuard/WildTeaming（安全）**。
- 方法：off-policy SFT vs on-policy（自生成数据 / PPO-RL / OPD）。
- **关键消融**：**固定数据集+teacher，只变 off-policy(curated) vs on-policy(自生成) 采样** → 隔离"数据分布"与"算法"。

**结果**：
- on-policy 比 off-policy SFT 在 **MMLU & AlpacaEval 上少忘 3–5%**；MATH 相当；安全保留 **+2–4pp**。
- **中心 claim：RL 抗遗忘主要来自 on-policy 数据本身、非算法**。

**知识 vs 格式**：说是**知识损失 + 格式漂移都有**（≠ TM 的"仅格式"）；on-policy 靠"输出更贴 in-distribution"同时缓解两者。

**问题所在**：最强的因果隔离（数据源独立）；但 7-8B instruct，非 4B-base；"forgetting 是知识+格式"与 TM 略有出入（Theory 要调和：可能任务/量级依赖）。

---

## 4. Self-Distillation Enables Continual Learning (arXiv 2601.19897, Shenfeld & Damani, 2026)

**设置**：Qwen2.5-7B-Instruct 主；scaling 3B/7B/14B；Olmo-3-7B-Think（推理）。任务：技能学习（Science QA/SciKnowEval、Tool Use/ToolAlpaca、Medical/HuatuoGPT-o1）、知识获取（2025 灾害 Wiki ~200K token）。保留 benchmark：HellaSwag/TruthfulQA/MMLU/IFEval/WinoGrande/HumanEval。多任务顺序学习。
- 方法：**SDFT（on-policy 自蒸馏，demonstration-conditioned teacher）** vs SFT vs DFT(importance-sampling 近似 on-policy) vs Re-invoke vs CPT vs Oracle RAG。

**结果**：
- *知识获取（Table 1）*：SFT strict **80%**/lenient 95%/**OOD 80%**；**SDFT 89%/100%/OOD 98%**；Oracle RAG 91%/100%/100%。→ **SDFT 的 OOD 98% vs SFT 80%（泛化远好）**，接近 Oracle。
- *技能学习+保留（Table 5，new / 先验保留）*：Science SDFT 70.2/64.5 vs SFT 66.2/**53.4**；Tool 70.6/65.4 vs 63.2/56.0；Medical 40.2/65.4 vs 35.5/60.2。→ **SDFT 在 new-task 和 retention 上双赢**。
- *多任务（Fig 3）*：SFT 学下一任务时前任务震荡退化；SDFT 稳定。
- *推理（Table 2，Olmo-3-7B-Think，answer-only 监督）*：base 31.2%（4612 tok）→ +SFT **23.5%（3273 tok，推理坍缩）** → +SDFT **43.7%（4180 tok，保住推理）**。
- **机制**：reverse KL $D_{KL}(\pi_\theta(\cdot|x)\,\|\,\pi(\cdot|x,c))$；demonstration-conditioned teacher 距 base **0.68 nats** vs SFT 模型 **1.26 nats** → SDFT 更贴 base 分布。

**问题所在**：强论文（Shenfeld=RL's Razor 作者）；但**self-distillation**（teacher=自身条件化 demo）≠ teacher-OPD；7B instruct 非 4B-base。→ **reverse-KL + 距 base 更近（0.68 vs 1.26 nats）机制直接支持我们"OPD 保分布/压缩"叙事**，且推理坍缩(格式)+OOD 98vs80 是"格式/泛化非知识"的强证据。KL-nats 距离 ≈ 我们的 drift-from-θ₀。

---

## 5. Revisiting Catastrophic Forgetting in LLM Tuning (arXiv 2406.04836, 2024) — ⚠️更正：flatness/SAM 论文

**设置**：TinyLlama-1.1B、Llama2-7B/13B。FT 数据：Alpaca/ShareGPT/MetaMathQA/Open-Platypus。遗忘：MMLU（知识）+ SuperGLUE/HellaSwag 等。

**结果/发现**：**loss landscape 越 sharp → 遗忘越多**（MMLU 随任务分歧 40.53→33.46→23.31）。提 **SAM（sharpness-aware minimization）** 缓解：Llama2-7B **+7.01** AVG，随规模增大（13B +9.78）；优于 Wise-FT(+0.97)/Rehearsal(+3.02)。

**问题所在（更正）**：这是 **loss-landscape flatness / SAM 缓解**论文，**不做知识 vs 格式分解**（general 分数统一看）。→ 与我们的 **activation 几何**不同（它是**参数损失面几何**）；作为"遗忘↔几何"另一支 + SAM mitigation baseline 有用，但**不是 Q5（知识 vs 格式）的证据来源**。我 earlier 的归类已纠正。

---

## 综合：五问逐答 + 给 Theory 的问题清单

| 问 | 答 | 主证据 |
|---|---|---|
| Q1 OPD OOD 衰退程度有讨论吗 | 有 | TM/nrehiew/Retaining/Self-Distillation |
| Q2 OPD vs SFT | **OPD 忘得远少于 SFT** | TM(83 vs 45 IF)、nrehiew(0.31 vs 0.29)、Self-Distill(retention 64.5 vs 53.4) |
| Q3 OPD vs off-policy KD | **OPD 更少**（on-policy 是抗遗忘因子） | Retaining 消融(on>off 3-5%)、nrehiew(数据源>teacher) |
| Q4 OPD vs RL | **RL 略优，OPD 贴近 RL** | nrehiew(SFT<OPD<RL)；TM(OPD 继承 RL 抗遗忘、1/10 成本、且能教知识) |
| Q5 知识 vs 格式 | **主要格式/行为/分布，知识大体保住** | TM(IF↓知识保)、nrehiew(分布收窄)、Self-Distill(推理坍缩 vs 保) |

**给 Theory 的四个"问题所在"（须表态）**：
1. **"OPD 保 OOD 好"已被确立** → 本项目**不能当新颖**；新颖=机制/几何（ER + dip-recovery + 白化谱 + 方向）。这是硬约束。
2. **我们 C08 测的 MMLU-Pro 是知识轴**，但文献说衰退主要是**格式/行为**。→ 要么(a) 我们的 MMLU-Pro"OOD-drop"是**输出控制/格式中介**（C09 output-control gate 正在查），要么(b) 4B 上确实动知识。**Theory 必须明确 OOD 轴测的是知识退化还是格式退化**，否则 claim 会被 reviewer 攻击。建议补**格式轴（IFEval）+ 知识轴（MMLU/闭卷QA）分离**看。
3. **KL 方向机制现成可用**：SFT=Forward-KL(mode-covering→侵蚀旧模式) vs OPD=on-policy/Reverse-KL(mode-seeking→保模式)；Self-Distill 给了"距 base 0.68 vs 1.26 nats"的量化 ↔ 我们的 drift-from-θ₀。**可把几何压缩叙事接到 KL-方向/分布收窄上**。
4. **"on-policy 是驱动"有张力**：Retaining/nrehiew 说 on-policy 数据是抗遗忘主因；但 2606.07082 的 control 说 off-policy rollout 仍保 weight-rank（on-policy 非 weight 几何驱动）。**Theory 需调和"行为抗遗忘由 on-policy 驱动"vs"weight 几何非 on-policy 驱动"**——可能 activation 几何与 weight 几何的驱动不同（这本身是我们的护城河点）。

**登记**：TM blog、nrehiew blog、Retaining(2510.18874,已在库)、Self-Distillation(2601.19897)、Revisiting-CF(2406.04836,flatness/SAM)、Instruction Vector(2406.12227)、When Does Online Imitation Help(2606.30445)、Superficial Alignment(LIMA)、Linearly Decoding Refused Knowledge(2507.00239)、Limitations of Instruction Tuning(2402.05119)。

---

## 补充精读（2026-07-07 第二轮，本地 PDF + 网上）

### A. Self-Distillation 详细分测试数据（本地 2601.19897v1.pdf 精读）
- 作者 Shenfeld/Damani/Hübotter/Agrawal（MIT/ETH）。方法 SDFT：teacher=同模型条件化(query+demo)，student=仅 query；**reverse KL $D_{KL}(\pi_\theta(\cdot|x)\|\pi(\cdot|x,c))$（Eq1，明确 reverse，设计选择）**；teacher=student 的 EMA。inverse-RL 解释。
- **Table 1**（知识获取 strict/lenient/OOD）：Base 0/0/0｜CPT 9/37/7｜SFT **80/95/80**｜SDFT **89/100/98**｜Oracle-RAG 91/100/100。
- **Table 2**（Olmo-3-7B-Think，answer-only）：base 31.2%(4612tok)→SFT **23.5%(3273tok,推理坍缩)**→SDFT **43.7%(4180tok)**。
- **Table 5**（技能 new/保留）：Science 70.2/64.5 vs SFT 66.2/53.4；Tool 70.6/65.4 vs 63.2/56.0；Medical 40.2/65.4 vs 35.5/60.2。先验能力=6 benchmark 均值（分项在附录）。
- KL 距 base：SFT 1.26 nats、teacher 0.68 nats。
- **⚠️ Fig5 左（SDFT−SFT 按规模）：3B −3.3（SDFT 反而更差）/ 7B +4.0 / 14B +6.9**。本项目 4B 在临界带——写入 Theory 风险。

### B. ⚠️ KL 方向更正（用户质疑正确，关键）
"OPD=reverse KL" 是 conflation。**KL 方向 ⊥ on-policy**（on-policy=数据源；KL 方向=损失几何）。
- Self-Distillation=reverse（**设计选择**，非 OPD 天生）；GKD=广义 JSD 可调；RL=reverse-ish；**本项目 verl OPD 默认=top-k FORWARD KL（forward_kl_topk）**，另有 PG-OPD(k1/k3=reverse-ish)。
- **关键 fork：须核实 Cycle 08 OPD 的 `loss_mode`。** 若 forward_kl_topk → 本项目 OPD 与 SFT **同为 forward KL**，差别仅在 **on-policy 数据 + top-k**，**"reverse-KL 保模式"机制不适用、不可借用**；反利好护城河（优势须归 on-policy+top-k+activation 几何）。若 PG k1/k3 → reverse-ish，机制可用。**Theory 必须先定这个。**

### C. 知识 vs 格式（针对性搜索 + Instruction Vector 精读）——共识强
- **Instruction Vector (2406.12227，LLaMA2-7B-Chat/LoRA)**：拆知识概率 P(y|x) vs 指令概率 P(y^c|c,x)；微调后**指令准确率 −10.24、知识准确率 +1.93**；IV 稳定(cos~0.95) → 表示未擦除，是**新 head 压制原计算图**；**重注入 IV 恢复 0.03→0.47** → 遗忘=抑制非丢失。
- 佐证：Superficial Alignment(LIMA，知识来自预训练/后训练塑格式)、Linearly Decoding Refused Knowledge(2507.00239，压制生成访问但知识可线性解码)、Limitations of Instruction Tuning(2402.05119，SFT 学表层格式非知识逻辑)。
- **结论：遗忘=格式/指令-访问退化，参数知识大体保住。**
- **重磅含义（给 Theory）**：C08 的 MMLU-Pro 是知识轴，但掉的是格式/访问 → 你的"OPD 保 MMLU-Pro"很可能是**"保住表达知识的格式/指令-访问"**，非保知识本身。**强力支持 C09 output-control gate**；精修几何链为 **压缩→格式/分布收窄→指令-访问丢失→表观 OOD-drop（压缩关格式非知识）**。claim 须写成"保表达格式"而非"保知识"。**建议分离 IFEval(格式轴)+闭卷知识探针(知识轴)**。

### D. When Does Online Imitation Help (2606.30445，realizability 理论)
- Thm1 **非可实现**（expert 不在学生函数类）：on-policy(OPD) 比 SFT **指数级优势、随 horizon 增长**（compounding error）。Thm2 **可实现**：优势**消失**，SFT 足够。
- → **OPD 优于 SFT 是 setting-dependent**：长 horizon + 非可实现才有优势。本项目（long-CoT 长 horizon、teacher-32B≫4B-student 非可实现）**正落在"OPD 该有优势"区间** → 给 OPD 优势一个理论依据 + caveat（可实现则无优势）。
