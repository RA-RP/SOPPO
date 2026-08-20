# QA: Cycle 06 Result Analysis

```yaml
artifact_type: result_phase_qa
cycle: cycle_06_sft_feasibility_and_degradation
phase: result_analysis
status: pending_user_answers
date: 2026-06-24
role: Result Analysis (write scope = result/ only)
```

读取来源：`local_experiment_results/cycle_06_.../run_01/RESULTS_06.md`、3 个 FINDING_06、
`code/code_evolution.md` handin 节、`exp/current_experiment_design.md`、
`exp/current_acceptance_criteria.md`、`result/current_picture.md`。

---

## 我先做的 gate 判定（待用户确认）

| Gate | 判定 | 依据 |
|---|---|---|
| A06 instruct feasibility | **FAIL** | instruct_base numina=0.533，阈值=base+1SE=0.549；4 个 sft arm numina 全 ≤0.411，无增益反而退化 |
| A06b base feasibility | **UNDETERMINED** | base numina 全是截断伪影（⚠️0.03–0.06，66% trunc），非能力测量，需 cycle07 stop-fix 重测 |
| B06 OOD-lite | **No degradation（preserved within noise）** | mmlu/tqa/wino/arc 全 arm <1.5% spread，与 cycle04/05 一致 |
| C06 engineering | **INCOMPLETE（in result workspace）** | geometry_metrics.csv / principal_evidence.csv 未 copyback，OverlapLift 无法分析 |

## 核心 narrative（待确认）

无 think 的简洁 SFT 在两个系列上都**不增加数学能力**：instruct 侧直接退化 numina/math500
（A06 fail）；base 侧 GSM8K/numina 的"SFT 增益梯度"是差异化去截断伪影，stop 修正后 base GSM8K
**变平 ~0.66**、SFT 中性（FINDING_06_base_gsm8k_stop）。think probe 进一步显示 no-think SFT
**摧毁 thinking 的功能**（形式在、功能失，instruct_base think +0.13 vs SFT −0.02~−0.06）。
OOD-lite 全程保住。→ 与前几轮"no-think concise SFT 不增能力"的故事一致并强化。

---

## 待用户拍板的 3 个问题

### Q1. A06b（base feasibility）怎么报？
base numina 是截断伪影、无法判 gate。
- (A) 报 **UNDETERMINED + 用 stop 修正后的 base GSM8K（flat ~0.66, SFT 中性）作定性代理**，结论"base 侧同样无 headroom"，numina 留 cycle07 重测。【推荐】
- (B) 严格只报 UNDETERMINED，不做任何 base feasibility 定性判断，全部 defer。

### Q2. 几何 / principalEvidence（C06）怎么处理？
geometry_metrics.csv 和 principal_evidence.csv 没 copyback、未分析。
- (A) 本轮 result 标 **C06 partial / geometry pending**，几何结论 defer cycle07，不阻塞 result 写作。【推荐，符合"快进 cycle07"】
- (B) 先回代码侧把两个 CSV 拷回再写 result（阻塞）。

### Q3. 本轮 allowed claim 的最强表述？
- (A) **"No-think concise SFT 在 instruct 和 base 两个系列、n∈{512..4096} 上均无可测 NuminaMath
  增益；instruct 侧反而退化 math 推理，并特异性摧毁 thinking 功能；OOD-lite 保住。"**【推荐】
- (B) 更保守：只 claim instruct A06 fail + OOD-lite preserved，think/base 部分标 indicative-only。

---

## 用户回答（2026-06-24 确认）

- **Q1 = 纯 UNDETERMINED**：base feasibility 不做任何定性结论，不用 GSM8K 作代理，base 系列 numina/gsm8k/numina-headroom 全部 defer 到 cycle07 stop-fix 重测。
- **Q2 = 标 pending，不阻塞**：result 标 C06 partial / geometry pending，几何与 principalEvidence 结论 defer cycle07，现在直接写 result。
- **Q3 = 保守表述**：只 claim instruct A06 fail + OOD-lite preserved within noise；think probe 与 base-stop 修正部分标 indicative-only（n=100 单 seed / 待 cycle07 重测），不写入 confirmatory claim。

### 内化后的本轮 gate 结论
| Gate | 最终判定 |
|---|---|
| A06 instruct | **FAIL**（confirmatory）：sft_n* numina 全 ≤ base+1SE，无增益且退化 math |
| A06b base | **UNDETERMINED**：numina 截断伪影，defer cycle07 |
| B06 OOD-lite | **No degradation, preserved within noise**（confirmatory） |
| C06 engineering | **PARTIAL**：eval 全 + provenance 全；principal_evidence/geometry 未 copyback → pending |
