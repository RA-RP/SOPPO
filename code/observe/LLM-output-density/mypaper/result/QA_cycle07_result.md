# QA: Cycle 07 Result Analysis

```yaml
artifact_type: result_phase_qa
cycle: cycle_07_base_sft_trajectory
phase: result_analysis
status: pending_user_answers
date: 2026-06-27
role: Result Analysis (write scope = result/ only)
```

读取来源：`local_experiment_results/cycle_07_base_sft_trajectory/run_01/`（RESULTS_07.md、
trajectory_scores.csv、response_length_trajectory.csv、geometry/*、run_provenance.json、
raw eval/*/math500|numina|aime24/*_samples.jsonl）、`exp/current_experiment_design.md`、
`exp/current_acceptance_criteria.md`。注：`code/code_evolution.md` 尚无 Cycle 07 handin 节。

---

## 🔴 中心问题：生成式数学轴被 4096 截断伪影污染（HIGH，第三次同类）

**事实**：design 规定 math eval `max_gen=32768`（"needed for think-format"），实际执行
`max_tokens math500/numina=4096, aime24=16384`（provenance）。模型 step_40+ 学会长 think-CoT
（math500 mean resp_len 670→3762，逼近 4096），触发大规模截断；末位/boxed 抽取在截断响应上判错。

**直接审计原始 samples（`finish=="length"` 字段）**：

| task | step_20 trunc | step_80 trunc | recovery trunc | non-trunc acc (完成样本) |
|---|---|---|---|---|
| math500 | 6.8% | **79.2%** | 47–55% | → **0.98** |
| numina (N=256) | 4.7% | **85.2%** | 56–59% | → 0.81 |
| aime24 (N=30) | 16.7% | **86.7%** | 87–93% | N=2–4（失效）|

**去截断 math500 轨迹（仅完成样本，作为方向性指示，非最终修正值）**：

```
step:        0     5     10    20↓   40    80    160   320   480   624
reported:  .636  .656  .642  .572  .592  .620  .670  .698  .660  .680
non-trunc: .662  .682  .675  .594  .673  .971  .987  .992  .976  .980
trunc%:     6%    6%    7%    7%   16%   79%   55%   47%   50%   50%
```

**对四个产出的影响**：
1. **A07 feasibility**：名义 PASS（0.680 > base+SE 0.658），但 0.680 是截断压低值；真实方向更强（完成 acc ~0.98）。门限数字本身建立在损坏测量上。
2. **B07 dip-recovery**：step_20 凹陷**真实**（低截断区，non-trunc 0.594 为早期最低）；但 recovery "平台 ~0.68" 是完成率天花板而非能力平台，恢复幅度/形状被扭曲。
3. **geometry↔capability 相关（本轮主打机制发现）**：Spearman −0.71/+0.77 用被污染 acc（step_80–624）算得 → 需用去截断 acc 重算。step_20 三重共位因落干净区**幸存**。
4. **C07 OOD-lite**：loglikelihood 不截断，**不受影响**。

**注意**：non-trunc acc 不是合法的修正值（丢弃截断样本会上偏：简单/短题更易完成）。正确修法是
**用 ≥16384（或 design 的 32768）token 重测 math500/numina/aime24**。

---

## 我核对的 gate 判定（基于当前数据）

| Gate | RESULTS_07 报 | 我的核验 |
|---|---|---|
| A07 feasibility | PASS (0.680>0.658) | 数字建立在截断测量上；方向 PASS 且真实更强，但报告值需重测后定稿 |
| B07 dip-recovery | FULL PASS | dip 真实；recovery 形状被截断扭曲，需重测后重画 |
| C07 OOD-lite | "pass"（GPQA +0.005）| 诚实读数=无真实增益：GPQA +0.005（≪1SE 噪声）、MMLU-Pro −0.029（~2SE 轻微真跌）；实际 N=1400 非 500 |

## 次要发现 / 规格偏差

- **OverlapLift 全程 < 1**（down_proj 0.48→0.74 单调升、不过 1；attention 0.04–0.28）。与 cycle06
  设计预期"SFT on-principal、OverlapLift >> 1"**相反**，更像 RLVR/off-principal。RESULTS_07 未报此数字。
  → 直接影响 cycle08 H-OPD 框架（"测 OPD 是否比 SFT 更像 RLVR"的前提被动摇）。
- numina 实测 N=256（design 1024）；MMLU-Pro 实测 N=1400（design 500，RESULTS_07 误标 500）。
- principalEvidence 按层存（`principal_layer_{9,18,27}_*.csv`）而非按步（design 写 `_step_{NNN}`）；数据完整含全 step 行。
- `overlap_lift_trajectory.png` 缺失（figures 仅 3 张）。
- temp=0.6 采样未见记录 generation seed（invalidation 条件之一，软标）。

---

## 待用户拍板的 3 个问题

### Q1. 生成式数学轴（A07/B07 主线 + geometry 相关）怎么处理？
- (A) **判 math 轴为"测量无效、需重测"**：发起 code-side 重测（math500/numina/aime24 用 ≥16384
  token），重测前不在 result/ 写任何 math 定量结论/门限数字/geometry-capability 相关；先写不受影响的部分。【推荐——与项目"截断/floor 必须重测"既有规则一致】
- (B) 用去截断（完成样本）acc 作 indicative 修正写入 result/，重测作 follow-up。
- (C) 按 RESULTS_07 原样写，加重 caveat。

我倾向是a，之前改小阶段是因为实验时测试发现不需要那么长的max token，max token小很多实际上不怎么改变截断率，同时能大幅度提升测试速度。我现在还是希望能够和之前保持一致的想法，在不严重影响截断率的情况下，尽量减少测试的截断长度。这需要在正式确认截断长度前，先预热测试一下大致多少的max token是合理的。

### Q2. 本轮 result/ 现在写哪些（不阻塞的部分）？
- (A) 现在写：C07 OOD-lite（无真实增益 + MMLU-Pro 轻跌）、OverlapLift<1 轨迹、step_20 干净凹陷
  与几何三重共位（定性）、feasibility 方向性结论（"4B+think-CoT 可行,真实恢复 ~0.98,待重测定稿"）；
  math 定量轴标 ⚠️ 待重测。【推荐】
- (B) 整轮 hold,等 math 重测完一起写。

B，我现在需要你根据现有的所有问题，整理一份补救的重测计划。

### Q3. OverlapLift < 1 与 cycle06 前提冲突,怎么记？
- (A) 在 result/ 记为**confirmatory 几何发现**（"4B LoRA think-format SFT 更新 off-principal,
  OverlapLift 沿轨迹 0.48→0.74 单调升但始终 <1"）,并标注它推翻 cycle06 设计的 on-principal 预期,
  作为 cycle08 H-OPD 的新输入。【推荐】
- (B) 仅记为 pending/indicative,留 cycle08 与 OPD 一起判。

b，这个要具体和后面的OPD一起判断，而且SFT本身就不是on-policy的呀，为什么推翻cycle06的预期。我甚至认为在lora的稀疏条件下，还能有这么高的overlap实际上也是很论文《the path not taken》很符合的呀

---

## 用户回答（2026-06-27 确认）

- **Q1 = A（重测）+ pilot 前置**：判 math 生成轴测量无效需重测，但不直接跳 32768。用户既往经验：之前
  调小 max_token 是因为当时测试发现"小很多也几乎不改变截断率、却大幅提速"——本轮模型学会长 think-CoT 后情况
  变了。原则不变：**在不严重抬高截断率的前提下尽量压低 max_token**。因此正式定 cap 前，先 pilot 预热测出
  合理值。
- **Q2 = B（整轮 hold）**：math 重测完再统一写 result/。本对话交付物改为**补救重测计划**（覆盖所有已发现问题）。
- **Q3 = B（pending/indicative，留 cycle08 与 OPD 同口径判）**。用户更正了我的 over-claim：
  (1) SFT 本就不是 on-policy，不必期待它像 RLVR；(2) LoRA 稀疏/低秩约束本会压低 overlap，能到 0.48–0.74
  且单调升向 1 反而与《The Path Not Taken》的 SFT-on-principal 倾向**相容**，不是反例。
  **采纳**：result/ 不写"推翻 cycle06 预期"，只记 OverlapLift 轨迹事实 + LoRA 同口径 caveat，结论 defer cycle08。
  （技术注：RandomExpected 基线已对 update_count 归一化，但未对低秩结构归一化——这正是需 OPD/全参或随机基线
  对照才能定论的原因，支持 defer。）

→ 交付：见 `result/cycle07_remediation_plan.md`（本对话产出，标 ⚠️ 协议问题待同步，须由 Experiment Design
对话内化到 exp/ + code_brief 后执行）。result/ 正式叙事 hold 到 math 重测完成。

---

## 解决（2026-06-27 晚 — Q1/Q2 hold 已解除，无需重跑）

用户在 Code 对话跑了 **cap-pilot**（`cap_pilot_findings_REVISION.md`），结论推翻了我"截断=假阴性、大 cap 能恢复真 acc"的前提：
- scorer 已能从截断响应抽 boxed（step_080 截断 396 个中 **209 判对**）；as-run **0.620@4096 ≈ 0.617@24576**（6× cap，acc 不变）→ **as-run math acc 有效、cap-robust，不用重跑**。
- 我那张"完成样本 acc→0.97"是 selection-on-short-problems，不能当修正值（自查时已警示，pilot 坐实）。
- 真正信号是 **非终止（non-termination）**，已从现有 samples 出 `nontermination_trajectory.csv`，0 GPU。

**我的 Result 侧增值**：从 finisher-acc 拆解发现两个信号**错时**——能力凹陷在 step_20（非终止仅 6.8%，干净）、非终止爆发在 step_80（79%，finisher 已恢复 0.97），修正了 REVISION 的"finishers 全程 ~100% 正确/concurrent"。

**Q1/Q2 状态**：hold 条件（等 math 重测）消失 → 已用 as-run 有效数据写完 result/（incremental_log + current_picture + claims_allowed）。`cycle07_remediation_plan.md` 标 ⛔ SUPERSEDED 保留作 provenance。
**Q3 状态**：按答复执行——OverlapLift 只记事实 + LoRA caveat，结论 defer cycle08。
