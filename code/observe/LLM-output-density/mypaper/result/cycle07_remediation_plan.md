# Cycle 07 Remediation & Re-test Plan

> ## ⛔ SUPERSEDED (2026-06-27) — 本计划的"重测"前提已被推翻，保留作 provenance
> Code 对话的 cap-pilot（`cap_pilot_findings_REVISION.md`）证明 **as-run math acc 是 cap-robust 的、近似有效，无需重跑**：
> scorer 已能从截断响应里抽出 boxed（step_080 截断 396 个中 209 个判对），as-run 0.620@4096 ≈ pilot 0.617@24576。
> 我原先 §1/§2 的"截断=假阴性、大 cap 能恢复真 acc"是**错的**；§4 的"用干净 acc 重算相关"也不需要（as-run acc 有效，相关成立）。
> 真正的信号是 **非终止（non-termination）**，已从现有 samples 出 `nontermination_trajectory.csv`，0 GPU。
> **唯一仍有效的是本文件的隔离/hand-off 结构**；具体结论以 `cap_pilot_findings_REVISION.md` 和后续 result/ 叙事为准。

```yaml
artifact_type: remediation_plan
cycle: cycle_07_base_sft_trajectory
status: ⛔ SUPERSEDED by cap_pilot_findings_REVISION.md (no re-run needed)
authored_by: Result Analysis conversation (2026-06-27)
baseline_principle: |
  以 as-run 为基准真实方案。run 与 exp 不一致处，一律以 run 为准（那是用户与 coder
  根据实操讨论后的真实方案），不视为偏离、不改回 exp。本次重测只针对一个**数据质量问题**
  ——math 生成轴的 4096/16384 截断让 boxed 缺失被误判、数字不可信——只动 max_token，
  其余 as-run 值全部保留。
write_scope_note: |
  本文件由 Result 对话产出，落在 result/（我的写入范围）。它是"协议问题待同步"标记，
  不直接改 exp/ 或 code/。用户将自行把本计划复制进 exp/ 并交 Code 对话执行。
decision: HOLD cycle07 result/ 正式叙事，先按本计划重测/重算 math 轴，再写 result/。
```

> 本轮决定（QA_cycle07_result.md 用户确认）：Q1=重测+pilot 前置，Q2=整轮 hold 先出本计划，
> Q3=OverlapLift 留 cycle08 同口径判。

---

## 1. 唯一需要重测的原因：math 生成轴截断（数据质量问题）

as-run 的 math eval `max_tokens math500/numina=4096, aime24=16384`。模型 step_40+ 学会长
think-CoT（math500 mean resp_len 670→3762），响应被切断 → 末位/boxed 抽取在截断响应上判错。
这是**数据不可信**，与 exp 写什么无关。

**原始 samples 截断审计（`finish=="length"`）**：

| task | step_20 trunc | step_80 trunc | recovery trunc | 完成样本 acc |
|---|---|---|---|---|
| math500 | 6.8% | **79.2%** | 47–55% | → **0.98** |
| numina (N=256) | 4.7% | **85.2%** | 56–59% | → 0.81 |
| aime24 (N=30) | 16.7% | **86.7%** | 87–93% | N=2–4（失效）|

**去截断 math500 轨迹（仅完成样本，方向性指示，非合法修正值）**：

```
step:        0     5     10    20↓   40    80    160   320   480   624
as-run:    .636  .656  .642  .572  .592  .620  .670  .698  .660  .680
完成样本:  .662  .682  .675  .594  .673  .971  .987  .992  .976  .980
trunc%:     6%    6%    7%    7%   16%   79%   55%   47%   50%   50%
```

**影响**：A07 名义 PASS 但报告值被截断压低；B07 dip 真实（step_20 在低截断区）但 recovery 形状被
完成率天花板扭曲；geometry↔capability 相关用被污染 acc 算得，需重算。（C07 OOD-lite 是 loglikelihood，
不截断，不受影响。）

> 注：完成样本 acc 会上偏（短题更易完成），不能当修正值。正确做法是 pilot 定 cap 后重测。

---

## 2. Pilot：定 max_token（重测前必做）

**为什么必须 pilot**（贯彻用户原则"不严重抬高截断率前提下尽量压低 max_token"）：当前完成样本 p95 已达
~3775–4092（贴住 4096），step_80 有 396/500 截在 4096 → 真实长度分布尾巴在 4096 之外**完全不可见**，
只有高 cap 探针能揭示，据此选最小够用 cap。

| 项 | 取值 |
|---|---|
| 探针 checkpoint | `step_080`（最长、截断最重）+ `step_624`（收敛行为）|
| 探针任务 | math500 子集 N≈150–200；aime24 全量 N=30（aime 单列）|
| 探针 max_token | 一个足够大的上限（如 32768）先把分布完全解censor |
| 采样 | temp=0.6 top_p=0.9（as-run 一致），固定并记录 seed 便于复跑 |
| 量取 | 解censor 后 resp_len 分布 p50/p90/p95/p99；候选 cap {8k,12k,16k} 下的截断率 |

**定 cap 规则**：每任务取使截断率回落到 **≤ ~6–8%**（对齐早期干净区 step_0–20 的 ~6%）的最小 cap。
经验预判：math500/numina 可能 8k–16k 达标；**aime24 大概率需更大 cap**（as-run 16384 时仍 87% 截断）。
具体由 pilot 实测定，不预设。

---

## 3. 重测：math 生成轴（10 ckpt × 3 task），只改 cap，其余保持 as-run

| 任务 | N（as-run，保留）| max_token | 说明 |
|---|---|---|---|
| math500 | 500 | pilot 定值 | — |
| numina | **256（as-run，保留）** | pilot 定值 | 次要退化检查轴，SE±3.1% 足够；不改回 exp 的 1024 |
| aime24 | 30 | pilot 定值（大概率较大）| — |

要求：
- 采样 temp=0.6 top_p=0.9（as-run 一致）；**记录 generation seed** 便于复跑（as-run 未记，重测顺手补上）。
- **截断率（finish=="length" 占比）作为一等列**写进结果表（像 RESULTS_06 那样），用于**验证新 cap 达标**——
  这是本轮缺的，补上后下次能自查。
- 早期 ckpt（0–40）响应短、遇 EOS 即停，高 cap 不增成本；成本集中在 step_80+，可接受。

**产出**：含截断列的 `trajectory_scores.csv`、更新 `trajectory_math500.png` / `response_length.png`、
A07/B07 用干净数字重判。

---

## 4. 仅重算（不重跑模型）

- **geometry↔capability 相关**：用 §3 干净 MATH500 acc 重算 Pearson/Spearman vs
  {effective_rank, xs_log_spectrum_gap, drift_from_base, spectral_gap}。几何 CSV 不动。
  预期 step_20 三重共位幸存，step_80–624 段相关方向可能变，需如实重述。
- **`overlap_lift_trajectory.png`**：从现有 `geometry/principal_layer_{9,18,27}_*.csv` 出图。

---

## 5. as-run 有效，无需任何动作（不改回 exp、不重跑）

- **OOD-lite 分数**：gpqa（N=198）、mmlu_pro（**N=1400，as-run**，`--limit 100/class ×14`）。loglikelihood
  不截断，数值有效。以 N=1400 记录为真实协议（不改回 exp 的 500）。诚实读数：GPQA +0.005（噪声）、
  MMLU-Pro −0.029（~2SE 轻微真跌）。
- **几何度量本身**：effective_rank / drift_from_base / xs_log_spectrum_gap / spectral_gap / OverlapLift
  数值、UAngle/VAngle——基于权重/激活，不受截断影响，有效。
- **step_20 干净凹陷 + 几何三重共位**：落在 6.8% 低截断区，de-truncation 后仍成立。
- **principalEvidence 文件结构**：按层存（`principal_layer_{9,18,27}_*.csv`，含全 step 行），as-run，数据完整。

---

## 6. 跨 cycle / 遗留

- **OverlapLift 解读**：result/ 只记事实——"4B LoRA think-SFT 的 OverlapLift 沿轨迹 0.48→0.74 单调升、
  全程 <1；UAngle/VAngle≈0°"——并标 **LoRA 同口径 caveat**（低秩约束会压低 overlap，RandomExpected 基线
  只归一化了 update_count、未归一化低秩结构，当前无全参/随机/OPD 对照不能判 on/off-principal）。
  **结论 defer cycle08** 与 OPD arm 同口径比较。不写"推翻 cycle06 预期"。
- **C06 copyback**：cycle07 eval 已完成，可把 cycle06 的 `geometry_metrics.csv` + `principal_evidence.csv`
  从源盘拷到 `local_experiment_results/cycle_06_.../run_01/`，关闭 C06 PARTIAL。（非关键路径。）

---

## 7. 执行顺序与 hand-off

```
本计划（result/，已出）
  └─→ 用户复制进 exp/（pilot+重测协议、as-run 值、截断审计列、seed 记录）
      └─→ Code 对话：§2 pilot → 定 cap → §3 重测 + §4 重算 + §6 copyback
          └─→ Result 对话（回到本角色）：用干净数字写 result/ 正式叙事，关闭 cycle07
```

**Result 侧现在不写正式 current_picture / claims 的 math 部分**（Q2=B hold）。本计划是唯一交付物。
