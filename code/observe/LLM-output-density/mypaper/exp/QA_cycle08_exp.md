# QA: Cycle 08 — Experiment Design Phase

```yaml
phase: experiment_design
cycle: cycle_08_h_opd_vs_sft_comparison
status: discussion_complete
created: 2026-06-29
discussion_closed: 2026-06-29
internalize_to_after_discussion:
  - exp/current_experiment_design.md
  - exp/current_acceptance_criteria.md
  - exp/current_code_handoff.md
  - code/current_code_brief.md
read_scope_confirmed:
  - readme.md
  - result/current_picture.md
  - result/claims_allowed.md
  - next_cycle/open_questions.md
  - next_cycle/current_cycle08_intent.md
  - theory/current_theory_update.md
  - theory/geometry_metric_definitions.md
  - related_work/QA_cycle08_related_work.md
  - related_work/source_matrix.md
  - related_work/source_notes/cycle08_opd_teacher_coldstart_tooling.md
  - related_work/source_notes/cycle08_capability_degradation_benchmarks.md
  - exp/README.md
  - exp/current_experiment_design.md
  - exp/current_acceptance_criteria.md
  - exp/experimental_methodology/model_and_data.md
  - exp/experimental_methodology/training_protocol.md
  - exp/experimental_methodology/evaluation_protocol.md
  - exp/experimental_methodology/geometry_protocol.md
  - local_experiment_results/cycle_07_base_sft_trajectory/run_01/RESULTS_07.md
write_scope:
  - exp/
  - code/current_code_brief.md
```

本 QA 文件用于 Cycle 08 Experiment Design 阶段开始前的审查。按照工作区规则，本阶段可以写 `exp/`，并且在实验设计确认后写 `code/current_code_brief.md` 作为 Code Execution 对话的任务说明；其他目录只读。

---

## 角色与边界确认

当前对话角色：**Experiment Design / exp**。

允许写入：

- `mypaper/exp/`
- `mypaper/code/current_code_brief.md`

不允许写入：

- `theory/`
- `related_work/`
- `result/`
- `next_cycle/`
- `research_cycles/`
- `local_experiment_results/`
- `paper_drafts/`

本阶段职责：

- 把 Cycle 08 的 H-OPD vs SFT 比较写成可执行、可判定、可交给 coder 的实验设计。
- 预注册 teacher、lambda、cold-start / no-cold-start 决策流程、checkpoint grid、eval suite、geometry suite、gate 阈值和 artifact 要求。
- 更新 `current_experiment_design.md`、`current_acceptance_criteria.md`、`current_code_handoff.md`，并写 `code/current_code_brief.md`。

本阶段不做：

- 不运行训练或评测。
- 不修改 theory / related_work / result 的结论。
- 不把尚未执行的 OPD arm 写成结果。
- 不把 supplementary benchmark 事后改成 primary gate。

---

## 当前 Cycle 判定

`exp/current_experiment_design.md` 和 `exp/current_acceptance_criteria.md` 仍是 Cycle 07 工作面；但 `next_cycle/current_cycle08_intent.md`、`theory/current_theory_update.md`、`related_work/QA_cycle08_related_work.md` 都已确认当前进入：

```text
cycle_08_h_opd_vs_sft_comparison
```

因此本阶段应覆盖当前 exp 工作面为 Cycle 08，并在代码 handoff 中替换 Cycle 07 的执行任务。

---

## 已读上下文摘要

Cycle 07 已经建立 Cycle 08 的 SFT reference trajectory：

- Model/data/format：Qwen3-4B-Base + Math-CoT-20k + think-format LoRA。
- Checkpoint grid：`step_000, 005, 010, 020, 040, 080, 160, 320, 480, 624`。
- MATH500：base 0.636，dip step_20 = 0.572，recover step_160 = 0.670，peak step_320 = 0.698，final step_624 = 0.680。
- step_20 dip 是 genuine capability dip：finisher acc 0.594，non-term 仅 6.8%。
- 后续 step_40-160 有 non-termination transient，final as-run acc 仍是有效 end-to-end task-success。
- MMLU-Pro final 相对 base 约 -0.029（约 -2.2 SE），GPQA-D 近似 flat。
- Geometry：step_20 triple co-location = argmin acc = argmax effective_rank = argmin xs_log_spectrum_gap。
- OverlapLift：SFT arm < 1 且上升；Cycle 08 theory 已允许 **OPD-vs-SFT matched-LoRA relative comparison**，不要求先做随机 LoRA baseline 才能比较两条轨迹。

Cycle 08 的核心实验对象：

- 单个 OPD arm，单个 lambda，不做 sweep。
- 同 model / data / LoRA / checkpoint grid，与 Cycle 07 SFT trajectory 做轨迹级 primary comparison。
- Teacher 根据 related work 建议为 Qwen3-8B-Instruct。
- Gate priority：B08 > C08 > D08 > A08。

---

## Q1：Cycle 08 是否采用“两阶段设计”：Phase 0 probe → Phase 1 main OPD？

**问题：** 用户倾向最纯的 no-cold-start / step_000 OPD，但 related work 警告 raw base → 8B-instruct + think-format teacher 的初始 overlap 可能太低。Experiment Design 是否应预注册：

```text
Phase 0:
  1. 对 step_000 和 Cycle 07 早期 SFT checkpoints 测 teacher-student top-k overlap。
  2. 可选跑一个极短 OPD smoke（几十步以内）看 overlap / loss / generation 是否上升或崩坏。
  3. 只用于决定 cold-start 起点和框架可行性，不作为结果。

Phase 1:
  在预注册的起点上跑唯一正式 OPD arm，并按 10 点 grid 评测。
```

- a. 是。Phase 0 是 Cycle 08 的正式前置设计，用来决定是否能从 step_000 纯 OPD。
- b. 否。直接从 step_000 跑完整 OPD，失败也接受为最纯 H-OPD 测试。
- c. 直接选最小 pre-recovery cold-start，不做 Phase 0。

**Codex 初步建议：** a。这样保留用户想要的 strict test，同时避免 OPD 因 teacher-overlap 太低而为错误原因失败。

**你的回答：**
a

---

## Q2：如果 Phase 0 显示 step_000 overlap 不足，cold-start 起点如何选？

**问题：** Theory 已禁止 recover-point cold-start 作为“avoid dip”证据；related work 建议选择“刚进入 teacher-overlap 区，但仍在 SFT recovery 巩固前”的最早点。Experiment Design 是否设定：

```text
Preferred:
  step_000 if overlap probe passes.

Fallback:
  choose the earliest checkpoint among {step_005, step_010, step_020, step_040, step_080}
  that satisfies overlap viability.

Avoid:
  step_160 only as last-resort conditional test.
  step_320/480/624 not allowed for H-OPD avoid-dip claim.
```

- a. 是。fallback 只在 `000-080` 中选；`160` 需要单独用户确认；`320+` 禁止作为 H-OPD avoid-dip 测试。
- b. 放宽到 `000-160`，只要仍能说明不是从 peak/final 起步。
- c. 不设硬边界，由 coder 根据 overlap 最大的点选。

**Codex 初步建议：** a。`160` 已是 Cycle 07 first-recovery checkpoint，作为 cold-start 会削弱“避开 dip”的解释；最好只作为 emergency redesign，而不是默认 fallback。

**你的回答：**
a.我认为需要在dip之前就结束，最好不要出现dip，这样实验结果比较纯净。同时我认为这可能也是一个可以研究的重点，现在的研究集中在对OPD-SFT的理论发现，但是如果能进一步发现dip和其关系，说不定可以给出一个OPD如何cold-start的理论指导。但是这不是现在需要考虑的，然而你需要将其进行注册为之后可能需要讨论的思想。

---

## Q3：lambda 选择是否预注册为 `lambda=1.0`？

**问题：** Theory 给出的原则是“单 arm 选仍可能保能力的最强 OPD contrast”。用户也认为理想情况是 `lambda=1.0` 满足要求时最能说明 OPD 性质。Experiment Design 是否把正式 OPD arm 设为：

```text
lambda = 1.0
```

并把风险控制交给 Phase 0 overlap / cold-start，而不是降低 lambda？

- a. 是。Cycle 08 主 arm 用 `lambda=1.0`，最大化机制对比。
- b. 用 `lambda=0.5`，优先保 B08 endpoint non-inferiority。
- c. Phase 0 后再二选一，但仍只跑一个正式 arm。

**Codex 初步建议：** a。既然只有一个 arm，`lambda=1.0` 是最清楚的机制测试；如果 Phase 0 已经发现 `lambda=1.0` 完全不可运行，再回到用户处确认是否降为 `0.5`。

**你的回答：**
a

---

## Q4：teacher 是否固定为 Qwen3-8B-Instruct？

**问题：** Related Work Q1 已由用户确认使用 Qwen3-8B-Instruct。Experiment Design 是否将 teacher 预注册为：

```text
teacher = Qwen3-8B-Instruct 同家族模型
```

并在 code handoff 中要求 coder 启动时确认本地路径；若路径不存在则先停止并回报，而不是自动改用 14B/32B/4B teacher？

- a. 是。teacher 固定为 Qwen3-8B-Instruct；路径缺失时回报，不自动替换。
- b. 允许 coder 在 8B 不存在时退回 Qwen3-4B-Instruct。
- c. 允许 coder 在 8B 不存在时改用 14B 或 32B。

**Codex 初步建议：** a。teacher 是理论与 related work 共同决定的设计变量，不应在代码阶段悄悄替换。

**你的回答：**
a

---

## Q5：训练框架策略：先 TRL probe，还是直接迁移 verl？

**问题：** Related Work 结论是 TRL/GKD 功能上可跑但不高速，verl async OPD 是高速专用但迁移成本高。Cycle 08 只有一个 OPD arm。Experiment Design 是否采用：

```text
Primary implementation path:
  use the existing TRL / local OPD tooling first.

Required Phase 0 framework probe:
  coder measures short-run throughput, memory, rollout behavior, and top-k KL feasibility.

Escalation:
  if TRL cannot finish the planned arm within acceptable time/memory,
  stop and report before migrating to verl; do not silently rewrite the experiment.
```

- a. 是。先 TRL probe，verl 只作为 blocked fallback。
- b. 直接要求 coder 迁移到 verl async OPD。
- c. 两条都写，让 coder 自由选择。

**Codex 初步建议：** a。Cycle 08 的科学问题已经很大，先减少工程迁移变量；但必须显式记录吞吐 probe，避免训练跑到一半才发现不可承受。

**你的回答：**
b，这个开销是在是比较大了。

---

## Q6：B08 的主判定是否使用 as-run MATH500，finisher acc 作为分解条件？

**问题：** Cycle 07 已确认 as-run MATH500 是有效 end-to-end task-success，non-termination 是真实 output-control 行为，不是单纯评测 artifact。Experiment Design 是否将 B08 写成：

```text
B08.1 final non-inferiority:
  OPD MATH500 as-run at final >= SFT final as-run - 1*SFT_SE
  numeric anchor: 0.680 - 0.0209 ≈ 0.659 (round to 0.660)

B08.2 dip avoidance / mitigation:
  compare the OPD trajectory's minimum against SFT step_20.
  Primary: as-run trajectory should not reproduce a severe step_20-level trough.
  Decomposition: finisher acc and non-term rate must be reported to separate
  capability dip from output-control transient.
```

- a. 是。as-run 是主判定；finisher/non-term 是必报分解，不替代主判定。
- b. B08.2 主要用 finisher acc，因为 step_20 dip 是 genuine capability dip。
- c. 同时要求 as-run 与 finisher 都超过 SFT step_20，才算 pass。

**Codex 初步建议：** a 或 c。最稳妥是 a 作为主 gate，c 作为强 pass 版本：若 OPD as-run 与 finisher 都优于 SFT trough，则 B08 很干净；若二者分歧，报告 mixed 而不是强 pass。

**你的回答：**
a

---

## Q7：C08 是否仍以 MMLU-Pro 为 primary preservation axis？

**问题：** 用户在 Theory QA 中指出 MMLU-Pro 不一定代表完整 OOD；Related Work 已给出更宽的 capability-degradation benchmark 候选。但在未跑 protocol feasibility 之前，是否应保持：

```text
C08 primary:
  MMLU-Pro preservation, all 10 checkpoints, same Cycle 07 protocol.

Supplementary preservation:
  optional Tier 1 cheap MC suite after code feasibility:
  MMLU / HellaSwag / WinoGrande / ARC-Challenge / PIQA / TruthfulQA-MC1.

Rule:
  supplementary suite can strengthen or weaken the preservation story,
  but cannot replace C08 after results are known.
```

- a. 是。MMLU-Pro 保持 primary；Tier 1 作为 supplementary，能跑则跑。
- b. 把 Tier 1 cheap MC suite 升级为 C08 primary，MMLU-Pro 只是其中之一。
- c. Cycle 08 不加任何 supplement，只跑 MMLU-Pro + GPQA-D。

**Codex 初步建议：** a。它尊重 theory 的 caveat，也避免把 Cycle 08 的 primary gate 扩得太散。

**你的回答：**
a

---

## Q8：supplementary Tier 1 若加入，评测范围如何控制成本？

**问题：** 若 Q7 选择加入 Tier 1，OPD 与 SFT 都需要可比较。成本选项：

- a. 全 trajectory：OPD 10 checkpoints + retro-evaluate Cycle 07 SFT 10 checkpoints。
- b. Landmark only：只评 `step_000 / 020 / 160 / 320 / 624`，覆盖 base / dip / recovery / peak / final。
- c. OPD-only：只看 OPD preservation，不和 SFT 做完整比较。

**Codex 初步建议：** b。作为 supplementary，landmark 足以检验“是否只有 MMLU-Pro 特例”，成本也不会压垮主实验。若 cheap enough，coder 可报告能否升级为 a。

**你的回答：**
我不认为是b，由于OPD机制不同，未必能和SFT的轨迹类似，还是现在设计的比较多的checkpoint比较合适。然后我认为cycle08的运行拆分成两部分，先按照cycle07测试方法进行最小的测试，我再根据测试结果结合cycle07的测试结果，选择几个checkpoint进行supplementary的测试。这样成本比较小也比较灵活。

---

## Q9：A08 是否取消“随机 LoRA baseline 必须先解决”的要求？

**问题：** `next_cycle/current_cycle08_intent.md` 仍写着 A08 需要 LoRA-aware reference；但 Cycle 08 Theory 已更新为：

```text
OPD-vs-SFT OverlapLift trajectories are valid relative matched-LoRA comparisons.
A random/full-rank reference is only required for absolute on/off-principal claims.
Principal-angle/subspace-rotation claims defer to Cycle 09.
```

Experiment Design 是否按 theory 最新规则写入：

- a. 是。Cycle 08 A08 只做 OPD-vs-SFT relative OverlapLift；不要求随机 LoRA baseline。
- b. 仍要求 random-LoRA baseline，否则 A08 不判定。
- c. 两者都做：relative comparison + random baseline。

**Codex 初步建议：** a。这样对齐用户在 Theory QA 的意见，也避免把 Cycle 08 变成几何方法学 cycle。

**你的回答：**
c，random baseline作为绝对值尤其意义。

---

## Q10：Geometry artifact 是否沿用 Cycle 07 全量要求？

**问题：** 即使 principal-angle 推迟到 Cycle 09，Cycle 08 仍需要 D08/A08。Experiment Design 是否要求：

```text
Geometry at all 10 checkpoints:
  effective_rank
  spectral_gap / xs_log_spectrum_gap
  drift_from_base
  OverlapLift / principalEvidence

Artifacts:
  keep per-checkpoint geometry CSVs
  keep principalEvidence outputs
  save any UV/sketch files required by the existing principalEvidence pipeline
```

- a. 是。沿用 Cycle 07 全量 geometry/OverlapLift artifact 要求。
- b. 只跑 summary geometry，不保存 UV/sketch。
- c. 只在 landmark checkpoints 跑 geometry。

**Codex 初步建议：** a。D08 是轨迹级 geometry gate，少 checkpoint 会直接削弱“是否避开 triple co-location”的解释。

**你的回答：**
a。但是cycle07不是也没跑UV吗，我认为UV可以延后到cycle09。

---

## Q11：是否把 overlap probe 产物列为必须 copyback artifact？

**问题：** cold-start 是 Cycle 08 的关键解释变量。Experiment Design 是否要求 Code handoff 必须产出：

```text
overlap_probe.csv
  checkpoint
  mean_student_teacher_topk_overlap
  teacher_topk_mass_on_shared_tokens
  student_entropy
  teacher_entropy
  notes / viability verdict

phase0_smoke.md or json
  if short OPD smoke is run:
    command, steps, loss/KL trend, sample generations, failure/success signs
```

- a. 是。Phase 0 artifacts 是必需项，决定 main OPD 起点。
- b. 只需要 coder 口头判断，不保存正式 artifact。
- c. 不做 overlap artifact。

**Codex 初步建议：** a。否则 cold-start 选择无法在 paper / result 阶段追溯。

**你的回答：**
a

---

## Q12：Cycle 08 是否需要重新评测 Cycle 07 SFT reference？

**问题：** Intent 说不要重跑 SFT arm；但若加入新的 supplementary benchmark，可能需要对 Cycle 07 SFT checkpoints 做补充评测。Experiment Design 是否区分：

```text
Core tasks:
  Do not re-run Cycle 07 SFT; use RESULTS_07 and copied artifacts.

New supplementary tasks:
  If selected, evaluate the already-saved SFT checkpoints only on those new tasks
  for comparability, without retraining SFT.
```

- a. 是。核心不重评，新增 supplement 可对 SFT checkpoint 补测。
- b. 完全不碰 Cycle 07 SFT checkpoint。
- c. 重新完整评测 SFT + OPD，保证同一代码版本。

**Codex 初步建议：** a。它保留已验证结果，又允许 supplementary benchmark 有可比 SFT reference。

**你的回答：**
我之前说过了OPD先检测完成之后，根据结果选择几个checkpoint和SFT对应checkpoint，进行supplement的测试，其他不再重复测试。

---

## Q13：正式设计完成后的写入目标是否正确？

**问题：** 用户确认 QA 后，本阶段将写：

```text
exp/current_experiment_design.md
exp/current_acceptance_criteria.md
exp/current_code_handoff.md
code/current_code_brief.md
```

并保留 `exp/QA_cycle08_exp.md` 原文，不写其他目录。

- a. 是。
- b. 还需要同步 `-ch` 中文版。
- c. 还需要追加 research_cycles evolution。

**Codex 初步建议：** a。按当前流程，QA 讨论后先更新工作面和 code brief；`research_cycles/` 通常在 cycle 收尾或明确要求时追加。若你希望本阶段同步 `-ch`，也可以选 b。

**你的回答：**
a

---

## 讨论结论（2026-06-29，已内化到正式文件）

1. **Q1 Phase 0**：采用两阶段设计。先做 overlap/smoke probe，再跑唯一正式 OPD arm。
2. **Q2 cold-start**：优先 strict `step_000`；若不行，只允许自动 fallback 到 dip 前的最小可行 checkpoint（正式设计写为 `step_005` / `step_010`）。`step_020` 是 dip 本身，只能诊断或经用户确认；`step_040+` 不作为自动 H-OPD avoid-dip 起点。另注册 future idea：overlap 与 dip/cold-start 的关系可能成为 Cycle 09 理论问题。
3. **Q3 lambda**：正式 arm 使用 `lambda=1.0`。
4. **Q4 teacher**：teacher 固定为 Qwen3-8B-Instruct；路径缺失时停止回报，不自动替换。
5. **Q5 framework**：用户选择 b，直接以 verl async/on-policy KD 为主实现方向；TRL 不作自动 fallback。
6. **Q6 B08**：as-run MATH500 是主判定；finisher acc / non-termination 是必报分解。
7. **Q7 C08**：MMLU-Pro 保持 primary preservation axis；supplementary benchmark 不替换 C08。
8. **Q8 supplementary**：Core OPD 跑完后再根据 OPD/SFT 轨迹选择 checkpoint pairs 和 supplementary tasks；不预先跑固定 landmark-only 或 full trajectory supplement。
9. **Q9 A08**：同时做 OPD-vs-SFT relative OverlapLift 和 random-LoRA reference；random baseline 用作绝对尺度参考。
10. **Q10 geometry/UV**：全 checkpoint geometry/OverlapLift 继续做；UV/principal-angle 相关直接子空间旋转分析延后到 Cycle 09。
11. **Q11 artifacts**：overlap probe 和 phase0 smoke artifacts 必须 copyback。
12. **Q12 SFT reference**：核心任务不重评 Cycle 07 SFT；supplementary 只在 core 后按用户选择补测对应 SFT checkpoints。
13. **Q13 write targets**：只更新 `exp/current_experiment_design.md`、`exp/current_acceptance_criteria.md`、`exp/current_code_handoff.md`、`code/current_code_brief.md`，不写 `-ch`，不追加 `research_cycles/`。

## 已执行写入（本阶段范围内）

1. `exp/current_experiment_design.md`：覆盖为 Cycle 08 H-OPD vs SFT 设计。
2. `exp/current_acceptance_criteria.md`：覆盖为 Phase0/B08/C08/D08/A08 gates。
3. `exp/current_code_handoff.md`：更新为 Cycle 08 redirect。
4. `code/current_code_brief.md`：覆盖为 Cycle 08 Code Execution handoff。

