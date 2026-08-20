# Current Cycle 08 Intent

```yaml
artifact_type: next_cycle_intent
cycle: cycle_08_h_opd_vs_sft_comparison
status: ready_for_experiment_design
created: 2026-06-27
source_qa: next_cycle/QA.md
write_owner_next: Experiment Design
```

This file is the handoff from Next Cycle Seed to Experiment Design for Cycle 08.

---

## Seed Decision

Cycle 08 is the **H-OPD vs SFT comparison cycle**.

Cycle 07 established the SFT reference trajectory on Qwen3-4B-Base + Math-CoT-20k
(LoRA, think-format): A07 PASS, B07 FULL PASS, geometry triple co-location at step_20,
OverlapLift < 1 rising monotonically. Cycle 08 trains the OPD arm on the same setup
and runs the first direct comparison.

> **One OPD arm, one lambda, same model + data + LoRA config as Cycle 07 SFT.**

---

## OPD Arm Design

| Parameter | Value |
|---|---|
| Model | Qwen3-4B-Base (same checkpoint as Cycle 07) |
| Training data | Math-CoT-20k (same as Cycle 07) |
| Training format | Think-format LoRA (same as Cycle 07) |
| LoRA config | Same rank / alpha / target modules as Cycle 07 |
| Checkpoint grid | Same 10-point grid: step_000…624 |
| OPD regularizer | Single arm, one best-guess lambda — NOT a sweep |
| Lambda selection | Experiment Design selects based on prior cycles (lmbda0.5 or lmbda1.0) and theory layer; justify the choice explicitly |

**Why single arm:** OPD training is expensive. A lambda sweep is not warranted
until feasibility is confirmed on the 4B+Math-CoT regime.

---

## Evaluation Protocol

**Same protocol as Cycle 07 for all generative tasks:**
- MATH500: no chat template, think-format generative, max_tokens=4096, `\boxed{}`-aware scorer
- as-run acc is the primary metric (cap-robust confirmed: 0.620@4096 ≈ 0.617@24576)
- Do NOT change max_tokens for Cycle 08 — the truncation is computationally valid

**For every checkpoint, additionally report:**
- Non-termination rate (parallel to Cycle 07 table)
- Finisher acc (correct / finisher count) — available for decomposition analysis

**OOD-lite — full trajectory tracking for OPD arm:**
- GPQA-Diamond (no chat template, loglikelihood) at all 10 checkpoints
- MMLU-Pro (no chat template, loglikelihood) at all 10 checkpoints
- Rationale: C07 found mild MMLU-Pro degradation (−2.2 SE at final); Gate C08 tests
  whether OPD arm preserves MMLU-Pro better; full trajectory needed to distinguish
  endpoint degradation from trajectory-timing effects

**Geometry — same protocol as Cycle 07:**
- effective_rank, xs_log_spectrum_gap at all 10 checkpoints
- OverlapLift (down_proj, attention modules) at all 10 checkpoints
- UV/sketch artifacts must be saved (same as Cycle 07; required for OverlapLift)
- Drift from base checkpoint (step_000) — same negative-control reference as Cycle 07

---

## Gate Definitions (seed-level; Experiment Design must pre-register exact thresholds)

Gates listed in order of importance: B08 (blocking) > C08 > D08 > A08.

### Gate B08 — Capability Preservation (Primary, Blocking)

**Question:** Does OPD arm match SFT final MATH500 (non-inferiority) and avoid
the step_20-level capability dip?

Sub-conditions:
1. **Final non-inferiority:** OPD arm final MATH500 ≥ SFT arm final − 1SE.
   (Threshold: 0.680 − 0.020 = 0.660; Experiment Design sets exact value from
   measured SFT SE at step_624.)
2. **Dip severity:** OPD arm minimum finisher acc > SFT step_20 finisher acc (0.594).
   OR: if OPD dip exists, it is shallower and/or at a different step.

**Pass:** OPD matches SFT capability and shows different dip dynamics.
Supports the core claim that OPD is not worse than SFT on the primary task.
**Fail:** OPD significantly underperforms SFT on MATH500 — investigate lambda,
LR, or training duration before advancing.

---

### Gate C08 — MMLU-Pro Preservation (Second)

**Question:** Does OPD arm show smaller MMLU-Pro degradation than SFT at the
final checkpoint?

**Pass condition:** OPD arm MMLU-Pro(step_624) − MMLU-Pro(step_000) > SFT arm
MMLU-Pro(step_624) − SFT arm MMLU-Pro(step_000). (i.e., less degradation.)

**Pass:** Supports claim that OPD regularization preserves general knowledge
better than SFT under matched training conditions.
**Fail:** OPD degrades MMLU-Pro equally or more than SFT — OPD does not help
on this preservation axis.

---

### Gate D08 — Geometry Pattern (Third)

**Question:** Does OPD geometry at its capability low-point differ from SFT
step_20 triple co-location?

Comparisons:
- Does OPD arm show triple co-location (argmin acc = argmax rank = argmin xs_gap)?
  If yes: at what step? If the OPD low-point is earlier, later, or absent,
  document the difference.
- Effective rank trajectory: does OPD show same peak-at-dip pattern as SFT?
- xs_gap trajectory: does OPD show same minimum-at-dip as SFT?

**This gate is diagnostic, not blocking.** Record the OPD geometry trajectory
and compare against the Cycle 07 SFT trajectory table.

---

### Gate A08 — OverlapLift Comparison (Fourth, Most Complex)

**Question:** Does OPD OverlapLift trajectory differ systematically from SFT?

SFT arm: OverlapLift < 1 throughout, rising monotonically (down_proj 0.48→0.74;
attention 0.04–0.28). Verdict on on/off-principal deferred from Cycle 07.

For Cycle 08:
1. Measure OPD arm OverlapLift at all 10 checkpoints (same modules as Cycle 07).
2. Establish or approximate the LoRA-aware reference for normalization. Experiment
   Design must specify the normalization approach before the run.
3. Compare OPD vs SFT OverlapLift trajectories; note whether OPD rises faster,
   slower, or reaches a different plateau.

**This gate is diagnostic, not blocking.** Goal: make the on/off-principal
verdict for both arms in Cycle 08 by resolving the LoRA-aware reference question.

---

## SFT Reference Trajectory (from Cycle 07)

All Cycle 08 comparisons use the Cycle 07 SFT trajectory as the reference control.
Do not re-run the SFT arm in Cycle 08.

Key reference points:
- **Dip checkpoint:** step_20 (finisher acc 0.594, non-term 6.8%, argmax rank, argmin xs_gap)
- **Recovery checkpoint:** step_160 (first step above base)
- **Peak checkpoint:** step_320 (MATH500 0.698)
- **Final checkpoint:** step_624 (MATH500 0.680, non-term ~50%, finisher acc 0.98)
- **MMLU-Pro at final (SFT):** −2.2 SE vs step_000

---

## Legacy Items — Closed

| Item | Decision |
|---|---|
| C06 PARTIAL (geometry copyback) | **CLOSED.** Not tracked. |
| A06b UNDETERMINED (base stop-fix) | **CLOSED.** Not tracked. |

---

## Required Experiment Design Decisions

1. **Lambda selection:** which lambda value for the OPD arm? Justify from prior
   cycles (lmbda0.5 or lmbda1.0) or theory layer. Record the justification in
   `exp/current_experiment_design.md`.
2. **LoRA-aware OverlapLift reference:** how to normalize for on/off-principal
   verdict? Options: random-LoRA baseline run, analytical bound, or alternative
   normalization from OverlapLift literature. Decide before the run.
3. **B08 exact thresholds:** pre-register the non-inferiority threshold and dip
   severity condition using the measured Cycle 07 numbers.
4. **C08 exact threshold:** pre-register MMLU-Pro preservation condition.

---

## Routing After Cycle 08

- **If B08 + C08 pass:** strong evidence that OPD matches SFT capability and
  preserves MMLU-Pro better. Core paper claim viable. Cycle 09 can focus on
  geometry mechanism and paper writing.
- **If B08 pass, C08 fail:** OPD matches capability but does not preserve better.
  OOD-lite claim is not supported; paper must narrow scope.
- **If B08 fail:** OPD underperforms SFT. Investigate lambda before Cycle 09.
- **D08 and A08:** regardless of B08/C08 outcome, record geometry and OverlapLift
  findings and advance the LoRA-aware OverlapLift interpretation.

---

## Read Next

Experiment Design should read, in order:

1. `next_cycle/current_cycle08_intent.md` (this file)
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `theory/current_theory_update.md` (for OPD lambda justification)
5. `exp/experimental_methodology/training_protocol.md`
6. `exp/experimental_methodology/evaluation_protocol.md`
7. `exp/experimental_methodology/geometry_protocol.md`
8. `local_experiment_results/cycle_07_base_sft_trajectory/run_01/RESULTS_07.md`

Then write `exp/current_experiment_design.md`, `exp/current_acceptance_criteria.md`,
and the cycle08 code handoff.
