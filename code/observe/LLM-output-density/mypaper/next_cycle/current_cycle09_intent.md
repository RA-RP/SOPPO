# Current Cycle 09 Intent

```yaml
artifact_type: next_cycle_intent
cycle: cycle_09_aaai_competitiveness_completion
status: ready_for_theory_and_experiment_design
created: 2026-07-07
source_qa: next_cycle/QA.md (cycle_09)
prior_draft: 2026-07-03 (updated from draft_seed_pending_cycle08_results)
write_owner_next: Related Work (scoop audit first), then Theory, then Experiment Design
```

This file is the Next Cycle Seed → Theory / Experiment Design handoff for Cycle 09.

---

## Cycle 08 Results Summary

Cycle 08 trained an OPD arm (Qwen3-4B-Base ← Qwen3-8B teacher, verl on-policy distillation,
LoRA r=32, lambda=1.0) and compared it to the Cycle 07 SFT reference trajectory.

**Gate verdicts:**

| Gate | Result |
|---|---|
| B08 — Math capability | **STRONG PASS**: OPD final 0.848 vs SFT 0.752 (+0.096); OPD beats SFT at every step (+0.10 to +0.24) at unified cap 16384. Conservative — OPD is more truncation-suppressed. |
| C08 — MMLU-Pro preservation | **PASS**: OPD net +0.016 vs SFT net −0.029; delta margin +0.045. Non-monotone trajectory: large dip step_40/80 (−0.12), recovery by step_160, net positive at final. |
| D08 — Geometry discriminant | **POSITIVE**: SFT shows L18 step_20 rank bump; OPD ER contracts monotonically with no bump. ER is the sole metric that cleanly separates the arms and tracks the OOD outcome. |
| A08 — Weight-space direction | **INCONCLUSIVE**: OverlapLift retracted (≈ random-LoRA null). ρ reads OPD on-principal / SFT early off-principal; external full-param lit (2606.13657/07082) says OPD off-principal. Contradiction unresolved. |

**New findings (not pre-registered):**
- **OPD early dip at step_5** (MATH500 −0.100, numina −0.070): earlier and deeper than SFT
  dip at step_20 (−0.064). OPD recovers by step_20; SFT by step_160. Both dips co-locate
  with ER uptick. Core mechanism story candidate.
- **LoRA ρ vs full-param contradiction**: LoRA ρ reads OPD on-principal; external full-param
  lit says OPD off-principal. LoRA structure may change the geometry regime.
- **Eval-cap correction**: OPD's long distilled CoT truncated under 4096. Unified cap 16384
  used for MATH500 comparison; late-step "decline" was an artifact.

---

## Seed Decision

Cycle 09 is the **AAAI competitiveness completion cycle**, but the path requires resolving
a prior strategic question before running more experiments:

> **First priority: define and narrow the novel contribution given the external lit scoop.**
> Then run fp32 retrain + P0 (stats + geometry + output-control) + P2 (theory + paper rewrite).
> P1 (controls + robustness) deferred to Cycle 10.

The external full-param literature (arXiv 2606.13657/07082) has published OPD off-principal
weight geometry results. Our activation-space ER evidence is NOT covered by that work,
but our weight-space ρ finding (OPD on-principal under LoRA) contradicts it. The novel
moat is the activation-space observation, not the weight-space direction.

---

## Contribution Framing (Three Candidates — must be chosen before Cycle 09 Experiment Design)

Three candidate framings for our paper's novel contribution:

**(a) LoRA supplement:**
LoRA adapters change the principal geometry rules relative to full-param. Our LoRA ρ
(OPD on-principal under LoRA) is a genuine LoRA-regime finding that the external full-param
work does not cover. If fp32 retrain shows OPD on-principal, this strengthens the
LoRA-specific argument. If fp32 shows OPD off-principal (matching external lit), then
the LoRA adapter itself changes the direction — which is also a finding.

**(b) New observation space:**
Activation-ER is a discriminating observable that is not covered by weight-space analyses.
We propose ER as the mechanism observable linking training dynamics to OOD outcome.
This framing does NOT depend on the weight-space direction result and is robust to the
external lit scoop.

**(c) Geometry-process-result chain (largest ambition, may need full-param + multi-task):**
ER state (activation compression level) → dip-recovery process (timing and depth) → OOD
outcome (MMLU-Pro preservation vs degradation). A causal-narrative framework integrating
geometry, training trajectory, and generalization.

**Cycle 09 Theory conversation must choose among these before Experiment Design begins.**
The answer likely affects whether the paper is a mechanism paper (b, c) or a measurement
paper (a), and how much compute is required.

---

## Required Workstreams

### Tier 0: Contribution Positioning (no experiments — Theory conversation)

Define the novel contribution and paper scope based on:
- Review external lit (arXiv 2606.13657/07082) in detail: what exactly do they claim?
  What is not covered (LoRA, activation space, training trajectory)?
- Decide among framings (a), (b), (c) or a combination.
- Update `result/claims_allowed.md` with the chosen claim boundary before writing.

**This is the blocking task before any new code runs.**

---

### Gate A09: Weight-Space Direction (Full-Param / fp32 Retrain)

Run a full-param or fp32-master retrain to resolve the LoRA ρ contradiction with external lit.

Required:
- Same student (Qwen3-4B-Base), same data (Math-CoT-20k), full-param instead of LoRA.
- Measure ρ (scale-invariant directional alignment) using the same protocol as Cycle 08.
- Compare OPD full-param ρ vs SFT full-param ρ vs LoRA ρ from Cycle 08.

Decision tree:
- If full-param OPD is also on-principal: LoRA finding is real, not LoRA-confounded.
  → Strengthens framing (a).
- If full-param OPD is off-principal (matching external lit): LoRA adapters change the
  direction → LoRA-vs-full-param geometry difference is itself a finding.
  → Enables framing (a) in a different way.

**A09 is a prerequisite for any weight-space direction claim in the paper.**

---

### Gate B09: Statistical Reliability (P0)

Run paired bootstrap 95% CI on all core OPD-vs-SFT comparisons. Every main-text claim
must have a CI or be explicitly marked as qualitative/diagnostic.

Required CIs:
- OPD final − SFT final (at unified cap 16384)
- OPD peak − SFT peak
- OPD dip depth − SFT dip depth (step_5 vs step_20)
- OPD trajectory AUC − SFT trajectory AUC
- OPD non-termination peak − SFT non-termination peak

**Without CI, the OPD early dip mechanism story (Q3=a) cannot be a primary claim.**

---

### OPD Early Dip Mechanism (Core Contribution)

Formalize the "sharper-shorter transient" story with evidence:

1. Confirm OPD ER uptick at step_5 (activation geometry timing).
2. Compare OPD ER trajectory shape vs SFT: does OPD ER peak earlier and compress faster?
3. Add CI on finisher acc at dip checkpoint and recovery speed (steps to recover).
4. Write mechanism hypothesis in non-causal language:
   > "On-policy distillation triggers an earlier and sharper compression transient
   > (step_5, −0.100) that resolves within one checkpoint interval. SFT triggers a
   > shallower but longer-lived transient (step_20, −0.064, resolving by step_160).
   > Both transients co-locate with the effective rank uptick in activation space."

---

### OPD Geometry Mechanism Package (P0, Diagnostic)

Convert Cycle 08 geometry into paper-level evidence comparable to Cycle 07 SFT:

- Effective rank trajectory: OPD vs SFT, same checkpoint grid, same layer/module.
- xs_log_spectrum_gap: OPD vs SFT.
- Drift from base checkpoint: OPD vs SFT (negative control).
- ER on OOD-domain inputs (MMLU-Pro prompts, if feasible): strengthen compression→OOD story.

**Gate B09: OPD and SFT geometry tables must be directly comparable before paper section is written.**

---

### Gate C09: Output-Control Analysis (P0)

Report and classify the OPD accuracy advantage:

- as-run accuracy (unified cap 16384 for math500, as-run 4096 for early steps)
- finisher accuracy (correct / finisher count)
- response length trajectory
- non-termination rate (parallel to Cycle 07 SFT table)
- truncation / boxed-answer-before-truncation audit

Classify: is OPD's improvement over SFT capability, output-control, or mixed?

Note: OPD produces very long responses (3974 tokens at step_624). The MMLU-Pro dip
at step_40/80 should be examined for a possible output-length or distribution-shift artifact.

---

### Theory Update and Paper Rewrite (P2)

Update theory and LaTeX draft around the chosen framing, e.g.:

> "SFT under token-level imitation amplifies a representation expansion transient that
> is shallower and longer-lived. OPD's on-policy alignment triggers a sharper, shorter
> transient that recovers quickly, maintaining representational compression and
> OOD-relevant directions."

This is a mechanism framework, not a theorem. Strict causality claims are not allowed.

Claim boundaries in `result/claims_allowed.md` must be updated before the draft makes
stronger statements.

The paper must not contain "Cycle N" language in its main narrative.

---

## Things Not To Do in Cycle 09

- Do not run Cycle 09 experiments before the contribution framing decision (Theory first).
- Do not claim OPD on/off-principal in weight space without A09 fp32 result.
- Do not claim the compression→OOD link is causal — it is correlational (n=2 arms).
- Do not start the off-policy KD control or robustness axis — deferred to Cycle 10.
- Do not use Qwen3-1.7B as a robustness model (insufficient headroom, prior evidence).
- Do not cite Cycle 08 OverlapLift as a signal — it was retracted.

---

## Deferred to Cycle 10

| Item | Gate | Reason |
|---|---|---|
| OPD-specific control (off-policy KD) | D09 | Core contribution not yet established. |
| Robustness axis (second data distribution) | E09 | Same reason. |
| Teacher / theta0 / headroom baselines | F09 | Same reason; revisit after core claims. |

---

## Cycle 09 Gate Summary

| Gate | Status | Priority |
|---|---|---|
| Contribution positioning | OPEN — must decide before experiments | BLOCKING (Tier 0) |
| A09: fp32 weight-space direction | OPEN — run needed | P0 |
| B09: statistical CI | OPEN — not yet computed | P0 |
| OPD dip mechanism (CI + ER timing) | OPEN | P0 |
| C09: output-control decomposition | OPEN — partial data exists | P0 |
| B09 geometry package (OPD vs SFT tables) | OPEN — data exists, needs analysis | P0 |
| P2: theory + paper rewrite | OPEN | P2 (after P0) |
| D09: off-policy KD control | DEFERRED | Cycle 10 |
| E09: robustness axis | DEFERRED | Cycle 10 |

---

## Read Next

### Related Work (first)

Related Work should read, in order:
1. `next_cycle/current_cycle09_intent.md` (this file)
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/RESULTS_08.md`

Key tasks for Related Work:
- Audit arXiv 2606.13657 and 2606.07082 in detail: what exactly is claimed?
  What is NOT covered (LoRA regime, activation-space ER, training trajectory)?
- Identify the remaining novelty space for each of the three contribution framings (a/b/c).
- Update `related_work/current_related_work.md` with the scoop assessment and gap map.

### Theory (second, after Related Work)

Theory reads Related Work output, then develops the mechanism framework:
- Choose among contribution framings (a) LoRA supplement / (b) new observation space /
  (c) geometry-process-result chain, or a combination.
- Update `theory/current_theory_update.md`.

### Experiment Design (third, after Theory)

Writes `exp/current_experiment_design.md` and `exp/current_acceptance_criteria.md`
for Cycle 09 experiments (A09 fp32 retrain, CI, geometry package, output-control).
