# Current Theory Update

```yaml
view_type: current_latest_view
status: active_cycle08_theory
source: (snapshot removed 2026-06-16; this file is primary)
maintenance_rule: preserve_valid_content_update_changed_content_add_missing_content
last_organized: 2026-06-29
```

> This file is maintained as the current workspace view. Source snapshot removed 2026-06-16; this file is now the primary source.

---

# Theory Update 05: From Matched-Gain Stability To Base-Aware Stability

## Material Passport

```yaml
artifact_type: theory_update
cycle: cycle_05_matched_control_id_ood
source_cycle: ../cycle_04_opd_stability_gain/
status: proposed
date: 2026-06-16
```

## Why The Theory Needs Updating

Cycle 03 supported a narrow K1-style story:

```text
At matched GSM8K gain, OPD-like training can show lower OOD-lite degradation
and lower spectral drift than continued SFT.
```

Cycle 04 showed that this story is under-specified. The matched-gain control can
select very different SFT anchors depending on `theta0` saturation. When the
matched SFT is almost unchanged from `theta0`, the comparison becomes unfair to
any method that actually moves.

The added base-model test shows an even more basic issue: the untrained
Qwen3-1.7B base is already approximately tied with `theta0` on ID and GSM8K,
and is higher than OPD on the ID axis. That means the current setting may have
too little headroom for any training method to show a meaningful gain.

## Updated Hypothesis

Use the following base-aware hypothesis for Cycle 05:

```text
K1-base-aware: OPD-like training can only be meaningfully compared with SFT
after the selected base/task/data setting shows measurable headroom above the
untrained base. Conditional on that headroom, OPD-like training is more stable
only if it preserves OOD behavior better at comparable ID learning or comparable
training budget, not merely at comparable GSM8K gain.
```

This version separates three quantities that Cycle 03 blurred:

- ID learning: performance on NuminaMath-test.
- OOD transfer: performance on GSM8K and OOD-lite.
- base headroom: trained-model improvement over the untrained base.
- Representation movement: spectral drift, effective rank, and subspace change.

## Key Variables

| Variable | Role |
|---|---|
| `base_performance` | Establishes whether the task has measurable headroom above the untrained model. |
| `theta0_saturation` | Determines how much room is left for ID/OOD gain and which SFT anchor is selected. |
| `id_performance` | Measures same-domain capability on NuminaMath-test. |
| `ood_retention` | Measures transfer/generalization on GSM8K and OOD-lite. |
| `training_budget` | Prevents comparing a trained OPD model to a nearly untrained SFT model. |
| `geometry_movement` | Measures representation change, but should be interpreted relative to learning amount. |
| `extraction_validity` | Blocks false conclusions from scoring artifacts. |

## Predictions

### P0: Base Headroom

If base, `theta0`, OPD, and SFT are tied within uncertainty on ID/OOD, then the
setting is not suitable for testing improvement or stability. The next action
should be task/base/data redesign, not another OPD-vs-SFT run.

### P1: Equal-ID Matching

If OPD has a real stability advantage, then at similar NuminaMath-test
performance it should retain GSM8K/OOD-lite at least as well as SFT.

### P2: Equal-Budget Matching

If the Cycle 04 negative result was mainly a control-anchor artifact, then
equal-budget comparison should be less hostile to OPD than matching against
`sft_n128`.

### P3: Theta0 Saturation

As `theta0` becomes more saturated, matched-gain selection should shift toward
larger or more degraded SFT controls. As `theta0` becomes less saturated,
matched-gain selection may collapse toward tiny SFT controls.

### P4: Geometry Normalization

Raw drift from `theta0` may penalize methods that learn more. A useful geometry
metric should be interpreted together with ID learning and OOD retention.

## Falsifiers

The base-aware K1 hypothesis is weakened if:

- no trained variant beats the untrained base beyond uncertainty;
- OPD fails to retain OOD better than SFT under equal-ID matching;
- OPD fails to retain OOD better than SFT under equal-budget matching;
- OPD's geometry movement is larger without corresponding OOD retention;
- conclusions continue to flip only because the matching anchor changes.

## Updated Paper Claim

Until Cycle 05 is complete, the safest paper-level claim is:

```text
Our early OPD-like experiments show that stability conclusions depend strongly
on base headroom and control-anchor construction; base-aware ID/OOD matching is
required before claiming an OPD stability advantage.
```

---

# Cycle 06 Theory Update: SFT Feasibility as K1 Precondition

```yaml
artifact_type: theory_update
cycle: cycle_06_sft_feasibility_and_degradation
date: 2026-06-23
status: active
qa_source: theory/QA_cycle06_theory.md
```

## What Changed and What Did Not

K1-base-aware remains the main hypothesis. Cycle 06 contributes minimally to
theory — its purpose is engineering validation: finding the right model, training
format, and evaluation protocol under a clean SFT baseline. SFT catastrophic
forgetting is a well-known phenomenon and does not require new theoretical variables.

The only theory-level additions for Cycle 06 are:
1. A precondition clause clarifying when K1 can be tested.
2. Model identity disambiguation in claim boundaries.
3. Gate interpretation rules to prevent post-hoc standard switching.
4. Updated OOD-lite and prohibition claims.

No new mechanism variables are added. The SFT degradation curve (scale →
degradation) is a motivating observation and engineering finding, handled in
Experiment Design.

## Cycle 06 Precondition for K1 Validity

K1-base-aware can only be meaningfully tested when both of the following hold:

1. **SFT learnability confirmed** (Gate A06 pass): SFT-from-base produces a
   measurable gain above the untrained base on the primary ID/downstream axis.
2. **Protocol validity confirmed** (Gate C06 pass): The train/eval format and
   scorer protocol have been validated before training runs.

If neither condition holds, the setting is not appropriate for OPD-vs-SFT
comparison. The next action should be model, task, data, or format redesign —
not another OPD training run.

## Model Identity Claim Boundary

```text
local_base_row ≠ official_pretraining_base_checkpoint

Local Qwen3-1.7B (instruct stack, post-trained):
  path: /root/autodl-tmp/model/Qwen/Qwen3-1.7B
  Has undergone instruction tuning. Cannot be treated as a pretraining-only LM
  starting point. Its 0-shot chat performance does not represent a
  pretraining-only capability floor.

Official Qwen3-1.7B-Base (pretraining-only):
  path: Qwen/Qwen3-1.7B-Base (HuggingFace hub official)
  No instruction tuning. Chat/think format assumptions do not automatically apply.
  Requires a separate evaluation protocol.
```

If Cycle 06 switches to the official Base checkpoint, theory must be updated in
the next cycle to distinguish the applicable assumptions (e.g., whether Base's
lower initial capability implies larger SFT headroom remains untested).

## Gate A06 / B06 Theory Interpretation Boundary

These are interpretation rules only. Exact numeric thresholds belong to Experiment Design.

```text
A06 pass:
  The current model/task/data/protocol setting has sufficient SFT learnability.
  This justifies considering an OPD comparison in a later cycle.
  It does NOT mean OPD comparison is ready — Experiment Design must still
  decide cost, control, and matching criteria.

A06 fail:
  The current setting has insufficient headroom or a protocol/task/model mismatch.
  Do not interpret this as OPD failure.
  Redesign model, task, data, or format before running OPD.

B06 degradation observed:
  SFT scale can induce a capability tradeoff in this setting.
  OPD forgetting-prevention test becomes meaningful in a future cycle.

B06 no degradation:
  The current setting appears stable under SFT at the tested scales.
  OPD forgetting-prevention is not yet testable in this setting.
```

## OOD-lite Claim Boundary (Formalized from Cycle 05)

OOD-lite (MMLU / TruthfulQA / WinoGrande / ARC-challenge) is a
**capability-preservation check only**, not a discriminating axis for OPD-vs-SFT
or SFT-scale comparisons, in the current 1.7B + NuminaMath setting.

Empirical basis: Cycle 05 showed spread <1.5% across all 8 models on all four
tasks, including `sft_n1024` which shows severe math regression. The axis lacks
discriminative power in this setting.

This boundary applies until a future experiment demonstrates larger, protocol-valid
spread on one or more OOD-lite tasks.

## Cycle 06 Claim Prohibitions

1. Do not claim OPD prevents catastrophic forgetting. Cycle 06 trains no OPD arm.
2. Do not frame Cycle 06 as an OPD-vs-SFT comparison.
3. Do not treat `sft_n1024` degradation as a proven mechanism (overfitting,
   forgetting, or otherwise). It is a motivating observation / consistency signal.
4. Do not use OOD-lite as a discriminating axis in the current setting.
5. Do not conflate the local Qwen3-1.7B instruct checkpoint with the official
   Qwen3-1.7B-Base pretraining-only checkpoint.
6. Do not treat geometry diagnostics as mechanism proofs.
7. Do not cite protocol-invalid scores as theoretical evidence:
   - Original Cycle 04 GSM8K (~0.41 band): broken protocol, not capability.
   - Original Cycle 04 MATH500 (0.000): extraction failure, not capability.
   - Cycle 05 MMLU with chat template (0.2295): wrong protocol.

---

# Cycle 07 Theory Update: From Base-Aware Boundary to a Trajectory Mechanism (OPD vs SFT_dip vs SFT_recover)

```yaml
artifact_type: theory_update
cycle: cycle_07_base_model_sft_feasibility
date: 2026-06-24
status: active
qa_source: theory/QA_cycle07_theory.md
scaffolding: related_work/source_notes/rethink_sft_experiment_tables.md §7
external_anchor: "Ren et al., Rethinking Generalization in Reasoning SFT, arXiv 2604.06628 (2026)"
```

## What Changed: The Project's First Positive Mechanism

Cycles 04–06 were **defensive**: they established *when a comparison is invalid*
(base saturation, no headroom, capability floor, protocol artifacts). Cycle 06
theory in particular was a minimal, engineering-level update.

Cycle 07 is different in kind. It introduces the project's **first positive,
falsifiable mechanism hypothesis**. The trigger is the external evidence from
Rethink SFT (Ren et al. 2026): cross-domain performance under reasoning SFT
follows a **dip-and-recovery** trajectory, not a monotone curve.

This does not replace the base-aware framework — it **nests inside it** (Q1=c).
The three layers form a single chain:

```
Layer 1 (entry condition)   K1-base-aware: a comparison is only meaningful with
                            measurable headroom above the untrained base.
        │  (Rethink SFT externally reinforces this: Qwen3-1.7B-Base reproduces
        │   the capability floor; 4B + long-CoT is the move to clear the gate.)
        ▼
Layer 2 (mechanism)         Dip-and-recovery: OOD-drop is a property of the
                            checkpoint's optimization stage, not of the method.
        ▼
Layer 3 (pre-registered)    H-OPD: OPD is a shortcut that reaches the recover
                            state without passing through the dip. (Cycle 08 test.)
```

## Core Reframing: OOD-drop Is a Checkpoint Property, Not a Method Property (Q2)

Earlier cycles implicitly treated OOD-drop / stability as a **fixed property of
the method** ("SFT drops OOD, OPD is more stable"). Rethink SFT's dip-and-recovery
shows this is wrong as stated: the *same* SFT run drops OOD at an early checkpoint
(dip) and rebounds — often surpassing base — at a later checkpoint (recover).

**Reframed central claim:**

```text
OOD-drop is a transient on the optimization trajectory, determined by the
checkpoint's optimization stage (dip vs recover). "SFT necessarily drops OOD"
mistakes a checkpoint property for a method property.
```

**Caveat (mandatory claim boundary):** This reframing is currently supported by
*external* evidence (Rethink SFT, on Qwen3-4B/8B/14B). This project has **not yet
reproduced dip-and-recovery in its own pipeline**. Cycle 07's multi-checkpoint
sampling on Qwen3-4B-Base is precisely the internal validation. Until then, treat
the reframing as a strong, externally-supported hypothesis, not a settled result.

## The Three-Way Framework: OPD vs SFT_dip vs SFT_recover

SFT is not a point but a trajectory SFT_dip → SFT_recover. The two states have
distinct, measurable signatures (Rethink SFT, Qwen3-4B-Base + Math-CoT-20k,
Table 9; behavioral reads from App C):

| Dimension | SFT_dip (~step 20–40) | SFT_recover (~step 320+) | Rethink SFT 4B anchor |
|---|---|---|---|
| ID (training domain) | not consolidated (MATH500 62.9 < base 70.8) | consolidated / surpassed (MATH500 86.2) | Table 9 |
| OOD reasoning | drops (GPQA 17.3, LCB 14.5) | rebounds / surpasses (GPQA 37.9, LCB 24.4) | Table 9 |
| response_length | surges (~28–30k) | contracts (~7–23k) | Sec 3.2; Fig 13 |
| behavior | shallow imitation; long but non-convergent | internalized procedural pattern (backtracks, self-verifies) | App C.3, C.6 |

The 4B run also shows a visible dip (MATH500 step 20 = 62.9, below base 70.8,
before rising to 86.2), so the framework's two states are expected to be
observable at this model scale.

## H-OPD: A Pre-Registered Hypothesis for Cycle 08 (Q3)

```text
H-OPD: OPD ≈ a shortcut that reaches the recover state without passing through
the dip transient. On-policy training reduces exposure bias, so the model takes
fewer detours through shallow-imitation states.
```

**Falsifiable predictions (Cycle 08, requires an OPD arm — NOT tested in Cycle 07):**

- If H-OPD holds: OPD's (ood_drop, response_length surge, geometry transient peak)
  are **significantly smaller than SFT_dip**, while its ID/OOD endpoints are
  **≈ SFT_recover**.
- If H-OPD fails: OPD passes through a dip comparable to SFT, or its endpoint does
  not match SFT_recover.

**Literature support (OPD leg, registered in Related Work):** Shenfeld et al. 2026
(online training forgets less and stays closer to the base distribution → directly
predicts flatter drift-from-base for OPD); Chen et al. 2025 (on-policy mitigates
forgetting); Tajwar et al. 2024 (value of on-policy sampling).

**Boundary:** Cycle 07 trains no OPD arm and produces no OPD evidence. H-OPD is
written here only as a pre-registered target so that Cycle 08 cannot retrofit a
story to whatever it finds.

## Cycle 07 Trajectory Variables (Q4)

Unlike Cycle 06 (which deliberately built no variable table because degradation
was an engineering given), Cycle 07 is a genuine mechanism and needs variables:

| Variable | Meaning | Role |
|---|---|---|
| `optimization_stage` | checkpoint's phase along the SFT trajectory (dip / transition / recover) | core independent variable to which OOD-drop is attributed |
| `response_length` | generation length; surges in dip, contracts in recover | **observable proxy** for dip↔recover (Rethink SFT Sec 3.2); a **required artifact** — Experiment Design must collect it along the trajectory |
| `ood_drop` | OOD-reasoning change relative to base | dependent variable; reframed from method property to checkpoint property |
| `id_consolidation` | whether ID (training-domain) performance consolidates / surpasses base | distinguishes shallow imitation from internalized learning |
| geometry (trajectory) | effective rank / eigengap γ_r / drift-from-base read **along the trajectory** | mechanism-layer evidence; see `geometry_metric_definitions.md` (Cycle 07 section) |

**Hard requirement for Experiment Design:** geometry and `response_length` must be
sampled at **multiple checkpoints** along the SFT trajectory, not only at the final
checkpoint — otherwise the dip transient is invisible and the Cycle 08 three-way
comparison has no baseline.

## Re-Characterization of Prior Negative Results (Q6)

Rethink SFT supplies an external explanation for the project's Cycle 04–06
negative results on Qwen3-1.7B + NuminaMath. They are most likely the **sum of two
independent defects**, not a terminal failure of the SFT method:

- **Capability floor:** Qwen3-1.7B cannot be trained into generalization. The paper
  reproduces this on the *official* Qwen3-1.7B-Base (Table 6: net-negative MATH500
  58.9 → 56.2 after 640 steps), so it is not an instruct/no-think artifact.
- **Data quality floor:** NuminaMath-1.5 short solutions are the paper's
  worst-performing data class (Table 2: drives OOD reasoning to the floor).

```text
Reframing: the Cycle 04–06 instruct negatives are consistent with a weak-model +
low-quality-data double pit, not a terminal failure of SFT as a method.
```

**Caveat (mandatory):** Rethink SFT is external (4B/8B/14B). This project has not
yet reproduced the 4B + long-CoT positive result in its own pipeline. Until Cycle 07
Gate A07 returns, this re-characterization is an **externally-supported hypothesis
pending internal validation**, not a confirmed conclusion. This converts the
project's first half from "a string of opaque negatives" into a methodological
chain that progressively isolated confounders (saturation → headroom → capability
floor + data quality) — but the chain's final link is tested, not assumed.

## Cycle 07 Claim Boundary and Prohibitions (Q7)

1. Do not write OOD-drop as an intrinsic property of the SFT method — it is a
   checkpoint / optimization-stage property (unless future evidence overturns this).
2. Do not treat Rethink SFT's 4B/8B/14B results as "validated by this project" —
   they are external evidence not yet reproduced in this pipeline.
3. Do not write H-OPD as a conclusion or as observed — Cycle 07 trains no OPD arm;
   H-OPD is a pre-registered hypothesis for Cycle 08.
4. Do not call SFT a "fail" before observing the full trajectory — the
   dip-and-recovery + under-optimization guardrail: short-epoch checkpoints
   systematically underestimate generalization (Rethink SFT Sec 3).
5. Do not write the old 1.7B + NuminaMath negatives as "SFT method failure" — they
   are now attributed to a weak-model + low-quality-data double pit (pending
   internal validation).
6. `response_length` / effective-rank / γ_r transient peaks are **diagnostic
   proxies / hypotheses under test**, not mechanism proofs; they require
   multi-checkpoint sampling.
7. Do not write dip-and-recovery as a phenomenon this project has reproduced — it
   comes from external work; Cycle 07 is the first internal test.

## Relationship to Prior Theory

K1-base-aware (Theory 05) and the Cycle 06 SFT-feasibility precondition both
**remain valid** and sit as Layer 1 (entry condition) of the chain above. The
model identity boundary (`local_base_row ≠ official_pretraining_base_checkpoint`)
becomes directly load-bearing in Cycle 07: the move is specifically to the
**official Qwen3-4B-Base** pretraining checkpoint with a long-CoT think-format,
which is the first deliberate attempt to clear the K1 headroom gate.


---

# Cycle 08 Theory Update: Active H-OPD Test and Matched-LoRA Trajectory Comparison

```yaml
artifact_type: theory_update
cycle: cycle_08_h_opd_vs_sft_comparison
date: 2026-06-29
status: active
qa_source: theory/QA_cycle08_theory.md
depends_on:
  - cycle_07_base_model_sft_feasibility
```

## What Changed from Cycle 07

Cycle 07 moved the framework from an externally supported hypothesis to an
internally observed SFT trajectory: Qwen3-4B-Base + Math-CoT-20k + think-format
LoRA is learnable, the SFT run has a real dip-and-recovery structure, and the
early dip is accompanied by a geometry/output-control transient. Cycle 08 is
therefore the first direct test of H-OPD rather than another preparatory cycle.

```text
Active H-OPD test:
OPD should flatten, shift, or avoid the SFT dip transient while matching the
SFT recovered endpoint. If OPD reproduces the same dip or fails to reach the
recovered endpoint, H-OPD is weakened under this teacher/lambda/cold-start setup.
```

This does not make Cycle 07 an OPD result. Cycle 07 remains the SFT trajectory
baseline; Cycle 08 adds the OPD arm needed to test the mechanism.

## Primary Comparison Unit

The primary comparison is the **matched checkpoint-grid trajectory**, not a
single selected checkpoint. The theory-level object is:

```text
SFT trajectory:  base -> SFT_dip -> transition -> SFT_recover
OPD trajectory:  base/cold-start -> OPD checkpoints on the same grid
```

Single checkpoints are landmarks, not the causal comparison itself. The main
landmarks are dip depth, dip timing, recovery timing, endpoint capability,
response-length transient, and geometry transient. Iso-ID or iso-capability
cross-sections are secondary summaries that may help interpret the trajectory,
but they should not replace the trajectory comparison.

B08/C08 remain useful gates, but they are interfaces onto trajectory landmarks:
they do not turn the design back into a single-point matched-control experiment.

## Cold-Start Interpretability

Cold-start choice changes the interpretation of the H-OPD test:

- **Level 0: no cold-start / step_000 OPD.** This is the strictest H-OPD test.
  If it works, it gives the strongest evidence that OPD can avoid the SFT dip
  from the original base. If it fails under very low teacher-overlap, the failure
  may be a conditioning or teacher-alignment failure rather than a direct
  falsification of H-OPD.
- **Level 1: minimal pre-recovery cold-start selected by an overlap probe.** This
  is a conditional H-OPD test. It remains interpretable only if the cold-start is
  before SFT recovery and is chosen solely to enter a viable teacher-overlap
  region, not to inherit the recovered SFT state.

Recover-point cold-start should not be used as a meaningful H-OPD design. It is
neither a useful ablation for "avoid the dip" nor a common training baseline in
this setup; it would only test continued OPD from an already recovered SFT state.

## Lambda Principle

Theory does not set a numeric lambda for Cycle 08. The decision rule is:

```text
With one OPD arm, choose the strongest OPD contrast that is still expected to
preserve endpoint capability.
```

`lambda=1.0` is the cleanest OPD mechanism contrast if it can satisfy B08; it
would best expose the nature of OPD. A smaller lambda is a conservative choice
only if capability-risk considerations make the pure contrast unlikely to pass.
The final numeric choice belongs to Experiment Design.

## Gate Interpretation

**B08: math capability and dip avoidance.**

- Pass: OPD is capability-compatible with SFT_recover and does not reproduce the
  same severe early SFT dip. H-OPD remains viable.
- Fail: this OPD configuration is not yet a viable replacement under the chosen
  lambda, teacher, cold-start, overlap, and implementation. Do not use positive
  geometry or OverlapLift signals to claim H-OPD if B08 fails badly.

**C08: non-target capability preservation.**

- Pass: OPD has preservation evidence relative to SFT on the chosen non-target
  axis.
- Fail or mixed: OPD may match math capability without showing preservation
  advantage.

MMLU-Pro is the current Cycle 08 axis, but it should not be treated as the whole
of OOD. Related Work has identified additional capability-degradation benchmarks;
the concrete suite may be selected or expanded in Experiment Design / Code after
protocol checks. Those supplementary benchmarks can strengthen or weaken the
preservation story, but they do not become post-hoc primary H-OPD gates.

**D08: geometry transient.**

Geometry is mechanism-supporting only when it aligns with B/C behavior. Geometry
cannot rescue a capability failure.

**A08: OverlapLift trajectory.**

OPD-vs-SFT OverlapLift trajectories are a valid relative comparison because the
two arms use the same model/data/checkpoint-grid and the same LoRA regime. A
separate LoRA-aware random reference is not required for the relative OPD-vs-SFT
comparison. However, absolute claims such as "on-principal" or "off-principal"
relative to a universal/random/full-rank reference remain qualified unless that
reference is explicitly defined.

Principal-angle or measured subspace-rotation claims are not part of Cycle 08
theory; they are registered for Cycle 09 consideration.

## Evidence Ladder for Cycle 08 Results

```text
Strong claim:
  B08 pass + C08 pass + D08/A08 mechanism signals aligned.
  OPD matches recovered math capability while reducing the SFT dip/preservation
  cost, with compatible trajectory-level mechanism signals.

Narrow claim:
  B08 pass, but C08 fails or is mixed.
  OPD can match math capability / avoid the severe math dip, but a broader
  preservation advantage is unsupported.

Negative or redesign:
  B08 fail.
  Investigate lambda, teacher, cold-start, overlap, and implementation before
  making any OPD stability claim.
```

## Cycle 08 Claim Prohibitions

1. Do not claim OPD or H-OPD is supported from Cycle 07 alone; Cycle 07 has no OPD
   arm.
2. Do not interpret a single Cycle 08 OPD-arm failure as "OPD theory is false"
   before separating lambda, teacher, cold-start, overlap, and implementation.
3. Do not use recover-point cold-start as evidence that OPD avoided the dip.
4. Do not make single-checkpoint comparison the primary causal comparison; the
   matched trajectory is primary.
5. Do not turn OPD-vs-SFT OverlapLift into an absolute on/off-principal verdict
   against an undefined reference. The relative matched-LoRA OPD-vs-SFT
   comparison itself is allowed.
6. Do not let geometry or OverlapLift positives rescue a B08 capability failure.
7. Do not mix official Qwen3 benchmark scores and this project's protocol scores
   as if they were the same measured quantity.
8. Do not write non-termination as merely an evaluation artifact. Cycle 07 treats
   it as real output-control behavior, and as-run accuracy remains a valid
   end-to-end task-success measure.
9. Do not rewrite supplementary capability-degradation benchmarks into post-hoc
   primary gates.

---

# Cycle 09 Theory Update: Contribution Repositioning After the Scoop (Tier-0)

```yaml
artifact_type: theory_update
cycle: cycle_09_aaai_competitiveness_completion
date: 2026-07-08
status: active
qa_source: theory/QA_cycle09_theory.md (Q1–Q8 = a; discussion resolutions E1–E5)
scoop_audit: related_work/source_notes/cycle09_scoop_audit_and_gap_map.md
unblocks: Experiment Design (Tier-0 framing decided below)
```

## Tier-0 Decision: What Died, What Survives

**Dead (scoop audit, final):** (1) "OPD off-principal weight direction" as a
contribution — published full-param (arXiv 2606.07082, 2606.13657). (2) "Introduce
activation ER as an observable" as a standalone selling point — 2605.30524
headlines effective rank. (3) "Activation compression ↔ OOD" standalone — partially
occupied, with the opposite direction.

**The surviving moat (the paper's core claim territory):**

```text
Under a matched OPD-vs-SFT trajectory comparison (same student / data /
checkpoint grid / LoRA), observed on the whitened OUTPUT-RELEVANT activation
spectrum (SVD-LLM lens): the two supervision channels produce OPPOSITE
spectral-transient signatures. SFT shows a mid-layer (L18) transient rank
expansion plus an early off-principal weight-direction reorientation
(ρ: steps 5–10 off → flips on by step 20), accompanied by OOD (MMLU-Pro)
erosion. OPD keeps a stable update direction throughout and contracts the
spectrum monotonically, accompanied by OOD preservation and a capability
advantage at every step (MATH500 +0.10–0.24; final 0.848 vs 0.752).
Contrary to the collapse literature's "compression → harm": what predicts the
OOD outcome is not the amount of compression but the presence and magnitude
of the transient break.
```

The evidence form no single competitor has: **three independent observation
spaces temporally interlocking** — ρ (weight direction, steps 5–10) → ER bump
(activation spectrum, step 20) → capability dip / OOD erosion (behavior).

**Field development line (the related-work backbone):** Stage 1, weight-space
statics ("where do updates land", completed by 2606.×2) → Stage 2,
representation-space statics/prognosis ("what representations become",
2605.30524) → Stage 3, training dynamics ("what happens along the way",
Rethink SFT / 2509.12235). **Confluence = this project**: process view ×
output-relevant spectral observable × parallel supervision-channel contrast ×
OOD outcome binding.

## Contribution Architecture (Q1 = layered)

| Layer | Content | Evidence dependency |
|---|---|---|
| **Primary** | Sharpened (b): OPD-vs-SFT × whitened output-relevant spectrum × trajectory × direction reversal | Tier A analysis only (complete or in hand) |
| **Mechanism layer** | (c) geometry→process→outcome chain as an explicit hypothesis framework; **upgrades to co-primary** if off-KD control + double-ER land clean and in time | Tier B |
| **Conditional secondary** | (a′) LoRA-regime difference vs full-param lit | A09 outcome (either branch publishable) |
| **Methods** | Measurement critique: OverlapLift retraction + bf16/LoRA invalidate weight-direction metrics | Complete (Cycle 08) + A09 dual-track gap |

Writing rule: primary claims must never depend on Tier B experiments; conditional
contributions carry their conditions explicitly in the text.

## (a′) LoRA-vs-Full-Param: Decision Tree and the Process/End-State Dual Track

**Framework (E1, user-proposed, adopted):** bf16 vs fp32 = **end-state vs
process**. The bf16 deployed artifact is the end-state (what the shipped model
carries — TPNT's ecologically valid object). The fp32 update is the process
(what training geometrically did — the attribution-clean object). The
LoRA-vs-full-param attribution question lives on the **process** side; both
tracks are reported, and their gap quantifies how much bf16 distorts directional
metrics (data for the methods contribution).

**Numerical-rank evidence (E2, recorded):** measured ΔW numerical rank ≈ 2121,
but a LoRA r=32 update BA has rank ≤ 32. The measured object
`bf16(W0+BA) − bf16(W0)` is therefore rank-structurally dominated by full-rank
quantization staircase noise; ρ² ≈ 0.02 ("diffuse") is the signature of the
bf16 noise floor, not of LoRA — a clean BA should be highly concentrated in 32
directions. (The initial conjecture "diffuseness comes from LoRA structure" was
reversed by this evidence in QA discussion.)

**A09 execution order (E2):** step 1 — locate the Cycle 08 LoRA adapter files
(both arms are LoRA r=32) and compute ρ on **fp32(B@A)** directly: zero training
cost, removes the measurement-level confound (adapter-relative rounding ~1e-3 of
the update, vs merge−subtract rounding relative to W0). Retrain only if adapters
are unavailable or the training-level bf16 effect must itself be excluded.

**Decision tree (pre-registered):**
- fp32/clean BA still on-principal (opposite to full-param lit) → "update
  geometry is parameterization-dependent, not method-intrinsic" — a boundary
  correction to 2606.×2's generality (their off-principal is a full-param result
  and does not extrapolate to LoRA deployment, which is the practical mainstream).
  (a′) stands. Two required clarifications: (i) reconcile with LoRA
  intruder-dimensions — a mostly-diffuse-plus-slight-on-bias reading and a few
  intruder directions can coexist; (ii) the LoRA differentiation holds **only
  against 2606.×2** — 2605.30524 is itself LoRA.
- fp32 flips off-principal (matching full-param lit) → bf16 quantization alone
  can flip a directional verdict — the methods warning sharpens; (a′) dies,
  reported honestly.
- **Before A09 resolves: no on/off-principal claim of any kind.**

## Off-Policy KD Control (Q3 = variant i)

Three confounds separate OPD from SFT: input distribution (on-policy rollouts vs
static corpus), supervision signal (dense teacher logits vs one-hot text), and
teacher identity (8B logits vs 32B-generated text). The control arm: same 8B
teacher, same forward_kl_topk loss, same LoRA r=32, same checkpoint grid — only
the input switches from student rollouts to **teacher-generated static
responses** (classic off-policy sequence-level KD). off-KD vs OPD isolates
on-policy sampling cleanly; off-KD vs SFT (density + teacher identity) stays
confounded, deferred to Cycle 10. Engineering note: no rollout loop; teacher
top-k precomputable offline — cheaper than the OPD arm.

**Interpretation tree (pre-registered):**

| off-KD signature | Conclusion |
|---|---|
| SFT-like (L18 bump + early off-principal + OOD erosion) | on-policy sampling drives the OPD signature (H-onpolicy supported) |
| OPD-like (no bump + OOD preserved) | dense logit supervision suffices; on-policy is not the driver — converges with 2606.07082's weight-side control; strengthens the Cycle 10 data-selection conjecture |
| Mixed (e.g., no bump but OOD erodes) | the bump and OOD preservation have decoupled drivers — highest information, most careful writing |

All three outcomes have a paper-ready reading.

## Double-ER Experiment (Q6): Engaging 2605.30524

Strategy = construct-difference as the floor, potential counterexample as the
attack. On the same checkpoints (three arms), compute both: our whitened GetSlice
ER (p ∝ σ, SVD-LLM lens) and their raw centered-covariance ER (λ = σ², d-normalized,
multi-domain probe — merged with the OOD-domain probe requirement). Construct
spec lives in `geometry_metric_definitions.md` (Cycle 09 section).

**Three-outcome tree (pre-registered):**
1. Raw ER also shows OPD compressing more while preserving OOD → **direct
   counterexample/refinement of 2605.30524's headline** ("compression amount is
   not the measure of OOD damage; the transient break is") — strongest outcome.
2. Only the whitened ER carries the discriminating signal → signal is specific to
   the output-relevant spectrum; narrower claim, observable-choice sensitivity.
3. Raw ER reproduces "compression → harm" while whitened ER reverses → the two
   compression constructs divide labor; still novel, requires the mechanism of
   the division to be spelled out.

## Rotation-vs-Values Adjudication (Q5): Answering 2509.12235

Defense premise: the whitening operator is input-covariance-dependent, so
weight-direction rotation manifests as spectral-shape change in whitened
activation space — our ER is not a pure "values" statistic. This must be tested,
not asserted: θ_r (activation principal angles, UV now saved) versus ER as
trackers of (i) arm separation and (ii) MMLU-Pro Δ. Full spec (r-grid, bootstrap,
probe battery) in `geometry_metric_definitions.md`.

**Adjudication tree:** θ_r tracks better → adopt a rotation+values two-axis
observable framework, cite 2509.12235 as cross-space convergent (theirs is
weight-space); ER remains the activation-side discriminator → an empirical reply
("in whitened activation space, value-distribution transients, not subspace
rotation, discriminate OPD from SFT"); complementary split (e.g., θ_r tracks OOD,
ER tracks the dip) → observable division-of-labor framework.

## Cycle 09 Claim Boundary (Q7, 10 items)

1. Before A09: no on/off-principal weight-direction claim (either side, LoRA
   included); never cite the retracted OverlapLift.
2. Before the off-KD control resolves: do not attribute the OPD signature to
   on-policy; mechanism language stays non-causal (co-locate / consistent with).
3. Before double-ER: the direction reversal vs 2605.30524 is "observed under the
   whitened construct; construct comparability pending".
4. Before θ_r: ER's status is "the only measured observable that cleanly
   separates the arms", not "the correct observable".
5. Any "compression" statement must name its construct (whitened output-relevant
   vs raw representational); no bare "compression is good/bad".
6. L18: the layer-selection criterion (largest discriminant effect) is
   pre-registered and the full-layer profile must be reported; L18-only reporting
   is disallowed.
7. n-honesty: one model, one dataset, one seed, ≤3 arms — every claim is
   "in this setting"; until CIs land, n=10 trajectory correlations are diagnostic.
8. The capability margin must ship with the C09 output-control decomposition and
   inference-cost note (OPD responses ~4k tokens at step_624; trunc 0.92 @16384).
9. No "Cycle N" language in the paper's main narrative.
10. Never compare 1.7B-series and 4B-series numbers.

## Execution Discipline (Q8 + E5)

| Tier | Content | Nature | Paper role |
|---|---|---|---|
| **A** (weeks 1–2, blocking) | B09 CIs; double-ER; L18 layer-resolved re-derivation + dip co-location adjudication; UV re-probe + θ_r; C09 output-control decomposition | analysis only | primary-contribution core |
| **B** (parallel, best-effort) | A09 (adapter-first, retrain fallback); off-KD control arm | ≤2 training runs | (a′) + (c)-upgrade channel; late → auto-downgrade, never blocks submission |
| **C** (after Tier A) | theory finalization + paper rewrite per the layered framing | writing | — |

**Contingency (pre-registered):** any Tier B miss → its contribution downgrades
automatically ((a′) → future work; (c) stays hypothesis); Tier A analysis and the
submission decision must not wait on Tier B.

**Mini-cycle protocol (E5):** Tier A analysis runs as
theory → coder → theory fast loops ("mini-cycle09"); coder outputs land under
`local_experiment_results/cycle_09.../mini/` for provenance; theory may hold
provisional readings. **Guardrail:** nothing enters claims_allowed /
current_picture without passing Result consolidation (batchable at Tier A close)
— the fast loop accelerates theory iteration, it does not waive the claim gate.
Full training runs (Tier B) take the complete theory → exp → coder → result path.

## Mini-Round 1 Adjudication (2026-07-08, E5 fast loop — provisional pending Result consolidation)

Inputs: `local_experiment_results/cycle_09.../run_01/mini/` (T1–T4, F1–F5 flags).
Rulings applied per the pre-registered trees; probed layers this round: {9, 18, 27} only.

### R1 (H1/F3 → E1 rule applied): the "SFT early off-principal reorientation" leg is RETIRED on the process track

All three adapter-product (BA) tracks — `sft_fp32_BA`, `sft_bf16_BA`, `opd_top32_approx` —
read **weakly on-principal at every step for both arms** (ρ²_U ≈ 0.02–0.026, z +5..+9, k=32).
The bf16-merged track's SFT step_5 off-principal (z=−6.7) does not replicate in any BA
track. Since `sft_bf16_BA` ≈ `sft_fp32_BA` (bf16 rounding of the update itself is harmless),
the corruptor is the **merge−subtract-vs-base pipeline**, concentrated where updates are
smallest (early steps).

Per E1 (attribution lives on the process side): the moat's "ρ: steps 5–10 off → flips on"
leg is retired as a claim; it is re-scoped as a **deployment-track measurement finding**
feeding the methods contribution ("merge−subtract on bf16 artifacts flips early-step
directional verdicts"). The three-space chain is restructured:

```text
Evidence legs:   activation transient (layer-scoped ER uptick, both arms)
                 + behavior (dip timing/recovery; capability CIs; OOD outcome)
Null/critique leg: clean-track weight direction does NOT discriminate the channels
                 (both weakly on-principal at k=32) — reinforcing the framing-(b)
                 thesis that the discriminating observable is activation-spectral.
```

OPD's clean track is an approximation (adapter pruned; top-32 SVD of merged−base);
the final OPD process-track ruling waits on A09 retrain — **A09 priority raised**.

### R2 (H2/F2 → Q4 branch fired): dip co-location is REAL and layer-scoped; "OPD no bump" was a mean-over-layers artifact

At L18 and L9, **both arms** show a dip-co-located ER uptick (OPD +1.3 ER @step_5;
SFT +3.6 ER @step_20 at L18); at L27 neither. D08's "OPD contracts monotonically, no
bump" was L27's monotone decline washing out the mid-layer upticks in the mean.
Q4's pre-registered branch **"co-location retained, scoped to mid layers"** fires; the
channel-heterogeneous-dip branch does not. The mechanism statement UNIFIES:

```text
Capability dips co-locate with mid-layer ER upticks in BOTH channels.
The channels differ in transient magnitude (SFT ≈ 3× OPD at L18), timing
(step_20 vs step_5), recovery span (step_160 vs step_20), and endpoint
compression depth (L27 divergence, OPD lower). Transient magnitude — not
compression amount — tracks the OOD outcome (SFT larger break → MMLU-Pro
erosion; OPD smaller break → preservation).
```

→ Result sync: `current_picture` "both dips co-locate" is CORRECT once scoped to
mid layers; D08's "no bump" wording must be corrected to layer-scoped.

### R2b (F1 → layer rule DECIDED by user, 2026-07-08)

The "largest discriminant" criterion is **rejected** (user ruling): it is dominated by
endpoint-ER divergence (picks L27) and misses the transient structure the paper is
about. The pre-registered criterion is now **clearest dip-transient** (ER-uptick ↔
capability-dip co-location with visible recovery), which selects **L18 as the single
headline layer** (L9 confirmatory). L27's endpoint divergence (40.5) remains reported
as supporting evidence inside the compression→OOD narrative — not a second headline.
Guardrails: (i) the full-36-layer profile (round 2) stays mandatory — as the
anti-cherry-pick exhibit and to check whether an unprobed layer shows an even clearer
transient (if so, report both; the criterion, not the layer index, is what is
pre-registered); (ii) the L18 uptick magnitudes are small (+1.3 / +3.6 ER) — round 2
must attach probe-bootstrap error bands to ER before the uptick is written as
established.

### R3 (H3 → E2 corrected; (a′) reshaped): diffuseness is a REAL LoRA property, not bf16 noise

Clean fp32 rank-32 SFT BA has ρ²_U(k=32) ≈ 0.024 — same level as bf16-merged. E2's
expectation ("pure BA should concentrate → high ρ²") is **falsified**: the 32 LoRA
directions mostly lie outside the base top-32 subspace, with only a small (but robust,
z>5) on-principal bias. E2's merge−subtract-noise diagnosis (rank 2121 vs true 32)
stands. **(a′) status:** SFT clean track = weakly on-principal, **consistent** with the
external full-param SFT-on finding — no contradiction on the SFT side. The
LoRA-vs-full-param contradiction now rests entirely on OPD's clean BA → A09 decides.

### R4 (H4/F4 → B09 guardrail applied): "deeper" is retired from the dip narrative

Paired CIs (B=4096; stable across draws 256/1024/4096): final +0.096 [+0.058,+0.134] ✓,
peak +0.114 ✓, AUC +84.1 ✓, non-term peak +0.180 ✓ — all exclude 0. **Dip-depth
+0.036 [−0.018,+0.090] includes 0.** The supported dip narrative is: OPD's transient is
**earlier** (step_5 vs step_20) and **shorter** (recovered by step_20 vs step_160) —
not "deeper/sharper". Capability superiority claims are now CI-backed.

### Stable additions (to provisional picture)

- C09 decomposition: late OPD accuracy is carried by **truncated-but-boxed** samples
  (non-term 0.9+, 85–92% boxed-before-cut, mean length ~15k @16384; finisher acc
  0.93–1.00). The Q7#8 inference-cost caveat is mandatory in the paper.
- CI practice: draws 256/1024/4096 indistinguishable → small draw counts suffice;
  report stability in the appendix (E4 spirit).

### Claim-boundary deltas (append to the Q7 list)

11. No "OPD dip deeper/sharper than SFT" claim — CI includes 0; only earlier/shorter.
12. No "SFT early off-principal reorientation" claim on the process track — it is a
    merge−subtract measurement artifact; usable only as a deployment-track methods finding.
13. Any ER bump / no-bump statement must be layer-scoped; mean-over-layers bump
    claims are disallowed (the mean hid both-arm mid-layer upticks).
14. No OPD process-track direction claim until A09 clean-BA lands (current OPD track
    is a top-32 approximation).

### Round-2 priorities (reordered by these rulings)

1. **A09 OPD clean-BA retrain** (H1/H3 hinge on it) — Tier B, now top.
2. UV re-probe + θ_r, focused on L18/L9 dip steps: does rotation discriminate where
   clean-track ρ does not?
3. Full-36-layer profile (validates the two-lens rule).
4. Double-ER (unchanged); off-KD control (unchanged).

## Addendum (2026-07-10): 2509.12235 Precise Reading — Threat #2 Downgraded; Intervention Experiment Deferred

Full-text reading of arXiv 2509.12235 v3 (PDF in workspace root) yields a
three-layer evidence assessment now governing our engage strategy:

1. **Observation layer (solid):** Σ changes during SFT/RL are ≤0.005 (their App F.2);
   weight-side principal-angle rotation is ≈0° at top ranks (k≤8), <1° for k=32–384,
   jumping to ~13.9° only at k≈512 (their Table 7). Rotation is real but lives in
   mid/tail ranks; top weight subspaces barely move.
2. **Protected-SFT layer (preliminary but genuine intervention):** training-time
   orthogonalization against top-k singular-vector subspaces preserves late-SFT OOD
   (their Table 6, Fig 11) — a forward intervention, not rollback-confounded. They
   label it preliminary (App A).
3. **Restoration layer (§4.3, Figs 5–8 — does NOT support the causal claim):** given
   their own Σ-stability, vector-restoration is algebraically ≈ rolling back the
   analyzed matrices (W_QKV+MLP only; o_proj/embeddings/LN stay at End) to the
   MaxOOD checkpoint; value-restoration ≈ identity. Their own numbers confirm: the
   restored LLaMA profile lands at the MaxOOD profile (ID → ~20 vs MaxOOD-era
   ~25–30; OOD → ~15–17 vs MaxOOD 17.5; Qwen anchors from their Table 3:
   MaxOOD ID≈20–25/OOD≈17.5–19.7, End ID 46.6/OOD 14.1). No reversion control, no
   magnitude-matched control; on Qwen the OOD does not even recover under the
   partial rollback. **The strong causal reading "OOD follows rotation, not values"
   rests only on this layer and does not hold.**

**Engage wording (paper):** acknowledge layers 1–2; critique layer 3 as
"indistinguishable from a partial checkpoint rollback under their own measured
spectral stability (≤0.005), lacking rollback and magnitude-matched controls."

**Consequences:**
- The rotation-vs-values pressure on our ER observable is downgraded: our reply is
  the analytical critique above plus our own two-space data (activation-side θ_r
  discriminates the arms at 17–20°; weight-side θ_r^W expected ≈0 under LoRA by
  Davis–Kahan — the cross-space contrast itself supports the observation-space
  thesis, echoing their Table 7 top-rank ≈0°).
- **The factorized weight-space intervention (swap experiment) is DEFERRED (user
  ruling, 2026-07-10):** it intervenes on the null/critique leg, not the theory's
  core observable; its proper place is a post-main-theory whitened-vs-parameter-space
  comparison, at much smaller scope. Recorded as a post-theory candidate together
  with a protected-SFT-style forward intervention (suppress the mid-layer transient
  during training — Tier B, training). The analytical critique itself requires no
  experiment.

## Mini-Round 3 Provisional Rulings (2026-07-11 — pending coder handin + Result consolidation)

Inputs: `mini/R3_*` artifacts. Applied per pre-registered trees; new standards from
the Q1–Q5 discussion (recorded in `geometry_metric_definitions.md` Round-3 additions).

**P-R1 — L18 story FINAL (sample-level).** True probe-sample bootstrap (n=32
windows × 256 draws): OPD step_5 uptick +0.83 [0.65, 1.08], SFT step_20 +4.76
[3.70, 5.70]; **all 14 module cells exclude zero**. Magnitude ratio ≈5.7× (updates
the r2 ~10× estimate). Module localization: MLP (gate/up/down) + o_proj carry the
transient; k/v smallest. R2's co-location claim is now established at the
pre-registered evidence standard.

**P-R2 — Same-domain eviction hypothesis FALSIFIED; break-not-movement holds
within-OOD.** X-conditioned whitened ER (L18, endpoint): OPD contracts most under
OOD conditioning (X_ood −21.3 > X_math −17.3) yet preserves OOD behavior; SFT
barely contracts X_ood (−4.7) yet erodes it — and SFT's X_ood spectrum shows its
own small uptick at steps 20–40. Interpretation ruling: **ER is an organization
statistic, not a capacity meter**; OPD's domain-universal smooth contraction is
the signature of function-level reorganization (consistent with distillation
toward the teacher's computation), not domain-level deletion. Direction-resolved
closure (does the retained subspace cover OOD-needed directions) pends R3-8
(tail/r_ε + output drift + output-space decomposition). New largest arm
divergence recorded: L27 X_math_hard — OPD −86.7 vs SFT **+17.4 (expansion)**.

**P-R3 — θ_w near-floor confirmed; two-space claim SPLIT per the Q4 standard.**
Weight-side rotation: SFT at the 1.23° identity floor through step_40, 2.35° at
624; OPD-approx 4.60° at 624 (top-32 caveat). Magnitude contrast vs activation
side (17–23°) is established **descriptively**; any "observation-space
superiority" claim awaits the paired discriminability test (R3-9). Claim
boundary #15 below.

**P-R4 — IFEval is a second discriminating OOD axis (same direction as
MMLU-Pro).** prompt_strict: OPD 0.272→0.316 (+4.4 pts, peak 0.355@160); SFT
0.272→0.214 (−5.7 pts). TruthfulQA flat (preservation check, as at 1.7B). IFEval
stays pre-registered as a preservation axis — corroborating evidence, not a new
primary gate (Cycle 08 prohibition #9).

**P-R5 — ID axis completion (unified cap).** numina @12288 (cap-robust by paired
pilot; N=200): OPD 0.64/0.67/0.61 vs SFT 0.54/0.535/0.56 at steps 40/160/624 —
OPD leads at every measured step (+0.05 to +0.135). **AIME24 avg@10** (N=30,
10 seeds, cap 24576, secondary): OPD 0.270@320 / 0.240@624 vs SFT 0.187@624
(+0.053). Mandatory caveat: AIME is truncation-dominated (OPD trunc 0.97, SFT
0.80) and the margin is partly output-control-carried (boxed-before-trunc: OPD
0.52 vs SFT 0.025) — report only with the C09 decomposition, never as a clean
capability number.

**P-R6 — Behavioral micro-transients near dip steps (exploratory).** IFEval local
minima: OPD 0.251@step_10 (one checkpoint after its geometry dip), SFT
0.253@step_20 (at its geometry dip); SFT MMLU-Pro dip @40; SFT's S/X/H H/ood ER
transient −45.7 @step_20. Suggestive cross-space alignment, but individual deltas
are ~1σ (IFEval SE ≈ 2 pts, single seed) — pends R3-10 (prompt-level paired
bootstrap + transient co-location matrix); exploratory, never a gate.

### Claim-boundary additions (append to the Q7 list)

15. "Observation-space superiority" (whitened vs raw; activation-θ vs weight-θ)
    may not be claimed until the paired same-scale discriminability test passes
    (spec in `geometry_metric_definitions.md`); until then such statements are
    descriptive ("construct-specific signal", "magnitude contrast").
16. No eviction-hypothesis claim in either direction without direction-resolved
    metrics (R3-8): ER contraction alone cannot show which directions survived.
17. Behavioral transient co-location (P-R6) is exploratory; it must never be
    written as a confirmed cross-space chain without the R3-10 statistics.
18. AIME24 numbers carry the truncation/output-control caveat inline wherever
    cited (trunc ≥0.8 both arms; margins partly boxed-before-trunc-carried).

### Round-4 amendments (2026-07-12, windowing-v2 design discussion)

- **P-R1 scope amendment:** the L18 uptick bands were measured on first-window
  (question + answer-opening) probes — internally valid, but demoted to **pilot
  tier** per the evidence-tier ruling; paper-grade status requires v2
  random-window re-derivation (expected robust; must be verified).
- **P-R2 amendment:** the cross-domain contraction ordering ("OPD contracts most
  under OOD conditioning, −21.3 > −17.3") is confounded by probe text nature
  (generated CoT vs question text) — retire the ordering claim; the load-bearing
  facts are within-probe: the arm contrast on E_ood (OPD −21.3 vs SFT −4.7) and
  each probe's trajectory shape. The same-domain eviction falsification stands on
  the within-probe evidence.
- Claim boundary 19: probe-based paper numbers must come from the v2 windowing
  standard (spec in `geometry_metric_definitions.md` Round-4 section); v1-based
  geometry readings may appear only as pilot-tier with their window-scope stated.
- Claim boundary 20: no cross-domain contraction-magnitude comparisons across
  probes of different text nature (S vs E types).

## Mini-Round 5 Provisional Rulings (2026-07-14 — pending θ_{r_ε}, bootstrap CI, and Result consolidation)

Thesis was frozen 2026-07-13 (see `paper_thesis_framework.md`); Round 5 tests the
two lines it implies. **A2 (θ_{r_ε}) and A4 (bootstrap CI) are still running** —
rulings below are provisional and CI-free unless stated.

### P5-1 — Functional compressibility: OPD wins on ALL FIVE probes (strongest result to date)

r_ε change vs base (ε=0.05, L18, cross-module mean, endpoint):

| Probe | OPD | SFT | ratio |
|---|---|---|---|
| Training-domain CoT | −10.4 | −4.9 | 2.1× |
| E_ood (MMLU-Pro text) | **−24.3** | −6.4 | 3.8× |
| E_general (wikitext) | −17.6 | −3.3 | 5.3× |
| E_math_hard (AIME statements) | −16.9 | −3.3 | 5.1× |
| S_bos (unprompted free generation) | −10.8 | −3.8 | 2.8× |

**Zero exceptions across five probes**, including S_bos — the most task-neutral
probe available. The compressibility gain is **domain-universal**, not a
training-domain artifact. This is the paper's most robust finding and it upgrades
the thesis's core clause from "on external text" to "across all probed domains".

### P5-2 — NEW: domain-specific co-location; SFT's transient SPREADS outward

SFT's ER-uptick step, per probe, mapped to the behavioural dip it co-locates with:

| Probe (domain) | SFT ER peak | Co-located benchmark dip |
|---|---|---|
| Training-domain CoT | **step 20** (+2.66) | **MATH500 (ID) @20** |
| E_ood (knowledge) | **step 40** (+1.18) | **MMLU-Pro @40** |
| E_math_hard | **step 40** (+0.88) | MMLU-Pro @40 |
| E_general | step 160 (+0.13) | IFEval @160 |

The transient is **domain-specific and temporally ordered**: it appears first in
the training domain (step 20), then in OOD-knowledge probes (step 40), then in
general text (step 160) — an **outward spread** from the trained domain. Each
geometric transient co-locates with the capability dip *of that same domain*.

**OPD shows no positive ER peak on any E-type probe** (peak heights are negative =
monotone contraction); its only upticks are weak and late (S_bos/training-domain
@160) and co-locate with nothing.

⚠️ This pattern was missed for several rounds because the analysis kept focusing on
the training-domain probe alone. **Methodological rule adopted: any mechanism claim
must first be laid out across ALL probes × ALL benchmarks before a headline is
chosen.**

### P5-3 — MMLU-Pro / GPQA dip at step 40 is SHARED by both arms (honesty item)

Both arms dip on MMLU-Pro and GPQA at step 40; **OPD's dip is deeper**
(−0.084 vs −0.036). This is not a channel difference — candidate explanation is a
shared training-schedule/epoch boundary. The arms differ in **recovery**: OPD
returns above base (net +0.016), SFT declines again (net −0.029). Any "OPD is
undamaged throughout" phrasing is disallowed.

### P5-4 — H-mismatch (B-line) : the borrowed exposure-bias framing failed, not the quantity

The pre-registered prediction (SFT's Mismatch peaks at its dip step; OPD flat)
failed on both counts: SFT peaks at step 160, OPD carries the **larger** mismatch
throughout, and cell-level SD (16–34) dwarfs the means (5–19).

**Ruling (user-directed, 2026-07-14):** what failed is the **exposure-bias framing
imported to interpret the quantity**, not the measurement. Mismatch =
r_ε(own output) − r_ε(static reference) measures a **width difference in functional
dimensions**: OPD's persistently positive Mismatch (+5…+19) means its own
generations *need more dimensions than the dataset CoT* — consistent with its
4k-token long-CoT rollouts (information injection), not with "misalignment".
Decisive counter-evidence against the exposure-bias account: **both arms dip**,
yet OPD is on-policy and should have no exposure bias.

**Replacement framing (user-proposed, to be tested):** the dip is a
**compression-scheme transition** — the old (base) functional compression breaks
before the new (task) one is established; during that window r_ε rises (SFT +3.9)
or has not yet begun to fall (OPD −0.6), and capability dips. The arms differ in
the **speed** of the transition and the **quality of the scheme they land on**
(OPD: dip@5→recovered@20, lands at r_ε −25; SFT: dip@20→recovered@160, lands at −4.9).
**Status: hypothesis, not yet tested** — θ_{r_ε} (running) is its main test.

### P5-5 — Two "mismatch" metrics point opposite ways because they measure different things (F-A2)

| | Mismatch (B-line) | xs_log_spectrum_gap (A5) |
|---|---|---|
| formula | r_ε(own) − r_ε(static ref) | mean(log σ_X − log σ_S) |
| measures | **width** (how many directions) | **height** (spectral energy level) |
| SFT | ≈0 throughout | **peaks at step 20 = its dip step** (.0795) |
| OPD | +5…+19 (larger) | .02–.04 (smaller) |

No contradiction — one measures dimensional width, the other spectral energy.
**Terminology fixed: "mismatch" may no longer be used as an umbrella term.**
Load-bearing reading: SFT's xs_gap peaking **at its own dip step** is a
within-arm shape fact (same text, only weights vary) and is valid. Cross-arm
xs_gap magnitudes are **not** comparable (OPD's X = its own rollouts, SFT's X =
dataset CoT — different text kinds).

### P5-6 — γ_{r_ε} is dominated by noise at r_ε; the informative gap lives at the spectrum's top (F-A1)

At r_ε (≈632 of 2560) the spectrum is nearly flat: γ ≈ 0.0014 while σ ≈ 0.45 —
i.e. **we are differencing two numbers that differ by ~0.3%**, under sampling
noise (n=32), SVD precision, bf16 weights, and a moving r_ε. The resulting
swings (−12%, +91%, −46%) are not credible. **The earlier reading ("γ collapses at
the dip, then rebounds +91%") is RETRACTED.**

At **k=64** (the spectrum's steep region) the signal is clean: OPD's γ rises +137%
vs SFT's +42% — OPD's functional/redundant boundary at the spectrum's top becomes
markedly sharper. **Compression *quality* should therefore be judged at the
spectrum's top, not at the flat r_ε boundary.** Robust redefinition (window-averaged
gap / local log-spectrum slope, plus bootstrap) is queued for the next round.

### Claim-boundary additions

21. Compressibility claims must be stated across all five probes (they hold with
    zero exceptions); single-probe compressibility claims are underpowered.
22. The step-40 MMLU-Pro/GPQA dip is shared by both arms — never present it as an
    SFT-specific failure; the arm difference is in recovery, not in dipping.
23. "Mismatch" is not an umbrella term: Mismatch (width) and xs_gap (height) are
    distinct quantities and may not be conflated.
24. No γ_{r_ε} reading may be cited until the robust redefinition + CI land;
    γ at the spectrum's top (k=64) is the currently usable form.
25. The compression-scheme-transition account of the dip is a HYPOTHESIS
    (θ_{r_ε} pending); the exposure-bias account is falsified (both arms dip).


---

## Off-KD control arm (Tier B) — behavioral rulings (2026-07-16, geometry pending)

**Run**: off-KD = teacher(Qwen3-8B) static responses + forward_kl_topk(32), same base /
LoRA(r32/α64/all-linear) / lr / 5k prompts / 624 steps as OPD; the only intended variable
is the response source (student on-policy → teacher static). Full 10-point grid evaluated
on MATH500 / Numina / AIME24 / MMLU-Pro / GPQA / IFEval / TruthfulQA.
**Provenance caveats (inherited from coder flags, must accompany any citation)**:
F-offKD-1 trainer is a config-aligned HF+PEFT loop, not the verl runner — state it as a
"configuration- and loss-path-aligned off-policy control", never "identical runner";
F-offKD-2 steps 80/320/480 are numerical backfills; F-offKD-3 OPD/SFT rows reuse earlier
formal evals (step-0 differences are eval noise, not training effects). Single seed,
point estimates — no CIs yet.

### P6-1 — The ID gain is carried mostly by dense soft targets; on-policy adds the rest

MATH500 endpoint: OPD .848 > off-KD .794 > SFT .752; peaks .866/.800/.752. The soft-target
arm without on-policy recovers ~55–65% of OPD's advantage over SFT on ID math.
(Numina endpoint even favors off-KD: .620 vs OPD .610 vs SFT .560 — 3-point grid only.)

### P6-2 — OOD retention/recovery is attributable to on-policy sampling (the causal cell we lacked)

MMLU-Pro: OPD dips (@40–80, .399/.361) then **fully recovers** (.511@480); SFT roughly holds;
**off-KD collapses monotonically and never recovers** (.478 → .354, −12.4 pts, lowest of the
three arms by >10 pts). IFEval reproduces the same three shapes (OPD ↑ to .392@80; SFT drifts
down; off-KD cliff at step 40, .314 → .176, partial recovery only). Since off-KD shares the
teacher, the loss, and the soft targets with OPD, **the OOD damage is caused by removing
on-policy sampling** — soft targets alone do not protect OOD; hard-label SFT on the original
dataset is actually safer for OOD than off-policy distillation on teacher text.
⚠️ This is the strongest causal statement the three-arm design licenses at the behavioral
level; CI still pending, keep as point-estimate ruling.

### P6-3 — Non-termination is an on-policy-specific pathology; dip timing follows supervision type

Termination: off-KD is the only arm that ends clean (MATH500 trunc .048 @624, mean length
1.7k — shorter than base) while OPD ends at trunc .918 / 15.3k with the same loss. The
teacher corpus (78.6% natural EOS) teaches stopping; on-policy training amplifies
non-termination instead.
Dip timing: OPD dips @5, off-KD dips @5–10, SFT dips @20 — the two KL arms dip together
and early, regardless of on/off-policy ⇒ **dip timing tracks the supervision-signal type
(dense soft targets vs hard labels), not the sampling policy**. This further narrows the
dip mechanism space after H-transition's main test failed.

### Geometry gate (open)

r_ε attribution for the third arm is pre-registered and frozen in
`theory/offkd_geometry_handoff.md` §4 (G1 soft-targets / G2 on-policy / G3 split, with
fixed metrics and majority rule) — no geometry interpretation may precede those readings.
H-offkd-transient (non-recovering E_ood transient co-located with non-recovering MMLU-Pro)
is pre-registered there as well.

### Claim-boundary additions

26. Any off-KD claim must carry F-offKD-1/2/3 provenance caveats; "identical runner" may
    not be asserted.
27. P6-2's causal attribution (on-policy → OOD retention) is behavioral and point-estimate;
    it may be cited as the design's headline causal cell only after CIs, and never as a
    geometry claim until the pre-registered tree (§4 of the geometry handoff) is adjudicated.
28. Off-KD's Numina endpoint advantage (.620) rests on a 3-point grid — trajectory claims
    on Numina are not licensed for the off-KD arm.

### P6-4 — Pre-registered geometry adjudication: G1 (off-KD compresses like OPD) — compression is a soft-target signature, and it DISSOCIATES from capability

Frozen rule executed mechanically (d = r_ε(624)−r_ε(0), L18, ε=0.05, nearest-reference,
majority over five probes): **5/5 probes fall on the OPD side**
(d_offkd = −14.3 / −18.6 / −15.0 / −14.3 / −9.0 vs d_OPD −10.4…−24.3, d_SFT −3.3…−6.4).
On the training domain off-KD compresses *more* than OPD (−14.3 vs −10.4).
Frozen consequence now in force: **the five-probe compressibility gain is driven by the
dense soft-target (KL) signal, not by on-policy sampling** — and since off-KD compresses
like OPD while its OOD behavior collapses (MMLU-Pro −12.4 pts, unrecovered), **endpoint
compression is decoupled from capability retention in the third arm**. The C-chain
(compression ↔ OOD/capability) is downgraded to an OPD-specific correlation; causal
phrasing is forbidden. Ruling is at the pilot tier (point estimates; keep_factors was off,
no sample-level CI; CI backfill ≈ 4–6h GPU).

### P6-5 — H-offkd-transient FALSIFIED; what survives is transient-amplitude ordering

My pre-registered prediction (off-KD's E_ood geometric transient would not recover,
mirroring its unrecovered MMLU-Pro) failed under its own frozen criterion: the transient
(+2.42 @20) fully recovered and overshot into deep compression (endpoint −14.29) while
MMLU-Pro never recovered — another geometry/behavior dissociation.
Recorded (not claimed): (i) transient-amplitude ordering OPD(none) < SFT(+1.82@40) <
off-KD(+2.42@20) matches OOD damage ordering (recovers / holds / collapses), consistent
with the mini-r1 ruling "transient amplitude, not compression amount, tracks OOD" — n=3
ordering, point estimates; (ii) off-KD's OOD damage window (steps 20–40) coincides with
its r_ε *expansion* phase, while its bulk compression (160→624) happens after MMLU-Pro
has already flattened at the bottom; (iii) F4: off-KD's M2(X0) endpoint is the highest of
all three arms (0.0312 vs 0.0244/0.0245) — movement doesn't explain the capability
ordering either.

### Claim-boundary additions

29. The five-probe compressibility gain may no longer be attributed to on-policy sampling;
    it is a soft-target/KL signature (G1, pilot tier). Any "compression ↔ capability"
    statement must carry the third-arm dissociation.
30. H-offkd-transient is falsified; geometric-transient recovery may not be cited as
    predicting behavioral recovery.
31. Transient-amplitude ↔ OOD-damage ordering (3 arms) is an observation, not a claim,
    until CIs; it is the strongest surviving geometry–behavior link.
