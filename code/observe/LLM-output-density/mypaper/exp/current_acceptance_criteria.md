# Current Acceptance Criteria

```yaml
view_type: current_latest_view
cycle: cycle_08_h_opd_vs_sft_comparison
last_updated: 2026-06-29
qa_source: exp/QA_cycle08_exp.md
```

---

# Acceptance Criteria 08: H-OPD vs SFT Trajectory

Gate priority:

```text
Phase0 viability > B08 > C08 > D08 > A08
```

Phase0 determines whether the official OPD arm can start from `step_000` or from
a minimal pre-dip cold-start. B08 is the blocking scientific gate. C08 is the
primary preservation gate. D08 and A08 are diagnostic/mechanism gates and cannot
rescue a B08 failure.

---

## Phase0 Gate — Start-Point and Implementation Viability

**Question:** Can Cycle 08 run the intended OPD arm from the strict base state, or
does it require a minimal pre-dip cold-start?

### Overlap Viability

Measure top-k overlap against Qwen3-8B-Instruct:

```text
top_k = 32
candidates = step_000, step_005, step_010, step_020 diagnostic
viability threshold = mean top-k overlap >= 0.70
```

Decision:

| Pattern | Decision |
|---|---|
| `step_000` viable and smoke stable | Start official OPD from `step_000`; strict H-OPD test. |
| `step_000` not viable, earliest of `step_005`/`step_010` viable | Use earliest viable pre-dip cold-start; conditional H-OPD test. |
| no pre-dip checkpoint viable | Stop and report. Do not automatically use `step_020` or later. |

### Implementation Viability

verl OPD path must demonstrate in a short smoke:

- teacher path resolves to Qwen3-8B-Instruct;
- rollout and teacher top-k logprob query run;
- loss/KL values are finite;
- memory does not OOM;
- sample generations are not obviously corrupted.

If verl is unavailable or blocked, Code must stop and report. Do not silently
switch to TRL.

---

## Gate B08 — Math Capability and Dip Avoidance (Primary, Blocking)

**Question:** Does OPD match SFT's recovered MATH500 endpoint and avoid or reduce
the step_20-level SFT capability dip?

### B08.1 Final Non-Inferiority

Use as-run MATH500 accuracy.

```text
SFT final = 0.680
SFT final SE = 0.0209
threshold = 0.680 - 0.0209 = 0.6591
pre-registered numeric threshold = 0.660

Pass if:
  OPD final MATH500 as-run >= 0.660
```

### B08.2 Dip Severity

Use as-run MATH500 as the primary dip metric. Report finisher accuracy and
non-termination as decomposition diagnostics.

Reference:

```text
SFT step_20 as-run MATH500 = 0.572
SFT step_20 finisher acc = 0.594
SFT step_20 non-term = 6.8%
```

Verdicts:

| Pattern | B08 verdict |
|---|---|
| final >= 0.660 and OPD min as-run >= 0.636 | **STRONG PASS**: endpoint matched and no below-base dip. |
| final >= 0.660 and 0.572 < OPD min as-run < 0.636 | **PASS / mitigated dip**: endpoint matched and SFT-level trough avoided, but a shallow dip remains. |
| final >= 0.660 but OPD min as-run <= 0.572 | **PARTIAL / mechanism fail**: endpoint matched but OPD reproduces an SFT-level trough. |
| final < 0.660 | **FAIL**: OPD configuration is not capability-compatible with SFT_recover. |

Finisher/non-term rule:

- If as-run and finisher readings agree, report the B08 verdict directly.
- If as-run passes but finisher acc reproduces the SFT step_20 capability trough,
  report B08 as mixed rather than strong.
- If B08 fails badly, D08/A08 positives cannot be used to claim H-OPD.

---

## Gate C08 — MMLU-Pro Preservation (Primary Preservation Axis)

**Question:** Does OPD preserve MMLU-Pro better than SFT at the final checkpoint?

Use the same MMLU-Pro protocol as Cycle 07:

```text
no chat template, no thinking mode
0-shot lm-eval
100/class, 14 subtasks = 1400
seed = 42
```

Reference:

```text
SFT MMLU-Pro delta = step_624 - step_000
                    = 0.4621 - 0.4907
                    = -0.0286
```

Pass condition:

```text
OPD MMLU-Pro delta > -0.0286
```

Verdicts:

| Pattern | C08 verdict |
|---|---|
| OPD delta > SFT delta by at least 1 OPD/SFT SE scale | **STRONG PASS** |
| OPD delta > -0.0286, but margin is small | **PASS** |
| OPD delta approximately equals SFT delta | **MIXED / no preservation edge** |
| OPD delta < -0.0286 | **FAIL** |

MMLU-Pro is the pre-registered preservation axis, but not the whole of OOD.
Supplementary benchmarks may contextualize C08 after the core run, not replace it.

---

## Gate D08 — Geometry Transient (Diagnostic)

**Question:** Does OPD show the same geometry transient as SFT step_20?

Reference SFT pattern:

```text
argmin MATH500 = step_20
argmax mean_effective_rank = step_20
argmin mean_xs_log_spectrum_gap = step_20
```

Supportive D08 evidence:

- OPD has no severe MATH500 trough and no co-located rank/gap transient.
- OPD has a shallower or shifted geometry transient relative to its capability
  low point.
- OPD drift_from_base does not merely reproduce SFT's monotone negative control.

Mixed or negative D08 evidence:

- OPD reproduces the same triple co-location at the same early dip.
- OPD geometry differs but B08 behavior does not improve.
- Geometry signals disagree with each other.

D08 is diagnostic only. It cannot rescue B08 failure.

---

## Gate A08 — OverlapLift and Random-LoRA Reference (Diagnostic)

**Question:** Does OPD's OverlapLift trajectory differ systematically from SFT,
and how does each compare with a random-LoRA reference?

Required comparisons:

1. OPD vs SFT matched-LoRA OverlapLift trajectory.
2. OPD/SFT vs random-LoRA reference with the same LoRA rank/targets and
   checkpoint-matched scale where feasible.

Supportive A08 evidence:

- OPD differs from SFT around the SFT dip/recovery landmarks in a direction
  consistent with B08/C08.
- OPD is distinguishable from random-LoRA reference at the same checkpoint scale.

Boundary:

- Cycle 08 may report random-reference-normalized OverlapLift.
- Cycle 08 must not make direct principal-angle or measured subspace-rotation
  claims; those are registered for Cycle 09.
- A08 cannot rescue B08 failure.

---

## Supplementary Evaluation Gate (Deferred)

Supplementary capability-degradation tests are selected only after the core OPD
trajectory is known.

Allowed process:

1. Complete core OPD trajectory and core gates.
2. User selects checkpoint pairs and supplementary tasks.
3. Code runs only those selected OPD/SFT checkpoint pairs.

Candidate tasks:

```text
MMLU, HellaSwag, WinoGrande, ARC-Challenge, PIQA, TruthfulQA-MC1
```

Supplementary results are never allowed to replace B08/C08 after the fact.

---

## Completion Conditions

Cycle 08 core is complete only if all required artifacts exist:

1. `phase0/overlap_probe.csv`
2. `phase0/phase0_smoke.md` or equivalent JSON
3. `phase0/selected_start.json`
4. OPD checkpoints on the agreed grid after the selected start
5. core evaluation outputs for all OPD checkpoints
6. `trajectory_scores_opd.csv`
7. `nontermination_trajectory_opd.csv`
8. `response_length_trajectory_opd.csv`
9. geometry CSVs and OverlapLift outputs for OPD checkpoints
10. random-LoRA reference outputs
11. `RESULTS_08_CORE.md`
12. `run_provenance.json`

---

## Run Invalidation Conditions

A run is invalid or blocked if:

- teacher is not Qwen3-8B-Instruct or an explicitly approved same-path equivalent;
- lambda is not `1.0`;
- OPD silently falls back to TRL without user approval;
- the official OPD arm starts at `step_020` or later without user confirmation;
- MATH500/Numina uses a larger cap than the Cycle 07 as-run protocol;
- GPQA-D or MMLU-Pro uses chat template;
- OPD checkpoint labels cannot be mapped to the shared absolute grid;
- Phase0 artifacts are missing;
- random-LoRA reference is omitted without a documented blocker;
- supplementary tasks are run before core results and user checkpoint selection.

---

## Pre-Registered Allowed Conclusions

| Result pattern | Allowed wording |
|---|---|
| B08 strong pass + C08 pass + D/A aligned | "In this Qwen3-4B-Base + Math-CoT-20k matched-LoRA setting, OPD matches the recovered SFT endpoint while reducing the SFT dip/preservation cost, with aligned trajectory-level mechanism signals." |
| B08 pass, C08 fail/mixed | "OPD can match math capability and avoid or reduce the severe SFT dip, but a broader preservation advantage is not supported by MMLU-Pro." |
| B08 endpoint pass, dip partial | "OPD reaches the recovered endpoint but still shows an early capability transient; H-OPD is weakened on the avoid-dip mechanism." |
| B08 fail | "This OPD configuration is not capability-compatible with SFT_recover under the chosen teacher/lambda/start/framework. Investigate setup before making OPD stability claims." |
| D/A positive but B08 fail | "Geometry/OverlapLift signals are diagnostic only and do not establish H-OPD without capability compatibility." |

## Not Allowed

- Do not claim OPD is better than SFT from a single checkpoint.
- Do not claim H-OPD is false from one failed configuration without separating
  teacher, lambda, start checkpoint, overlap, and implementation.
- Do not use a recover-point cold-start as evidence that OPD avoided the dip.
- Do not replace C08 with supplementary tasks after seeing results.
- Do not treat random-LoRA OverlapLift as a direct measured principal-angle claim.
- Do not write non-termination as a mere harness artifact.
