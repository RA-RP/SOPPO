# Current Experiment Design

```yaml
view_type: current_latest_view
status: active_cycle08_design
cycle: cycle_08_h_opd_vs_sft_comparison
last_organized: 2026-06-29
qa_source: exp/QA_cycle08_exp.md
```

---

# Experiment Design 08: H-OPD vs SFT Trajectory Comparison

## Core Questions

1. **B08, primary blocking:** Can one OPD arm match the recovered SFT endpoint on
   MATH500 while avoiding or reducing the SFT step_20 capability dip?
2. **C08, preservation:** Does OPD preserve MMLU-Pro better than the matched SFT
   trajectory under the same model/data/LoRA/checkpoint regime?
3. **D08, geometry:** Does OPD avoid, flatten, or shift the SFT step_20 geometry
   transient: argmin accuracy = argmax effective rank = argmin X-S log-spectrum gap?
4. **A08, OverlapLift:** Does OPD have a systematically different OverlapLift
   trajectory from SFT under matched LoRA, and where does it sit relative to a
   registered random-LoRA reference?

The primary comparison unit is the **matched checkpoint-grid trajectory**, not a
single selected checkpoint. Single checkpoints are landmarks: dip depth/timing,
recovery step, peak, final endpoint, response-length transient, and geometry
transient.

---

## QA-Confirmed Design Decisions

| Decision | Cycle 08 setting |
|---|---|
| Overall structure | Two-stage run: Phase 0 overlap/smoke probe, then Phase 1 main OPD arm. |
| Teacher | Qwen3-8B-Instruct, same family. If the local path is missing, stop and report; do not substitute another teacher. |
| OPD strength | `lambda=1.0` for the single official arm. |
| Cold-start preference | Prefer strict step_000 OPD if viable. If not, fallback must end before the SFT dip whenever possible. |
| Allowed automatic fallback | `step_005` or `step_010`, chosen as the earliest viable pre-dip checkpoint. |
| Not automatic | `step_020` is the SFT dip itself; use only after explicit user confirmation. `step_040+` is not valid for an automatic H-OPD avoid-dip claim. |
| Framework | Use verl async/on-policy KD as the primary implementation direction because OPD rollout cost is expected to be high. TRL is not an automatic fallback. |
| B08 metric | as-run MATH500 is primary; finisher accuracy and non-termination are required decomposition diagnostics. |
| C08 metric | MMLU-Pro remains the primary preservation axis; it is not treated as the whole of OOD. |
| Supplementary eval | Deferred until after core OPD results. Then choose a small set of OPD/SFT checkpoint pairs and tasks for supplementary capability-degradation tests. |
| A08 reference | Do both OPD-vs-SFT relative OverlapLift and a random-LoRA reference for absolute scale. |
| UV/principal angles | Direct UV/principal-angle or subspace-rotation claims are deferred to Cycle 09. Do not require full UV saves for Cycle 08 beyond what the existing OverlapLift pipeline needs. |

**Registered future idea:** the relationship between teacher-student overlap,
minimal cold-start, and dip avoidance may itself become a theory object. If Cycle
08 shows that only a narrow pre-dip cold-start window makes OPD viable, Cycle 09
should consider whether overlap probes can become a principled OPD cold-start
selection rule.

---

## SFT Reference Trajectory

Cycle 08 does **not** retrain the SFT arm. The reference is Cycle 07:

```text
local_experiment_results/cycle_07_base_sft_trajectory/run_01/
```

Key reference values:

| Landmark | Step | MATH500 as-run | Notes |
|---|---:|---:|---|
| base | 0 | 0.636 | Qwen3-4B-Base |
| dip | 20 | 0.572 | finisher acc 0.594, non-term 6.8% |
| first recovery | 160 | 0.670 | first above base |
| peak | 320 | 0.698 | best MATH500 |
| final | 624 | 0.680 | non-term about 50%, finisher acc about 0.98 |

Geometry reference:

```text
step_20 = argmin MATH500 = argmax mean_effective_rank = argmin mean_xs_log_spectrum_gap
```

MMLU-Pro reference:

```text
SFT final delta = MMLU-Pro(step_624) - MMLU-Pro(step_000)
                = 0.4621 - 0.4907
                = -0.0286
```

---

## Phase 0: Overlap and Implementation Probe

Phase 0 is part of the design, not an informal code convenience. Its purpose is
to decide whether the strict step_000 OPD test is viable without silently turning
an overlap/conditioning failure into an H-OPD failure.

### Candidate Start Checkpoints

Use Cycle 07 checkpoints as candidate initial states:

| Candidate | Status |
|---|---|
| `step_000` | preferred strict H-OPD test |
| `step_005` | allowed automatic fallback if viable |
| `step_010` | allowed automatic fallback if viable |
| `step_020` | diagnostic only by default; this is the SFT dip, so it needs user confirmation before use |
| `step_040+` | not allowed as automatic fallback for an avoid-dip claim |

### Overlap Probe

Use the selected Qwen3-8B-Instruct teacher and a fixed prompt subset from the
Cycle 07 Math-CoT-20k seed=42 training sample.

Required measurement:

```text
top_k = 32
candidate checkpoints = step_000, step_005, step_010, step_020 diagnostic
metrics = mean top-k overlap, teacher mass on shared top-k tokens,
          student entropy, teacher entropy, sample-generation notes
```

Viability rule:

```text
Preferred:
  use step_000 if mean top-k overlap >= 0.70 and the short smoke run is finite/stable.

Fallback:
  otherwise use the earliest checkpoint in {step_005, step_010} satisfying the
  same viability condition.

If no pre-dip checkpoint is viable:
  stop and report. Do not automatically use step_020 or later.
```

The 0.70 threshold is a design anchor from the OPD/cold-start literature; the
exact measured values must be reported, not hidden behind the verdict.

### Short OPD Smoke

After choosing the tentative start checkpoint, run a short verl OPD smoke only
to verify implementation viability:

- teacher path resolves to Qwen3-8B-Instruct;
- rollout, teacher top-k logprob query, and student update all run without OOM;
- loss/KL values are finite;
- generations remain parseable enough to continue.

The smoke is not a result. It should produce a `phase0_smoke.md` or JSON artifact.

---

## Phase 1: Main OPD Arm

### Model, Data, and Adapter

| Item | Value |
|---|---|
| Student/base | `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base` |
| Teacher | Qwen3-8B-Instruct local path, to be verified by Code |
| Training prompts | Math-CoT-20k, same 5000-row seed=42 sample as Cycle 07 |
| Prompt format | same user prompt as Cycle 07: problem + boxed-answer instruction, no system prompt |
| Student completion source | on-policy student rollouts |
| Adapter | LoRA r=32, alpha=64, dropout=0.05, targets q/k/v/o/gate/up/down_proj |
| OPD lambda | `1.0` |
| Main framework | verl async/on-policy KD style implementation, top-k KL if teacher server returns top-k logprobs |

The exact OPD loss formula and any beta / KL-direction choices must be recorded
by Code from the implementation actually used. Do not invent a paper-facing
formula in this design file.

### Checkpoint Grid

Use the same absolute budget labels as Cycle 07:

```text
[0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
```

If Phase 0 selects `step_000`, all nonzero checkpoints are OPD checkpoints. If
Phase 0 selects a pre-dip cold-start checkpoint such as `step_005` or `step_010`,
then:

- the selected checkpoint is recorded as `cold_start_step`;
- OPD checkpoints are saved at subsequent absolute-budget grid points;
- earlier grid entries are marked as inherited / not OPD, not silently filled in.

This keeps comparison against SFT budget landmarks interpretable while preserving
the cold-start caveat.

---

## Core Evaluation Protocol

Core Cycle 08 evaluation follows the **actual Cycle 07 run provenance**, not the
older pre-run draft that used a larger token cap.

Generative tasks:

| Task | Scope | Format | Max generation | Sampling | Output diagnostics |
|---|---|---|---:|---|---|
| MATH500 | full N=500 | chat template, think-format, no `enable_thinking=False` | 4096 | temp=0.6, top_p=0.9 | as-run acc, SE, response length, non-term rate, finisher acc |
| NuminaMath-test | N=256 as in Cycle 07 | same as MATH500 | 4096 | temp=0.6, top_p=0.9 | acc, SE |
| AIME24 | full N=30 | same as MATH500 | 16384 | temp=0.6, top_p=0.9 | acc, SE; qualitative only if non-term dominates |

Loglikelihood tasks:

| Task | Scope | Format | Protocol |
|---|---|---|---|
| GPQA-Diamond | full N=198 | no chat template, no thinking mode | lm-eval 0-shot |
| MMLU-Pro | 100/class, 14 subtasks = 1400, seed=42 | no chat template, no thinking mode | lm-eval 0-shot |

**Do not increase MATH500/Numina max generation from 4096 to 32768.** Cycle 07
cap-pilot showed 4096 is cap-robust for the scorer and is the valid as-run
task-success protocol.

Evaluate every OPD checkpoint that exists on the grid. Compare against Cycle 07
SFT results without re-running the SFT core suite.

---

## Supplementary Capability-Degradation Evaluation

Supplementary evaluation is **not run before the core OPD trajectory is known**.
The cycle is intentionally split:

1. **Core run:** OPD training + Cycle 07 core eval suite + geometry.
2. **Supplementary selection:** user reviews the OPD trajectory with the SFT
   trajectory, then selects a small set of OPD/SFT checkpoint pairs and tasks.

Candidate supplementary tasks:

```text
MMLU, HellaSwag, WinoGrande, ARC-Challenge, PIQA, TruthfulQA-MC1
```

Design rule:

- supplementary tasks can strengthen or weaken the preservation story;
- they do not replace C08's pre-registered MMLU-Pro primary gate;
- supplementary SFT measurements should only be run on the selected matching
  SFT checkpoints, not on the whole Cycle 07 trajectory by default.

---

## Geometry and OverlapLift

Geometry remains all-checkpoint for the core OPD trajectory.

### Standard Geometry Metrics

For each OPD checkpoint on the grid:

- layers `{9, 18, 27}`;
- modules `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`;
- probe rows drawn from the same Cycle 07 Math-CoT-20k seed=42 sample;
- metrics: effective_rank, spectral_gap, drift_from_base, X-S log-spectrum gap.

Reference for drift and X probe remains `step_000` Qwen3-4B-Base.

### OverlapLift

Run the existing principalEvidence / OverlapLift pipeline needed for:

1. OPD-vs-SFT relative matched-LoRA comparison.
2. OPD and SFT relative to a random-LoRA reference.

Random-LoRA reference:

- same base model;
- same LoRA rank, alpha, target modules;
- module-wise scale matched to the corresponding trained adapter where feasible;
- minimum one random seed per grid checkpoint, with more seeds optional if cheap.

Interpretation boundary:

- Cycle 08 may compare OPD, SFT, and random-LoRA OverlapLift values.
- Cycle 08 still does not make direct principal-angle / subspace-rotation claims.
- Full UV/sketch saves for measured principal angles are deferred to Cycle 09.

---

## Expected Outputs

```text
local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/
  phase0/
    overlap_probe.csv
    phase0_smoke.md
    selected_start.json
  checkpoints/
    opd_step_{NNN}/
  eval/
    step_{NNN}/...
  geometry/
    geometry_metrics_step_{NNN}.csv
    principal_evidence_step_{NNN}.csv
    random_lora_reference_step_{NNN}.csv
    geometry_summary.csv
    overlap_lift_summary.csv
  figures/
    trajectory_math500_opd_vs_sft.png
    trajectory_ood_opd_vs_sft.png
    nontermination_opd_vs_sft.png
    geometry_opd_vs_sft.png
    overlap_lift_opd_vs_sft_random.png
  trajectory_scores_opd.csv
  nontermination_trajectory_opd.csv
  response_length_trajectory_opd.csv
  RESULTS_08_CORE.md
  run_provenance.json
```

Supplementary outputs, if a second pass is launched after user selection:

```text
local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/supplementary/
  selected_checkpoints.json
  supplementary_scores.csv
  RESULTS_08_SUPPLEMENTARY.md
```

---

## Non-Goals

- No lambda sweep.
- No teacher sweep.
- No automatic recover-point cold-start.
- No retraining of the SFT reference trajectory.
- No post-hoc replacement of C08 with supplementary benchmarks.
- No direct principal-angle or subspace-rotation claim in Cycle 08.
- No TRL fallback unless the user explicitly approves it after a verl blocker.
