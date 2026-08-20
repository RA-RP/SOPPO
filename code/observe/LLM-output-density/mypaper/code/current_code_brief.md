# Current Code Brief — Cycle 08

```yaml
view_type: current_code_brief
cycle: cycle_08_h_opd_vs_sft_comparison
status: ready_for_coding_phase0_and_core
scope: opd_phase0_verl_core_eval_geometry
created: 2026-06-29
qa_source: exp/QA_cycle08_exp.md
exp_design: exp/current_experiment_design.md
acceptance: exp/current_acceptance_criteria.md
copyback_target: local_experiment_results/cycle_08_h_opd_vs_sft_comparison/
handin_target: code/code_evolution.md (append handin section)
```

## Before You Start — What to Read

```text
readme.md
↓ code/current_code_brief.md
↓ code/current_code_structure.md
↓ exp/experimental_methodology/ (4 files)
↓ exp/current_experiment_design.md
↓ exp/current_acceptance_criteria.md
↓ local_experiment_results/cycle_07_base_sft_trajectory/run_01/RESULTS_07.md
↓ local_experiment_results/cycle_07_base_sft_trajectory/run_01/run_provenance.json
```

**Cycle 08 override note:** if the stable `exp/experimental_methodology/` files still mention older Cycle04-07 details such as Qwen3-4B teacher, TRL-first OPD, SFT-only training, or 32768-token pre-run caps, treat this brief plus `exp/current_experiment_design.md` and `exp/current_acceptance_criteria.md` as authoritative for Cycle 08. The stable methodology layer should be updated later only after Code verifies the actual verl implementation and OPD loss/config.

---

## What This Cycle Does

Cycle 08 trains **one** OPD arm and compares it to the existing Cycle 07 SFT
trajectory.

This brief covers the **core run only**:

1. Phase 0 overlap + verl smoke probe.
2. Main OPD arm with `lambda=1.0`.
3. Core eval suite matching Cycle 07 as-run protocol.
4. Geometry + OverlapLift + random-LoRA reference.
5. Core copyback and handin.

**Do not run supplementary capability-degradation benchmarks yet.** After core
results land, the user will select a small set of checkpoint pairs and tasks for
supplementary eval.

---

## Hard Design Decisions

| Item | Required setting |
|---|---|
| Teacher | Qwen3-8B-Instruct. If missing, stop and report. |
| Student/base | `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base` |
| Data | Math-CoT-20k, same 5000-row seed=42 sample as Cycle 07 |
| OPD lambda | `1.0` |
| Framework | verl async/on-policy KD path. Do not silently fall back to TRL. |
| Cold-start | Prefer `step_000`; automatic fallback only to `step_005` or `step_010` if overlap/smoke requires it. |
| Forbidden automatic start | `step_020` or later. Stop and ask if no pre-dip start is viable. |
| Core comparison | OPD trajectory vs existing Cycle 07 SFT trajectory. Do not retrain SFT. |
| MATH500 cap | Keep Cycle 07 as-run `max_tokens=4096`; do not change to 32768. |

---

## Important Existing Paths

| Item | Path |
|---|---|
| Student/base | `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base` |
| Math-CoT-20k data | `/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet` |
| Cycle 07 run root | `/root/autodl-tmp/cycle07_base_sft_trajectory` |
| Cycle 07 copied results | `/root/LLM-output-density/mypaper/local_experiment_results/cycle_07_base_sft_trajectory/run_01` |
| Cycle 07 SFT checkpoints | `/root/autodl-tmp/cycle07_base_sft_trajectory/checkpoints/step_NNN` |
| Cycle 07 train JSONL | `/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/train_5k.jsonl` |
| Cycle 07 probe rows | `/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/probe_rows.jsonl` |

Teacher path is deliberately not guessed here. Find the local Qwen3-8B-Instruct
path at startup; if it is absent, write a blocker note and stop.

---

## Output Root

```text
EXP_ROOT=/root/LLM-output-density/mypaper/local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01
```

Create:

```text
run_01/
  phase0/
  checkpoints/
  eval/
  geometry/
  figures/
  supplementary/        # placeholder only; do not populate before user selection
  RESULTS_08_CORE.md
  trajectory_scores_opd.csv
  nontermination_trajectory_opd.csv
  response_length_trajectory_opd.csv
  run_provenance.json
```

---

## Phase 0: Overlap Probe and verl Smoke

### 0.1 Verify Required Resources

Check:

- student config exists;
- Qwen3-8B-Instruct teacher path exists;
- Math-CoT-20k parquet exists;
- Cycle 07 checkpoints `step_005`, `step_010`, `step_020` exist;
- verl environment / recipe is available.

If teacher or verl is missing, stop and write:

```text
{EXP_ROOT}/phase0/BLOCKED.md
```

Do not replace teacher, and do not fall back to TRL without user approval.

### 0.2 Reuse Cycle 07 Data Split

Use the Cycle 07 seed=42 split. Prefer:

```text
/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/train_5k.jsonl
/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/probe_rows.jsonl
```

If those are missing, recreate from:

```python
df = pd.read_parquet("/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet")
df_sample = df.sample(n=5000, random_state=42).reset_index(drop=True)
q_field = "question"
r_field = "response"
```

The Cycle 07 provenance confirms columns:

```text
data_source, question, answer, message, response, response_length, advantage
```

### 0.3 Candidate Checkpoints

Run overlap on:

```text
step_000, step_005, step_010, step_020
```

`step_020` is diagnostic only. It is the SFT dip and is not an automatic start.

Load candidates:

- `step_000`: base model only;
- `step_005`, `step_010`, `step_020`: base + Cycle 07 LoRA adapter.

### 0.4 Top-k Overlap Probe

Use a fixed prompt subset from the Cycle 07 probe rows. Compute teacher/student
top-k distributions on the same contexts.

Required settings:

```text
top_k = 32
recommended prompt count = 128 if affordable, otherwise >=32
contexts = prompt-only plus short rollout contexts if implementation supports it
```

Write:

```text
{EXP_ROOT}/phase0/overlap_probe.csv
```

Required columns:

```text
checkpoint
candidate_role
n_prompts
top_k
mean_student_teacher_topk_overlap
median_student_teacher_topk_overlap
teacher_topk_mass_on_shared_tokens
student_entropy_mean
teacher_entropy_mean
sample_generation_notes
viability_verdict
```

Viability rule:

```text
Use step_000 if overlap >= 0.70 and smoke is stable.
Otherwise use earliest viable checkpoint in {step_005, step_010}.
If none is viable, stop. Do not auto-start from step_020 or later.
```

### 0.5 verl Smoke

For the tentative start checkpoint, run a short OPD smoke:

```text
20-50 update steps, or the smallest run that verifies the full rollout -> teacher top-k -> update path
lambda = 1.0
teacher = Qwen3-8B-Instruct
student = selected start checkpoint
```

Record:

```text
{EXP_ROOT}/phase0/phase0_smoke.md
{EXP_ROOT}/phase0/selected_start.json
```

`selected_start.json` must include:

```json
{
  "selected_start_step": 0,
  "selection_reason": "...",
  "overlap_threshold": 0.70,
  "teacher_path": "...",
  "student_start_path": "...",
  "smoke_status": "pass"
}
```

---

## Phase 1: Main OPD Training

Run only after Phase 0 passes.

### 1.1 OPD Configuration

Required:

```text
student/base = Qwen3-4B-Base
teacher = Qwen3-8B-Instruct
lambda = 1.0
LoRA = r=32, alpha=64, dropout=0.05
target_modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
prompt sample = Cycle 07 Math-CoT-20k seed=42 5k
framework = verl async/on-policy KD path
teacher signal = top-k logprobs if using serving path
top_k = 32 unless implementation requires another value; record if changed
```

Record the exact loss/KL formula from code. Do not infer it from memory.

### 1.2 Checkpoint Grid

Use shared absolute budget labels:

```text
GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
```

If `selected_start_step = 0`, save OPD checkpoints:

```text
opd_step_005, opd_step_010, ..., opd_step_624
```

If `selected_start_step = 5` or `10`, mark earlier entries as inherited and save
OPD checkpoints only at later grid points. Do not fabricate missing OPD entries.

Write checkpoint metadata:

```text
{EXP_ROOT}/checkpoints/checkpoint_manifest.json
```

Fields:

```text
absolute_step
opd_updates_since_start
path
is_inherited_cold_start
source_sft_checkpoint
```

---

## Phase 2: Core Evaluation

Evaluate every available OPD checkpoint on the core suite. Compare to existing
Cycle 07 SFT tables; do not re-run SFT core eval.

### 2.1 Generative Math Tasks

Follow Cycle 07 `run_provenance.json`:

```text
chat template, think-format, no enable_thinking=False
temperature = 0.6
top_p = 0.9
math500 max_tokens = 4096
numina max_tokens = 4096
aime24 max_tokens = 16384
```

Tasks:

| Task | Scope | Required output |
|---|---|---|
| MATH500 | full N=500 | acc, SE, response length, non-term rate, finisher acc |
| NuminaMath-test | N=256, matching Cycle 07 | acc, SE |
| AIME24 | full N=30 | acc, SE, non-term notes |

MATH500 extraction:

- use last `\boxed{}` + math_verify;
- do not score by first/last dollar-span;
- manually inspect a small sample if score is suspiciously low or uniform.

Non-termination:

- use the same definition as Cycle 07;
- write per-checkpoint non-term rate;
- report finisher acc as `correct among terminating/scorable finishers`.

### 2.2 Loglikelihood Tasks

Use no-chat, no-thinking lm-eval protocol:

| Task | Protocol |
|---|---|
| GPQA-Diamond | `gpqa_diamond`, 0-shot, full N=198 |
| MMLU-Pro | 0-shot, `--limit 100/class`, 14 subtasks = 1400, `--seed 42` |

Merge LoRA checkpoints only one at a time for lm-eval, then delete merged copies
after use to save disk.

### 2.3 Core Aggregates

Write:

```text
{EXP_ROOT}/trajectory_scores_opd.csv
{EXP_ROOT}/response_length_trajectory_opd.csv
{EXP_ROOT}/nontermination_trajectory_opd.csv
```

`trajectory_scores_opd.csv` columns:

```text
step, inherited_or_opd, math500_acc, math500_se, numina_acc, numina_se,
aime24_acc, aime24_se, gpqa_acc, gpqa_se, mmlu_pro_acc, mmlu_pro_se
```

---

## Phase 3: Geometry and OverlapLift

Run geometry for all available OPD checkpoints on the shared grid.

### 3.1 Standard Geometry

Use:

```text
reference = step_000 Qwen3-4B-Base
layers = [9, 18, 27]
modules = q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
probe rows = Cycle 07 probe rows if available
```

Write:

```text
{EXP_ROOT}/geometry/geometry_metrics_step_{NNN}.csv
{EXP_ROOT}/geometry/geometry_summary.csv
```

Required metrics:

```text
effective_rank
spectral_gap
drift_from_base
xs_log_spectrum_gap
```

### 3.2 OverlapLift

Run the existing principalEvidence / OverlapLift path needed to compare:

```text
OPD vs SFT
OPD vs random-LoRA reference
SFT vs random-LoRA reference where available/needed
```

Write:

```text
{EXP_ROOT}/geometry/principal_evidence_step_{NNN}.csv
{EXP_ROOT}/geometry/overlap_lift_summary.csv
```

### 3.3 Random-LoRA Reference

Generate a random-LoRA reference:

```text
same base model
same LoRA rank/alpha/target modules
module-wise scale matched to trained adapter where feasible
minimum 1 random seed per checkpoint
optional extra seeds if cheap
```

Write:

```text
{EXP_ROOT}/geometry/random_lora_reference_step_{NNN}.csv
{EXP_ROOT}/geometry/random_lora_reference_manifest.json
```

Boundary:

- Do not implement full principal-angle / direct subspace-rotation analysis.
- Do not turn random baseline into an unqualified universal on/off-principal claim.
- If existing OverlapLift code requires intermediate UV-like files, save only what
  is required for that pipeline; full UV/principal-angle artifacts can wait for Cycle 09.

---

## Phase 4: Core Results File

Write:

```text
{EXP_ROOT}/RESULTS_08_CORE.md
```

Required sections:

1. Run summary: model paths, teacher path, data path, seed, framework, lambda.
2. Phase 0: overlap table, selected start, smoke verdict.
3. OPD checkpoint manifest.
4. Core trajectory table: OPD scores on all tasks.
5. SFT reference table or linked reference values from Cycle 07.
6. B08 verdict: final non-inferiority + dip severity + finisher/non-term decomposition.
7. C08 verdict: MMLU-Pro preservation vs SFT final delta.
8. D08 geometry summary: triple co-location comparison.
9. A08 OverlapLift summary: OPD vs SFT vs random-LoRA reference.
10. Known limitations and deviations from this brief.

Also generate figures if feasible:

```text
figures/trajectory_math500_opd_vs_sft.png
figures/trajectory_ood_opd_vs_sft.png
figures/nontermination_opd_vs_sft.png
figures/geometry_opd_vs_sft.png
figures/overlap_lift_opd_vs_sft_random.png
```

---

## Supplementary Evaluation Is Deferred

Do not run supplementary benchmarks in this pass.

After `RESULTS_08_CORE.md` is available, stop and return the core results. The
user will choose:

- which OPD checkpoints matter;
- which matched SFT checkpoints to supplement;
- which tasks from MMLU / HellaSwag / WinoGrande / ARC-Challenge / PIQA /
  TruthfulQA-MC1 to run.

---

## Disk and Resume Notes

- Check disk before merging checkpoints for lm-eval.
- Merge one LoRA checkpoint at a time; delete merged copy after eval.
- All phases should be resume-safe: skip a task/checkpoint if its complete output exists.
- Do not delete raw checkpoints or Phase0 artifacts.

---

## Code Handin After Core Run

Append a `## Cycle 08 Handin` section to `code/code_evolution.md` with:

1. Teacher path actually used.
2. verl recipe / command / config actually used.
3. Exact OPD loss/KL settings from code.
4. Phase0 overlap values and selected start.
5. Smoke run result and any instability.
6. Checkpoint manifest and any missing grid points.
7. Eval commands/protocols, including generation caps and MMLU-Pro limit.
8. Geometry/OverlapLift implementation notes.
9. Random-LoRA reference construction.
10. Deviations from this brief.
11. New scripts added or changed.

Also update `code/current_code_structure.md` if implementation adds scripts,
configs, or changes the code workflow.
