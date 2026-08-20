# QA: Cycle 08 — Code Execution Phase

```yaml
phase: code_execution
cycle: cycle_08_h_opd_vs_sft_comparison
status: waiting_user_response
created: 2026-06-29
brief: code/current_code_brief.md
exp_design: exp/current_experiment_design.md
acceptance: exp/current_acceptance_criteria.md
copyback_target: local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/
blocked_artifact: local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/phase0/BLOCKED.md
```

This QA file starts the Cycle 08 coder phase. Per `mypaper/readme.md`, the code
phase must create this file, get user confirmation, internalize the decisions,
and only then execute the run.

## Role And Write Boundary

Current role: **Code Execution / coder**.

Allowed write targets for this phase:

- `local_experiment_results/cycle_08_h_opd_vs_sft_comparison/`
- `code/code_evolution.md` handin section, after execution or blocker handoff
- `code/current_code_structure.md` append section, if scripts/configs change
- codebase scripts/configs required to run the experiment, with changes recorded
  back into the two code documentation files above

Not allowed in this phase:

- changing `theory/`, `related_work/`, `result/`, `next_cycle/`, or `exp/`
- changing the teacher, lambda, checkpoint-start rules, or primary gates
- silently falling back from `verl` to TRL
- running supplementary benchmarks before core OPD results and user selection

## Context Read

Startup files read:

- `readme.md`
- `code/current_code_brief.md`
- `code/current_code_structure.md`
- `exp/experimental_methodology/model_and_data.md`
- `exp/experimental_methodology/training_protocol.md`
- `exp/experimental_methodology/evaluation_protocol.md`
- `exp/experimental_methodology/geometry_protocol.md`
- `exp/current_experiment_design.md`
- `exp/current_acceptance_criteria.md`
- `exp/QA_cycle08_exp.md`
- `local_experiment_results/cycle_07_base_sft_trajectory/run_01/RESULTS_07.md`
- `local_experiment_results/cycle_07_base_sft_trajectory/run_01/run_provenance.json`

Cycle confirmed from `exp/current_experiment_design.md`:

```text
cycle_08_h_opd_vs_sft_comparison
```

## Preliminary Resource Probe

I ran only read-only resource checks. No training, overlap probe, eval, or
geometry job has started.

Confirmed present:

- `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base`
- `/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet`
- Cycle 07 checkpoints: `step_005` through `step_624` plus `step_632`
- Python packages in `density`: `vllm`, `transformers`, `datasets`

Blocking findings:

- `Qwen3-8B-Instruct` was not found under the expected model roots.
- `verl` is not importable from `/root/miniconda3/envs/density/bin/python`.
- No local `verl` checkout was found under `/root`.

The blocker has been written to:

```text
local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/phase0/BLOCKED.md
```

## Q1: Required Teacher Path

Cycle 08 requires `Qwen3-8B-Instruct`, and the code brief forbids substituting
another teacher automatically.

Please confirm one:

- Provide the exact local path to an already available `Qwen3-8B-Instruct`.
- Approve downloading/installing `Qwen3-8B-Instruct` to a specific local path.
- Pause Code Execution until the teacher model is prepared externally.

## Q2: Required `verl` Implementation Path

Cycle 08 requires the `verl` async/on-policy KD path. TRL fallback is explicitly
not automatic.

Please confirm one:

- Provide the Python environment or local checkout where `verl` is available.
- Approve installing/cloning `verl` for this run.
- Pause Code Execution until `verl` is prepared externally.
- If you want TRL fallback, treat that as an Experiment Design change rather
  than a coder-side substitution.

## Q3: After Resources Are Fixed

Once teacher and `verl` are available, I propose to proceed in this order:

1. Add a Cycle 08 runner/worker around Phase 0 overlap and minimal smoke.
2. Reuse Cycle 07 data split and checkpoint-grid machinery where possible.
3. Run Phase 0 only: resource verification, top-k overlap probe, selected start,
   and minimal smoke.
4. Stop for a checkpoint if Phase 0 selects anything other than strict
   `step_000`, or if the smoke reveals instability.

Please confirm whether this Phase 0-first execution plan is acceptable after the
two blockers are resolved.
