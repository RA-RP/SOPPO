# Experiment Design Evolution (Cycle 03 → 05)

```yaml
view_type: cross_cycle_evolution_log
section: exp_design
status: active
maintenance_rule: append_one_dated_delta_entry_per_cycle_do_not_restate_full_text
last_organized: 2026-06-16
```

This file tracks **what changed and why**, cycle to cycle, in the project's
experiment design and acceptance gates. It does not restate each cycle's full
design document. Stable implementation methodology (model/data/training/eval/geometry
protocol) now lives separately in `../exp/experimental_methodology/`; this file
only covers what changed about the *design and gates* from cycle to cycle.

Note: cycle snapshot folders have been removed (2026-06-16). The current working-face
design docs live in `../exp/`; key result artifacts live in
`../local_experiment_results/<cycle_name>/`. The "Full source" lines below are
kept as historical provenance references even though the files no longer exist.

## Cycle 03 — Starting Point

Full source: `cycle_03_trl_opd_like/03_experiment_design/` (snapshot removed;
content summarized in this entry).

The original design (`initial_experiment_design_03.md`) specified an
ambitious 8-experiment program (GetSlice sanity check, SFT cold-start,
LR-swept continued SFT, `pi_mix`-swept OPD, matched-MATH500 comparison,
geometry validation, optional calibration, optional selection extension).
What was **actually executed** was a much smaller TRL-first pilot: one
`theta0` cold-start (512 prompts), one OPD arm (`lmbda=1.0`), four SFT
controls (256/512/1024/2048), matched on **GSM8K** gain (not MATH500),
single-layer (14) GetSlice geometry, `eval_limit=200`. The gap between the
ambitious design and the minimal pilot was deliberate (feasibility-first),
and is explicitly acknowledged in `00_cycle_index.md`'s "Next Gate" list:
increase/remove `eval_limit`, add more OPD settings/seeds, use a less
saturated `theta0`, expand geometry beyond layer 14.

## Cycle 03 → Cycle 04: Acted On Every Cycle-03 Gate, Added Pre-Registered Gates

Full source: `cycle_04_opd_stability_gain/03_experiment_design/` (snapshot removed;
key artifacts in `../local_experiment_results/cycle_04_opd_stability_gain/`).

All four Cycle 03 "Next Gate" items were incorporated: `theta0` reduced to
256 prompts (less saturated), two OPD arms (`lmbda` 1.0 and 0.5), four new
SFT sizes (128/256/512/1024), GSM8K eval with `eval_limit=None` (full
N=1319), and geometry expanded to 3 layers (6/14/22) with vLLM colocate added
for speed. The bigger change was **process**: Cycle 04 introduced
pre-registered Gates A–D (`acceptance_criteria_04.md`) with hard pass/fail
thresholds (`GSM8K_gain_gap <= 0.02`, OOD-lite-penalty comparison, an
80%-of-cells drift requirement, a 0.02 GSM8K-gain threshold for Gate D),
replacing Cycle 03's looser five-bucket interpretation scheme. Result: Gates
A/B passed, C/D failed, root-caused to the matched-gain anchor collapsing to
`sft_n128` under the less-saturated `theta0` (see
`result_interpretation_evolution.md`).

## Cycle 04 → Cycle 05: Redesigned Around The Cycle 04 Failure Diagnosis

Current design: `../exp/current_experiment_design.md`, `../exp/current_acceptance_criteria.md`.
Key protocol findings: `../local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_*.md`.

New Gates A–F (`acceptance_criteria_05.md`) added two structural requirements
Cycle 04 lacked: a **mandatory base-ceiling row** (Gate B) and an explicit
**ID/OOD axis separation** (Gate C: NuminaMath-test = ID, GSM8K = OOD/transfer,
replacing GSM8K's previous dual role). Equal-ID and equal-budget matching
regimes were added alongside the old equal-GSM8K rule, now demoted to
"diagnostic only" (Gate D).

**CYCLE 05 CLOSED (2026-06-18).** Summary of what actually happened:

Phase 1 reuse audit initially read Cycle 04's GSM8K base-ceiling as confirmed (base ties theta0) → nearly triggered a non-instruct base model pivot → pre-registered pre-pivot capability check caught Cycle 04's GSM8K eval had never applied the chat template → corrected re-evaluation reversed headroom verdict → base-swap pivot abandoned. This is the most consequential design self-correction in the project so far, caught by a pre-registered check, not by chance.

**All remaining design tasks completed:**
- `opd_lmbda1`, `sft_n256`, `sft_n512`, `sft_n1024` re-evaluated under corrected GSM8K/MATH500 protocol. Full 8-model Gate B–F re-analysis done with corrected numbers (see `result_interpretation_evolution.md` cycle05 closed entry).
- OOD-lite suite: MMLU collapse (FINDING_05_mmlu_chat_template_collapse.md) resolved via base model mode (no chat template, 5-shot, standard loglikelihood — Qwen3 official protocol arXiv 2505.09388). TruthfulQA and WinoGrande confirmed. ARC-challenge added as 4th OOD-lite task (25-shot, no chat template, acc_norm, N=1172).
- The "apply chat template" distinction is now fully established: required for generate_until tasks (GSM8K, MATH500, TruthfulQA); harmful for loglikelihood tasks (MMLU, WinoGrande, ARC). This is codified in `../exp/experimental_methodology/evaluation_protocol.md` per-task protocol table.

**Eval stack redesign (2026-06-18):** All 7 evaluation tasks unified under `Eval/run_eval.py` single entry point with `--task` and `--backend {custom,lm_eval}` parameters. Custom scripts moved to `Eval/component/{gsm8k,math500,numina}/runner.py`. `Eval/run_cycle05_eval.py` updated to route all subprocess calls through `run_eval.py`. Unused files deleted.
