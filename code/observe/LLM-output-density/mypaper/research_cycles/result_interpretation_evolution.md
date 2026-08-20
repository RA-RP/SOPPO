# Result Interpretation Evolution (Cycle 03 → 05)

```yaml
view_type: cross_cycle_evolution_log
section: result_interpretation
status: active
maintenance_rule: append_one_dated_delta_entry_per_cycle_do_not_restate_full_text
last_organized: 2026-06-16
```

This file tracks **what changed and why**, cycle to cycle, in how results
were read and what claims were allowed. It does not restate each cycle's full
result/claims documents — for full text, follow the path given in each entry.
The current, stable result view lives in `../result/current_picture.md`,
`../result/claims_allowed.md`, `../result/incremental_log.md`.

Note: cycle snapshot folders have been removed (2026-06-16). Key artifacts live
in `../local_experiment_results/<cycle_name>/`. The "Full source" lines below are
kept as historical provenance even though the snapshot files no longer exist.

## Cycle 03 — Starting Point

Full source: `cycle_03_trl_opd_like/05_result_interpretation/` (snapshot removed;
content summarized in this entry).

Verdict: engineering-feasibility pass with **narrow, preliminary** K1
support. Under the one valid matched pair found (`opd_lmbda1`(800) vs
`sft_n256`, `GSM8K_gain_gap=0.010`), OPD showed lower OOD-lite penalty, lower
worst OOD-lite drop, and lower layer-14 spectral drift across all 7 measured
modules than the matched SFT. Explicitly **not allowed**: full OPD
replication, a GSM8K-improvement claim, or any principal-angle claim. Flagged
limits: `eval_limit=200` noise, only one OPD run, geometry limited to layer
14.

## Cycle 03 → Cycle 04: Stricter Re-Run Reverses The Stability Reading

Full source: `cycle_04_opd_stability_gain/05_result_interpretation/` (snapshot removed).
Key artifacts now in `../local_experiment_results/cycle_04_opd_stability_gain/`:
`RESULTS_04.md`, `FINDING_04_eval_extraction_artifact.md`, `FINDING_04_base_ceiling.md`.

What changed: under the new pre-registered gates, Gates A/B passed but C/D
**failed** — "Cycle 03's stability signal was not reproduced." The
failure-analysis document is explicit that the cause was not "OPD got worse"
but a **control-anchor failure mode**: a less-saturated `theta0` forced the
nearest matched-gain SFT down to `sft_n128`, which had barely moved from
`theta0` and was therefore trivially "stable" to lose against. Two
independent major findings were produced in the same cycle: (1) `MATH500=0`
for every model was an **extraction artifact** (the scorer's strict-only
path failed on this output format; the same model scored 0.413 under
flexible extraction), which led to adopting NuminaMath-test as a clean ID
axis; (2) the untrained base **tied `theta0`** and **beat both OPD arms** on
that new ID axis — a base-ceiling finding suggesting no method had
demonstrated net capability gain in this setting. Official verdict:
"engineering success, pre-registered claim failure" — reframed as a
methodological lesson about control-anchor sensitivity to `theta0`
saturation, not a verdict against OPD's value.

## Cycle 04 → Cycle 05 (CLOSED 2026-06-18): The Base-Ceiling Reading Itself Gets Corrected

Key artifacts: `../local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_gsm8k_chat_template_mismatch.md`,
`FINDING_05_math500_dollar_span_extraction_bug.md`, `FINDING_05_mmlu_chat_template_collapse.md`.
Full result view: `../result/current_picture.md`, `../result/claims_allowed.md`, `../result/incremental_log.md`.
Detailed tables: `../local_experiment_results/cycle_05_matched_control_id_ood/run_01/tables/id_ood_trajectory.csv`.

**What changed:** Cycle 05 started as a pre-pivot reuse audit. Cycle 04's GSM8K
base-ceiling finding was initially read as confirmed — the same GSM8K numbers
(~0.41 for all models) suggested base ≈ theta0, no headroom. A pre-registered
pre-pivot capability check (triggered before downloading a non-instruct base
model) compared those numbers against Qwen's official benchmark and found a
~35-point gap, tracing it to Cycle 04's GSM8K eval never applying the chat
template (same artifact shape as MATH500=0 one cycle earlier: every model had
strict-match = 0.000). Three compounding eval protocol bugs were identified and
corrected across all 8 models:

1. **Finding 05a (GSM8K):** missing chat template + enable_thinking=False + 3072-token budget.
   GSM8K: ~0.41 → ~0.61–0.73. Reverses the base-ceiling / no-headroom reading on GSM8K axis.
2. **Finding 05b (MATH500):** `$`-span extractor slices across intermediate LaTeX in CoT,
   not the final `\boxed{}`. MATH500: 0.000 → ~0.54–0.73.
3. **Finding 05c (MMLU):** `--apply_chat_template` without enable_thinking=False collapses all
   models to 0.2295 (4-choice random chance). Resolution: base model mode (no chat template,
   5-shot, standard loglikelihood) — Qwen3 official protocol (arXiv 2505.09388). Yields 0.60–0.61.

**Complete 8-model corrected results (cycle05):**

| Model | NuminaMath ID | GSM8K | MATH500 | MMLU | TQA-MC1 | WinoGrande | ARC acc_norm |
|---|---|---|---|---|---|---|---|
| base | 0.5516 | 0.7005 | 0.7140 | 0.6061 | 0.3341 | 0.6172 | 0.4565 |
| theta0 | 0.5527 | 0.7180 | 0.7260 | 0.6019 | 0.3415 | 0.6235 | 0.4590 |
| opd_lmbda05 | 0.5392 | 0.7180 | 0.7080 | 0.6004 | 0.3390 | 0.6251 | 0.4650 |
| opd_lmbda1 | 0.5348 | **0.7346** | 0.7140 | 0.6061 | 0.3427 | 0.6235 | 0.4616 |
| sft_n128 | 0.5370 | 0.7096 | 0.7260 | 0.6049 | 0.3439 | 0.6290 | 0.4599 |
| sft_n256 | 0.5516 | 0.6839 | 0.7080 | 0.6033 | 0.3415 | 0.6235 | 0.4667 |
| sft_n512 | 0.5179 | 0.6732 | 0.7140 | 0.6086 | 0.3378 | 0.6322 | 0.4710 |
| sft_n1024 | 0.3890 | 0.6141 | 0.5400 | 0.6065 | 0.3317 | 0.6322 | 0.4667 |

**Key result shifts from Cycle 04:**
- **Headroom restored**: theta0 is ahead of base on both GSM8K (+0.0175) and MATH500 (+0.0120). The Cycle 04 "no headroom" reading was entirely a protocol artifact.
- **opd_lmbda1 highest GSM8K**: 0.7346, +0.0341 over base, +0.0166 over theta0. But below base on NuminaMath ID (0.5348 vs 0.5516).
- **sft_n1024 outlier**: severe math regression (NuminaMath ID −0.163, GSM8K −0.086, MATH500 −0.174 vs base) while OOD-lite stable. Pattern consistent with catastrophic forgetting at n=1024 SFT samples on 1.7B model.
- **OOD-lite non-discriminative**: all 4 tasks (MMLU, TruthfulQA, WinoGrande, ARC-challenge) show spread <1.5% across all 8 models. Gate C04 formally confirmed FAIL, but this is a non-signal: insufficient discriminative power, not OPD harming generalization.
- **ARC-challenge added** as 4th OOD-lite task (25-shot, no chat template, acc_norm, N=1172). Spread = 0.0145 ≈ 1 SE (±0.0146). Confirms OOD-lite capability preservation pattern.

**Gate re-analysis (cycle05 corrected numbers):**
- Gate B04: CONFIRMED PASS. Matching basis is NuminaMath-test ID; sft_n128 remains nearest SFT anchor.
- Gate C04: CONFIRMED FAIL. OPD does not beat sft_n128 on any OOD-lite metric. Non-signal (spread <1.5%).
- Gate D04: CONFIRMED FAIL. Depends on Gate C04.

**Eval stack:** All 8 evaluation tasks unified under `Eval/run_eval.py` single entry point with `--task` and `--backend {custom,lm_eval}` control. Custom scripts moved to `Eval/component/{gsm8k,math500,numina}/runner.py` with clean `run()` API (2026-06-18).
