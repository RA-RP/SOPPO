# Cycle 09 Reviewer-Robustness Coder Handoff

```yaml
view_type: reviewer_derived_execution_handoff
cycle: cycle_09_aaai_competitiveness_completion
created: 2026-07-27
status: READY_FOR_PREFLIGHT
execution_gate:
  tier_1_cpu_or_reuse: allowed_after_RR0_inventory
  tier_2_new_forward: requires_explicit_theory_GO
review_source: /root/.codex/attachments/228fb847-de5d-47b8-aebb-c400037d8a82/pasted-text.txt
equivalent_review_copy: /root/.codex/attachments/f142e59f-191b-499e-8d5e-1a23989a7bd8/pasted-text.txt
fact_source: /root/LLM-output-density/mypaper/theory/human_read-ch.md
paper_grid_source: /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/d10_5_integrated_state_equal7.csv
output_root: /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/
guard: zero training; raw readings only; no theory adjudication; no paper edits
```

## 1. Purpose

This handoff converts the latest reviewer questions into executable robustness
tasks. It does not authorize a new research direction. The immediate target is
to answer:

1. how probe/sample resampling changes \(r_\varepsilon\) and the 23/24 early
   OPD ordering;
   separately, how reducing the number of probe samples changes the rank,
   uncertainty, and OPD ordering;
2. whether the discrete threshold is close to a tail-energy boundary and
   whether continuous spectral summaries retain the result;
3. whether centered covariance changes between-arm ordering or output links;
4. how much teacher probability mass is retained by the top-32 trainer data;
5. whether raw-activation features add information jointly with
   \(c_\varepsilon\);
6. which coarse rollout statistics differ between OPD and frozenSelf0-KD, if
   matched rollout artifacts exist.

Coder reports measurements and provenance only. Theory decides what the
measurements mean and whether they enter `human_read-ch.md`.

## 2. Frozen Main-Paper Grid

Use the formal D10/D4 matched grid and do not silently substitute legacy
geometry.

| item | Qwen | Llama |
|---|---|---|
| student | Qwen3-4B-Base | Llama-3.2-3B |
| headline layer | L18 | L14 |
| arms | OPD, SFT, off-KD, seqKD | OPD, SFT, off-KD, seqKD |
| common early steps | 20, 40, 80 | 20, 40, 80 |
| formal probes | `E_general`, `E_math`, `E_ood`, `E_if` | same |
| paper aliases | general, mathHeld, mmluPro, ifeval | same |
| modules | q/k/v/o, gate/up/down projections | same |
| headline epsilon | .05 | .05 |
| sensitivity epsilons | .01, .025, .05, .10 | same |

All base-relative quantities must pair each current cell with the formal
step-0 cell from the same model, probe, layer, module, numerical protocol, and
sample resample. Equal-seven summaries are module-first means.

## 3. Known State Before Coding

Do not relabel these partial artifacts as completed analyses:

- `state_displacement_sample_count_bootstrap.csv` contains 13,056
  `resample_ready` sample-index hashes at sample count eight. It is an
  inventory, not a recomputation of \(r_\varepsilon\) or a confidence interval.
- Existing C3 code performs real sample bootstrap for selected OPD cells:
  `experiments/opd_sft_h1/scripts/cycle09_c2c3_bootstrap.py`.
- Stage-4 `.direction.pt` cells retain per-module singular values and can
  support tail-margin and continuous-spectrum calculations without model
  forward passes.
- Centered covariance currently covers only a partial grid. The existing
  aggregate is
  `mini/d10_5_a5_centered_state_cells.csv`.
- Teacher top-32 arrays exist for some off-KD, alpha=.5, and frozen-self
  pipelines, but a complete retained-mass table has not been produced.
- Raw activation ER/PR/CKA comparisons exist. A same-fold
  `raw activation + C` incremental model has not been established in
  `human_read-ch.md`.

## 4. RR0: Artifact And Protocol Inventory

**Cost:** CPU, read-only.  
**Run first and stop if the formal grid cannot be identified.**

Inventory, with size and SHA256 where practical:

- formal D10/D4 state tables and output tables;
- all singular-spectrum or `.direction.pt` files;
- per-sample factor bundles usable for exact sample bootstrap;
- centered and uncentered cells;
- raw-activation feature tables;
- teacher top-32 ids/log-prob arrays and their manifests;
- OPD and frozenSelf0-KD rollout dumps usable for matched text diagnostics.

Required outputs:

```text
RR0_artifact_inventory.csv
RR0_grid_coverage.csv
RR0_protocol_map.json
RR0_blockers.md
```

`RR0_grid_coverage.csv` must have one row per requested
`task x model x arm x checkpoint x probe`, with:

```text
task, model, arm, checkpoint, probe_name, layer,
formal_source, input_path, input_sha256, protocol_id,
has_spectrum, has_sample_factors, has_centered,
sample_count, status, blocker_reason
```

Allowed statuses:

```text
READY_REUSE
READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT
BLOCKED_MISSING_ARTIFACT
BLOCKED_PROTOCOL_MISMATCH
```

No legacy/fp16/cholesky cell may fill a missing D10/D4 cell without an explicit
Theory ruling.

## 5. RR1: Actual Sample Bootstrap For Functional Rank

**Priority:** highest.  
**Tier:** reuse factors when available; otherwise new forward requires Theory
GO.  
**Primary draws:** 1,024; seed 42. Also emit the first 256 draws as a parity
summary.

RR1 has two distinct estimands and must not merge them:

```text
RR1A = fixed-N empirical-distribution uncertainty:
       sample bootstrap with replacement

RR1B = sample-count sensitivity:
       repeated subsampling without replacement at several n
```

RR1A answers uncertainty conditional on the frozen probe composition and
formal sample count. RR1B answers how much the estimator changes when fewer
samples are available. Neither task establishes robustness to a different
semantic probe composition, source mixture, or domain definition.

### RR1A: fixed-N paired bootstrap

Reuse the sample-with-windows-nested bootstrap mechanics from C3, but apply it
to the frozen grid in Section 2.

For each draw:

1. resample probe samples with replacement;
2. use the same sampled indices for the base and every arm sharing that fixed
   probe;
3. reconstruct each module Gram/second moment using the exact per-model formal
   numeric protocol below; a generic shared `FP32 Gram` implementation is
   forbidden;
4. use the formal deployed checkpoint weight and the matching D10/D4
   eig/whitening/SVD path;
5. recompute module \(r_{.05}\), base-relative \(\Delta r_{.05}\), and the
   equal-seven mean;
6. compute the paired OPD-minus-nearest-offline margin within each
   `model x probe x checkpoint` cell.

The frozen numeric protocols are:

```text
Qwen D4:
  checkpoint/model forward = BF16
  Gram accumulation and whitening/eigh = FP64
  W S matmul = FP32
  SVD input and singular-energy accumulation = FP64

Llama D10:
  checkpoint/model forward = BF16
  hidden states cast to FP32
  Gram accumulation = FP32
  Gram factorization/eigh = FP64
  W S matmul = FP32
  SVD input and singular-energy accumulation = FP64
```

The manifest must record `gram_accumulation_dtype` and
`gram_factorization_dtype` separately. Do not infer one from the other.

Required outputs:

```text
RR1_rank_bootstrap_draws.parquet
RR1_rank_cell_intervals.csv
RR1_opd_ordering_persistence.csv
RR1_23of24_distribution.csv
RR1_bootstrap_manifest.json
```

Required cell-level columns:

```text
model, arm, checkpoint, probe_name, layer, epsilon,
point_delta_r_equal7, bootstrap_mean, ci_low, ci_high,
draws, seed, bootstrap_unit, sample_count
```

Required ordering outputs:

- probability that OPD is the most contracted arm in each of the 24 cells;
- probability that OPD beats the nearest offline arm by more than zero;
- distribution over draws of the number of cells, out of 24, in which OPD is
  deepest;
- model-separated versions out of 12 cells;
- 256-draw versus 1,024-draw differences in means and interval endpoints.

For the 23-of-24 distribution, a draw ID must define one coordinated bootstrap
replicate across the complete panel: each probe may use its own seeded sample
resample, but the resulting cell indicators must be aggregated only within
the same draw ID. Do not form the count distribution by independently
shuffling already-computed cell intervals.

### RR1B: formal sample-count sensitivity

Reuse exactly the same formal checkpoints, weights, sample identities, and
per-model numeric protocols as RR1A. At each \(n\), draw 200 repeated subsets
without replacement with seed 42:

```text
E_math (formal N=32):       n = 8, 16, 24, 32
E_general:                  n = 16, 32, 64, 128
E_if and E_ood:             n = 16, 32, 64, 128
```

If a formal probe has fewer than a requested \(n\), cap the grid at its full
formal \(N\) and record the skipped value; never sample duplicated examples
to simulate a larger sample count. Use the same subset indices for the shared
base and all four arms.

For every `model x arm x checkpoint x probe x n`, report:

```text
median, mean, IQR, ci_low, ci_high,
absolute_difference_from_full_N,
delta_r_sign_persistence,
opd_deepest_probability
```

Also rerun the model-separated 12-cell and combined 24-cell OPD ordering count
at every common \(n\). The full-\(N\) RR1A bootstrap remains the primary
uncertainty result; RR1B is a finite-sample sensitivity curve.

Additional required outputs:

```text
RR1_sample_count_draws.parquet
RR1_sample_count_intervals.csv
RR1_sample_count_ordering.csv
RR1_sample_count_manifest.json
```

Do not treat probes, modules, checkpoints, or bootstrap draws as independent
training seeds.

## 6. RR2: Tail Margins And Continuous Spectrum Summaries

**Priority:** high.  
**Tier:** CPU/reuse only when full singular values are present.

For each formal headline state spectrum, let

\[
p_i=\sigma_i^2/\sum_j\sigma_j^2,\qquad
T_k=\sum_{i>k}p_i,\qquad r=r_{.05}.
\]

Report:

```text
tail_at_r
tail_at_r_minus_1
margin_below = .05 - tail_at_r
margin_above = tail_at_r_minus_1 - .05
two_sided_tail_margin = min(margin_below, margin_above)
stable_rank = sum(sigma^2) / max(sigma^2)
entropy_effective_rank = exp(-sum(p * log(p)))
top1_energy_share
top10_energy_share
```

Use the convention \(0\log 0=0\), and record the number of exactly zero
singular-energy probabilities after the formal numerical clamp.

Create module-level, equal-seven, and base-relative tables. For stable rank and
entropy effective rank, define relative contraction module-first using the
same base normalization as \(c_\varepsilon\).

Required outputs:

```text
RR2_spectrum_stability_module.csv
RR2_spectrum_stability_equal7.csv
RR2_continuous_ordering.csv
RR2_continuous_output_links.csv
RR2_spectrum_manifest.json
```

`RR2_continuous_ordering.csv` must rerun the frozen 23/24 early-window check
separately for stable-rank contraction and entropy-rank contraction.
`RR2_continuous_output_links.csv` must use the same matched cells as D10.5 and
report within-arm Spearman correlations with cumulative KL and absolute NLL.

If a formal cell has only threshold ranks but no singular values, mark it
`BLOCKED_MISSING_SPECTRUM`; do not infer a continuous rank from four
\(\varepsilon\) values.

## 7. RR3: Balanced Centered-Covariance Audit

**Priority:** high but expensive.  
**Tier:** new forward; do not start without explicit Theory GO.

Target the complete common early grid:

```text
2 models x 4 arms x 3 steps x 4 probes
+ one shared base per model/probe
= 104 model-arm-step-probe profiles
```

Use the same samples, checkpoints, headline layers, seven modules, dtype
protocol, and epsilon set as the formal uncentered cells. Change only:

```text
uncentered E[h h^T]
    ->
centered E[(h - mean(h))(h - mean(h))^T]
```

Required outputs:

```text
RR3_centered_module.csv
RR3_centered_equal7.csv
RR3_centered_vs_uncentered.csv
RR3_centered_ordering.csv
RR3_centered_output_links.csv
RR3_centered_manifest.json
```

Report:

- absolute and base-relative rank changes under both estimands;
- how often between-arm ordering changes;
- centered analogue of the 23/24 early-window count;
- centered \(c\)-KL and \(c\)-absolute-NLL within-arm correlations;
- exact cells where centering changes the identity of the deepest arm.

All centered-versus-uncentered ordering and output-link comparisons must use
the exact same 104-profile early grid, sample IDs, checkpoints, and available
output rows. Recompute the uncentered comparator on that subset. Do not compare
an early-grid centered correlation against the 24/36-row full-trajectory
uncentered correlation from the main text as though only centering changed.

This is a construct audit. Do not replace the headline uncentered estimand.

## 8. RR4: Teacher Top-32 Retained Probability Mass

**Priority:** medium-high.  
**Tier:** CPU if stored arrays are raw teacher log probabilities.

First verify from code/manifests whether stored values are raw full-vocabulary
teacher log probabilities or top-32-renormalized values. Only raw values allow

\[
m_{32}=\sum_{i\in\mathrm{top32}}\exp(\log p_i).
\]

For every valid source, report:

```text
model, arm_or_pipeline, checkpoint_or_rollout_source,
n_sequences, n_tokens, mean, std, min,
p01, p05, p10, p25, median, p75, p90, p95, p99,
frac_below_0_90, frac_below_0_95, frac_below_0_99
```

Required outputs:

```text
RR4_top32_retained_mass_summary.csv
RR4_top32_retained_mass_by_sequence.csv
RR4_top32_coverage.csv
RR4_top32_manifest.json
```

If arrays were renormalized, report `BLOCKED_RENORMALIZED_TOPK`; do not return a
retained mass of one. Do not launch top-64/128 or exact-KL training/evaluation
unless Theory issues a separate GO after reading RR4.

## 9. RR5: Raw-Activation Hybrid Incremental Models

**Priority:** medium.  
**Tier:** CPU/reuse.

Join raw-activation features to the exact D11 same-cell matrix using explicit
keys:

```text
model, arm, checkpoint, probe_name, layer
```

No imputation and no nearest-checkpoint/probe matching. Report the exact common
coverage before fitting.

Candidate feature blocks:

```text
A     = raw activation ER, PR, top-share, anisotropy, CKA where available
C     = c_epsilon
Pk    = strict source-principal p_k block used by D11
A+C
Pk+A
Pk+C
Pk+A+C
```

Use the same checkpoint-grouped folds and train-fold-only standardization as
D11. Targets:

```text
cumulative KL
absolute NLL
signed NLL (secondary)
OPD vs non-OPD AUC
```

Required outputs:

```text
RR5_hybrid_coverage.csv
RR5_hybrid_grouped_models.csv
RR5_hybrid_predictions.parquet
RR5_hybrid_manifest.json
```

Report pooled and model-separated results. If the common grid is too small or
misses a full model/arm, return coverage and stop rather than imputing.

## 10. RR6: Frozen-Self Coarse Mediator Diagnostics

**Priority:** optional, artifact-dependent.  
**Tier:** CPU/reuse only.

This task does not identify causal mediation. It only describes whether
matched OPD and frozenSelf0-KD rollout dumps differ in coarse observables.

On exactly matched Llama prompts/checkpoints, report:

```text
response_tokens
EOS rate
truncation rate
4-gram repetition
distinct-2
exact duplicate rate
boxed rate
think-tag rate
```

Required outputs:

```text
RR6_frozen_self_text_stats.csv
RR6_frozen_self_paired_differences.csv
RR6_frozen_self_coverage.csv
RR6_frozen_self_manifest.json
```

If current-OPD rollout text is unavailable at a frozen-self checkpoint, mark
that cell missing. Do not compare unmatched prompt pools as if paired.

## 11. Explicitly Out Of Scope

Coder must not start any of the following from this handoff:

- independent training seeds;
- full-parameter fine-tuning;
- new LoRA ranks or adapter-placement training;
- output-gradient/Fisher/K-FAC variants;
- top-64/128 trainer reruns or exact full-vocabulary KL;
- new policy-lag or support-fraction arms;
- paper figures, prose, claims, or reviewer rebuttal.

These require Theory/Experiment Design decisions because they change the
paper's scope or consume new training budget.

## 12. Validation And Return Contract

Before any expensive run:

1. run RR0;
2. produce estimated CPU/GPU time and scratch-space requirements per task;
3. execute RR1A/RR1B/RR2/RR4/RR5/RR6 only where RR0 says `READY_REUSE`;
4. request Theory GO for every `READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT` cell
   and all of RR3.

Every output manifest must record:

```text
created_utc
git_commit
command
input_paths_and_sha256
formal_protocol_id
models
arms
checkpoints
probes
layers
epsilons
draws_and_seeds
row_counts
blocked_cells
output_sha256
```

After completion:

- append an execution handin to `code/code_evolution.md`;
- copy final compact artifacts to `output_root`;
- keep large factors/spectra in scratch and reference them by path/hash;
- create `reviewer_robustness_theory_handoff.md` containing raw tables,
  coverage, blockers, and no interpretation;
- do not edit `theory/`, `result/`, `related_work/`, `lecture/`, or the paper.

## 13. Execution Return: 2026-07-27

```yaml
execution_status: COMPLETE_REUSE_PASS_WITH_BLOCKERS
script: /root/LLM-output-density/experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py
rr0_command: python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py --rr0
reuse_command: python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py --run-ready-reuse
theory_handoff: /root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/reviewer_robustness_theory_handoff.md
code_evolution_updated: true
new_training: false
new_forward: false
paper_or_theory_edits: false
```

RR0 gate result:

| task | status | n |
|---|---:|---:|
| RR1A | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT | 96 |
| RR1B | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT | 96 |
| RR2 | READY_REUSE | 60 |
| RR2 | BLOCKED_MISSING_ARTIFACT | 36 |
| RR3 | READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT | 96 |
| RR4 | READY_REUSE | 96 |
| RR5 | READY_REUSE | 96 |
| RR6 | READY_REUSE | 48 |
| RR6 | BLOCKED_PROTOCOL_MISMATCH | 48 |

Completed reuse outputs:

| task | manifest status | key outputs |
|---|---|---|
| RR2 | COMPLETE_RR2_READY_REUSE_PARTIAL_GRID | `RR2_spectrum_stability_module.csv`, `RR2_spectrum_stability_equal7.csv`, `RR2_continuous_ordering.csv`, `RR2_continuous_output_links.csv` |
| RR4 | COMPLETE_RR4_READY_REUSE | `RR4_top32_retained_mass_summary.csv`, `RR4_top32_retained_mass_by_sequence.csv`, `RR4_top32_coverage.csv` |
| RR5 | BLOCKED_INSUFFICIENT_COMMON_GRID | `RR5_hybrid_coverage.csv`, `RR5_hybrid_grouped_models.csv`, `RR5_hybrid_predictions.parquet` |
| RR6 | COMPLETE_RR6_MATH500_MATCHED_TEXT_STATS | `RR6_frozen_self_text_stats.csv`, `RR6_frozen_self_paired_differences.csv`, `RR6_frozen_self_coverage.csv` |

Blocked or gated items:

| item | status | reason |
|---|---|---|
| RR1A/RR1B | not run | no formal per-sample factor bundle found; exact sample bootstrap would require new forward / Theory GO |
| RR2 Qwen sft/offkd/seqkd | blocked | missing Stage4 current/base `direction.pt` singular spectra for 36 cells |
| RR3 | not run | centered covariance audit is new forward and requires explicit Theory GO |
| RR4 Qwen alpha05 frozen_external | blocked | stored top32 manifest does not prove raw full-vocabulary teacher log probabilities |
| RR5 fitting | stopped by protocol | exact raw-activation join covers 76/192 D11 cells; no imputation or nearest matching allowed |

Raw readback is in `reviewer_robustness_theory_handoff.md`; no interpretation or
paper claim was added by Coder.

## 14. Correction Return: 2026-07-27

Theory复核后的 CPU correction pass 已完成；未修改论文、`human_read-ch.md` 或理论结论。

输出目录：

```text
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/
```

执行命令：

```bash
python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness.py --correction-pass
```

状态：

| 项 | 状态 |
|---|---|
| 旧 RR2 | `SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION` |
| RR2S state spectrum | `READY_RECOMPUTE_FROM_FORMAL_CHECKPOINT`，未启动 new forward |
| RR2D displacement spectrum | auxiliary only，Llama only；Qwen top-128 blocked |
| RR5 | Llama-only exact common-grid grouped models 完成；Qwen raw activation exact-grid blocked |
| RR6 | 改名为 matched behavioral-readout diagnostics，并加入 paired item-bootstrap CI |
| RR4 | provenance/clamped/omitted mass 补充完成；Qwen alpha=.5 仍 blocked |
| RR1/RR3 | shared forward/cache preflight only，未启动 new forward |

正式 theory handoff：

```text
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/reviewer_robustness_theory_handoff.md
```

## 15. Correction2 Reuse Return: 2026-07-27

按 Theory 复核意见完成第二轮 reviewer-robustness correction；未修改论文、`human_read-ch.md` 或理论判断。

执行命令：

```bash
python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_correction2.py
```

新增/更新产物目录：

```text
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/
```

状态摘要：

| 项 | 状态 |
|---|---|
| RR5-A | `RR5_hybrid_fold_performance.csv` 已补真正逐 fold metrics |
| RR5-B | `RR5_checkpoint_demeaned_correlations.csv` / cells 已补 |
| RR5-C | `RR5_nested_regularization_*` 已补 nested train-fold-only regularization |
| RR2S Llama | 使用 D10 formal state spectra 完成，无 forward |
| RR3 Llama | 使用 D10 formal Gram + input_sample_means + BF16 merged weights 完成 centered audit，无 forward |
| RR2D | equal7 / ordering / output_links 已补，继续标注 auxiliary |
| RR1/RR2S/RR3 preflight | 已拆成 task-specific availability；RR1 仍需 future new forward，未启动 |

正式 handoff：

```text
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/reviewer_robustness_theory_handoff.md
```

## 16. Module-Level Robustness Return: 2026-07-27

低成本收尾已完成并改为 tie-aware 口径：从正式 RR2S/RR3 module 表分别统计 OPD 严格最深、并列最深及 offline 严格更深；无 forward、无训练、未修改论文或 `human_read-ch.md`。

输出：

```text
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/RR_module_level_robustness_summary.csv
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/RR_module_level_robustness_cells.csv
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/reviewer_robustness/RR_module_level_robustness_manifest.json
```

正式读数：

| analysis                          | formal_use_class                 | source_table                         | metric                             | metric_direction   | module_filter   |   opd_among_best_count |   opd_strict_deepest_count |   opd_tied_deepest_count |   offline_strictly_deeper_count |   total_cells |   opd_among_best_fraction |   opd_strict_deepest_fraction |   opd_tied_deepest_fraction |   offline_strictly_deeper_fraction |   mean_opd_minus_nearest_offline_margin |   min_opd_minus_nearest_offline_margin |   median_opd_minus_nearest_offline_margin |
|:----------------------------------|:---------------------------------|:-------------------------------------|:-----------------------------------|:-------------------|:----------------|-----------------------:|---------------------------:|-------------------------:|--------------------------------:|--------------:|--------------------------:|------------------------------:|----------------------------:|-----------------------------------:|----------------------------------------:|---------------------------------------:|------------------------------------------:|
| uncentered_r_epsilon              | formal_rr2s_llama_state_spectrum | RR2S_llama_state_spectrum_module.csv | absolute_contraction               | max_is_deepest     | all             |                    330 |                        311 |                       19 |                               6 |           336 |                  0.982143 |                      0.925595 |                  0.0565476  |                          0.0178571 |                                9.29167  |                             -3         |                                 8         |
| uncentered_stable_rank            | formal_rr2s_llama_state_spectrum | RR2S_llama_state_spectrum_module.csv | stable_rank_contraction            | max_is_deepest     | all             |                    328 |                        328 |                        0 |                               8 |           336 |                  0.97619  |                      0.97619  |                  0          |                          0.0238095 |                                0.2061   |                             -0.138437  |                                 0.0970498 |
| uncentered_entropy_effective_rank | formal_rr2s_llama_state_spectrum | RR2S_llama_state_spectrum_module.csv | entropy_effective_rank_contraction | max_is_deepest     | all             |                    336 |                        336 |                        0 |                               0 |           336 |                  1        |                      1        |                  0          |                          0         |                                6.78884  |                              0.0278649 |                                 5.7625    |
| centered_r_epsilon                | formal_rr3_llama_centered_audit  | RR3_llama_centered_module.csv        | centered_absolute_contraction      | max_is_deepest     | all             |                    297 |                        269 |                       28 |                              39 |           336 |                  0.883929 |                      0.800595 |                  0.0833333  |                          0.116071  |                                7.08929  |                             -5         |                                 6         |
| centered_r_epsilon_non_qk_modules | formal_rr3_llama_centered_audit  | RR3_llama_centered_module.csv        | centered_absolute_contraction      | max_is_deepest     | non_qk          |                    240 |                        238 |                        2 |                               0 |           240 |                  1        |                      0.991667 |                  0.00833333 |                          0         |                                9.95     |                              0         |                                 9         |
| centered_r_epsilon_q_proj         | formal_rr3_llama_centered_audit  | RR3_llama_centered_module.csv        | centered_absolute_contraction      | max_is_deepest     | q_proj          |                     35 |                         22 |                       13 |                              13 |            48 |                  0.729167 |                      0.458333 |                  0.270833   |                          0.270833  |                                0.208333 |                             -5         |                                 0         |
| centered_r_epsilon_k_proj         | formal_rr3_llama_centered_audit  | RR3_llama_centered_module.csv        | centered_absolute_contraction      | max_is_deepest     | k_proj          |                     22 |                          9 |                       13 |                              26 |            48 |                  0.458333 |                      0.1875   |                  0.270833   |                          0.541667  |                               -0.333333 |                             -2         |                                -1         |

## EQUAL5_NON_QK Reuse Return: 2026-07-28T06:17:19.563050+00:00

- Task: measurement-side non-q/k equal-5 aggregation from existing module artifacts.
- Boundary: GPU used=false; new_forward=false; new_training=false; new_behavior_eval=false; new_svd=false.
- Command: `experiments/opd_sft_h1/scripts/cycle09_equal5_non_qk.py`
- Output directory: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk`
- Completed outputs:
  - `EQUAL5_coverage_inventory.csv`: 9156 rows
  - `EQUAL5_functional_cells.csv`: 6520 rows
  - `EQUAL5_functional_trajectories.csv`: 1304 rows
  - `EQUAL5_dominance_cells.csv`: 96 rows
  - `EQUAL5_dominance_summary.csv`: 12 rows
  - `EQUAL5_ncd.csv`: 32 rows
  - `EQUAL5_spectrum_robustness.csv`: 469 rows
  - `EQUAL5_support_controls.csv`: 10 rows
  - `EQUAL5_weight_baselines_module.csv`: 12000 rows
  - `EQUAL5_weight_baselines_aggregate.csv`: 2401 rows
  - `EQUAL5_output_link_correlations.csv`: 384 rows
  - `EQUAL5_checkpoint_demeaned_correlations.csv`: 24 rows
  - `EQUAL5_nested_metrics.csv`: 28 rows
  - `EQUAL5_nested_folds.csv`: 112 rows
  - `EQUAL5_nested_predictions.parquet`: 1792 rows
  - `EQUAL5_d11_parity_metrics.csv`: 41 rows
  - `EQUAL5_equal7_paired_comparison.csv`: 1306 rows
- Blocked items were recorded in `EQUAL5_coverage_inventory.csv` and `EQUAL5_manifest.json`; blocked rows were not reverse-engineered from equal-7 aggregates.

## EQUAL5_NON_QK Reuse Return: 2026-07-28T06:18:15.495260+00:00

- Task: measurement-side non-q/k equal-5 aggregation from existing module artifacts.
- Boundary: GPU used=false; new_forward=false; new_training=false; new_behavior_eval=false; new_svd=false.
- Command: `experiments/opd_sft_h1/scripts/cycle09_equal5_non_qk.py`
- Output directory: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/equal5_non_qk`
- Completed outputs:
  - `EQUAL5_coverage_inventory.csv`: 9156 rows
  - `EQUAL5_functional_cells.csv`: 6520 rows
  - `EQUAL5_functional_trajectories.csv`: 1304 rows
  - `EQUAL5_dominance_cells.csv`: 96 rows
  - `EQUAL5_dominance_summary.csv`: 12 rows
  - `EQUAL5_ncd.csv`: 32 rows
  - `EQUAL5_spectrum_robustness.csv`: 517 rows
  - `EQUAL5_support_controls.csv`: 10 rows
  - `EQUAL5_weight_baselines_module.csv`: 12000 rows
  - `EQUAL5_weight_baselines_aggregate.csv`: 2401 rows
  - `EQUAL5_output_link_correlations.csv`: 384 rows
  - `EQUAL5_checkpoint_demeaned_correlations.csv`: 24 rows
  - `EQUAL5_nested_metrics.csv`: 28 rows
  - `EQUAL5_nested_folds.csv`: 112 rows
  - `EQUAL5_nested_predictions.parquet`: 1792 rows
  - `EQUAL5_d11_parity_metrics.csv`: 41 rows
  - `EQUAL5_equal7_paired_comparison.csv`: 1306 rows
- Blocked items were recorded in `EQUAL5_coverage_inventory.csv` and `EQUAL5_manifest.json`; blocked rows were not reverse-engineered from equal-7 aggregates.



## 2026-07-28T14:43:06.984242+00:00 FAT-R1-v2 equal-5 correction return
- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_r1_v2_equal5_final.py`
- Output root: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_link_equal5`
- Manifest: `/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/fat_outlink_round1_v2_link_equal5/fat_r1_v2_equal5_manifest.json`
- Status: `COMPLETE_EQUAL5_WITH_QWEN_PK_STEP10_EXCLUDED`
- Boundary: CPU-only reuse; no training/forward/rollout/Eval/new SVD.
- Headline aggregation: non-QK equal-5 modules `v/o/gate/up/down`; q/k excluded; equal-7 retained as sensitivity.
- Qwen step10 `p_k` remains excluded, not imputed.
