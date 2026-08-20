# D11 PK-TPNT handoff

## Status

- status: `COMPLETE_D11_CORE`
- tag: `formal`
- created_utc: `2026-07-27T02:33:00.912825+00:00`
- protocol: `D11_PK_TPNT` from `mypaper/theory/stage_plan_handoff.md`
- Llama strict p_k official track: `bf16_merged_minus_base`
- Qwen strict p_k official track: reused existing deployed-effective result
- no training, no free-generation behavior eval, no c_epsilon/r_epsilon recomputation

## Coverage

| artifact | rows |
|---|---:|
| d11_llama_merged_pk.csv | 672 |
| d11_llama_pk_numeric_audit.csv | 672 |
| d11_tpnt_principal_mask.csv | 2016 |
| d11_tpnt_angles_pabs_nss.csv | 1344 |
| d11_same_cell_feature_matrix.csv | 192 |
| d11_same_cell_incremental_comparison.csv | 132 |

## Raw Comparison Head

| analysis | model_scope | target | feature_set | features | n | heldout_r2 | mae | auc | balanced_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | W | raw_update_energy_equal7 | 192 | -0.0542214 | 0.0476514 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | p_k | p_k4,p_k8,p_k16,p_k32 | 192 | 0.577759 | 0.0302773 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | C | c_epsilon | 192 | 0.503778 | 0.0310494 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | W_plus_C | raw_update_energy_equal7,c_epsilon | 192 | 0.579226 | 0.03176 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | p_k_plus_C | p_k4,p_k8,p_k16,p_k32,c_epsilon | 192 | 0.598257 | 0.0280954 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | TPNT | tpnt_overlap_lift,tpnt_lift_minus_null,pabs_joint_mean_cos,nss_l1_top32 | 192 | 0.229989 | 0.0380365 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | TPNT_plus_C | tpnt_overlap_lift,tpnt_lift_minus_null,pabs_joint_mean_cos,nss_l1_top32,c_epsilon | 192 | 0.460166 | 0.0352395 |  |  |
| checkpoint_grouped_regression | pooled | cumulative_kl_base_to_current | p_k_TPNT_C | p_k4,p_k8,p_k16,p_k32,tpnt_overlap_lift,tpnt_lift_minus_null,pabs_joint_mean_cos,nss_l1_top32,c_epsilon | 192 | 0.579358 | 0.0306232 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | W | raw_update_energy_equal7 | 192 | -0.1136 | 0.0998108 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | p_k | p_k4,p_k8,p_k16,p_k32 | 192 | 0.618975 | 0.0567024 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | C | c_epsilon | 192 | 0.501251 | 0.0648346 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | W_plus_C | raw_update_energy_equal7,c_epsilon | 192 | 0.551257 | 0.0644467 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | p_k_plus_C | p_k4,p_k8,p_k16,p_k32,c_epsilon | 192 | 0.628385 | 0.0543319 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | TPNT | tpnt_overlap_lift,tpnt_lift_minus_null,pabs_joint_mean_cos,nss_l1_top32 | 192 | 0.318462 | 0.0749796 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | TPNT_plus_C | tpnt_overlap_lift,tpnt_lift_minus_null,pabs_joint_mean_cos,nss_l1_top32,c_epsilon | 192 | 0.455408 | 0.0693206 |  |  |
| checkpoint_grouped_regression | pooled | absolute_delta_nll_cumulative | p_k_TPNT_C | p_k4,p_k8,p_k16,p_k32,tpnt_overlap_lift,tpnt_lift_minus_null,pabs_joint_mean_cos,nss_l1_top32,c_epsilon | 192 | 0.585432 | 0.0604283 |  |  |
| checkpoint_grouped_regression | pooled | delta_nll_cumulative | W | raw_update_energy_equal7 | 192 | -0.0120278 | 0.0347211 |  |  |
| checkpoint_grouped_regression | pooled | delta_nll_cumulative | p_k | p_k4,p_k8,p_k16,p_k32 | 192 | 0.52345 | 0.0256056 |  |  |
| checkpoint_grouped_regression | pooled | delta_nll_cumulative | C | c_epsilon | 192 | 0.336307 | 0.0292893 |  |  |
| checkpoint_grouped_regression | pooled | delta_nll_cumulative | W_plus_C | raw_update_energy_equal7,c_epsilon | 192 | 0.409843 | 0.0281334 |  |  |

## Boundaries

- TPNT random-null column uses a rank-32 Frobenius-norm-matched low-rank BF16 update mask with fixed deterministic seeds.
- E5--E7 optional enhancements are now complete in the optional addendum below.
- This handoff reports raw tables and mechanical status only; it does not adjudicate Theory claims.

## Optional E5--E7 Addendum

| optional task | status | rows |
|---|---|---:|
| E5 layer robustness | `COMPLETE` | 3024 raw / 432 summary |
| E6 alpha sensitivity | `COMPLETE` | 2016 raw / 288 summary |
| E7 spectrum-matched random-subspace null | `COMPLETE` | 1008 aggregate / 144 summary |

Optional handoff:
`/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_e5_e7_optional_handoff.md`.

Optional manifest:
`/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_e5_e7_manifest.json`.

## Output Files

- `d11_pk_tpnt_task_status.csv`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_pk_tpnt_task_status.csv`
- `d11_llama_merged_pk.csv`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_llama_merged_pk.csv`
- `d11_llama_pk_numeric_audit.csv`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_llama_pk_numeric_audit.csv`
- `d11_tpnt_principal_mask.csv`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_tpnt_principal_mask.csv`
- `d11_tpnt_angles_pabs_nss.csv`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_tpnt_angles_pabs_nss.csv`
- `d11_same_cell_incremental_comparison.csv`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_same_cell_incremental_comparison.csv`
- `d11_modelwise_and_crossmodel_summary.md`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_modelwise_and_crossmodel_summary.md`
- `d11_pk_tpnt_manifest.json`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_pk_tpnt_manifest.json`
- `d11_e5_e7_optional_handoff.md`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_e5_e7_optional_handoff.md`
- `d11_e5_e7_manifest.json`: `/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_e5_e7_manifest.json`
