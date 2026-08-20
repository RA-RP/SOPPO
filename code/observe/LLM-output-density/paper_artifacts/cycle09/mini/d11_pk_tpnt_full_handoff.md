# D11 PK-TPNT full handoff

## Status

- status: `COMPLETE_D11_CORE_PLUS_OPTIONAL_E5_E7`
- created_utc: `2026-07-27T02:45:03Z`
- protocol: `D11_PK_TPNT` from `mypaper/theory/stage_plan_handoff.md`
- machine: `2 x RTX 5090 32G`
- scope completed: `E0,E1,E2,E3,E4,E5,E6,E7`
- not run: no training, no free-generation behavior eval, no `c_epsilon/r_epsilon` recomputation
- official Llama strict `p_k` track: `bf16_merged_minus_base`
- official Qwen strict `p_k` track: reused existing deployed-effective result

This file records raw mechanical completion and readback numbers only. It does not adjudicate Theory claims.

## Runtime Readback

| lane | task block | seconds | status |
|---|---|---:|---|
| GPU0 | Llama E0/E2/E3 core | 99.278 | `COMPLETE` |
| GPU1 | Qwen E2/E3 core | 65.400 | `COMPLETE` |
| GPU0 | Llama E5/E6/E7 optional | 217.197 | `COMPLETE` |
| GPU1 | Qwen E5/E6/E7 optional | 156.356 | `COMPLETE` |
| CPU | E1/E4 + summary/handoff | completed | `COMPLETE` |

## Coverage

| artifact | rows | note |
|---|---:|---|
| `d11_llama_merged_pk.csv` | 672 | Llama four arms x six checkpoints x seven modules x four fixed-k values |
| `d11_llama_pk_numeric_audit.csv` | 672 | old adapter-BA versus deployed merged-BF16 audit |
| `d11_tpnt_principal_mask.csv` | 2016 | E2 principal-mask core table |
| `d11_tpnt_angles_pabs_nss.csv` | 1344 | E3 angles/PABS/NSS core table |
| `d11_same_cell_feature_matrix.csv` | 192 | E1/E4 same-cell feature matrix |
| `d11_same_cell_incremental_comparison.csv` | 132 | E1/E4 grouped comparison |
| `d11_e5_layer_robustness.csv` | 3024 | E5 raw layer robustness |
| `d11_e5_layer_robustness_summary.csv` | 432 | E5 summary |
| `d11_e6_alpha_sensitivity.csv` | 2016 | E6 raw alpha sensitivity |
| `d11_e6_alpha_sensitivity_summary.csv` | 288 | E6 summary |
| `d11_e7_spectrum_matched_null_seed_rows_llama.csv` | 5040 | Llama E7 10 seed rows |
| `d11_e7_spectrum_matched_null_seed_rows_qwen.csv` | 5040 | Qwen E7 10 seed rows |
| `d11_e7_spectrum_matched_null.csv` | 1008 | E7 aggregate rows |
| `d11_e7_spectrum_matched_null_summary.csv` | 144 | E7 summary |

## Protocol Checks

| check | readback |
|---|---|
| Llama E7 seed count per cell | all 504 cells have 10 seeds |
| Qwen E7 seed count per cell | all 504 cells have 10 seeds |
| Llama E7 singular values used | 1024 or 3072, max 3072 |
| Qwen E7 singular values used | 1024 or 2560, max 2560 |
| E7 max-rank setting | `0`, meaning all positive singular values retained |
| GPU residual process | none after completion |

## Llama Adapter-vs-Merged `p_k` Audit

| statistic | value |
|---|---:|
| mean `abs_diff` | 0.0000192916 |
| min `abs_diff` | 0.000000000621 |
| max `abs_diff` | 0.000255601 |
| mean `merged_minus_adapter` | -0.0000191324 |
| min `merged_minus_adapter` | -0.000255601 |
| max `merged_minus_adapter` | 0.00000825798 |

## E1/E4 Pooled Same-Cell Comparison

| target | feature set | n | heldout R2 | MAE |
|---|---|---:|---:|---:|
| cumulative KL | W | 192 | -0.054221 | 0.047651 |
| cumulative KL | `p_k` | 192 | 0.577759 | 0.030277 |
| cumulative KL | C | 192 | 0.503778 | 0.031049 |
| cumulative KL | W+C | 192 | 0.579226 | 0.031760 |
| cumulative KL | `p_k`+C | 192 | 0.598257 | 0.028095 |
| cumulative KL | TPNT | 192 | 0.229989 | 0.038037 |
| cumulative KL | TPNT+C | 192 | 0.460166 | 0.035240 |
| cumulative KL | `p_k`+TPNT+C | 192 | 0.579358 | 0.030623 |
| absolute NLL | W | 192 | -0.113600 | 0.099811 |
| absolute NLL | `p_k` | 192 | 0.618975 | 0.056702 |
| absolute NLL | C | 192 | 0.501251 | 0.064835 |
| absolute NLL | W+C | 192 | 0.551257 | 0.064447 |
| absolute NLL | `p_k`+C | 192 | 0.628385 | 0.054332 |
| absolute NLL | TPNT | 192 | 0.318462 | 0.074980 |
| absolute NLL | TPNT+C | 192 | 0.455408 | 0.069321 |
| absolute NLL | `p_k`+TPNT+C | 192 | 0.585432 | 0.060428 |
| signed NLL | W | 192 | -0.012028 | 0.034721 |
| signed NLL | `p_k` | 192 | 0.523450 | 0.025606 |
| signed NLL | C | 192 | 0.336307 | 0.029289 |
| signed NLL | W+C | 192 | 0.409843 | 0.028133 |
| signed NLL | `p_k`+C | 192 | 0.508595 | 0.026493 |
| signed NLL | TPNT | 192 | 0.105505 | 0.030971 |
| signed NLL | TPNT+C | 192 | 0.313769 | 0.031647 |
| signed NLL | `p_k`+TPNT+C | 192 | 0.573449 | 0.025070 |

## Delta-R2 Readback

| model scope | target | comparison | delta R2 |
|---|---|---|---:|
| pooled | cumulative KL | W+C minus W | 0.633447 |
| pooled | cumulative KL | `p_k`+C minus `p_k` | 0.020499 |
| pooled | cumulative KL | TPNT+C minus TPNT | 0.230178 |
| pooled | cumulative KL | `p_k`+TPNT+C minus `p_k` | 0.001599 |
| pooled | absolute NLL | W+C minus W | 0.664857 |
| pooled | absolute NLL | `p_k`+C minus `p_k` | 0.009410 |
| pooled | absolute NLL | TPNT+C minus TPNT | 0.136945 |
| pooled | absolute NLL | `p_k`+TPNT+C minus `p_k` | -0.033543 |
| pooled | signed NLL | W+C minus W | 0.421871 |
| pooled | signed NLL | `p_k`+C minus `p_k` | -0.014856 |
| pooled | signed NLL | TPNT+C minus TPNT | 0.208264 |
| pooled | signed NLL | `p_k`+TPNT+C minus `p_k` | 0.049999 |

## E5-E7 Aggregate Readback

| table | statistic | mean | min | max |
|---|---|---:|---:|---:|
| E5 layer robustness | mean overlap lift | 0.766630 | 0.310753 | 0.962359 |
| E5 layer robustness | mean PABS joint cos | 0.999640 | 0.996976 | 0.999787 |
| E5 layer robustness | mean NSS L1 top32 | 0.0000650812 | 0.0000376302 | 0.000251220 |
| E6 alpha sensitivity | mean overlap lift | 0.803779 | 0.310753 | 0.970437 |
| E6 alpha sensitivity | mean coverage | 0.078835 | 0.003108 | 0.194087 |
| E7 spectrum null | mean real overlap lift | 0.749852 | 0.310753 | 0.951327 |
| E7 spectrum null | mean null overlap lift | 0.747134 | 0.308085 | 0.951495 |
| E7 spectrum null | mean z TPNT | 0.711935 | -2.227837 | 3.002072 |

## Output Paths

Primary handoffs:

```text
/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_pk_tpnt_full_handoff.md
/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_modelwise_and_crossmodel_summary.md
/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_e5_e7_optional_handoff.md
```

Primary manifests:

```text
/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_pk_tpnt_manifest.json
/root/autodl-tmp/cycle09_relative_functional_contraction/d11_pk_tpnt/formal/final/d11_e5_e7_manifest.json
```

Mini mirrors:

```text
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/d11_pk_tpnt_full_handoff.md
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/d11_pk_tpnt_manifest.json
/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/d11_e5_e7_manifest.json
```

Code handoff log:

```text
/root/LLM-output-density/mypaper/code/code_evolution.md
```
