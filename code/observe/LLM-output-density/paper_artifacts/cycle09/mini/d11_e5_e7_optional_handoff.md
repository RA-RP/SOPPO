# D11 E5-E7 optional enhancement handoff

## Status

- status: `COMPLETE_D11_OPTIONAL_E5_E7`
- created_utc: `2026-07-27T02:45:03.008673+00:00`
- no training, no behavior eval, no c_epsilon/r_epsilon recomputation

## Coverage

| artifact | rows |
|---|---:|
| d11_e5_layer_robustness.csv | 3024 |
| d11_e5_layer_robustness_summary.csv | 432 |
| d11_e6_alpha_sensitivity.csv | 2016 |
| d11_e6_alpha_sensitivity_summary.csv | 288 |
| d11_e7_spectrum_matched_null.csv | 1008 |
| d11_e7_spectrum_matched_null_summary.csv | 144 |

## E7 Summary Head

| model | arm | checkpoint | source_rank_k | mask_density_alpha | mean_real_overlap_lift | mean_null_overlap_lift | mean_z_tpnt | median_z_tpnt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama | offkd | 20 | 16 | 0.01 | 0.608629 | 0.592432 | 2.38552 | 2.06756 |
| llama | offkd | 20 | 16 | 0.1 | 0.85214 | 0.843791 | 2.12287 | 1.59488 |
| llama | offkd | 20 | 32 | 0.01 | 0.508544 | 0.496361 | 2.28617 | 1.88832 |
| llama | offkd | 20 | 32 | 0.1 | 0.798122 | 0.790813 | 2.35885 | 1.77904 |
| llama | offkd | 20 | 50 | 0.01 | 0.432362 | 0.422039 | 2.25975 | 2.51915 |
| llama | offkd | 20 | 50 | 0.1 | 0.7503 | 0.743994 | 2.15982 | 1.9988 |
| llama | offkd | 160 | 16 | 0.01 | 0.745968 | 0.744455 | 0.53792 | 0.0215246 |
| llama | offkd | 160 | 16 | 0.1 | 0.909985 | 0.911248 | 0.153461 | -0.541032 |
| llama | offkd | 160 | 32 | 0.01 | 0.66845 | 0.671106 | -0.233734 | -1.20315 |
| llama | offkd | 160 | 32 | 0.1 | 0.875734 | 0.879078 | -0.711361 | -1.79875 |
| llama | offkd | 160 | 50 | 0.01 | 0.605944 | 0.611911 | -0.938019 | -1.74671 |
| llama | offkd | 160 | 50 | 0.1 | 0.84566 | 0.850749 | -1.52029 | -2.49135 |
| llama | offkd | 320 | 16 | 0.01 | 0.785135 | 0.78686 | -0.743881 | -0.736592 |
| llama | offkd | 320 | 16 | 0.1 | 0.926147 | 0.928492 | -0.501388 | -1.02176 |
| llama | offkd | 320 | 32 | 0.01 | 0.720504 | 0.722952 | -0.491398 | -0.360963 |
| llama | offkd | 320 | 32 | 0.1 | 0.898591 | 0.901733 | -0.669031 | -1.24282 |
| llama | offkd | 320 | 50 | 0.01 | 0.664839 | 0.670676 | -1.29164 | -1.5343 |
| llama | offkd | 320 | 50 | 0.1 | 0.873715 | 0.878262 | -1.23137 | -1.49656 |
| llama | opd | 20 | 16 | 0.01 | 0.641318 | 0.624133 | 1.9786 | 1.84185 |
| llama | opd | 20 | 16 | 0.1 | 0.868604 | 0.860215 | 2.23129 | 3.10576 |

## Boundaries

- E7 uses 10 fixed spectrum-matched random-subspace seeds per landmark cell.
- Spectrum-matched null preserves the selected positive singular values of the deployed BF16 merged-minus-base update.
- Layer x module rows are reported as mechanical cells, not independent statistical seeds.
