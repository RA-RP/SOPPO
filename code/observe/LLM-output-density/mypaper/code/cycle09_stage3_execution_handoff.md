# Cycle 09 Stage 3 Execution Handoff

Snapshot: 2026-07-20 UTC. Formal execution completed and all seven manifests passed
strict validation.

## Frozen scope

This block implements C2, C3, C5, C8, C11, C14, and C15 from Appendix C of
`mypaper/theory/human_read-ch.md`.

- C2: sample-bootstrap CI for fixed-probe raw ER, L18, 256 draws.
- C3: OPD step 0/40/160 overcompression and rebound CI, L18, five probes.
- C5: fixed IFEval prompt probe `E_if`, four arms x ten checkpoints x L9/L18/L27.
- C8: response-only conditional PPL on three frozen training-text corpora.
- C11: full-vocabulary next-token entropy at the frozen MMLU-Pro answer position.
  Its secondary candidate mass uses all valid options present in each frozen row; the
  audited source has 3-10 options rather than a uniform ten.
- C14: `per_checkpoint` L9/L18/L27 backfill and descriptive C7 event tables.
- C15: corrected cap-hit statistic on the existing 60-sample pilot.

C4 is a separate large held-out campaign. C6 remains unidentified, and C13 requires a
new controlled training design. They are not silently included in this block.

The frozen contracts are:

- `mypaper/code/cycle09_stage3_execution_contract.json`
- `mypaper/code/cycle09_c7_prospective_contract.json`

## Completed before GPU launch

C14 and C15 are complete because they only read existing artifacts.

| artifact | rows | sha256 |
|---|---:|---|
| `C14_per_checkpoint_layer_sensitivity.csv` | 600 | `b1293bcf3311bfbc47d7348cf989eee8353b78dd8ac82aa28cb352941f055ab7` |
| `C14_c7_geometry_events.csv` | 12 | `82687b5208136b1330df5618ecd0fa81b5e8b6905c8750d886a2347b792d0ffd` |
| `C14_c7_behavior_extrema.csv` | 56 | `9ccd7b329806f5e1706a0585925d2e7eb537a0437b0f4420712efaa66a4c555e` |
| `C14_c7_lead_lag_descriptive.csv` | 140 | `63d815496e6aaa058f9afcd4d97d6e47120a34233e1a0de50a4035c7e0b47e5e` |
| `C15_cap_pilot_corrected.csv` | 1 | `14dfd3213beb1a0ab096baff4e4a528e33df398a21457fcb18fbc6e30a27bab7` |
| `C15_cap_pilot_samples_corrected.csv` | 60 | `868d2776007d3c1e1afe80f7872cedba26b2429cea122a6dc0b019ba930f3be7` |
| `C15_cap_unpaired_aggregate_comparison.csv` | 2 | `26fc9db83b1971b6e38719d394c1420103d52bbdc2acc87c4fc8fcdd22fd1ea3` |

C15 raw reading: cap-hit = 38/60 under
`finish == length OR response_tokens >= generation_cap`. The N=60 and N=500 runs
remain unpaired.

## Current data gate

The latest 2026-07-20 preflight found both 96 GiB GPUs, the base model, all nonzero
four-arm checkpoint models, and the three 32-row base `S_bos` seed corpora. An earlier
preflight report incorrectly looked under an extra `base/step_000` path component; the
formal files were already present under `corpora/generated/S/bos/`. The path gate now
matches the R4 common resolver.

C5 (541 rows), C8 (3 x 500 rows), and C11 (1400 rows) have already been tokenized
and frozen under `/root/autodl-tmp/cycle09_stage3/`. The C11 source audit found this
option-count distribution: `{3:6, 4:80, 5:8, 6:9, 7:21, 8:43, 9:94, 10:1139}`.

The machine copy may still continue for files outside this scope. The supervisor always regenerates
`/root/autodl-tmp/cycle09_stage3/supervisor/preflight.json` and requires all model
shards and required files to stay size-stable before launch.

## Detached launch

From `/root/LLM-output-density`, launch after the copy is ready:

```bash
/root/miniconda3/envs/density/bin/python \
  experiments/opd_sft_h1/scripts/cycle09_stage3_supervisor.py \
  --detach --wait-for-data --shutdown never
```

This process is independent of VS Code. To request AutoDL shutdown only after strict
success validation, replace `--shutdown never` with `--shutdown success`. During its
120-second grace window, cancel shutdown with:

```bash
touch /root/autodl-tmp/cycle09_stage3/supervisor/ABORT_SHUTDOWN
```

Status and logs:

```bash
/root/miniconda3/envs/density/bin/python \
  experiments/opd_sft_h1/scripts/cycle09_stage3_supervisor.py --status
```

- status: `/root/autodl-tmp/cycle09_stage3/supervisor/status.json`
- supervisor log: `/root/autodl-tmp/cycle09_stage3/supervisor/logs/supervisor.log`
- per-task logs: `/root/autodl-tmp/cycle09_stage3/supervisor/logs/`

GPU0 runs OPD/SFT plus the OPD-only C3 task. GPU1 runs off-KD/seqKD. Shared base cells
use file locks; every checkpoint cell is resumable. Final CSVs are produced only after
both lanes succeed, then seven completion manifests and all listed output hashes are
validated.

## Verification already passed

- Python compilation for all seven task scripts and the supervisor.
- C2 raw-ER synthetic bootstrap smoke.
- C5 spectrum/rank smoke against the squared singular-value energy definition.
- C8 dynamic batching, causal shift, response token count, and token-weighted NLL.
- C11 right-padding answer-position selection and full-vocabulary entropy.
- A real-base smoke compares full-wrapper logits with selective output-head logits
  before either GPU lane starts.
- The completed real-base smoke measured C8 NLL absolute difference `5.59e-8` and
  C11 entropy absolute difference `3.70e-6`; its raw manifest is
  `/root/autodl-tmp/cycle09_stage3/supervisor/model_smoke.json`.
- C14/C15 source-grid and artifact-manifest checks.
- Supervisor command dry-run and two-GPU inventory check.

<!-- cycle09-stage3-final-start -->

## Final execution status

The detached supervisor started at `2026-07-19T22:02:54.679591+00:00` and finished with final state `SHUTDOWN_CANCELLED` at `2026-07-20T06:32:27.144413+00:00`. `failure` is null, both GPU lanes are complete, and all finalizers are complete. The user-requested marker `/root/autodl-tmp/cycle09_stage3/supervisor/ABORT_SHUTDOWN` cancelled machine shutdown after validation.

| task | status | primary rows | retained detailed rows |
|---|---:|---:|---:|
| C2 raw ER bootstrap | complete | 24 bootstrap + 4 transient + 2 audit | sample factors in scratch |
| C3 OPD overcompression/rebound | complete | 120 | sample factors in scratch |
| C5 fixed E_if geometry | complete | 3360 | 40 spectra inventory |
| C8 training-corpus PPL | complete | 120 | 60000 sample + 40 inventory |
| C11 answer-position entropy | complete | 40 | 560 category + 56000 sample + 40 inventory |
| C14 main-track/C7 backfill | complete | 600 + 12 + 56 + 140 | source artifacts unchanged |
| C15 cap-pilot repair | complete | 1 + 60 + 2 | source artifacts unchanged |

## C2 raw ER readings

Protocol: `E_ood`, L18, normalized raw ER current minus base, 256 bootstrap draws, sample resampling with windows nested. The common step-0 value is `0.00043118358007632196`; all four step-0 deltas and CIs are zero.

| arm | step | point delta | bootstrap mean | 95% CI | excludes 0 |
|---|---:|---:|---:|---:|---:|
| opd | 5 | -2.82598194e-07 | -2.42388978e-07 | [-9.14882548e-07, 3.10672476e-07] | False |
| opd | 10 | -9.54023562e-08 | -8.30107183e-08 | [-5.76601451e-07, 3.57453246e-07] | False |
| opd | 20 | 7.27479346e-07 | 7.67857955e-07 | [1.25281076e-07, 1.33472713e-06] | True |
| opd | 40 | 3.59781552e-07 | 3.9322515e-07 | [-7.48543243e-07, 1.22764686e-06] | False |
| opd | 80 | 1.93907181e-06 | 1.96741439e-06 | [7.70120096e-07, 3.10510586e-06] | True |
| sft | 5 | 2.20898073e-08 | 1.76548838e-08 | [-4.22805897e-08, 8.12906364e-08] | False |
| sft | 10 | -2.02853698e-08 | -1.57402837e-08 | [-1.52562279e-07, 9.9043973e-08] | False |
| sft | 20 | -2.44559487e-07 | -2.30651835e-07 | [-7.85883458e-07, 1.4791658e-07] | False |
| sft | 40 | 3.57860699e-07 | 3.89428465e-07 | [-1.47654646e-07, 7.94279913e-07] | False |
| sft | 80 | 2.9738294e-07 | 3.60387503e-07 | [-8.61920853e-07, 1.20100231e-06] | False |
| offkd | 5 | -3.11294571e-07 | -3.15101829e-07 | [-9.46405635e-07, 1.51914719e-07] | False |
| offkd | 10 | 1.62573997e-07 | 1.79480253e-07 | [-1.69460691e-07, 4.71231033e-07] | False |
| offkd | 20 | 7.04691047e-07 | 7.21288643e-07 | [2.27824785e-07, 1.13190617e-06] | True |
| offkd | 40 | 5.92903234e-07 | 6.06698109e-07 | [-1.42303179e-07, 1.31971319e-06] | False |
| offkd | 80 | -5.70085831e-07 | -5.29530439e-07 | [-1.66541577e-06, 4.25035978e-07] | False |
| seqkd | 5 | -1.40425982e-07 | -1.37275492e-07 | [-5.46333467e-07, 1.35627488e-07] | False |
| seqkd | 10 | 5.0116796e-07 | 4.93824587e-07 | [6.6429493e-08, 8.6685759e-07] | True |
| seqkd | 20 | 6.26780093e-07 | 6.24121299e-07 | [-3.09715688e-07, 1.52657958e-06] | False |
| seqkd | 40 | 7.00412784e-07 | 6.71342832e-07 | [-2.36494088e-07, 1.45719605e-06] | False |
| seqkd | 80 | -7.57572707e-07 | -8.40897769e-07 | [-2.33040555e-06, 5.90232958e-07] | False |

| arm | max-transient bootstrap mean | 95% CI | P(max > 0) | peak-step draw counts |
|---|---:|---:|---:|---:|
| opd | 1.96905421e-06 | [7.70120096e-07, 3.10510586e-06] | 1.000000 | 5:0, 10:0, 20:5, 40:0, 80:251 |
| sft | 5.38755557e-07 | [6.1118044e-10, 1.20100231e-06] | 0.972656 | 5:13, 10:2, 20:0, 40:108, 80:133 |
| offkd | 7.62897685e-07 | [3.07092705e-07, 1.36133167e-06] | 1.000000 | 5:0, 10:8, 20:171, 40:77, 80:0 |
| seqkd | 7.63267167e-07 | [1.44089427e-07, 1.52657958e-06] | 0.996094 | 5:0, 10:65, 20:74, 40:117, 80:0 |

The protocol audit marks both `R5_raw_er_fixed_ckpt.csv` and `R5_raw_er_fixed.csv` as `directly_mergeable_with_C2=False`.

## C3 OPD readings

Protocol: per-checkpoint L18 `r_epsilon@0.05`, five fixed probes, 256 sample-bootstrap draws. The table reports the fixed seven-module mean rows.

| probe | contrast | point | bootstrap mean | 95% CI | excludes 0 |
|---|---:|---:|---:|---:|---:|
| legacy_S_math | overcompression_depth_r0_minus_r40 | 25.000000 | 25.400112 | [24.053571, 26.660714] | True |
| legacy_S_math | rebound_r160_minus_r40 | 16.285714 | 15.738839 | [14.857143, 16.571429] | True |
| legacy_S_math | net_r160_minus_r0 | -8.714286 | -9.661272 | [-11.428571, -8.053571] | True |
| E_ood | overcompression_depth_r0_minus_r40 | 10.000000 | 10.064174 | [9.339286, 10.857143] | True |
| E_ood | rebound_r160_minus_r40 | -4.000000 | -3.332589 | [-4.142857, -2.482143] | True |
| E_ood | net_r160_minus_r0 | -14.000000 | -13.396763 | [-14.428571, -12.339286] | True |
| E_general | overcompression_depth_r0_minus_r40 | 10.571429 | 9.974888 | [9.428571, 10.517857] | True |
| E_general | rebound_r160_minus_r40 | 0.857143 | 0.674107 | [0.142857, 1.285714] | True |
| E_general | net_r160_minus_r0 | -9.714286 | -9.300781 | [-10.000000, -8.714286] | True |
| E_math_hard | overcompression_depth_r0_minus_r40 | 4.428571 | 3.362165 | [2.428571, 4.142857] | True |
| E_math_hard | rebound_r160_minus_r40 | -2.571429 | -1.982143 | [-2.857143, -1.142857] | True |
| E_math_hard | net_r160_minus_r0 | -7.000000 | -5.344308 | [-6.714286, -3.714286] | True |
| S_bos | overcompression_depth_r0_minus_r40 | 5.285714 | 4.552269 | [3.238095, 5.934524] | True |
| S_bos | rebound_r160_minus_r40 | 4.476190 | 4.155320 | [3.333333, 5.029762] | True |
| S_bos | net_r160_minus_r0 | -0.809524 | -0.396949 | [-1.857143, 1.095238] | False |

## C5 fixed E_if readings

The table is the mechanical mean of the seven module rows for `track=per_checkpoint`, `epsilon=0.05`, and `task_id=E_if`. The full output retains both tracks, both epsilon values, all seven modules, and L9/L18/L27.

| arm | step | L9 mean r_epsilon_delta | L18 mean r_epsilon_delta | L27 mean r_epsilon_delta |
|---|---:|---:|---:|---:|
| opd | 0 | 0.000000 | 0.000000 | 0.000000 |
| opd | 5 | -0.571429 | -0.142857 | -7.571429 |
| opd | 10 | -1.142857 | 0.000000 | -15.571429 |
| opd | 20 | -1.142857 | 0.285714 | -33.714286 |
| opd | 40 | -2.571429 | -5.857143 | -54.571429 |
| opd | 80 | -4.000000 | -5.285714 | -53.714286 |
| opd | 160 | -7.857143 | -9.571429 | -54.000000 |
| opd | 320 | -9.428571 | -16.714286 | -59.142857 |
| opd | 480 | -10.714286 | -20.000000 | -61.142857 |
| opd | 624 | -12.428571 | -26.571429 | -65.714286 |
| sft | 0 | 0.000000 | 0.000000 | 0.000000 |
| sft | 5 | 0.000000 | 0.142857 | 0.000000 |
| sft | 10 | 0.000000 | 0.142857 | -0.285714 |
| sft | 20 | -0.285714 | 0.428571 | -3.142857 |
| sft | 40 | 0.571429 | 2.428571 | -1.428571 |
| sft | 80 | -0.857143 | -1.571429 | 21.428571 |
| sft | 160 | -2.428571 | -4.285714 | 13.857143 |
| sft | 320 | -1.857143 | -8.142857 | 23.857143 |
| sft | 480 | -1.714286 | -9.857143 | 24.571429 |
| sft | 624 | -1.285714 | -9.714286 | 27.428571 |
| offkd | 0 | 0.000000 | 0.000000 | 0.000000 |
| offkd | 5 | -0.571429 | 0.714286 | -5.142857 |
| offkd | 10 | -0.285714 | 2.000000 | -8.428571 |
| offkd | 20 | 0.142857 | 3.142857 | -14.428571 |
| offkd | 40 | -1.571429 | 2.000000 | -14.000000 |
| offkd | 80 | -3.714286 | -2.142857 | -6.285714 |
| offkd | 160 | -6.571429 | -8.000000 | -2.142857 |
| offkd | 320 | -6.285714 | -12.285714 | -3.000000 |
| offkd | 480 | -8.000000 | -15.142857 | -5.142857 |
| offkd | 624 | -7.857143 | -15.285714 | -5.142857 |
| seqkd | 0 | 0.000000 | 0.000000 | 0.000000 |
| seqkd | 5 | -0.285714 | 0.428571 | -6.142857 |
| seqkd | 10 | 0.142857 | 1.285714 | -9.857143 |
| seqkd | 20 | 0.285714 | 2.714286 | -14.142857 |
| seqkd | 40 | -0.428571 | 1.714286 | -9.714286 |
| seqkd | 80 | -2.857143 | -2.714286 | -3.714286 |
| seqkd | 160 | -4.428571 | -9.857143 | 0.000000 |
| seqkd | 320 | -4.142857 | -15.857143 | 2.285714 |
| seqkd | 480 | -6.428571 | -19.142857 | -0.857143 |
| seqkd | 624 | -7.000000 | -18.714286 | 0.857143 |

## C8 training-corpus PPL readings

PPL is response-only conditional perplexity, aggregated by token-weighted NLL over 500 frozen samples per corpus and cell.

| corpus | step | OPD PPL | SFT PPL | off-KD PPL | seqKD PPL |
|---|---:|---:|---:|---:|---:|
| X_OPD_reconstructed | 0 | 1.038410 | 1.038410 | 1.038410 | 1.038410 |
| X_OPD_reconstructed | 5 | 1.052450 | 1.038363 | 1.043615 | 1.042405 |
| X_OPD_reconstructed | 10 | 1.090591 | 1.038589 | 1.043840 | 1.043319 |
| X_OPD_reconstructed | 20 | 1.186531 | 1.040697 | 1.050849 | 1.053398 |
| X_OPD_reconstructed | 40 | 1.196647 | 1.045368 | 1.060784 | 1.063706 |
| X_OPD_reconstructed | 80 | 1.181129 | 1.072261 | 1.070220 | 1.073973 |
| X_OPD_reconstructed | 160 | 1.173722 | 1.090940 | 1.080307 | 1.085441 |
| X_OPD_reconstructed | 320 | 1.171417 | 1.100265 | 1.082125 | 1.093052 |
| X_OPD_reconstructed | 480 | 1.180622 | 1.106412 | 1.084073 | 1.092131 |
| X_OPD_reconstructed | 624 | 1.187636 | 1.107400 | 1.083953 | 1.098189 |
| X_SFT_dataset | 0 | 1.868036 | 1.868036 | 1.868036 | 1.868036 |
| X_SFT_dataset | 5 | 1.830098 | 1.866482 | 1.814799 | 1.784036 |
| X_SFT_dataset | 10 | 1.859069 | 1.849829 | 1.748470 | 1.728783 |
| X_SFT_dataset | 20 | 2.110997 | 1.805714 | 1.696091 | 1.676019 |
| X_SFT_dataset | 40 | 2.065181 | 1.708430 | 1.646132 | 1.631828 |
| X_SFT_dataset | 80 | 1.994358 | 1.603631 | 1.606372 | 1.595584 |
| X_SFT_dataset | 160 | 1.903762 | 1.555613 | 1.578966 | 1.570566 |
| X_SFT_dataset | 320 | 1.829888 | 1.525279 | 1.558638 | 1.554362 |
| X_SFT_dataset | 480 | 1.806039 | 1.512817 | 1.547963 | 1.546525 |
| X_SFT_dataset | 624 | 1.785221 | 1.510596 | 1.540342 | 1.539495 |
| X_teacher | 0 | 1.684757 | 1.684757 | 1.684757 | 1.684757 |
| X_teacher | 5 | 1.653995 | 1.683617 | 1.639821 | 1.613499 |
| X_teacher | 10 | 1.676241 | 1.670891 | 1.583165 | 1.563319 |
| X_teacher | 20 | 1.932194 | 1.632375 | 1.533167 | 1.511665 |
| X_teacher | 40 | 1.928439 | 1.549209 | 1.487115 | 1.465857 |
| X_teacher | 80 | 1.864933 | 1.463699 | 1.450861 | 1.429264 |
| X_teacher | 160 | 1.785449 | 1.426905 | 1.424371 | 1.401370 |
| X_teacher | 320 | 1.716648 | 1.406260 | 1.400276 | 1.376276 |
| X_teacher | 480 | 1.694113 | 1.398363 | 1.387354 | 1.362354 |
| X_teacher | 624 | 1.673447 | 1.397432 | 1.378267 | 1.351308 |

## C11 MMLU-Pro answer-position readings

The frozen prompt has 1400 questions. `H_vocab` is full-vocabulary next-token entropy in nats; option-token quantities use every available option in each row. The audited option count ranges from 3 to 10.

| arm | step | H_vocab mean | effective vocabulary mean | gold option-token p mean | option-token mass mean | H_option restricted mean |
|---|---:|---:|---:|---:|---:|---:|
| opd | 0 | 2.401994 | 15.751856 | 0.089281 | 0.191499 | 0.630959 |
| opd | 5 | 2.453950 | 16.415218 | 0.087531 | 0.189929 | 0.638103 |
| opd | 10 | 2.627936 | 19.698211 | 0.084037 | 0.186044 | 0.648252 |
| opd | 20 | 3.390581 | 43.074087 | 0.070640 | 0.164539 | 0.668242 |
| opd | 40 | 3.720261 | 56.716026 | 0.063405 | 0.152437 | 0.695480 |
| opd | 80 | 3.958606 | 75.722250 | 0.064330 | 0.152054 | 0.713288 |
| opd | 160 | 4.273904 | 105.560347 | 0.057971 | 0.140809 | 0.720107 |
| opd | 320 | 4.538243 | 137.962884 | 0.054662 | 0.134133 | 0.728233 |
| opd | 480 | 4.657324 | 157.694780 | 0.054246 | 0.132467 | 0.733049 |
| opd | 624 | 4.745845 | 172.601932 | 0.052293 | 0.130130 | 0.735301 |
| sft | 0 | 2.401994 | 15.751856 | 0.089281 | 0.191499 | 0.630959 |
| sft | 5 | 2.404403 | 15.819620 | 0.089262 | 0.191327 | 0.631676 |
| sft | 10 | 2.414821 | 15.957646 | 0.089104 | 0.191164 | 0.632621 |
| sft | 20 | 2.505777 | 17.607342 | 0.088476 | 0.190183 | 0.635012 |
| sft | 40 | 2.629302 | 20.054174 | 0.087903 | 0.188050 | 0.637022 |
| sft | 80 | 2.622531 | 19.645775 | 0.090469 | 0.189701 | 0.631870 |
| sft | 160 | 2.628267 | 19.869975 | 0.089180 | 0.188705 | 0.639254 |
| sft | 320 | 2.650055 | 20.381970 | 0.086975 | 0.187739 | 0.640059 |
| sft | 480 | 2.670085 | 20.715346 | 0.086101 | 0.186172 | 0.640053 |
| sft | 624 | 2.658563 | 20.446700 | 0.086155 | 0.186529 | 0.640888 |
| offkd | 0 | 2.401994 | 15.751856 | 0.089281 | 0.191499 | 0.630959 |
| offkd | 5 | 2.594879 | 19.622889 | 0.088158 | 0.189171 | 0.636937 |
| offkd | 10 | 2.612339 | 19.824182 | 0.087605 | 0.188192 | 0.636716 |
| offkd | 20 | 2.680612 | 21.099058 | 0.087023 | 0.185855 | 0.635201 |
| offkd | 40 | 2.649728 | 20.231481 | 0.088783 | 0.186900 | 0.628909 |
| offkd | 80 | 2.672975 | 21.196053 | 0.089720 | 0.186961 | 0.622501 |
| offkd | 160 | 2.685540 | 21.324764 | 0.087665 | 0.185497 | 0.625909 |
| offkd | 320 | 2.758792 | 22.628499 | 0.084865 | 0.180928 | 0.628328 |
| offkd | 480 | 2.735418 | 21.754168 | 0.083375 | 0.180090 | 0.628142 |
| offkd | 624 | 2.776655 | 22.322989 | 0.081414 | 0.176705 | 0.627578 |
| seqkd | 0 | 2.401994 | 15.751856 | 0.089281 | 0.191499 | 0.630959 |
| seqkd | 5 | 2.565424 | 18.898397 | 0.088362 | 0.189557 | 0.635909 |
| seqkd | 10 | 2.594210 | 19.430736 | 0.088052 | 0.188523 | 0.634566 |
| seqkd | 20 | 2.687736 | 21.239810 | 0.087184 | 0.186007 | 0.632599 |
| seqkd | 40 | 2.667340 | 20.665887 | 0.089558 | 0.187410 | 0.624511 |
| seqkd | 80 | 2.668760 | 20.976177 | 0.090162 | 0.186866 | 0.617335 |
| seqkd | 160 | 2.682571 | 21.087648 | 0.087163 | 0.184415 | 0.620767 |
| seqkd | 320 | 2.752214 | 21.958223 | 0.082299 | 0.177917 | 0.618449 |
| seqkd | 480 | 2.762230 | 22.236449 | 0.082134 | 0.178574 | 0.628192 |
| seqkd | 624 | 2.810407 | 23.073088 | 0.078669 | 0.172833 | 0.624310 |

## C14 and C15 mechanical readings

C14 C7 geometry events use `E_ood`, `track=per_checkpoint`, L18, `epsilon=0.05`, and seven modules.

| arm | event | defined | step | r_epsilon_delta_mean |
|---|---:|---:|---:|---:|
| opd | first_positive_checkpoint | False |  |  |
| opd | positive_peak | False |  |  |
| opd | signed_minimum | True | 624 | -24.285714 |
| sft | first_positive_checkpoint | True | 5 | 0.142857 |
| sft | positive_peak | True | 40 | 2.714286 |
| sft | signed_minimum | True | 480 | -7.142857 |
| offkd | first_positive_checkpoint | True | 5 | 1.714286 |
| offkd | positive_peak | True | 20 | 3.142857 |
| offkd | signed_minimum | True | 624 | -18.571429 |
| seqkd | first_positive_checkpoint | True | 5 | 1.428571 |
| seqkd | positive_peak | True | 20 | 2.714286 |
| seqkd | signed_minimum | True | 624 | -22.428571 |

C15 corrected pilot and explicitly unpaired aggregate rows:

| run | cap | n | accuracy | cap-hit rate | paired |
|---|---:|---:|---:|---:|---:|
| formal_step80 | 4096 | 500 | 0.620000 | 0.792000 | False |
| large_cap_pilot_step80_corrected | 24576 | 60 | 0.616667 | 0.633333 | False |

For the corrected pilot, `cap_hit_n=38`, `accuracy_given_cap_hit=0.39473684210526316`, `stopped_n=22`, and `accuracy_given_stopped=1.0`. Cap hit is defined as `finish == length OR response_tokens >= generation_cap`.

## Validated artifact manifests

| manifest | bytes | sha256 |
|---|---:|---:|
| `C2_raw_er_manifest.json` | 1328 | `4fc789aaf2d4cbe047ed11e55163c02aa13f97ec5af49d8b207a4805d00e8d79` |
| `C3_opd_overcompression_rebound_manifest.json` | 841 | `9346f8ba1f478fc453d07530be22210b26555f4083a323293f402015dcf2a798` |
| `C5_eif_geometry_manifest.json` | 1818 | `482446854b35018d64e8d3c7db645634a63e636519cdb11df5171c178c85ac73` |
| `C8_training_corpus_ppl_manifest.json` | 1730 | `ff769869ee388f887118c49860b773c7c486c9c40ff01212884b67557aa42976` |
| `C11_mmlupro_answer_token_entropy_manifest.json` | 2299 | `165131457dbc281f23e6788a52759d6cd685afeec9622a1b73f96f1c35293f5c` |
| `C14_main_track_backfill_manifest.json` | 2195 | `c396c5117ea61e9bab405ea51d7a58739ed0588760085f72abe3df845dc998c8` |
| `C15_cap_pilot_repair_manifest.json` | 1762 | `18ce2cd5a8cb39ddee901f9726be9e1991b488fe03e4939ab0a1bc89105dd14e` |

All artifacts are under `mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/`. The supervisor generated final CSVs only after both canonical lanes completed, then verified every manifest output path, byte count, and SHA-256.

Real-base selective-logit smoke: C8 NLL absolute difference `5.58793544769e-08`; C11 entropy absolute difference `3.69548797607e-06`; status `complete`.

After GPU1's canonical off-KD/seqKD lane completed early, a detached helper processed remaining SFT C8/C11 cells. Per-cell locks preserved single-writer semantics; the canonical finalizer remained the only final CSV/manifest writer.

## Post-Stage-3 frozen theory note T-ONPOLICY: evidence hierarchy and deferred freshness control

Status: `accepted theory decision`. Preserve this ordering when the pending tasks are consolidated.

### Main on-policy story, prioritized by no-retraining evidence

1. OPD and ordinary offline post-training exhibit different checkpoint-wise $r_\varepsilon$ transients.
2. OPD–off-KD, under the same dense KL objective, localizes the observed difference to the sequence-support side at the level of the current design. This is the primary matched on/off-policy contrast.
3. The $\alpha=.5$ arm tests whether the trajectory moves in an ordered fashion with partial on-policy exposure; use only the checkpoints actually observed and do not infer a full continuous dose-response curve from three exposure levels.
4. The general-adjusted DiD in T-DID tests whether the on-policy effect is domain-selective rather than a uniform global shift. State this as evidence only after the residual trajectories have been computed and shown stable.
5. A direct OPD–off-KD functional-subspace comparison on existing checkpoints tests whether on-policy training changes not only $r_\varepsilon$ counts but also the retained/compressed directions. This requires no retraining and should precede new-arm training. Compare output-space projectors in their common coordinates; input-side comparisons must use a common fixed-whitening coordinate or an explicit transport, because independently per-checkpoint-whitened right singular vectors are not directly commensurate.
6. off-KD–seqKD remains a secondary matched-support boundary result: target/loss/readout can change final behavior even when functional-rank paths are similar. It is not the main soft-versus-hard-label story.

This hierarchy is a target narrative, not permission to convert pending analyses into confirmed findings. Items 3–5 retain their actual evidence status until their frozen analyses pass.

### Two meanings of “frozen base 0”

- **Geometric base:** subtracting $r_{\varepsilon,D,a,0}$ from every arm/checkpoint. This is already required for $A_{D,a,t}$ and makes changes comparable, but it is only a reference point; OPD-versus-step0 by itself does not isolate on-policy training.
- **Frozen step0-self support arm (`frozenSelf0-KD`):** initialize from the same step0 student, generate the OPD prompt rollouts once with that student, freeze those sequences/prefixes, and then train the evolving student on them with the same fixed teacher dense-KL targets, optimizer, schedule, and prompt pool as OPD. The inputs are fixed, but training remains nontrivial because gradients compare the current student distribution with the teacher distribution on those fixed prefixes.

The new arm enables the clean hierarchy:

- OPD versus `frozenSelf0-KD`: total effect of dynamically refreshed current-student support (“online freshness”), including the downstream length/EOS/style changes caused by refreshing;
- `frozenSelf0-KD` versus teacher-generated off-KD: generator/style difference under fixed off-policy support;
- OPD versus existing off-KD: the current broad matched-objective on/off-policy contrast.

### Total effect first; sequence-property mechanisms second

Freeze the intended causal ordering as

$$
\text{freshness}
\longrightarrow
\{\text{length, EOS, repetition, token frequency/entropy, style}\}
\longrightarrow
r_\varepsilon\text{ trajectory},
$$

with a possible additional direct freshness-to-trajectory path. Length, EOS placement, repetition, and related generated-text properties are post-treatment mediators: current-self support may change them, and those changes may be precisely how freshness affects activation covariance, gradient exposure, and the resulting functional geometry. They are not baseline confounders to be matched away in the primary experiment.

The Stage-1 estimand is therefore the natural total effect at every checkpoint,

$$
\mathrm{TE}_{D,t}
=A_{D,\mathrm{currentSelf},t}
-A_{D,\mathrm{frozenSelf0},t},
$$

holding fixed pre-treatment/design variables—initial student, prompt pool/order, teacher, top-32 KL definition, optimizer, LoRA configuration, number of updates, and nominal prompt batch size—while allowing the generated sequence distribution to change naturally. Do **not** length-match, EOS-match, repetition-match, truncate to a common realized response distribution, or jointly reweight these mediators in the headline comparison. Report realized response-token exposure and compute cost transparently; a token-budget-matched result, if added, is a secondary controlled sensitivity analysis rather than the total effect.

Stage 2 asks how the total effect is carried. Save per-checkpoint/refresh-block sequence statistics, align their changes with $A_D/G_D$ trajectories, and then examine one mediator family at a time through predeclared stratification, reweighting, or direct controlled interventions. Ordinary regression that conditions on all observed mediators is descriptive: post-treatment conditioning can block the real mechanism or introduce collider bias, so it cannot by itself identify a causal mediation fraction. Paper language should distinguish “total freshness effect,” “mechanism-consistent association,” and “controlled sensitivity”; it must not call the residual after mediator matching the complete on-policy effect.

`frozenSelf0-KD` is a strengthening experiment, not a prerequisite for the present paper story. Record it as a low-priority retraining task after all analyses that reuse existing checkpoints (T-DID, strict T-PK, direct functional-subspace comparison, and T-BEH). It must not delay the current submission unless the no-retraining evidence fails materially.

## Post-Stage-3 frozen analysis note T-DID: checkpoint-wise absolute and general-adjusted geometry

Status: `accepted theory decision`, recorded now and to be integrated into the consolidated task organization after discussion.

For every domain $D$, arm $a$, and available checkpoint $t$, retain the primary within-domain change

$$
A_{D,a,t}=r_{\varepsilon,D,a,t}-r_{\varepsilon,D,a,0}.
$$

This remains the headline quantity because it directly states whether the model's domain-conditioned functional dimensionality expanded or compressed relative to its own base value. In parallel, compute the general-adjusted difference-in-differences

$$
G_{D,a,t}
=A_{D,a,t}-A_{\mathrm{general},a,t}
=\bigl(r_{D,a,t}-r_{D,a,0}\bigr)
-\bigl(r_{\mathrm{general},a,t}-r_{\mathrm{general},a,0}\bigr).
$$

$G$ is secondary and answers a different question: whether domain $D$ gained or lost functional dimensionality **relative to the general-domain change**. It is a domain-reallocation view, not a replacement for the absolute $A$ trajectory and not itself a direct capability score.

Both quantities must be reported checkpoint by checkpoint, not only at the terminal checkpoint. Use every common Qwen checkpoint `{0,5,10,20,40,80,160,320,480,624}` and all available landmarks for $\alpha=.5$/Llama without interpolation. Curves are primary; endpoint, signed minimum, positive maximum, rebound, and step-aware trajectory area are summaries only.

The main on-policy contrast is the same-objective OPD–off-KD trajectory at every common checkpoint:

$$
C^{\mathrm{on}}_{D,t}=A_{D,\mathrm{OPD},t}-A_{D,\mathrm{offKD},t},
\qquad
C^{\mathrm{on,rel}}_{D,t}=G_{D,\mathrm{OPD},t}-G_{D,\mathrm{offKD},t}.
$$

Use $C^{\mathrm{on}}$ to ask whether on-policy training changes the domain-conditioned path and $C^{\mathrm{on,rel}}$ to ask whether it changes domain allocation relative to general. The observed terminal clustering that appears related to CE versus KL is a hypothesis to test, not a conclusion: off-KD–seqKD is the matched-support objective contrast, while the incomplete factorial design prevents a global loss-type attribution. In particular, a weak terminal on-policy contrast does not erase a systematic transient contrast.

All calculations must use the same $r_\varepsilon$ configuration within a comparison (per-checkpoint whitening, L18 headline, $\varepsilon=.05$, seven-module equal mean). The consolidated task should also add $A$ as the primary and $G$ as the predeclared secondary geometry predictor in T-BEH; they must be reported separately rather than selected post hoc by whichever correlates better with Eval.

## Post-Stage-3 mandatory follow-up T-INC: fair incremental-information experiment

Status: `todo`, high priority, no retraining. This is a main-text requirement and depends on strict T-PK being available. The question is not whether $r_\varepsilon$ works in isolation, but whether it supplies information not already available in the strongest weight-space and unwhitened baselines.

### Frozen fairness rules

- Use one row per genuine arm×checkpoint observation. Run domain-specific analyses separately and macro-average domains with equal weight; never duplicate the domain-independent $p_k$ across probe/module/domain rows and count those copies as independent evidence.
- Use exactly the same observed checkpoint intersection, outer folds, train-fold-only standardization, regularization family, hyperparameter rule, and missing-data exclusions for every nested comparison. Exclude the shared step0 from arm classification because identical observations cannot identify an arm label; retain it only as the trajectory reference.
- Keep feature budgets explicit. The primary comparison adds one scalar domain-matched $r_\varepsilon$ feature to one scalar strict $p_k$ feature. Multimetric baselines must use fixed regularization and be compared with an otherwise identical model after adding $r_\varepsilon$; do not give one family more tuning or hidden feature selection.
- Preserve checkpoint/arm dependence in folds and uncertainty. Modules, probes, domains, and checkpoints are not training seeds. Report all frozen domains/outcomes, including null and adverse increments.

### Track A: trajectory/arm discriminability

Primary binary task: distinguish OPD from off-KD at common nonzero checkpoints, because this is the matched-KL on/off-policy contrast. Use paired leave-one-checkpoint-out validation and report out-of-fold log loss, ROC-AUC, balanced accuracy, calibration, and paired uncertainty.

Fit the nested sequence:

1. $M_P$: strict source-principal $p_k$ only;
2. $M_{P+r}$: the identical model with domain-matched $A_{D,a,t}=\Delta r_{\varepsilon,D,a,t}$ added;
3. secondary repetitions replacing or augmenting $p_k$ with the predeclared unwhitened baselines raw ER and PR;
4. construct ablations, where available, using the same $r_\varepsilon$ estimator with weight-only/unwhitened, fixed-whitening $S_{D,0}$, and per-checkpoint whitening $S_{D,t}$ inputs.

The headline is the paired out-of-fold increment from $M_P$ to $M_{P+r}$, not a comparison of raw coefficient sizes. Report $\Delta$log-loss, $\Delta$AUC, and the fold-level prediction changes. The four/five-arm task and ordinal $\alpha\in\{0,.5,1\}$ exposure task are secondary because their small common-checkpoint sample sizes limit inference.

### Track B: incremental behavior explanation

Extend T-BEH with a nested comparison rather than only one-metric-at-a-time replacement:

$$
M_N=\text{arm/progress nuisance model},\qquad
M_{N+P}=M_N+p_k,\qquad
M_{N+P+r}=M_N+p_k+A_D.
$$

Report the out-of-fold increment $M_{N+P+r}-M_{N+P}$ for each predeclared capability outcome and its equal-domain macro average. Repeat with the general-adjusted $G_D$ only as the frozen secondary predictor and with raw ER/PR as secondary comparator baselines. Do not select between $A_D$ and $G_D$ after seeing which correlates better. This track inherits T-BEH's domain pairing, checkpoint blocking, arm/objective identifiability rule, and null-result policy.

The estimand is **incremental explanation, not perfect behavioral prediction**. Report absolute out-of-fold fit for context, but judge the metric by the residual reduction conditional on nuisance variables and $p_k$:

$$
R^2_{\mathrm{partial},r}
=\frac{\mathrm{SSE}(M_{N+P})-\mathrm{SSE}(M_{N+P+r})}
{\mathrm{SSE}(M_{N+P})},
$$

together with out-of-fold $\Delta R^2$, $\Delta\mathrm{MAE}$, and conditional log-loss where the outcome is modeled probabilistically. A modest total $R^2$ does not refute the claim if the conditional increment is reproducibly positive across folds and is not driven by one arm/domain. Conversely, high in-sample correlation or strong standalone prediction does not establish added information if $M_{N+P+r}$ fails to improve over $M_{N+P}$. Paper wording must remain “adds partial/conditional information about behavior,” never “determines,” “fully explains,” or “perfectly predicts” behavior.

### Interpretation gate and required artifacts

- Positive Track A means $r_\varepsilon$ contains checkpoint-wise on/off-policy trajectory information beyond source-weight update location. Positive Track B means it additionally provides partial behavioral information beyond nuisance variables and $p_k$.
- Track A positive / Track B null still supports the paper's geometry and trajectory contribution, while delimiting end-to-end behavioral sufficiency. Both null means the claimed empirical advantage over $p_k$ must be removed or sharply narrowed.
- A positive increment from per-checkpoint whitening over both weight-only and fixed-whitening variants is the cleanest evidence that current domain activations—not merely another spectral transform of weights—supply the added information.

Required outputs: `TINC_frozen_splits.json`, `TINC_trackA_oof.csv`, `TINC_trackA_nested_comparison.csv`, `TINC_trackB_nested_comparison.csv`, `TINC_whitening_ablation.csv`, `TINC_report.md`, and `TINC_manifest.json`, all in a new immutable run directory with exact input/output hashes.

## Post-Stage-3 queued follow-up T-PK: strict source-principal comparison

Status: `todo`, recorded now and to be integrated into the final task organization after the current theory discussion. This is the primary related-work comparison, not an appendix-only raw-metric check.

- Reproduce the strict joint source-principal statistic
  $$
  p_k(W,\Delta W)=\frac{\|U_k^\top\Delta W V_k\|_F^2}{\|\Delta W\|_F^2},
  $$
  with $U_k,V_k$ taken from the source weight SVD. The existing left-only $\rho_U^2$ and top-32 approximation are not substitutes.
- Freeze source checkpoint, $\Delta W$ construction, merge precision, rank-fraction grid, layer/module aggregation, checkpoint landmarks, and random/null controls before execution. At minimum cover Qwen OPD/SFT/off-KD/seqKD; include $\alpha=.5$ and Llama landmarks when their matched inputs are ready.
- Compare $p_k$ with $r_\varepsilon$ on the same frozen arm/trajectory discrimination task using standardized or studentized effect sizes. Do not compare their raw numerical magnitudes: $p_k$ measures where the update lies in source-weight coordinates, whereas $r_\varepsilon$ measures domain-conditioned functional dimensionality after activation whitening.
- Preserve both possible outcomes. If strict $p_k$ separates the trajectories, report the complementary information supplied by activation conditioning; if it does not, quantify the incremental discriminability of $r_\varepsilon$. Do not prestate that conventional weight geometry fails.
- The later consolidated task specification must define scripts, immutable output paths, manifests, acceptance checks, and the exact linkage from T-PK outputs into the T-BEH comparator table.

## Post-Stage-3 follow-up T-BEH: incremental geometry-to-behavior explanation

Status: `todo`, exploratory, zero retraining, and non-blocking for the paper's main claim. A positive result may strengthen the paper; a null or unstable result must be retained as a measurement boundary and does not invalidate the trajectory/discriminability contribution.

### Question and frozen estimand

Test whether domain-matched $r_\varepsilon$ explains behavioral variation beyond arm identity and training progress, and whether its out-of-sample increment is larger than that of raw ER, participation ratio (PR), and strict source-principal $p_k$.

Fit each primary behavior domain separately on the common observed arm×checkpoint cells. Do not impute missing checkpoints. Let $a$ denote arm and $t$ normalized training progress:

$$
\begin{aligned}
M_0:\quad B_{D,a,t} &= \alpha_D + f_D(t) + \gamma_{D,a} + \eta_{D,a,t},\\
M_r:\quad B_{D,a,t} &= M_0 + \beta_D\,z_{\mathrm{train}}\!\left(\Delta r_{\varepsilon,D,a,t}\right).
\end{aligned}
$$

The metric is the frozen headline configuration: per-checkpoint whitening, L18, $\varepsilon=.05$, seven-module equal mean, expressed as change from the shared step-0 model. Standardization must be learned on each training fold only. Fit comparator models by replacing $\Delta r_\varepsilon$ with exactly one of raw ER, PR, or strict $p_k$; never compare raw coefficients or unstandardized scales.

Important identifiability rule: do **not** include both a complete set of arm fixed effects and an objective main effect. Objective is deterministic given the four-arm label and is therefore redundant. If support and objective are analyzed explicitly, do so only as separately reported matched contrasts: OPD versus off-KD for the common KL objective, and off-KD versus seqKD for common frozen-teacher sequence support. The incomplete factorial design does not identify a global support×objective decomposition.

### Frozen domain pairing and outcomes

Primary capability outcomes, fixed before looking at regressions:

| behavior $B_D$ | matched geometry domain | primary behavioral column |
|---|---|---|
| MATH500 | existing 32-item formal-MATH held-out probe, uniformly named $E_{\mathrm{mathHeld}}$ (historical artifact/task alias `E_math`; all MATH500 prompts and training overlap removed) | canonical flexible/extracted accuracy |
| MMLU-Pro | $E_{\mathrm{mmluPro}}$ | flexible/extracted accuracy |
| IFEval | $E_{\mathrm{ifeval}}$ | prompt-level strict accuracy |

The MATH pairing is domain-matched but not item-matched; it must not be described as a same-question probe. Secondary readout/error outcomes are MATH truncation/cap-hit and length, MMLU strict accuracy and extraction-failure rate, and IFEval instruction-level strict accuracy. They are reported as secondary outcomes and cannot replace a failed primary outcome after inspection.

Use the full Qwen four-arm checkpoint intersection where both geometry and the relevant Eval exist. The $\alpha=.5$ arm has only sparse behavioral landmarks and is therefore an optional held-out directional check, not an additional independent seed and not a source of pseudo-replication. Llama enters only after the same domain/metric pairing exists; until then it is not pooled with Qwen.

### Validation and reporting

1. Report in-sample fit only as a diagnostic. Headline quantities are out-of-fold $\Delta R^2=R^2(M_m)-R^2(M_0)$ and $\Delta\mathrm{MAE}=\mathrm{MAE}(M_m)-\mathrm{MAE}(M_0)$ under checkpoint-blocked or leave-one-checkpoint-out validation.
2. Preserve arm/checkpoint dependence in resampling. Do not treat domains, modules, probes, or checkpoints as independent training seeds. With one training seed, confidence intervals describe trajectory-cell stability, not training-run reproducibility.
3. In addition to each domain, report a predeclared equal-domain macro average. A claim of “partial behavioral explanation” requires positive out-of-fold $\Delta R^2$, negative $\Delta\mathrm{MAE}$, and a direction not caused by only one arm or one domain. Otherwise state that no stable incremental explanation was detected.
4. Plot observed versus out-of-fold predicted behavior and residuals by arm/checkpoint. Report matched OPD–off-KD and off-KD–seqKD contrasts separately so that a pooled fit cannot hide the mechanism boundary.
5. Record all metrics, including negative and null results. The comparison asks what activation-whitened functional rank adds over weight-space or unwhitened summaries; it is not a search over outcomes for the best correlation.

Required outputs under a new immutable run directory:

- `TBEH_model_comparison.csv`: domain, outcome, metric, fold scheme, $n$, baseline/augmented $R^2$, $\Delta R^2$, baseline/augmented MAE, $\Delta$MAE, and uncertainty summary.
- `TBEH_oof_predictions.csv`: arm, step, domain, outcome, observed value, baseline prediction, augmented prediction, fold id.
- `TBEH_matched_contrasts.csv`: OPD–off-KD and off-KD–seqKD geometry/behavior contrasts by common checkpoint.
- `TBEH_report.md`: protocol, exclusions, plots, positive/null interpretation, and explicit limitations.
- `TBEH_manifest.json`: input paths and hashes, exact column mappings, code revision, random seed, validation splits, and output hashes.

## Consolidated prioritized task queue — 2026-07-23 16:47 UTC

This section supersedes the ordering of the earlier queued notes, but not their frozen definitions. Times below are engineering estimates grounded in current logs. `2×96G` and `1×96G` mean elapsed wall time with that many RTX PRO 6000 96 GiB GPUs available; they are not total GPU-hours. The final column is **non-GPU engineering/analysis time**, not literal CPU occupancy: it includes coding, artifact audit, interpretation, plots, manifests, and document edits, but excludes waiting for user replies or an open-ended discussion loop. “Unsupported” means the present OPD runner allocates separate actor and teacher GPUs; converting it to one GPU would change the execution protocol and is not estimated as a simple 2× slowdown.

### Verified starting state

- Qwen OPD/SFT/off-KD/seqKD ten-checkpoint behavior and headline geometry: complete.
- Qwen $\alpha=.5$ through step160: complete. Stage-B 160→320 is currently running on both GPUs; observed progress was step229/320 at about 188 seconds/step.
- Llama OPD/SFT/off-KD/seqKD behavior and per-checkpoint-whitened geometry at `{0,5,20,40,80,160,320}`: complete. Existing `human_read-ch.md` text that says only Llama OPD early exists is stale and must be replaced after result adjudication.
- Llama SFT/off-KD/seqKD adapters exist through step624, but the accepted paper scope stops the four-arm comparison at step320. **Llama624 is removed from the task queue.**

### Priority table

| rank | task | status / why it matters | GPU? | 2×96G wall | 1×96G wall | non-GPU engineering/analysis |
|---:|---|---|:---:|---:|---:|---:|
| 1 | **Q1-320: finish and validate $\alpha=.5$ step320** | Already running; completes the exposure trajectory now that its cost is sunk. Then export, behavior, six-probe geometry, support statistics, manifests. Do not interrupt. | yes, 2 required for training | **5.5–7 h remaining** | current runner unsupported | 1–2 h finalization |
| 2 | **RESYNC: adjudicate new Llama four-arm + $\alpha=.5$ results and update human_read** | Highest immediate theory task. Replace stale Llama status, freeze which findings reproduce, normalize probe names, and rebuild current tables before interpreting new analyses. | no | 0 | 0 | **3–6 h** |
| 3 | **T-PK: strict joint source-principal $p_k$** | Primary related-work baseline and dependency of T-INC/T-BEH. Qwen four arms + $\alpha=.5$ landmarks; Llama four arms through320. Use joint $U_k^\top\Delta W V_k$, not left-only $\rho_U^2$. | yes, recommended | **2–4 h** | **4–7 h** | 2–4 h implementation/audit |
| 4 | **T-WHITE: construct ablation of activation conditioning** | Compare weight-only/unwhitened, fixed $S_{D,0}$, and per-checkpoint $S_{D,t}$ under the same $r_\varepsilon$ estimator. Qwen is headline; add Llama where affordable. This is required to show what activation whitening adds. | yes for missing tracks | **2–4 h** | **4–7 h** | 1–2 h aggregation |
| 5 | **T-DID: checkpoint-wise absolute $A_D$ and general-adjusted $G_D$** | Main on-policy contrast OPD–off-KD at every common checkpoint; endpoint alone is insufficient. Also test the tentative CE/KL endpoint pattern without promoting it to a factorial conclusion. | no | 0 | 0 | **1–3 h** |
| 6 | **T-INC + T-BEH: fair incremental information and behavior explanation** | Main-text gate. Track A asks whether $r_\varepsilon$ adds OPD/off-KD trajectory information beyond $p_k$; Track B asks whether it reduces behavioral residuals beyond arm/progress/$p_k$. Includes $A_D$ primary, $G_D$ secondary, raw ER/PR controls, frozen folds, and null reporting. | no after inputs exist | 0 | 0 | **5–10 h** |
| 7 | **T-SUB: direct OPD–off-KD functional-subspace comparison** | Tests whether on-policy changes compressed/retained directions, not only rank counts. Output-side projectors use common coordinates; input-side comparison must use fixed whitening/transport. Existing checkpoints suffice; no training. | yes | **3–6 h** | **6–12 h** | 2–3 h audit/plots |
| 8 | **T-SUPPORT: sequence-support and mediator statistics on existing rollouts** | Compute length, EOS, truncation, duplicate/near-duplicate, distinct-2/4, token share, sequence entropy, and source-specific KL/loss by checkpoint. This strengthens the candidate path without matching post-treatment mediators away. | no | 0 | 0 | **3–6 h** |
| 9 | **PAPER-MAIN: rewrite current report, figures, abstract decision, and main manuscript** | Start in parallel after rank2; finalize after ranks3–7. Keep $r_\varepsilon$ as the sole main method, on-policy trajectories as the main empirical story, and support–readout separation as boundary/secondary result. | no | 0 | 0 | **10–18 h** |
| 10 | **PROBE-CORE: exact MATH500-aligned $E_{\mathrm{math}}$ + $\alpha=.5$ $E_{\mathrm{aime24}}$ backfill** | Repairs the remaining strict behavior–probe pairings. Keep the existing 32-item non-overlap formal-MATH probe as $E_{\mathrm{mathHeld}}$, not as exact MATH500. Use landmark scope first. | yes | **2–4 h** | **4–8 h** | 1–2 h naming/audit |
| 11 | **STABILITY: epsilon synthesis and finite-sample table** | Summarize $\varepsilon\in\{.01,.025,.05,.10\}$; compile existing sample-count tests into one table. Recompute only if the historical artifacts cannot be recovered. | normally no; fallback yes | 0; fallback **1–2 h** | 0; fallback **2–4 h** | **2–5 h** |
| 12 | **MATHCOT-HOLD: $D_{\mathrm{mathCoThold}}$, Eval_mathCoThold, $E_{\mathrm{mathCoTtrain}}/E_{\mathrm{mathCoThold}}$** | Use **256 fixed held-out Eval questions** at `{0,20,40,160,end}`. Use 32 actual-training questions for $E_{\mathrm{mathCoTtrain}}$ and a separate 32-question held-out probe for $E_{\mathrm{mathCoThold}}$, disjoint from the 256 Eval questions. Expand Eval to500 only under a predeclared ambiguity rule. | yes | **6–10 h** | **12–20 h** | 2–4 h data/audit |
| 13 | **SUPPLEMENT: Numina and AIME25 completion** | Add seqKD Numina behavior, $E_{\mathrm{numina}}$, and AIME25 cross-year robustness. Appendix-level unless results unexpectedly alter the core story. | yes | **3–6 h** | **6–12 h** | 1–3 h |
| 14 | **LEAD-LAG: matched geometry–behavior timing** | MATH↔math probe, MMLU↔$E_{\mathrm{mmluPro}}$, IFEval↔$E_{\mathrm{ifeval}}$; descriptive timing only, no mediation claim. Run after strict pairings are frozen. | no | 0 | 0 | **1–3 h** |
| 15 | **FROZEN-SELF: new `frozenSelf0-KD` arm** | Best new-training control for the total effect of online freshness. One step0-student rollout pass + teacher top-32 labeling + offline KL training + landmark behavior/geometry. Keep natural length/EOS/repetition differences in the primary contrast. | yes | **12–18 h** | **18–28 h** | 3–5 h validation |
| 16 | **MEDIATOR-2: controlled mechanism analysis after frozen-self** | First measure mediation associations; then optionally reweight/intervene one family at a time. Do not condition on every post-treatment mediator or call the residual the total on-policy effect. | no for reweighting; yes for new interventions | 0; intervention TBD | 0; intervention TBD | **3–6 h** initially |

### Execution order and stopping rule

1. While rank1 occupies both GPUs, execute CPU ranks2, 5, and 8 where inputs already exist; prepare but do not launch GPU ranks3/4/7.
2. When $\alpha=.5$ step320 and its postprocessing finish, run T-PK, then T-WHITE, then T-SUB. T-DID can finalize as soon as step320 lands.
3. Run T-INC/T-BEH only after T-PK and the frozen whitening baselines exist. Then update the paper before committing to new training.
4. If T-PK/T-INC/T-WHITE establish clear incremental information and Llama through320 reproduces the transient story, frozen-self is a strengthening experiment rather than a submission blocker. If those core tests fail, reassess claims before spending GPU time on supplementary probes or new training.
5. Do not add independent seeds in the current budget. **Llama624 and clean lag-80 have been explicitly removed from the current task queue.**


### Intermediate Theory handoff protocol

The core results should return in small, audit-complete packages rather than waiting for all of
`M2` or `M3`. A package is eligible only when every listed cell has a raw artifact, a row-count
or checkpoint inventory, a manifest with hashes and command/config provenance, and an explicit
missing-cell list (which may be empty). Coder reports **raw tables and provenance only**;
Theory owns interpretation, adjudication, and the GO/NO-GO decision for the next optional block.

| gate | when it returns | contents frozen for Theory | does not wait for | next action while Theory reads |
|---|---|---|---|---|
| **H0 - active endpoint** | Q1-320 export, behavior, six-probe geometry, and support statistics all validate | Qwen $\alpha=.5$ `{0,5,20,40,80,160,320}` endpoint package; checkpoint inventory; behavior tables; per-probe geometry tables; support statistics; manifests | RESYNC conclusions, T-PK, T-WHITE, T-SUB | Start/continue M1 CPU reconciliation; prepare M2 inputs only. Do not launch a new training arm. |
| **H1 - reconciled baseline** | M1 naming/coverage audit and provisional DiD/support tables validate | One current cross-arm inventory for Qwen four arms + $\alpha=.5$ and Llama four arms through 320; canonical probe-name map; stale-document replacement inputs; raw $A_D/G_D$ and support tables | New GPU analyses; T-INC/T-BEH | Freeze the common checkpoint/probe universe and launch M2. H1 is the first package suitable for a provisional paper-results outline. |
| **H2 - metric validity** | T-PK and T-WHITE each complete their prespecified grids and checks | Strict joint $p_k$ tables; three whitening-condition tables under the same $r_\varepsilon$ estimator; factor/cache provenance; missing-track list | T-SUB, exact probe backfill, behavioral regression | Theory decides whether the headline metric survives the principal-energy and whitening challenges. Coder may continue with T-SUB and PROBE-CORE without waiting for prose review. |
| **H3 - geometry closure** | T-SUB and PROBE-CORE validate; all completed M2 pieces are merged without overwriting raw sources | Direct functional-subspace comparison; exact $E_{\mathrm{math}}$ and $E_{\mathrm{aime24}}$ backfill; consolidated M2 index and reproducible figure-input tables | T-INC/T-BEH and manuscript wording | Freeze the geometry input schema and begin M3 frozen-fold analyses. This is the main geometry package for a core-results review. |
| **H4 - claim-ready core** | T-INC, T-BEH, final DiD, stability synthesis, and lead-lag complete under frozen schemas/folds | Incremental-information and behavior-residual outputs including nulls; final DiD/stability/lead-lag tables; analysis manifest; explicit scope and exclusions | Frozen-self, MathCoT/Numina/AIME25 supplements | Theory makes the M4 wording decision: freeze the main-paper claim set, revise it, or authorize M5 frozen-self. |
| **H5 - frozen-self total-effect control** | M5 `frozenSelf0-KD` training, landmark behavior, and geometry validate | Step-0-self rollout/teacher-label provenance; training inventory; landmark behavior/geometry and total-effect readouts; direct comparability audit against OPD/off-KD | M6 supplements and M7 analysis | Theory decides whether the new control strengthens, narrows, or changes the core mechanism account. |
| **H6 - mediator follow-through** | M7 association/reweighting work completes, plus any separately authorized intervention | Mediator association and reweighting tables; intervention provenance if applicable; explicit distinction between descriptive association and controlled intervention | Nothing | Final mechanism appendix/claim-boundary review. This gate exists only if H5 is authorized and completes. |

#### Required layout for every handoff package

Each gate writes one immutable markdown handoff plus one JSON manifest under
`local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/`:

```text
mini_stage3_H{0..6}_theory_handoff.md
stage3_H{0..6}_handoff_manifest.json
```

The markdown handoff contains: (1) status and exact completed grid, (2) raw readout tables or
links to their immutable CSVs, (3) coverage/missing-cell table, and (4) provenance. The manifest
records source paths, byte counts, SHA-256 hashes, commands/config versions, model/checkpoint IDs,
probe and sampling seeds, and timestamps. Existing source CSVs are append-only or versioned;
the package must never silently replace a prior raw artifact. `code/code_evolution.md` receives
one short entry per gate pointing to both files.

#### Delivery discipline

- Return H0 immediately after Q1 validation, even if M1 is still being reconciled.
- Return H1 independently as soon as its audit is complete; it may arrive before or after H0.
- Return H2 as soon as T-PK and T-WHITE are both complete. Do not hold it for T-SUB.
- Return H3 after the remaining M2 geometry/probe work completes.
- Return H4 before spending budget on M5 or M6. M5 and M7 return as H5 and H6 respectively; M6
  remains a separate appendix package. Neither is retroactively folded into the core H0-H4 record.
- Launch H5 only after Theory authorizes M5 at the H4 decision gate. Launch H6 only after H5 and
  Theory approval of the M7 scope.

### Aggregate planning ranges

- **Submission-critical merged remainder excluding new training:** current $\alpha=.5$ completion plus the merged core-geometry bundle is approximately **12–17 additional 2×96G wall-hours**, with non-GPU work overlapped.
- **Add frozen-self end to end:** approximately **+12–18 two-GPU wall-hours**.

### Merged execution plan

Merging is preferred because model loading, SVD/whitening factors, checkpoint inventories, frozen folds, and report rendering can be shared.

| order | merged bundle | included tasks | GPU? | 2×96G wall | 1×96G wall | literal CPU compute | analyst/engineering |
|---:|---|---|:---:|---:|---:|---:|---:|
| M0 | **ACTIVE-320** | finish $\alpha=.5$ 160→320; export; behavior; geometry; support manifest | yes, 2 required for training | **5.5–7 h remaining** | unsupported | <0.5 h | 1–2 h |
| M1 | **CORE-RESYNC** | Llama/$\alpha$ result adjudication + naming audit + provisional $A_D/G_D$ + existing-rollout support stats + stability-artifact inventory | no | 0 | 0 | **0.5–1.5 h** | **3–5 h** |
| M2 | **CORE-GEOMETRY** | strict T-PK + weight/fixed/current whitening ablation + OPD–offKD subspace + exact $E_{\mathrm{math}}$/$\alpha$ $E_{\mathrm{aime24}}$ backfill | yes | **6–10 h** | **12–20 h** | 0.5–1 h | 3–5 h |
| M3 | **CORE-INCREMENT** | frozen folds + T-INC Track A + T-BEH Track B + final DiD + epsilon/sample-count summary + matched lead–lag | no | 0 | 0 | **0.3–1 h** | **3–6 h** |
| M4 | **PAPER-SYNTHESIS** | human_read rewrite, current tables, main figures, abstract wording gate, paper/appendix prose | no | 0 | 0 | <0.5 h | **8–14 h** |
| M5 | **FROZEN-SELF** | one-time step0-self rollout, teacher top-32, offline KL, landmarks, total-effect analysis | yes | **12–18 h** | **18–28 h** | 1–2 h | 3–5 h |
| M6 | **SUPPLEMENT-BATCH** | Math-CoT held-out 256 + two 32-item probes; Numina; AIME25 | yes | **7–12 h** | **14–24 h** | 0.5–1 h | 3–5 h |
| M7 | **MEDIATOR-2** | post-frozen-self association/reweighting; any new intervention requires a separate decision | initially no | 0 | 0 | **0.5–2 h** | **3–6 h** |

Recommended order is `M0 || M1`, then `M2`, then `M3`, then freeze the main paper in `M4`. Only after that decision gate run `M5`; `M6` is appendix closure and can run before or after M5 according to remaining GPU budget. M7 requires M5. User discussion/review time is not included in these estimates.

### Maximum-parallel schedule and critical path

With exactly two 96 GiB GPUs and one analyst/agent, use the following resource schedule:

| elapsed phase | GPU0 | GPU1 | CPU/analyst lane |
|---|---|---|---|
| P0, now → about 5.5–6.5 h | M0 $\alpha=.5$ training/postprocess | M0 teacher/rollout/postprocess | M1 result inventory, provisional DiD/support statistics, M2 preparation, paper skeleton |
| P1, next 6–10 h | M2 Qwen $p_k$/whitening/probe/subspace lane | M2 Llama $p_k$/whitening/subspace lane | finish M1; freeze M3 folds and schemas; continue method/related-work writing |
| P2, next 18–28 h | M5 frozen-self end to end on one GPU | M6 supplement batch on one GPU (14–24 h) | M3 incremental analyses, then M4 manuscript synthesis |
| P3, after M5, 3–6 h | idle or artifact repair | finish M6 if needed | M7 mediator association/reweighting and final integration |

The all-task critical path is approximately

$$
T_{\mathrm{all}}
=T_{M0}+T_{M2}
+\max\!\left(T_{M5}^{1\mathrm{GPU}}+T_{M7},\ T_{M6}^{1\mathrm{GPU}},\ T_{M3+M4}\right),
$$

which gives an optimistic arithmetic lower bound of about **32–33 hours**, an aggressive but credible target of **40–48 hours**, and an upper end of roughly **50–51 hours** within the current ranges. The recommended commitment is therefore **two wall-clock days** for every retained task, assuming no failed job or artifact repair. Submission-critical results through M3 need about **12–17 hours of two-GPU availability**; a polished M4 manuscript package is more realistically **22–32 hours from now** because interpretation and writing remain serial work.

This schedule excludes user review latency, Llama624, lag-80, new independent seeds, and any new training intervention added under M7. It assumes the two GPUs split after M2; running M5 and M6 sequentially on both GPUs is not faster than assigning one GPU to each.

<!-- cycle09-stage3-final-end -->
