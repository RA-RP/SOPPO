# FAT-R1-v2 equal-5 handoff

created_utc: 2026-07-28T15:42:04.358859+00:00
status: COMPLETE_EQUAL5_WITH_QWEN_PK_STEP10_EXCLUDED

## Scope

- Formal headline module aggregation is non-QK equal-5: v/o/gate/up/down.
- q_proj and k_proj are excluded. This is measurement-side module exclusion, not adapter ablation.
- CPU-only reuse: no training, forward, rollout, behavior Eval, or new SVD.
- FAT domains are domain-matched, not item-matched: MMLU-Pro -> E_mmluPro, MATH500 -> E_mathHeld.
- Qwen step10 p_k remains excluded from matched C-vs-p analyses; no interpolation or adapter-BA substitute.

## Coverage

| model   | domain   |   expected_c5_rows |   observed_c5_rows |   expected_pk5_rows |   observed_pk5_rows |   missing_pk5_cells | missing_pk5_reason                       |   module_count | included_modules                                                          | excluded_modules                  | domain_match_status             |
|:--------|:---------|-------------------:|-------------------:|--------------------:|--------------------:|--------------------:|:-----------------------------------------|---------------:|:--------------------------------------------------------------------------|:----------------------------------|:--------------------------------|
| llama   | math     |                 24 |                 24 |                  24 |                  24 |                   0 |                                          |              5 | self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj | self_attn.q_proj,self_attn.k_proj | domain_matched_not_item_matched |
| llama   | mmlu     |                 24 |                 24 |                  24 |                  24 |                   0 |                                          |              5 | self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj | self_attn.q_proj,self_attn.k_proj | domain_matched_not_item_matched |
| qwen    | math     |                 36 |                 36 |                  32 |                  32 |                   4 | QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION |              5 | self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj | self_attn.q_proj,self_attn.k_proj | domain_matched_not_item_matched |
| qwen    | mmlu     |                 36 |                 36 |                  32 |                  32 |                   4 | QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION |              5 | self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj | self_attn.q_proj,self_attn.k_proj | domain_matched_not_item_matched |
| ALL     | ALL      |                120 |                120 |                 112 |                 112 |                   8 | QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION |              5 | self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj | self_attn.q_proj,self_attn.k_proj | domain_matched_not_item_matched |

## Outputs

- `equal5_task_status.csv`: rows=11 sha256=bb908a33fab8043d4c4fa71fae911db9f0120e479bbaab8e69375d79f7b12e0b
- `equal5_feature_matrix.csv`: rows=120 sha256=38404dec2f61ce157be84650e9b7ad005b92059798970b8c2a622d1c260b776e
- `equal5_coverage_audit.csv`: rows=5 sha256=283ce4831256b487547020209dd4dc7587ee5e5fec074eb47a4f39161a7a4421
- `equal5_standalone_correlations.csv`: rows=960 sha256=9570e3ae947aa7991593895b1c16ffdbd00ef1b4e93989824f123aff43385d8a
- `equal5_checkpoint_demeaned_correlations.csv`: rows=240 sha256=8cc64411bb838a6efeb2759c159dd0d875bbd4eb63417224b7a2772d1c74006b
- `equal5_progress_residual_diagnostic.csv`: rows=960 sha256=76386e652f7bf3228795955891d65002829ae7c3e0a4ecb258abd32451db99d7
- `equal5_epsilon_sensitivity.csv`: rows=768 sha256=904f2ecf1378d45fc2a871ae9a51117714adf268281c741a7c15cf1d618e5d65
- `equal5_standalone_grouped_models.csv`: rows=336 sha256=ba99433b7c43d01cc09e3a6a5de132638f3d6755b892ccab772fa8a1a632bb2a
- `equal5_incremental_grouped_models.csv`: rows=192 sha256=8a14923df803621de50a0a3ec28e6492c95d2f08393d48ff880939892d1f08bd
- `equal5_grouped_predictions.parquet`: rows=14880 sha256=0e8599b097faa6ad473c234213c52062755757d436f4191b6a93db11e94594e8
- `equal5_foldwise_results.csv`: rows=3864 sha256=9191e4fadf02b393ec10e0e60c8aeb78ad68c0694bd7025df923a9aab1c6e61e
- `equal5_related_work_metrics.csv`: rows=61 sha256=acad85d8c0ad1bcccaadc691ba64af9818d486e2c97ff999d5624c5c772b7a58
- `equal5_vs_equal7_paired.csv`: rows=1306 sha256=1b35759771374dade0bead03edc2fe7bde4cdefaee5f15273814461f985f9309
- `equal5_behavior_join.csv`: rows=60 sha256=313a88a0aa4c7a2eb48d0b1758064233df84fc888019d1337d1a386c711e4cb6
- `equal5_format_realization_table.csv`: rows=8 sha256=4e64178c3c1390671b6f9ac5b1354a81233ab8f12cecfce76506adf4809911e6
- `fat_r1_v2_documentation_correction.json`: rows=1 sha256=8f437bd84fa5e4038edbb2c6a45fa52f3f98556ab62f88af1c55eaf5aece7fa6
- `fat_r1_v2_documentation_correction.md`: rows=1 sha256=938c2321daa6e1f896a15c0ed5861441f1c87bfcd3a78043fd812b2d301c1c2e
- `figure1_equal5_trajectory_llama.png`: rows=1 sha256=3e5193deb1592d898dfee39ddd28b44700e99558d76a05818feacc49f26cd6a9
- `figure1_equal5_trajectory_qwen.png`: rows=1 sha256=39c55753fb61bfa2574ac292c92f67d0d6097b43fd24bc188b9d2dd380d1f472
- `figure2_equal5_grouped_models.png`: rows=1 sha256=3e3266ccbdd53ace42ed3e59d2648a3ebd1cf1ccb58e7cbd70660854fefc8b11
- `figure2b_equal5_incremental_models.png`: rows=1 sha256=62920737b85c96d17d60063e60b47797d004a505cb5db0cdf166d0a5af3ab776
- `figure3_format_gap.png`: rows=1 sha256=75b243916cdea3e1d795618d7b98263323cb10b79041499ab39afde34e7eef72
- `figure4_equal5_vs_equal7.png`: rows=1 sha256=7357d050865f7974479a3334c15af870c1fbc616707458c22e044290cd2c026e

## Grouped Model Snapshot

| model   | domain   | target      | feature_block   |      OOF_R2 |   OOF_MAE |   OOF_prediction_spearman |   n_checkpoint_groups |   n_state_rows | analysis_role                     | outer_split                    | standardization   | alpha_selection                |
|:--------|:---------|:------------|:----------------|------------:|----------:|--------------------------:|----------------------:|---------------:|:----------------------------------|:-------------------------------|:------------------|:-------------------------------|
| llama   | math     | delta_nll_p | C-only          |  0.785077   | 0.0343083 |                  0.76087  |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | C-only-matched  |  0.785077   | 0.0343083 |                  0.76087  |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | Pk4-only        |  0.245101   | 0.0616867 |                  0.581739 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | Pk8-only        |  0.414914   | 0.0541844 |                  0.693913 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | Pk16-only       |  0.571196   | 0.048874  |                  0.650435 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | Pk32-only       |  0.610241   | 0.0483331 |                  0.738261 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | PkAll-only      |  0.407122   | 0.0558999 |                  0.724348 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | M0              |  0.711684   | 0.0356693 |                  0.74     |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | M0+C5           |  0.818025   | 0.0276357 |                  0.704348 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | M0+PkAll5       |  0.724571   | 0.0336437 |                  0.732174 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_p | M0+PkAll5+C5    |  0.837679   | 0.0260315 |                  0.676522 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | C-only          |  0.80764    | 0.0419779 |                  0.833913 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | C-only-matched  |  0.80764    | 0.0419779 |                  0.833913 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | Pk4-only        |  0.190246   | 0.0826636 |                  0.632174 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | Pk8-only        |  0.417382   | 0.0655877 |                  0.695652 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | Pk16-only       |  0.654947   | 0.0534849 |                  0.785217 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | Pk32-only       |  0.776214   | 0.0446531 |                  0.86087  |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | PkAll-only      |  0.624172   | 0.0546601 |                  0.848696 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | M0              |  0.71142    | 0.0430092 |                  0.752174 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | M0+C5           |  0.839299   | 0.0330746 |                  0.792174 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | M0+PkAll5       |  0.66574    | 0.0463944 |                  0.776522 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_c | M0+PkAll5+C5    |  0.830733   | 0.0354808 |                  0.786957 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | C-only          | -0.0658344  | 0.139221  |                 -0.265217 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | C-only-matched  | -0.0658344  | 0.139221  |                 -0.265217 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | Pk4-only        | -0.163516   | 0.146507  |                 -0.125217 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | Pk8-only        | -0.0443463  | 0.135661  |                 -0.184348 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | Pk16-only       | -0.0839395  | 0.136088  |                 -0.290435 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | Pk32-only       | -0.0716157  | 0.13514   |                 -0.284348 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | PkAll-only      | -0.00265959 | 0.135032  |                  0.106087 |                     6 |             24 | standalone                        | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |
| llama   | math     | delta_nll_b | M0              |  0.531838   | 0.0826976 |                  0.691304 |                     6 |             24 | nuisance_or_incremental_secondary | leave_one_checkpoint_group_out | train_fold_only   | inner_checkpoint_cv_train_only |

## Branch Codes

- formal_usable: equal5 feature matrix, standalone correlations, checkpoint-demeaned correlations, epsilon sensitivity, grouped held-out models, canonical behavior join, related-work pk5 rebuild, equal5/equal7 sensitivity.
- auxiliary: progress residual diagnostic only; equal-7 sensitivity.
- blocked: optional Qwen step10 pk backfill deferred; main matched pk analysis remains 112 states.
- superseded: previous equal-7 headline interpretation is sensitivity only for this FAT related-work link.
