# Result Incremental Log

```yaml
view_type: incremental_log
status: active
maintenance_rule: append_only_per_cycle
last_updated: 2026-07-07
```

This file records what each cycle added or changed relative to the previous cycle. It is append-only: past entries are not rewritten. For detailed numbers and evidence, see `local_experiment_results/<cycle_name>/`. For the current aggregate picture, see `current_picture.md`.

---

## Cycle 03: First OPD-Like Run (`cycle_03_trl_opd_like`)

**What was new:** First end-to-end TRL-based OPD-like vs SFT comparison on Qwen3-1.7B + NuminaMath. GSM8K used as the primary evaluation axis.

**What the results showed:** OPD-like training appeared more stable than SFT under matched GSM8K gain: OPD maintained or slightly improved GSM8K while larger SFT variants degraded. This was treated as a preliminary positive signal.

**What this cycle could not claim:** The signal was confounded by `theta0` saturation level. The matched SFT anchor was a relatively degraded large-SFT control, making the comparison easy to pass. No ID/OOD axis separation; no geometry; GSM8K was the only axis.

**Key limitations carried forward:** GSM8K used as both training-domain and eval-domain signal; high-noise small-sample evaluation; no untrained base reference.

---

## Cycle 04: Stricter Replication (`cycle_04_opd_stability_gain`)

**What was new:** Stricter matched-control design. `theta0` initialized from a less-saturated starting point (256 prompts) to leave room for SFT to gain. Full-sample GSM8K (N=1319). OOD-lite suite (MMLU/TruthfulQA/WinoGrande). Multi-layer GetSlice geometry. NuminaMath-test introduced as a separate ID axis. Untrained `Qwen3-1.7B` instruct checkpoint added as a reference row.

**What the results showed:**
- Gate A/B passed: experiment ran, both OPD arms found a valid GSM8K-gain match.
- Gate C/D failed: OPD did not beat matched SFT on OOD-lite penalty or spectral drift. The matched SFT collapsed to `sft_n128` (barely moved from `theta0`), making it trivially stable — OPD compared against an almost-stationary anchor.
- Appeared to show base-ceiling: untrained base ≈ theta0 on both NuminaMath ID and GSM8K, both higher than OPD arms on ID. Suggested no net learning gain from any training arm.
- Geometry: mixed signal — OPD lowered effective rank (more concentrated), SFT raised it; but OPD did not consistently reduce drift from `theta0`.
- MATH500 = 0.0 across all models: discovered to be a scoring artifact (`FINDING_04_eval_extraction_artifact.md`), not incapability.

**Two major findings documented:**
- `FINDING_04_eval_extraction_artifact.md`: MATH500 = 0.0 is a `$`-span extraction artifact; NuminaMath-test introduced as ID axis.
- `FINDING_04_base_ceiling.md`: untrained base ties theta0, suggests capability headroom is exhausted at this scale/data.

Detailed results: `local_experiment_results/cycle_04_opd_stability_gain/RESULTS_04.md`.

---

## Cycle 05: Eval Protocol Audit (`cycle_05_matched_control_id_ood`)

**What was new:** Pre-pivot capability audit revealed three compounding eval bugs in the Cycle 04 measurement stack. No new training; all findings are protocol corrections on existing Cycle 04 checkpoints. All 8 models re-evaluated under corrected protocols.

### Finding 05a: GSM8K Chat Template Missing
Every Cycle 04 GSM8K score used `lm_eval`'s default raw-completion prompt on an instruction-tuned chat model (no chat template, no `enable_thinking=False`). Correcting the protocol (chat template + `enable_thinking=False` + 3072 tokens + `\boxed{}`-aware extraction) raised every model's GSM8K score by ~+0.28-0.30 absolute. The original ~0.41 band was almost entirely a measurement floor. See `local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_gsm8k_chat_template_mismatch.md`.

### Finding 05b: MATH500 Dollar-Span Extraction Bug
`lm_eval`'s `hendrycks_math500` scorer extracts "the answer" by slicing between the first and last `$` in the entire generation — catastrophically wrong for any verbose CoT generation with intermediate LaTeX. Re-running with chat template and 3072 tokens still scores 0.0. The fix (dedicated script using `last_boxed` extraction + `math_verify` symbolic scoring) raises every model to ~0.54-0.73. See `local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_math500_dollar_span_extraction_bug.md`.

### Finding 05c: MMLU Collapse — Resolution via Base Model Mode
Applying `--apply_chat_template` to MMLU (a loglikelihood/MC task) collapsed all models to exactly 0.2295 (≈ 4-choice random chance). Four configurations were attempted; all failed. Resolution: adopt Qwen3 official evaluation protocol (arXiv 2505.09388) — base model mode (no chat template, 5-shot, standard loglikelihood). This yields scores in the 0.60-0.61 range for all 8 models, consistent with known Qwen3-1.7B capability. Community independent confirmation: lm-eval-harness issues #3405/#3576/#3322. `mmlu_protocol_version=2`. See `local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_mmlu_chat_template_collapse.md`.

**Full 8-model corrected results (2026-06-17):**

| Model | NuminaMath ID | GSM8K | ±SE | MATH500 | ±SE | MMLU | TruthfulQA MC1 | WinoGrande |
|---|---|---|---|---|---|---|---|---|
| base | 0.5516 | 0.7005 | 0.0126 | 0.7140 | 0.0202 | 0.6061 | 0.3341 | 0.6172 |
| theta0 | 0.5527 | 0.7180 | 0.0124 | 0.7260 | 0.0199 | 0.6019 | 0.3415 | 0.6235 |
| opd_lmbda05 | 0.5392 | 0.7180 | 0.0124 | 0.7080 | 0.0203 | 0.6004 | 0.3390 | 0.6251 |
| opd_lmbda1 | 0.5348 | 0.7346 | 0.0122 | 0.7140 | 0.0202 | 0.6061 | 0.3427 | 0.6235 |
| sft_n128 | 0.5370 | 0.7096 | 0.0125 | 0.7260 | 0.0199 | 0.6049 | 0.3439 | 0.6290 |
| sft_n256 | 0.5516 | 0.6839 | 0.0128 | 0.7080 | 0.0203 | 0.6033 | 0.3415 | 0.6235 |
| sft_n512 | 0.5179 | 0.6732 | 0.0129 | 0.7140 | 0.0202 | 0.6086 | 0.3378 | 0.6322 |
| sft_n1024 | 0.3890 | 0.6141 | 0.0134 | 0.5400 | 0.0223 | 0.6065 | 0.3317 | 0.6322 |

**Eval protocol (all 8 models):**
- NuminaMath-test ID: chat template, `enable_thinking=False`, open-answer subset
- GSM8K: chat template + `enable_thinking=False` + 3072 tokens + `\boxed{}`-aware extraction
- MATH500: chat template + `enable_thinking=False` + 3072 tokens + `math_verify` + `last_boxed` extraction
- MMLU: base model mode (no chat template, 5-shot, standard loglikelihood) — `mmlu_protocol_version=2`
- TruthfulQA-MC1: chat template, loglikelihood
- WinoGrande: no chat template, 5-shot, loglikelihood

**Cycle 05 Gates:**
- Gate A05: ✅ PASS — all 8 models re-evaluated under corrected protocol
- Gate B05: ✅ PASS — MMLU resolved via base model mode (Qwen3 official protocol)
- Gate C05: ✅ PASS — numbers self-consistent, no new scoring artifact

**Cycle 04 Gate re-analysis with corrected numbers:**

*Gate B04 (matched control anchor):* Matching basis is NuminaMath-test ID, not GSM8K. Under this basis, sft_n128 (0.5370) remains the nearest SFT anchor for opd_lmbda05 (0.5392, Δ=+0.0022) and opd_lmbda1 (0.5348, Δ=−0.0022). **Gate B04: CONFIRMED PASS.**

*Gate C04/D04 (OOD-lite stability):* OOD-lite now has valid numbers for all 8 models. Spread across all 8 models: MMLU ±0.0041 (range 0.0082), TruthfulQA ±0.0061 (range 0.0122), WinoGrande ±0.0075 (range 0.0150). OPD arms do not beat sft_n128 on any OOD-lite metric. **Gate C04: CONFIRMED FAIL.** However, the failure is a non-signal: all 8 models including base cluster within <1.5% on every metric; the axis has insufficient discriminative power in this setting. **Gate D04: CONFIRMED FAIL** (depends on Gate C04).

**Notable finding — sft_n1024 math degradation:** sft_n1024 shows severe math regression (NuminaMath ID −0.163, GSM8K −0.086, MATH500 −0.174 vs base) while OOD-lite remains stable. Train-eval format alignment verified (empty think block injected identically in training and eval). Pattern is consistent with math catastrophic forgetting at n=1024 SFT samples on Qwen3-1.7B. This data point is recorded as an outlier; it does not change the OPD vs SFT comparison (sft_n128/256/512 are the relevant SFT controls).

### ARC-Challenge OOD-lite (incremental, 2026-06-17)

ARC-challenge run completed after initial Cycle 05 result write-in. Protocol: no chat template, `enable_thinking=False`, 25-shot (lm-eval arc_challenge task default — confirmed by run log "Overwriting default num_fewshot of arc_challenge from None to 25"), `acc_norm`, N=1172 — same no-chat-template loglikelihood protocol as MMLU and WinoGrande. (Note: earlier drafts incorrectly stated "0-shot"; 25-shot is confirmed from the actual run log.)

| Model | ARC acc_norm | vs base |
|---|---|---|
| base | 0.4565 | — |
| theta0 | 0.4590 | +0.0026 |
| opd_lmbda05 | 0.4650 | +0.0086 |
| opd_lmbda1 | 0.4616 | +0.0051 |
| sft_n128 | 0.4599 | +0.0034 |
| sft_n256 | 0.4667 | +0.0103 |
| sft_n512 | 0.4710 | +0.0145 |
| sft_n1024 | 0.4667 | +0.0103 |

All SE = ±0.0146. Spread range = 0.0145. Confirms the OOD-lite non-discriminative pattern: all 8 models cluster within <1.5% — identical conclusion to MMLU (0.0082), TruthfulQA (0.0122), WinoGrande (0.0150). sft_n1024 shows no ARC regression (+0.0103 over base), consistent with OOD-lite capability preservation seen across all four OOD-lite tasks.

Detailed results: `local_experiment_results/cycle_05_matched_control_id_ood/run_01/RESULTS_05.md`.

**Cycle 05 status: CLOSED (2026-06-18).** All 8 models evaluated under corrected protocols on all 7 tasks. Eval stack unified under `Eval/run_eval.py`. See `research_cycles/result_interpretation_evolution.md` and `research_cycles/exp_design_evolution.md` for closed cycle delta entries.

---

## Cycle 06: SFT-Only Feasibility & Degradation (`cycle_06_sft_feasibility_and_degradation`)

**What was new:** First SFT-only feasibility + degradation sweep (no OPD arm). 8 SFT arms
trained (instruct ×4 + base ×4 over n∈{512,1024,2048,4096}) + 2 untrained references
(`instruct_base`, `base_base`) = 10 models, each on 7 tasks. New base-series generative-math
eval (4-shot CoT plain text, no chat template, aligned with Qwen3 official base protocol).
GetSlice geometry + principalEvidence also run (diagnostic-only, no gate).
Detail: `local_experiment_results/cycle_06_sft_feasibility_and_degradation/run_01/RESULTS_06.md`.

**Gate verdicts (pre-registered, this cycle):**

| Gate | Verdict | Basis |
|---|---|---|
| A06 instruct feasibility | **FAIL** (confirmatory) | instruct_base numina=0.533, threshold=base+1SE=0.549; sft_n512/1024/2048/4096 numina = 0.411/0.381/0.359/0.380, all below base. No-think concise SFT adds no ID gain and degrades math. |
| A06b base feasibility | **UNDETERMINED** | base-series numina is a truncation artifact (⚠️0.03–0.06, ~66% finish_reason=="length"), not a capability measurement. Deferred to Cycle 07 stop-fix re-collection. No base feasibility conclusion drawn this cycle. |
| B06 OOD-lite | **No degradation — preserved within noise** (confirmatory) | mmlu*/tqa/wino/arc all cluster across arms; same <1.5%-spread non-signal as Cycle 04/05. |
| C06 engineering | **PARTIAL** | All 10 models × 7 tasks evaluated + provenance complete; but geometry_metrics.csv / principal_evidence.csv were NOT copied back → OverlapLift not analyzable. Geometry pending → Cycle 07. |

**What the results showed (confirmatory):**
- Instruct no-think concise SFT does not improve NuminaMath-test at any n in {512..4096}; it
  degrades it (0.533 → 0.36–0.41) and also pulls MATH500 down (0.724 → 0.52–0.57). GSM8K mostly
  held (0.717 → 0.61–0.71). A06 FAIL.
- OOD-lite (MMLU 6-subtask*, TruthfulQA, WinoGrande, ARC) preserved within noise across all arms,
  both series. B06 no-degradation.

**What the results showed (indicative-only, NOT confirmatory):**
- *Base-series generative-math eval artifact (FINDING_06_base_gsm8k_stop, HIGH):* base
  `runner_base.py` has no stop sequence; a 4-shot base model never emits EOS and fabricates new
  `Problem:` continuations, the last-number scorer grabs a hallucinated digit → item marked wrong.
  This halved base GSM8K (0.368 vs official 0.7544) and **manufactured a fake monotonic "SFT +0.28"
  gradient**. With `stop=["\n\nProblem:"]`, base GSM8K is **flat ~0.66** and SFT is neutral —
  the gradient was differential de-truncation of a broken baseline. math500 (~8–10% trunc)
  unaffected. base numina (66% trunc) is artifact-dominated and not yet trustworthy. → Cycle 07
  must add stop seqs to all three `runner_base.py` and re-collect base gsm8k + numina.
- *No-think SFT removes the function of thinking (FINDING_06_nothink_sft, MED, n=100 single seed):*
  with `enable_thinking=True` on math500-first100, untrained `instruct_base` gains +0.13
  (0.76→0.89) but every no-think-SFT arm is flat-to-negative (−0.02 to −0.06) despite closing the
  `</think>` block 100% of the time — SFT preserves the *form* of a thinking block while destroying
  its *function*. Indicative; needs full math500 + multiple seeds before confirmatory.

**Allowed claim this cycle (conservative):** Only A06 FAIL (instruct no-think SFT shows no
measurable NuminaMath gain at n∈{512..4096} and degrades math) and B06 OOD-lite-preserved are
stated as confirmatory. The base stop-fix and the no-think-think-probe findings are recorded as
indicative-only (artifact-correction pending / n=100 single seed). No base feasibility claim; no
geometry/principal claim (pending copyback).

**Engineering findings carried to Cycle 07:** (1) add stop seq to base math runners; (2) MMLU full
too slow (~99 min/model) → 6-subtask subset used (`lm_eval --limit` doesn't propagate to group
tasks), absolute value ~+4.5 above official, intra-experiment only; (3) GetSlice profiling cache
hardcodes batch=1 — S/X batching must stay 1, parallelize via layer-chunk only
(FINDING_06_getslice_batch_size_constraint).

**Cycle 06 status: result analysis complete (2026-06-24); geometry/principalEvidence copyback +
base stop-fix re-collection deferred to Cycle 07.**

---

## Cycle 07: Base-Model SFT Trajectory (`cycle_07_base_sft_trajectory`)

**What was new (biggest change since Cycle 06):** Model and data both changed —
**Qwen3-4B-Base** (not 1.7B) + **Math-CoT-20k** (Rethink SFT data, think-format long-CoT,
n_train=5000), a **single SFT training run** profiled over a 10-checkpoint trajectory
(step_000…step_624, dense early grid), not the multi-arm n-sweep. New eval axes: AIME24,
GPQA-Diamond, MMLU-Pro. Geometry + principalEvidence collected at all 10 checkpoints (the two
items Cycle 06 left PARTIAL). Detail:
`local_experiment_results/cycle_07_base_sft_trajectory/run_01/RESULTS_07.md` +
`cap_pilot_findings_REVISION.md` + `nontermination_trajectory.csv`.

**Measurement correction (cap-pilot, no re-run):** RESULTS_07's generative-math eval used
`max_tokens=4096` (math500/numina). A first read (this analyst) flagged this as a truncation
artifact suppressing accuracy. The Code-side **cap-pilot disproved that**: the scorer
(`math_verify`/last-`\boxed`) already extracts the boxed answer from truncated responses
(step_080 math500: of 396 truncated, **209 scored correct**), so **as-run acc is cap-robust and
approximately valid** — `0.620@4096 ≈ 0.617@24576` (6× cap, acc unchanged). **No re-run.** The
earlier "completed-sample acc → 0.97" was selection-on-short-problems, not a valid correction.

**The real signal is non-termination, and it is time-offset from the accuracy dip** (this
analyst's refinement of the Code revision, from finisher-acc decomposition):

| step | as-run MATH500 | non-term % | finisher acc |
|---|---|---|---|
| 20 | 0.572 (dip) | 6.8% | **0.594** (finishers also dip; non-term still low) |
| 80 | 0.620 | **79.2%** | 0.971 (finishers recovered; non-term peaks) |
| 320 | 0.698 (peak) | 46.8% | 0.992 |
| 624 | 0.680 | 50.2% | 0.980 |

Two distinct transients, NOT concurrent: **(1) a clean accuracy/capability dip at step_20**
(finisher acc 0.594, non-term only 6.8% → genuine, not an output-control artifact); **(2) a
later non-termination explosion at step_40–160** (peaks 79% at step_80, by which point finisher
acc has already recovered to 0.97). as-run acc is a **valid end-to-end task-success/capability
measure** (the model was evaluated exactly this way; non-termination is real model behavior — the
harness respects EOS, 37–50% of samples terminate). The finisher/termination decomposition is an
**interpretive enrichment**, not a reason to distrust the number: the high-step plateau ~0.68 is
**termination-limited, not reasoning-limited** (finisher acc ~0.98), which matters only for
cross-model comparison and the geometry/mechanism story. (Also corrects the Code revision's
"finishers stay ~100% correct throughout" — false at step_20, where finisher acc dips to 0.594.)

**Gate verdicts (pre-registered):**

| Gate | Verdict | Basis (as-run, valid) |
|---|---|---|
| A07 feasibility | **PASS** | MATH500 step_624 0.680 > base+1SE 0.658. (Understated: finisher acc 0.98 → capability gain is larger; as-run depressed by non-termination.) |
| B07 dip-recovery | **FULL PASS** | dip at step_20 (clean, real), recovery above base by step_160, peak 0.698 at step_320. Holds on as-run numbers. |
| C07 OOD-lite | **No transfer + mild MMLU-Pro degradation** | GPQA-D flat (all ‖Δ‖<0.75 SE, noise); MMLU-Pro final −0.029 (−2.2 SE, real mild degradation; −3 SE dips at step_40 & step_480; non-monotone). C07's literal "pass" rests on GPQA +0.005 noise → honest read = no OOD-lite gain. |

**Geometry (diagnostic, valid — as-run acc valid so correlation stands):**
- **Triple co-location at step_20** (robust, clean regime): argmin(acc)=argmax(effective_rank)=
  argmin(xs_log_spectrum_gap)=step_20. The capability dip co-locates with max activation-spectrum
  expansion + max S–X spectral gap.
- effective_rank ↔ as-run acc Spearman −0.71; xs_log_spectrum_gap ↔ acc +0.77 (n=10, p≈0.01–0.04,
  small effect sizes ~1.8% rank span, concurrent-not-causal). drift_from_base monotone → negative
  control, does not explain the non-monotone dip.
- **OverlapLift < 1 throughout, monotone rising** (down_proj 0.48→0.74; attention 0.04–0.28),
  UAngle/VAngle≈0°. Recorded as fact only; under LoRA r=32 the RandomExpected baseline normalizes
  update_count but NOT the low-rank structure → cannot judge on/off-principal without a full-param/
  random/OPD reference. **Interpretation deferred to Cycle 08** (with OPD arm, same protocol). NOT
  framed as overturning any prior expectation.

**aime24:** non-terminating from step_40+ (86–93%), finishers N=2–4, acc floor-y/unreliable →
**qualitative only** ("4B+LoRA cannot solve AIME in a terminating way"); far-OOD N=30 weak probe.

**Allowed claim this cycle:** A07 PASS + B07 dip-recovery (as-run, cap-robust) are confirmatory;
the capability-vs-termination decomposition is confirmatory (from existing samples). C07 = no
OOD-lite transfer + mild MMLU-Pro degradation. OverlapLift recorded but interpretation deferred to
Cycle 08. No comparison to Cycle 04/05/06 (different model/data/format). No H-OPD claim (no OPD arm).

**Cycle 07 status: data valid as-run (no re-run); result analysis complete (2026-06-27).**
Remediation plan superseded by `cap_pilot_findings_REVISION.md`.

---

## Cycle 08: H-OPD vs SFT Trajectory (`cycle_08_h_opd_vs_sft_comparison`)

**Setup:** OPD arm — Qwen3-4B-Base student ← Qwen3-8B teacher, on-policy distillation (verl,
forward_kl_topk, LoRA r=32), same 10-checkpoint grid as Cycle 07 SFT. Compared head-to-head vs the
Cycle 07 SFT trajectory. Detail: `local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/`
(RESULTS_08.md, opd_vs_sft_unified.md, geometry/, rho_trajectory.csv, cap_pilot_findings, cycle08_rho_metric_spec).

**B08 — capability (CONFIRMATORY, the headline positive):** Both arms re-tested at a **unified cap
16384** (OPD's long distilled CoT needed it; cap-calibrated step_320: 4096→0.680, 16384→0.880,
24576→0.910). **Corrected MATH500 OPD ≫ SFT at every step, +0.10 to +0.24** (step_624 OPD 0.848 vs
SFT 0.752). The as-run "post-peak decline" was a truncation artifact. Note OPD is MORE
truncation-suppressed than SFT at 16384 (OPD trunc 0.92 vs SFT 0.35 at late steps) → OPD's true
margin is if anything larger. A08 feasibility PASS, B08 dip-recovery holds.

**Dip locations (as-run, dip is early → truncation-immune) — corrects the earlier "dip = SFT@20" framing:**

| bench | axis | OPD dip@step (Δ) | SFT dip@step (Δ) |
|---|---|---|---|
| math500 | ID | **step_5 (−0.100)** | step_20 (−0.064) |
| numina | ID | **step_5 (−0.070)** | step_10 (−0.016) |
| aime24 | ID(floor) | none | none |
| gpqa | OOD | none | step_5 (−0.025) |
| mmlu_pro | OOD | step_80 (−0.116, recovers, net +0.016) | step_480 (−0.041, net −0.029) |

**OPD dips earlier AND deeper on ID math than SFT.** Both arms' ID dips co-locate with an
activation-ER rank uptick (OPD +1.3 @ step_5-10; SFT +4.56 @ step_20, layer 18).

**C08 — OOD-lite:** GPQA flat (noise). **MMLU-Pro: OPD preserves (net +0.016) vs SFT erodes (net
−0.029).** OPD's OOD is deep-transient-dip-then-recover; SFT's is slow-late-erode.

**D08 — activation geometry (the real discriminator):** effective_rank — **SFT has a mid-layer
(L18) step_20 rank BUMP (+4.56); OPD monotone contraction, no bump.** This is the ONE metric that
cleanly separates OPD from SFT and tracks the OOD outcome.

**A08 — weight-space principal geometry (NULL / RETRACTED / superseded):**
- **OverlapLift RETRACTED:** a scale/rank/bf16-matched random-LoRA null (`cycle08_random_lora_ref.py`)
  gives OverlapLift ≈ OPD at every step (0.355→0.772 vs 0.369→0.760). The metric is
  magnitude/bf16-mask-driven, not directional; its growth over training is a ‖ΔW‖×bf16 artifact, not
  learning. (Caveat: null is OPD-scaled → OPD-vs-null is clean; SFT-vs-null is NOT scale-matched.)
- **ρ (scale-invariant subspace-energy, `rho_probe.py`, spec `cycle08_rho_metric_spec.md`) added:**
  both arms weakly ON-principal (z≈+6, ρ²_U≈2% at k=32), indistinguishable from each other once both
  have real updates. Early "SFT off-principal" is a ΔW≈0 artifact (LoRA zero-init: SFT dW_fro 0.000
  @step_5). **BUT this ON-principal reading is bf16-biased-toward-on + LoRA-confounded → NOT reliable.**
- **External literature supersedes our weight-space read:** two June-2026 full-param papers
  (arXiv 2606.13657, 2606.07082) find **OPD is OFF-principal** (relaxed regime between SFT-on and
  RLVR-tight-off), in weight space. Our LoRA+bf16 "weakly on-principal" is the opposite sign → almost
  certainly our confound (bf16→on bias). **Weight-space on/off verdict deferred to full-param + fp32
  (cycle09); our LoRA ρ reading is not load-bearing.**

**Mechanism hypothesis (compression, from D08):** ER = representational compression (low rank =
compressed). The ID dip co-locates with a compression-BREAK (rank uptick); recovery = re-compression
("capability injection"). **Break MAGNITUDE tracks OOD damage** (SFT big break +4.56 → OOD erodes;
OPD small +1.3 → OOD preserved). ID-dip DEPTH tracks update speed/coherence instead (OPD's fast
teacher-coherent early jump → deep dip but small break). Suggestive, not causal.

**Allowed (confirmatory):** B08 (OPD ≫ SFT MATH500 +0.10–0.24, unified cap); C08 (OPD preserves
MMLU-Pro, SFT erodes); D08 (SFT step_20 mid-layer ER bump, OPD none). **Not allowed / pending:** any
OPD on/off-principal weight verdict (LoRA+bf16 confounded; ext. lit says off-principal — needs
full-param/fp32); H-OPD "reward-density-not-on-policy" claim (ext. lit suggests on-policy may drive
off-principal — theory under revision); compression→OOD causality (correlational, n=2 arms).

**⚠️ related_work 待同步 (NOT written here — Related Work conversation):** the OPD-off-principal
weight-geometry finding is now PUBLISHED (2606.13657, 2606.07082, full-param, weight-space). Scoop
risk on that specific claim. Candidate remaining novelty (moat): activation-whitening/compression
observation space + compression→OOD mechanism + the LoRA/bf16 measurement-critique. See memory
`cycle08-opd-geometry-lit-scoop` and Related Work seed below.

**Cycle 08 status: result analysis complete (2026-07-07). Weight-space on/off-principal deferred to
Cycle 09 (full-param or fp32-master retrain). Theory rebuild + related_work update pending (user).**
