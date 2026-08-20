# Claims Allowed

```yaml
view_type: current_claim_boundary
status: active
last_updated: 2026-07-07
maintenance_rule: update_when_results_change_or_new_cycle_lands
```

This file defines what claims are currently supported by valid experimental results. See `current_picture.md` for the underlying numbers. See `incremental_log.md` for how each finding was established.

---

## Allowed Claims

**On the experimental pipeline:**
- Cycle 04 completed a stricter OPD-like vs continued SFT comparison than Cycle 03.
- The engineering loop is now stronger: high-sample GSM8K, OOD-lite evaluation, multi-layer GetSlice geometry, matched-pair selection, and ID/OOD evaluation are all available.
- NuminaMath-test is a usable ID evaluation axis (holdout split is zero-overlap with training prompts).

**On the Cycle 03 → 04 replication:**
- Cycle 03's OPD-like stability signal was not reproduced under Cycle 04's stricter conditions.
- The matched-gain comparison is sensitive to `theta0` saturation state: a less-saturated `theta0` shifts the matched SFT anchor to a nearly-stationary control.
- In Cycle 04, the matched SFT anchor collapsed to `sft_n128`, which barely moved from `theta0`.
- This is a control-anchor failure mode, not evidence that OPD is worse than SFT in general.

**On the ID axis (NuminaMath-test, valid):**
- The untrained `Qwen3-1.7B` instruct checkpoint is approximately tied with `theta0` on NuminaMath ID (0.5516 vs 0.5527, open-answer), and is above both OPD arms on ID.
- In the current 1.7B instruct + ≤1024 NuminaMath training-data setting, no training arm shows a net ID gain over the untrained base.
- GSM8K should be treated as an OOD/transfer axis, not as the primary performance axis.
- The matching basis for OPD vs SFT comparisons is NuminaMath-test ID. Under this basis, sft_n128 (0.5370) is the nearest SFT anchor for opd_lmbda05 (0.5392, Δ=+0.0022) and opd_lmbda1 (0.5348, Δ=−0.0022).

**On the eval protocol corrections (Cycle 05 findings):**
- The original Cycle 04 GSM8K (~0.41 band) and MATH500 (0.000) numbers were overwhelmingly a measurement-protocol floor: missing chat template, insufficient generation budget, broken dollar-span extractor (MATH500), and a thinking/non-thinking train–eval format mismatch.
- Correcting the protocol raises GSM8K to ~0.61-0.73 and MATH500 to ~0.54-0.73 for all 8 models.
- `theta0` is ahead of the untrained `base` on both GSM8K (+0.0175) and MATH500 (+0.0120); both deltas are below ~1σ individually but directionally consistent across axes.
- Headroom above the untrained base does exist on the corrected protocol, reversing the Cycle 04 base-ceiling reading on these axes.
- `opd_lmbda05` matches `theta0` on GSM8K (+0.0175 over base) but is slightly below `base` on MATH500 (−0.006), reproducing the pattern that OPD's GSM8K edge is not universal.
- `opd_lmbda1` is the highest GSM8K model of all 8 (+0.0341 over base, +0.0166 over theta0), but is below base on NuminaMath-test ID (0.5348 vs 0.5516).
- Any MATH500 = 0.0 or near-zero math evaluation must be audited for extraction failure before being interpreted as model inability (project-level rule).

**On OOD-lite (MMLU / TruthfulQA / WinoGrande / ARC-challenge, valid from Cycle 05):**
- Valid MMLU numbers are obtained via base model mode (no chat template, 5-shot, standard loglikelihood) — Qwen3 official protocol (arXiv 2505.09388, `mmlu_protocol_version=2`).
- ARC-challenge uses the same no-chat-template loglikelihood protocol as MMLU and WinoGrande (25-shot lm-eval task default, acc_norm, N=1172).
- OOD-lite spread across all 8 models is extremely small on all four tasks: MMLU ±0.0041, TruthfulQA ±0.0061, WinoGrande ±0.0075, ARC-challenge ±0.0073. No model is distinguishably better or worse on any OOD-lite metric.
- OOD-lite functions as a capability preservation check only: fine-tuning (all arms) does not meaningfully alter general language ability in this setting.
- TruthfulQA-MC1 of 0.33-0.34 is normal for 1.7B models (reference: Qwen2.5-1.5B-Instruct ≈ 0.30, Llama-2-7B ≈ 0.306). No anomaly.
- Gate C04 is formally confirmed FAIL (OPD does not beat matched SFT on OOD-lite), but this is a non-signal: all models cluster within <1.5% on every metric across all four OOD-lite tasks. The failure reflects insufficient discriminative power, not that OPD harms generalization.

**On sft_n1024 degradation:**
- `sft_n1024` shows severe math regression (NuminaMath ID −0.163, GSM8K −0.086, MATH500 −0.174 vs base) with stable OOD-lite performance.
- Train-eval format alignment is verified (Qwen3 tokenizer injects identical empty think block in both training and eval paths); the degradation is not a format artifact.
- This pattern is consistent with math catastrophic forgetting at n=1024 SFT samples on a 1.7B model.
- `sft_n1024` is an outlier and is not used as a comparison anchor for OPD analysis; sft_n128/256/512 are the relevant SFT controls.

**On geometry:**
- OPD lowers effective rank relative to `theta0` (more concentrated activation spectra); SFT raises it.
- These are directional signals only; UV/principal-angle artifacts were not saved so no subspace-overlap claim is possible.

**On Cycle 06 SFT-only feasibility (confirmatory):**
- Instruct no-think concise SFT shows no measurable NuminaMath-test gain at any n∈{512,1024,2048,4096} over the untrained `instruct_base` (all arms below base+1SE; threshold 0.549). **Gate A06 FAIL.**
- Instruct SFT degrades math reasoning monotonically-ish (NuminaMath −0.12 to −0.17, MATH500 −0.15 to −0.21 vs base); GSM8K mostly held.
- OOD-lite (MMLU 6-subtask subset, TruthfulQA, WinoGrande, ARC) is preserved within noise across all 10 models and both series — same non-signal as Cycle 04/05. **Gate B06 = no degradation.**

**On Cycle 06 base series (indicative-only / pending):**
- Base-series generative-math eval is a truncation artifact (no stop sequence; ~66% gsm8k/numina truncation from runaway few-shot continuation). The apparent "base SFT +0.28 on GSM8K" is differential de-truncation of a broken baseline: with `stop=["\n\nProblem:"]` base GSM8K is flat ~0.66 and SFT is neutral. This is an engineering finding pending re-collection, NOT a base feasibility result. **Gate A06b UNDETERMINED.**
- No-think SFT removes the *function* of thinking (think-mode probe: instruct_base +0.13 vs SFT −0.02..−0.06 despite 100% `</think>` closure). Indicative only (n=100, single seed).

**On Cycle 07 Qwen3-4B-Base trajectory (confirmatory; as-run valid, cap-robust, no re-run):**
- SFT from Qwen3-4B-Base on Math-CoT-20k is feasible: MATH500 step_624 0.680 > base+1SE 0.658. **Gate A07 PASS.**
- The trajectory shows dip-and-recovery: MATH500 dips to 0.572 at step_20, recovers above base by step_160, peaks 0.698 at step_320. **Gate B07 FULL PASS.**
- as-run generative-math accuracy is **cap-robust and approximately valid** — the scorer extracts `\boxed` from truncated responses (cap-pilot: 0.620@4096 ≈ 0.617@24576). `max_tokens=4096` is not a false-negative artifact.
- as-run MATH500 accuracy is a **valid end-to-end task-success (capability) measure** — the model was evaluated exactly this way, non-termination is real model behavior (the harness respects EOS: 37–50% of samples do terminate), and failing to produce a terminating scorable answer is a genuine failure. 0.680 is the real number.
- **Additionally** (interpretive enrichment, not a reason to distrust the number): as-run acc can be decomposed into capability (finisher acc) + output-control (termination rate). The step_20 dip is genuine (finisher acc 0.594, non-term only 6.8%); a separate, later non-termination explosion peaks at step_80 (79%, finisher acc already ~0.97). So the high-step plateau ~0.68 is **termination-limited, not reasoning-limited** (finisher acc ~0.98) — relevant for cross-model comparison (don't read 0.68 as "reasons worse than Rethink-SFT 0.86") and the geometry/mechanism story.
- OOD-lite: GPQA-Diamond flat (|Δ|<0.75 SE, no transfer); **MMLU-Pro shows mild real degradation** (final −2.2 SE, −3 SE dips at step_40/480; non-monotone). C07 = no OOD-lite transfer + mild MMLU-Pro degradation.
- Geometry: the step_20 **triple co-location** (argmin acc = argmax effective_rank = argmin xs_log_spectrum_gap) is robust; effective_rank↔acc Spearman −0.71, xs_gap↔acc +0.77 (n=10, small effect, concurrent-not-causal).

**On Cycle 07 OverlapLift (recorded fact; interpretation deferred):**
- OverlapLift is < 1 across all modules/steps, rising monotonically along the trajectory (down_proj 0.48→0.74; attention 0.04–0.28); UAngle/VAngle ≈ 0°. This is a recorded measurement, not an on/off-principal verdict.

**On Cycle 08 OPD vs SFT (confirmatory):**
- OPD (4B ← 8B teacher, on-policy distillation) beats SFT on MATH500 at every checkpoint by +0.10 to +0.24 under a unified cap (16384); OPD final 0.848 vs SFT 0.752. The margin is conservative (OPD more truncation-suppressed).
- OPD preserves MMLU-Pro (net +0.016) while SFT erodes it (net −0.029); GPQA flat both.
- Activation effective_rank cleanly separates the arms: SFT shows a mid-layer (L18) step_20 rank bump; OPD contracts monotonically with no bump. This is the discriminating geometry observable and it tracks the OOD outcome.
- OPD dips earlier and deeper than SFT on ID math (math500/numina at step_5); both ID dips co-locate with the ER uptick.

**On Cycle 08 weight-space principal geometry (recorded / NOT a verdict):**
- OverlapLift is retracted as an on/off-principal signal (indistinguishable from a scale-matched random-LoRA null; magnitude/bf16-driven). ρ (scale-invariant) reads weakly on-principal but is bf16-biased and LoRA-confounded. External full-param work (arXiv 2606.13657, 2606.07082) finds OPD off-principal. No weight-space on/off-principal verdict is claimed this cycle.

---

## Not Allowed Claims

- Do not claim any Cycle 08 OPD on/off-principal WEIGHT-space verdict — LoRA+bf16 confounded (OverlapLift retracted; ρ bf16-biased-toward-on); external full-param lit says OPD is off-principal, opposite to our LoRA reading. Deferred to full-param/fp32 (Cycle 09).
- Do not claim "on-policy is not the driver / reward-density is" — external full-param evidence (OPD off-principal like relaxed RL) currently points the other way; theory is under revision.
- Do not claim the Cycle 08 OPD-off-principal weight-geometry finding as novel — it is already published (arXiv 2606.13657, 2606.07082, June 2026, full-param, weight-space). Candidate novelty is the activation-whitening/compression space + compression→OOD mechanism, pending verification.
- Do not claim compression→OOD causality — the ER↔OOD link is correlational (n=2 arms) and ER is measured on math-probe activations, not OOD-domain inputs.
- Do not claim full OPD replication or that OPD improves reasoning accuracy.
- Do not claim OPD is more stable than SFT in general.
- Do not claim broad OOD generalization from Cycle 04.
- Do not use any MATH500 = 0.0 (original Cycle 04) number as evidence of model inability.
- Do not write the local unfinetuned `Qwen3-1.7B` instruct checkpoint as the official `Qwen3-1.7B-Base` pretraining-only checkpoint.
- Do not cite OOD-lite numbers from the original Cycle 04 run or the Cycle 05 chat-template rerun (MMLU collapsed to 0.2295 under those protocols); use Cycle 05 base-model-mode numbers only.
- Do not use OOD-lite as a discriminating axis between OPD and SFT — the spread is <1.5% across all 8 models, insufficient for any reliable comparison.
- Do not cite any uncorrected GSM8K numbers (0.33-0.43 band) for any model.
- Do not treat the Cycle 04 base-ceiling / no-headroom reading as confirmed for GSM8K or MATH500.
- Do not claim principal-angle or subspace-overlap evidence (UV artifacts not saved).
- Do not treat the Cycle 03 positive and Cycle 04 negative results as a simple contradiction; both are confounded by the matched-control anchor.
- Do not write Gate D as "nearly passed." It failed under the pre-registered criteria.
- Do not run further OPD-vs-SFT comparisons without keeping the untrained base as a first-class reference point.
- Do not use `sft_n1024` as a comparison anchor for OPD analysis; it is an outlier showing catastrophic math forgetting at n=1024.
- Do not interpret Gate C04 failure as evidence that OPD harms generalization — the axis lacks discriminative power in this setting.
- Do not cite any Cycle 06 base-series numina or GSM8K number as a capability or feasibility result — they are truncation artifacts until re-collected with a stop sequence (Cycle 07).
- Do not make any base-series SFT feasibility (A06b) claim from Cycle 06; it is UNDETERMINED.
- Do not attribute the Cycle 07 high-step plateau (~0.68) to reasoning saturation — finisher acc is ~0.98; the plateau is termination-limited. (The as-run number itself IS a valid end-to-end task-success/capability measure; this caution is only against mis-reading WHAT limits it.)
- Do not claim the Cycle 07 `max_tokens=4096` truncation suppressed accuracy / that a larger cap recovers higher acc — the cap-pilot disproved this (acc is cap-robust; the scorer is truncation-robust).
- Do not use the Cycle 07 "completed-sample acc → 0.97" numbers as a trajectory — they are selection-on-short-problems (upward biased), not a valid capability measure.
- Do not judge Cycle 07 OverlapLift as on- or off-principal — under LoRA r=32 the random baseline does not normalize the low-rank structure; deferred to Cycle 08 (needs OPD arm + LoRA-aware reference).
- Do not frame Cycle 07 OverlapLift<1 as overturning the Cycle 06 design's on-principal expectation — SFT is not on-policy, and a high-but-<1 overlap under LoRA sparsity is not inconsistent with the SFT-on-principal literature.
- Do not cite Cycle 07 AIME24 accuracy as a capability measure — it is non-terminating from step_40+ (finishers N=2–4); qualitative only.
- Do not compare Cycle 07 (4B / Math-CoT / think-format) numbers against Cycle 04/05/06 (1.7B / NuminaMath); different model, data, and format.
- Do not make any H-OPD or OPD claim from Cycle 07 — no OPD arm was trained; H-OPD is a Cycle 08 hypothesis.
- Do not state the no-think-SFT "destroys thinking function" finding as confirmatory — it is n=100, single seed, indicative only.
- Do not cite any Cycle 06 OverlapLift / principal-evidence / geometry number — those CSVs were not copied back and are unanalyzed (C06 PARTIAL).
- Do not compare Cycle 06 instruct/base-series scores across series (different prompt formats), nor against Cycle 04/05 SFT arms (different checkpoint and n range).
- Do not read Cycle 06 MMLU absolute values against official MMLU — it is a 6-subtask subset, ~+4.5 inflated, intra-experiment only.

---

## Recommended Wording

**For the main Cycle 04 conclusion:**
```
Cycle 04 provides a stricter negative replication of the preliminary Cycle 03
stability signal. The result suggests that matched-gain OPD-vs-SFT comparisons
are sensitive to theta0 saturation and control-anchor selection.
```

**For the narrow OPD signal on GSM8K:**
```
OPD-like training preserved or slightly improved GSM8K relative to theta0, but
the effect was below the pre-registered improvement threshold and did not come
with a robust OOD-lite or geometry stability advantage.
```

**For the corrected base-ceiling:**
```
After correcting the GSM8K and MATH500 evaluation protocols, theta0 is ahead of
the untrained base on both axes (+0.0175 and +0.0120 respectively), though
neither delta is individually significant. Headroom above base exists; the
original base-ceiling reading was a measurement artifact.
```

**For OOD-lite:**
```
All training arms — OPD and SFT alike — maintain general language ability within
±1.5% on MMLU (5-shot, base model mode), TruthfulQA-MC1, WinoGrande, and ARC-challenge
(25-shot, no chat template). This axis confirms capability preservation but has
insufficient discriminative power to distinguish OPD from SFT in this experimental setting.
```

**For sft_n1024:**
```
sft_n1024 shows severe math regression (NuminaMath ID −0.163, MATH500 −0.174
relative to base) with stable OOD-lite performance, consistent with catastrophic
forgetting of math reasoning at this SFT data scale on a 1.7B model. It is
excluded from the primary OPD-vs-SFT comparison.
```

**Avoid:**
```
OPD failed.
SFT is better than OPD.
OPD improves GSM8K.
Cycle 04 proves the mechanism false.
MATH500 shows the models cannot do competition math.
The base and theta0 are tied on GSM8K. [outdated — only valid under the broken protocol]
OPD degrades generalization. [Gate C failure is a non-signal, not a negative]
```
