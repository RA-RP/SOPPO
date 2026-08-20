# Current Picture: All Valid Experimental Results

```yaml
view_type: current_aggregate
status: active
last_updated: 2026-07-07
maintenance_rule: update_when_new_results_land_or_supersede_prior_results
```

This file answers: **what does the full set of currently valid experimental results tell us?**

Valid = latest corrected version where a correction exists; original result where it has not been superseded. Results based on a known-defective protocol are excluded even if not yet replaced.

For how each cycle got here, see `incremental_log.md`. For detailed tables and evidence, see `local_experiment_results/`.

---

## What We Can Currently Read

### ID Axis — NuminaMath-test (valid, all 8 models, from Cycle 05)

Protocol: chat template applied, `enable_thinking=False`, `\boxed{}`-aware scoring, open-answer subset, N=1024 (clean holdout, zero overlap with training prompts).

| Model | NuminaMath ID (open-answer) |
|---|---|
| base (untrained Qwen3-1.7B instruct) | 0.5516 |
| theta0 | 0.5527 |
| opd_lmbda05 | 0.5392 |
| opd_lmbda1 | 0.5348 |
| sft_n128 | 0.5370 |
| sft_n256 | 0.5516 |
| sft_n512 | 0.5179 |
| sft_n1024 | 0.3890 |

**Reading:** base and theta0 are approximately tied at the top. Both OPD arms are below base on ID. sft_n128/256 are near base; sft_n512 slightly below. sft_n1024 severely degraded (−0.163 vs base — math catastrophic forgetting). No training arm shows a net ID gain over the untrained base.

**Matched control anchor:** The NuminaMath-test ID axis is the matching basis for OPD vs SFT comparisons (not GSM8K). Under this basis, sft_n128 (0.5370) is the nearest SFT anchor for opd_lmbda05 (0.5392, Δ=+0.0022) and opd_lmbda1 (0.5348, Δ=−0.0022).

---

### GSM8K Axis — corrected, all 8 models valid (from Cycle 05)

Protocol: chat template + `enable_thinking=False` + 3072 tokens + `\boxed{}`-aware extraction. N=1319. Treated as OOD/transfer axis (not training-domain score).

| Model | GSM8K | ±SE |
|---|---|---|
| base | 0.7005 | 0.0126 |
| theta0 | 0.7180 | 0.0124 |
| opd_lmbda05 | 0.7180 | 0.0124 |
| **opd_lmbda1** | **0.7346** | 0.0122 |
| sft_n128 | 0.7096 | 0.0125 |
| sft_n256 | 0.6839 | 0.0128 |
| sft_n512 | 0.6732 | 0.0129 |
| sft_n1024 | 0.6141 | 0.0134 |

**Reading:** opd_lmbda1 is highest of all 8 models (+0.0341 over base, +0.0166 over theta0). Both OPD arms match or exceed theta0. SFT shows a degradation trend with n: sft_n128 is near theta0, sft_n256/512 drop below base, sft_n1024 severely regressed (−0.086).

---

### MATH500 Axis — corrected, all 8 models valid (from Cycle 05)

Protocol: chat template + `enable_thinking=False` + 3072 tokens + `math_verify` symbolic scoring + `last_boxed` extraction. N=500.

| Model | MATH500 | ±SE |
|---|---|---|
| base | 0.7140 | 0.0202 |
| theta0 | 0.7260 | 0.0199 |
| opd_lmbda05 | 0.7080 | 0.0203 |
| opd_lmbda1 | 0.7140 | 0.0202 |
| sft_n128 | 0.7260 | 0.0199 |
| sft_n256 | 0.7080 | 0.0203 |
| sft_n512 | 0.7140 | 0.0202 |
| sft_n1024 | 0.5400 | 0.0223 |

**Reading:** theta0 and sft_n128 are tied at the top (+0.0120 over base). opd_lmbda05 and sft_n256 are slightly below base (−0.006). opd_lmbda1 and sft_n512 tie base. Unlike GSM8K, OPD does not show a consistent edge here. sft_n1024 severely regressed (−0.174).

---

### OOD-lite (MMLU / TruthfulQA / WinoGrande / ARC-challenge) — valid, all 8 models (from Cycle 05)

**Protocol notes:**
- MMLU: base model mode (no chat template, 5-shot, standard loglikelihood) — Qwen3 official protocol (arXiv 2505.09388, `mmlu_protocol_version=2`)
- TruthfulQA-MC1: chat template + loglikelihood, 0-shot
- WinoGrande: no chat template, 0-shot, loglikelihood
- ARC-challenge: no chat template, 25-shot (lm-eval task default), `acc_norm`, N=1172 — same no-chat-template loglikelihood protocol as MMLU/WinoGrande

| Model | MMLU | TruthfulQA MC1 | WinoGrande | ARC acc_norm |
|---|---|---|---|---|
| base | 0.6061 | 0.3341 | 0.6172 | 0.4565 |
| theta0 | 0.6019 | 0.3415 | 0.6235 | 0.4590 |
| opd_lmbda05 | 0.6004 | 0.3390 | 0.6251 | 0.4650 |
| opd_lmbda1 | 0.6061 | 0.3427 | 0.6235 | 0.4616 |
| sft_n128 | 0.6049 | 0.3439 | 0.6290 | 0.4599 |
| sft_n256 | 0.6033 | 0.3415 | 0.6235 | 0.4667 |
| sft_n512 | 0.6086 | 0.3378 | 0.6322 | 0.4710 |
| sft_n1024 | 0.6065 | 0.3317 | 0.6322 | 0.4667 |

**Reading:** Spread across all 8 models: MMLU ±0.0041 (range 0.0082), TruthfulQA ±0.0061 (range 0.0122), WinoGrande ±0.0075 (range 0.0150), ARC-challenge ±0.0073 (range 0.0145). All models — including sft_n1024 which shows severe math regression — cluster within <1.5% on every OOD-lite metric across all four tasks. This axis functions as a capability preservation check, not a discriminating signal between OPD and SFT.

Reference context: 0.33-0.34 TruthfulQA-MC1 is normal for 1.7B models (Qwen2.5-1.5B-Instruct ≈ 0.30, Llama-2-7B ≈ 0.306). No anomaly.

---

### sft_n1024 Degradation Summary

sft_n1024 shows a qualitatively distinct pattern from all other models:

| Axis | sft_n1024 | base | Δ |
|---|---|---|---|
| NuminaMath ID | 0.3890 | 0.5516 | **−0.163** |
| GSM8K | 0.6141 | 0.7005 | **−0.086** |
| MATH500 | 0.5400 | 0.7140 | **−0.174** |
| MMLU | 0.6065 | 0.6061 | +0.000 |
| TruthfulQA | 0.3317 | 0.3341 | −0.002 |
| WinoGrande | 0.6322 | 0.6172 | +0.015 |
| ARC-challenge | 0.4667 | 0.4565 | +0.010 |

Math axes uniformly degraded; OOD-lite stable or slightly higher. Train-eval format alignment has been verified (Qwen3 tokenizer injects identical empty think block in training and eval paths). This pattern is consistent with catastrophic forgetting of math reasoning at n=1024 SFT samples on a 1.7B model, not a format artifact. sft_n1024 is excluded from the OPD vs SFT comparison; sft_n128/256/512 are the relevant SFT controls.

---

### Gate Analysis — Cycle 04 re-analysis with Cycle 05 corrected numbers

| Gate | Cycle 04 Verdict | Cycle 05 Re-analysis | Change |
|---|---|---|---|
| A04: feasibility | PASS | CONFIRMED PASS | No change |
| B04: matched control | PASS | CONFIRMED PASS | Matching basis clarified as NuminaMath-test ID; sft_n128 remains nearest anchor for both OPD arms |
| C04: stability | FAIL | CONFIRMED FAIL | OPD does not beat sft_n128 on any OOD-lite metric; but the axis has <1.5% spread — non-signal, not a negative finding |
| D04: gain | FAIL | CONFIRMED FAIL | Depends on Gate C04 |

**Clarification on Gate C04:** The original Gate C04 failure was based on invalid MMLU numbers. With correct OOD-lite numbers, OPD formally remains below sft_n128 on all four metrics (formal fail stands). However, all 8 models including base cluster within <1.5% on every OOD-lite metric — the failure reflects insufficient discriminative power in this setting, not that OPD harms generalization.

---

### Geometry — from Cycle 04 (directional signal only)

GetSlice (multi-layer, 7 modules × 3 layers × 2 prompt types):
- OPD lowers effective rank relative to `theta0` (more concentrated spectra).
- SFT raises effective rank (more dispersed).
- OPD does not consistently reduce drift from `theta0`; `opd_lmbda1` layer-14 drift exceeds all SFT controls.
- UV/principal-angle artifacts not saved → no subspace-overlap claim possible.

---

### Cycle 06 — SFT-only feasibility & degradation (instruct A06 FAIL; base UNDETERMINED)

New start point this cycle: a fresh 10-model SFT sweep (instruct ×4 + base ×4 over
n∈{512,1024,2048,4096} + 2 untrained refs), NOT comparable to the Cycle 04/05 8-model set
(different checkpoint, n range, base series new). Detail:
`local_experiment_results/cycle_06_sft_feasibility_and_degradation/run_01/RESULTS_06.md`.

**Instruct series — NuminaMath ID (this cycle's measurement):**

| Model | numina (ID) | gsm8k | math500 |
|---|---|---|---|
| instruct_base | 0.533 | 0.717 | 0.724 |
| instruct_sft_n512 | 0.411 | 0.611 | 0.572 |
| instruct_sft_n1024 | 0.381 | 0.607 | 0.548 |
| instruct_sft_n2048 | 0.359 | 0.646 | 0.516 |
| instruct_sft_n4096 | 0.380 | 0.710 | 0.546 |

Threshold = base + 1SE = 0.549; all SFT arms below base. **A06 FAIL (confirmatory):** no-think
concise SFT adds no measurable NuminaMath-test gain at any scale and degrades math reasoning
(numina −0.12 to −0.17, math500 −0.15 to −0.21). Reinforces the standing project reading that
no-think concise SFT does not add capability over the untrained base.

**OOD-lite (both series):** preserved within noise across all arms — same <1.5%-spread
non-signal as Cycle 04/05. **B06 = no degradation (confirmatory).** (MMLU here is a 6-subtask
subset, ~+4.5 above official full MMLU, intra-experiment only.)

**Base series — NOT yet valid (excluded from the picture):** base generative-math runners had no
stop sequence; a 4-shot base model fabricates new `Problem:` continuations and the last-number
scorer reads a hallucinated digit (FINDING_06_base_gsm8k_stop, HIGH). The apparent monotonic
"base SFT +0.28" on GSM8K is **differential de-truncation of a broken baseline**: with
`stop=["\n\nProblem:"]` base GSM8K is flat ~0.66 and SFT is neutral. base numina (~66% trunc) is
artifact-dominated. **A06b UNDETERMINED** — base feasibility deferred to Cycle 07 stop-fix
re-collection. (math500, ~8–10% trunc, is unaffected: base math500 0.306→0.362 is a small real
gain but not load-bearing.)

**Indicative-only (not in the valid picture):** with `enable_thinking=True` on math500-first100
(n=100, single seed), untrained instruct_base gains +0.13 from thinking while every no-think-SFT
arm is flat-to-negative despite 100% `</think>` closure — no-think SFT removes the *function* of
thinking, not its form (FINDING_06_nothink_sft, MED). Needs full math500 + multi-seed before
confirmatory.

---

### Cycle 07 — Qwen3-4B-Base SFT trajectory on Math-CoT-20k (A07 PASS, B07 dip-recovery, valid as-run)

New regime: **Qwen3-4B-Base** + **Math-CoT-20k** (think-format long-CoT), single SFT run profiled
over 10 checkpoints (step_000…624). NOT comparable to the 1.7B / NuminaMath cycles above (different
model, data, format). Detail:
`local_experiment_results/cycle_07_base_sft_trajectory/run_01/{RESULTS_07.md, cap_pilot_findings_REVISION.md, nontermination_trajectory.csv}`.

**Data validity:** as-run generative-math acc is **cap-robust and approximately valid** — the
scorer extracts `\boxed` from truncated responses (cap-pilot: 0.620@4096 ≈ 0.617@24576). No re-run.
The `max_tokens=4096` is not a false-negative artifact. (An initial truncation-suppression reading
by this analyst was disproven by the Code cap-pilot.)

**MATH500 (primary, valid) — decomposed into capability vs output-control:**

| step | as-run acc | non-term % | finisher acc |
|---|---|---|---|
| 0 (base) | 0.636 | 6.0% | 0.662 |
| 20 | **0.572** (dip) | 6.8% | **0.594** |
| 80 | 0.620 | **79.2%** | 0.971 |
| 320 | **0.698** (peak) | 46.8% | 0.992 |
| 624 | 0.680 | 50.2% | 0.980 |

Two time-offset transients: **(1) clean accuracy dip at step_20** (finisher acc 0.594, non-term
only 6.8% → genuine capability dip, not output-control); **(2) non-termination explosion at
step_40–160** (peaks 79% @ step_80, after finisher acc already recovered to ~0.97). as-run acc is a
**valid end-to-end task-success/capability measure** (the model was tested exactly this way;
non-termination is real model behavior, not a harness artifact). The finisher/termination
decomposition is an **interpretive enrichment**: it shows the high-step plateau ~0.68 is
**termination-limited, not reasoning-limited** (finisher acc ~0.98) — which matters for cross-model
comparison and the geometry/mechanism story, not for trusting the 0.68 number.

**Gates (as-run, valid):** A07 **PASS** (0.680 > base+1SE 0.658; capability gain understated by
non-termination — finisher acc 0.98). B07 **FULL PASS** (dip@20, recovery > base by step_160, peak
0.698@320). C07 **no OOD-lite transfer + mild MMLU-Pro degradation** (see below).

**OOD-lite (loglikelihood, unaffected by truncation, valid):**
- **GPQA-Diamond:** flat — all |Δ| < 0.75 SE across the trajectory (base 0.394 → 0.399). No
  transfer, no degradation (far-OOD reasoning the 4B cannot do either way).
- **MMLU-Pro:** **mild real degradation** — final step_624 −0.029 (−2.2 SE), with −3 SE dips at
  step_40 and step_480; non-monotone (briefly +0.8 SE at step_160–320). Think-format math SFT does
  not transfer to OOD knowledge and mildly degrades MMLU-Pro by the end.

**Geometry (diagnostic, valid):**
- **Triple co-location at step_20** (robust, clean regime): argmin(acc) = argmax(effective_rank) =
  argmin(xs_log_spectrum_gap) = step_20. Capability dip co-locates with max activation-spectrum
  expansion + max S–X gap. effective_rank↔acc Spearman −0.71, xs_gap↔acc +0.77 (n=10, small effect
  ~1.8% rank span, concurrent-not-causal); drift_from_base monotone = negative control.
- **OverlapLift < 1 throughout, rising** (down_proj 0.48→0.74; attention 0.04–0.28; UAngle/VAngle
  ≈0°). Recorded as fact; under LoRA r=32 the random baseline normalizes update_count but not the
  low-rank structure → on/off-principal judgment **deferred to Cycle 08** (with OPD arm). Not framed
  as overturning prior expectation.

**aime24:** non-terminating from step_40+ (86–93%, finishers N=2–4) → acc unreliable, **qualitative
only**; far-OOD N=30 weak probe.

---

### Cycle 08 — OPD vs SFT (4B, same grid): OPD ≫ SFT capability; discriminator is activation-geometry

OPD arm (Qwen3-4B-Base ← Qwen3-8B teacher, verl on-policy distillation, LoRA r=32) vs the Cycle 07
SFT trajectory. Detail: `local_experiment_results/cycle_08_h_opd_vs_sft_comparison/run_01/`.

- **Capability (B08, valid, unified cap 16384):** OPD ≫ SFT on MATH500 at **every** step, **+0.10 to
  +0.24** (step_624 OPD **0.848** vs SFT 0.752). Conservative — OPD is more truncation-suppressed at
  16384 (trunc 0.92 vs 0.35), so true margin ≥ this.
- **OOD (C08):** OPD **preserves** MMLU-Pro (net +0.016) vs SFT **erodes** (net −0.029); GPQA flat.
- **Activation geometry (D08) — the real discriminator:** effective_rank shows **SFT a mid-layer
  (L18) step_20 rank BUMP; OPD monotone contraction, no bump.** Only this metric cleanly separates
  the arms and tracks the OOD outcome.
- **Dip:** OPD dips EARLIER and DEEPER on ID math (math500 step_5 −0.100; numina step_5 −0.070) than
  SFT (step_10–20). Both dips co-locate with the ER uptick.
- **Weight-space principal geometry (A08) — NOT usable / superseded:** OverlapLift RETRACTED
  (≈ scale-matched random-LoRA; magnitude/bf16-driven). ρ (scale-invariant) gives a weak on-principal
  reading but is **bf16-biased-toward-on + LoRA-confounded → not reliable**. **Two full-param
  papers (arXiv 2606.13657, 2606.07082, 2026-06) find OPD is OFF-principal** (relaxed regime, SFT-on
  → OPD-relaxed-off → RLVR-tight-off); our LoRA reading has the opposite sign → our confound.
  **Weight-space on/off verdict is NOT in the valid picture; deferred to full-param/fp32 (Cycle 09).**
- **Mechanism hypothesis (compression):** ER = representational compression; ID dip ↔ compression-
  break (rank uptick); OOD damage ↔ break MAGNITUDE (SFT big break → erode; OPD small → preserve).
  Suggestive, n=2 arms, correlational.

---

## What Is Still Pending

| Open item | Status |
|---|---|
| **Cycle 08 weight-space OPD on/off-principal** | LoRA+bf16 confounded (OverlapLift retracted; ρ bf16-biased). Full-param OR fp32-master retrain needed. External full-param lit (2606.13657/07082) says OPD **off-principal** — our LoRA "weakly on" is unreliable. → Cycle 09 |
| **Theory rebuild (on-policy vs reward-density)** | Full-param OPD=off-principal (like relaxed RL) weakens the "on-policy is not the driver" hypothesis; on-policy may drive the off-principal direction, density modulating tightness. Needs an on-policy-vs-density ablation. → Cycle 09 + Theory conversation |
| **Related-work sync (SCOOP RISK)** | OPD-off-principal weight-geometry is now PUBLISHED (2606.13657, 2606.07082, full-param). Moat candidate = activation-whitening/compression space + compression→OOD + LoRA/bf16 measurement critique. → Related Work conversation |
| **Compression→OOD causality** | ER measured on MATH-probe activations only. To explain OOD-preservation: measure activation-spectrum on OOD-domain inputs, show SFT re-compression evicts OOD directions, predict MMLU-Pro Δ. → Cycle 09 |
| Geometry re-interpretation under corrected matched-control framework | Not started — Cycle 04 GetSlice data is valid (checkpoints unchanged); needs re-read using NuminaMath ID matching instead of GSM8K matching |
| Cycle 06 geometry / principalEvidence | Ran (rc=0) but `geometry_metrics.csv` / `principal_evidence.csv` NOT copied back → OverlapLift not analyzed. C06 PARTIAL; copyback deferred (Cycle 07 design scheduled it post-eval; Cycle 07 eval now done → can copy back) |
| Cycle 06 base-series math (numina/gsm8k) | Truncation-artifact-dominated (no stop seq). Re-collect with `stop=["\n\nProblem:"]` on all 3 `runner_base.py` before any base feasibility (A06b) judgment |
| No-think SFT think-function damage | Indicative only (n=100, single seed). Scale to full math500 + multiple seeds for confirmatory |
| Cycle 07 OverlapLift on/off-principal interpretation | Deferred to Cycle 08 — needs OPD arm (same protocol) + LoRA-aware reference before judging on/off-principal |
| Cycle 08 H-OPD | Pre-registered hypothesis: train OPD arm on the same 4B+Math-CoT setup; compare OverlapLift & geometry vs the SFT trajectory (dip / recover checkpoints as reference points) |
