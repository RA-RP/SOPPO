# Geometry Metric Definitions

```yaml
view_type: current_metric_definition
status: active_cycle08_metric_definition
primary_code: /root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/geometry_metrics.py
main_historical_definition: (snapshot removed 2026-06-16)
current_cycles_using_metrics:
  - cycle_04_opd_stability_gain
  - cycle_05_matched_control_id_ood
  - cycle_06_sft_feasibility_and_degradation
  - cycle_07_base_model_sft_feasibility
  - cycle_08_h_opd_vs_sft_comparison
last_organized: 2026-06-29
```

## Scope

These metrics summarize GetSlice singular spectra for activation-conditioned whitened module matrices. They are descriptive mechanism and stability signals. In current claim gates, drift is the confirmatory stability metric, while effective rank, spectral gap, and X-S spectrum level gap are auxiliary but important evidence.

## Design Rationale: Why A Spectral Lens, And Why These Four Metrics

This project needed a mechanism-level signal for "how much did training move the
model's internal representations," not just an output-accuracy number, because
Cycle 03/04's central question is about *stability* (does OPD move the
representation less than SFT for the same amount of learning), and output
accuracy alone cannot distinguish "did not move" from "moved but landed in an
equally good place." Singular-spectrum analysis of activation-conditioned
matrices was chosen over alternatives (e.g. raw weight-norm distance, gradient
norms, or output-token KL divergence) because related fine-tuning/forgetting
work (`LoRA vs Full Fine-tuning: An Illusion of Equivalence`, see
`../related_work/current_source_matrix.md`) shows that singular-vector
structure can expose qualitatively different update behavior (e.g. "intruder
dimensions") that scalar distance metrics miss. GetSlice was the specific
implementation available in this project's toolchain (see
`../exp/experimental_methodology/geometry_protocol.md` for how it is run).

Given that choice, the four metrics serve distinct, deliberately
non-redundant roles:

- `spectral_drift_from_theta0_diag` is the **confirmatory** metric because it
  directly operationalizes "how far did the model move from its starting
  point," which is exactly what a stability gate needs to threshold on.
- `effective_rank` and `spectral_gap` are **auxiliary** because they describe
  the *shape* of the spectrum (concentrated vs spread, dominant-direction
  strength) rather than movement — useful for distinguishing "the model moved
  a lot but stayed well-conditioned" from "the model moved a little but
  collapsed onto fewer directions," which raw drift alone cannot tell apart.
- `X_S_spectrum_level_gap` is auxiliary because it answers a different
  question from the other three: not "how much did this model change," but
  "how far is this model's training-target-conditioned spectrum from a
  shared reference probe," which is the metric that can in principle expose
  whether a model is overfitting to its own S-source (see
  `../exp/experimental_methodology/geometry_protocol.md` for why S differs by
  model role) rather than learning something that generalizes.
- `principal_angle` was planned but is not currently computable (UV/sketch
  artifacts are not saved by default); it would add subspace-overlap evidence
  that none of the other four metrics can provide, which is why it remains
  listed as a target rather than removed from the table.

Let `sigma = (sigma_1, ..., sigma_n)` be the non-negative singular spectrum after removing non-finite values, and let `sigma0` be the theta0 diagonal S-probe baseline spectrum for the same layer and module. The implementation uses `eps = 1e-12` in log-space metrics.

## Metrics

| Metric | Meaning | Formula / Implementation | Interpretation |
|---|---|---|---|
| `effective_rank` | Entropy-based dimensional spread of the spectrum. | Normalize `p_i = sigma_i / sum_j sigma_j`; compute `exp(-sum_i p_i log p_i)`. | Higher means the spectrum is spread across more directions; lower means more concentration. Lower is not automatically better. |
| `spectral_gap` | Dominance of the top singular direction. | For `k=1`, `sigma_1 - sigma_2`. Implemented as `spectral_gap(sigma, 1)`. | Larger can indicate a stronger leading direction, but it is local to the top spectrum and should be read with effective rank. |
| `spectral_drift_from_theta0_diag` | Overall log-spectrum movement from theta0. | `sqrt(mean((log(sigma_t + eps) - log(sigma0 + eps)) ** 2))` over the shared prefix length. | Direct measure of movement from the starting representation; used in Cycle 04 Gate C. Raw drift can penalize models that learned more, so Cycle 05 interprets it with ID learning. |
| `X_S_spectrum_level_gap` | Signed log-level gap between a shared X probe and the model S probe. | `mean(log(sigma_X + eps) - log(sigma_S + eps))` over the shared prefix length. | Measures how far a general/reference probe spectrum sits from the training-target probe spectrum for the same model. Nearer zero can indicate better alignment, but it is auxiliary. |
| `spectral_gap_abs_delta_from_start` | Absolute change in spectral gap from theta0. | `abs(gap_1(sigma_t) - gap_1(sigma0))`. | Designed in early plans, not always stored as a CSV column after v2; can be recomputed from `spectral_gap` and theta0 rows. |
| `principal_angle` | Subspace overlap between singular-vector spaces. | Requires saved UV/sketch artifacts. | Unavailable in Cycle 04 because UV/sketch artifacts were not saved. Do not claim principal-angle evidence without these artifacts. |

## Implementation References

- Metric functions: `/root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/geometry_metrics.py`
- Cycle 04 metric rows: `/root/LLM-output-density/local_experiment_results/cycle_04_opd_stability_gain/geometry_metrics.csv`
- Cycle 04 full singular spectra: `/root/autodl-tmp/cycle04_opd_stability_gain/tables/geometry_long.csv`

## Current Interpretation Rule

Use drift for pre-registered geometry stability gates, but always report effective rank, spectral gap, and X-S gap as auxiliary geometry evidence. If these auxiliary metrics diverge from the drift verdict, the report should state that the mechanism signal is mixed rather than discard the auxiliary metrics.

---

## Cycle 06 Interpretation Notes (SFT-Only Runs)

```yaml
cycle: cycle_06_sft_feasibility_and_degradation
date: 2026-06-23
qa_source: theory/QA_cycle06_theory.md (Q6=a)
```

### Diagnostic Status

For Cycle 06 SFT-only runs, all geometry metrics are **diagnostic signals only**.
Because no OPD arm is trained in this cycle, they cannot support OPD mechanism
claims. Do not use Cycle 06 geometry results as evidence that OPD is more or less
stable than SFT.

Permitted use: geometry may be used to check whether representation changes
correlate with SFT learning (Gate A06 pass scenario) or degradation (Gate B06
scenario). These are correlational signals, not mechanistic proof.

### lower effective rank: Interpretation Caveat

Lower effective rank after SFT is **not automatically a stability signal**. It can
also indicate:
- Harmful representational concentration or collapse
- Task specialization without generalization capacity

Always interpret effective rank together with the amount of ID learning achieved.
If effective rank decreases without corresponding ID gain on the primary axis,
treat it as a warning (possible harmful collapse), not as evidence of stability.

### drift Reference Point for Cycle 06

Cycles 04 and 05 used `theta0` as the drift reference point
(`spectral_drift_from_theta0_diag`). For Cycle 06:

- If training starts from the local base checkpoint or the official
  `Qwen3-1.7B-Base` rather than from `theta0`, `spectral_drift_from_theta0_diag`
  is no longer the appropriate primary drift metric.
- Experiment Design must specify the reference checkpoint before geometry runs.
- If both base-relative and theta0-relative drift are computed, label them
  clearly (`drift_from_base`, `drift_from_theta0`) and do not conflate them.
- Raw drift still penalizes models that learn more (P4 from K1-base-aware);
  always report drift together with ID learning amount.

---

## Cycle 07 Update: Trajectory Reading and Subspace-Stability Metrics

```yaml
cycle: cycle_07_base_model_sft_feasibility
date: 2026-06-24
qa_source: theory/QA_cycle07_theory.md (Q5)
governs: theory definitions and interpretation only; implementation (changing the
         k argument, computing stable rank / energy ratio, multi-checkpoint
         sampling) belongs to exp/experimental_methodology/geometry_protocol.md
```

### From Static Endpoint to Trajectory

Cycles 04/05 read geometry as a **static endpoint contrast** ("OPD lowers
effective rank, SFT raises it"). The Cycle 07 three-way framework (see
`current_theory_update.md`, Cycle 07 section) requires reading every geometry
metric **along the SFT trajectory at multiple checkpoints**. A single endpoint
checkpoint hides the dip transient, which is exactly the structure the three-way
framework is about. This applies to **all four metrics**, not only effective rank
and drift — `spectral_gap` (now read as γ_r, below) and `X_S_spectrum_level_gap`
are part of the trajectory reading too.

### Subspace Stability: From top-1 gap to the eigengap γ_r (Davis–Kahan)

The Cycle 04 use of `spectral_gap(sigma, 1) = σ_1 − σ_2` measures only how far the
**single** leading direction leads the second. It cannot judge the stability of a
**subspace**. The stability of the principal subspace is a property of the top-r
singular directions jointly, not of the top-1 vector.

**Davis–Kahan bound (the governing math):** for two matrices M and M+E, the largest
principal angle θ_r between their top-r singular subspaces satisfies

```
sin θ_r  ≤  ‖E‖ / (σ_r − σ_{r+1})
                   └──── eigengap at r, denoted γ_r ────┘
```

So the eigengap **γ_r = σ_r − σ_{r+1}** is the standard proxy for top-r subspace
stability: a larger γ_r means the top-r subspace is harder to rotate under a
training perturbation. This is the correct generalization of the top-1 gap, and it
is **the metric previously remembered in this project as "gamma"** (confirmed
2026-06-24; no `gamma` metric had actually been implemented — `gamma_s.jsonl` is the
OPD S-probe storage file, unrelated). Operationally, γ_r is just `spectral_gap`
called with `k = r` instead of `k = 1`.

**Choosing r:** r is the principal-subspace dimension, set adaptively per
(layer, module) — e.g. the dimension capturing ~90% spectral energy
(energy-ratio@r), or `round(effective_rank)` — not a single global k.

### New Auxiliary Metrics (computable from the existing spectrum, no UV needed)

| Metric | Formula | Role |
|---|---|---|
| `eigengap_gamma_r` | σ_r − σ_{r+1} (i.e. `spectral_gap(sigma, r)`) | **capacity proxy** for top-r subspace stability (Davis–Kahan denominator) |
| `stable_rank` | ‖A‖_F² / σ_1² = Σ σ_i² / σ_1² | continuous "effective dimension"; noise-robust; complements entropy-based `effective_rank` (no k choice needed) |
| `energy_ratio_at_r` | Σ_{i≤r} σ_i² / Σ σ_i² | fraction of spectral energy in the top-r subspace; also used to set r |

These use only the singular spectrum already collected, so they can be added
without re-running GetSlice or saving UV.

### Principal Angle: Deferred Beyond Cycle 07 (user decision 2026-06-24; superseded 2026-06-29)

The **direct, measured** subspace-stability quantity is the principal angle θ_r
itself (the Davis–Kahan left-hand side), which compares the actual top-r subspaces
of two checkpoints. It requires saved UV/sketch artifacts.

**Cycle 07 decision:** UV saving is **not turned on in Cycle 07**; principal angle
and the question of enabling UV were deferred beyond Cycle 07. **Cycle 08
superseding decision:** OPD-vs-SFT OverlapLift comparison proceeds as a
matched-LoRA relative trajectory comparison, while principal-angle /
subspace-rotation interpretation is registered for Cycle 09 rather than used in
Cycle 08. Consequence and boundary:

- γ_r is only a **capacity proxy** (large gap ⇒ subspace *can* resist rotation),
  computed from a single model's spectrum. It is **not** a measurement of how much
  the subspace actually rotated between checkpoints.
- Therefore Cycle 07 can describe subspace-stability *capacity* via γ_r, but cannot
  make a measured subspace-rotation claim. Do not write γ_r as if it were θ_r.

### Transient-Peak Hypothesis (to be tested by Cycle 07 multi-checkpoint data)

Under the three-way framework, the geometry signatures are predicted to be
**transient structures along the trajectory**, not endpoint levels:

- **SFT_dip:** possible transient rise in effective rank / drift-from-base, and
  collapse of γ_r (the principal subspace becomes ill-defined / easy to rotate) —
  the geometric signature of shallow imitation.
- **SFT_recover:** effective rank / drift settle, γ_r recovers — internalized.
- **(Cycle 08, H-OPD):** does OPD flatten this transient peak, i.e. keep γ_r from
  collapsing and effective rank from spiking, all along the trajectory?

This is the mechanistic answer to the long-standing open question "is lower
effective rank a stability signal or harmful collapse?" — it depends on the
optimization stage: a transient effective-rank spike with γ_r collapse in the dip
is harmful instability, not stability. **This is a hypothesis pending Cycle 07
multi-checkpoint data, not an established reading.**

### Reminder to Experiment Design

`geometry_metric_definitions.md` is theory (definitions + interpretation). The
following implementation items belong to `geometry_protocol.md` and must be
decided/pre-registered by Experiment Design:

1. Call `spectral_gap` with `k = r` (adaptive per layer/module), and implement
   `stable_rank` and `energy_ratio_at_r` in `geometry_metrics.py`.
2. Sample geometry **and `response_length`** at multiple checkpoints along each SFT
   trajectory (not just the endpoint).
3. Measure drift from the **base** checkpoint (Qwen3-4B-Base), not `theta0`.
4. UV saving stayed **off** in Cycle 07; Cycle 08 registers principal-angle /
   measured subspace-rotation interpretation for Cycle 09.


---

## Cycle 08 Update: OverlapLift Under Matched-LoRA Trajectories

```yaml
cycle: cycle_08_h_opd_vs_sft_comparison
date: 2026-06-29
qa_source: theory/QA_cycle08_theory.md (Q6, Q9, Q10)
governs: theory interpretation only; implementation and artifact-saving decisions
         belong to exp/code
```

### Relative OPD-vs-SFT Comparison Is Allowed

Cycle 08 compares OPD and SFT under the same model, data family, checkpoint grid,
and LoRA regime. Under this matched-LoRA design, OverlapLift trajectories can be
compared directly as a **relative OPD-vs-SFT geometry signal**. The comparison is:

```text
Does OPD's teacher-subspace overlap trajectory differ systematically from the
matched SFT trajectory, especially around the SFT dip and recovery landmarks?
```

This does not require a separate LoRA-aware random baseline before any comparison
can be made. LoRA is controlled by matching the adaptation regime across OPD and
SFT.

### Random-LoRA Baseline Caveat

A random-LoRA baseline may normalize update count, adapter footprint, or some
training-budget features, but it does not necessarily normalize the low-rank
structure induced by the actual training objective. Therefore, Cycle 07's
SFT-only OverlapLift < 1 and monotone increase should not be upgraded into an
absolute on/off-principal verdict by itself.

Cycle 08 resolves the relative question by comparing OPD against SFT under the
same LoRA regime. It still does not by itself settle an absolute claim such as
"this update is on-principal relative to the correct random/full-rank reference"
unless that reference is explicitly defined.

### Principal Angle Deferred to Cycle 09

Principal-angle or measured subspace-rotation claims require saved UV/sketch
artifacts and an explicit top-r definition. Cycle 08 theory does not make those
claims and does not require them for the OPD-vs-SFT OverlapLift comparison.

The open question is registered for Cycle 09: decide whether to save the required
artifacts and how to define the top-r subspaces before making direct
subspace-rotation claims.

### Gate Status

OverlapLift is mechanism-supporting only when aligned with behavior:

- If B08/C08 support OPD and OverlapLift differs favorably from SFT, A08
  strengthens the H-OPD story.
- If B08 fails badly, favorable OverlapLift cannot rescue the capability failure.
- If OverlapLift is mixed while B08 passes, report the mechanism signal as mixed
  rather than forcing an on/off-principal label.

> **Cycle 08 post-hoc note (2026-07-05, recorded for continuity):** OverlapLift was
> RETRACTED as an on/off-principal signal (indistinguishable from a scale-matched
> random-LoRA null; magnitude/bf16-driven) and superseded by the scale-invariant ρ
> (`result/cycle08_rho_metric_spec.md`). See the Cycle 09 section below for the
> measurement correction.

---

## Cycle 09 Update: Layer-Resolved Reading, Measured Rotation (θ_r), Dual-ER Constructs, Probe Battery v2.1

```yaml
cycle: cycle_09_aaai_competitiveness_completion
date: 2026-07-08
qa_source: theory/QA_cycle09_theory.md (Q4/Q5/Q6 + E1–E4)
governs: definitions and interpretation only; implementation belongs to exp/code
supersedes: "principal angle deferred" — UV saving is now ON (user decision D2, 2026-07-08)
```

### Layer-Resolved Reading Rule (D3 / Q4)

The primary geometry lens is now **layer-resolved**, not mean-over-layers:

1. **Headline layer** = the layer with the largest OPD-vs-SFT discriminant effect
   (currently L18). The selection criterion is pre-registered as exactly that —
   "largest discriminant effect" — and the **full-layer profile must always be
   reported alongside** (anti-cherry-pick). External anchor: 2605.30524 also finds
   collapse concentrated in mid-late layers, so a mid-layer headline has
   cross-paper consistency.
2. Mean-over-layers tables are demoted to appendix summaries. (The Cycle 08 D08
   discriminant — SFT L18 step_20 rank bump vs OPD no bump — is a layer-level
   finding that the mean table dilutes.)

### Dip Co-location Adjudication Procedure (Q4 — recorded contradiction)

**The contradiction (found 2026-07-08):** `result/current_picture.md` states "Both
dips co-locate with the ER uptick", but the mean-over-layers table
(`opd_vs_sft_geometry.md`) shows OPD effective_rank monotone-decreasing from
step_0 (argmax = step_0, no uptick); the OPD-side quantity nearest its step_5 dip
is the **xs_log_spectrum_gap argmin (step_10)**, not an ER uptick.

**Procedure (pre-registered):** re-derive both arms' dip co-location at the
headline layer with four quantities together (ER, xs_gap, drift, θ_r):

- If OPD's dip shows an L18-level ER uptick → keep the co-location wording,
  scoped to the headline layer.
- If it still does not → rewrite the mechanism story as **channel-heterogeneous
  dip geometry** (SFT: ER-bump type; OPD: xs-gap type — two geometrically
  different transients), and flag Result to correct `current_picture.md`.

Both branches are publishable; the second may be the more interesting one. Do not
average the two arms' dip signatures into one story before this adjudication.

### θ_r Activated: Measured Subspace Rotation (D2 / Q5 + E4)

UV/sketch saving is **ON** for all Cycle 09 probe runs (three arms × full
checkpoint grid) — probe-time cost only, **no retraining required**.

- **Definition:** θ_r = principal angles between top-r activation singular
  subspaces, computed (i) between consecutive checkpoints and (ii) each
  checkpoint vs base, per layer × module.
- **r-grid (E4):** r ∈ {8, 16, 32, 64, 128} ∪ {energy@80%, 90%, 95%}. Primary
  r = energy@90% (headline); the full θ_r(r) robustness curve goes to the
  appendix. This mirrors the ρ two-k reporting logic and blocks the
  "you picked r" attack.
- **Uncertainty (E4):** bootstrap over probe samples → confidence band on θ_r;
  plus a one-time probe-count sensitivity sweep (e.g. n_samples ∈ {8,16,32,64})
  — the user's earlier ablation suggests insensitivity; confirm once, fix n,
  report as robustness appendix.
- **Prediction (pre-registered):** SFT shows a headline-layer θ_r rotation spike
  near steps 10–20 co-locating with its ER bump; OPD stays smooth.
- **Adjudication tree (answers 2509.12235):** compare ER vs θ_r as trackers of
  (i) arm separation and (ii) MMLU-Pro Δ. θ_r better → adopt the
  rotation+values two-axis observable framework and cite 2509.12235 as
  cross-space convergent (theirs is weight-space). ER better → empirical reply:
  in whitened activation space, value-distribution transients, not subspace
  rotation, discriminate the channels. Split roles → observable
  division-of-labor framework.
- **γ_r vs θ_r:** γ_r (eigengap) remains the *capacity* proxy; θ_r is the
  *measured* rotation. Report both; never write γ_r as if it were θ_r.

### Dual-ER Construct Specification (Q6)

Two ER constructs are computed side by side on the same checkpoints and probes.
Any "compression" statement must name which construct it uses.

| | This project (whitened) | 2605.30524 (raw) |
|---|---|---|
| Input | GetSlice whitened activation-conditioned module matrices (Cholesky-jitter covariance whitening) | Raw residual-stream hidden states, centered only |
| Spectrum weights | p_i ∝ σ_i | p_i ∝ λ_i = σ_i² |
| Normalization | none | × 1/d (range [0,1]) |
| Granularity | per module (q/k/v/o/gate/up/down) × layer | whole-layer residual stream |
| Probe | probe battery (below), incl. per-arm S | fixed multi-domain probe, teacher-forced |
| Meaning | **output-relevant spectrum** — whitening makes σ_i correspond to output-error-optimal truncation (SVD-LLM, arXiv 2403.07378 — formal methodological citation) | representational spread |

Pre-registered three-outcome tree for the comparison lives in
`current_theory_update.md` (Cycle 09 section).

### Probe Battery v2.1 (E3)

All X probes frozen, versioned, shared across arms and checkpoints; probe-time
cost only.

| Probe | Domain | Status | Serves |
|---|---|---|---|
| `X_math` (MATH500-style) | ID | existing | main trajectory |
| `X_math_hard` (AIME-style; use historical AIME to avoid eval-set coupling) | ID-hard | **new** | difficulty-stratified ID; **generation-free, so it bypasses AIME24's non-termination failure** and gives the hard-math regime an observable channel |
| `X_bos` | unconditioned | existing | reference |
| `X_ood_knowledge` (MMLU-Pro-style) | OOD knowledge | **new, top priority** | makes compression→OOD a same-domain observation instead of a cross-domain inference |
| `X_teacher` (fixed 8B-teacher response set) | supervision source | **new** | one shared supervision-source ruler across all three arms; teacher-alignment dynamics for the off-KD control |
| `X_general` (open-domain text) | general | **new** | construct comparability with 2605.30524's multi-domain probe |
| `S` (per-arm, unchanged; off-KD arm's S = its teacher static training responses) | training target | existing | xs_gap |

If budget-constrained: X_ood_knowledge > X_teacher > X_general ≈ X_math_hard.

### Weight-Side Measurement Correction (E1 / E2 — pointer)

ρ (scale-invariant weight-direction alignment) is specified in
`result/cycle08_rho_metric_spec.md`. Cycle 09 correction, recorded here because it
changes how the metric must be computed:

- **Dual track (E1):** bf16 = end-state (what the deployed artifact carries,
  TPNT-conform); fp32 = process (what training geometrically did,
  attribution-clean). Report both; their gap quantifies bf16's distortion of
  directional metrics.
- **Measurement fix (E2):** compute ΔW = **B@A directly from saved LoRA
  adapters** (fp32 arithmetic; bf16-stored adapters acceptable — rounding is
  relative to adapter-entry magnitude, ~1e-3 of the update) instead of
  `bf16(W0+BA) − bf16(W0)` (rounding relative to W0, ~100× the update).
- **Numerical-rank evidence:** measured ΔW rank ≈ 2121 vs true LoRA rank ≤ 32 —
  the merge−subtract object is rank-structurally dominated by quantization
  noise; ρ² ≈ 0.02 "diffuseness" is the noise floor's signature, not LoRA's.
- **A09 order:** adapters-first (zero training cost); retrain only if adapters
  are unavailable or the training-level bf16 effect must itself be excluded.

### Reminder to Experiment Design (Cycle 09)

1. Turn UV/sketch saving ON for all probe runs; budget the storage.
2. Build probe battery v2.1 (three arms × grid); version and freeze it.
3. Implement θ_r with the r-grid + probe bootstrap + one probe-count sweep.
4. Compute both ER constructs on the same probe outputs.
5. A09: locate Cycle 08 adapter files first; fp32(B@A) before any retrain.
6. Required artifacts: full-layer profiles (not just L18), θ_r(r) curves,
   dual-ER tables, per-arm dip co-location tables at the headline layer.

### Mini-Round 1 Measurement Rulings (2026-07-08 — provisional pending Result consolidation)

Source: `local_experiment_results/cycle_09.../run_01/mini/` (T1/T4). These change how
two metrics must be computed and read:

1. **ρ estimator rule:** the adapter product **BA is the required estimator** of the
   LoRA update direction. The merge−subtract object `bf16(W0+BA) − bf16(W0)` flips
   early-step directional verdicts (SFT step_5: off z=−6.7 merged vs on z=+7.9 clean
   BA) and inflates numerical rank (≈2121 vs true ≤32). bf16 rounding **of the
   adapters** is harmless (`sft_bf16_BA` ≈ `sft_fp32_BA` in ρ² and verdicts) — the
   corruptor is the merge−subtract pipeline at small update magnitude. Dual-track
   reporting stands: BA = process/claim track; merged = deployment/ecological track.
2. **E2 correction (recorded):** ρ²_U(k=32) ≈ 0.022–0.026 on the clean rank-32 BA —
   diffuseness relative to the base top-k subspace is a genuine LoRA-regime property
   with a small robust on-principal bias (z > 5), not a bf16 noise floor.
3. **OPD clean-BA gap:** the OPD adapter was pruned; `opd_top32_approx` (top-32 SVD
   of merged−base) is an APPROXIMATION — no OPD process-track direction claim until
   A09 retrain provides a true adapter.
4. **Layer-scoping rule (user-decided 2026-07-08; supersedes the single-headline
   formulation above):** mean-over-layers ER bump/no-bump statements are
   **disallowed** — the mean hid dip-co-located upticks present in BOTH arms at
   L9/L18 (OPD +1.3, SFT +3.6 ER at L18) while L27 declines monotonically.
   Headline layer = **L18**, selected by the pre-registered criterion **"clearest
   dip-transient (uptick↔dip co-location + visible recovery)"** — NOT by largest
   endpoint discriminant (which picks L27 and misses the transient). L9 =
   confirmatory; L27 endpoint divergence = supporting evidence in the
   compression→OOD story, not a second headline. The full-36-layer profile
   (round 2) stays mandatory (anti-cherry-pick + checks for an even clearer
   transient layer; the criterion, not the layer index, is pre-registered), and
   ER upticks need probe-bootstrap error bands before being written as established.
5. **CI practice:** paired-bootstrap CIs are stable across draws {256, 1024, 4096};
   small draw counts suffice — report the stability check in the appendix.

### S/X/H Probe Semantics (Round 3 — new axis, user-proposed 2026-07-09, formalized 2026-07-10)

A **new probe taxonomy** that encodes the supervision channel into the probe
semantics. It **coexists with** (does not replace) the legacy S/X design: legacy
probes remain the basis for all existing adjudications (L18 uptick, θ_r, dual-ER);
S/X/H opens a new observation axis. **Never mix ER values across the two
taxonomies — different conditioning, separate tables.**

| Role | Definition | OPD arm | SFT arm | Domains |
|---|---|---|---|---|
| **S** (source) | base (step_0) generations — the initial output distribution, shared across arms | base-generated | same (shared) | every probe domain |
| **X** (training signal) | the *actual* supervision text per arm | the checkpoint's **own on-policy rollouts** (regenerated per checkpoint) | the **dataset CoT** (fixed external text; = legacy S for SFT) | training domain (math) only |
| **H** (half: generated-but-not-trained) | the checkpoint's self-generated text that was NOT trained on | non-training domains (MMLU-Pro-style, general, …) | math self-generations AND non-training domains (SFT trains on the dataset, not its own outputs) | per domain |

Design payoffs: (i) on-policy vs off-policy supervision is now *encoded in the
probes* — OPD's X is self-generated, SFT's X is external-fixed, both arms' H is
self-generated-untrained; (ii) it repairs the legacy asymmetry caveat (legacy S is
off-target for OPD, whose true training signal is its rollouts).

**Pre-registered interpretation rules:**
1. Dip / OOD-drop readings must be reported **jointly across S-, X-, and
   H-conditioned spectra** (user requirement): e.g., does the L18 transient appear
   under the initial distribution (S), only under the training signal (X), or in
   generated-untrained text (H)?
2. This round the axis is **diagnostic only** — no mechanism claim from S/X/H alone.
   Two pre-registered hypotheses it can inform: (a) OPD's X-conditioned spectrum
   tracks its rollout-distribution shift along training; (b) H-conditioned spectra
   in non-trained domains make the "compression evicts untrained directions"
   hypothesis observable same-domain.
3. Generation parameters (sampling, length caps, n per domain) must be frozen,
   versioned, and recorded in the manifest; X_OPD and H require **per-checkpoint
   generation** — provenance is mandatory.
4. Comparability guard: S/X/H requires its own baseline column (step_0 = S by
   construction); trajectory deltas are computed within-taxonomy only.

### Output-Precision Metrics + Discriminability Standard (Round-3 additions, 2026-07-11)

Motivated by the Q1 gap (ER is a shape summary — it cannot distinguish "pruned
task-needed directions" from "consolidated the same function into fewer
directions") and by the user's SVD-LLM-native proposal. Three metric additions:

1. **Domain-conditioned tail energy / functional rank.**
   `tail_energy(r) = Σ_{i>r} σ_i² / Σ σ_i²` = 1 − `energy_ratio_at_r` (Cycle 07
   definition, now read domain-conditioned); under the SVD-LLM whitening this IS
   the output-precision loss of rank-r compression on that domain.
   `functional_rank r_ε = min r s.t. tail_energy(r) ≤ ε` — how many directions the
   function actually needs for that domain, at output-error tolerance ε.
2. **Module output drift** (direction-aware, supersedes the direction-blindness of
   log-spectrum drift): `‖ΔW·X‖_F / ‖W₀·X‖_F` per (module, layer, probe domain) —
   the module's actual output change on that domain.
   **Zero re-probe cost:** `‖ΔW·X‖_F² = tr(ΔWᵀ ΔW Σ_X)` with Σ_X built from the
   per-probe input factors saved by the R3-4 campaign, and ΔW = BA in hand.
3. **Output-space decomposition** ("activation-side ρ", optional): fraction of
   `‖ΔW·X‖²` energy landing inside base's top-r output directions vs the tail —
   the direction-resolved test of whether a contraction touched the directions a
   domain needs.

**Pre-registered predictions (before R3-8 runs):**
- P-tail-1: SFT shows a transient tail_energy / r_ε spike on X_ood at step_20
  (co-located with its ER uptick); OPD's r_ε(X_ood) declines smoothly.
- P-tail-2 (eviction test): if OPD's OOD-conditioned contraction were evicting
  OOD-needed directions, its output drift restricted to base's top-r_ε(X_ood)
  subspace should be disproportionately large; the benign-consolidation reading
  predicts OPD's top-subspace-restricted drift stays comparable to SFT's despite
  larger total drift.

**Discriminability standard (user ruling 2026-07-11, Q4):**
"Space A discriminates the arms better than space B" may NOT be claimed from a
presence/absence pattern or from raw magnitudes across different-scale constructs
(e.g., whitened −15.8 vs raw −0.11 is scale-incomparable). Required test:

```text
On the SAME probe samples (whitened and raw come from the same forward passes),
compute per space a dimensionless discriminability statistic
   d_space = |mean(OPD) − mean(SFT)| / pooled bootstrap SD,
then bootstrap the PAIRED gap d_whitened − d_raw over probe samples;
the superiority claim requires the gap's 95% CI to exclude 0.
```

Until this passes: (i) whitened-vs-raw statements stay at "construct-specific
signal" (T8 wording); (ii) the weight-vs-activation θ contrast is split into a
**magnitude contrast** (17–23° vs ≤2–5° incl. the 1.2° identity floor —
descriptive, established) and a **discriminability contrast** (requires error
bands on both θ's — pending). Applies to all "observation-space superiority"
language in the paper.

---

## Round-4 Spec (2026-07-12): Probe Nomenclature S/E/X/H, Windowing v2, Metric Suite Final Form

Settled in the 2026-07-11/12 design discussion (user rulings). **Supersedes v1
windowing and the Round-3 S/X/H instantiation for all paper-grade measurements.**

### Probe nomenclature (final): four types

| Type | Definition | Legacy mapping |
|---|---|---|
| **S_d** | base-generated text in domain d — frozen, shared across arms | S_math = legacy `X_math`/X_base (base rollouts!); S_bos = legacy `X_bos`; S_ood / S_general = base generations on the E_d prompt pools |
| **E_d** | external fixed corpus text (no generation part) | E_ood = MMLU-Pro question+options (14 subjects, seed42, 128); E_general = wikitext slices (128); E_math_hard = AIME statements; E_math (textbook math text) = optional, deferred |
| **X_arm** | the actual training signal | X_OPD = per-checkpoint own math rollouts; X_SFT = dataset CoT (question masked in v2) |
| **H_d** | per-checkpoint self-generation, **non-training domains only, both arms symmetric** | H_ood, H_general, **H_bos (new)**; SFT math self-gen REMOVED from H (user ruling) |

**Cross-domain magnitude caveat:** contraction orderings across domains (e.g.,
E_ood −21.3 vs S_math −17.3) are confounded by text nature (generated CoT vs
question text). Load-bearing comparisons are **within-probe**: arm contrasts and
trajectories on the same probe.

### Windowing v2 (the length-decoupled design)

1. Window = 512 tokens, placed **entirely inside the generation region** (prompt
   stays in the forward context, is excluded from statistics). E-type text has no
   generation split — whole text eligible.
2. **Fixed k = 3 uniform-random window offsets per sample** (overlap allowed when
   generation < 3×512; record actual k). Window seeds separate from generation
   seeds. Relative position recorded per window → early/mid/late 3-bin
   diagnostics for free.
3. Generation-based probes (S / X_OPD / H): **≥3 generation seed batches**;
   report mean ± SD across batches.
4. **Hierarchical normalization (the rule that decouples weight from length):**
   window-level token-mean gram `(1/n_w)XᵀX` → sample-level mean over its windows
   → corpus-level **equal-weight mean over samples**. Estimand = the
   sample-equal processing profile, deliberately orthogonal to `response_length`
   (tracked as its own pre-registered variable). k affects estimator *precision*
   only, never *weight*.
5. Whitening matrices are built from the hierarchical covariance (constant
   rescaling of Σ leaves ER/tail/EC ratios invariant).
6. **Bootstrap unit = sample** (windows nested within) — upgrades the Round-3
   `probe_window` unit.

### Evidence-tier rule (user ruling)

v2 = the primary measurement standard. v1 generation-probes (Round-3 S/X/H):
superseded. Fixed-text results — including the L18 uptick sample bands — become
**pilot tier**: internally valid (identical fixed text across arms/checkpoints)
with scope restated as "question + answer-opening windows"; paper-grade status
requires **v2 random-window re-derivation**. Behavioral evals, ρ, θ_w, and the
capability CIs are windowing-independent and keep their tier. A **window
composition audit** (prompt-token share of v1 first-windows per probe cell) is a
mandatory CPU task before the v2 campaign.

### Metric suite — final form

**M1: tail_energy / r_ε + effective-compression (EC) criterion.**
Per (module, layer, arm, step, probe): `tail(r) = Σ_{i>r}σ²/Σσ²`; r_ε at
ε ∈ {0.05, 0.01}. Whitening tracks: **per-checkpoint** (primary — the current
function's compression budget, SVD-LLM-faithful) and **frozen-base** (secondary —
pure weight-side view; the difference quantifies input-covariance drift).
User-proposed EC criterion:

```text
EC (vs base, same probe) :=  r_ε(t) < r_ε(0)                (redundancy shrinks)
                         AND drift_core small,               (core stays stable)
     drift_core = RMS[ log σ̂_i(t) − log σ̂_i(0) ],  i ≤ r_ε,  σ̂ = σ/Σσ
```

Singular-value proxy only — what was damaged is checked by M3. Pre-registered:
OPD endpoint satisfies EC on all domains; SFT dip steps transiently violate core
stability (co-located with the uptick); SFT endpoint near-neutral.

**M2: module output drift.** `‖ΔW·X‖_F / ‖W₀·X‖_F`, **raw** (unwhitened) inputs;
X₀ (base factors) primary reference, X_t secondary; computed as
`tr(ΔWᵀΔW·Σ_X)` from saved factors + BA (OPD = top-32 approx, caveat inline).
Module-local by construction; end-to-end complement = **M2b representation
drift** `‖h_t − h₀‖/‖h₀‖` on the same text.

**M3: retention & aim — BOTH sides (U output / V input).** From the base
whitened matrix M₀ = W₀S₀ = U₀Σ₀V₀ᵀ per probe domain; r linked to M1's r_ε.

| Side | Retention | Aim | Question |
|---|---|---|---|
| U (output) | e_keep_U(r) = ‖U₀ᵣᵀW_tX₀‖²/‖U₀ᵣᵀW₀X₀‖² | φ_U = ‖U₀ᵣᵀΔWX₀‖²/‖ΔWX₀‖² | still *writing* the outputs downstream needs? |
| V (input) | e_keep_V(r) = ‖W_tS₀V₀ᵣ‖²/‖W₀S₀V₀ᵣ‖² | φ_V = ‖ΔWS₀V₀ᵣ‖²/‖ΔWS₀‖² | still *reading* the domain's principal input directions? |

Joint reading: φ high + e_keep ≈ 1 → refinement inside needed directions;
φ high + e_keep ↓ → overwrite; e_keep ↓ → eviction (V-side deafness is the more
likely forgetting mechanism; ties to 2509.12235's V-rotation and our θ_V). This
family lives entirely in **frozen-base geometry** — the structural reason the
frozen-basis secondary track exists. Pre-registered (P-tail-2 final form):
benign consolidation → OPD e_keep_U/V(E_ood) ≈ 1 despite ER contraction;
eviction → e_keep drops.
