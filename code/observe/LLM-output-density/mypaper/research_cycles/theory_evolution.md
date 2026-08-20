# Theory Evolution (Cycle 03 → 05)

```yaml
view_type: cross_cycle_evolution_log
section: theory
status: active
maintenance_rule: append_one_dated_delta_entry_per_cycle_do_not_restate_full_text
last_organized: 2026-06-16
```

This file tracks **what changed and why**, cycle to cycle, in the project's
theory. It does not restate each cycle's full theory document. The current,
stable theory and metric definitions live in `../theory/current_theory_update.md`
and `../theory/geometry_metric_definitions.md`.

Note: cycle snapshot folders have been removed (2026-06-16). The "Full source"
lines below are kept as historical provenance even though those files no longer exist.

## Cycle 03 — Starting Point

Full source: `cycle_03_trl_opd_like/02_theory_design/` (snapshot removed;
content summarized in this entry).

Cycle 03 defined the project's core theoretical object: the
activation-conditioned operator `A_t = W_t L_t` (weight matrix observed
through an activation-induced whitening transform `L_t`, rather than raw
weight space). `my_existing_method.md` is the user's own original informal
derivation of this same idea — the direct predecessor of what is now
`geometry_metric_definitions.md`. The primary metric was **spectral gap
stability**; auxiliary metrics were spectral drift, effective rank, X-S
geometry gap, and (optionally) CKA. The fairness anchor was **matched
MATH500 ID gain**, with an explicit `SFT-small-LR` control required to rule
out "OPD just updates less." The headline hypothesis (K1):

> At matched MATH500 gain, OPD preserves General OOD better than continued
> SFT because it reaches the same ID improvement with less disruption to
> activation-conditioned principal geometry.

**Design-vs-execution gap already present at Cycle 03**: the actual minimal
TRL run (`RESULTS_03_v2.md`) matched on **GSM8K** gain, not MATH500, and
measured geometry at **layer 14 only**, not the full multi-probe design this
theory document specifies. This gap was not an error so much as a deliberate
feasibility-first pilot — but it foreshadows the measurement issues found
later (see `result_interpretation_evolution.md` Cycle 04/05 entries).

## Cycle 03 → Cycle 04: No Theory Rewrite, A Result-Driven Reading Shift

Current working-face: `../theory/current_theory_update.md` (snapshot removed).

The theory's substance did not change in Cycle 04 — K1 and the
matched-gain-anchor framing were kept. What changed was the *result*, which
forced a reinterpretation: the matched-gain anchor turned out to be
confounded by `theta0` saturation (a less-saturated `theta0` forces the
matched SFT control toward an almost-untrained checkpoint, which is trivially
"stable"). This was discovered empirically, not predicted by the Cycle 03
theory document. Cycle 04 also produced the `MATH500=0` extraction-artifact
finding — an evaluation-validity finding, not a theory revision, but one that
directly motivated introducing NuminaMath-test as a cleaner ID axis going
into Cycle 05.

## Cycle 04 → Cycle 05: Formal Revision To "K1-Base-Aware"

Current working-face: `../theory/current_theory_update.md` (snapshot removed).

This is the project's first **explicit, pre-registered** theory revision in
response to evidence (Cycle 04's shift was an interpretive reframing; this
one rewrites the hypothesis). New variables made explicit: `base_performance`
(headroom gate), `theta0_saturation`, `id_performance` vs `ood_retention`
(previously conflated under "GSM8K"), `training_budget`, `geometry_movement`
(now read jointly with ID learning rather than as raw drift). Revised
hypothesis:

> K1-base-aware: OPD-like training can only be meaningfully compared with
> SFT after the selected base/task/data setting shows measurable headroom
> above the untrained base. Conditional on that headroom, OPD-like training
> is more stable only if it preserves OOD behavior better at comparable ID
> learning or comparable training budget, not merely at comparable GSM8K
> gain.

Why: Cycle 04's base-model addition showed the untrained base tied `theta0`
and beat OPD on ID — suggesting no headroom existed to test improvement at
all. Falsifiers were also formalized (see source) rather than left implicit
as in Cycle 03/04. This revision's empirical premise (no headroom) was
itself later partially reversed within Cycle 05 by a measurement-protocol
fix — see `result_interpretation_evolution.md`'s Cycle 05 entry; the
**theory document itself has not yet been re-edited** to reflect that
reversal, which is an open item for whoever next updates `theory/`.
