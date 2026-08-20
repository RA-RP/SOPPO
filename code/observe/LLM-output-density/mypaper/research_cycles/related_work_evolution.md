# Related Work Evolution (Cycle 03 → 05)

```yaml
view_type: cross_cycle_evolution_log
section: related_work
status: active
maintenance_rule: append_one_dated_delta_entry_per_cycle_do_not_restate_full_text
last_organized: 2026-06-16
```

This file tracks **what changed and why**, cycle to cycle, in the project's
literature base. It does not restate each cycle's full literature review.
The current, stable cross-cycle literature positioning lives in
`../related_work/current_source_matrix.md`.

Note: cycle snapshot folders have been removed (2026-06-16). The "Full source"
lines below are kept as historical provenance even though those files no longer
exist. `source_notes/ood_id_positioning_with_ppl.md` has been moved to
`../related_work/source_notes/ood_id_positioning_with_ppl.md`.

## Cycle 03 — Starting Point

Full source: `cycle_03_trl_opd_like/01_related_work/` (snapshot removed).
`source_notes/ood_id_positioning_with_ppl.md` now at `../related_work/source_notes/`.

Cycle 03 established the broadest literature base the project has had,
organized around three strands that are still the project's positioning today
(see `../related_work/current_source_matrix.md`): OPD mechanism papers (GKD,
MiniLLM, OPD survey, Lightning-OPD, NPD, On-Policy Context Distillation),
SFT-degradation/conditionality papers (SFT Doesn't Always Hurt, Rethinking
Generalization in Reasoning SFT), and geometry/spectral mechanism papers (The
Path Not Taken, SVD-LLM/ASVD/FWSVD, LoRA vs Full Fine-tuning). It also carried
a fourth, separate strand that is **not** part of the current OPD-vs-SFT main
line: a PPL-plus-custom-geometry-score data-selection design for synthetic
math SFT data (`source_notes/ood_id_positioning_with_ppl.md`). This is the
user's own earlier work on whether a geometry-based score can select SFT data
that mitigates OOD degradation — it is retained as legitimate ID/OOD-relevant
related work and a candidate future extension, not discarded, even though it
has not yet been integrated into the active K1/OPD-vs-SFT thread.

## Cycle 03 → Cycle 04: Narrowed To The Executable Core

Current working-face: `../related_work/current_related_work_update.md`,
`../related_work/current_source_matrix.md` (snapshot removed).

What changed: the broad Cycle 03 matrix (dozens of papers across must-cite,
discussion, and appendix tiers) was narrowed to a focused 7-source list
directly supporting the matched-gain OPD-vs-SFT-with-geometry design that
Cycle 04 actually ran. Why: Cycle 04 was a feasibility-focused, stricter
re-run of the same K1 question, not a literature-expansion cycle — the wider
Cycle 03 discussion-only literature (data selection, calibration, loss
landscape alternatives) was set aside as out of scope for this cycle, not
judged wrong. Cycle 04 also added a "Local Evidence From Cycle 03" table that
explicitly carries forward Cycle 03's empirical findings (matched pair gap,
OOD-lite/drift comparison, saturated-`theta0` warning) as constraints on
Cycle 04's design.

## Cycle 04 → Cycle 05: No New Search, New Question Inventory

Current working-face: `../next_cycle/current_open_questions.md` (snapshot removed).

What changed: Cycle 05 did not run a new literature search. Instead, Cycle
04's results generated a structured question inventory (base-ceiling
construction, control construction, ID/OOD measurement, geometry/mechanism,
paper positioning) that names what a *future* related-work pass should look
for — e.g. prior work on reporting base-model baselines before fine-tuning
comparisons, and on defining "matched" controls when the baseline is already
partially saturated. As of this writing, that follow-up search has not been
run; it remains an open item for whichever cycle resumes related-work work.
