# Next-Cycle Seed Evolution (Cycle 03 → 05)

```yaml
view_type: cross_cycle_evolution_log
section: next_cycle_seed
status: active
maintenance_rule: append_one_dated_delta_entry_per_cycle_do_not_restate_full_text
last_organized: 2026-06-16
```

This file tracks, cycle to cycle, what each cycle asked for next and whether
the following cycle actually acted on it. It does not restate each seed
document in full. The current, stable next-cycle entry point lives in `../next_cycle/`.

Note: cycle snapshot folders have been removed (2026-06-16). The "Full source"
lines below are kept as historical provenance even though those files no longer exist.

## Cycle 03 → Cycle 04 Seed

Full source: `cycle_03_trl_opd_like/07_next_cycle_seed/` (snapshot removed;
content summarized in this entry).

Asked for: increase/remove `eval_limit`, add more OPD settings/seeds, use a
less saturated `theta0`, expand geometry beyond layer 14.

**Acted on:** all four, fully, in Cycle 04's design (see
`exp_design_evolution.md`).

## Cycle 04 → Cycle 05 Seed

Full source: `cycle_04_opd_stability_gain/07_next_cycle_seed/` (snapshot removed;
content summarized in this entry).

Asked for: a base-ceiling audit first; only then test whether the OPD
stability signal survives under equal-budget, equal-ID, and a `theta0`
saturation sweep, with matched-GSM8K kept only as a diagnostic; mandatory
base row in every ID/OOD table; confidence/bootstrap intervals; saved
UV/sketch artifacts for future principal-angle analysis; extraction-floored
tasks explicitly flagged.

**Acted on:**

- Base-ceiling audit: done (Gate B), and its first reading was later
  corrected by the chat-template fix (see `result_interpretation_evolution.md`).
- ID/OOD axis separation via NuminaMath-test: done (Gate C design).
- Equal-ID / equal-budget matching: **not yet executed** — deprioritized
  after the (since-revised) Phase 1 reading that the setting was
  headroom-limited, per the Decision Log in
  `../exp/current_experiment_design.md`. Now that headroom has been
  re-established, these regimes are back in scope but still pending.
- UV/sketch artifacts: **still not saved** — open item, carried forward in
  `../exp/experimental_methodology/geometry_protocol.md`.
- Extraction-floored task flagging: done and then some — this practice
  directly led to catching the GSM8K chat-template bug.
- Confidence/bootstrap intervals: partially done (binomial stderr reported
  in the FINDING_05 documents for the 4 re-evaluated models; not yet
  extended to the remaining 4).

## Cycle 05 → Cycle 06 Seed

**Not yet written.** Cycle 05 is not closed. The immediate next step
(re-evaluate `opd_lmbda1`, `sft_n256/512/1024` under the corrected protocol,
then redo Gates B–F on the full 8-model set) is itself still inside Cycle 05.
Writing a genuine Cycle 06 seed is deferred until Cycle 05 concludes.
