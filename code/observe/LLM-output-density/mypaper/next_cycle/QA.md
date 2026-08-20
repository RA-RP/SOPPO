# QA — Next Cycle Seed · Cycle 09

```yaml
phase: next_cycle
cycle: cycle_09
status: answered_internalized
created: 2026-07-07
maintenance_rule: overwrite_each_cycle; do_not_append
internalize_to: next_cycle/open_questions.md, next_cycle/history_questions.md, next_cycle/current_cycle09_intent.md
```

**Purpose:** Cycle 08 is now complete. This QA closes Cycle 08 and finalizes the
Cycle 09 work scope. Please answer in `QA-ch.md` (inline, Chinese is fine).

Historical QA content lives in `history_questions.md`.

---

## Background (from Cycle 08 results — 2026-07-07)

Cycle 08 trained an OPD arm (Qwen3-4B-Base ← Qwen3-8B teacher, verl on-policy distillation,
LoRA r=32) and compared it to the Cycle 07 SFT reference trajectory.

**Gate verdicts (formal, using acceptance_criteria.md labels):**

| Gate | Result |
|---|---|
| B08 (math capability + dip avoidance) | **STRONG PASS**: OPD final 0.848 vs SFT 0.752 (+0.096); OPD beats SFT at every step (+0.10 to +0.24) at unified cap 16384. |
| C08 (MMLU-Pro preservation) | **PASS**: OPD net +0.016 vs SFT net −0.029 (delta margin +0.045). Trajectory has mid-run dip (step_40/80) then recovery. |
| D08 (geometry transient) | **POSITIVE**: SFT shows L18 step_20 rank bump; OPD ER contracts monotonically with no bump. ER is the only metric that cleanly separates the arms and tracks the OOD outcome. |
| A08 (weight-space direction) | **INCONCLUSIVE**: OverlapLift retracted (≈ random-LoRA null, magnitude/bf16-dominated). ρ (scale-invariant) reads OPD on-principal / SFT off-principal at steps 5–10; external full-param lit (arXiv 2606.13657/07082) says OPD is OFF-principal. Contradiction flagged; bf16+LoRA confound suspected. |

**OPD early dip (new finding):** OPD dips at step_5 (MATH500 0.552, numina 0.418) — earlier and
deeper than SFT's dip at step_20. Both dips co-locate with ER uptick. The OPD dip recovers
by step_20; SFT recovers by step_160. This was not pre-registered in the Cycle 09 draft.

**Draft Cycle 09 intent exists** (`current_cycle09_intent.md`, status: draft_seed_pending_cycle08_results,
written 2026-07-03). It proposes: P0 (stats CI + geometry + output-control), P1 (OPD-specific
control + robustness + teacher baselines), P2 (theory + paper rewrite). Needs activation now
that results are in.

---

## Questions

### Q1 — Cycle 09 scope: does the pre-drafted intent hold given actual results?

The draft Cycle 09 intent was written before Cycle 08 ran. Now that B08 STRONG PASS + C08 PASS
are confirmed and D08 provides a positive geometry discriminant, is the planned scope still correct?

Pre-registered P0 workstreams:
- Paired bootstrap CI on all OPD-vs-SFT deltas (P0 statistical reliability)
- OPD geometry mechanism package — same protocol as Cycle 07 SFT, paper-level comparisons (P0)
- Output-control analysis: as-run acc / finisher acc / response length / non-term rate decomposition (P0)

Pre-registered P1 workstreams:
- OPD-specific control: off-policy KD or teacher-forced distillation (P1)
- Robustness axis: second data / prompt distribution (P1)
- Teacher / theta0 / headroom baselines (P1)

P2: theory update + paper rewrite.

Options:
- **(a) Proceed with the draft as-is.** B08/C08 confirmation means the paper scope is correct;
  start P0 immediately, P1 in parallel where feasible.
- **(b) Narrow P1.** B08/C08 are already so strong that the OPD-specific control and/or robustness
  axis can be deferred; focus Cycle 09 on P0 (CI + geometry + output-control) and P2.
- **(c) Expand.** The ρ contradiction with external lit is important enough to add a P0 gate for
  resolving weight-space direction (fp32 retrain or alternative).
- (d) Other.

---

### Q2 — ρ vs external literature: what to do about the weight-space direction contradiction?

Our ρ metric (scale-invariant directional alignment, computed on bf16 LoRA adapters) reads:
- OPD on-principal throughout (z +5.7 to +7.3)
- SFT off-principal at early steps (z −6.7 at step_5), then flips on by step_20

External full-param literature (arXiv 2606.13657, 2606.07082, June 2026) says OPD is
**OFF-principal** (relaxed regime). Our LoRA reading has the **opposite sign**.

Possible explanations: LoRA-structure confound, bf16 bias, regime difference (full-param vs LoRA).
The current_picture.md and claims_allowed.md flag this as "not a valid verdict — deferred to
full-param / fp32."

Options:
- **(a) Run full-param or fp32-master retrain** to resolve the measurement ambiguity. Expensive,
  but would give an authoritative reading comparable to the external literature.
- **(b) Accept ρ as a LoRA-specific measurement; treat it as a genuine LoRA-regime finding that
  may not extrapolate to full-param.** Cite external lit for the full-param regime, frame the
  LoRA difference honestly. A08 produces a real-but-limited finding.
- **(c) Drop weight-space principal geometry.** Rely on activation-space ER as the sole mechanism
  metric. Acknowledge weight-space direction is inconclusive, cite external lit.
- (d) Other.

---

### Q3 — OPD early dip (step_5) as mechanism evidence: major story or secondary?

OPD dips at step_5 (MATH500 0.552 ← base 0.652, −0.100; numina −0.070), deeper and earlier than
SFT's dip at step_20. OPD recovers by step_20; SFT doesn't recover until step_160. Both dips
co-locate with ER uptick.

The "compression transient" hypothesis: OPD forces a sharper early compression break (sudden
ER uptick at step_5) that resolves quickly because on-policy distillation re-anchors the
trajectory. SFT forces a shallower break (ER uptick at step_20) that takes longer to recover
because it is misaligned with the model's generation distribution.

This finding was not pre-registered in the Cycle 09 draft. The question is whether it deserves
a central role in the mechanism narrative.

Options:
- **(a) Make it a central mechanism story.** The sharper-but-shorter OPD transient vs the
  shallower-but-longer SFT transient is the key evidence for on-policy trajectory alignment
  as the causal mechanism. Add this as a primary Section 4 claim in Cycle 09.
- **(b) Report as secondary.** Document the timing difference in trajectory tables and mechanism
  section; do not claim it as primary evidence. The ER-vs-OOD link is the primary mechanism.
- **(c) Other.**

---

### Q4 — P1 priority: OPD-specific control vs robustness axis?

The draft Cycle 09 intent lists both as P1 but does not strictly order them. Both require
non-trivial compute. The question is which to prioritize if they cannot both be completed in Cycle 09.

- **OPD-specific control (off-policy KD or teacher-forced distillation, Gate D09):**
  Directly closes "teacher logits are sufficient; on-policy doesn't matter." If OPD outperforms
  this control on dip/AUC/non-termination, it becomes a major paper contribution. If not, the
  claim must narrow from "on-policy matching" to "teacher distribution alignment."

- **Robustness axis (second data / prompt distribution, Gate E09):**
  Directly closes "single distribution result." A second math data distribution with the same
  Qwen3-4B student and same training pipeline. Simpler to implement than a new model.

Options:
- **(a) Off-policy KD control first (D09).** Closes the harder-to-dismiss objection: if OPD is
  just distillation, the paper contribution weakens significantly.
- **(b) Robustness axis first (E09).** The single-distribution objection is the easier reviewer
  rebuttal, but showing B08/C08 generalize to a second distribution adds direct value.
- **(c) Run both in parallel in Cycle 09** (if compute allows).
- **(d) Defer both P1 items** — focus Cycle 09 entirely on P0 and P2; revisit in Cycle 10.

---

*请在 `QA-ch.md` 中用中文直接在问题后写下你的回答。完成后告知。*
