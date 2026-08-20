# Current Cycle 06 Intent

```yaml
artifact_type: next_cycle_intent
cycle: cycle_06_sft_feasibility_and_degradation
status: ready_for_experiment_design
created: 2026-06-18
source_qa: next_cycle/QA.md
write_owner_next: Experiment Design
```

This file is the handoff from Next Cycle Seed to Experiment Design. It records
the Cycle 06 seed decision after QA discussion.

---

## Seed Decision

Cycle 06 is an **SFT-only feasibility and degradation cycle**.

Do not run a full OPD-vs-SFT comparison yet. Do not train new OPD arms in this
cycle unless the user explicitly reopens scope in Experiment Design.

The purpose is to rebuild the experimental ladder step by step:

1. Confirm whether plain SFT can improve the selected downstream / ID target.
2. Measure whether plain SFT causes ID/OOD degradation as data scale increases.
3. Use those results to decide whether a later OPD experiment is justified.
4. Keep geometry as a diagnostic layer, not the main gate.

---

## Primary Cycle 06 Questions

1. **SFT learnability:** Can SFT from the unfinetuned student/base produce a real
   gain on the chosen primary downstream / ID axis?
2. **SFT degradation:** At what data scale, if any, does SFT degrade ID and/or
   OOD performance?
3. **Protocol selection:** Which model, data scale, task pair, and input/output
   format are appropriate for a clean SFT baseline?
4. **Geometry diagnostic:** Do lightweight S/X and PE-style geometry probes
   (with the exact PE metric defined in Experiment Design) show changes that
   track SFT learning or degradation?

---

## Active Scope

**Train / evaluate:**

- SFT-from-base arms only.
- Candidate initial data scales: n=256, n=512, n=1024. Experiment Design may add
  n=128 for onset resolution or n=2048 for stress testing only with explicit
  resource justification.
- Always include the unfinetuned student/base as a reference row.

**Defer:**

- New OPD arms.
- OPD forgetting-prevention test at matched budget.
- Full OPD-vs-SFT stability comparison.
- Paper-level mechanism claims from geometry.

---

## Required Experiment Design Decisions

Experiment Design must explicitly decide and pre-register:

1. **Model identity:** whether to keep `/root/autodl-tmp/model/Qwen/Qwen3-1.7B`
   as the student/base or change model/task because of headroom limits.
2. **Training data:** exact SFT data source, sample sizes, split rule, and
   whether the run starts directly from base with no `theta0` warmup.
3. **Training format:** chat/plain format, thinking mode, target text structure,
   and whether it remains aligned with `exp/experimental_methodology/model_and_data.md`.
4. **Evaluation axes:** primary ID/downstream axis and OOD axes. Existing default
   candidates are NuminaMath-test, GSM8K, MATH500, and OOD-lite preservation
   checks, but the design must justify whether NuminaMath is too ceilinged.
5. **Scoring audit:** extraction and chat/think/token settings for every task,
   following `exp/experimental_methodology/evaluation_protocol.md`.
6. **Geometry outputs:** whether to run S/X and PE-style probes; what "PE"
   means operationally in this workspace; which layers, modules, artifacts, and
   compressed UV/sketch outputs must be saved for post-hoc audit.

---

## Suggested Gate Shape

These are seed-level gates; Experiment Design should formalize exact thresholds.

| Gate | Purpose | Pass meaning | Fail meaning |
|---|---|---|---|
| A06: SFT learnability | Test whether SFT improves the selected target over base | The setting can support a later OPD comparison | Change model, task, data, or format before OPD |
| B06: SFT degradation | Track ID/OOD deltas across SFT scale | Degradation curve is measurable and interpretable | Current axes do not diagnose forgetting cleanly |
| C06: protocol validity | Confirm every task has explicit think/chat/token/extractor settings | Scores are interpretable | Stop and fix protocol before writing results |
| D06: geometry diagnostic | Collect S/X and PE-style signals if cheap | Geometry can guide later mechanism design | No mechanism claim; proceed from capability results only |

---

## Routing After Cycle 06

- If SFT does not improve the target: do not run OPD yet. Redesign model, task,
  data scale, or input/output format.
- If SFT improves and does not degrade: the setting may be viable for a future
  OPD-vs-SFT comparison.
- If SFT improves but degrades at larger scale: the next OPD cycle can test
  forgetting prevention at the identified scale.
- If geometry signals are collected: treat them as diagnostic until OPD and SFT
  are compared under a valid capability/degradation setup.

---

## Read Next

Experiment Design should read, in order:

1. `next_cycle/current_cycle06_intent.md`
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `exp/experimental_methodology/model_and_data.md`
5. `exp/experimental_methodology/training_protocol.md`
6. `exp/experimental_methodology/evaluation_protocol.md`
7. `exp/experimental_methodology/geometry_protocol.md`

Then write `exp/current_experiment_design.md`, `exp/current_acceptance_criteria.md`,
and the cycle06 code handoff.
