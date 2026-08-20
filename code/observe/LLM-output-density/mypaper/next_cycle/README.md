# Current Next Cycle

```yaml
view_type: current_workspace_section
section: next_cycle
status: active
current_cycle: cycle_06_sft_scale_and_forgetting
last_organized: 2026-06-18
```

This directory is the current next-cycle entry point. It tracks what should happen
next, while `../research_cycles/` preserves older next-cycle seeds.

## Current Files

| File | Role | Maintenance |
|---|---|---|
| `open_questions.md` / `-ch.md` | All currently open questions (unresolved from prior cycles + new from current cycle). Resolved items are moved out to `history_questions.md`. | Dynamic incremental |
| `history_questions.md` / `-ch.md` | Per-cycle log: each cycle appends one section recording questions resolved and questions newly proposed that cycle. | Append-only per cycle |
| `QA.md` / `-ch.md` | Entry point for Next Cycle Seed conversation. Overwritten each time the phase is entered: pose questions → user answers → discuss → internalize to `open_questions.md` + `history_questions.md`. | Overwrite each cycle |
| `current_cycle<N>_intent.md` / `-ch.md` | Active cycle seed (written by Next Cycle Seed conversation, read by Experiment Design). Deleted after Experiment Design formalizes it into `exp/current_experiment_design.md`. | One file per active cycle; delete when Exp Design is written |

## Next Cycle Seed Conversation — Startup Sequence

1. `../readme.md` (workspace rules and read/write boundaries)
2. `../result/current_picture.md`
3. `../result/claims_allowed.md`
4. `../exp/current_acceptance_criteria.md` (if exists)
5. `open_questions.md` (all currently open questions)
6. `history_questions.md` (context on how questions evolved)
7. Overwrite `QA.md` with questions for this cycle → discuss → internalize

## Current Status

Cycle 05 closed 2026-06-18. Gates A05 / B05 / C05 all passed. Gate C04/D04 confirmed
fail (no flip). Cycle 06 seed was revised after QA and narrowed to an SFT-only
feasibility/degradation cycle.

**Cycle 06 core question:** Before running another OPD-vs-SFT comparison, can
plain SFT produce a real downstream / ID gain under a well-specified model,
data, task, and train/eval format, and where does SFT begin to degrade ID/OOD
behavior?

**Active experiment direction:**
- SFT-first diagnostic: train SFT from the unfinetuned student/base under
  pre-registered data scales and evaluation protocols. Decide whether the
  current model/data/task/format can support a later OPD-vs-SFT comparison.
  OPD forgetting-prevention training is deferred until this SFT baseline is
  understood.

**Next step:** Experiment Design phase for cycle06. Read
`current_cycle06_intent.md`, `exp/experimental_methodology/model_and_data.md`,
`exp/experimental_methodology/training_protocol.md`,
`exp/experimental_methodology/evaluation_protocol.md`, and
`exp/experimental_methodology/geometry_protocol.md`, then write
`exp/current_experiment_design.md` and `exp/current_acceptance_criteria.md`
for cycle06.
