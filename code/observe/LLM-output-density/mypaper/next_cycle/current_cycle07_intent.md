# Current Cycle 07 Intent

```yaml
artifact_type: next_cycle_intent
cycle: cycle_07_base_model_sft_feasibility
status: ready_for_related_work_then_experiment_design
created: 2026-06-24
source_qa: next_cycle/QA.md
hard_dependency: Related Work phase must complete before Experiment Design
write_owner_next: Related Work → Experiment Design
```

This file is the handoff from Next Cycle Seed to Related Work and then to Experiment
Design. It records the Cycle 07 seed decision after QA discussion.

---

## Seed Decision

Cycle 07 is a **base model SFT feasibility cycle**, with the following key shift:

> Move the primary experimental model from **Qwen3-1.7B instruct** to
> **Qwen3-4B base** (pretraining checkpoint, no instruct tuning, no chat/think format).

This is driven by two findings from Cycle 06:
- No-think concise SFT **destroys the function** of thinking in instruct models
  (Cycle 06 think probe, indicative, n=100). The instruct training regime is not a
  clean starting point for SFT feasibility experiments.
- Related work (Rethink SFT and related) uses **base model without instruct/think
  format** for SFT comparison experiments.

**Instruct-model experiments are deferred to supplementary.** Core SFT vs OPD
comparison will be built from a clean base pretraining checkpoint.

---

## Hard Dependency: Related Work

**Experiment Design is BLOCKED until Related Work completes.**

Related Work must confirm or update:

| Decision point | What to confirm |
|---|---|
| Model choice | Is Qwen3-4B base the right checkpoint? Size, checkpoint path, availability. |
| Training format | What data format (plain Q+A, no chat template, no think block) does literature use for base model SFT feasibility experiments? |
| Training data | Should NuminaMath be kept, or does literature suggest a change? |
| Primary evaluation axis | What is the right ID / downstream axis for a base (non-instruct) model? Is NuminaMath-test still appropriate, or is a different task more standard? |
| Stop sequence | Confirm base model eval requires explicit stop sequences for generative math tasks. |

---

## Immediate Pre-Experiment Tasks

These tasks are cheap and do not require Related Work to complete. They should be
done at the start of Cycle 07, before Experiment Design begins.

### Task 1: Base series stop-fix (resolves A06b UNDETERMINED)

Re-run all base-series numina and GSM8K evaluations with `stop=["\n\nProblem:"]`
added to every base-model runner. This was confirmed in Cycle 06 to eliminate
the ~66% runaway-continuation truncation artifact.

- Result: produces the first **valid** base-series NuminaMath and GSM8K numbers.
- Gate outcome: closes A06b with a definitive PASS or FAIL.
- Prerequisites: none. Can be run before Related Work.

### Task 2: Cycle 06 geometry copyback (closes C06 PARTIAL)

Copy `geometry_metrics.csv` and `principal_evidence.csv` from the Cycle 06 run
back to `local_experiment_results/cycle_06_sft_feasibility_and_degradation/run_01/`.
Run the OverlapLift analysis. Record results in `result/current_picture.md` under
the Cycle 06 geometry section and update `result/claims_allowed.md`.

- Gate outcome: closes C06 PARTIAL → DONE or FAIL.
- Prerequisites: the CSV files from the Cycle 06 training run must be located.

**Note:** Think-probe expansion (no-think SFT destroys thinking function, indicative
n=100) is **deferred**. With the model shift to base (non-instruct), the think-probe
finding becomes secondary and no confirmatory run is needed now.

---

## Primary Cycle 07 Goals

1. **A07 — Base SFT feasibility:** Can SFT from Qwen3-4B base (pretraining
   checkpoint) produce a measurable gain on the primary downstream / ID axis?
   Specific axis to be pre-registered in Experiment Design after Related Work.

2. **D07 — A06b reconfirm:** With the stop-fix applied, what is the base-model
   numina and GSM8K feasibility? (From pre-experiment Task 1.)

3. **Degradation curve:** Does base-model SFT degrade ID performance as data scale
   increases, as seen in instruct no-think SFT? Or is the pattern different for a
   pretraining checkpoint?

4. **Eval protocol for base model:** Establish the correct evaluation protocol for
   a base (non-instruct) model: chat template, think mode, few-shot format, stop
   sequences. Must be finalized in Experiment Design.

---

## Deferred Items (explicit deferrals from Cycle 07 QA)

| Item | Reason for deferral |
|---|---|
| Think-format SFT | Not needed given model shift to base; related work may confirm base-format SFT is the right direction |
| Think-probe confirmatory expansion | Becomes secondary if primary experiments use base model (no instruct tuning, no think mode) |
| OPD training arms | First establish a clean base-model SFT baseline; OPD comparison requires valid SFT gain first |
| B07 gate (format comparison: think vs no-think) | Not applicable if model shifts to base pretraining checkpoint |
| C07 gate (think-SFT forgetting) | Not applicable without think-format SFT |

---

## Suggested Gate Shape

These are seed-level gates. Experiment Design must formalize exact thresholds after
Related Work review.

| Gate | Purpose | Prerequisite | Pass → | Fail → |
|---|---|---|---|---|
| D07 (A06b reconfirm) | Validate base-series eval with stop sequence | Task 1 done | Base feasibility judgment possible | Check stop-sequence implementation |
| A07: SFT learnability | Base SFT improves ID/downstream axis over untrained base checkpoint | Related Work + Experiment Design | Proceed to degradation characterization and OPD design in Cycle 08 | Redesign: different scale, data, or eval axis |
| (optional) Geometry | Collect base-model geometry (OverlapLift, effective rank, drift from base checkpoint) | Training runs complete | Diagnostic for mechanism layer | No geometry claim; proceed from capability results |

---

## Active Scope

**Do in Cycle 07:**
- Immediate pre-experiment tasks (stop-fix + geometry copyback)
- Related Work phase (Rethink SFT + related): confirm model, format, data, eval axis
- Base model SFT training (Qwen3-4B base, format TBD)
- Primary eval on pre-registered ID/downstream axis
- Geometry collection alongside training runs (OverlapLift, effective rank, drift)

**Defer to Cycle 08 or later:**
- OPD training arms and OPD-vs-SFT comparison
- Think-format SFT
- Think-probe confirmatory expansion
- Paper-level mechanism claims

---

## Required Experiment Design Decisions

After Related Work phase, Experiment Design must explicitly decide and pre-register:

1. **Model checkpoint:** Exact path to Qwen3-4B base (pretraining checkpoint).
   Confirm: is this `Qwen3-4B-Base` on HuggingFace, or a different checkpoint?
2. **Training format:** data format (plain Q+A, no chat template, no think block),
   target text structure, tokenization. Must match Related Work protocol.
3. **Training data:** keep NuminaMath (likely), confirm sample sizes, split rule.
4. **Primary evaluation axis:** pre-register the ID axis for Gate A07. If not
   NuminaMath-test (possible if eval format changes for base model), justify choice.
5. **Evaluation protocol for base model:** chat template off, think mode off,
   few-shot format, stop sequences for all generative tasks.
6. **Geometry protocol:** reference checkpoint for drift (must be base, not theta0),
   which layers, and compressed UV/sketch artifacts to save.

---

## Routing After Cycle 07

- **If A07 passes (SFT gain > pre-registered threshold):** proceed to degradation
  characterization and OPD comparison design in Cycle 08. Base model is confirmed
  as the experimental platform.
- **If A07 fails:** redesign before OPD. Options: different base size, different
  task, more data, or a different training format per Related Work.
- **D07:** whichever outcome, record the base-series stop-fix numbers as the new
  ground truth and update `result/current_picture.md` and `result/claims_allowed.md`.

---

## Read Next

Related Work should read, in order:

1. `next_cycle/current_cycle07_intent.md` (this file)
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `exp/experimental_methodology/model_and_data.md`

Then locate and review:
- "Rethink SFT" (primary reference)
- Additional base-model SFT feasibility papers as relevant

After Related Work, Experiment Design should read:

1. `next_cycle/current_cycle07_intent.md`
2. `result/current_picture.md`
3. `result/claims_allowed.md`
4. `exp/experimental_methodology/model_and_data.md`
5. `exp/experimental_methodology/training_protocol.md`
6. `exp/experimental_methodology/evaluation_protocol.md`
7. `exp/experimental_methodology/geometry_protocol.md`
8. Related Work output

Then write `exp/current_experiment_design.md`, `exp/current_acceptance_criteria.md`,
and the cycle07 code handoff.
