# Open Questions

```yaml
view_type: open_questions
status: active_cycle09_seed
maintenance_rule: dynamic_incremental; resolved_items_moved_to_history_questions.md; new_items_appended
last_updated: 2026-07-07
```

All currently open questions entering Cycle 09. Resolved Cycle 08 questions are in
`history_questions.md` (Cycle 08 Close section).

**Project state after Cycle 08:** B08 STRONG PASS (OPD final 0.848 vs SFT 0.752,
+0.10–+0.24 at every step, unified cap 16384). C08 PASS (OPD MMLU-Pro net +0.016 vs
SFT −0.029). D08 positive (ER cleanly separates arms; SFT rank bump at step_20; OPD
monotone contraction). A08 inconclusive (OverlapLift retracted; ρ contradicts external
full-param lit). OPD dips earlier and deeper (step_5, −0.100) than SFT (step_20, −0.064),
but recovers much faster (step_20 vs step_160).

**Central open question:** What is our novel contribution given external full-param lit
(arXiv 2606.13657/07082) already published OPD off-principal weight geometry? The candidate
moat is the activation-space ER observation + the geometry-process-result chain, not the
weight-space principal direction.

**Cycle 09 gate priority:** A09 (contribution positioning + fp32 measurement) > B09
(stats CI + geometry) > C09 (output-control) > D09/E09 deferred to Cycle 10.

---

## Cycle 09 Primary: Contribution Positioning (URGENT — before new experiments)

1. **What is the exact novel claim given the external lit scoop?**
   External lit covers OPD off-principal in weight space (full-param). Three candidate
   framings for our paper:
   - **(a) LoRA supplement:** LoRA adapters change the principal geometry rules; weight-space
     principal direction under LoRA is different from full-param. Our LoRA ρ finding
     (OPD on-principal under LoRA) is a genuine LoRA-regime result.
   - **(b) New observation space:** Activation-ER is a discriminating observable that is NOT
     covered by weight-space analyses. We propose ER as the mechanism observable linking
     training dynamics to OOD outcome.
   - **(c) Geometry-process-result chain:** ER state (compression level) → dip-recovery process
     → OOD outcome. A narrative that integrates activation geometry, training trajectory, and
     generalization. This chain may need full-param + multi-task validation.
   **Status: URGENT. Assign to Theory conversation + user decision before any new experiments.**

2. **Which framing yields the strongest paper given compute constraints?**
   **Status: depends on fp32 retrain result (Gate A09) and user's judgment on scope.**

---

## Gate A09 — Weight-Space Direction (Full-Param / fp32 Retrain)

1. **Does full-param or fp32-master retrain confirm LoRA ρ (OPD on-principal) or match
   external lit (OPD off-principal)?**
   Current state: LoRA ρ reads OPD on-principal throughout (z +5.7 to +7.3); SFT off-principal
   at steps 5–10 then flips. External full-param lit says OPD off-principal. Contradiction.
   **Status: must run. Result determines whether framing (a) (LoRA supplement) is defensible.**

2. **Can the LoRA-vs-full-param geometry difference become an independent contribution?**
   If fp32 retrain shows OPD off-principal (matching external lit), then the LoRA-vs-full-param
   discrepancy itself is a finding: "LoRA training changes the weight-space principal geometry."
   **Status: contingent on A09 fp32 result.**

---

## Gate B09 — Statistical Reliability

1. **Have paired bootstrap 95% CI been computed for all OPD-vs-SFT deltas?**
   Required: OPD final − SFT final, OPD peak − SFT peak, OPD dip depth − SFT dip depth,
   trajectory AUC difference, non-termination peak difference.
   **Status: not yet run. Every main-text OPD-vs-SFT claim must have CI or be marked
   qualitative/diagnostic. Assign to Cycle 09 Code conversation.**

2. **Is the OPD early dip mechanism claim (step_5, −0.100) statistically significant?**
   Need: CI on finisher acc at dip checkpoint; CI on recovery speed difference (steps to
   recover vs SFT). This is required to elevate the "sharper-shorter transient" to a paper claim.
   **Status: open. Must be resolved in Cycle 09 before the dip mechanism can be stated as
   a primary finding (Q3=a decision).**

---

## OPD Early Dip Mechanism (Core Contribution Candidate)

1. **Can the "sharper-shorter transient" story be formalized?**
   OPD: step_5 dip (−0.100), recovers by step_20 (4 steps of dip). SFT: step_20 dip (−0.064),
   recovers by step_160 (7 steps of dip). Both dips co-locate with ER uptick.
   Hypothesis: on-policy distillation triggers rapid compression break, then re-anchors
   quickly. SFT triggers slower break without live guidance. This predicts:
   - OPD ER uptick earlier (step_5) and shorter-lived than SFT ER uptick (step_20, longer).
   **Status: requires ER timing analysis (confirm OPD ER uptick at step_5) + CI on dip metrics.
   Assign to Cycle 09 geometry + stats tasks.**

2. **Does the ER timing confirm the "sharper-shorter" story at the activation level?**
   Needed: OPD ER at step_5 vs step_0 (is there an uptick?), and the OPD ER trajectory
   shape (does it compress faster after step_5 than SFT does after step_20?).
   **Status: open. Must be extracted from Cycle 08 geometry artifacts.**

---

## Compression → OOD Causality (Diagnostic, Cycle 09)

1. **Can ER be measured on OOD-domain inputs (MMLU-Pro prompts) to strengthen the
   compression → OOD causal story?**
   Currently: ER measured on math-probe activations only. To build a causal story,
   need to show SFT's rank bump on MMLU-domain activations evicts OOD-relevant directions,
   while OPD's monotone contraction preserves them.
   **Status: open. High-value but requires running additional ER measurements on held-out
   MMLU-Pro prompts through the saved checkpoints.**

2. **Does OPD MMLU-Pro trajectory (large dip step_40/80, recovery by step_160) reflect
   a transient compression break on OOD inputs, parallel to the ID dip pattern?**
   Step_40/80 OPD MMLU-Pro dips (0.4764 → 0.3993 → 0.3607) are large — almost −0.12.
   This might be a second compression transient on the OOD axis, distinct from the ID dip.
   **Status: open. Needs interpretation in light of ER measurements.**

---

## Gate C09 — Output-Control Analysis (P0)

1. **Is the OPD accuracy advantage over SFT capability, output-control, or mixed?**
   OPD produces long responses (3974 tokens at final) with high truncation under 4096 cap.
   Unified cap 16384 was used for MATH500 comparison; but the output-control behavior
   differs substantially from SFT.
   **Status: decomposition into finisher acc + non-term rate not yet done for all checkpoints
   under unified cap. Assign to Cycle 09 output-control analysis.**

2. **Does OPD improve non-termination behavior vs SFT at matched training steps?**
   SFT peaks at ~79% non-termination at step_80. OPD's non-term pattern differs (longer
   responses but different termination dynamics). Direct comparison needed.
   **Status: open. Assign to Cycle 09 Code conversation.**

---

## Paper Rewrite and Theory (P2, Cycle 09)

1. **Can the paper adopt the geometry-process-result framework as the main narrative?**
   - Geometry: activation ER as state indicator (compression level)
   - Process: dip-recovery transient (timing and depth as mechanism signal)
   - Result: OOD outcome (MMLU-Pro preservation vs degradation)
   **Status: open. Theory conversation must develop this framework before the LaTeX draft
   is reorganized. Non-causal wording required throughout.**

2. **How to frame the weight-space ρ finding relative to external lit?**
   If A09 fp32 confirms OPD off-principal (matching external lit): our LoRA finding was
   LoRA-confounded → acknowledge, cite, and focus on activation-space moat.
   If A09 fp32 shows OPD on-principal (matching our LoRA ρ): LoRA changes the geometry
   rules → this is a genuine finding, argue the LoRA supplement contribution.
   **Status: contingent on A09. Must be resolved before paper framing is finalized.**

3. **How to remove cycle-language ("Cycle 07/08") from paper main text?**
   **Status: paper rewrite task in P2. Assign to paper-writing conversation.**

---

## Deferred to Cycle 10

| Item | Reason for deferral |
|---|---|
| **Gate D09: OPD-specific control (off-policy KD)** | Main contributions not yet established. Core claims (LoRA geometry supplement OR new observation space OR geometry-process-result chain) must be finalized first. |
| **Gate E09: Robustness axis (second data distribution)** | Same reason as D09. |
| **Gate F09: Teacher / theta0 / headroom baselines** | Same reason. Lower priority once core claims are established. |
