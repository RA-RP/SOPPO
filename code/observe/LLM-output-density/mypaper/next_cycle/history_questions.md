# History Questions

```yaml
view_type: history_questions
status: active
maintenance_rule: append_only_per_cycle; never_overwrite_existing_entries
last_updated: 2026-06-18
```

Per-cycle log of questions resolved and questions newly proposed.
Each cycle appends one section at the end. Do not edit existing entries.

---

## Cycle 04 — Initial Setup (2026-06-15)

### Newly Proposed

Initial question set established:

- **Base-Ceiling** (Q1–Q4): untrained base as reference, headroom thresholds, task/base selection
- **Control Construction** (Q1–Q3): right control for negative SFT slope, matching criteria, multi-anchor reporting
- **ID/OOD Measurement** (Q1–Q4): NuminaMath-test as ID axis, GSM8K as OOD axis, per-source buckets, extraction audit
- **Geometry And Mechanism** (Q1–Q4): effective rank as signal, drift normalization, principal-angle necessity, artifact sufficiency
- **Paper Positioning** (Q1–Q3): methodological warning pivot, smaller empirical contribution, K1 falsifier

### Resolved

None.

---

## Cycle 05 — Eval Protocol Fix (closed 2026-06-17/18)

### Newly Proposed

| Question | Category | Reason for addition |
|---|---|---|
| Does OPD prevent catastrophic forgetting at matched budget? | Forgetting Prevention Q1 | sft_n1024 showed severe NuminaMath ID collapse (0.5516→0.3890); OPD at equivalent budget becomes the natural comparison |
| At what training scale does catastrophic forgetting onset? | Forgetting Prevention Q2 | n=512 stable, n=1024 severe — onset curve unknown |
| Is forgetting-prevention the more defensible claim given absent OPD gain evidence? | Forgetting Prevention Q3 | If Gate A06 fails, forgetting prevention may be the only supported framing |
| MMLU with thinking models | Eval Protocol Q1 | lm-eval with chat template produced near-random scores (~0.23); correct protocol unclear before Qwen3 arXiv review |
| Three-axis protocol checklist | Eval Protocol Q2 | Missing think-axis / chat-axis / cutoff documentation caused measurement bugs (MATH500=0.0 artifact, MMLU collapse) |

### Resolved

| Question | Resolution | Date |
|---|---|---|
| **Base-Ceiling Q1**: Must untrained base always be included as reference row? | **Yes.** Rule established: untrained base is a mandatory reference row in all eval tables. No exceptions. | 2026-06-17 |
| **ID/OOD Q1**: Should NuminaMath-test become the default ID axis? | **Yes.** NuminaMath-test IS the default ID axis. Matching basis for OPD vs SFT comparisons is NuminaMath-test ID. GSM8K is the OOD/transfer axis. (`claims_allowed.md`) | 2026-06-18 |
| **ID/OOD Q2**: Should GSM8K be reported as OOD/transfer? | **Yes.** GSM8K should be treated as OOD/transfer axis, not primary performance axis. (`claims_allowed.md`) | 2026-06-18 |
| **ID/OOD Q4**: What extraction audit is mandatory before any math score? | **Partial.** Project-level rule established: any math score must verify scorer logic against model output format before writing conclusions. Specific per-task protocols written into `exp/experimental_methodology/evaluation_protocol.md`. Ongoing: must apply to all new tasks. | 2026-06-17 |
| **Eval Protocol Q1**: MMLU with thinking models | Adopt Qwen3 official evaluation protocol (arXiv 2505.09388): base model mode (no chat template, 5-shot, standard loglikelihood), `mmlu_protocol_version=2`. Valid scores: 0.60–0.61 for all 8 models. | 2026-06-17 |
| **Eval Protocol Q2**: Three-axis protocol checklist | Written into `exp/experimental_methodology/evaluation_protocol.md`. Every task must document think-axis / chat-axis / token-cutoff-axis settings explicitly. | 2026-06-17 |

---

## Cycle 06 — Seed QA: SFT-first narrowing (2026-06-18)

### Newly Proposed

| Question | Category | Reason for addition |
|---|---|---|
| Can plain SFT improve the selected downstream / ID target at all? | Cycle 06 SFT Feasibility Q1 | User requested a stepwise reset before OPD comparison: first prove that SFT can produce a meaningful learning signal. |
| Which model, data scale, task pair, and train/eval format produce a reliable SFT baseline? | Cycle 06 SFT Feasibility Q2 | Prior cycles bundled many checkpoints and axes at once; Cycle 06 should first settle the engineering/protocol foundation. |
| Does SFT itself cause ID/OOD degradation as scale increases? | Forgetting Prevention / SFT Degradation Q2 | `sft_n1024` showed severe math regression; before asking whether OPD prevents it, the SFT degradation curve must be understood. |
| Can lightweight S/X and PE-style geometry diagnostics be collected alongside SFT without becoming the primary gate? | Geometry Q4 | User prefers collecting cheap geometry signals, while keeping capability/degradation as the main decision layer. PE must be operationally defined in Experiment Design before use. |

### Resolved

| Question / decision | Resolution | Date |
|---|---|---|
| Cycle 06 scope | **Narrowed to SFT-only.** Do not train new OPD arms or run a full OPD-vs-SFT pipeline in Cycle 06. | 2026-06-18 |
| OPD forgetting-prevention direction | **Deferred.** First establish whether SFT can improve the target and where SFT degradation appears. | 2026-06-18 |
| Starting point for the active SFT diagnostic | **Use SFT-from-base as the active direction.** This removes the `theta0` saturation/confound that affected the previous matched-anchor comparison. | 2026-06-18 |
| Geometry role in Cycle 06 | **Diagnostic, not primary gate.** S/X and PE-style probes may be included if cheap, with PE defined before execution; the main gates remain SFT learning and ID/OOD degradation. | 2026-06-18 |
| Paper positioning | **No major pivot yet.** The project remains about activation-space differences and ID/OOD degradation differences between SFT and OPD; Cycle 06 is an engineering/measurement foundation step. | 2026-06-18 |

---

## Cycle 06 — Close (2026-06-24)

### Newly Proposed

| Question | Category | Reason for addition |
|---|---|---|
| Does no-think concise SFT fail to improve ID capability at all scales for instruct models? | SFT Feasibility | A06 FAIL confirmed: all instruct SFT arms below base on NuminaMath at n∈{512,1024,2048,4096} |
| Does no-think SFT destroy the *function* of thinking in Qwen3 thinking models? | Training Format | Think probe (indicative, n=100): instruct_base think-mode +0.13, no-think SFT arms −0.02..−0.06 — form preserved, function removed |
| Is base-model SFT (pretraining checkpoint, no instruct tuning, no think/chat format) the right experimental design? | Model/Format | User flagged "Rethink SFT" reference using base model without think/chat; instruct models "too complex" for clean baseline |
| Should the project shift from Qwen3-1.7B instruct to Qwen3-4B base? | Model Choice | User: "大概率是要换成qwen3-4B-base"; instruct models reserved for supplementary experiments after core paper is written |
| What is the base-model SFT feasibility (A06b) with a correct stop sequence? | Eval Artifact | Base series numina/gsm8k are truncation artifacts (no stop seq, ~66% trunc). Fix: `stop=["\n\nProblem:"]`. Deferred to Cycle 07. |

### Resolved / Newly Answered

| Question | Resolution | Date |
|---|---|---|
| **SFT Feasibility Q1 (instruct + no-think)**: Can SFT improve ID? | **NO (confirmatory).** A06 FAIL: instruct no-think SFT degrades NuminaMath −0.12 to −0.17 at all scales tested. | 2026-06-24 |
| **SFT Degradation Q1 (instruct + no-think)**: Does SFT cause ID degradation? | **YES (confirmatory).** All instruct SFT arms degrade NuminaMath and MATH500. GSM8K mostly held. OOD-lite preserved within noise (B06 = no degradation on general ability). | 2026-06-24 |
| **Training data quantity**: Is too little training data the core constraint? | **NO.** n=4096 does not improve over n=512; more data monotonically worsens ID. Quantity is not the bottleneck; format is. | 2026-06-24 |
| **Base GSM8K artifact**: Are the base-series GSM8K/numina gains real? | **NO.** Without stop sequence, ~66% truncation; apparent +0.28 GSM8K gain is differential de-truncation artifact. Fix confirmed: with stop, base GSM8K flat ~0.66 and SFT is neutral. | 2026-06-24 |
| **Geometry copyback (C06)**: Were cycle06 geometry CSVs analyzed? | **NO.** `geometry_metrics.csv` and `principal_evidence.csv` not copied back; OverlapLift unanalyzed. C06 PARTIAL. Carry to Cycle 07 as first task. | 2026-06-24 |

---

## Cycle 07 — Seed QA (2026-06-24)

### Key Decisions

| Decision | Content |
|---|---|
| **Cycle 07 model direction** | **Shift to Qwen3-4B base** (pretraining checkpoint, not instruct). Pending related work (Rethink SFT and related) confirmation before Experiment Design. Instruct models deferred to supplementary experiments. |
| **Cycle 07 training format** | Base model SFT — no think-format requirement, no chat-format requirement. Specific format decision pending related work. |
| **Think-format SFT (B)** | Deferred; may not be necessary once base model protocol is confirmed from related work. |
| **Training data** | Likely continue with NuminaMath; pending related work. Do not change data source without related work justification. |
| **Cycle 07 immediate tasks** | (1) Base series stop-fix re-collection; (2) Cycle 06 geometry copyback + OverlapLift analysis. Both are cheap and should precede new training design. |
| **Think-probe confirmatory extension** | **Deferred.** If project moves to base model (no instruct), the instruct think-probe finding becomes secondary. No need to expand to full math500+multi-seed now. |
| **Gate shape** | Gate A07 (base SFT feasibility) + Gate D07 (A06b reconfirm) are primary. B07 (format comparison instruct vs no-think) and C07 (think-SFT forgetting) likely not needed given model shift. Downstream task for A07 to be confirmed in related work. |
| **Related work dependency** | Cycle 07 Experiment Design is blocked on Related Work phase completing. Specifically: confirm model choice, training format, and downstream task axes from literature. |

### Newly Proposed

| Question | Category |
|---|---|
| Can Qwen3-4B base model SFT (base-format, no instruct/think) produce measurable ID/downstream gain? | Cycle 07 SFT Feasibility (base model) |
| What is the right downstream / ID evaluation axis for a base model? (NuminaMath open-answer vs few-shot generative format?) | Eval Protocol for Base Model |
| What training format (data structure, target, chat/no-chat) is established for base-model SFT in related work? | Training Format (base model) |
| What does "Rethink SFT" and related literature say about base model vs instruct model for SFT comparison experiments? | Related Work dependency |

---

## Cycle 07 — Close (2026-06-27)

### Newly Proposed

| Question | Category | Reason for addition |
|---|---|---|
| Does the Rethink SFT dip-and-recovery trajectory appear under LoRA fine-tuning? | SFT Trajectory | B07 FULL PASS confirmed: dip at step_20 (finisher acc 0.594), recovery by step_160, peak 0.698@step_320 |
| Is as-run acc cap-robust under max_tokens=4096 for think-format CoT? | Eval Protocol | Cap-pilot confirmed: 0.620@4096 ≈ 0.617@24576; truncation is computationally valid and acc is faithfully measured |
| What does step_20 triple co-location (argmin acc = argmax rank = argmin xs_gap) indicate mechanistically? | Geometry | Robust co-location found; concurrent-not-causal interpretation; effect size small (~1.8% rank span) |
| How to normalize OverlapLift under LoRA r=32 for on/off-principal verdict? | OverlapLift / Methodology | OverlapLift < 1 throughout (rising 0.48→0.74); random baseline doesn't normalize low-rank structure under LoRA; interpretation deferred to Cycle 08 |
| Does think-format math SFT cause OOD degradation? | OOD Generalization | GPQA-D flat; MMLU-Pro mild real degradation −2.2 SE at final (non-monotone); SFT gain is math-domain-confined for LoRA variant |

### Resolved / Newly Answered

| Question | Resolution | Date |
|---|---|---|
| **A07: Base SFT feasibility (Qwen3-4B-Base + Math-CoT-20k)** | **PASS.** Final MATH500 0.680 > base+1SE 0.658. Finisher acc 0.98 at final step. SFT from base is feasible with think-format. | 2026-06-27 |
| **B07: Dip-and-recovery trajectory under LoRA** | **FULL PASS.** Dip at step_20 (finisher acc 0.594, non-term 6.8% — genuine capability dip). Recovery above base by step_160. Peak 0.698@step_320. | 2026-06-27 |
| **C07: OOD-lite transfer** | **No transfer + mild MMLU-Pro degradation.** GPQA-Diamond flat (|Δ|<0.75 SE). MMLU-Pro −2.2 SE at final (mild, real, non-monotone). | 2026-06-27 |
| **Model choice** | **Qwen3-4B-Base confirmed.** SFT feasibility established on this checkpoint. | 2026-06-27 |
| **Training format** | **Think-format LoRA on Math-CoT-20k confirmed.** Dip-and-recovery matches Rethink SFT dynamics for LoRA variant. | 2026-06-27 |
| **Eval protocol for base model** | **Established:** MATH500 generative, no chat template, think-format, max_tokens=4096, as-run acc (cap-robust). | 2026-06-27 |
| **As-run acc vs finisher acc** | **as-run acc is the primary metric.** Cap-pilot proved cap-robust (0.620@4096 ≈ 0.617@24576). Non-termination is a known property of the regime, computationally accepted. Finisher acc available for decomposition analysis. | 2026-06-27 |
| **Drift reference checkpoint** | **Base checkpoint (step_000) is the drift reference**, not theta0. drift_from_base monotone = negative control confirmed in Cycle 07. | 2026-06-27 |
| **C06 PARTIAL (geometry copyback)** | **CLOSED.** Decision: skip; no longer relevant given model shift to 4B+Math-CoT. | 2026-06-27 |
| **A06b UNDETERMINED (base stop-fix)** | **CLOSED.** Decision: skip; 1.7B base experiments are no longer the main line. | 2026-06-27 |

---

## Cycle 08 — Seed QA (2026-06-27)

### Key Decisions

| Decision | Content |
|---|---|
| **Cycle 08 primary goal** | Train OPD arm on same Qwen3-4B-Base + Math-CoT-20k + LoRA setup. Compare OverlapLift, geometry, and capability/OOD-lite vs the Cycle 07 SFT trajectory. |
| **OPD arm design** | Single arm, one best-guess lambda (not a sweep). OPD is expensive; Experiment Design selects the most appropriate lambda for the 4B+Math-CoT regime. |
| **Non-termination handling** | Accept max_tokens=4096 as-is (cap-robust confirmed). Report as-run acc as primary metric. Decompose finisher acc + non-term rate for analysis when comparing arms. Do NOT change max_tokens for Cycle 08. |
| **OOD-lite coverage** | Full trajectory tracking: all 10 checkpoints, GPQA-D + MMLU-Pro for OPD arm. |
| **C06 + A06b** | Both CLOSED. Not tracked further. |
| **Gate priority** | B08 (capability preservation) > C08 (MMLU-Pro preservation) > D08 (geometry pattern) > A08 (OverlapLift trajectory). All four are active; B08 is the primary blocking gate. |

### Newly Proposed

| Question | Category |
|---|---|
| Does OPD arm final MATH500 match SFT (non-inferiority), and does OPD avoid the step_20-level dip? | Gate B08: Capability Preservation |
| Does OPD arm show smaller MMLU-Pro degradation than SFT at final checkpoint and across trajectory? | Gate C08: MMLU-Pro Preservation |
| Does OPD geometry at its capability low-point differ from SFT step_20 triple co-location? | Gate D08: Geometry Pattern |
| Does OPD OverlapLift trajectory differ systematically from SFT (rising 0.48→0.74)? | Gate A08: OverlapLift Comparison |
| What lambda is best for OPD on Qwen3-4B-Base + Math-CoT regime? | OPD Design |
| What is the LoRA-aware reference for OverlapLift normalization, enabling an on/off-principal verdict in Cycle 08? | OverlapLift Methodology |
| Does OPD arm show similar non-termination rate (~50%) as SFT, or does OPD affect output-control? | Output-Control |

---

## Cycle 08 — Close (2026-07-07)

### Newly Proposed (from Cycle 08 Seed QA) → Resolved

| Question | Category | Outcome |
|---|---|---|
| Does OPD arm final MATH500 match SFT (non-inferiority), and does OPD avoid the step_20-level dip? | Gate B08: Capability Preservation | **STRONG PASS**: OPD final 0.848 vs SFT 0.752 (+0.096) at unified cap 16384; OPD beats SFT at every step (+0.10 to +0.24). |
| Does OPD arm show smaller MMLU-Pro degradation than SFT at final checkpoint and trajectory? | Gate C08: MMLU-Pro Preservation | **PASS**: OPD net +0.016 vs SFT net −0.029; delta margin +0.045. Non-monotone trajectory (large dip step_40/80, recovery by step_160). |
| Does OPD geometry at its capability low-point differ from SFT step_20 triple co-location? | Gate D08: Geometry Pattern | **POSITIVE**: SFT shows L18 step_20 rank bump; OPD ER contracts monotonically with no bump. ER is the sole metric cleanly separating the arms and tracking OOD outcome. |
| Does OPD OverlapLift trajectory differ systematically from SFT? | Gate A08: Weight-Space Direction | **INCONCLUSIVE**: OverlapLift retracted (≈ random-LoRA null, magnitude/bf16-dominated). ρ reads OPD on-principal / SFT early off-principal; contradicts external full-param lit (arXiv 2606.13657/07082 says OPD off-principal). Full-param/fp32 measurement deferred. |
| What lambda is best for OPD on Qwen3-4B-Base + Math-CoT regime? | OPD Design | **RESOLVED**: lambda=1.0 used (verl supervised forward_kl_topk, topk=32). |
| What is the LoRA-aware reference for OverlapLift normalization? | OverlapLift Methodology | **RETRACTED**: random-LoRA null disproves OverlapLift as meaningful. ρ introduced but contradicts external lit → unresolved for now. |
| Does OPD arm show similar non-termination rate (~50%) as SFT? | Output-Control | **NEW FINDING**: OPD response length monotonically increases (626→3974 tokens). Very high truncation at late steps under 4096 → unified cap 16384 required; corrected numbers used for B08. OPD termination pattern different from SFT. |

### Newly Discovered (not pre-registered in Cycle 08 Seed)

| Finding | Significance |
|---|---|
| **OPD early dip at step_5** (MATH500 −0.100; numina −0.070) | Earlier and deeper than SFT dip at step_20 (−0.064). OPD recovers by step_20; SFT by step_160. Both co-locate with ER uptick. Supports "sharper-shorter transient" mechanism story. |
| **LoRA ρ vs full-param lit contradiction** | LoRA ρ reads OPD on-principal; external full-param lit says OPD off-principal. LoRA structure may change principal geometry rules relative to full-param → new contribution angle. |
| **Eval-cap truncation required for OPD** | OPD long distilled CoT systematically truncated at 4096 → as-run late-step MATH500 biased low. Unified cap 16384 applied. "Post-peak decline" was an artifact, not real capability drop. |

---

## Cycle 09 — Seed QA (2026-07-07)

### Key Decisions

| Decision | Content |
|---|---|
| **Cycle 09 primary task (Q1=c, expanded scope)** | First: define and narrow the paper's novel contributions given external full-param lit. Then: run fp32/full-param retrain + P0 (stats CI + geometry + output-control) + P2 (theory + paper rewrite). P1 deferred to Cycle 10. |
| **Weight-space direction (Q2=a, LoRA supplement + new observation space vision)** | Run full-param/fp32 retrain to resolve ρ contradiction. Three-part contribution framing: (1) LoRA changes principal geometry rules vs full-param → LoRA supplement; (2) Activation-ER as new observation space with discriminative power; (3) Geometry-process-result chain: ER state → dip-recovery process → OOD outcome. May need full-param + multi-task validation. |
| **OPD early dip as core contribution (Q3=a)** | OPD step_5 "sharper-shorter transient" vs SFT step_20 "shallower-longer transient" is a core mechanism contribution. Must be combined with CI and final results. |
| **P1 deferred to Cycle 10 (Q4=d)** | Off-policy KD control (Gate D09) and robustness axis (Gate E09) deferred. Main contributions not yet established; secondary validations come after. |
| **Cycle 09 scope** | P0 expanded (contribution positioning + fp32 measurement + stats CI + geometry + output-control) + P2 (theory + paper rewrite). P1 (control + robustness + teacher baselines) → Cycle 10. |

### Newly Proposed

| Question | Category |
|---|---|
| What is the exact novel contribution given external full-param lit (OPD off-principal in weight space) already published? | Contribution Positioning (URGENT) |
| Does full-param/fp32 retrain confirm LoRA ρ reading (OPD on-principal) or match external lit (OPD off-principal)? | Gate A09: Weight-Space Direction |
| Can the LoRA-vs-full-param geometry difference become its own contribution ("LoRA changes principal direction")? | Contribution Positioning |
| Can paired bootstrap CI establish OPD early dip depth/recovery speed difference as statistically significant? | Gate for OPD Dip Mechanism Claim |
| Can ER be measured on OOD-domain inputs (MMLU-Pro prompts) to strengthen compression→OOD causality? | Compression→OOD Causal Chain |
| Is the geometry-process-result chain (ER state → dip-recovery → OOD outcome) defensible as a paper narrative? | Theory / Paper Framing |
| Does OPD output-control (response length, non-term) explain any accuracy difference vs SFT, or is it pure capability? | Output-Control Analysis (C09) |
| Have all OPD-vs-SFT main-text claims been accompanied by paired bootstrap 95% CI? | Statistical Reliability (A09) |
