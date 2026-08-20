# Evaluation Protocol

```yaml
view_type: stable_methodology
section: exp
status: active
maintenance_rule: edit_in_place_as_methodology_changes_not_per_cycle
grounded_in:
  - /root/LLM-output-density/experiments/opd_sft_h1/scripts/run_opd_minimal_closure_v2.py
  - /root/LLM-output-density/mypaper/local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_gsm8k_chat_template_mismatch.md
  - /root/LLM-output-density/mypaper/local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_math500_dollar_span_extraction_bug.md
  - /root/LLM-output-density/mypaper/local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_mmlu_chat_template_collapse.md
  - /root/LLM-output-density/mypaper/exp/current_experiment_design.md
last_organized: 2026-06-17
```

This file exists specifically because two real measurement bugs
(GSM8K missing chat template, MATH500's broken `$`-span extractor) were only
caught after the fact, and were not caught sooner because this protocol was
never written down in one place. Read this before interpreting any new score,
and update it the moment a task's protocol changes.

**Update policy:** Pre-result-validated content is written directly. Content
pending full 8-model validation is marked `[待 cycle05 code_handin 确认]`.
Final confirmation happens when the cycle05 code_handin is merged back before
writing `result/`.

## Three-Axis Protocol Framework (established cycle05, QA-confirmed 2026-06-17)

Every evaluation task must be explicitly specified along three axes. Failure
to check all three axes was the direct cause of the GSM8K and MMLU failures.

| Axis | Description | Impact |
|---|---|---|
| **think axis** | `enable_thinking=True/False` | Generation tasks: False prevents think-block; loglikelihood tasks: True + think_end_token or switch to generative |
| **chat format axis** | `--apply_chat_template` on/off | Instruct models need it for generation tasks; WinoGrande is the established exception (no chat template — field standard) |
| **token cutoff axis** | `max_gen_toks` for generation tasks | 3072 required for CoT + boxed answer; not relevant for loglikelihood tasks |

**Task type classification:**
- **Generation tasks** (GSM8K, MATH500, NuminaMath-test, MMLU-generative): all three axes matter
- **Loglikelihood tasks** (TruthfulQA-MC1, WinoGrande): token cutoff not relevant; think and chat axes affect loglikelihood mass distribution

**Per-task protocol (QA-confirmed 2026-06-17, MMLU decision finalized 2026-06-17):**

| Task | think | chat | token / fewshot | extractor | validation status |
|---|---|---|---|---|---|
| NuminaMath-test | `False` | ✅ | 3072 tok, 0-shot | last_boxed + math_verify | ✅ cycle04 |
| GSM8K | `False` | ✅ | 3072 tok, 0-shot | extract_pred → last_boxed | ✅ 8/8 models (cycle05) |
| MATH500 | `False` | ✅ | 3072 tok, 0-shot | last_boxed + math_verify | ✅ 8/8 models (cycle05) |
| MMLU | `False` | ❌ **no chat template** | loglikelihood, 5-shot | lm_eval MC default | ✅ 8/8 models (cycle05, base model mode, 0.60-0.61) |
| TruthfulQA-MC1 | `False` | ✅ | loglikelihood, 0-shot | lm_eval MC default | ✅ 8/8 models (cycle05, chat template) |
| WinoGrande | `False` | ❌ **no chat template** | loglikelihood, 0-shot | lm_eval MC default | ✅ 8/8 models (cycle05, no chat template) |
| ARC-challenge | `False` | ❌ **no chat template** | loglikelihood, 25-shot (task default) | lm_eval acc_norm | ✅ 8/8 models (cycle05, 0.4565–0.4710) |

> **MMLU format decision (2026-06-17):** MMLU 使用 **base model 模式**（不加 chat template），
> 与 Qwen3 官方技术报告及 lm_eval 社区标准一致。原因：
> (1) lm_eval loglikelihood MMLU 对 A/B/C/D token 做直接对数概率比较，不涉及文本生成，
> chat template 的格式前缀对 loglikelihood 比较有干扰而非帮助；
> (2) `--apply_chat_template` 在未设置 `enable_thinking=False` 时将 MMLU 崩溃到随机基线
> 0.2295（FINDING_05_mmlu_chat_template_collapse.md），即使加 `enable_thinking=False`
> 也只能恢复到 ~0.47-0.57，仍低于 base model 模式的 0.55-0.70；
> (3) Qwen3-4B 官方 MMLU 72.99% 与 no-chat-template lm_eval 运行对齐（0.7014），
> 与 chat-template+thinking-off 模式（0.5721）不对齐——说明官方基准本身用 base model 模式；
> (4) 社区 lm-eval-harness Issue #3405、#3576 均确认 no-chat-template 是该任务的正确协议。
> 全部 8 模型已在 base model 模式下完成重测（cycle05，2026-06-17），得分范围 0.6004-0.6086（`mmlu_protocol_version=2`）。

> **WinoGrande 与 MMLU 一致性说明：** WinoGrande 同为 no-chat-template 协议（field standard），
> 与 MMLU 的 base model 模式保持一致，这两个任务都是 loglikelihood 评测，不需要 chat 格式。

## Task-To-Axis Mapping

| Task | Axis | Status |
|---|---|---|
| NuminaMath-test (1024-row held-out split) | **ID** (in-domain, primary) | Correct protocol from the start (chat template + `enable_thinking=False` applied via `eval_numina_full.py`). |
| GSM8K (full, N=1319) | **OOD / transfer** | Cycle 04 ran this **without** the chat template (protocol bug, see below). Corrected for 4/8 models in Cycle 05. |
| MATH500 (hendrycks_math500, N=500) | OOD / transfer, **currently uninformative until fully re-scored** | Cycle 04's `0.000` for every model was a scoring-extractor bug, not a capability result (see below). Corrected for 4/8 models in Cycle 05. |
| OOD-lite: MMLU (+ STEM/Humanities/Social Sciences/Other breakdown), TruthfulQA-MC1/MC2, WinoGrande, ARC-challenge | OOD-lite, secondary stability axis | **全部完成（cycle05，2026-06-17/18）。** MMLU: base model 模式，no chat template，5-shot（`mmlu_protocol_version=2`），得分 0.6004-0.6086。TruthfulQA-MC1: chat template + loglikelihood，0-shot，得分 0.3317-0.3439。WinoGrande: no chat template，0-shot，得分 0.6172-0.6322。ARC-challenge: no chat template，25-shot（lm-eval 任务默认值），acc_norm，N=1172，得分 0.4565-0.4710。OOD-lite 可引用，但注意：4 项任务 8 模型间 spread 均极小（<1.5%），仅可作为能力保持检验，不可作为 OPD vs SFT 判别轴。 |

## Why GSM8K And MATH500 Were Wrong (Cycle 04), And What Fixed Them (Cycle 05, partial)

- **GSM8K**: ran through `lm_eval`'s default `gsm8k` task, `num_fewshot=0`,
  no chat template (`chat_template: None` in the result JSON metadata), on an
  instruction-tuned chat model. `exact_match,strict-match=0.000` for every
  single model was the tell — the same artifact shape later confirmed for
  MATH500. Fix: a dedicated `eval_gsm8k_full.py` applying the chat template,
  `enable_thinking=False`, `max_gen_toks=3072`, sampling
  `temp=0.7/top_p=0.8/top_k=20` (matching the ID-axis protocol below) — not
  `lm_eval`'s flexible-extract path.
- **MATH500**: `lm_eval`'s `hendrycks_math.utils.process_results` extracts
  the model's answer by slicing between the **first** and **last** literal
  `$` character in the whole generated response. For a `<think>`-style
  response full of intermediate LaTeX, this slices across most of the
  reasoning trace, not the final `\boxed{}` answer — so it scores `0.0`
  almost unconditionally regardless of whether the model's real answer was
  correct. Fix: a dedicated `eval_math500_full.py` using
  `last_boxed_only_string`/`remove_boxed` on the **model's own response**
  (not a `$`-position heuristic) plus `math_verify` symbolic equivalence —
  the same extractor pattern already used and validated on the ID axis.
- **Both fixes together** raised GSM8K from a ~0.33-0.43 band to ~0.70-0.72,
  and MATH500 from a uniform `0.000` to ~0.71-0.73, for the 4 models
  re-evaluated so far (`base`, `theta0`, `opd_lmbda05`, `sft_n128`).
  `opd_lmbda1`, `sft_n256`, `sft_n512`, `sft_n1024` have **not** been
  re-evaluated under the corrected protocol yet.
- Landed numbers (N=1319 GSM8K, N=500 MATH500, binomial stderr in
  parentheses): `base` GSM8K 0.7005 (±0.0126) / MATH500 0.7140 (±0.0202);
  `theta0` GSM8K 0.7180 (±0.0124) / MATH500 0.7260 (±0.0199); `opd_lmbda05`
  GSM8K 0.7180 (±0.0124) / MATH500 0.7080 (±0.0203); `sft_n128` GSM8K 0.7096
  (±0.0125) / MATH500 0.7260 (±0.0199). `theta0` is the only model ahead of
  `base` on both axes (+0.0174 GSM8K, +0.0120 MATH500), neither individually
  past ~1σ. This reverses Cycle 04's "no headroom above base" reading on
  these two axes — see `FINDING_05_gsm8k_chat_template_mismatch.md` and
  `FINDING_05_math500_dollar_span_extraction_bug.md` for the full numbers and
  cross-axis read.
- **Important scope limit**: the OOD-lite fix attempt surfaced a third,
  unrelated regression (MMLU collapsing to near-chance under
  `--apply_chat_template` without `enable_thinking=False`) — see
  `FINDING_05_mmlu_chat_template_collapse.md`. The chat-template fix is not a
  universal "always apply it" rule; `generate_until` tasks (GSM8K, MATH500)
  needed it, but the `loglikelihood`/multiple-choice OOD-lite tasks need
  further investigation before any number from that suite is trusted.

## Standing Rule: Extraction-Validity Audit

Any near-zero score, or any score that is suspiciously uniform across every
model on one axis, must be audited for an extraction/prompting artifact
**before** being interpreted as a capability finding. This rule exists
because it has already produced two real findings
(`FINDING_04_eval_extraction_artifact.md`, `FINDING_05_gsm8k_chat_template_mismatch.md`,
`FINDING_05_math500_dollar_span_extraction_bug.md`). A minimal audit checks:
chat template applied, `enable_thinking` matches the training format,
generation budget is large enough to avoid truncation before a final boxed
answer, and the extractor is verified against at least one manually-inspected
correct generation.

## Sampling / Generation Settings

| Setting | ID axis (NuminaMath) | GSM8K / MATH500 (corrected, Cycle 05) | OOD-lite (MMLU/TruthfulQA/WinoGrande/ARC) |
|---|---|---|---|
| Chat template | applied | applied | MMLU: ❌ no chat. TruthfulQA: ✅ chat. WinoGrande: ❌ no chat. ARC: ❌ no chat. |
| `enable_thinking` | `False` | `False` | `False` (via model_args for all four tasks) |
| Generation budget | 3072 tokens (v2 clean) | 3072 tokens | task-default (loglikelihood, not generation-length-sensitive) |
| Sampling | temp 0.7 / top_p 0.8 / top_k 20 (v2 clean; v1 was greedy) | temp 0.7 / top_p 0.8 / top_k 20 | deterministic scoring (loglikelihood MC tasks) |
| Scorer | `math_verify` + last `\boxed{}` extraction (`scorer_v2.py`) | same `math_verify`/boxed pattern (dedicated scripts, not `lm_eval` defaults) | `lm_eval` task defaults |

## Matching Regimes For OPD-vs-SFT Comparison

Established in `experiment_design_05.md`; this is the stable analysis-method
layer that future cycles should reuse rather than re-derive:

| Regime | Rule | Use |
|---|---|---|
| Base ceiling | Compare trained variants against the untrained base with uncertainty | Mandatory headroom gate before any stability claim |
| Equal GSM8K | Nearest `GSM8K_gain` match (the original Cycle 03/04 rule) | Diagnostic continuity only — known to be sensitive to `theta0` saturation, see `research_cycles/exp_design_evolution.md` Cycle 04 entry |
| Equal ID | Nearest NuminaMath-test score within tolerance | Primary fair-control comparison |
| Equal budget | Comparable training/update/sample budget | Primary fair-control comparison |

## Base Model Evaluation Protocol (added cycle06, 2026-06-23)

For `Qwen3-1.7B-Base` and any model trained from it, use **base model mode**
throughout — no chat template, no `enable_thinking` argument. This is a
distinct protocol track from the instruct-series protocol above.

**Source:** Qwen3 technical report (arXiv:2505.09388), Table 8; community
standard for base-model evaluation (no chat template for loglikelihood tasks).
Added to source_matrix per protocol research rule.

| Task | think | chat | format | axis |
|---|---|---|---|---|
| NuminaMath-test | n/a | ❌ | 3072 tok, 4-shot CoT plain | ID |
| GSM8K | n/a | ❌ | 3072 tok, 4-shot CoT plain | OOD |
| MATH500 | n/a | ❌ | 3072 tok, 4-shot CoT plain | OOD |
| MMLU | n/a | ❌ | loglikelihood, 5-shot | OOD-lite |
| TruthfulQA-MC1 | n/a | ❌ | loglikelihood, 0-shot | OOD-lite |
| WinoGrande | n/a | ❌ | loglikelihood, 0-shot | OOD-lite |
| ARC-challenge | n/a | ❌ | loglikelihood, 25-shot | OOD-lite |

**4-shot CoT format for math tasks:** Demonstrations provided as plain text
(`Problem: ...\n\nSolution: ...` blocks) without any chat template. Same
`last_boxed + math_verify` extractor as instruct protocol.

**Cross-series comparability:** Instruct-series and base-series scores are
NOT directly comparable (different prompt format). Within-series comparisons
only: instruct SFT vs `instruct_base`, base SFT vs `base_base`.

**When to use:** Apply this protocol to any model whose starting checkpoint
is a pretrain-only base (no instruction tuning), OR any model trained from
such a starting checkpoint in plain-text format. Do not mix with instruct
protocol for the same model.

---

## Cycle 07 Evaluation Protocol (2026-06-24): Think-Format + Multi-Checkpoint

Cycle 07 introduces a new eval track for the Qwen3-4B-Base + think-format SFT trajectory.
This track **replaces** the no-think instruct protocol and the 4-shot CoT base protocol as
the primary evaluation axis from Cycle 07 onward.

### New Task Set

| Task | N | think | chat | format | axis | notes |
|---|---|---|---|---|---|---|
| MATH500 | 500 | ✅ (natural) | ✅ | 0-shot, temp=0.6, max 32768 | ID primary + response_length diagnostic | last `\boxed{}` + math_verify |
| NuminaMath-test | 1024 | ✅ (natural) | ✅ | 0-shot, temp=0.6, max 32768 | ID (degradation check) | last `\boxed{}` + math_verify |
| AIME24 | 30 | ✅ (natural) | ✅ | 0-shot, temp=0.6, max 32768 | OOD (hard math, pass@1) | last `\boxed{}` + math_verify |
| GPQA-Diamond | 198 | ❌ | ❌ | loglikelihood, 0-shot | OOD reasoning | lm_eval, no chat, no thinking |
| MMLU-Pro | 500 (limit) | ❌ | ❌ | loglikelihood, 0-shot, `--limit 500 --seed 42` | OOD knowledge | lm_eval, no chat, no thinking |

**"Think: natural"** means: after think-format SFT, the model naturally produces `<think>...</think>` output.
There is no explicit `enable_thinking=True/False` flag needed — the model was trained on this format.
For the untrained base (step_000), it will produce direct output; extract `\boxed{}` wherever it appears.

### Response Length Measurement

For MATH500 at every checkpoint, record `mean_response_len` = mean tokens in the generated
response (assistant turn only). This is a required trajectory diagnostic (dip-surge, recover-contract).
Output: `response_length_trajectory.csv`.

### Checkpoint Evaluation Protocol

Evaluate all 10 targets (step_000 through step_624) on all 5 tasks.
For lm_eval tasks (GPQA-D, MMLU-Pro): merge LoRA adapter before running lm_eval
(`model.merge_and_unload()`), save merged model temporarily, delete after eval.

### Why MMLU-Pro instead of MMLU (change from Cycle 06)

Cycle 07 switched from MMLU (12k questions, 5-shot) to MMLU-Pro with `--limit 500 --seed 42`.
Reasons: (1) MMLU is too coarse a signal at 5k+ questions; (2) MMLU-Pro is harder and more
discriminative; (3) `--limit 500` brings it to the same scale as MATH500 (N≈500) for
balanced per-task time; (4) Rethink SFT Table 9 reports MMLU-Pro (not MMLU), so our
results are directly comparable to the external anchor.

**MMLU-Pro scores at --limit 500 are NOT comparable to full MMLU-Pro official numbers.**
Use intra-experiment comparison only (step vs. step).

---

## Resolved Items (closed cycle05, 2026-06-17/18)

All items that were pending or unconfirmed as of the initial cycle05 architecture pass are now resolved:

- **OOD-lite suite**: MMLU + TruthfulQA + WinoGrande + ARC-challenge all completed for all 8 models under the correct protocols. See per-task protocol table above. Numbers are citable; discriminative caveat applies (spread <1.5% across all 8 models on all 4 tasks).
- **Extractor implementation**: `component/gsm8k/runner.py` calls `scorer.py`'s `extract_pred()` first (drops `<think>` block, prefers last `\boxed{...}` via brace-depth-aware parsing, falls back to "answer is X" regex, then last non-empty line) and only then applies the last-number regex on the narrowed span. `component/math500/runner.py` uses the same `extract_pred()` span with `scorer_v2.score()` (math_verify symbolic equivalence). Scripts are under `Eval/component/` (moved from `Eval/` root as part of the cycle05 eval stack unification, 2026-06-18).
- **All 8 models corrected**: `opd_lmbda1`, `sft_n256`, `sft_n512`, `sft_n1024` were re-evaluated under the corrected GSM8K/MATH500 protocol in cycle05. Full 8-model results are in `../../local_experiment_results/cycle_05_matched_control_id_ood/run_01/tables/id_ood_trajectory.csv`.
