# Model And Data

```yaml
view_type: stable_methodology
section: exp
status: active
maintenance_rule: edit_in_place_as_methodology_changes_not_per_cycle
grounded_in:
  - /root/LLM-output-density/experiments/opd_sft_h1/configs/trl_first_minimal.yaml
  - /root/LLM-output-density/experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py
  - /root/LLM-output-density/experiments/opd_sft_h1/scripts/run_opd_minimal_closure_v2.py
  - /root/LLM-output-density/mypaper/local_experiment_results/cycle_05_matched_control_id_ood/FINDING_05_gsm8k_chat_template_mismatch.md
last_organized: 2026-06-16
```

This file is the stable cross-cycle reference for "what models we use and what
format their inputs/outputs are in." It is edited in place when the
methodology actually changes, not restated per cycle. Cycle-specific choices
(which arms, which data sizes) live in `../current_experiment_design.md`.

## Model Identity

| Role | Checkpoint | Notes |
|---|---|---|
| Teacher | `/root/autodl-tmp/model/Qwen/Qwen3-4B` | Instruct/post-trained release. Used for OPD teacher feedback and for generating the `theta0`/cold-start `S`-probe rollouts. |
| Student / base | `/root/autodl-tmp/model/Qwen/Qwen3-1.7B` | **Instruct/post-trained release**, not the official pretrain-only `Qwen3-1.7B-Base`. The "untrained base" row in every result table is this instruct checkpoint, unfinetuned. This distinction matters for capability comparisons against Qwen's official base-model benchmark numbers (see `FINDING_05_gsm8k_chat_template_mismatch.md`). |
| Adapter method | LoRA, `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias=none`, `target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` | Same LoRA config for both OPD-like training and continued-SFT training, so geometry/training comparisons are not confounded by adapter capacity. |

## Chat / Think Format (the thing that caused the Cycle 04→05 measurement bug)

- All training data and all rollouts are built through the tokenizer's chat
  template (`apply_chat_template`). A generation or training example that
  skips the chat template is a **protocol bug**, not a valid ablation —
  this is exactly what happened to the Cycle 04 GSM8K/OOD-lite eval (see
  `evaluation_protocol.md`).
- The supervised targets (`train_sft.jsonl`, used both as continued-SFT data
  and as the teacher-completion targets that produce `theta0`) contain **zero**
  `<think>` tags. Every teacher completion goes problem → step-by-step
  reasoning → `\boxed{...}` directly, with no thinking-block wrapper.
- Therefore every model in this project is trained and must be evaluated in
  **non-thinking mode** (`enable_thinking=False`). Qwen3's chat template
  defaults to `enable_thinking=True`; leaving the default on at eval time
  re-introduces a train/eval format mismatch even after the chat-template bug
  is fixed (confirmed empirically: GSM8K stayed in the broken ~0.39-0.43 band
  with thinking left on, and only recovered to ~0.70-0.72 once
  `enable_thinking=False` was also applied — see
  `FINDING_05_gsm8k_chat_template_mismatch.md`).
- **Standing rule**: any new eval task added to this project must explicitly
  set `apply_chat_template=True` and `enable_thinking=False`. Do not assume a
  task's default behavior matches this; the GSM8K/MATH500 bugs were both
  silent defaults.

## Training / Probe Data

| Pool | Source | Role |
|---|---|---|
| `train.parquet` (NuminaMath-1.5) | `/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet` | Source of all training prompts: cold-start (`theta0`), OPD rollout prompts, and continued-SFT supervised targets. Built into one **unified prompt pool** per run (`build_unified_pool`, single seeded split) so cold-start / OPD / SFT / held-out-eval / GetSlice-probe prompts are drawn from disjoint, bias-controlled slices of the same pool rather than separately sampled subsets. |
| `test.jsonl` (NuminaMath-1.5) | `/root/autodl-tmp/prepared/NuminaMath-1___5/test.jsonl` | Held-out ID evaluation split. Verified disjoint from the training row IDs (`train_row_ids ∩ test_row_ids = 0`). This is the project's primary ID axis (see `evaluation_protocol.md`). |
| SFT supervised format | chat `messages` list (`role: user` / `role: assistant`), rendered via `apply_chat_template(..., add_generation_prompt=False)` | The assistant turn is the teacher's full non-thinking completion (problem → reasoning → `\boxed{}`). This is the literal text continued-SFT trains on and the literal text whose tokenizer-rendered form theta0's cold-start training also targets. |

---

## Cycle 07 Pivot (2026-06-24): Base Model + Think-Format + Long-CoT Data

**Effective from Cycle 07 onward, the prior methodology above is superseded for the main experimental line.** The instruct-series / NuminaMath-1.5 / no-think protocol remains valid for Cycle 04–06 historical results only.

### New Model Identity (Cycle 07+)

| Role | Checkpoint | Notes |
|---|---|---|
| Student / base | `/root/autodl-tmp/model/Qwen/Qwen3-4B-Base` | **Pretraining-only checkpoint** (no instruction tuning). This is the critical shift from Cycle 06: pretraining checkpoints have more headroom for long-CoT SFT acquisition than already-instruction-tuned models. |
| Teacher | `/root/autodl-tmp/model/Qwen/Qwen3-4B` | Instruct release. No longer used for training in Cycle 07 (SFT uses dataset-provided targets). Available as reference eval only. |
| Adapter | LoRA `r=32`, `lora_alpha=64`, `dropout=0.05`, `target_modules=[q/k/v/o/gate/up/down_proj]` | Upgraded from r=16/alpha=32 for the 4B model scale + complex reasoning. |

### New Training Format (Cycle 07+)

- **Think-format:** assistant training target is `<think>...reasoning...</think>answer\n\boxed{}`.
  This is a 180° reversal from Cycle 04–06 which explicitly suppressed think tags.
- **No `enable_thinking=False`** in eval — the model must produce think blocks naturally.
  Evaluation uses `max_new_tokens=32768` to accommodate long reasoning chains.
- **No system prompt** in training or eval.
- The standing rule from Cycle 04–06 ("always set `enable_thinking=False`") does NOT apply
  to Cycle 07+ — that rule was specific to the no-think training regime.

### New Training Data (Cycle 07+)

| Pool | Source | Role |
|---|---|---|
| Math-CoT-20k | `/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet` | Rethink SFT paper official training data. OpenR1-Math queries + Qwen3-32B think-format long-CoT responses + math-verify filter. ~20.5k rows. |
| Training subset | 5000 rows, `seed=42` from Math-CoT-20k | Sampled with `df.sample(n=5000, random_state=42)`. |

NuminaMath-1.5 is **no longer used for training** from Cycle 07 onward. It remains as an
eval task (NuminaMath-test held-out split) for ID-degradation tracking.

---

## Open / Unconfirmed Items

- The exact mathematical form of the OPD-like distillation loss (`lmbda`,
  `beta`, `loss_top_k` semantics) is set via TRL's `DistillationConfig` /
  `DistillationTrainer` (see `trl_runner.py`) but has not yet been transcribed
  into this workspace as an explicit formula. **Needs confirmation in a future
  code conversation** before any paper-facing claim describes the OPD loss
  mathematically — do not infer or invent the formula here.
