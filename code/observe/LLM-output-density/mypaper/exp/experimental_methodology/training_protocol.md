# Training Protocol

```yaml
view_type: stable_methodology
section: exp
status: active
maintenance_rule: edit_in_place_as_methodology_changes_not_per_cycle
grounded_in:
  - /root/LLM-output-density/experiments/opd_sft_h1/scripts/run_opd_stability_gain_cycle04.py
  - /root/LLM-output-density/experiments/opd_sft_h1/scripts/run_opd_minimal_closure_v2.py
  - /root/LLM-output-density/experiments/opd_sft_h1/opd_sft_h1/trl_runner.py
last_organized: 2026-06-16
```

This file describes how `theta0`, the OPD-like arms, and the continued-SFT
controls are actually produced. See `model_and_data.md` for model identity and
`evaluation_protocol.md` for how they are scored afterward.

## Training Arms

| Arm | What it is | Key config |
|---|---|---|
| `theta0` (cold start) | OPD-like training (`lmbda=1.0`) of the student on `n_cold=256` prompts, used as the common starting point for every OPD arm and (indirectly, via its merged weights) the comparison baseline for geometry drift. | `max_steps = n_cold // cold_steps_div` (=64 at `cold_steps_div=4`), `grad_accum=4` → reported `DataSize=256` consumed. |
| `opd_lmbda1` / `opd_lmbda05` | OPD-like training continuing from `theta0`'s merged checkpoint, `lmbda ∈ {1.0, 0.5}`, `seed=42`. | `opd_steps=200`, `opd_grad=4` → reported `DataSize=800` consumed (`200*4`). Two arms only differ in `lmbda`. |
| `sft_n{128,256,512,1024}` | Continued supervised fine-tuning from `theta0`'s merged checkpoint on `num_samples ∈ {128,256,512,1024}` supervised examples (chat-rendered teacher completions, see `model_and_data.md`). | `learning_rate=1e-5` (vs `3e-5` for OPD arms), `num_train_epochs=3`, cosine schedule, `warmup_ratio=0.05`, `load_best_model_at_end` on held-out `eval_loss`. |

All arms share the same LoRA adapter shape (`r=16, alpha=32, dropout=0.05`,
`q/k/v/o/gate/up/down_proj`) — see `model_and_data.md` — so geometry and
capability comparisons are not confounded by adapter capacity differences.

## OPD-Like Training Mechanics

- Implemented via TRL's on-policy-distillation trainer (`trl_runner.py`),
  configured per-arm by a generated YAML (see
  `_write_opd_arm_config` in `run_opd_stability_gain_cycle04.py`).
- `max_length=4096`, `max_completion_length=512`,
  `per_device_train_batch_size=1`, `bf16=True`, optimizer `adamw_torch`,
  `gradient_checkpointing=False`.
- **vLLM colocate acceleration**: when enabled (`use_vllm=True`, the default
  for non-smoke runs), the student rollout step (sampling completions during
  OPD training) runs through a same-process vLLM engine with continuous
  batching, rather than HuggingFace `generate`. Settings:
  - `vllm_mode: colocate`, `vllm_gpu_memory_utilization=0.30`,
    `vllm_max_model_length=4096`, `vllm_tensor_parallel_size=1`.
  - `vllm_enable_sleep_mode=True`: the vLLM engine sleeps during the optimizer
    step to free GPU memory for the teacher model and the backward pass.
  - `vllm_sync_frequency=1`: the LoRA weights are re-synced into the vLLM
    engine every optimizer step (so rollouts always reflect the latest
    policy).
  - Run on a single 48GB GPU, no OOM observed in Cycle 04.
- **Process isolation**: each OPD arm (and `theta0`) trains in an **isolated
  subprocess** (`run_trl_distill_smoke.py` invoked via `subprocess.run`), not
  inline in the orchestrating script. Reason: vLLM colocate initializes
  `torch.distributed` (NCCL); a second colocate training in the same process
  would collide with an already-destroyed/reused process group, and PEFT
  adapter loading collides with `EmbeddingParallel` while distributed is
  initialized. The subprocess exits and tears down its process group cleanly
  after each arm; LoRA-adapter merging happens back in the parent process
  (which never initializes distributed), so both steps stay clean.

## Continued-SFT Mechanics

- Implemented via TRL's standard `SFTTrainer`/`SFTConfig` (not the
  distillation trainer) — i.e. continued SFT is plain supervised
  next-token training on the chat-rendered teacher completions, not a
  teacher-feedback method.
- Starts from `theta0`'s merged checkpoint, same LoRA config as the OPD arms.
- `gradient_checkpointing=True` (OPD arms use `False`), `max_grad_norm=0.5`,
  `lr_scheduler_type=cosine`, `dataset_text_field="text"`, `packing=False`.
- Held-out evaluation (`eval_loss` on a fixed 64-example held-out split from
  the same unified pool) gates checkpoint selection
  (`load_best_model_at_end=True`, `metric_for_best_model=eval_loss`), which is
  what "matched-gain" selection in `evaluation_protocol.md` operates on top of.

---

## Cycle 07 Training Protocol (2026-06-24): Base SFT Trajectory

Cycle 07 is **SFT-only**, no OPD arm, no vLLM colocate, no subprocess isolation.

### Arms

Single arm: Qwen3-4B-Base → LoRA SFT on Math-CoT-20k (5k rows, think-format).

### Key Config

| Parameter | Value |
|---|---|
| Base model | Qwen3-4B-Base |
| LoRA | r=32, alpha=64, dropout=0.05, targets=q/k/v/o/gate/up/down_proj |
| LR | 5e-5, cosine, warmup_ratio=0.10, weight_decay=0.01 |
| Epochs | 8 |
| Effective bsz | 64 (per_device=1, grad_accum=64) |
| Total steps | ~624 |
| Max seq length | 19456 (max_prompt=3072 + max_response=16384) |
| Trainer | TRL SFTTrainer (standard, no distillation) |
| dtype | bf16 |

### Multi-Checkpoint Grid

Checkpoints saved at steps {5, 10, 20, 40, 80, 160, 320, 480, 624} via a custom
`TrainerCallback`. This is a **hard requirement** from Theory: multi-checkpoint sampling
is necessary to observe dip-and-recovery dynamics and associate geometry/response_length
with trajectory phase.

- LoRA adapters saved unmerged at each grid point.
- Merge-on-demand strategy: merge one checkpoint at a time for lm_eval, delete merged
  copy after eval to preserve disk.

### Relationship to Rethink SFT Reference

The original (Ren et al. 2026) used full-parameter fine-tuning with bsz=256. Our variant:
- LoRA (not full-param): faster convergence expected; dip may be shallower or earlier.
- bsz=64 (not 256): total steps ~624 ≈ paper's 640. Gradient magnitude not normalized by
  bsz under mean-reduction loss — bsz primarily affects stability, not optimization depth.
  Step count preserved intentionally to align trajectory with paper's Table 9 anchors.
- 5k samples (not 20k): fewer multi-epoch passes through the full distribution; dynamics
  may be compressed relative to paper's 640-step trajectory.

---

## Open / Unconfirmed Items

- Whether Cycle 03's training used the same vLLM-colocate / subprocess
  isolation pattern, or the earlier non-colocate path, has not been
  cross-checked against `run_opd_minimal_closure.py` (v1) line-by-line. The
  v1/v2 distinction is described in `run_opd_minimal_closure_v2.py`'s module
  docstring (5 methodology fixes) but a full diff has not been done for this
  document. Treat Cycle 03 process details as **lower-confidence** than
  Cycle 04/05 until checked.
