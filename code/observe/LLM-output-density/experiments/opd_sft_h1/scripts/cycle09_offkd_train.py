#!/usr/bin/env python3
"""Cycle 09 off-KD Stage 2: offline top-32 forward-KL LoRA training.

The response tokens and teacher distributions are static artifacts from
cycle09_offkd_rollout.py. This loop deliberately removes both online student
rollout and the live teacher server while preserving Cycle 08's student, LoRA,
optimizer, loss, batching order, epoch count, and checkpoint grid.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path("/root/LLM-output-density")
EXP_ROOT = Path("/root/autodl-tmp/cycle09_offkd")
STUDENT = Path("/root/autodl-tmp/model/Qwen/Qwen3-4B-Base")
ROLLOUT_ROOT = EXP_ROOT / "rollout"
COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/offkd"
)

SEED = 42
MAX_PROMPT_TOKENS = 1024
TOPK = 32
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
GRAD_CLIP = 1.0
LOGPROB_MIN = -10.0
LOSS_MAX = 10.0
CHECKPOINT_GRID = (0, 5, 10, 20, 40, 160, 624)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "checkpoints")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or a checkpoint directory containing trainer_state.pt",
    )
    parser.add_argument(
        "--checkpoint-grid",
        default=",".join(map(str, CHECKPOINT_GRID)),
        help="Comma-separated save steps; formal training uses the preregistered default.",
    )
    parser.add_argument(
        "--no-copyback",
        action="store_true",
        help="Keep metadata under output-dir (used by deterministic checkpoint backfills).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional early stop for engineering tests; formal training leaves this at zero.",
    )
    args = parser.parse_args()
    args.checkpoint_grid = tuple(
        int(item.strip()) for item in args.checkpoint_grid.split(",") if item.strip()
    )
    if not args.checkpoint_grid or any(step < 0 or step > 624 for step in args.checkpoint_grid):
        raise ValueError(f"Invalid checkpoint grid: {args.checkpoint_grid}")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_artifacts(
    rollout_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    records = read_jsonl(rollout_dir / "teacher_rollout.jsonl")
    with np.load(rollout_dir / "teacher_top32_logprob.npz") as archive:
        top_ids = archive["top32_ids"]
        top_logprobs = archive["top32_logprob"]
        offsets = archive["row_offsets"]
    return records, top_ids, top_logprobs, offsets


def validate_artifacts(
    rollout_dir: Path,
    records: list[dict[str, Any]],
    top_ids: np.ndarray,
    top_logprobs: np.ndarray,
    offsets: np.ndarray,
    *,
    smoke: bool,
) -> tuple[list[int], dict[str, Any]]:
    expected_records = 8 if smoke else 5000
    errors: list[str] = []
    if len(records) != expected_records:
        errors.append(f"records={len(records)} expected={expected_records}")
    if offsets.shape != (len(records), 2):
        errors.append(f"offsets_shape={offsets.shape} expected=({len(records)}, 2)")
    if top_ids.shape != top_logprobs.shape:
        errors.append(f"ids_shape={top_ids.shape} logprob_shape={top_logprobs.shape}")
    if top_ids.ndim != 2 or top_ids.shape[1] != TOPK:
        errors.append(f"topk_shape={top_ids.shape} expected=(*, {TOPK})")

    total_tokens = sum(int(record["n_tokens"]) for record in records)
    if top_ids.shape[0] != total_tokens:
        errors.append(f"topk_rows={top_ids.shape[0]} total_tokens={total_tokens}")

    cursor = 0
    for row, record in enumerate(records):
        start, end = map(int, offsets[row])
        expected_end = cursor + int(record["n_tokens"])
        if (start, end) != (cursor, expected_end):
            errors.append(
                f"row={row} offset=({start},{end}) expected=({cursor},{expected_end})"
            )
            break
        if (int(record["logprob_row_start"]), int(record["logprob_row_end"])) != (
            start,
            end,
        ):
            errors.append(f"row={row} JSON/NPZ offset mismatch")
            break
        if len(record["generation_token_ids"]) != int(record["n_tokens"]):
            errors.append(f"row={row} generation length mismatch")
            break
        if len(record["prompt_token_ids"]) != int(record["n_prompt_tokens"]):
            errors.append(f"row={row} prompt length mismatch")
            break
        cursor = end

    chunk = 250_000
    missing_ids = 0
    nonfinite = 0
    positive = 0
    unsorted = 0
    for start in range(0, top_ids.shape[0], chunk):
        end = min(start + chunk, top_ids.shape[0])
        ids_part = top_ids[start:end]
        lp_part = top_logprobs[start:end]
        missing_ids += int(np.count_nonzero(ids_part < 0))
        nonfinite += int(np.count_nonzero(~np.isfinite(lp_part)))
        positive += int(np.count_nonzero(lp_part > 1e-5))
        unsorted += int(np.count_nonzero(np.diff(lp_part, axis=1) > 1e-5))
    if missing_ids:
        errors.append(f"missing_token_ids={missing_ids}")
    if nonfinite:
        errors.append(f"nonfinite_logprobs={nonfinite}")
    if positive:
        errors.append(f"positive_logprobs={positive}")
    if unsorted:
        errors.append(f"unsorted_adjacent_pairs={unsorted}")

    pass1_path = rollout_dir / "teacher_rollout_pass1.jsonl"
    pass1_exact = None
    if pass1_path.exists():
        pass1 = read_jsonl(pass1_path)
        pass1_exact = len(pass1) == len(records) and all(
            before["prompt_token_ids"] == after["prompt_token_ids"]
            and before["generation_token_ids"] == after["generation_token_ids"]
            for before, after in zip(pass1, records)
        )
        if not pass1_exact:
            errors.append("pass1/final token sequences differ")

    eligible = [
        row
        for row, record in enumerate(records)
        if int(record["n_prompt_tokens"]) <= MAX_PROMPT_TOKENS
    ]
    if not smoke and len(eligible) != 4999:
        errors.append(f"eligible_prompts={len(eligible)} expected=4999")

    report = {
        "status": "pass" if not errors else "fail",
        "rollout_dir": str(rollout_dir),
        "n_records": len(records),
        "total_response_tokens": total_tokens,
        "topk_shape": list(top_ids.shape),
        "eligible_prompt_count": len(eligible),
        "overlong_prompt_count": len(records) - len(eligible),
        "pass1_final_tokens_exact": pass1_exact,
        "missing_token_ids": missing_ids,
        "nonfinite_logprobs": nonfinite,
        "positive_logprobs": positive,
        "unsorted_adjacent_pairs": unsorted,
        "errors": errors,
    }
    report_path = rollout_dir / "pass2_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError("pass2 validation failed: " + "; ".join(errors[:8]))
    return eligible, report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "trainer_state.pt").exists():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume(resume: str, output_dir: Path) -> Path | None:
    if resume == "none":
        return None
    if resume == "auto":
        return latest_checkpoint(output_dir)
    path = Path(resume)
    if not (path / "trainer_state.pt").exists():
        raise FileNotFoundError(f"resume checkpoint is incomplete: {path}")
    return path


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
    }
    torch.save(state, checkpoint / "trainer_state.pt")
    (checkpoint / "complete.json").write_text(
        json.dumps({"step": step, "saved_at_unix": time.time()}, indent=2),
        encoding="utf-8",
    )
    return checkpoint


def build_model(resume_path: Path | None) -> torch.nn.Module:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        STUDENT,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    if resume_path is None:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            target_modules="all-linear",
            bias="none",
            lora_dropout=0.0,
        )
        model = get_peft_model(base, config)
    else:
        model = PeftModel.from_pretrained(base, resume_path, is_trainable=True)

    # Cycle 08's FSDP builder explicitly casts trainable adapters to the bf16
    # base dtype before wrapping (transformer_impl.py:346-360).
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.to("cuda")
    model.train()
    return model


def sample_loss(
    model: torch.nn.Module,
    record: dict[str, Any],
    teacher_ids_np: np.ndarray,
    teacher_logprobs_np: np.ndarray,
) -> tuple[torch.Tensor, dict[str, float]]:
    prompt_ids = record["prompt_token_ids"]
    generation_ids = record["generation_token_ids"]
    sequence = torch.tensor(
        prompt_ids + generation_ids, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    prompt_length = len(prompt_ids)
    response_length = len(generation_ids)
    prediction_positions = torch.arange(
        prompt_length - 1,
        prompt_length + response_length - 1,
        dtype=torch.long,
        device="cuda",
    )
    output = model(
        input_ids=sequence,
        use_cache=False,
        logits_to_keep=prediction_positions,
    )
    student_logits = output.logits[0]
    if student_logits.shape[0] != response_length:
        raise RuntimeError(
            f"student logits rows={student_logits.shape[0]} response={response_length}"
        )

    teacher_ids = torch.as_tensor(
        teacher_ids_np.astype(np.int64, copy=False), dtype=torch.long, device="cuda"
    )
    teacher_logprobs = torch.as_tensor(
        teacher_logprobs_np, dtype=torch.float32, device="cuda"
    )

    # Cycle 08 default path: log_softmax, gather teacher ids, clamp, then KL.
    student_logprobs = F.log_softmax(student_logits, dim=-1)
    student_topk = torch.gather(student_logprobs, dim=-1, index=teacher_ids)
    student_mass = student_topk.exp().sum(dim=-1)
    teacher_mass = teacher_logprobs.exp().sum(dim=-1)
    student_topk = student_topk.clamp_min(LOGPROB_MIN).float()
    teacher_clamped = teacher_logprobs.clamp_min(LOGPROB_MIN)
    token_loss = (
        teacher_clamped.exp() * (teacher_clamped - student_topk)
    ).sum(dim=-1)
    token_loss = token_loss.clamp_min(0.0).clamp_max(LOSS_MAX)
    metrics = {
        "loss_sum": float(token_loss.detach().sum().item()),
        "student_mass_sum": float(student_mass.detach().float().sum().item()),
        "teacher_mass_sum": float(teacher_mass.detach().float().sum().item()),
    }
    return token_loss.sum(), metrics


def train(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    top_ids: np.ndarray,
    top_logprobs: np.ndarray,
    offsets: np.ndarray,
    eligible: list[int],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for off-KD training")
    set_seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1 if args.smoke else BATCH_SIZE
    epochs = 1 if args.smoke else EPOCHS
    if args.smoke:
        eligible = [max(eligible, key=lambda row: int(records[row]["n_tokens"]))]
    steps_per_epoch = len(eligible) // batch_size
    total_steps = steps_per_epoch * epochs
    if not args.smoke and total_steps != 624:
        raise RuntimeError(
            f"formal schedule produced {total_steps} steps, expected 624 "
            f"({len(eligible)} eligible / {batch_size} * {epochs})"
        )
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    resume_path = resolve_resume(args.resume, args.output_dir)
    model = build_model(resume_path)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    start_step = 1
    if resume_path is not None:
        state = torch.load(
            resume_path / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"[offkd-train] resumed from {resume_path} at step {start_step}", flush=True)
    else:
        saved = save_checkpoint(model, optimizer, args.output_dir, 0)
        print(f"[offkd-train] saved initial adapter -> {saved}", flush=True)

    trainable_count = sum(parameter.numel() for parameter in trainable)
    manifest = {
        "schema_version": 1,
        "status": "running",
        "implementation": "standalone HF+PEFT offline forward-KL loop",
        "stage2_spec_status": (
            "No separate Stage-2 handoff exists in the repository. Parameters are "
            "sourced from offkd_rollout_handoff.md's closing summary and Cycle 08's "
            "formal Hydra config (outputs/2026-07-02/10-40-01/.hydra/config.yaml)."
        ),
        "only_experimental_variable": (
            "static teacher-generated responses replace online student rollouts"
        ),
        "student_model": str(STUDENT),
        "rollout_dir": str(args.rollout_dir),
        "seed": SEED,
        "eligible_prompts": len(eligible) if not args.smoke else 1,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "shuffle": False,
        "drop_last": True,
        "lora": {
            "r": 32,
            "alpha": 64,
            "target_modules": "all-linear",
            "dropout": 0.0,
            "bias": "none",
            "adapter_dtype": "bfloat16 (matches Cycle 08 FSDP builder)",
            "trainable_parameters": trainable_count,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": LEARNING_RATE,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "scheduler": "constant",
            "warmup_steps": 0,
            "clip_grad_norm": GRAD_CLIP,
        },
        "loss": {
            "name": "forward_kl_topk",
            "topk": TOPK,
            "aggregation": "token-mean over each global batch",
            "student_path": "bf16 F.log_softmax then gather teacher token ids",
            "teacher_logprobs": "raw temperature=1.0",
            "logprob_min_clamp": LOGPROB_MIN,
            "token_loss_min_clamp": 0.0,
            "loss_max_clamp": LOSS_MAX,
        },
        "checkpoint_grid": list(args.checkpoint_grid if not args.smoke else (0, 1)),
        "resume_from": str(resume_path) if resume_path else None,
        "started_at_unix": time.time(),
    }
    atomic_json(args.output_dir / "training_manifest.json", manifest)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    if resume_path is not None and metrics_path.exists():
        # A crash can happen after a metric row but before the next landmark
        # checkpoint. Keep the log aligned with the state we actually resumed.
        completed_step = start_step - 1
        retained = [
            line
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(json.loads(line)["step"]) <= completed_step
        ]
        metrics_path.write_text(
            ("\n".join(retained) + "\n") if retained else "", encoding="utf-8"
        )

    if start_step > total_steps:
        manifest["status"] = "complete"
        manifest["completed_steps"] = total_steps
        manifest["completed_at_unix"] = time.time()
        atomic_json(args.output_dir / "training_manifest.json", manifest)
        print("[offkd-train] requested schedule already complete", flush=True)
        return

    run_started = time.time()
    for global_step in range(start_step, total_steps + 1):
        torch.cuda.reset_peak_memory_stats()
        step_started = time.time()
        epoch = (global_step - 1) // steps_per_epoch
        batch_in_epoch = (global_step - 1) % steps_per_epoch
        begin = batch_in_epoch * batch_size
        batch_rows = eligible[begin : begin + batch_size]
        token_denominator = sum(int(records[row]["n_tokens"]) for row in batch_rows)
        if token_denominator <= 0:
            raise RuntimeError(f"empty response batch at step {global_step}")

        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        student_mass_sum = 0.0
        teacher_mass_sum = 0.0
        for sample_number, row in enumerate(batch_rows, start=1):
            record = records[row]
            offset_start, offset_end = map(int, offsets[row])
            sample_loss_sum, sample_metrics = sample_loss(
                model,
                record,
                top_ids[offset_start:offset_end],
                top_logprobs[offset_start:offset_end],
            )
            (sample_loss_sum / token_denominator).backward()
            loss_sum += sample_metrics["loss_sum"]
            student_mass_sum += sample_metrics["student_mass_sum"]
            teacher_mass_sum += sample_metrics["teacher_mass_sum"]
            del sample_loss_sum
            print(
                f"[offkd-train] step={global_step}/{total_steps} "
                f"sample={sample_number}/{batch_size} row={row} "
                f"tokens={record['n_tokens']}",
                flush=True,
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        elapsed = time.time() - step_started
        metric = {
            "step": global_step,
            "epoch_zero_based": epoch,
            "batch_in_epoch_zero_based": batch_in_epoch,
            "response_tokens": token_denominator,
            "loss": loss_sum / token_denominator,
            "student_top32_mass": student_mass_sum / token_denominator,
            "teacher_top32_mass": teacher_mass_sum / token_denominator,
            "grad_norm_before_clip": float(grad_norm),
            "seconds": elapsed,
            "tokens_per_second": token_denominator / elapsed,
            "gpu_max_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
            handle.flush()
        print("[offkd-train] " + json.dumps(metric), flush=True)

        save_grid = (0, 1) if args.smoke else args.checkpoint_grid
        if global_step in save_grid:
            saved = save_checkpoint(model, optimizer, args.output_dir, global_step)
            print(f"[offkd-train] checkpoint -> {saved}", flush=True)
        manifest["completed_steps"] = global_step
        manifest["last_step_seconds"] = elapsed
        atomic_json(args.output_dir / "training_manifest.json", manifest)

    manifest["status"] = "complete"
    manifest["completed_steps"] = total_steps
    manifest["training_seconds_this_invocation"] = time.time() - run_started
    manifest["completed_at_unix"] = time.time()
    atomic_json(args.output_dir / "training_manifest.json", manifest)
    print(f"[offkd-train] complete: {total_steps} steps", flush=True)

    if not args.smoke and not args.no_copyback:
        COPYBACK.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output_dir / "training_manifest.json", COPYBACK)
        shutil.copy2(metrics_path, COPYBACK)
        validation = args.rollout_dir / "pass2_validation.json"
        if validation.exists():
            shutil.copy2(validation, COPYBACK)
        print(f"[offkd-train] metadata copyback -> {COPYBACK}", flush=True)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.rollout_dir = EXP_ROOT / "smoke/rollout"
        args.output_dir = EXP_ROOT / "smoke/train_pass2"
    records, top_ids, top_logprobs, offsets = load_artifacts(args.rollout_dir)
    eligible, _ = validate_artifacts(
        args.rollout_dir,
        records,
        top_ids,
        top_logprobs,
        offsets,
        smoke=args.smoke,
    )
    if args.validate_only:
        return
    train(args, records, top_ids, top_logprobs, offsets, eligible)


if __name__ == "__main__":
    main()
