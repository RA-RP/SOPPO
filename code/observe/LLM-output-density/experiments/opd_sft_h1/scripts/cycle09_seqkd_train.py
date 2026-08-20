#!/usr/bin/env python3
"""Cycle 09 block 2 G1: offline seqKD LoRA training on teacher responses."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import cycle09_offkd_train as offkd


REPO = Path("/root/LLM-output-density")
EXP_ROOT = Path("/root/autodl-tmp/cycle09_seqkd")
ROLLOUT_DIR = Path("/root/autodl-tmp/cycle09_offkd/rollout")
COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/seqkd"
)
CHECKPOINT_GRID = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, default=ROLLOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "checkpoints")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume",
        default="auto",
        help="'auto', 'none', or a checkpoint directory with trainer_state.pt",
    )
    parser.add_argument(
        "--checkpoint-grid", default=",".join(map(str, CHECKPOINT_GRID))
    )
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-copyback", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="sdpa",
        help="PyTorch SDPA is the portable formal default on this machine.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory; disabled by default on the 96 GiB GPU.",
    )
    args = parser.parse_args()
    args.checkpoint_grid = tuple(
        int(item.strip()) for item in args.checkpoint_grid.split(",") if item.strip()
    )
    if not args.checkpoint_grid or any(step < 0 or step > 624 for step in args.checkpoint_grid):
        parser.error(f"invalid checkpoint grid: {args.checkpoint_grid}")
    return args


def validate_records(
    rollout_dir: Path, records: list[dict[str, Any]], smoke: bool
) -> tuple[list[int], dict[str, Any]]:
    expected = 8 if smoke and "smoke" in str(rollout_dir) else 5000
    errors: list[str] = []
    if len(records) != expected:
        errors.append(f"records={len(records)} expected={expected}")

    total_tokens = 0
    max_token_id = -1
    for row, record in enumerate(records):
        required = {
            "prompt_token_ids",
            "generation_token_ids",
            "n_prompt_tokens",
            "n_tokens",
        }
        missing = required - set(record)
        if missing:
            errors.append(f"row={row} missing={sorted(missing)}")
            break
        prompt_ids = record["prompt_token_ids"]
        generation_ids = record["generation_token_ids"]
        if len(prompt_ids) != int(record["n_prompt_tokens"]):
            errors.append(f"row={row} prompt length mismatch")
            break
        if len(generation_ids) != int(record["n_tokens"]):
            errors.append(f"row={row} generation length mismatch")
            break
        if not prompt_ids or not generation_ids:
            errors.append(f"row={row} empty prompt or generation")
            break
        total_tokens += len(generation_ids)
        max_token_id = max(max_token_id, max(prompt_ids), max(generation_ids))

    eligible = [
        row
        for row, record in enumerate(records)
        if int(record.get("n_prompt_tokens", offkd.MAX_PROMPT_TOKENS + 1))
        <= offkd.MAX_PROMPT_TOKENS
    ]
    if expected == 5000 and len(eligible) != 4999:
        errors.append(f"eligible_prompts={len(eligible)} expected=4999")

    report = {
        "status": "pass" if not errors else "fail",
        "rollout_dir": str(rollout_dir),
        "n_records": len(records),
        "eligible_prompt_count": len(eligible),
        "overlong_prompt_count": len(records) - len(eligible),
        "total_response_tokens": total_tokens,
        "max_token_id": max_token_id,
        "question_tokens_used_as_labels": False,
        "errors": errors,
    }
    report_path = rollout_dir / "seqkd_input_validation.json"
    offkd.atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError("seqKD input validation failed: " + "; ".join(errors))
    return eligible, report


def response_ce(
    model: torch.nn.Module, record: dict[str, Any]
) -> tuple[torch.Tensor, int]:
    prompt_ids = record["prompt_token_ids"]
    response_ids = record["generation_token_ids"]
    sequence = torch.tensor(
        prompt_ids + response_ids, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    prompt_length = len(prompt_ids)
    response_length = len(response_ids)
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
    logits = output.logits[0]
    if logits.shape[0] != response_length:
        raise RuntimeError(
            f"student logits rows={logits.shape[0]} response={response_length}"
        )
    targets = torch.as_tensor(response_ids, dtype=torch.long, device="cuda")
    loss_sum = F.cross_entropy(logits.float(), targets, reduction="sum")
    return loss_sum, response_length


def build_model(
    resume_path: Path | None,
    attn_implementation: str,
    gradient_checkpointing: bool,
) -> torch.nn.Module:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        offkd.STUDENT,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    if gradient_checkpointing:
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
    if resume_path is None:
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=32,
                lora_alpha=64,
                target_modules="all-linear",
                bias="none",
                lora_dropout=0.0,
            ),
        )
    else:
        model = PeftModel.from_pretrained(base, resume_path, is_trainable=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    model.to("cuda")
    model.train()
    return model


def train(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    eligible: list[int],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for seqKD training")
    offkd.set_seed(offkd.SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = 1 if args.smoke else offkd.BATCH_SIZE
    epochs = 1 if args.smoke else offkd.EPOCHS
    if args.smoke:
        eligible = [max(eligible, key=lambda row: int(records[row]["n_tokens"]))]
    steps_per_epoch = len(eligible) // batch_size
    scheduled_steps = steps_per_epoch * epochs
    if not args.smoke and scheduled_steps != 624:
        raise RuntimeError(
            f"formal schedule produced {scheduled_steps} steps, expected 624"
        )
    total_steps = min(scheduled_steps, args.max_steps) if args.max_steps else scheduled_steps

    resume_path = offkd.resolve_resume(args.resume, args.output_dir)
    model = build_model(
        resume_path,
        args.attn_implementation,
        args.gradient_checkpointing,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=offkd.LEARNING_RATE,
        betas=offkd.BETAS,
        weight_decay=offkd.WEIGHT_DECAY,
    )
    start_step = 1
    if resume_path is None:
        saved = offkd.save_checkpoint(model, optimizer, args.output_dir, 0)
        print(f"[seqkd-train] initial adapter -> {saved}", flush=True)
    else:
        state = torch.load(
            resume_path / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"[seqkd-train] resume {resume_path} at step {start_step}", flush=True)

    trainable_count = sum(parameter.numel() for parameter in trainable)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "task": "Cycle 09 block 2 G1 seqKD",
        "implementation": "standalone HF+PEFT offline response CE loop",
        "only_experimental_variable_vs_offkd": "CE replaces forward-KL top-32",
        "student_model": str(offkd.STUDENT),
        "attention_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "rollout_dir": str(args.rollout_dir),
        "seed": offkd.SEED,
        "eligible_prompts": len(eligible),
        "max_prompt_tokens": offkd.MAX_PROMPT_TOKENS,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "scheduled_steps": scheduled_steps,
        "total_steps_this_run": total_steps,
        "shuffle": False,
        "drop_last": True,
        "lora": {
            "r": 32,
            "alpha": 64,
            "target_modules": "all-linear",
            "dropout": 0.0,
            "bias": "none",
            "adapter_dtype": "bfloat16",
            "trainable_parameters": trainable_count,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "lr": offkd.LEARNING_RATE,
            "betas": list(offkd.BETAS),
            "weight_decay": offkd.WEIGHT_DECAY,
            "scheduler": "constant",
            "warmup_steps": 0,
            "clip_grad_norm": offkd.GRAD_CLIP,
        },
        "loss": {
            "name": "standard_cross_entropy",
            "labels": "teacher generation tokens only",
            "question_masked": True,
            "aggregation": "token-mean over each global batch",
            "logits_dtype_for_ce": "float32",
        },
        "checkpoint_grid": list((0, 1) if args.smoke else args.checkpoint_grid),
        "resume_from": str(resume_path) if resume_path else None,
        "started_at_unix": time.time(),
    }
    offkd.atomic_json(args.output_dir / "training_manifest.json", manifest)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    if resume_path is not None and metrics_path.exists():
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
        offkd.atomic_json(args.output_dir / "training_manifest.json", manifest)
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
        loss_sum_value = 0.0
        for sample_number, row in enumerate(batch_rows, start=1):
            sample_loss_sum, response_length = response_ce(model, records[row])
            (sample_loss_sum / token_denominator).backward()
            loss_sum_value += float(sample_loss_sum.detach().item())
            del sample_loss_sum
            print(
                f"[seqkd-train] step={global_step}/{total_steps} "
                f"sample={sample_number}/{batch_size} row={row} "
                f"tokens={response_length}",
                flush=True,
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, offkd.GRAD_CLIP)
        optimizer.step()
        elapsed = time.time() - step_started
        metric = {
            "step": global_step,
            "epoch_zero_based": epoch,
            "batch_in_epoch_zero_based": batch_in_epoch,
            "response_tokens": token_denominator,
            "loss": loss_sum_value / token_denominator,
            "grad_norm_before_clip": float(grad_norm),
            "seconds": elapsed,
            "tokens_per_second": token_denominator / elapsed,
            "gpu_max_memory_gib": torch.cuda.max_memory_allocated() / (1024**3),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
            handle.flush()
        print("[seqkd-train] " + json.dumps(metric), flush=True)

        save_grid = (0, 1) if args.smoke else args.checkpoint_grid
        if global_step in save_grid:
            saved = offkd.save_checkpoint(model, optimizer, args.output_dir, global_step)
            print(f"[seqkd-train] checkpoint -> {saved}", flush=True)
        manifest["completed_steps"] = global_step
        manifest["last_step_seconds"] = elapsed
        offkd.atomic_json(args.output_dir / "training_manifest.json", manifest)

    manifest["status"] = "complete"
    manifest["completed_steps"] = total_steps
    manifest["training_seconds_this_invocation"] = time.time() - run_started
    manifest["completed_at_unix"] = time.time()
    offkd.atomic_json(args.output_dir / "training_manifest.json", manifest)
    print(f"[seqkd-train] complete: {total_steps} steps", flush=True)

    if not args.smoke and not args.no_copyback:
        COPYBACK.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output_dir / "training_manifest.json", COPYBACK)
        shutil.copy2(metrics_path, COPYBACK)
        validation = args.rollout_dir / "seqkd_input_validation.json"
        if validation.exists():
            shutil.copy2(validation, COPYBACK)


def main() -> None:
    args = parse_args()
    records = offkd.read_jsonl(args.rollout_dir / "teacher_rollout.jsonl")
    eligible, _ = validate_records(args.rollout_dir, records, args.smoke)
    if args.validate_only:
        return
    train(args, records, eligible)


if __name__ == "__main__":
    main()
