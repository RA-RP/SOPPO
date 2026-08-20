#!/usr/bin/env python3
"""Cycle 09 block 2 G6: three matched Llama offline LoRA arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO = Path("/root/LLM-output-density")
ROOT = Path("/root/autodl-tmp/cycle09_block2/model2_llama")
BASE = Path("/root/autodl-tmp/model/Meta/modelscope/Llama-3.2-3B")
TEACHER = Path(
    "/root/autodl-tmp/model/Meta/modelscope/Meta-Llama-3.1-8B-Instruct"
)
SOURCE_PARQUET = Path("/root/autodl-tmp/dataset/Math-CoT-20k/Math-CoT-20k.parquet")
PROMPTS_PARQUET = Path(
    "/root/autodl-tmp/cycle08_opd_trajectory/data/opd_prompts_5k.parquet"
)
FORMAL_ROLLOUT = ROOT / "rollout"
COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/model2_llama"
)

SEED = 42
MAX_PROMPT_TOKENS = 1024
MAX_RESPONSE_TOKENS = 10240
TOPK = 32
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 5e-5
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOGPROB_MIN = -10.0
LOSS_MAX = 10.0
CHECKPOINT_GRID = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("sft", "offkd", "seqkd"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    branch = "smoke/g6" if args.smoke else "g6"
    args.output_dir = ROOT / branch / args.arm / "checkpoints"
    args.rollout_dir = ROOT / ("smoke/rollout" if args.smoke else "rollout")
    args.sft_cache = ROOT / ("smoke/sft_data" if args.smoke else "sft_data")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_sft(cache: Path, smoke: bool) -> list[dict[str, Any]]:
    records_path = cache / "records.jsonl"
    manifest_path = cache / "manifest.json"
    expected = 8 if smoke else 5000
    if records_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("n_records", -1)) != expected:
            raise RuntimeError(f"stale SFT cache: {manifest_path}")
        return read_jsonl(records_path)

    from transformers import AutoTokenizer

    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    frame = pd.read_parquet(SOURCE_PARQUET)
    sampled = frame.sample(n=5000, random_state=SEED).reset_index(drop=True)
    if smoke:
        sampled = sampled.iloc[:expected]

    prompt_frame = pd.read_parquet(PROMPTS_PARQUET)
    if not smoke and len(prompt_frame) != 5000:
        raise RuntimeError(f"prompt parquet rows={len(prompt_frame)}")

    records: list[dict[str, Any]] = []
    truncated = 0
    for index, row in sampled.iterrows():
        question = str(row["question"])
        response = str(row["response"])
        user = {"role": "user", "content": question + INSTR}
        prompt_ids = teacher_tokenizer.apply_chat_template(
            [user], tokenize=True, add_generation_prompt=True
        )
        full_ids = teacher_tokenizer.apply_chat_template(
            [user, {"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError(f"SFT prompt is not a full-sequence prefix at row={index}")
        generation_ids = full_ids[len(prompt_ids) :]
        original_length = len(generation_ids)
        if original_length > MAX_RESPONSE_TOKENS:
            generation_ids = generation_ids[:MAX_RESPONSE_TOKENS]
            truncated += 1
        records.append(
            {
                "index": int(index),
                "data_source": "Math-CoT-20k",
                "prompt_token_ids": prompt_ids,
                "generation_token_ids": generation_ids,
                "n_prompt_tokens": len(prompt_ids),
                "n_tokens": len(generation_ids),
                "original_response_tokens": original_length,
                "finish_reason": "length" if original_length > MAX_RESPONSE_TOKENS else "data_eos",
            }
        )

    if not smoke:
        for index in range(min(32, len(records))):
            parquet_prompt = prompt_frame.iloc[index]["prompt"]
            messages = parquet_prompt.tolist() if hasattr(parquet_prompt, "tolist") else list(parquet_prompt)
            expected_content = str(messages[0]["content"])
            source_content = str(sampled.iloc[index]["question"]) + INSTR
            if expected_content != source_content:
                raise RuntimeError(f"SFT/OPD sampled question mismatch at row={index}")

    cache.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    lengths = np.asarray([record["n_tokens"] for record in records])
    manifest = {
        "status": "complete",
        "n_records": len(records),
        "source": str(SOURCE_PARQUET),
        "selection": "pandas sample(n=5000, random_state=42), identical to Cycle 07/08",
        "teacher_chat_template_source": str(TEACHER / "tokenizer_config.json"),
        "student_tokenizer_sha256": sha256_file(BASE / "tokenizer.json"),
        "teacher_tokenizer_sha256": sha256_file(TEACHER / "tokenizer.json"),
        "question_masked": True,
        "response_cap": MAX_RESPONSE_TOKENS,
        "n_truncated": truncated,
        "truncation_rate": truncated / len(records),
        "response_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        },
        "records": str(records_path),
    }
    if manifest["student_tokenizer_sha256"] != manifest["teacher_tokenizer_sha256"]:
        raise RuntimeError("Llama teacher/student tokenizer mismatch")
    atomic_json(manifest_path, manifest)
    return records


def load_teacher_records(rollout_dir: Path) -> list[dict[str, Any]]:
    return read_jsonl(rollout_dir / "teacher_rollout.jsonl")


def validate_records(
    records: list[dict[str, Any]], *, smoke: bool, arm: str
) -> tuple[list[int], dict[str, Any]]:
    expected = 8 if smoke else 5000
    errors: list[str] = []
    if len(records) != expected:
        errors.append(f"records={len(records)} expected={expected}")
    total_tokens = 0
    max_token_id = -1
    for row, record in enumerate(records):
        prompt_ids = record["prompt_token_ids"]
        generation_ids = record["generation_token_ids"]
        if len(prompt_ids) != int(record["n_prompt_tokens"]):
            errors.append(f"row={row} prompt length mismatch")
            break
        if len(generation_ids) != int(record["n_tokens"]):
            errors.append(f"row={row} response length mismatch")
            break
        if not generation_ids or len(generation_ids) > MAX_RESPONSE_TOKENS:
            errors.append(f"row={row} invalid response length={len(generation_ids)}")
            break
        total_tokens += len(generation_ids)
        max_token_id = max(max_token_id, max(prompt_ids), max(generation_ids))
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    if max_token_id >= int(config["vocab_size"]):
        errors.append(f"max token id {max_token_id} >= vocab {config['vocab_size']}")
    eligible = [
        row
        for row, record in enumerate(records)
        if int(record["n_prompt_tokens"]) <= MAX_PROMPT_TOKENS
    ]
    if not smoke and len(eligible) < 4992:
        errors.append(f"eligible={len(eligible)} below 4992 needed for 624 steps")
    report = {
        "status": "pass" if not errors else "fail",
        "arm": arm,
        "n_records": len(records),
        "eligible_prompt_count": len(eligible),
        "overlong_prompt_count": len(records) - len(eligible),
        "total_response_tokens": total_tokens,
        "max_token_id": max_token_id,
        "question_tokens_used_as_labels": False,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("record validation failed: " + "; ".join(errors))
    return eligible, report


def load_top32(
    rollout_dir: Path, records: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    with np.load(rollout_dir / "teacher_top32_logprob.npz") as archive:
        top_ids = archive["top32_ids"]
        top_logprobs = archive["top32_logprob"]
        offsets = archive["row_offsets"]
    errors: list[str] = []
    total_tokens = sum(int(record["n_tokens"]) for record in records)
    if top_ids.shape != (total_tokens, TOPK):
        errors.append(f"ids shape={top_ids.shape} expected=({total_tokens},{TOPK})")
    if top_logprobs.shape != top_ids.shape:
        errors.append(f"logprobs shape={top_logprobs.shape}")
    if offsets.shape != (len(records), 2):
        errors.append(f"offsets shape={offsets.shape}")
    cursor = 0
    for row, record in enumerate(records):
        start, end = map(int, offsets[row])
        expected_end = cursor + int(record["n_tokens"])
        if (start, end) != (cursor, expected_end):
            errors.append(f"offset mismatch row={row}")
            break
        cursor = end
    for start in range(0, total_tokens, 250_000):
        end = min(start + 250_000, total_tokens)
        ids = top_ids[start:end]
        logprobs = top_logprobs[start:end]
        if np.any(ids < 0) or not np.isfinite(logprobs).all():
            errors.append(f"invalid top32 values in rows {start}:{end}")
            break
        if np.any(np.diff(logprobs, axis=1) > 1e-5):
            errors.append(f"unsorted top32 values in rows {start}:{end}")
            break
    report = {
        "status": "pass" if not errors else "fail",
        "shape": list(top_ids.shape),
        "offsets_shape": list(offsets.shape),
        "raw_temperature": 1.0,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("top32 validation failed: " + "; ".join(errors))
    return top_ids, top_logprobs, offsets, report


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match and (path / "trainer_state.pt").is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def resolve_resume(value: str, output_dir: Path) -> Path | None:
    if value == "none":
        return None
    if value == "auto":
        return latest_checkpoint(output_dir)
    path = Path(value)
    if not (path / "trainer_state.pt").is_file():
        raise FileNotFoundError(path / "trainer_state.pt")
    return path


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    step: int,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{step:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        checkpoint / "trainer_state.pt",
    )
    atomic_json(
        checkpoint / "complete.json",
        {"step": step, "saved_at_unix": time.time()},
    )
    return checkpoint


def build_model(resume_path: Path | None, gradient_checkpointing: bool) -> torch.nn.Module:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
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


def model_response_logits(
    model: torch.nn.Module, record: dict[str, Any]
) -> torch.Tensor:
    prompt_ids = record["prompt_token_ids"]
    generation_ids = record["generation_token_ids"]
    sequence = torch.as_tensor(
        prompt_ids + generation_ids, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    prompt_length = len(prompt_ids)
    response_length = len(generation_ids)
    positions = torch.arange(
        prompt_length - 1,
        prompt_length + response_length - 1,
        dtype=torch.long,
        device="cuda",
    )
    logits = model(
        input_ids=sequence,
        use_cache=False,
        logits_to_keep=positions,
    ).logits[0]
    if logits.shape[0] != response_length:
        raise RuntimeError(f"logits rows={logits.shape[0]} response={response_length}")
    return logits


def ce_loss(model: torch.nn.Module, record: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model_response_logits(model, record)
    targets = torch.as_tensor(
        record["generation_token_ids"], dtype=torch.long, device="cuda"
    )
    loss_sum = F.cross_entropy(logits.float(), targets, reduction="sum")
    return loss_sum, {"loss_sum": float(loss_sum.detach().item())}


def kl_loss(
    model: torch.nn.Module,
    record: dict[str, Any],
    teacher_ids_np: np.ndarray,
    teacher_logprobs_np: np.ndarray,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = model_response_logits(model, record)
    teacher_ids = torch.as_tensor(
        teacher_ids_np.astype(np.int64, copy=False), dtype=torch.long, device="cuda"
    )
    teacher_logprobs = torch.as_tensor(
        teacher_logprobs_np, dtype=torch.float32, device="cuda"
    )
    student_logprobs = F.log_softmax(logits, dim=-1)
    student_topk = torch.gather(student_logprobs, dim=-1, index=teacher_ids)
    student_mass = student_topk.exp().sum(dim=-1)
    teacher_mass = teacher_logprobs.exp().sum(dim=-1)
    student_topk = student_topk.clamp_min(LOGPROB_MIN).float()
    teacher_clamped = teacher_logprobs.clamp_min(LOGPROB_MIN)
    token_loss = (
        teacher_clamped.exp() * (teacher_clamped - student_topk)
    ).sum(dim=-1)
    token_loss = token_loss.clamp_min(0.0).clamp_max(LOSS_MAX)
    return token_loss.sum(), {
        "loss_sum": float(token_loss.detach().sum().item()),
        "student_mass_sum": float(student_mass.detach().float().sum().item()),
        "teacher_mass_sum": float(teacher_mass.detach().float().sum().item()),
    }


def train(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    eligible: list[int],
    top_ids: np.ndarray | None,
    top_logprobs: np.ndarray | None,
    offsets: np.ndarray | None,
    validation: dict[str, Any],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
            f"formal schedule={total_steps}, expected 624 from eligible={len(eligible)}"
        )
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)

    resume_path = resolve_resume(args.resume, args.output_dir)
    gradient_checkpointing = not args.no_gradient_checkpointing
    model = build_model(resume_path, gradient_checkpointing)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    start_step = 1
    if resume_path is None:
        saved = save_checkpoint(model, optimizer, args.output_dir, 0)
        print(f"[G6 {args.arm}] initial checkpoint -> {saved}", flush=True)
    else:
        state = torch.load(
            resume_path / "trainer_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        torch.set_rng_state(state["cpu_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        print(f"[G6 {args.arm}] resume {resume_path} at {start_step}", flush=True)

    loss_name = "forward_kl_topk32_raw" if args.arm == "offkd" else "response_ce"
    manifest = {
        "schema_version": 1,
        "status": "running",
        "task": "Cycle 09 block 2 G6",
        "arm": args.arm,
        "student_model": str(BASE),
        "data_source": (
            str(args.sft_cache / "records.jsonl")
            if args.arm == "sft"
            else str(args.rollout_dir / "teacher_rollout.jsonl")
        ),
        "seed": SEED,
        "eligible_prompts": len(eligible) if not args.smoke else 1,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "batch_size": batch_size,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "scheduled_steps": total_steps,
        "shuffle": False,
        "drop_last": True,
        "question_masked": True,
        "gradient_checkpointing": gradient_checkpointing,
        "attention_implementation": "sdpa",
        "lora": {
            "r": 32,
            "alpha": 64,
            "target_modules": "all-linear",
            "dropout": 0.0,
            "bias": "none",
            "adapter_dtype": "bfloat16",
            "trainable_parameters": sum(p.numel() for p in trainable),
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
            "name": loss_name,
            "aggregation": "token-mean over each global batch",
            "teacher_logprobs": "raw temperature=1.0" if args.arm == "offkd" else None,
            "topk": TOPK if args.arm == "offkd" else None,
        },
        "checkpoint_grid": list((0, 1) if args.smoke else CHECKPOINT_GRID),
        "resume_from": str(resume_path) if resume_path else None,
        "input_validation": validation,
        "started_at_unix": time.time(),
    }
    atomic_json(args.output_dir / "training_manifest.json", manifest)
    metrics_path = args.output_dir / "train_metrics.jsonl"
    if resume_path is not None and metrics_path.is_file():
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
        manifest.update(
            status="complete", completed_steps=total_steps, completed_at_unix=time.time()
        )
        atomic_json(args.output_dir / "training_manifest.json", manifest)
        return

    run_started = time.time()
    for global_step in range(start_step, total_steps + 1):
        torch.cuda.reset_peak_memory_stats()
        step_started = time.time()
        batch_in_epoch = (global_step - 1) % steps_per_epoch
        begin = batch_in_epoch * batch_size
        batch_rows = eligible[begin : begin + batch_size]
        token_denominator = sum(int(records[row]["n_tokens"]) for row in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        sums = {"loss_sum": 0.0, "student_mass_sum": 0.0, "teacher_mass_sum": 0.0}
        for sample_number, row in enumerate(batch_rows, start=1):
            record = records[row]
            if args.arm == "offkd":
                assert top_ids is not None and top_logprobs is not None and offsets is not None
                offset_start, offset_end = map(int, offsets[row])
                sample_loss, metrics = kl_loss(
                    model,
                    record,
                    top_ids[offset_start:offset_end],
                    top_logprobs[offset_start:offset_end],
                )
            else:
                sample_loss, metrics = ce_loss(model, record)
            (sample_loss / token_denominator).backward()
            for key, value in metrics.items():
                sums[key] += value
            del sample_loss
            print(
                f"[G6 {args.arm}] step={global_step}/{total_steps} "
                f"sample={sample_number}/{batch_size} row={row} "
                f"tokens={record['n_tokens']}",
                flush=True,
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
        optimizer.step()
        elapsed = time.time() - step_started
        metric = {
            "step": global_step,
            "epoch_zero_based": (global_step - 1) // steps_per_epoch,
            "batch_in_epoch_zero_based": batch_in_epoch,
            "response_tokens": token_denominator,
            "loss": sums["loss_sum"] / token_denominator,
            "grad_norm_before_clip": float(grad_norm),
            "seconds": elapsed,
            "tokens_per_second": token_denominator / elapsed,
            "gpu_max_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if args.arm == "offkd":
            metric["student_top32_mass"] = sums["student_mass_sum"] / token_denominator
            metric["teacher_top32_mass"] = sums["teacher_mass_sum"] / token_denominator
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric) + "\n")
        print(f"[G6 {args.arm}] {json.dumps(metric)}", flush=True)

        save_grid = (0, 1) if args.smoke else CHECKPOINT_GRID
        if global_step in save_grid:
            saved = save_checkpoint(model, optimizer, args.output_dir, global_step)
            print(f"[G6 {args.arm}] checkpoint -> {saved}", flush=True)
        manifest["completed_steps"] = global_step
        manifest["last_step_seconds"] = elapsed
        atomic_json(args.output_dir / "training_manifest.json", manifest)

    manifest.update(
        status="complete",
        completed_steps=total_steps,
        training_seconds_this_invocation=time.time() - run_started,
        completed_at_unix=time.time(),
    )
    atomic_json(args.output_dir / "training_manifest.json", manifest)
    if not args.smoke:
        destination = COPYBACK / args.arm
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output_dir / "training_manifest.json", destination)
        shutil.copy2(metrics_path, destination)
    print(f"[G6 {args.arm}] complete {total_steps}", flush=True)


def main() -> None:
    args = parse_args()
    for path in (BASE / "config.json", BASE / "tokenizer.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256_file(BASE / "tokenizer.json") != sha256_file(TEACHER / "tokenizer.json"):
        raise RuntimeError("teacher/student tokenizer.json mismatch")

    if args.arm == "sft":
        records = prepare_sft(args.sft_cache, args.smoke)
    else:
        records = load_teacher_records(args.rollout_dir)
    eligible, record_report = validate_records(
        records, smoke=args.smoke, arm=args.arm
    )
    top_ids = top_logprobs = offsets = None
    validation: dict[str, Any] = {"records": record_report}
    if args.arm == "offkd":
        top_ids, top_logprobs, offsets, topk_report = load_top32(
            args.rollout_dir, records
        )
        validation["top32"] = topk_report
    validation_path = args.output_dir.parent / "input_validation.json"
    atomic_json(validation_path, validation)
    print(json.dumps(validation, indent=2), flush=True)
    if args.validate_only or args.prepare_only:
        return
    train(
        args,
        records,
        eligible,
        top_ids,
        top_logprobs,
        offsets,
        validation,
    )


if __name__ == "__main__":
    main()
