#!/usr/bin/env python3
"""Validate and export the Q1 alpha=.5 Stage-A trajectory without resuming training."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c


STEPS = (0, 5, 20, 40, 80, 160)
NONZERO_STEPS = STEPS[1:]
FORMAL_MANIFEST = c.MINI / "qwen_alpha05_stage_a_training_manifest.json"
SUPPORT_STATS = c.MINI / "qwen_alpha05_support_stats.csv"
SUPPORT_MANIFEST = c.MINI / "qwen_alpha05_support_stats_manifest.json"
MODELS = c.Q1_ROOT / "_merged_models"
CONVERTER = c.SCRIPT_DIR / "cycle08_convert_ckpt.py"


def checkpoint(step: int) -> Path:
    return c.Q1_CHECKPOINTS / f"global_step_{step}"


def merged(step: int) -> Path:
    return c.QWEN_STUDENT if step == 0 else MODELS / f"step_{step:03d}"


def checkpoint_file(path: Path) -> dict[str, Any]:
    """Avoid re-hashing the multi-gigabyte FSDP shards during a metadata audit."""
    payload = {"path": str(path), "bytes": path.stat().st_size}
    if payload["bytes"] <= 8 << 20:
        payload["sha256"] = c.sha256_file(path)
    else:
        payload["sha256"] = None
        payload["integrity"] = "VERL tracker commit + nonzero shard size; loadability checked at export"
    return payload


def require_checkpoint(step: int) -> dict[str, Any]:
    root = checkpoint(step)
    required = (
        root / "actor/model_world_size_1_rank_0.pt",
        root / "actor/optim_world_size_1_rank_0.pt",
        root / "actor/extra_state_world_size_1_rank_0.pt",
        root / "actor/lora_train_meta.json",
        root / "data.pt",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"incomplete Q1 checkpoint step {step}: {missing}")
    return {
        "step": step,
        "root": str(root),
        "files": [checkpoint_file(path) for path in required],
        "lora_meta": c.read_json(root / "actor/lora_train_meta.json", {}),
    }


def validate_stage_a() -> dict[str, Any]:
    tracker = c.Q1_CHECKPOINTS / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != "160":
        raise RuntimeError("Q1 Stage A tracker must be committed at exactly step 160")
    cells = [require_checkpoint(step) for step in NONZERO_STEPS]
    log = c.Q1_LOGS / "formal_train.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    final_validation_failed = bool(
        re.search(r"ActorDiedError|Resource temporarily unavailable", text)
    )
    if not final_validation_failed:
        raise RuntimeError("expected final-validation failure signature was not found for audit")
    payload = {
        "schema_version": 1,
        "status": "complete_checkpoint_validated",
        "arm": "qwen_alpha05_support_mixture",
        "stage": "A",
        "completed_steps": 160,
        "checkpoint_grid": list(STEPS),
        "checkpoint_cells": cells,
        "training_schedule": c.artifact(c.Q1_DATA / "qwen_alpha05_schedule_624.parquet"),
        "prepare_manifest": c.artifact(c.Q1_DATA / "qwen_alpha05_prepare_manifest.json"),
        "supervisor_status": c.read_json(c.Q1_ROOT / "supervisor/status.json", {}),
        "terminal_validation": {
            "status": "failed_after_step160_checkpoint_commit",
            "effect_on_checkpoint": "none; final actor update and checkpoint commit precede validation",
            "failure_signature": "AgentLoopWorkerTQ Resource temporarily unavailable / ActorDiedError",
            "log": c.artifact(log),
        },
        "stage_b": {
            "status": "not_started",
            "final_endpoint": 320,
            "requires_explicit_go": True,
        },
        "created_utc": c.utc_now(),
    }
    c.atomic_json(FORMAL_MANIFEST, payload)
    return payload


def support_stats() -> dict[str, Any]:
    schedule = pd.read_parquet(c.Q1_DATA / "qwen_alpha05_schedule_624.parquet")
    rows: list[dict[str, Any]] = []
    missing_steps: list[int] = []
    for step in range(1, 161):
        rollout = c.Q1_ROOT / "rollouts" / f"{step}.jsonl"
        if not rollout.is_file():
            missing_steps.append(step)
            continue
        records = c.read_jsonl(rollout)
        if len(records) != c.TRAIN_BATCH_SIZE:
            raise RuntimeError(f"Q1 rollout {step} rows={len(records)} != {c.TRAIN_BATCH_SIZE}")
        batch = schedule.iloc[(step - 1) * c.TRAIN_BATCH_SIZE : step * c.TRAIN_BATCH_SIZE]
        if len(batch) != c.TRAIN_BATCH_SIZE:
            raise RuntimeError(f"schedule lacks Q1 physical batch {step}")
        for record, (_, source) in zip(records, batch.iterrows(), strict=True):
            text = str(record.get("output", ""))
            source_name = str(source["support_source"])
            rows.append(
                {
                    "step": step,
                    "support_source": source_name,
                    "sample": 1,
                    "response_chars": len(text),
                    "has_boxed": int("\\\\boxed" in text),
                    "has_think": int("<think>" in text),
                    "source_prompt_identity": int(source["prompt_identity"]),
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame["support_source"]) != {"self", "external"}:
        raise RuntimeError("Q1 saved rollout source join is incomplete")
    summary = (
        frame.groupby("support_source", as_index=False)
        .agg(
            n_samples=("sample", "sum"),
            n_steps=("step", "nunique"),
            response_chars_mean=("response_chars", "mean"),
            response_chars_median=("response_chars", "median"),
            boxed_rate=("has_boxed", "mean"),
            think_rate=("has_think", "mean"),
        )
        .sort_values("support_source", kind="stable")
    )
    c.atomic_csv(SUPPORT_STATS, summary.to_dict("records"))
    manifest = {
        "schema_version": 1,
        "status": "complete_with_terminal_rollout_gap",
        "task": "Q1 Stage-A source-separated saved-rollout statistics",
        "saved_rollout_steps": sorted(set(frame["step"])),
        "missing_rollout_steps": missing_steps,
        "gap_reason": "step160 actor update/checkpoint committed; final validation failed before the asynchronous training-rollout dump",
        "source_assignment": "joined to physical schedule rows; 8 self + 8 external per saved update",
        "response_length_unit": "Unicode code points; token ids were not retained by the trainer rollout dump",
        "raw_samples": len(frame),
        "per_source_counts": dict(Counter(frame["support_source"])),
        "output": c.artifact(SUPPORT_STATS),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(SUPPORT_MANIFEST, manifest)
    return manifest


def export_models(steps: list[int]) -> dict[str, Any]:
    validate_stage_a()
    results = []
    for step in steps:
        if step == 0:
            results.append(c.model_check(c.QWEN_STUDENT))
            continue
        target = merged(step)
        command = [
            str(c.VERL_PYTHON),
            str(CONVERTER),
            "--ckpt-root",
            str(c.Q1_CHECKPOINTS),
            "--step",
            str(step),
            "--out-dir",
            str(target),
            "--base",
            str(c.QWEN_STUDENT),
        ]
        completed = subprocess.run(command, cwd=c.REPO)
        if completed.returncode != 0:
            raise RuntimeError(f"Q1 model export failed at step {step}: rc={completed.returncode}")
        check = c.model_check(target)
        if not check["complete"]:
            raise RuntimeError(f"Q1 merged model incomplete at step {step}: {check}")
        results.append(check | {"step": step})
    payload = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 Stage-A FSDP checkpoint export to merged Qwen models",
        "steps": steps,
        "models": results,
        "created_utc": c.utc_now(),
    }
    c.atomic_json(c.Q1_ROOT / "qwen_alpha05_model_export_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate", "support-stats", "export"), required=True)
    parser.add_argument("--steps", default=",".join(map(str, NONZERO_STEPS)))
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(",") if value.strip()]
    if any(step not in STEPS for step in steps):
        raise ValueError(f"unsupported Q1 export steps: {steps}")
    if args.phase == "validate":
        result = validate_stage_a()
    elif args.phase == "support-stats":
        result = support_stats()
    else:
        result = export_models(steps)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
