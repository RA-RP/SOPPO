#!/usr/bin/env python3
"""Validate and export the explicitly authorized Q1 alpha=.5 Stage-B endpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c


STEP = 320
STAGE_A = c.MINI / "qwen_alpha05_stage_a_training_manifest.json"
MANIFEST = c.MINI / "qwen_alpha05_stage_b_training_manifest.json"
EXPORT_MANIFEST = c.Q1_ROOT / "qwen_alpha05_stage_b_model_export_manifest.json"
SUPPORT_STATS = c.MINI / "qwen_alpha05_stage_b_support_stats.csv"
SUPPORT_MANIFEST = c.MINI / "qwen_alpha05_stage_b_support_stats_manifest.json"
CONVERTER = c.SCRIPT_DIR / "cycle08_convert_ckpt.py"


def checkpoint_file(path: Path) -> dict[str, Any]:
    payload = {"path": str(path), "bytes": path.stat().st_size}
    if payload["bytes"] <= 8 << 20:
        payload["sha256"] = c.sha256_file(path)
    else:
        payload["sha256"] = None
        payload["integrity"] = "VERL tracker commit + nonzero shard size; loadability checked at export"
    return payload


def require_checkpoint() -> dict[str, Any]:
    root = c.Q1_CHECKPOINTS / f"global_step_{STEP}"
    required = (
        root / "actor/model_world_size_1_rank_0.pt",
        root / "actor/optim_world_size_1_rank_0.pt",
        root / "actor/extra_state_world_size_1_rank_0.pt",
        root / "actor/lora_train_meta.json",
        root / "data.pt",
    )
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"incomplete Q1 Stage-B checkpoint: {missing}")
    return {
        "step": STEP,
        "root": str(root),
        "files": [checkpoint_file(path) for path in required],
        "lora_meta": c.read_json(root / "actor/lora_train_meta.json", {}),
    }


def validate() -> dict[str, Any]:
    stage_a = c.read_json(STAGE_A, {})
    if stage_a.get("status") != "complete_checkpoint_validated":
        raise RuntimeError("Q1 Stage A must be checkpoint-validated before Stage B")
    tracker = c.Q1_CHECKPOINTS / "latest_checkpointed_iteration.txt"
    if not tracker.is_file() or tracker.read_text(encoding="utf-8").strip() != str(STEP):
        raise RuntimeError("Q1 Stage-B tracker must be committed at exactly step 320")
    schedule = c.Q1_DATA / "qwen_alpha05_schedule_320.parquet"
    if not schedule.is_file() or schedule.stat().st_size == 0:
        raise FileNotFoundError(schedule)
    payload = {
        "schema_version": 1,
        "status": "complete_checkpoint_validated",
        "arm": "qwen_alpha05_support_mixture",
        "stage": "B",
        "resume_from_step": 160,
        "completed_steps": STEP,
        "checkpoint_grid": [0, 5, 10, 20, 40, 80, 160, 320],
        "checkpoint_cell": require_checkpoint(),
        "training_schedule": c.artifact(schedule),
        "stage_a_manifest": c.artifact(STAGE_A),
        "training_log": c.artifact(c.Q1_LOGS / "stage_b_train.log"),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, payload)
    return payload


def export() -> dict[str, Any]:
    validate()
    target = c.Q1_ROOT / "_merged_models" / "step_320"
    before = c.model_check(target)
    if not before["complete"]:
        command = [
            str(c.VERL_PYTHON), str(CONVERTER), "--ckpt-root", str(c.Q1_CHECKPOINTS),
            "--step", str(STEP), "--out-dir", str(target), "--base", str(c.QWEN_STUDENT),
        ]
        completed = subprocess.run(command, cwd=c.REPO)
        if completed.returncode != 0:
            raise RuntimeError(f"Q1 Stage-B model export failed: rc={completed.returncode}")
    model = c.model_check(target)
    if not model["complete"]:
        raise RuntimeError(f"Q1 Stage-B merged model incomplete: {model}")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 alpha=.5 Stage-B FSDP checkpoint export",
        "step": STEP,
        "model": model,
        "delta_w_policy": "adapter BA fp32 only for any update-space analysis; merge-subtract forbidden",
        "created_utc": c.utc_now(),
    }
    c.atomic_json(EXPORT_MANIFEST, payload)
    return payload


def support_stats() -> dict[str, Any]:
    """Summarize every persisted Stage-A/B rollout without filling terminal gaps."""
    schedule = pd.read_parquet(c.Q1_DATA / "qwen_alpha05_schedule_320.parquet")
    if len(schedule) != STEP * c.TRAIN_BATCH_SIZE:
        raise RuntimeError(f"Q1 schedule rows={len(schedule)} expected={STEP * c.TRAIN_BATCH_SIZE}")
    rows: list[dict[str, Any]] = []
    missing_steps: list[int] = []
    for step in range(1, STEP + 1):
        rollout = c.Q1_ROOT / "rollouts" / f"{step}.jsonl"
        if not rollout.is_file():
            missing_steps.append(step)
            continue
        records = c.read_jsonl(rollout)
        if len(records) != c.TRAIN_BATCH_SIZE:
            raise RuntimeError(f"Q1 rollout {step} rows={len(records)} != {c.TRAIN_BATCH_SIZE}")
        batch = schedule.iloc[(step - 1) * c.TRAIN_BATCH_SIZE : step * c.TRAIN_BATCH_SIZE]
        for record, (_, source) in zip(records, batch.iterrows(), strict=True):
            text = str(record.get("output", ""))
            finish = str(record.get("finish_reason", "")).lower()
            rows.append(
                {
                    "stage": "A" if step <= 160 else "B",
                    "step": step,
                    "support_source": str(source["support_source"]),
                    "response_chars": len(text),
                    "response_tokens": int(record.get("response_token_length", 0)),
                    "eos": int(finish in {"eos", "stop", "stop_sequence"}),
                    "truncated": int(finish in {"length", "max_tokens"}),
                    "has_boxed": int("\\boxed{" in text),
                    "has_think": int("<think>" in text),
                    "output": text,
                }
            )
    frame = pd.DataFrame(rows)
    if set(frame["support_source"]) != {"self", "external"}:
        raise RuntimeError("Q1 saved rollout source join is incomplete")
    if missing_steps != [160, 320]:
        raise RuntimeError(f"unexpected Q1 rollout gaps: {missing_steps}")

    summaries: list[dict[str, Any]] = []
    for stage, part in [*frame.groupby("stage", sort=True), ("all", frame)]:
        for source, cell in part.groupby("support_source", sort=True):
            summaries.append(
                {
                    "stage": stage,
                    "support_source": source,
                    "n_samples": len(cell),
                    "n_steps": cell["step"].nunique(),
                    "response_chars_mean": cell["response_chars"].mean(),
                    "response_chars_median": cell["response_chars"].median(),
                    "response_tokens_mean": cell["response_tokens"].mean(),
                    "response_tokens_median": cell["response_tokens"].median(),
                    "eos_rate": cell["eos"].mean(),
                    "truncation_rate": cell["truncated"].mean(),
                    "exact_duplicate_rate": 1 - cell["output"].nunique() / len(cell),
                    "boxed_rate": cell["has_boxed"].mean(),
                    "think_rate": cell["has_think"].mean(),
                }
            )
    c.atomic_csv(SUPPORT_STATS, summaries)
    payload = {
        "schema_version": 1,
        "status": "complete_with_declared_terminal_gaps",
        "task": "Q1 alpha=.5 Stage-A/B source-separated support statistics",
        "checkpoint_range": [1, STEP],
        "saved_rollout_steps": sorted(frame["step"].unique().tolist()),
        "missing_rollout_steps": missing_steps,
        "gap_reason": "terminal validation occurs after checkpoint commit but before asynchronous rollout dump",
        "source_assignment": "joined by physical schedule row; eight self and eight external per saved update",
        "raw_samples": len(frame),
        "per_source_counts": dict(Counter(frame["support_source"])),
        "output": c.artifact(SUPPORT_STATS),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(SUPPORT_MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("validate", "export", "support-stats"), required=True)
    args = parser.parse_args()
    if args.phase == "validate":
        result = validate()
    elif args.phase == "export":
        result = export()
    else:
        result = support_stats()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
