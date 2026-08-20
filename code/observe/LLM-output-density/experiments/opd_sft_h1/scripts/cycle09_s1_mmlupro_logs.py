#!/usr/bin/env python3
"""Backfill formal MMLU-Pro per-sample logs for the three-arm ten-step grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cycle09_r4_common as c4


STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ARMS = ("opd", "sft", "offkd")
OFFKD = Path("/root/autodl-tmp/cycle09_offkd/_merged_models")
R4_BEHAVIOR = Path("/root/autodl-tmp/cycle09_r4/behavior/mmlu_pro")
RUN_ROOT = Path("/root/autodl-tmp/cycle09_s1/mmlu_pro_logs")
MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
SUBJECTS = (
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def model_path(arm: str, step: int) -> Path:
    if step == 0:
        return c4.BASE_MODEL
    if arm == "offkd":
        path = OFFKD / c4.step_label(step)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path)
        return path
    return c4.model_path(arm, step)


def result_files(root: Path) -> list[Path]:
    return sorted(root.rglob("results_*.json")) if root.is_dir() else []


def sample_files(root: Path) -> list[Path]:
    return sorted(root.rglob("samples_mmlu_pro_*.jsonl")) if root.is_dir() else []


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def validate(root: Path, limit: int) -> dict | None:
    results = result_files(root)
    samples = sample_files(root)
    if len(results) != 1 or len(samples) != len(SUBJECTS):
        return None
    by_subject = {}
    for path in samples:
        matches = [
            subject
            for subject in SUBJECTS
            if f"samples_mmlu_pro_{subject}_" in path.name
        ]
        if len(matches) != 1:
            return None
        by_subject[matches[0]] = count_lines(path)
    if set(by_subject) != set(SUBJECTS):
        return None
    if any(count != limit for count in by_subject.values()):
        return None
    return {
        "root": str(root),
        "result_path": str(results[0]),
        "result_sha256": sha256_file(results[0]),
        "sample_files": [
            {
                "path": str(path),
                "subject": next(
                    subject
                    for subject in SUBJECTS
                    if f"samples_mmlu_pro_{subject}_" in path.name
                ),
                "n_rows": count_lines(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in samples
        ],
        "n_rows": sum(by_subject.values()),
    }


def r4_source(arm: str, step: int, limit: int) -> dict | None:
    source_arm = "opd" if step == 0 else arm
    if source_arm not in ("opd", "sft"):
        return None
    root = R4_BEHAVIOR / source_arm / c4.step_label(step)
    return validate(root, limit)


def formal_root(arm: str, step: int) -> Path:
    return RUN_ROOT / arm / c4.step_label(step)


def run_one(
    *,
    arm: str,
    step: int,
    limit: int,
    gpu_memory: float,
    max_model_len: int,
    seed: int,
    output: Path,
) -> dict:
    complete = validate(output, limit)
    if complete is not None:
        print(f"[S1 MMLU cached] {arm}/{c4.step_label(step)}", flush=True)
        return complete
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"incomplete non-empty MMLU output requires audit: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = model_path(arm, step)
    model_args = (
        f"pretrained={model},dtype=bfloat16,gpu_memory_utilization={gpu_memory},"
        f"max_model_len={max_model_len}"
    )
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "vllm",
        "--model_args",
        model_args,
        "--tasks",
        "mmlu_pro",
        "--num_fewshot",
        "0",
        "--batch_size",
        "auto",
        "--seed",
        str(seed),
        "--output_path",
        str(output),
        "--limit",
        str(limit),
        "--log_samples",
    ]
    env = dict(os.environ)
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"[S1 MMLU] {arm}/{c4.step_label(step)}", flush=True)
    result = subprocess.run(command, cwd=str(c4.REPO), env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"MMLU-Pro failed {arm}/{c4.step_label(step)} rc={result.returncode}"
        )
    complete = validate(output, limit)
    if complete is None:
        raise RuntimeError(f"MMLU-Pro output failed validation: {output}")
    return complete


def run_grid(args) -> list[dict]:
    if args.smoke:
        cell = run_one(
            arm="opd",
            step=0,
            limit=1,
            gpu_memory=args.gpu_memory,
            max_model_len=args.max_model_len,
            seed=args.seed,
            output=RUN_ROOT / "smoke" / "opd" / "step_000",
        )
        return [
            {
                "arm": "opd",
                "step": 0,
                "source_kind": "new_smoke",
                **cell,
            }
        ]

    rows = []
    unique = {}
    for arm in ARMS:
        for step in STEPS:
            key = ("base", 0) if step == 0 else (arm, step)
            if key not in unique:
                reused = r4_source(arm, step, args.limit)
                if reused is not None:
                    source_kind = "reused_R4_same_protocol"
                    cell = reused
                    print(
                        f"[S1 MMLU reuse R4] {arm}/{c4.step_label(step)}",
                        flush=True,
                    )
                else:
                    source_kind = "new_Stage1_log_backfill"
                    cell = run_one(
                        arm=arm,
                        step=step,
                        limit=args.limit,
                        gpu_memory=args.gpu_memory,
                        max_model_len=args.max_model_len,
                        seed=args.seed,
                        output=formal_root(arm, step),
                    )
                unique[key] = (source_kind, cell)
            source_kind, cell = unique[key]
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "model_path": str(model_path(arm, step)),
                    "source_kind": source_kind,
                    **cell,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    if not args.smoke and args.limit != 100:
        raise ValueError("formal S1-1/2 protocol requires --limit 100")
    rows = run_grid(args)
    if args.smoke:
        print(json.dumps(rows, indent=2))
        return
    if len(rows) != 30 or any(row["n_rows"] != 1400 for row in rows):
        raise RuntimeError(f"incomplete three-arm grid: rows={len(rows)}")
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-1/S1-2 per-sample log prerequisite",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "backend": "lm_eval vllm",
                "task": "mmlu_pro",
                "num_fewshot": 0,
                "limit_per_subject": args.limit,
                "subjects": list(SUBJECTS),
                "seed": args.seed,
                "dtype": "bfloat16",
                "max_model_len": args.max_model_len,
                "chat_template": False,
                "log_samples": True,
            },
            "deviation": (
                "The formal trajectory result directories did not contain --log_samples; "
                "missing cells were rerun with the frozen formal protocol. Existing R4 "
                "cells with the same protocol were reused."
            ),
            "arms": list(ARMS),
            "steps": list(STEPS),
            "n_grid_rows": len(rows),
            "n_unique_log_cells": len(
                {(row["root"], row["result_path"]) for row in rows}
            ),
            "cells": rows,
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MINI / "S1_mmlupro_log_manifest.json",
    )
    print(
        f"[S1 MMLU] complete grid_rows={len(rows)} "
        f"unique={len({row['root'] for row in rows})}"
    )


if __name__ == "__main__":
    main()
