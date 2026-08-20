#!/usr/bin/env python3
"""GPU sidecar for atomically precomputing expensive seqKD Math500 cells."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cycle09_offkd_eval as oe
import cycle09_seqkd_eval as g2


RUN_ROOT = Path("/root/autodl-tmp/cycle09_seqkd")
FORMAL_ROOT = RUN_ROOT / "eval/formal"
WORKER_ROOT = RUN_ROOT / "eval/workers"
BLOCK_ROOT = Path("/root/autodl-tmp/cycle09_block2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default="320,480")
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--worker-id", default="gpu1_math")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.steps = g2.parse_steps(args.steps)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.worker_id):
        parser.error("worker-id may contain only letters, digits, dot, dash, underscore")
    return args


def summary_path(root: Path, step: int) -> Path:
    label = oe.step_label(step)
    return root / "generative" / label / "math500" / f"{label}.json"


def validate_summary(path: Path, step: int, n: int, max_tokens: int) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "task": "math500",
        "label": oe.step_label(step),
        "model": str(oe.model_path(RUN_ROOT, step)),
        "n": n,
        "max_tokens": max_tokens,
        "seed": oe.SEED,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"summary provenance mismatch at {path}: {mismatches}")
    return payload


def publish_step(staging_root: Path, step: int, n: int, max_tokens: int) -> str:
    label = oe.step_label(step)
    staged_dir = staging_root / "generative" / label / "math500"
    official_dir = FORMAL_ROOT / "generative" / label / "math500"
    official_summary = official_dir / f"{label}.json"
    if official_summary.is_file():
        validate_summary(official_summary, step, n, max_tokens)
        return "official_cached"
    if official_dir.exists():
        if any(official_dir.iterdir()):
            raise RuntimeError(
                f"official cell became non-empty before publish; refusing overwrite: {official_dir}"
            )
        official_dir.rmdir()
    official_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged_dir, official_dir)
    validate_summary(official_summary, step, n, max_tokens)
    return "published"


def run(args: argparse.Namespace) -> None:
    mode = "smoke" if args.smoke else "formal"
    staging_root = WORKER_ROOT / mode / args.worker_id
    status_path = BLOCK_ROOT / f"g2_{args.worker_id}_status.json"
    lock_path = BLOCK_ROOT / f"g2_{args.worker_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"worker already running: {lock_path}") from exc

        completed: list[int] = []
        atomic_json(
            status_path,
            {
                "status": "running",
                "worker_id": args.worker_id,
                "pid": os.getpid(),
                "steps": args.steps,
                "task": "math500",
                "mode": mode,
                "shared_outputs_written": False,
                "started_at": utc_now(),
            },
        )
        try:
            g2.configure_shared_runner()
            worker_args = SimpleNamespace(
                steps=args.steps,
                smoke=args.smoke,
                run_root=RUN_ROOT,
                eval_root=staging_root,
                copyback_root=g2.COPYBACK,
                gpu_mem=args.gpu_mem,
            )
            g2.preflight(worker_args)
            for step in args.steps:
                max_tokens, max_model_len = oe.math500_budget(step)
                n = 500
                if args.smoke:
                    n, max_tokens, max_model_len = 2, 256, 2048

                official = summary_path(FORMAL_ROOT, step)
                if not args.smoke and official.is_file():
                    validate_summary(official, step, n, max_tokens)
                    completed.append(step)
                    continue

                staged = summary_path(staging_root, step)
                if staged.is_file():
                    validate_summary(staged, step, n, max_tokens)
                else:
                    staged_dir = staged.parent
                    if staged_dir.exists() and any(staged_dir.iterdir()):
                        raise RuntimeError(
                            f"incomplete staging cell requires audit: {staged_dir}"
                        )
                    atomic_json(
                        status_path,
                        {
                            "status": "running",
                            "worker_id": args.worker_id,
                            "pid": os.getpid(),
                            "steps": args.steps,
                            "task": "math500",
                            "mode": mode,
                            "current_step": step,
                            "completed_steps": completed,
                            "shared_outputs_written": False,
                            "started_at": json.loads(
                                status_path.read_text(encoding="utf-8")
                            )["started_at"],
                            "updated_at": utc_now(),
                        },
                    )
                    oe.ensure_merged_model(RUN_ROOT, step)
                    oe.run_think_eval(
                        worker_args,
                        task="math500",
                        step=step,
                        n=n,
                        max_tokens=max_tokens,
                        max_model_len=max_model_len,
                    )
                    validate_summary(staged, step, n, max_tokens)

                publish_state = "smoke_only"
                if not args.smoke:
                    publish_state = publish_step(staging_root, step, n, max_tokens)
                completed.append(step)
                print(f"[G2 worker] step={step} state={publish_state}", flush=True)

            atomic_json(
                status_path,
                {
                    "status": "complete",
                    "worker_id": args.worker_id,
                    "pid": os.getpid(),
                    "steps": args.steps,
                    "task": "math500",
                    "mode": mode,
                    "completed_steps": completed,
                    "shared_outputs_written": False,
                    "completed_at": utc_now(),
                },
            )
        except Exception as exc:
            atomic_json(
                status_path,
                {
                    "status": "failed",
                    "worker_id": args.worker_id,
                    "pid": os.getpid(),
                    "steps": args.steps,
                    "task": "math500",
                    "mode": mode,
                    "completed_steps": completed,
                    "error": repr(exc),
                    "failed_at": utc_now(),
                },
            )
            raise


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
