#!/usr/bin/env python3
"""Atomic standalone pipeline registry updates; uses only the standard library."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def initialize(args: argparse.Namespace) -> None:
    path = Path(args.path)
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite registry: {path}")
    order = args.stages.split(",")
    training_gpu_ids = args.training_gpus.split(",")
    payload = {
        "schema_version": 2,
        "cycle_id": "cycle-20260818-01",
        "experiment_id": args.experiment,
        "experiment_design": "v0.6-sspo-aligned-30k",
        "execution_platform": "standalone",
        "git_commit": args.commit,
        "pipeline_status": "running",
        "current_stage": "preflight",
        "started_at": now(),
        "finished_at": None,
        "pid": int(args.pid),
        "process_group_id": int(args.pgid),
        "gpu_contract": {
            "training_gpu_ids": training_gpu_ids,
            "training_gpu_count": len(training_gpu_ids),
            "postprocess_gpu_id": args.post_gpu,
            "minimum_memory_mib": int(args.minimum_memory_mib),
        },
        "stage_order": order,
        "stages": {
            name: {
                "state": "pending",
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "log": f"logs/{name}.log",
            }
            for name in order
        },
    }
    write(path, payload)


def set_stage(args: argparse.Namespace) -> None:
    path = Path(args.path)
    payload = read(path)
    if args.stage not in payload["stages"]:
        raise KeyError(f"Unknown stage: {args.stage}")
    stage = payload["stages"][args.stage]
    stage["state"] = args.state
    if args.state == "running":
        stage["started_at"] = now()
        payload["current_stage"] = args.stage
    if args.state in {"completed", "failed", "interrupted"}:
        stage["finished_at"] = now()
        stage["exit_code"] = args.exit_code
    write(path, payload)


def set_pipeline(args: argparse.Namespace) -> None:
    path = Path(args.path)
    payload = read(path)
    payload["pipeline_status"] = args.state
    if args.current_stage is not None:
        payload["current_stage"] = args.current_stage or None
    if args.state in {"completed", "failed", "interrupted"}:
        payload["finished_at"] = now()
    write(path, payload)


def show(args: argparse.Namespace) -> None:
    payload = read(Path(args.path))
    print(f"Experiment: {payload['experiment_id']}")
    print(f"Platform: {payload['execution_platform']}")
    print(f"Git commit: {payload['git_commit']}")
    print(f"Pipeline: {payload['pipeline_status']}")
    print(f"Current stage: {payload.get('current_stage') or '-'}")
    print(f"PID/PGID: {payload['pid']}/{payload['process_group_id']}")
    print()
    print(f"{'stage':<28} {'state':<12} {'exit':<6} log")
    print(f"{'-' * 28} {'-' * 12} {'-' * 6} {'-' * 24}")
    for name in payload["stage_order"]:
        stage = payload["stages"][name]
        exit_code = "-" if stage["exit_code"] is None else str(stage["exit_code"])
        print(f"{name:<28} {stage['state']:<12} {exit_code:<6} {stage['log']}")


def get_value(args: argparse.Namespace) -> None:
    value = read(Path(args.path))[args.field]
    if isinstance(value, (dict, list)):
        print(json.dumps(value, sort_keys=True))
    elif value is not None:
        print(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--path", required=True)
    init_parser.add_argument("--experiment", required=True)
    init_parser.add_argument("--commit", required=True)
    init_parser.add_argument("--pid", required=True)
    init_parser.add_argument("--pgid", required=True)
    init_parser.add_argument("--training-gpus", required=True)
    init_parser.add_argument("--post-gpu", required=True)
    init_parser.add_argument("--minimum-memory-mib", required=True)
    init_parser.add_argument("--stages", required=True)
    init_parser.set_defaults(func=initialize)

    stage_parser = subparsers.add_parser("set-stage")
    stage_parser.add_argument("--path", required=True)
    stage_parser.add_argument("--stage", required=True)
    stage_parser.add_argument(
        "--state", required=True, choices=("running", "completed", "failed", "interrupted")
    )
    stage_parser.add_argument("--exit-code", type=int)
    stage_parser.set_defaults(func=set_stage)

    pipeline_parser = subparsers.add_parser("set-pipeline")
    pipeline_parser.add_argument("--path", required=True)
    pipeline_parser.add_argument(
        "--state", required=True, choices=("running", "completed", "failed", "interrupted")
    )
    pipeline_parser.add_argument("--current-stage")
    pipeline_parser.set_defaults(func=set_pipeline)

    show_parser = subparsers.add_parser("show")
    show_parser.add_argument("--path", required=True)
    show_parser.set_defaults(func=show)

    get_parser = subparsers.add_parser("get")
    get_parser.add_argument("--path", required=True)
    get_parser.add_argument("--field", required=True)
    get_parser.set_defaults(func=get_value)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
