#!/usr/bin/env python3
"""Cycle 09 block 2 C1: extend S1-6 direction analysis to every static probe."""

from __future__ import annotations

import argparse
import gc
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

import cycle09_r4_campaign as camp
import cycle09_r4_common as c4
import cycle09_s1_6_direction as direction


RUN_ROOT = Path("/root/autodl-tmp/cycle09_s1/s1_6_cache_all")
MINI = c4.MINI_ROOT
TASKS = (
    "legacy_S_math",
    "E_general",
    "E_math_hard",
    "S_bos__g3",
    "S_bos__g17",
    "S_bos__g31",
)
R5_PARITY_TASKS = {"legacy_S_math", "E_general", "E_math_hard"}
OUTPUTS = {
    "analysis": MINI / "S1_direction_analysis.csv",
    "principal": MINI / "S1_direction_principal_angles.csv",
    "ranks": MINI / "S1_direction_rank_distribution.csv",
    "overlap": MINI / "S1_direction_overlap.csv",
}
MANIFEST = MINI / "C1_direction_all_probes_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    unknown = sorted(set(args.tasks) - set(TASKS))
    if not args.tasks or unknown:
        parser.error(f"invalid tasks; unknown={unknown}")
    return args


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def merge_task(path: Path, task: str, frame: pd.DataFrame) -> None:
    if path.is_file():
        existing = pd.read_csv(path)
        existing = existing[existing["task_id"] != task]
        frame = frame.reindex(columns=existing.columns)
        combined = pd.concat([existing, frame], ignore_index=True)
    else:
        combined = frame.copy()
    sort_columns = [name for name in ("task_id", "arm", "step", "module", "space") if name in combined]
    atomic_csv(path, combined.sort_values(sort_columns, kind="stable").reset_index(drop=True))


def task_protocol(task: str, args: argparse.Namespace, modules: tuple[str, ...], arms: tuple[str, ...], steps: tuple[int, ...]) -> tuple[dict, str]:
    reference = c4.RUN_ROOT / "scratch/references" / f"{task}.pt"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    protocol = {
        "version": "block2-c1-all-static-probes-v1",
        "mode": "smoke" if args.smoke else "formal",
        "task": task,
        "layer": direction.LAYER,
        "epsilon": direction.EPSILON,
        "arms": list(arms),
        "steps": list(steps),
        "modules": list(modules),
        "reference_path": str(reference),
        "reference_sha256": direction.sha256_file(reference),
        "weight_load_dtype": "float16, matching R5-A2",
        "svd_dtype": "float64",
        "reorthogonalization": "float64 reduced QR",
        "principal_angles": "Bjorck-Golub arccos(svdvals(Q0.T @ Qt))",
        "base_direction_angle": "arccos(||Qt.T @ q0_j||_2)",
        "large_rotation_threshold_deg": 5.0,
        "device": args.device,
    }
    return protocol, direction.sha256_json(protocol)


def run_task(task: str, args: argparse.Namespace):
    arms = ("opd",) if args.smoke else direction.ARMS
    steps = (5,) if args.smoke else direction.STEPS
    modules = (c4.MODULES[0],) if args.smoke else c4.MODULES
    reference = c4.RUN_ROOT / "scratch/references" / f"{task}.pt"
    protocol, protocol_id = task_protocol(task, args, modules, arms, steps)

    direction.TASK = task
    direction.REFERENCE = reference
    scales = direction.reference_scales(args.device)
    base_model = camp.load_model(c4.BASE_MODEL, args.device)
    try:
        base_bases = direction.orthogonal_bases(base_model, scales, modules, args.device)
        direction._BASE_SIGMA.clear()
        direction._BASE_SIGMA.update(
            {module: base_bases[module]["sigma"].copy() for module in modules}
        )
        cells = {}
        cache_root = RUN_ROOT / ("smoke" if args.smoke else "formal") / task
        for arm in arms:
            for step in steps:
                cells[(arm, step)] = direction.load_or_compute(
                    root=cache_root,
                    arm=arm,
                    step=step,
                    base_bases=base_bases,
                    scales=scales,
                    modules=modules,
                    device=args.device,
                    protocol_id=protocol_id,
                )
    finally:
        camp.unload_model(base_model)
        scales.clear()
        if "base_bases" in locals():
            base_bases.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    analysis, principal, ranks, overlap = direction.build_tables(cells, modules)
    parity = (
        direction.parity_check(analysis, args.smoke)
        if task in R5_PARITY_TASKS
        else []
    )
    return analysis, principal, ranks, overlap, protocol, protocol_id, parity


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(max(1, min(32, torch.get_num_threads())))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else {
        "schema_version": 1,
        "status": "running",
        "task": "Cycle 09 block 2 C1",
        "requested_tasks": list(args.tasks),
        "existing_task_preserved": "E_ood",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_tasks": [],
        "task_protocols": {},
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    manifest["status"] = "running"
    manifest["requested_tasks"] = list(args.tasks)
    atomic_json(MANIFEST, manifest)

    for task in args.tasks:
        if not args.smoke and task in manifest["completed_tasks"]:
            print(f"[C1] cached complete {task}", flush=True)
            continue
        print(f"[C1] begin {task}", flush=True)
        analysis, principal, ranks, overlap, protocol, protocol_id, parity = run_task(task, args)
        if args.smoke:
            print(analysis.to_string(index=False), flush=True)
            print(ranks.to_string(index=False), flush=True)
            break
        expected_analysis = len(direction.ARMS) * len(direction.STEPS) * len(c4.MODULES) * 2
        expected_overlap = len(direction.STEPS) * len(c4.MODULES) * 2 * 3
        if len(analysis) != expected_analysis or len(overlap) != expected_overlap:
            raise RuntimeError(
                f"{task} row mismatch analysis={len(analysis)}/{expected_analysis} "
                f"overlap={len(overlap)}/{expected_overlap}"
            )
        for name, frame in (
            ("analysis", analysis),
            ("principal", principal),
            ("ranks", ranks),
            ("overlap", overlap),
        ):
            merge_task(OUTPUTS[name], task, frame)
        if task not in manifest["completed_tasks"]:
            manifest["completed_tasks"].append(task)
        manifest["task_protocols"][task] = {
            "protocol": protocol,
            "protocol_id": protocol_id,
            "r5_theta_parity": parity,
            "rows": {
                "analysis": len(analysis),
                "principal": len(principal),
                "ranks": len(ranks),
                "overlap": len(overlap),
            },
        }
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(MANIFEST, manifest)
        print(f"[C1] complete {task}", flush=True)

    if not args.smoke:
        expected_tasks = {"E_ood", *args.tasks}
        for name, path in OUTPUTS.items():
            frame = pd.read_csv(path)
            observed = set(frame["task_id"])
            missing = expected_tasks - observed
            if missing:
                raise RuntimeError(f"{name} missing tasks: {sorted(missing)}")
            manifest.setdefault("outputs", {})[name] = {
                "path": str(path),
                "rows": len(frame),
                "sha256": direction.sha256_file(path),
            }
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(MANIFEST, manifest)


if __name__ == "__main__":
    main()
