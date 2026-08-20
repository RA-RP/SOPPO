#!/usr/bin/env python3
"""Detached, restart-safe supervisor for the approved O0--O5 closure.

The supervisor stops after O5.  O6 (full Llama/Qwen expansion) remains behind the
explicit Theory gate in the current handoff and is never auto-started here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import sys

REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_actual_output_trajectory as actual  # noqa: E402


SCRIPT = SCRIPTS / "cycle09_actual_output_trajectory.py"
PYTHON = b3.DENSITY_PYTHON
STATE = actual.ROOT / "supervisor_state.json"
LOGS = actual.ROOT / "logs"
PROBES = actual.PROBES
STEPS = (0, 20, 160, 320)


def write_state(phase: str, status: str, completed: list[dict[str, Any]], error: str | None = None) -> None:
    actual.atomic_json(STATE, {
        "schema_version": "cycle09_actual_output_supervisor_v1",
        "phase": phase,
        "status": status,
        "completed": completed,
        "error": error,
        "no_auto_shutdown": True,
        "stops_after": "O5; O6 requires Theory GO",
        "created_utc": actual.utc_now(),
    })


def command(*arguments: str) -> list[str]:
    return [str(PYTHON), str(SCRIPT), *arguments]


def run(*arguments: str) -> dict[str, Any]:
    cmd = command(*arguments)
    print("[actual-output] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)
    return {"command": arguments, "completed_utc": actual.utc_now()}


def smoke() -> None:
    completed: list[dict[str, Any]] = []
    write_state("smoke", "running", completed)
    completed.append(run("--phase", "audit"))
    completed.append(run(
        "--phase", "b2-smoke", "--model", "llama", "--arm", "opd", "--step", "20",
        "--probe", "E_math", "--measurement-n", "1", "--device", "cuda:0",
    ))
    completed.append(run(
        "--phase", "o4-cell", "--model", "llama", "--arm", "opd", "--step", "20",
        "--probe", "E_math", "--measurement-n", "1", "--selected-token-cap", "8", "--device", "cuda:1",
    ))
    completed.append(run("--phase", "finalize"))
    completed.append(run("--phase", "o5"))
    # Smoke-only caps are quarantined by a path suffix and cannot be consumed by formal cap=0 cells.
    token = actual.FINAL / "actual_checkpoint_token_kl_all.parquet"
    geometry = actual.FINAL / "actual_update_cumulative_geometry.csv"
    if not token.is_file() or not geometry.is_file() or token.stat().st_size == 0 or geometry.stat().st_size == 0:
        raise RuntimeError("smoke artifacts missing or empty")
    selection = actual.read_json(actual.AUDIT / "o1_effective_weight_selection.json", {})
    if selection.get("selected_effective_weight_object") != "serialized_merged_bf16_effective_difference":
        raise RuntimeError("O1 source-of-forward selection was not frozen to merged effective delta")
    write_state("smoke", "complete", completed)


def base_cells() -> list[tuple[str, int, str]]:
    return [("base", 0, probe) for probe in PROBES]


def arm_cells(arm: str) -> list[tuple[str, int, str]]:
    return [(arm, step, probe) for step in STEPS if step != 0 for probe in PROBES]


def execute_cells(cells: list[tuple[str, int, str]], device: str) -> list[dict[str, Any]]:
    completed = []
    for arm, step, probe in cells:
        completed.append(run(
            "--phase", "o4-cell", "--model", "llama", "--arm", arm, "--step", str(step),
            "--probe", probe, "--measurement-n", "0", "--selected-token-cap", "0", "--device", device,
        ))
    return completed


def formal() -> None:
    completed: list[dict[str, Any]] = []
    write_state("formal", "running", completed)
    completed.append(run("--phase", "audit"))
    # The O1 audit is cacheable.  Formal cells refuse to run without it.
    completed.append(run(
        "--phase", "b2-smoke", "--model", "llama", "--arm", "opd", "--step", "20",
        "--probe", "E_math", "--measurement-n", "1", "--device", "cuda:0",
    ))
    # Base profile/logit cache is built once then shared by both independent arm lanes.
    completed.extend(execute_cells(base_cells(), "cuda:0"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(execute_cells, arm_cells("opd"), "cuda:0"),
            pool.submit(execute_cells, arm_cells("offkd"), "cuda:1"),
        ]
        for future in futures:
            completed.extend(future.result())
    completed.append(run("--phase", "finalize"))
    completed.append(run("--phase", "o5"))
    write_state("formal", "complete_at_O5_gate", completed)


def dry_run() -> None:
    print(json.dumps({
        "scope": "O0/O1/B2-smoke -> O2/O3/O4 Llama OPD/off-KD -> O5; stop before O6",
        "gpu0": "B2 smoke, base cache, Llama OPD {20,160,320} x four probes",
        "gpu1": "Llama off-KD {20,160,320} x four probes",
        "cpu": "audit, parquet/CSV finalization, O5 grouped analysis",
        "formal_token_policy": "measurement_n=all; selected_token_cap=0",
        "smoke_token_policy": "measurement_n=1; selected_token_cap=8; separate paths",
        "restart": "atomic profile/base-logit/geometry/output manifests skip completed cells",
        "no_auto_shutdown": True,
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", "formal", "dry-run"))
    args = parser.parse_args()
    if args.phase == "dry-run":
        dry_run()
        return
    try:
        {"smoke": smoke, "formal": formal}[args.phase]()
    except Exception as error:
        write_state(args.phase, "failed", [], f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
