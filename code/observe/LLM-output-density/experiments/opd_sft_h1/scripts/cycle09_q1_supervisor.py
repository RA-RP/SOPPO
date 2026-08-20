#!/usr/bin/env python3
"""Detached dual-GPU Q1 Stage-A postprocessing supervisor; never starts Stage B."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cycle09_block3_common as c


ROOT = c.Q1_ROOT / "postprocess_supervisor"
STATUS = ROOT / "status.json"
PID_FILE = ROOT / "supervisor.pid"
LOG_ROOT = ROOT / "logs"
SUPERVISOR_LOG = LOG_ROOT / "supervisor.log"
PYTHON = str(c.DENSITY_PYTHON)
POST = c.SCRIPT_DIR / "cycle09_q1_postprocess.py"
GEOMETRY = c.SCRIPT_DIR / "cycle09_q1_geometry.py"
BEHAVIOR = c.SCRIPT_DIR / "cycle09_q1_behavior.py"
PROBE_SOURCES = c.SCRIPT_DIR / "cycle09_q1_probe_sources.py"
HANDOFF = c.SCRIPT_DIR / "cycle09_q1_handoff.py"
STEPS = (0, 5, 20, 40, 80, 160)
PROBES = ("S_math", "E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general")


@dataclass(frozen=True)
class Unit:
    name: str
    argv: tuple[str, ...]
    planned_gpu_hours: float
    complete: Callable[[], bool]


class State:
    def __init__(self, mode: str):
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "pid": os.getpid(),
            "mode": mode,
            "state": "INITIALIZING",
            "started_utc": c.utc_now(),
            "updated_utc": c.utc_now(),
            "shutdown_policy": "never",
            "lanes": {str(gpu): {"state": "PENDING", "current": None, "completed": []} for gpu in (0, 1)},
            "failure": None,
        }
        self.write()

    def write(self) -> None:
        self.payload["updated_utc"] = c.utc_now()
        c.atomic_json(STATUS, self.payload)

    def update(self, **values: Any) -> None:
        with self.lock:
            self.payload.update(values)
            self.write()

    def lane(self, gpu: int, **values: Any) -> None:
        with self.lock:
            self.payload["lanes"][str(gpu)].update(values)
            self.write()

    def completed(self, gpu: int, name: str) -> None:
        with self.lock:
            self.payload["lanes"][str(gpu)]["completed"].append(name)
            self.write()


def live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ensure_singleton() -> None:
    if PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = -1
        if pid > 0 and pid != os.getpid() and live(pid):
            raise RuntimeError(f"Q1 postprocess supervisor already running as PID {pid}")


def environment(gpu: int | None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run_cpu(name: str, argv: list[str], state: State) -> None:
    log = LOG_ROOT / f"cpu__{name}.log"
    state.update(state="CPU", current_cpu=name, cpu_log=str(log))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{c.utc_now()}] RUN {' '.join(argv)}\n")
        handle.flush()
        completed = subprocess.run(argv, cwd=c.REPO, env=environment(None), stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} exited rc={completed.returncode}; log={log}")


def run_unit(unit: Unit, gpu: int, state: State) -> None:
    if unit.complete():
        state.completed(gpu, unit.name + ":cached")
        return
    run_id = c.budget_start(f"Q1_{unit.name}", gpu_count=1, planned_upper_gpu_hours=unit.planned_gpu_hours)
    log = LOG_ROOT / f"gpu{gpu}__{unit.name}.log"
    status, detail = "failed", ""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{c.utc_now()}] RUN {' '.join(unit.argv)}\n")
        handle.flush()
        process = subprocess.Popen(unit.argv, cwd=c.REPO, env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        state.lane(gpu, state="RUNNING", current=unit.name, child_pid=process.pid, log=str(log), budget_run_id=run_id)
        try:
            while process.poll() is None:
                time.sleep(20)
            if process.wait() != 0:
                detail = f"unit exited rc={process.returncode}"
                raise RuntimeError(f"{unit.name}: {detail}; log={log}")
            if not unit.complete():
                detail = "strict completion artifact absent"
                raise RuntimeError(f"{unit.name}: {detail}; log={log}")
            status, detail = "complete", "strict completion artifact verified"
        finally:
            ledger = c.budget_finish(run_id, status=status, detail=detail)
            state.lane(gpu, child_pid=None, current=None, consumed_gpu_hours=ledger["consumed_gpu_hours"], remaining_gpu_hours=ledger["remaining_gpu_hours"])
    state.completed(gpu, unit.name)


def geometry_complete(step: int, probe: str) -> bool:
    arm = "base" if step == 0 else "alpha05"
    path = c.Q1_ROOT / "geometry/cells" / arm / f"step_{step:03d}" / f"{probe}.json"
    return c.read_json(path, {}).get("status") == "complete"


def behavior_complete(step: int, smoke: bool) -> bool:
    branch = "smoke" if smoke else "formal"
    path = c.Q1_ROOT / "behavior" / branch / "cells" / f"step_{step:03d}" / "cell_manifest.json"
    return c.read_json(path, {}).get("status") == "complete"


def queue(units: list[Unit], state: State) -> None:
    pending = deque(units)
    guard = threading.Lock()
    failed = threading.Event()
    errors: list[str] = []

    def worker(gpu: int) -> None:
        try:
            while not failed.is_set():
                with guard:
                    if not pending:
                        break
                    unit = pending.popleft()
                run_unit(unit, gpu, state)
            state.lane(gpu, state="COMPLETE" if not failed.is_set() else "STOPPED")
        except Exception as error:
            failed.set()
            with guard:
                errors.append(f"gpu{gpu}: {error}")
            state.lane(gpu, state="FAILED", error=str(error), current=None)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))


def preflight() -> dict[str, Any]:
    stage_a = c.read_json(c.MINI / "qwen_alpha05_stage_a_training_manifest.json", {})
    gpus = c.gpu_inventory()
    payload = {
        "stage_a_validated": stage_a.get("status") == "complete_checkpoint_validated",
        "stage_b_not_started": stage_a.get("stage_b", {}).get("status") == "not_started",
        "gpus": gpus,
        "disk_free_bytes": __import__("shutil").disk_usage(c.AUTODL).free,
        "shutdown_policy": "never",
    }
    payload["complete"] = bool(
        payload["stage_a_validated"]
        and payload["stage_b_not_started"]
        and len(gpus) >= 2
        and all(row["memory_total_mib"] >= 90000 for row in gpus[:2])
        and payload["disk_free_bytes"] >= 100 * 2**30
    )
    c.atomic_json(ROOT / "preflight.json", payload)
    return payload


def smoke_units() -> list[Unit]:
    return [
        Unit("geometry_step5_e_math_smoke", (PYTHON, str(GEOMETRY), "--phase", "cell", "--step", "5", "--probe", "E_math", "--forward-batch-size", "2", "--max-batch-tokens", "4096", "--no-retain-factor"), 0.2, lambda: geometry_complete(5, "E_math")),
        Unit("behavior_step5_smoke", (PYTHON, str(BEHAVIOR), "--phase", "cell", "--step", "5", "--smoke"), 0.3, lambda: behavior_complete(5, True)),
    ]


def smoke_reference_units() -> list[Unit]:
    return [
        Unit(
            "geometry_base_e_math_smoke",
            (PYTHON, str(GEOMETRY), "--phase", "reference", "--probe", "E_math", "--forward-batch-size", "2", "--max-batch-tokens", "4096"),
            0.2,
            lambda: geometry_complete(0, "E_math"),
        )
    ]


def behavior_smoke_units() -> list[Unit]:
    return [
        Unit(
            "behavior_step5_smoke",
            (PYTHON, str(BEHAVIOR), "--phase", "cell", "--step", "5", "--smoke"),
            0.3,
            lambda: behavior_complete(5, True),
        )
    ]


def formal_units() -> list[Unit]:
    units: list[Unit] = []
    units.append(Unit("behavior_base", (PYTHON, str(BEHAVIOR), "--phase", "cell", "--step", "0"), 2.0, lambda: behavior_complete(0, False)))
    for step in STEPS[1:]:
        for probe in PROBES:
            units.append(Unit(f"geometry_{step:03d}_{probe}", (PYTHON, str(GEOMETRY), "--phase", "cell", "--step", str(step), "--probe", probe, "--no-retain-factor"), 0.4, lambda step=step, probe=probe: geometry_complete(step, probe)))
    for step in (20, 40, 160):
        units.append(Unit(f"behavior_{step:03d}", (PYTHON, str(BEHAVIOR), "--phase", "cell", "--step", str(step)), 4.0, lambda step=step: behavior_complete(step, False)))
    return units


def formal_reference_units() -> list[Unit]:
    return [
        Unit(
            f"geometry_base_{probe}",
            (PYTHON, str(GEOMETRY), "--phase", "reference", "--probe", probe, "--no-retain-factor"),
            0.4,
            lambda probe=probe: geometry_complete(0, probe),
        )
        for probe in PROBES
    ]


def behavior_formal_units() -> list[Unit]:
    return [
        Unit(
            f"behavior_{step:03d}",
            (PYTHON, str(BEHAVIOR), "--phase", "cell", "--step", str(step)),
            4.0,
            lambda step=step: behavior_complete(step, False),
        )
        for step in (0, 20, 40, 160)
    ]


def run(args: argparse.Namespace, state: State) -> None:
    check = preflight()
    if not check["complete"]:
        raise RuntimeError(f"Q1 postprocess preflight failed: {ROOT / 'preflight.json'}")
    if args.preflight_only:
        state.update(state="PREFLIGHT_COMPLETE")
        return
    if args.mode == "formal":
        smoke = c.read_json(ROOT / "smoke_manifest.json", {})
        if smoke.get("status") != "complete":
            raise RuntimeError("Q1 formal postprocessing requires a complete Q1 smoke manifest")
    if args.mode == "behavior-formal":
        smoke = c.read_json(ROOT / "behavior_smoke_manifest.json", {})
        if smoke.get("status") != "complete":
            raise RuntimeError("Q1 behavior formal postprocessing requires a complete behavior smoke manifest")
    run_cpu("validate", [PYTHON, str(POST), "--phase", "validate"], state)
    run_cpu("support_stats", [PYTHON, str(POST), "--phase", "support-stats"], state)
    export_steps = "5" if args.mode in {"smoke", "behavior-smoke"} else "5,20,40,80,160"
    run_cpu("export", [PYTHON, str(POST), "--phase", "export", "--steps", export_steps], state)
    if args.mode in {"smoke", "formal"}:
        run_cpu("probe_sources", [PYTHON, str(PROBE_SOURCES), "--phase", "prepare"], state)
        run_cpu("geometry_prepare", [PYTHON, str(GEOMETRY), "--phase", "prepare"], state)
    if args.mode == "smoke":
        state.update(state="GPU_REFERENCE_QUEUE")
        queue(smoke_reference_units(), state)
        state.update(state="GPU_QUEUE")
        queue(smoke_units(), state)
    elif args.mode == "formal":
        state.update(state="GPU_REFERENCE_QUEUE")
        queue(formal_reference_units(), state)
        state.update(state="GPU_QUEUE")
        queue(formal_units(), state)
    else:
        state.update(state="GPU_QUEUE")
        queue(
            behavior_smoke_units() if args.mode == "behavior-smoke" else behavior_formal_units(),
            state,
        )
    if args.mode == "smoke":
        payload = {"status": "complete", "mode": "smoke", "created_utc": c.utc_now(), "shutdown_policy": "never"}
        c.atomic_json(ROOT / "smoke_manifest.json", payload)
        state.update(state="COMPLETE", manifest=str(ROOT / "smoke_manifest.json"))
        return
    if args.mode == "behavior-smoke":
        payload = {"status": "complete", "mode": "behavior-smoke", "created_utc": c.utc_now(), "shutdown_policy": "never"}
        c.atomic_json(ROOT / "behavior_smoke_manifest.json", payload)
        state.update(state="COMPLETE", manifest=str(ROOT / "behavior_smoke_manifest.json"))
        return
    if args.mode == "behavior-formal":
        state.update(state="FINALIZING")
        run_cpu("behavior_finalize", [PYTHON, str(BEHAVIOR), "--phase", "finalize"], state)
        payload = {
            "status": "complete",
            "mode": "behavior-formal",
            "created_utc": c.utc_now(),
            "stage_b": "not_started",
            "behavior": c.artifact(c.MINI / "qwen_alpha05_behavior_manifest.json"),
            "gpu_budget_ledger": c.artifact(c.BUDGET_LEDGER),
            "shutdown_policy": "never",
        }
        c.atomic_json(ROOT / "behavior_formal_manifest.json", payload)
        state.update(state="COMPLETE", manifest=str(ROOT / "behavior_formal_manifest.json"))
        return
    state.update(state="FINALIZING")
    run_cpu("geometry_finalize", [PYTHON, str(GEOMETRY), "--phase", "finalize"], state)
    run_cpu("behavior_finalize", [PYTHON, str(BEHAVIOR), "--phase", "finalize"], state)
    payload = {
        "status": "complete",
        "mode": "formal",
        "created_utc": c.utc_now(),
        "stage_b": "not_started",
        "geometry": c.artifact(c.MINI / "qwen_alpha05_geometry_manifest.json"),
        "behavior": c.artifact(c.MINI / "qwen_alpha05_behavior_manifest.json"),
        "gpu_budget_ledger": c.artifact(c.BUDGET_LEDGER),
        "shutdown_policy": "never",
    }
    c.atomic_json(ROOT / "formal_manifest.json", payload)
    run_cpu("handoff", [PYTHON, str(HANDOFF)], state)
    state.update(state="COMPLETE", manifest=str(ROOT / "formal_manifest.json"))


def detach() -> int:
    ensure_singleton()
    ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    child = [item for item in sys.argv[1:] if item != "--detach"]
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), *child], cwd=c.REPO, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"pid": process.pid, "status": str(STATUS), "log": str(SUPERVISOR_LOG)}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal", "behavior-smoke", "behavior-formal"), required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else "{}")
        return 0
    if args.detach:
        return detach()
    ensure_singleton()
    ROOT.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    state = State(args.mode)
    try:
        run(args, state)
        return 0
    except Exception as error:
        state.update(state="FAILED", failure=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
