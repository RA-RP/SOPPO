#!/usr/bin/env python3
"""Detached two-GPU scheduler for Cycle09 block-3 Llama L2/L3."""

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


ROOT = c.RUN_ROOT / "p1_supervisor"
STATUS = ROOT / "status.json"
PID_FILE = ROOT / "supervisor.pid"
LOG_ROOT = ROOT / "logs"
SUPERVISOR_LOG = LOG_ROOT / "supervisor.log"
EXPORT = c.SCRIPT_DIR / "cycle09_llama_model_export.py"
PROBES = c.SCRIPT_DIR / "cycle09_llama_probe_prepare.py"
BEHAVIOR = c.SCRIPT_DIR / "cycle09_llama_behavior.py"
GEOMETRY = c.SCRIPT_DIR / "cycle09_llama_geometry.py"
CAP_PILOT = c.SCRIPT_DIR / "cycle09_llama_cap_pilot.py"
Q1_BEHAVIOR = c.SCRIPT_DIR / "cycle09_q1_behavior.py"
PYTHON = str(c.DENSITY_PYTHON)
EARLY_STEPS = (5, 20, 40, 80, 160)
SCOPE_ARMS = {
    "opd_early": ("opd",),
    "offline_early": ("sft", "offkd", "seqkd"),
}


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
            "lanes": {
                "gpu0": {"state": "PENDING", "current": None, "completed": []},
                "gpu1": {"state": "PENDING", "current": None, "completed": []},
            },
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

    def lane(self, lane: str, **values: Any) -> None:
        with self.lock:
            self.payload["lanes"][lane].update(values)
            self.write()

    def completed(self, lane: str, name: str) -> None:
        with self.lock:
            self.payload["lanes"][lane]["completed"].append(name)
            self.write()


def process_live(pid: int) -> bool:
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
        if pid > 0 and pid != os.getpid() and process_live(pid):
            raise RuntimeError(f"P1 supervisor already running as PID {pid}")
    status = c.read_json(STATUS, {})
    for lane in status.get("lanes", {}).values():
        child = lane.get("child_pid")
        if child and process_live(int(child)):
            raise RuntimeError(f"P1 child still running as PID {child}")


def environment(gpu: int | None) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
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
    return value


def stop_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=45)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_cpu(name: str, argv: list[str], state: State) -> None:
    log = LOG_ROOT / f"cpu__{name}.log"
    state.update(state="CPU_PREPARATION", current_cpu=name, cpu_log=str(log))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{c.utc_now()}] RUN {' '.join(argv)}\n")
        handle.flush()
        result = subprocess.run(
            argv,
            cwd=c.REPO,
            env=environment(None),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"CPU preparation failed rc={result.returncode}: {name}; log={log}")


def ledger_consumed_now() -> float:
    payload = c.load_ledger()
    c._refresh_ledger(payload)
    return float(payload["consumed_gpu_hours"])


def run_unit(unit: Unit, gpu: int, state: State, lane: str) -> None:
    if unit.complete():
        state.completed(lane, unit.name + ":cached")
        return
    run_id = c.budget_start(
        f"P1_{unit.name}", gpu_count=1, planned_upper_gpu_hours=unit.planned_gpu_hours
    )
    log = LOG_ROOT / f"{lane}__{unit.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    status = "failed"
    detail = ""
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{c.utc_now()}] RUN {' '.join(unit.argv)}\n")
        handle.flush()
        process = subprocess.Popen(
            unit.argv,
            cwd=c.REPO,
            env=environment(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state.lane(
            lane,
            state="RUNNING",
            current=unit.name,
            child_pid=process.pid,
            log=str(log),
            budget_run_id=run_id,
        )
        try:
            while process.poll() is None:
                consumed = ledger_consumed_now()
                state.lane(lane, consumed_gpu_hours=consumed)
                if consumed >= c.GPU_BUDGET_HOURS:
                    detail = "hard 72 GPU-hour budget reached"
                    stop_group(process)
                    status = "budget_stopped"
                    break
                time.sleep(20)
            return_code = process.wait()
            if status != "budget_stopped":
                if return_code != 0:
                    detail = f"unit exited rc={return_code}"
                    raise RuntimeError(f"{unit.name}: {detail}; log={log}")
                if not unit.complete():
                    detail = "unit exited without strict completion artifact"
                    raise RuntimeError(f"{unit.name}: {detail}; log={log}")
                status, detail = "complete", "strict completion artifact verified"
        finally:
            ledger = c.budget_finish(run_id, status=status, detail=detail)
            state.lane(
                lane,
                child_pid=None,
                current=None,
                consumed_gpu_hours=ledger["consumed_gpu_hours"],
                remaining_gpu_hours=ledger["remaining_gpu_hours"],
            )
    if status == "budget_stopped":
        raise RuntimeError(f"{unit.name}: {detail}")
    state.completed(lane, unit.name)


def behavior_complete(arm: str, step: int, smoke: bool) -> bool:
    branch = "smoke" if smoke else "formal"
    label = "base" if step == 0 else arm
    path = c.RUN_ROOT / "llama_behavior" / branch / label / f"step_{step:03d}"
    return c.read_json(path / "cell_manifest.json", {}).get("status") == "complete"


def geometry_complete(arm: str, step: int, smoke: bool) -> bool:
    branch = "smoke" if smoke else "formal"
    path = c.RUN_ROOT / "llama_geometry/cells" / branch / arm / f"step_{step:03d}.json"
    return c.read_json(path, {}).get("status") == "complete"


def geometry_base_complete(smoke: bool) -> bool:
    path = c.RUN_ROOT / (
        "llama_geometry/cells/smoke/base.json"
        if smoke
        else "llama_geometry/cells/base.json"
    )
    return c.read_json(path, {}).get("status") == "complete"


def cap_pilot_cell_complete(arm: str, step: int, cap: int) -> bool:
    path = (
        c.RUN_ROOT
        / "llama_cap_pilot/cells"
        / arm
        / f"step_{step:03d}"
        / f"cap_{cap}/cell_manifest.json"
    )
    return c.read_json(path, {}).get("status") == "complete"


def cap_pilot_complete() -> bool:
    return c.read_json(
        c.RUN_ROOT / "llama_cap_pilot/llama_early_cap_pilot_manifest.json", {}
    ).get("status") == "complete"


def q1_behavior_complete(step: int) -> bool:
    path = c.Q1_ROOT / "behavior/formal/cells" / f"step_{step:03d}/cell_manifest.json"
    return c.read_json(path, {}).get("status") == "complete"


def probe_args(args: argparse.Namespace, phase: str) -> list[str]:
    command = [PYTHON, str(PROBES), "--phase", phase]
    if args.math_source:
        command.extend(["--math-source", str(args.math_source)])
    if args.aime25_source:
        command.extend(["--aime25-source", str(args.aime25_source)])
    return command


def smoke_units() -> list[Unit]:
    return [
        Unit(
            "behavior_base_smoke",
            (PYTHON, str(BEHAVIOR), "--phase", "cell", "--arm", "base", "--step", "0", "--smoke"),
            0.5,
            lambda: behavior_complete("base", 0, True),
        ),
        Unit(
            "geometry_reference_smoke",
            (PYTHON, str(GEOMETRY), "--phase", "reference", "--smoke", "--measurement-n", "2"),
            0.5,
            lambda: geometry_base_complete(True),
        ),
        Unit(
            "behavior_opd_005_smoke",
            (PYTHON, str(BEHAVIOR), "--phase", "cell", "--arm", "opd", "--step", "5", "--smoke"),
            0.5,
            lambda: behavior_complete("opd", 5, True),
        ),
        Unit(
            "geometry_opd_005_smoke",
            (PYTHON, str(GEOMETRY), "--phase", "cell", "--arm", "opd", "--step", "5", "--smoke", "--measurement-n", "2"),
            0.5,
            lambda: geometry_complete("opd", 5, True),
        ),
    ]


def cap_pilot_units() -> list[Unit]:
    units = []
    for arm, step in (("base", 0), ("opd", 20)):
        for cap in (4096, 16384):
            units.append(
                Unit(
                    f"cap_pilot_{arm}_{step:03d}_{cap}",
                    (
                        PYTHON,
                        str(CAP_PILOT),
                        "--phase",
                        "cell",
                        "--arm",
                        arm,
                        "--step",
                        str(step),
                        "--cap",
                        str(cap),
                    ),
                    0.5,
                    lambda arm=arm, step=step, cap=cap: cap_pilot_cell_complete(
                        arm, step, cap
                    ),
                )
            )
    return units


def formal_units(scope: str) -> list[Unit]:
    units = []
    if scope == "opd_early":
        for step in (5, 80):
            units.append(
                Unit(
                    f"q1_behavior_{step:03d}",
                    (PYTHON, str(Q1_BEHAVIOR), "--phase", "cell", "--step", str(step)),
                    1.5,
                    lambda step=step: q1_behavior_complete(step),
                )
            )
    for step in EARLY_STEPS:
        for arm in SCOPE_ARMS[scope]:
            units.append(
                Unit(
                    f"behavior_{arm}_{step:03d}",
                    (PYTHON, str(BEHAVIOR), "--phase", "cell", "--arm", arm, "--step", str(step)),
                    1.5,
                    lambda arm=arm, step=step: behavior_complete(arm, step, False),
                )
            )
        for arm in SCOPE_ARMS[scope]:
            units.append(
                Unit(
                    f"geometry_{arm}_{step:03d}",
                    (PYTHON, str(GEOMETRY), "--phase", "cell", "--arm", arm, "--step", str(step)),
                    1.0,
                    lambda arm=arm, step=step: geometry_complete(arm, step, False),
                )
            )
    return units


def run_queue(units: list[Unit], state: State) -> None:
    queue = deque(units)
    queue_lock = threading.Lock()
    failed = threading.Event()
    errors: list[str] = []

    def worker(gpu: int) -> None:
        lane = f"gpu{gpu}"
        try:
            while not failed.is_set():
                with queue_lock:
                    if not queue:
                        break
                    unit = queue.popleft()
                run_unit(unit, gpu, state, lane)
            state.lane(lane, state="COMPLETE" if not failed.is_set() else "STOPPED", current=None)
        except Exception as error:
            failed.set()
            with queue_lock:
                errors.append(f"{lane}: {error}")
            state.lane(lane, state="FAILED", error=str(error), current=None)

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise RuntimeError("; ".join(errors))


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    l1 = c.read_json(c.MINI / "llama_opd_training_manifest.json", {})
    offline = {
        arm: c.read_json(c.LLAMA_OFFLINE_ROOT / f"{arm}/checkpoints/training_manifest.json", {})
        for arm in ("sft", "offkd", "seqkd")
    }
    math_candidates = (
        [args.math_source]
        if args.math_source
        else [
            c.AUTODL / "dataset/hendrycks_math/test.jsonl",
            c.AUTODL / "dataset/MATH/test.jsonl",
            c.AUTODL / "prepared/hendrycks_math_test.jsonl",
        ]
    )
    aime25_candidates = (
        [args.aime25_source]
        if args.aime25_source
        else [
            c.AUTODL / "dataset/aime25/test.parquet",
            c.AUTODL / "dataset/aime25/test.jsonl",
            c.AUTODL / "dataset/math-ai/aime25/test.parquet",
            c.AUTODL / "prepared/aime25.jsonl",
        ]
    )
    probe_sources = {
        "math_source": next((str(path) for path in math_candidates if path and path.is_file()), None),
        "aime25_source": next(
            (str(path) for path in aime25_candidates if path and path.is_file()), None
        ),
        "E_ood": str(c.AUTODL / "cycle09_r4/corpora/fixed/E_ood.jsonl"),
        "E_general": str(c.AUTODL / "cycle09_r4/corpora/fixed/E_general.jsonl"),
        "E_if": str(c.REPO / "Eval/tasks/data/ifeval/train.jsonl"),
        "MATH500_dedup": str(c.REPO / "Eval/tasks/data/hendrycks_math500/test.jsonl"),
        "AIME24_dedup": str(c.REPO / "Eval/tasks/data/aime24/train.jsonl"),
    }
    probe_missing = [
        name
        for name, path in probe_sources.items()
        if path is None or not Path(path).is_file()
    ]
    payload = {
        "l1_complete": l1.get("status")
        in {"complete", "complete_with_terminal_rollout_gap"},
        "offline_complete": {
            arm: manifest.get("status") == "complete" and int(manifest.get("completed_steps", -1)) == 624
            for arm, manifest in offline.items()
        },
        "probe_sources": probe_sources,
        "probe_source_candidates": {
            "math_source": [str(path) for path in math_candidates],
            "aime25_source": [str(path) for path in aime25_candidates],
        },
        "probe_sources_missing": probe_missing,
        "probe_sources_complete": not probe_missing,
        "gpus": c.gpu_inventory(),
        "shutdown_policy": "never",
    }
    payload["complete"] = bool(
        payload["l1_complete"]
        and all(payload["offline_complete"].values())
        and payload["probe_sources_complete"]
        and len(payload["gpus"]) >= 2
        and all(row["memory_total_mib"] >= 90000 for row in payload["gpus"][:2])
    )
    c.atomic_json(ROOT / "preflight.json", payload)
    return payload


def prepare(args: argparse.Namespace, state: State) -> None:
    if args.mode == "smoke":
        run_cpu(
            "export_smoke",
            [PYTHON, str(EXPORT), "--arms", "opd", "--steps", "0,5"],
            state,
        )
    elif args.mode == "cap-pilot":
        run_cpu(
            "export_cap_pilot",
            [PYTHON, str(EXPORT), "--arms", "opd", "--steps", "0,20"],
            state,
        )
        return
    else:
        run_cpu(
            f"export_{args.scope}",
            [
                PYTHON,
                str(EXPORT),
                "--arms",
                ",".join(SCOPE_ARMS[args.scope]),
                "--steps",
                "0," + ",".join(map(str, EARLY_STEPS)),
            ],
            state,
        )
    run_cpu("probe_fixed", probe_args(args, "fixed"), state)
    s_math = Unit(
        f"s_math_{args.mode}",
        tuple(probe_args(args, "s_math")),
        0.25,
        lambda: c.read_json(probes_manifest(), {}).get("status") == "complete",
    )
    run_unit(s_math, 0, state, "gpu0")


def probes_manifest() -> Path:
    return c.RUN_ROOT / "llama_geometry/corpora/probe_manifest.json"


def run(args: argparse.Namespace, state: State) -> None:
    payload = preflight(args)
    if not payload["complete"]:
        raise RuntimeError(f"P1 preflight incomplete: {ROOT / 'preflight.json'}")
    if args.preflight_only:
        state.update(state="PREFLIGHT_COMPLETE")
        return
    if args.mode in {"cap-pilot", "formal"}:
        smoke = c.read_json(ROOT / "smoke_manifest.json", {})
        if smoke.get("status") != "complete":
            raise RuntimeError("P1 cap-pilot/formal work is gated on complete P1 smoke")
    if args.mode == "formal" and not cap_pilot_complete():
        raise RuntimeError("formal P1 is gated on the complete paired Llama cap pilot")
    prepare(args, state)
    state.update(state="GPU_QUEUE")
    if args.mode == "smoke":
        run_queue(smoke_units(), state)
        manifest = {
            "status": "complete",
            "mode": "smoke",
            "created_utc": c.utc_now(),
            "units": [unit.name for unit in smoke_units()],
        }
        c.atomic_json(ROOT / "smoke_manifest.json", manifest)
        state.update(state="COMPLETE", manifest=str(ROOT / "smoke_manifest.json"))
        return

    if args.mode == "cap-pilot":
        run_queue(cap_pilot_units(), state)
        state.update(state="FINALIZING")
        run_cpu(
            "cap_pilot_finalize",
            [PYTHON, str(CAP_PILOT), "--phase", "finalize"],
            state,
        )
        pilot_manifest = c.RUN_ROOT / "llama_cap_pilot/llama_early_cap_pilot_manifest.json"
        if c.read_json(pilot_manifest, {}).get("status") != "complete":
            raise RuntimeError("cap pilot finalizer exited without a strict manifest")
        state.update(state="COMPLETE", manifest=str(pilot_manifest))
        return

    base_units = [
        Unit(
            "behavior_base",
            (PYTHON, str(BEHAVIOR), "--phase", "cell", "--arm", "base", "--step", "0"),
            1.5,
            lambda: behavior_complete("base", 0, False),
        ),
        Unit(
            "geometry_reference",
            (PYTHON, str(GEOMETRY), "--phase", "reference"),
            1.5,
            lambda: geometry_base_complete(False),
        ),
    ]
    run_queue(base_units, state)
    run_queue(formal_units(args.scope), state)
    state.update(state="FINALIZING")
    final_arms = c.ARMS if args.scope == "offline_early" else SCOPE_ARMS[args.scope]
    final_scope = "early" if args.scope == "offline_early" else args.scope
    shared = [
        "--arms",
        ",".join(final_arms),
        "--steps",
        ",".join(map(str, (0, *EARLY_STEPS))),
        "--scope",
        final_scope,
    ]
    run_cpu("behavior_finalize", [PYTHON, str(BEHAVIOR), "--phase", "finalize", *shared], state)
    run_cpu("geometry_finalize", [PYTHON, str(GEOMETRY), "--phase", "finalize", *shared], state)
    if args.scope == "opd_early":
        run_cpu(
            "q1_behavior_finalize",
            [PYTHON, str(Q1_BEHAVIOR), "--phase", "finalize"],
            state,
        )
    manifest = {
        "status": "complete",
        "mode": "formal",
        "scope": args.scope,
        "created_utc": c.utc_now(),
        "behavior_manifest": c.artifact(
            c.MINI
            / (
                "llama_early_behavior_manifest.json"
                if args.scope == "offline_early"
                else "llama_opd_early_behavior_manifest.json"
            )
        ),
        "geometry_manifest": c.artifact(
            c.MINI
            / (
                "llama_early_geometry_manifest.json"
                if args.scope == "offline_early"
                else "llama_opd_early_geometry_manifest.json"
            )
        ),
        "q1_alpha05_behavior_manifest": (
            c.artifact(c.MINI / "qwen_alpha05_behavior_manifest.json")
            if args.scope == "opd_early"
            else None
        ),
        "gpu_budget_ledger": c.artifact(c.BUDGET_LEDGER),
        "shutdown_policy": "never",
    }
    manifest_path = ROOT / f"p1_{args.scope}_manifest.json"
    c.atomic_json(manifest_path, manifest)
    state.update(state="COMPLETE", manifest=str(manifest_path))


def detach() -> int:
    ensure_singleton()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    child = [argument for argument in sys.argv[1:] if argument != "--detach"]
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [PYTHON, str(Path(__file__).resolve()), *child],
            cwd=c.REPO,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(json.dumps({"pid": process.pid, "status": str(STATUS), "log": str(SUPERVISOR_LOG)}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "cap-pilot", "formal"), required=True)
    parser.add_argument("--scope", choices=tuple(SCOPE_ARMS), default="opd_early")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--math-source", type=Path)
    parser.add_argument("--aime25-source", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else "{}")
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preflight": preflight(args),
                    "mode": args.mode,
                    "units": [
                        unit.name
                        for unit in (
                            smoke_units()
                            if args.mode == "smoke"
                            else cap_pilot_units()
                            if args.mode == "cap-pilot"
                            else formal_units(args.scope)
                        )
                    ],
                    "shutdown_policy": "never",
                },
                indent=2,
            )
        )
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
