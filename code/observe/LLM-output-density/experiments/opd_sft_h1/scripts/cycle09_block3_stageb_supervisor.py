#!/usr/bin/env python3
"""Detached, fail-stop controller for the authorized Cycle09 160->320 delivery."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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


ROOT = c.RUN_ROOT / "stageb_320"
STATUS = ROOT / "status.json"
PID_FILE = ROOT / "supervisor.pid"
LOGS = ROOT / "logs"
LEDGER = ROOT / "gpu_ledger.json"
PYTHON = str(c.DENSITY_PYTHON)
SCRIPTS = c.SCRIPT_DIR
LLAMA_TRAIN = SCRIPTS / "run_cycle09_llama_opd.sh"
QWEN_TRAIN = SCRIPTS / "run_cycle09_q1_alpha05.sh"
PRUNER = SCRIPTS / "cycle08_ckpt_pruner.py"
LLAMA_POST = SCRIPTS / "cycle09_llama_opd_postprocess.py"
LLAMA_EXPORT = SCRIPTS / "cycle09_llama_model_export.py"
LLAMA_BEHAVIOR = SCRIPTS / "cycle09_llama_behavior.py"
LLAMA_GEOMETRY = SCRIPTS / "cycle09_llama_geometry.py"
QWEN_PREPARE = SCRIPTS / "cycle09_q1_prepare.py"
QWEN_POST = SCRIPTS / "cycle09_q1_stageb_postprocess.py"
QWEN_BEHAVIOR = SCRIPTS / "cycle09_q1_behavior.py"
QWEN_GEOMETRY = SCRIPTS / "cycle09_q1_geometry.py"
ARMS = ("opd", "sft", "offkd", "seqkd")
GRID = (0, 5, 10, 20, 40, 80, 160, 320)
QWEN_PROBES = ("S_math", "E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general")


@dataclass(frozen=True)
class Unit:
    name: str
    argv: tuple[str, ...]
    complete: Callable[[], bool]


class State:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "schema_version": 1, "status": "running", "state": "starting",
            "pid": os.getpid(), "started_utc": c.utc_now(), "updated_utc": c.utc_now(),
            "shutdown_policy": "never", "scope": "160_to_320_only",
            "lanes": {str(gpu): {"state": "idle", "current": None, "completed": []} for gpu in (0, 1)},
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
            self.payload["completed_units"] = int(self.payload.get("completed_units", 0)) + 1
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
            raise RuntimeError(f"Stage-B supervisor already running as PID {pid}")


def env_for(gpu: int | None = None, *, verl: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu),
    })
    if verl:
        env["PATH"] = f"{c.VERL_PYTHON.parent}:{env.get('PATH', '')}"
        env["PYTHONPATH"] = f"{c.VERL_ROOT}:{SCRIPTS}:{env.get('PYTHONPATH', '')}"
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("PYTORCH_ALLOC_CONF", None)
    else:
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return env


def complete_json(path: Path) -> bool:
    return c.read_json(path, {}).get("status") == "complete"


def checkpoint_complete(root: Path, step: int) -> bool:
    path = root / f"global_step_{step}" / "actor" / "model_world_size_1_rank_0.pt"
    return path.is_file() and path.stat().st_size > 0


def atomic_ledger(entry: dict[str, Any]) -> None:
    payload = c.read_json(LEDGER, {"schema_version": 1, "runs": []})
    payload["runs"].append(entry)
    payload["updated_utc"] = c.utc_now()
    c.atomic_json(LEDGER, payload)


def run_process(name: str, argv: list[str], state: State, *, gpu: int | None = None, verl: bool = False, completion: Callable[[], bool] | None = None) -> None:
    log = LOGS / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{c.utc_now()}] RUN {' '.join(argv)}\n")
        handle.flush()
        proc = subprocess.Popen(argv, cwd=c.REPO, env=env_for(gpu, verl=verl), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        if gpu is None:
            state.update(current_cpu=name, cpu_pid=proc.pid, state=name)
        else:
            state.lane(gpu, state="running", current=name, child_pid=proc.pid, log=str(log))
        while proc.poll() is None:
            time.sleep(20)
        rc = proc.wait()
    elapsed = time.monotonic() - started
    if gpu is not None:
        state.lane(gpu, state="idle", current=None, child_pid=None)
    if rc != 0:
        raise RuntimeError(f"{name} exited rc={rc}; log={log}")
    if completion is not None and not completion():
        raise RuntimeError(f"{name} returned but completion artifact is absent; log={log}")
    if gpu is not None:
        atomic_ledger({"name": name, "started_utc": c.utc_now(), "wall_seconds": elapsed, "gpu_count": 1, "gpu_hours": elapsed / 3600, "status": "complete"})


def run_training(name: str, script: Path, state: State, *, qwen: bool) -> None:
    root = c.Q1_CHECKPOINTS if qwen else c.L1_CHECKPOINTS
    if checkpoint_complete(root, 320):
        state.update(state=f"{name}_cached")
        return
    pruner_log = LOGS / f"{name}_pruner.log"
    pruner = subprocess.Popen([PYTHON, str(PRUNER), "--ckpt-root", str(root), "--grid", ",".join(map(str, GRID)), "--poll", "20"], cwd=c.REPO, env=env_for(None), stdout=pruner_log.open("a", encoding="utf-8"), stderr=subprocess.STDOUT, start_new_session=True)
    env_extra = ["MODE=formal", "L1_TOTAL_STEPS=320"] if not qwen else ["MODE=formal", "Q1_STAGE=stage_b", "TOTAL_TRAINING_STEPS=320"]
    command = ["env", *env_extra, "bash", str(script)]
    started = time.monotonic()
    committed_since: float | None = None
    try:
        # The postprocessors audit the continuous trainer log, including the first
        # 160 steps.  Keep the resumed output in the same durable log namespace.
        log = c.Q1_LOGS / "stage_b_train.log" if qwen else c.L1_LOGS / "train.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{c.utc_now()}] RUN {' '.join(command)}\n")
            handle.flush()
            proc = subprocess.Popen(command, cwd=c.REPO, env=env_for("0,1", verl=True), stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(state=name, training_pid=proc.pid, training_log=str(log), pruner_pid=pruner.pid)
            while proc.poll() is None:
                committed_now = checkpoint_complete(root, 320) and (
                    root / "latest_checkpointed_iteration.txt"
                ).read_text(encoding="utf-8").strip() == "320"
                if committed_now and committed_since is None:
                    committed_since = time.monotonic()
                    state.update(
                        state=f"{name}_checkpoint_320_committed_waiting_for_terminal_exit",
                        checkpoint_committed_utc=c.utc_now(),
                    )
                # The terminal async rollout is non-authoritative. If it stalls after
                # checkpoint persistence, release both cards for postprocessing.
                if committed_since is not None and time.monotonic() - committed_since > 600:
                    handle.write(f"[{c.utc_now()}] terminal callback exceeded 600s after checkpoint 320; terminating process group\n")
                    handle.flush()
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    break
                time.sleep(30)
            rc = proc.wait()
        # A final validation callback can fail after the durable 320 checkpoint; only the explicit
        # checkpoint commit decides whether this training stage is resumable/complete.
        committed = checkpoint_complete(root, 320) and (root / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip() == "320"
        if not committed:
            raise RuntimeError(f"{name} did not commit checkpoint 320 (rc={rc}); log={log}")
        atomic_ledger({"name": name, "ended_utc": c.utc_now(), "wall_seconds": time.monotonic() - started, "gpu_count": 2, "gpu_hours": 2 * (time.monotonic() - started) / 3600, "process_returncode": rc, "status": "checkpoint_committed"})
        state.update(training_pid=None, pruner_pid=None, state=f"{name}_checkpoint_320_committed", process_returncode=rc)
    finally:
        if pruner.poll() is None:
            os.killpg(os.getpgid(pruner.pid), signal.SIGTERM)
            pruner.wait(timeout=30)


def queue(units: list[Unit], state: State, label: str) -> None:
    pending = deque(units)
    lock = threading.Lock()
    failed = threading.Event()
    errors: list[str] = []
    state.update(state=label, total_units=len(units), completed_units=0)

    def worker(gpu: int) -> None:
        try:
            while not failed.is_set():
                with lock:
                    if not pending:
                        break
                    unit = pending.popleft()
                if unit.complete():
                    state.completed(gpu, unit.name + ":cached")
                    continue
                run_process(unit.name, list(unit.argv), state, gpu=gpu, completion=unit.complete)
                state.completed(gpu, unit.name)
            state.lane(gpu, state="complete" if not failed.is_set() else "stopped")
        except Exception as caught:  # fail-stop avoids contaminating a comparable campaign
            failed.set()
            with lock:
                errors.append(f"gpu{gpu}: {caught}")
            state.lane(gpu, state="failed", error=str(caught), current=None)

    workers = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in (0, 1)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if errors:
        raise RuntimeError("; ".join(errors))


def preflight() -> dict[str, Any]:
    early = c.read_json(c.RUN_ROOT / "p1_supervisor/p1_offline_early_manifest.json", {})
    gpus = c.gpu_inventory()
    payload = {
        "early_delivery_complete": early.get("status") == "complete",
        "gpu_inventory": gpus,
        "disk_free_bytes": shutil.disk_usage(c.AUTODL).free,
        "llama_step160": checkpoint_complete(c.L1_CHECKPOINTS, 160),
        "qwen_step160": checkpoint_complete(c.Q1_CHECKPOINTS, 160),
        "qwen_schedule320": c.file_check(c.Q1_DATA / "qwen_alpha05_schedule_320.parquet"),
        "shutdown_policy": "never",
    }
    payload["complete"] = bool(payload["early_delivery_complete"] and payload["llama_step160"] and payload["qwen_step160"] and payload["qwen_schedule320"]["complete"] and len(gpus) >= 2 and all(row["memory_total_mib"] >= 90000 for row in gpus[:2]) and payload["disk_free_bytes"] >= 120 * 2**30)
    c.atomic_json(ROOT / "preflight.json", payload)
    return payload


def llama_units() -> list[Unit]:
    out = []
    for arm in ARMS:
        behavior = c.RUN_ROOT / "llama_behavior/formal" / arm / "step_320/cell_manifest.json"
        geometry = c.RUN_ROOT / "llama_geometry/cells/formal" / arm / "step_320.json"
        out.extend((
            Unit(f"llama_behavior_{arm}_320", (PYTHON, str(LLAMA_BEHAVIOR), "--phase", "cell", "--arm", arm, "--step", "320"), lambda path=behavior: complete_json(path)),
            Unit(f"llama_geometry_{arm}_320", (PYTHON, str(LLAMA_GEOMETRY), "--phase", "cell", "--arm", arm, "--step", "320", "--device", "cuda:0"), lambda path=geometry: complete_json(path)),
        ))
    return out


def qwen_units() -> list[Unit]:
    behavior = c.Q1_ROOT / "behavior/formal/cells/step_320/cell_manifest.json"
    units = [Unit("qwen_alpha05_behavior_320", (PYTHON, str(QWEN_BEHAVIOR), "--phase", "cell", "--step", "320"), lambda: complete_json(behavior))]
    for probe in QWEN_PROBES:
        path = c.Q1_ROOT / "geometry/cells/alpha05/step_320" / f"{probe}.json"
        units.append(Unit(f"qwen_alpha05_geometry_320_{probe}", (PYTHON, str(QWEN_GEOMETRY), "--phase", "cell", "--step", "320", "--probe", probe, "--device", "cuda:0", "--no-retain-factor"), lambda path=path: complete_json(path)))
    return units


def run() -> None:
    ensure_singleton()
    ROOT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    state = State()
    try:
        check = preflight()
        if not check["complete"]:
            raise RuntimeError(f"Stage-B preflight failed: {ROOT / 'preflight.json'}")
        run_training("llama_opd_160_to_320", LLAMA_TRAIN, state, qwen=False)
        run_process("llama_postprocess", [PYTHON, str(LLAMA_POST), "--log", str(c.L1_LOGS / "train.log"), "--total-steps", "320", "--allow-rollout-gaps", "160,320"], state)
        run_process("llama_export_320", [PYTHON, str(LLAMA_EXPORT), "--arms", ",".join(ARMS), "--steps", "320"], state)
        queue(llama_units(), state, "llama_320_measurements")
        run_process("llama_behavior_finalize", [PYTHON, str(LLAMA_BEHAVIOR), "--phase", "finalize", "--arms", ",".join(ARMS), "--steps", ",".join(map(str, (0, 5, 20, 40, 80, 160, 320))), "--scope", "early_320"], state)
        run_process("llama_geometry_finalize", [PYTHON, str(LLAMA_GEOMETRY), "--phase", "finalize", "--arms", ",".join(ARMS), "--steps", ",".join(map(str, (0, 5, 20, 40, 80, 160, 320))), "--scope", "early_320"], state)
        run_process("qwen_schedule_validate", [PYTHON, str(QWEN_PREPARE), "--validate-only"], state)
        run_training("qwen_alpha05_160_to_320", QWEN_TRAIN, state, qwen=True)
        run_process("qwen_stageb_validate", [PYTHON, str(QWEN_POST), "--phase", "validate"], state)
        run_process("qwen_stageb_export", [PYTHON, str(QWEN_POST), "--phase", "export"], state)
        queue(qwen_units(), state, "qwen_alpha05_320_measurements")
        run_process("qwen_geometry_finalize", [PYTHON, str(QWEN_GEOMETRY), "--phase", "finalize"], state)
        run_process("qwen_behavior_finalize", [PYTHON, str(QWEN_BEHAVIOR), "--phase", "finalize"], state)
        manifest = {
            "schema_version": 1, "status": "complete", "scope": "authorized 160->320 only",
            "shutdown_policy": "never", "preflight": c.artifact(ROOT / "preflight.json"),
            "llama_behavior": c.artifact(c.MINI / "llama_early_320_behavior_manifest.json"),
            "llama_geometry": c.artifact(c.MINI / "llama_early_320_geometry_manifest.json"),
            "qwen_behavior": c.artifact(c.MINI / "qwen_alpha05_behavior_manifest.json"),
            "qwen_geometry": c.artifact(c.MINI / "qwen_alpha05_geometry_manifest.json"),
            "qwen_stageb": c.artifact(c.MINI / "qwen_alpha05_stage_b_training_manifest.json"),
            "created_utc": c.utc_now(),
        }
        c.atomic_json(ROOT / "stageb_320_manifest.json", manifest)
        state.update(status="complete", state="complete", manifest=str(ROOT / "stageb_320_manifest.json"), current_cpu=None)
    except Exception as caught:
        state.update(status="failed", state="failed", failure=repr(caught), current_cpu=None)
        raise
    finally:
        PID_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), ensure_ascii=False, indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
