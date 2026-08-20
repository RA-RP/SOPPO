#!/usr/bin/env python3
"""Detached, resumable supervisor for Cycle 09 block-2 Qwen GPU work."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/root/LLM-output-density")
PYTHON = Path("/root/miniconda3/envs/density/bin/python")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
BLOCK_ROOT = Path("/root/autodl-tmp/cycle09_block2")
LOG_ROOT = BLOCK_ROOT / "logs"
SEQKD_ROOT = Path("/root/autodl-tmp/cycle09_seqkd")
TRAIN_MANIFEST = SEQKD_ROOT / "checkpoints/training_manifest.json"
STATUS = BLOCK_ROOT / "qwen_chain_status.json"
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
POLL_SECONDS = 20
MAX_TRAIN_RESTARTS = 4

TRAIN_COMMAND = (
    str(PYTHON),
    str(SCRIPTS / "cycle09_seqkd_train.py"),
    "--resume",
    "auto",
    "--gradient-checkpointing",
)

STAGES = (
    (
        "G2_seqkd_eval",
        (str(PYTHON), str(SCRIPTS / "cycle09_seqkd_eval.py")),
        SEQKD_ROOT / "eval/formal/evaluation_manifest.json",
    ),
    (
        "G3_seqkd_geometry",
        (
            str(PYTHON),
            str(SCRIPTS / "cycle09_seqkd_geometry.py"),
            "--steps",
            "0,5,10,20,40,80,160,320,480,624",
            "--device",
            "cuda",
        ),
        MINI / "seqkd_geometry_manifest.json",
    ),
    (
        "G8_adapter_ablation",
        (str(PYTHON), str(SCRIPTS / "cycle09_g8_adapter_ablation.py")),
        MINI / "G8_adapter_ablation_manifest.json",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def atomic_status(**updates: Any) -> dict[str, Any]:
    current = read_json(STATUS)
    current.update(updates)
    current["updated_at"] = utc_now()
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(current, indent=2), encoding="utf-8")
    temporary.replace(STATUS)
    print(f"[supervisor] {json.dumps(updates, sort_keys=True)}", flush=True)
    return current


def process_running(script_name: str) -> bool:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if script_name in command and "--smoke" not in command:
            return True
    return False


def training_complete() -> bool:
    manifest = read_json(TRAIN_MANIFEST)
    return (
        manifest.get("status") == "complete"
        and int(manifest.get("completed_steps", -1)) == 624
    )


def launch_training(restart_number: int) -> int:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    log_path = LOG_ROOT / "g1_seqkd_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            TRAIN_COMMAND,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    atomic_status(
        state="waiting_for_G1",
        train_pid=process.pid,
        train_restart_count=restart_number,
    )
    return process.pid


def wait_for_training() -> None:
    restarts = int(read_json(STATUS).get("train_restart_count", 0))
    missing_since: float | None = None
    last_reported_step = -1
    while not training_complete():
        manifest = read_json(TRAIN_MANIFEST)
        step = int(manifest.get("completed_steps", -1))
        if step != last_reported_step:
            atomic_status(
                state="waiting_for_G1",
                g1_status=manifest.get("status", "missing"),
                g1_completed_steps=step,
            )
            last_reported_step = step

        if process_running("cycle09_seqkd_train.py"):
            missing_since = None
        else:
            missing_since = missing_since or time.monotonic()
            if time.monotonic() - missing_since >= 30:
                if restarts >= MAX_TRAIN_RESTARTS:
                    raise RuntimeError(
                        f"G1 exited repeatedly; restart limit={MAX_TRAIN_RESTARTS}"
                    )
                restarts += 1
                launch_training(restarts)
                missing_since = None
        time.sleep(POLL_SECONDS)
    atomic_status(state="G1_complete", g1_completed_steps=624)


def manifest_complete(path: Path) -> bool:
    return read_json(path).get("status") == "complete"


def run_stage(name: str, command: tuple[str, ...], manifest: Path) -> None:
    if manifest_complete(manifest):
        atomic_status(state=f"{name}_cached", current_stage=name)
        return
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1"})
    log_path = LOG_ROOT / f"{name}.log"
    atomic_status(
        state=f"running_{name}",
        current_stage=name,
        stage_command=list(command),
        stage_log=str(log_path),
    )
    with log_path.open("ab", buffering=0) as log:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"{name} exited with code {result.returncode}; log={log_path}")
    if not manifest_complete(manifest):
        raise RuntimeError(f"{name} exited 0 without complete manifest: {manifest}")
    atomic_status(state=f"{name}_complete", current_stage=name)


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_status(
        status="running",
        state="starting",
        supervisor_pid=os.getpid(),
        supervisor_sid=os.getsid(0),
        started_at=read_json(STATUS).get("started_at", utc_now()),
    )
    try:
        wait_for_training()
        for name, command, manifest in STAGES:
            run_stage(name, command, manifest)
    except Exception as exc:
        atomic_status(status="failed", state="failed", error=repr(exc))
        raise
    atomic_status(
        status="complete",
        state="complete",
        current_stage=None,
        completed_at=utc_now(),
    )


if __name__ == "__main__":
    main()
