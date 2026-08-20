#!/usr/bin/env python3
"""Detached GPU1 continuation for the Cycle 09 block-2 dual-card schedule."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO = Path("/root/LLM-output-density")
PYTHON = Path("/root/miniconda3/envs/density/bin/python")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
BLOCK_ROOT = Path("/root/autodl-tmp/cycle09_block2")
LOG_ROOT = BLOCK_ROOT / "logs"
STATUS = BLOCK_ROOT / "gpu1_accelerator_status.json"
G2_WORKER_STATUS = BLOCK_ROOT / "g2_gpu1_math_status.json"
G2_MANIFEST = Path("/root/autodl-tmp/cycle09_seqkd/eval/formal/evaluation_manifest.json")
QWEN_STATUS = BLOCK_ROOT / "qwen_chain_status.json"
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
G3_MANIFEST = MINI / "seqkd_geometry_manifest.json"
G3_WORKER_STATUS = Path(
    "/root/autodl-tmp/cycle09_seqkd/geometry/workers/gpu1_high.json"
)
G8_WORKER_STATUS = BLOCK_ROOT / "g8_gpu1_high_status.json"
FINAL_MANIFEST = MINI / "block2_completion_manifest.json"
POLL_SECONDS = 20

G3_COMMAND = (
    str(PYTHON),
    str(SCRIPTS / "cycle09_seqkd_geometry.py"),
    "--steps",
    "0,5,10,20,40,80,160,320,480,624",
    "--worker-only",
    "--worker-steps",
    "624,480,320,160,80",
    "--worker-id",
    "gpu1_high",
    "--device",
    "cuda",
)
G8_COMMAND = (
    str(PYTHON),
    str(SCRIPTS / "cycle09_g8_adapter_ablation.py"),
    "--configs",
    "all_closed,close_30_35,close_24_29,close_18_23",
    "--worker-only",
    "--worker-id",
    "gpu1_high",
)
FINALIZE_COMMAND = (
    str(PYTHON),
    str(SCRIPTS / "cycle09_block2_finalize.py"),
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
    os.replace(temporary, STATUS)
    return current


def pid_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def wait_for(
    label: str,
    path: Path,
    predicate: Callable[[dict[str, Any]], bool],
    *,
    worker_process_required: bool = False,
) -> dict[str, Any]:
    missing_process_since: float | None = None
    atomic_status(state=f"waiting_{label}", wait_path=str(path))
    while True:
        payload = read_json(path)
        if predicate(payload):
            return payload
        if payload.get("status") == "failed":
            raise RuntimeError(f"{label} failed: {payload.get('error', payload)}")
        qwen = read_json(QWEN_STATUS)
        if label != "G2_worker" and qwen.get("status") == "failed":
            raise RuntimeError(f"Qwen supervisor failed while waiting for {label}: {qwen}")
        if worker_process_required and payload:
            pid = int(payload.get("pid", -1))
            if pid_alive(pid):
                missing_process_since = None
            else:
                missing_process_since = missing_process_since or time.monotonic()
                if time.monotonic() - missing_process_since >= 60:
                    raise RuntimeError(
                        f"{label} process {pid} disappeared without complete status"
                    )
        time.sleep(POLL_SECONDS)


def run_stage(
    label: str,
    command: tuple[str, ...],
    log_path: Path,
    worker_status: Path,
) -> None:
    if read_json(worker_status).get("status") == "complete":
        atomic_status(state=f"{label}_cached")
        return
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_status(
        state=f"running_{label}",
        current_stage=label,
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
        raise RuntimeError(f"{label} exited {result.returncode}; log={log_path}")
    worker = read_json(worker_status)
    if worker.get("status") != "complete":
        raise RuntimeError(f"{label} exited 0 without complete worker status: {worker}")
    atomic_status(state=f"{label}_complete", current_stage=label)


def main() -> None:
    atomic_status(
        status="running",
        state="starting",
        supervisor_pid=os.getpid(),
        supervisor_sid=os.getsid(0),
        started_at=read_json(STATUS).get("started_at", utc_now()),
    )
    try:
        wait_for(
            "G2_worker",
            G2_WORKER_STATUS,
            lambda payload: payload.get("status") == "complete"
            and payload.get("completed_steps") == [320, 480],
            worker_process_required=True,
        )
        wait_for(
            "G2_main",
            G2_MANIFEST,
            lambda payload: payload.get("status") == "complete"
            and set(payload.get("completed_steps", []))
            == {0, 5, 10, 20, 40, 80, 160, 320, 480, 624},
        )
        run_stage(
            "G3_gpu1_high",
            G3_COMMAND,
            LOG_ROOT / "G3_gpu1_high.log",
            G3_WORKER_STATUS,
        )
        wait_for(
            "G3_main_finalize",
            G3_MANIFEST,
            lambda payload: payload.get("status") == "complete"
            and payload.get("steps") == [0, 5, 10, 20, 40, 80, 160, 320, 480, 624],
        )
        run_stage(
            "G8_gpu1_high",
            G8_COMMAND,
            LOG_ROOT / "G8_gpu1_high.log",
            G8_WORKER_STATUS,
        )
        wait_for(
            "Qwen_main_finalize",
            QWEN_STATUS,
            lambda payload: payload.get("status") == "complete",
        )
        run_stage(
            "block2_finalize",
            FINALIZE_COMMAND,
            LOG_ROOT / "block2_finalize.log",
            FINAL_MANIFEST,
        )
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
