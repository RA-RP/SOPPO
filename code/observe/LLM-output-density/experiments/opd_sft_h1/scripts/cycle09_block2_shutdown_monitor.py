#!/usr/bin/env python3
"""Fail-stop completion validator and AutoDL shutdown monitor for block 2."""

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
BLOCK = Path("/root/autodl-tmp/cycle09_block2")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
QWEN = BLOCK / "qwen_chain_status.json"
LLAMA = BLOCK / "llama_chain_status.json"
ACCELERATOR = BLOCK / "gpu1_accelerator_status.json"
FINAL = MINI / "block2_completion_manifest.json"
STATUS = BLOCK / "shutdown_monitor_status.json"
ABORT = BLOCK / "ABORT_SHUTDOWN"
LOG = BLOCK / "logs/block2_shutdown_monitor.log"
POLL_SECONDS = 30
GRACE_SECONDS = 300


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_status(state: str, detail: str) -> None:
    payload = {
        "state": state,
        "detail": detail,
        "monitor_pid": os.getpid(),
        "updated_at": utc_now(),
        "abort_path": str(ABORT),
    }
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, STATUS)


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{utc_now()}] {message}\n")


def run_finalizer() -> None:
    if read_json(FINAL).get("status") == "complete":
        return
    command = [str(PYTHON), str(SCRIPTS / "cycle09_block2_finalize.py")]
    finalizer_log = BLOCK / "logs/block2_finalize.log"
    with finalizer_log.open("ab", buffering=0) as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0 or read_json(FINAL).get("status") != "complete":
        raise RuntimeError(
            f"block2 finalizer failed with code {result.returncode}; log={finalizer_log}"
        )


def all_complete() -> bool:
    return all(
        read_json(path).get("status") == "complete"
        for path in (QWEN, LLAMA, ACCELERATOR)
    )


def failed_status() -> tuple[str, dict[str, Any]] | None:
    for label, path in (("qwen", QWEN), ("llama", LLAMA), ("accelerator", ACCELERATOR)):
        payload = read_json(path)
        if payload.get("status") == "failed":
            return label, payload
    return None


def main() -> None:
    write_status("ARMED", "waiting for Qwen, Llama, and GPU1 accelerator completion")
    log("monitor armed")
    while not all_complete():
        failure = failed_status()
        if failure is not None:
            label, payload = failure
            write_status("STOPPED_ON_FAILURE", f"{label} failed: {payload.get('error')}")
            log(f"{label} failed; leaving instance online")
            return
        if ABORT.exists():
            write_status("CANCELLED", "ABORT_SHUTDOWN exists")
            log("shutdown cancelled before completion")
            return
        time.sleep(POLL_SECONDS)

    try:
        write_status("FINALIZING", "all execution chains complete; validating artifacts")
        run_finalizer()
    except Exception as exc:
        write_status("STOPPED_ON_FAILURE", repr(exc))
        log(f"finalizer failed; leaving instance online: {exc!r}")
        return

    write_status("SHUTDOWN_PENDING", f"validated; grace_seconds={GRACE_SECONDS}")
    log(f"validated completion; shutdown in {GRACE_SECONDS}s; cancel with touch {ABORT}")
    for _remaining in range(GRACE_SECONDS, 0, -1):
        if ABORT.exists():
            write_status("CANCELLED", "ABORT_SHUTDOWN found during grace")
            log("shutdown cancelled during grace")
            return
        time.sleep(1)

    write_status("SHUTDOWN_REQUESTED", "grace elapsed; calling /usr/bin/shutdown")
    log("calling AutoDL shutdown")
    os.sync()
    subprocess.run(["/usr/bin/shutdown"], check=False)


if __name__ == "__main__":
    main()
