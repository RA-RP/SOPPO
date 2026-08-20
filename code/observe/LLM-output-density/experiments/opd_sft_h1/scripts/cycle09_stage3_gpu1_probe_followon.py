#!/usr/bin/env python3
"""Run the independent Qwen PROBE-CORE partition on GPU1 after T-SUB.

This is deliberately separate from the long T-WHITE child on GPU0.  The
merged PROBE-CORE artifact is idempotent; the main Stage3 supervisor observes
it and skips its own duplicate partition step when it reaches that barrier.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import cycle09_stage3_followup_common as c


PYTHON = "/root/miniconda3/envs/density/bin/python"
ROOT = c.RUN_ROOT / "gpu1_probe_followon"
LOG = ROOT / "supervisor.log"
STATUS = ROOT / "status.json"
PID_FILE = ROOT / "supervisor.pid"
QWEN_SUB = c.RUN_ROOT / "H2_sub/T_SUB_qwen3_4b_manifest.json"
MERGED = c.RUN_ROOT / "H2_probe_core/PROBE_CORE_manifest.json"


def payload(path: Path) -> dict[str, Any]:
    return c.read_json(path, {}) if path.is_file() else {}


def complete(path: Path) -> bool:
    return str(payload(path).get("status", "")).startswith("complete")


def pid_alive() -> bool:
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def write(state: str, **extra: Any) -> None:
    c.atomic_json(
        STATUS,
        {
            "schema_version": 1,
            "status": state,
            "pid": os.getpid(),
            "auto_shutdown": False,
            "log": str(LOG),
            "updated_utc": c.utc_now(),
            **extra,
        },
    )


def child(argv: list[str], *, gpu: int | None = None, scope: str | None = None) -> None:
    env = os.environ.copy()
    env.pop("CYCLE09_STAGE3_SCOPE", None)
    if scope:
        env["CYCLE09_STAGE3_SCOPE"] = scope
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{c.utc_now()}] START {' '.join(argv)}\n")
        handle.flush()
        process = subprocess.Popen(argv, cwd=c.REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        rc = process.wait()
        handle.write(f"[{c.utc_now()}] END rc={rc}\n")
    if rc:
        raise RuntimeError(f"child failed rc={rc}")


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    c.atomic_text(PID_FILE, f"{os.getpid()}\n")
    if not complete(QWEN_SUB):
        raise RuntimeError(f"Qwen T-SUB is not complete: {QWEN_SUB}")
    if complete(MERGED):
        write("complete", skipped=True, artifact=str(MERGED))
        return
    try:
        write("running_qwen_probe_partition", prerequisite=str(QWEN_SUB))
        child(
            [
                PYTHON,
                str(c.SCRIPT_DIR / "cycle09_stage3_probe_core.py"),
                "--families",
                "qwen3_4b",
                "--phase",
                "all",
                "--device",
                "cuda:0",
            ],
            gpu=1,
            scope="partition_probe_qwen_20260723",
        )
        write("merging_probe_partitions")
        child(
            [
                PYTHON,
                str(c.SCRIPT_DIR / "cycle09_stage3_probe_core_merge.py"),
                "--phase",
                "merge",
            ]
        )
        if not complete(MERGED):
            raise RuntimeError(f"merge did not create a complete artifact: {MERGED}")
        write("complete", artifact=str(MERGED))
    except BaseException as error:
        write("failed", failure=f"{type(error).__name__}: {error}")
        raise


def detach() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    if pid_alive():
        raise RuntimeError(f"GPU1 follow-on already running: pid={PID_FILE.read_text().strip()}")
    with LOG.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            [PYTHON, str(Path(__file__).resolve()), "--run"],
            cwd=c.REPO,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    c.atomic_text(PID_FILE, f"{process.pid}\n")
    c.atomic_json(STATUS, {"schema_version": 1, "status": "detached", "pid": process.pid, "auto_shutdown": False, "log": str(LOG), "created_utc": c.utc_now()})
    return process.pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--detach", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.detach:
        print(json.dumps({"pid": detach(), "auto_shutdown": False}, indent=2))
    elif args.status:
        print(json.dumps(payload(STATUS), indent=2))
    else:
        run()
