#!/usr/bin/env python3
"""Detached, fail-closed two-GPU supervisor for the Cycle09 Stage-3 block."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cycle09_stage3_common as s3


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = Path("/root/miniconda3/envs/density/bin/python")
ROOT = s3.RUN_ROOT / "supervisor"
LOG_ROOT = ROOT / "logs"
STATUS = ROOT / "status.json"
PID_FILE = ROOT / "supervisor.pid"
ABORT_SHUTDOWN = ROOT / "ABORT_SHUTDOWN"
PREFLIGHT = ROOT / "preflight.json"


@dataclass(frozen=True)
class Command:
    name: str
    argv: tuple[str, ...]


def script(name: str, *arguments: str) -> tuple[str, ...]:
    return (str(PYTHON), str(SCRIPT_DIR / name), *arguments)


PREPARE = (
    Command(
        "prepare_c5",
        script("cycle09_c5_eif_geometry.py", "--phase", "prepare"),
    ),
    Command(
        "prepare_c8",
        script("cycle09_c8_training_ppl.py", "--phase", "prepare"),
    ),
    Command(
        "prepare_c11",
        script("cycle09_c11_answer_entropy.py", "--phase", "prepare"),
    ),
    Command(
        "mechanical_c14_c15",
        script("cycle09_c14c15_mechanical.py", "--task", "all"),
    ),
)

MODEL_SMOKE = Command(
    "real_base_model_smoke",
    script("cycle09_stage3_model_smoke.py"),
)

LANES = {
    "gpu0": (
        Command(
            "c2_opd_sft",
            script(
                "cycle09_c2c3_bootstrap.py",
                "--task", "c2", "--phase", "cells", "--arms", "opd,sft",
            ),
        ),
        Command(
            "c3_opd_rebound",
            script(
                "cycle09_c2c3_bootstrap.py",
                "--task", "c3", "--phase", "cells",
            ),
        ),
        Command(
            "c5_opd_sft",
            script(
                "cycle09_c5_eif_geometry.py",
                "--phase", "cells", "--arms", "opd,sft",
            ),
        ),
        Command(
            "c8_opd_sft",
            script(
                "cycle09_c8_training_ppl.py",
                "--phase", "cells", "--arms", "opd,sft",
            ),
        ),
        Command(
            "c11_opd_sft",
            script(
                "cycle09_c11_answer_entropy.py",
                "--phase", "cells", "--arms", "opd,sft",
            ),
        ),
    ),
    "gpu1": (
        Command(
            "c2_offkd_seqkd",
            script(
                "cycle09_c2c3_bootstrap.py",
                "--task", "c2", "--phase", "cells", "--arms", "offkd,seqkd",
            ),
        ),
        Command(
            "c5_offkd_seqkd",
            script(
                "cycle09_c5_eif_geometry.py",
                "--phase", "cells", "--arms", "offkd,seqkd",
            ),
        ),
        Command(
            "c8_offkd_seqkd",
            script(
                "cycle09_c8_training_ppl.py",
                "--phase", "cells", "--arms", "offkd,seqkd",
            ),
        ),
        Command(
            "c11_offkd_seqkd",
            script(
                "cycle09_c11_answer_entropy.py",
                "--phase", "cells", "--arms", "offkd,seqkd",
            ),
        ),
    ),
}

FINALIZE = (
    Command(
        "finalize_c2",
        script("cycle09_c2c3_bootstrap.py", "--task", "c2", "--phase", "finalize"),
    ),
    Command(
        "finalize_c3",
        script("cycle09_c2c3_bootstrap.py", "--task", "c3", "--phase", "finalize"),
    ),
    Command(
        "finalize_c5",
        script("cycle09_c5_eif_geometry.py", "--phase", "finalize"),
    ),
    Command(
        "finalize_c8",
        script("cycle09_c8_training_ppl.py", "--phase", "finalize"),
    ),
    Command(
        "finalize_c11",
        script("cycle09_c11_answer_entropy.py", "--phase", "finalize"),
    ),
)

EXPECTED_MANIFESTS = (
    s3.MINI / "C2_raw_er_manifest.json",
    s3.MINI / "C3_opd_overcompression_rebound_manifest.json",
    s3.MINI / "C5_eif_geometry_manifest.json",
    s3.MINI / "C8_training_corpus_ppl_manifest.json",
    s3.MINI / "C11_mmlupro_answer_token_entropy_manifest.json",
    s3.MINI / "C14_main_track_backfill_manifest.json",
    s3.MINI / "C15_cap_pilot_repair_manifest.json",
)


class State:
    def __init__(self, shutdown_mode: str):
        self.lock = threading.Lock()
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "state": "INITIALIZING",
            "pid": os.getpid(),
            "started_utc": s3.utc_now(),
            "updated_utc": s3.utc_now(),
            "shutdown_mode": shutdown_mode,
            "lanes": {
                name: {"state": "PENDING", "current": None, "completed": []}
                for name in LANES
            },
            "completed_global": [],
            "failure": None,
        }
        self.write()

    def write(self) -> None:
        self.payload["updated_utc"] = s3.utc_now()
        s3.atomic_json(STATUS, self.payload)

    def update(self, **values: Any) -> None:
        with self.lock:
            self.payload.update(values)
            self.write()

    def lane(self, lane: str, **values: Any) -> None:
        with self.lock:
            self.payload["lanes"][lane].update(values)
            self.write()

    def complete_command(self, lane: str | None, name: str) -> None:
        with self.lock:
            if lane is None:
                self.payload["completed_global"].append(name)
            else:
                self.payload["lanes"][lane]["completed"].append(name)
            self.write()


def gpu_inventory() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name, memory = [part.strip() for part in line.split(",", 2)]
        output.append(
            {"index": int(index), "name": name, "memory_total_mib": int(memory)}
        )
    return output


def preflight() -> dict[str, Any]:
    payload = s3.preflight_payload(
        ["c2", "c3", "c5", "c8", "c11", "c15"],
        list(s3.ARMS),
        list(s3.STEPS),
    )
    try:
        payload["gpus"] = gpu_inventory()
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        payload["gpus"] = []
        payload["gpu_error"] = str(error)
    payload["gpu_complete"] = (
        len(payload["gpus"]) >= 2
        and all(item["memory_total_mib"] >= 90000 for item in payload["gpus"][:2])
    )
    disk = shutil.disk_usage(s3.AUTODL)
    payload["disk"] = {
        "total_bytes": disk.total,
        "used_bytes": disk.used,
        "free_bytes": disk.free,
    }
    payload["complete"] = bool(payload["complete"] and payload["gpu_complete"])
    s3.atomic_json(PREFLIGHT, payload)
    return payload


def preflight_signature(payload: dict[str, Any]) -> str:
    return s3.sha256_json(
        {
            "files": [
                (item["path"], item["complete"], item["bytes"])
                for item in payload["files"]
            ],
            "models": [
                (
                    item["path"], item["complete"], item["config_bytes"],
                    item["weight_files"], item["weight_bytes"],
                )
                for item in payload["models"]
            ],
            "gpus": payload["gpus"],
        }
    )


def missing_summary(payload: dict[str, Any]) -> list[str]:
    missing = []
    for item in payload["files"]:
        if not item["complete"]:
            missing.append(item["path"])
    for item in payload["models"]:
        if not item["complete"]:
            missing.append(f"{item['arm']}/{item['step']}: {item['path']}")
    if not payload["gpu_complete"]:
        missing.append("two GPUs with at least 90000 MiB each")
    return missing


def wait_for_preflight(args: argparse.Namespace, state: State) -> dict[str, Any]:
    previous = None
    stable_since = None
    while True:
        payload = preflight()
        signature = preflight_signature(payload)
        if payload["complete"]:
            if signature == previous:
                stable_since = stable_since or time.monotonic()
            else:
                stable_since = time.monotonic()
            stable_for = time.monotonic() - stable_since
        else:
            stable_since = None
            stable_for = 0.0
        state.update(
            state="WAITING_FOR_DATA" if not payload["complete"] else "VERIFYING_DATA_STABILITY",
            preflight_path=str(PREFLIGHT),
            preflight_complete=payload["complete"],
            preflight_stable_seconds=stable_for,
            missing=missing_summary(payload),
        )
        if payload["complete"] and stable_for >= args.stable_seconds:
            return payload
        if not args.wait_for_data:
            raise RuntimeError(
                "preflight incomplete or not stable; see " + str(PREFLIGHT)
            )
        previous = signature
        time.sleep(args.poll_seconds)


def command_environment(gpu: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else gpu
    return environment


def run_command(
    command: Command,
    state: State,
    lane: str | None,
    gpu: str | None,
) -> None:
    log_path = LOG_ROOT / f"{lane or 'global'}__{command.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if lane is None:
        state.update(current_global=command.name)
    else:
        state.lane(lane, state="RUNNING", current=command.name, log=str(log_path))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{s3.utc_now()}] START {' '.join(command.argv)} gpu={gpu}\n"
        )
        log.flush()
        process = subprocess.Popen(
            command.argv,
            cwd=s3.REPO,
            env=command_environment(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        if lane is None:
            state.update(child_pid=process.pid)
        else:
            state.lane(lane, child_pid=process.pid)
        return_code = process.wait()
        log.write(f"[{s3.utc_now()}] END return_code={return_code}\n")
        log.flush()
    if return_code != 0:
        raise RuntimeError(
            f"command {command.name} failed with {return_code}; log={log_path}"
        )
    state.complete_command(lane, command.name)


def run_lane(
    lane: str,
    gpu: str,
    commands: tuple[Command, ...],
    state: State,
    failed: threading.Event,
) -> None:
    try:
        for command in commands:
            if failed.is_set():
                state.lane(lane, state="CANCELLED_AFTER_PEER_FAILURE", current=None)
                return
            run_command(command, state, lane, gpu)
        state.lane(lane, state="COMPLETE", current=None, child_pid=None)
    except Exception as error:
        failed.set()
        state.lane(lane, state="FAILED", current=None, error=str(error))
        raise


def validate_manifests() -> list[dict[str, Any]]:
    validated = []
    for path in EXPECTED_MANIFESTS:
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"manifest is not complete: {path}")
        outputs = list(payload.get("outputs", []))
        if isinstance(payload.get("output"), dict):
            outputs.append(payload["output"])
        for output in outputs:
            output_path = Path(output["path"])
            if not output_path.is_file() or output_path.stat().st_size == 0:
                raise FileNotFoundError(output_path)
            if output.get("sha256") and s3.sha256_file(output_path) != output["sha256"]:
                raise RuntimeError(f"artifact hash mismatch: {output_path}")
        validated.append(s3.artifact(path))
    return validated


def maybe_shutdown(args: argparse.Namespace, state: State, success: bool) -> None:
    should_shutdown = args.shutdown == "always" or (
        args.shutdown == "success" and success
    )
    if not should_shutdown:
        return
    state.update(
        state="SHUTDOWN_GRACE",
        shutdown_at_epoch=time.time() + args.shutdown_grace,
        shutdown_cancel_file=str(ABORT_SHUTDOWN),
    )
    deadline = time.monotonic() + args.shutdown_grace
    while time.monotonic() < deadline:
        if ABORT_SHUTDOWN.exists():
            state.update(state="SHUTDOWN_CANCELLED")
            return
        time.sleep(min(5.0, max(0.1, deadline - time.monotonic())))
    subprocess.run(["sync"], check=False)
    state.update(state="SHUTDOWN_REQUESTED")
    subprocess.run(["/usr/bin/shutdown"], check=False)


def process_is_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def ensure_not_running() -> None:
    if not PID_FILE.is_file():
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if pid != os.getpid() and process_is_live(pid):
        raise RuntimeError(f"Stage-3 supervisor already running as PID {pid}")


def detach() -> int:
    ensure_not_running()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    supervisor_log = LOG_ROOT / "supervisor.log"
    child_args = [argument for argument in sys.argv[1:] if argument != "--detach"]
    with supervisor_log.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(PYTHON), str(Path(__file__).resolve()), *child_args],
            cwd=s3.REPO,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=command_environment(None),
        )
    print(
        json.dumps(
            {
                "pid": process.pid,
                "status": str(STATUS),
                "log": str(supervisor_log),
            }
        )
    )
    return 0


def dry_run() -> None:
    payload = preflight()
    print(
        json.dumps(
            {
                "preflight_complete": payload["complete"],
                "missing": missing_summary(payload),
                "prepare": [list(item.argv) for item in PREPARE],
                "model_smoke": list(MODEL_SMOKE.argv),
                "lanes": {
                    lane: [list(item.argv) for item in commands]
                    for lane, commands in LANES.items()
                },
                "finalize": [list(item.argv) for item in FINALIZE],
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wait-for-data", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-seconds", type=int, default=180)
    parser.add_argument("--shutdown", choices=("never", "success", "always"), default="never")
    parser.add_argument("--shutdown-grace", type=int, default=120)
    args = parser.parse_args()
    if args.status:
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else "{}")
        return 0
    if args.dry_run:
        dry_run()
        return 0
    if args.detach:
        return detach()
    ensure_not_running()
    ROOT.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    state = State(args.shutdown)
    success = False
    try:
        s3.assert_contract()
        wait_for_preflight(args, state)
        if args.preflight_only:
            state.update(state="PREFLIGHT_COMPLETE")
            return 0
        state.update(state="PREPARING", missing=[])
        for command in PREPARE:
            run_command(command, state, None, None)
        run_command(MODEL_SMOKE, state, None, "0")
        state.update(state="GPU_LANES_RUNNING", current_global=None, child_pid=None)
        failed = threading.Event()
        errors = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(run_lane, "gpu0", "0", LANES["gpu0"], state, failed),
                executor.submit(run_lane, "gpu1", "1", LANES["gpu1"], state, failed),
            ]
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))
        state.update(state="FINALIZING")
        for command in FINALIZE:
            run_command(command, state, None, None)
        state.update(state="VALIDATING", current_global=None, child_pid=None)
        manifests = validate_manifests()
        state.update(state="COMPLETE", validated_manifests=manifests)
        success = True
        return 0
    except Exception as error:
        state.update(state="FAILED", failure=str(error))
        return 1
    finally:
        maybe_shutdown(args, state, success)


if __name__ == "__main__":
    raise SystemExit(main())
