#!/usr/bin/env python3
"""Detached, resumable supervisor for Cycle09 block-3 L1 smoke/formal runs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cycle09_block3_common as c


STATUS = c.RUN_ROOT / "supervisor/status.json"
PID_FILE = c.RUN_ROOT / "supervisor/supervisor.pid"
SUPERVISOR_LOG = c.RUN_ROOT / "supervisor/supervisor.log"
PREFLIGHT = c.RUN_ROOT / "supervisor/l1_preflight.json"
LAUNCH = c.SCRIPT_DIR / "run_cycle09_llama_opd.sh"
PATCHES = c.SCRIPT_DIR / "apply_verl_patches.sh"
PREPARE = c.SCRIPT_DIR / "cycle09_block3_prepare.py"
POSTPROCESS = c.SCRIPT_DIR / "cycle09_llama_opd_postprocess.py"
PRUNER = c.SCRIPT_DIR / "cycle08_ckpt_pruner.py"
PYTHON = str(c.DENSITY_PYTHON)


class State:
    def __init__(self, mode: str):
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "pid": os.getpid(),
            "mode": mode,
            "state": "INITIALIZING",
            "started_utc": c.utc_now(),
            "updated_utc": c.utc_now(),
            "shutdown_policy": "never",
            "child_pid": None,
            "pruner_pid": None,
            "failure": None,
        }
        self.write()

    def write(self) -> None:
        self.payload["updated_utc"] = c.utc_now()
        c.atomic_json(STATUS, self.payload)

    def update(self, **values: Any) -> None:
        self.payload.update(values)
        self.write()


def process_live(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ensure_singleton() -> None:
    if not PID_FILE.is_file():
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return
    if pid != os.getpid() and process_live(pid):
        raise RuntimeError(f"block-3 supervisor already running as PID {pid}")
    status = c.read_json(STATUS, {})
    child_pid = status.get("child_pid")
    if child_pid and process_live(int(child_pid)):
        raise RuntimeError(f"L1 training child is still running as PID {child_pid}")


def recover_stale_budget() -> None:
    payload = c.load_ledger()
    active = [row for row in payload["runs"] if "ended_at_unix" not in row]
    if not active:
        return
    status = c.read_json(STATUS, {})
    child_pid = status.get("child_pid")
    if child_pid and process_live(int(child_pid)):
        raise RuntimeError(f"cannot recover budget while child PID {child_pid} is live")
    cutoff = STATUS.stat().st_mtime if STATUS.is_file() else time.time()
    for row in active:
        row["ended_at_unix"] = max(float(row["started_at_unix"]), cutoff)
        row["ended_utc"] = c.utc_now()
        row["status"] = "interrupted_recovered"
        row["detail"] = "closed at last supervisor status mtime after process loss"
    c._refresh_ledger(payload)
    c.atomic_json(c.BUDGET_LEDGER, payload)


def signature(payload: dict[str, Any]) -> str:
    return c.sha256_json(
        {
            "models": [
                (row["path"], row["complete"], row["weight_files"], row["weight_bytes"])
                for row in payload["models"]
            ],
            "files": [(row["path"], row["complete"], row["bytes"]) for row in payload["files"]],
            "runtime": [(row["path"], row["complete"]) for row in payload["runtime"]],
            "gpus": payload["gpus"],
        }
    )


def missing(payload: dict[str, Any]) -> list[str]:
    rows = payload["models"] + payload["files"] + payload["runtime"]
    output = [row["path"] for row in rows if not row["complete"]]
    if not payload["gpu_complete"]:
        output.append("two GPUs with >=90000 MiB each")
    if payload["disk"]["free_bytes"] < 120 * 2**30:
        output.append("at least 120 GiB free under /root/autodl-tmp")
    return output


def wait_preflight(args: argparse.Namespace, state: State) -> dict[str, Any]:
    previous = None
    stable_since = None
    while True:
        payload = c.l1_preflight()
        c.atomic_json(PREFLIGHT, payload)
        current = signature(payload)
        if payload["complete"] and current == previous:
            stable_since = stable_since or time.monotonic()
        elif payload["complete"]:
            stable_since = time.monotonic()
        else:
            stable_since = None
        stable = time.monotonic() - stable_since if stable_since is not None else 0.0
        state.update(
            state="VERIFYING_DATA_STABILITY" if payload["complete"] else "WAITING_FOR_DATA",
            preflight=str(PREFLIGHT),
            preflight_complete=payload["complete"],
            stable_seconds=stable,
            missing=missing(payload),
        )
        if payload["complete"] and (
            not args.wait_for_data or stable >= args.stable_seconds
        ):
            return payload
        if not args.wait_for_data:
            raise RuntimeError(f"L1 preflight is incomplete/unstable; see {PREFLIGHT}")
        previous = current
        time.sleep(args.poll_seconds)


def base_environment(mode: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "VERL_ROOT": str(c.VERL_ROOT),
            "MODE": mode,
            "L1_TOTAL_STEPS": str(c.L1_FINAL_STEP),
        }
    )
    environment["PATH"] = f"{c.VERL_PYTHON.parent}:{environment.get('PATH', '')}"
    environment["PYTHONPATH"] = (
        f"{c.VERL_ROOT}:{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else str(c.VERL_ROOT)
    )
    environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    environment.pop("PYTORCH_ALLOC_CONF", None)
    return environment


def run_checked(argv: list[str], *, log: Path | None = None, env: dict[str, str] | None = None) -> None:
    if log is None:
        result = subprocess.run(argv, cwd=c.REPO, env=env)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{c.utc_now()}] RUN {' '.join(argv)}\n")
            handle.flush()
            result = subprocess.run(argv, cwd=c.REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}: {' '.join(argv)}")


def latest_step(root: Path) -> int:
    values = []
    for path in root.glob("global_step_*"):
        try:
            values.append(int(path.name.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return max(values, default=0)


def latest_rollout(root: Path) -> int:
    values = []
    for path in root.glob("*.jsonl"):
        try:
            values.append(int(path.stem.removeprefix("step_")))
        except ValueError:
            pass
    return max(values, default=0)


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, util, used, total = [item.strip() for item in line.split(",")]
        rows.append(
            {
                "index": int(index),
                "utilization_percent": int(util),
                "memory_used_mib": int(used),
                "memory_total_mib": int(total),
            }
        )
    return rows


def completed(mode: str) -> bool:
    root = c.L1_ROOT / "smoke" if mode == "smoke" else c.L1_ROOT
    target = 2 if mode == "smoke" else c.L1_FINAL_STEP
    return (root / f"checkpoints/global_step_{target}/actor").is_dir() and latest_rollout(
        root / "rollouts/raw"
    ) >= target


def stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=45)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def train(args: argparse.Namespace, state: State) -> Path:
    mode = args.mode
    root = c.L1_ROOT / "smoke" if mode == "smoke" else c.L1_ROOT
    checkpoint_root = root / "checkpoints"
    rollout_root = root / "rollouts/raw"
    log = root / "logs/train.log"
    target = 2 if mode == "smoke" else c.L1_FINAL_STEP
    grid = (1, 2) if mode == "smoke" else tuple(step for step in c.L1_CHECKPOINT_GRID if step)
    if completed(mode):
        state.update(state="TRAIN_ALREADY_COMPLETE", latest_step=target)
        return log

    recover_stale_budget()
    planned = args.planned_gpu_hours
    if planned is None:
        planned = 1.0 if mode == "smoke" else 42.0
    run_id = c.budget_start(
        f"L1_llama_opd_{mode}", gpu_count=2, planned_upper_gpu_hours=planned
    )
    environment = base_environment(mode)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    rollout_root.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    pruner_log = root / "logs/checkpoint_pruner.log"
    pruner_handle = pruner_log.open("a", encoding="utf-8")
    pruner = subprocess.Popen(
        [
            PYTHON,
            str(PRUNER),
            "--ckpt-root",
            str(checkpoint_root),
            "--grid",
            ",".join(map(str, grid)),
        ],
        cwd=c.REPO,
        stdout=pruner_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    status = "failed"
    detail = ""
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{c.utc_now()}] L1 {mode} launch\n")
        handle.flush()
        process = subprocess.Popen(
            ["/usr/bin/bash", str(LAUNCH)],
            cwd=c.REPO,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state.update(
            state="TRAINING",
            child_pid=process.pid,
            pruner_pid=pruner.pid,
            training_log=str(log),
            budget_run_id=run_id,
        )
        try:
            while process.poll() is None:
                ledger = c.load_ledger()
                active = next(row for row in ledger["runs"] if row["run_id"] == run_id)
                prior = sum(
                    float(row.get("gpu_hours", 0.0))
                    for row in ledger["runs"]
                    if row["run_id"] != run_id
                )
                active_gpu_hours = (time.time() - active["started_at_unix"]) / 3600 * 2
                if prior + active_gpu_hours >= c.GPU_BUDGET_HOURS:
                    detail = "hard GPU budget limit reached; stopped at latest recoverable checkpoint"
                    stop_process_group(process)
                    status = "budget_stopped"
                    break
                try:
                    gpus = gpu_snapshot()
                except (OSError, subprocess.SubprocessError, ValueError):
                    gpus = []
                state.update(
                    state="TRAINING",
                    latest_checkpoint=latest_step(checkpoint_root),
                    latest_rollout=latest_rollout(rollout_root),
                    target_step=target,
                    active_gpu_hours=active_gpu_hours,
                    projected_consumed_gpu_hours=prior + active_gpu_hours,
                    gpus=gpus,
                )
                time.sleep(args.monitor_seconds)
            return_code = process.wait()
            if status != "budget_stopped":
                if return_code != 0:
                    detail = f"training exited rc={return_code}"
                    raise RuntimeError(detail)
                if not completed(mode):
                    detail = f"training exited without complete step {target} + rollout dump"
                    raise RuntimeError(detail)
                status = "complete"
                detail = f"completed step {target}"
        finally:
            if pruner.poll() is None:
                pruner.terminate()
                try:
                    pruner.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pruner.kill()
            pruner_handle.close()
            ledger = c.budget_finish(run_id, status=status, detail=detail)
            state.update(
                child_pid=None,
                pruner_pid=None,
                gpu_budget_consumed=ledger["consumed_gpu_hours"],
                gpu_budget_remaining=ledger["remaining_gpu_hours"],
            )
    if status == "budget_stopped":
        raise RuntimeError(detail)
    return log


def prepare_and_patch(state: State) -> None:
    state.update(state="PREPARING_PROMPTS")
    run_checked([PYTHON, str(PREPARE)])
    state.update(state="PATCHING_VERL")
    run_checked(["/usr/bin/bash", str(PATCHES)], env=base_environment("formal"))
    run_checked(
        [
            PYTHON,
            str(c.SCRIPT_DIR / "patch_cycle09_verl_rollout_dump.py"),
            "--verl-root",
            str(c.VERL_ROOT),
            "--check",
        ]
    )


def postprocess(args: argparse.Namespace, state: State, log: Path) -> None:
    state.update(state="POSTPROCESSING", training_log=str(log))
    command = [PYTHON, str(POSTPROCESS), "--log", str(log)]
    if args.mode == "smoke":
        command.append("--smoke")
    else:
        command.extend(["--total-steps", str(c.L1_FINAL_STEP)])
    run_checked(command, log=(c.L1_ROOT / args.mode / "logs/postprocess.log") if args.mode == "smoke" else c.L1_LOGS / "postprocess.log")


def detach(args: argparse.Namespace) -> int:
    ensure_singleton()
    SUPERVISOR_LOG.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--postprocess-only", action="store_true")
    parser.add_argument("--wait-for-data", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-seconds", type=int, default=180)
    parser.add_argument("--monitor-seconds", type=int, default=20)
    parser.add_argument("--planned-gpu-hours", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        print(STATUS.read_text(encoding="utf-8") if STATUS.is_file() else "{}")
        return 0
    if args.dry_run:
        payload = c.l1_preflight()
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "preflight_complete": payload["complete"],
                    "missing": missing(payload),
                    "prepare": [PYTHON, str(PREPARE)],
                    "patch": ["/usr/bin/bash", str(PATCHES)],
                    "train": ["/usr/bin/bash", str(LAUNCH)],
                    "postprocess": [PYTHON, str(POSTPROCESS)],
                    "shutdown_policy": "never",
                },
                indent=2,
            )
        )
        return 0
    if args.detach:
        return detach(args)
    ensure_singleton()
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(f"{os.getpid()}\n", encoding="utf-8")
    state = State(args.mode)
    try:
        if args.postprocess_only:
            root = c.L1_ROOT / "smoke" if args.mode == "smoke" else c.L1_ROOT
            postprocess(args, state, root / "logs/train.log")
            state.update(state="COMPLETE")
            return 0
        wait_preflight(args, state)
        if args.preflight_only:
            state.update(state="PREFLIGHT_COMPLETE")
            return 0
        prepare_and_patch(state)
        if args.mode == "formal":
            smoke_manifest = c.L1_ROOT / "smoke/rollouts/canonical/llama_opd_training_manifest.json"
            payload = c.read_json(smoke_manifest, {})
            if payload.get("status") != "complete":
                raise RuntimeError(f"formal L1 is gated on a complete smoke: {smoke_manifest}")
        log = train(args, state)
        postprocess(args, state, log)
        state.update(state="COMPLETE")
        return 0
    except Exception as error:
        state.update(state="FAILED", failure=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
