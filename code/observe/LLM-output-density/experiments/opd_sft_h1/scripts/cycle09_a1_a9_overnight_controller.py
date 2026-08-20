#!/usr/bin/env python3
"""Detached A1--A9 controller for the current 2x32G machine.

The controller is intentionally limited to A1--A9.  A10 is AUTO-GO in the
handoff only after a 2x96G GPU preflight, so this file records a hardware HOLD
on 32G cards and never invokes the seed43 training runner.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_block3_common as b3  # noqa: E402


PY = b3.DENSITY_PYTHON if b3.DENSITY_PYTHON.is_file() else Path(sys.executable)
D10 = SCRIPTS / "cycle09_d10_llama_numeric_parity.py"
CONTRACTION = SCRIPTS / "cycle09_contraction_completion.py"
D5D7 = SCRIPTS / "cycle09_d5_d7_tables.py"
TPK = SCRIPTS / "cycle09_stage3_tpk.py"
ROOT = b3.AUTODL / "cycle09_relative_functional_contraction/d10_llama_numeric_parity"
LOGDIR = ROOT / "logs"
STATE = ROOT / "overnight_controller_state.json"
LAUNCH_LOG = LOGDIR / "a1_a9_overnight.log"

ARMS_GPU0 = ("opd", "sft")
ARMS_GPU1 = ("offkd", "seqkd")
STEPS = (5, 20, 40, 80, 160, 320)
CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")


def now() -> str:
    return b3.utc_now()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_state(phase: str, status: str, detail: dict[str, Any] | None = None) -> None:
    payload = {
        "schema_version": "cycle09_a1_a9_overnight_controller_v1",
        "phase": phase,
        "status": status,
        "updated_utc": now(),
        "a10": {
            "status": "HOLD_NOT_STARTED_BY_A1_A9_CONTROLLER",
            "reason": "current controller is limited to A1-A9; A10 requires 2x96G protocol preflight",
        },
    }
    if detail:
        payload.update(detail)
    atomic_json(STATE, payload)


def run(command: list[str], *, log_name: str | None = None) -> dict[str, Any]:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log_path = LOGDIR / (log_name or (Path(command[1]).stem + ".log"))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now()}] CMD {' '.join(command)}\n")
        handle.flush()
        result = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, text=True)
        handle.write(f"[{now()}] RC {result.returncode} elapsed_s={time.time() - started:.1f}\n")
    if result.returncode != 0:
        raise RuntimeError(f"command failed rc={result.returncode}; log={log_path}; cmd={' '.join(command)}")
    return {"cmd": command, "log": str(log_path), "elapsed_s": time.time() - started}


def d10(*args: str) -> list[str]:
    return [str(PY), str(D10), *args]


def optional(command: list[str], name: str, required: list[Path] | None = None) -> dict[str, Any]:
    missing = [str(path) for path in (required or []) if not path.is_file()]
    if missing:
        return {"name": name, "status": "SKIPPED_MISSING_UPSTREAM", "missing": missing, "created_utc": now()}
    try:
        return {"name": name, "status": "complete", "result": run(command, log_name=f"{name}.log"), "created_utc": now()}
    except Exception as error:
        return {"name": name, "status": "FAILED_NONBLOCKING", "error": f"{type(error).__name__}: {error}", "created_utc": now()}


def preflight(strict: bool) -> dict[str, Any]:
    command = d10("--phase", "preflight", "--min-free-gb", "50")
    if strict:
        command.append("--strict")
    run(command, log_name="preflight.log")
    payload = read_json(ROOT / "preflight.json", {})
    write_state("preflight", payload.get("status", "unknown"), {"preflight": payload})
    return payload


def wait_for_data(poll_seconds: int, max_wait_hours: float, strict_after_wait: bool) -> dict[str, Any]:
    started = time.time()
    attempt = 0
    while True:
        attempt += 1
        payload = preflight(strict=False)
        if payload.get("complete"):
            if strict_after_wait:
                preflight(strict=True)
            return payload
        elapsed = time.time() - started
        if elapsed > max_wait_hours * 3600:
            missing = [item for item in payload.get("checks", []) if not item.get("complete")]
            raise TimeoutError(f"data preflight still incomplete after {max_wait_hours} h; first missing={missing[:5]}")
        missing_count = sum(1 for item in payload.get("checks", []) if not item.get("complete"))
        write_state("wait_for_data", "waiting", {
            "attempt": attempt,
            "elapsed_s": elapsed,
            "poll_seconds": poll_seconds,
            "missing_count": missing_count,
            "last_preflight": str(ROOT / "preflight.json"),
        })
        time.sleep(poll_seconds)


def smoke() -> dict[str, Any]:
    write_state("smoke", "running")
    result = run(
        d10("--phase", "smoke", "--device", "cuda:0", "--smoke-samples", "1", "--max-batch-tokens", "2048"),
        log_name="smoke.log",
    )
    payload = read_json(ROOT / "smoke_result.json", {})
    write_state("smoke", "complete", {"smoke": payload, "run": result})
    return payload


def run_arm_lane(device: str, arms: tuple[str, ...], poll_seconds: int) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for arm in arms:
        for step in STEPS:
            started = time.time()
            result = run(
                d10(
                    "--phase", "checkpoint",
                    "--tag", "formal",
                    "--arm", arm,
                    "--step", str(step),
                    "--probes", "all",
                    "--device", device,
                    "--forward-batch-size", "1",
                    "--max-batch-tokens", "4096",
                ),
                log_name=f"a1_{device.replace(':', '')}_{arm}_{step}.log",
            )
            completed.append({"arm": arm, "step": step, "device": device, "result": result})
            if time.time() - started > poll_seconds:
                write_state("formal", "running", {"last_completed": completed[-1], "completed_count": len(completed)})
    return completed


def run_base() -> dict[str, Any]:
    return run(
        d10(
            "--phase", "checkpoint",
            "--tag", "formal",
            "--arm", "base",
            "--step", "0",
            "--probes", "all",
            "--device", "cuda:0",
            "--forward-batch-size", "1",
            "--max-batch-tokens", "4096",
        ),
        log_name="a1_base.log",
    )


def run_a3_tpk() -> list[dict[str, Any]]:
    """Best-effort A3 T-PK lane using the existing strict-pk helper."""
    commands = [
        [
            str(PY), str(TPK), "--phase", "run", "--family", "llama3_2_3b",
            "--arms", "opd,sft,offkd,seqkd", "--steps", ",".join(map(str, STEPS)),
            "--layers", "14", "--modules", "all", "--ks", "4,8,16,32",
            "--delta-mode", "adapter_ba", "--device", "cuda:0",
        ],
        [
            str(PY), str(TPK), "--phase", "run", "--family", "qwen3_4b",
            "--arms", "opd,sft,offkd,seqkd", "--steps", "5,20,40,80,160,320,480,624",
            "--layers", "18", "--modules", "all", "--ks", "4,8,16,32",
            "--delta-mode", "bf16_merged_minus_base", "--device", "cuda:1",
        ],
    ]
    outputs = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(optional, command, f"a3_tpk_{idx}") for idx, command in enumerate(commands)]
        for future in as_completed(futures):
            outputs.append(future.result())
    atomic_json(ROOT / "a3_tpk_results.json", {"status": "complete", "results": outputs, "created_utc": now()})
    return outputs


def cpu_postprocess() -> dict[str, Any]:
    final = b3.AUTODL / "cycle09_relative_functional_contraction/final"
    results = [
        optional(
            [str(PY), str(CONTRACTION), "--phase", "cpu-all"],
            "a2_contraction_completion_cpu_all",
            [
                final / "relative_contraction_matched_cumulative_outputs.csv",
                final / "qwen_d4_merged_state_all_cells.csv",
                final / "qwen_d4_merged_state_outputs.csv",
            ],
        ),
        optional(
            [str(PY), str(D5D7), "--phase", "formal"],
            "a4_d5_d7_tables",
            [
                final / "relative_functional_contraction_module_audit.csv",
                final / "relative_contraction_matched_cumulative_outputs.csv",
                final / "relative_contraction_matched_stepwise_outputs.csv",
                final / "qwen_d4_merged_state_module_audit.csv",
                final / "qwen_d4_merged_state_all_cells.csv",
                final / "qwen_d4_merged_state_outputs.csv",
            ],
        ),
    ]
    payload = {"schema_version": "cycle09_a1_a9_cpu_postprocess_v1", "status": "complete", "results": results, "created_utc": now()}
    atomic_json(ROOT / "cpu_postprocess_results.json", payload)
    return payload


def formal(poll_seconds: int) -> dict[str, Any]:
    write_state("formal", "running")
    base = run_base()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_arm_lane, "cuda:0", ARMS_GPU0, poll_seconds),
            pool.submit(run_arm_lane, "cuda:1", ARMS_GPU1, poll_seconds),
        ]
        completed = []
        for future in as_completed(futures):
            completed.append(future.result())
            write_state("formal", "running", {"completed_lanes": len(completed), "base": base})
    a3 = run_a3_tpk()
    final = run(d10("--phase", "finalize", "--tag", "formal"), log_name="finalize.log")
    cpu = cpu_postprocess()
    payload = {
        "schema_version": "cycle09_a1_a9_formal_v1",
        "status": "complete",
        "base": base,
        "lanes": completed,
        "a3_tpk": a3,
        "finalize": final,
        "cpu_postprocess": cpu,
        "created_utc": now(),
    }
    atomic_json(ROOT / "a1_a9_formal_result.json", payload)
    write_state("formal", "complete", payload)
    return payload


def status() -> dict[str, Any]:
    payload = read_json(STATE, {"status": "not_started"})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def launch(args: argparse.Namespace) -> dict[str, Any]:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    command = [
        "nohup",
        str(PY),
        str(Path(__file__).resolve()),
        "--phase",
        "pipeline",
        "--poll-seconds",
        str(args.poll_seconds),
        "--max-wait-hours",
        str(args.max_wait_hours),
    ]
    if args.wait_for_data:
        command.append("--wait-for-data")
    if args.smoke_only:
        command.append("--smoke-only")
    with LAUNCH_LOG.open("ab") as out:
        process = subprocess.Popen(command, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    payload = {
        "schema_version": "cycle09_a1_a9_launch_v1",
        "status": "launched",
        "pid": process.pid,
        "command": command,
        "log": str(LAUNCH_LOG),
        "state": str(STATE),
        "created_utc": now(),
    }
    atomic_json(ROOT / "launch.json", payload)
    write_state("launch", "launched", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def pipeline(args: argparse.Namespace) -> dict[str, Any]:
    try:
        if args.wait_for_data:
            wait_for_data(args.poll_seconds, args.max_wait_hours, strict_after_wait=True)
        else:
            preflight(strict=True)
        smoke()
        if args.smoke_only:
            write_state("pipeline", "complete_smoke_only")
            return {"status": "complete_smoke_only"}
        result = formal(args.poll_seconds)
        write_state("pipeline", "complete", result)
        return result
    except Exception as error:
        payload = {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "created_utc": now(),
        }
        atomic_json(ROOT / "pipeline_error.json", payload)
        write_state("pipeline", "failed", payload)
        raise


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "wait", "smoke", "formal", "pipeline", "launch", "status"))
    parser.add_argument("--wait-for-data", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--max-wait-hours", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.phase == "preflight":
        value = preflight(strict=False)
    elif args.phase == "wait":
        value = wait_for_data(args.poll_seconds, args.max_wait_hours, strict_after_wait=False)
    elif args.phase == "smoke":
        value = smoke()
    elif args.phase == "formal":
        value = formal(args.poll_seconds)
    elif args.phase == "pipeline":
        value = pipeline(args)
    elif args.phase == "launch":
        value = launch(args)
    else:
        value = status()
    if args.phase != "status":
        print(json.dumps({"status": value.get("status"), "phase": args.phase, "created_utc": now()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
