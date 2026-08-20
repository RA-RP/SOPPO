#!/usr/bin/env python3
"""Detached, resumable supervisor for Cycle 09 block-2 Llama GPU work."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO = Path("/root/LLM-output-density")
PYTHON = Path("/root/miniconda3/envs/density/bin/python")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
BLOCK_ROOT = Path("/root/autodl-tmp/cycle09_block2")
ROOT = BLOCK_ROOT / "model2_llama"
LOG_ROOT = BLOCK_ROOT / "logs"
STATUS = BLOCK_ROOT / "llama_chain_status.json"
G4_MANIFEST = ROOT / "g4_preflight/formal/manifest.json"
G5_RAW = ROOT / "rollout/teacher_rollout_pass1.jsonl"
G5_MANIFEST = ROOT / "rollout/rollout_manifest.json"
ARMS = ("sft", "offkd", "seqkd")
EXPECTED_ROLLOUT_ROWS = 5000


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
    print(f"[llama-supervisor] {json.dumps(updates, sort_keys=True)}", flush=True)
    return current


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def g4_complete() -> bool:
    manifest = read_json(G4_MANIFEST)
    return manifest.get("status") == "complete" and int(manifest.get("n", -1)) == 100


def g5_complete() -> bool:
    manifest = read_json(G5_MANIFEST)
    return (
        int(manifest.get("n_prompts", -1)) == EXPECTED_ROLLOUT_ROWS
        and manifest.get("arm") == "model2_llama"
        and (ROOT / "rollout/teacher_rollout.jsonl").is_file()
        and (ROOT / "rollout/teacher_top32_logprob.npz").is_file()
    )


def arm_complete(arm: str) -> bool:
    manifest = read_json(ROOT / f"g6/{arm}/checkpoints/training_manifest.json")
    return (
        manifest.get("status") == "complete"
        and manifest.get("arm") == arm
        and int(manifest.get("completed_steps", -1)) == 624
    )


def run_command(
    name: str,
    command: tuple[str, ...],
    complete: Callable[[], bool],
    *,
    attempts: int = 1,
) -> None:
    if complete():
        atomic_status(state=f"{name}_cached", current_stage=name)
        return
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    log_path = LOG_ROOT / f"{name}.log"
    for attempt in range(1, attempts + 1):
        atomic_status(
            state=f"running_{name}",
            current_stage=name,
            stage_attempt=attempt,
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
        if result.returncode == 0 and complete():
            atomic_status(state=f"{name}_complete", current_stage=name)
            return
        atomic_status(
            state=f"{name}_attempt_failed",
            current_stage=name,
            stage_attempt=attempt,
            stage_returncode=result.returncode,
        )
    raise RuntimeError(f"{name} did not complete after {attempts} attempt(s); log={log_path}")


def run_g4() -> None:
    run_command(
        "G4_llama_preflight",
        (str(PYTHON), str(SCRIPTS / "cycle09_llama_g4_preflight.py")),
        g4_complete,
    )
    decision = read_json(G4_MANIFEST).get("decision")
    atomic_status(g4_decision=decision)
    if decision != "GO":
        atomic_status(status="stopped", state="stopped_at_G4_gate")
        raise SystemExit(0)


def run_g5() -> None:
    if g5_complete():
        atomic_status(state="G5_llama_rollout_cached", current_stage="G5")
        return
    if line_count(G5_RAW) != EXPECTED_ROLLOUT_ROWS:
        run_command(
            "G5_pass1",
            (
                str(PYTHON),
                str(SCRIPTS / "cycle09_llama_g5_rollout.py"),
                "--stage",
                "pass1",
            ),
            lambda: line_count(G5_RAW) == EXPECTED_ROLLOUT_ROWS,
        )
    else:
        atomic_status(state="G5_pass1_cached", current_stage="G5_pass1")
    run_command(
        "G5_pass2",
        (
            str(PYTHON),
            str(SCRIPTS / "cycle09_llama_g5_rollout.py"),
            "--stage",
            "pass2",
        ),
        g5_complete,
        attempts=2,
    )


def run_g6() -> None:
    for arm in ARMS:
        run_command(
            f"G6_{arm}",
            (
                str(PYTHON),
                str(SCRIPTS / "cycle09_llama_g6_train.py"),
                "--arm",
                arm,
            ),
            lambda arm=arm: arm_complete(arm),
            attempts=2,
        )


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
        run_g4()
        run_g5()
        run_g6()
    except SystemExit:
        raise
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
