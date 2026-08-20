#!/usr/bin/env python3
"""Detached controller: resume H5 only after all Stage-4 GPU work completed."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path("/root/autodl-tmp/cycle09_stage4_state_displacement")
FORMAL = ROOT / "supervisor_state.json"
ASSIST = ROOT / "llama_p1_gpu0_assist.json"
STATE = ROOT / "stage4_then_h5.json"
H5_RESUME = Path("/root/LLM-output-density/experiments/opd_sft_h1/scripts/run_cycle09_h5_postprocess_resume.sh")
POLL_SECONDS = 60


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(status: str, *, reason: str | None = None) -> None:
    payload = {
        "schema_version": "cycle09_stage4_then_h5_v1",
        "status": status,
        "reason": reason,
        "formal": read(FORMAL),
        "llama_p1_assist": read(ASSIST),
        "updated_unix": time.time(),
        "no_auto_shutdown": True,
    }
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=STATE.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, STATE)


def h5_active() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "cycle09_stage3_frozen_self_postprocess.py --phase worker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def main() -> None:
    while True:
        formal = read(FORMAL)
        assist = read(ASSIST)
        formal_status = formal.get("status")
        assist_status = assist.get("status")
        if formal.get("phase") == "formal" and formal_status == "failed":
            write("blocked", reason="Stage4 formal failed; H5 intentionally remains paused")
            return
        if assist_status == "failed":
            write("blocked", reason="GPU0 Llama P1 assistant failed; H5 intentionally remains paused")
            return
        if formal.get("phase") == "formal" and formal_status == "complete" and assist_status == "complete":
            if h5_active():
                write("complete", reason="H5 was already active; no duplicate resume")
                return
            write("launching_h5")
            result = subprocess.run([str(H5_RESUME)], check=False)
            if result.returncode == 0:
                write("complete", reason="H5 detached resume launched after Stage4 completion")
            else:
                write("blocked", reason=f"H5 resume launcher exited rc={result.returncode}")
            return
        write("waiting")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
