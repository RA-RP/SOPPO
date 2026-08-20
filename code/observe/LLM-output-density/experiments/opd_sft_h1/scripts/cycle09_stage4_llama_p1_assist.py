#!/usr/bin/env python3
"""GPU0 work-stealing helper: run Llama P1 while GPU1 finishes Llama A1/A3."""
from __future__ import annotations

import json
import traceback

import cycle09_block3_common as b3
import cycle09_stage4_state_displacement as s4
import cycle09_stage4_supervisor as supervisor

STATE = s4.ROOT / "llama_p1_gpu0_assist.json"


def write(status: str, error: str | None = None) -> None:
    s4.atomic_json(STATE, {
        "schema_version": "cycle09_stage4_llama_p1_assist_v1",
        "status": status,
        "model": "llama",
        "device": "cuda:0",
        "scope": "formal_p1_lane: retained profiles, A5 local output, A6 zeroing",
        "error": error,
        "created_utc": b3.utc_now(),
    })


def main() -> None:
    write("running")
    try:
        completed = supervisor.execute_serial(supervisor.formal_p1_lane("llama", "cuda:0"))
    except Exception as error:
        write("failed", f"{type(error).__name__}: {error}")
        raise
    s4.atomic_json(STATE, {
        "schema_version": "cycle09_stage4_llama_p1_assist_v1",
        "status": "complete",
        "model": "llama",
        "device": "cuda:0",
        "scope": "formal_p1_lane: retained profiles, A5 local output, A6 zeroing",
        "completed": completed,
        "created_utc": b3.utc_now(),
    })
    print(json.dumps({"status": "complete", "cells": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
