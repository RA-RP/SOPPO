#!/usr/bin/env python3
"""Online checkpoint pruner for 624-step VERL runs.

verl saves the FULL model state dict per checkpoint (~16 GB fp32 for the 4B, NOT
just the small LoRA adapter). With save_freq=5 over 624 steps that is ~125×16 GB ≈
2 TB — far over disk. We only need the cycle07 evaluated grid steps
{5,10,20,40,80,160,320,480,624}. This watcher deletes every COMPLETED non-grid
`global_step_N/actor` as it appears, so only the ~9 grid checkpoints (~144 GB) ever
accumulate; run_cycle08.py converts those to merged HF afterwards.

The tracker file is VERL's commit marker: it is written only after actor and
dataloader state finish saving. During a save, preserve both the tracker step
and the highest directory being written. Once the tracker advances, the old
non-grid rolling checkpoint becomes reclaimable.

Run as a background process for the lifetime of training; it exits on SIGTERM.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

STEP_RE = re.compile(r"global_step_(\d+)$")
TRACKER = "latest_checkpointed_iteration.txt"


def _steps(ckpt_root: Path) -> list[int]:
    out = []
    for d in ckpt_root.glob("global_step_*"):
        m = STEP_RE.search(d.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _tracked_step(ckpt_root: Path) -> tuple[bool, int | None]:
    tracker = ckpt_root / TRACKER
    if not tracker.exists():
        return False, None
    try:
        value = tracker.read_text(encoding="utf-8").strip()
        return True, int(value)
    except (OSError, ValueError):
        return True, None


def _deletable_steps(ckpt_root: Path, grid: set[int]) -> tuple[list[int], dict]:
    steps = _steps(ckpt_root)
    if not steps:
        return [], {"highest": None, "tracked": None, "tracker_exists": False}
    tracker_exists, tracked = _tracked_step(ckpt_root)
    highest = steps[-1]
    state = {
        "highest": highest,
        "tracked": tracked,
        "tracker_exists": tracker_exists,
    }
    if tracker_exists and (tracked is None or tracked not in steps):
        state["guarded_reason"] = "tracker unreadable or target missing"
        return [], state
    protected = set(grid)
    protected.add(highest)
    if tracked is not None:
        protected.add(tracked)
    return [step for step in steps if step not in protected], state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", type=Path, required=True)
    ap.add_argument("--grid", required=True, help="comma-separated grid steps to KEEP")
    ap.add_argument("--poll", type=float, default=20.0)
    args = ap.parse_args()
    grid = {int(x) for x in args.grid.split(",") if x.strip()}
    print(f"[pruner] watching {args.ckpt_root}; keeping grid {sorted(grid)}", flush=True)

    while True:
        try:
            deletable, state = _deletable_steps(args.ckpt_root, grid)
            if state.get("guarded_reason"):
                print(
                    f"[pruner] guard: {state['guarded_reason']} "
                    f"(tracked={state['tracked']} highest={state['highest']})",
                    flush=True,
                )
            for n in deletable:
                d = args.ckpt_root / f"global_step_{n}"
                shutil.rmtree(d, ignore_errors=True)
                print(f"[pruner] removed non-grid {d.name} (reclaimed ~16GB)", flush=True)
        except Exception as e:  # noqa: BLE001 — a pruner must never crash the run
            print(f"[pruner] warn: {e}", flush=True)
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
