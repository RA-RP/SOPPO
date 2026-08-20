#!/usr/bin/env python3
"""Stop the AutoDL instance after the C0--C5 supervisor terminates or stalls."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path("/root/autodl-tmp/cycle09_relative_functional_contraction")
STATE = ROOT / "supervisor_state.json"
QUEUE = ROOT / "audit/forward_queue.json"
GUARD_STATE = ROOT / "shutdown_guard_state.json"
LOG = ROOT / "shutdown_guard.log"


def now() -> datetime:
    return datetime.now(timezone.utc)


def read_json(path: Path, default: dict | None = None) -> dict:
    return json.loads(path.read_text()) if path.is_file() else (default or {})


def write_state(payload: dict) -> None:
    temporary = GUARD_STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, GUARD_STATE)


def log(message: str) -> None:
    line = f"{now().isoformat()} {message}\n"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def queue_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in read_json(QUEUE, {}).get("tasks", []):
        status = task.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def poweroff(reason: str, dry_run: bool) -> None:
    log(f"shutdown_trigger reason={reason} dry_run={dry_run}")
    write_state({"status": "triggered", "reason": reason, "triggered_utc": now().isoformat(), "dry_run": dry_run})
    if dry_run:
        return
    shutdown = next(
        (candidate for candidate in ("/sbin/shutdown", "/usr/sbin/shutdown", "/usr/bin/shutdown", "/bin/shutdown")
         if Path(candidate).is_file()),
        shutil.which("shutdown") or "shutdown",
    )
    subprocess.run([shutdown, "-h", "now"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supervisor-pid", required=True, type=int)
    parser.add_argument("--deadline-hours", type=float, default=10.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stalled-minutes", type=float, default=45.0)
    parser.add_argument("--supervisor-grace-minutes", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    started = now()
    deadline = started + timedelta(hours=args.deadline_hours)
    last_progress = started
    last_complete = -1
    log(f"armed supervisor_pid={args.supervisor_pid} deadline={deadline.isoformat()} stalled_minutes={args.stalled_minutes}")
    while True:
        supervisor = read_json(STATE, {})
        queue = queue_counts()
        completed = queue.get("complete", 0) + queue.get("error", 0)
        if completed > last_complete:
            last_complete = completed
            last_progress = now()
        snapshot = {
            "status": "armed", "supervisor_pid": args.supervisor_pid, "supervisor_status": supervisor.get("status"),
            "queue_counts": queue, "started_utc": started.isoformat(), "deadline_utc": deadline.isoformat(),
            "last_progress_utc": last_progress.isoformat(), "updated_utc": now().isoformat(), "dry_run": args.dry_run,
        }
        write_state(snapshot)
        if supervisor.get("status") in {"complete", "error"}:
            poweroff(f"supervisor_{supervisor.get('status')}", args.dry_run)
            return
        if queue.get("error", 0) > 0:
            poweroff("forward_queue_error", args.dry_run)
            return
        if now() >= deadline:
            poweroff("deadline", args.dry_run)
            return
        if not pid_alive(args.supervisor_pid) and now() - started >= timedelta(minutes=args.supervisor_grace_minutes):
            poweroff("supervisor_pid_missing", args.dry_run)
            return
        if now() - last_progress >= timedelta(minutes=args.stalled_minutes):
            poweroff("queue_stalled", args.dry_run)
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
