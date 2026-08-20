#!/usr/bin/env python3
"""Retain only H5 landmark checkpoints plus one durable resume checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

LANDMARKS = frozenset((5, 20, 40, 80, 160, 320))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def latest_step(root: Path) -> int:
    try:
        return int((root / "checkpoints" / "latest_checkpointed_iteration.txt").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return 0


def checkpoint_steps(root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in (root / "checkpoints").glob("global_step_*"):
        if not path.is_dir():
            continue
        try:
            result[int(path.name.rsplit("_", 1)[1])] = path
        except ValueError:
            continue
    return result


def remove_obsolete(root: Path, dry_run: bool, allow_orphans: bool) -> dict:
    latest = latest_step(root)
    protected = set(LANDMARKS)
    if latest:
        protected.add(latest)
    deleted: list[int] = []
    deferred: list[int] = []
    for step, path in sorted(checkpoint_steps(root).items()):
        if step in protected:
            continue
        if step > latest and not allow_orphans:
            deferred.append(step)
            continue
        if dry_run:
            deleted.append(step)
            continue
        shutil.rmtree(path)
        deleted.append(step)
    payload = {
        "schema_version": 1,
        "task": "H5 checkpoint retention",
        "latest_durable_step": latest,
        "landmark_steps": sorted(LANDMARKS),
        "protected_steps": sorted(protected),
        "deleted_steps": deleted,
        "deferred_writing_or_orphan_steps": deferred,
        "dry_run": dry_run,
        "allow_orphans": allow_orphans,
        "updated_unix": time.time(),
    }
    if not dry_run:
        atomic_json(root / "H5_checkpoint_retention_status.json", payload)
    return payload


def parent_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.once:
        print(json.dumps(remove_obsolete(args.root, args.dry_run, allow_orphans=False), indent=2))
        return
    if not args.parent_pid:
        parser.error("--parent-pid is required unless --once is used")
    while parent_alive(args.parent_pid):
        payload = remove_obsolete(args.root, args.dry_run, allow_orphans=False)
        print(json.dumps(payload, sort_keys=True), flush=True)
        time.sleep(args.poll_seconds)
    payload = remove_obsolete(args.root, args.dry_run, allow_orphans=True)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
