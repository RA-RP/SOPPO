"""Read-only Slurm pipeline status; never reads logs or sample-level artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


def command(args):
    result = subprocess.run(args, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    jobs = registry["jobs"]
    ids = [str(value) for value in jobs.values()]
    queue_text = command(["squeue", "-h", "-j", ",".join(ids), "-o", "%A|%K|%T|%M|%R"])
    live = defaultdict(list)
    for line in queue_text.splitlines():
        if line:
            job_id, task_id, state, elapsed, reason = line.split("|", 4)
            live[job_id].append((task_id, state, elapsed, reason))
    print(f"Experiment: {registry['experiment_id']}")
    print(f"Git commit: {registry['git_commit']}")
    print("stage | job_id | state | elapsed/reason")
    print("--- | --- | --- | ---")
    for stage, job_id in jobs.items():
        job_id = str(job_id)
        if job_id in live:
            entries = live[job_id]
            state = ",".join(
                f"{name}:{count}" for name, count in sorted(Counter(row[1] for row in entries).items())
            )
            detail = "; ".join(
                f"task={task_id} {elapsed} {reason}" for task_id, _, elapsed, reason in entries
            )
        else:
            accounting = command(
                ["sacct", "-n", "-X", "-j", job_id, "--format=State,Elapsed", "--parsable2"]
            )
            state_line = next((line for line in accounting.splitlines() if line.strip()), "UNKNOWN|-")
            state, detail = state_line.split("|", 1)
        print(f"{stage} | {job_id} | {state} | {detail}")


if __name__ == "__main__":
    main()
