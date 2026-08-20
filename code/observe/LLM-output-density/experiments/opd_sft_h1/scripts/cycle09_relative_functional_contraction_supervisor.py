#!/usr/bin/env python3
"""Detached two-GPU supervisor for Cycle 09 C0--C5 relative contraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
PYTHON = Path("/root/miniconda3/envs/density/bin/python")
SCRIPT = SCRIPTS / "cycle09_relative_functional_contraction.py"
ROOT = Path("/root/autodl-tmp/cycle09_relative_functional_contraction")
AUDIT = ROOT / "audit"
FINAL = ROOT / "final"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
STATE = ROOT / "supervisor_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def invoke(phase: str, device: str | None = None) -> None:
    command = [str(PYTHON), str(SCRIPT), "--phase", phase]
    if device:
        command += ["--device", device]
    subprocess.run(command, check=True)


def write_state(**changes: Any) -> None:
    state = read_json(STATE, {})
    state.update(changes)
    state.setdefault("schema_version", "cycle09_relative_functional_contraction_supervisor_v1")
    state.setdefault("created_utc", utc_now())
    atomic_json(STATE, state)


def write_handoff() -> dict[str, Any]:
    queue = read_json(AUDIT / "forward_queue.json", {})
    tasks = queue.get("tasks", [])
    queue_counts: dict[str, int] = {}
    for task in tasks:
        queue_counts[task.get("status", "unknown")] = queue_counts.get(task.get("status", "unknown"), 0) + 1
    coverage_path = FINAL / "relative_functional_contraction_coverage.csv"
    coverage = pd.read_csv(coverage_path) if coverage_path.is_file() else pd.DataFrame()
    coverage_counts = coverage["status"].value_counts().to_dict() if not coverage.empty else {}
    artifacts = {}
    for name in (
        "relative_functional_contraction_coverage.csv",
        "relative_functional_contraction_missing_registry.csv",
        "relative_functional_contraction_all_cells.csv",
        "relative_functional_contraction_module_audit.csv",
        "relative_functional_contraction_aggregation_sensitivity.csv",
        "relative_contraction_matched_cumulative_outputs.csv",
        "relative_contraction_matched_stepwise_outputs.csv",
        "relative_contraction_output_correlations.csv",
        "relative_contraction_within_checkpoint_correlations.csv",
        "relative_contraction_grouped_models.csv",
        "relative_contraction_leave_arm_domain_out.csv",
        "relative_contraction_raw_predictions.csv",
        "relative_contraction_stepwise_diagnostics.csv",
    ):
        path = FINAL / name
        if path.is_file():
            rows = len(pd.read_csv(path))
            artifacts[name] = {"path": str(path), "rows": rows, "bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema_version": "full_relative_functional_contraction_handoff_v1",
        "status": "complete_with_declared_coverage",
        "created_utc": utc_now(),
        "queue_status_counts": queue_counts,
        "coverage_status_counts": coverage_counts,
        "artifacts": artifacts,
        "no_new_behavior_eval": True,
        "no_training_backward_fisher_or_rankk_intervention": True,
    }
    MINI.mkdir(parents=True, exist_ok=True)
    manifest_path = MINI / "full_relative_functional_contraction_manifest.json"
    atomic_json(manifest_path, manifest)
    lines = [
        "# Full relative functional contraction: raw Theory handoff",
        "",
        "Status: `COMPLETE_WITH_DECLARED_COVERAGE`  ",
        f"Created: {manifest['created_utc']}  ",
        "Scope: C1--C5 coverage, raw values, missing cells, and provenance only. No Theory decision.",
        "",
        "## Queue Coverage",
        "",
        "| task status | count |",
        "| --- | ---: |",
    ]
    lines += [f"| {key} | {value} |" for key, value in sorted(queue_counts.items())]
    lines += ["", "## Registry Coverage", "", "| registry status | count |", "| --- | ---: |"]
    lines += [f"| {key} | {value} |" for key, value in sorted(coverage_counts.items())]
    lines += ["", "## Artifact Index", "", "| artifact | rows | bytes | SHA-256 |", "| --- | ---: | ---: | --- |"]
    lines += [f"| {name} | {info['rows']} | {info['bytes']} | `{info['sha256']}` |" for name, info in sorted(artifacts.items())]
    lines += [
        "",
        "## Transfer Boundary",
        "",
        "This handoff records only coverage, raw result files and declared missing/pending cells. It does not adjudicate the C6 decision tree, launch behavior evaluation, training, backward/Fisher, or rank-k interventions.",
        "",
        f"Machine-readable manifest: `{manifest_path}`.",
    ]
    handoff_path = MINI / "full_relative_functional_contraction_theory_handoff.md"
    handoff_path.write_text("\n".join(lines) + "\n")
    evolution = REPO / "mypaper/code/code_evolution.md"
    marker = "<!-- cycle09-relative-functional-contraction-handoff -->"
    existing = evolution.read_text()
    if marker not in existing:
        evolution.write_text(existing.rstrip() + f'''\n\n---\n\n{marker}\n\n## Cycle 09 full relative functional contraction raw transfer\n\nCompleted C1--C5 coverage/derivation/fixed-token forward/analysis pipeline. The handoff records final queue and registry states, including `PENDING_UPSTREAM` model cells and any output errors, without interpretation.\n\nRaw Theory handoff:\n`mini/full_relative_functional_contraction_theory_handoff.md`.\nMachine-readable manifest:\n`mini/full_relative_functional_contraction_manifest.json`.\n\n<!-- cycle09-relative-functional-contraction-handoff-end -->\n''')
    return {"handoff": str(handoff_path), "manifest": str(manifest_path), "queue_counts": queue_counts}


def formal() -> None:
    write_state(status="running_cpu_preparation", started_utc=utc_now(), no_auto_shutdown=True)
    invoke("audit")
    invoke("derive")
    invoke("plan")
    write_state(status="running_gpu_workers", gpu_workers=["cuda:0", "cuda:1"], gpu_started_utc=utc_now())
    workers = [
        subprocess.Popen([str(PYTHON), str(SCRIPT), "--phase", "worker", "--device", device])
        for device in ("cuda:0", "cuda:1")
    ]
    return_codes = [worker.wait() for worker in workers]
    if any(code != 0 for code in return_codes):
        raise RuntimeError(f"GPU worker process failed: {return_codes}")
    write_state(status="running_cpu_finalization", gpu_workers_returncodes=return_codes, gpu_finished_utc=utc_now())
    invoke("aggregate-outputs")
    invoke("audit")
    invoke("analyze")
    result = write_handoff()
    write_state(status="complete", completed_utc=utc_now(), **result)


def smoke(device: str) -> None:
    write_state(status="smoke_running", smoke_started_utc=utc_now(), no_auto_shutdown=True)
    invoke("smoke", device)
    write_state(status="smoke_complete", smoke_completed_utc=utc_now())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        if args.mode == "smoke":
            smoke(args.device)
        else:
            formal()
    except Exception as error:
        write_state(status="error", error=f"{type(error).__name__}: {error}", failed_utc=utc_now())
        raise


if __name__ == "__main__":
    main()
