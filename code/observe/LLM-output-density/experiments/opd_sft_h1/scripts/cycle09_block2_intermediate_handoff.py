#!/usr/bin/env python3
"""Build the immutable Cycle 09 block-2 intermediate Theory handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path("/root/LLM-output-density")
MINI = (
    ROOT
    / "mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
CODE_EVOLUTION = ROOT / "mypaper/code/code_evolution.md"
SPEC = ROOT / "mypaper/theory/stage_plan_handoff.md"

HANDOFF = MINI / "mini_block2_intermediate_theory_handoff.md"
MANIFEST = MINI / "block2_intermediate_manifest.json"
G2_BEHAVIOR_SNAPSHOT = MINI / "block2_intermediate_g2_behavior.csv"
G2_EXTRACT_SNAPSHOT = MINI / "block2_intermediate_g2_mmlupro_extract.csv"
G2_FLEXIBLE_SNAPSHOT = MINI / "block2_intermediate_g2_mmlupro_flexible.csv"
G2_IFEVAL_SNAPSHOT = MINI / "block2_intermediate_g2_ifeval_breakdown.csv"

TRAJECTORY = MINI / "three_arm_full_trajectory.csv"
MMLU_EXTRACT = MINI / "S1_mmlupro_extract_audit.csv"
MMLU_FLEXIBLE = MINI / "S1_mmlupro_flexible.csv"
IFEVAL_BREAKDOWN = MINI / "S1_ifeval_breakdown.csv"
C2_CSV = MINI / "C2_dose_response.csv"

G1_MANIFEST = Path("/root/autodl-tmp/cycle09_seqkd/checkpoints/training_manifest.json")
G4_MANIFEST = Path(
    "/root/autodl-tmp/cycle09_block2/model2_llama/g4_preflight/formal/manifest.json"
)
G5_DIR = Path("/root/autodl-tmp/cycle09_block2/model2_llama/rollout")
G5_MANIFEST = G5_DIR / "rollout_manifest.json"
G6_ROOT = Path("/root/autodl-tmp/cycle09_block2/model2_llama/g6")
G7_MANIFEST = MINI / "offkd_h_geometry_manifest.json"
C1_MANIFEST = MINI / "C1_direction_all_probes_manifest.json"
C2_MANIFEST = MINI / "C2_dose_response_manifest.json"

CRITICAL_STEPS = [0, 5, 20, 40, 624]
FULL_GRID = [0, 5, 10, 20, 40, 80, 160, 320, 480, 624]
PENDING_G2_STEPS = [80, 160, 320, 480]
MARKER_START = "<!-- cycle09-block2-intermediate-start -->"
MARKER_END = "<!-- cycle09-block2-intermediate-end -->"

BEHAVIOR_COLUMNS = [
    "arm",
    "step",
    "checkpoint_source_type",
    "math500_n",
    "math500_cap",
    "math500_acc",
    "math500_trunc_rate",
    "math500_mean_response_len",
    "mmlu_pro_n",
    "mmlu_pro_exact_match",
    "mmlu_pro_flexible",
    "strict_extract_fail_rate",
    "n_extract_fail",
    "ifeval_n",
    "ifeval_prompt_strict",
    "ifeval_instruction_strict",
    "ifeval_prompt_loose",
    "ifeval_instruction_loose",
    "gpqa_diamond_n",
    "gpqa_diamond_acc",
    "truthfulqa_mc1_n",
    "truthfulqa_mc1_acc",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def read_csv_stable(path: Path, attempts: int = 5) -> tuple[list[str], list[dict[str, str]]]:
    """Read a live CSV only when its stat is unchanged across the read."""
    for attempt in range(attempts):
        before = path.stat()
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            return fieldnames, rows
        if attempt + 1 < attempts:
            time.sleep(1)
    raise RuntimeError(f"CSV changed during every read attempt: {path}")


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def write_csv_atomic(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def step_number(row: Mapping[str, str]) -> int:
    return int(float(row["step"]))


def select_unique_steps(
    rows: Sequence[dict[str, str]], arm: str, steps: Sequence[int], source: Path
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for step in steps:
        matches = [
            row for row in rows if row.get("arm") == arm and step_number(row) == step
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one {arm} row at step {step} in {source}; found {len(matches)}"
            )
        selected.append(matches[0])
    return selected


def row_index(rows: Sequence[dict[str, str]]) -> dict[tuple[str, int], dict[str, str]]:
    output: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        key = (row["arm"], step_number(row))
        if key in output:
            raise RuntimeError(f"Duplicate row for {key}")
        output[key] = row
    return output


def markdown_table(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "|" + "|".join("---" for _ in fieldnames) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(name, "")) for name in fieldnames) + " |")
    return "\n".join(lines)


def require_complete_training(path: Path, expected_arm: str | None = None) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"Training manifest is not complete: {path}")
    if int(manifest.get("completed_steps", -1)) != 624:
        raise RuntimeError(f"Training did not reach step 624: {path}")
    if [int(x) for x in manifest.get("checkpoint_grid", [])] != FULL_GRID:
        raise RuntimeError(f"Unexpected checkpoint grid: {path}")
    if expected_arm is not None and manifest.get("arm") != expected_arm:
        raise RuntimeError(f"Expected arm {expected_arm}: {path}")
    checkpoint_root = path.parent
    for step in FULL_GRID[1:]:
        checkpoint = checkpoint_root / f"checkpoint-{step:06d}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
    return manifest


def file_record(path: Path, hash_file: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
    if hash_file:
        record["sha256"] = sha256(path)
        record["hash_policy"] = "sha256_current_file"
    else:
        record["hash_policy"] = "not_rehashed_large_raw"
    return record


def replace_or_append_marked_block(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if MARKER_START in text or MARKER_END in text:
        if text.count(MARKER_START) != 1 or text.count(MARKER_END) != 1:
            raise RuntimeError(f"Malformed block-2 intermediate markers in {path}")
        start = text.index(MARKER_START)
        end = text.index(MARKER_END, start) + len(MARKER_END)
        updated = text[:start].rstrip() + "\n\n" + block.rstrip() + "\n" + text[end:].lstrip("\n")
        write_text_atomic(path, updated)
        return
    with path.open("a", encoding="utf-8") as handle:
        if text and not text.endswith("\n"):
            handle.write("\n")
        handle.write("\n" + block.rstrip() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    generated_at = utc_now()

    trajectory_fields, trajectory_rows = read_csv_stable(TRAJECTORY)
    extract_fields, extract_rows = read_csv_stable(MMLU_EXTRACT)
    flexible_fields, flexible_rows = read_csv_stable(MMLU_FLEXIBLE)
    ifeval_fields, ifeval_rows = read_csv_stable(IFEVAL_BREAKDOWN)
    c2_fields, c2_rows = read_csv_stable(C2_CSV)

    critical_trajectory = select_unique_steps(
        trajectory_rows, "seqkd", CRITICAL_STEPS, TRAJECTORY
    )
    critical_extract = select_unique_steps(
        extract_rows, "seqkd", CRITICAL_STEPS, MMLU_EXTRACT
    )
    critical_flexible = select_unique_steps(
        flexible_rows, "seqkd", CRITICAL_STEPS, MMLU_FLEXIBLE
    )
    critical_ifeval = [
        row
        for row in ifeval_rows
        if row.get("arm") == "seqkd" and step_number(row) in CRITICAL_STEPS
    ]
    critical_ifeval.sort(
        key=lambda row: (CRITICAL_STEPS.index(step_number(row)), row["instruction_category"])
    )
    if len(critical_ifeval) != 45:
        raise RuntimeError(f"Expected 45 critical IFEval rows; found {len(critical_ifeval)}")
    for step in CRITICAL_STEPS:
        count = sum(step_number(row) == step for row in critical_ifeval)
        if count != 9:
            raise RuntimeError(f"Expected 9 IFEval categories at step {step}; found {count}")

    extract_by_key = row_index(critical_extract)
    flexible_by_key = row_index(critical_flexible)
    behavior_rows: list[dict[str, str]] = []
    for trajectory_row in critical_trajectory:
        key = ("seqkd", step_number(trajectory_row))
        extract_row = extract_by_key[key]
        flexible_row = flexible_by_key[key]
        merged = dict(trajectory_row)
        merged["mmlu_pro_flexible"] = flexible_row["mmlu_pro_flexible"]
        merged["strict_extract_fail_rate"] = flexible_row["strict_extract_fail_rate"]
        merged["n_extract_fail"] = extract_row["n_extract_fail"]
        behavior_rows.append({name: merged.get(name, "") for name in BEHAVIOR_COLUMNS})

    write_csv_atomic(G2_BEHAVIOR_SNAPSHOT, BEHAVIOR_COLUMNS, behavior_rows)
    write_csv_atomic(G2_EXTRACT_SNAPSHOT, extract_fields, critical_extract)
    write_csv_atomic(G2_FLEXIBLE_SNAPSHOT, flexible_fields, critical_flexible)
    write_csv_atomic(G2_IFEVAL_SNAPSHOT, ifeval_fields, critical_ifeval)

    g1 = require_complete_training(G1_MANIFEST)
    g4 = load_json(G4_MANIFEST)
    if g4.get("status") != "complete" or g4.get("decision") != "GO":
        raise RuntimeError("G4 is not complete with decision GO")
    for artifact_path in g4.get("artifacts", {}).values():
        if not Path(artifact_path).is_file():
            raise FileNotFoundError(artifact_path)

    g5 = load_json(G5_MANIFEST)
    if int(g5.get("n_prompts", -1)) != 5000:
        raise RuntimeError("G5 rollout manifest does not contain 5000 prompts")
    g5_raw_paths = [
        G5_DIR / "teacher_rollout_pass1.jsonl",
        G5_DIR / "pass2_stream/top32_ids.npy",
        G5_DIR / "pass2_stream/top32_logprob.npy",
        G5_DIR / "pass2_stream/row_offsets.npy",
        G5_DIR / "pass2_stream/progress.json",
        G5_DIR / "teacher_rollout.jsonl",
        G5_DIR / "teacher_top32_logprob.npz",
    ]
    for path in g5_raw_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    g6: dict[str, dict[str, Any]] = {}
    for arm in ("sft", "offkd", "seqkd"):
        g6[arm] = require_complete_training(
            G6_ROOT / arm / "checkpoints/training_manifest.json", expected_arm=arm
        )

    g7 = load_json(G7_MANIFEST)
    if g7.get("status") != "complete":
        raise RuntimeError("G7 manifest is not complete")
    for info in g7.get("files", {}).values():
        if not Path(info["path"]).is_file():
            raise FileNotFoundError(info["path"])

    c1 = load_json(C1_MANIFEST)
    if c1.get("status") != "complete":
        raise RuntimeError("C1 manifest is not complete")
    for info in c1.get("outputs", {}).values():
        if not Path(info["path"]).is_file():
            raise FileNotFoundError(info["path"])

    c2 = load_json(C2_MANIFEST)
    if c2.get("status") != "complete" or int(c2.get("rows", -1)) != len(c2_rows):
        raise RuntimeError("C2 manifest and CSV row count do not agree")

    snapshot_records = {
        path.name: file_record(path)
        for path in (
            G2_BEHAVIOR_SNAPSHOT,
            G2_EXTRACT_SNAPSHOT,
            G2_FLEXIBLE_SNAPSHOT,
            G2_IFEVAL_SNAPSHOT,
        )
    }
    compact_sources = {
        str(path): file_record(path)
        for path in (
            G1_MANIFEST,
            G4_MANIFEST,
            G5_MANIFEST,
            G6_ROOT / "sft/checkpoints/training_manifest.json",
            G6_ROOT / "offkd/checkpoints/training_manifest.json",
            G6_ROOT / "seqkd/checkpoints/training_manifest.json",
            G7_MANIFEST,
            C1_MANIFEST,
            C2_MANIFEST,
            C2_CSV,
        )
    }
    g5_large_files = [file_record(path, hash_file=False) for path in g5_raw_paths]

    completion_rows = [
        {"item": "G1", "status": "complete", "raw_scope": "seqKD training through step 624; ten checkpoints"},
        {"item": "G2 critical", "status": "complete_intermediate", "raw_scope": "seqKD behavior and audits at 0,5,20,40,624"},
        {"item": "G4", "status": "complete", "raw_scope": "Llama-3.2-3B preflight; decision GO"},
        {"item": "G5", "status": "complete", "raw_scope": "5000-prompt two-pass rollout; raw top-32"},
        {"item": "G6 SFT", "status": "complete", "raw_scope": "Llama training through step 624; ten checkpoints"},
        {"item": "G6 off-KD", "status": "complete", "raw_scope": "Llama training through step 624; ten checkpoints"},
        {"item": "G6 seqKD", "status": "complete", "raw_scope": "Llama training through step 624; ten checkpoints"},
        {"item": "G7", "status": "complete", "raw_scope": "off-KD H_bos/H_ood seven-step geometry"},
        {"item": "C1", "status": "complete", "raw_scope": "direction analysis extended to all static probes"},
        {"item": "C2", "status": "complete", "raw_scope": "dose-response table; 9 rows"},
    ]

    g1_fields = [
        "status", "completed_steps", "eligible_prompts", "batch_size", "epochs",
        "steps_per_epoch", "training_seconds_this_invocation", "checkpoint_grid",
        "resume_from", "student_model", "rollout_dir",
    ]
    g1_row = {key: g1.get(key, "") for key in g1_fields}

    g4_fields = list(g4["summary"][0].keys())
    g4_rows = [dict(row) for row in g4["summary"]]

    g5_fields = [
        "n_prompts", "truncation_rate", "n_truncated", "truncated_kept",
        "has_boxed_rate", "length_mean", "length_median", "length_p90",
        "length_max", "pass2_minutes", "topk", "raw_convention",
    ]
    g5_row = {
        "n_prompts": g5["n_prompts"],
        "truncation_rate": g5["truncation_rate"],
        "n_truncated": g5["n_truncated"],
        "truncated_kept": g5["truncated_kept"],
        "has_boxed_rate": g5["has_boxed_rate"],
        "length_mean": g5["length_stats"]["mean"],
        "length_median": g5["length_stats"]["median"],
        "length_p90": g5["length_stats"]["p90"],
        "length_max": g5["length_stats"]["max"],
        "pass2_minutes": g5["timing_minutes"]["pass2_logprobs"],
        "topk": g5["logprob_pass2"]["topk"],
        "raw_convention": g5["logprob_pass2"]["convention"],
    }

    g6_fields = [
        "arm", "status", "completed_steps", "eligible_prompts", "batch_size",
        "epochs", "steps_per_epoch", "training_seconds_this_invocation",
        "loss_name", "checkpoint_grid", "student_model", "data_source",
    ]
    g6_rows = []
    for arm in ("sft", "offkd", "seqkd"):
        source = g6[arm]
        g6_rows.append(
            {
                "arm": arm,
                "status": source["status"],
                "completed_steps": source["completed_steps"],
                "eligible_prompts": source["eligible_prompts"],
                "batch_size": source["batch_size"],
                "epochs": source["epochs"],
                "steps_per_epoch": source["steps_per_epoch"],
                "training_seconds_this_invocation": source["training_seconds_this_invocation"],
                "loss_name": source["loss"]["name"],
                "checkpoint_grid": source["checkpoint_grid"],
                "student_model": source["student_model"],
                "data_source": source["data_source"],
            }
        )

    g7_rows = [
        {"field": "status", "value": g7["status"]},
        {"field": "steps", "value": g7["steps"]},
        {"field": "domains", "value": g7["domains"]},
        {"field": "generation_seeds", "value": g7["generation_seeds"]},
        {"field": "n_generated_per_cell", "value": g7["n_generated_per_cell"]},
        {"field": "dW_track", "value": g7["dW_track"]},
        {"field": "theta_numerics", "value": g7["theta_numerics"]},
        {"field": "spectra_rows", "value": g7["rows"]["spectra"]},
        {"field": "m1_rows", "value": g7["rows"]["m1"]},
        {"field": "m2_rows", "value": g7["rows"]["m2"]},
        {"field": "theta_rows", "value": g7["rows"]["theta"]},
    ]

    c1_rows = []
    for name in ("analysis", "principal", "ranks", "overlap"):
        info = c1["outputs"][name]
        c1_rows.append(
            {"output": name, "path": info["path"], "rows": info["rows"], "sha256": info["sha256"]}
        )

    g5_inventory_rows = [
        {"path": item["path"], "bytes": item["bytes"], "hash_policy": item["hash_policy"]}
        for item in g5_large_files
    ]

    source_inventory_rows = []
    for path_text, info in compact_sources.items():
        source_inventory_rows.append(
            {"path": path_text, "bytes": info["bytes"], "sha256": info["sha256"]}
        )
    for info in snapshot_records.values():
        source_inventory_rows.append(
            {"path": info["path"], "bytes": info["bytes"], "sha256": info["sha256"]}
        )

    handoff_parts = [
        "# Cycle 09 Block 2 Intermediate Theory Handoff",
        "",
        f"Generated UTC: `{generated_at}`",
        "",
        f"Source specification: `{SPEC}` (`Second execution block (2026-07-18)`).",
        "",
        "Reporting guard: raw readings and provenance only; no interpretation or adjudication.",
        "",
        "## Completion Register",
        "",
        markdown_table(["item", "status", "raw_scope"], completion_rows),
        "",
        "## G2 Critical-Step Behavior Snapshot",
        "",
        markdown_table(BEHAVIOR_COLUMNS, behavior_rows),
        "",
        "Snapshot: `" + str(G2_BEHAVIOR_SNAPSHOT) + "`.",
        "",
        "## G2 MMLU-Pro Strict Extraction Snapshot",
        "",
        markdown_table(extract_fields, critical_extract),
        "",
        "Snapshot: `" + str(G2_EXTRACT_SNAPSHOT) + "`.",
        "",
        "## G2 MMLU-Pro Flexible Extraction Snapshot",
        "",
        markdown_table(flexible_fields, critical_flexible),
        "",
        "Snapshot: `" + str(G2_FLEXIBLE_SNAPSHOT) + "`.",
        "",
        "## G2 IFEval Native-Category Snapshot",
        "",
        markdown_table(ifeval_fields, critical_ifeval),
        "",
        "Snapshot: `" + str(G2_IFEVAL_SNAPSHOT) + "`.",
        "",
        "## G1 SeqKD Training Provenance",
        "",
        markdown_table(g1_fields, [g1_row]),
        "",
        f"Manifest: `{G1_MANIFEST}`.",
        "",
        "## G4 Llama Preflight Raw Summary",
        "",
        markdown_table(g4_fields, g4_rows),
        "",
        f"Frozen gate: `{g4['frozen_gate']}`. Recorded decision: `{g4['decision']}`.",
        "",
        f"Manifest: `{G4_MANIFEST}`.",
        "",
        "## G5 Llama Two-Pass Rollout Provenance",
        "",
        markdown_table(g5_fields, [g5_row]),
        "",
        markdown_table(["path", "bytes", "hash_policy"], g5_inventory_rows),
        "",
        f"Manifest: `{G5_MANIFEST}`.",
        "",
        "## G6 Llama Offline Training Provenance",
        "",
        markdown_table(g6_fields, g6_rows),
        "",
        "## G7 off-KD H Geometry Completion",
        "",
        markdown_table(["field", "value"], g7_rows),
        "",
        f"Manifest: `{G7_MANIFEST}`.",
        "",
        "G7 source-manifest artifact records:",
        "",
        markdown_table(
            ["artifact", "path", "sha256"],
            [
                {"artifact": name, "path": info["path"], "sha256": info["sha256"]}
                for name, info in g7["files"].items()
            ],
        ),
        "",
        "## C1 Direction Analysis Completion",
        "",
        "Requested completed tasks: `" + json.dumps(c1["completed_tasks"], ensure_ascii=True) + "`.",
        "",
        f"Existing task preserved: `{c1['existing_task_preserved']}`.",
        "",
        markdown_table(["output", "path", "rows", "sha256"], c1_rows),
        "",
        f"Manifest: `{C1_MANIFEST}`.",
        "",
        "## C2 Dose-Response Raw Table",
        "",
        markdown_table(c2_fields, c2_rows),
        "",
        f"Manifest: `{C2_MANIFEST}`.",
        "",
        "## Compact Artifact Inventory",
        "",
        markdown_table(["path", "bytes", "sha256"], source_inventory_rows),
        "",
        "## Pending Outside This Intermediate Handoff",
        "",
        "| item | pending raw scope |",
        "|---|---|",
        "| G2 remaining grid | seqKD steps 80,160,320,480 |",
        "| G3 | seqKD ten-step geometry |",
        "| G8 | adapter layer ablation |",
        "",
        "This file records an intermediate handoff and does not mark the full second execution block complete.",
        "",
    ]
    write_text_atomic(HANDOFF, "\n".join(handoff_parts))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete_intermediate",
        "created_at": generated_at,
        "task": "Cycle 09 second execution block intermediate Theory handoff",
        "source_specification": str(SPEC),
        "reporting_guard": "raw readings and provenance only; no interpretation or adjudication",
        "critical_grid": CRITICAL_STEPS,
        "full_grid": FULL_GRID,
        "completed_items": [
            "G1",
            "G2_critical_behavior_and_audits",
            "G4",
            "G5",
            "G6_sft",
            "G6_offkd",
            "G6_seqkd",
            "G7",
            "C1",
            "C2",
        ],
        "pending_items": {
            "G2_remaining_steps": PENDING_G2_STEPS,
            "G3": "pending",
            "G8": "pending",
        },
        "snapshot_rows": {
            "g2_behavior": len(behavior_rows),
            "g2_mmlupro_extract": len(critical_extract),
            "g2_mmlupro_flexible": len(critical_flexible),
            "g2_ifeval_breakdown": len(critical_ifeval),
        },
        "handoff": {"path": str(HANDOFF), "sha256": sha256(HANDOFF)},
        "snapshots": snapshot_records,
        "compact_sources": compact_sources,
        "large_raw_files": g5_large_files,
        "source_manifest_records": {
            "G7": g7.get("files", {}),
            "C1": c1.get("outputs", {}),
            "C2_output": {"path": c2["output"], "sha256": c2["output_sha256"]},
        },
        "machine_shutdown_requested": False,
    }
    write_json_atomic(MANIFEST, manifest)

    evolution_block = "\n".join(
        [
            MARKER_START,
            "",
            "## Cycle 09 Second Execution Block - Intermediate Raw Handin",
            "",
            "The confirmed T+7-8h intermediate scope is frozen as immutable snapshots. "
            "This entry records raw readings and provenance only; it makes no interpretation or adjudication.",
            "",
            "| item | status | artifact |",
            "|---|---|---|",
            f"| G2 critical behavior | complete_intermediate (steps {CRITICAL_STEPS}) | {G2_BEHAVIOR_SNAPSHOT} |",
            f"| G2 MMLU-Pro strict extraction | complete_intermediate (5 rows) | {G2_EXTRACT_SNAPSHOT} |",
            f"| G2 MMLU-Pro flexible extraction | complete_intermediate (5 rows) | {G2_FLEXIBLE_SNAPSHOT} |",
            f"| G2 IFEval native categories | complete_intermediate (45 rows) | {G2_IFEVAL_SNAPSHOT} |",
            f"| G1/G4/G5/G6/G7/C1/C2 provenance and raw tables | handed in | {HANDOFF} |",
            "",
            f"Machine-readable manifest: `{MANIFEST}` (`sha256={sha256(MANIFEST)}`).",
            "",
            "Pending outside this intermediate handoff: G2 steps 80/160/320/480, G3, and G8.",
            "",
            MARKER_END,
        ]
    )
    replace_or_append_marked_block(CODE_EVOLUTION, evolution_block)

    print(
        json.dumps(
            {
                "status": "complete_intermediate",
                "handoff": str(HANDOFF),
                "manifest": str(MANIFEST),
                "snapshot_rows": manifest["snapshot_rows"],
                "handoff_sha256": manifest["handoff"]["sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
