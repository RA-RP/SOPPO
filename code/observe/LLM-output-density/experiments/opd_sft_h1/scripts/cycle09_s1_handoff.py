#!/usr/bin/env python3
"""Validate Stage 1 artifacts and write the raw Theory handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/root/LLM-output-density")
MINI = REPO / (
    "mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
EVOLUTION = REPO / "mypaper/code/code_evolution.md"
HANDOFF = MINI / "mini_stage1_theory_handoff.md"
START_MARKER = "<!-- cycle09-stage1-start -->"
END_MARKER = "<!-- cycle09-stage1-end -->"

TASKS = (
    (
        "S1-1 MMLU-Pro extraction audit",
        "S1_mmlupro_extract_audit.csv",
        "S1_mmlupro_audit_manifest.json",
        30,
    ),
    (
        "S1-2 MMLU-Pro flexible extraction",
        "S1_mmlupro_flexible.csv",
        "S1_mmlupro_audit_manifest.json",
        30,
    ),
    (
        "S1-3 transient-window paired CI",
        "S1_transient_ci.csv",
        "S1_transient_ci_manifest.json",
        48,
    ),
    (
        "S1-4 fixed wikitext-family PPL trajectory",
        "S1_wikitext_ppl.csv",
        "S1_wikitext_ppl_manifest.json",
        30,
    ),
    (
        "S1-5 base-PPL on three training corpora",
        "S1_train_corpus_base_ppl.csv",
        "S1_train_corpus_base_ppl_manifest.json",
        3,
    ),
    (
        "S1-6 R6-2 direction analysis",
        "S1_direction_analysis.csv",
        "S1_direction_analysis_manifest.json",
        210,
    ),
    (
        "S1-7 H/B1 generated-text statistics",
        "S1_h_text_stats.csv",
        "S1_h_text_stats_manifest.json",
        70,
    ),
)

COMPANIONS = (
    ("S1-6 all principal angles", "S1_direction_principal_angles.csv", None),
    ("S1-6 top-10 base-sigma ranks", "S1_direction_rank_distribution.csv", 2100),
    ("S1-6 pairwise direction-set overlap", "S1_direction_overlap.csv", 210),
    ("S1-7 per-generation-seed statistics", "S1_h_text_stats_by_seed.csv", 210),
    ("S1-5 per-sample conditional NLL", "S1_train_corpus_base_ppl_samples.csv", 1500),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        handle.write(text)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    atomic_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        path,
    )


def load_csv(path: Path, expected_rows: int | None) -> tuple[list[str], list[list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as error:
            raise RuntimeError(f"empty CSV: {path}") from error
        rows = list(reader)
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError(
            f"{path.name}: expected {expected_rows} rows, found {len(rows)}"
        )
    if not rows:
        raise RuntimeError(f"no data rows: {path}")
    if any(len(row) != len(header) for row in rows):
        raise RuntimeError(f"ragged CSV: {path}")
    return header, rows


def cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(cell(value) for value in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def provenance_line(csv_path: Path, manifest_path: Path) -> str:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = (
        manifest.get("protocol_id")
        or manifest.get("protocol_version")
        or manifest.get("schema_version")
        or "not_recorded"
    )
    created = (
        manifest.get("created_at")
        or manifest.get("created_utc")
        or manifest.get("completed_at")
        or manifest.get("generated_at")
        or "not_recorded"
    )
    return (
        f"Provenance: `{manifest_path}` "
        f"(sha256 `{sha256_file(manifest_path)}`; protocol `{protocol}`; "
        f"created `{created}`); CSV sha256 `{sha256_file(csv_path)}`."
    )


def artifact_record(label: str, path: Path, expected: int | None) -> dict:
    header, rows = load_csv(path, expected)
    return {
        "label": label,
        "path": str(path),
        "rows": len(rows),
        "columns": len(header),
        "sha256": sha256_file(path),
    }


def validate_task_grids(records: dict[str, tuple[list[str], list[list[str]]]]) -> None:
    for name in (TASKS[0][1], TASKS[1][1], TASKS[3][1]):
        header, rows = records[name]
        arm = header.index("arm")
        step = header.index("step")
        cells = {(row[arm], int(float(row[step]))) for row in rows}
        expected = {
            (a, s)
            for a in ("opd", "sft", "offkd")
            for s in (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
        }
        if cells != expected:
            raise RuntimeError(f"incomplete three-arm ten-step grid: {name}")

    header, rows = records["S1_h_text_stats.csv"]
    arm = header.index("arm")
    step = header.index("step")
    probe = header.index("probe")
    expected_h = {
        (p, a, s)
        for p in ("H_bos", "H_general", "H_ood")
        for a in ("opd", "sft")
        for s in (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
    }
    expected_h |= {
        ("B1_X_sft_math", "sft", s)
        for s in (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
    }
    actual_h = {(row[probe], row[arm], int(float(row[step]))) for row in rows}
    if actual_h != expected_h:
        raise RuntimeError("S1-7 is not the user-approved ten-checkpoint H/B1 grid")


def main() -> None:
    created = datetime.now(timezone.utc).isoformat()
    sections = [
        "# Mini-cycle 09 Stage 1 raw handoff",
        "",
        f"Generated: `{created}`",
        "",
        "Guard: raw readings and provenance only; no interpretation or adjudication.",
        "",
        "Source specification: `mypaper/theory/stage_plan_handoff.md`, confirmed Stage 1 section.",
    ]
    task_records: dict[str, tuple[list[str], list[list[str]]]] = {}
    artifact_records: list[dict] = []
    for title, csv_name, manifest_name, expected_rows in TASKS:
        csv_path = MINI / csv_name
        manifest_path = MINI / manifest_name
        header, rows = load_csv(csv_path, expected_rows)
        task_records[csv_name] = (header, rows)
        artifact_records.append(
            {
                "label": title,
                "path": str(csv_path),
                "rows": len(rows),
                "columns": len(header),
                "sha256": sha256_file(csv_path),
            }
        )
        sections.extend(
            [
                "",
                f"## {title}",
                "",
                markdown_table(header, rows),
                "",
                provenance_line(csv_path, manifest_path),
            ]
        )
    validate_task_grids(task_records)

    companion_records = [
        artifact_record(label, MINI / name, expected)
        for label, name, expected in COMPANIONS
    ]
    artifact_records.extend(companion_records)
    companion_rows = [
        [
            record["label"],
            record["path"],
            str(record["rows"]),
            str(record["columns"]),
            record["sha256"],
        ]
        for record in companion_records
    ]
    sections.extend(
        [
            "",
            "## Companion raw artifacts",
            "",
            markdown_table(
                ["artifact", "path", "rows", "columns", "sha256"], companion_rows
            ),
        ]
    )

    inventory = MINI / "S1_machine_migration_inventory.csv"
    inv_manifest = MINI / "S1_machine_migration_inventory_manifest.json"
    inv_header, inv_rows = load_csv(inventory, 54)
    status_index = inv_header.index("status")
    if any(row[status_index] != "READY" for row in inv_rows):
        raise RuntimeError("migration inventory contains non-READY required items")
    inventory_record = {
        "label": "machine migration inventory",
        "path": str(inventory),
        "rows": len(inv_rows),
        "columns": len(inv_header),
        "sha256": sha256_file(inventory),
    }
    artifact_records.append(inventory_record)
    sections.extend(
        [
            "",
            "## Machine migration inventory",
            "",
            markdown_table(inv_header, inv_rows),
            "",
            provenance_line(inventory, inv_manifest),
            "",
            "Operation: `inventory_only_no_sync`.",
        ]
    )
    atomic_text("\n".join(sections) + "\n", HANDOFF)

    evolution_rows = [
        [
            record["label"],
            record["path"],
            str(record["rows"]),
            record["sha256"],
        ]
        for record in artifact_records
    ]
    evolution_block = "\n".join(
        [
            START_MARKER,
            "",
            "## Cycle 09 Stage 1 - Raw handin",
            "",
            "Specification: `mypaper/theory/stage_plan_handoff.md` confirmed Stage 1. "
            "No tentative night-block task or data synchronization was executed.",
            "",
            markdown_table(["artifact", "path", "rows", "sha256"], evolution_rows),
            "",
            f"Raw Theory handoff: `{HANDOFF}`.",
            "",
            "Migration operation: `inventory_only_no_sync`; all required inventory rows are `READY`.",
            "",
            END_MARKER,
        ]
    )
    existing = EVOLUTION.read_text(encoding="utf-8")
    pattern = re.compile(
        re.escape(START_MARKER) + ".*?" + re.escape(END_MARKER), re.DOTALL
    )
    if pattern.search(existing):
        updated = pattern.sub(evolution_block, existing)
    else:
        updated = existing.rstrip() + "\n\n---\n\n" + evolution_block + "\n"
    atomic_text(updated, EVOLUTION)

    manifest_path = MINI / "S1_stage1_handoff_manifest.json"
    atomic_json(
        {
            "schema_version": 1,
            "created_at": created,
            "status": "complete",
            "source_spec": str(REPO / "mypaper/theory/stage_plan_handoff.md"),
            "guard": "raw_readings_and_provenance_only_no_interpretation_no_adjudication",
            "night_block_executed": False,
            "data_sync_executed": False,
            "machine_shutdown_requested": False,
            "handoff": str(HANDOFF),
            "handoff_sha256": sha256_file(HANDOFF),
            "code_evolution": str(EVOLUTION),
            "artifacts": artifact_records,
        },
        manifest_path,
    )
    print(
        f"[S1 handoff] validated={len(artifact_records)} artifacts -> {HANDOFF}",
        flush=True,
    )


if __name__ == "__main__":
    main()
