#!/usr/bin/env python3
"""Write the raw Q1 Stage-A handoff after all strict artifacts are complete."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c


ROOT = c.Q1_ROOT / "postprocess_supervisor"
FORMAL = ROOT / "formal_manifest.json"
OUTPUT = c.MINI / "mini_qwen_alpha05_stage_a_handoff.md"
MANIFEST = c.MINI / "qwen_alpha05_stage_a_handoff_manifest.json"
CODE_EVOLUTION = c.REPO / "mypaper/code/code_evolution.md"
MARKER_START = "<!-- cycle09-q1-stage-a-start -->"
MARKER_END = "<!-- cycle09-q1-stage-a-end -->"


def csv_rows(path: Path) -> int:
    return len(pd.read_csv(path))


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")).replace("|", "\\|") for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def artifact_row(name: str, path: Path, *, rows: int | str = "-") -> dict[str, Any]:
    item = c.artifact(path)
    return {"artifact": name, "rows": rows, "path": item["path"], "sha256": item["sha256"]}


def geometry_readout(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    subset = frame[frame["epsilon"] == 0.05]
    expected = 6 * 6 * 7
    if len(subset) != expected:
        raise RuntimeError(f"Q1 geometry epsilon=.05 rows={len(subset)} expected={expected}")
    grouped = (
        subset.groupby(["step", "probe"], as_index=False)
        .agg(
            module_equal_r_epsilon=("r_epsilon", "mean"),
            module_equal_base_r_epsilon=("base_r_epsilon", "mean"),
            module_equal_delta=("r_epsilon_delta", "mean"),
        )
        .sort_values(["probe", "step"], kind="stable")
    )
    return grouped.to_dict("records")


def code_evolution_block() -> str:
    return "\n".join(
        [
            MARKER_START,
            "",
            "## Cycle 09 Q1 alpha=.5 Stage A",
            "",
            "Implemented the detached Stage-A postprocessing chain: checkpoint validation and source-separated "
            "support statistics; revision-pinned MATH/AIME25 acquisition with frozen deduplication; six-probe "
            "L18 whitened `r_epsilon` geometry; and keypoint MATH500/MMLU-Pro/IFEval evaluation. "
            "The supervisor runs a smoke gate before the formal two-GPU queue, keeps base references ahead of "
            "dependent checkpoint cells, remains resumable by completion artifacts, never shuts the instance down, "
            "and has no path to start the gated `160 -> 320` Stage B.",
            "",
            f"Raw handoff: `{OUTPUT}`.",
            "",
            MARKER_END,
            "",
        ]
    )


def append_code_evolution() -> None:
    current = CODE_EVOLUTION.read_text(encoding="utf-8") if CODE_EVOLUTION.is_file() else ""
    if MARKER_START in current:
        return
    c.atomic_text(CODE_EVOLUTION, current.rstrip() + "\n\n---\n\n" + code_evolution_block())


def run() -> dict[str, Any]:
    formal = c.read_json(FORMAL, {})
    if formal.get("status") != "complete":
        raise RuntimeError(f"Q1 formal supervisor is not complete: {FORMAL}")
    training = c.MINI / "qwen_alpha05_stage_a_training_manifest.json"
    support = c.MINI / "qwen_alpha05_support_stats.csv"
    support_manifest = c.MINI / "qwen_alpha05_support_stats_manifest.json"
    geometry = c.MINI / "qwen_alpha05_r_epsilon.csv"
    geometry_manifest = c.MINI / "qwen_alpha05_geometry_manifest.json"
    behavior = c.MINI / "qwen_alpha05_behavior_keypoints.csv"
    behavior_extract = c.MINI / "qwen_alpha05_mmlupro_extract_audit.csv"
    behavior_flexible = c.MINI / "qwen_alpha05_mmlupro_flexible.csv"
    behavior_ifeval = c.MINI / "qwen_alpha05_ifeval_breakdown.csv"
    behavior_manifest = c.MINI / "qwen_alpha05_behavior_manifest.json"
    required = (
        training,
        support,
        support_manifest,
        geometry,
        geometry_manifest,
        behavior,
        behavior_extract,
        behavior_flexible,
        behavior_ifeval,
        behavior_manifest,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Q1 handoff missing artifacts: {missing}")
    if c.read_json(geometry_manifest, {}).get("status") != "complete":
        raise RuntimeError("Q1 geometry manifest is incomplete")
    if c.read_json(behavior_manifest, {}).get("status") != "complete":
        raise RuntimeError("Q1 behavior manifest is incomplete")

    inventory = [
        artifact_row("Stage-A checkpoint validation", training),
        artifact_row("support statistics", support, rows=csv_rows(support)),
        artifact_row("support-statistics manifest", support_manifest),
        artifact_row("six-probe geometry", geometry, rows=csv_rows(geometry)),
        artifact_row("geometry manifest", geometry_manifest),
        artifact_row("behavior keypoints", behavior, rows=csv_rows(behavior)),
        artifact_row("MMLU-Pro strict extraction audit", behavior_extract, rows=csv_rows(behavior_extract)),
        artifact_row("MMLU-Pro flexible audit", behavior_flexible, rows=csv_rows(behavior_flexible)),
        artifact_row("IFEval category audit", behavior_ifeval, rows=csv_rows(behavior_ifeval)),
        artifact_row("behavior manifest", behavior_manifest),
        artifact_row("formal supervisor manifest", FORMAL),
    ]
    geometry_rows = geometry_readout(geometry)
    behavior_rows = pd.read_csv(behavior).sort_values("step", kind="stable").to_dict("records")
    support_rows = pd.read_csv(support).to_dict("records")
    training_payload = c.read_json(training, {})
    support_payload = c.read_json(support_manifest, {})
    lines = [
        "# Q1 alpha=.5 Stage-A Raw Handoff",
        "",
        f"Generated UTC: {c.utc_now()}",
        "",
        "## Artifact Inventory",
        "",
        markdown_table(inventory, ["artifact", "rows", "path", "sha256"]),
        "",
        "## Source-Separated Saved-Rollout Statistics",
        "",
        markdown_table(support_rows, list(pd.read_csv(support).columns)),
        "",
        "## Geometry: L18, epsilon=.05, Seven-Module Equal Mean",
        "",
        markdown_table(
            geometry_rows,
            ["probe", "step", "module_equal_r_epsilon", "module_equal_base_r_epsilon", "module_equal_delta"],
        ),
        "",
        "## Behavior Keypoints",
        "",
        markdown_table(behavior_rows, list(pd.read_csv(behavior).columns)),
        "",
        "## Provenance",
        "",
        f"- checkpoint validation: `{training_payload.get('status')}`; completed step `{training_payload.get('completed_steps')}`; terminal validation `{training_payload.get('terminal_validation', {}).get('status')}`.",
        f"- saved rollout coverage: `{support_payload.get('status')}`; saved steps `{support_payload.get('saved_rollout_steps')}`; missing `{support_payload.get('missing_rollout_steps')}`.",
        "- geometry: six fixed probes, L18, per-checkpoint whitening, epsilon grid `{.01,.025,.05,.10}`, seven-module equal weighting, and window-token -> sample-window -> sample-equal normalization.",
        "- behavior: MATH500, MMLU-Pro strict/flexible, and IFEval; raw per-cell files and manifests are listed above.",
        "- Stage B: `not_started`; no command in this handoff resumes past step160.",
        "",
    ]
    c.atomic_text(OUTPUT, "\n".join(lines))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 alpha=.5 Stage-A raw Theory handoff",
        "stage_b": "not_started",
        "inventory": inventory,
        "geometry_readout_rows": len(geometry_rows),
        "behavior_rows": len(behavior_rows),
        "handoff": c.artifact(OUTPUT),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, payload)
    append_code_evolution()
    return payload


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
