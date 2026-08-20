#!/usr/bin/env python3
"""Freeze the complete Q1 alpha=.5 step-320 endpoint package for Stage3 H0."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c


OUTPUT = c.MINI / "mini_qwen_alpha05_stage_b_320_handoff.md"
MANIFEST = c.MINI / "qwen_alpha05_stage_b_320_handoff_manifest.json"
TRAINING = c.MINI / "qwen_alpha05_stage_b_training_manifest.json"
EXPORT = c.Q1_ROOT / "qwen_alpha05_stage_b_model_export_manifest.json"
BEHAVIOR = c.MINI / "qwen_alpha05_behavior_manifest.json"
GEOMETRY = c.MINI / "qwen_alpha05_geometry_manifest.json"
SUPPORT = c.MINI / "qwen_alpha05_stage_b_support_stats_manifest.json"
STEPS = [0, 5, 20, 40, 80, 160, 320]


def require_status(path: Path, allowed: set[str]) -> dict[str, Any]:
    payload = c.read_json(path, {})
    if payload.get("status") not in allowed:
        raise RuntimeError(f"{path} status={payload.get('status')!r}, expected one of {sorted(allowed)}")
    return payload


def markdown_table(path: Path) -> str:
    frame = pd.read_csv(path)
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def run() -> dict[str, Any]:
    training = require_status(TRAINING, {"complete_checkpoint_validated"})
    export = require_status(EXPORT, {"complete"})
    behavior = require_status(BEHAVIOR, {"complete"})
    geometry = require_status(GEOMETRY, {"complete"})
    support = require_status(SUPPORT, {"complete_with_declared_terminal_gaps"})

    if behavior.get("steps") != STEPS:
        raise RuntimeError(f"behavior step grid drift: {behavior.get('steps')}")
    if geometry.get("steps") != STEPS:
        raise RuntimeError(f"geometry step grid drift: {geometry.get('steps')}")
    probes = list(geometry.get("probes", []))
    if len(probes) != 6:
        raise RuntimeError(f"geometry probe count={len(probes)} expected=6")
    if support.get("missing_rollout_steps") != [160, 320]:
        raise RuntimeError(f"support gap drift: {support.get('missing_rollout_steps')}")

    behavior_csv = c.MINI / "qwen_alpha05_behavior_keypoints.csv"
    geometry_csv = c.MINI / "qwen_alpha05_r_epsilon.csv"
    support_csv = c.MINI / "qwen_alpha05_stage_b_support_stats.csv"
    required_csvs = (behavior_csv, geometry_csv, support_csv)
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_csvs):
        raise FileNotFoundError("one or more Q1 H0 readout tables are absent")

    inputs = (TRAINING, EXPORT, BEHAVIOR, GEOMETRY, SUPPORT)
    inventory = [c.artifact(path) for path in (*inputs, *required_csvs)]
    coverage = {
        "checkpoint_grid": STEPS,
        "behavior_steps": behavior["steps"],
        "geometry_steps": geometry["steps"],
        "geometry_probes": probes,
        "geometry_layer": geometry.get("layer"),
        "support_saved_steps": support["saved_rollout_steps"],
        "support_missing_steps": support["missing_rollout_steps"],
        "missing_measurement_cells": [],
    }
    markdown = [
        "# Q1 alpha=.5 Stage-B 320 Raw Handoff",
        "",
        f"Generated UTC: {c.utc_now()}",
        "",
        "## Coverage",
        "",
        "```json",
        json.dumps(coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Behavior",
        "",
        markdown_table(behavior_csv),
        "",
        "## Support",
        "",
        markdown_table(support_csv),
        "",
        "## Geometry",
        "",
        f"Full six-probe table: `{geometry_csv}` ({geometry.get('output', {}).get('bytes', geometry_csv.stat().st_size)} bytes).",
        "",
        "## Provenance",
        "",
        "Raw readouts and declared coverage only. No interpretation or adjudication is included.",
    ]
    c.atomic_text(OUTPUT, "\n".join(markdown) + "\n")
    result = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 alpha=.5 step320 H0 endpoint package",
        "checkpoint_grid": STEPS,
        "coverage": coverage,
        "inputs": [c.artifact(path) for path in inputs],
        "inventory": inventory,
        "handoff": c.artifact(OUTPUT),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
