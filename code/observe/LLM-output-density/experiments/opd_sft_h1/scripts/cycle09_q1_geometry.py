#!/usr/bin/env python3
"""Q1 alpha=.5 L18 six-probe whitened functional-rank campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cycle09_block3_common as c
import cycle09_block3_qwen_probe_geometry as base
import cycle09_r4_campaign as campaign
import cycle09_stage3_common as stage3


ARM = "alpha05"
STEPS = (0, 5, 20, 40, 80, 160, 320)
ROOT = c.Q1_ROOT / "geometry"
OUTPUT = c.MINI / "qwen_alpha05_r_epsilon.csv"
MANIFEST = c.MINI / "qwen_alpha05_geometry_manifest.json"


def configure() -> None:
    from transformers import AutoTokenizer

    base.ROOT = ROOT
    base.CORPUS_ROOT = ROOT / "corpora"
    base.CELL_ROOT = ROOT / "cells"
    base.REFERENCE_ROOT = ROOT / "references"
    base.FACTOR_ROOT = ROOT / "factors"
    base.PROBE_MANIFEST = base.CORPUS_ROOT / "probe_manifest.json"
    base.R2_OUTPUT = ROOT / "unused_r2_output.csv"
    base.R2_MANIFEST = ROOT / "unused_r2_manifest.json"
    if ARM not in stage3.ARMS:
        stage3.ARMS = (*stage3.ARMS, ARM)
    original = getattr(stage3, "_q1_original_model_path", stage3.model_path)
    stage3._q1_original_model_path = original

    def model_path(arm: str, step: int) -> Path:
        if arm == ARM:
            return c.QWEN_STUDENT if step == 0 else c.Q1_ROOT / "_merged_models" / f"step_{step:03d}"
        return original(arm, step)

    stage3.model_path = model_path

    if not hasattr(campaign, "load_tokenizer"):
        def load_tokenizer(path: Path):
            return AutoTokenizer.from_pretrained(
                str(path), local_files_only=True, trust_remote_code=True
            )

        campaign.load_tokenizer = load_tokenizer


def finalize() -> dict:
    configure()
    rows = []
    cells = []
    for probe in base.ALL_PROBES:
        for step in STEPS:
            cell_arm = "base" if step == 0 else ARM
            payload = c.read_json(base.cell_path(cell_arm, step, probe), {})
            if payload.get("status") != "complete":
                raise RuntimeError(f"incomplete Q1 geometry cell: {cell_arm}/{step}/{probe}")
            for row in payload["rows"]:
                rows.append({**row, "arm": ARM, "shared_base_compute": step == 0})
            cells.append(c.artifact(base.cell_path(cell_arm, step, probe)))
    expected = len(STEPS) * len(base.ALL_PROBES) * len(base.c4.MODULES) * len(base.EPSILONS)
    if len(rows) != expected:
        raise RuntimeError(f"Q1 geometry row drift: {len(rows)} != {expected}")
    c.atomic_csv(OUTPUT, rows)
    payload = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 alpha=.5 L18 six-probe per-checkpoint whitened functional rank",
        "arm": ARM,
        "steps": list(STEPS),
        "probes": list(base.ALL_PROBES),
        "layer": base.LAYER,
        "modules": list(base.c4.MODULES),
        "epsilons": list(base.EPSILONS),
        "window_protocol": "window token mean -> sample window mean -> sample equal mean",
        "track": "per_checkpoint_only",
        "cells": cells,
        "output": c.artifact(OUTPUT),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "reference", "cell", "finalize"), required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--probe", choices=base.ALL_PROBES, default="E_math")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=16384)
    parser.add_argument("--no-retain-factor", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = base.prepare_corpora()
    elif args.phase == "reference":
        args.arm = "base"
        args.step = 0
        result = base.run_reference(args)
    elif args.phase == "cell":
        if args.step not in STEPS[1:]:
            raise ValueError(f"Q1 nonbase geometry step must be one of {STEPS[1:]}")
        args.arm = ARM
        result = base.run_cell(args)
    else:
        result = finalize()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
