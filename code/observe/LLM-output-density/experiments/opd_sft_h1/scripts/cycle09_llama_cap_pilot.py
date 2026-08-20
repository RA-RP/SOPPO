#!/usr/bin/env python3
"""Paired early-cap audit for the Llama P1 behavior campaign.

This is an audit between smoke and formal evaluation.  It deliberately uses
the same first 60 MATH500 rows, prompt order, temperature, top-p, and seed for
both caps.  It does not replace or alter any formal behavior cell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cycle09_block3_common as c
import cycle09_llama_model_export as export


ROOT = c.RUN_ROOT / "llama_cap_pilot"
CELL_ROOT = ROOT / "cells"
OUTPUT = ROOT / "llama_early_cap_pilot.csv"
MANIFEST = ROOT / "llama_early_cap_pilot_manifest.json"
THINK_RUNNER = c.REPO / "Eval/component/think_math/runner_think.py"
CASES = (("base", 0), ("opd", 20))
CAPS = (4096, 16384)
N = 60


def label(arm: str, step: int, cap: int) -> str:
    return f"{arm}_step_{step:03d}_cap_{cap}"


def cell_root(arm: str, step: int, cap: int) -> Path:
    return CELL_ROOT / arm / f"step_{step:03d}" / f"cap_{cap}"


def model_path(arm: str, step: int) -> Path:
    return export.merged_target("opd", 0) if step == 0 else export.merged_target(arm, step)


def cell_manifest(arm: str, step: int, cap: int) -> Path:
    return cell_root(arm, step, cap) / "cell_manifest.json"


def complete(arm: str, step: int, cap: int) -> bool:
    return c.read_json(cell_manifest(arm, step, cap), {}).get("status") == "complete"


def cap_model_len(cap: int) -> int:
    return 6144 if cap == 4096 else 18432


def run_cell(arm: str, step: int, cap: int, gpu_mem: float) -> dict[str, Any]:
    if (arm, step) not in CASES or cap not in CAPS:
        raise ValueError(f"unsupported pilot cell {arm}/{step}/cap{cap}")
    target = cell_manifest(arm, step, cap)
    cached = c.read_json(target, {})
    if cached.get("status") == "complete":
        return cached
    model = model_path(arm, step)
    if not c.model_check(model)["complete"]:
        raise FileNotFoundError(f"pilot model export incomplete: {model}")
    root = cell_root(arm, step, cap)
    root.mkdir(parents=True, exist_ok=True)
    run_label = label(arm, step, cap)
    command = [
        sys.executable,
        str(THINK_RUNNER),
        "--task", "math500",
        "--model", str(model),
        "--label", run_label,
        "--n", str(N),
        "--max-tokens", str(cap),
        "--max-model-len", str(cap_model_len(cap)),
        "--gpu-mem", str(gpu_mem),
        "--temperature", "0.6",
        "--top-p", "0.9",
        "--seed", "42",
        "--outdir", str(root),
    ]
    result = subprocess.run(command, cwd=c.REPO)
    if result.returncode != 0:
        raise RuntimeError(f"cap pilot failed rc={result.returncode}: {' '.join(command)}")
    summary_path = root / f"{run_label}.json"
    samples_path = root / f"{run_label}_samples.jsonl"
    generation_manifest = root / f"{run_label}_generations_manifest.json"
    if not summary_path.is_file() or not samples_path.is_file() or not generation_manifest.is_file():
        raise RuntimeError(f"pilot runner did not produce complete artifacts: {root}")
    summary = c.read_json(summary_path, {})
    samples = c.read_jsonl(samples_path)
    protocol = c.read_json(generation_manifest, {}).get("protocol", {})
    if len(samples) != N or int(summary.get("max_tokens", -1)) != cap:
        raise RuntimeError(f"pilot output drift for {arm}/{step}/cap{cap}")
    payload = {
        "schema_version": 1,
        "status": "complete",
        "arm": arm,
        "step": step,
        "cap": cap,
        "n": N,
        "seed": 42,
        "temperature": 0.6,
        "top_p": 0.9,
        "sample_order": "first 60 frozen MATH500 rows; paired across caps",
        "rows_sha256": protocol.get("rows_sha256"),
        "summary": c.artifact(summary_path),
        "samples": c.artifact(samples_path),
        "generation_manifest": c.artifact(generation_manifest),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(target, payload)
    return payload


def finalize() -> dict[str, Any]:
    rows = []
    for arm, step in CASES:
        paired: dict[int, list[dict[str, Any]]] = {}
        cell_payloads = {}
        for cap in CAPS:
            payload = c.read_json(cell_manifest(arm, step, cap), {})
            if payload.get("status") != "complete":
                raise RuntimeError(f"incomplete cap pilot cell: {arm}/{step}/cap{cap}")
            sample_path = Path(payload["samples"]["path"])
            paired[cap] = c.read_jsonl(sample_path)
            cell_payloads[cap] = payload
        if len(paired[CAPS[0]]) != N or len(paired[CAPS[1]]) != N:
            raise RuntimeError(f"pilot sample count drift: {arm}/{step}")
        for low, high in zip(paired[CAPS[0]], paired[CAPS[1]], strict=True):
            if low["gold"] != high["gold"]:
                raise RuntimeError(f"non-paired pilot samples: {arm}/{step}")
        for cap in CAPS:
            samples = paired[cap]
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "cap": cap,
                    "n": N,
                    "accuracy": sum(bool(row["ok"]) for row in samples) / N,
                    "cap_hit_rate": sum(row["finish"] == "length" for row in samples) / N,
                    "eos_rate": sum(row["finish"] == "stop" for row in samples) / N,
                    "boxed_rate": sum("\\boxed" in row["gen"] for row in samples) / N,
                    "mean_response_len": sum(int(row["resp_len"]) for row in samples) / N,
                    "paired_vs_16384_accuracy_delta": (
                        sum(bool(low["ok"]) for low in paired[4096]) / N
                        - sum(bool(high["ok"]) for high in paired[16384]) / N
                        if cap == 4096 else 0.0
                    ),
                    "paired_same_correctness_rate": sum(
                        bool(low["ok"]) == bool(high["ok"])
                        for low, high in zip(paired[4096], paired[16384], strict=True)
                    ) / N,
                    "sample_rows_sha256": cell_payloads[cap]["rows_sha256"],
                }
            )
    c.atomic_csv(OUTPUT, rows)
    c.atomic_csv(c.MINI / OUTPUT.name, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 Llama paired early MATH500 cap audit",
        "cases": [{"arm": arm, "step": step} for arm, step in CASES],
        "caps": list(CAPS),
        "n": N,
        "seed": 42,
        "pairing": "identical first-60 MATH500 rows, prompt order, temperature, top-p, and seed",
        "reporting": "raw paired readout only; formal Llama evaluation retains the established Qwen stepwise cap protocol",
        "output": c.artifact(OUTPUT),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, manifest)
    c.atomic_json(c.MINI / MANIFEST.name, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("cell", "finalize"), required=True)
    parser.add_argument("--arm", choices=("base", "opd"), default="base")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--cap", type=int, choices=CAPS, default=4096)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()
    result = finalize() if args.phase == "finalize" else run_cell(args.arm, args.step, args.cap, args.gpu_mem)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
