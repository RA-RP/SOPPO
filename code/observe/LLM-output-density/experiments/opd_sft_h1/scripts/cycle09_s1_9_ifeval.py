#!/usr/bin/env python3
"""Build the Stage-1 IFEval prompt-strict breakdown from saved samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ARMS = ("opd", "sft", "offkd")
REPO = Path("/root/LLM-output-density")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
OUTPUT = MINI / "S1_ifeval_breakdown.csv"
MANIFEST = MINI / "S1_ifeval_breakdown_manifest.json"
MISSING = MINI / "S1_ifeval_missing.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sample_root(arm: str, step: int) -> Path:
    label = f"step_{step:03d}"
    if arm == "offkd":
        return Path("/root/autodl-tmp/cycle09_offkd/eval/ood_expansion") / label
    return Path("/root/autodl-tmp/cycle09_r3/ood_expansion") / arm / label


def find_sample(arm: str, step: int) -> tuple[Path | None, str]:
    root = sample_root(arm, step)
    paths = sorted(root.rglob("samples_ifeval_*.jsonl")) if root.is_dir() else []
    if len(paths) == 1:
        return paths[0], "ok"
    if not paths:
        return None, "missing"
    return None, f"ambiguous:{len(paths)}"


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc = row.get("doc") or {}
            ids = doc.get("instruction_id_list")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"{path}:{line_number}: missing instruction_id_list")
            strict = row.get("prompt_level_strict_acc")
            if not isinstance(strict, bool):
                raise ValueError(f"{path}:{line_number}: invalid prompt strict metric")
            inst = row.get("inst_level_strict_acc")
            if not isinstance(inst, list) or len(inst) != len(ids):
                raise ValueError(f"{path}:{line_number}: instruction metric length mismatch")
            rows.append(row)
    return rows


def response_text(row: dict) -> str:
    resps = row.get("resps")
    if (
        isinstance(resps, list)
        and resps
        and isinstance(resps[0], list)
        and resps[0]
    ):
        return str(resps[0][0])
    filtered = row.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        return str(filtered[0])
    return ""


def fingerprint(rows: list[dict]) -> str:
    payload = []
    for row in rows:
        payload.append(
            {
                "doc_hash": row.get("doc_hash"),
                "instruction_id_list": row["doc"]["instruction_id_list"],
            }
        )
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mean(values: list[int]) -> float:
    return float(statistics.fmean(values))


def build_breakdown(cells: list[dict]) -> list[dict]:
    output = []
    for cell in cells:
        groups: dict[str, list[tuple[bool, int, int]]] = defaultdict(list)
        for row in cell["rows"]:
            text = response_text(row)
            chars = len(text)
            words = len(re.findall(r"\S+", text))
            categories = {
                str(instruction_id).split(":", 1)[0]
                for instruction_id in row["doc"]["instruction_id_list"]
            }
            for category in categories:
                groups[category].append(
                    (bool(row["prompt_level_strict_acc"]), chars, words)
                )
        for category in sorted(groups):
            values = groups[category]
            passes = [int(value[0]) for value in values]
            chars = [value[1] for value in values]
            words = [value[2] for value in values]
            output.append(
                {
                    "arm": cell["arm"],
                    "step": cell["step"],
                    "instruction_category": category,
                    "n_prompts": len(values),
                    "n_pass": sum(passes),
                    "pass_rate": mean(passes),
                    "resp_len": mean(chars),
                    "resp_len_median": float(statistics.median(chars)),
                    "resp_words_mean": mean(words),
                    "resp_words_median": float(statistics.median(words)),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    selected = (("opd", 0),) if args.smoke else tuple(
        (arm, step) for arm in ARMS for step in STEPS
    )
    missing = []
    cells = []
    reference_fingerprint = None
    for arm, step in selected:
        path, state = find_sample(arm, step)
        if path is None:
            missing.append(
                {
                    "arm": arm,
                    "step": step,
                    "expected_root": str(sample_root(arm, step)),
                    "state": state,
                }
            )
            continue
        rows = read_rows(path)
        current_fingerprint = fingerprint(rows)
        if len(rows) != 541:
            missing.append(
                {
                    "arm": arm,
                    "step": step,
                    "expected_root": str(sample_root(arm, step)),
                    "state": f"invalid_row_count:{len(rows)}",
                }
            )
            continue
        if reference_fingerprint is None:
            reference_fingerprint = current_fingerprint
        elif current_fingerprint != reference_fingerprint:
            missing.append(
                {
                    "arm": arm,
                    "step": step,
                    "expected_root": str(sample_root(arm, step)),
                    "state": "doc_or_instruction_fingerprint_mismatch",
                }
            )
            continue
        cells.append(
            {
                "arm": arm,
                "step": step,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "n_rows": len(rows),
                "rows": rows,
            }
        )

    if missing:
        if not args.smoke:
            atomic_csv(
                missing,
                ["arm", "step", "expected_root", "state"],
                MISSING,
            )
            atomic_json(
                {
                    "status": "MISSING_INPUTS",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "missing": missing,
                    "generation_rerun": False,
                },
                MANIFEST,
            )
        raise SystemExit(f"IFEval sample inventory incomplete: {missing}")

    breakdown = build_breakdown(cells)
    expected_rows = 9 if args.smoke else 270
    if len(breakdown) != expected_rows:
        raise RuntimeError(
            f"unexpected breakdown row count: {len(breakdown)} != {expected_rows}"
        )
    if args.smoke:
        print(json.dumps(breakdown, ensure_ascii=False, indent=2))
        return

    fields = [
        "arm",
        "step",
        "instruction_category",
        "n_prompts",
        "n_pass",
        "pass_rate",
        "resp_len",
        "resp_len_median",
        "resp_words_mean",
        "resp_words_median",
    ]
    atomic_csv(breakdown, fields, OUTPUT)
    manifest_cells = [
        {key: value for key, value in cell.items() if key != "rows"} for cell in cells
    ]
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-9 IFEval native instruction-category audit",
            "status": "COMPLETE",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "source": "saved lm-eval per-sample JSONL only",
                "generation_rerun": False,
                "instruction_category": "prefix before ':' in instruction_id_list",
                "prompt_pass": "prompt_level_strict_acc from the saved sample",
                "multi_category_prompt": "counted once in every distinct native prefix present",
                "resp_len": "mean Unicode code-point count of raw resps[0][0]",
                "resp_len_median": "median Unicode code-point count",
                "resp_words": "whitespace-delimited non-empty spans",
            },
            "arms": list(ARMS),
            "steps": list(STEPS),
            "n_cells": len(cells),
            "n_input_rows_per_cell": 541,
            "n_output_rows": len(breakdown),
            "sample_fingerprint": reference_fingerprint,
            "cells": manifest_cells,
            "output": str(OUTPUT),
            "output_sha256": sha256_file(OUTPUT),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MANIFEST,
    )
    print(f"[S1-9] complete cells={len(cells)} rows={len(breakdown)}")


if __name__ == "__main__":
    main()
