#!/usr/bin/env python3
"""Validate and aggregate the full merged-state Qwen D4.1 matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO = Path("/root/LLM-output-density")
ROOT = Path("/root/autodl-tmp/cycle09_relative_functional_contraction/d4_merged_state/formal")
FINAL = Path("/root/autodl-tmp/cycle09_relative_functional_contraction/final")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
ARMS = ("base", "opd", "sft", "offkd", "seqkd")
CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")
STEPS = {"base": (0,), "opd": (5, 10, 20, 40, 80, 160, 320, 480, 624), "sft": (5, 10, 20, 40, 80, 160, 320, 480, 624), "offkd": (5, 10, 20, 40, 80, 160, 320, 480, 624), "seqkd": (5, 10, 20, 40, 80, 160, 320, 480, 624)}
REQUIRED_PROTOCOL = (
    "checkpoint_storage_dtype", "merge_compute_dtype", "model_load_dtype", "activation_dtype",
    "gram_and_whitening_dtype", "WS_matmul_dtype", "svd_input_dtype", "singular_value_accumulation_dtype",
    "logit_forward_dtype", "logit_storage_dtype", "KL_NLL_compute_dtype",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label(step: int) -> str:
    return f"step_{step:03d}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expected_cells() -> set[tuple[str, int, str]]:
    return {(arm, step, probe) for arm in ARMS for step in STEPS[arm] for probe in CORE_PROBES}


def finalize() -> dict[str, Any]:
    state_cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    output_cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for arm, step, probe in sorted(expected_cells()):
        state_path = ROOT / "state" / arm / label(step) / f"{probe}.json"
        output_path = ROOT / "outputs" / arm / label(step) / f"{probe}.json"
        key = (arm, step, probe)
        if state_path.is_file():
            state_cells[key] = read_json(state_path)
        if output_path.is_file():
            output_cells[key] = read_json(output_path)
    expected = expected_cells()
    complete = {
        key for key in expected
        if state_cells.get(key, {}).get("status") == "complete" and output_cells.get(key, {}).get("status") == "complete"
    }
    missing = sorted(expected - complete)
    if missing:
        raise RuntimeError(f"D4.1 formal matrix incomplete: {len(missing)} missing cells; first={missing[:5]}")

    base_rank: dict[tuple[str, str, float], float] = {}
    for probe in CORE_PROBES:
        item = state_cells[("base", 0, probe)]
        for row in item["state_rows"]:
            base_rank[(probe, row["module"], float(row["epsilon"]))] = float(row["r_epsilon"])

    module_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    sample_digests: dict[str, set[str]] = {probe: set() for probe in CORE_PROBES}
    for key in sorted(expected):
        arm, step, probe = key
        state, output = state_cells[key], output_cells[key]
        if state["sample_ids_sha256"] != output["sample_ids_sha256"]:
            raise RuntimeError(f"sample ID mismatch in state/output for {key}")
        sample_digests[probe].add(state["sample_ids_sha256"])
        protocol = state.get("numerical_protocol", {})
        missing_protocol = [field for field in REQUIRED_PROTOCOL if not protocol.get(field)]
        if missing_protocol:
            raise RuntimeError(f"missing numerical protocol fields for {key}: {missing_protocol}")
        protocol_rows.append({
            "arm": arm, "checkpoint": step, "probe_name": probe,
            "state_sha256": digest(ROOT / "state" / arm / label(step) / f"{probe}.json"),
            "output_sha256": digest(ROOT / "outputs" / arm / label(step) / f"{probe}.json"),
            "sample_ids_sha256": state["sample_ids_sha256"], **protocol,
        })
        for row in state["state_rows"]:
            epsilon = float(row["epsilon"])
            reference = base_rank[(probe, row["module"], epsilon)]
            current = float(row["r_epsilon"])
            absolute = reference - current
            module_rows.append({
                "model": "qwen", "arm": arm, "checkpoint": step, "probe_name": probe, "layer": state["layer"],
                "module": row["module"], "epsilon": epsilon, "state_rank_base": reference,
                "state_rank_current": current, "state_rank_delta": current - reference,
                "absolute_contraction": absolute,
                "relative_functional_contraction_module": absolute / reference if reference else None,
                "source_name": "qwen_d4_merged_state", "source_protocol": "D4.1_current_merged_state",
            })
        frame = pd.DataFrame(output["rows"])
        if len(frame) != int(output["sample_count"]):
            raise RuntimeError(f"output sample count mismatch for {key}")
        means = frame[[
            "cumulative_kl_base_to_current", "nll_base", "nll_current", "delta_nll_cumulative", "absolute_delta_nll_cumulative",
        ]].mean(numeric_only=True).to_dict()
        output_rows.append({
            "model": "qwen", "arm": arm, "checkpoint": step, "probe_name": probe,
            "sample_count": int(output["sample_count"]), "aggregation": "sample_equal_mean_of_token_weighted_rows", **means,
        })

    module = pd.DataFrame(module_rows)
    keys = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "source_name", "source_protocol"]
    aggregates = module.groupby(keys, dropna=False).agg(
        module_count=("module", "nunique"), state_rank_base_mean=("state_rank_base", "mean"),
        state_rank_current_mean=("state_rank_current", "mean"), state_rank_delta_mean=("state_rank_delta", "mean"),
        absolute_contraction_mean=("absolute_contraction", "mean"),
        relative_functional_contraction_equal7=("relative_functional_contraction_module", "mean"),
    ).reset_index()
    aggregates["relative_functional_contraction_ratio_of_means_sensitivity"] = (
        (aggregates["state_rank_base_mean"] - aggregates["state_rank_current_mean"]) / aggregates["state_rank_base_mean"]
    )
    atomic_csv(FINAL / "qwen_d4_merged_state_module_audit.csv", module_rows)
    atomic_csv(FINAL / "qwen_d4_merged_state_all_cells.csv", aggregates.to_dict("records"))
    atomic_csv(FINAL / "qwen_d4_merged_state_outputs.csv", output_rows)
    protocol_payload = {
        "schema_version": "cycle09_qwen_d4_numeric_protocol_v1", "status": "complete", "created_utc": now(),
        "target_cells": len(expected), "complete_cells": len(complete),
        "sample_id_digest_per_probe": {probe: sorted(values) for probe, values in sample_digests.items()},
        "artifacts": protocol_rows,
    }
    atomic_json(MINI / "qwen_merged_state_numeric_protocol.json", protocol_payload)
    manifest = {
        "schema_version": "cycle09_qwen_d4_finalize_v1", "status": "complete", "created_utc": now(),
        "target_cells": len(expected), "complete_cells": len(complete),
        "module_rows": len(module_rows), "aggregate_rows": len(aggregates), "output_rows": len(output_rows),
        "outputs": [str(FINAL / name) for name in (
            "qwen_d4_merged_state_module_audit.csv", "qwen_d4_merged_state_all_cells.csv", "qwen_d4_merged_state_outputs.csv",
        )],
    }
    atomic_json(ROOT / "finalize_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("finalize",), default="finalize")
    args = parser.parse_args()
    print(json.dumps(finalize(), ensure_ascii=False))


if __name__ == "__main__":
    main()
