#!/usr/bin/env python3
"""Validate and merge disjoint Qwen/Llama PROBE-CORE partitions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_stage3_followup_common as c


ROOT = c.RUN_ROOT / "H2_probe_core"
OUTPUT = ROOT / "PROBE_CORE_r_epsilon.csv"
MANIFEST = ROOT / "PROBE_CORE_manifest.json"
PARTITIONS = {
    "qwen3_4b": c.RUN_ROOT
    / "partitions/partition_probe_qwen_20260723/H2_probe_core",
    "llama3_2_3b": c.RUN_ROOT
    / "partitions/partition_probe_llama_20260723/H2_probe_core",
}
KEY = ["family", "arm", "step", "probe", "layer", "module", "epsilon", "track"]


def require_partition(family: str, root: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = root / "PROBE_CORE_manifest.json"
    payload = c.read_json(manifest_path, {})
    if payload.get("status") != "complete":
        raise RuntimeError(f"{family} partition status={payload.get('status')!r}")
    if payload.get("families") != [family]:
        raise RuntimeError(f"{family} partition family drift: {payload.get('families')}")
    output = Path(payload.get("output", {}).get("path", ""))
    if not output.is_file() or output.stat().st_size == 0:
        raise FileNotFoundError(output)
    frame = pd.read_csv(output)
    if len(frame) != int(payload.get("rows", -1)):
        raise RuntimeError(f"{family} rows={len(frame)} manifest={payload.get('rows')}")
    if set(frame["family"]) != {family}:
        raise RuntimeError(f"{family} CSV contains families={sorted(set(frame['family']))}")
    return payload, frame


def merge() -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    frames = []
    for family, root in PARTITIONS.items():
        payload, frame = require_partition(family, root)
        payloads[family] = payload
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    duplicates = combined.duplicated(KEY, keep=False)
    if duplicates.any():
        raise RuntimeError(
            "duplicate PROBE-CORE keys: "
            + combined.loc[duplicates, KEY].head(10).to_json(orient="records")
        )
    combined = combined.sort_values(KEY, kind="stable").reset_index(drop=True)
    c.atomic_csv(OUTPUT, combined.to_dict("records"), list(combined.columns))
    result = {
        "schema_version": 2,
        "status": "complete",
        "task": "PROBE-CORE exact landmark geometry, merged disjoint family partitions",
        "families": list(PARTITIONS),
        "probes": ["E_math", "E_aime24"],
        "partition_manifests": {
            family: c.artifact(root / "PROBE_CORE_manifest.json")
            for family, root in PARTITIONS.items()
        },
        "partition_outputs": {
            family: payloads[family]["output"] for family in PARTITIONS
        },
        "merge_key": KEY,
        "duplicate_keys": 0,
        "rows": len(combined),
        "output": c.artifact(OUTPUT),
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("merge",), required=True)
    parser.parse_args()
    print(json.dumps(merge(), indent=2))
