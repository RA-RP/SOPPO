#!/usr/bin/env python3
"""Verify the immutable outputs of Round4 UltraFeedback/UltraChat preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def source_revision(path: Path) -> str:
    manifest = read_json(path / "ROUND4_ASSET_MANIFEST.json")
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError(f"Invalid frozen source revision: {path}")
    return revision


def row_type(row: Any, label: str) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"{label}: non-object row")
    fields = [row.get(key) for key in ("instruction", "chosen", "rejected", "unlabeled")]
    if not all(isinstance(value, str) for value in fields) or not fields[0].strip():
        raise ValueError(f"{label}: malformed preference row")
    labeled = bool(fields[1].strip() and fields[2].strip()) and not fields[3].strip()
    unlabeled = bool(fields[3].strip()) and not fields[1].strip() and not fields[2].strip()
    if labeled == unlabeled:
        raise ValueError(f"{label}: row is not exclusively labeled or unlabeled")
    return "labeled" if labeled else "unlabeled"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--ultrafeedback-source", required=True)
    parser.add_argument("--ultrachat-source", required=True)
    args = parser.parse_args()

    prepared = Path(args.prepared_dir).expanduser().resolve(strict=True)
    ultrafeedback = Path(args.ultrafeedback_source).expanduser().resolve(strict=True)
    ultrachat = Path(args.ultrachat_source).expanduser().resolve(strict=True)
    manifest_path = prepared / "ROUND4_PREPROCESS_MANIFEST.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "round4-preprocessing-v1":
        raise RuntimeError("Unsupported Round4 preprocessing manifest")
    expected_sources = {
        "ultrafeedback": (ultrafeedback, source_revision(ultrafeedback)),
        "ultrachat": (ultrachat, source_revision(ultrachat)),
    }
    for name, (path, revision) in expected_sources.items():
        observed = manifest.get("sources", {}).get(name, {})
        if observed.get("source") != str(path) or observed.get("revision") != revision:
            raise RuntimeError(f"Prepared data source mismatch for {name}")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or len(outputs) != 3:
        raise RuntimeError("Prepared manifest must describe exactly three datasets")
    summary = {}
    for name, record in outputs.items():
        path = prepared / record["file_name"]
        if sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Prepared data SHA mismatch: {name}")
        rows = read_json(path)
        if not isinstance(rows, list):
            raise ValueError(f"Prepared dataset is not a list: {name}")
        types = [row_type(row, name) for row in rows]
        observed = {
            "rows": len(rows),
            "labeled_rows": types.count("labeled"),
            "unlabeled_rows": types.count("unlabeled"),
        }
        for key, value in observed.items():
            if record.get(key) != value:
                raise RuntimeError(f"Prepared data count mismatch: {name}.{key}")
        summary[name] = observed
    dataset_info_path = prepared / "dataset_info.json"
    if sha256_file(dataset_info_path) != manifest.get("dataset_info_sha256"):
        raise RuntimeError("Prepared dataset_info SHA mismatch")
    print(json.dumps({"status": "PASS", "datasets": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
