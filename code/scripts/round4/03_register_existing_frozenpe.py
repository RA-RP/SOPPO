#!/usr/bin/env python3
"""Register an already-generated FrozenPE fixture for a resumed smoke run.

Candidate generation intentionally refuses to overwrite an existing fixture.
When a smoke run is resumed after candidate generation but before FrozenPE
training, this utility restores only the matching ``dataset_info.json`` entry.
It never reads or rewrites the candidate rows themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-data-dir", required=True)
    parser.add_argument("--dataset-name", default="round4_smoke_frozenpe_train")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.smoke_data_dir).expanduser().resolve(strict=True)
    dataset_info_path = data_dir / "dataset_info.json"
    candidate_path = data_dir / "frozenpe_train.json"
    manifest_path = candidate_path.with_suffix(candidate_path.suffix + ".manifest.json")
    for path in (dataset_info_path, candidate_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required FrozenPE resume artifact is missing: {path}")

    rows = read_json(candidate_path)
    manifest = read_json(manifest_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError("FrozenPE candidate fixture must be a non-empty JSON list.")
    if not isinstance(manifest, dict):
        raise ValueError("FrozenPE candidate manifest must be a JSON object.")

    unlabeled = [row for row in rows if isinstance(row, dict) and str(row.get("unlabeled", "")).strip()]
    labeled = [row for row in rows if isinstance(row, dict) and str(row.get("chosen", "")).strip()]
    if not labeled or not unlabeled:
        raise ValueError("FrozenPE resumed fixture must retain both labeled and unlabeled rows.")
    if any(not str(row.get("unlabeled_b", "")).strip() for row in unlabeled):
        raise ValueError("FrozenPE resumed fixture has unlabeled rows without candidate B.")

    dataset_info = read_json(dataset_info_path)
    if not isinstance(dataset_info, dict):
        raise ValueError("dataset_info.json must contain an object.")
    dataset_info[args.dataset_name] = {
        "file_name": candidate_path.name,
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "chosen": "chosen",
            "rejected": "rejected",
            "unlabeled": "unlabeled",
            "unlabeled_b": "unlabeled_b",
        },
    }
    atomic_json(dataset_info_path, dataset_info)
    print(
        json.dumps(
            {
                "dataset_name": args.dataset_name,
                "candidate_rows": len(rows),
                "labeled_rows": len(labeled),
                "unlabeled_rows": len(unlabeled),
                "candidate_sha256": sha256_file(candidate_path),
                "manifest_counts": manifest.get("counts"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
