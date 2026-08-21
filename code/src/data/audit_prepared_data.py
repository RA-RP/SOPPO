"""Fail-closed audit of the frozen 30k server dataset and private label joins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonlines

from .dataset import data_file_sha256


PUBLIC_FILES = {
    "labeled_train.jsonl": (2700, True),
    "labeled_val.jsonl": (300, True),
    "unlabeled_train.jsonl": (24000, False),
    "test_inputs.jsonl": (3000, False),
}
RATIOS = {
    "labeled_total": 0.10,
    "labeled_train": 0.09,
    "labeled_val": 0.01,
    "unlabeled_train": 0.80,
    "test": 0.10,
}


def read_public(path: Path, expected_rows: int, labeled: bool, global_ids: set[str]) -> set[str]:
    local_ids = set()
    with jsonlines.open(path) as reader:
        for row in reader:
            required = ("sample_id", "prompt", "response_a", "response_b")
            if any(not isinstance(row.get(key), str) or not row[key] for key in required):
                raise ValueError(f"Malformed preference row in {path}: {row.get('sample_id')}")
            sample_id = row["sample_id"]
            if sample_id in local_ids or sample_id in global_ids:
                raise ValueError(f"Duplicate or cross-split sample ID: {sample_id}")
            local_ids.add(sample_id)
            global_ids.add(sample_id)
            if labeled:
                if int(row.get("label", -1)) not in {0, 1}:
                    raise ValueError(f"Missing/nonbinary public label: {sample_id}")
            else:
                forbidden = {"label", "original_chosen", "original_rejected"} & set(row)
                if forbidden:
                    raise ValueError(f"Public hidden-label leak for {sample_id}: {sorted(forbidden)}")
    if len(local_ids) != expected_rows:
        raise ValueError(f"Row-count mismatch for {path}: {len(local_ids)} != {expected_rows}")
    return local_ids


def read_private_labels(path: Path) -> set[str]:
    values = set()
    with jsonlines.open(path) as reader:
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in values:
                raise ValueError(f"Duplicate private-label ID in {path}: {sample_id}")
            if int(row.get("label", -1)) not in {0, 1}:
                raise ValueError(f"Missing/nonbinary private label: {sample_id}")
            values.add(sample_id)
    return values


def audit(data_dir: Path) -> dict:
    manifest_path = data_dir / "manifest_public.json"
    private_manifest = data_dir / "manifest_private.json"
    if not manifest_path.is_file() or not private_manifest.is_file():
        raise FileNotFoundError(f"Prepared-data manifests are incomplete: {data_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    private_manifest_value = json.loads(private_manifest.read_text(encoding="utf-8"))
    expected_manifest_counts = {
        "labeled_train": 2700,
        "labeled_val": 300,
        "unlabeled_train": 24000,
        "test": 3000,
    }
    actual_counts = {key: int(manifest.get(key, -1)) for key in expected_manifest_counts}
    if actual_counts != expected_manifest_counts or manifest.get("split_ratios") != RATIOS:
        raise ValueError(f"Frozen 30k manifest contract failed: {actual_counts}")

    checksums = manifest.get("checksums") or {}
    global_ids: set[str] = set()
    split_ids = {}
    for filename, (expected_rows, labeled) in PUBLIC_FILES.items():
        path = data_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Prepared-data file is missing: {path}")
        actual_sha = data_file_sha256(path)
        if checksums.get(filename) != actual_sha:
            raise ValueError(f"Prepared-data checksum mismatch: {path}")
        split_ids[filename] = read_public(path, expected_rows, labeled, global_ids)

    private_dir = data_dir / "private_labels"
    private_paths = {
        "unlabeled_labels.jsonl": private_dir / "unlabeled_labels.jsonl",
        "test_labels.jsonl": private_dir / "test_labels.jsonl",
    }
    private_checksums = {name: data_file_sha256(path) for name, path in private_paths.items()}
    recorded_private_checksums = (
        manifest.get("private_checksums") or private_manifest_value.get("private_checksums")
    )
    if recorded_private_checksums and recorded_private_checksums != private_checksums:
        raise ValueError("Prepared private-label checksum mismatch")
    unlabeled_private = read_private_labels(private_paths["unlabeled_labels.jsonl"])
    test_private = read_private_labels(private_paths["test_labels.jsonl"])
    if unlabeled_private != split_ids["unlabeled_train.jsonl"]:
        raise ValueError("Private unlabeled-label IDs do not exactly match public unlabeled inputs")
    if test_private != split_ids["test_inputs.jsonl"]:
        raise ValueError("Private test-label IDs do not exactly match public test inputs")
    return {
        "status": "succeeded",
        "rows": len(global_ids),
        "counts": expected_manifest_counts,
        "checksums": {name: checksums[name] for name in PUBLIC_FILES},
        "private_checksums": private_checksums,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = audit(Path(args.data_dir).resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).resolve()
        if output.exists():
            raise FileExistsError(f"Refuse to overwrite data audit: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
