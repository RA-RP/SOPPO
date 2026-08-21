"""Build the server-private DPO-100 training file in an isolated stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonlines

from .dataset import data_file_sha256


def load_labels(path: Path) -> dict:
    labels = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            sample_id = row["sample_id"]
            label = int(row["label"])
            if sample_id in labels:
                raise ValueError(f"Duplicate private unlabeled-label ID: {sample_id}")
            if label not in {0, 1}:
                raise ValueError(f"Private unlabeled label must be 0/1: {sample_id}")
            labels[sample_id] = label
    return labels


def verify_existing(paths: dict[str, Path]) -> None:
    output = paths["output"]
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if not output.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Oracle materialization is incomplete: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_inputs = {
        str(paths[name]): data_file_sha256(paths[name])
        for name in ("labeled", "unlabeled", "private_labels")
    }
    if (
        manifest.get("classification") != "SERVER_PRIVATE_LABEL_DERIVATIVE"
        or int(manifest.get("rows", -1)) != 26700
        or manifest.get("inputs") != expected_inputs
        or manifest.get("output_sha256") != data_file_sha256(output)
    ):
        raise ValueError(f"Existing oracle materialization is stale or mutated: {output}")
    print(f"Verified existing oracle training rows: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", required=True)
    parser.add_argument("--unlabeled", required=True)
    parser.add_argument("--private-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    paths = {
        key: Path(value).resolve()
        for key, value in vars(args).items()
        if key != "verify"
    }
    output = paths["output"]
    if args.verify:
        verify_existing(paths)
        return
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite oracle materialization: {output}")
    labels = load_labels(paths["private_labels"])
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    unlabeled_ids = set()
    with jsonlines.open(output, "w") as writer:
        with jsonlines.open(paths["labeled"]) as reader:
            for row in reader:
                if row["sample_id"] in seen:
                    raise ValueError(f"Duplicate labeled sample: {row['sample_id']}")
                seen.add(row["sample_id"])
                writer.write(row)
        with jsonlines.open(paths["unlabeled"]) as reader:
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id in seen or sample_id not in labels:
                    raise ValueError(f"Oracle label join failed: {sample_id}")
                writer.write({**row, "label": labels[sample_id]})
                seen.add(sample_id)
                unlabeled_ids.add(sample_id)
    if set(labels) != unlabeled_ids:
        raise ValueError("Private unlabeled-label IDs do not exactly match unlabeled inputs")
    manifest = {
        "schema_version": 1,
        "rows": len(seen),
        "inputs": {
            str(paths[name]): data_file_sha256(paths[name])
            for name in ("labeled", "unlabeled", "private_labels")
        },
        "output_sha256": data_file_sha256(output),
        "classification": "SERVER_PRIVATE_LABEL_DERIVATIVE",
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Materialized {len(seen)} oracle training rows: {output}")
    verify_existing(paths)


if __name__ == "__main__":
    main()
