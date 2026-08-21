"""Create a tiny server-only fixture from prepared data for the GPU smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

import jsonlines


def take(input_path: Path, output_path: Path, count: int, labels=None) -> list:
    rows = []
    with jsonlines.open(input_path) as reader:
        for row in reader:
            if len(rows) >= count:
                break
            if labels is not None:
                row = {**row, "label": labels[row["sample_id"]]}
            rows.append(row)
    if len(rows) < count:
        raise ValueError(f"Smoke fixture source too small: {input_path}")
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite smoke fixture: {output}")
    output.mkdir(parents=True)
    take(data / "labeled_train.jsonl", output / "labeled_train.jsonl", 16)
    take(data / "labeled_val.jsonl", output / "labeled_val.jsonl", 4)
    unlabeled = take(data / "unlabeled_train.jsonl", output / "unlabeled_train.jsonl", 16)
    private = {}
    with jsonlines.open(data / "private_labels" / "unlabeled_labels.jsonl") as reader:
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in private:
                raise ValueError(f"Duplicate private unlabeled-label ID: {sample_id}")
            private[sample_id] = int(row["label"])
    labeled_rows = []
    with jsonlines.open(output / "labeled_train.jsonl") as reader:
        labeled_rows.extend(reader)
    oracle = labeled_rows + [{**row, "label": private[row["sample_id"]]} for row in unlabeled]
    with jsonlines.open(output / "oracle_train.private.jsonl", "w") as writer:
        writer.write_all(oracle)
    print(f"Smoke fixture ready: {output}")


if __name__ == "__main__":
    main()
