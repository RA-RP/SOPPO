"""Create a tiny server-only fixture from prepared data for the GPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonlines

from ..data.dataset import PreferenceDataset
from ..model.model_utils import load_tokenizer


def take(
    input_path: Path,
    output_path: Path,
    count: int,
    labels=None,
    prefer_longest: bool = False,
) -> list:
    candidates = []
    with jsonlines.open(input_path) as reader:
        for row in reader:
            if labels is not None:
                row = {**row, "label": labels[row["sample_id"]]}
            candidates.append(row)
            if not prefer_longest and len(candidates) >= count:
                break
    if prefer_longest:
        candidates.sort(
            key=lambda row: (
                -(
                    len(row["prompt"])
                    + max(len(row["response_a"]), len(row["response_b"]))
                ),
                -max(len(row["response_a"]), len(row["response_b"])),
                row["sample_id"],
            )
        )
    rows = candidates[:count]
    if len(rows) < count:
        raise ValueError(f"Smoke fixture source too small: {input_path}")
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    data = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite smoke fixture: {output}")
    output.mkdir(parents=True)
    take(
        data / "labeled_train.jsonl",
        output / "labeled_train.jsonl",
        16,
        prefer_longest=True,
    )
    take(
        data / "labeled_val.jsonl",
        output / "labeled_val.jsonl",
        4,
        prefer_longest=True,
    )
    unlabeled = take(
        data / "unlabeled_train.jsonl",
        output / "unlabeled_train.jsonl",
        16,
        prefer_longest=True,
    )
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
    tokenizer = load_tokenizer(args.model)
    length_profile = {}
    for filename in (
        "labeled_train.jsonl",
        "labeled_val.jsonl",
        "unlabeled_train.jsonl",
    ):
        dataset = PreferenceDataset(
            str(output / filename),
            tokenizer,
            max_length=args.max_length,
            enable_thinking=False,
        )
        lengths = [
            max(len(row["input_ids_a"]), len(row["input_ids_b"]))
            for row in (dataset[index] for index in range(len(dataset)))
        ]
        length_profile[filename] = {"rows": len(lengths), "max_tokens": max(lengths)}
    missing_limit = [
        filename
        for filename, profile in length_profile.items()
        if profile["max_tokens"] != args.max_length
    ]
    if missing_limit:
        raise ValueError(
            "Smoke fixture did not exercise max-length truncation in every split: "
            f"missing={missing_limit}, required={args.max_length}"
        )
    (output / "length_profile.json").write_text(
        json.dumps(length_profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Smoke fixture ready: {output}")


if __name__ == "__main__":
    main()
