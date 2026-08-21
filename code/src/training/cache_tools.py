"""Prepare unique cache inputs and bind a combined reference cache to target files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jsonlines

from ..data.dataset import TOKENIZATION_CONTRACT, data_file_sha256


def combine(inputs, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite combined cache input: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    with jsonlines.open(output, "w") as writer:
        for input_name in inputs:
            with jsonlines.open(input_name) as reader:
                for row in reader:
                    sample_id = row["sample_id"]
                    identity = (row["prompt"], row["response_a"], row["response_b"])
                    if sample_id in seen and seen[sample_id] != identity:
                        raise ValueError(f"Conflicting content for sample_id={sample_id}")
                    if sample_id not in seen:
                        writer.write(row)
                        seen[sample_id] = identity
    print(f"Combined {len(seen)} unique pairs: {output}")


def split(combined_cache: Path, targets, output_dir: Path, model_manifest: Path, max_length: int) -> None:
    cache = {}
    with jsonlines.open(combined_cache) as reader:
        for row in reader:
            cache[row["sample_id"]] = row
    output_dir.mkdir(parents=True, exist_ok=True)
    model_sha = hashlib.sha256(model_manifest.read_bytes()).hexdigest()
    for target_name in targets:
        target = Path(target_name).resolve()
        output = output_dir / f"{target.name}.ref.jsonl"
        manifest_path = output.with_suffix(".manifest.json")
        if output.exists() or manifest_path.exists():
            raise FileExistsError(f"Refuse to overwrite target reference cache: {output}")
        count = 0
        with jsonlines.open(target) as reader, jsonlines.open(output, "w") as writer:
            for row in reader:
                sample_id = row["sample_id"]
                if sample_id not in cache:
                    raise KeyError(f"Combined reference cache misses {sample_id} from {target}")
                writer.write(cache[sample_id])
                count += 1
        manifest = {
            "schema_version": 1,
            "model_manifest_sha256": model_sha,
            "input_sha256": data_file_sha256(target),
            "rows": count,
            "max_length": max_length,
            "enable_thinking": False,
            "response_only": True,
            "tokenization_contract": TOKENIZATION_CONTRACT,
            "cache_sha256": data_file_sha256(output),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Bound reference cache: {target} -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    combine_parser = sub.add_parser("combine")
    combine_parser.add_argument("--input", action="append", required=True)
    combine_parser.add_argument("--output", required=True)
    split_parser = sub.add_parser("split")
    split_parser.add_argument("--combined-cache", required=True)
    split_parser.add_argument("--target", action="append", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--model-manifest", required=True)
    split_parser.add_argument("--max-length", type=int, required=True)
    args = parser.parse_args()
    if args.command == "combine":
        combine(args.input, Path(args.output).resolve())
    else:
        split(
            Path(args.combined_cache).resolve(),
            args.target,
            Path(args.output_dir).resolve(),
            Path(args.model_manifest).resolve(),
            args.max_length,
        )


if __name__ == "__main__":
    main()
