"""Derive the frozen Round2 single-response anchor from public MVP data.

This server-only utility never reads private labels.  It deterministically maps
each public ``unlabeled_train.jsonl`` row to its already position-randomized
``response_a`` and writes a separate immutable Round2 artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from .sft_schema import SFT_SCHEMA_VERSION, validate_sft_corpus


SELECTION_RULE = "mvp_unlabeled_public_response_a_after_seed42_position_randomization"
FORBIDDEN_SOURCE_FIELDS = {
    "label",
    "chosen",
    "rejected",
    "original_chosen",
    "original_rejected",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_public_source_manifest(
    source: Path, expected_rows: int
) -> Dict[str, Any]:
    manifest_path = source.parent / "manifest_public.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frozen public data manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Frozen public data manifest must be a JSON object")
    if manifest.get("dataset") != "openbmb/UltraFeedback":
        raise ValueError("Round2 anchor requires the frozen UltraFeedback source")
    if int(manifest.get("unlabeled_train", -1)) != int(expected_rows):
        raise ValueError("Frozen public manifest has the wrong unlabeled row count")
    ratios = manifest.get("split_ratios") or {}
    if ratios.get("unlabeled_train") != 0.8:
        raise ValueError("Frozen public manifest has the wrong unlabeled split ratio")
    position_ratio = (manifest.get("position_randomization_ratio") or {}).get(
        "unlabeled"
    )
    if not isinstance(position_ratio, (int, float)) or not 0.45 <= float(
        position_ratio
    ) <= 0.55:
        raise ValueError(
            "Frozen public manifest does not prove balanced A/B position randomization"
        )
    source_sha256 = _file_sha256(source)
    if (manifest.get("checksums") or {}).get(source.name) != source_sha256:
        raise ValueError("Frozen public unlabeled source checksum mismatch")
    return {
        "path": str(manifest_path.resolve()),
        "sha256": _file_sha256(manifest_path),
        "source_sha256": source_sha256,
        "position_randomization_ratio": float(position_ratio),
    }


def _iter_anchor_rows(source: Path) -> Iterable[Dict[str, str]]:
    seen = set()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid unlabeled JSON at line {line_number}: {source}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Unlabeled row {line_number} must be an object")
            leaked = FORBIDDEN_SOURCE_FIELDS & set(row)
            if leaked:
                raise ValueError(
                    "Public unlabeled source exposes forbidden preference fields: "
                    f"line={line_number}, fields={sorted(leaked)}"
                )
            for key in ("sample_id", "prompt", "response_a", "response_b"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(
                        f"Unlabeled row {line_number} requires non-empty {key}"
                    )
            sample_id = row["sample_id"]
            if sample_id in seen:
                raise ValueError(f"Duplicate unlabeled sample_id: {sample_id}")
            seen.add(sample_id)
            yield {
                "schema_version": SFT_SCHEMA_VERSION,
                "sample_id": sample_id,
                "prompt": row["prompt"],
                "response": row["response_a"],
            }


def _validate_existing(
    source: Path, output_dir: Path, expected_rows: int
) -> Dict[str, Any]:
    anchor = output_dir / "sft_anchor.jsonl"
    manifest_path = output_dir / "manifest.json"
    if not anchor.is_file() or not manifest_path.is_file():
        raise RuntimeError(
            f"Existing Round2 anchor directory is incomplete: {output_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = _validate_public_source_manifest(source, expected_rows)
    expected_manifest = {
        "schema_version": SFT_SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "source_path": str(source),
        "source_sha256": source_manifest["source_sha256"],
        "source_manifest_path": source_manifest["path"],
        "source_manifest_sha256": source_manifest["sha256"],
        "source_position_randomization_ratio": source_manifest[
            "position_randomization_ratio"
        ],
        "output_path": str(anchor),
        "output_sha256": _file_sha256(anchor),
        "rows": int(expected_rows),
        "status": "succeeded",
    }
    if manifest != expected_manifest:
        raise RuntimeError(
            "Existing Round2 anchor manifest differs from the frozen derivation contract"
        )
    evidence = validate_sft_corpus(anchor, source, expected_rows)
    return {**manifest, "validation": evidence, "reused": True}


def prepare_sft_anchor(
    source: str | Path, output_dir: str | Path, expected_rows: int = 24000
) -> Tuple[Path, Dict[str, Any]]:
    source_path = Path(source).resolve()
    destination = Path(output_dir).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Frozen public unlabeled source is missing: {source_path}")
    source_manifest = _validate_public_source_manifest(source_path, expected_rows)
    if destination.exists():
        evidence = _validate_existing(source_path, destination, expected_rows)
        return destination / "sft_anchor.jsonl", evidence

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    partial.mkdir()
    anchor_partial = partial / "sft_anchor.jsonl"
    count = 0
    with anchor_partial.open("x", encoding="utf-8") as handle:
        for row in _iter_anchor_rows(source_path):
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    if count != int(expected_rows):
        raise ValueError(
            f"Round2 anchor row count mismatch: actual={count}, expected={expected_rows}"
        )

    anchor_final = destination / "sft_anchor.jsonl"
    manifest = {
        "schema_version": SFT_SCHEMA_VERSION,
        "selection_rule": SELECTION_RULE,
        "source_path": str(source_path),
        "source_sha256": source_manifest["source_sha256"],
        "source_manifest_path": source_manifest["path"],
        "source_manifest_sha256": source_manifest["sha256"],
        "source_position_randomization_ratio": source_manifest[
            "position_randomization_ratio"
        ],
        "output_path": str(anchor_final),
        "output_sha256": _file_sha256(anchor_partial),
        "rows": count,
        "status": "succeeded",
    }
    (partial / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(destination)
    evidence = validate_sft_corpus(anchor_final, source_path, expected_rows)
    return anchor_final, {**manifest, "validation": evidence, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=24000)
    args = parser.parse_args()
    anchor, evidence = prepare_sft_anchor(
        args.unlabeled, args.output_dir, args.expected_rows
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print(f"Round2 SFT anchor ready: {anchor}")


if __name__ == "__main__":
    main()
