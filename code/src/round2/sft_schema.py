"""Fail-closed schema for the single-response SFT corpus used by round2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


SFT_SCHEMA_VERSION = "round2.sft.v1"
FORBIDDEN_FIELDS = {
    "label",
    "chosen",
    "rejected",
    "original_chosen",
    "original_rejected",
    "response_a",
    "response_b",
}
ALLOWED_FIELDS = {"schema_version", "sample_id", "prompt", "response"}


def validate_sft_record(record: Dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("An SFT row must be a JSON object")
    required = ("sample_id", "prompt", "response")
    for key in required:
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"SFT row requires a non-empty string field: {key}")
    unexpected = set(record) - ALLOWED_FIELDS
    if unexpected:
        raise ValueError(f"Round2 SFT row has unregistered fields: {sorted(unexpected)}")
    forbidden = FORBIDDEN_FIELDS & set(record)
    if forbidden:
        raise ValueError(
            "Round2 SFT data must not expose preference labels or pairs: "
            f"{sorted(forbidden)}"
        )
    version = record.get("schema_version", SFT_SCHEMA_VERSION)
    if version != SFT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported SFT schema: {version}")


def load_sft_jsonl(path: str | Path) -> List[Dict[str, str]]:
    input_path = Path(path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Round2 SFT corpus is missing: {input_path}")
    rows: List[Dict[str, str]] = []
    seen = set()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid SFT JSON at line {line_number}") from exc
            validate_sft_record(row)
            if row["sample_id"] in seen:
                raise ValueError(f"Duplicate SFT sample_id: {row['sample_id']}")
            seen.add(row["sample_id"])
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
            )
    if not rows:
        raise ValueError(f"Round2 SFT corpus is empty: {input_path}")
    return rows


def _load_unlabeled_prompts(path: Path) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid unlabeled JSON at line {line_number}: {path}"
                ) from exc
            sample_id = row.get("sample_id")
            prompt = row.get("prompt")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"Malformed unlabeled sample_id at line {line_number}")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(f"Malformed unlabeled prompt at line {line_number}")
            if sample_id in prompts:
                raise ValueError(f"Duplicate unlabeled sample_id: {sample_id}")
            prompts[sample_id] = prompt
    return prompts


def validate_sft_corpus(
    sft_path: str | Path,
    unlabeled_path: str | Path,
    expected_rows: int,
) -> Dict[str, Any]:
    """Require one label-free SFT response for every frozen unlabeled prompt."""
    sft_file = Path(sft_path).resolve()
    unlabeled_file = Path(unlabeled_path).resolve()
    rows = load_sft_jsonl(sft_file)
    prompts = _load_unlabeled_prompts(unlabeled_file)
    if len(rows) != int(expected_rows):
        raise ValueError(
            f"Round2 SFT row count mismatch: actual={len(rows)}, expected={expected_rows}"
        )
    if len(prompts) != int(expected_rows):
        raise ValueError(
            "Frozen unlabeled split count does not match the round2 contract: "
            f"actual={len(prompts)}, expected={expected_rows}"
        )
    sft_ids = {row["sample_id"] for row in rows}
    if sft_ids != set(prompts):
        missing = sorted(set(prompts) - sft_ids)[:5]
        extra = sorted(sft_ids - set(prompts))[:5]
        raise ValueError(
            f"SFT/unlabeled sample-id mismatch: missing={missing}, extra={extra}"
        )
    for row in rows:
        if row["prompt"] != prompts[row["sample_id"]]:
            raise ValueError(f"SFT prompt mismatch for sample_id={row['sample_id']}")
    digest = hashlib.sha256(sft_file.read_bytes()).hexdigest()
    return {
        "path": str(sft_file),
        "rows": len(rows),
        "sha256": digest,
        "schema_version": SFT_SCHEMA_VERSION,
        "matches_unlabeled_split": True,
    }
