"""Versioned, sample-safe rollout artifact validation for round2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable


ROLLOUT_SCHEMA_VERSION = "round2.rollout.v1"

_REQUIRED = {
    "schema_version",
    "sample_id",
    "source_split",
    "policy_checkpoint",
    "policy_source",
    "generation",
    "responses",
}


def _sha256_payload(record: Dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_rollout_record(record: Dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("Rollout record must be a JSON object")
    missing = _REQUIRED - set(record)
    if missing:
        raise ValueError(f"Rollout record is missing fields: {sorted(missing)}")
    if record["schema_version"] != ROLLOUT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported rollout schema: {record['schema_version']}")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValueError("Rollout sample_id must be a non-empty string")
    if record["source_split"] != "unlabeled_train":
        raise ValueError("Round2 rollout may only consume unlabeled_train")
    if record["policy_source"] not in {"sft_rollout", "rollout_only"}:
        raise ValueError("Unknown round2 rollout policy_source")
    if not isinstance(record["policy_checkpoint"], str) or not record["policy_checkpoint"]:
        raise ValueError("Rollout policy_checkpoint must be recorded")
    if not isinstance(record["generation"], dict):
        raise ValueError("Rollout generation must be a JSON object")
    responses = record["responses"]
    if not isinstance(responses, list) or not responses:
        raise ValueError("Rollout responses must be a non-empty list")
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("Each rollout response must be an object")
        if not isinstance(response.get("response_id"), str) or not response["response_id"]:
            raise ValueError("Each response requires response_id")
        if not isinstance(response.get("text"), str):
            raise ValueError("Each response requires text")


def validate_rollout_jsonl(path: str | Path) -> Dict[str, Any]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Rollout artifact is missing: {input_path}")
    count = 0
    sample_ids = set()
    digest = hashlib.sha256()
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid rollout JSON at line {line_number}") from exc
            validate_rollout_record(record)
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate rollout sample_id: {sample_id}")
            sample_ids.add(sample_id)
            digest.update(line.encode("utf-8"))
            count += 1
    if count == 0:
        raise ValueError(f"Rollout artifact is empty: {input_path}")
    return {"path": str(input_path.resolve()), "records": count, "sha256": digest.hexdigest()}


def write_rollout_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> Dict[str, Any]:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite rollout artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    sample_ids = set()
    digest = hashlib.sha256()
    with output.open("x", encoding="utf-8") as handle:
        for record in records:
            validate_rollout_record(record)
            sample_id = record["sample_id"]
            if sample_id in sample_ids:
                raise ValueError(f"Duplicate rollout sample_id: {sample_id}")
            sample_ids.add(sample_id)
            line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    if count == 0:
        raise ValueError("Refuse to write an empty rollout artifact")
    return {"path": str(output.resolve()), "records": count, "sha256": digest.hexdigest()}
