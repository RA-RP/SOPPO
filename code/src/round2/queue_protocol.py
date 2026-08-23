"""Atomic filesystem protocol between TP training and the vLLM worker."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict


REQUEST_SCHEMA_VERSION = "round2.rollout_request.v1"
RESPONSE_SCHEMA_VERSION = "round2.rollout_response.v1"


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Refuse to overwrite queue artifact: {output}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def read_json(path: str | Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Queue artifact must contain one JSON object: {path}")
    return value


def wait_for_json(
    path: str | Path,
    timeout: float,
    poll_interval: float,
    failure_path: str | Path | None = None,
) -> Dict[str, Any]:
    target = Path(path).resolve()
    failure = Path(failure_path).resolve() if failure_path is not None else None
    started = time.monotonic()
    while not target.is_file():
        if failure is not None and failure.is_file():
            payload = read_json(failure)
            raise RuntimeError(
                "Rollout worker reported a request failure: "
                f"{payload.get('error_type')}: {payload.get('message')}"
            )
        if time.monotonic() - started > float(timeout):
            raise TimeoutError(f"Timed out waiting for queue artifact: {target}")
        time.sleep(float(poll_interval))
    return read_json(target)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_request(payload: Dict[str, Any]) -> None:
    if payload.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError("Unsupported round2 rollout request schema")
    if not isinstance(payload.get("step"), int) or payload["step"] < 0:
        raise ValueError("Rollout request step must be a non-negative integer")
    if payload.get("method") not in {
        "soppo_pe_sft_rollout_exp",
        "soppo_pe_rollout_only_exp",
    }:
        raise ValueError("Rollout request has an unsupported method")
    checkpoint = Path(str(payload.get("policy_checkpoint", "")))
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise ValueError("Rollout request policy checkpoint is not ready")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != 56:
        raise ValueError("Every formal rollout request must contain exactly 56 samples")
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("Rollout request samples must be objects")
        for key in ("sample_id", "prompt"):
            if not isinstance(sample.get(key), str) or not sample[key]:
                raise ValueError(f"Rollout request sample requires {key}")
        if payload["method"] == "soppo_pe_sft_rollout_exp":
            if not isinstance(sample.get("sft_response"), str) or not sample["sft_response"]:
                raise ValueError("SFT+rollout request requires sft_response")
        elif "sft_response" in sample:
            raise ValueError("Rollout-only request must not carry an SFT response")
        if sample["sample_id"] in seen:
            raise ValueError(f"Duplicate rollout request sample: {sample['sample_id']}")
        seen.add(sample["sample_id"])


def validate_response(payload: Dict[str, Any], request: Dict[str, Any]) -> None:
    if payload.get("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("Unsupported round2 rollout response schema")
    if payload.get("step") != request.get("step"):
        raise ValueError("Rollout response/request step mismatch")
    if payload.get("method") != request.get("method"):
        raise ValueError("Rollout response/request method mismatch")
    if payload.get("policy_checkpoint") != request.get("policy_checkpoint"):
        raise ValueError("Rollout response used the wrong policy checkpoint")
    if payload.get("generation") != request.get("generation"):
        raise ValueError("Rollout response generation settings changed in transit")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(request["samples"]):
        raise ValueError("Rollout response population is incomplete")
    expected = [row["sample_id"] for row in request["samples"]]
    actual = []
    for pair, requested in zip(pairs, request["samples"]):
        if not isinstance(pair, dict):
            raise ValueError("Rollout response pairs must be objects")
        for key in ("sample_id", "prompt", "response_a", "response_b"):
            if not isinstance(pair.get(key), str) or not pair[key]:
                raise ValueError(f"Rollout response pair requires {key}")
        if set(pair) & {"label", "chosen", "rejected", "original_chosen", "original_rejected"}:
            raise ValueError("Rollout response leaked a preference label")
        if pair["prompt"] != requested["prompt"]:
            raise ValueError("Rollout response prompt differs from the request")
        sources = {pair.get("response_a_source"), pair.get("response_b_source")}
        if request["method"] == "soppo_pe_sft_rollout_exp":
            if sources != {"sft", "rollout"}:
                raise ValueError("SFT+rollout response sources are malformed")
            sft_side = (
                "response_a"
                if pair.get("response_a_source") == "sft"
                else "response_b"
            )
            if pair[sft_side] != requested["sft_response"]:
                raise ValueError("SFT response changed during rollout construction")
        elif sources != {"rollout_0", "rollout_1"}:
            raise ValueError("Rollout-only response sources are malformed")
        actual.append(pair["sample_id"])
    if actual != expected:
        raise ValueError("Rollout response order/sample IDs differ from the request")
    statistics = payload.get("statistics")
    if not isinstance(statistics, dict):
        raise ValueError("Rollout response is missing generation statistics")
    expected_sequences = len(request["samples"]) * (
        1 if request["method"] == "soppo_pe_sft_rollout_exp" else 2
    )
    if int(statistics.get("generated_sequences", -1)) != expected_sequences:
        raise ValueError("Rollout response generated-sequence count is incomplete")
