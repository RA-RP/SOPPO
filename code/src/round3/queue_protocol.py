"""Two-replica atomic rollout protocol for Round3 dynamic PE."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REQUEST_SCHEMA = "round3.rollout_request.v1"
RESPONSE_SCHEMA = "round3.rollout_replica_response.v1"
ROUTING_NAMESPACE = "round3-rollout-replica-v1"
DYNAMIC_METHODS = {"dpo_pe_sft_rollout", "dpo_pe_rollout_only"}


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Refuse to overwrite queue artifact: {output}")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(output)


def read_json(path: str | Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Queue artifact must be a JSON object: {path}")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_sha256(checkpoint: str | Path) -> str:
    path = Path(checkpoint).resolve() / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Published adapter is missing: {path}")
    return file_sha256(path)


def route_replica(method_id: str, optimizer_step: int, sample_id: str, draw_index: int) -> int:
    payload = (
        f"{ROUTING_NAMESPACE}\0{method_id}\0{int(optimizer_step)}\0"
        f"{sample_id}\0{int(draw_index)}"
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()[-1] & 1


def rollout_seed(base_seed: int, optimizer_step: int, sample_id: str, draw_index: int) -> int:
    payload = f"round3-rollout-seed-v1\0{int(base_seed)}\0{int(optimizer_step)}\0{sample_id}\0{int(draw_index)}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def make_request(
    method_id: str,
    optimizer_step: int,
    checkpoint: str | Path,
    samples: Iterable[Dict[str, str]],
    generation: Dict[str, Any],
    base_seed: int = 42,
) -> Dict[str, Any]:
    if method_id not in DYNAMIC_METHODS:
        raise ValueError("Round3 rollout request method is not dynamic PE")
    checkpoint_path = Path(checkpoint).resolve()
    digest = adapter_sha256(checkpoint_path)
    jobs: List[Dict[str, Any]] = []
    seen = set()
    draws = 1 if method_id == "dpo_pe_sft_rollout" else 2
    sample_rows = list(samples)
    if len(sample_rows) != 28:
        raise ValueError("Round3 dynamic request requires exactly 28 source samples")
    for sample in sample_rows:
        sample_id = sample.get("sample_id")
        prompt = sample.get("prompt")
        if not isinstance(sample_id, str) or not sample_id or not isinstance(prompt, str) or not prompt:
            raise ValueError("Rollout source sample requires sample_id and prompt")
        if sample_id in seen:
            raise ValueError(f"Duplicate dynamic source sample: {sample_id}")
        seen.add(sample_id)
        if method_id == "dpo_pe_sft_rollout":
            response = sample.get("response")
            if not isinstance(response, str) or not response:
                raise ValueError("SFT+rollout source requires a fixed response")
        for draw_index in range(draws):
            jobs.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "draw_index": draw_index,
                    "replica_id": route_replica(method_id, optimizer_step, sample_id, draw_index),
                    "seed": rollout_seed(base_seed, optimizer_step, sample_id, draw_index),
                }
            )
    return {
        "schema_version": REQUEST_SCHEMA,
        "method_id": method_id,
        "optimizer_step": int(optimizer_step),
        "policy_checkpoint": str(checkpoint_path),
        "adapter_sha256": digest,
        "generation": generation,
        "jobs": jobs,
        "anchors": (
            {row["sample_id"]: row["response"] for row in sample_rows}
            if method_id == "dpo_pe_sft_rollout"
            else None
        ),
    }


def validate_request(payload: Dict[str, Any]) -> None:
    if payload.get("schema_version") != REQUEST_SCHEMA:
        raise ValueError("Unsupported Round3 rollout request schema")
    method_id = payload.get("method_id")
    if method_id not in DYNAMIC_METHODS:
        raise ValueError("Unsupported Round3 dynamic method")
    step = payload.get("optimizer_step")
    if not isinstance(step, int) or not 0 <= step < 250:
        raise ValueError("Round3 rollout optimizer_step must be in [0,249]")
    checkpoint = Path(str(payload.get("policy_checkpoint", "")))
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise ValueError("Round3 rollout policy checkpoint is not ready")
    digest = payload.get("adapter_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or digest != adapter_sha256(checkpoint):
        raise ValueError("Round3 rollout adapter checksum mismatch")
    jobs = payload.get("jobs")
    expected_jobs = 28 if method_id == "dpo_pe_sft_rollout" else 56
    if not isinstance(jobs, list) or len(jobs) != expected_jobs:
        raise ValueError("Round3 rollout request has an incomplete logical population")
    keys = set()
    sample_ids = set()
    for job in jobs:
        key = (job.get("sample_id"), job.get("draw_index"))
        if not isinstance(key[0], str) or not key[0]:
            raise ValueError("Round3 rollout job sample_id is empty")
        if not isinstance(key[1], int) or key[1] not in {0, 1}:
            raise ValueError("Round3 rollout draw_index must be 0/1")
        if key in keys:
            raise ValueError(f"Duplicate rollout generation job: {key}")
        keys.add(key)
        sample_ids.add(job.get("sample_id"))
        if job.get("replica_id") != route_replica(method_id, step, key[0], key[1]):
            raise ValueError("Round3 rollout job violates stable replica routing")
        if job.get("seed") != rollout_seed(42, step, key[0], key[1]):
            raise ValueError("Round3 rollout job seed mismatch")
        if not isinstance(job.get("prompt"), str) or not job["prompt"]:
            raise ValueError("Round3 rollout job prompt is empty")
    if len(sample_ids) != 28:
        raise ValueError("Round3 rollout request must cover exactly 28 source IDs")
    anchors = payload.get("anchors")
    if method_id == "dpo_pe_sft_rollout":
        if not isinstance(anchors, dict) or set(anchors) != sample_ids or any(not value for value in anchors.values()):
            raise ValueError("SFT+rollout anchors are missing or incomplete")
    elif anchors is not None:
        raise ValueError("Rollout-only request must not carry fixed anchors")


def validate_replica_response(response: Dict[str, Any], request: Dict[str, Any], replica_id: int) -> None:
    if response.get("schema_version") != RESPONSE_SCHEMA:
        raise ValueError("Unsupported Round3 replica response schema")
    expected_ack = {
        "method_id": request["method_id"],
        "optimizer_step": request["optimizer_step"],
        "adapter_sha256": request["adapter_sha256"],
    }
    if response.get("replica_id") != int(replica_id) or response.get("ack") != expected_ack:
        raise ValueError("Round3 rollout replica ACK mismatch")
    expected_request_sha = hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest()
    if response.get("request_sha256") != expected_request_sha:
        raise ValueError("Round3 rollout replica response/request SHA mismatch")
    expected = [
        (job["sample_id"], job["draw_index"])
        for job in request["jobs"]
        if int(job["replica_id"]) == int(replica_id)
    ]
    outputs = response.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Round3 replica response outputs must be a list")
    actual = []
    for output in outputs:
        key = (output.get("sample_id"), output.get("draw_index"))
        actual.append(key)
        if not isinstance(output.get("text"), str) or not output["text"]:
            raise ValueError(f"Round3 rollout produced an empty response: {key}")
        if output.get("finish_reason") not in {"stop", "length"}:
            raise ValueError(f"Unexpected rollout finish reason: {output.get('finish_reason')}")
        token_count = output.get("token_count")
        if not isinstance(token_count, int) or not 1 <= token_count <= 1024:
            raise ValueError(f"Round3 rollout token count is outside [1,1024]: {key}")
        raw_prompt = output.get("raw_prompt_tokens")
        effective_prompt = output.get("effective_prompt_tokens")
        removed_prompt = output.get("prompt_truncated_tokens")
        if (
            not isinstance(raw_prompt, int)
            or not isinstance(effective_prompt, int)
            or not isinstance(removed_prompt, int)
            or effective_prompt != min(raw_prompt, 1024)
            or removed_prompt != max(0, raw_prompt - 1024)
        ):
            raise ValueError(f"Round3 rollout prompt truncation audit mismatch: {key}")
    if actual != expected:
        raise ValueError("Round3 replica response jobs are incomplete or reordered")


def merge_replica_responses(
    request: Dict[str, Any], responses: Iterable[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    response_list = list(responses)
    if len(response_list) != 2:
        raise ValueError("Round3 requires responses from exactly two replicas")
    by_replica = {int(item.get("replica_id", -1)): item for item in response_list}
    if set(by_replica) != {0, 1}:
        raise ValueError("Round3 requires replica IDs 0 and 1")
    generated: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for replica_id in (0, 1):
        response = by_replica[replica_id]
        validate_replica_response(response, request, replica_id)
        for output in response["outputs"]:
            generated[(output["sample_id"], int(output["draw_index"]))] = output
    prompts = {job["sample_id"]: job["prompt"] for job in request["jobs"]}
    pairs = []
    for sample_id in dict.fromkeys(job["sample_id"] for job in request["jobs"]):
        if request["method_id"] == "dpo_pe_sft_rollout":
            candidates = [
                (request["anchors"][sample_id], "sft"),
                (generated[(sample_id, 0)]["text"], "rollout_0"),
            ]
        else:
            candidates = [
                (generated[(sample_id, 0)]["text"], "rollout_0"),
                (generated[(sample_id, 1)]["text"], "rollout_1"),
            ]
        reverse = hashlib.sha256(
            f"round3-dynamic-ab-swap-v1\0{request['method_id']}\0{request['optimizer_step']}\0{sample_id}".encode("utf-8")
        ).digest()[-1] & 1
        left, right = (candidates[1], candidates[0]) if reverse else candidates
        pairs.append(
            {
                "sample_id": sample_id,
                "prompt": prompts[sample_id],
                "response_a": left[0],
                "response_b": right[0],
                "response_a_source": left[1],
                "response_b_source": right[1],
            }
        )
    if len(pairs) != 28:
        raise ValueError("Merged Round3 rollout population is incomplete")
    statistics = {
        "generated_sequences": len(generated),
        "replica_job_counts": {str(replica_id): len(by_replica[replica_id]["outputs"]) for replica_id in (0, 1)},
        "finish_reason_counts": {
            reason: sum(int(output["finish_reason"] == reason) for output in generated.values())
            for reason in ("stop", "length")
        },
        "response_tokens": [int(output["token_count"]) for output in generated.values()],
        "prompt_truncated_count": sum(
            int(output.get("prompt_truncated_tokens", 0) > 0)
            for output in generated.values()
        ),
        "prompt_truncated_tokens": sum(
            int(output.get("prompt_truncated_tokens", 0))
            for output in generated.values()
        ),
        "generation_seconds_by_replica": {
            str(replica_id): float(by_replica[replica_id]["generation_seconds"])
            for replica_id in (0, 1)
        },
    }
    return pairs, statistics


def wait_for_replica_responses(
    artifact_dir: str | Path,
    request: Dict[str, Any],
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> List[Dict[str, Any]]:
    root = Path(artifact_dir).resolve()
    step = int(request["optimizer_step"])
    targets = [root / "responses" / f"replica_{replica_id}" / f"step_{step:06d}.json" for replica_id in (0, 1)]
    started = time.monotonic()
    while not all(path.is_file() for path in targets):
        for replica_id in (0, 1):
            failure = root / f"replica_{replica_id}.failed.json"
            if failure.is_file():
                payload = read_json(failure)
                raise RuntimeError(f"Rollout replica {replica_id} failed: {payload.get('error_type')}: {payload.get('message')}")
        if time.monotonic() - started > float(timeout_seconds):
            raise TimeoutError("Timed out waiting for both Round3 rollout replicas")
        time.sleep(float(poll_interval_seconds))
    return [read_json(path) for path in targets]
