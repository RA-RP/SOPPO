"""Persistent one-GPU vLLM worker for current-policy round2 candidates."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from ..config import canonical_json
from .config import load_round2_config, validate_round2_config
from .queue_protocol import (
    RESPONSE_SCHEMA_VERSION,
    atomic_write_json,
    file_sha256,
    read_json,
    validate_request,
)
from .rollout_backend import validate_rollout_runtime


def _position_is_reversed(seed: int, step: int, sample_id: str) -> bool:
    payload = f"{seed}:{step}:{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).digest()[0] & 1 == 1


def _format_prompts(tokenizer, samples: List[Dict[str, str]]) -> List[str]:
    prompts = []
    for sample in samples:
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": sample["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return prompts


def _build_pairs(
    request: Dict[str, Any], generated: List[List[str]], seed: int
) -> List[Dict[str, Any]]:
    pairs = []
    method = request["method"]
    for sample, candidates in zip(request["samples"], generated):
        if method == "soppo_pe_sft_rollout_exp":
            if len(candidates) != 1:
                raise ValueError("SFT+rollout requires exactly one generated response")
            left = (sample["sft_response"], "sft")
            right = (candidates[0], "rollout")
        else:
            if len(candidates) != 2:
                raise ValueError("Rollout-only requires exactly two generated responses")
            left = (candidates[0], "rollout_0")
            right = (candidates[1], "rollout_1")
        reversed_position = _position_is_reversed(
            seed, int(request["step"]), sample["sample_id"]
        )
        response_a, response_b = (right, left) if reversed_position else (left, right)
        if not response_a[0] or not response_b[0]:
            raise ValueError(f"Empty dynamic response for sample {sample['sample_id']}")
        pairs.append(
            {
                "sample_id": sample["sample_id"],
                "prompt": sample["prompt"],
                "response_a": response_a[0],
                "response_b": response_b[0],
                "response_a_source": response_a[1],
                "response_b_source": response_b[1],
                "position_randomized": True,
            }
        )
    return pairs


def _process_request(llm, tokenizer, config: Dict[str, Any], request: Dict[str, Any]):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    validate_request(request)
    method = config["method"]["name"]
    if request["method"] != method:
        raise ValueError("Worker received a request for a different method")
    checkpoint = Path(request["policy_checkpoint"]).resolve()
    for required in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint_meta.json",
        "READY.json",
        "run_config.yaml",
    ):
        if not (checkpoint / required).is_file():
            raise FileNotFoundError(
                f"Published TP adapter is incomplete ({required}): {checkpoint}"
            )
    ready = read_json(checkpoint / "READY.json")
    expected_config_sha256 = hashlib.sha256(
        canonical_json(config).encode("utf-8")
    ).hexdigest()
    if int(ready.get("step", -1)) != int(request["step"]):
        raise ValueError("Published adapter step differs from rollout request")
    if ready.get("config_sha256") != expected_config_sha256:
        raise ValueError("Published adapter was produced by another resolved config")
    if ready.get("adapter_sha256") != file_sha256(
        checkpoint / "adapter_model.safetensors"
    ):
        raise ValueError("Published adapter checksum mismatch")
    if Path(ready.get("base_model", "")).resolve() != Path(
        config["model"]["name_or_path"]
    ).resolve():
        raise ValueError("Published adapter/base model mismatch")
    if Path(ready.get("model_manifest", "")).resolve() != Path(
        config["model"]["manifest_path"]
    ).resolve():
        raise ValueError("Published adapter/model manifest mismatch")
    rollout = config["rollout"]
    requested_generation = request.get("generation")
    configured_generation = {
        "temperature": float(rollout["temperature"]),
        "top_p": float(rollout["top_p"]),
        "top_k": int(rollout["top_k"]),
        "min_p": float(rollout["min_p"]),
        "max_new_tokens": int(rollout["max_new_tokens"]),
        "min_new_tokens": int(rollout["min_new_tokens"]),
        "max_model_len": int(rollout["max_model_len"]),
    }
    if requested_generation != configured_generation:
        raise ValueError("Rollout request generation settings differ from resolved config")
    n = 1 if method == "soppo_pe_sft_rollout_exp" else 2
    sampling = SamplingParams(
        n=n,
        temperature=configured_generation["temperature"],
        top_p=configured_generation["top_p"],
        top_k=configured_generation["top_k"],
        min_p=configured_generation["min_p"],
        max_tokens=configured_generation["max_new_tokens"],
        min_tokens=configured_generation["min_new_tokens"],
        seed=int(config["training"]["seed"]) + int(request["step"]),
        truncate_prompt_tokens=(
            configured_generation["max_model_len"]
            - configured_generation["max_new_tokens"]
            - 1  # reserve the EOS appended by response-only training tokenization
        ),
    )
    lora_id = int(request["step"]) + 1
    lora_request = LoRARequest(
        lora_name=f"round2-step-{request['step']:06d}",
        lora_int_id=lora_id,
        lora_path=str(checkpoint),
    )
    prompts = _format_prompts(tokenizer, request["samples"])
    started = time.monotonic()
    outputs = llm.generate(
        prompts,
        sampling,
        lora_request=lora_request,
        use_tqdm=False,
    )
    elapsed = time.monotonic() - started
    if len(outputs) != len(request["samples"]):
        raise ValueError("vLLM returned an incomplete rollout population")
    generated = [[candidate.text for candidate in item.outputs] for item in outputs]
    generated_token_counts = [
        len(candidate.token_ids) for item in outputs for candidate in item.outputs
    ]
    raw_prompt_token_counts = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    prompt_limit = (
        configured_generation["max_model_len"]
        - configured_generation["max_new_tokens"]
        - 1
    )
    pairs = _build_pairs(request, generated, int(config["training"]["seed"]))
    duplicate_pair_count = sum(
        int(pair["response_a"] == pair["response_b"]) for pair in pairs
    )
    if method == "soppo_pe_rollout_only_exp" and duplicate_pair_count == len(pairs):
        raise RuntimeError(
            "All rollout-only candidate pairs are identical; PE would degenerate"
        )
    removed = llm.llm_engine.remove_lora(lora_id)
    if removed is False:
        raise RuntimeError(f"vLLM did not unload adapter id {lora_id}")
    return {
        "schema_version": RESPONSE_SCHEMA_VERSION,
        "step": request["step"],
        "method": method,
        "policy_checkpoint": str(checkpoint),
        "generation": configured_generation,
        "generation_seconds": elapsed,
        "statistics": {
            "requested_prompts": len(prompts),
            "generated_sequences": len(generated_token_counts),
            "min_generated_tokens": min(generated_token_counts),
            "max_generated_tokens": max(generated_token_counts),
            "max_raw_prompt_tokens": max(raw_prompt_token_counts),
            "max_effective_prompt_tokens": max(
                min(value, prompt_limit) for value in raw_prompt_token_counts
            ),
            "duplicate_pair_count": duplicate_pair_count,
        },
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_round2_config(args.config)
    validate_round2_config(config)
    versions = validate_rollout_runtime(config)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected = str(config["rollout"]["gpu_ids"])
    if visible != expected:
        raise RuntimeError(
            f"Rollout CUDA_VISIBLE_DEVICES mismatch: actual={visible}, expected={expected}"
        )

    from transformers import AutoTokenizer
    from vllm import LLM

    rollout = config["rollout"]
    model_path = str(config["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=int(config["model"]["lora"]["r"]),
        max_model_len=int(rollout["max_model_len"]),
        max_num_seqs=int(rollout["max_num_seqs"]),
        gpu_memory_utilization=float(rollout["gpu_memory_utilization"]),
        enforce_eager=bool(rollout["enforce_eager"]),
        enable_prefix_caching=False,
        disable_log_stats=True,
    )

    artifact_dir = Path(rollout["artifact_dir"]).resolve()
    request_dir = artifact_dir / "requests"
    response_dir = artifact_dir / "responses"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        artifact_dir / "worker.ready.json",
        {
            "state": "ready",
            "engine": "vllm",
            "git_commit": config["provenance"]["git_commit"],
            "config_sha256": hashlib.sha256(
                canonical_json(config).encode("utf-8")
            ).hexdigest(),
            "versions": versions,
            "cuda_visible_devices": visible,
            "model": model_path,
            "sampling": {
                key: rollout[key]
                for key in ("temperature", "top_p", "top_k", "min_p")
            },
        },
    )

    processed = 0
    stop_path = artifact_dir / "STOP"
    poll_interval = float(rollout["poll_interval_seconds"])
    while not stop_path.exists():
        requests = sorted(request_dir.glob("step_*.json"))
        pending = [path for path in requests if not (response_dir / path.name).exists()]
        if not pending:
            if args.once and processed:
                break
            time.sleep(poll_interval)
            continue
        for request_path in pending:
            try:
                request = read_json(request_path)
                response = _process_request(llm, tokenizer, config, request)
                atomic_write_json(response_dir / request_path.name, response)
            except Exception as exc:
                atomic_write_json(
                    artifact_dir / "worker.failed.json",
                    {
                        "state": "failed",
                        "request": str(request_path),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            processed += 1
            if args.once:
                break


if __name__ == "__main__":
    main()
