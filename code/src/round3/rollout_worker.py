"""Persistent single-GPU vLLM worker for one Round3 rollout replica."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from .config import DYNAMIC_METHODS, load_round3_config, validate_round3_config
from .queue_protocol import (
    RESPONSE_SCHEMA,
    atomic_write_json,
    canonical_json,
    file_sha256,
    read_json,
    validate_request,
)


def _validate_publication(config: Dict[str, Any], request: Dict[str, Any]) -> Path:
    checkpoint = Path(request["policy_checkpoint"]).resolve()
    for name in ("adapter_config.json", "adapter_model.safetensors", "checkpoint_meta.json", "READY.json"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Incomplete Round3 policy publication ({name}): {checkpoint}")
    ready = read_json(checkpoint / "READY.json")
    expected = {
        "method_id": request["method_id"],
        "optimizer_step": request["optimizer_step"],
        "adapter_sha256": request["adapter_sha256"],
    }
    if {key: ready.get(key) for key in expected} != expected:
        raise ValueError("Published Round3 adapter does not match request ACK tuple")
    if ready.get("adapter_sha256") != file_sha256(checkpoint / "adapter_model.safetensors"):
        raise ValueError("Published Round3 adapter SHA-256 is invalid")
    if Path(ready.get("base_model", "")).resolve() != Path(config["model"]["name_or_path"]).resolve():
        raise ValueError("Published Round3 adapter/base model mismatch")
    if Path(ready.get("model_manifest", "")).resolve() != Path(config["model"]["manifest_path"]).resolve():
        raise ValueError("Published Round3 adapter/model manifest mismatch")
    expected_config_hash = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if ready.get("config_sha256") != expected_config_hash:
        raise ValueError("Published Round3 adapter/resolved config mismatch")
    return checkpoint


def _generation_contract(config: Dict[str, Any]) -> Dict[str, Any]:
    rollout = config["rollout"]
    return {
        "enable_thinking": False,
        "do_sample": True,
        "temperature": float(rollout["temperature"]),
        "top_p": float(rollout["top_p"]),
        "top_k": int(rollout["top_k"]),
        "min_p": float(rollout["min_p"]),
        "repetition_penalty": float(rollout["repetition_penalty"]),
        "presence_penalty": float(rollout["presence_penalty"]),
        "max_prompt_tokens": 1024,
        "max_new_tokens": 1024,
        "max_model_len": 2048,
        "eos_token_id": list(rollout["eos_token_id"]),
        "pad_token_id": int(rollout["pad_token_id"]),
    }


def process_request(llm, tokenizer, config: Dict[str, Any], request: Dict[str, Any], replica_id: int) -> Dict[str, Any]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    validate_request(request)
    if request["method_id"] != config["method"]["name"]:
        raise ValueError("Round3 rollout worker received another method's request")
    if request.get("generation") != _generation_contract(config):
        raise ValueError("Round3 rollout request changed the generation contract")
    checkpoint = _validate_publication(config, request)
    jobs = [job for job in request["jobs"] if int(job["replica_id"]) == int(replica_id)]
    prompts: List[str] = []
    sampling: List[Any] = []
    raw_prompt_tokens: List[int] = []
    for job in jobs:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": job["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        raw_prompt_tokens.append(raw_count)
        prompts.append(prompt)
        sampling.append(
            SamplingParams(
                n=1,
                temperature=float(config["rollout"]["temperature"]),
                top_p=float(config["rollout"]["top_p"]),
                top_k=int(config["rollout"]["top_k"]),
                min_p=float(config["rollout"]["min_p"]),
                repetition_penalty=float(config["rollout"]["repetition_penalty"]),
                presence_penalty=float(config["rollout"]["presence_penalty"]),
                max_tokens=1024,
                seed=int(job["seed"]),
                truncate_prompt_tokens=1024,
                stop_token_ids=list(config["rollout"]["eos_token_id"]),
            )
        )
    lora_id = int(request["optimizer_step"]) + 1
    lora_request = LoRARequest(
        lora_name=f"round3-{request['method_id']}-step-{request['optimizer_step']:06d}",
        lora_int_id=lora_id,
        lora_path=str(checkpoint),
    )
    started = time.monotonic()
    generated = (
        llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=False)
        if jobs
        else []
    )
    elapsed = time.monotonic() - started
    if len(generated) != len(jobs):
        raise ValueError("vLLM returned an incomplete Round3 replica population")
    outputs = []
    for job, item in zip(jobs, generated):
        if len(item.outputs) != 1:
            raise ValueError("Every Round3 generation job must return exactly one text")
        effective_prompt_tokens = len(item.prompt_token_ids)
        expected_prompt_tokens = min(raw_prompt_tokens[len(outputs)], 1024)
        if effective_prompt_tokens != expected_prompt_tokens:
            raise ValueError(
                "vLLM effective prompt length differs from the Round3 truncation contract"
            )
        candidate = item.outputs[0]
        raw_reason = str(candidate.finish_reason)
        if raw_reason not in {"stop", "length"}:
            raise ValueError(f"Unexpected vLLM finish reason: {raw_reason}")
        reason = raw_reason
        outputs.append(
            {
                "sample_id": job["sample_id"],
                "draw_index": int(job["draw_index"]),
                "text": candidate.text,
                "token_count": len(candidate.token_ids),
                "finish_reason": reason,
                "raw_prompt_tokens": raw_prompt_tokens[len(outputs)],
                "effective_prompt_tokens": effective_prompt_tokens,
                "prompt_truncated_tokens": max(0, raw_prompt_tokens[len(outputs)] - 1024),
            }
        )
    if jobs:
        removed = llm.llm_engine.remove_lora(lora_id)
        if removed is False:
            raise RuntimeError("vLLM did not unload the current Round3 LoRA adapter")
    return {
        "schema_version": RESPONSE_SCHEMA,
        "replica_id": int(replica_id),
        "ack": {
            "method_id": request["method_id"],
            "optimizer_step": request["optimizer_step"],
            "adapter_sha256": request["adapter_sha256"],
        },
        "request_sha256": hashlib.sha256(canonical_json(request).encode("utf-8")).hexdigest(),
        "generation_seconds": elapsed,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--replica-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    if config["method"]["name"] not in DYNAMIC_METHODS:
        raise ValueError("Round3 rollout worker requires a dynamic method config")
    expected_visible = str(config["rollout"]["gpu_ids"][args.replica_id])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError(
            f"Replica {args.replica_id} CUDA_VISIBLE_DEVICES mismatch: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')} != {expected_visible}"
        )

    from transformers import AutoTokenizer
    from vllm import LLM

    model_path = str(config["model"]["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=False,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=int(config["model"]["lora"]["r"]),
        max_model_len=2048,
        max_num_seqs=int(config["rollout"]["max_num_seqs"]),
        gpu_memory_utilization=float(config["rollout"]["gpu_memory_utilization"]),
        enforce_eager=bool(config["rollout"]["enforce_eager"]),
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    artifact_dir = Path(config["rollout"]["artifact_dir"]).resolve()
    request_dir = artifact_dir / "requests"
    response_dir = artifact_dir / "responses" / f"replica_{args.replica_id}"
    request_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        artifact_dir / f"replica_{args.replica_id}.ready.json",
        {
            "state": "ready",
            "replica_id": args.replica_id,
            "cuda_visible_devices": expected_visible,
            "method_id": config["method"]["name"],
            "git_commit": config["provenance"]["git_commit"],
            "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        },
    )
    processed = 0
    while not (artifact_dir / "STOP").exists():
        pending = [
            path
            for path in sorted(request_dir.glob("step_*.json"))
            if not (response_dir / path.name).exists()
        ]
        if not pending:
            if args.once and processed:
                break
            time.sleep(float(config["rollout"]["poll_interval_seconds"]))
            continue
        for request_path in pending:
            try:
                response = process_request(llm, tokenizer, config, read_json(request_path), args.replica_id)
                atomic_write_json(response_dir / request_path.name, response)
            except Exception as exc:
                atomic_write_json(
                    artifact_dir / f"replica_{args.replica_id}.failed.json",
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
