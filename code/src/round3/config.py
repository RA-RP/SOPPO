"""Fail-closed configuration contract for Round3."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable

from ..config import apply_overrides, load_config


METHODS = {
    "dpo_1k",
    "sspo_code_loss_stratified_ultrachat_2df9e9a",
    "dpo_8k",
    "dpo_pe_sft_rollout",
    "dpo_pe_rollout_only",
}
DYNAMIC_METHODS = {"dpo_pe_sft_rollout", "dpo_pe_rollout_only"}
DPO_METHODS = {"dpo_1k", "dpo_8k"}
SSPO_METHOD = "sspo_code_loss_stratified_ultrachat_2df9e9a"


def load_round3_config(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    return apply_overrides(load_config(path), overrides)


def _absolute(value: Any, key: str) -> Path:
    path = Path(str(value or ""))
    if not path.is_absolute():
        raise ValueError(f"{key} must be an absolute server-local path")
    return path


def _full_sha(value: Any, key: str) -> str:
    text = str(value or "")
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{key} must be a full lowercase 40-character Git SHA")
    return text


def validate_round3_config(config: Dict[str, Any]) -> None:
    for section in (
        "contract",
        "provenance",
        "execution",
        "model",
        "data",
        "training",
        "method",
        "rollout",
        "evaluation",
        "output",
        "storage",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Round3 config missing mapping: {section}")
    if config["contract"].get("theory") != "r3-theory-v0.9":
        raise ValueError("Wrong Round3 theory contract")
    if config["contract"].get("experiment") != "round3-exp-v1.4":
        raise ValueError("Wrong Round3 experiment contract")
    _full_sha(config["provenance"].get("git_commit"), "provenance.git_commit")
    experiment_id = config["provenance"].get("experiment_id")
    if (
        not isinstance(experiment_id, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id)
    ):
        raise ValueError(
            "Round3 experiment_id must be an explicit path-safe identifier at server resolve time"
        )
    execution_mode = config["execution"].get("mode")
    if execution_mode not in {"formal", "strong_smoke"}:
        raise ValueError("Round3 execution.mode must be formal or strong_smoke")
    if execution_mode == "formal" and config["execution"].get("smoke_max_steps") is not None:
        raise ValueError("Formal Round3 config must not cap optimizer steps")
    if execution_mode == "strong_smoke" and int(config["execution"].get("smoke_max_steps") or 0) != 1:
        raise ValueError("Round3 strong smoke must execute exactly one optimizer step")

    model = config["model"]
    if model.get("repo_id") != "Qwen/Qwen3-1.7B":
        raise ValueError("Round3 requires Qwen/Qwen3-1.7B, not the Base variant")
    _absolute(model.get("name_or_path"), "model.name_or_path")
    _absolute(model.get("manifest_path"), "model.manifest_path")
    if model.get("torch_dtype") != "bfloat16" or model.get("attention_implementation") != "sdpa":
        raise ValueError("Round3 requires BF16 mixed precision with SDPA")
    if bool(model.get("quantized")) or bool(model.get("enable_thinking")):
        raise ValueError("Round3 requires non-quantized native non-thinking Qwen3")
    if not bool(model.get("gradient_checkpointing")) or bool(model.get("use_cache")):
        raise ValueError("Round3 requires gradient checkpointing and use_cache=false")
    lengths = (model.get("max_length"), model.get("max_prompt_length"), model.get("max_completion_length"))
    if tuple(int(value or 0) for value in lengths) != (2048, 1024, 1024):
        raise ValueError("Round3 sequence caps must be 2048/1024/1024")
    lora = model.get("lora", {})
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    if (
        int(lora.get("r", 0)) != 8
        or int(lora.get("alpha", 0)) != 16
        or float(lora.get("dropout", -1)) != 0.0
        or lora.get("bias") != "none"
        or lora.get("task_type") != "CAUSAL_LM"
        or set(lora.get("target_modules", [])) != expected_targets
    ):
        raise ValueError("Round3 LoRA contract mismatch")
    _full_sha(model.get("resolved_revision"), "model.resolved_revision")

    data = config["data"]
    _absolute(data.get("data_dir"), "data.data_dir")
    _absolute(data.get("reference_cache_dir"), "data.reference_cache_dir")
    _full_sha(data.get("ultrafeedback_revision"), "data.ultrafeedback_revision")
    _full_sha(data.get("ultrachat_revision"), "data.ultrachat_revision")
    if data.get("ultrafeedback_repo") != "HuggingFaceH4/ultrafeedback_binarized":
        raise ValueError("Wrong Round3 paired dataset")
    if data.get("ultrachat_repo") != "HuggingFaceH4/ultrachat_200k":
        raise ValueError("Wrong Round3 unpaired dataset")
    counts = {
        "paired_master": 8000,
        "paired_limited": 1000,
        "unpaired_train": 7000,
        "validation": 1000,
        "test": 997,
    }
    for key, expected in counts.items():
        if int(data.get(key, -1)) != expected:
            raise ValueError(f"Round3 requires data.{key}={expected}")
    if int(data.get("seed", -1)) != 42:
        raise ValueError("Round3 data seed must be 42")

    training = config["training"]
    exact = {
        "epochs": 1,
        "optimizer_steps": 250,
        "save_steps": 25,
        "eval_steps": 25,
        "seed": 42,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "cosine",
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"Round3 requires training.{key}={expected}")
    floats = {
        "learning_rate": 1e-5,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "dpo_beta": 0.1,
        "pe_beta": 10.0,
        "pe_epsilon": 1e-8,
    }
    for key, expected in floats.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"Round3 requires training.{key}={expected}")
    if int(training.get("train_gpu", -1)) != 0:
        raise ValueError("All Round3 training must use physical GPU0")
    physical = int(training.get("physical_pair_subbatch", 0))
    if physical < 1:
        raise ValueError("physical_pair_subbatch is resolved by strong smoke and must be positive")

    method = config["method"]
    name = method.get("name")
    if name not in METHODS:
        raise ValueError(f"Unsupported Round3 method: {name}")
    required_batch = {
        "dpo_1k": (4, 0),
        SSPO_METHOD: (4, 28),
        "dpo_8k": (32, 0),
        "dpo_pe_sft_rollout": (4, 28),
        "dpo_pe_rollout_only": (4, 28),
    }[name]
    actual_batch = (int(method.get("labeled_pairs", -1)), int(method.get("unpaired_units", -1)))
    if actual_batch != required_batch:
        raise ValueError(f"Round3 {name} requires logical batch {required_batch}")
    if name == SSPO_METHOD:
        sspo = method.get("sspo", {})
        required = {
            "source_commit": "2df9e9a1d5fb9202a583cb66eb081e0cb60e873d",
            "beta": 10.0,
            "margin": 2.0,
            "prior": 0.5,
            "reward_norm_momentum": 0.95,
            "reward_clip_range": 5.0,
            "running_state_initialization": "none",
            "threshold": "min_normalized_chosen",
            "gamma_min": 0.125,
            "gamma_decay": 0.001,
        }
        for key, expected in required.items():
            if sspo.get(key) != expected:
                raise ValueError(f"Round3 SSPO requires method.sspo.{key}={expected}")
        forbidden = {"kde", "threshold_ema", "joint_reward_statistics"} & set(sspo)
        if forbidden:
            raise ValueError(f"Paper-v3 SSPO fields are forbidden: {sorted(forbidden)}")
    if name in DYNAMIC_METHODS and float(method.get("lambda_pe", -1)) != 0.1:
        raise ValueError("Round3 dynamic methods require lambda_pe=0.1")
    if name not in DYNAMIC_METHODS and method.get("lambda_pe") is not None:
        raise ValueError("Only dynamic PE methods may define lambda_pe")

    rollout = config["rollout"]
    if name in DYNAMIC_METHODS:
        if rollout.get("enabled") is not True:
            raise ValueError("Round3 dynamic methods must enable rollout")
        _absolute(rollout.get("artifact_dir"), "rollout.artifact_dir")
        if rollout.get("engine") != "vllm" or rollout.get("gpu_ids") != [1, 2]:
            raise ValueError("Round3 dynamic rollout requires vLLM replicas on GPU1/2")
        sampling = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "presence_penalty": 0.0,
        }
        for key, expected in sampling.items():
            if rollout.get(key) != expected:
                raise ValueError(f"Round3 rollout requires {key}={expected}")
        if int(rollout.get("replicas", 0)) != 2 or int(rollout.get("max_model_len", 0)) != 2048:
            raise ValueError("Round3 rollout requires two independent 2048-context replicas")
        if int(rollout.get("max_num_seqs", 0)) < 1:
            raise ValueError("Round3 rollout.max_num_seqs must be positive")
        utilization = float(rollout.get("gpu_memory_utilization", 0.0))
        if not 0.0 < utilization < 1.0:
            raise ValueError("Round3 rollout.gpu_memory_utilization must be between zero and one")
        if float(rollout.get("poll_interval_seconds", 0.0)) <= 0.0:
            raise ValueError("Round3 rollout.poll_interval_seconds must be positive")
        if float(rollout.get("request_timeout_seconds", 0.0)) <= 0.0:
            raise ValueError("Round3 rollout.request_timeout_seconds must be positive")
        if rollout.get("eos_token_id") != [151645, 151643] or int(rollout.get("pad_token_id", -1)) != 151643:
            raise ValueError("Round3 rollout special-token IDs differ from the approved contract")
    elif bool(rollout.get("enabled")):
        raise ValueError("Static Round3 methods must not start rollout replicas")

    evaluation = config["evaluation"]
    if (
        int(evaluation.get("selection_pairs", 0)) != 1000
        or int(evaluation.get("selection_batch_size", 0)) != 4
        or float(evaluation.get("selection_dpo_beta", -1)) != 0.1
        or int(evaluation.get("test_pairs", 0)) != 997
        or int(evaluation.get("ece_bins", 0)) != 15
    ):
        raise ValueError("Round3 selection/final-test contract mismatch")
    if set(evaluation.get("test_heads", [])) != {
        "dpo_reference_delta_beta_0.1",
        "raw_mean_logp_delta_beta_10",
    }:
        raise ValueError("Round3 requires both pre-registered test score heads")
    if evaluation.get("alpacaeval") != "deferred_round4" or evaluation.get("mt_bench") != "deferred_round4":
        raise ValueError("Round3 must defer AlpacaEval and MT-Bench")
    if evaluation.get("pe_static") != "deferred_round5":
        raise ValueError("Round3 must not implement PE-static")
    _absolute(config["output"].get("run_dir"), "output.run_dir")
    if config["output"].get("retain_all_durable_checkpoints") is not True:
        raise ValueError("Round3 must retain all durable checkpoints")
    if config["output"].get("automatic_pruner") is not False:
        raise ValueError("Round3 forbids an automatic checkpoint pruner")
    storage = config["storage"]
    if float(storage.get("require_free_multiplier", 0)) != 2.0:
        raise ValueError("Round3 storage multiplier must be 2.0")
    projected = storage.get("projected_peak_bytes")
    if execution_mode == "formal" and int(projected or 0) <= 0:
        raise ValueError("Formal Round3 config requires strong-smoke projected_peak_bytes")
    if execution_mode == "strong_smoke" and projected is not None:
        raise ValueError("Strong smoke measures projected peak; it must not predeclare it")
