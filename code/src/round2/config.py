"""Round2 configuration loading and fail-closed validation.

Round2 deliberately has its own validation layer. The first-round validator
continues to describe the frozen MVP and is not widened to accept new active
methods by accident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from ..config import apply_overrides, load_config


ROUND2_METHODS = {
    "soppo_pe_sft_rollout_exp",
    "soppo_pe_rollout_only_exp",
}


def load_round2_config(path: str | Path, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    config = load_config(path)
    return apply_overrides(config, overrides)


def _require_absolute(value: Any, name: str) -> Path:
    if not value or not Path(str(value)).is_absolute():
        raise ValueError(f"{name} must be an absolute server-local path")
    return Path(str(value))


def validate_round2_config(config: Dict[str, Any]) -> None:
    for section in (
        "provenance",
        "model",
        "data",
        "training",
        "method",
        "output",
        "tensor_parallel",
        "rollout",
        "evaluation",
    ):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Round2 config missing mapping: {section}")

    method = config["method"].get("name")
    if method not in ROUND2_METHODS:
        raise ValueError(f"Unsupported round2 method: {method}")

    model = config["model"]
    data = config["data"]
    training = config["training"]
    output = config["output"]
    tensor_parallel = config["tensor_parallel"]
    rollout = config["rollout"]
    evaluation = config["evaluation"]
    git_commit = str(config["provenance"].get("git_commit") or "")
    if len(git_commit) != 40 or any(
        value not in "0123456789abcdef" for value in git_commit
    ):
        raise ValueError("Round2 requires a full lowercase Git commit in provenance")

    _require_absolute(model.get("name_or_path"), "model.name_or_path")
    _require_absolute(model.get("manifest_path"), "model.manifest_path")
    _require_absolute(data.get("data_dir"), "data.data_dir")
    _require_absolute(output.get("run_dir"), "output.run_dir")
    _require_absolute(rollout.get("artifact_dir"), "rollout.artifact_dir")
    _require_absolute(rollout.get("sft_data_file"), "rollout.sft_data_file")

    if int(model.get("max_seq_len", 0)) != 2048:
        raise ValueError("Round2 requires max_seq_len=2048")
    if model.get("repo_id") != "Qwen/Qwen3-4B":
        raise ValueError("Round2 requires the frozen Qwen/Qwen3-4B base")
    if model.get("torch_dtype") != "bfloat16":
        raise ValueError("Round2 requires bfloat16")
    if model.get("attention_implementation") != "sdpa":
        raise ValueError("Round2 requires SDPA attention")
    if bool(model.get("gradient_checkpointing")) is not True:
        raise ValueError("Round2 requires gradient checkpointing")
    if bool(model.get("enable_thinking")) or bool(model.get("use_cache")):
        raise ValueError("Round2 requires enable_thinking=false and use_cache=false")

    lora = model.get("lora", {})
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    if int(lora.get("r", 0)) != 8 or int(lora.get("alpha", 0)) != 16:
        raise ValueError("Round2 requires the frozen LoRA r=8/alpha=16 contract")
    if float(lora.get("dropout", -1)) != 0.0 or lora.get("bias") != "none":
        raise ValueError("Round2 requires LoRA dropout=0 and bias=none")
    if lora.get("task_type") != "CAUSAL_LM":
        raise ValueError("Round2 requires LoRA task_type=CAUSAL_LM")
    if set(lora.get("target_modules", [])) != expected_targets:
        raise ValueError("Round2 LoRA target modules do not match the frozen contract")

    data_contract = {
        "total_samples": 30000,
        "labeled_train_samples": 2700,
        "labeled_val_samples": 300,
        "unlabeled_train_samples": 24000,
        "test_samples": 3000,
    }
    for key, expected in data_contract.items():
        if int(data.get(key, -1)) != expected:
            raise ValueError(f"Round2 data contract requires {key}={expected}")
    if data.get("labeled_ratio") != 0.1 or data.get("unlabeled_ratio") != 0.8 or data.get("test_ratio") != 0.1:
        raise ValueError("Round2 split ratios must remain 0.1/0.8/0.1")
    if data.get("dataset_name") != "openbmb/UltraFeedback" or int(
        data.get("seed", -1)
    ) != 42:
        raise ValueError("Round2 requires the frozen UltraFeedback/seed42 data contract")

    if int(training.get("joint_labeled_global_batch_size", 0)) != 8:
        raise ValueError("Round2 requires labeled global batch size 8")
    if int(training.get("joint_unlabeled_global_batch_size", 0)) != 56:
        raise ValueError("Round2 requires unlabeled global batch size 56")
    if int(training.get("global_batch_size", 0)) != 64:
        raise ValueError("Round2 requires logical global batch size 64")
    if int(training.get("num_devices", 0)) != 2:
        raise ValueError("Round2 TP execution requires training.num_devices=2")
    if int(training.get("epochs", 0)) != 2:
        raise ValueError("Round2 requires the frozen two-epoch contract")
    if float(training.get("lr", 0)) != 1e-5 or float(
        training.get("learning_rate", 0)
    ) != 1e-5:
        raise ValueError("Round2 requires the frozen lr=1e-5 contract")
    if int(training.get("backward_subbatch_size_per_device", 0)) != 1:
        raise ValueError("Round2 24GB profile requires physical pair subbatch size 1")
    if int(training.get("gradient_accumulation_steps", 0)) != 64:
        raise ValueError("Round2 TP profile requires 64 physical pair microsteps")
    if any(
        training.get(key) is not None
        for key in (
            "dpo_batch_size_per_device",
            "joint_labeled_batch_size_per_device",
            "joint_labeled_microsteps",
            "joint_unlabeled_microbatch_pattern",
        )
    ):
        raise ValueError("Round2 must not carry first-round DDP microbatch fields")
    if training.get("distributed_backend") != "transformers_tp":
        raise ValueError("Round2 requires the transformers_tp backend")
    optimization_contract = {
        "optimizer": "adamw",
        "lr_scheduler_type": "cosine",
        "seed": 42,
        "save_steps": 1,
        "eval_steps": 40,
    }
    for key, expected in optimization_contract.items():
        if training.get(key) != expected:
            raise ValueError(f"Round2 requires training.{key}={expected}")
    float_contract = {
        "simpo_beta": 10.0,
        "simpo_margin": 2.0,
        "epsilon": 1e-8,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.1,
    }
    for key, expected in float_contract.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"Round2 requires training.{key}={expected}")
    smoke_mode = bool(training.get("smoke_mode"))
    eval_max_samples = training.get("eval_max_samples")
    if smoke_mode:
        if int(training.get("max_steps") or 0) != 1:
            raise ValueError("Round2 strong smoke requires max_steps=1")
        if not 1 <= int(eval_max_samples or 0) <= 8:
            raise ValueError("Round2 strong smoke requires 1-8 validation samples")
        if int(training.get("smoke_objective_step") or 0) != 1:
            raise ValueError("Round2 strong smoke must exercise scheduler step 1")
    elif eval_max_samples is not None:
        raise ValueError("Formal round2 evaluation must use all validation samples")
    elif training.get("smoke_objective_step") is not None:
        raise ValueError("Formal round2 must not override the objective scheduler step")
    elif training.get("max_steps") is not None:
        raise ValueError("Formal round2 must run the frozen two-epoch plan")

    method_contract = {
        "use_unlabeled": True,
        "weighting": "exponential_gamma",
        "gamma0": 1.0,
        "gamma_decay": 0.01,
        "pe_distance": "l1",
        "detach_denominator": False,
        "population_batch_contract": "exact_global_optimizer_batch",
    }
    for key, expected in method_contract.items():
        if config["method"].get(key) != expected:
            raise ValueError(f"Round2 requires method.{key}={expected}")
    expected_gamma_min = 2700 / (2700 + 24000)
    if abs(float(config["method"].get("gamma_min", -1)) - expected_gamma_min) > 1e-10:
        raise ValueError("Round2 requires paper gamma_min=2700/26700")
    if int(tensor_parallel.get("tensor_model_parallel_size", 0)) != 2:
        raise ValueError("Round2 requires tensor_model_parallel_size=2")
    if int(tensor_parallel.get("pipeline_model_parallel_size", 0)) != 1:
        raise ValueError("Round2 requires pipeline_model_parallel_size=1")
    if int(tensor_parallel.get("data_parallel_size", 0)) != 1:
        raise ValueError("Round2 requires data_parallel_size=1")
    if int(tensor_parallel.get("micro_batch_size", 0)) != 1:
        raise ValueError("Round2 requires one physical preference pair per microbatch")
    if tensor_parallel.get("backend") != "transformers-native-tp":
        raise ValueError("Round2 requires the transformers-native-tp backend")
    if bool(tensor_parallel.get("sequence_parallel")):
        raise ValueError("Round2 does not combine sequence parallelism with TP-LoRA")
    if bool(tensor_parallel.get("require_tp_lora")) is not True:
        raise ValueError("Round2 must require TP-aware LoRA")
    expected_versions = {
        "minimum_transformers_version": "5.4.0",
        "minimum_peft_version": "0.19.0",
        "minimum_torch_version": "2.7.0",
    }
    for key, expected in expected_versions.items():
        if str(tensor_parallel.get(key)) != expected:
            raise ValueError(f"Round2 requires {key}={expected}")

    train_gpus = [
        value
        for value in str(tensor_parallel.get("gpu_ids", "")).split(",")
        if value != ""
    ]
    rollout_gpus = [
        value for value in str(rollout.get("gpu_ids", "")).split(",") if value != ""
    ]
    if not all(value.isdigit() for value in train_gpus + rollout_gpus):
        raise ValueError("Round2 GPU IDs must be comma-separated non-negative integers")
    if not train_gpus:
        raise ValueError("tensor_parallel.gpu_ids must not be empty")
    if not rollout_gpus:
        raise ValueError("Rollout gpu_ids must not be empty")
    if set(train_gpus) & set(rollout_gpus):
        raise ValueError("Training and rollout GPU sets must be disjoint")
    if len(train_gpus) != 2 or len(set(train_gpus)) != 2:
        raise ValueError("Round2 requires exactly two distinct training GPUs")
    if len(rollout_gpus) != 1 or len(set(rollout_gpus)) != 1:
        raise ValueError("Round2 requires exactly one rollout GPU")
    if int(rollout.get("tensor_parallel_size", 0)) != 1:
        raise ValueError("The one-GPU rollout worker requires tensor_parallel_size=1")
    if rollout.get("engine") != "vllm":
        raise ValueError("Round2 rollout requires vLLM")
    if str(rollout.get("minimum_vllm_version")) != "0.9.2":
        raise ValueError("Round2 requires minimum_vllm_version=0.9.2")
    if int(rollout.get("max_model_len", 0)) != 2048:
        raise ValueError("Round2 rollout requires max_model_len=2048")
    if int(rollout.get("max_new_tokens", 0)) != 512:
        raise ValueError("Round2 rollout requires max_new_tokens=512")
    min_new_tokens = int(rollout.get("min_new_tokens", -1))
    if not 0 <= min_new_tokens <= int(rollout["max_new_tokens"]):
        raise ValueError("rollout.min_new_tokens must be in [0, max_new_tokens]")
    if smoke_mode and min_new_tokens != int(rollout["max_new_tokens"]):
        raise ValueError("Round2 strong smoke must force 512 generated tokens")
    if not smoke_mode and min_new_tokens != 0:
        raise ValueError("Formal round2 requires min_new_tokens=0")
    if int(rollout.get("max_num_seqs", 0)) < 1:
        raise ValueError("Round2 rollout max_num_seqs must be positive")
    if not 0 < float(rollout.get("gpu_memory_utilization", 0)) < 1:
        raise ValueError("Round2 rollout gpu_memory_utilization must be in (0, 1)")
    if bool(rollout.get("enforce_eager")) is not True:
        raise ValueError("The 24GB rollout profile requires enforce_eager=true")
    if int(rollout.get("sync_interval_steps", 0)) != 1:
        raise ValueError("Round2 current-policy rollout requires sync_interval_steps=1")
    temperature = rollout.get("temperature")
    top_p = rollout.get("top_p")
    top_k = rollout.get("top_k")
    min_p = rollout.get("min_p")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("rollout.temperature must be an explicit numeric value")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        raise ValueError("rollout.top_p must be an explicit numeric value")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("rollout.top_k must be an explicit integer value")
    if isinstance(min_p, bool) or not isinstance(min_p, (int, float)):
        raise ValueError("rollout.min_p must be an explicit numeric value")
    sampling_contract = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
    }
    for key, expected in sampling_contract.items():
        if rollout.get(key) != expected:
            raise ValueError(
                f"Round2 requires the preregistered Qwen3 non-thinking {key}={expected}"
            )
    if float(rollout.get("poll_interval_seconds", 0)) <= 0 or float(
        rollout.get("request_timeout_seconds", 0)
    ) <= 0:
        raise ValueError("Round2 rollout polling and timeout values must be positive")

    if method == "soppo_pe_sft_rollout_exp" and rollout.get("source") != "sft_rollout":
        raise ValueError("soppo_pe_sft_rollout_exp requires rollout.source=sft_rollout")
    if method == "soppo_pe_rollout_only_exp" and rollout.get("source") != "rollout_only":
        raise ValueError("soppo_pe_rollout_only_exp requires rollout.source=rollout_only")

    evaluation_gpus = [
        value for value in str(evaluation.get("gpu_id", "")).split(",") if value
    ]
    if len(evaluation_gpus) != 1 or not evaluation_gpus[0].isdigit():
        raise ValueError("Round2 evaluation requires exactly one GPU")
    if int(evaluation.get("batch_size", 0)) != 1:
        raise ValueError("Round2 24GB evaluation requires batch_size=1")
    if evaluation.get("score_type") != "simpo_mean_logp_delta_margin_free":
        raise ValueError("Round2 evaluation score type changed")

    if bool(output.get("refuse_overwrite", True)) is not True:
        raise ValueError("Round2 requires refuse_overwrite=true")
