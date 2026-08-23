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
    for section in ("model", "data", "training", "method", "output", "megatron", "rollout"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Round2 config missing mapping: {section}")

    method = config["method"].get("name")
    if method not in ROUND2_METHODS:
        raise ValueError(f"Unsupported round2 method: {method}")

    model = config["model"]
    data = config["data"]
    training = config["training"]
    output = config["output"]
    megatron = config["megatron"]
    rollout = config["rollout"]

    _require_absolute(model.get("name_or_path"), "model.name_or_path")
    _require_absolute(model.get("manifest_path"), "model.manifest_path")
    _require_absolute(data.get("data_dir"), "data.data_dir")
    _require_absolute(output.get("run_dir"), "output.run_dir")
    _require_absolute(megatron.get("entrypoint"), "megatron.entrypoint")
    _require_absolute(megatron.get("working_dir"), "megatron.working_dir")
    _require_absolute(rollout.get("artifact_dir"), "rollout.artifact_dir")

    if int(model.get("max_seq_len", 0)) != 2048:
        raise ValueError("Round2 requires max_seq_len=2048")
    if model.get("torch_dtype") != "bfloat16":
        raise ValueError("Round2 requires bfloat16")
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

    if int(training.get("joint_labeled_global_batch_size", 0)) != 8:
        raise ValueError("Round2 requires labeled global batch size 8")
    if int(training.get("joint_unlabeled_global_batch_size", 0)) != 56:
        raise ValueError("Round2 requires unlabeled global batch size 56")
    if int(training.get("backward_subbatch_size_per_device", 0)) != 2:
        raise ValueError("Round2 requires backward subbatch size 2")
    if int(megatron.get("tensor_model_parallel_size", 0)) < 1:
        raise ValueError("Megatron tensor_model_parallel_size must be >= 1")
    if int(megatron.get("pipeline_model_parallel_size", 0)) < 1:
        raise ValueError("Megatron pipeline_model_parallel_size must be >= 1")
    if int(megatron.get("data_parallel_size", 0)) < 1:
        raise ValueError("Megatron data_parallel_size must be >= 1")

    train_gpus = [str(value) for value in str(megatron.get("gpu_ids", "")).split(",") if value != ""]
    rollout_gpus = [str(value) for value in str(rollout.get("gpu_ids", "")).split(",") if value != ""]
    if not train_gpus:
        raise ValueError("Megatron gpu_ids must not be empty")
    if not rollout_gpus:
        raise ValueError("Rollout gpu_ids must not be empty")
    if set(train_gpus) & set(rollout_gpus):
        raise ValueError("Training and rollout GPU sets must be disjoint")

    if method == "soppo_pe_sft_rollout_exp" and rollout.get("source") != "sft_rollout":
        raise ValueError("soppo_pe_sft_rollout_exp requires rollout.source=sft_rollout")
    if method == "soppo_pe_rollout_only_exp" and rollout.get("source") != "rollout_only":
        raise ValueError("soppo_pe_rollout_only_exp requires rollout.source=rollout_only")

    if bool(output.get("refuse_overwrite", True)) is not True:
        raise ValueError("Round2 requires refuse_overwrite=true")
