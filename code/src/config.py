"""Recursive YAML config loading, CLI overrides, and fail-closed validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_name = current.pop("_base_", None)
    if base_name is None:
        return current
    return _deep_merge(load_config(config_path.parent / base_name), current)


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid override value: {raw}") from exc


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be dotted.path=value: {item}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = _parse_value(raw)
    return result


def validate_config(config: Dict[str, Any], world_size: int | None = None) -> None:
    for section in ("model", "data", "training", "method", "output"):
        if section not in config or not isinstance(config[section], dict):
            raise ValueError(f"Missing config mapping: {section}")

    method = config["method"].get("name")
    supported = {"dpo10", "dpo100", "sspo_hard_exp", "soppo_pe_exp", "soppo_pe_static"}
    if method not in supported:
        raise ValueError(f"Unsupported method: {method}")

    def require_absolute(value: Any, name: str) -> None:
        if not value or not Path(str(value)).is_absolute():
            raise ValueError(f"{name} must be an absolute server-local path")

    model_path = config["model"].get("name_or_path")
    require_absolute(model_path, "model.name_or_path")
    require_absolute(config["model"].get("manifest_path"), "model.manifest_path")
    require_absolute(config["data"].get("data_dir"), "data.data_dir")
    require_absolute(config["output"].get("run_dir"), "output.run_dir")

    adapter = config["model"].get("lora", {})
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    if (
        int(adapter.get("r", 0)) != 8
        or int(adapter.get("alpha", 0)) != 16
        or float(adapter.get("dropout", -1)) != 0.0
        or set(adapter.get("target_modules", [])) != expected_targets
        or len(adapter.get("target_modules", [])) != len(expected_targets)
        or adapter.get("bias") != "none"
        or adapter.get("task_type") != "CAUSAL_LM"
    ):
        raise ValueError("v0.6 requires the frozen Qwen3 LoRA r=8/alpha=16/dropout=0/all-projection contract")

    expected_use_unlabeled = method.startswith("sspo") or method.startswith("soppo")
    if bool(config["method"].get("use_unlabeled")) != expected_use_unlabeled:
        raise ValueError(f"method.use_unlabeled contradicts method.name={method}")
    if bool(config["model"].get("enable_thinking")) or bool(config["model"].get("use_cache")):
        raise ValueError("v0.6 requires enable_thinking=false and use_cache=false")
    if config["model"].get("attention_implementation") != "sdpa":
        raise ValueError("v0.6 requires SDPA attention")
    if config["data"].get("train_file"):
        require_absolute(config["data"]["train_file"], "data.train_file")
    if config["output"].get("refuse_overwrite") is not True:
        raise ValueError("v0.6 output must refuse overwrite")

    training = config["training"]
    devices = world_size if world_size is not None else int(training["num_devices"])
    expected_dpo = (
        int(training["dpo_batch_size_per_device"])
        * int(training["gradient_accumulation_steps"])
        * devices
    )
    if method.startswith("dpo") and int(training["global_batch_size"]) != expected_dpo:
        raise ValueError(
            "global_batch_size contract violated: "
            f"configured={training['global_batch_size']}, computed={expected_dpo}"
        )
    if method.startswith("dpo"):
        require_absolute(config["data"].get("reference_cache"), "data.reference_cache")

    if method.startswith("sspo") or method.startswith("soppo"):
        pattern = [int(value) for value in training["joint_unlabeled_microbatch_pattern"]]
        labeled_steps = [int(value) for value in training["joint_labeled_microsteps"]]
        accumulation = int(training["gradient_accumulation_steps"])
        if len(pattern) != accumulation or len(set(labeled_steps)) != len(labeled_steps):
            raise ValueError("Joint microbatch pattern must cover one optimizer step without duplicate labeled steps")
        if any(step < 0 or step >= accumulation for step in labeled_steps):
            raise ValueError("joint_labeled_microsteps contains an out-of-range index")
        labeled = len(labeled_steps) * int(training["joint_labeled_batch_size_per_device"]) * devices
        unlabeled = sum(pattern) * devices
        if labeled != int(training["joint_labeled_global_batch_size"]):
            raise ValueError("joint_labeled_global_batch_size contract violated")
        if unlabeled != int(training["joint_unlabeled_global_batch_size"]):
            raise ValueError("joint_unlabeled_global_batch_size contract violated")
        if labeled + unlabeled != int(training["global_batch_size"]):
            raise ValueError("Joint labeled+unlabeled batches must equal global_batch_size")
        if method == "soppo_pe_static":
            value = float(config["method"]["fixed_lambda"])
            if value not in {0.1, 0.3, 0.5, 1.0}:
                raise ValueError("SOPPO-PE-static lambda must be one of {0.1,0.3,0.5,1.0}")
            if config["method"].get("weighting") != "normalized_fixed_lambda":
                raise ValueError("SOPPO-PE-static requires normalized_fixed_lambda weighting")
        elif config["method"].get("weighting") != "exponential_gamma":
            raise ValueError(f"{method} requires the paper exponential gamma scheduler")

    if not bool(training.get("smoke_mode", False)):
        common_actual = {
            "torch_dtype": config["model"]["torch_dtype"],
            "attention_implementation": config["model"]["attention_implementation"],
            "max_seq_len": int(config["model"]["max_seq_len"]),
            "enable_thinking": bool(config["model"]["enable_thinking"]),
            "gradient_checkpointing": bool(config["model"]["gradient_checkpointing"]),
            "use_cache": bool(config["model"]["use_cache"]),
            "num_devices": int(training["num_devices"]),
            "global_batch_size": int(training["global_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "backward_subbatch_size_per_device": int(
                training["backward_subbatch_size_per_device"]
            ),
            "optimizer": training["optimizer"],
            "weight_decay": float(training["weight_decay"]),
            "warmup_ratio": float(training["warmup_ratio"]),
            "lr_scheduler_type": training["lr_scheduler_type"],
            "max_grad_norm": float(training["max_grad_norm"]),
            "seed": int(training["seed"]),
        }
        common_expected = {
            "torch_dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "max_seq_len": 2048,
            "enable_thinking": False,
            "gradient_checkpointing": True,
            "use_cache": False,
            "num_devices": 2,
            "global_batch_size": 64,
            "gradient_accumulation_steps": 8,
            "backward_subbatch_size_per_device": 2,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 1.0,
            "seed": 42,
        }
        if common_actual != common_expected:
            raise ValueError(f"v0.6 formal optimization contract violated: {common_actual}")
        if method in {"dpo10", "dpo100"}:
            if (
                int(training["epochs"]) != 1
                or float(training["lr"]) != 1e-6
                or int(training["dpo_batch_size_per_device"]) != 4
                or float(training["dpo_beta"]) != 0.1
                or float(training["simpo_beta"]) != 10.0
            ):
                raise ValueError(
                    "DPO-10/DPO-100 require 1 epoch, lr=1e-6, batch=64, beta=0.1, "
                    "and beta=10 for the common headroom score"
                )
        else:
            expected_gamma_min = 2700 / (2700 + 24000)
            if (
                int(training["epochs"]) != 2
                or float(training["lr"]) != 1e-5
                or float(training["simpo_beta"]) != 10.0
                or float(training["simpo_margin"]) != 2.0
                or labeled_steps != [0, 2, 4, 6]
                or pattern != [3, 4, 3, 4, 3, 4, 3, 4]
                or float(config["method"]["gamma0"]) != 1.0
                or abs(float(config["method"]["gamma_min"]) - expected_gamma_min) > 1e-10
                or float(config["method"]["gamma_decay"]) != 0.01
                or bool(config["method"]["detach_denominator"])
            ):
                raise ValueError("SSPO/SOPPO formal hyperparameter or 8/56 population contract violated")
            if method == "sspo_hard_exp" and (
                float(config["method"]["pseudo_prior"]) != 0.5
                or float(config["method"]["ema_momentum"]) != 0.95
                or float(config["method"]["reward_std_epsilon"]) != 1e-6
                or int(config["method"]["kde_grid_points"]) != 200
                or config["method"]["kde_bandwidth"] != "scott"
            ):
                raise ValueError("SSPO-hard KDE/prior/EMA reproduction contract violated")
            if method in {"soppo_pe_exp", "soppo_pe_static"} and (
                float(training["epsilon"]) != 1e-8
                or config["method"]["pe_distance"] != "l1"
                or bool(config["method"]["detach_denominator"])
            ):
                raise ValueError("SOPPO-PE epsilon/distance/gradient contract violated")

    data = config["data"]
    frozen_counts = {
        "total_samples": 30000,
        "labeled_train_samples": 2700,
        "labeled_val_samples": 300,
        "unlabeled_train_samples": 24000,
        "test_samples": 3000,
    }
    actual_counts = {key: int(data[key]) for key in frozen_counts}
    if actual_counts != frozen_counts:
        raise ValueError(
            "v0.6 data contract must remain 30k with 2,700 labeled-train, "
            f"300 labeled-val, 24,000 unlabeled, and 3,000 test; got {actual_counts}"
        )
    frozen_ratios = {"labeled_ratio": 0.10, "unlabeled_ratio": 0.80, "test_ratio": 0.10}
    actual_ratios = {key: float(data[key]) for key in frozen_ratios}
    if actual_ratios != frozen_ratios:
        raise ValueError(f"v0.6 split-ratio contract violated: {actual_ratios}")


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)


def canonical_json(config: Dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
