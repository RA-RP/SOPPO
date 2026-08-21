"""Offline Qwen3 loading plus standard PEFT LoRA adapter checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_manifest import verify_manifest


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _verify_adapter_contract(checkpoint: Path, config: Dict) -> None:
    adapter_path = checkpoint / "adapter_config.json"
    metadata_path = checkpoint / "checkpoint_meta.json"
    if not adapter_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"LoRA adapter checkpoint is incomplete: {checkpoint}")
    recorded = json.loads(adapter_path.read_text(encoding="utf-8"))
    expected = config["model"]["lora"]
    actual_contract = {
        "r": int(recorded.get("r", -1)),
        "alpha": int(recorded.get("lora_alpha", -1)),
        "dropout": float(recorded.get("lora_dropout", -1)),
        "bias": recorded.get("bias"),
        "task_type": recorded.get("task_type"),
        "target_modules": set(recorded.get("target_modules") or []),
    }
    expected_contract = {
        "r": int(expected["r"]),
        "alpha": int(expected["alpha"]),
        "dropout": float(expected["dropout"]),
        "bias": expected["bias"],
        "task_type": expected["task_type"],
        "target_modules": set(expected["target_modules"]),
    }
    if actual_contract != expected_contract:
        raise ValueError(
            f"Adapter/config LoRA contract mismatch: actual={actual_contract}, "
            f"expected={expected_contract}"
        )
    expected_base = Path(config["model"]["name_or_path"]).resolve()
    recorded_base = recorded.get("base_model_name_or_path")
    if not recorded_base or Path(recorded_base).resolve() != expected_base:
        raise ValueError("Adapter was not created from the configured frozen base model")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(metadata.get("base_model", "")).resolve() != expected_base:
        raise ValueError("Adapter checkpoint metadata/base-model mismatch")


def _verify_trainable_lora_parameters(policy, target_modules) -> None:
    expected_targets = set(target_modules)
    observed_targets = set()
    unexpected = []
    for name, parameter in policy.named_parameters():
        if not parameter.requires_grad:
            continue
        pieces = set(name.split("."))
        targets = pieces & expected_targets
        if len(targets) != 1 or not ({"lora_A", "lora_B"} & pieces):
            unexpected.append(name)
        else:
            observed_targets.update(targets)
    if unexpected:
        raise ValueError(f"Non-contract trainable parameters found: {unexpected[:8]}")
    if observed_targets != expected_targets:
        raise ValueError(
            f"LoRA target coverage mismatch: observed={sorted(observed_targets)}, "
            f"expected={sorted(expected_targets)}"
        )


def verify_frozen_model(model_dir: str, manifest_path: str) -> None:
    verify_manifest(Path(model_dir).resolve(), Path(manifest_path).resolve())


def load_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_frozen_base(
    model_dir: str,
    manifest_path: str,
    dtype_name: str,
    attention_implementation: str,
    gradient_checkpointing: bool,
):
    verify_frozen_model(model_dir, manifest_path)
    if dtype_name not in DTYPES:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=DTYPES[dtype_name],
        attn_implementation=attention_implementation,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # A frozen base with gradient checkpointing needs a differentiable input
        # so checkpointed blocks retain the LoRA adapter graph.
        model.enable_input_require_grads()
    return model


def load_policy_model(
    model_dir: str,
    manifest_path: str,
    dtype_name: str = "bfloat16",
    attention_implementation: str = "sdpa",
    gradient_checkpointing: bool = False,
):
    """Load the immutable base model (reference-cache and manifest paths)."""
    return _load_frozen_base(
        model_dir,
        manifest_path,
        dtype_name,
        attention_implementation,
        gradient_checkpointing,
    )


def _lora_config(config: Dict) -> LoraConfig:
    values = config["model"]["lora"]
    return LoraConfig(
        r=int(values["r"]),
        lora_alpha=int(values["alpha"]),
        lora_dropout=float(values["dropout"]),
        bias=values["bias"],
        task_type=values["task_type"],
        target_modules=list(values["target_modules"]),
    )


def load_trainable_policy(config: Dict, adapter_checkpoint: str | None = None):
    model_cfg = config["model"]
    base = _load_frozen_base(
        model_cfg["name_or_path"],
        model_cfg["manifest_path"],
        model_cfg["torch_dtype"],
        model_cfg["attention_implementation"],
        bool(model_cfg["gradient_checkpointing"]),
    )
    if adapter_checkpoint:
        checkpoint = Path(adapter_checkpoint).resolve()
        _verify_adapter_contract(checkpoint, config)
        policy = PeftModel.from_pretrained(
            base,
            str(checkpoint),
            is_trainable=True,
            local_files_only=True,
        )
    else:
        policy = get_peft_model(base, _lora_config(config))
    trainable, total = count_parameters(policy)
    if trainable <= 0 or trainable >= total:
        raise ValueError(f"Expected LoRA-only training, got trainable={trainable}, total={total}")
    _verify_trainable_lora_parameters(policy, model_cfg["lora"]["target_modules"])
    return policy


def load_adapter_for_inference(
    checkpoint: str,
    base_model_dir: str,
    manifest_path: str,
    dtype_name: str,
    attention_implementation: str = "sdpa",
    merge: bool = False,
):
    checkpoint_path = Path(checkpoint).resolve()
    adapter_path = checkpoint_path / "adapter_config.json"
    metadata_path = checkpoint_path / "checkpoint_meta.json"
    if not adapter_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint is incomplete: {checkpoint_path}")
    adapter_config = json.loads(
        adapter_path.read_text(encoding="utf-8")
    )
    recorded_base = adapter_config.get("base_model_name_or_path")
    if not recorded_base or Path(recorded_base).resolve() != Path(base_model_dir).resolve():
        raise ValueError("Adapter/base-model mismatch during inference")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        Path(metadata.get("base_model", "")).resolve() != Path(base_model_dir).resolve()
        or Path(metadata.get("model_manifest", "")).resolve() != Path(manifest_path).resolve()
    ):
        raise ValueError("Adapter checkpoint metadata does not match the frozen inference base")
    base = _load_frozen_base(
        base_model_dir,
        manifest_path,
        dtype_name,
        attention_implementation,
        gradient_checkpointing=False,
    )
    model = PeftModel.from_pretrained(
        base,
        str(checkpoint_path),
        is_trainable=False,
        local_files_only=True,
    )
    return model.merge_and_unload(safe_merge=True) if merge else model


def count_parameters(model) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_adapter_checkpoint(
    model,
    tokenizer,
    output_dir: str,
    config: Dict,
    rank: int,
    training_state: Dict | None = None,
) -> None:
    """Write a small, HF/PEFT-loadable LoRA adapter on rank zero."""
    if rank != 0:
        return
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite checkpoint: {output}")
    output.mkdir(parents=True)
    policy = unwrap_model(model)
    if not isinstance(policy, PeftModel):
        raise TypeError("Checkpoint contract requires a PEFT LoRA policy")
    policy.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    import yaml

    with (output / "run_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True, allow_unicode=True)
    metadata = {
        "format": "peft_lora_adapter",
        "base_model": str(Path(config["model"]["name_or_path"]).resolve()),
        "model_manifest": str(Path(config["model"]["manifest_path"]).resolve()),
        "training_state": training_state or {},
    }
    (output / "checkpoint_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
