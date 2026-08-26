"""Durable and staging LoRA checkpoint contracts for Round3."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from peft import PeftModel

from .queue_protocol import canonical_json, file_sha256


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(payload: Dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(payload) != required:
        raise ValueError("Round3 checkpoint RNG state is incomplete")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    torch.cuda.set_rng_state_all(payload["torch_cuda"])


def _metadata(config: Dict[str, Any], step: int, adapter_sha: str) -> Dict[str, Any]:
    return {
        "format": "round3.peft_lora.v1",
        "method_id": config["method"]["name"],
        "optimizer_step": int(step),
        "base_model": str(Path(config["model"]["name_or_path"]).resolve()),
        "model_manifest": str(Path(config["model"]["manifest_path"]).resolve()),
        "git_commit": config["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "adapter_sha256": adapter_sha,
    }


def _save_adapter(policy: PeftModel, tokenizer, directory: Path, config: Dict[str, Any]) -> str:
    if not isinstance(policy, PeftModel):
        raise TypeError("Round3 checkpoint requires a PEFT LoRA policy")
    policy.save_pretrained(directory, safe_serialization=True, save_embedding_layers=False)
    tokenizer.save_pretrained(directory)
    (directory / "run_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8"
    )
    return file_sha256(directory / "adapter_model.safetensors")


def publish_staging_adapter(
    policy: PeftModel,
    tokenizer,
    root: str | Path,
    config: Dict[str, Any],
    step: int,
) -> Path:
    """Atomically publish an immutable current-policy adapter for both replicas.

    Staging adapters are retained through the run and are not a keep-N pruner.
    Cleanup is deliberately outside this implementation and requires a later
    result-retention decision.
    """
    root_path = Path(root).resolve()
    final = root_path / f"step_{int(step):06d}"
    partial = root_path / f".step_{int(step):06d}.partial"
    root_path.mkdir(parents=True, exist_ok=True)
    if final.exists() or partial.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 staging adapter: {final}")
    partial.mkdir()
    adapter_sha = _save_adapter(policy, tokenizer, partial, config)
    metadata = _metadata(config, step, adapter_sha)
    (partial / "checkpoint_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (partial / "READY.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(final)
    return final


def save_durable_checkpoint(
    policy: PeftModel,
    tokenizer,
    optimizer,
    scheduler,
    root: str | Path,
    config: Dict[str, Any],
    step: int,
    sspo_state: Dict[str, Any] | None,
    allow_smoke_step: bool = False,
) -> Path:
    if int(step) not in range(25, 251, 25) and not (allow_smoke_step and int(step) == 1):
        raise ValueError("Round3 durable checkpoints are only steps 25..250 by 25")
    root_path = Path(root).resolve()
    final = root_path / f"step_{int(step):06d}"
    partial = root_path / f".step_{int(step):06d}.partial"
    root_path.mkdir(parents=True, exist_ok=True)
    if final.exists() or partial.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 durable checkpoint: {final}")
    partial.mkdir()
    adapter_sha = _save_adapter(policy, tokenizer, partial, config)
    training_state = {
        "schema_version": "round3.training_state.v1",
        "method_id": config["method"]["name"],
        "global_step": int(step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "sspo": sspo_state,
    }
    torch.save(training_state, partial / "training_state.pt")
    state_sha = file_sha256(partial / "training_state.pt")
    metadata = {**_metadata(config, step, adapter_sha), "training_state_sha256": state_sha}
    (partial / "checkpoint_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (partial / "COMPLETE.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    partial.replace(final)
    return final


def load_training_state(
    checkpoint: str | Path,
    config: Dict[str, Any],
    optimizer,
    scheduler,
    require_sspo: bool,
) -> Dict[str, Any]:
    root = Path(checkpoint).resolve()
    metadata_path = root / "checkpoint_meta.json"
    state_path = root / "training_state.pt"
    complete_path = root / "COMPLETE.json"
    if not metadata_path.is_file() or not state_path.is_file() or not complete_path.is_file():
        raise FileNotFoundError(f"Round3 durable checkpoint is incomplete: {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete != metadata:
        raise ValueError("Round3 durable COMPLETE marker/metadata mismatch")
    if metadata.get("method_id") != config["method"]["name"]:
        raise ValueError("Round3 resume method mismatch")
    if metadata.get("git_commit") != config["provenance"]["git_commit"]:
        raise ValueError("Round3 resume Git commit mismatch")
    if metadata.get("config_sha256") != hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest():
        raise ValueError("Round3 resume config mismatch")
    if metadata.get("adapter_sha256") != file_sha256(root / "adapter_model.safetensors"):
        raise ValueError("Round3 durable adapter checksum mismatch")
    if metadata.get("training_state_sha256") != file_sha256(state_path):
        raise ValueError("Round3 durable training-state checksum mismatch")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != "round3.training_state.v1":
        raise ValueError("Unsupported Round3 training-state schema")
    if payload.get("method_id") != config["method"]["name"]:
        raise ValueError("Round3 training-state method mismatch")
    if int(payload.get("global_step", -1)) != int(metadata.get("optimizer_step", -2)):
        raise ValueError("Round3 checkpoint global step mismatch")
    if require_sspo and not isinstance(payload.get("sspo"), dict):
        raise ValueError("Round3 SSPO checkpoint is missing running state")
    if not require_sspo and payload.get("sspo") is not None:
        raise ValueError("Non-SSPO Round3 checkpoint unexpectedly contains SSPO state")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return payload
