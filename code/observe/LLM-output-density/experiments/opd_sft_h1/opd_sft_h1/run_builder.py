from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .registry import append_jsonl, validate_run_record


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return data


def build_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name).strip("-")
    return f"{safe_name}-{timestamp}"


def build_run_record(config: dict[str, Any], *, run_id: str | None = None, status: str = "planned") -> dict[str, Any]:
    experiment = config.get("experiment") or {}
    model = config.get("model") or {}
    data = config.get("data") or {}
    trl = config.get("trl") or {}
    registry = config.get("registry") or {}
    name = str(experiment.get("name") or "trl_first_minimal")

    record = {
        "run_id": run_id or build_run_id(name),
        "trajectory_group_id": registry.get("trajectory_group_id") or name,
        "method": registry.get("method", "trl_opd_like"),
        "role_label": registry.get("role_label", "TRL-OPD-lmbda-1.0"),
        "parent_run_id": registry.get("parent_run_id"),
        "start_checkpoint": model.get("student_start_checkpoint") or model.get("cold_start_checkpoint"),
        "seed": experiment.get("seed"),
        "model": model,
        "data": data,
        "training": trl,
        "artifacts": {"output_root": experiment.get("output_root")},
        "status": status,
        "teacher_model": model.get("teacher_model"),
        "teacher_mode": "server" if trl.get("use_teacher_server") else "local",
        "lmbda": trl.get("lmbda"),
        "beta": trl.get("beta"),
        "loss_top_k": trl.get("loss_top_k"),
        "use_vllm": bool(trl.get("use_vllm", False)),
        "use_teacher_server": bool(trl.get("use_teacher_server", False)),
        "teacher_model_server_url": trl.get("teacher_model_server_url"),
        "pi_mix_lambda": registry.get("pi_mix_lambda", trl.get("lmbda")),
    }
    validate_run_record(record)
    return record


def write_run_record(config_path: str | Path, registry_path: str | Path, *, status: str = "planned") -> dict[str, Any]:
    config = load_yaml_config(config_path)
    record = build_run_record(config, status=status)
    append_jsonl(registry_path, record)
    return record
