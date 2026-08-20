from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .paths import DEFAULT_REGISTRY_DIR, ensure_dir


RUN_REGISTRY_FILENAME = "run_registry.jsonl"
CHECKPOINT_REGISTRY_FILENAME = "checkpoints.jsonl"

REQUIRED_RUN_FIELDS = (
    "run_id",
    "trajectory_group_id",
    "method",
    "role_label",
    "parent_run_id",
    "start_checkpoint",
    "seed",
    "model",
    "data",
    "training",
    "artifacts",
    "status",
)

TRL_EXTRA_FIELDS = (
    "teacher_model",
    "teacher_mode",
    "lmbda",
    "beta",
    "loss_top_k",
    "use_vllm",
    "use_teacher_server",
    "teacher_model_server_url",
)

ALLOWED_METHODS = {
    "trl_opd_like",
    "standard_opd",
    "verl_opd",
    "lightning_opd",
    "npd",
    "sft",
    "continued_sft",
    "cold_start",
    "geometry_probe",
}


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default))
        f.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path_obj}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL record at {path_obj}:{line_no} is not an object")
            rows.append(row)
    return rows


def _require_keys(record: dict[str, Any], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def _validate_method(record: dict[str, Any]) -> None:
    method = record.get("method")
    if method == "opd":
        raise ValueError("Use a specific method label such as 'trl_opd_like'; do not use method='opd'.")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"Unknown method label: {method!r}")


def validate_run_record(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        raise TypeError("run record must be a dictionary")
    _require_keys(record, REQUIRED_RUN_FIELDS)
    _validate_method(record)
    if not str(record.get("run_id") or "").strip():
        raise ValueError("run_id must be non-empty")
    if record["method"] == "trl_opd_like":
        _require_keys(record, TRL_EXTRA_FIELDS)
    return True


def validate_checkpoint_record(record: dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        raise TypeError("checkpoint record must be a dictionary")
    _require_keys(record, ("checkpoint_id", *REQUIRED_RUN_FIELDS))
    _validate_method(record)
    if not str(record.get("checkpoint_id") or "").strip():
        raise ValueError("checkpoint_id must be non-empty")
    if record["method"] == "trl_opd_like":
        _require_keys(record, TRL_EXTRA_FIELDS)
    return True


def default_run_registry_path(registry_dir: str | Path | None = None) -> Path:
    root = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    return root / RUN_REGISTRY_FILENAME


def default_checkpoint_registry_path(registry_dir: str | Path | None = None) -> Path:
    root = Path(registry_dir) if registry_dir is not None else DEFAULT_REGISTRY_DIR
    return root / CHECKPOINT_REGISTRY_FILENAME


def update_status(run_id: str, status: str, path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else default_run_registry_path()
    rows = load_jsonl(registry_path)
    updated_record: dict[str, Any] | None = None

    for row in rows:
        if row.get("run_id") == run_id:
            row["status"] = status
            updated_record = row

    if updated_record is None:
        raise ValueError(f"run_id not found in registry: {run_id}")

    ensure_dir(registry_path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=registry_path.name, suffix=".tmp", dir=str(registry_path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default))
            f.write("\n")
    os.replace(tmp_name, registry_path)
    return updated_record
