"""Command adapter for the independent vLLM rollout worker."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class RolloutLaunchSpec:
    python_executable: str
    entrypoint: Path
    working_dir: Path
    model_path: Path
    checkpoint_path: Path
    artifact_path: Path
    gpu_ids: str
    tensor_parallel_size: int
    max_model_len: int
    max_new_tokens: int
    gpu_memory_utilization: float
    source: str


def _path(value: Any, name: str) -> Path:
    result = Path(str(value)).resolve()
    if not result.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return result


def launch_spec_from_config(config: Dict[str, Any]) -> RolloutLaunchSpec:
    model = config["model"]
    rollout = config["rollout"]
    return RolloutLaunchSpec(
        python_executable=str(rollout.get("python_executable") or "python"),
        entrypoint=_path(rollout["entrypoint"], "rollout.entrypoint"),
        working_dir=_path(rollout["working_dir"], "rollout.working_dir"),
        model_path=_path(model["name_or_path"], "model.name_or_path"),
        checkpoint_path=_path(rollout["policy_checkpoint"], "rollout.policy_checkpoint"),
        artifact_path=_path(rollout["artifact_path"], "rollout.artifact_path"),
        gpu_ids=str(rollout["gpu_ids"]),
        tensor_parallel_size=int(rollout["tensor_parallel_size"]),
        max_model_len=int(rollout["max_model_len"]),
        max_new_tokens=int(rollout["max_new_tokens"]),
        gpu_memory_utilization=float(rollout["gpu_memory_utilization"]),
        source=str(rollout["source"]),
    )


def build_rollout_command(spec: RolloutLaunchSpec) -> List[str]:
    """Call a project-owned rollout entrypoint, not an undocumented vLLM CLI."""
    return [
        spec.python_executable,
        str(spec.entrypoint),
        "--base-model",
        str(spec.model_path),
        "--policy-checkpoint",
        str(spec.checkpoint_path),
        "--artifact-path",
        str(spec.artifact_path),
        "--policy-source",
        spec.source,
        "--tensor-parallel-size",
        str(spec.tensor_parallel_size),
        "--max-model-len",
        str(spec.max_model_len),
        "--max-new-tokens",
        str(spec.max_new_tokens),
        "--gpu-memory-utilization",
        str(spec.gpu_memory_utilization),
    ]


def shell_command(spec: RolloutLaunchSpec) -> str:
    return " ".join(shlex.quote(item) for item in build_rollout_command(spec))
