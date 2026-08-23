"""Launch planning and runtime checks for the project-owned vLLM worker."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from packaging.version import Version


@dataclass(frozen=True)
class RolloutLaunchSpec:
    python_executable: str
    config_path: Path
    gpu_ids: str
    artifact_dir: Path


def validate_rollout_runtime(config: Dict[str, Any]) -> Dict[str, str]:
    minimum = str(config["rollout"]["minimum_vllm_version"])
    try:
        actual = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Round2 rollout requires vllm>={minimum}") from exc
    if Version(actual) < Version(minimum):
        raise RuntimeError(
            f"Round2 rollout requires vllm>={minimum}, but the environment has {actual}"
        )
    return {
        "vllm": actual,
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }


def launch_spec_from_config(
    config: Dict[str, Any], config_path: str | Path, python_executable: str
) -> RolloutLaunchSpec:
    rollout = config["rollout"]
    return RolloutLaunchSpec(
        python_executable=str(python_executable),
        config_path=Path(config_path).resolve(),
        gpu_ids=str(rollout["gpu_ids"]),
        artifact_dir=Path(rollout["artifact_dir"]).resolve(),
    )


def build_rollout_command(spec: RolloutLaunchSpec) -> List[str]:
    return [
        spec.python_executable,
        "-m",
        "src.round2.run_rollout",
        "--config",
        str(spec.config_path),
    ]
