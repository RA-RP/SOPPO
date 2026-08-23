"""Transformers-native TP=2 loading and launch planning for round2."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from packaging.version import Version


@dataclass(frozen=True)
class TPLaunchSpec:
    python_executable: str
    config_path: Path
    gpu_ids: str
    nproc_per_node: int
    output_dir: Path


def require_package_version(package: str, minimum: str) -> str:
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"Round2 requires package {package}>={minimum}") from exc
    if Version(actual) < Version(str(minimum)):
        raise RuntimeError(
            f"Round2 requires {package}>={minimum}, but the environment has {actual}"
        )
    return actual


def validate_training_runtime(config: Dict[str, Any]) -> Dict[str, str]:
    tp = config["tensor_parallel"]
    versions = {
        "torch": require_package_version("torch", tp["minimum_torch_version"]),
        "transformers": require_package_version(
            "transformers", tp["minimum_transformers_version"]
        ),
        "peft": require_package_version("peft", tp["minimum_peft_version"]),
    }
    return versions


def launch_spec_from_config(
    config: Dict[str, Any], config_path: str | Path, python_executable: str
) -> TPLaunchSpec:
    tp = config["tensor_parallel"]
    gpu_ids = str(tp["gpu_ids"])
    nproc = len([value for value in gpu_ids.split(",") if value])
    return TPLaunchSpec(
        python_executable=str(python_executable),
        config_path=Path(config_path).resolve(),
        gpu_ids=gpu_ids,
        nproc_per_node=nproc,
        output_dir=Path(config["output"]["run_dir"]).resolve(),
    )


def build_tp_command(spec: TPLaunchSpec) -> List[str]:
    return [
        spec.python_executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={spec.nproc_per_node}",
        "-m",
        "src.round2.tp_trainer",
        "--config",
        str(spec.config_path),
    ]


def describe_tp_parameters(model) -> Dict[str, Any]:
    """Return evidence that base weights are sharded instead of DDP replicas."""
    try:
        from torch.distributed.tensor import DTensor, Shard
    except ImportError as exc:
        raise RuntimeError("The configured torch build has no DTensor support") from exc
    sharded: List[Tuple[str, str]] = []
    replicated = 0
    for name, parameter in model.named_parameters():
        if isinstance(parameter, DTensor):
            placements = tuple(parameter.placements)
            if any(isinstance(placement, Shard) for placement in placements):
                sharded.append((name, repr(placements)))
            else:
                replicated += 1
    if not sharded:
        raise RuntimeError(
            "TP verification failed: no sharded DTensor parameter was found; "
            "refuse to run a replicated two-GPU model"
        )
    return {
        "backend": "transformers-native-tp",
        "sharded_parameter_count": len(sharded),
        "replicated_dtensor_parameter_count": replicated,
        "sample_sharded_parameters": [
            {"name": name, "placements": placements}
            for name, placements in sharded[:12]
        ],
    }


def write_launch_record(path: str | Path, spec: TPLaunchSpec, command: List[str]) -> None:
    output = Path(path).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite TP launch record: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "backend": "transformers-native-tp",
                "gpu_ids": spec.gpu_ids,
                "nproc_per_node": spec.nproc_per_node,
                "command": command,
                "config": str(spec.config_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
