"""Megatron launch adapter for round2.

This module intentionally does not import Megatron. The target server owns the
Megatron/Megatron-Core installation and exposes its training entrypoint through
the round2 config. Keeping the adapter command-oriented makes dependency
availability explicit and prevents a silent fallback to DDP.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class MegatronLaunchSpec:
    python_executable: str
    entrypoint: Path
    working_dir: Path
    model_path: Path
    manifest_path: Path
    data_dir: Path
    output_dir: Path
    gpu_ids: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    max_seq_len: int
    global_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    seed: int
    method: str
    rollout_artifact: Path
    dry_run: bool = False


def _as_path(value: Any, name: str) -> Path:
    path = Path(str(value)).resolve()
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    return path


def launch_spec_from_config(config: Dict[str, Any]) -> MegatronLaunchSpec:
    model = config["model"]
    data = config["data"]
    training = config["training"]
    method = config["method"]
    output = config["output"]
    megatron = config["megatron"]
    rollout = config["rollout"]
    return MegatronLaunchSpec(
        python_executable=str(megatron.get("python_executable") or "python"),
        entrypoint=_as_path(megatron["entrypoint"], "megatron.entrypoint"),
        working_dir=_as_path(megatron["working_dir"], "megatron.working_dir"),
        model_path=_as_path(model["name_or_path"], "model.name_or_path"),
        manifest_path=_as_path(model["manifest_path"], "model.manifest_path"),
        data_dir=_as_path(data["data_dir"], "data.data_dir"),
        output_dir=_as_path(output["run_dir"], "output.run_dir"),
        gpu_ids=str(megatron["gpu_ids"]),
        tensor_parallel_size=int(megatron["tensor_model_parallel_size"]),
        pipeline_parallel_size=int(megatron["pipeline_model_parallel_size"]),
        data_parallel_size=int(megatron["data_parallel_size"]),
        max_seq_len=int(model["max_seq_len"]),
        global_batch_size=int(training["global_batch_size"]),
        micro_batch_size=int(megatron["micro_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(training["seed"]),
        method=str(method["name"]),
        rollout_artifact=_as_path(rollout["artifact_path"], "rollout.artifact_path"),
    )


def build_megatron_command(spec: MegatronLaunchSpec) -> List[str]:
    """Build an explicit command; never substitute the DDP trainer."""
    return [
        spec.python_executable,
        str(spec.entrypoint),
        "--model-path",
        str(spec.model_path),
        "--model-manifest",
        str(spec.manifest_path),
        "--data-dir",
        str(spec.data_dir),
        "--output-dir",
        str(spec.output_dir),
        "--rollout-artifact",
        str(spec.rollout_artifact),
        "--method",
        spec.method,
        "--max-seq-len",
        str(spec.max_seq_len),
        "--global-batch-size",
        str(spec.global_batch_size),
        "--micro-batch-size",
        str(spec.micro_batch_size),
        "--gradient-accumulation-steps",
        str(spec.gradient_accumulation_steps),
        "--learning-rate",
        str(spec.learning_rate),
        "--seed",
        str(spec.seed),
        "--tensor-model-parallel-size",
        str(spec.tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(spec.pipeline_parallel_size),
        "--data-parallel-size",
        str(spec.data_parallel_size),
    ]


def shell_command(spec: MegatronLaunchSpec) -> str:
    return " ".join(shlex.quote(item) for item in build_megatron_command(spec))
