"""Transformers-native TP=2 loading and launch planning for round2."""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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


_SHARDED_WEIGHT_DIMS = {
    "colwise": -2,
    "colwise_gather_output": -2,
    "packed_colwise": -2,
    "rowwise": -1,
    "rowwise_split_input": -1,
    "packed_rowwise": -1,
}


def _checkpoint_tensor_shapes(model_path: str | Path) -> Dict[str, Tuple[int, ...]]:
    """Read full tensor shapes from safetensors headers without loading weights."""
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("Round2 TP verification requires safetensors") from exc

    root = Path(model_path).resolve()
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError(f"Invalid safetensors index: {index_path}")
        files = sorted({root / str(value) for value in weight_map.values()})
        expected_names = set(weight_map)
    else:
        single = root / "model.safetensors"
        if not single.is_file():
            raise RuntimeError(
                "TP verification requires model.safetensors or "
                f"model.safetensors.index.json under {root}"
            )
        files = [single]
        expected_names = None

    shapes: Dict[str, Tuple[int, ...]] = {}
    for checkpoint_file in files:
        if not checkpoint_file.is_file():
            raise RuntimeError(f"Missing safetensors shard: {checkpoint_file}")
        with safe_open(str(checkpoint_file), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in shapes:
                    raise RuntimeError(f"Duplicate tensor in model checkpoint: {name}")
                shapes[name] = tuple(
                    int(value) for value in handle.get_slice(name).get_shape()
                )
    if expected_names is not None and set(shapes) != expected_names:
        missing = sorted(expected_names - set(shapes))
        unexpected = sorted(set(shapes) - expected_names)
        raise RuntimeError(
            "Safetensors index/header mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    if not shapes:
        raise RuntimeError(f"No tensors found in model checkpoint: {root}")
    return shapes


def _generic_module_name(name: str) -> str:
    return re.sub(r"\d+", "*", name)


def _expected_local_shape(
    full_shape: Sequence[int], plan: str, tp_size: int, rank: int
) -> Tuple[int, ...]:
    if plan not in _SHARDED_WEIGHT_DIMS:
        raise ValueError(f"Unsupported sharded TP plan for verification: {plan}")
    shape = [int(value) for value in full_shape]
    if len(shape) < 2:
        raise ValueError(f"TP weight must have at least two dimensions: {shape}")
    dimension = _SHARDED_WEIGHT_DIMS[plan] % len(shape)
    shard_size = math.ceil(shape[dimension] / int(tp_size))
    start = int(rank) * shard_size
    end = min(start + shard_size, shape[dimension])
    shape[dimension] = max(0, end - start)
    return tuple(shape)


def _verify_local_tp_shapes(
    model,
    checkpoint_shapes: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    """Verify Transformers-native local Tensor shards against full checkpoint shapes."""
    tp_size = int(getattr(model, "_tp_size", 0) or 0)
    device_mesh = getattr(model, "_device_mesh", None)
    tp_plan = dict(getattr(model, "tp_plan", {}) or {})
    if tp_size != 2 or device_mesh is None or int(device_mesh.size()) != tp_size:
        raise RuntimeError(
            "TP verification failed: model does not expose the required TP=2 device mesh"
        )
    if not tp_plan:
        raise RuntimeError("TP verification failed: model TP plan is empty")
    rank = int(device_mesh.get_local_rank())
    if not 0 <= rank < tp_size:
        raise RuntimeError(f"TP verification failed: invalid mesh rank {rank}")

    sharded_rules = {
        name: plan for name, plan in tp_plan.items() if plan in _SHARDED_WEIGHT_DIMS
    }
    if not sharded_rules:
        raise RuntimeError("TP verification failed: TP plan contains no sharded weights")

    applied_rules = set()
    sharded: List[Dict[str, Any]] = []
    replicated_tp_modules = 0
    for module_name, module in model.named_modules():
        generic_name = _generic_module_name(module_name)
        expected_plan = tp_plan.get(generic_name)
        if expected_plan is None:
            continue
        module_plan = getattr(module, "_hf_tp_plan", None)
        if module_plan is None:
            raise RuntimeError(
                f"TP verification failed: TP hooks are missing for {module_name}"
            )
        module_mesh = getattr(module, "_hf_device_mesh", None)
        if module_mesh is None or int(module_mesh.size()) != tp_size:
            raise RuntimeError(
                f"TP verification failed: invalid module mesh for {module_name}"
            )
        if expected_plan != module_plan:
            raise RuntimeError(
                "TP verification failed: module plan mismatch for "
                f"{module_name}: expected={expected_plan}, actual={module_plan}"
            )
        if module_plan not in _SHARDED_WEIGHT_DIMS:
            replicated_tp_modules += 1
            continue

        parameter = getattr(module, "weight", None)
        parameter_name = f"{module_name}.weight"
        if parameter is None:
            raise RuntimeError(
                f"TP verification failed: sharded module has no weight: {module_name}"
            )
        if parameter_name not in checkpoint_shapes:
            raise RuntimeError(
                "TP verification failed: checkpoint shape is missing for "
                f"{parameter_name}"
            )
        full_shape = tuple(int(value) for value in checkpoint_shapes[parameter_name])
        expected_local_shape = _expected_local_shape(
            full_shape, str(module_plan), tp_size, rank
        )
        actual_local_shape = tuple(int(value) for value in parameter.shape)
        if actual_local_shape != expected_local_shape:
            raise RuntimeError(
                "TP verification failed: local shard shape mismatch for "
                f"{parameter_name}: full={full_shape}, expected_local="
                f"{expected_local_shape}, actual_local={actual_local_shape}"
            )
        applied_rules.add(generic_name)
        sharded.append(
            {
                "name": parameter_name,
                "plan": str(module_plan),
                "full_shape": list(full_shape),
                "local_shape": list(actual_local_shape),
            }
        )

    missing_rules = sorted(set(sharded_rules) - applied_rules)
    if missing_rules:
        raise RuntimeError(
            "TP verification failed: sharded TP rules were not applied: "
            f"{missing_rules}"
        )
    if not sharded:
        raise RuntimeError(
            "TP verification failed: no checkpoint-backed local shard was found; "
            "refuse to run a replicated two-GPU model"
        )
    return {
        "backend": "transformers-native-tp",
        "sharding_representation": "checkpoint-verified-local-tensor-slices",
        "tp_size": tp_size,
        "mesh_rank": rank,
        "tp_plan_rule_count": len(tp_plan),
        "sharded_rule_count": len(sharded_rules),
        "sharded_parameter_count": len(sharded),
        "replicated_tp_module_count": replicated_tp_modules,
        "sample_sharded_parameters": sharded[:12],
    }


def describe_tp_parameters(model, model_path: str | Path) -> Dict[str, Any]:
    """Return fail-closed evidence that base weights are real local TP shards."""
    checkpoint_shapes = _checkpoint_tensor_shapes(model_path)
    evidence = _verify_local_tp_shapes(model, checkpoint_shapes)
    evidence["checkpoint_tensor_count"] = len(checkpoint_shapes)
    return evidence


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
