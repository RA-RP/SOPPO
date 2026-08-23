"""Server-only environment, data, model, and GPU checks for round2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from ..config import canonical_json
from ..model.model_manifest import verify_manifest
from .config import load_round2_config, validate_round2_config
from .rollout_backend import validate_rollout_runtime
from .sft_schema import validate_sft_corpus
from .tp_backend import validate_training_runtime


def _cuda_evidence(expected_devices: int, expected_visible: str) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected round2 environment")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(expected_visible):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES differs from the resolved config: "
            f"actual={visible}, expected={expected_visible}"
        )
    count = torch.cuda.device_count()
    if count != int(expected_devices):
        raise RuntimeError(
            f"CUDA device count mismatch: actual={count}, expected={expected_devices}"
        )
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        if "4090" not in properties.name:
            raise RuntimeError(
                f"The approved round2 profile requires RTX 4090, got {properties.name}"
            )
        if properties.total_memory < 23 * 1024**3:
            raise RuntimeError(
                f"GPU {index} has less than 23 GiB physical memory: {properties.total_memory}"
            )
        devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    return {
        "cuda_visible_devices": visible,
        "torch_cuda": torch.version.cuda,
        "devices": devices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("training", "rollout"), required=True)
    args = parser.parse_args()
    config = load_round2_config(args.config)
    validate_round2_config(config)

    model_dir = Path(config["model"]["name_or_path"]).resolve()
    manifest = Path(config["model"]["manifest_path"]).resolve()
    verify_manifest(model_dir, manifest)
    data_dir = Path(config["data"]["data_dir"]).resolve()
    for name in ("labeled_train.jsonl", "labeled_val.jsonl", "unlabeled_train.jsonl"):
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Round2 data file is missing: {data_dir / name}")
    sft = validate_sft_corpus(
        config["rollout"]["sft_data_file"],
        data_dir / "unlabeled_train.jsonl",
        int(config["data"]["unlabeled_train_samples"]),
    )

    if args.role == "training":
        versions = validate_training_runtime(config)
        expected_devices = 2
        expected_visible = str(config["tensor_parallel"]["gpu_ids"])
    else:
        versions = validate_rollout_runtime(config)
        expected_devices = 1
        expected_visible = str(config["rollout"]["gpu_ids"])
    evidence = {
        "role": args.role,
        "git_commit": config["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "versions": versions,
        "cuda": _cuda_evidence(expected_devices, expected_visible),
        "model": str(model_dir),
        "model_manifest": str(manifest),
        "sft": sft,
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
