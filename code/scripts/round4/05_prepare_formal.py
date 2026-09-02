#!/usr/bin/env python3
"""Create immutable Round4 formal configs and a separate dataset view.

The prepared data directory is treated as immutable.  FrozenPE candidate B is
instead written into a run-specific data view whose other files are symlinks to
the validated prepared files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_REVISION = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
METHODS = ("dpo", "sspo", "staticpe", "frozenpe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data-dir", required=True)
    parser.add_argument("--formal-data-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    return parser.parse_args()


def absolute(value: str, label: str, *, exists: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    return path.resolve(strict=exists)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def create_new_directory(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to reuse {label}: {path}")
    path.mkdir(parents=True)


def formal_config(method: str, model_path: Path, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    per_device_train = 1 if method == "dpo" else 4
    dataset = "ultrafeedback_fb0.1_dpo" if method == "dpo" else "ultra_combined_fb0.1_ch0.1"
    if method == "frozenpe":
        dataset = "ultra_combined_fb0.1_ch0.1_frozenpe"
    config: dict[str, Any] = {
        "model_name_or_path": str(model_path),
        "model_revision": MODEL_REVISION,
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_target": "all",
        "dataset": dataset,
        "eval_dataset": "ultrafeedback_round4_eval",
        "dataset_dir": str(data_dir),
        "template": "qwen3",
        "cutoff_len": 1024,
        "max_samples": 10_000_000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 12,
        "val_size": 0.0,
        "pref_loss": "sigmoid" if method == "dpo" else method,
        "pref_beta": 10.0 if method == "staticpe" else 0.1,
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": 1.0,
        "per_device_train_batch_size": per_device_train,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "learning_rate": 1.0e-5,
        "bf16": True,
        "fp16": False,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
        "seed": 42,
        "data_seed": 42,
        "logging_first_step": True,
        "logging_steps": 20,
        "save_strategy": "steps",
        "save_steps": 300,
        "save_total_limit": 2,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "load_best_model_at_end": False,
        "plot_loss": True,
        "report_to": "none",
        "ddp_timeout": 180000000,
    }
    if method == "sspo":
        config.update(
            {
                "sspo_base": "dpo",
                "sspo_gamma_decay": 0.001,
                "sspo_gamma_0": 1.0,
                "sspo_gamma_min": 0.2273,
                "sspo_prior": 0.5,
                "sspo_min_labeled_per_batch": 0,
                "sspo_min_unlabeled_per_batch": 2,
            }
        )
    elif method == "staticpe":
        config.update(
            {
                "staticpe_lambda": 0.1,
                "staticpe_epsilon": 1.0e-8,
                "staticpe_temperature": 1.0,
                "staticpe_reward_norm_momentum": 0.95,
                "staticpe_reward_clip_range": 5.0,
                "simpo_gamma": 2.0,
                "staticpe_min_labeled_per_batch": 0,
                "staticpe_min_unlabeled_per_batch": 2,
                "pe_contract": "simpo_single_response_ema_v1",
            }
        )
    elif method == "frozenpe":
        config.update(
            {
                "frozenpe_lambda": 0.1,
                "frozenpe_epsilon": 1.0e-8,
                "frozenpe_min_labeled_per_batch": 0,
                "frozenpe_min_unlabeled_per_batch": 2,
                "pe_contract": "dpo_frozen_pair_v1",
            }
        )
    return config


def export_config(model_path: Path, adapter_dir: Path, export_dir: Path) -> dict[str, Any]:
    return {
        "model_name_or_path": str(model_path),
        "model_revision": MODEL_REVISION,
        "adapter_name_or_path": str(adapter_dir),
        "template": "qwen3",
        "finetuning_type": "lora",
        "trust_remote_code": True,
        "infer_dtype": "bfloat16",
        "export_dir": str(export_dir),
        "export_size": 2,
        "export_device": "cpu",
        "export_legacy_format": False,
    }


def main() -> None:
    args = parse_args()
    if len(args.code_commit) != 40:
        raise ValueError("--code-commit must be a full Git SHA")
    if not args.run_id.strip():
        raise ValueError("--run-id must be non-empty")
    if args.model_revision != MODEL_REVISION:
        raise ValueError("Round4 formal model revision is fixed")
    prepared = absolute(args.prepared_data_dir, "prepared data", exists=True)
    formal_data = absolute(args.formal_data_dir, "formal data")
    run_root = absolute(args.run_root, "run root")
    export_root = absolute(args.export_root, "export root")
    model = absolute(args.model_path, "model", exists=True)
    model_manifest = read_json(model / "ROUND4_ASSET_MANIFEST.json")
    if model_manifest.get("repo_id") != "Qwen/Qwen3-1.7B" or model_manifest.get("resolved_revision") != args.model_revision:
        raise ValueError("model directory is not the frozen Qwen3 Round4 revision")
    for path, label in ((formal_data, "formal data"), (run_root, "run root"), (export_root, "export root")):
        create_new_directory(path, label)

    manifest_path = prepared / "ROUND4_PREPROCESS_MANIFEST.json"
    info_path = prepared / "dataset_info.json"
    manifest = read_json(manifest_path)
    info = read_json(info_path)
    if manifest.get("schema") != "round4-preprocessing-v2":
        raise ValueError("prepared manifest must use the Round4 v2 schema")
    expected = {
        "ultrafeedback_fb0.1_dpo": (6105, 6105, 0),
        "ultra_combined_fb0.1_ch0.1": (26888, 6105, 20783),
        "ultrafeedback_round4_eval": (1997, 1997, 0),
    }
    for name, counts in expected.items():
        observed = manifest.get("outputs", {}).get(name, {})
        if tuple(observed.get(key) for key in ("rows", "labeled_rows", "unlabeled_rows")) != counts:
            raise ValueError(f"unexpected validated counts for {name}: {observed}")
        filename = observed.get("file_name")
        if not isinstance(filename, str) or info.get(name, {}).get("file_name") != filename:
            raise ValueError(f"dataset_info is inconsistent for {name}")
        source = prepared / filename
        if not source.is_file() or sha256_file(source) != observed.get("sha256"):
            raise ValueError(f"prepared dataset checksum mismatch for {name}")
        os.symlink(source, formal_data / filename)

    atomic_json(formal_data / "dataset_info.json", info)
    atomic_json(
        formal_data / "FORMAL_DATA_VIEW.json",
        {
            "schema": "round4-formal-data-view-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "prepared_manifest_sha256": sha256_file(manifest_path),
            "prepared_dataset_info_sha256": sha256_file(info_path),
            "symlinked_datasets": {name: payload["file_name"] for name, payload in info.items()},
        },
    )

    configs = run_root / "configs"
    configs.mkdir()
    method_plan: dict[str, Any] = {}
    for method in METHODS:
        adapter = run_root / method / "adapter"
        merged = export_root / method / "merged"
        train = formal_config(method, model, formal_data, adapter)
        train_path = configs / f"{method}_train.json"
        export_path = configs / f"{method}_export.json"
        atomic_json(train_path, train)
        atomic_json(export_path, export_config(model, adapter, merged))
        method_plan[method] = {
            "train_config_sha256": sha256_file(train_path),
            "export_config_sha256": sha256_file(export_path),
            "dataset": train["dataset"],
            "effective_batch_size": 16 if method == "dpo" else 64,
        }
    atomic_json(
        run_root / "FORMAL_PLAN.json",
        {
            "schema": "round4-formal-plan-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "code_commit": args.code_commit,
            "model_path": str(model),
            "model_revision": args.model_revision,
            "formal_data_view": str(formal_data),
            "prepared_manifest_sha256": sha256_file(manifest_path),
            "methods": method_plan,
        },
    )
    print(run_root / "FORMAL_PLAN.json")


if __name__ == "__main__":
    main()
