#!/usr/bin/env python3
"""Create deterministic Round4 smoke fixtures and standalone run configs.

This utility uses only the Python standard library.  It consumes the immutable
full Round4 preprocessing outputs, selects enough rows for exactly two
optimizer steps, and writes configs whose batch/GA settings remain identical
to the approved formal experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_REVISION = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
METHODS = ("dpo", "sspo", "staticpe", "frozenpe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data-dir", required=True)
    parser.add_argument("--smoke-data-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpo-labeled-rows", type=int, default=32)
    parser.add_argument("--mixed-labeled-rows", type=int, default=64)
    parser.add_argument("--mixed-unlabeled-rows", type=int, default=64)
    parser.add_argument("--eval-rows", type=int, default=8)
    return parser.parse_args()


def absolute_path(value: str, label: str, *, must_exist: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {value}")
    return path.resolve(strict=must_exist)


def ensure_new_directory(path: Path, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to reuse existing {label}: {path}")
    path.mkdir(parents=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(row: Any, index: int, source: Path) -> str:
    if not isinstance(row, dict):
        raise ValueError(f"{source} row {index} is not an object")
    instruction = row.get("instruction")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    unlabeled = row.get("unlabeled")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{source} row {index} has no instruction")
    if not all(isinstance(value, str) for value in (chosen, rejected, unlabeled)):
        raise ValueError(f"{source} row {index} has non-string response fields")
    labeled = bool(chosen.strip() and rejected.strip()) and not unlabeled.strip()
    unlabeled_only = bool(unlabeled.strip()) and not chosen.strip() and not rejected.strip()
    if labeled == unlabeled_only:
        raise ValueError(f"{source} row {index} is not an exclusive labeled/unlabeled row")
    return "labeled" if labeled else "unlabeled"


def validated_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    types = [classify(row, index, path) for index, row in enumerate(rows)]
    return rows, types


def select(rows: list[dict[str, Any]], types: list[str], row_type: str, count: int) -> list[dict[str, Any]]:
    selected = [dict(row) for row, observed in zip(rows, types) if observed == row_type][:count]
    if len(selected) != count:
        raise ValueError(f"Need {count} {row_type} rows, found only {len(selected)}")
    return selected


def training_config(
    *,
    method: str,
    model_path: Path,
    smoke_data_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    per_device_train = 1 if method == "dpo" else 4
    config: dict[str, Any] = {
        "model_name_or_path": str(model_path),
        "trust_remote_code": True,
        "stage": "dpo",
        "do_train": True,
        "do_eval": True,
        "finetuning_type": "lora",
        "lora_rank": 8,
        "lora_target": "all",
        "dataset": f"round4_smoke_{method}_train",
        "eval_dataset": "round4_smoke_eval",
        "dataset_dir": str(smoke_data_dir),
        "template": "qwen3",
        "cutoff_len": 1024,
        "max_samples": 10000000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 4,
        "val_size": 0.0,
        "pref_loss": "sigmoid" if method == "dpo" else method,
        "pref_beta": 10.0 if method == "staticpe" else 0.1,
        "output_dir": str(output_dir),
        "overwrite_output_dir": False,
        "max_steps": 2,
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
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": 2,
        "save_total_limit": 1,
        "eval_strategy": "steps",
        "eval_steps": 1,
        "load_best_model_at_end": False,
        "plot_loss": False,
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
                "sspo_min_labeled_per_batch": 2,
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
                "staticpe_min_labeled_per_batch": 2,
                "staticpe_min_unlabeled_per_batch": 2,
                "pe_contract": "simpo_single_response_ema_v1",
            }
        )
    elif method == "frozenpe":
        config.update(
            {
                "frozenpe_lambda": 0.1,
                "frozenpe_epsilon": 1.0e-8,
                "frozenpe_min_labeled_per_batch": 2,
                "frozenpe_min_unlabeled_per_batch": 2,
                "pe_contract": "dpo_frozen_pair_v1",
            }
        )
    return config


def export_config(*, model_path: Path, adapter_dir: Path, export_dir: Path) -> dict[str, Any]:
    return {
        "model_name_or_path": str(model_path),
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


def assert_config_contract(method: str, config: dict[str, Any]) -> None:
    expected_train = 1 if method == "dpo" else 4
    expected_loss = "sigmoid" if method == "dpo" else method
    checks = {
        "max_steps": 2,
        "per_device_train_batch_size": expected_train,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "pref_loss": expected_loss,
        "eval_steps": 1,
    }
    for key, expected in checks.items():
        if config.get(key) != expected:
            raise AssertionError(f"{method} {key}: expected {expected!r}, got {config.get(key)!r}")
    effective = expected_train * 2 * config["gradient_accumulation_steps"]
    expected_effective = 16 if method == "dpo" else 64
    if effective != expected_effective:
        raise AssertionError(f"{method} effective batch changed: {effective}")


def main() -> None:
    args = parse_args()
    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in args.run_id):
        raise ValueError("--run-id may contain only letters, digits, '-' and '_'")
    if len(args.code_commit) != 40 or any(character not in "0123456789abcdef" for character in args.code_commit):
        raise ValueError("--code-commit must be a full lowercase Git SHA")
    counts = [args.dpo_labeled_rows, args.mixed_labeled_rows, args.mixed_unlabeled_rows, args.eval_rows]
    if any(count <= 0 for count in counts):
        raise ValueError("All fixture row counts must be positive")
    if args.dpo_labeled_rows > args.mixed_labeled_rows:
        raise ValueError("DPO smoke rows must be a subset of the mixed labeled rows")

    prepared_dir = absolute_path(args.prepared_data_dir, "prepared data", must_exist=True)
    smoke_data_dir = absolute_path(args.smoke_data_dir, "smoke data")
    run_root = absolute_path(args.run_root, "run root")
    export_root = absolute_path(args.export_root, "export root")
    model_path = absolute_path(args.model_path, "model path", must_exist=True)
    model_manifest_path = model_path / "ROUND4_ASSET_MANIFEST.json"
    model_manifest = read_json(model_manifest_path)
    if model_manifest.get("repo_id") != "Qwen/Qwen3-1.7B" or model_manifest.get("resolved_revision") != args.model_revision:
        raise RuntimeError("Smoke model directory is not the frozen Qwen3 Round4 revision")
    ensure_new_directory(smoke_data_dir, "smoke data directory")
    ensure_new_directory(run_root, "run directory")
    ensure_new_directory(export_root, "export directory")

    dpo_source = prepared_dir / "ultrafeedback_fb0.1_dpo.json"
    mixed_source = prepared_dir / "ultra_combined_fb0.1_ch0.1.json"
    eval_source = prepared_dir / "ultrafeedback_round4_eval.json"
    dpo_rows, dpo_types = validated_rows(dpo_source)
    mixed_rows, mixed_types = validated_rows(mixed_source)
    eval_rows, eval_types = validated_rows(eval_source)
    if set(dpo_types) != {"labeled"} or set(eval_types) != {"labeled"}:
        raise ValueError("DPO and eval source views must be label-only")

    shared_labeled = select(dpo_rows, dpo_types, "labeled", args.mixed_labeled_rows)
    unlabeled = select(mixed_rows, mixed_types, "unlabeled", args.mixed_unlabeled_rows)
    mixed_fixture = shared_labeled + unlabeled
    random.Random(args.seed).shuffle(mixed_fixture)
    fixture_payloads = {
        "round4_smoke_dpo_train": shared_labeled[: args.dpo_labeled_rows],
        "round4_smoke_sspo_train": mixed_fixture,
        "round4_smoke_staticpe_train": mixed_fixture,
        "round4_smoke_frozenpe_input": mixed_fixture,
        "round4_smoke_eval": select(eval_rows, eval_types, "labeled", args.eval_rows),
    }
    fixture_files = {
        "round4_smoke_dpo_train": "dpo_train.json",
        "round4_smoke_sspo_train": "sspo_train.json",
        "round4_smoke_staticpe_train": "staticpe_train.json",
        "round4_smoke_frozenpe_input": "frozenpe_input.json",
        "round4_smoke_eval": "eval.json",
    }
    dataset_info: dict[str, Any] = {}
    for name, payload in fixture_payloads.items():
        file_name = fixture_files[name]
        atomic_json(smoke_data_dir / file_name, payload)
        dataset_info[name] = {
            "file_name": file_name,
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "chosen": "chosen",
                "rejected": "rejected",
                "unlabeled": "unlabeled",
            },
        }
    atomic_json(smoke_data_dir / "dataset_info.json", dataset_info)

    configs_dir = run_root / "configs"
    configs_dir.mkdir()
    method_records: dict[str, Any] = {}
    for method in METHODS:
        adapter_dir = run_root / method / "adapter"
        merged_dir = export_root / method / "merged"
        train_config = training_config(
            method=method,
            model_path=model_path,
            smoke_data_dir=smoke_data_dir,
            output_dir=adapter_dir,
        )
        assert_config_contract(method, train_config)
        train_path = configs_dir / f"{method}_train.json"
        export_path = configs_dir / f"{method}_export.json"
        atomic_json(train_path, train_config)
        atomic_json(
            export_path,
            export_config(model_path=model_path, adapter_dir=adapter_dir, export_dir=merged_dir),
        )
        method_records[method] = {
            "train_config": str(train_path),
            "export_config": str(export_path),
            "adapter_dir": str(adapter_dir),
            "merged_dir": str(merged_dir),
            "effective_batch": train_config["per_device_train_batch_size"] * 2 * 8,
        }

    source_records = {}
    for label, path, rows, types in (
        ("dpo", dpo_source, dpo_rows, dpo_types),
        ("mixed", mixed_source, mixed_rows, mixed_types),
        ("eval", eval_source, eval_rows, eval_types),
    ):
        source_records[label] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "labeled_rows": types.count("labeled"),
            "unlabeled_rows": types.count("unlabeled"),
        }
    fixture_records = {}
    for name, file_name in fixture_files.items():
        path = smoke_data_dir / file_name
        payload = fixture_payloads[name]
        fixture_records[name] = {"path": str(path), "sha256": sha256_file(path), "rows": len(payload)}

    manifest = {
        "schema": "round4-smoke-plan-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "model": {
            "path": str(model_path),
            "revision": args.model_revision,
            "asset_manifest_sha256": sha256_file(model_manifest_path),
        },
        "contract": {
            "optimizer_steps": 2,
            "gpu_count": 2,
            "gradient_accumulation_steps": 8,
            "eval_every_optimizer_steps": 1,
            "seed": args.seed,
            "formal_batch_settings_preserved": True,
            "smoke_two_stream_minima": {"labeled": 2, "unlabeled": 2},
        },
        "sources": source_records,
        "fixtures": fixture_records,
        "methods": method_records,
    }
    if any(not math.isfinite(float(item["rows"])) for item in fixture_records.values()):
        raise AssertionError("Non-finite fixture count")
    atomic_json(run_root / "SMOKE_PLAN.json", manifest)
    print(json.dumps({"run_id": args.run_id, "fixtures": fixture_records, "methods": method_records}, indent=2))


if __name__ == "__main__":
    main()
