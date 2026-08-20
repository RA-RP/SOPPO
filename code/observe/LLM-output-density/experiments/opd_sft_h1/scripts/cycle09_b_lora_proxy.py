#!/usr/bin/env python3
"""Cycle 09 B0/B1 provenance inventory for the LoRA proxy program.

This CPU-only entry point freezes the two weight objects and the matched-top-k
contract before any B2/B3/B4 GPU work begins.  It intentionally does not
construct a Qwen effective difference when a direct adapter is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
AUTODL = Path("/root/autodl-tmp")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

import sys

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402


K_VALUES = (4, 8, 16, 32)
PROBES = ("E_general", "E_math", "E_ood", "E_if")
SPECS = {
    "qwen": {
        "layer": 18,
        "arms": ("opd", "offkd"),
        "steps": (20, 160, 624),
    },
    "llama": {
        "layer": 14,
        "arms": ("opd", "offkd"),
        "steps": (20, 160, 320),
    },
}
MODULES = ("self_attn.o_proj", "mlp.down_proj")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row}) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_fingerprint(path: Path) -> str | None:
    """Cheap, explicit pre-M6 fingerprint; never label it as a content SHA-256."""
    if not path.exists():
        return None
    entries: list[tuple[str, int, int]] = []
    if path.is_file():
        stat = path.stat()
        entries.append((path.name, stat.st_size, stat.st_mtime_ns))
    else:
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            stat = child.stat()
            entries.append((str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size).decode("utf-8"))


def safetensor_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": str(path) if path else None, "status": "NOT_FOUND"}
    try:
        header = safetensor_header(path)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if not tensors:
            raise ValueError("no tensors in header")
        key, value = min(
            tensors.items(),
            key=lambda item: (int(__import__("math").prod(item[1].get("shape", [1]))), item[0]),
        )
        return {
            "path": str(path), "status": "AVAILABLE", "dtype": value.get("dtype"),
            "sample_key": key, "tensor_count": len(tensors),
            "metadata": header.get("__metadata__", {}),
        }
    except Exception as error:
        return {"path": str(path), "status": "HEADER_ERROR", "error": repr(error)}


def safetensor_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file() and path.suffix == ".safetensors":
        return [path]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        payload = json.loads(index.read_text())
        names = sorted(set(payload.get("weight_map", {}).values()))
        return [path / name for name in names if (path / name).is_file()]
    return sorted(path.glob("*.safetensors"))


def merged_summary(path: Path) -> dict[str, Any]:
    files = safetensor_files(path)
    if not files:
        return {"path": str(path), "status": "NOT_FOUND"}
    summary = safetensor_summary(files[0])
    summary.update({
        "model_path": str(path),
        "shard_count": len(files),
        "metadata_fingerprint": metadata_fingerprint(path),
        "content_sha256": "PENDING_FULL_CONTENT_SHA256_AFTER_M6",
    })
    return summary


def find_adapter_file(path: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def adapter_summary(path: Path | None, layer: int, module: str) -> dict[str, Any]:
    if path is None or not path.is_dir():
        return {"path": str(path) if path else None, "status": "NOT_FOUND"}
    config_path = path / "adapter_config.json"
    weights = find_adapter_file(path)
    if not config_path.is_file() or weights is None:
        return {
            "path": str(path), "status": "INCOMPLETE",
            "config_exists": config_path.is_file(), "weights": str(weights) if weights else None,
        }
    config = json.loads(config_path.read_text())
    result: dict[str, Any] = {
        "path": str(path), "status": "AVAILABLE", "weights": str(weights),
        "config_sha256": sha256_file(config_path), "weights_sha256": sha256_file(weights),
        "rank": config.get("r"), "alpha": config.get("lora_alpha"),
        "scaling": (float(config["lora_alpha"]) / float(config["r"])
                    if config.get("r") not in (None, 0) and config.get("lora_alpha") is not None else None),
        "metadata_fingerprint": metadata_fingerprint(path),
    }
    if weights.suffix != ".safetensors":
        result.update({"factor_storage_dtype": "UNKNOWN_NON_SAFETENSOR", "module_keys_status": "UNINSPECTED"})
        return result
    header = safetensor_header(weights)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    suffix_a = f"layers.{layer}.{module}.lora_A.weight"
    suffix_b = f"layers.{layer}.{module}.lora_B.weight"
    a_keys = [key for key in tensors if key.endswith(suffix_a)]
    b_keys = [key for key in tensors if key.endswith(suffix_b)]
    factor_dtypes = {tensors[key].get("dtype") for key in a_keys + b_keys}
    result.update({
        "factor_storage_dtype": ",".join(sorted(dtype for dtype in factor_dtypes if dtype)) or None,
        "a_key_count": len(a_keys), "b_key_count": len(b_keys),
        "module_keys_status": "AVAILABLE" if len(a_keys) == len(b_keys) == 1 else "MISSING_OR_AMBIGUOUS",
    })
    if a_keys:
        result["a_key"] = a_keys[0]
    if b_keys:
        result["b_key"] = b_keys[0]
    return result


def qwen_adapter_root(arm: str) -> Path:
    return {
        "opd": AUTODL / "cycle08_opd_trajectory",
        "offkd": AUTODL / "cycle09_offkd",
    }[arm]


def qwen_adapter_path(arm: str, step: int) -> Path | None:
    """Find only serialized PEFT adapters; merged checkpoints never qualify as BA."""
    root = qwen_adapter_root(arm)
    if not root.is_dir():
        return None
    pattern = re.compile(rf"(?:^|[^0-9])0*{int(step)}(?:[^0-9]|$)")
    matches = [candidate.parent for candidate in root.rglob("adapter_config.json") if pattern.search(candidate.parent.name)]
    exact = [path for path in matches if re.search(rf"(?:checkpoint[-_]|step[_-]?)0*{int(step)}$", path.name)]
    found = sorted(exact or matches)
    return found[0] if len(found) == 1 else None


def source_paths(model: str, arm: str, step: int) -> tuple[Path, Path | None]:
    if model == "llama":
        return lexport.merged_target(arm, step), lexport.adapter_target(arm, step)
    return qstage.model_path(arm, step), qwen_adapter_path(arm, step)


def merge_version(path: Path) -> str:
    for name in ("merge_manifest.json", "merge_provenance.json", "manifest.json"):
        if (path / name).is_file():
            return f"serialized:{name}"
    return "UNVERIFIED_NO_SERIALIZED_MERGE_VERSION"


def provenance_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, spec in SPECS.items():
        for arm in spec["arms"]:
            for step in spec["steps"]:
                merged_path, adapter_path = source_paths(model, arm, step)
                merged = merged_summary(merged_path)
                for module in MODULES:
                    adapter = adapter_summary(adapter_path, spec["layer"], module)
                    common = {
                        "model": model, "arm": arm, "checkpoint": step, "layer": spec["layer"],
                        "module": module, "probes": ";".join(PROBES),
                        "merged_checkpoint_path": str(merged_path),
                        "merged_storage_dtype": merged.get("dtype"),
                        "merged_shard_count": merged.get("shard_count"),
                        "merged_metadata_fingerprint": merged.get("metadata_fingerprint"),
                        "merged_content_sha256": merged.get("content_sha256"),
                        "merge_version": merge_version(merged_path),
                        "factor_checkpoint_path": adapter.get("path"),
                        "factor_config_sha256": adapter.get("config_sha256"),
                        "factor_weights_sha256": adapter.get("weights_sha256"),
                        "factor_metadata_fingerprint": adapter.get("metadata_fingerprint"),
                        "factor_storage_dtype": adapter.get("factor_storage_dtype"),
                        "lora_rank": adapter.get("rank"), "lora_alpha": adapter.get("alpha"),
                        "lora_scaling": adapter.get("scaling"),
                        "matmul_dtype": "fp32", "subtraction_dtype": "fp32",
                        "svd_dtype": "fp64", "required_fixed_k": ";".join(map(str, K_VALUES)),
                    }
                    direct_status = ("READY" if adapter.get("status") == "AVAILABLE" and
                                     adapter.get("module_keys_status") == "AVAILABLE" else
                                     "BLOCKED_DIRECT_BA_ARTIFACT_MISSING")
                    rows.append({
                        **common, "weight_object": "direct_BA_from_bf16_factors_fp32_matmul",
                        "object_status": direct_status, "adapter_key_status": adapter.get("module_keys_status"),
                        "checkpoint_hash": adapter.get("weights_sha256"),
                        "checkpoint_hash_status": "EXACT_ADAPTER_CONTENT_SHA256" if adapter.get("weights_sha256") else "UNAVAILABLE",
                    })
                    rows.append({
                        **common, "weight_object": "serialized_merged_bf16_effective_difference",
                        "object_status": "READY" if merged.get("status") == "AVAILABLE" else "BLOCKED_MERGED_CHECKPOINT_MISSING",
                        "adapter_key_status": None,
                        "checkpoint_hash": merged.get("content_sha256"),
                        "checkpoint_hash_status": "PENDING_FULL_CONTENT_SHA256_AFTER_M6",
                    })
    # B4's step-0 G0/S0 cache is a base reference, not an artificial zero-rank adapter.
    base_path = lexport.merged_target("opd", 0)
    base = merged_summary(base_path)
    for module in MODULES:
        rows.append({
            "model": "llama", "arm": "base", "checkpoint": 0, "layer": 14, "module": module,
            "probes": ";".join(PROBES), "weight_object": "base_reference_for_G0_S0",
            "object_status": "READY" if base.get("status") == "AVAILABLE" else "BLOCKED_MERGED_CHECKPOINT_MISSING",
            "merged_checkpoint_path": str(base_path), "merged_storage_dtype": base.get("dtype"),
            "merged_shard_count": base.get("shard_count"), "merged_metadata_fingerprint": base.get("metadata_fingerprint"),
            "merged_content_sha256": base.get("content_sha256"), "merge_version": "base_serialized_checkpoint",
            "factor_checkpoint_path": None, "factor_config_sha256": None, "factor_weights_sha256": None,
            "factor_metadata_fingerprint": None, "factor_storage_dtype": None, "lora_rank": None,
            "lora_alpha": None, "lora_scaling": None, "matmul_dtype": "fp32", "subtraction_dtype": "fp32",
            "svd_dtype": "fp64", "required_fixed_k": ";".join(map(str, K_VALUES)),
            "adapter_key_status": None, "checkpoint_hash": base.get("content_sha256"),
            "checkpoint_hash_status": "PENDING_FULL_CONTENT_SHA256_AFTER_M6",
        })
    return rows


def b0b1() -> dict[str, Any]:
    rows = provenance_rows()
    inventory = MINI / "lora_dual_track_inventory.csv"
    atomic_csv(inventory, rows)
    protocol = {
        "schema_version": "cycle09_b0_dual_weight_object_v1",
        "status": "complete_with_declared_hash_deferrals",
        "created_utc": utc_now(),
        "tracks": {
            "direct_BA_from_bf16_factors_fp32_matmul": {
                "formula": "Delta W = scaling * (B @ A)",
                "factor_policy": "serialized BF16 factors are converted to FP32 before matmul; no FP32 master is implied",
                "matmul_dtype": "fp32", "use": ["adapter_branch", "rollback", "LoRA-rank", "recompression"],
            },
            "serialized_merged_bf16_effective_difference": {
                "formula": "FP32(W_merged_bf16_t) - FP32(W_merged_bf16_0)",
                "subtraction_dtype": "fp32", "use": ["related-work", "serialized-state", "merge-audit"],
            },
        },
        "required_fields": [
            "weight_object", "factor_storage_dtype", "merged_storage_dtype", "matmul_dtype",
            "subtraction_dtype", "svd_dtype", "lora_scaling", "merge_version", "checkpoint_hash",
        ],
        "full_merged_content_hash_policy": "deferred until M6 releases data-disk I/O; metadata fingerprints are explicitly not content SHA-256 values",
        "inventory": str(inventory),
    }
    atomic_json(MINI / "lora_dual_track_protocol.json", protocol)
    matched = {
        "schema_version": "cycle09_b1_matched_topk_v1", "status": "complete",
        "created_utc": utc_now(), "fixed_k": list(K_VALUES),
        "cell_identity": ["model", "arm", "checkpoint", "layer", "module", "probe", "weight_object", "matrix_side", "precision"],
        "rank_limit_policy": "If either object rank < k, record NA_RANK_LIMIT. Never zero-fill or compare unequal k.",
        "adaptive_rank_policy": "r_epsilon is descriptive only and never substitutes for matched fixed-k comparisons.",
        "metrics": ["cosine", "PABS", "left_subspace_overlap", "right_subspace_overlap", "principal_angles", "tail_energy"],
    }
    atomic_json(MINI / "matched_topk_protocol.json", matched)
    registry_rows = []
    for object_name, intended_use, status in (
        ("direct_BA_from_bf16_factors_fp32_matmul", "primary LoRA action / proxy / rollback", "PENDING_B2"),
        ("serialized_merged_bf16_effective_difference", "related-work fixed-k geometry / serialized-state audit", "PENDING_B2"),
        ("related_work_native_k_or_rank_fraction", "native reporting only; not method-comparison headline", "PENDING_SOURCE_REGISTRY"),
        ("adaptive_r_epsilon", "within-method descriptive trajectory only", "NOT_COMPARABLE_AS_MATCHED_K"),
    ):
        registry_rows.append({
            "source_or_weight_object": object_name, "permitted_use": intended_use,
            "comparison_basis": "fixed matched k" if "r_epsilon" not in object_name else "descriptive only",
            "status": status, "fixed_k": ";".join(map(str, K_VALUES)),
        })
    atomic_csv(MINI / "related_work_native_vs_matched_k_registry.csv", registry_rows)
    blocked = [row for row in rows if row["weight_object"].startswith("direct_BA") and row["object_status"] != "READY"]
    manifest = {
        "schema_version": "cycle09_b0b1_manifest_v1", "status": "complete_with_declared_blocks",
        "created_utc": utc_now(), "inventory_rows": len(rows), "blocked_direct_ba_rows": len(blocked),
        "outputs": {
            "inventory": str(inventory), "dual_track_protocol": str(MINI / "lora_dual_track_protocol.json"),
            "matched_topk_protocol": str(MINI / "matched_topk_protocol.json"),
            "registry": str(MINI / "related_work_native_vs_matched_k_registry.csv"),
        },
        "blocked_direct_ba_cells": [
            {key: row[key] for key in ("model", "arm", "checkpoint", "layer", "module", "object_status")}
            for row in blocked
        ],
        "next_action": "Run exact full content hashes after M6 and resolve any direct-BA artifact gaps before B2 measurement.",
    }
    atomic_json(MINI / "lora_b0b1_manifest.json", manifest)
    return manifest


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("b0b1",))
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "b0b1":
        result = b0b1()
    else:
        raise AssertionError(args.command)
    print(json.dumps({"status": result["status"], "inventory_rows": result["inventory_rows"],
                      "blocked_direct_ba_rows": result["blocked_direct_ba_rows"]}, indent=2))


if __name__ == "__main__":
    main()
