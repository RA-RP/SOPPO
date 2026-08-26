"""Server-only Round3 provenance, isolation, hardware, and storage gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

import jsonlines
import torch

from ..model.model_manifest import verify_manifest
from ..model.model_utils import load_tokenizer
from .config import DYNAMIC_METHODS, load_round3_config, validate_round3_config
from .data import (
    MALFORMED_SOURCE_ROWS,
    NAMESPACES,
    SEED,
    SOURCE_AUDIT_CONTRACT,
    SOURCE_MANIFEST_ROWS,
    TOKENIZATION_CONTRACT,
    VIEW_COUNTS,
    canonical_text,
    file_sha256,
    sha256_text,
)
from .queue_protocol import canonical_json


def _git_evidence(repo: Path, expected_commit: str) -> Dict[str, Any]:
    actual = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if actual != expected_commit or dirty:
        raise RuntimeError("Round3 preflight requires the exact clean reviewed Git commit")
    return {"commit": actual, "clean": True}


def _rows(path: Path) -> List[Dict[str, Any]]:
    with jsonlines.open(path) as reader:
        return list(reader)


def _source_resolution_evidence(config: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(config["output"]["run_dir"]).resolve()
    evidence_path = run_dir.parents[1] / "source_revisions.json"
    value = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected = {
        "model": (config["model"]["repo_id"], config["model"]["resolved_revision"]),
        "ultrafeedback": (
            config["data"]["ultrafeedback_repo"],
            config["data"]["ultrafeedback_revision"],
        ),
        "ultrachat": (
            config["data"]["ultrachat_repo"],
            config["data"]["ultrachat_revision"],
        ),
    }
    if value.get("schema_version") != "round3.source_revisions.v1" or set(
        value.get("sources", {})
    ) != set(expected):
        raise ValueError("Round3 source-revision evidence schema/inventory mismatch")
    for name, (repo_id, resolved_sha) in expected.items():
        source = value["sources"][name]
        if (
            source.get("repo_id") != repo_id
            or source.get("resolved_sha") != resolved_sha
            or source.get("transport") != "public_git_ls_remote"
            or not isinstance(source.get("requested_ref"), str)
            or not source["requested_ref"].startswith(("refs/heads/", "refs/tags/"))
        ):
            raise ValueError(f"Round3 resolved source evidence mismatch: {name}")
    return {"path": str(evidence_path), "sha256": file_sha256(evidence_path), "sources": expected}


def _data_evidence(config: Dict[str, Any], require_reference_cache: bool) -> Dict[str, Any]:
    root = Path(config["data"]["data_dir"]).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "round3.data_manifest.v2"
        or manifest.get("seed") != SEED
        or manifest.get("namespaces") != NAMESPACES
        or manifest.get("source_audit") != SOURCE_AUDIT_CONTRACT
    ):
        raise ValueError("Round3 data manifest contract/seed/namespaces mismatch")
    repositories = manifest.get("repositories", {})
    if (
        repositories.get("ultrafeedback", {}).get("repo")
        != config["data"]["ultrafeedback_repo"]
        or repositories.get("ultrafeedback", {}).get("revision")
        != config["data"]["ultrafeedback_revision"]
    ):
        raise ValueError("Round3 UltraFeedback revision/manifest mismatch")
    if (
        repositories.get("ultrachat", {}).get("repo") != config["data"]["ultrachat_repo"]
        or repositories.get("ultrachat", {}).get("revision")
        != config["data"]["ultrachat_revision"]
    ):
        raise ValueError("Round3 UltraChat revision/manifest mismatch")
    parquet_evidence = manifest.get("source_parquet_files")
    if not isinstance(parquet_evidence, dict) or set(parquet_evidence) != {"ultrafeedback", "ultrachat"}:
        raise ValueError("Round3 data manifest lacks source parquet evidence")
    for repo_name, entries in parquet_evidence.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Round3 source parquet evidence is empty: {repo_name}")
        for entry in entries:
            path = Path(entry["path"]).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != int(entry["bytes"])
                or file_sha256(path) != entry["sha256"]
            ):
                raise ValueError(f"Round3 source parquet evidence mismatch: {path}")
    source_cache_evidence = manifest.get("source_cache_files")
    if not isinstance(source_cache_evidence, dict) or set(source_cache_evidence) != {
        "train_prefs", "test_prefs", "train_sft"
    }:
        raise ValueError("Round3 data manifest lacks source Arrow-cache evidence")
    for view_name, entries in source_cache_evidence.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Round3 source Arrow-cache evidence is empty: {view_name}")
        for entry in entries:
            path = Path(entry["path"]).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != int(entry["bytes"])
                or file_sha256(path) != entry["sha256"]
            ):
                raise ValueError(f"Round3 source Arrow-cache evidence mismatch: {path}")
    expected = {
        "paired_train_8k.jsonl": VIEW_COUNTS["paired_train_8k"],
        "paired_train_1k.jsonl": VIEW_COUNTS["paired_train_1k"],
        "unpaired_train_7k.jsonl": VIEW_COUNTS["unpaired_train_7k"],
        "validation_1k.jsonl": VIEW_COUNTS["validation"],
        "test.public.jsonl": VIEW_COUNTS["test"],
        "test.private_labels.jsonl": VIEW_COUNTS["test"],
    }
    audit_name = "malformed_source_rows.jsonl"
    if set(manifest.get("files", {})) != set(expected) | {"source_manifest.jsonl", audit_name}:
        raise ValueError("Round3 data manifest file inventory mismatch")
    loaded = {}
    evidence = {}
    for name, count in expected.items():
        path = root / name
        rows = _rows(path)
        if len(rows) != count:
            raise ValueError(f"Round3 data count mismatch for {name}: {len(rows)} != {count}")
        recorded = manifest.get("files", {}).get(name, {})
        digest = file_sha256(path)
        if recorded.get("sha256") != digest or int(recorded.get("rows", -1)) != count:
            raise ValueError(f"Round3 data file/manifest mismatch: {name}")
        loaded[name] = rows
        evidence[name] = {"rows": count, "sha256": digest}
    exact_keys = {
        "paired_train_8k.jsonl": {"sample_id", "prompt", "response_a", "response_b", "label"},
        "paired_train_1k.jsonl": {"sample_id", "prompt", "response_a", "response_b", "label"},
        "unpaired_train_7k.jsonl": {"sample_id", "prompt", "response"},
        "validation_1k.jsonl": {"sample_id", "prompt", "response_a", "response_b", "label"},
        "test.public.jsonl": {"sample_id", "prompt", "response_a", "response_b"},
        "test.private_labels.jsonl": {"sample_id", "label"},
    }
    for name, rows in loaded.items():
        if any(set(row) != exact_keys[name] for row in rows):
            raise ValueError(f"Round3 canonical row schema mismatch: {name}")
    if any(
        int(row["label"]) not in {0, 1}
        for name in ("paired_train_8k.jsonl", "paired_train_1k.jsonl", "validation_1k.jsonl", "test.private_labels.jsonl")
        for row in loaded[name]
    ):
        raise ValueError("Round3 pair/private labels must all be binary")
    if (
        loaded["paired_train_8k.jsonl"][: VIEW_COUNTS["paired_train_1k"]]
        != loaded["paired_train_1k.jsonl"]
    ):
        raise ValueError("Round3 1K labeled view is not the ordered prefix of the 8K master")
    public_ids = [row["sample_id"] for row in loaded["test.public.jsonl"]]
    private_ids = [row["sample_id"] for row in loaded["test.private_labels.jsonl"]]
    if public_ids != private_ids or any("label" in row for row in loaded["test.public.jsonl"]):
        raise ValueError("Round3 independent-test public/private isolation failed")

    audit_path = root / audit_name
    audit_rows = _rows(audit_path)
    recorded_audit = manifest.get("files", {}).get(audit_name, {})
    if (
        len(audit_rows) != MALFORMED_SOURCE_ROWS
        or int(recorded_audit.get("rows", -1)) != MALFORMED_SOURCE_ROWS
        or recorded_audit.get("sha256") != file_sha256(audit_path)
    ):
        raise ValueError("Round3 malformed-source audit count/SHA mismatch")
    audit_keys = {
        "sample_id",
        "source_id",
        "canonical_prompt_sha256",
        "reason_codes",
        "dataset_id",
        "resolved_revision",
        "split",
        "prompt_id",
        "source_row_index",
    }
    audit_key_by_split = {
        (config["data"]["ultrafeedback_repo"], "train_prefs"): "train_prefs",
        (config["data"]["ultrafeedback_repo"], "test_prefs"): "test_prefs",
        (config["data"]["ultrachat_repo"], "train_sft"): "train_sft",
    }
    observed_audit = {
        key: {"rows": 0, "reasons": Counter()} for key in SOURCE_AUDIT_CONTRACT
    }
    observed_order = []
    for row in audit_rows:
        if set(row) != audit_keys:
            raise ValueError("Round3 malformed-source audit schema mismatch")
        key = audit_key_by_split.get((row["dataset_id"], row["split"]))
        if key is None:
            raise ValueError("Round3 malformed-source audit dataset/split mismatch")
        expected_revision = (
            config["data"]["ultrafeedback_revision"]
            if row["dataset_id"] == config["data"]["ultrafeedback_repo"]
            else config["data"]["ultrachat_revision"]
        )
        reasons = row["reason_codes"]
        prompt_hash = row["canonical_prompt_sha256"]
        if (
            row["resolved_revision"] != expected_revision
            or not isinstance(row["source_row_index"], int)
            or isinstance(row["source_row_index"], bool)
            or row["source_row_index"] < 0
            or not isinstance(row["prompt_id"], str)
            or not isinstance(reasons, list)
            or not reasons
            or reasons != sorted(set(reasons))
            or any(not isinstance(reason, str) or not reason for reason in reasons)
            or (
                prompt_hash is not None
                and (
                    not isinstance(prompt_hash, str)
                    or len(prompt_hash) != 64
                    or any(character not in "0123456789abcdef" for character in prompt_hash)
                )
            )
        ):
            raise ValueError("Round3 malformed-source audit field mismatch")
        provenance = tuple(
            str(row[name])
            for name in (
                "dataset_id",
                "resolved_revision",
                "split",
                "prompt_id",
                "source_row_index",
            )
        )
        if row["sample_id"] != sha256_text("\0".join(provenance)):
            raise ValueError("Round3 malformed-source sample ID mismatch")
        if row["source_id"] != ":".join(provenance):
            raise ValueError("Round3 malformed-source source ID mismatch")
        observed_audit[key]["rows"] += 1
        observed_audit[key]["reasons"].update(reasons)
        observed_order.append((list(SOURCE_AUDIT_CONTRACT).index(key), row["source_row_index"]))
    if observed_order != sorted(observed_order):
        raise ValueError("Round3 malformed-source audit order changed")
    for key, expected_audit in SOURCE_AUDIT_CONTRACT.items():
        if (
            observed_audit[key]["rows"] != expected_audit["malformed_rows"]
            or dict(sorted(observed_audit[key]["reasons"].items()))
            != expected_audit["malformed_reason_counts"]
        ):
            raise ValueError(f"Round3 malformed-source aggregate mismatch: {key}")
    evidence[audit_name] = {
        "rows": MALFORMED_SOURCE_ROWS,
        "sha256": file_sha256(audit_path),
    }

    source_path = root / "source_manifest.jsonl"
    source_rows = _rows(source_path)
    recorded_source = manifest.get("files", {}).get("source_manifest.jsonl", {})
    if (
        len(source_rows) != SOURCE_MANIFEST_ROWS
        or int(recorded_source.get("rows", -1)) != SOURCE_MANIFEST_ROWS
        or recorded_source.get("sha256") != file_sha256(source_path)
    ):
        raise ValueError("Round3 source manifest count/SHA mismatch")
    view_files = {
        "paired_train_8k": "paired_train_8k.jsonl",
        "paired_train_1k": "paired_train_1k.jsonl",
        "unpaired_train_7k": "unpaired_train_7k.jsonl",
        "validation": "validation_1k.jsonl",
        "test": "test.public.jsonl",
    }
    source_by_view = {view: [] for view in view_files}
    expected_source_keys = {
        "view", "sample_id", "source_id", "canonical_prompt_sha256", "dataset_id",
        "resolved_revision", "split", "prompt_id", "source_row_index",
    }
    for row in source_rows:
        if set(row) != expected_source_keys or row.get("view") not in source_by_view:
            raise ValueError("Round3 source manifest schema/view mismatch")
        if (
            not isinstance(row["source_row_index"], int)
            or isinstance(row["source_row_index"], bool)
            or row["source_row_index"] < 0
            or not isinstance(row["prompt_id"], str)
        ):
            raise ValueError("Round3 source manifest row index/prompt ID is malformed")
        provenance = tuple(
            str(row[key])
            for key in ("dataset_id", "resolved_revision", "split", "prompt_id", "source_row_index")
        )
        if row["sample_id"] != sha256_text("\0".join(provenance)):
            raise ValueError("Round3 sample_id/source provenance formula mismatch")
        expected_source_id = ":".join(provenance)
        if row["source_id"] != expected_source_id:
            raise ValueError("Round3 source_id/source provenance mismatch")
        expected_source = {
            "paired_train_8k": (config["data"]["ultrafeedback_repo"], "train_prefs"),
            "paired_train_1k": (config["data"]["ultrafeedback_repo"], "train_prefs"),
            "unpaired_train_7k": (config["data"]["ultrachat_repo"], "train_sft"),
            "validation": (config["data"]["ultrafeedback_repo"], "test_prefs"),
            "test": (config["data"]["ultrafeedback_repo"], "test_prefs"),
        }[row["view"]]
        if (row["dataset_id"], row["split"]) != expected_source:
            raise ValueError("Round3 source view uses the wrong dataset/split")
        if row["dataset_id"] == config["data"]["ultrafeedback_repo"]:
            if (
                row["resolved_revision"] != config["data"]["ultrafeedback_revision"]
                or row["split"] not in {"train_prefs", "test_prefs"}
            ):
                raise ValueError("Round3 UltraFeedback row provenance mismatch")
        elif row["dataset_id"] == config["data"]["ultrachat_repo"]:
            if (
                row["resolved_revision"] != config["data"]["ultrachat_revision"]
                or row["split"] != "train_sft"
            ):
                raise ValueError("Round3 UltraChat row provenance mismatch")
        else:
            raise ValueError("Round3 source manifest contains another dataset")
        source_by_view[row["view"]].append(row)
    for view, filename in view_files.items():
        rows = loaded[filename]
        source_view = source_by_view[view]
        if [row["sample_id"] for row in source_view] != [row["sample_id"] for row in rows]:
            raise ValueError(f"Round3 source manifest order/ID mismatch: {view}")
        if any(
            row["canonical_prompt_sha256"] != sha256_text(canonical_text(public["prompt"]))
            for row, public in zip(source_view, rows)
        ):
            raise ValueError(f"Round3 source manifest prompt hash mismatch: {view}")
    evidence["source_manifest.jsonl"] = {
        "rows": SOURCE_MANIFEST_ROWS,
        "sha256": file_sha256(source_path),
    }
    view_prompts = {
        "paired_train": {canonical_text(row["prompt"]) for row in loaded["paired_train_8k.jsonl"]},
        "unpaired_train": {canonical_text(row["prompt"]) for row in loaded["unpaired_train_7k.jsonl"]},
        "validation": {canonical_text(row["prompt"]) for row in loaded["validation_1k.jsonl"]},
        "test": {canonical_text(row["prompt"]) for row in loaded["test.public.jsonl"]},
    }
    for name, values in view_prompts.items():
        expected_count = {
            "paired_train": VIEW_COUNTS["paired_train_8k"],
            "unpaired_train": VIEW_COUNTS["unpaired_train_7k"],
            "validation": VIEW_COUNTS["validation"],
            "test": VIEW_COUNTS["test"],
        }[name]
        if len(values) != expected_count:
            raise ValueError(f"Round3 canonical prompt duplicates remain in {name}")
    names = list(view_prompts)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = view_prompts[left] & view_prompts[right]
            if overlap:
                raise ValueError(f"Round3 canonical prompt leakage: {left}/{right}={len(overlap)}")
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "files": evidence,
        "prompt_isolation": "exact_zero_overlap",
        "reference_cache_required": require_reference_cache,
    }
    if not require_reference_cache:
        return result
    cache_root = Path(config["data"]["reference_cache_dir"]).resolve()
    cache_manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if cache_manifest.get("git_commit") != config["provenance"]["git_commit"]:
        raise ValueError("Round3 reference cache Git commit mismatch")
    model_manifest = Path(config["model"]["manifest_path"]).resolve()
    if (
        Path(cache_manifest.get("model_path", "")).resolve()
        != Path(config["model"]["name_or_path"]).resolve()
        or Path(cache_manifest.get("model_manifest", "")).resolve() != model_manifest
        or cache_manifest.get("model_manifest_sha256") != file_sha256(model_manifest)
        or cache_manifest.get("tokenization_contract") != TOKENIZATION_CONTRACT
    ):
        raise ValueError("Round3 reference cache model/tokenization provenance mismatch")
    cache_specs = (
        (
            "paired_train_8k.reference.jsonl",
            "paired_train_8k.jsonl",
            VIEW_COUNTS["paired_train_8k"],
        ),
        (
            "paired_train_1k.reference.jsonl",
            "paired_train_1k.jsonl",
            VIEW_COUNTS["paired_train_1k"],
        ),
        (
            "validation_1k.reference.jsonl",
            "validation_1k.jsonl",
            VIEW_COUNTS["validation"],
        ),
        ("test.reference.jsonl", "test.public.jsonl", VIEW_COUNTS["test"]),
    )
    if set(cache_manifest.get("files", {})) != {spec[0] for spec in cache_specs}:
        raise ValueError("Round3 reference-cache manifest file inventory mismatch")
    for name, input_name, count in cache_specs:
        rows = _rows(cache_root / name)
        if len(rows) != count:
            raise ValueError(f"Round3 reference-cache count mismatch: {name}")
        if (
            [row.get("sample_id") for row in rows]
            != [row["sample_id"] for row in loaded[input_name]]
            or any(set(row) != {"sample_id", "ref_logp_a", "ref_logp_b"} for row in rows)
            or any(
                not math.isfinite(float(row[key]))
                for row in rows
                for key in ("ref_logp_a", "ref_logp_b")
            )
        ):
            raise ValueError(f"Round3 reference-cache rows/order/scores mismatch: {name}")
        recorded = cache_manifest.get("files", {}).get(name, {})
        if (
            recorded.get("output_sha256") != file_sha256(cache_root / name)
            or recorded.get("input_sha256") != evidence[input_name]["sha256"]
            or int(recorded.get("rows", -1)) != count
        ):
            raise ValueError(f"Round3 reference-cache provenance mismatch: {name}")
    result["reference_cache_manifest"] = str(cache_root / "manifest.json")
    result["reference_cache_manifest_sha256"] = file_sha256(cache_root / "manifest.json")
    return result


def _gpu_evidence() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0,1,2":
        raise RuntimeError("Round3 preflight requires CUDA_VISIBLE_DEVICES=0,1,2")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError("Round3 preflight requires exactly three visible GPUs")
    devices = []
    for index in range(3):
        properties = torch.cuda.get_device_properties(index)
        if "4090" not in properties.name or properties.total_memory < 23 * 1024**3:
            raise RuntimeError(f"Round3 requires RTX 4090-class 24GB GPU, got {properties.name}")
        devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
        )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if processes:
        raise RuntimeError("Round3 preflight found existing GPU compute processes; it will not stop them")
    return {"cuda_visible_devices": visible, "torch_cuda": torch.version.cuda, "devices": devices, "compute_processes": []}


def _storage_evidence(config: Dict[str, Any], global_evidence_path: Path | None) -> Dict[str, Any]:
    run_dir = Path(config["output"]["run_dir"]).resolve()
    probe = run_dir.parent
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    projected = config["storage"].get("projected_peak_bytes")
    mode = config["execution"]["mode"]
    if mode == "formal":
        required = 2 * int(projected)
        if global_evidence_path is None or not global_evidence_path.is_file():
            raise FileNotFoundError("Formal method preflight requires one-time storage-gate evidence")
        global_evidence = json.loads(global_evidence_path.read_text(encoding="utf-8"))
        if (
            global_evidence.get("schema_version") != "round3.formal_storage_gate.v1"
            or global_evidence.get("status") != "passed"
            or global_evidence.get("experiment_id") != config["provenance"]["experiment_id"]
            or int(global_evidence.get("projected_peak_bytes", -1)) != int(projected)
            or int(global_evidence.get("required_free_bytes", -1)) != required
            or int(global_evidence.get("free_bytes_at_gate", -1)) < required
        ):
            raise ValueError("Formal Round3 storage-gate evidence is inconsistent")
    else:
        required = None
    return {
        "filesystem_probe": str(probe),
        "free_bytes": usage.free,
        "total_bytes": usage.total,
        "projected_peak_bytes": projected,
        "required_free_bytes": required,
        "multiplier": 2.0,
        "automatic_deletion": False,
        "one_time_gate_evidence": str(global_evidence_path) if global_evidence_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--global-storage-evidence")
    parser.add_argument("--phase", choices=("inputs", "method"), default="method")
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    repo = Path(args.repo_root).resolve()
    model_dir = Path(config["model"]["name_or_path"]).resolve()
    model_manifest = Path(config["model"]["manifest_path"]).resolve()
    verify_manifest(model_dir, model_manifest)
    recorded_model_manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    if recorded_model_manifest.get("resolved_revision") != config["model"]["resolved_revision"]:
        raise ValueError("Round3 model resolved revision/config mismatch")
    tokenizer = load_tokenizer(str(model_dir))
    recorded_special_tokens = recorded_model_manifest.get("special_tokens")
    actual_special_tokens = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if recorded_special_tokens != actual_special_tokens:
        raise ValueError("Round3 tokenizer special-token IDs differ from model manifest")
    if tokenizer.pad_token_id != int(config["rollout"]["pad_token_id"]):
        raise ValueError("Round3 tokenizer pad_token_id differs from rollout contract")
    if tokenizer.eos_token_id not in set(config["rollout"]["eos_token_id"]):
        raise ValueError("Round3 tokenizer eos_token_id differs from rollout contract")
    evidence = {
        "schema_version": "round3.preflight.v1",
        "phase": args.phase,
        "experiment_id": config["provenance"]["experiment_id"],
        "method_id": config["method"]["name"],
        "execution_mode": config["execution"]["mode"],
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "git": _git_evidence(repo, config["provenance"]["git_commit"]),
        "model": {
            "path": str(model_dir),
            "manifest": str(model_manifest),
            "manifest_sha256": file_sha256(model_manifest),
            "resolved_revision": config["model"]["resolved_revision"],
        },
        "source_resolution": _source_resolution_evidence(config),
        "data": _data_evidence(config, require_reference_cache=args.phase == "method"),
        "gpus": _gpu_evidence(),
        "storage": _storage_evidence(
            config,
            Path(args.global_storage_evidence).resolve()
            if args.global_storage_evidence
            else None,
        ),
        "gpu_roles": {
            "training": [0],
            "rollout": [1, 2] if config["method"]["name"] in DYNAMIC_METHODS else [],
        },
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 preflight evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
