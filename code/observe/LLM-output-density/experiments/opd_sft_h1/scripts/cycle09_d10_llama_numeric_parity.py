#!/usr/bin/env python3
"""D10 Llama matched-numeric state/output parity.

This is the Llama analogue of ``cycle09_qwen_d4_merged_state.py``.  It uses
the same frozen probe corpora as the existing Llama D8 geometry, but reruns the
headline L14 cells with the D10 numeric protocol: BF16 load/forward, FP32 Gram
accumulation, FP64 Gram eig/SVD, and fresh BF16 fixed-token output caches.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_actual_output_trajectory as actual  # noqa: E402
import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_geometry as lgeom  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402


ROOT = b3.AUTODL / "cycle09_relative_functional_contraction/d10_llama_numeric_parity"
FORMAL = ROOT / "formal"
FINAL = FORMAL / "final"
SCRATCH = FORMAL / "scratch"
MINI = b3.MINI

ARMS = tuple(b3.ARMS)
CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")
STEPS = (5, 20, 40, 80, 160, 320)
LAYER = 14
MODULES = tuple(b3.MODULES)
EPSILONS = (0.01, 0.025, 0.05, 0.10)
REQUIRED_PROTOCOL = (
    "checkpoint_storage_dtype",
    "merge_compute_dtype",
    "model_load_dtype",
    "activation_dtype",
    "gram_accumulation_dtype",
    "gram_factorization_dtype",
    "gram_factorization_method",
    "WS_matmul_dtype",
    "svd_input_dtype",
    "singular_value_accumulation_dtype",
    "logit_forward_dtype",
    "logit_storage_dtype",
    "KL_NLL_compute_dtype",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label(step: int) -> str:
    return f"step_{int(step):03d}"


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            value.update(part)
    return value.hexdigest()


def json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@contextmanager
def lock(path: Path):
    import fcntl

    target = path.with_suffix(path.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def state_path(tag: str, arm: str, step: int, probe: str) -> Path:
    return ROOT / tag / "state" / arm / label(step) / f"{probe}.json"


def output_path(tag: str, arm: str, step: int, probe: str) -> Path:
    return ROOT / tag / "outputs" / arm / label(step) / f"{probe}.json"


def profile_path(tag: str, arm: str, step: int, probe: str) -> Path:
    return ROOT / tag / "profiles" / arm / label(step) / f"{probe}.pt"


def profile_meta_path(tag: str, arm: str, step: int, probe: str) -> Path:
    return profile_path(tag, arm, step, probe).with_suffix(".json")


def base_cache_path(tag: str, probe: str) -> Path:
    return ROOT / tag / "scratch/base_logits" / f"{probe}.pt"


def base_cache_meta(tag: str, probe: str) -> Path:
    return base_cache_path(tag, probe).with_suffix(".json")


def legacy_manifest() -> dict[str, Any]:
    path = b3.AUTODL / "cycle09_block3/llama_geometry/corpora/probe_manifest.json"
    return read_json(path, {}) | {
        "manifest_path": str(path),
        "manifest_sha256": digest(path) if path.is_file() else None,
    }


def model_dir(arm: str, step: int) -> Path:
    if step == 0:
        check = b3.model_check(b3.LLAMA_STUDENT)
        if not check["complete"]:
            raise FileNotFoundError(f"missing/incomplete Llama BF16 base model: {b3.LLAMA_STUDENT}; {check['error']}")
        return b3.LLAMA_STUDENT
    path = lexport.merged_target(arm, step)
    check = b3.model_check(path)
    if not check["complete"]:
        raise FileNotFoundError(f"missing/incomplete serialized Llama BF16 merged checkpoint: {path}; {check['error']}")
    return path


def load_bf16(path: Path, device: str) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval().to(device)
    return model


def unload(model: Any) -> None:
    if model is not None:
        model.to("cpu")
        del model
    torch.cuda.empty_cache()


def samples_for(probe: str, sample_limit: int) -> list[Any]:
    if probe not in CORE_PROBES:
        raise ValueError(f"unsupported D10 core probe: {probe}")
    # The frozen D8 probe corpus already contains full_token_ids and window
    # boundaries; c4.prepare_samples does not re-tokenize these rows.
    samples = lgeom.prepare_samples(None, probe, sample_limit)
    if not samples:
        raise RuntimeError(f"empty Llama probe after preparation: {probe}")
    return samples


def sample_summary(probe: str, samples: list[Any]) -> dict[str, Any]:
    ids = [sample.sample_id for sample in samples]
    manifest = legacy_manifest()
    corpus = lgeom.corpus_path(probe)
    return {
        "probe": probe,
        "sample_count": len(samples),
        "sample_ids_sha256": json_digest(ids),
        "sample_ids": ids,
        "corpus_path": str(corpus),
        "corpus_sha256": digest(corpus) if corpus.is_file() else None,
        "legacy_probe_manifest_path": manifest.get("manifest_path"),
        "legacy_probe_manifest_sha256": manifest.get("manifest_sha256"),
        "legacy_sample_ids_sha256_status": "reconstructed_from_D8_corpus_and_tokenizer",
        "window_seed": 42,
        "window_protocol": manifest.get("window_protocol"),
    }


def ranks(singular: torch.Tensor) -> dict[str, int]:
    energy = singular.to(torch.float64).square()
    total = energy.sum()
    if float(total) == 0.0:
        return {str(epsilon): 0 for epsilon in EPSILONS}
    cumulative = energy.cumsum(0)
    return {
        str(epsilon): int(torch.searchsorted(cumulative, (1.0 - epsilon) * total).item() + 1)
        for epsilon in EPSILONS
    }


def sqrt_gram(gram: torch.Tensor, device: str) -> tuple[torch.Tensor, dict[str, Any]]:
    symmetric = ((gram + gram.T) / 2).to(device=device, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(symmetric)
    negative = values[values < 0]
    audit = {
        "min_eigenvalue": float(values.min().item()) if values.numel() else 0.0,
        "negative_eigenvalue_count": int(negative.numel()),
        "negative_eigenvalue_mass": float((-negative).sum().item()) if negative.numel() else 0.0,
        "lambda_max": float(values.max().item()) if values.numel() else 0.0,
        "width": int(values.numel()),
    }
    scale = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    return scale.to(dtype=torch.float32), audit


def protocol(arm: str, step: int) -> dict[str, str]:
    merge_route = "shared_base_bf16_model_dir" if step == 0 else "existing_serialized_bf16_merged_checkpoint"
    return {
        "checkpoint_storage_dtype": "bf16_safetensors_merged_state",
        "merge_compute_dtype": merge_route,
        "model_load_dtype": "bf16",
        "activation_dtype": "bf16_forward_hidden_states_cast_to_fp32_for_gram",
        "gram_accumulation_dtype": "fp32",
        "gram_factorization_dtype": "fp64",
        "gram_factorization_method": "symmetric_eigh_clamp_min_0_no_jitter",
        "WS_matmul_dtype": "fp32",
        "svd_input_dtype": "fp64",
        "singular_value_accumulation_dtype": "fp64",
        "logit_forward_dtype": "bf16",
        "logit_storage_dtype": "bf16",
        "KL_NLL_compute_dtype": "fp32_full_vocabulary",
        "arm": arm,
        "checkpoint": str(step),
    }


def profile_state(
    model: Any,
    samples: list[Any],
    device: str,
    batch_size: int,
    max_batch_tokens: int,
) -> dict[str, Any]:
    return campaign.collect_profile(
        model,
        samples,
        [LAYER],
        device,
        keep_factors=False,
        keep_residual_samples=False,
        keep_input_sample_means=True,
        factor_layers=(LAYER,),
        forward_batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        early_stop=True,
    )


def build_base_cache(tag: str, probe: str, samples: list[Any], device: str) -> dict[str, Any]:
    target = base_cache_path(tag, probe)
    meta = base_cache_meta(tag, probe)
    with lock(target):
        cached = read_json(meta, {})
        if cached.get("status") == "complete" and target.is_file():
            return torch.load(target, map_location="cpu", weights_only=True)
        model = load_bf16(model_dir("base", 0), device)
        try:
            records = []
            for sample in samples:
                positions = actual.selected_positions(sample, 0)
                records.append({
                    "sample_id": sample.sample_id,
                    "positions": positions.cpu(),
                    "token_weights": sample.token_weights[positions].float().cpu(),
                    "target_ids": sample.input_ids[0, positions].long().cpu(),
                    "logits": actual.selected_logits(model, sample, positions, device).cpu(),
                })
        finally:
            unload(model)
        summary = sample_summary(probe, samples)
        value = {
            "schema_version": "cycle09_d10_llama_bf16_base_logits_v1",
            "status": "complete",
            "model": "llama",
            "arm": "base",
            "checkpoint": 0,
            "probe_name": probe,
            "records": records,
            "logit_forward_dtype": "bf16",
            "logit_storage_dtype": "bf16",
            "created_utc": now(),
            **{key: value for key, value in summary.items() if key != "sample_ids"},
        }
        atomic_torch(target, value)
        atomic_json(meta, {
            "schema_version": value["schema_version"],
            "status": "complete",
            "model": "llama",
            "probe_name": probe,
            "sample_count": value["sample_count"],
            "sample_ids_sha256": value["sample_ids_sha256"],
            "logit_forward_dtype": value["logit_forward_dtype"],
            "logit_storage_dtype": value["logit_storage_dtype"],
            "artifact": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest(target),
            "created_utc": value["created_utc"],
        })
        return value


def output_rows(model: Any, samples: list[Any], baseline: dict[str, Any], device: str) -> list[dict[str, Any]]:
    source = {sample.sample_id: sample for sample in samples}
    rows: list[dict[str, Any]] = []
    for record in baseline["records"]:
        sample = source[record["sample_id"]]
        positions = record["positions"].long()
        base_logits = record["logits"].to(device=device, dtype=torch.float32)
        current_logits = actual.selected_logits(model, sample, positions, device).to(device=device, dtype=torch.float32)
        log_base = torch.log_softmax(base_logits, dim=-1)
        log_current = torch.log_softmax(current_logits, dim=-1)
        probability = log_base.exp()
        weights = record["token_weights"].to(device=device, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-30)
        token_ids = record["target_ids"].to(device=device, dtype=torch.long)
        kl = (probability * (log_base - log_current)).sum(dim=-1)
        nll_base = -log_base.gather(1, token_ids[:, None]).squeeze(1)
        nll_current = -log_current.gather(1, token_ids[:, None]).squeeze(1)
        delta = nll_current - nll_base
        rows.append({
            "sample_id": record["sample_id"],
            "token_count": int(len(positions)),
            "cumulative_kl_base_to_current": float((weights * kl).sum()),
            "nll_base": float((weights * nll_base).sum()),
            "nll_current": float((weights * nll_current).sum()),
            "delta_nll_cumulative": float((weights * delta).sum()),
            "absolute_delta_nll_cumulative": float((weights * delta.abs()).sum()),
        })
        del base_logits, current_logits, log_base, log_current, probability, kl, nll_base, nll_current, delta
        torch.cuda.empty_cache()
    return rows


def run_loaded_cell(
    args: argparse.Namespace,
    model: Any,
    arm: str,
    probe: str,
    samples: list[Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    target = state_path(args.tag, arm, args.step, probe)
    output = output_path(args.tag, arm, args.step, probe)
    with lock(target):
        cached = read_json(target, {})
        if cached.get("status") == "complete" and read_json(output, {}).get("status") == "complete":
            return cached | {"artifact": str(target), "output_artifact": str(output), "cached": True}
        measured = profile_state(model, samples, args.device, args.forward_batch_size, args.max_batch_tokens)
        rows = output_rows(model, samples, baseline, args.device)
        spectra: dict[str, list[float]] = {}
        eig_audit: dict[str, dict[str, Any]] = {}
        state_rows: list[dict[str, Any]] = []
        for module in MODULES:
            group = lgeom.c4.MODULE_TO_GROUP[module]
            scale, audit = sqrt_gram(measured["grams"][LAYER][group], args.device)
            weight = campaign.module_at(model, LAYER, module).weight.detach().to(args.device, dtype=torch.float32)
            singular = torch.linalg.svdvals((weight @ scale).to(torch.float64))
            for epsilon, rank in ranks(singular).items():
                state_rows.append({"module": module, "epsilon": float(epsilon), "r_epsilon": int(rank)})
            spectra[module] = singular.cpu().tolist()
            eig_audit[module] = audit
            del scale, weight, singular
            torch.cuda.empty_cache()
        proto = protocol(arm, args.step)
        summary = sample_summary(probe, samples)
        profile_artifact = profile_path(args.tag, arm, args.step, probe)
        profile_payload = {
            "schema_version": "cycle09_d10_llama_profile_v1",
            "status": "complete",
            "model": "llama",
            "arm": arm,
            "checkpoint": args.step,
            "probe_name": probe,
            "layer": LAYER,
            "sample_count": len(samples),
            "sample_ids_sha256": summary["sample_ids_sha256"],
            "grams": measured["grams"],
            "input_sample_means": measured["input_sample_means"],
            "forward_execution": measured["forward_execution"],
            "created_utc": now(),
        }
        atomic_torch(profile_artifact, profile_payload)
        atomic_json(profile_meta_path(args.tag, arm, args.step, probe), {
            key: profile_payload[key]
            for key in ("schema_version", "status", "model", "arm", "checkpoint", "probe_name", "layer", "sample_count", "sample_ids_sha256", "forward_execution", "created_utc")
        } | {"artifact": str(profile_artifact), "bytes": profile_artifact.stat().st_size, "sha256": digest(profile_artifact)})
        payload = {
            "schema_version": "cycle09_d10_llama_matched_state_v1",
            "status": "complete",
            "model": "llama",
            "arm": arm,
            "checkpoint": args.step,
            "probe_name": probe,
            "layer": LAYER,
            "state_rows": state_rows,
            "spectra": spectra,
            "eigen_audit": eig_audit,
            "numerical_protocol": proto,
            "profile_artifact": str(profile_artifact),
            "created_utc": now(),
            **{key: value for key, value in summary.items() if key != "sample_ids"},
        }
        output_payload = {
            "schema_version": "cycle09_d10_llama_fixed_token_output_v1",
            "status": "complete",
            "model": "llama",
            "arm": arm,
            "checkpoint": args.step,
            "probe_name": probe,
            "sample_count": len(samples),
            "sample_ids_sha256": summary["sample_ids_sha256"],
            "rows": rows,
            "numerical_protocol": proto,
            "aggregation": "sample_equal_mean_of_token_weighted_rows",
            "created_utc": now(),
        }
        atomic_json(target, payload)
        atomic_json(output, output_payload)
        return payload | {"artifact": str(target), "output_artifact": str(output)}


def prepare_cell(args: argparse.Namespace, arm: str, probe: str) -> tuple[list[Any], dict[str, Any]] | None:
    target = state_path(args.tag, arm, args.step, probe)
    output = output_path(args.tag, arm, args.step, probe)
    if read_json(target, {}).get("status") == "complete" and read_json(output, {}).get("status") == "complete":
        return None
    samples = samples_for(probe, args.sample_limit)
    baseline = build_base_cache(args.tag, probe, samples, args.device)
    if baseline["sample_ids_sha256"] != sample_summary(probe, samples)["sample_ids_sha256"]:
        raise RuntimeError(f"base-logit sample IDs mismatch for {probe}")
    return samples, baseline


def run_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    arm = "base" if args.step == 0 else args.arm
    probes = CORE_PROBES if args.probes == "all" else tuple(item.strip() for item in args.probes.split(",") if item.strip())
    invalid = sorted(set(probes) - set(CORE_PROBES))
    if invalid:
        raise ValueError(f"unsupported D10 probes: {invalid}")
    prepared = [(probe, data) for probe in probes if (data := prepare_cell(args, arm, probe)) is not None]
    if not prepared:
        return {"status": "complete", "arm": arm, "checkpoint": args.step, "cells": [], "cached": True}
    model = load_bf16(model_dir(arm, args.step), args.device)
    try:
        cells = [
            run_loaded_cell(args, model, arm, probe, samples, baseline)
            for probe, (samples, baseline) in prepared
        ]
    finally:
        unload(model)
    return {"status": "complete", "arm": arm, "checkpoint": args.step, "cells": cells, "created_utc": now()}


def expected_cells() -> set[tuple[str, int, str]]:
    return {("base", 0, probe) for probe in CORE_PROBES} | {
        (arm, step, probe) for arm in ARMS for step in STEPS for probe in CORE_PROBES
    }


def read_cell(root: Path, kind: str, arm: str, step: int, probe: str) -> dict[str, Any]:
    path = root / kind / arm / label(step) / f"{probe}.json"
    return read_json(path, {})


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = ROOT / args.tag
    state_cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    output_cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for arm, step, probe in sorted(expected_cells()):
        state_cells[(arm, step, probe)] = read_cell(root, "state", arm, step, probe)
        output_cells[(arm, step, probe)] = read_cell(root, "outputs", arm, step, probe)
    complete = {
        key for key in expected_cells()
        if state_cells.get(key, {}).get("status") == "complete" and output_cells.get(key, {}).get("status") == "complete"
    }
    missing = sorted(expected_cells() - complete)
    if missing:
        raise RuntimeError(f"D10 matrix incomplete: {len(missing)} missing cells; first={missing[:5]}")

    base_rank: dict[tuple[str, str, float], float] = {}
    for probe in CORE_PROBES:
        for row in state_cells[("base", 0, probe)]["state_rows"]:
            base_rank[(probe, row["module"], float(row["epsilon"]))] = float(row["r_epsilon"])

    module_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []
    sample_digests: dict[str, set[str]] = {probe: set() for probe in CORE_PROBES}
    for key in sorted(expected_cells()):
        arm, step, probe = key
        state = state_cells[key]
        output = output_cells[key]
        if state["sample_ids_sha256"] != output["sample_ids_sha256"]:
            raise RuntimeError(f"state/output sample digest mismatch for {key}")
        sample_digests[probe].add(state["sample_ids_sha256"])
        for required in REQUIRED_PROTOCOL:
            if required not in state.get("numerical_protocol", {}):
                raise RuntimeError(f"missing protocol field {required} for {key}")
        for module, audit in state.get("eigen_audit", {}).items():
            protocol_rows.append({
                "model": "llama",
                "arm": arm,
                "checkpoint": step,
                "probe_name": probe,
                "layer": state["layer"],
                "module": module,
                **audit,
                **state["numerical_protocol"],
                "state_artifact": str(root / "state" / arm / label(step) / f"{probe}.json"),
                "output_artifact": str(root / "outputs" / arm / label(step) / f"{probe}.json"),
            })
        for row in state["state_rows"]:
            epsilon = float(row["epsilon"])
            reference = base_rank[(probe, row["module"], epsilon)]
            current = float(row["r_epsilon"])
            absolute = reference - current
            module_rows.append({
                "model": "llama",
                "arm": arm,
                "checkpoint": step,
                "probe_name": probe,
                "layer": state["layer"],
                "module": row["module"],
                "epsilon": epsilon,
                "state_rank_base": reference,
                "state_rank_current": current,
                "state_rank_delta": current - reference,
                "absolute_contraction": absolute,
                "relative_functional_contraction_module": absolute / reference if reference else None,
                "source_name": "llama_d10_matched_numeric",
                "source_protocol": "D10_bf16_forward_fp64_eigh_svd",
                "sample_count": state["sample_count"],
                "sample_ids_sha256": state["sample_ids_sha256"],
            })
        frame = pd.DataFrame(output["rows"])
        means = frame[[
            "cumulative_kl_base_to_current",
            "nll_base",
            "nll_current",
            "delta_nll_cumulative",
            "absolute_delta_nll_cumulative",
        ]].mean(numeric_only=True).to_dict()
        output_rows.append({
            "model": "llama",
            "arm": arm,
            "checkpoint": step,
            "probe_name": probe,
            "sample_count": int(output["sample_count"]),
            "aggregation": "sample_equal_mean_of_token_weighted_rows",
            **means,
        })

    module_frame = pd.DataFrame(module_rows)
    keys = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "source_name", "source_protocol"]
    aggregate = module_frame.groupby(keys, dropna=False).agg(
        module_count=("module", "nunique"),
        state_rank_base_mean=("state_rank_base", "mean"),
        state_rank_current_mean=("state_rank_current", "mean"),
        state_rank_delta_mean=("state_rank_delta", "mean"),
        absolute_contraction_mean=("absolute_contraction", "mean"),
        relative_functional_contraction_equal7=("relative_functional_contraction_module", "mean"),
        sample_count=("sample_count", "first"),
    ).reset_index()
    aggregate["relative_functional_contraction_ratio_of_means_sensitivity"] = (
        (aggregate["state_rank_base_mean"] - aggregate["state_rank_current_mean"])
        / aggregate["state_rank_base_mean"]
    )

    FINAL.mkdir(parents=True, exist_ok=True)
    atomic_csv(FINAL / "llama_matched_state_module_ranks.csv", module_rows)
    atomic_csv(FINAL / "llama_matched_state_equal7.csv", aggregate.to_dict("records"))
    atomic_csv(FINAL / "llama_matched_fixed_token_outputs.csv", output_rows)
    atomic_csv(FINAL / "llama_matched_numeric_protocol_audit.csv", protocol_rows)
    # Matched-numeric names used by downstream analysis without overwriting legacy artifacts.
    atomic_csv(FINAL / "relative_functional_contraction_all_cells_matched_numeric.csv", aggregate.to_dict("records"))
    atomic_csv(FINAL / "relative_contraction_matched_cumulative_outputs_matched_numeric.csv", output_rows)

    parity = legacy_parity(module_rows)
    atomic_csv(FINAL / "llama_legacy_matched_numeric_parity.csv", parity["rows"])
    atomic_csv(FINAL / "llama_legacy_matched_numeric_parity_summary.csv", parity["summary"])

    protocol_payload = {
        "schema_version": "cycle09_d10_llama_numeric_protocol_v1",
        "status": "complete",
        "created_utc": now(),
        "target_cells": len(expected_cells()),
        "complete_cells": len(complete),
        "sample_id_digest_per_probe": {probe: sorted(values) for probe, values in sample_digests.items()},
        "required_protocol_fields": list(REQUIRED_PROTOCOL),
        "artifacts": protocol_rows,
    }
    atomic_json(FINAL / "llama_matched_state_numeric_protocol.json", protocol_payload)
    mirror_outputs(FINAL)
    handoff = write_handoff(args.tag, len(module_rows), len(aggregate), len(output_rows), parity)
    manifest = {
        "schema_version": "cycle09_d10_llama_numeric_parity_manifest_v1",
        "status": "COMPLETE_MATCHED_NUMERIC_PARITY",
        "created_utc": now(),
        "tag": args.tag,
        "target_cells": len(expected_cells()),
        "complete_cells": len(complete),
        "module_rows": len(module_rows),
        "aggregate_rows": len(aggregate),
        "output_rows": len(output_rows),
        "handoff": str(handoff),
        "outputs": [str(FINAL / name) for name in (
            "llama_matched_state_numeric_protocol.json",
            "llama_matched_state_module_ranks.csv",
            "llama_matched_state_equal7.csv",
            "llama_matched_fixed_token_outputs.csv",
            "llama_legacy_matched_numeric_parity.csv",
            "llama_legacy_matched_numeric_parity_summary.csv",
            "relative_functional_contraction_all_cells_matched_numeric.csv",
            "relative_contraction_matched_cumulative_outputs_matched_numeric.csv",
        )],
    }
    atomic_json(FINAL / "d10_llama_numeric_parity_manifest.json", manifest)
    atomic_json(MINI / "d10_llama_numeric_parity_manifest.json", manifest)
    return manifest


def legacy_parity(module_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    legacy_path = b3.AUTODL / "cycle09_block3/llama_geometry/llama_early_320_r_epsilon.csv"
    if not legacy_path.is_file():
        return {
            "rows": [],
            "summary": [{
                "status": "LEGACY_NOT_AVAILABLE",
                "legacy_path": str(legacy_path),
                "created_utc": now(),
            }],
        }
    legacy = pd.read_csv(legacy_path)
    legacy = legacy[
        legacy["probe"].isin(CORE_PROBES)
        & legacy["track"].eq("per_checkpoint")
        & legacy["layer"].eq(LAYER)
    ].copy()
    legacy = legacy.rename(columns={"probe": "probe_name", "step": "checkpoint", "r_epsilon": "module_r_old"})
    matched = pd.DataFrame(module_rows).rename(columns={"state_rank_current": "module_r_matched"})
    joined = matched.merge(
        legacy[["arm", "checkpoint", "probe_name", "layer", "module", "epsilon", "module_r_old"]],
        on=["arm", "checkpoint", "probe_name", "layer", "module", "epsilon"],
        how="left",
    )
    joined["module_rank_difference"] = joined["module_r_matched"] - joined["module_r_old"]
    rows = joined.to_dict("records")
    summary_rows: list[dict[str, Any]] = []
    for arm, group in joined[joined["arm"].isin(ARMS)].groupby("arm", sort=True):
        diff = group["module_rank_difference"].dropna().astype(float)
        summary_rows.append({
            "arm": arm,
            "status": "complete" if len(diff) else "NO_OVERLAP",
            "rows": int(len(group)),
            "overlap_rows": int(len(diff)),
            "exact_rank_match_fraction": float((diff == 0).mean()) if len(diff) else None,
            "abs_rank_difference_le_1_fraction": float((diff.abs() <= 1).mean()) if len(diff) else None,
            "mean_absolute_rank_difference": float(diff.abs().mean()) if len(diff) else None,
            "median_absolute_rank_difference": float(diff.abs().median()) if len(diff) else None,
            "max_absolute_rank_difference": float(diff.abs().max()) if len(diff) else None,
            "created_utc": now(),
        })
    return {"rows": rows, "summary": summary_rows}


def mirror_outputs(source: Path) -> None:
    MINI.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.csv"):
        shutil.copy2(path, MINI / path.name)
    for path in source.glob("*.json"):
        shutil.copy2(path, MINI / path.name)


def write_handoff(tag: str, module_rows: int, aggregate_rows: int, output_rows: int, parity: dict[str, list[dict[str, Any]]]) -> Path:
    path = FINAL / "d10_llama_numeric_parity_handoff.md"
    lines = [
        "# D10 Llama matched-numeric parity handoff",
        "",
        f"- status: `COMPLETE_MATCHED_NUMERIC_PARITY`",
        f"- tag: `{tag}`",
        f"- created_utc: `{now()}`",
        f"- cells: `100/100`",
        f"- module rows: `{module_rows}`",
        f"- equal-seven rows: `{aggregate_rows}`",
        f"- output rows: `{output_rows}`",
        f"- legacy parity summary rows: `{len(parity['summary'])}`",
        "",
        "Outputs are mirrored into `mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/`.",
        "No theory interpretation is included.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(path, MINI / path.name)
    return path


def gpu_inventory() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        return [{"error": f"{type(error).__name__}: {error}"}]
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, name, total, used, util = [part.strip() for part in line.split(",", 4)]
        rows.append({
            "index": int(index),
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "utilization_gpu_percent": int(util),
        })
    return rows


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add_file(path: Path, name: str) -> None:
        checks.append({
            "name": name,
            "path": str(path),
            "kind": "file",
            "complete": path.is_file() and path.stat().st_size > 0,
            "bytes": path.stat().st_size if path.is_file() else 0,
        })

    def add_model(path: Path, name: str) -> None:
        item = b3.model_check(path)
        checks.append({"name": name, "kind": "model", **item})

    for probe in CORE_PROBES:
        add_file(lgeom.corpus_path(probe), f"probe_corpus:{probe}")
    add_file(b3.AUTODL / "cycle09_block3/llama_geometry/corpora/probe_manifest.json", "legacy_probe_manifest")
    add_model(b3.LLAMA_STUDENT, "llama_base_model")
    for arm in ARMS:
        for step in STEPS:
            add_model(lexport.merged_target(arm, step), f"merged:{arm}:{step}")
    free = shutil.disk_usage(b3.AUTODL).free
    checks.append({
        "name": "autodl_tmp_free_space",
        "kind": "disk",
        "path": str(b3.AUTODL),
        "complete": free >= args.min_free_gb * (1 << 30),
        "free_bytes": free,
        "required_free_gb": args.min_free_gb,
    })
    gpus = gpu_inventory()
    a10_hold = {
        "task": "A10",
        "status": "HOLD_HARDWARE_INSUFFICIENT_FOR_2X96_PROTOCOL"
        if not (len(gpus) >= 2 and all(g.get("memory_total_mib", 0) >= 90000 for g in gpus[:2]))
        else "HARDWARE_PREFLIGHT_OK_NOT_STARTED_BY_D10_CONTROLLER",
        "gpus": gpus,
        "note": "This controller only runs A1-A9. It never starts A10.",
    }
    payload = {
        "schema_version": "cycle09_d10_preflight_v1",
        "status": "complete" if all(item.get("complete") for item in checks) else "incomplete",
        "complete": all(item.get("complete") for item in checks),
        "created_utc": now(),
        "checks": checks,
        "a10_hardware_gate": a10_hold,
    }
    atomic_json(ROOT / "preflight.json", payload)
    if args.strict and not payload["complete"]:
        missing = [item for item in checks if not item.get("complete")]
        raise RuntimeError(f"D10 preflight incomplete: {len(missing)} missing; first={missing[:5]}")
    return payload


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    smoke_args = argparse.Namespace(**vars(args))
    smoke_args.tag = "smoke"
    smoke_args.sample_limit = args.smoke_samples
    smoke_args.probes = "E_math"
    smoke_args.forward_batch_size = 1
    smoke_args.max_batch_tokens = min(args.max_batch_tokens, 2048)
    base = argparse.Namespace(**vars(smoke_args))
    base.arm, base.step = "base", 0
    first = run_checkpoint(base)
    current = argparse.Namespace(**vars(smoke_args))
    current.arm, current.step = "opd", 20
    second = run_checkpoint(current)
    result = {
        "schema_version": "cycle09_d10_smoke_v1",
        "status": "complete",
        "base": first,
        "opd20": second,
        "created_utc": now(),
    }
    atomic_json(ROOT / "smoke_result.json", result)
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "smoke", "checkpoint", "finalize"))
    parser.add_argument("--arm", default="opd", choices=("base", *ARMS))
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--probes", default="all")
    parser.add_argument("--tag", default="formal")
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--smoke-samples", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-batch-size", type=int, default=1)
    parser.add_argument("--max-batch-tokens", type=int, default=4096)
    parser.add_argument("--min-free-gb", type=int, default=50)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.phase == "preflight":
        value = preflight(args)
    elif args.phase == "smoke":
        value = smoke(args)
    elif args.phase == "checkpoint":
        value = run_checkpoint(args)
    else:
        value = finalize(args)
    print(json.dumps({"phase": args.phase, "status": value.get("status"), "created_utc": now()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
