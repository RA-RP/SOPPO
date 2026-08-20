#!/usr/bin/env python3
"""Cycle 09 C0--C5: baseline-normalized functional-rank contraction.

This implements the 2026-07-26 superseding Theory handoff.  It deliberately
does not create new training, behavior evaluation, backward/Fisher, or
counterfactual models.  C2 reads immutable state-rank artifacts; C3 only
performs fixed-token, full-vocabulary checkpoint forwards for model directories
that pass the registry audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

import sys

REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_actual_output_trajectory as actual  # noqa: E402
import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_r4_campaign as campaign  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402


ROOT = b3.AUTODL / "cycle09_relative_functional_contraction"
AUDIT = ROOT / "audit"
OUTPUT = ROOT / "output_cells"
BASE_LOGITS = ROOT / "base_logits"
FINAL = ROOT / "final"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

SCHEMA = "cycle09_relative_functional_contraction_v1"
ARMS = ("opd", "sft", "offkd", "seqkd")
CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")
EPSILONS = (0.01, 0.025, 0.05, 0.10)
MODULES = tuple(b3.MODULES)
LAYER_SCOPE = {"llama": (7, 14, 21), "qwen": (9, 18, 27)}
HEADLINE_LAYER = {"llama": 14, "qwen": 18}

LLAMA_STATE = b3.AUTODL / "cycle09_block3/llama_geometry/llama_early_320_r_epsilon.csv"
LLAMA_FROZEN = MINI / "llama_frozen_self_r_epsilon.csv"
QWEN_R4 = MINI / "R4_m1_tail_ec.csv"
QWEN_M6 = MINI / "M6_geometry_r_epsilon.csv"
QWEN_ALPHA05 = MINI / "qwen_alpha05_r_epsilon.csv"
ACTUAL_FINAL = b3.AUTODL / "cycle09_actual_output_trajectory/final"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def label(step: int) -> str:
    return f"step_{int(step):03d}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
    fields = sorted({key for row in rows for key in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


@contextmanager
def lock(path: Path):
    import fcntl

    target = path.with_suffix(path.suffix + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def ordered_steps(model: str) -> tuple[int, ...]:
    return tuple(b3.MEASURED_CHECKPOINTS) if model == "llama" else tuple(qstage.STEPS)


def path_for(model: str, arm: str, step: int) -> Path:
    if model == "qwen" and step == 0:
        return qstage.BASE_MODEL
    return actual.model_path(model, "base" if step == 0 else arm, step)


def model_is_available(model: str, arm: str, step: int) -> tuple[bool, str]:
    path = path_for(model, arm, step)
    if not path.is_dir():
        return False, f"missing_model_dir:{path}"
    try:
        complete = actual.model_integrity(model, path)
    except Exception as error:  # Registry should preserve the reason rather than guess.
        return False, f"model_integrity_error:{type(error).__name__}:{error}"
    return (True, "complete") if complete else (False, "incomplete_model_dir")


def available_steps(model: str, arm: str) -> tuple[int, ...]:
    return tuple(step for step in ordered_steps(model) if model_is_available(model, arm, step)[0])


def source_frame_llama(path: Path, arm_override: str | None = None, source_name: str = "llama_block3") -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    result = pd.DataFrame({
        "model": "llama",
        "arm": arm_override if arm_override else frame["arm"].astype(str),
        "checkpoint": frame["step"].astype(int),
        "probe_name": frame["probe"].astype(str),
        "layer": frame["layer"].astype(int),
        "module": frame["module"].astype(str),
        "epsilon": frame["epsilon"].astype(float),
        "state_rank_base": frame["base_r_epsilon"].astype(float),
        "state_rank_current": frame["r_epsilon"].astype(float),
        "sample_count": frame.get("n_samples", pd.Series(np.nan, index=frame.index)),
        "track": frame.get("track", pd.Series("per_checkpoint", index=frame.index)).astype(str),
        "source_name": source_name,
        "source_path": str(path),
        "source_protocol": "v2_three_level_equal_sample_declared",
    })
    return result


def source_frame_qwen_r4() -> pd.DataFrame:
    if not QWEN_R4.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(QWEN_R4)
    frame = frame[frame["track"].eq("per_checkpoint") & frame["probe_type"].eq("E")].copy()
    return pd.DataFrame({
        "model": "qwen",
        "arm": frame["arm"].astype(str),
        "checkpoint": frame["step"].astype(int),
        "probe_name": frame["task_id"].astype(str),
        "layer": frame["layer"].astype(int),
        "module": frame["module"].astype(str),
        "epsilon": frame["epsilon"].astype(float),
        "state_rank_base": frame["r_epsilon_base"].astype(float),
        "state_rank_current": frame["r_epsilon_current"].astype(float),
        "sample_count": np.nan,
        "track": frame["track"].astype(str),
        "source_name": "qwen_r4_v2",
        "source_path": str(QWEN_R4),
        "source_protocol": "v2_three_level_equal_sample_declared",
    })


def source_frame_qwen_m6() -> pd.DataFrame:
    if not QWEN_M6.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(QWEN_M6)
    frame = frame[frame["checkpoint"].notna()].copy()
    return pd.DataFrame({
        "model": "qwen",
        "arm": frame["arm"].astype(str),
        "checkpoint": frame["checkpoint"].astype(int),
        "probe_name": frame["probe_name"].astype(str),
        "layer": frame["layer"].astype(int),
        "module": frame["module"].astype(str),
        "epsilon": frame["epsilon"].astype(float),
        "state_rank_base": frame["base_r_epsilon"].astype(float),
        "state_rank_current": frame["r_epsilon"].astype(float),
        "sample_count": frame["sample_count"].astype(float),
        "track": frame["track"].astype(str),
        "source_name": "qwen_m6",
        "source_path": str(QWEN_M6),
        "source_protocol": "v2_three_level_equal_sample_declared",
    })


def source_frame_qwen_alpha() -> pd.DataFrame:
    if not QWEN_ALPHA05.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(QWEN_ALPHA05)
    return pd.DataFrame({
        "model": "qwen",
        "arm": "alpha05",
        "checkpoint": frame["step"].astype(int),
        "probe_name": frame["probe"].astype(str),
        "layer": frame["layer"].astype(int),
        "module": frame["module"].astype(str),
        "epsilon": frame["epsilon"].astype(float),
        "state_rank_base": frame["base_r_epsilon"].astype(float),
        "state_rank_current": frame["r_epsilon"].astype(float),
        "sample_count": frame.get("n_samples", pd.Series(np.nan, index=frame.index)),
        "track": frame["track"].astype(str),
        "source_name": "qwen_alpha05",
        "source_path": str(QWEN_ALPHA05),
        "source_protocol": "v2_three_level_equal_sample_declared",
    })


def load_state_sources() -> pd.DataFrame:
    frames = [
        source_frame_llama(LLAMA_STATE),
        source_frame_llama(LLAMA_FROZEN, arm_override="frozen_self", source_name="llama_frozen_self"),
        source_frame_qwen_r4(),
        source_frame_qwen_m6(),
        source_frame_qwen_alpha(),
    ]
    result = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    result = result[result["module"].isin(MODULES)].copy()
    result["state_rank_delta"] = result["state_rank_current"] - result["state_rank_base"]
    result["absolute_contraction"] = result["state_rank_base"] - result["state_rank_current"]
    valid = result["state_rank_base"] > 0
    result["relative_functional_contraction_module"] = np.where(
        valid, result["absolute_contraction"] / result["state_rank_base"], np.nan
    )
    result["attention_or_mlp"] = np.where(result["module"].str.startswith("self_attn"), "attention", "mlp")
    key = ["model", "arm", "checkpoint", "probe_name", "layer", "module", "epsilon", "source_name"]
    conflicts = result.groupby(key, dropna=False).agg(
        base_n=("state_rank_base", "nunique"), current_n=("state_rank_current", "nunique")
    ).reset_index()
    if (conflicts[["base_n", "current_n"]] > 1).any(axis=None):
        raise RuntimeError("state-rank source has conflicting duplicate rows")
    return result.drop_duplicates(key).sort_values(key).reset_index(drop=True)


def output_meta_path(model: str, arm: str, step: int, probe: str, n: int, cap: int) -> Path:
    return OUTPUT / model / arm / label(step) / f"{probe}.n{n or 'all'}.cap{cap or 'all'}.json"


def output_data_path(model: str, arm: str, step: int, probe: str, n: int, cap: int) -> Path:
    return output_meta_path(model, arm, step, probe, n, cap).with_suffix(".parquet")


def coverage_status(
    model: str, arm: str, step: int, probe: str, layer: int, epsilon: float, source: pd.DataFrame
) -> tuple[str, str, int]:
    available, reason = model_is_available(model, arm, step)
    subset = source[
        (source["model"] == model) & (source["arm"] == arm) & (source["checkpoint"] == step)
        & (source["probe_name"] == probe) & (source["layer"] == layer) & (np.isclose(source["epsilon"], epsilon))
    ]
    modules = int(subset["module"].nunique())
    if not available:
        return "PENDING_UPSTREAM", reason, modules
    if modules == 0:
        return "MISSING_STATE_RANK", "no matching r_epsilon state row", modules
    if subset["state_rank_base"].isna().any() or (subset["state_rank_base"] <= 0).any():
        return "MISSING_BASE_RANK", "nonpositive or missing base r_epsilon", modules
    if modules != len(MODULES):
        return "MISSING_STATE_RANK", f"expected 7 modules, found {modules}", modules
    metadata = read_json(output_meta_path(model, arm, step, probe, 0, 0), {})
    if metadata.get("status") != "complete":
        return "MISSING_MATCHED_OUTPUT", "fixed-token output cell not complete", modules
    if metadata.get("sample_ids_sha256") != metadata.get("source_sample_ids_sha256"):
        return "PROTOCOL_MISMATCH", "output sample manifest did not match its fixed source manifest", modules
    return "AVAILABLE_COMPLETE", "complete", modules


def audit(_: argparse.Namespace) -> dict[str, Any]:
    source = load_state_sources()
    atomic_csv(AUDIT / "state_rank_source_rows.csv", source.to_dict("records"))
    rows: list[dict[str, Any]] = []
    for model in ("llama", "qwen"):
        registry_arms = (("base", (0,)),) + tuple((arm, tuple(step for step in ordered_steps(model) if step != 0)) for arm in ARMS)
        for arm, steps in registry_arms:
            for step in steps:
                for probe in CORE_PROBES:
                    for layer in LAYER_SCOPE[model]:
                        for epsilon in EPSILONS:
                            status, reason, modules = coverage_status(model, arm, step, probe, layer, epsilon, source)
                            rows.append({
                                "model": model, "arm": arm, "checkpoint": step, "probe_name": probe,
                                "layer": layer, "epsilon": epsilon, "module_rows": modules,
                                "status": status, "reason": reason,
                            })
    coverage = pd.DataFrame(rows)
    atomic_csv(FINAL / "relative_functional_contraction_coverage.csv", coverage.to_dict("records"))
    missing = coverage[coverage["status"] != "AVAILABLE_COMPLETE"].copy()
    atomic_csv(FINAL / "relative_functional_contraction_missing_registry.csv", missing.to_dict("records"))
    payload = {
        "schema_version": SCHEMA, "status": "complete", "source_rows": int(len(source)),
        "coverage_rows": int(len(coverage)), "status_counts": coverage["status"].value_counts().to_dict(),
        "sources": sorted(source["source_path"].drop_duplicates().tolist()),
        "created_utc": utc_now(),
    }
    atomic_json(AUDIT / "coverage_manifest.json", payload)
    return payload


def derive(_: argparse.Namespace) -> dict[str, Any]:
    source = load_state_sources()
    valid = source[source["state_rank_base"] > 0].copy()
    group_keys = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "source_name", "source_protocol"]
    aggregate = valid.groupby(group_keys, dropna=False).agg(
        module_count=("module", "nunique"),
        state_rank_base_mean=("state_rank_base", "mean"),
        state_rank_current_mean=("state_rank_current", "mean"),
        state_rank_delta_mean=("state_rank_delta", "mean"),
        absolute_contraction_mean=("absolute_contraction", "mean"),
        relative_functional_contraction_equal7=("relative_functional_contraction_module", "mean"),
    ).reset_index()
    aggregate["relative_functional_contraction_ratio_of_means_sensitivity"] = (
        (aggregate["state_rank_base_mean"] - aggregate["state_rank_current_mean"])
        / aggregate["state_rank_base_mean"]
    )
    for family, modules in (("attention", [m for m in MODULES if m.startswith("self_attn")]), ("mlp", [m for m in MODULES if m.startswith("mlp")])):
        part = valid[valid["module"].isin(modules)].groupby(group_keys, dropna=False)["relative_functional_contraction_module"].mean().rename(
            f"{family}_group_relative_contraction"
        )
        aggregate = aggregate.merge(part.reset_index(), on=group_keys, how="left")
    source["relative_functional_contraction_equal7"] = source.merge(
        aggregate[group_keys + ["relative_functional_contraction_equal7"]], on=group_keys, how="left"
    )["relative_functional_contraction_equal7"]
    atomic_csv(FINAL / "relative_functional_contraction_module_audit.csv", source.to_dict("records"))
    atomic_csv(FINAL / "relative_functional_contraction_all_cells.csv", aggregate.to_dict("records"))
    sensitivity = aggregate[group_keys + [
        "module_count", "relative_functional_contraction_equal7",
        "relative_functional_contraction_ratio_of_means_sensitivity",
        "attention_group_relative_contraction", "mlp_group_relative_contraction",
    ]]
    atomic_csv(FINAL / "relative_functional_contraction_aggregation_sensitivity.csv", sensitivity.to_dict("records"))
    payload = {
        "schema_version": SCHEMA, "status": "complete", "module_rows": int(len(source)),
        "aggregate_rows": int(len(aggregate)), "complete_equal7_cells": int((aggregate["module_count"] == 7).sum()),
        "created_utc": utc_now(),
    }
    atomic_json(AUDIT / "derive_manifest.json", payload)
    return payload


def samples_for(model: str, probe: str, n: int) -> list[Any]:
    return actual.samples_for(model, probe, n)


def base_logits_path(model: str, probe: str, n: int, cap: int) -> Path:
    return BASE_LOGITS / model / f"{probe}.n{n or 'all'}.cap{cap or 'all'}.pt"


def ensure_base_logits(model: str, probe: str, n: int, cap: int, device: str) -> dict[str, Any]:
    target = base_logits_path(model, probe, n, cap)
    metadata = target.with_suffix(".json")
    with lock(target):
        cached = read_json(metadata, {})
        if cached.get("status") == "complete" and target.is_file():
            return torch.load(target, map_location="cpu", weights_only=True)
        samples = samples_for(model, probe, n)
        model_obj = campaign.load_model(path_for(model, "base", 0), device)
        try:
            records = []
            for sample in samples:
                positions = actual.selected_positions(sample, cap)
                records.append({
                    "sample_id": sample.sample_id,
                    "positions": positions.cpu(),
                    "token_weights": sample.token_weights[positions].float().cpu(),
                    "target_ids": sample.input_ids[0, positions].long().cpu(),
                    "logits": actual.selected_logits(model_obj, sample, positions, device),
                })
        finally:
            campaign.unload_model(model_obj)
        sample_ids = [record["sample_id"] for record in records]
        value = {
            "schema_version": SCHEMA, "status": "complete", "model": model, "probe": probe,
            "measurement_n": n, "selected_token_cap": cap, "records": records,
            "sample_ids_sha256": sha256_json(sample_ids), "logit_storage_dtype": "bf16_exact_forward_logits",
            "log_softmax_dtype": "fp32_at_metric_time", "created_utc": utc_now(),
        }
        atomic_torch(target, value)
        atomic_json(metadata, {
            key: value[key] for key in (
                "schema_version", "status", "model", "probe", "measurement_n", "selected_token_cap",
                "sample_ids_sha256", "logit_storage_dtype", "log_softmax_dtype", "created_utc"
            )
        } | {"artifact": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target)})
        return value


def previous_available_step(model: str, arm: str, step: int) -> int | None:
    # The base model is the cumulative reference, not a member of a training arm.
    if step == 0 or arm == "base":
        return None
    values = [item for item in available_steps(model, arm) if item <= step]
    index = values.index(step)
    return values[index - 1] if index else None


def forward_cell(args: argparse.Namespace) -> dict[str, Any]:
    arm = "base" if args.step == 0 else args.arm
    target = output_data_path(args.model, arm, args.step, args.probe, args.measurement_n, args.selected_token_cap)
    metadata = target.with_suffix(".json")
    with lock(target):
        cached = read_json(metadata, {})
        if cached.get("status") == "complete" and target.is_file():
            return cached
        available, reason = model_is_available(args.model, arm, args.step)
        if not available:
            payload = {"schema_version": SCHEMA, "status": "PENDING_UPSTREAM", "reason": reason, "created_utc": utc_now()}
            atomic_json(metadata, payload)
            return payload
        samples = samples_for(args.model, args.probe, args.measurement_n)
        source_ids = [sample.sample_id for sample in samples]
        baseline = ensure_base_logits(args.model, args.probe, args.measurement_n, args.selected_token_cap, args.device)
        if baseline["sample_ids_sha256"] != sha256_json(source_ids):
            raise RuntimeError("base logits fixed-token manifest mismatches current samples")
        previous = previous_available_step(args.model, arm, args.step)
        current = campaign.load_model(path_for(args.model, arm, args.step), args.device)
        previous_model = None
        if previous not in (None, 0):
            previous_model = campaign.load_model(path_for(args.model, arm, previous), args.device)
        sample_map = {sample.sample_id: sample for sample in samples}
        rows: list[dict[str, Any]] = []
        try:
            for record in baseline["records"]:
                sample = sample_map[record["sample_id"]]
                positions = record["positions"].long()
                base_logits = record["logits"].float()
                current_logits = actual.selected_logits(current, sample, positions, args.device).float()
                source_logits = base_logits if previous in (None, 0) else actual.selected_logits(previous_model, sample, positions, args.device).float()
                logp0 = torch.log_softmax(base_logits, dim=-1)
                logpt = torch.log_softmax(current_logits, dim=-1)
                logpp = torch.log_softmax(source_logits, dim=-1)
                p0, pp = logp0.exp(), logpp.exp()
                cumulative_kl = (p0 * (logp0 - logpt)).sum(dim=-1)
                stepwise_kl = (pp * (logpp - logpt)).sum(dim=-1)
                token_ids = record["target_ids"].long()
                nll0 = -logp0.gather(1, token_ids[:, None]).squeeze(1)
                nllt = -logpt.gather(1, token_ids[:, None]).squeeze(1)
                nllp = -logpp.gather(1, token_ids[:, None]).squeeze(1)
                for index, position in enumerate(positions.tolist()):
                    cumulative_delta = float(nllt[index] - nll0[index])
                    step_delta = float(nllt[index] - nllp[index])
                    rows.append({
                        "schema_version": SCHEMA, "model": args.model, "arm": arm, "checkpoint": args.step,
                        "source_checkpoint": previous if previous is not None else 0, "probe_name": args.probe,
                        "sample_id": record["sample_id"], "token_position": int(position),
                        "target_token_id": int(token_ids[index]), "token_weight": float(record["token_weights"][index]),
                        "cumulative_kl_base_to_current": float(cumulative_kl[index]),
                        "stepwise_kl_source_to_current": float(stepwise_kl[index]),
                        "nll_base": float(nll0[index]), "nll_current": float(nllt[index]), "nll_source": float(nllp[index]),
                        "delta_nll_cumulative": cumulative_delta, "delta_nll_stepwise": step_delta,
                        "absolute_delta_nll_cumulative": abs(cumulative_delta),
                        "absolute_delta_nll_stepwise": abs(step_delta), "full_vocabulary": True,
                        "log_softmax_dtype": "fp32", "base_logit_cache": str(base_logits_path(args.model, args.probe, args.measurement_n, args.selected_token_cap)),
                    })
        finally:
            if previous_model is not None:
                campaign.unload_model(previous_model)
            campaign.unload_model(current)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        frame = pd.DataFrame(rows)
        atomic_parquet(target, frame)
        payload = {
            "schema_version": SCHEMA, "status": "complete", "model": args.model, "arm": arm,
            "checkpoint": args.step, "source_checkpoint": previous if previous is not None else 0,
            "probe_name": args.probe, "rows": len(rows), "samples": len(baseline["records"]),
            "measurement_n": args.measurement_n, "selected_token_cap": args.selected_token_cap,
            "sample_ids_sha256": baseline["sample_ids_sha256"], "source_sample_ids_sha256": sha256_json(source_ids),
            "artifact": str(target), "bytes": target.stat().st_size, "sha256": sha256_file(target),
            "full_vocabulary": True, "log_softmax_dtype": "fp32", "created_utc": utc_now(),
        }
        atomic_json(metadata, payload)
        return payload


def plan(_: argparse.Namespace) -> dict[str, Any]:
    tasks = []
    for model in ("llama", "qwen"):
        base_ok, _ = model_is_available(model, "base", 0)
        if base_ok:
            for probe in CORE_PROBES:
                tasks.append({"model": model, "arm": "base", "step": 0, "probe": probe, "status": "pending"})
        for arm in ARMS:
            for step in available_steps(model, arm):
                if step == 0:
                    continue
                for probe in CORE_PROBES:
                    tasks.append({"model": model, "arm": arm, "step": step, "probe": probe, "status": "pending"})
    for task in tasks:
        meta = read_json(output_meta_path(task["model"], task["arm"], task["step"], task["probe"], 0, 0), {})
        if meta.get("status") == "complete":
            task["status"] = "complete"
    payload = {"schema_version": SCHEMA, "tasks": tasks, "created_utc": utc_now()}
    atomic_json(AUDIT / "forward_queue.json", payload)
    return {"status": "complete", "tasks": len(tasks), "pending": sum(t["status"] == "pending" for t in tasks)}


def claim_task(queue_path: Path, device: str) -> dict[str, Any] | None:
    with lock(queue_path):
        payload = read_json(queue_path, {})
        for task in payload.get("tasks", []):
            if task.get("status") == "pending":
                task.update({"status": "running", "device": device, "pid": os.getpid(), "started_utc": utc_now()})
                atomic_json(queue_path, payload)
                return dict(task)
    return None


def complete_task(queue_path: Path, task: dict[str, Any], result: dict[str, Any] | None, error: str | None) -> None:
    key = (task["model"], task["arm"], int(task["step"]), task["probe"])
    with lock(queue_path):
        payload = read_json(queue_path, {})
        for current in payload.get("tasks", []):
            if (current["model"], current["arm"], int(current["step"]), current["probe"]) == key:
                current.update({"status": "complete" if error is None else "error", "completed_utc": utc_now()})
                if result:
                    current["result_status"] = result.get("status")
                    current["rows"] = result.get("rows")
                if error:
                    current["error"] = error
                break
        atomic_json(queue_path, payload)


def worker(args: argparse.Namespace) -> dict[str, Any]:
    queue_path = AUDIT / "forward_queue.json"
    done = 0
    while task := claim_task(queue_path, args.device):
        namespace = argparse.Namespace(
            model=task["model"], arm=task["arm"], step=int(task["step"]), probe=task["probe"],
            measurement_n=0, selected_token_cap=0, device=args.device,
        )
        try:
            result = forward_cell(namespace)
            complete_task(queue_path, task, result, None)
            done += 1
        except Exception as error:
            complete_task(queue_path, task, None, f"{type(error).__name__}: {error}")
    return {"status": "complete", "claimed": done}


def weighted_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "arm", "checkpoint", "source_checkpoint", "probe_name"]
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        weights = group["token_weight"].to_numpy(dtype=np.float64)
        if float(weights.sum()) <= 0:
            continue
        row = dict(zip(keys, values, strict=True))
        row.update({
            "sample_count": int(group["sample_id"].nunique()), "token_count": int(len(group)),
            **{column: float(np.average(group[column], weights=weights)) for column in (
                "cumulative_kl_base_to_current", "stepwise_kl_source_to_current", "delta_nll_cumulative",
                "delta_nll_stepwise", "absolute_delta_nll_cumulative", "absolute_delta_nll_stepwise",
                "nll_base", "nll_current", "nll_source",
            )},
        })
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_outputs(_: argparse.Namespace) -> dict[str, Any]:
    paths = []
    for metadata in sorted(OUTPUT.rglob("*.json")):
        item = read_json(metadata, {})
        if item.get("status") == "complete" and int(item.get("measurement_n", -1)) == 0 and int(item.get("selected_token_cap", -1)) == 0:
            candidate = Path(item["artifact"])
            if candidate.is_file():
                paths.append(candidate)
    tokens = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()
    aggregate = weighted_aggregate(tokens) if not tokens.empty else pd.DataFrame()
    cumulative = aggregate.drop(columns=[column for column in aggregate if column.startswith("stepwise_") or column.endswith("_stepwise")], errors="ignore")
    stepwise = aggregate.drop(columns=[column for column in aggregate if column.startswith("cumulative_") or column.endswith("_cumulative")], errors="ignore")
    atomic_csv(FINAL / "relative_contraction_matched_cumulative_outputs.csv", cumulative.to_dict("records"))
    atomic_csv(FINAL / "relative_contraction_matched_stepwise_outputs.csv", stepwise.to_dict("records"))
    payload = {
        "schema_version": SCHEMA, "status": "complete", "cell_files": len(paths), "token_rows": int(len(tokens)),
        "aggregate_rows": int(len(aggregate)), "created_utc": utc_now(),
    }
    atomic_json(AUDIT / "output_aggregate_manifest.json", payload)
    return payload


def correlation_rows(frame: pd.DataFrame, grouping: list[str], grouping_label: str, metrics: list[str], targets: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key, group in frame.groupby(grouping, dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        common = dict(zip(grouping, key_values, strict=True))
        for metric in metrics:
            for target in targets:
                subset = group[[metric, target]].dropna()
                rows.append(common | {
                    "association_scope": grouping_label, "metric": metric, "target": target, "rows": int(len(subset)),
                    "spearman": float(subset[metric].corr(subset[target], method="spearman")) if len(subset) >= 3 else np.nan,
                    "kendall": float(subset[metric].corr(subset[target], method="kendall")) if len(subset) >= 3 else np.nan,
                })
    return rows


def grouped_oof(local: pd.DataFrame, fields: list[str], target: str, group_field: str, protocol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    complete = np.isfinite(local[target].to_numpy(float)) & np.isfinite(local[fields].to_numpy(float)).all(axis=1)
    data = local.loc[complete].copy()
    groups = data[group_field].to_numpy()
    if len(data) < 12 or len(np.unique(groups)) < 3:
        return {"status": "DEFERRED_INSUFFICIENT_GROUPS", "rows": int(len(data)), "groups": int(len(np.unique(groups)))}, []
    x, y = data[fields].to_numpy(float), data[target].to_numpy(float)
    pred = np.full(len(data), np.nan)
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for train, test in splitter.split(x, y, groups):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    rows = []
    for position, (_, item) in enumerate(data.iterrows()):
        rows.append({
            "evaluation_protocol": protocol, "model": item["model"], "arm": item["arm"], "checkpoint": int(item["checkpoint"]),
            "probe_name": item["probe_name"], "target": target, "actual": float(y[position]), "predicted": float(pred[position]),
        })
    return {
        "status": "complete", "rows": int(len(data)), "groups": int(len(np.unique(groups))),
        "heldout_mae": float(mean_absolute_error(y, pred)), "heldout_r2": float(r2_score(y, pred)),
        "heldout_spearman": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
    }, rows


def train_predict(train: pd.DataFrame, test: pd.DataFrame, fields: list[str], target: str, protocol: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_ok = np.isfinite(train[target].to_numpy(float)) & np.isfinite(train[fields].to_numpy(float)).all(axis=1)
    test_ok = np.isfinite(test[target].to_numpy(float)) & np.isfinite(test[fields].to_numpy(float)).all(axis=1)
    train, test = train.loc[train_ok].copy(), test.loc[test_ok].copy()
    if len(train) < 8 or len(test) < 1:
        return {"status": "DEFERRED_INSUFFICIENT_TRAIN_OR_TEST", "rows": int(len(test)), "train_rows": int(len(train))}, []
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    reg.fit(train[fields].to_numpy(float), train[target].to_numpy(float))
    actuals = test[target].to_numpy(float)
    predictions = reg.predict(test[fields].to_numpy(float))
    rows = []
    for position, (_, item) in enumerate(test.iterrows()):
        rows.append({
            "evaluation_protocol": protocol, "model": item["model"], "arm": item["arm"], "checkpoint": int(item["checkpoint"]),
            "probe_name": item["probe_name"], "target": target, "actual": float(actuals[position]), "predicted": float(predictions[position]),
        })
    return {
        "status": "complete", "rows": int(len(test)), "train_rows": int(len(train)),
        "heldout_mae": float(mean_absolute_error(actuals, predictions)),
        "heldout_r2": float(r2_score(actuals, predictions)) if len(test) > 2 else np.nan,
        "heldout_spearman": float(pd.Series(actuals).corr(pd.Series(predictions), method="spearman")) if len(test) > 2 else np.nan,
    }, rows


def add_weight_features(frame: pd.DataFrame) -> pd.DataFrame:
    path = ACTUAL_FINAL / "actual_update_cumulative_geometry.csv"
    if not path.is_file():
        frame["raw_update_energy_equal7"] = np.nan
        frame["whitened_update_energy_equal7"] = np.nan
        return frame
    geometry = pd.read_csv(path)
    geometry = geometry[(geometry["epsilon"] == 0.05) & geometry["layer"].eq(14)].copy()
    keys = ["model", "arm", "checkpoint", "probe_name", "layer"]
    weights = geometry.groupby(keys, dropna=False).agg(
        raw_update_energy_equal7=("raw_weight_energy", "mean"),
        whitened_update_energy_equal7=("whitened_update_energy_current", "mean"),
    ).reset_index()
    return frame.merge(weights, on=keys, how="left")


def analyze(_: argparse.Namespace) -> dict[str, Any]:
    state_path = FINAL / "relative_functional_contraction_all_cells.csv"
    output_path = FINAL / "relative_contraction_matched_cumulative_outputs.csv"
    stepwise_path = FINAL / "relative_contraction_matched_stepwise_outputs.csv"
    if not state_path.is_file() or not output_path.is_file():
        raise FileNotFoundError("run derive and aggregate-outputs before analyze")
    state = pd.read_csv(state_path)
    outputs = pd.read_csv(output_path)
    state = state[(state["module_count"] == 7) & state["epsilon"].eq(0.05)].copy()
    state = state[state.apply(lambda row: int(row["layer"]) == HEADLINE_LAYER[row["model"]], axis=1)]
    joined = state.merge(outputs, on=["model", "arm", "checkpoint", "probe_name"], how="inner")
    joined = joined[(joined["arm"] != "base") & joined["probe_name"].isin(CORE_PROBES)].copy()
    joined = add_weight_features(joined)
    joined["absolute_state_rank_mean"] = joined["state_rank_current_mean"].abs()
    joined["absolute_delta_nll_cumulative"] = joined["delta_nll_cumulative"].abs()
    metrics = [
        "absolute_state_rank_mean", "state_rank_delta_mean", "absolute_contraction_mean",
        "relative_functional_contraction_equal7", "raw_update_energy_equal7", "whitened_update_energy_equal7",
    ]
    targets = ["cumulative_kl_base_to_current", "delta_nll_cumulative", "absolute_delta_nll_cumulative"]
    rows: list[dict[str, Any]] = []
    rows += correlation_rows(joined, ["model", "arm"], "model_arm_checkpoint_domain", metrics, targets)
    rows += correlation_rows(joined, ["model", "checkpoint"], "model_checkpoint_within_domain", metrics, targets)
    rows += correlation_rows(joined, ["model", "arm", "probe_name"], "model_arm_domain_time", metrics, targets)
    demean = joined.copy()
    for metric in metrics + targets:
        demean[f"demean_{metric}"] = demean[metric] - demean.groupby(["model", "checkpoint"])[metric].transform("mean")
    rows += correlation_rows(
        demean, ["model"], "checkpoint_demeaned_pooled",
        [f"demean_{metric}" for metric in metrics], [f"demean_{target}" for target in targets],
    )
    atomic_csv(FINAL / "relative_contraction_output_correlations.csv", rows)
    within = pd.DataFrame(rows)
    atomic_csv(FINAL / "relative_contraction_within_checkpoint_correlations.csv", within[within["association_scope"].eq("model_checkpoint_within_domain")].to_dict("records"))

    required = ["raw_update_energy_equal7", "whitened_update_energy_equal7", "relative_functional_contraction_equal7"]
    model_data = joined.dropna(subset=required).copy()
    specs = {
        "Model-W": ["raw_update_energy_equal7"],
        "Model-C": ["relative_functional_contraction_equal7"],
        "Model-WC": ["raw_update_energy_equal7", "relative_functional_contraction_equal7"],
        "Model-WS": ["raw_update_energy_equal7", "whitened_update_energy_equal7"],
        "Model-WSC": ["raw_update_energy_equal7", "whitened_update_energy_equal7", "relative_functional_contraction_equal7"],
    }
    grouped_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    leave_rows: list[dict[str, Any]] = []
    for model in sorted(model_data["model"].unique()):
        local = model_data[model_data["model"] == model].copy()
        for target in targets:
            for scale in ("linear", "signed_log1p_abs"):
                transformed = local.copy()
                name = target if scale == "linear" else f"{target}__signed_log1p_abs"
                transformed[name] = transformed[target] if scale == "linear" else np.sign(transformed[target]) * np.log1p(np.abs(transformed[target]))
                for feature_set, fields in specs.items():
                    result, preds = grouped_oof(transformed, fields, name, "checkpoint", "leave_one_checkpoint_out")
                    grouped_rows.append({"model": model, "target": target, "scale": scale, "feature_set": feature_set} | result)
                    for row in preds:
                        row["feature_set"] = feature_set
                        row["scale"] = scale
                    prediction_rows.extend(preds)
                    for held_out in sorted(arm for arm in transformed["arm"].unique() if arm != "base"):
                        train = transformed[(transformed["arm"] != held_out) & (transformed["arm"] != "base")]
                        test = transformed[transformed["arm"] == held_out]
                        result_arm, preds_arm = train_predict(train, test, fields, name, f"leave_one_arm_out:{held_out}")
                        if result_arm.get("status") == "complete":
                            for row in preds_arm:
                                row["feature_set"] = feature_set
                                row["scale"] = scale
                                row["held_out_arm"] = held_out
                                leave_rows.append(row)
                    for held_out in sorted(transformed["probe_name"].unique()):
                        train = transformed[transformed["probe_name"] != held_out]
                        test = transformed[transformed["probe_name"] == held_out]
                        result_domain, preds_domain = train_predict(train, test, fields, name, f"leave_one_domain_out:{held_out}")
                        if result_domain.get("status") == "complete":
                            for row in preds_domain:
                                row["feature_set"] = feature_set
                                row["scale"] = scale
                                row["held_out_domain"] = held_out
                                leave_rows.append(row)
    atomic_csv(FINAL / "relative_contraction_grouped_models.csv", grouped_rows)
    atomic_csv(FINAL / "relative_contraction_leave_arm_domain_out.csv", leave_rows)
    atomic_csv(FINAL / "relative_contraction_raw_predictions.csv", prediction_rows)

    stepwise_rows: list[dict[str, Any]] = []
    if stepwise_path.is_file():
        stepwise = pd.read_csv(stepwise_path)
        current = state[(state["module_count"] == 7) & state["epsilon"].eq(0.05)].copy()
        current = current[current.apply(lambda row: int(row["layer"]) == HEADLINE_LAYER[row["model"]], axis=1)]
        current = current.rename(columns={"relative_functional_contraction_equal7": "c_current"})
        current = current[current["arm"] != "base"].copy()
        current["source_checkpoint"] = np.nan
        for (_, arm, probe), indices in current.groupby(["model", "arm", "probe_name"]).groups.items():
            ordered = current.loc[indices].sort_values("checkpoint").index.tolist()
            previous = 0
            for index in ordered:
                current.loc[index, "source_checkpoint"] = previous
                previous = int(current.loc[index, "checkpoint"])
        source = current[["model", "arm", "checkpoint", "probe_name", "c_current"]].rename(columns={"checkpoint": "source_checkpoint", "c_current": "c_source"})
        base = state[(state["module_count"] == 7) & state["epsilon"].eq(0.05) & state["arm"].eq("base")].copy()
        base = base[base.apply(lambda row: int(row["layer"]) == HEADLINE_LAYER[row["model"]], axis=1)]
        base_rows = []
        for _, item in base.iterrows():
            for arm in ARMS:
                base_rows.append({"model": item["model"], "arm": arm, "source_checkpoint": 0, "probe_name": item["probe_name"], "c_source": 0.0})
        source = pd.concat([source, pd.DataFrame(base_rows)], ignore_index=True)
        pair = current.merge(source, on=["model", "arm", "source_checkpoint", "probe_name"], how="left")
        pair["delta_c"] = pair["c_current"] - pair["c_source"]
        stepwise_rows = pair.merge(stepwise, on=["model", "arm", "checkpoint", "source_checkpoint", "probe_name"], how="inner").to_dict("records")
    atomic_csv(FINAL / "relative_contraction_stepwise_diagnostics.csv", stepwise_rows)
    payload = {
        "schema_version": SCHEMA, "status": "complete", "matched_state_output_rows": int(len(joined)),
        "model_rows": len(grouped_rows), "prediction_rows": len(prediction_rows), "stepwise_rows": len(stepwise_rows),
        "created_utc": utc_now(),
    }
    atomic_json(AUDIT / "analysis_manifest.json", payload)
    return payload


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    audit(args)
    derive(args)
    cell = argparse.Namespace(
        model="llama", arm="opd", step=20, probe="E_math", measurement_n=2, selected_token_cap=16, device=args.device,
    )
    result = forward_cell(cell)
    if result.get("status") != "complete" or result.get("rows", 0) <= 0:
        raise RuntimeError(f"smoke forward did not complete: {result}")
    return {"status": "complete", "smoke_rows": result["rows"], "created_utc": utc_now()}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("audit", "derive", "forward-cell", "plan", "worker", "aggregate-outputs", "analyze", "smoke"))
    parser.add_argument("--model", choices=("llama", "qwen"), default="llama")
    parser.add_argument("--arm", choices=("base", *ARMS), default="opd")
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--probe", choices=CORE_PROBES, default="E_math")
    parser.add_argument("--measurement-n", type=int, default=0)
    parser.add_argument("--selected-token-cap", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    actions = {
        "audit": audit, "derive": derive, "forward-cell": forward_cell, "plan": plan,
        "worker": worker, "aggregate-outputs": aggregate_outputs, "analyze": analyze, "smoke": smoke,
    }
    value = actions[args.phase](args)
    print(json.dumps({"phase": args.phase, "status": value.get("status"), "rows": value.get("rows"), "output": value.get("artifact")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
