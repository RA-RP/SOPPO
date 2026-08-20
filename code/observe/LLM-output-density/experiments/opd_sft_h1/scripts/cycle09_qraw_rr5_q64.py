#!/usr/bin/env python3
"""QRAW-RR5-Q64: Qwen raw-activation strict common grid."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPTS = Path("/root/LLM-output-density/experiments/opd_sft_h1/scripts")
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_r4_common as c4  # noqa: E402

qprobe = None
fatr1 = None
campaign = None
s3 = None


def ensure_heavy_imports() -> None:
    global qprobe, fatr1, campaign, s3
    if qprobe is not None:
        return
    import cycle09_block3_qwen_probe_geometry as _qprobe  # noqa: E402
    import cycle09_fat_outlink_round1 as _fatr1  # noqa: E402
    import cycle09_r4_campaign as _campaign  # noqa: E402
    import cycle09_stage3_common as _s3  # noqa: E402

    qprobe = _qprobe
    fatr1 = _fatr1
    campaign = _campaign
    s3 = _s3


TASK = "QRAW-RR5-Q64"
SCHEMA = "cycle09_qraw_rr5_q64_v1"
REPO = Path("/root/LLM-output-density")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
OUT = MINI / "qwen_raw_activation_rr5_q64"
SCRATCH = Path("/root/autodl-tmp/cycle09_qraw_rr5_q64")
CELL_ROOT = SCRATCH / "cells"
BASE_PROFILE_ROOT = SCRATCH / "base_profiles"

MODEL = "qwen"
LAYER = 18
ARMS = ("opd", "sft", "offkd", "seqkd")
CHECKPOINTS = (5, 20, 40, 160)
PROBES = ("E_general", "E_if", "E_math", "E_ood")
MODULES5 = (
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
RAW_FEATURES = [
    "normalized_entropy_effective_rank",
    "participation_ratio",
    "top1_explained_share",
    "top8_explained_share",
    "top32_explained_share",
    "raw_anisotropy",
    "centered_anisotropy",
    "linear_cka_vs_step0",
]
REG_TARGETS = [
    "cumulative_kl_base_to_current",
    "absolute_delta_nll_cumulative",
    "delta_nll_cumulative",
]
TARGETS = REG_TARGETS + ["is_opd"]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(value)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    return b3.sha256_file(path)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def exact_keys() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for step in CHECKPOINTS:
            for probe in PROBES:
                rows.append({"model": MODEL, "arm": arm, "checkpoint": step, "probe_name": probe, "layer": LAYER})
    return pd.DataFrame(rows)


def load_c5() -> pd.DataFrame:
    fc = pd.read_csv(MINI / "equal5_non_qk/EQUAL5_functional_cells.csv")
    q = fc[
        (fc["model"].eq("qwen"))
        & (fc["layer"].eq(LAYER))
        & (fc["epsilon"].eq(0.05))
        & (fc["module"].isin(MODULES5))
    ].copy()
    q["probe_name"] = q["artifact_probe_name"].fillna(q["probe_name"])
    out = (
        q.groupby(["model", "arm", "checkpoint", "probe_name", "layer"], as_index=False)
        .agg(
            c_epsilon=("relative_functional_contraction_module", "mean"),
            c5_module_count=("module", "nunique"),
            c5_source_probe_names=("probe_name", lambda s: ",".join(sorted(set(map(str, s))))),
        )
    )
    return out


def load_pk() -> pd.DataFrame:
    fm = pd.read_csv(MINI / "d11_a4_feature_matrix_replaced_pk.csv")
    cols = ["model", "arm", "checkpoint", "probe_name", "layer", "p_k4", "p_k8", "p_k16", "p_k32", "pk_track"]
    pk = fm[
        (fm["model"].eq("qwen"))
        & (fm["layer"].eq(LAYER))
        & (fm["epsilon"].eq(0.05))
    ][cols].drop_duplicates()
    return pk


def load_outputs() -> pd.DataFrame:
    out = pd.read_csv(MINI / "d10_5_integrated_outputs.csv")
    out = out[out["model"].eq("qwen")][
        [
            "model",
            "arm",
            "checkpoint",
            "probe_name",
            "cumulative_kl_base_to_current",
            "absolute_delta_nll_cumulative",
            "delta_nll_cumulative",
            "sample_count",
            "track",
        ]
    ].drop_duplicates()
    out["layer"] = LAYER
    return out


def checkpoint_inventory() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for step in CHECKPOINTS:
            path = s3.model_path(arm, step)
            check = s3.model_integrity(path)
            source_type = "direct_merged"
            source_path = path
            adapter_complete = False
            if not check["complete"] and arm != "opd":
                adapter = fatr1.qwen_adapter_path(arm, step)
                source_type = "ephemeral_bf16_merge_from_adapter"
                source_path = adapter
                adapter_complete = fatr1.adapter_complete(adapter, arm in ("offkd", "seqkd"))
            rows.append({
                "model": MODEL,
                "arm": arm,
                "checkpoint": step,
                **check,
                "source_type": source_type,
                "source_path": str(source_path),
                "state_loadable": bool(check["complete"] or adapter_complete),
                "adapter_complete": bool(adapter_complete),
            })
    return pd.DataFrame(rows)


def probe_inventory() -> pd.DataFrame:
    rows = []
    manifest = b3.read_json(qprobe.PROBE_MANIFEST, {})
    details = manifest.get("probe_details", {})
    for probe in PROBES:
        path = qprobe.corpus_path(probe)
        n = sum(1 for _ in path.open(encoding="utf-8")) if path.is_file() else 0
        rows.append(
            {
                "probe_name": probe,
                "path": str(path),
                "exists": path.is_file(),
                "source_rows": n,
                "used_rows": 32,
                "sample_rule": "first_32_from_formal_probe_order",
                "source_sha256": sha256(path) if path.is_file() else None,
                "manifest_sample_ids_sha256": details.get(probe, {}).get("sample_ids_sha256"),
                "manifest_text_sha256": details.get(probe, {}).get("text_sha256"),
            }
        )
    return pd.DataFrame(rows)


def preflight(write: bool = True) -> dict[str, Any]:
    ensure_heavy_imports()
    OUT.mkdir(parents=True, exist_ok=True)
    keys = exact_keys()
    inv = checkpoint_inventory()
    probes = probe_inventory()
    c5 = load_c5()
    pk = load_pk()
    outputs = load_outputs()
    audits = []
    joined = keys.copy()
    for name, df in [("C5", c5), ("Pk5", pk), ("output", outputs)]:
        dup = int(df.duplicated(["model", "arm", "checkpoint", "probe_name", "layer"]).sum())
        m = keys.merge(df, on=["model", "arm", "checkpoint", "probe_name", "layer"], how="left", indicator=True)
        audits.append(
            {
                "component": name,
                "coverage": int((m["_merge"] == "both").sum()),
                "expected": 64,
                "duplicates": dup,
                "missing_keys": json.dumps(m[m["_merge"] != "both"][["arm", "checkpoint", "probe_name"]].to_dict("records"), sort_keys=True),
            }
        )
        joined = joined.merge(df, on=["model", "arm", "checkpoint", "probe_name", "layer"], how="left")
    checks = {
        "checkpoint_states": int(inv["state_loadable"].sum()),
        "probe_manifests": int((probes["exists"] & probes["used_rows"].eq(32)).sum()),
        "state_cells": 64,
        "base_reference_cells": 4,
        "c5_coverage": audits[0]["coverage"],
        "pk5_coverage": audits[1]["coverage"],
        "output_coverage": audits[2]["coverage"],
        "duplicate_exact_keys": int(joined.duplicated(["model", "arm", "checkpoint", "probe_name", "layer"]).sum()),
    }
    complete = (
        checks["checkpoint_states"] == 16
        and checks["probe_manifests"] == 4
        and checks["c5_coverage"] == 64
        and checks["pk5_coverage"] == 64
        and checks["output_coverage"] == 64
        and checks["duplicate_exact_keys"] == 0
        and all(item["duplicates"] == 0 for item in audits)
    )
    status = "PREFLIGHT_COMPLETE" if complete else "BLOCKED_QRAW_RR5_Q64_PREFLIGHT"
    task = pd.DataFrame([{"task": "S0_preflight", "status": status, **checks}])
    if write:
        atomic_csv(OUT / "QRAW_RR5_checkpoint_inventory.csv", inv)
        atomic_csv(OUT / "QRAW_RR5_probe_inventory.csv", probes)
        atomic_csv(OUT / "QRAW_RR5_join_audit.csv", pd.DataFrame(audits))
        atomic_csv(OUT / "QRAW_RR5_preflight_common_grid_preview.csv", joined)
        atomic_csv(OUT / "QRAW_RR5_task_status.csv", task)
        atomic_json(
            OUT / "QRAW_RR5_probe_manifest.json",
            {
                "schema_version": SCHEMA,
                "status": "complete" if complete else "blocked",
                "probe_inventory": str(OUT / "QRAW_RR5_probe_inventory.csv"),
                "source_probe_manifest": str(qprobe.PROBE_MANIFEST),
                "used_rows_per_probe": 32,
                "sample_rule": "first_32_from_formal_probe_order",
                "created_utc": b3.utc_now(),
            },
        )
    return {"status": status, "complete": complete, "checks": checks}


def samples_for(probe: str, tokenizer: Any, n: int = 32) -> list[c4.PreparedSample]:
    samples = c4.prepare_samples(
        qprobe.corpus_path(probe),
        tokenizer,
        corpus_id=f"qraw_rr5_q64:{probe}",
        window_seed=c4.WINDOW_SEED,
        max_context_tokens=c4.MAX_CONTEXT_TOKENS,
    )
    samples = samples[:n]
    if len(samples) != n:
        raise RuntimeError(f"{probe} has {len(samples)} samples, expected {n}")
    return samples


def profile_model(model: Any, samples: list[c4.PreparedSample], device: str, args: argparse.Namespace) -> dict[str, Any]:
    return campaign.collect_profile(
        model,
        samples,
        [LAYER],
        device,
        keep_factors=False,
        keep_residual_samples=True,
        factor_layers=(),
        forward_batch_size=args.forward_batch_size,
        max_batch_tokens=args.max_batch_tokens,
        early_stop=True,
    )


def sample_mean_matrix(profile: dict[str, Any]) -> torch.Tensor:
    return torch.stack([sample[LAYER].float() for sample in profile["residual_sample_means"]], dim=0)


def pairwise_cosine_mean(matrix: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(matrix.float(), dim=1, eps=1e-12)
    gram = normalized @ normalized.T
    count = gram.shape[0]
    if count <= 1:
        return 0.0
    return float((gram.sum() - gram.diagonal().sum()) / (count * (count - 1)))


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    width = min(left.shape[0], right.shape[0])
    left = left[:width].float()
    right = right[:width].float()
    left -= left.mean(dim=0, keepdim=True)
    right -= right.mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    numerator = torch.sum(left_gram * right_gram)
    denominator = torch.linalg.vector_norm(left_gram) * torch.linalg.vector_norm(right_gram)
    return float(numerator / denominator.clamp_min(1e-30))


def raw_row(profile: dict[str, Any], base_profile: dict[str, Any], arm: str, step: int, probe: str, device: str) -> dict[str, Any]:
    second = profile["residual_second"][LAYER].to(device=device, dtype=torch.float32)
    mean = profile["residual_mean"][LAYER].to(device=device, dtype=torch.float32)
    covariance = second - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).float().cpu().numpy()[::-1].astype(np.float64)
    trace = max(float(eigenvalues.sum()), 1e-300)
    probabilities = eigenvalues / trace
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, None))))
    participation = trace * trace / max(float(np.square(eigenvalues).sum()), 1e-300)
    current_means = sample_mean_matrix(profile).to(device)
    base_means = sample_mean_matrix(base_profile).to(device)
    centered = current_means - current_means.mean(dim=0, keepdim=True)
    row = {
        "model": MODEL,
        "arm": arm,
        "checkpoint": step,
        "probe_name": probe,
        "layer": LAYER,
        "native_object": "raw residual-stream sample means and centered covariance",
        "n_samples": int(profile["n_samples"]),
        "normalized_entropy_effective_rank": math.exp(entropy),
        "participation_ratio": participation,
        "top1_explained_share": float(eigenvalues[:1].sum() / trace),
        "top8_explained_share": float(eigenvalues[:8].sum() / trace),
        "top32_explained_share": float(eigenvalues[:32].sum() / trace),
        "raw_anisotropy": pairwise_cosine_mean(current_means),
        "centered_anisotropy": pairwise_cosine_mean(centered),
        "linear_cka_vs_step0": linear_cka(base_means, current_means),
    }
    del second, mean, covariance, current_means, base_means, centered
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def base_profile_path(probe: str, smoke: bool) -> Path:
    branch = "smoke" if smoke else "formal"
    return BASE_PROFILE_ROOT / branch / f"{probe}.pt"


def cell_path(arm: str, step: int, probe: str, smoke: bool) -> Path:
    branch = "smoke" if smoke else "formal"
    return CELL_ROOT / branch / arm / s3.step_label(step) / f"{probe}.json"


def run_base_profiles(args: argparse.Namespace, smoke: bool) -> list[dict[str, Any]]:
    tokenizer = qprobe.load_qwen_tokenizer()
    model = campaign.load_model(s3.BASE_MODEL, args.device)
    rows = []
    probes = PROBES[:1] if smoke else PROBES
    sample_n = 2 if smoke else 32
    try:
        for probe in probes:
            target = base_profile_path(probe, smoke)
            if target.is_file():
                payload = torch.load(target, map_location="cpu", weights_only=False)
                rows.append(payload["manifest"])
                continue
            samples = samples_for(probe, tokenizer, sample_n)
            prof = profile_model(model, samples, args.device, args)
            manifest = {
                "model": MODEL,
                "arm": "base",
                "checkpoint": 0,
                "probe_name": probe,
                "layer": LAYER,
                "n_samples": len(samples),
                "sample_ids_sha256": b3.sha256_json([s.sample_id for s in samples]),
                "corpus": str(qprobe.corpus_path(probe)),
                "forward_execution": prof["forward_execution"],
                "created_utc": b3.utc_now(),
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"profile": prof, "manifest": manifest}, target)
            rows.append(manifest | {"artifact": str(target), "sha256": sha256(target)})
            del prof, samples
            gc.collect()
    finally:
        campaign.unload_model(model)
    return rows


def run_forward(args: argparse.Namespace, smoke: bool = False) -> None:
    ensure_heavy_imports()
    if not smoke:
        pf = preflight(write=True)
        if not pf["complete"]:
            raise RuntimeError(pf["status"])
    base_rows = run_base_profiles(args, smoke)
    tokenizer = qprobe.load_qwen_tokenizer()
    arms = ("opd",) if smoke else ARMS
    steps = (5,) if smoke else CHECKPOINTS
    probes = (PROBES[0],) if smoke else PROBES
    sample_n = 2 if smoke else 32
    raw_rows = []
    status_rows = []
    for arm in arms:
        for step in steps:
            with fatr1.materialized_model("qwen", arm, step) as model_path:
                model = campaign.load_model(model_path, args.device)
                try:
                    for probe in probes:
                        target = cell_path(arm, step, probe, smoke)
                        cached = b3.read_json(target, {})
                        if cached.get("status") == "complete" and cached.get("n_samples") == sample_n:
                            raw_rows.append(cached["raw_row"])
                            status_rows.append({k: cached[k] for k in ("model", "arm", "checkpoint", "probe_name", "status", "n_samples", "wall_seconds")})
                            continue
                        start = time.time()
                        samples = samples_for(probe, tokenizer, sample_n)
                        profile = profile_model(model, samples, args.device, args)
                        base_payload = torch.load(base_profile_path(probe, smoke), map_location="cpu", weights_only=False)
                        row = raw_row(profile, base_payload["profile"], arm, step, probe, args.device)
                        payload = {
                            "schema_version": SCHEMA,
                            "task": TASK,
                            "status": "complete",
                            "model": MODEL,
                            "arm": arm,
                            "checkpoint": step,
                            "probe_name": probe,
                            "layer": LAYER,
                            "n_samples": sample_n,
                            "raw_row": row,
                            "wall_seconds": round(time.time() - start, 3),
                            "model_path": str(model_path),
                            "model_hash": s3.model_integrity(model_path),
                            "forward_execution": profile["forward_execution"],
                            "created_utc": b3.utc_now(),
                        }
                        atomic_json(target, payload)
                        raw_rows.append(row)
                        status_rows.append({k: payload[k] for k in ("model", "arm", "checkpoint", "probe_name", "status", "n_samples", "wall_seconds")})
                        del profile, samples, base_payload
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                finally:
                    campaign.unload_model(model)
    branch = "smoke" if smoke else "formal"
    atomic_csv(OUT / f"QRAW_RR5_{branch}_raw_representation_suite.csv", pd.DataFrame(raw_rows))
    atomic_csv(OUT / f"QRAW_RR5_{branch}_forward_status.csv", pd.DataFrame(status_rows))
    if not smoke:
        atomic_json(OUT / "QRAW_qwen_l18_base_profiles_manifest.json", {"schema_version": SCHEMA, "status": "complete", "rows": base_rows})


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    return float(pd.Series(a[valid]).rank(method="average").corr(pd.Series(b[valid]).rank(method="average")))


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def regression_metrics(y: np.ndarray, pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    ss_res = float(np.square(y - pred).sum())
    ss_tot = float(np.square(y - y.mean()).sum()) if len(y) else 0.0
    return {
        f"{prefix}r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        f"{prefix}mae": float(np.mean(np.abs(y - pred))) if len(y) else float("nan"),
        f"{prefix}spearman": rank_corr(y, pred),
    }


def classification_metrics(y: np.ndarray, prob: np.ndarray, prefix: str = "") -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(prob)
    y = y[valid].astype(int)
    prob = np.clip(prob[valid], 1e-6, 1 - 1e-6)
    pred = (prob >= 0.5).astype(int)
    pos = y == 1
    neg = y == 0
    tpr = float(np.mean(pred[pos] == 1)) if pos.any() else float("nan")
    tnr = float(np.mean(pred[neg] == 0)) if neg.any() else float("nan")
    return {
        f"{prefix}auc": auc_score(y, prob),
        f"{prefix}log_loss": float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))) if len(y) else float("nan"),
        f"{prefix}balanced_accuracy": float(np.nanmean([tpr, tnr])),
    }


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    Xd = np.column_stack([np.ones(X.shape[0]), X])
    penalty = np.eye(Xd.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    if alpha == 0:
        return np.linalg.pinv(Xd) @ y
    return np.linalg.pinv(Xd.T @ Xd + penalty) @ Xd.T @ y


def predict_linear(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(X.shape[0]), X]) @ beta


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    Xd = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(Xd.shape[1], dtype=float)
    lr = 0.1
    for _ in range(2500):
        z = np.clip(Xd @ beta, -40, 40)
        p = 1 / (1 + np.exp(-z))
        grad = Xd.T @ (p - y) / len(y)
        grad[1:] += float(l2) * beta[1:]
        beta -= lr * grad
    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(X.shape[0]), X]) @ beta, -40, 40)))


def standardize(X: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0)
    std[std == 0] = 1.0
    return (X[train] - mean) / std, (X[test] - mean) / std


def nested_select(X: np.ndarray, y: np.ndarray, groups: np.ndarray, train: np.ndarray, task: str) -> tuple[float, list[dict[str, float]]]:
    grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0] if task == "regression" else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    rows = []
    for alpha in grid:
        losses = []
        for held in sorted(set(groups[train])):
            tr = train & (groups != held)
            va = train & (groups == held)
            if va.sum() == 0:
                continue
            Xt, Xv = standardize(X, tr, va)
            if task == "regression":
                pred = predict_linear(Xv, fit_ridge(Xt, y[tr], alpha))
                losses.append(float(np.mean(np.abs(y[va] - pred))))
            else:
                if len(set(y[tr].astype(int))) < 2:
                    continue
                prob = np.clip(predict_logistic(Xv, fit_logistic(Xt, y[tr], alpha)), 1e-6, 1 - 1e-6)
                yy = y[va]
                losses.append(float(-np.mean(yy * np.log(prob) + (1 - yy) * np.log(1 - prob))))
        rows.append({"regularization": float(alpha), "inner_loss": float(np.mean(losses)) if losses else float("inf")})
    best = min(rows, key=lambda r: (r["inner_loss"], r["regularization"]))
    return float(best["regularization"]), rows


def feature_blocks() -> dict[str, list[str]]:
    return {
        "A": RAW_FEATURES,
        "C5": ["c_epsilon"],
        "Pk5": ["p_k4", "p_k8", "p_k16", "p_k32"],
        "A+C5": RAW_FEATURES + ["c_epsilon"],
        "Pk5+A": ["p_k4", "p_k8", "p_k16", "p_k32"] + RAW_FEATURES,
        "Pk5+C5": ["p_k4", "p_k8", "p_k16", "p_k32", "c_epsilon"],
        "Pk5+A+C5": ["p_k4", "p_k8", "p_k16", "p_k32"] + RAW_FEATURES + ["c_epsilon"],
    }


def build_common_grid(raw: pd.DataFrame) -> pd.DataFrame:
    keys = exact_keys()
    c5 = load_c5()
    pk = load_pk()
    outputs = load_outputs()
    common = keys.merge(raw, on=["model", "arm", "checkpoint", "probe_name", "layer"], how="left")
    common = common.merge(c5, on=["model", "arm", "checkpoint", "probe_name", "layer"], how="left")
    common = common.merge(pk, on=["model", "arm", "checkpoint", "probe_name", "layer"], how="left")
    common = common.merge(outputs, on=["model", "arm", "checkpoint", "probe_name", "layer"], how="left")
    common["is_opd"] = common["arm"].eq("opd").astype(int)
    common["row_id"] = np.arange(len(common))
    return common


def nested_models(common: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    blocks = feature_blocks()
    metric_rows, fold_rows, pred_rows = [], [], []
    for target in TARGETS:
        task = "classification" if target == "is_opd" else "regression"
        for block, cols in blocks.items():
            subset = common.dropna(subset=cols + [target]).copy().reset_index(drop=True)
            X = subset[cols].to_numpy(float)
            y = subset[target].to_numpy(float)
            groups = subset["checkpoint"].to_numpy(int)
            oof = np.full(len(y), np.nan)
            for held in CHECKPOINTS:
                train = groups != held
                test = groups == held
                Xt, Xv = standardize(X, train, test)
                selected, inner = nested_select(X, y, groups, train, task)
                if task == "regression":
                    pred = predict_linear(Xv, fit_ridge(Xt, y[train], selected))
                    oof[test] = pred
                    metrics = regression_metrics(y[test], pred, "test_")
                    extra = {"target_mean": float(y[test].mean()), "target_std": float(y[test].std(ddof=0))}
                else:
                    prob = predict_logistic(Xv, fit_logistic(Xt, y[train], selected))
                    oof[test] = prob
                    metrics = classification_metrics(y[test], prob, "test_")
                    extra = {"n_positive": int((y[test] == 1).sum()), "n_negative": int((y[test] == 0).sum())}
                fold_rows.append(
                    {
                        "model": MODEL,
                        "target": target,
                        "task_type": task,
                        "feature_block": block,
                        "heldout_checkpoint": held,
                        "selected_regularization": selected,
                        "selection_metric": "inner_mae" if task == "regression" else "inner_log_loss",
                        "inner_grid": json.dumps(inner, sort_keys=True),
                        "test_n": int(test.sum()),
                        **metrics,
                        **extra,
                    }
                )
                for idx, pred_value in zip(np.where(test)[0], oof[test], strict=True):
                    pred_rows.append(
                        {
                            "model": MODEL,
                            "target": target,
                            "task_type": task,
                            "feature_block": block,
                            "row_id": int(subset.loc[idx, "row_id"]),
                            "arm": subset.loc[idx, "arm"],
                            "checkpoint": int(subset.loc[idx, "checkpoint"]),
                            "probe_name": subset.loc[idx, "probe_name"],
                            "y_true": float(y[idx]),
                            "y_pred": float(pred_value),
                        }
                    )
            if task == "regression":
                om = regression_metrics(y, oof, "")
                metric_rows.append(
                    {
                        "model": MODEL,
                        "target": target,
                        "task_type": task,
                        "feature_block": block,
                        "features": ",".join(cols),
                        "n_common": int(len(y)),
                        "n_checkpoint_groups": int(len(set(groups))),
                        "n_oof": int(np.isfinite(oof).sum()),
                        "r2_oof": om["r2"],
                        "mae_oof": om["mae"],
                        "spearman_oof": om["spearman"],
                    }
                )
            else:
                om = classification_metrics(y, oof, "")
                metric_rows.append(
                    {
                        "model": MODEL,
                        "target": target,
                        "task_type": task,
                        "feature_block": block,
                        "features": ",".join(cols),
                        "n_common": int(len(y)),
                        "n_checkpoint_groups": int(len(set(groups))),
                        "n_oof": int(np.isfinite(oof).sum()),
                        "auc_oof": om["auc"],
                        "log_loss_oof": om["log_loss"],
                        "balanced_accuracy_oof": om["balanced_accuracy"],
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def side_by_side(qmetrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    llama_source = MINI / "equal5_non_qk/EQUAL5_nested_metrics.csv"
    llama = pd.read_csv(llama_source).copy()
    keep = ["model", "target", "task_type", "feature_block", "n_common", "r2_oof", "mae_oof", "spearman_oof", "auc_oof", "log_loss_oof", "balanced_accuracy_oof"]
    side = pd.concat([llama[[c for c in keep if c in llama.columns]], qmetrics[[c for c in keep if c in qmetrics.columns]]], ignore_index=True, sort=False)
    rows = []
    for model, g in side.groupby("model"):
        for target, gt in g.groupby("target"):
            by = gt.set_index("feature_block")
            for comparison, metric, bigger in [
                ("C5_vs_A", "r2_oof" if target != "is_opd" else "auc_oof", True),
                ("C5_vs_Pk5", "r2_oof" if target != "is_opd" else "auc_oof", True),
                ("A+C5_vs_C5", "r2_oof" if target != "is_opd" else "auc_oof", True),
                ("Pk5+C5_vs_Pk5", "r2_oof" if target != "is_opd" else "auc_oof", True),
            ]:
                left, right = comparison.split("_vs_")
                if left in by.index and right in by.index and metric in by.columns:
                    rows.append(
                        {
                            "model": model,
                            "target": target,
                            "comparison": comparison,
                            "metric": metric,
                            "left_value": float(by.loc[left, metric]),
                            "right_value": float(by.loc[right, metric]),
                            "left_minus_right": float(by.loc[left, metric] - by.loc[right, metric]),
                        }
                    )
    return side, pd.DataFrame(rows)


def write_handoff_and_manifest(status: str, metrics: pd.DataFrame, outputs: dict[str, dict[str, object]]) -> None:
    outputs = dict(outputs)
    outputs.pop("QRAW_RR5_manifest.json", None)
    outputs.pop("QRAW_RR5_theory_handoff.md", None)
    manifest = {
        "schema_version": SCHEMA,
        "task": TASK,
        "status": status,
        "created_utc": b3.utc_now(),
        "git_commit": git_commit(),
        "numeric_protocol": {
            "model_forward_dtype": "existing Qwen formal inference dtype via campaign.load_model",
            "accumulation_dtype": "FP32 residual mean/second/sample means",
            "covariance_eig_dtype": "torch FP32 eigvalsh, eigenvalues exported as FP64 numpy for ratios",
            "layer": LAYER,
            "sample_rule": "first_32_from_formal_probe_order",
            "early_stop": True,
            "training": "none",
            "rollout": "none",
            "new_functional_svd": "none",
        },
        "cross_model_protocol": {
            "llama_source": str(MINI / "equal5_non_qk/EQUAL5_nested_metrics.csv"),
            "qwen_source": str(OUT / "QRAW_RR5_qwen_nested_metrics.csv"),
            "feature_block_renaming": "none",
            "legacy_rr5_source_used": False,
        },
        "acceptance": {
            "preflight_state_cells": "64/64",
            "raw_activation_state_rows": "64/64",
            "eight_raw_features_finite": "512/512",
            "outer_folds": 4,
            "test_rows_per_fold": 16,
            "grouped_metrics": 28,
            "fold_rows": 112,
            "oof_predictions": 1792,
            "imputation": "none",
            "nearest_matching": "none",
            "probe_substitution": "none",
            "cross_model_llama_equal5_source": "equal5_non_qk/EQUAL5_nested_metrics.csv",
            "cross_model_feature_block_renaming": "none",
        },
        "outputs": outputs,
    }
    atomic_json(OUT / "QRAW_RR5_manifest.json", manifest)
    lines = [
        "# QRAW-RR5-Q64 handoff",
        "",
        f"- created_utc: `{manifest['created_utc']}`",
        f"- status: `{status}`",
        "- scope: Qwen L18 raw residual-stream activation suite and strict RR5 common grid.",
        "- correction: cross-model side-by-side now reads Llama from `equal5_non_qk/EQUAL5_nested_metrics.csv`.",
        "- correction: no legacy RR5 source and no C-to-C5 feature-block renaming are used.",
        "- exclusions: no training, no rollout, no behavior Eval, no new WS SVD, no paper/human_read edits.",
        "- sample rule: first 32 rows from each formal Qwen probe corpus, preserving sample order and text.",
        "",
        "## Row Counts",
        "",
        "| file | rows | sha256 |",
        "|---|---:|---|",
    ]
    for name, info in outputs.items():
        lines.append(f"| `{name}` | {info['rows'] if info['rows'] is not None else 'NA'} | `{info['sha256']}` |")
    lines += [
        "",
        "## Headline Metrics",
        "",
        metrics.to_markdown(index=False),
    ]
    atomic_text(OUT / "QRAW_RR5_theory_handoff.md", "\n".join(lines) + "\n")
    manifest["outputs"]["QRAW_RR5_theory_handoff.md"] = {"path": str(OUT / "QRAW_RR5_theory_handoff.md"), "rows": None, "sha256": sha256(OUT / "QRAW_RR5_theory_handoff.md")}
    atomic_json(OUT / "QRAW_RR5_manifest.json", manifest)


def collect_outputs() -> dict[str, dict[str, object]]:
    outputs = {}
    for path in sorted(OUT.glob("QRAW_RR5_*")) + sorted(OUT.glob("QRAW_qwen_*")):
        if path.is_file():
            rows = None
            if path.suffix == ".csv":
                rows = max(0, sum(1 for _ in path.open(encoding="utf-8")) - 1)
            elif path.suffix == ".parquet":
                rows = int(pd.read_parquet(path).shape[0])
            outputs[path.name] = {"path": str(path), "rows": rows, "sha256": sha256(path)}
    return outputs


def finalize_cross_model_only() -> None:
    metrics = pd.read_csv(OUT / "QRAW_RR5_qwen_nested_metrics.csv")
    side, wins = side_by_side(metrics)
    atomic_csv(OUT / "QRAW_RR5_cross_model_side_by_side.csv", side)
    atomic_csv(OUT / "QRAW_RR5_cross_model_wins.csv", wins)
    outputs = collect_outputs()
    status = "COMPLETE_QRAW_RR5_Q64_EQUAL5_CROSS_MODEL_CORRECTED"
    write_handoff_and_manifest(status, metrics, outputs)


def finalize() -> None:
    raw = pd.read_csv(OUT / "QRAW_RR5_formal_raw_representation_suite.csv")
    raw = raw[raw["checkpoint"].isin(CHECKPOINTS) & raw["probe_name"].isin(PROBES)]
    atomic_csv(OUT / "QRAW_RR5_qwen_raw_representation_suite.csv", raw)
    common = build_common_grid(raw)
    atomic_csv(OUT / "QRAW_RR5_qwen_common_grid.csv", common)
    metrics, folds, preds = nested_models(common)
    atomic_csv(OUT / "QRAW_RR5_qwen_nested_metrics.csv", metrics)
    atomic_csv(OUT / "QRAW_RR5_qwen_nested_folds.csv", folds)
    preds.to_parquet(OUT / "QRAW_RR5_qwen_oof_predictions.parquet", index=False)
    side, wins = side_by_side(metrics)
    atomic_csv(OUT / "QRAW_RR5_cross_model_side_by_side.csv", side)
    atomic_csv(OUT / "QRAW_RR5_cross_model_wins.csv", wins)
    finite = int(np.isfinite(raw[RAW_FEATURES].to_numpy(float)).sum())
    status = "COMPLETE_QRAW_RR5_Q64_EXACT_COMMON_GRID" if (len(raw) == 64 and len(common) == 64 and len(metrics) == 28 and len(folds) == 112 and len(preds) == 1792 and finite == 512) else "PARTIAL_QRAW_RR5_Q64"
    task = pd.read_csv(OUT / "QRAW_RR5_task_status.csv") if (OUT / "QRAW_RR5_task_status.csv").is_file() else pd.DataFrame()
    task = pd.concat([task, pd.DataFrame([{"task": "formal_finalize", "status": status, "raw_rows": len(raw), "common_rows": len(common), "finite_raw_features": finite, "metrics_rows": len(metrics), "fold_rows": len(folds), "prediction_rows": len(preds)}])], ignore_index=True)
    atomic_csv(OUT / "QRAW_RR5_task_status.csv", task)
    outputs = collect_outputs()
    write_handoff_and_manifest(status, metrics, outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal", "finalize", "finalize-cross", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--forward-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode in ("preflight", "all"):
        payload = preflight(write=True)
        if args.mode == "preflight":
            print(json.dumps(payload, indent=2, ensure_ascii=False))
            return
    if args.mode in ("smoke",):
        run_forward(args, smoke=True)
        return
    if args.mode in ("formal", "all"):
        run_forward(args, smoke=False)
    if args.mode in ("finalize", "formal", "all"):
        finalize()
    if args.mode == "finalize-cross":
        finalize_cross_model_only()


if __name__ == "__main__":
    main()
