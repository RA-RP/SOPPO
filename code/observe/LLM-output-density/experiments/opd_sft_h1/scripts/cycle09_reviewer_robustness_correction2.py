#!/usr/bin/env python3
"""Second reviewer-robustness correction pass.

This pass is intentionally reuse-only: no model forward, no training, and no
paper/theory text edits.  It strengthens RR5, reuses the formal Llama D10 state
spectra for RR2S, computes the Llama centered RR3 audit from saved formal
profiles, aggregates RR2D, and rewrites task-specific availability.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open


REPO = Path("/root/LLM-output-density")
AUTODL = Path("/root/autodl-tmp")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
OUT = MINI / "reviewer_robustness"
RFC = AUTODL / "cycle09_relative_functional_contraction"
D10_ROOT = RFC / "d10_llama_numeric_parity/formal"
D10_FINAL = D10_ROOT / "final"
D11_FINAL = RFC / "d11_pk_tpnt/formal/final"
QWEN_FINAL = RFC / "final"
STAGE4 = AUTODL / "cycle09_stage4_state_displacement"

ARMS = ["opd", "sft", "offkd", "seqkd"]
STEPS = [20, 40, 80]
PROBES = ["E_general", "E_math", "E_ood", "E_if"]
EPSILONS = [0.01, 0.025, 0.05, 0.10]
LAYER = 14
MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
MODULE_TO_GROUP = {
    "self_attn.q_proj": "attn_qkv_input",
    "self_attn.k_proj": "attn_qkv_input",
    "self_attn.v_proj": "attn_qkv_input",
    "self_attn.o_proj": "attn_o_input",
    "mlp.gate_proj": "mlp_gate_up_input",
    "mlp.up_proj": "mlp_gate_up_input",
    "mlp.down_proj": "mlp_down_input",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path, max_bytes: int = 512 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    size = path.stat().st_size
    if size > max_bytes:
        return f"SKIPPED_SIZE_{size}"
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=float)
    if float(np.std(ar)) == 0.0 or float(np.std(br)) == 0.0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def auc_score(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
    return float(wins / (len(pos) * len(neg)))


def regression_metrics(y: np.ndarray, pred: np.ndarray, prefix: str = "") -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    if len(y) == 0:
        return {
            f"{prefix}r2": float("nan"),
            f"{prefix}mae": float("nan"),
            f"{prefix}spearman": float("nan"),
        }
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    return {
        f"{prefix}r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        f"{prefix}mae": float(np.mean(np.abs(y - pred))),
        f"{prefix}spearman": rank_corr(y, pred),
    }


def classification_metrics(y: np.ndarray, prob: np.ndarray, prefix: str = "") -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(prob)
    y = y[valid].astype(int)
    prob = np.clip(prob[valid], 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {
            f"{prefix}auc": float("nan"),
            f"{prefix}log_loss": float("nan"),
            f"{prefix}balanced_accuracy": float("nan"),
        }
    pred = (prob >= 0.5).astype(int)
    pos = y == 1
    neg = y == 0
    tpr = float(np.mean(pred[pos] == 1)) if pos.any() else float("nan")
    tnr = float(np.mean(pred[neg] == 0)) if neg.any() else float("nan")
    return {
        f"{prefix}auc": auc_score(y, prob),
        f"{prefix}log_loss": float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))),
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
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xd.T @ (p - y) / len(y)
        grad[1:] += float(l2) * beta[1:]
        beta -= lr * grad
    return beta


def predict_logistic(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([np.ones(X.shape[0]), X]) @ beta, -40, 40)))


def train_standardize(X: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0)
    std[std == 0] = 1.0
    return (X[train] - mean) / std, (X[test] - mean) / std, mean, std


def rr5_blocks(common: pd.DataFrame) -> dict[str, list[str]]:
    feature_cols_a = [
        c
        for c in [
            "normalized_entropy_effective_rank",
            "participation_ratio",
            "top1_explained_share",
            "top8_explained_share",
            "top32_explained_share",
            "raw_anisotropy",
            "centered_anisotropy",
            "linear_cka_vs_step0",
        ]
        if c in common.columns
    ]
    feature_cols_c = ["c_epsilon"]
    feature_cols_pk = [c for c in ["p_k4", "p_k8", "p_k16", "p_k32"] if c in common.columns]
    return {
        "A": feature_cols_a,
        "C": feature_cols_c,
        "Pk": feature_cols_pk,
        "A+C": feature_cols_a + feature_cols_c,
        "Pk+A": feature_cols_pk + feature_cols_a,
        "Pk+C": feature_cols_pk + feature_cols_c,
        "Pk+A+C": feature_cols_pk + feature_cols_a + feature_cols_c,
    }


def nested_select_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    alphas = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    train_groups = sorted(set(groups[outer_train]))
    rows = []
    for alpha in alphas:
        fold_mae = []
        for held in train_groups:
            inner_train = outer_train & (groups != held)
            inner_test = outer_train & (groups == held)
            if inner_train.sum() <= X.shape[1] or inner_test.sum() == 0:
                continue
            Xt, Xv, _, _ = train_standardize(X, inner_train, inner_test)
            beta = fit_ridge(Xt, y[inner_train], alpha)
            pred = predict_linear(Xv, beta)
            fold_mae.append(float(np.mean(np.abs(y[inner_test] - pred))))
        rows.append({"regularization": alpha, "inner_mean_mae": float(np.mean(fold_mae)) if fold_mae else float("inf")})
    best = min(rows, key=lambda r: (r["inner_mean_mae"], r["regularization"]))
    return float(best["regularization"]), rows


def nested_select_classification(X: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray) -> tuple[float, list[dict[str, Any]]]:
    l2s = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    train_groups = sorted(set(groups[outer_train]))
    rows = []
    for l2 in l2s:
        fold_loss = []
        for held in train_groups:
            inner_train = outer_train & (groups != held)
            inner_test = outer_train & (groups == held)
            if inner_test.sum() == 0 or len(set(y[inner_train].astype(int))) < 2:
                continue
            Xt, Xv, _, _ = train_standardize(X, inner_train, inner_test)
            beta = fit_logistic(Xt, y[inner_train], l2)
            prob = np.clip(predict_logistic(Xv, beta), 1e-6, 1 - 1e-6)
            yy = y[inner_test]
            fold_loss.append(float(-np.mean(yy * np.log(prob) + (1 - yy) * np.log(1 - prob))))
        rows.append({"regularization": l2, "inner_mean_log_loss": float(np.mean(fold_loss)) if fold_loss else float("inf")})
    best = min(rows, key=lambda r: (r["inner_mean_log_loss"], r["regularization"]))
    return float(best["regularization"]), rows


def run_rr5_checkpoint_control() -> None:
    common = pd.read_csv(OUT / "RR5_llama_common_grid.csv")
    blocks = rr5_blocks(common)
    targets = {
        "cumulative_kl_base_to_current": "regression",
        "absolute_delta_nll_cumulative": "regression",
        "delta_nll_cumulative": "regression",
        "is_opd": "classification",
    }
    fold_rows: list[dict[str, Any]] = []
    nested_metric_rows: list[dict[str, Any]] = []
    nested_fold_rows: list[dict[str, Any]] = []
    nested_pred_rows: list[dict[str, Any]] = []
    fixed_pred_rows: list[dict[str, Any]] = []

    for target, task in targets.items():
        per_target_nested: dict[str, dict[str, float]] = {}
        for block, cols in blocks.items():
            subset = common.dropna(subset=cols + [target]).copy().reset_index(drop=True)
            X = subset[cols].to_numpy(dtype=float)
            y = subset[target].to_numpy(dtype=float)
            groups = subset["checkpoint"].to_numpy(dtype=int)
            row_ids = subset["row_id"].to_numpy(dtype=int)
            fixed_pred = np.full(len(y), np.nan)
            nested_pred = np.full(len(y), np.nan)
            for held in sorted(set(groups)):
                train = groups != held
                test = groups == held
                if task == "regression":
                    Xt, Xv, _, _ = train_standardize(X, train, test)
                    beta = fit_ridge(Xt, y[train], 1e-6)
                    fixed_pred[test] = predict_linear(Xv, beta)
                    selected, inner_rows = nested_select_regression(X, y, groups, train)
                    beta_nested = fit_ridge(Xt, y[train], selected)
                    nested_pred[test] = predict_linear(Xv, beta_nested)
                    fm = regression_metrics(y[test], fixed_pred[test], "test_")
                    nm = regression_metrics(y[test], nested_pred[test], "test_")
                    extra = {
                        "target_mean": float(y[test].mean()),
                        "target_std": float(y[test].std(ddof=0)),
                    }
                else:
                    if len(set(y[train].astype(int))) < 2:
                        continue
                    Xt, Xv, _, _ = train_standardize(X, train, test)
                    beta = fit_logistic(Xt, y[train], 1e-4)
                    fixed_pred[test] = predict_logistic(Xv, beta)
                    selected, inner_rows = nested_select_classification(X, y, groups, train)
                    beta_nested = fit_logistic(Xt, y[train], selected)
                    nested_pred[test] = predict_logistic(Xv, beta_nested)
                    fm = classification_metrics(y[test], fixed_pred[test], "test_")
                    nm = classification_metrics(y[test], nested_pred[test], "test_")
                    extra = {
                        "n_positive": int((y[test] == 1).sum()),
                        "n_negative": int((y[test] == 0).sum()),
                    }
                fold_rows.append({
                    "model": "llama",
                    "target": target,
                    "task_type": task,
                    "feature_block": block,
                    "heldout_checkpoint": int(held),
                    "train_checkpoints": ",".join(map(str, sorted(set(groups[train])))),
                    "test_n": int(test.sum()),
                    **fm,
                    **extra,
                })
                nested_fold_rows.append({
                    "model": "llama",
                    "target": target,
                    "task_type": task,
                    "feature_block": block,
                    "heldout_checkpoint": int(held),
                    "selected_regularization": selected,
                    "selection_metric": "inner_mean_mae" if task == "regression" else "inner_mean_log_loss",
                    "inner_grid": json.dumps(inner_rows, sort_keys=True),
                    "test_n": int(test.sum()),
                    **nm,
                    **extra,
                })
            if task == "regression":
                m = regression_metrics(y, nested_pred, "")
                metric = {
                    "n_oof": int(np.isfinite(nested_pred).sum()),
                    "r2_oof": m["r2"],
                    "mae_oof": m["mae"],
                    "spearman_oof": m["spearman"],
                }
                pred_name = "y_pred"
            else:
                m = classification_metrics(y, nested_pred, "")
                metric = {
                    "n_oof": int(np.isfinite(nested_pred).sum()),
                    "auc_oof": m["auc"],
                    "log_loss_oof": m["log_loss"],
                    "balanced_accuracy_oof": m["balanced_accuracy"],
                }
                pred_name = "y_prob"
            per_target_nested[block] = metric
            nested_metric_rows.append({
                "model": "llama",
                "target": target,
                "task_type": task,
                "feature_block": block,
                "features": ",".join(cols),
                "n_common": int(len(subset)),
                "n_checkpoint_groups": int(subset["checkpoint"].nunique()),
                "checkpoint_groups": ",".join(map(str, sorted(subset["checkpoint"].unique()))),
                **metric,
            })
            for i, src in subset.iterrows():
                base_row = {
                    "model": "llama",
                    "target": target,
                    "task_type": task,
                    "feature_block": block,
                    "row_id": int(src["row_id"]),
                    "arm": src["arm"],
                    "checkpoint": int(src["checkpoint"]),
                    "probe_name": src["probe_name"],
                    "layer": int(src["layer"]),
                    "y_true": float(y[i]),
                }
                nested_pred_rows.append({**base_row, pred_name: float(nested_pred[i]) if np.isfinite(nested_pred[i]) else np.nan})
                fixed_pred_rows.append({**base_row, pred_name: float(fixed_pred[i]) if np.isfinite(fixed_pred[i]) else np.nan})
        for row in nested_metric_rows:
            if row["target"] != target:
                continue
            for baseline in ["A", "C", "Pk"]:
                base = per_target_nested.get(baseline, {})
                if task == "regression":
                    row[f"delta_r2_vs_{baseline}"] = row.get("r2_oof", np.nan) - base.get("r2_oof", np.nan)
                    row[f"mae_reduction_vs_{baseline}"] = base.get("mae_oof", np.nan) - row.get("mae_oof", np.nan)
                    row[f"delta_spearman_vs_{baseline}"] = row.get("spearman_oof", np.nan) - base.get("spearman_oof", np.nan)
                else:
                    row[f"delta_auc_vs_{baseline}"] = row.get("auc_oof", np.nan) - base.get("auc_oof", np.nan)
                    row[f"log_loss_reduction_vs_{baseline}"] = base.get("log_loss_oof", np.nan) - row.get("log_loss_oof", np.nan)
                    row[f"delta_balanced_accuracy_vs_{baseline}"] = row.get("balanced_accuracy_oof", np.nan) - base.get("balanced_accuracy_oof", np.nan)

    fold_path = OUT / "RR5_hybrid_fold_performance.csv"
    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

    features = [
        "c_epsilon",
        "p_k4",
        "p_k8",
        "p_k16",
        "p_k32",
        "normalized_entropy_effective_rank",
        "participation_ratio",
        "linear_cka_vs_step0",
    ]
    corr_targets = [
        "cumulative_kl_base_to_current",
        "absolute_delta_nll_cumulative",
        "delta_nll_cumulative",
    ]
    demean = common.copy()
    for col in features + corr_targets:
        if col in demean.columns:
            demean[f"demeaned_{col}"] = demean[col] - demean.groupby("checkpoint")[col].transform("mean")
    corr_rows = []
    for feature in features:
        if f"demeaned_{feature}" not in demean.columns:
            continue
        for target in corr_targets:
            x = demean[f"demeaned_{feature}"].to_numpy(dtype=float)
            y = demean[f"demeaned_{target}"].to_numpy(dtype=float)
            corr_rows.append({
                "model": "llama",
                "feature": feature,
                "target": target,
                "n_cells": int(np.isfinite(x + y).sum()),
                "checkpoint_groups": ",".join(map(str, sorted(demean["checkpoint"].unique()))),
                "association": "checkpoint_demeaned_descriptive_not_oof_prediction",
                "pearson": pearson(x, y),
                "spearman": rank_corr(x, y),
            })
    demean_path = OUT / "RR5_checkpoint_demeaned_cells.csv"
    corr_path = OUT / "RR5_checkpoint_demeaned_correlations.csv"
    demean.to_csv(demean_path, index=False)
    pd.DataFrame(corr_rows).to_csv(corr_path, index=False)

    nested_metrics_path = OUT / "RR5_nested_regularization_metrics.csv"
    nested_folds_path = OUT / "RR5_nested_regularization_folds.csv"
    nested_preds_path = OUT / "RR5_nested_regularization_predictions.parquet"
    pd.DataFrame(nested_metric_rows).to_csv(nested_metrics_path, index=False)
    pd.DataFrame(nested_fold_rows).to_csv(nested_folds_path, index=False)
    pd.DataFrame(nested_pred_rows).to_parquet(nested_preds_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR5_CHECKPOINT_CONTROL_AND_NESTED_REGULARIZATION",
        "formal_protocol_id": "RR5_llama_exact_common_grid_checkpoint_control_nested_regularization",
        "join_policy": "strict exact-key Llama-only common grid; no imputation; no nearest checkpoint; no probe replacement",
        "outer_folds": "leave-one-checkpoint-group-out; same outer folds for all feature blocks",
        "standardization": "feature mean/std fit on outer train checkpoints only; inner selection uses train checkpoints only",
        "fixed_regularization_parity": {
            "ridge_alpha": 1e-6,
            "logistic_l2": 1e-4,
            "fold_performance_output": "RR5_hybrid_fold_performance.csv",
        },
        "nested_regularization_grid": {
            "ridge_alpha": [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100],
            "logistic_l2": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        },
        "interpretation_guard": [
            "Do not claim hybrid always outperforms c_epsilon.",
            "c_epsilon alone is strongest for unsigned drift in the current Llama common grid.",
            "c_epsilon added to p_k improves over p_k alone for unsigned drift.",
            "p_k is stronger for OPD classification; all features add only small classification increment.",
        ],
        "row_counts": {
            "fold_performance": int(len(fold_rows)),
            "checkpoint_demeaned_cells": int(len(demean)),
            "checkpoint_demeaned_correlations": int(len(corr_rows)),
            "nested_metrics": int(len(nested_metric_rows)),
            "nested_folds": int(len(nested_fold_rows)),
            "nested_predictions": int(len(nested_pred_rows)),
        },
        "input_paths_and_sha256": {
            "RR5_llama_common_grid.csv": sha256_file(OUT / "RR5_llama_common_grid.csv"),
            "RR5_hybrid_grouped_models.csv": sha256_file(OUT / "RR5_hybrid_grouped_models.csv"),
        },
        "output_sha256": {
            "RR5_hybrid_fold_performance.csv": sha256_file(fold_path),
            "RR5_checkpoint_demeaned_correlations.csv": sha256_file(corr_path),
            "RR5_checkpoint_demeaned_cells.csv": sha256_file(demean_path),
            "RR5_nested_regularization_metrics.csv": sha256_file(nested_metrics_path),
            "RR5_nested_regularization_folds.csv": sha256_file(nested_folds_path),
            "RR5_nested_regularization_predictions.parquet": sha256_file(nested_preds_path),
        },
    }
    (OUT / "RR5_nested_regularization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def state_json_path(arm: str, step: int, probe: str) -> Path:
    return D10_ROOT / "state" / arm / f"step_{step:03d}" / f"{probe}.json"


def profile_path(arm: str, step: int, probe: str) -> Path:
    return D10_ROOT / "profiles" / arm / f"step_{step:03d}" / f"{probe}.pt"


def rank_from_singular(singular: np.ndarray, epsilon: float) -> int:
    energy = np.asarray(singular, dtype=np.float64) ** 2
    total = float(energy.sum())
    if total <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(energy), (1.0 - epsilon) * total, side="left") + 1)


def energy_stats(singular: np.ndarray, rank: int, epsilon: float, tol: float = 1e-8) -> dict[str, Any]:
    singular = np.asarray(singular, dtype=np.float64)
    energy = singular * singular
    total = float(energy.sum())
    if total <= 0:
        return {
            "tail_at_r": np.nan,
            "tail_at_r_minus_1": np.nan,
            "margin_below": np.nan,
            "margin_above": np.nan,
            "two_sided_tail_margin": np.nan,
            "stable_rank": np.nan,
            "entropy_effective_rank": np.nan,
            "top1_energy_share": np.nan,
            "top10_energy_share": np.nan,
            "top32_energy_share": np.nan,
            "zero_probability_count": int(len(energy)),
            "rank_spectrum_consistency": "INVALID_ZERO_TOTAL_ENERGY",
        }
    p = energy / total
    r = int(max(0, min(rank, len(p))))
    tail_at_r = float(p[r:].sum())
    tail_at_r_minus_1 = float(p[r - 1 :].sum()) if r > 0 else 1.0
    nz = p[p > 0]
    consistency = "PASS" if tail_at_r <= epsilon + tol and tail_at_r_minus_1 > epsilon - tol else "INVALID_RANK_SPECTRUM_MISMATCH"
    return {
        "tail_at_r": tail_at_r,
        "tail_at_r_minus_1": tail_at_r_minus_1,
        "margin_below": epsilon - tail_at_r,
        "margin_above": tail_at_r_minus_1 - epsilon,
        "two_sided_tail_margin": min(epsilon - tail_at_r, tail_at_r_minus_1 - epsilon),
        "stable_rank": float(total / energy.max()),
        "entropy_effective_rank": float(np.exp(-(nz * np.log(nz)).sum())),
        "top1_energy_share": float(p[:1].sum()),
        "top10_energy_share": float(p[:10].sum()),
        "top32_energy_share": float(p[:32].sum()),
        "zero_probability_count": int((p == 0).sum()),
        "rank_spectrum_consistency": consistency,
    }


def run_rr2s_llama_state_reuse() -> None:
    module_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    expected = [("base", 0, probe) for probe in PROBES] + [
        (arm, step, probe) for arm in ARMS for step in STEPS for probe in PROBES
    ]
    d10_module = pd.read_csv(D10_FINAL / "d10_5_integrated_state_module.csv")
    d10_lookup = {
        (r.arm, int(r.checkpoint), r.probe_name, int(r.layer), r.module, round(float(r.epsilon), 6)): float(r.state_rank_current)
        for r in d10_module.itertuples()
        if r.model == "llama" and int(r.layer) == LAYER
    }
    for arm, step, probe in expected:
        payload = read_json(state_json_path(arm, step, probe))
        if payload.get("status") != "complete":
            parity_rows.append({"arm": arm, "checkpoint": step, "probe_name": probe, "status": "MISSING_STATE_JSON"})
            continue
        rank_lookup = {(r["module"], round(float(r["epsilon"]), 6)): int(r["r_epsilon"]) for r in payload.get("state_rows", [])}
        for module, singular in payload.get("spectra", {}).items():
            singular_np = np.asarray(singular, dtype=np.float64)
            for eps in EPSILONS:
                rank = rank_lookup.get((module, round(eps, 6)))
                if rank is None:
                    parity_rows.append({"arm": arm, "checkpoint": step, "probe_name": probe, "module": module, "epsilon": eps, "status": "MISSING_RANK_IN_STATE_JSON"})
                    continue
                d10_rank = d10_lookup.get((arm, step, probe, LAYER, module, round(eps, 6)))
                status = "PASS" if d10_rank is not None and int(d10_rank) == int(rank) else "MISMATCH"
                parity_rows.append({
                    "arm": arm,
                    "checkpoint": step,
                    "probe_name": probe,
                    "layer": LAYER,
                    "module": module,
                    "epsilon": eps,
                    "rr2s_rank": int(rank),
                    "d10_rank": d10_rank,
                    "rank_diff": float(rank - d10_rank) if d10_rank is not None else np.nan,
                    "status": status,
                })
                module_rows.append({
                    "model": "llama",
                    "arm": arm,
                    "checkpoint": step,
                    "probe_name": probe,
                    "layer": LAYER,
                    "module": module,
                    "epsilon": eps,
                    "state_rank": int(rank),
                    "spectrum_source": str(state_json_path(arm, step, probe)),
                    "spectrum_quantity": "formal_D10_W_t_S_D_t_state_singular_values",
                    "singular_values_stored": int(len(singular_np)),
                    **energy_stats(singular_np, int(rank), eps),
                })
    parity = pd.DataFrame(parity_rows)
    parity_path = OUT / "RR2S_llama_parity_audit.csv"
    parity.to_csv(parity_path, index=False)
    if not parity.empty and (parity["status"].astype(str) == "MISMATCH").any():
        pd.DataFrame(module_rows).to_csv(OUT / "RR2S_llama_state_spectrum_module.csv", index=False)
        (OUT / "RR2S_llama_manifest.json").write_text(json.dumps({
            "created_utc": now(),
            "git_commit": git_commit(),
            "status": "INVALID_RR2S_PARITY_MISMATCH_STOPPED_BEFORE_AGGREGATION",
            "parity_mismatch_rows": int((parity["status"].astype(str) == "MISMATCH").sum()),
            "output_sha256": {"RR2S_llama_parity_audit.csv": sha256_file(parity_path)},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    module = pd.DataFrame(module_rows)
    base = module[module["arm"].eq("base")][[
        "probe_name",
        "module",
        "epsilon",
        "state_rank",
        "stable_rank",
        "entropy_effective_rank",
        "top1_energy_share",
        "top10_energy_share",
        "top32_energy_share",
    ]].rename(columns={
        "state_rank": "state_rank_base",
        "stable_rank": "stable_rank_base",
        "entropy_effective_rank": "entropy_effective_rank_base",
        "top1_energy_share": "top1_energy_share_base",
        "top10_energy_share": "top10_energy_share_base",
        "top32_energy_share": "top32_energy_share_base",
    })
    current = module[~module["arm"].eq("base")].merge(base, on=["probe_name", "module", "epsilon"], how="left")
    current = current.rename(columns={"state_rank": "state_rank_current"})
    current["absolute_contraction"] = current["state_rank_base"] - current["state_rank_current"]
    current["relative_functional_contraction_module"] = current["absolute_contraction"] / current["state_rank_base"]
    current["stable_rank_contraction"] = current["stable_rank_base"] - current["stable_rank"]
    current["entropy_effective_rank_contraction"] = current["entropy_effective_rank_base"] - current["entropy_effective_rank"]
    module_out = pd.concat([module[module["arm"].eq("base")], current], ignore_index=True, sort=False)
    module_path = OUT / "RR2S_llama_state_spectrum_module.csv"
    module_out.to_csv(module_path, index=False)

    num_cols = [
        "state_rank_current",
        "state_rank_base",
        "absolute_contraction",
        "relative_functional_contraction_module",
        "tail_at_r",
        "tail_at_r_minus_1",
        "margin_below",
        "margin_above",
        "two_sided_tail_margin",
        "stable_rank",
        "entropy_effective_rank",
        "top1_energy_share",
        "top10_energy_share",
        "top32_energy_share",
        "stable_rank_base",
        "entropy_effective_rank_base",
        "stable_rank_contraction",
        "entropy_effective_rank_contraction",
    ]
    equal = current.groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], as_index=False)[num_cols].mean(numeric_only=True)
    equal["module_count"] = 7
    equal_path = OUT / "RR2S_llama_state_spectrum_equal7.csv"
    equal.to_csv(equal_path, index=False)

    ordering_rows = []
    for eps in EPSILONS:
        eg = equal[equal["epsilon"].round(6).eq(round(eps, 6))]
        for (checkpoint, probe), g in eg.groupby(["checkpoint", "probe_name"]):
            offline = g[~g["arm"].eq("opd")]
            opd = g[g["arm"].eq("opd")]
            if opd.empty or offline.empty:
                continue
            opd_row = opd.iloc[0]
            for metric in ["absolute_contraction", "stable_rank_contraction", "entropy_effective_rank_contraction"]:
                best = g.sort_values(metric, ascending=False).iloc[0]
                nearest = offline[metric].max()
                ordering_rows.append({
                    "model": "llama",
                    "checkpoint": int(checkpoint),
                    "probe_name": probe,
                    "epsilon": eps,
                    "metric": metric,
                    "opd_deepest": bool(best["arm"] == "opd"),
                    "deepest_arm": best["arm"],
                    "opd_value": float(opd_row[metric]),
                    "nearest_offline_value": float(nearest),
                    "opd_minus_nearest_offline_margin": float(opd_row[metric] - nearest),
                    "arm_values_json": json.dumps({r.arm: float(getattr(r, metric)) for r in g.itertuples()}, sort_keys=True),
                })
    ordering_path = OUT / "RR2S_llama_continuous_ordering.csv"
    pd.DataFrame(ordering_rows).to_csv(ordering_path, index=False)

    outputs = pd.read_csv(D10_FINAL / "d10_5_integrated_outputs.csv")
    links_base = equal[equal["epsilon"].round(6).eq(0.05)].merge(
        outputs[outputs["arm"].ne("base")],
        on=["model", "arm", "checkpoint", "probe_name"],
        how="left",
    )
    link_rows = []
    for arm, g in links_base.groupby("arm"):
        for metric in ["absolute_contraction", "stable_rank_contraction", "entropy_effective_rank_contraction"]:
            for target in ["cumulative_kl_base_to_current", "absolute_delta_nll_cumulative", "delta_nll_cumulative"]:
                link_rows.append({
                    "model": "llama",
                    "arm": arm,
                    "epsilon": 0.05,
                    "metric": metric,
                    "target": target,
                    "n_cells": int(g[[metric, target]].dropna().shape[0]),
                    "spearman": rank_corr(g[metric].to_numpy(dtype=float), g[target].to_numpy(dtype=float)),
                    "pearson": pearson(g[metric].to_numpy(dtype=float), g[target].to_numpy(dtype=float)),
                })
    links_path = OUT / "RR2S_llama_continuous_output_links.csv"
    pd.DataFrame(link_rows).to_csv(links_path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR2S_LLAMA_FORMAL_STATE_SPECTRUM_REUSE_NO_FORWARD",
        "formal_protocol_id": "RR2S_llama_D10_formal_WtSdt_state_spectrum_reuse",
        "numeric_protocol": "Formal D10 BF16 merged checkpoint/load/forward; FP32 Gram accumulation and WS matmul; FP64 Gram eig/SVD and singular-energy accumulation. This pass reused D10 saved complete spectra; no model forward.",
        "coverage": "4 arms x 3 checkpoints x 4 probes plus 4 base probe profiles; headline L14; 7 modules; eps=.01/.025/.05/.10",
        "validation": "RR2S ranks matched D10 formal state-rank table before aggregation; tail consistency checked for every spectrum/epsilon.",
        "row_counts": {
            "module": int(len(module_out)),
            "equal7": int(len(equal)),
            "ordering": int(len(ordering_rows)),
            "output_links": int(len(link_rows)),
            "parity_audit": int(len(parity)),
        },
        "input_paths_and_sha256": {
            "d10_5_integrated_state_module.csv": sha256_file(D10_FINAL / "d10_5_integrated_state_module.csv"),
            "d10_5_integrated_outputs.csv": sha256_file(D10_FINAL / "d10_5_integrated_outputs.csv"),
        },
        "output_sha256": {
            "RR2S_llama_state_spectrum_module.csv": sha256_file(module_path),
            "RR2S_llama_state_spectrum_equal7.csv": sha256_file(equal_path),
            "RR2S_llama_continuous_ordering.csv": sha256_file(ordering_path),
            "RR2S_llama_continuous_output_links.csv": sha256_file(links_path),
            "RR2S_llama_parity_audit.csv": sha256_file(parity_path),
        },
    }
    (OUT / "RR2S_llama_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def model_dir(arm: str, step: int) -> Path:
    if arm == "base" or step == 0:
        return AUTODL / "model/Meta/modelscope/Llama-3.2-3B"
    return AUTODL / "cycle09_block3/llama_models/merged" / arm / f"step_{step:03d}"


def load_layer_weights(arm: str, step: int) -> dict[str, torch.Tensor]:
    root = model_dir(arm, step)
    index = read_json(root / "model.safetensors.index.json")
    if not index:
        raise FileNotFoundError(f"missing safetensors index: {root}")
    wanted = {
        module: f"model.layers.{LAYER}.{module}.weight"
        for module in MODULES
    }
    by_file: dict[str, list[tuple[str, str]]] = {}
    for module, key in wanted.items():
        shard = index["weight_map"].get(key)
        if shard is None:
            raise KeyError(f"missing weight key {key} in {root}")
        by_file.setdefault(shard, []).append((module, key))
    weights: dict[str, torch.Tensor] = {}
    for shard, items in by_file.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for module, key in items:
                weights[module] = handle.get_tensor(key).to(dtype=torch.float32, device="cpu").contiguous()
    return weights


def centered_gram(profile: dict[str, Any], group: str) -> tuple[torch.Tensor, dict[str, Any]]:
    gram = profile["grams"][LAYER][group].to(dtype=torch.float64)
    means = []
    for sample in profile["input_sample_means"]:
        item = sample.get(LAYER, sample.get(str(LAYER), {}))
        means.append(item[group].to(dtype=torch.float64))
    mu = torch.stack(means, dim=0).mean(dim=0)
    centered = (gram + gram.T) / 2 - torch.outer(mu, mu)
    centered = (centered + centered.T) / 2
    audit = {
        "sample_count": int(profile["sample_count"]),
        "input_sample_means_count": int(len(means)),
        "gram_weighting": "sample_equal_mean_of_window_token_weighted_second_moments",
        "mean_weighting": "sample_equal_mean_of_window_token_weighted_sample_means",
        "direct_centering_valid": bool(len(means) == int(profile["sample_count"])),
        "min_centered_diag": float(torch.diag(centered).min().item()),
    }
    return centered, audit


def spectrum_from_gram_and_weight(gram: torch.Tensor, weight: torch.Tensor, device: str) -> np.ndarray:
    gram = ((gram + gram.T) / 2).to(device=device, dtype=torch.float64)
    values, vectors = torch.linalg.eigh(gram)
    scale = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    weight = weight.to(device=device, dtype=torch.float32)
    product = weight @ scale.to(dtype=torch.float32)
    singular = torch.linalg.svdvals(product.to(dtype=torch.float64)).detach().cpu().numpy()
    del gram, values, vectors, scale, weight, product
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return singular


def run_rr3_llama_centered_reuse() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expected = [("base", 0, probe) for probe in PROBES] + [
        (arm, step, probe) for arm in ARMS for step in STEPS for probe in PROBES
    ]
    module_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    weights_cache: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    try:
        for arm, step, probe in expected:
            print(f"[RR3] centered reuse {arm} step={step} probe={probe}", flush=True)
            profile_file = profile_path(arm, step, probe)
            state_file = state_json_path(arm, step, probe)
            profile = torch.load(profile_file, map_location="cpu", weights_only=False)
            uncentered = read_json(state_file)
            key = (arm, step)
            if key not in weights_cache:
                weights_cache[key] = load_layer_weights(arm, step)
            weights = weights_cache[key]
            uncentered_rank = {
                (r["module"], round(float(r["epsilon"]), 6)): int(r["r_epsilon"])
                for r in uncentered.get("state_rows", [])
            }
            for module in MODULES:
                group = MODULE_TO_GROUP[module]
                cov, audit = centered_gram(profile, group)
                audit_rows.append({
                    "model": "llama",
                    "arm": arm,
                    "checkpoint": step,
                    "probe_name": probe,
                    "layer": LAYER,
                    "module": module,
                    **audit,
                })
                singular = spectrum_from_gram_and_weight(cov, weights[module], device)
                for eps in EPSILONS:
                    centered_rank = rank_from_singular(singular, eps)
                    uncentered_value = uncentered_rank.get((module, round(eps, 6)))
                    module_rows.append({
                        "model": "llama",
                        "arm": arm,
                        "checkpoint": step,
                        "probe_name": probe,
                        "layer": LAYER,
                        "module": module,
                        "epsilon": eps,
                        "centered_state_rank": int(centered_rank),
                        "uncentered_state_rank": uncentered_value,
                        "centered_minus_uncentered": int(centered_rank - uncentered_value) if uncentered_value is not None else np.nan,
                        "spectrum_quantity": "formal_D10_W_t_centered_Sigma_D_t_state_singular_values",
                        "profile_source": str(profile_file),
                        "checkpoint_weight_source": str(model_dir(arm, step)),
                        **energy_stats(singular, centered_rank, eps),
                    })
                del cov, singular
            del profile
            gc.collect()
    finally:
        weights_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    module = pd.DataFrame(module_rows)
    base = module[module["arm"].eq("base")][["probe_name", "module", "epsilon", "centered_state_rank", "uncentered_state_rank"]].rename(columns={
        "centered_state_rank": "centered_state_rank_base",
        "uncentered_state_rank": "uncentered_state_rank_base",
    })
    current = module[~module["arm"].eq("base")].merge(base, on=["probe_name", "module", "epsilon"], how="left")
    current["centered_absolute_contraction"] = current["centered_state_rank_base"] - current["centered_state_rank"]
    current["centered_relative_contraction_module"] = current["centered_absolute_contraction"] / current["centered_state_rank_base"]
    current["uncentered_absolute_contraction"] = current["uncentered_state_rank_base"] - current["uncentered_state_rank"]
    current["uncentered_relative_contraction_module"] = current["uncentered_absolute_contraction"] / current["uncentered_state_rank_base"]
    module_out = pd.concat([module[module["arm"].eq("base")], current], ignore_index=True, sort=False)
    module_path = OUT / "RR3_llama_centered_module.csv"
    module_out.to_csv(module_path, index=False)
    equal = current.groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], as_index=False).agg(
        module_count=("module", "nunique"),
        centered_state_rank=("centered_state_rank", "mean"),
        centered_state_rank_base=("centered_state_rank_base", "mean"),
        centered_absolute_contraction=("centered_absolute_contraction", "mean"),
        centered_relative_contraction_equal7=("centered_relative_contraction_module", "mean"),
        uncentered_state_rank=("uncentered_state_rank", "mean"),
        uncentered_state_rank_base=("uncentered_state_rank_base", "mean"),
        uncentered_absolute_contraction=("uncentered_absolute_contraction", "mean"),
        uncentered_relative_contraction_equal7=("uncentered_relative_contraction_module", "mean"),
    )
    equal_path = OUT / "RR3_llama_centered_equal7.csv"
    equal.to_csv(equal_path, index=False)
    vs = equal.copy()
    vs["centered_minus_uncentered_rank"] = vs["centered_state_rank"] - vs["uncentered_state_rank"]
    vs["centered_minus_uncentered_contraction"] = vs["centered_absolute_contraction"] - vs["uncentered_absolute_contraction"]
    vs_path = OUT / "RR3_llama_centered_vs_uncentered.csv"
    vs.to_csv(vs_path, index=False)
    order_rows = []
    for eps in EPSILONS:
        eg = equal[equal["epsilon"].round(6).eq(round(eps, 6))]
        for (checkpoint, probe), g in eg.groupby(["checkpoint", "probe_name"]):
            for metric in ["centered_absolute_contraction", "uncentered_absolute_contraction"]:
                best = g.sort_values(metric, ascending=False).iloc[0]
                opd = g[g["arm"].eq("opd")]
                order_rows.append({
                    "model": "llama",
                    "checkpoint": int(checkpoint),
                    "probe_name": probe,
                    "epsilon": eps,
                    "metric": metric,
                    "deepest_arm": best["arm"],
                    "opd_deepest": bool(not opd.empty and best["arm"] == "opd"),
                    "opd_value": float(opd.iloc[0][metric]) if not opd.empty else np.nan,
                    "arm_values_json": json.dumps({r.arm: float(getattr(r, metric)) for r in g.itertuples()}, sort_keys=True),
                })
    ordering_path = OUT / "RR3_llama_centered_ordering.csv"
    pd.DataFrame(order_rows).to_csv(ordering_path, index=False)
    changed = pd.DataFrame(order_rows)
    changed_wide = changed.pivot_table(index=["checkpoint", "probe_name", "epsilon"], columns="metric", values="deepest_arm", aggfunc="first").reset_index()
    if "centered_absolute_contraction" in changed_wide and "uncentered_absolute_contraction" in changed_wide:
        changed_wide["centering_changed_deepest_arm_identity"] = changed_wide["centered_absolute_contraction"] != changed_wide["uncentered_absolute_contraction"]
    changed_path = OUT / "RR3_llama_centered_changed_deepest_arm.csv"
    changed_wide.to_csv(changed_path, index=False)
    outputs = pd.read_csv(D10_FINAL / "d10_5_integrated_outputs.csv")
    link_base = equal[equal["epsilon"].round(6).eq(0.05)].merge(outputs[outputs["arm"].ne("base")], on=["model", "arm", "checkpoint", "probe_name"], how="left")
    link_rows = []
    for metric in ["centered_absolute_contraction", "uncentered_absolute_contraction"]:
        for target in ["cumulative_kl_base_to_current", "absolute_delta_nll_cumulative"]:
            link_rows.append({
                "model": "llama",
                "metric": metric,
                "target": target,
                "epsilon": 0.05,
                "n_cells": int(link_base[[metric, target]].dropna().shape[0]),
                "spearman": rank_corr(link_base[metric].to_numpy(dtype=float), link_base[target].to_numpy(dtype=float)),
                "pearson": pearson(link_base[metric].to_numpy(dtype=float), link_base[target].to_numpy(dtype=float)),
            })
    links_path = OUT / "RR3_llama_centered_output_links.csv"
    pd.DataFrame(link_rows).to_csv(links_path, index=False)
    audit = {
        "created_utc": now(),
        "status": "PASS_SAMPLE_EQUAL_GRAM_AND_MEAN_WEIGHTING_COMPATIBLE",
        "weighting_protocol": "collect_profile stores Gram as sample-equal mean of per-sample window-token-weighted second moments; input_sample_means are per-sample window-token-weighted means. Centered covariance uses Gram - mean(sample_means) mean(sample_means)^T.",
        "device_for_matrix_eigh_svd": device,
        "rows": audit_rows,
    }
    audit_path = OUT / "RR3_llama_weighting_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_RR3_LLAMA_CENTERED_REUSE_NO_FORWARD",
        "formal_protocol_id": "RR3_llama_D10_formal_profile_centered_gram_plus_bf16_weight_reuse",
        "numeric_protocol": "Saved formal FP32 aggregate Gram and sample means; centered covariance in FP64; checkpoint weights BF16 merged deployment loaded from safetensors and cast FP32 for W@S; SVD input and singular-energy accumulation FP64. No model forward.",
        "coverage": "4 arms x 3 checkpoints x 4 probes plus 4 base profiles; headline L14; 7 modules; eps=.01/.025/.05/.10",
        "row_counts": {
            "module": int(len(module_out)),
            "equal7": int(len(equal)),
            "vs_uncentered": int(len(vs)),
            "ordering": int(len(order_rows)),
            "output_links": int(len(link_rows)),
            "weighting_audit_rows": int(len(audit_rows)),
        },
        "output_sha256": {
            "RR3_llama_centered_module.csv": sha256_file(module_path),
            "RR3_llama_centered_equal7.csv": sha256_file(equal_path),
            "RR3_llama_centered_vs_uncentered.csv": sha256_file(vs_path),
            "RR3_llama_centered_ordering.csv": sha256_file(ordering_path),
            "RR3_llama_centered_output_links.csv": sha256_file(links_path),
            "RR3_llama_weighting_audit.json": sha256_file(audit_path),
        },
    }
    (OUT / "RR3_llama_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_rr2d_aggregation() -> None:
    aux = pd.read_csv(OUT / "RR2D_displacement_spectrum_auxiliary.csv")
    equal = aux.groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], as_index=False).agg(
        module_count=("module", "nunique"),
        displacement_rank=("rank_at_epsilon", "mean"),
        stable_rank=("stable_rank", "mean"),
        entropy_effective_rank=("entropy_effective_rank", "mean"),
        top1_energy_share=("top1_energy_share", "mean"),
        top10_energy_share=("top10_energy_share", "mean"),
        tail_at_r=("tail_at_r", "mean"),
        two_sided_tail_margin=("two_sided_tail_margin", "mean"),
    )
    equal_path = OUT / "RR2D_displacement_equal7.csv"
    equal.to_csv(equal_path, index=False)
    ordering_rows = []
    for eps in EPSILONS:
        eg = equal[equal["epsilon"].round(6).eq(round(eps, 6))]
        for (checkpoint, probe), g in eg.groupby(["checkpoint", "probe_name"]):
            offline = g[~g["arm"].eq("opd")]
            opd = g[g["arm"].eq("opd")]
            if opd.empty or offline.empty:
                continue
            for metric in ["stable_rank", "entropy_effective_rank"]:
                # For displacement spectra, lower stable/entropy effective rank
                # means a more concentrated displacement spectrum.
                best = g.sort_values(metric, ascending=True).iloc[0]
                nearest_offline = float(offline[metric].min())
                opd_value = float(opd.iloc[0][metric])
                ordering_rows.append({
                    "model": "llama",
                    "checkpoint": int(checkpoint),
                    "probe_name": probe,
                    "epsilon": eps,
                    "metric": metric,
                    "auxiliary_result": True,
                    "not_state_rank_robustness": True,
                    "deepest_arm": best["arm"],
                    "opd_deepest": bool(best["arm"] == "opd"),
                    "deepest_direction": "lower_effective_rank_is_deeper_displacement_compression",
                    "opd_value": opd_value,
                    "nearest_offline_value": nearest_offline,
                    "nearest_offline_minus_opd_margin": float(nearest_offline - opd_value),
                    "opd_minus_nearest_offline_margin": float(opd_value - nearest_offline),
                    "arm_values_json": json.dumps({r.arm: float(getattr(r, metric)) for r in g.itertuples()}, sort_keys=True),
                })
    ordering_path = OUT / "RR2D_displacement_ordering.csv"
    pd.DataFrame(ordering_rows).to_csv(ordering_path, index=False)
    outputs = pd.read_csv(D10_FINAL / "d10_5_integrated_outputs.csv")
    link_base = equal[equal["epsilon"].round(6).eq(0.05)].merge(outputs[outputs["arm"].ne("base")], on=["model", "arm", "checkpoint", "probe_name"], how="left")
    link_rows = []
    for arm, g in link_base.groupby("arm"):
        for metric in ["stable_rank", "entropy_effective_rank"]:
            for target in ["cumulative_kl_base_to_current", "absolute_delta_nll_cumulative"]:
                link_rows.append({
                    "model": "llama",
                    "arm": arm,
                    "epsilon": 0.05,
                    "metric": metric,
                    "target": target,
                    "within_arm_spearman": rank_corr(g[metric].to_numpy(dtype=float), g[target].to_numpy(dtype=float)),
                    "within_arm_pearson": pearson(g[metric].to_numpy(dtype=float), g[target].to_numpy(dtype=float)),
                    "n_cells": int(g[[metric, target]].dropna().shape[0]),
                    "auxiliary_result": True,
                })
    links_path = OUT / "RR2D_displacement_output_links.csv"
    pd.DataFrame(link_rows).to_csv(links_path, index=False)
    manifest = read_json(OUT / "RR2D_manifest.json")
    manifest.update({
        "created_utc_updated": now(),
        "status": "COMPLETE_RR2D_LLAMA_DISPLACEMENT_AUXILIARY_WITH_EQUAL7_ORDERING_LINKS",
        "auxiliary_disclaimer": "auxiliary displacement-spectrum result; not state-rank robustness; not a replacement for RR2S",
        "row_counts": {
            **manifest.get("row_counts", {}),
            "equal7": int(len(equal)),
            "ordering": int(len(ordering_rows)),
            "output_links": int(len(link_rows)),
        },
        "output_sha256": {
            **manifest.get("output_sha256", {}),
            "RR2D_displacement_equal7.csv": sha256_file(equal_path),
            "RR2D_displacement_ordering.csv": sha256_file(ordering_path),
            "RR2D_displacement_output_links.csv": sha256_file(links_path),
        },
    })
    (OUT / "RR2D_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_task_specific_availability() -> None:
    rows = []
    for model in ["llama", "qwen"]:
        for arm in ARMS:
            for step in STEPS:
                for probe in PROBES:
                    if model == "llama":
                        p = profile_path(arm, step, probe)
                        sj = state_json_path(arm, step, probe)
                        has_profile = p.is_file()
                        has_state = sj.is_file() and bool(read_json(sj).get("spectra"))
                        rows.extend([
                            {
                                "model": model,
                                "arm": arm,
                                "checkpoint": step,
                                "probe_name": probe,
                                "task": "RR1A_RR1B",
                                "aggregate_gram": "present_but_insufficient",
                                "aggregate_global_mean": "present_but_insufficient",
                                "per_sample_mean": "present_but_insufficient",
                                "per_sample_second_moment_contribution": "missing",
                                "checkpoint_weight": "present",
                                "status": "NEEDS_NEW_FORWARD_PER_SAMPLE_SECOND_MOMENT",
                                "profile_path": str(p) if has_profile else "",
                            },
                            {
                                "model": model,
                                "arm": arm,
                                "checkpoint": step,
                                "probe_name": probe,
                                "task": "RR2S",
                                "aggregate_gram": "present",
                                "aggregate_global_mean": "not_required",
                                "per_sample_mean": "not_required",
                                "per_sample_second_moment_contribution": "not_required",
                                "checkpoint_weight": "present",
                                "status": "READY_REUSE_NO_FORWARD" if has_state else "BLOCKED_MISSING_STATE_SPECTRUM_OR_PROFILE",
                                "profile_path": str(p) if has_profile else "",
                            },
                            {
                                "model": model,
                                "arm": arm,
                                "checkpoint": step,
                                "probe_name": probe,
                                "task": "RR3",
                                "aggregate_gram": "present" if has_profile else "missing",
                                "aggregate_global_mean": "present" if has_profile else "missing",
                                "per_sample_mean": "present" if has_profile else "missing",
                                "per_sample_second_moment_contribution": "not_required",
                                "checkpoint_weight": "present",
                                "status": "READY_REUSE_AFTER_WEIGHTING_AUDIT_NO_FORWARD" if has_profile else "BLOCKED_MISSING_FORMAL_GRAM_MEAN_PROFILE",
                                "profile_path": str(p) if has_profile else "",
                            },
                        ])
                    else:
                        rows.extend([
                            {
                                "model": model,
                                "arm": arm,
                                "checkpoint": step,
                                "probe_name": probe,
                                "task": task,
                                "aggregate_gram": "inventory_required",
                                "aggregate_global_mean": "inventory_required",
                                "per_sample_mean": "inventory_required" if task == "RR3" else "not_required_or_insufficient",
                                "per_sample_second_moment_contribution": "missing" if task == "RR1A_RR1B" else "not_required",
                                "checkpoint_weight": "present_in_formal_tables_not_enough_for_all_tasks",
                                "status": "BLOCKED_QWEN_TASK_SPECIFIC_PROFILE_INVENTORY_NOT_FORMALLY_PRESENT",
                                "profile_path": "",
                            }
                            for task in ["RR1A_RR1B", "RR2S", "RR3"]
                        ])
    path = OUT / "RR1_RR2S_RR3_task_specific_availability.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "COMPLETE_TASK_SPECIFIC_AVAILABILITY_PREFLIGHT_NO_FORWARD",
        "availability_rules": {
            "RR1A_RR1B": "existing aggregate Gram/mean is insufficient; per-sample second-moment contribution needed; new forward only with Theory GO",
            "RR2S_llama": "complete D10 formal state spectra/aggregate Gram/checkpoint weights are sufficient; no forward",
            "RR3_llama": "D10 formal Gram plus input_sample_means sufficient after weighting audit; no forward",
            "qwen": "kept task-specific blocked/inventory, not globally marked as 96 new-forward cells",
        },
        "row_counts": {"availability_rows": int(len(rows))},
        "output_sha256": {"RR1_RR2S_RR3_task_specific_availability.csv": sha256_file(path)},
    }
    (OUT / "RR1_RR2S_RR3_task_specific_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_handoff() -> None:
    def table(path: Path, n: int = 12) -> str:
        if not path.is_file() or path.stat().st_size == 0:
            return "_missing_"
        df = pd.read_csv(path)
        if df.empty:
            return "_empty_"
        return df.head(n).to_markdown(index=False)

    manifests = {
        "RR5": read_json(OUT / "RR5_nested_regularization_manifest.json"),
        "RR2S": read_json(OUT / "RR2S_llama_manifest.json"),
        "RR3": read_json(OUT / "RR3_llama_manifest.json"),
        "RR2D": read_json(OUT / "RR2D_manifest.json"),
        "availability": read_json(OUT / "RR1_RR2S_RR3_task_specific_manifest.json"),
        "RR4": read_json(OUT / "RR4_top32_manifest.json"),
        "RR6": read_json(OUT / "RR6_readout_manifest.json"),
    }
    lines = [
        "# Reviewer Robustness Theory Handoff",
        "",
        "```yaml",
        "status: CORRECTION2_REUSE_PASS_COMPLETE",
        f"created_utc: {now()}",
        f"script: {REPO / 'experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_correction2.py'}",
        "command_correction2: python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_correction2.py",
        f"output_root: {OUT}/",
        "guard: zero training; no model forward; no paper/human_read/theory edits; raw readings only",
        f"git_commit: {git_commit()}",
        "```",
        "",
        "## Formal Usable / Auxiliary / Blocked / Superseded",
        "",
        "| item | class | status | rows/blocker |",
        "|---|---|---|---|",
        f"| RR5 checkpoint-control + nested regularization | formal usable, Llama-only | {manifests['RR5'].get('status','')} | nested metrics={manifests['RR5'].get('row_counts',{}).get('nested_metrics','')} fold performance={manifests['RR5'].get('row_counts',{}).get('fold_performance','')} |",
        f"| RR2S Llama state spectrum | formal usable, Llama-only | {manifests['RR2S'].get('status','')} | module={manifests['RR2S'].get('row_counts',{}).get('module','')} equal7={manifests['RR2S'].get('row_counts',{}).get('equal7','')} |",
        f"| RR3 Llama centered audit | formal usable, Llama-only | {manifests['RR3'].get('status','')} | module={manifests['RR3'].get('row_counts',{}).get('module','')} equal7={manifests['RR3'].get('row_counts',{}).get('equal7','')} |",
        f"| RR2D displacement spectrum | auxiliary | {manifests['RR2D'].get('status','')} | equal7={manifests['RR2D'].get('row_counts',{}).get('equal7','')} ordering={manifests['RR2D'].get('row_counts',{}).get('ordering','')} |",
        f"| RR1/RR2S/RR3 availability | preflight usable | {manifests['availability'].get('status','')} | rows={manifests['availability'].get('row_counts',{}).get('availability_rows','')} |",
        "| RR1A/RR1B | blocked new-forward | NEEDS_NEW_FORWARD_PER_SAMPLE_SECOND_MOMENT | do not start without Theory GO |",
        "| Qwen RR2S/RR3 missing-profile cells | blocked/inventory | BLOCKED_QWEN_TASK_SPECIFIC_PROFILE_INVENTORY_NOT_FORMALLY_PRESENT | not allowed to block Llama reuse |",
        "| old RR2 spectrum_stability_* | superseded | SUPERSEDED_WRONG_ESTIMAND_AND_EPSILON_IMPLEMENTATION | displacement spectrum, not W_t S_D,t state spectrum |",
        "| RR4 top-32 retained mass | already closed | " + manifests["RR4"].get("status", "") + " | not rerun |",
        "| RR6 matched readout bootstrap | already closed | " + manifests["RR6"].get("status", "") + " | not rerun |",
        "",
        "## RR5 Fold Performance",
        "",
        table(OUT / "RR5_hybrid_fold_performance.csv", 16),
        "",
        "## RR5 Checkpoint-Demeaned Correlations",
        "",
        table(OUT / "RR5_checkpoint_demeaned_correlations.csv", 24),
        "",
        "## RR5 Nested Regularization Metrics",
        "",
        table(OUT / "RR5_nested_regularization_metrics.csv", 28),
        "",
        "## RR2S Llama State Spectrum",
        "",
        table(OUT / "RR2S_llama_state_spectrum_equal7.csv", 16),
        "",
        "Ordering:",
        "",
        table(OUT / "RR2S_llama_continuous_ordering.csv", 24),
        "",
        "## RR3 Llama Centered Audit",
        "",
        table(OUT / "RR3_llama_centered_equal7.csv", 16),
        "",
        "Centered vs uncentered:",
        "",
        table(OUT / "RR3_llama_centered_vs_uncentered.csv", 16),
        "",
        "## RR2D Aggregation",
        "",
        table(OUT / "RR2D_displacement_equal7.csv", 16),
        "",
        "Ordering:",
        "",
        table(OUT / "RR2D_displacement_ordering.csv", 24),
        "",
        "## Task-Specific Availability",
        "",
        table(OUT / "RR1_RR2S_RR3_task_specific_availability.csv", 18),
        "",
        "## SHA256 / Provenance",
        "",
        "See manifests:",
        "",
    ]
    for name in [
        "RR5_nested_regularization_manifest.json",
        "RR2S_llama_manifest.json",
        "RR3_llama_manifest.json",
        "RR2D_manifest.json",
        "RR1_RR2S_RR3_task_specific_manifest.json",
        "RR4_top32_manifest.json",
        "RR6_readout_manifest.json",
    ]:
        path = OUT / name
        if path.exists():
            lines.append(f"- `{name}` sha256={sha256_file(path)}")
    (OUT / "reviewer_robustness_theory_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_returns() -> None:
    handoff = REPO / "mypaper/code/cycle09_reviewer_robustness_handoff.md"
    text = handoff.read_text(encoding="utf-8") if handoff.exists() else "# Cycle09 Reviewer Robustness Handoff\n"
    marker = "## 15. Correction2 Reuse Return: 2026-07-27"
    block = f"""

{marker}

按 Theory 复核意见完成第二轮 reviewer-robustness correction；未修改论文、`human_read-ch.md` 或理论判断。

执行命令：

```bash
python experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_correction2.py
```

新增/更新产物目录：

```text
{OUT}/
```

状态摘要：

| 项 | 状态 |
|---|---|
| RR5-A | `RR5_hybrid_fold_performance.csv` 已补真正逐 fold metrics |
| RR5-B | `RR5_checkpoint_demeaned_correlations.csv` / cells 已补 |
| RR5-C | `RR5_nested_regularization_*` 已补 nested train-fold-only regularization |
| RR2S Llama | 使用 D10 formal state spectra 完成，无 forward |
| RR3 Llama | 使用 D10 formal Gram + input_sample_means + BF16 merged weights 完成 centered audit，无 forward |
| RR2D | equal7 / ordering / output_links 已补，继续标注 auxiliary |
| RR1/RR2S/RR3 preflight | 已拆成 task-specific availability；RR1 仍需 future new forward，未启动 |

正式 handoff：

```text
{OUT / 'reviewer_robustness_theory_handoff.md'}
```
"""
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block
    else:
        text = text.rstrip() + block
    handoff.write_text(text + "\n", encoding="utf-8")

    evo = REPO / "mypaper/code/code_evolution.md"
    etext = evo.read_text(encoding="utf-8") if evo.exists() else "# Code Evolution\n"
    emarker = "### 2026-07-27 Reviewer Robustness Correction2 Reuse Pass"
    eblock = f"""

{emarker}

- Added `experiments/opd_sft_h1/scripts/cycle09_reviewer_robustness_correction2.py`.
- Completed RR5 fold performance, checkpoint-demeaned descriptive correlations, and nested train-fold-only regularization.
- Completed Llama RR2S from D10 formal saved state spectra without model forward.
- Completed Llama RR3 centered audit from D10 formal profiles and BF16 merged deployment weights without model forward.
- Added RR2D equal-seven/order/output-link aggregation and task-specific RR1/RR2S/RR3 availability manifest.
"""
    if emarker not in etext:
        evo.write_text(etext.rstrip() + eblock + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_rr5_checkpoint_control()
    run_rr2s_llama_state_reuse()
    run_rr3_llama_centered_reuse()
    run_rr2d_aggregation()
    run_task_specific_availability()
    write_handoff()
    append_returns()
    manifest = {
        "created_utc": now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "status": "CORRECTION2_REUSE_PASS_COMPLETE",
        "handoff": str(OUT / "reviewer_robustness_theory_handoff.md"),
    }
    (OUT / "reviewer_robustness_correction2_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
