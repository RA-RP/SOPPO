#!/usr/bin/env python3
"""EQUAL5_NON_QK formal reuse task.

This script performs measurement-side non-q/k module exclusion using existing
module-level artifacts only.  It does not train, does not load models, does not
run forward passes, does not recompute SVDs, and does not alter equal-7 sources.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/root/LLM-output-density")
AUTODL = Path("/root/autodl-tmp")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
RR = MINI / "reviewer_robustness"
OUT = MINI / "equal5_non_qk"
SCRATCH = AUTODL / "cycle09_equal5_non_qk"
RFC = AUTODL / "cycle09_relative_functional_contraction"
D10_FINAL = RFC / "d10_llama_numeric_parity/formal/final"
D11_FINAL = RFC / "d11_pk_tpnt/formal/final"
QWEN_FINAL = RFC / "final"

M5 = [
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
EXCLUDED = ["self_attn.q_proj", "self_attn.k_proj"]
M7 = EXCLUDED + M5
EPSILONS = [0.01, 0.025, 0.05, 0.10]
CORE_ARTIFACT_PROBES = ["E_general", "E_math", "E_ood", "E_if"]
CANONICAL_PROBE = {
    "E_general": "E_general",
    "E_math": "E_mathHeld",
    "E_math_hard_v2": "E_AIME24",
    "E_ood": "E_mmluPro",
    "E_if": "E_ifeval",
    "S_math": "legacy_S_math",
}
HEADLINE_LAYER = {"qwen": 18, "llama": 14}
ARMS = ["opd", "sft", "offkd", "seqkd"]
EARLY_STEPS = [20, 40, 80]
TIE_RTOL = 1e-9
TIE_ATOL = 1e-12

SOURCES = {
    "qwen_state_module": QWEN_FINAL / "qwen_d4_merged_state_module_audit.csv",
    "llama_state_module": MINI / "llama_matched_state_module_ranks.csv",
    "llama_rr2s_spectrum_module": RR / "RR2S_llama_state_spectrum_module.csv",
    "llama_rr3_centered_module": RR / "RR3_llama_centered_module.csv",
    "qwen_alpha05_module": MINI / "qwen_alpha05_r_epsilon.csv",
    "llama_frozen_self_module": AUTODL / "cycle09_stage3_followup/H5_frozen_self/geometry/llama_frozen_self_r_epsilon.csv",
    "d5_fairness_update_module": QWEN_FINAL / "d5_fairness_update_module.csv",
    "d5_fairness_update_equal7": QWEN_FINAL / "d5_fairness_update_equal7.csv",
    "d11_llama_merged_pk": D11_FINAL / "d11_llama_merged_pk.csv",
    "d11_tpnt_principal_mask": D11_FINAL / "d11_tpnt_principal_mask.csv",
    "d11_tpnt_angles_pabs_nss": D11_FINAL / "d11_tpnt_angles_pabs_nss.csv",
    "d11_e5_layer_robustness": D11_FINAL / "d11_e5_layer_robustness.csv",
    "d11_e6_alpha_sensitivity": D11_FINAL / "d11_e6_alpha_sensitivity.csv",
    "d11_e7_spectrum_matched_null": D11_FINAL / "d11_e7_spectrum_matched_null.csv",
    "d11_e7_null_seed_rows_llama": D11_FINAL / "d11_e7_spectrum_matched_null_seed_rows_llama.csv",
    "d11_e7_null_seed_rows_qwen": D11_FINAL / "d11_e7_spectrum_matched_null_seed_rows_qwen.csv",
    "d11_same_cell_feature_matrix": D11_FINAL / "d11_same_cell_feature_matrix.csv",
    "rr5_common_grid": RR / "RR5_llama_common_grid.csv",
    "rr5_nested_manifest": RR / "RR5_nested_regularization_manifest.json",
    "rr5_nested_metrics": RR / "RR5_nested_regularization_metrics.csv",
    "rr5_nested_folds": RR / "RR5_nested_regularization_folds.csv",
    "rr5_checkpoint_demeaned_cells": RR / "RR5_checkpoint_demeaned_cells.csv",
    "rr5_checkpoint_demeaned_correlations": RR / "RR5_checkpoint_demeaned_correlations.csv",
    "qwen_outputs": QWEN_FINAL / "qwen_d4_merged_state_outputs.csv",
    "llama_outputs": D10_FINAL / "d10_5_integrated_outputs.csv",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "UNKNOWN"


def sha256_file(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(float)
    br = pd.Series(b).rank(method="average").to_numpy(float)
    if np.std(ar) == 0 or np.std(br) == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
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


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(pred)
    y = y[valid]
    pred = pred[valid]
    if len(y) == 0:
        return {"r2_oof": np.nan, "mae_oof": np.nan, "spearman_oof": np.nan}
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    ss_res = float(np.sum((y - pred) ** 2))
    return {
        "r2_oof": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "mae_oof": float(np.mean(np.abs(y - pred))),
        "spearman_oof": rank_corr(y, pred),
    }


def classification_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y) & np.isfinite(prob)
    y = y[valid].astype(int)
    prob = np.clip(prob[valid], 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"auc_oof": np.nan, "log_loss_oof": np.nan, "balanced_accuracy_oof": np.nan}
    pred = (prob >= 0.5).astype(int)
    pos = y == 1
    neg = y == 0
    tpr = float(np.mean(pred[pos] == 1)) if pos.any() else np.nan
    tnr = float(np.mean(pred[neg] == 0)) if neg.any() else np.nan
    return {
        "auc_oof": auc_score(y, prob),
        "log_loss_oof": float(-np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))),
        "balanced_accuracy_oof": float(np.nanmean([tpr, tnr])),
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


def train_standardize(X: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X[train].mean(axis=0)
    std = X[train].std(axis=0)
    std[std == 0] = 1.0
    return (X[train] - mean) / std, (X[test] - mean) / std


def nested_select_regression(X: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    rows = []
    for alpha in grid:
        losses = []
        for held in sorted(set(groups[outer_train])):
            inner_train = outer_train & (groups != held)
            inner_test = outer_train & (groups == held)
            if inner_test.sum() == 0:
                continue
            Xt, Xv = train_standardize(X, inner_train, inner_test)
            pred = predict_linear(Xv, fit_ridge(Xt, y[inner_train], alpha))
            losses.append(float(np.mean(np.abs(y[inner_test] - pred))))
        rows.append({"regularization": float(alpha), "inner_mean_mae": float(np.mean(losses)) if losses else float("inf")})
    best = min(rows, key=lambda r: (r["inner_mean_mae"], r["regularization"]))
    return float(best["regularization"]), rows


def nested_select_classification(X: np.ndarray, y: np.ndarray, groups: np.ndarray, outer_train: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    rows = []
    for l2 in grid:
        losses = []
        for held in sorted(set(groups[outer_train])):
            inner_train = outer_train & (groups != held)
            inner_test = outer_train & (groups == held)
            if inner_test.sum() == 0 or len(set(y[inner_train].astype(int))) < 2:
                continue
            Xt, Xv = train_standardize(X, inner_train, inner_test)
            prob = np.clip(predict_logistic(Xv, fit_logistic(Xt, y[inner_train], l2)), 1e-6, 1 - 1e-6)
            yy = y[inner_test]
            losses.append(float(-np.mean(yy * np.log(prob) + (1 - yy) * np.log(1 - prob))))
        rows.append({"regularization": float(l2), "inner_mean_log_loss": float(np.mean(losses)) if losses else float("inf")})
    best = min(rows, key=lambda r: (r["inner_mean_log_loss"], r["regularization"]))
    return float(best["regularization"]), rows


def canonical_probe(s: str) -> str:
    return CANONICAL_PROBE.get(str(s), str(s))


def normalize_state_sources() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    qwen = read_csv(SOURCES["qwen_state_module"])
    if not qwen.empty:
        x = qwen.copy()
        x["source_key"] = "qwen_state_module"
        x["artifact_probe_name"] = x["probe_name"]
        x["probe_name"] = x["artifact_probe_name"].map(canonical_probe)
        frames.append(x)

    llama = read_csv(SOURCES["llama_state_module"])
    if not llama.empty:
        x = llama.copy()
        x["source_key"] = "llama_state_module"
        x["artifact_probe_name"] = x["probe_name"]
        x["probe_name"] = x["artifact_probe_name"].map(canonical_probe)
        frames.append(x)

    alpha = read_csv(SOURCES["qwen_alpha05_module"])
    if not alpha.empty:
        x = pd.DataFrame()
        x["model"] = "qwen"
        x["arm"] = "alpha05"
        x["checkpoint"] = alpha["step"]
        x["epsilon"] = alpha["epsilon"]
        x["layer"] = alpha["layer"]
        x["module"] = alpha["module"]
        x["artifact_probe_name"] = alpha["probe"]
        x["probe_name"] = alpha["probe"].map(canonical_probe)
        x["state_rank_base"] = alpha["base_r_epsilon"]
        x["state_rank_current"] = alpha["r_epsilon"]
        x["state_rank_delta"] = alpha["r_epsilon"] - alpha["base_r_epsilon"]
        x["absolute_contraction"] = alpha["base_r_epsilon"] - alpha["r_epsilon"]
        x["relative_functional_contraction_module"] = np.where(alpha["base_r_epsilon"] != 0, (alpha["base_r_epsilon"] - alpha["r_epsilon"]) / alpha["base_r_epsilon"], np.nan)
        x["source_name"] = "qwen_alpha05_r_epsilon"
        x["source_protocol"] = "qwen_alpha05_existing_module_r_epsilon_reuse"
        x["source_key"] = "qwen_alpha05_module"
        frames.append(x)

    frozen = read_csv(SOURCES["llama_frozen_self_module"])
    if not frozen.empty:
        h = frozen[(frozen["layer"] == HEADLINE_LAYER["llama"]) & (frozen["arm"] == "frozen_self")].copy()
        x = pd.DataFrame()
        x["model"] = "llama"
        x["arm"] = "frozenSelf"
        x["checkpoint"] = h["step"]
        x["epsilon"] = h["epsilon"]
        x["layer"] = h["layer"]
        x["module"] = h["module"]
        x["artifact_probe_name"] = h["probe"]
        x["probe_name"] = h["probe"].map(canonical_probe)
        x["state_rank_base"] = h["base_r_epsilon"]
        x["state_rank_current"] = h["r_epsilon"]
        x["state_rank_delta"] = h["r_epsilon"] - h["base_r_epsilon"]
        x["absolute_contraction"] = h["base_r_epsilon"] - h["r_epsilon"]
        x["relative_functional_contraction_module"] = np.where(h["base_r_epsilon"] != 0, (h["base_r_epsilon"] - h["r_epsilon"]) / h["base_r_epsilon"], np.nan)
        x["source_name"] = "llama_frozen_self_r_epsilon"
        x["source_protocol"] = "H5_frozen_self_existing_module_r_epsilon_reuse"
        x["source_key"] = "llama_frozen_self_module"
        frames.append(x)

    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not out.empty:
        out["checkpoint"] = out["checkpoint"].astype(int)
        out["epsilon"] = out["epsilon"].astype(float)
        out["state_rank_delta"] = out["state_rank_current"].astype(float) - out["state_rank_base"].astype(float)
        out["absolute_contraction"] = out["state_rank_base"].astype(float) - out["state_rank_current"].astype(float)
        out["relative_functional_contraction_module"] = np.where(
            out["state_rank_base"].astype(float) != 0,
            out["absolute_contraction"].astype(float) / out["state_rank_base"].astype(float),
            np.nan,
        )
        out["source_path"] = out["source_key"].map(lambda k: str(SOURCES.get(k, "")))
    return out


def aggregate_functional(module_df: pd.DataFrame, modules: list[str], suffix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["model", "arm", "checkpoint", "artifact_probe_name", "probe_name", "epsilon", "layer"]
    for key, g in module_df[module_df["module"].isin(modules)].groupby(keys, dropna=False):
        mods = sorted(g["module"].unique().tolist())
        complete = sorted(mods) == sorted(modules)
        row = dict(zip(keys, key))
        row["module_count"] = len(mods)
        row["module_set"] = suffix
        row["source_rows_complete"] = bool(complete)
        row["included_modules"] = ",".join(mods)
        row["excluded_modules"] = ",".join(EXCLUDED if suffix == "equal5_non_qk" else [])
        if complete:
            base = g.set_index("module").loc[modules, "state_rank_base"].astype(float)
            cur = g.set_index("module").loc[modules, "state_rank_current"].astype(float)
            delta = cur - base
            cmod = np.where(base.to_numpy() != 0, (base.to_numpy() - cur.to_numpy()) / base.to_numpy(), np.nan)
            row.update({
                f"r_epsilon_{suffix}": float(cur.mean()),
                f"state_rank_base_{suffix}": float(base.mean()),
                f"state_rank_current_{suffix}": float(cur.mean()),
                f"delta_r_{suffix}": float(delta.mean()),
                f"c_{suffix}": float(np.nanmean(cmod)),
                f"ratio_of_means_sensitivity_{suffix}": float((base.mean() - cur.mean()) / base.mean()) if base.mean() != 0 else np.nan,
                f"base_rank_modules_json_{suffix}": json.dumps({m: float(base.loc[m]) for m in modules}, sort_keys=True),
                f"current_rank_modules_json_{suffix}": json.dumps({m: float(cur.loc[m]) for m in modules}, sort_keys=True),
                f"delta_r_modules_json_{suffix}": json.dumps({m: float(delta.loc[m]) for m in modules}, sort_keys=True),
                f"c_modules_json_{suffix}": json.dumps({m: float(cmod[i]) for i, m in enumerate(modules)}, sort_keys=True),
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and suffix == "equal5_non_qk":
        gcols = ["model", "arm", "checkpoint", "epsilon", "layer"]
        gen = out[out["probe_name"] == "E_general"][gcols + ["c_equal5_non_qk"]].rename(columns={"c_equal5_non_qk": "c_equal5_E_general"})
        out = out.merge(gen, on=gcols, how="left")
        out["G_equal5_non_qk"] = out["c_equal5_non_qk"] - out["c_equal5_E_general"]
    return out


def make_functional_outputs(module_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    equal5 = aggregate_functional(module_df, M5, "equal5_non_qk")
    equal7 = aggregate_functional(module_df, M7, "equal7")
    key = ["model", "arm", "checkpoint", "artifact_probe_name", "probe_name", "epsilon", "layer"]
    paired = equal5.merge(equal7, on=key, how="left", suffixes=("", "_eq7src"))
    paired["equal5_minus_equal7_c"] = paired["c_equal5_non_qk"] - paired["c_equal7"]
    paired["equal5_minus_equal7_delta_r"] = paired["delta_r_equal5_non_qk"] - paired["delta_r_equal7"]
    paired["equal5_minus_equal7_r"] = paired["r_epsilon_equal5_non_qk"] - paired["r_epsilon_equal7"]
    paired["sign_changed_c"] = np.sign(paired["c_equal5_non_qk"]) != np.sign(paired["c_equal7"])
    paired["excluded_modules_exactly_qk"] = True
    traj = paired.rename(columns={
        "r_epsilon_equal5_non_qk": "r_epsilon_equal5",
        "delta_r_equal5_non_qk": "delta_r_equal5",
        "c_equal5_non_qk": "c_equal5",
        "ratio_of_means_sensitivity_equal5_non_qk": "ratio_of_means_sensitivity_equal5",
        "G_equal5_non_qk": "G_equal5",
    }).copy()
    return equal5, traj, paired


def make_coverage_inventory(module_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for src, path in SOURCES.items():
        if path.suffix != ".csv":
            continue
        exists = path.is_file()
        try:
            row_count = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1 if exists else 0
        except Exception:
            row_count = 0
        rows.append({
            "inventory_type": "source_file",
            "status": "READY_REUSE_MODULE_ROWS" if exists and src in {
                "qwen_state_module", "llama_state_module", "llama_rr2s_spectrum_module", "llama_rr3_centered_module",
                "qwen_alpha05_module", "llama_frozen_self_module", "d5_fairness_update_module", "d11_llama_merged_pk",
                "d11_tpnt_principal_mask", "d11_tpnt_angles_pabs_nss", "d11_e5_layer_robustness",
                "d11_e6_alpha_sensitivity", "d11_e7_spectrum_matched_null",
            } else ("NOT_APPLICABLE_NO_MODULE_AXIS" if exists else "BLOCKED_NO_MODULE_SOURCE"),
            "source_key": src,
            "source_path": str(path),
            "row_count": row_count,
            "sha256": sha256_file(path) if exists else "MISSING",
        })

    if not module_df.empty:
        for key, g in module_df.groupby(["model", "arm", "checkpoint", "artifact_probe_name", "probe_name", "epsilon", "layer", "module"], dropna=False):
            row = dict(zip(["model", "arm", "checkpoint", "artifact_probe_name", "probe_name", "epsilon", "layer", "module"], key))
            row.update({
                "inventory_type": "functional_module_cell",
                "status": "READY_REUSE_MODULE_ROWS",
                "source_key": ",".join(sorted(g["source_key"].dropna().unique())),
                "source_path": ",".join(sorted(g["source_path"].dropna().unique())),
                "row_count": int(len(g)),
            })
            rows.append(row)
    rows.extend([
        {
            "inventory_type": "blocked_task_family",
            "status": "SUPERSEDED_SOURCE_FORBIDDEN",
            "task": "RR2D_displacement_spectrum",
            "reason": "RR2 displacement spectrum is superseded and forbidden for state-rank robustness; keep auxiliary only.",
        },
        {
            "inventory_type": "blocked_task_family",
            "status": "BLOCKED_NO_MODULE_SOURCE",
            "task": "Qwen_centered_state_spectrum_equal5",
            "reason": "No formal Qwen centered module spectrum source; no new forward permitted.",
        },
        {
            "inventory_type": "blocked_task_family",
            "status": "BLOCKED_NO_MODULE_SOURCE",
            "task": "Qwen_full_state_singular_spectrum_equal5",
            "reason": "Qwen state-rank module rows exist, but full singular spectrum/stable-rank/entropy source is not present in the approved input list.",
        },
        {
            "inventory_type": "blocked_task_family",
            "status": "BLOCKED_NO_MODULE_SOURCE",
            "task": "Qwen_p_k_equal5",
            "reason": "No Qwen formal per-module p_k table found; cannot infer equal5 from equal7 feature matrix.",
        },
    ])
    return pd.DataFrame(rows)


def dominance(functional: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    src = functional[
        functional["arm"].isin(ARMS)
        & functional["checkpoint"].isin(EARLY_STEPS)
        & functional["probe_name"].isin([canonical_probe(p) for p in CORE_ARTIFACT_PROBES])
    ].copy()
    rows: list[dict[str, Any]] = []
    for key, g in src.groupby(["model", "checkpoint", "probe_name", "epsilon", "layer"], dropna=False):
        vals = dict(zip(g["arm"], g["delta_r_equal5"]))
        if "opd" not in vals or any(a not in vals for a in ["sft", "offkd", "seqkd"]):
            continue
        opd = vals["opd"]
        offline = {a: vals[a] for a in ["sft", "offkd", "seqkd"]}
        min_off = min(offline.values())
        strict = all((v - opd) > TIE_ATOL and not np.isclose(v, opd, rtol=TIE_RTOL, atol=TIE_ATOL) for v in offline.values())
        tied = any(np.isclose(v, opd, rtol=TIE_RTOL, atol=TIE_ATOL) for v in offline.values())
        offline_deeper = any((opd - v) > TIE_ATOL and not np.isclose(v, opd, rtol=TIE_RTOL, atol=TIE_ATOL) for v in offline.values())
        rows.append({
            "model": key[0],
            "checkpoint": int(key[1]),
            "probe_name": key[2],
            "epsilon": float(key[3]),
            "layer": int(key[4]),
            "opd_delta_r_equal5": float(opd),
            "nearest_offline_delta_r_equal5": float(min_off),
            "continuous_margin": float(min_off - opd),
            "nearest_offline_arm": min(offline, key=offline.get),
            "OPD_strict_deepest": bool(strict),
            "OPD_tied_deepest": bool(tied),
            "offline_strictly_deeper": bool(offline_deeper),
        })
    cells = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    if not cells.empty:
        for key, g in cells.groupby(["model", "epsilon"], dropna=False):
            summary_rows.append({
                "scope": "by_model_epsilon",
                "model": key[0],
                "epsilon": float(key[1]),
                "n_cells": int(len(g)),
                "OPD_strict_deepest": int(g["OPD_strict_deepest"].sum()),
                "OPD_tied_deepest": int(g["OPD_tied_deepest"].sum()),
                "offline_strictly_deeper": int(g["offline_strictly_deeper"].sum()),
                "mean_continuous_margin": float(g["continuous_margin"].mean()),
                "min_continuous_margin": float(g["continuous_margin"].min()),
            })
        for eps, g in cells.groupby("epsilon"):
            summary_rows.append({
                "scope": "pooled_models_epsilon",
                "model": "pooled_qwen_llama",
                "epsilon": float(eps),
                "n_cells": int(len(g)),
                "OPD_strict_deepest": int(g["OPD_strict_deepest"].sum()),
                "OPD_tied_deepest": int(g["OPD_tied_deepest"].sum()),
                "offline_strictly_deeper": int(g["offline_strictly_deeper"].sum()),
                "mean_continuous_margin": float(g["continuous_margin"].mean()),
                "min_continuous_margin": float(g["continuous_margin"].min()),
            })
    return cells, pd.DataFrame(summary_rows)


def ncd(functional: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, value_col in [("equal5_non_qk", "delta_r_equal5"), ("equal7", "delta_r_equal7")]:
        src = paired if label == "equal7" else functional
        col = value_col
        for key, g in src[src["arm"].isin(ARMS)].groupby(["model", "arm", "probe_name", "epsilon", "layer"], dropna=False):
            h = g.sort_values("checkpoint")
            if h.empty:
                continue
            t = h["checkpoint"].to_numpy(float)
            y = np.maximum(-h[col].to_numpy(float), 0.0)
            tau = np.log1p(t)
            order = np.argsort(tau)
            area = float(np.trapz(y[order], tau[order])) if len(tau) >= 2 else 0.0
            rows.append({
                "aggregation": label,
                "model": key[0],
                "arm": key[1],
                "probe_name": key[2],
                "epsilon": float(key[3]),
                "layer": int(key[4]),
                "n_points": int(len(h)),
                "ncd_probe": area,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = df.groupby(["aggregation", "model", "arm", "epsilon", "layer"], dropna=False).agg(
        n_probes=("probe_name", "nunique"),
        ncd=("ncd_probe", "mean"),
    ).reset_index()
    wide = agg.pivot_table(index=["model", "arm", "epsilon", "layer"], columns="aggregation", values="ncd").reset_index()
    wide.columns.name = None
    if "equal5_non_qk" in wide and "equal7" in wide:
        wide["equal5_minus_equal7_ncd"] = wide["equal5_non_qk"] - wide["equal7"]
    return wide


def spectrum_robustness() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rr2s = read_csv(SOURCES["llama_rr2s_spectrum_module"])
    if not rr2s.empty:
        for key, g in rr2s[rr2s["module"].isin(M5)].groupby(["model", "arm", "checkpoint", "probe_name", "layer"], dropna=False):
            if len(set(g["module"])) != 5:
                continue
            first = g.drop_duplicates(["module"])
            rows.append({
                "analysis": "RR2S_uncentered_continuous_spectrum",
                "model": key[0],
                "arm": key[1],
                "checkpoint": int(key[2]),
                "probe_name": canonical_probe(key[3]),
                "artifact_probe_name": key[3],
                "layer": int(key[4]),
                "module_count": 5,
                "stable_rank_contraction_equal5": float(first["stable_rank_contraction"].mean()) if "stable_rank_contraction" in first else np.nan,
                "entropy_effective_rank_contraction_equal5": float(first["entropy_effective_rank_contraction"].mean()) if "entropy_effective_rank_contraction" in first else np.nan,
                "source_status": "READY_REUSE_MODULE_ROWS",
            })
        eps = rr2s[rr2s["module"].isin(M5)].groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], dropna=False).agg(
            uncentered_r_epsilon_contraction_equal5=("relative_functional_contraction_module", "mean"),
            module_count=("module", "nunique"),
        ).reset_index()
        for r in eps.itertuples():
            if int(r.module_count) == 5:
                rows.append({
                    "analysis": "RR2S_uncentered_r_epsilon_epsilon_expanded_audit",
                    "model": r.model,
                    "arm": r.arm,
                    "checkpoint": int(r.checkpoint),
                    "probe_name": canonical_probe(r.probe_name),
                    "artifact_probe_name": r.probe_name,
                    "layer": int(r.layer),
                    "epsilon": float(r.epsilon),
                    "module_count": 5,
                    "uncentered_r_epsilon_contraction_equal5": float(r.uncentered_r_epsilon_contraction_equal5),
                    "source_status": "READY_REUSE_MODULE_ROWS",
                })
    rr3 = read_csv(SOURCES["llama_rr3_centered_module"])
    if not rr3.empty:
        eps = rr3[rr3["module"].isin(M5)].groupby(["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], dropna=False).agg(
            centered_r_epsilon_contraction_equal5=("centered_relative_contraction_module", "mean"),
            uncentered_r_epsilon_contraction_equal5=("uncentered_relative_contraction_module", "mean"),
            module_count=("module", "nunique"),
        ).reset_index()
        eps["centered_minus_uncentered_equal5"] = eps["centered_r_epsilon_contraction_equal5"] - eps["uncentered_r_epsilon_contraction_equal5"]
        for r in eps.itertuples():
            if int(r.module_count) == 5:
                rows.append({
                    "analysis": "RR3_centered_vs_uncentered_epsilon_expanded_audit",
                    "model": r.model,
                    "arm": r.arm,
                    "checkpoint": int(r.checkpoint),
                    "probe_name": canonical_probe(r.probe_name),
                    "artifact_probe_name": r.probe_name,
                    "layer": int(r.layer),
                    "epsilon": float(r.epsilon),
                    "module_count": 5,
                    "centered_r_epsilon_contraction_equal5": float(r.centered_r_epsilon_contraction_equal5),
                    "uncentered_r_epsilon_contraction_equal5": float(r.uncentered_r_epsilon_contraction_equal5),
                    "centered_minus_uncentered_equal5": float(r.centered_minus_uncentered_equal5),
                    "source_status": "READY_REUSE_MODULE_ROWS",
                })
        early = eps[
            eps["arm"].isin(ARMS)
            & eps["checkpoint"].isin(EARLY_STEPS)
            & eps["probe_name"].isin(CORE_ARTIFACT_PROBES)
        ].copy()
        for key, g in early.groupby(["model", "checkpoint", "probe_name", "layer", "epsilon"], dropna=False):
            vals = dict(zip(g["arm"], g["centered_r_epsilon_contraction_equal5"]))
            if "opd" not in vals or any(a not in vals for a in ["sft", "offkd", "seqkd"]):
                continue
            # Larger contraction means deeper compression for centered c_epsilon.
            opd = vals["opd"]
            offline = {a: vals[a] for a in ["sft", "offkd", "seqkd"]}
            max_off = max(offline.values())
            rows.append({
                "analysis": "RR3_centered_equal5_OPD_dominance",
                "model": key[0],
                "arm": "opd",
                "checkpoint": int(key[1]),
                "probe_name": canonical_probe(key[2]),
                "artifact_probe_name": key[2],
                "layer": int(key[3]),
                "epsilon": float(key[4]),
                "module_count": 5,
                "centered_opd_contraction_equal5": float(opd),
                "centered_nearest_offline_contraction_equal5": float(max_off),
                "centered_opd_minus_nearest_offline_margin": float(opd - max_off),
                "OPD_strict_deepest": bool(opd - max_off > TIE_ATOL and not np.isclose(opd, max_off, rtol=TIE_RTOL, atol=TIE_ATOL)),
                "nearest_offline_arm": max(offline, key=offline.get),
                "source_status": "READY_REUSE_MODULE_ROWS",
            })
    rows.append({
        "analysis": "Qwen_centered_and_full_spectrum",
        "model": "qwen",
        "source_status": "BLOCKED_NO_MODULE_SOURCE",
        "reason": "No approved formal Qwen centered/full singular-spectrum module source; no new forward/SVD permitted.",
    })
    return pd.DataFrame(rows)


def generic_module_aggregate(path_key: str, value_cols: list[str], extra_keys: list[str], metric_family: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = read_csv(SOURCES[path_key])
    if df.empty or "module" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df = df[df["module"].isin(M5)].copy()
    key_cols = [c for c in ["model", "family", "arm", "checkpoint", "step", "layer"] + extra_keys if c in df.columns]
    mod = df.copy()
    mod["metric_family"] = metric_family
    mod["source_key"] = path_key
    mod["aggregation"] = "module_rows_non_qk_source"
    present_vals = [c for c in value_cols if c in mod.columns]
    agg_spec = {c: (c, "mean") for c in present_vals}
    agg = df.groupby(key_cols, dropna=False).agg(**agg_spec, module_count=("module", "nunique")).reset_index()
    agg = agg[agg["module_count"] == 5].copy()
    agg["metric_family"] = metric_family
    agg["source_key"] = path_key
    agg["aggregation"] = "equal5_non_qk"
    agg["included_modules"] = ",".join(M5)
    agg["excluded_modules"] = ",".join(EXCLUDED)
    return mod, agg


def weight_baselines() -> tuple[pd.DataFrame, pd.DataFrame]:
    module_frames: list[pd.DataFrame] = []
    agg_frames: list[pd.DataFrame] = []

    specs = [
        ("d5_fairness_update_module", ["raw_weight_energy", "whitened_update_energy_current", "whitened_update_energy_fixed", "activation_exposure_ratio"], ["probe_name", "epsilon"], "update_energy_activation_exposure"),
        ("d11_llama_merged_pk", ["p_k"], ["k", "rank_spec_kind"], "p_k_fixed"),
        ("d11_tpnt_principal_mask", ["coverage", "overlap_lift", "overlap_lift_minus_random_null", "update_density"], ["source_rank_k", "mask_density_alpha", "random_null_rank"], "tpnt_principal_mask"),
        ("d11_tpnt_angles_pabs_nss", ["theta_u_mean_deg", "theta_u_max_deg", "theta_v_mean_deg", "theta_v_max_deg", "pabs_mean_cos_u", "pabs_mean_cos_v", "pabs_joint_mean_cos", "nss_l1_top32", "nss_l2_top32"], ["angle_k"], "pabs_angles_nss"),
        ("d11_e5_layer_robustness", ["coverage", "overlap_lift", "pabs_joint_mean_cos_k32", "theta_u_mean_deg_k32", "theta_v_mean_deg_k32", "nss_l1_top32", "nss_l2_top32"], ["source_rank_k", "mask_density_alpha", "headline_layer"], "layer_robustness"),
        ("d11_e6_alpha_sensitivity", ["coverage", "overlap_lift", "overlap_lift_minus_random_null", "update_density"], ["source_rank_k", "mask_density_alpha", "random_null_rank"], "alpha_mask_density_sensitivity"),
        ("d11_e7_spectrum_matched_null", ["real_overlap_lift", "null_overlap_lift_mean", "null_overlap_lift_std", "z_tpnt"], ["source_rank_k", "mask_density_alpha"], "tpnt_spectrum_matched_null"),
    ]
    for spec in specs:
        mod, agg = generic_module_aggregate(*spec)
        if not mod.empty:
            module_frames.append(mod)
        if not agg.empty:
            agg_frames.append(agg)

    blocked = pd.DataFrame([{
        "metric_family": "p_k_fixed",
        "model": "qwen",
        "aggregation": "equal5_non_qk",
        "source_key": "missing_qwen_p_k_module",
        "source_status": "BLOCKED_NO_MODULE_SOURCE",
        "reason": "No approved Qwen formal per-module p_k table was found; equal7 p_k cannot be reverse-aggregated.",
    }])
    agg_frames.append(blocked)
    return (
        pd.concat(module_frames, ignore_index=True, sort=False) if module_frames else pd.DataFrame(),
        pd.concat(agg_frames, ignore_index=True, sort=False) if agg_frames else pd.DataFrame(),
    )


def join_outputs(functional: pd.DataFrame) -> pd.DataFrame:
    q = read_csv(SOURCES["qwen_outputs"])
    l = read_csv(SOURCES["llama_outputs"])
    out = pd.concat([q, l], ignore_index=True, sort=False)
    if out.empty:
        return functional
    out["probe_name"] = out["probe_name"].map(canonical_probe)
    return functional.merge(
        out[["model", "arm", "checkpoint", "probe_name", "cumulative_kl_base_to_current", "absolute_delta_nll_cumulative", "delta_nll_cumulative", "nll_base", "nll_current", "sample_count"]],
        on=["model", "arm", "checkpoint", "probe_name"],
        how="left",
        suffixes=("", "_output"),
    )


def output_link_correlations(functional: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = ["cumulative_kl_base_to_current", "absolute_delta_nll_cumulative", "delta_nll_cumulative"]
    joined = join_outputs(functional)
    paired_joined = join_outputs(paired.rename(columns={"c_equal5_non_qk": "c_equal5"}))
    rows: list[dict[str, Any]] = []

    def add_corr(src: pd.DataFrame, value_col: str, aggregation: str, assoc: str) -> None:
        for key, g in src[src["arm"].isin(ARMS)].groupby(["model", "arm", "epsilon"], dropna=False):
            for target in targets:
                rows.append({
                    "association_type": assoc,
                    "aggregation": aggregation,
                    "model": key[0],
                    "arm": key[1],
                    "epsilon": float(key[2]),
                    "feature": value_col,
                    "target": target,
                    "n_cells": int(g[[value_col, target]].dropna().shape[0]),
                    "pearson": pearson(g[value_col].to_numpy(float), g[target].to_numpy(float)),
                    "spearman": rank_corr(g[value_col].to_numpy(float), g[target].to_numpy(float)),
                })

    add_corr(joined, "c_equal5", "equal5_non_qk", "within_model_arm")
    if "c_equal7" in paired_joined.columns:
        add_corr(paired_joined, "c_equal7", "equal7", "within_model_arm")

    demean = joined.copy()
    for col in ["c_equal5"] + targets:
        demean[f"demeaned_{col}"] = demean[col] - demean.groupby(["model", "checkpoint", "epsilon"])[col].transform("mean")
    demean_rows: list[dict[str, Any]] = []
    for key, g in demean[demean["arm"].isin(ARMS)].groupby(["model", "epsilon"], dropna=False):
        for target in targets:
            x = g["demeaned_c_equal5"].to_numpy(float)
            y = g[f"demeaned_{target}"].to_numpy(float)
            demean_rows.append({
                "association_type": "checkpoint_demeaned_descriptive_not_oof_prediction",
                "aggregation": "equal5_non_qk",
                "model": key[0],
                "epsilon": float(key[1]),
                "feature": "c_equal5",
                "target": target,
                "n_cells": int(np.isfinite(x + y).sum()),
                "pearson": pearson(x, y),
                "spearman": rank_corr(x, y),
            })

    for key, g in joined[joined["arm"].isin(ARMS)].groupby(["model", "arm", "epsilon"], dropna=False):
        tau = np.log1p(g["checkpoint"].to_numpy(float))
        X = np.column_stack([np.ones(len(tau)), tau])
        for target in targets:
            x = g["c_equal5"].to_numpy(float)
            y = g[target].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(tau)
            if valid.sum() >= 4:
                bx = np.linalg.pinv(X[valid]) @ x[valid]
                by = np.linalg.pinv(X[valid]) @ y[valid]
                xr = x[valid] - X[valid] @ bx
                yr = y[valid] - X[valid] @ by
                rows.append({
                    "association_type": "logstep_progress_residual",
                    "aggregation": "equal5_non_qk",
                    "model": key[0],
                    "arm": key[1],
                    "epsilon": float(key[2]),
                    "feature": "c_equal5_residual_logstep",
                    "target": target,
                    "n_cells": int(valid.sum()),
                    "pearson": pearson(xr, yr),
                    "spearman": rank_corr(xr, yr),
                })

    # Equal5 minus equal7 correlation deltas where the same grouping exists.
    corr = pd.DataFrame(rows)
    if not corr.empty:
        eq5 = corr[(corr["association_type"] == "within_model_arm") & (corr["aggregation"] == "equal5_non_qk")]
        eq7 = corr[(corr["association_type"] == "within_model_arm") & (corr["aggregation"] == "equal7")]
        delta = eq5.merge(eq7, on=["model", "arm", "epsilon", "target"], suffixes=("_equal5", "_equal7"))
        for r in delta.itertuples():
            rows.append({
                "association_type": "equal5_minus_equal7_correlation_delta",
                "aggregation": "paired_delta",
                "model": r.model,
                "arm": r.arm,
                "epsilon": float(r.epsilon),
                "feature": "c_equal5_minus_c_equal7",
                "target": r.target,
                "n_cells": int(min(r.n_cells_equal5, r.n_cells_equal7)),
                "pearson": float(r.pearson_equal5 - r.pearson_equal7),
                "spearman": float(r.spearman_equal5 - r.spearman_equal7),
            })

    return pd.DataFrame(rows), pd.DataFrame(demean_rows)


def build_equal5_feature_matrix(functional: pd.DataFrame, weight_agg: pd.DataFrame) -> pd.DataFrame:
    base = join_outputs(functional)
    base = base[(base["arm"].isin(ARMS)) & (base["epsilon"].round(6) == 0.05)].copy()
    # Keep D11 same-cell support checkpoints/probes where outputs exist.
    base = base[base["probe_name"].isin([canonical_probe(p) for p in CORE_ARTIFACT_PROBES])]
    out = base[[
        "model", "arm", "checkpoint", "epsilon", "layer", "artifact_probe_name", "probe_name",
        "c_equal5", "delta_r_equal5", "r_epsilon_equal5", "ratio_of_means_sensitivity_equal5",
        "cumulative_kl_base_to_current", "absolute_delta_nll_cumulative", "delta_nll_cumulative",
    ]].copy()

    # Update-energy features are probe/epsilon dependent.
    upd = weight_agg[weight_agg.get("metric_family", "") == "update_energy_activation_exposure"].copy()
    if not upd.empty:
        upd["probe_name"] = upd["probe_name"].map(canonical_probe)
        upd = upd.rename(columns={
            "raw_weight_energy": "raw_update_energy_equal5",
            "whitened_update_energy_current": "whitened_update_energy_equal5",
            "whitened_update_energy_fixed": "whitened_update_energy_fixed_equal5",
            "activation_exposure_ratio": "activation_exposure_ratio_equal5",
        })
        out = out.merge(
            upd[["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "raw_update_energy_equal5", "whitened_update_energy_equal5", "whitened_update_energy_fixed_equal5", "activation_exposure_ratio_equal5"]],
            on=["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"],
            how="left",
        )

    pk = weight_agg[(weight_agg.get("metric_family", "") == "p_k_fixed") & (weight_agg.get("model", "") == "llama")].copy()
    if not pk.empty:
        pkw = pk.pivot_table(index=["model", "arm", "checkpoint", "layer"], columns="k", values="p_k").reset_index()
        pkw.columns = [f"p_k{int(c)}_equal5" if isinstance(c, (int, float)) else c for c in pkw.columns]
        out = out.merge(pkw, on=["model", "arm", "checkpoint", "layer"], how="left")

    tpnt = weight_agg[(weight_agg.get("metric_family", "") == "tpnt_principal_mask") & (weight_agg.get("source_rank_k", np.nan) == 16) & (weight_agg.get("mask_density_alpha", np.nan) == 0.01)].copy()
    if not tpnt.empty:
        tpnt = tpnt.rename(columns={"overlap_lift": "tpnt_overlap_lift_equal5", "overlap_lift_minus_random_null": "tpnt_lift_minus_null_equal5", "coverage": "tpnt_coverage_equal5"})
        out = out.merge(tpnt[["model", "arm", "checkpoint", "layer", "tpnt_overlap_lift_equal5", "tpnt_lift_minus_null_equal5", "tpnt_coverage_equal5"]], on=["model", "arm", "checkpoint", "layer"], how="left")

    pabs = weight_agg[(weight_agg.get("metric_family", "") == "pabs_angles_nss") & (weight_agg.get("angle_k", np.nan) == 32)].copy()
    if not pabs.empty:
        pabs = pabs.rename(columns={"pabs_joint_mean_cos": "pabs_joint_mean_cos_equal5", "nss_l1_top32": "nss_l1_top32_equal5", "nss_l2_top32": "nss_l2_top32_equal5"})
        out = out.merge(pabs[["model", "arm", "checkpoint", "layer", "pabs_joint_mean_cos_equal5", "nss_l1_top32_equal5", "nss_l2_top32_equal5"]], on=["model", "arm", "checkpoint", "layer"], how="left")
    out["is_opd"] = (out["arm"] == "opd").astype(int)
    return out


def nested_equal5(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rr5 = read_csv(SOURCES["rr5_common_grid"])
    old_metrics = read_csv(SOURCES["rr5_nested_metrics"])
    if rr5.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rr5 = rr5[["arm", "checkpoint", "probe_name", "row_id"] + [
        c for c in [
            "normalized_entropy_effective_rank", "participation_ratio", "top1_explained_share",
            "top8_explained_share", "top32_explained_share", "raw_anisotropy",
            "centered_anisotropy", "linear_cka_vs_step0",
            "cumulative_kl_base_to_current", "absolute_delta_nll_cumulative", "delta_nll_cumulative", "is_opd",
        ] if c in rr5.columns
    ]]
    rr5["probe_name_canonical"] = rr5["probe_name"].map(canonical_probe)
    fm = feature_matrix[(feature_matrix["model"] == "llama") & (feature_matrix["epsilon"].round(6) == 0.05)].copy()
    common = rr5.merge(
        fm[["arm", "checkpoint", "probe_name", "c_equal5", "p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5"]],
        left_on=["arm", "checkpoint", "probe_name_canonical"],
        right_on=["arm", "checkpoint", "probe_name"],
        how="inner",
        suffixes=("", "_equal5src"),
    )

    feature_cols_a = [c for c in [
        "normalized_entropy_effective_rank", "participation_ratio", "top1_explained_share",
        "top8_explained_share", "top32_explained_share", "raw_anisotropy",
        "centered_anisotropy", "linear_cka_vs_step0",
    ] if c in common.columns]
    blocks = {
        "A": feature_cols_a,
        "C5": ["c_equal5"],
        "Pk5": ["p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5"],
        "A+C5": feature_cols_a + ["c_equal5"],
        "Pk5+A": ["p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5"] + feature_cols_a,
        "Pk5+C5": ["p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5", "c_equal5"],
        "Pk5+A+C5": ["p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5"] + feature_cols_a + ["c_equal5"],
    }
    targets = {
        "cumulative_kl_base_to_current": "regression",
        "absolute_delta_nll_cumulative": "regression",
        "delta_nll_cumulative": "regression",
        "is_opd": "classification",
    }
    metric_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for target, task in targets.items():
        per_target: dict[str, dict[str, float]] = {}
        for block, cols in blocks.items():
            subset = common.dropna(subset=cols + [target]).copy().reset_index(drop=True)
            if subset.empty:
                continue
            X = subset[cols].to_numpy(float)
            y = subset[target].to_numpy(float)
            groups = subset["checkpoint"].to_numpy(int)
            pred = np.full(len(y), np.nan)
            for held in sorted(set(groups)):
                train = groups != held
                test = groups == held
                if task == "regression":
                    selected, inner = nested_select_regression(X, y, groups, train)
                    Xt, Xv = train_standardize(X, train, test)
                    pred[test] = predict_linear(Xv, fit_ridge(Xt, y[train], selected))
                    fold_metric = regression_metrics(y[test], pred[test])
                else:
                    if len(set(y[train].astype(int))) < 2:
                        continue
                    selected, inner = nested_select_classification(X, y, groups, train)
                    Xt, Xv = train_standardize(X, train, test)
                    pred[test] = predict_logistic(Xv, fit_logistic(Xt, y[train], selected))
                    fold_metric = classification_metrics(y[test], pred[test])
                fold_rows.append({
                    "model": "llama",
                    "target": target,
                    "task_type": task,
                    "feature_block": block,
                    "heldout_checkpoint": int(held),
                    "selected_regularization": float(selected),
                    "inner_grid": json.dumps(inner, sort_keys=True),
                    "test_n": int(test.sum()),
                    **fold_metric,
                })
            metric = regression_metrics(y, pred) if task == "regression" else classification_metrics(y, pred)
            metric.update({
                "model": "llama",
                "target": target,
                "task_type": task,
                "feature_block": block,
                "features": ",".join(cols),
                "n_common": int(len(subset)),
                "n_checkpoint_groups": int(subset["checkpoint"].nunique()),
            })
            per_target[block] = metric
            metric_rows.append(metric)
            pred_name = "y_pred" if task == "regression" else "y_prob"
            for i, r in subset.iterrows():
                pred_rows.append({
                    "model": "llama",
                    "target": target,
                    "task_type": task,
                    "feature_block": block,
                    "row_id": int(r["row_id"]),
                    "arm": r["arm"],
                    "checkpoint": int(r["checkpoint"]),
                    "probe_name": r["probe_name"],
                    "y_true": float(y[i]),
                    pred_name: float(pred[i]) if np.isfinite(pred[i]) else np.nan,
                })
        for m in metric_rows:
            if m["target"] != target:
                continue
            for baseline in ["A", "C5", "Pk5"]:
                base = per_target.get(baseline, {})
                if task == "regression":
                    m[f"delta_r2_vs_{baseline}"] = m.get("r2_oof", np.nan) - base.get("r2_oof", np.nan)
                    m[f"mae_reduction_vs_{baseline}"] = base.get("mae_oof", np.nan) - m.get("mae_oof", np.nan)
                    m[f"delta_spearman_vs_{baseline}"] = m.get("spearman_oof", np.nan) - base.get("spearman_oof", np.nan)
                else:
                    m[f"delta_auc_vs_{baseline}"] = m.get("auc_oof", np.nan) - base.get("auc_oof", np.nan)
                    m[f"log_loss_reduction_vs_{baseline}"] = base.get("log_loss_oof", np.nan) - m.get("log_loss_oof", np.nan)
                    m[f"delta_balanced_accuracy_vs_{baseline}"] = m.get("balanced_accuracy_oof", np.nan) - base.get("balanced_accuracy_oof", np.nan)

    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty and not old_metrics.empty:
        map_block = {"A": "A", "C5": "C", "Pk5": "Pk", "A+C5": "A+C", "Pk5+A": "Pk+A", "Pk5+C5": "Pk+C", "Pk5+A+C5": "Pk+A+C"}
        metrics["equal7_feature_block"] = metrics["feature_block"].map(map_block)
        old = old_metrics.rename(columns={
            "feature_block": "equal7_feature_block",
            "r2_oof": "r2_oof_equal7",
            "mae_oof": "mae_oof_equal7",
            "spearman_oof": "spearman_oof_equal7",
            "auc_oof": "auc_oof_equal7",
            "log_loss_oof": "log_loss_oof_equal7",
            "balanced_accuracy_oof": "balanced_accuracy_oof_equal7",
        })
        metrics = metrics.merge(
            old[["target", "equal7_feature_block"] + [c for c in old.columns if c.endswith("_equal7")]],
            on=["target", "equal7_feature_block"],
            how="left",
        )
        for c in ["r2_oof", "mae_oof", "spearman_oof", "auc_oof", "log_loss_oof", "balanced_accuracy_oof"]:
            if c in metrics and f"{c}_equal7" in metrics:
                metrics[f"delta_{c}_vs_equal7"] = metrics[c] - metrics[f"{c}_equal7"]
    return metrics, pd.DataFrame(fold_rows), pd.DataFrame(pred_rows)


def d11_parity(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    targets = {
        "cumulative_kl_base_to_current": "regression",
        "absolute_delta_nll_cumulative": "regression",
        "delta_nll_cumulative": "regression",
        "is_opd": "classification",
    }
    feature_sets = {
        "C5": ["c_equal5"],
        "W5": ["raw_update_energy_equal5"],
        "Pk5": ["p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5"],
        "TPNT5": ["tpnt_overlap_lift_equal5", "tpnt_lift_minus_null_equal5", "pabs_joint_mean_cos_equal5", "nss_l1_top32_equal5"],
        "Pk5_TPNT5_C5": ["p_k4_equal5", "p_k8_equal5", "p_k16_equal5", "p_k32_equal5", "tpnt_overlap_lift_equal5", "tpnt_lift_minus_null_equal5", "pabs_joint_mean_cos_equal5", "nss_l1_top32_equal5", "c_equal5"],
    }
    for scope, df in [("pooled_available_complete_rows", feature_matrix), ("llama_only_complete_rows", feature_matrix[feature_matrix["model"] == "llama"])]:
        for target, task in targets.items():
            for name, cols in feature_sets.items():
                if any(c not in df.columns for c in cols) or target not in df.columns:
                    rows.append({"analysis": "D11_broad_parity_equal5", "model_scope": scope, "target": target, "feature_set": name, "status": "BLOCKED_NO_MODULE_SOURCE"})
                    continue
                sub = df.dropna(subset=cols + [target]).copy().reset_index(drop=True)
                if len(sub) < 12 or sub["checkpoint"].nunique() < 2:
                    rows.append({"analysis": "D11_broad_parity_equal5", "model_scope": scope, "target": target, "feature_set": name, "n": int(len(sub)), "status": "BLOCKED_INSUFFICIENT_COMPLETE_ROWS"})
                    continue
                X = sub[cols].to_numpy(float)
                y = sub[target].to_numpy(float)
                groups = sub["checkpoint"].to_numpy(int)
                pred = np.full(len(y), np.nan)
                for held in sorted(set(groups)):
                    train = groups != held
                    test = groups == held
                    if task == "classification":
                        if len(set(y[train].astype(int))) < 2:
                            continue
                        Xt, Xv = train_standardize(X, train, test)
                        pred[test] = predict_logistic(Xv, fit_logistic(Xt, y[train], 1e-4))
                    else:
                        Xt, Xv = train_standardize(X, train, test)
                        pred[test] = predict_linear(Xv, fit_ridge(Xt, y[train], 1e-4))
                metric = classification_metrics(y, pred) if task == "classification" else regression_metrics(y, pred)
                rows.append({
                    "analysis": "D11_broad_parity_equal5",
                    "model_scope": scope,
                    "target": target,
                    "feature_set": name,
                    "features": ",".join(cols),
                    "n": int(len(sub)),
                    "status": "DESCRIPTIVE_PARITY_ONLY_NOT_RR5_FORMAL",
                    **metric,
                })
    rows.append({
        "analysis": "D11_broad_parity_equal5",
        "model_scope": "dual_model_all_features",
        "status": "BLOCKED_NO_MODULE_SOURCE",
        "reason": "Qwen per-module p_k source unavailable; full D11 same-cell matrix cannot be exactly rebuilt for both models.",
    })
    return pd.DataFrame(rows)


def paired_comparison(functional: pd.DataFrame, paired: pd.DataFrame, dominance_cells: pd.DataFrame, ncd_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in paired.itertuples():
        rows.append({
            "comparison_type": "functional_cell",
            "model": r.model,
            "arm": r.arm,
            "checkpoint": int(r.checkpoint),
            "probe_name": r.probe_name,
            "artifact_probe_name": r.artifact_probe_name,
            "epsilon": float(r.epsilon),
            "layer": int(r.layer),
            "equal7_value": float(r.c_equal7) if hasattr(r, "c_equal7") and pd.notna(r.c_equal7) else np.nan,
            "equal5_value": float(r.c_equal5_non_qk) if hasattr(r, "c_equal5_non_qk") and pd.notna(r.c_equal5_non_qk) else np.nan,
            "equal5_minus_equal7": float(r.equal5_minus_equal7_c) if hasattr(r, "equal5_minus_equal7_c") and pd.notna(r.equal5_minus_equal7_c) else np.nan,
            "sign_changed": bool(r.sign_changed_c) if hasattr(r, "sign_changed_c") else False,
            "ordering_changed": np.nan,
            "deepest_arm_changed": np.nan,
            "rank_or_score_changed": bool(abs(r.equal5_minus_equal7_c) > TIE_ATOL) if hasattr(r, "equal5_minus_equal7_c") and pd.notna(r.equal5_minus_equal7_c) else np.nan,
            "source_rows_complete": bool(r.source_rows_complete),
            "excluded_modules_exactly_qk": True,
        })
    if not dominance_cells.empty:
        rows.append({
            "comparison_type": "dominance_summary",
            "n_cells": int(len(dominance_cells)),
            "OPD_strict_deepest": int(dominance_cells["OPD_strict_deepest"].sum()),
            "offline_strictly_deeper": int(dominance_cells["offline_strictly_deeper"].sum()),
            "mechanical_readback": "dominance recomputed from equal5; compare to equal7 via EQUAL5_dominance_cells plus source equal7 audit if needed",
        })
    if not ncd_df.empty and "equal5_minus_equal7_ncd" in ncd_df:
        rows.append({
            "comparison_type": "NCD_summary",
            "n_cells": int(ncd_df["equal5_minus_equal7_ncd"].notna().sum()),
            "mean_equal5_minus_equal7": float(ncd_df["equal5_minus_equal7_ncd"].mean()),
            "mechanical_readback": "NCD paired equal5-equal7 generated",
        })
    return pd.DataFrame(rows)


def support_controls(inventory: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([
        {"control": "GPU used", "value": False, "status": "PASS"},
        {"control": "new_forward", "value": False, "status": "PASS"},
        {"control": "new_training", "value": False, "status": "PASS"},
        {"control": "new_behavior_eval", "value": False, "status": "PASS"},
        {"control": "new_svd", "value": False, "status": "PASS"},
        {"control": "measurement_side_exclusion", "value": True, "status": "PASS"},
        {"control": "included_modules", "value": ",".join(M5), "status": "PASS"},
        {"control": "excluded_modules", "value": ",".join(EXCLUDED), "status": "PASS"},
        {"control": "blocked_cells", "value": int((inventory.get("status", pd.Series(dtype=str)) == "BLOCKED_NO_MODULE_SOURCE").sum()), "status": "RECORDED"},
        {"control": "superseded_sources_forbidden", "value": "RR2D displacement spectrum not used for state-rank robustness", "status": "PASS"},
    ])


def sanity_checks(dominance_summary: pd.DataFrame, dominance_cells: pd.DataFrame, spectrum: pd.DataFrame) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    def count(model: str, eps: float | None = None) -> tuple[int, int]:
        g = dominance_cells[dominance_cells["model"] == model]
        if eps is not None:
            g = g[np.isclose(g["epsilon"], eps)]
        return int(g["OPD_strict_deepest"].sum()), int(len(g))
    l_all = count("llama")
    q_005 = count("qwen", 0.05)
    q_all = count("qwen")
    pooled_005 = dominance_cells[np.isclose(dominance_cells["epsilon"], 0.05)]
    checks.append({"check": "Llama uncentered equal5 all epsilon early cells", "observed": f"{l_all[0]}/{l_all[1]}", "expected_temporary_theory_read": "48/48", "pass": l_all == (48, 48)})
    checks.append({"check": "Qwen uncentered equal5 epsilon .05 early cells", "observed": f"{q_005[0]}/{q_005[1]}", "expected_temporary_theory_read": "12/12", "pass": q_005 == (12, 12)})
    checks.append({"check": "Qwen uncentered equal5 all epsilon early cells", "observed": f"{q_all[0]}/{q_all[1]}", "expected_temporary_theory_read": "47/48", "pass": q_all == (47, 48)})
    checks.append({"check": "Pooled Llama+Qwen epsilon .05 early cells", "observed": f"{int(pooled_005['OPD_strict_deepest'].sum())}/{len(pooled_005)}", "expected_temporary_theory_read": "24/24", "pass": int(pooled_005["OPD_strict_deepest"].sum()) == 24 and len(pooled_005) == 24})
    centered = spectrum[spectrum.get("analysis", pd.Series(dtype=str)) == "RR3_centered_equal5_OPD_dominance"]
    checks.append({
        "check": "Llama centered equal5 all epsilon early cells",
        "observed": f"{int(centered.get('OPD_strict_deepest', pd.Series(dtype=bool)).sum())}/{len(centered)}",
        "expected_temporary_theory_read": "48/48",
        "pass": int(centered.get("OPD_strict_deepest", pd.Series(dtype=bool)).sum()) == 48 and len(centered) == 48,
    })
    return checks


def write_handoff(
    manifest: dict[str, Any],
    dominance_summary: pd.DataFrame,
    ncd_df: pd.DataFrame,
    spectrum: pd.DataFrame,
    nested_metrics: pd.DataFrame,
    d11_metrics: pd.DataFrame,
    paired: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# EQUAL5_NON_QK Theory Handoff")
    lines.append("")
    lines.append(f"created_utc: `{manifest['created_utc']}`")
    lines.append("")
    lines.append("Boundary: reuse-only CPU aggregation. No GPU, no training, no model forward, no behavior eval, no new SVD. Module exclusion is measurement-side non-q/k equal-5 aggregation, not adapter ablation.")
    lines.append("")
    lines.append("## Module Set")
    lines.append("")
    lines.append(f"included: `{', '.join(M5)}`")
    lines.append(f"excluded: `{', '.join(EXCLUDED)}`")
    lines.append("")
    lines.append("## Dominance Sanity")
    lines.append("")
    if dominance_summary.empty:
        lines.append("No dominance rows produced.")
    else:
        lines.append(dominance_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## NCD")
    lines.append("")
    if ncd_df.empty:
        lines.append("No NCD rows produced.")
    else:
        lines.append(ncd_df.head(80).to_markdown(index=False))
    lines.append("")
    lines.append("## Spectrum Robustness")
    lines.append("")
    if spectrum.empty:
        lines.append("No spectrum rows produced.")
    else:
        counts = spectrum.groupby(["analysis", "source_status"], dropna=False).size().reset_index(name="rows")
        lines.append(counts.to_markdown(index=False))
    lines.append("")
    lines.append("## RR5 Nested Equal5")
    lines.append("")
    if nested_metrics.empty:
        lines.append("No nested rows produced.")
    else:
        lines.append(nested_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## D11 Parity")
    lines.append("")
    if d11_metrics.empty:
        lines.append("No D11 parity rows produced.")
    else:
        lines.append(d11_metrics.to_markdown(index=False))
    lines.append("")
    lines.append("## Paired Audit")
    lines.append("")
    lines.append(f"paired rows: `{len(paired)}`")
    lines.append("")
    lines.append("## Blocked")
    lines.append("")
    for row in manifest["blocked_cells"]:
        lines.append(f"- `{row.get('task')}`: `{row.get('status')}`; {row.get('reason')}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for name, count in manifest["row_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.append("")
    (OUT / "EQUAL5_theory_handoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_code_handoffs(manifest: dict[str, Any]) -> None:
    section = [
        "",
        f"## EQUAL5_NON_QK Reuse Return: {manifest['created_utc']}",
        "",
        "- Task: measurement-side non-q/k equal-5 aggregation from existing module artifacts.",
        "- Boundary: GPU used=false; new_forward=false; new_training=false; new_behavior_eval=false; new_svd=false.",
        f"- Command: `{manifest['command']}`",
        f"- Output directory: `{OUT}`",
        "- Completed outputs:",
    ]
    for name, count in manifest["row_counts"].items():
        section.append(f"  - `{name}`: {count} rows")
    section.append("- Blocked items were recorded in `EQUAL5_coverage_inventory.csv` and `EQUAL5_manifest.json`; blocked rows were not reverse-engineered from equal-7 aggregates.")
    text = "\n".join(section) + "\n"
    for path in [REPO / "mypaper/code/cycle09_reviewer_robustness_handoff.md", REPO / "mypaper/code/code_evolution.md"]:
        if path.exists():
            with path.open("a", encoding="utf-8") as f:
                f.write(text)


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    module_df = normalize_state_sources()
    inventory = make_coverage_inventory(module_df)
    inventory.to_csv(OUT / "EQUAL5_coverage_inventory.csv", index=False)

    _, functional, paired = make_functional_outputs(module_df)
    functional.to_csv(OUT / "EQUAL5_functional_trajectories.csv", index=False)
    # Module-detail table uses the normalized module rows, limited to included M5.
    detail = module_df[module_df["module"].isin(M5)].copy()
    detail["module_set"] = "equal5_non_qk"
    detail.to_csv(OUT / "EQUAL5_functional_cells.csv", index=False)

    dom_cells, dom_summary = dominance(functional)
    dom_cells.to_csv(OUT / "EQUAL5_dominance_cells.csv", index=False)
    dom_summary.to_csv(OUT / "EQUAL5_dominance_summary.csv", index=False)

    ncd_df = ncd(functional, paired)
    ncd_df.to_csv(OUT / "EQUAL5_ncd.csv", index=False)

    spectrum = spectrum_robustness()
    spectrum.to_csv(OUT / "EQUAL5_spectrum_robustness.csv", index=False)

    controls = support_controls(inventory)
    controls.to_csv(OUT / "EQUAL5_support_controls.csv", index=False)

    weight_module, weight_agg = weight_baselines()
    weight_module.to_csv(OUT / "EQUAL5_weight_baselines_module.csv", index=False)
    weight_agg.to_csv(OUT / "EQUAL5_weight_baselines_aggregate.csv", index=False)

    out_corr, demean_corr = output_link_correlations(functional, paired)
    out_corr.to_csv(OUT / "EQUAL5_output_link_correlations.csv", index=False)
    demean_corr.to_csv(OUT / "EQUAL5_checkpoint_demeaned_correlations.csv", index=False)

    feature_matrix = build_equal5_feature_matrix(functional, weight_agg)
    feature_matrix.to_csv(SCRATCH / "EQUAL5_same_cell_feature_matrix.csv", index=False)
    nested_metrics, nested_folds, nested_preds = nested_equal5(feature_matrix)
    nested_metrics.to_csv(OUT / "EQUAL5_nested_metrics.csv", index=False)
    nested_folds.to_csv(OUT / "EQUAL5_nested_folds.csv", index=False)
    nested_preds.to_parquet(OUT / "EQUAL5_nested_predictions.parquet", index=False)

    d11_metrics = d11_parity(feature_matrix)
    d11_metrics.to_csv(OUT / "EQUAL5_d11_parity_metrics.csv", index=False)

    pair_audit = paired_comparison(functional, paired, dom_cells, ncd_df)
    pair_audit.to_csv(OUT / "EQUAL5_equal7_paired_comparison.csv", index=False)

    checks = sanity_checks(dom_summary, dom_cells, spectrum)
    blocked = inventory[inventory["status"].isin(["BLOCKED_NO_MODULE_SOURCE", "SUPERSEDED_SOURCE_FORBIDDEN"])].fillna("").to_dict("records")
    manifest = {
        "created_utc": utc_now(),
        "git_commit": git_commit(),
        "command": " ".join(os.sys.argv),
        "task": "EQUAL5_NON_QK",
        "module_set_include": M5,
        "module_set_exclude": EXCLUDED,
        "measurement_side_exclusion": True,
        "model_checkpoint_unchanged": True,
        "activation_cache_from_full_checkpoint": True,
        "gpu_used": False,
        "new_forward": False,
        "new_training": False,
        "new_behavior_eval": False,
        "new_svd": False,
        "tie_policy": {"np_isclose_rtol": TIE_RTOL, "np_isclose_atol": TIE_ATOL},
        "nested_folds_and_regularization_grid": read_json(SOURCES["rr5_nested_manifest"]).get("nested_regularization_grid", {}),
        "models": sorted(module_df["model"].dropna().unique().tolist()) if not module_df.empty else [],
        "arms": sorted(module_df["arm"].dropna().unique().tolist()) if not module_df.empty else [],
        "checkpoints": sorted([int(x) for x in module_df["checkpoint"].dropna().unique().tolist()]) if not module_df.empty else [],
        "probes_canonical": sorted(module_df["probe_name"].dropna().unique().tolist()) if not module_df.empty else [],
        "layers": sorted([int(x) for x in module_df["layer"].dropna().unique().tolist()]) if not module_df.empty else [],
        "epsilons": EPSILONS,
        "sanity_checks": checks,
        "blocked_cells": blocked,
        "runtime_seconds": round(time.time() - start, 3),
        "input_paths_and_sha256": {k: {"path": str(p), "sha256": sha256_file(p), "exists": p.exists()} for k, p in SOURCES.items()},
        "source_protocols": {
            "qwen_state_module": "D4.1 current merged/deployed state rank module rows",
            "llama_state_module": "D10 BF16 formal numeric protocol module rows",
            "llama_rr2s_spectrum_module": "RR2S formal state spectrum reuse; no forward",
            "llama_rr3_centered_module": "RR3 formal centered spectrum reuse; no forward",
            "d11_weight_baselines": "D11 formal BF16 merged-minus-base/deployed-effective module rows",
            "rr5_nested": "RR5 leave-one-checkpoint nested regularization inherited protocol",
        },
        "dtype_merge_protocols": {
            "llama_pk": "formal BF16 merged-minus-base track; adapter-BA not used",
            "qwen_pk": "blocked/no per-module source found",
            "state_rank": "existing formal module CSV only",
        },
        "row_counts": {
            "EQUAL5_coverage_inventory.csv": int(len(inventory)),
            "EQUAL5_functional_cells.csv": int(len(detail)),
            "EQUAL5_functional_trajectories.csv": int(len(functional)),
            "EQUAL5_dominance_cells.csv": int(len(dom_cells)),
            "EQUAL5_dominance_summary.csv": int(len(dom_summary)),
            "EQUAL5_ncd.csv": int(len(ncd_df)),
            "EQUAL5_spectrum_robustness.csv": int(len(spectrum)),
            "EQUAL5_support_controls.csv": int(len(controls)),
            "EQUAL5_weight_baselines_module.csv": int(len(weight_module)),
            "EQUAL5_weight_baselines_aggregate.csv": int(len(weight_agg)),
            "EQUAL5_output_link_correlations.csv": int(len(out_corr)),
            "EQUAL5_checkpoint_demeaned_correlations.csv": int(len(demean_corr)),
            "EQUAL5_nested_metrics.csv": int(len(nested_metrics)),
            "EQUAL5_nested_folds.csv": int(len(nested_folds)),
            "EQUAL5_nested_predictions.parquet": int(len(nested_preds)),
            "EQUAL5_d11_parity_metrics.csv": int(len(d11_metrics)),
            "EQUAL5_equal7_paired_comparison.csv": int(len(pair_audit)),
        },
    }
    write_handoff(manifest, dom_summary, ncd_df, spectrum, nested_metrics, d11_metrics, pair_audit)
    manifest["output_sha256"] = {p.name: sha256_file(p) for p in OUT.iterdir() if p.is_file()}
    write_json(OUT / "EQUAL5_manifest.json", manifest)
    append_code_handoffs(manifest)
    print(json.dumps({
        "status": "COMPLETE_EQUAL5_NON_QK_REUSE_ONLY",
        "out": str(OUT),
        "runtime_seconds": manifest["runtime_seconds"],
        "sanity_checks": checks,
        "row_counts": manifest["row_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
