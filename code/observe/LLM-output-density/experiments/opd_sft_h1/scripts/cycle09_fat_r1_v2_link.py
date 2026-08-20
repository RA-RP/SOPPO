#!/usr/bin/env python3
"""FAT-R1-v2 output-link reuse analysis.

This script performs CPU-only joins and statistics over existing FAT-R1-v2
regional NLL/KL outputs, c_epsilon feature rows, and deployed fixed-k p_k rows.
It does not load models, run forward passes, or modify previous artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path("/root/LLM-output-density")
MINI = ROOT / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
FAT = MINI / "fat_outlink_round1_v2"
OUT = MINI / "fat_outlink_round1_v2_link"

MMLU_TARGETS = [
    "delta_nll_p",
    "delta_nll_f",
    "delta_nll_a",
    "delta_nll_t",
    "kl_f",
    "kl_a",
    "kl_t",
    "abs_delta_nll_f",
    "abs_delta_nll_a",
    "abs_delta_nll_t",
    "delta_nll_f_minus_a",
    "delta_nll_f_minus_p",
    "kl_f_minus_a",
]
MATH_TARGETS = [
    "delta_nll_p",
    "delta_nll_c",
    "delta_nll_b",
    "delta_nll_t",
    "kl_b",
    "kl_t",
    "abs_delta_nll_c",
    "abs_delta_nll_b",
    "abs_delta_nll_t",
    "delta_nll_b_minus_c",
    "delta_nll_b_minus_p",
]
SIGNED_TARGETS = {
    "delta_nll_p",
    "delta_nll_f",
    "delta_nll_a",
    "delta_nll_t",
    "delta_nll_c",
    "delta_nll_b",
}
PK_COLS = ["p_k4", "p_k8", "p_k16", "p_k32"]
ARMS = ["offkd", "opd", "seqkd", "sft"]
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        y = float(x)
    except Exception:
        return float("nan")
    return y if math.isfinite(y) else float("nan")


def pearson(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a = pd.Series(list(x), dtype="float64")
    b = pd.Series(list(y), dtype="float64")
    m = a.notna() & b.notna()
    if int(m.sum()) < 2:
        return float("nan")
    ar = a[m].rank(method="average")
    br = b[m].rank(method="average")
    return pearson(ar, br)


def kendall_tau_b(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 2:
        return float("nan")
    concordant = discordant = tie_x = tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(a[i] - a[j])
            dy = np.sign(b[i] - b[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_x += 1
            elif dy == 0:
                tie_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[m], pred[m]
    if len(y) == 0:
        return float("nan")
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mae_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - pred[m])))


def add_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_nll_f_minus_a"] = out.get("delta_nll_f") - out.get("delta_nll_a")
    out["delta_nll_f_minus_p"] = out.get("delta_nll_f") - out.get("delta_nll_p")
    out["kl_f_minus_a"] = out.get("kl_f") - out.get("kl_a")
    out["delta_nll_b_minus_c"] = out.get("delta_nll_b") - out.get("delta_nll_c")
    out["delta_nll_b_minus_p"] = out.get("delta_nll_b") - out.get("delta_nll_p")
    out["kl_b_minus_c"] = np.nan
    return out


def load_output_cells() -> pd.DataFrame:
    mmlu = pd.read_csv(FAT / "fat_r1_v2_mmlu_cells.csv")
    mmlu = mmlu[mmlu["aggregation"].eq("sample_macro")].copy()
    math_df = pd.read_csv(FAT / "fat_r1_v2_math_cells.csv")
    cells = pd.concat([mmlu, math_df], ignore_index=True, sort=False)
    cells = add_contrasts(cells)
    cells["output_probe_name"] = np.where(cells["domain"].eq("mmlu"), "E_ood", "E_math")
    cells["display_domain"] = np.where(cells["domain"].eq("mmlu"), "E_mmluPro", "E_math")
    return cells


def load_output_samples() -> pd.DataFrame:
    mmlu = pd.read_csv(FAT / "fat_r1_v2_mmlu_samples.csv")
    math_df = pd.read_csv(FAT / "fat_r1_v2_math_samples.csv")
    samples = pd.concat([mmlu, math_df], ignore_index=True, sort=False)
    samples = add_contrasts(samples)
    return samples


def load_c_features() -> pd.DataFrame:
    df = pd.read_csv(MINI / "d10_5_a4_feature_matrix.csv")
    df = df[(df["epsilon"].astype(float).round(6) == 0.05)].copy()
    df = df[((df["model"] == "qwen") & (df["layer"] == 18)) | ((df["model"] == "llama") & (df["layer"] == 14))]
    df = df[df["probe_name"].isin(["E_ood", "E_math"])].copy()
    keep = [
        "model",
        "arm",
        "checkpoint",
        "probe_name",
        "c_epsilon",
        "state_rank_base_mean",
        "state_rank_current_mean",
        "state_rank_delta_mean",
        "source_name",
        "source_protocol",
        "track",
    ]
    df = df[keep].drop_duplicates(["model", "arm", "checkpoint", "probe_name"])
    return df


def aggregate_pk() -> pd.DataFrame:
    frames = []
    q = pd.read_csv(MINI / "T_PK_qwen3_4b_fixedk.csv")
    q = q[(q["layer"] == 18) & (q["delta_construction"].eq("bf16_merged_minus_base"))].copy()
    q["model"] = "qwen"
    q["checkpoint"] = q["step"].astype(int)
    q["pk_track"] = "qwen_deployed_bf16_merged_minus_base"
    frames.append(q)

    l = pd.read_csv(MINI / "d11_llama_merged_pk.csv")
    l = l[(l["layer"] == 14) & (l["delta_construction"].eq("bf16_merged_minus_base"))].copy()
    l["model"] = "llama"
    l["checkpoint"] = l["checkpoint"].astype(int)
    l["pk_track"] = "llama_d11_bf16_merged_minus_base"
    frames.append(l)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    rows = []
    for (model, arm, checkpoint, k), g in raw.groupby(["model", "arm", "checkpoint", "k"]):
        rows.append(
            {
                "model": model,
                "arm": arm,
                "checkpoint": int(checkpoint),
                "k": int(k),
                "p_k": float(g["p_k"].mean()),
                "module_count_pk": int(g["module"].nunique()),
                "pk_track": "|".join(sorted(map(str, g["pk_track"].dropna().unique()))),
            }
        )
    pk_long = pd.DataFrame(rows)
    pk = pk_long.pivot_table(index=["model", "arm", "checkpoint"], columns="k", values="p_k", aggfunc="first").reset_index()
    pk.columns = [f"p_k{c}" if isinstance(c, (int, np.integer)) else c for c in pk.columns]
    meta = pk_long.groupby(["model", "arm", "checkpoint"]).agg(
        module_count_pk=("module_count_pk", "min"), pk_track=("pk_track", "first")
    ).reset_index()
    pk = pk.merge(meta, on=["model", "arm", "checkpoint"], how="left")
    return pk


def build_feature_matrix() -> pd.DataFrame:
    cells = load_output_cells()
    cells = cells[cells["arm"].isin(ARMS)].copy()
    c = load_c_features()
    pk = aggregate_pk()
    joined = cells.merge(
        c,
        left_on=["model", "arm", "checkpoint", "output_probe_name"],
        right_on=["model", "arm", "checkpoint", "probe_name"],
        how="left",
        validate="one_to_one",
    )
    joined = joined.merge(pk, on=["model", "arm", "checkpoint"], how="left", validate="many_to_one")
    joined["has_c_epsilon"] = joined["c_epsilon"].notna()
    joined["has_pk"] = joined[PK_COLS].notna().all(axis=1)
    joined["pk_exclusion_reason"] = ""
    joined.loc[joined["model"].eq("qwen") & joined["checkpoint"].eq(10) & ~joined["has_pk"], "pk_exclusion_reason"] = (
        "QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION"
    )
    joined.loc[~joined["has_pk"] & joined["pk_exclusion_reason"].eq(""), "pk_exclusion_reason"] = "PK_MISSING_UNEXPECTED"
    joined["link_schema"] = "cycle09_fat_outlink_round1_v2_link"
    return joined


def coverage_table(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = {
        ("qwen", "mmlu"): 4 * 9,
        ("qwen", "math"): 4 * 9,
        ("llama", "mmlu"): 4 * 6,
        ("llama", "math"): 4 * 6,
    }
    for (model, domain), g in feature.groupby(["model", "domain"]):
        exp = expected[(model, domain)]
        rows.append(
            {
                "model": model,
                "domain": domain,
                "expected_c_rows": exp,
                "observed_c_rows": int(g["has_c_epsilon"].sum()),
                "expected_pk_rows": exp - (4 if model == "qwen" else 0),
                "observed_pk_rows": int(g["has_pk"].sum()),
                "missing_pk_cells": int((~g["has_pk"]).sum()),
                "missing_pk_reason": "|".join(sorted(g.loc[~g["has_pk"], "pk_exclusion_reason"].dropna().unique())),
            }
        )
    total = {
        "model": "ALL",
        "domain": "ALL",
        "expected_c_rows": 120,
        "observed_c_rows": int(feature["has_c_epsilon"].sum()),
        "expected_pk_rows": 112,
        "observed_pk_rows": int(feature["has_pk"].sum()),
        "missing_pk_cells": int((~feature["has_pk"]).sum()),
        "missing_pk_reason": "|".join(sorted(feature.loc[~feature["has_pk"], "pk_exclusion_reason"].dropna().unique())),
    }
    rows.append(total)
    return pd.DataFrame(rows)


def bootstrap_contrast_ci(samples: pd.DataFrame, draws: int = 1024, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contrast_by_domain = {
        "mmlu": ["delta_nll_f_minus_a", "delta_nll_f_minus_p", "kl_f_minus_a"],
        "math": ["delta_nll_b_minus_c", "delta_nll_b_minus_p"],
    }
    rows = []
    for (model, domain, arm, checkpoint), g in samples.groupby(["model", "domain", "arm", "checkpoint"], sort=True):
        for contrast in contrast_by_domain[domain]:
            vals = g[contrast].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            means = []
            for _ in range(draws):
                idx = rng.integers(0, len(vals), len(vals))
                means.append(float(np.mean(vals[idx])))
            lo, hi = np.quantile(means, [0.025, 0.975])
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "arm": arm,
                    "checkpoint": int(checkpoint),
                    "contrast": contrast,
                    "mean": float(np.mean(vals)),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "n_items": int(len(vals)),
                    "draws": draws,
                    "seed": seed,
                    "bootstrap_unit": "paired_item_rows_same_resample_indices",
                }
            )
    return pd.DataFrame(rows)


def targets_for_domain(domain: str) -> list[str]:
    return MMLU_TARGETS if domain == "mmlu" else MATH_TARGETS


def correlation_rows(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, domain, arm), g0 in feature.groupby(["model", "domain", "arm"]):
        g = g0[g0["checkpoint"] > 0].copy()
        for target in targets_for_domain(domain):
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "arm": arm,
                    "target": target,
                    "feature": "c_epsilon",
                    "n": int(g[["c_epsilon", target]].dropna().shape[0]),
                    "spearman": spearman(g["c_epsilon"], g[target]),
                    "pearson": pearson(g["c_epsilon"], g[target]),
                    "track": "c_epsilon_only_120_rows",
                }
            )
            gp = g[g["has_pk"]].copy()
            for pcol in PK_COLS:
                rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "arm": arm,
                        "target": target,
                        "feature": pcol,
                        "n": int(gp[[pcol, target]].dropna().shape[0]),
                        "spearman": spearman(gp[pcol], gp[target]),
                        "pearson": pearson(gp[pcol], gp[target]),
                        "track": "same_cell_pk_112_rows_qwen_step10_excluded",
                    }
                )
    return pd.DataFrame(rows)


def checkpoint_demeaned(feature: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    corr_rows = []
    all_features = ["c_epsilon"] + PK_COLS
    target_cols = sorted(set(MMLU_TARGETS + MATH_TARGETS))
    for (model, domain, checkpoint), g in feature.groupby(["model", "domain", "checkpoint"]):
        for feat in all_features:
            if feat != "c_epsilon" and not bool(g["has_pk"].all()):
                continue
            if g["arm"].nunique() != 4:
                continue
            feat_mean = g[feat].mean(skipna=True)
            for target in targets_for_domain(domain):
                target_mean = g[target].mean(skipna=True)
                for _, r in g.iterrows():
                    cell_rows.append(
                        {
                            "model": model,
                            "domain": domain,
                            "checkpoint": int(checkpoint),
                            "arm": r["arm"],
                            "feature": feat,
                            "target": target,
                            "feature_demeaned": safe_float(r[feat] - feat_mean),
                            "target_demeaned": safe_float(r[target] - target_mean),
                            "n_arms_in_checkpoint": int(g["arm"].nunique()),
                        }
                    )
    cells = pd.DataFrame(cell_rows)
    if cells.empty:
        return cells, pd.DataFrame()
    for (model, domain, feature_name, target), g in cells.groupby(["model", "domain", "feature", "target"]):
        corr_rows.append(
            {
                "model": model,
                "domain": domain,
                "feature": feature_name,
                "target": target,
                "n": int(g[["feature_demeaned", "target_demeaned"]].dropna().shape[0]),
                "spearman": spearman(g["feature_demeaned"], g["target_demeaned"]),
                "pearson": pearson(g["feature_demeaned"], g["target_demeaned"]),
                "demean_unit": "model_domain_checkpoint_four_arm_mean",
            }
        )
    return cells, pd.DataFrame(corr_rows)


def residualize_on_progress(values: pd.Series, checkpoints: pd.Series) -> np.ndarray:
    y = values.to_numpy(dtype=float)
    x = np.log1p(checkpoints.to_numpy(dtype=float))
    m = np.isfinite(y) & np.isfinite(x)
    resid = np.full_like(y, np.nan, dtype=float)
    if int(m.sum()) < 2:
        return resid
    X = np.column_stack([np.ones(int(m.sum())), x[m]])
    beta = np.linalg.pinv(X) @ y[m]
    resid[m] = y[m] - X @ beta
    return resid


def progress_residual_correlations(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, domain, arm), g0 in feature.groupby(["model", "domain", "arm"]):
        g = g0[g0["checkpoint"] > 0].sort_values("checkpoint").copy()
        for target in targets_for_domain(domain):
            for feat in ["c_epsilon"] + PK_COLS:
                if feat != "c_epsilon":
                    gg = g[g["has_pk"]].copy()
                else:
                    gg = g
                rx = residualize_on_progress(gg[feat], gg["checkpoint"])
                ry = residualize_on_progress(gg[target], gg["checkpoint"])
                rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "arm": arm,
                        "feature": feat,
                        "target": target,
                        "n": int((np.isfinite(rx) & np.isfinite(ry)).sum()) if len(rx) else 0,
                        "spearman": spearman(rx, ry),
                        "pearson": pearson(rx, ry),
                        "residual_model": "intercept_plus_log1p_checkpoint_within_model_domain_arm",
                    }
                )
    return pd.DataFrame(rows)


def within_checkpoint_rank(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, domain, checkpoint), g in feature.groupby(["model", "domain", "checkpoint"]):
        if g["arm"].nunique() != 4:
            continue
        for target in targets_for_domain(domain):
            for feat in ["c_epsilon"] + PK_COLS:
                if feat != "c_epsilon" and not bool(g["has_pk"].all()):
                    continue
                sp = spearman(g[feat], g[target])
                kt = kendall_tau_b(g[feat], g[target])
                rows.append(
                    {
                        "row_type": "cell",
                        "model": model,
                        "domain": domain,
                        "checkpoint": int(checkpoint),
                        "feature": feat,
                        "target": target,
                        "n_arms": int(g["arm"].nunique()),
                        "spearman": sp,
                        "kendall_tau_b": kt,
                        "median_spearman": np.nan,
                        "sign_consistency_rate": np.nan,
                    }
                )
    cell_df = pd.DataFrame(rows)
    summaries = []
    if not cell_df.empty:
        for (model, domain, feature_name, target), g in cell_df.groupby(["model", "domain", "feature", "target"]):
            vals = g["spearman"].dropna()
            summaries.append(
                {
                    "row_type": "summary",
                    "model": model,
                    "domain": domain,
                    "checkpoint": "ALL",
                    "feature": feature_name,
                    "target": target,
                    "n_arms": 4,
                    "spearman": np.nan,
                    "kendall_tau_b": np.nan,
                    "median_spearman": float(vals.median()) if len(vals) else np.nan,
                    "sign_consistency_rate": float((np.sign(vals) == np.sign(vals.median())).mean()) if len(vals) else np.nan,
                }
            )
    return pd.concat([cell_df, pd.DataFrame(summaries)], ignore_index=True, sort=False)


def design_matrix(df: pd.DataFrame, block: str) -> tuple[np.ndarray, list[str]]:
    cols = ["intercept"]
    mats = [np.ones((len(df), 1), dtype=float)]
    for arm in ARMS:
        cols.append(f"arm_{arm}")
        mats.append((df["arm"].to_numpy() == arm).astype(float).reshape(-1, 1))
    cols.append("log1p_checkpoint")
    mats.append(np.log1p(df["checkpoint"].to_numpy(dtype=float)).reshape(-1, 1))
    block_cols: list[str] = []
    if block == "MC":
        block_cols = ["c_epsilon"]
    elif block.startswith("MP("):
        block_cols = [block[3:-1]]
    elif block == "MPall":
        block_cols = PK_COLS
    elif block == "MPC":
        block_cols = PK_COLS + ["c_epsilon"]
    elif block == "M0":
        block_cols = []
    else:
        raise ValueError(block)
    for c in block_cols:
        cols.append(c)
        mats.append(df[c].to_numpy(dtype=float).reshape(-1, 1))
    return np.hstack(mats), cols


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = X.copy().astype(float)
        if X.shape[1] > 1:
            out[:, 1:] = (out[:, 1:] - self.mean[1:]) / self.std[1:]
        return out


def fit_standardizer(X: np.ndarray) -> Standardizer:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    mean[0] = 0.0
    std[0] = 1.0
    return Standardizer(mean, std)


def ridge_fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, target: str, block: str, alpha: float
) -> tuple[np.ndarray, list[str]]:
    X_train, cols = design_matrix(train, block)
    X_test, _ = design_matrix(test, block)
    st = fit_standardizer(X_train)
    Xtr = st.transform(X_train)
    Xte = st.transform(X_test)
    y = train[target].to_numpy(dtype=float)
    reg = np.eye(Xtr.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(Xtr.T @ Xtr + reg) @ Xtr.T @ y
    return Xte @ beta, cols


def choose_alpha(train: pd.DataFrame, target: str, block: str) -> float:
    checkpoints = sorted(train["checkpoint"].unique())
    if len(checkpoints) < 3:
        return 1e-3
    scores = []
    for alpha in ALPHAS:
        fold_mae = []
        for ckpt in checkpoints:
            tr = train[train["checkpoint"] != ckpt]
            va = train[train["checkpoint"] == ckpt]
            if len(tr) == 0 or len(va) == 0:
                continue
            pred, _ = ridge_fit_predict(tr, va, target, block, alpha)
            fold_mae.append(mae_score(va[target].to_numpy(dtype=float), pred))
        scores.append((float(np.nanmean(fold_mae)) if fold_mae else np.inf, alpha))
    scores.sort(key=lambda z: (z[0], z[1]))
    return float(scores[0][1])


def incremental_models(feature: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    pred_rows = []
    blocks = ["M0", "MC", "MP(p_k4)", "MP(p_k8)", "MP(p_k16)", "MP(p_k32)", "MPall", "MPC"]
    pk_feature = feature[feature["has_pk"] & feature["has_c_epsilon"] & (feature["checkpoint"] > 0)].copy()
    pred_id = 0
    for (model, domain), gd in pk_feature.groupby(["model", "domain"]):
        targets = targets_for_domain(domain)
        checkpoints = sorted(gd["checkpoint"].unique())
        for target in targets:
            if gd[target].notna().sum() < 8:
                continue
            block_preds: dict[str, list[float]] = {b: [] for b in blocks}
            block_y: dict[str, list[float]] = {b: [] for b in blocks}
            fold_win_mae: dict[str, int] = {b: 0 for b in blocks}
            selected_alphas: dict[str, list[float]] = {b: [] for b in blocks}
            for block in blocks:
                for ckpt in checkpoints:
                    train = gd[gd["checkpoint"] != ckpt].dropna(subset=[target] + (["c_epsilon"] if block in ["MC", "MPC"] else []))
                    test = gd[gd["checkpoint"] == ckpt].dropna(subset=[target])
                    needed = []
                    if block == "MC":
                        needed = ["c_epsilon"]
                    elif block.startswith("MP("):
                        needed = [block[3:-1]]
                    elif block == "MPall":
                        needed = PK_COLS
                    elif block == "MPC":
                        needed = PK_COLS + ["c_epsilon"]
                    train = train.dropna(subset=needed)
                    test = test.dropna(subset=needed)
                    if len(train) == 0 or len(test) == 0:
                        continue
                    alpha = choose_alpha(train, target, block)
                    selected_alphas[block].append(alpha)
                    pred, cols = ridge_fit_predict(train, test, target, block, alpha)
                    y = test[target].to_numpy(dtype=float)
                    block_preds[block].extend(pred.tolist())
                    block_y[block].extend(y.tolist())
                    for idx, (_, r) in enumerate(test.iterrows()):
                        pred_rows.append(
                            {
                                "prediction_id": pred_id,
                                "model": model,
                                "domain": domain,
                                "target": target,
                                "feature_block": block,
                                "heldout_checkpoint": int(ckpt),
                                "arm": r["arm"],
                                "checkpoint": int(r["checkpoint"]),
                                "y_true": float(y[idx]),
                                "y_pred": float(pred[idx]),
                                "selected_alpha": alpha,
                                "feature_columns": "|".join(cols),
                            }
                        )
                        pred_id += 1
            metrics = {}
            for block in blocks:
                y = np.asarray(block_y[block], dtype=float)
                pred = np.asarray(block_preds[block], dtype=float)
                metrics[block] = {
                    "heldout_r2": r2_score(y, pred),
                    "mae": mae_score(y, pred),
                    "prediction_spearman": spearman(y, pred),
                    "n_predictions": int(len(y)),
                    "selected_alpha_median": float(np.nanmedian(selected_alphas[block])) if selected_alphas[block] else np.nan,
                }

            # Foldwise win counts are MAE comparisons at held-out checkpoint level.
            pred_df_tmp = pd.DataFrame([r for r in pred_rows if r["model"] == model and r["domain"] == domain and r["target"] == target])
            n_outer_folds = int(len(checkpoints))

            def fold_win_count(candidate: str, baseline: str) -> int:
                if pred_df_tmp.empty:
                    return 0
                wins = 0
                for ckpt, fg in pred_df_tmp.groupby("heldout_checkpoint"):
                    gc = fg[fg["feature_block"] == candidate]
                    gb = fg[fg["feature_block"] == baseline]
                    if gc.empty or gb.empty:
                        continue
                    mc = mae_score(gc["y_true"].to_numpy(dtype=float), gc["y_pred"].to_numpy(dtype=float))
                    mb = mae_score(gb["y_true"].to_numpy(dtype=float), gb["y_pred"].to_numpy(dtype=float))
                    if np.isfinite(mc) and np.isfinite(mb) and mc < mb:
                        wins += 1
                return wins

            wins_mc_m0 = fold_win_count("MC", "M0")
            wins_mpc_mpall = fold_win_count("MPC", "MPall")
            for block in blocks:
                rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "target": target,
                        "feature_block": block,
                        **metrics[block],
                        "delta_r2_MC_minus_M0": metrics["MC"]["heldout_r2"] - metrics["M0"]["heldout_r2"],
                        "delta_r2_MPC_minus_MPall": metrics["MPC"]["heldout_r2"] - metrics["MPall"]["heldout_r2"],
                        "delta_mae_MPC_minus_MPall": metrics["MPC"]["mae"] - metrics["MPall"]["mae"],
                        "foldwise_win_count_MC_vs_M0": wins_mc_m0,
                        "foldwise_win_count_MPC_vs_MPall": wins_mpc_mpall,
                        "n_outer_folds": n_outer_folds,
                        "outer_group": "checkpoint",
                        "inner_group": "checkpoint_train_only",
                        "alpha_grid": "|".join(map(str, ALPHAS)),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(pred_rows)


def model_interactions(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = feature[feature["has_c_epsilon"] & feature["has_pk"] & (feature["checkpoint"] > 0)].copy()
    for domain, gd in df.groupby("domain"):
        for target in targets_for_domain(domain):
            if target not in SIGNED_TARGETS:
                continue
            for feat in ["c_epsilon"] + PK_COLS:
                dd = gd.dropna(subset=[target, feat]).copy()
                if dd["model"].nunique() < 2 or len(dd) < 16:
                    continue
                model_llama = (dd["model"] == "llama").astype(float).to_numpy()
                x = dd[feat].to_numpy(dtype=float)
                x_std = np.std(x)
                if x_std == 0:
                    continue
                x = (x - np.mean(x)) / x_std
                logt = np.log1p(dd["checkpoint"].to_numpy(dtype=float))
                arm_cols = [(dd["arm"].to_numpy() == arm).astype(float) for arm in ARMS]
                X = np.column_stack([np.ones(len(dd)), model_llama, logt, *arm_cols, x, x * model_llama])
                y = dd[target].to_numpy(dtype=float)
                beta = np.linalg.pinv(X) @ y
                pred = X @ beta
                rows.append(
                    {
                        "domain": domain,
                        "target": target,
                        "feature": feat,
                        "n": int(len(dd)),
                        "coef_feature_qwen_reference": float(beta[-2]),
                        "coef_model_llama_x_feature": float(beta[-1]),
                        "coef_feature_llama": float(beta[-2] + beta[-1]),
                        "pooled_r2": r2_score(y, pred),
                        "pooled_mae": mae_score(y, pred),
                        "protocol": "pooled_signed_target_linear_model_with_model_x_feature_interaction",
                    }
                )
    return pd.DataFrame(rows)


def write_handoff(
    manifest: dict,
    coverage: pd.DataFrame,
    corrs: pd.DataFrame,
    inc: pd.DataFrame,
    status: str,
) -> None:
    lines = []
    lines.append("# FAT-R1-v2 link handoff")
    lines.append("")
    lines.append(f"created_utc: {manifest['created_utc']}")
    lines.append(f"status: {status}")
    lines.append("schema: cycle09_fat_outlink_round1_v2_link")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- CPU-only reuse analysis; no model forward, rollout, training, or Eval rerun.")
    lines.append("- Joins FAT-R1-v2 regional NLL/KL cells to headline c_epsilon and deployed strict joint fixed-k p_k.")
    lines.append("- MMLU uses sample_macro only; Math uses FAT-R1-v2 token-closed boxed-answer protocol.")
    lines.append("- Qwen p_k step10 remains excluded as `QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION`.")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for item in manifest["outputs"]:
        lines.append(f"- `{Path(item['path']).name}`: rows={item['rows']} sha256={item['sha256']}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(coverage.to_markdown(index=False))
    lines.append("")
    lines.append("## Correlation Snapshot")
    lines.append("")
    snap = corrs[corrs["feature"].eq("c_epsilon")].head(20)
    lines.append(snap.to_markdown(index=False))
    lines.append("")
    lines.append("## Incremental Model Snapshot")
    lines.append("")
    lines.append(inc.head(24).to_markdown(index=False) if not inc.empty else "EMPTY")
    lines.append("")
    lines.append("## Branch Codes")
    lines.append("")
    lines.append(f"- final_status: {status}")
    lines.append("- formal_usable: feature matrix, contrast bootstrap CI, descriptive correlations, checkpoint-demeaned correlations, progress residual correlations, within-checkpoint rank, grouped held-out incremental models.")
    lines.append("- blocked: Qwen deployed p_k step10 only; excluded rather than imputed.")
    lines.append("- superseded: none generated by this script; FAT-R1-v1 remains blocked audit in its own directory.")
    (OUT / "fat_r1_v2_link_handoff.md").write_text("\n".join(lines) + "\n")


def write_code_evolution(manifest: dict, status: str) -> None:
    path = ROOT / "mypaper/code/code_evolution.md"
    text = (
        f"\n## {manifest['created_utc']} FAT-R1-v2 link return\n"
        f"- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_r1_v2_link.py`\n"
        f"- Output root: `{OUT}`\n"
        f"- Manifest: `{OUT / 'fat_r1_v2_link_manifest.json'}`\n"
        f"- Status: `{status}`\n"
        "- Scope: CPU-only reuse linking of FAT-R1-v2 regional output cells with headline `c_epsilon` and deployed merged strict-joint `p_k`; no forward/training/Eval rerun.\n"
        "- Fixed exclusions: Qwen `p_k` step10 kept missing and marked `QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION`.\n"
    )
    with path.open("a") as f:
        f.write(text)


def write_csv(df: pd.DataFrame, name: str, outputs: list[dict]) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    outputs.append({"path": str(path), "rows": int(len(df)), "sha256": sha256_file(path)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-code-evolution", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [
        FAT / "fat_r1_v2_mmlu_cells.csv",
        FAT / "fat_r1_v2_math_cells.csv",
        FAT / "fat_r1_v2_mmlu_samples.csv",
        FAT / "fat_r1_v2_math_samples.csv",
        FAT / "fat_r1_v2_region_contrasts.csv",
        FAT / "fat_r1_v2_bootstrap_ci.csv",
        FAT / "fat_r1_v2_manifest.json",
        MINI / "d10_5_a4_feature_matrix.csv",
        MINI / "T_PK_qwen3_4b_fixedk.csv",
        MINI / "d11_llama_merged_pk.csv",
    ]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    feature = build_feature_matrix()
    coverage = coverage_table(feature)
    samples = load_output_samples()
    contrast_ci = bootstrap_contrast_ci(samples)
    corrs = correlation_rows(feature)
    demean_cells, demean_corrs = checkpoint_demeaned(feature)
    progress_corrs = progress_residual_correlations(feature)
    rank = within_checkpoint_rank(feature)
    inc, preds = incremental_models(feature)
    interactions = model_interactions(feature)

    observed_c = int(feature["has_c_epsilon"].sum())
    observed_pk = int(feature["has_pk"].sum())
    unexpected_missing = feature[~feature["has_pk"] & ~feature["pk_exclusion_reason"].eq("QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION")]
    if observed_c != 120 or observed_pk != 112 or len(unexpected_missing):
        status = "FAILED_FAT_R1_LINK_JOIN_PARITY"
    else:
        status = "PARTIAL_FAT_R1_LINK_PK_STEP10_EXCLUDED"

    outputs: list[dict] = []
    task_status = pd.DataFrame(
        [
            {"task": "L0_join_parity", "status": "complete" if observed_c == 120 and observed_pk == 112 else "failed", "rows": len(feature)},
            {"task": "L1_region_contrast_bootstrap_ci", "status": "complete", "rows": len(contrast_ci)},
            {"task": "L2_within_arm_correlations", "status": "complete", "rows": len(corrs)},
            {"task": "L2_checkpoint_demeaned", "status": "complete", "rows": len(demean_corrs)},
            {"task": "L2_progress_residual", "status": "complete", "rows": len(progress_corrs)},
            {"task": "L2_within_checkpoint_rank", "status": "complete", "rows": len(rank)},
            {"task": "L3_grouped_heldout_incremental", "status": "complete", "rows": len(inc)},
            {"task": "L3_model_feature_interactions", "status": "complete", "rows": len(interactions)},
        ]
    )
    write_csv(task_status, "fat_r1_v2_link_task_status.csv", outputs)
    write_csv(feature, "fat_r1_v2_link_feature_matrix.csv", outputs)
    write_csv(coverage, "fat_r1_v2_link_coverage.csv", outputs)
    write_csv(contrast_ci, "fat_r1_v2_link_region_contrast_bootstrap_ci.csv", outputs)
    write_csv(corrs, "fat_r1_v2_link_within_arm_correlations.csv", outputs)
    write_csv(demean_corrs, "fat_r1_v2_link_checkpoint_demeaned_correlations.csv", outputs)
    write_csv(demean_cells, "fat_r1_v2_link_checkpoint_demeaned_cells.csv", outputs)
    write_csv(progress_corrs, "fat_r1_v2_link_progress_residual_correlations.csv", outputs)
    write_csv(rank, "fat_r1_v2_link_within_checkpoint_rank.csv", outputs)
    write_csv(inc, "fat_r1_v2_link_incremental_models.csv", outputs)
    preds_path = OUT / "fat_r1_v2_link_incremental_predictions.csv"
    preds.to_csv(preds_path, index=False)
    outputs.append({"path": str(preds_path), "rows": int(len(preds)), "sha256": sha256_file(preds_path)})
    write_csv(interactions, "fat_r1_v2_link_model_interactions.csv", outputs)

    manifest = {
        "created_utc": now_iso(),
        "schema": "cycle09_fat_outlink_round1_v2_link",
        "status": status,
        "git_rev": git_rev(),
        "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs],
        "outputs": outputs,
        "numeric_protocol": {
            "reuse_only": True,
            "no_forward": True,
            "no_training": True,
            "no_eval_rerun": True,
            "bootstrap_seed": 42,
            "bootstrap_draws": 1024,
            "ridge_alpha_grid": ALPHAS,
            "outer_group": "checkpoint",
            "inner_group": "checkpoint_train_only",
        },
        "coverage": coverage.to_dict(orient="records"),
    }
    write_handoff(manifest, coverage, corrs, inc, status)
    manifest["outputs"].append(
        {"path": str(OUT / "fat_r1_v2_link_handoff.md"), "rows": 1, "sha256": sha256_file(OUT / "fat_r1_v2_link_handoff.md")}
    )
    (OUT / "fat_r1_v2_link_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.skip_code_evolution:
        write_code_evolution(manifest, status)
    print(json.dumps({"status": status, "output_root": str(OUT), "observed_c_rows": observed_c, "observed_pk_rows": observed_pk}, indent=2))


if __name__ == "__main__":
    main()
