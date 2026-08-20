#!/usr/bin/env python3
"""FAT-R1-v2 equal-5 correction and related-work linkage.

CPU-only reuse task. It builds non-QK equal-5 feature matrices from existing
module artifacts, joins them to FAT-R1-v2 regional NLL/KL cells, reconstructs
deployed merged-minus-base p_k equal-5 baselines from per-module rows, and
writes handoff/manifest/figures without modifying legacy artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path("/root/LLM-output-density")
MINI = ROOT / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
FAT = MINI / "fat_outlink_round1_v2"
EQ5 = MINI / "equal5_non_qk"
OUT = MINI / "fat_outlink_round1_v2_link_equal5"
FIG = OUT / "figures"

M5 = [
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]
EXCLUDED = ["self_attn.q_proj", "self_attn.k_proj"]
ARMS = ["offkd", "opd", "seqkd", "sft"]
EPS_GRID = [0.01, 0.025, 0.05, 0.10]
ALPHAS = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
PK = ["p4_equal5", "p8_equal5", "p16_equal5", "p32_equal5"]

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
    return pearson(a[m].rank(method="average"), b[m].rank(method="average"))


def r2_score(y: Iterable[float], pred: Iterable[float]) -> float:
    y = np.asarray(list(y), dtype=float)
    p = np.asarray(list(pred), dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return float("nan")
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - float(np.sum((y - p) ** 2)) / ss_tot


def mae_score(y: Iterable[float], pred: Iterable[float]) -> float:
    y = np.asarray(list(y), dtype=float)
    p = np.asarray(list(pred), dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - p[m])))


def add_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["delta_nll_f_minus_a"] = out["delta_nll_f"] - out["delta_nll_a"]
    out["delta_nll_f_minus_p"] = out["delta_nll_f"] - out["delta_nll_p"]
    out["kl_f_minus_a"] = out["kl_f"] - out["kl_a"]
    out["delta_nll_b_minus_c"] = out["delta_nll_b"] - out["delta_nll_c"]
    out["delta_nll_b_minus_p"] = out["delta_nll_b"] - out["delta_nll_p"]
    return out


def targets_for_domain(domain: str) -> list[str]:
    return MMLU_TARGETS if domain == "mmlu" else MATH_TARGETS


def load_fat_cells() -> pd.DataFrame:
    mmlu = pd.read_csv(FAT / "fat_r1_v2_mmlu_cells.csv")
    mmlu = mmlu[mmlu["aggregation"].eq("sample_macro")].copy()
    math_df = pd.read_csv(FAT / "fat_r1_v2_math_cells.csv")
    df = pd.concat([mmlu, math_df], ignore_index=True, sort=False)
    df = add_contrasts(df)
    df["probe_name"] = np.where(df["domain"].eq("mmlu"), "E_mmluPro", "E_mathHeld")
    df["domain_display"] = df["probe_name"]
    df["domain_match_status"] = "domain_matched_not_item_matched"
    return df


def load_c5_all_eps() -> pd.DataFrame:
    df = pd.read_csv(EQ5 / "EQUAL5_functional_trajectories.csv")
    df = df[df["probe_name"].isin(["E_mmluPro", "E_mathHeld"])].copy()
    df = df[df["arm"].isin(ARMS)].copy()
    df = df[df["module_count"].eq(5) & df["excluded_modules_exactly_qk"].eq(True)]
    keep = [
        "model",
        "arm",
        "checkpoint",
        "probe_name",
        "epsilon",
        "layer",
        "module_count",
        "module_set",
        "included_modules",
        "excluded_modules",
        "source_rows_complete",
        "r_epsilon_equal5",
        "state_rank_base_equal5_non_qk",
        "state_rank_current_equal5_non_qk",
        "delta_r_equal5",
        "c_equal5",
        "ratio_of_means_sensitivity_equal5",
        "c_equal7",
        "equal5_minus_equal7_c",
    ]
    return df[keep].drop_duplicates(["model", "arm", "checkpoint", "probe_name", "epsilon"])


def pk5_from_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    module_rows = []
    specs = [
        ("qwen", MINI / "T_PK_qwen3_4b_fixedk.csv", 18),
        ("llama", MINI / "d11_llama_merged_pk.csv", 14),
    ]
    for model, path, layer in specs:
        df = pd.read_csv(path)
        step_col = "checkpoint" if "checkpoint" in df.columns else "step"
        df = df[(df["layer"].eq(layer)) & df["delta_construction"].eq("bf16_merged_minus_base")]
        df = df[df["module"].isin(M5)].copy()
        df["model"] = model
        df["checkpoint"] = df[step_col].astype(int)
        df["included_modules"] = ",".join(M5)
        df["excluded_modules"] = ",".join(EXCLUDED)
        df["aggregation"] = "equal_mean_of_per_module_scores"
        df["delta_source"] = "deployed_BF16_merged_minus_base"
        module_rows.append(df)
        for (arm, checkpoint, k), g in df.groupby(["arm", "checkpoint", "k"]):
            rows.append(
                {
                    "model": model,
                    "arm": arm,
                    "checkpoint": int(checkpoint),
                    "k": int(k),
                    "p_k_equal5": float(g["p_k"].mean()),
                    "module_count": int(g["module"].nunique()),
                    "included_modules": ",".join(sorted(g["module"].unique())),
                    "excluded_modules": ",".join(EXCLUDED),
                    "aggregation": "equal_mean_of_per_module_scores",
                    "delta_source": "deployed_BF16_merged_minus_base",
                }
            )
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["model", "arm", "checkpoint"], columns="k", values="p_k_equal5", aggfunc="first").reset_index()
    wide.columns = [f"p{c}_equal5" if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    meta = long.groupby(["model", "arm", "checkpoint"]).agg(
        pk_module_count=("module_count", "min"),
        pk_included_modules=("included_modules", "first"),
        pk_excluded_modules=("excluded_modules", "first"),
        pk_aggregation=("aggregation", "first"),
        pk_delta_source=("delta_source", "first"),
    ).reset_index()
    return wide.merge(meta, on=["model", "arm", "checkpoint"], how="left"), pd.concat(module_rows, ignore_index=True, sort=False)


def build_feature_matrix(all_eps: bool = False) -> pd.DataFrame:
    fat = load_fat_cells()
    c5 = load_c5_all_eps()
    if not all_eps:
        c5 = c5[np.isclose(c5["epsilon"], 0.05)].copy()
    pk5, _ = pk5_from_raw()
    df = fat[fat["arm"].isin(ARMS)].merge(
        c5,
        on=["model", "arm", "checkpoint", "probe_name"],
        how="left",
        validate="many_to_many" if all_eps else "one_to_one",
    )
    df = df.merge(pk5, on=["model", "arm", "checkpoint"], how="left", validate="many_to_one")
    df["has_c_equal5"] = df["c_equal5"].notna()
    df["has_pk_equal5"] = df[PK].notna().all(axis=1)
    df["pk_missing_reason"] = ""
    q10 = df["model"].eq("qwen") & df["checkpoint"].eq(10) & ~df["has_pk_equal5"]
    df.loc[q10, "pk_missing_reason"] = "QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION"
    df.loc[~df["has_pk_equal5"] & df["pk_missing_reason"].eq(""), "pk_missing_reason"] = "PK5_MISSING_UNEXPECTED"
    df["kl_b_minus_c"] = np.nan
    df["kl_b_minus_c_status"] = "NA_NO_KL_C_IN_FAT_R1_V2"
    return df


def coverage_audit(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected = {("qwen", "mmlu"): 36, ("qwen", "math"): 36, ("llama", "mmlu"): 24, ("llama", "math"): 24}
    for (model, domain), g in feature.groupby(["model", "domain"]):
        rows.append(
            {
                "model": model,
                "domain": domain,
                "expected_c5_rows": expected[(model, domain)],
                "observed_c5_rows": int(g["has_c_equal5"].sum()),
                "expected_pk5_rows": expected[(model, domain)] - (4 if model == "qwen" else 0),
                "observed_pk5_rows": int(g["has_pk_equal5"].sum()),
                "missing_pk5_cells": int((~g["has_pk_equal5"]).sum()),
                "missing_pk5_reason": "|".join(sorted(g.loc[~g["has_pk_equal5"], "pk_missing_reason"].dropna().unique())),
                "module_count": 5,
                "included_modules": ",".join(M5),
                "excluded_modules": ",".join(EXCLUDED),
                "domain_match_status": "domain_matched_not_item_matched",
            }
        )
    rows.append(
        {
            "model": "ALL",
            "domain": "ALL",
            "expected_c5_rows": 120,
            "observed_c5_rows": int(feature["has_c_equal5"].sum()),
            "expected_pk5_rows": 112,
            "observed_pk5_rows": int(feature["has_pk_equal5"].sum()),
            "missing_pk5_cells": int((~feature["has_pk_equal5"]).sum()),
            "missing_pk5_reason": "|".join(sorted(feature.loc[~feature["has_pk_equal5"], "pk_missing_reason"].dropna().unique())),
            "module_count": 5,
            "included_modules": ",".join(M5),
            "excluded_modules": ",".join(EXCLUDED),
            "domain_match_status": "domain_matched_not_item_matched",
        }
    )
    return pd.DataFrame(rows)


def correlation_tables(feature: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    eps_rows = []
    for (model, domain, arm), g0 in feature[feature["checkpoint"] > 0].groupby(["model", "domain", "arm"]):
        for target in targets_for_domain(domain):
            g = g0[np.isclose(g0["epsilon"], 0.05)].copy()
            rows.append(
                {
                    "model": model,
                    "domain": domain,
                    "arm": arm,
                    "feature": "c_equal5",
                    "target": target,
                    "epsilon": 0.05,
                    "n": int(g[["c_equal5", target]].dropna().shape[0]),
                    "pearson": pearson(g["c_equal5"], g[target]),
                    "spearman": spearman(g["c_equal5"], g[target]),
                    "analysis_role": "standalone_c_equal5_primary",
                }
            )
            gp = g[g["has_pk_equal5"]]
            for p in PK:
                rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "arm": arm,
                        "feature": p,
                        "target": target,
                        "epsilon": 0.05,
                        "n": int(gp[[p, target]].dropna().shape[0]),
                        "pearson": pearson(gp[p], gp[target]),
                        "spearman": spearman(gp[p], gp[target]),
                        "analysis_role": "pk_equal5_standalone_matched_112",
                    }
                )
            for eps, ge in g0.groupby("epsilon"):
                eps_rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "arm": arm,
                        "feature": "c_equal5",
                        "target": target,
                        "epsilon": float(eps),
                        "n": int(ge[["c_equal5", target]].dropna().shape[0]),
                        "pearson": pearson(ge["c_equal5"], ge[target]),
                        "spearman": spearman(ge["c_equal5"], ge[target]),
                    }
                )

    demean = []
    primary = feature[np.isclose(feature["epsilon"], 0.05) & (feature["checkpoint"] > 0)].copy()
    for (model, domain, checkpoint), g in primary.groupby(["model", "domain", "checkpoint"]):
        if g["arm"].nunique() != 4:
            continue
        for target in targets_for_domain(domain):
            for feat in ["c_equal5"] + PK:
                if feat != "c_equal5" and not g["has_pk_equal5"].all():
                    continue
                xm, ym = g[feat].mean(skipna=True), g[target].mean(skipna=True)
                for _, r in g.iterrows():
                    demean.append(
                        {
                            "model": model,
                            "domain": domain,
                            "checkpoint": int(checkpoint),
                            "arm": r["arm"],
                            "feature": feat,
                            "target": target,
                            "feature_demeaned": r[feat] - xm,
                            "target_demeaned": r[target] - ym,
                            "demean_unit": "model_domain_checkpoint_four_arm_mean",
                        }
                    )
    dcells = pd.DataFrame(demean)
    drows = []
    for (model, domain, feat, target), g in dcells.groupby(["model", "domain", "feature", "target"]):
        drows.append(
            {
                "model": model,
                "domain": domain,
                "feature": feat,
                "target": target,
                "n": int(g[["feature_demeaned", "target_demeaned"]].dropna().shape[0]),
                "pearson": pearson(g["feature_demeaned"], g["target_demeaned"]),
                "spearman": spearman(g["feature_demeaned"], g["target_demeaned"]),
                "demean_unit": "model_domain_checkpoint_four_arm_mean",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(eps_rows)


def residualize(values: pd.Series, checkpoints: pd.Series) -> np.ndarray:
    y = values.to_numpy(dtype=float)
    x = np.log1p(checkpoints.to_numpy(dtype=float))
    m = np.isfinite(y) & np.isfinite(x)
    out = np.full(len(y), np.nan)
    if m.sum() < 2:
        return out
    X = np.column_stack([np.ones(m.sum()), x[m]])
    beta = np.linalg.pinv(X) @ y[m]
    out[m] = y[m] - X @ beta
    return out


def progress_residual(feature: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = feature[np.isclose(feature["epsilon"], 0.05) & (feature["checkpoint"] > 0)].copy()
    for (model, domain, arm), g in df.groupby(["model", "domain", "arm"]):
        for target in targets_for_domain(domain):
            for feat in ["c_equal5"] + PK:
                gg = g[g["has_pk_equal5"]] if feat != "c_equal5" else g
                rx = residualize(gg[feat], gg["checkpoint"])
                ry = residualize(gg[target], gg["checkpoint"])
                rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "arm": arm,
                        "feature": feat,
                        "target": target,
                        "n": int((np.isfinite(rx) & np.isfinite(ry)).sum()),
                        "pearson": pearson(rx, ry),
                        "spearman": spearman(rx, ry),
                        "diagnostic_only": True,
                        "residual_model": "within_model_domain_arm_intercept_plus_log1p_checkpoint",
                    }
                )
    return pd.DataFrame(rows)


def design(df: pd.DataFrame, block: str) -> tuple[np.ndarray, list[str]]:
    mats = [np.ones((len(df), 1))]
    cols = ["intercept"]
    features: list[str]
    if block in ["C-only", "C-only-matched"]:
        features = ["c_equal5"]
    elif block.startswith("Pk") and block.endswith("-only") and block != "PkAll-only":
        k = block.replace("Pk", "").replace("-only", "")
        features = [f"p{k}_equal5"]
    elif block == "PkAll-only":
        features = PK
    else:
        for arm in ARMS:
            cols.append(f"arm_{arm}")
            mats.append((df["arm"].to_numpy() == arm).astype(float).reshape(-1, 1))
        cols.append("log1p_checkpoint")
        mats.append(np.log1p(df["checkpoint"].to_numpy(dtype=float)).reshape(-1, 1))
        if block == "M0":
            features = []
        elif block == "M0+C5":
            features = ["c_equal5"]
        elif block == "M0+PkAll5":
            features = PK
        elif block == "M0+PkAll5+C5":
            features = PK + ["c_equal5"]
        else:
            raise ValueError(block)
    for f in features:
        cols.append(f)
        mats.append(df[f].to_numpy(dtype=float).reshape(-1, 1))
    return np.hstack(mats), cols


def standardize_train(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    mean[0] = 0
    std[0] = 1
    std[~np.isfinite(std) | (std == 0)] = 1
    mean[~np.isfinite(mean)] = 0
    return mean, std


def ridge_predict(train: pd.DataFrame, test: pd.DataFrame, target: str, block: str, alpha: float) -> tuple[np.ndarray, list[str]]:
    Xtr, cols = design(train, block)
    Xte, _ = design(test, block)
    mean, std = standardize_train(Xtr)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    y = train[target].to_numpy(dtype=float)
    reg = np.eye(Xtr.shape[1]) * alpha
    reg[0, 0] = 0
    beta = np.linalg.pinv(Xtr.T @ Xtr + reg) @ Xtr.T @ y
    return Xte @ beta, cols


def features_for_block(block: str) -> list[str]:
    if block in ["C-only", "C-only-matched"]:
        return ["c_equal5"]
    if block in ["Pk4-only", "Pk8-only", "Pk16-only", "Pk32-only"]:
        return [f"p{block[2:].replace('-only','')}_equal5"]
    if block == "PkAll-only":
        return PK
    if block == "M0":
        return []
    if block == "M0+C5":
        return ["c_equal5"]
    if block == "M0+PkAll5":
        return PK
    if block == "M0+PkAll5+C5":
        return PK + ["c_equal5"]
    raise ValueError(block)


def choose_alpha(train: pd.DataFrame, target: str, block: str) -> float:
    scores = []
    for alpha in ALPHAS:
        fold_mae = []
        for ck in sorted(train["checkpoint"].unique()):
            tr = train[train["checkpoint"] != ck]
            va = train[train["checkpoint"] == ck]
            if len(tr) == 0 or len(va) == 0:
                continue
            pred, _ = ridge_predict(tr, va, target, block, alpha)
            fold_mae.append(mae_score(va[target], pred))
        scores.append((np.nanmean(fold_mae) if fold_mae else np.inf, alpha))
    return float(sorted(scores, key=lambda x: (x[0], x[1]))[0][1])


def grouped_models(feature: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = feature[np.isclose(feature["epsilon"], 0.05) & (feature["checkpoint"] > 0)].copy()
    blocks_standalone = [
        "C-only",
        "C-only-matched",
        "Pk4-only",
        "Pk8-only",
        "Pk16-only",
        "Pk32-only",
        "PkAll-only",
    ]
    blocks_inc = ["M0", "M0+C5", "M0+PkAll5", "M0+PkAll5+C5"]
    pred_rows, fold_rows, metric_rows = [], [], []
    pid = 0
    for (model, domain), gd0 in primary.groupby(["model", "domain"]):
        for target in targets_for_domain(domain):
            for block in blocks_standalone + blocks_inc:
                needed = [target] + features_for_block(block)
                gd = gd0.dropna(subset=needed).copy()
                # The complete C-only track retains every checkpoint. All fair
                # C-vs-p and nuisance comparisons use the p_k-matched grid.
                if block != "C-only":
                    gd = gd[gd["has_pk_equal5"]]
                ckpts = sorted(gd["checkpoint"].unique())
                if len(ckpts) < 3:
                    continue
                y_all, p_all = [], []
                for ck in ckpts:
                    train = gd[gd["checkpoint"] != ck]
                    test = gd[gd["checkpoint"] == ck]
                    alpha = choose_alpha(train, target, block)
                    pred, cols = ridge_predict(train, test, target, block, alpha)
                    y = test[target].to_numpy(dtype=float)
                    y_all.extend(y.tolist())
                    p_all.extend(pred.tolist())
                    fold_rows.append(
                        {
                            "model": model,
                            "domain": domain,
                            "target": target,
                            "feature_block": block,
                            "heldout_checkpoint": int(ck),
                            "fold_r2": r2_score(y, pred),
                            "fold_mae": mae_score(y, pred),
                            "fold_spearman": spearman(y, pred),
                            "selected_alpha": alpha,
                            "n_test_rows": len(test),
                            "feature_columns": "|".join(cols),
                        }
                    )
                    for i, (_, r) in enumerate(test.iterrows()):
                        pred_rows.append(
                            {
                                "prediction_id": pid,
                                "model": model,
                                "domain": domain,
                                "target": target,
                                "feature_block": block,
                                "heldout_checkpoint": int(ck),
                                "arm": r["arm"],
                                "checkpoint": int(r["checkpoint"]),
                                "y_true": float(y[i]),
                                "y_pred": float(pred[i]),
                                "selected_alpha": alpha,
                            }
                        )
                        pid += 1
                metric_rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "target": target,
                        "feature_block": block,
                        "OOF_R2": r2_score(y_all, p_all),
                        "OOF_MAE": mae_score(y_all, p_all),
                        "OOF_prediction_spearman": spearman(y_all, p_all),
                        "n_checkpoint_groups": len(ckpts),
                        "n_state_rows": len(gd),
                        "analysis_role": "standalone" if block in blocks_standalone else "nuisance_or_incremental_secondary",
                        "outer_split": "leave_one_checkpoint_group_out",
                        "standardization": "train_fold_only",
                        "alpha_selection": "inner_checkpoint_cv_train_only",
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    folds = pd.DataFrame(fold_rows)
    preds = pd.DataFrame(pred_rows)
    win_rows = []
    if not folds.empty:
        for (model, domain, target), g in folds.groupby(["model", "domain", "target"]):
            for cand, base in [
                ("M0+C5", "M0"),
                ("M0+PkAll5+C5", "M0+PkAll5"),
                ("C-only-matched", "PkAll-only"),
            ]:
                c = g[g["feature_block"].eq(cand)].set_index("heldout_checkpoint")
                b = g[g["feature_block"].eq(base)].set_index("heldout_checkpoint")
                common = sorted(set(c.index) & set(b.index))
                win_rows.append(
                    {
                        "model": model,
                        "domain": domain,
                        "target": target,
                        "candidate": cand,
                        "baseline": base,
                        "foldwise_wins_by_MAE": int(sum(c.loc[x, "fold_mae"] < b.loc[x, "fold_mae"] for x in common)),
                        "n_common_folds": len(common),
                    }
                )
    return metrics, preds, pd.concat([folds, pd.DataFrame(win_rows)], ignore_index=True, sort=False)


def behavior_join(feature: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = feature[np.isclose(feature["epsilon"], 0.05) & feature["domain"].eq("mmlu") & (feature["checkpoint"] > 0)].copy()
    qflex = pd.read_csv(MINI / "S1_mmlupro_flexible.csv")
    qaudit = pd.read_csv(MINI / "S1_mmlupro_extract_audit.csv")
    q = qflex.merge(qaudit, on=["arm", "step", "n"], how="left", suffixes=("", "_audit"))
    q["model"] = "qwen"
    q["checkpoint"] = q["step"].astype(int)
    q["strict_accuracy"] = q["exact_match"]
    q["flexible_accuracy"] = q["mmlu_pro_flexible"]
    q["format_gap"] = q["flexible_accuracy"] - q["strict_accuracy"]
    q["extract_failure_rate"] = q["strict_extract_fail_rate"]
    q["recovery_rate"] = q["strict_fail_recovery_rate"]
    q["bad_format_count"] = q["failure_letter_bad_format_n"]
    q["bad_format_rate"] = q["failure_letter_bad_format_rate_all_samples"]
    q["truncated_count"] = q["failure_truncated_n"]
    q["truncated_rate"] = q["failure_truncated_rate_all_samples"]
    q["no_letter_count"] = q["failure_no_uppercase_standalone_A_to_J_n"]
    q["no_letter_rate"] = q["failure_no_uppercase_standalone_A_to_J_rate_all_samples"]
    q["behavior_source"] = "S1_mmlupro_flexible+S1_mmlupro_extract_audit"

    l = pd.read_csv(MINI / "llama_early_320_behavior.csv")
    l = l[l["task"].eq("mmlu_pro")].copy()
    l["model"] = "llama"
    l["checkpoint"] = l["step"].astype(int)
    l["format_gap"] = l["flexible_accuracy"] - l["strict_accuracy"]
    l["recovery_rate"] = np.nan
    for col in ["bad_format_count", "bad_format_rate", "truncated_count", "truncated_rate", "no_letter_count", "no_letter_rate"]:
        l[col] = np.nan
    l["behavior_source"] = "llama_early_320_behavior task=mmlu_pro"
    cols = [
        "model",
        "arm",
        "checkpoint",
        "strict_accuracy",
        "flexible_accuracy",
        "format_gap",
        "extract_failure_rate",
        "recovery_rate",
        "bad_format_count",
        "bad_format_rate",
        "truncated_count",
        "truncated_rate",
        "no_letter_count",
        "no_letter_rate",
        "behavior_source",
    ]
    beh = pd.concat([q[cols], l[cols]], ignore_index=True, sort=False)
    out = f.merge(beh, on=["model", "arm", "checkpoint"], how="left", validate="one_to_one")
    table = []
    for (model, arm), g in out.groupby(["model", "arm"]):
        table.append(
            {
                "model": model,
                "arm": arm,
                "n": len(g),
                "spearman_F_minus_A_vs_format_gap": spearman(g["delta_nll_f_minus_a"], g["format_gap"]),
                "spearman_delta_nll_f_vs_extract_failure": spearman(g["delta_nll_f"], g["extract_failure_rate"]),
                "spearman_c_equal5_vs_format_gap": spearman(g["c_equal5"], g["format_gap"]),
                "pooled_forbidden_note": "model_separate_only_signed_direction_not_pooled",
            }
        )
    return out, pd.DataFrame(table)


def related_work_metrics(feature: pd.DataFrame, pk_module: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm, checkpoint), g in pk_module.groupby(["model", "arm", "checkpoint"]):
        rows.append(
            {
                "row_type": "pk5_rebuilt_from_raw_modules",
                "model": model,
                "arm": arm,
                "checkpoint": int(checkpoint),
                "module_count": int(g["module"].nunique()),
                "included_modules": ",".join(sorted(g["module"].unique())),
                "excluded_modules": ",".join(EXCLUDED),
                "aggregation": "equal_mean_of_per_module_scores",
                "delta_source": "deployed_BF16_merged_minus_base",
                "k_values": ",".join(map(str, sorted(g["k"].unique()))),
                "source_status": "formal_available",
            }
        )
    # Add standalone metric summary rows so this file is useful without opening grouped outputs.
    for f in ["c_equal5"] + PK:
        rows.append(
            {
                "row_type": "feature_coverage",
                "model": "ALL",
                "arm": "ALL",
                "checkpoint": -1,
                "module_count": 5,
                "included_modules": ",".join(M5),
                "excluded_modules": ",".join(EXCLUDED),
                "aggregation": "feature_matrix_nonmissing_rows",
                "delta_source": "mixed_c5_state_or_deployed_pk5",
                "k_values": f,
                "source_status": int(feature[f].notna().sum()) if f in feature else np.nan,
            }
        )
    return pd.DataFrame(rows)


def equal5_vs_equal7() -> pd.DataFrame:
    src = pd.read_csv(EQ5 / "EQUAL5_equal7_paired_comparison.csv")
    return src.copy()


def doc_correction() -> tuple[dict, str]:
    p = FAT / "fat_r1_v2_mask_audit.csv"
    logical = len(pd.read_csv(p))
    with p.open("rb") as f:
        physical = sum(1 for _ in f) - 1
    data = {
        "created_utc": now_iso(),
        "artifact": str(p),
        "logical_record_count": logical,
        "physical_line_count": physical,
        "reason": "CSV quoted token text contains embedded newlines; physical wc-style line count is not logical CSV row count.",
        "stored_logits_description": {
            "MMLU": "full-vocabulary KL on F/A/T; scalar NLL also available on P/F/A/T",
            "MATH": "full-vocabulary KL on B/T; scalar NLL also available on P/C/B/T",
        },
        "naming_alias": {
            "P": "true prompt in FAT-R1-v2",
            "R": "legacy fixed reference-token stream formerly called P in older artifacts",
        },
        "fat_r1_v1_status": "blocked because character-defined boxed spans crossed token boundaries",
        "fat_r1_v2_status": "formal result using token-clean combined boxed spans",
    }
    md = "\n".join(
        [
            "# FAT-R1-v2 documentation correction",
            "",
            f"created_utc: {data['created_utc']}",
            "",
            f"- logical_record_count: {logical}",
            f"- physical_line_count: {physical}",
            "- reason: CSV quoted token text contains embedded newlines; use logical CSV records for row counts.",
            "- MMLU stored logits/NLL: full-vocabulary KL on F/A/T; scalar NLL on P/F/A/T.",
            "- MATH stored logits/NLL: full-vocabulary KL on B/T; scalar NLL on P/C/B/T.",
            "- naming: FAT-R1-v2 `P` is true prompt; legacy fixed reference-token stream is `R`.",
            "- FAT-R1-v1 is blocked; FAT-R1-v2 is formal.",
            "",
        ]
    )
    return data, md


def make_figures(feature: pd.DataFrame, metrics: pd.DataFrame, behavior: pd.DataFrame, eq7: pd.DataFrame) -> list[Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = []
    primary = feature[np.isclose(feature["epsilon"], 0.05)].copy()
    for model in sorted(primary["model"].unique()):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
        for ax, domain in zip(axes, ["mmlu", "math"]):
            gd = primary[(primary["model"] == model) & (primary["domain"] == domain)]
            for arm, g in gd.groupby("arm"):
                g = g.sort_values("checkpoint")
                ax.plot(g["checkpoint"], g["c_equal5"], marker="o", label=arm)
            ax.set_title(f"{model} {domain} c_equal5")
            ax.set_xlabel("checkpoint")
            ax.set_ylabel("c_equal5")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        path = FIG / f"figure1_equal5_trajectory_{model}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    # Keep model, domain, and target visible. Averaging OOF R2 over all of them
    # hides the cross-model/domain boundary that this analysis is meant to show.
    fig, axes = plt.subplots(2, 2, figsize=(19, 14), squeeze=False)
    standalone = metrics[
        metrics["feature_block"].isin(
            [
                "C-only-matched",
                "Pk4-only",
                "Pk8-only",
                "Pk16-only",
                "Pk32-only",
                "PkAll-only",
            ]
        )
    ].copy()
    for row, model in enumerate(["llama", "qwen"]):
        for col, domain in enumerate(["mmlu", "math"]):
            ax = axes[row, col]
            gd = standalone[(standalone["model"] == model) & (standalone["domain"] == domain)]
            targets = list(dict.fromkeys(gd["target"]))
            y = np.arange(len(targets), dtype=float)
            c_values = []
            best_scalar_values = []
            pkall_values = []
            for target in targets:
                gt = gd[gd["target"].eq(target)].set_index("feature_block")
                c_values.append(gt.loc["C-only-matched", "OOF_R2"])
                best_scalar_values.append(
                    gt.loc[["Pk4-only", "Pk8-only", "Pk16-only", "Pk32-only"], "OOF_R2"].max()
                )
                pkall_values.append(gt.loc["PkAll-only", "OOF_R2"])
            width = 0.25
            ax.barh(y - width, c_values, height=width, label=r"$c_\varepsilon^{(5)}$ only")
            ax.barh(
                y,
                best_scalar_values,
                height=width,
                label=r"best scalar $p_k^{(5)}$ (oracle over $k$)",
            )
            ax.barh(y + width, pkall_values, height=width, label=r"$p_{4,8,16,32}^{(5)}$ block")
            ax.axvline(0.0, color="black", linewidth=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(targets, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("leave-one-checkpoint-group-out OOF $R^2$")
            ax.set_title(
                f"{model.upper()} — {domain.upper()} "
                f"(matched n={int(gd['n_state_rows'].min())} states)"
            )
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Figure 2: standalone equal-5 functional compression versus matched weight-space baselines",
        fontsize=15,
    )
    fig.tight_layout()
    path = FIG / "figure2_equal5_grouped_models.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(19, 14), squeeze=False)
    incremental = metrics[
        metrics["feature_block"].isin(["M0", "M0+C5", "M0+PkAll5", "M0+PkAll5+C5"])
    ].copy()
    incremental_order = ["M0", "M0+C5", "M0+PkAll5", "M0+PkAll5+C5"]
    incremental_labels = {
        "M0": r"arm + $\log(1+t)$",
        "M0+C5": r"M0 + $c_\varepsilon^{(5)}$",
        "M0+PkAll5": r"M0 + $p_{4,8,16,32}^{(5)}$",
        "M0+PkAll5+C5": r"M0 + $p_{4,8,16,32}^{(5)}$ + $c_\varepsilon^{(5)}$",
    }
    for row, model in enumerate(["llama", "qwen"]):
        for col, domain in enumerate(["mmlu", "math"]):
            ax = axes[row, col]
            gd = incremental[(incremental["model"] == model) & (incremental["domain"] == domain)]
            targets = list(dict.fromkeys(gd["target"]))
            y = np.arange(len(targets), dtype=float)
            width = 0.19
            offsets = np.linspace(-1.5 * width, 1.5 * width, len(incremental_order))
            for offset, block in zip(offsets, incremental_order):
                gt = gd[gd["feature_block"].eq(block)].set_index("target")
                ax.barh(
                    y + offset,
                    [gt.loc[target, "OOF_R2"] for target in targets],
                    height=width,
                    label=incremental_labels[block],
                )
            ax.axvline(0.0, color="black", linewidth=0.7)
            ax.set_yticks(y)
            ax.set_yticklabels(targets, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("leave-one-checkpoint-group-out OOF $R^2$")
            ax.set_title(f"{model.upper()} — {domain.upper()} (n={int(gd['n_state_rows'].max())} states)")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Figure 2b: equal-5 incremental information after arm/progress and weight-space controls",
        fontsize=15,
    )
    fig.tight_layout()
    path = FIG / "figure2b_equal5_incremental_models.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # Regional signed NLL and realized behavior are shown together but never
    # pooled across model families. Arm is encoded by color; F/A/T by linestyle.
    fig, axes = plt.subplots(4, 2, figsize=(16, 17), sharex="col", squeeze=False)
    arm_colors = dict(zip(ARMS, plt.get_cmap("tab10").colors[: len(ARMS)]))
    signal_styles = {
        "delta_nll_f": ("F", "-"),
        "delta_nll_a": ("A", "--"),
        "delta_nll_t": ("T", ":"),
    }
    for col, model in enumerate(["llama", "qwen"]):
        gm = behavior[behavior["model"].eq(model)]
        ax = axes[0, col]
        for arm in ARMS:
            g = gm[gm["arm"].eq(arm)].sort_values("checkpoint")
            for signal, (_, linestyle) in signal_styles.items():
                ax.plot(
                    g["checkpoint"],
                    g[signal],
                    color=arm_colors[arm],
                    linestyle=linestyle,
                    marker="o",
                    markersize=3,
                )
        ax.axhline(0.0, color="black", linewidth=0.7)
        ax.set_title(f"{model.upper()}: regional signed NLL")
        ax.set_ylabel(r"$\Delta$NLL")

        for row, (signal, ylabel, title) in enumerate(
            [
                ("delta_nll_f_minus_a", r"$\Delta$NLL$_F-\Delta$NLL$_A$", "format–answer contrast"),
                ("format_gap", "flexible − strict", "realized format gap"),
                ("extract_failure_rate", "failure rate", "strict extraction failure"),
            ],
            start=1,
        ):
            ax = axes[row, col]
            for arm in ARMS:
                g = gm[gm["arm"].eq(arm)].sort_values("checkpoint")
                ax.plot(
                    g["checkpoint"],
                    g[signal],
                    color=arm_colors[arm],
                    marker="o",
                    markersize=4,
                    label=arm,
                )
            if row == 1:
                ax.axhline(0.0, color="black", linewidth=0.7)
            ax.set_title(f"{model.upper()}: {title}")
            ax.set_ylabel(ylabel)
            if row == 3:
                ax.set_xlabel("checkpoint")
    arm_handles = [
        Line2D([0], [0], color=arm_colors[arm], marker="o", label=arm) for arm in ARMS
    ]
    signal_handles = [
        Line2D([0], [0], color="black", linestyle=linestyle, label=label)
        for label, linestyle in signal_styles.values()
    ]
    axes[0, 0].legend(
        handles=arm_handles + signal_handles,
        fontsize=8,
        ncol=2,
        loc="best",
        title="arm / regional signal",
    )
    axes[1, 0].legend(handles=arm_handles, fontsize=8, ncol=2, loc="best")
    fig.suptitle(
        "Figure 3: MMLU regional output drift and sequence-level format realization",
        fontsize=15,
    )
    fig.tight_layout()
    path = FIG / "figure3_format_gap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    sub = eq7[eq7["comparison_type"].eq("functional_cell")].dropna(subset=["equal5_value", "equal7_value"])
    ax.scatter(sub["equal7_value"], sub["equal5_value"], s=8, alpha=0.5)
    ax.set_xlabel("equal7 sensitivity")
    ax.set_ylabel("equal5 headline")
    ax.set_title("Figure 4: equal5 vs equal7 paired sensitivity")
    fig.tight_layout()
    path = FIG / "figure4_equal5_vs_equal7.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_csv(df: pd.DataFrame, name: str, outputs: list[dict]) -> None:
    path = OUT / name
    df.to_csv(path, index=False)
    outputs.append({"path": str(path), "rows": int(len(df)), "sha256": sha256_file(path)})


def write_handoff(manifest: dict, coverage: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# FAT-R1-v2 equal-5 handoff",
        "",
        f"created_utc: {manifest['created_utc']}",
        f"status: {manifest['status']}",
        "",
        "## Scope",
        "",
        "- Formal headline module aggregation is non-QK equal-5: v/o/gate/up/down.",
        "- q_proj and k_proj are excluded. This is measurement-side module exclusion, not adapter ablation.",
        "- CPU-only reuse: no training, forward, rollout, behavior Eval, or new SVD.",
        "- FAT domains are domain-matched, not item-matched: MMLU-Pro -> E_mmluPro, MATH500 -> E_mathHeld.",
        "- Qwen step10 p_k remains excluded from matched C-vs-p analyses; no interpolation or adapter-BA substitute.",
        "",
        "## Coverage",
        "",
        coverage.to_markdown(index=False),
        "",
        "## Outputs",
        "",
    ]
    for o in manifest["outputs"]:
        lines.append(f"- `{Path(o['path']).name}`: rows={o['rows']} sha256={o['sha256']}")
    lines.extend(
        [
            "",
            "## Grouped Model Snapshot",
            "",
            metrics.head(30).to_markdown(index=False),
            "",
            "## Branch Codes",
            "",
            "- formal_usable: equal5 feature matrix, standalone correlations, checkpoint-demeaned correlations, epsilon sensitivity, grouped held-out models, canonical behavior join, related-work pk5 rebuild, equal5/equal7 sensitivity.",
            "- auxiliary: progress residual diagnostic only; equal-7 sensitivity.",
            "- blocked: optional Qwen step10 pk backfill deferred; main matched pk analysis remains 112 states.",
            "- superseded: previous equal-7 headline interpretation is sensitivity only for this FAT related-work link.",
            "",
        ]
    )
    (OUT / "fat_r1_v2_equal5_handoff.md").write_text("\n".join(lines))


def append_returns(manifest: dict) -> None:
    text = (
        f"\n## {manifest['created_utc']} FAT-R1-v2 equal-5 correction return\n"
        f"- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_r1_v2_equal5_final.py`\n"
        f"- Output root: `{OUT}`\n"
        f"- Manifest: `{OUT / 'fat_r1_v2_equal5_manifest.json'}`\n"
        f"- Status: `{manifest['status']}`\n"
        "- Boundary: CPU-only reuse; no training/forward/rollout/Eval/new SVD.\n"
        "- Headline aggregation: non-QK equal-5 modules `v/o/gate/up/down`; q/k excluded; equal-7 retained as sensitivity.\n"
        "- Qwen step10 `p_k` remains excluded, not imputed.\n"
    )
    with (ROOT / "mypaper/code/code_evolution.md").open("a") as f:
        f.write(text)
    rh = ROOT / "mypaper/code/cycle09_reviewer_robustness_handoff.md"
    with rh.open("a") as f:
        f.write("\n\n" + text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-append", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    inputs = [
        EQ5 / "EQUAL5_functional_trajectories.csv",
        FAT / "fat_r1_v2_mmlu_cells.csv",
        FAT / "fat_r1_v2_math_cells.csv",
        MINI / "T_PK_qwen3_4b_fixedk.csv",
        MINI / "d11_llama_merged_pk.csv",
        MINI / "S1_mmlupro_flexible.csv",
        MINI / "S1_mmlupro_extract_audit.csv",
        MINI / "llama_early_320_behavior.csv",
    ]
    missing = [str(p) for p in inputs if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)

    feature = build_feature_matrix(all_eps=False)
    feature_all_eps = build_feature_matrix(all_eps=True)
    coverage = coverage_audit(feature)
    observed_c = int(feature["has_c_equal5"].sum())
    observed_pk = int(feature["has_pk_equal5"].sum())
    unexpected_pk_missing = feature[~feature["has_pk_equal5"] & ~feature["pk_missing_reason"].eq("QWEN_PK_STEP10_MISSING_MATCHED_EXCLUSION")]
    if observed_c != 120 or observed_pk != 112 or len(unexpected_pk_missing):
        status = "FAILED_EQUAL5_COVERAGE_OR_PK_JOIN"
    else:
        status = "COMPLETE_EQUAL5_WITH_QWEN_PK_STEP10_EXCLUDED"

    corrs, demean, epssens = correlation_tables(feature_all_eps)
    prog = progress_residual(feature)
    metrics, preds, folds = grouped_models(feature)
    standalone = metrics[metrics["analysis_role"].eq("standalone")].copy()
    incremental = metrics[metrics["analysis_role"].ne("standalone")].copy()
    pk5, pk_module = pk5_from_raw()
    related = related_work_metrics(feature, pk_module)
    eq7 = equal5_vs_equal7()
    behavior, format_table = behavior_join(feature)
    doc_json, doc_md = doc_correction()
    fig_paths = make_figures(feature, metrics, behavior, eq7)

    outputs: list[dict] = []
    task_status = pd.DataFrame(
        [
            {"task": "H10_equal5_feature_matrix", "status": "complete" if observed_c == 120 else "failed", "rows": len(feature)},
            {"task": "H10_standalone_correlations", "status": "complete", "rows": len(corrs)},
            {"task": "H10_checkpoint_demeaned", "status": "complete", "rows": len(demean)},
            {"task": "H10_progress_residual_diagnostic", "status": "complete", "rows": len(prog)},
            {"task": "H10_epsilon_sensitivity", "status": "complete", "rows": len(epssens)},
            {"task": "H11_grouped_heldout", "status": "complete", "rows": len(metrics)},
            {"task": "H4_related_work_pk5_rebuild", "status": "complete" if observed_pk == 112 else "partial", "rows": len(related)},
            {"task": "canonical_behavior_join", "status": "complete", "rows": len(behavior)},
            {"task": "figures", "status": "complete", "rows": len(fig_paths)},
            {"task": "fat_r1_v2_documentation_correction", "status": "complete", "rows": 2},
            {"task": "optional_qwen_step10_pk", "status": "OPTIONAL_DEFERRED", "rows": 0},
        ]
    )
    write_csv(task_status, "equal5_task_status.csv", outputs)
    write_csv(feature, "equal5_feature_matrix.csv", outputs)
    write_csv(coverage, "equal5_coverage_audit.csv", outputs)
    write_csv(corrs, "equal5_standalone_correlations.csv", outputs)
    write_csv(demean, "equal5_checkpoint_demeaned_correlations.csv", outputs)
    write_csv(prog, "equal5_progress_residual_diagnostic.csv", outputs)
    write_csv(epssens, "equal5_epsilon_sensitivity.csv", outputs)
    write_csv(standalone, "equal5_standalone_grouped_models.csv", outputs)
    write_csv(incremental, "equal5_incremental_grouped_models.csv", outputs)
    pred_path = OUT / "equal5_grouped_predictions.parquet"
    preds.to_parquet(pred_path, index=False)
    outputs.append({"path": str(pred_path), "rows": int(len(preds)), "sha256": sha256_file(pred_path)})
    write_csv(folds, "equal5_foldwise_results.csv", outputs)
    write_csv(related, "equal5_related_work_metrics.csv", outputs)
    write_csv(eq7, "equal5_vs_equal7_paired.csv", outputs)
    write_csv(behavior, "equal5_behavior_join.csv", outputs)
    write_csv(format_table, "equal5_format_realization_table.csv", outputs)
    doc_json_path = OUT / "fat_r1_v2_documentation_correction.json"
    doc_json_path.write_text(json.dumps(doc_json, indent=2, sort_keys=True) + "\n")
    outputs.append({"path": str(doc_json_path), "rows": 1, "sha256": sha256_file(doc_json_path)})
    doc_md_path = OUT / "fat_r1_v2_documentation_correction.md"
    doc_md_path.write_text(doc_md)
    outputs.append({"path": str(doc_md_path), "rows": 1, "sha256": sha256_file(doc_md_path)})
    for p in fig_paths:
        outputs.append({"path": str(p), "rows": 1, "sha256": sha256_file(p)})

    manifest = {
        "created_utc": now_iso(),
        "schema": "cycle09_fat_r1_v2_equal5_correction",
        "status": status,
        "git_rev": git_rev(),
        "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs],
        "outputs": outputs,
        "protocol": {
            "included_modules": M5,
            "excluded_modules": EXCLUDED,
            "module_count": 5,
            "c_aggregation": "per_module_base_current_contraction_then_equal_mean",
            "pk_aggregation": "equal_mean_of_per_module_scores",
            "main_epsilon": 0.05,
            "epsilon_sensitivity": EPS_GRID,
            "step0_excluded_from_correlations_and_cv": True,
            "mmlu_aggregation": "sample_macro",
            "domain_match_status": "domain_matched_not_item_matched",
            "no_training_forward_rollout_eval_or_new_svd": True,
            "progress_residual_diagnostic_only": True,
            "qwen_step10_pk_status": "OPTIONAL_DEFERRED_not_imputed",
        },
        "coverage": coverage.to_dict(orient="records"),
    }
    write_handoff(manifest, coverage, metrics)
    outputs.append({"path": str(OUT / "fat_r1_v2_equal5_handoff.md"), "rows": 1, "sha256": sha256_file(OUT / "fat_r1_v2_equal5_handoff.md")})
    manifest["outputs"] = outputs
    manifest_path = OUT / "fat_r1_v2_equal5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if not args.skip_append:
        append_returns(manifest)
    print(json.dumps({"status": status, "output_root": str(OUT), "c5_rows": observed_c, "pk5_rows": observed_pk}, indent=2))


if __name__ == "__main__":
    main()
