#!/usr/bin/env python3
"""Finalize D10.5/A2/A4 CPU tables without mixing legacy Llama rows.

This script consumes the completed D10 matched-numeric Llama state/output
artifacts, the existing Qwen D4 merged-state artifacts, and existing weight
baseline tables. It writes D10-specific downstream tables instead of overwriting
the historical D5/D7 outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path("/root/LLM-output-density")
ROOT = Path("/root/autodl-tmp/cycle09_relative_functional_contraction")
D10 = ROOT / "d10_llama_numeric_parity/formal/final"
FINAL = ROOT / "final"
STAGE3 = Path("/root/autodl-tmp/cycle09_stage3_followup")
STAGE4 = Path("/root/autodl-tmp/cycle09_stage4_state_displacement")
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"

ARMS = ("opd", "sft", "offkd", "seqkd")
CORE_PROBES = ("E_general", "E_math", "E_ood", "E_if")
SHARED_PROBES = ("E_general", "E_ood", "E_if")
TARGETS = (
    "cumulative_kl_base_to_current",
    "delta_nll_cumulative",
    "absolute_delta_nll_cumulative",
)
RNG = np.random.default_rng(20260727)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def mirror(path: Path) -> Path:
    target = MINI / path.name
    if path.suffix == ".csv":
        pd.read_csv(path).to_csv(target, index=False)
    else:
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def rank_corr(x: pd.Series, y: pd.Series) -> float:
    local = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 3:
        return float("nan")
    return float(local.iloc[:, 0].rank(method="average").corr(local.iloc[:, 1].rank(method="average")))


def kendall_tau_b(x: pd.Series, y: pd.Series) -> float:
    local = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 3:
        return float("nan")
    xs = local.iloc[:, 0].to_numpy()
    ys = local.iloc[:, 1].to_numpy()
    concordant = discordant = tie_x = tie_y = 0
    for i in range(len(xs) - 1):
        dx = xs[i + 1 :] - xs[i]
        dy = ys[i + 1 :] - ys[i]
        sx = np.sign(dx)
        sy = np.sign(dy)
        concordant += int(np.sum((sx * sy) > 0))
        discordant += int(np.sum((sx * sy) < 0))
        tie_x += int(np.sum((sx == 0) & (sy != 0)))
        tie_y += int(np.sum((sx != 0) & (sy == 0)))
    denom = math.sqrt((concordant + discordant + tie_x) * (concordant + discordant + tie_y))
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def group_folds(groups: np.ndarray, max_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.array(sorted(pd.unique(groups)))
    splits = min(max_splits, len(unique))
    buckets = np.array_split(unique, splits)
    folds = []
    for bucket in buckets:
        test = np.isin(groups, bucket)
        train = ~test
        if train.any() and test.any():
            folds.append((np.flatnonzero(train), np.flatnonzero(test)))
    return folds


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    mu = train_x.mean(axis=0)
    sigma = train_x.std(axis=0)
    sigma[sigma == 0] = 1.0
    x_train = (train_x - mu) / sigma
    x_test = (test_x - mu) / sigma
    x_train = np.column_stack([np.ones(len(x_train)), x_train])
    x_test = np.column_stack([np.ones(len(x_test)), x_test])
    penalty = np.eye(x_train.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x_train.T @ x_train + penalty, x_train.T @ train_y)
    return x_test @ beta


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value > neg)) + 0.5 * float(np.sum(value == neg))
    return float(wins / (len(pos) * len(neg)))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            vals.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(vals)) if vals else float("nan")


def binary_log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def corr_row(frame: pd.DataFrame, x: str, y: str) -> dict[str, Any]:
    local = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 3:
        return {"rows": int(len(local)), "pearson": np.nan, "spearman": np.nan, "kendall": np.nan}
    return {
        "rows": int(len(local)),
        "pearson": float(local[x].corr(local[y])),
        "spearman": rank_corr(local[x], local[y]),
        "kendall": kendall_tau_b(local[x], local[y]),
    }


def grouped_corr(frame: pd.DataFrame, keys: list[str], x: str, targets: tuple[str, ...] = TARGETS) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        base = dict(zip(keys, values if isinstance(values, tuple) else (values,), strict=True))
        for target in targets:
            rows.append(base | {"metric": x, "target": target} | corr_row(group, x, target))
    return pd.DataFrame(rows)


def residual(values: pd.Series, steps: pd.Series) -> np.ndarray:
    y = values.to_numpy(float)
    x = np.column_stack([np.ones(len(y)), np.log1p(steps.to_numpy(float))])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def load_state() -> tuple[pd.DataFrame, pd.DataFrame]:
    llama_module = pd.read_csv(D10 / "llama_legacy_matched_numeric_parity.csv")
    llama_module = llama_module.rename(columns={"module_r_matched": "state_rank_current"})
    llama_module["state_rank_delta"] = llama_module["state_rank_current"] - llama_module["state_rank_base"]
    llama_module["source_name"] = "llama_d10_matched_numeric"
    llama_module["source_protocol"] = "D10_bf16_forward_fp64_eigh_svd"
    llama_module["track"] = "llama_d10_matched_numeric"
    keep = [
        "absolute_contraction",
        "arm",
        "checkpoint",
        "epsilon",
        "layer",
        "model",
        "module",
        "probe_name",
        "relative_functional_contraction_module",
        "source_name",
        "source_protocol",
        "state_rank_base",
        "state_rank_current",
        "state_rank_delta",
        "track",
    ]
    llama_module = llama_module[keep].copy()
    qwen_module = pd.read_csv(FINAL / "qwen_d4_merged_state_module_audit.csv")
    qwen_module["track"] = "qwen_d4_merged_state"
    qwen_module = qwen_module[keep].copy()
    module = pd.concat([llama_module, qwen_module], ignore_index=True, sort=False)
    module = module[module["probe_name"].isin(CORE_PROBES)].copy()
    module["attention_or_mlp"] = np.where(module["module"].str.startswith("self_attn"), "attention", "mlp")

    llama_equal = pd.read_csv(D10 / "llama_matched_state_equal7.csv")
    llama_equal["track"] = "llama_d10_matched_numeric"
    qwen_equal = pd.read_csv(FINAL / "qwen_d4_merged_state_all_cells.csv")
    qwen_equal["track"] = "qwen_d4_merged_state"
    equal = pd.concat([llama_equal, qwen_equal], ignore_index=True, sort=False)
    equal = equal[equal["probe_name"].isin(CORE_PROBES)].copy()
    return module, equal


def load_outputs() -> pd.DataFrame:
    llama = pd.read_csv(D10 / "llama_matched_fixed_token_outputs.csv")
    llama["track"] = "llama_d10_matched_numeric"
    qwen = pd.read_csv(FINAL / "qwen_d4_merged_state_outputs.csv")
    qwen["track"] = "qwen_d4_merged_state"
    out = pd.concat([llama, qwen], ignore_index=True, sort=False)
    return out[out["probe_name"].isin(CORE_PROBES)].copy()


def build_joined(equal: pd.DataFrame, output: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "arm", "checkpoint", "probe_name", "track"]
    joined = equal[equal["arm"].isin(ARMS)].merge(output[output["arm"].isin(ARMS)], on=keys, how="inner")
    joined = joined[joined["checkpoint"].gt(0)].copy()
    joined = joined.rename(columns={"relative_functional_contraction_equal7": "c_epsilon"})
    return joined


def build_stepwise(output: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in output.groupby(["model", "arm", "probe_name", "track"], sort=True):
        group = group.sort_values("checkpoint")
        prev: dict[str, Any] | None = None
        for row in group.to_dict("records"):
            if prev is None:
                prev = row
                continue
            rows.append(
                {
                    "model": keys[0],
                    "arm": keys[1],
                    "probe_name": keys[2],
                    "track": keys[3],
                    "source_checkpoint": int(prev["checkpoint"]),
                    "checkpoint": int(row["checkpoint"]),
                    "delta_nll_stepwise": float(row["nll_current"] - prev["nll_current"]),
                    "absolute_delta_nll_stepwise": float(abs(row["nll_current"] - prev["nll_current"])),
                    "stepwise_kl_source_to_current": np.nan,
                    "stepwise_kl_status": "UNAVAILABLE_NO_ADJACENT_SOURCE_LOGITS_IN_D10_D4_AGGREGATE",
                }
            )
            prev = row
    return pd.DataFrame(rows)


def model_cv(frame: pd.DataFrame, feature_sets: dict[str, list[str]], targets: tuple[str, ...] = TARGETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for (model, layer, epsilon, track), local in frame.groupby(["model", "layer", "epsilon", "track"], sort=True):
        for target in targets:
            for name, fields in feature_sets.items():
                clean = local.dropna(subset=fields + [target]).copy()
                groups = clean["checkpoint"].to_numpy()
                if len(clean) < 12 or len(np.unique(groups)) < 3:
                    results.append(
                        {
                            "model": model,
                            "layer": layer,
                            "epsilon": epsilon,
                            "track": track,
                            "target": target,
                            "feature_set": name,
                            "status": "DEFERRED_INSUFFICIENT_GROUPS",
                            "rows": int(len(clean)),
                            "checkpoint_groups": int(len(np.unique(groups))),
                        }
                    )
                    continue
                x = clean[fields].to_numpy(float)
                y = clean[target].to_numpy(float)
                pred = np.full(len(clean), np.nan)
                for train, test in group_folds(groups, 5):
                    pred[test] = ridge_predict(x[train], y[train], x[test])
                results.append(
                    {
                        "model": model,
                        "layer": layer,
                        "epsilon": epsilon,
                        "track": track,
                        "target": target,
                        "feature_set": name,
                        "status": "complete",
                        "rows": int(len(clean)),
                        "checkpoint_groups": int(len(np.unique(groups))),
                        "heldout_mae": mae(y, pred),
                        "heldout_r2": r2(y, pred),
                        "heldout_spearman": rank_corr(pd.Series(y), pd.Series(pred)),
                    }
                )
                for (_, item), actual, estimate in zip(clean.iterrows(), y, pred, strict=True):
                    predictions.append(
                        {
                            "model": model,
                            "layer": layer,
                            "epsilon": epsilon,
                            "track": track,
                            "target": target,
                            "feature_set": name,
                            "arm": item["arm"],
                            "checkpoint": item["checkpoint"],
                            "probe_name": item["probe_name"],
                            "actual": actual,
                            "predicted": estimate,
                        }
                    )
    return pd.DataFrame(results), pd.DataFrame(predictions)


def d10_5_tables(module: pd.DataFrame, equal: pd.DataFrame, output: pd.DataFrame, joined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    tables["d10_5_integrated_state_module.csv"] = module
    tables["d10_5_integrated_state_equal7.csv"] = equal
    tables["d10_5_integrated_outputs.csv"] = output
    stepwise = build_stepwise(output)
    tables["d10_5_stepwise_outputs.csv"] = stepwise

    tables["d10_5_output_correlations.csv"] = grouped_corr(
        joined, ["model", "arm", "layer", "epsilon", "track"], "c_epsilon"
    )
    for group_name in ("attention", "mlp"):
        g = module[module["attention_or_mlp"].eq(group_name)]
        values = g.groupby(
            ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "track"], dropna=False
        )["relative_functional_contraction_module"].mean().rename(f"c_{group_name}").reset_index()
        joined = joined.merge(values, on=["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "track"], how="left")
    tables["d10_5_attention_mlp_correlations.csv"] = pd.concat(
        [
            grouped_corr(joined, ["model", "arm", "layer", "epsilon", "track"], "c_attention"),
            grouped_corr(joined, ["model", "arm", "layer", "epsilon", "track"], "c_mlp"),
        ],
        ignore_index=True,
    )
    tables["d10_5_within_arm_domain_correlations.csv"] = grouped_corr(
        joined, ["model", "arm", "checkpoint", "layer", "epsilon", "track"], "c_epsilon"
    )
    tables["d10_5_within_domain_arm_correlations.csv"] = grouped_corr(
        joined, ["model", "probe_name", "checkpoint", "layer", "epsilon", "track"], "c_epsilon"
    )

    demean = joined.copy()
    for col in ("c_epsilon", *TARGETS):
        demean[f"demean_{col}"] = demean[col] - demean.groupby(["model", "layer", "checkpoint", "track"])[col].transform("mean")
    tables["d10_5_checkpoint_demeaned_correlations.csv"] = grouped_corr(
        demean,
        ["model", "layer", "epsilon", "track"],
        "demean_c_epsilon",
        tuple(f"demean_{t}" for t in TARGETS),
    )

    det_rows: list[dict[str, Any]] = []
    for values, group in joined.groupby(["model", "arm", "layer", "epsilon", "track"], sort=True):
        item = dict(zip(["model", "arm", "layer", "epsilon", "track"], values, strict=True))
        clean = group.dropna(subset=["c_epsilon", "delta_nll_cumulative"]).copy()
        if len(clean) >= 4 and clean["checkpoint"].nunique() >= 3:
            clean["c_residual"] = residual(clean["c_epsilon"], clean["checkpoint"])
            clean["nll_residual"] = residual(clean["delta_nll_cumulative"], clean["checkpoint"])
            det_rows.append(item | corr_row(clean, "c_residual", "nll_residual"))
        else:
            det_rows.append(item | {"rows": int(len(clean)), "pearson": np.nan, "spearman": np.nan, "kendall": np.nan})
    tables["d10_5_logstep_detrended_signed_nll.csv"] = pd.DataFrame(det_rows)

    signed_models, signed_predictions = model_cv(joined, {"Model-C": ["c_epsilon"]}, ("delta_nll_cumulative",))
    tables["d10_5_signed_nll_grouped_models.csv"] = signed_models
    tables["d10_5_signed_nll_predictions.csv"] = signed_predictions

    interaction_rows: list[dict[str, Any]] = []
    for (model, layer, epsilon, track), local in joined.groupby(["model", "layer", "epsilon", "track"], sort=True):
        clean = local.dropna(subset=["c_epsilon", "delta_nll_cumulative"]).copy()
        base = {"model": model, "layer": layer, "epsilon": epsilon, "track": track}
        if len(clean) < 16 or clean["arm"].nunique() < 2:
            interaction_rows.append(base | {"status": "DEFERRED_INSUFFICIENT_COMMON_GRID", "rows": int(len(clean))})
            continue
        domains = pd.get_dummies(clean["probe_name"], drop_first=True, dtype=float)
        opd = clean["arm"].eq("opd").astype(float).to_numpy()
        x = np.column_stack(
            [np.ones(len(clean)), clean["c_epsilon"], opd, clean["c_epsilon"].to_numpy() * opd, np.log1p(clean["checkpoint"]), domains.to_numpy()]
        )
        y = clean["delta_nll_cumulative"].to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        groups = clean["checkpoint"].unique()
        draws: list[float] = []
        for _ in range(256):
            sampled = RNG.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([clean[clean["checkpoint"].eq(g)] for g in sampled], ignore_index=True)
            b_domains = pd.get_dummies(boot["probe_name"], drop_first=True, dtype=float).reindex(columns=domains.columns, fill_value=0)
            b_opd = boot["arm"].eq("opd").astype(float).to_numpy()
            b_x = np.column_stack(
                [np.ones(len(boot)), boot["c_epsilon"], b_opd, boot["c_epsilon"].to_numpy() * b_opd, np.log1p(boot["checkpoint"]), b_domains.to_numpy()]
            )
            draws.append(float(np.linalg.lstsq(b_x, boot["delta_nll_cumulative"].to_numpy(float), rcond=None)[0][3]))
        interaction_rows.append(
            base
            | {
                "status": "complete",
                "rows": int(len(clean)),
                "checkpoint_groups": int(len(groups)),
                "opd_c_interaction": float(beta[3]),
                "grouped_bootstrap_ci_low": float(np.quantile(draws, 0.025)),
                "grouped_bootstrap_ci_high": float(np.quantile(draws, 0.975)),
            }
        )
    tables["d10_5_signed_nll_interaction.csv"] = pd.DataFrame(interaction_rows)

    energy = pd.read_csv(FINAL / "d5_fairness_update_equal7.csv")
    d5 = joined.merge(energy, on=["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], how="inner")
    feature_sets = {
        "Model-W": ["raw_update_energy_equal7"],
        "Model-C": ["c_epsilon"],
        "Model-WC": ["raw_update_energy_equal7", "c_epsilon"],
        "Model-WS": ["raw_update_energy_equal7", "whitened_update_energy_equal7"],
        "Model-WSC": ["raw_update_energy_equal7", "whitened_update_energy_equal7", "c_epsilon"],
    }
    d5_models, d5_predictions = model_cv(d5, feature_sets)
    tables["d10_5_d5_common_grid_models.csv"] = d5_models
    tables["d10_5_d5_common_grid_predictions.csv"] = d5_predictions

    llama96 = d5[
        (d5["model"].eq("llama"))
        & (d5["layer"].eq(14))
        & (np.isclose(d5["epsilon"].astype(float), 0.05))
        & (d5["track"].eq("llama_d10_matched_numeric"))
    ].copy()
    tables["d10_5_llama_model_c_full_availability_96.csv"] = llama96[
        ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "track", "c_epsilon", *TARGETS]
    ].sort_values(["arm", "checkpoint", "probe_name"]).reset_index(drop=True)

    tables["d10_5_dominance_ncd.csv"] = dominance_ncd(equal)
    tables["d10_5_legacy_matched_parity_summary.csv"] = parity_summary()
    tables["d10_5_branch_codes.csv"] = branch_codes(tables)
    return tables


def dominance_ncd(equal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eps = equal[np.isclose(equal["epsilon"].astype(float), 0.05)].copy()
    for model, layer, probes, label in [
        ("llama", 14, CORE_PROBES, "d10_four_core_llama"),
        ("qwen", 18, CORE_PROBES, "d4_four_core_qwen"),
        ("llama", 14, SHARED_PROBES, "shared_axis_llama"),
        ("qwen", 18, SHARED_PROBES, "shared_axis_qwen"),
    ]:
        local = eps[(eps["model"].eq(model)) & (eps["layer"].eq(layer)) & (eps["probe_name"].isin(probes))]
        for ck in (20, 40, 80):
            for probe in probes:
                g = local[(local["checkpoint"].eq(ck)) & (local["probe_name"].eq(probe))]
                if set(ARMS).issubset(set(g["arm"])):
                    opd = g[g["arm"].eq("opd")].iloc[0]
                    offline = g[g["arm"].isin(("sft", "offkd", "seqkd"))]
                    rows.append(
                        {
                            "scope": label,
                            "model": model,
                            "layer": layer,
                            "probe_name": probe,
                            "checkpoint": ck,
                            "opd_deepest_c": bool(opd["relative_functional_contraction_equal7"] > offline["relative_functional_contraction_equal7"].max()),
                            "opd_deepest_rank": bool(opd["state_rank_current_mean"] < offline["state_rank_current_mean"].min()),
                            "rank_margin_nearest_offline": float(offline["state_rank_current_mean"].min() - opd["state_rank_current_mean"]),
                        }
                    )
    steps = [5, 20, 40, 80, 160, 320]
    tau = np.log1p(np.array([0] + steps, dtype=float))
    for (model, arm, layer), g in eps[eps["arm"].isin(ARMS)].groupby(["model", "arm", "layer"], sort=True):
        values = []
        for probe, gp in g.groupby("probe_name"):
            series = gp.set_index("checkpoint")["state_rank_delta_mean"].reindex(steps)
            if series.isna().any():
                continue
            y = [0.0] + list(np.maximum(-series.to_numpy(float), 0.0))
            values.append(float(np.trapz(y, tau) / tau[-1]))
        rows.append(
            {
                "scope": "ncd_eps05_logtime_state_rank_delta",
                "model": model,
                "layer": int(layer),
                "arm": arm,
                "ncd": float(np.mean(values)) if values else np.nan,
                "probe_count": len(values),
            }
        )
    old = pd.read_csv(FINAL / "relative_functional_contraction_all_cells.csv")
    legacy_rows = []
    for model, layer in (("llama", 14), ("qwen", 18)):
        local = old[
            (old["model"].eq(model))
            & (old["layer"].eq(layer))
            & (np.isclose(old["epsilon"].astype(float), 0.05))
            & (old["checkpoint"].isin((20, 40, 80)))
            & (old["arm"].isin(ARMS))
        ]
        for ck in (20, 40, 80):
            for probe in sorted(local["probe_name"].dropna().unique()):
                g = local[(local["checkpoint"].eq(ck)) & (local["probe_name"].eq(probe))]
                if set(ARMS).issubset(set(g["arm"])):
                    opd = g[g["arm"].eq("opd")].iloc[0]
                    offline = g[g["arm"].isin(("sft", "offkd", "seqkd"))]
                    legacy_rows.append(
                        {
                            "scope": "legacy_six_probe_sensitivity",
                            "model": model,
                            "layer": layer,
                            "probe_name": probe,
                            "checkpoint": ck,
                            "opd_deepest_c": bool(opd["relative_functional_contraction_equal7"] > offline["relative_functional_contraction_equal7"].max()),
                        }
                    )
    return pd.concat([pd.DataFrame(rows), pd.DataFrame(legacy_rows)], ignore_index=True, sort=False)


def parity_summary() -> pd.DataFrame:
    old = pd.read_csv(FINAL / "relative_functional_contraction_all_cells.csv")
    new = pd.read_csv(D10 / "llama_matched_state_equal7.csv")
    old = old[
        (old["model"].eq("llama"))
        & (old["layer"].eq(14))
        & (old["arm"].isin(ARMS))
        & (old["probe_name"].isin(CORE_PROBES))
        & (old["checkpoint"].isin([5, 20, 40, 80, 160, 320]))
    ]
    new = new[new["arm"].isin(ARMS)]
    keys = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"]
    merged = old[keys + ["relative_functional_contraction_equal7"]].merge(
        new[keys + ["relative_functional_contraction_equal7"]], on=keys, suffixes=("_old", "_matched")
    )
    rows = []
    for label, group in {"all": merged, "eps05": merged[np.isclose(merged["epsilon"].astype(float), 0.05)]}.items():
        diff = (group["relative_functional_contraction_equal7_old"] - group["relative_functional_contraction_equal7_matched"]).abs()
        rows.append(
            {
                "slice": label,
                "rows": int(len(group)),
                "pearson": float(group["relative_functional_contraction_equal7_old"].corr(group["relative_functional_contraction_equal7_matched"])),
                "spearman": rank_corr(group["relative_functional_contraction_equal7_old"], group["relative_functional_contraction_equal7_matched"]),
                "mae": float(diff.mean()),
                "max_abs_diff": float(diff.max()),
            }
        )
    return pd.DataFrame(rows)


def branch_codes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signed = tables["d10_5_output_correlations.csv"]
    det = tables["d10_5_logstep_detrended_signed_nll.csv"]
    inter = tables["d10_5_signed_nll_interaction.csv"]

    def value(frame: pd.DataFrame, model: str, arm: str | None, col: str) -> float:
        local = frame[(frame["model"].eq(model)) & (np.isclose(frame["epsilon"].astype(float), 0.05))]
        if arm is not None and "arm" in local.columns:
            local = local[local["arm"].eq(arm)]
        if "target" in local.columns:
            local = local[local["target"].eq("delta_nll_cumulative")]
        if "layer" in local.columns:
            local = local[local["layer"].isin([14, 18])]
        return float(local[col].dropna().mean()) if len(local[col].dropna()) else np.nan

    qwen_nonopd = []
    for arm in ("sft", "offkd", "seqkd"):
        qwen_nonopd.append(value(signed, "qwen", arm, "spearman"))
    if all(v > 0 for v in qwen_nonopd if not np.isnan(v)):
        code = "C"
        rule = "Qwen D4 headline signed-NLL Spearman is positive for SFT, off-KD, and seqKD."
    else:
        code = "NON_C"
        rule = "Mechanical branch did not satisfy the old C condition."
    return pd.DataFrame(
        [
            {
                "branch_code": code,
                "coder_scope": "d10_5_mechanical_pre_registered_branch_code_no_theory_interpretation",
                "headline_epsilon": 0.05,
                "llama_opd_signed_spearman": value(signed, "llama", "opd", "spearman"),
                "llama_opd_detrended_spearman": value(det, "llama", "opd", "spearman"),
                "llama_opd_interaction": value(inter, "llama", None, "opd_c_interaction"),
                "qwen_opd_signed_spearman": value(signed, "qwen", "opd", "spearman"),
                "qwen_sft_signed_spearman": value(signed, "qwen", "sft", "spearman"),
                "qwen_offkd_signed_spearman": value(signed, "qwen", "offkd", "spearman"),
                "qwen_seqkd_signed_spearman": value(signed, "qwen", "seqkd", "spearman"),
                "qwen_opd_detrended_spearman": value(det, "qwen", "opd", "spearman"),
                "qwen_opd_interaction": value(inter, "qwen", None, "opd_c_interaction"),
                "selection_rule": rule,
            }
        ]
    )


def load_pk_cell() -> pd.DataFrame:
    paths = [
        STAGE3 / "H2_tpk/T_PK_llama3_2_3b.csv",
        STAGE3 / "H2_tpk/T_PK_qwen3_4b.csv",
    ]
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["model"] = np.where(frame["family"].str.contains("llama"), "llama", "qwen")
        frame = frame.rename(columns={"step": "checkpoint"})
        pivot = (
            frame.pivot_table(
                index=["model", "arm", "checkpoint", "layer", "module"],
                columns="k",
                values="p_k",
                aggfunc="mean",
            )
            .reset_index()
            .rename(columns={4: "p_k4", 8: "p_k8", 16: "p_k16", 32: "p_k32"})
        )
        frames.append(pivot)
    return pd.concat(frames, ignore_index=True, sort=False)


def a4_tables(module: pd.DataFrame, joined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    energy = pd.read_csv(FINAL / "d5_fairness_update_equal7.csv")
    pk = load_pk_cell()
    pk_cell = pk.groupby(["model", "arm", "checkpoint", "layer"], as_index=False)[["p_k4", "p_k8", "p_k16", "p_k32"]].mean()
    frame = joined.merge(energy, on=["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"], how="left")
    frame = frame.merge(pk_cell, on=["model", "arm", "checkpoint", "layer"], how="left")
    feature_sets = {
        "W_raw": ["raw_update_energy_equal7"],
        "W_WS": ["raw_update_energy_equal7", "whitened_update_energy_equal7"],
        "p_k": ["p_k4", "p_k8", "p_k16", "p_k32"],
        "C": ["c_epsilon"],
        "W_plus_C": ["raw_update_energy_equal7", "c_epsilon"],
        "p_k_plus_C": ["p_k4", "p_k8", "p_k16", "p_k32", "c_epsilon"],
        "W_p_k_C": ["raw_update_energy_equal7", "p_k4", "p_k8", "p_k16", "p_k32", "c_epsilon"],
        "WSC": ["raw_update_energy_equal7", "whitened_update_energy_equal7", "c_epsilon"],
    }
    models, predictions = model_cv(frame, feature_sets)

    discrim_rows: list[dict[str, Any]] = []
    for (model, layer, epsilon, track), local in frame.groupby(["model", "layer", "epsilon", "track"], sort=True):
        y = local["arm"].eq("opd").astype(int).to_numpy()
        groups = local["checkpoint"].to_numpy()
        for name, fields in feature_sets.items():
            clean = local.dropna(subset=fields).copy()
            y = clean["arm"].eq("opd").astype(int).to_numpy()
            groups = clean["checkpoint"].to_numpy()
            if len(clean) < 16 or len(np.unique(y)) < 2 or len(np.unique(groups)) < 3:
                discrim_rows.append(
                    {
                        "model": model,
                        "layer": layer,
                        "epsilon": epsilon,
                        "track": track,
                        "feature_set": name,
                        "status": "DEFERRED_INSUFFICIENT_GROUPS",
                        "rows": int(len(clean)),
                    }
                )
                continue
            prob = np.full(len(clean), np.nan)
            x = clean[fields].to_numpy(float)
            for train, test in group_folds(groups, 5):
                raw = ridge_predict(x[train], y[train].astype(float), x[test])
                prob[test] = sigmoid(raw)
            pred = (prob >= 0.5).astype(int)
            discrim_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "epsilon": epsilon,
                    "track": track,
                    "feature_set": name,
                    "status": "complete",
                    "rows": int(len(clean)),
                    "checkpoint_groups": int(len(np.unique(groups))),
                    "heldout_auc": roc_auc(y, prob),
                    "heldout_balanced_accuracy": balanced_accuracy(y, pred),
                    "heldout_log_loss": binary_log_loss(y, prob),
                }
            )
    sub_tables = []
    for path in [STAGE3 / "H2_sub/T_SUB_llama3_2_3b.csv", STAGE3 / "H2_sub/T_SUB_qwen3_4b.csv"]:
        sub = pd.read_csv(path)
        sub["model"] = np.where(sub["family"].str.contains("llama"), "llama", "qwen")
        sub_tables.append(sub)
    sub = pd.concat(sub_tables, ignore_index=True, sort=False)
    sub_summary = sub.groupby(["model", "step", "layer"], as_index=False).agg(
        rows=("family", "size"),
        mean_output_projector_overlap=("output_projector_overlap", "mean"),
        mean_input_projector_overlap_fixed=("input_projector_overlap_fixed", "mean"),
        mean_output_angle_deg=("output_angle_mean_deg", "mean"),
        mean_input_angle_deg_fixed=("input_angle_mean_deg_fixed", "mean"),
    )
    coverage = pd.DataFrame(
        [
            {
                "component": "strict_fixed_k_p_k",
                "status": "complete_raw",
                "rows": int(sum(len(pd.read_csv(p)) for p in [STAGE3 / "H2_tpk/T_PK_llama3_2_3b.csv", STAGE3 / "H2_tpk/T_PK_qwen3_4b.csv"])),
                "caveat": "Llama adapter_BA; Qwen BF16 merged-minus-base.",
            },
            {
                "component": "PABS_NSS_direction",
                "status": "partial_pairwise_overlap_available",
                "rows": int(len(sub)),
                "caveat": "T_SUB provides pairwise projector overlaps/angles, not full same-cell PABS/NSS masks.",
            },
            {
                "component": "same_fold_incremental_cv",
                "status": "complete_core",
                "rows": int(len(models)),
                "caveat": "Core feature sets W, WS, p_k, C, and combinations; optional full PABS/NSS not asserted.",
            },
        ]
    )
    return {
        "d10_5_a4_feature_matrix.csv": frame,
        "d10_5_a4_incremental_models.csv": models,
        "d10_5_a4_incremental_predictions.csv": predictions,
        "d10_5_a4_arm_discrimination.csv": pd.DataFrame(discrim_rows),
        "d10_5_a4_direction_overlap_summary.csv": sub_summary,
        "d10_5_a4_coverage.csv": coverage,
    }


def a5_a9_acceptance() -> pd.DataFrame:
    rows = []
    checks = [
        ("A5_current_fixed", STAGE4 / "outputs/state_rank_full_cells.csv", "complete_from_existing_stage4_formal_artifact"),
        ("A5_centered", STAGE4 / "outputs/centered_state_rank_full_cells.csv", "blocked_missing_centered_activation_mean_or_forward_artifact"),
        ("A6_sample_tail_epsilon", STAGE4 / "cpu/state_displacement_sample_count_bootstrap.csv", "complete_from_existing_stage4_formal_artifact"),
        ("A7_schema_provenance", STAGE4 / "cpu/state_displacement_schema.json", "complete_from_existing_stage4_formal_artifact"),
        ("A8_trainer_top32_audit", STAGE4 / "cpu/trainer_arm_implementation_audit.json", "complete_from_existing_stage4_formal_artifact"),
        ("A9_behavior_uncertainty", MINI / "relative_contraction_signed_nll_branch_codes.csv", "complete_legacy_behavior_protocol_available_not_d10_specific_item_bootstrap"),
    ]
    for task, path, ok_status in checks:
        if path.exists():
            rows.append(
                {
                    "task": task,
                    "status": ok_status,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        else:
            rows.append({"task": task, "status": ok_status if task == "A5_centered" else "blocked_missing_artifact", "path": str(path), "bytes": 0, "sha256": None})
    return pd.DataFrame(rows)


def a5_centered_tables() -> dict[str, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for path in sorted((STAGE4 / "cells").glob("*/*/step_*/*.main.centered.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") != "complete":
            continue
        for row in payload.get("rows", []):
            item = dict(row)
            item["cell_json"] = str(path)
            rows.append(item)
    centered = pd.DataFrame(rows)
    if centered.empty:
        return {
            "d10_5_a5_centered_state_cells.csv": centered,
            "d10_5_a5_construct_ablation.csv": pd.DataFrame(),
            "d10_5_a5_centered_coverage.csv": pd.DataFrame(
                [{"status": "blocked_missing_centered_cell_json", "rows": 0}]
            ),
        }
    current = pd.read_csv(STAGE4 / "outputs/state_rank_full_cells.csv")
    keys = ["model", "arm", "checkpoint", "probe_name", "layer", "module", "epsilon"]
    comp = centered.merge(
        current[keys + ["state_rank", "displacement_rank", "displacement_norm_normalized"]].rename(
            columns={
                "state_rank": "state_rank_uncentered",
                "displacement_rank": "displacement_rank_uncentered",
                "displacement_norm_normalized": "displacement_norm_normalized_uncentered",
            }
        ),
        on=keys,
        how="left",
    )
    comp = comp.rename(
        columns={
            "state_rank": "state_rank_centered",
            "displacement_rank": "displacement_rank_centered",
            "displacement_norm_normalized": "displacement_norm_normalized_centered",
        }
    )
    comp["state_rank_centered_minus_uncentered"] = comp["state_rank_centered"] - comp["state_rank_uncentered"]
    comp["displacement_rank_centered_minus_uncentered"] = (
        comp["displacement_rank_centered"] - comp["displacement_rank_uncentered"]
    )
    coverage = centered.groupby(["model", "arm", "checkpoint", "probe_name", "layer"], as_index=False).agg(
        rows=("module", "size"),
        modules=("module", "nunique"),
        epsilons=("epsilon", "nunique"),
    )
    return {
        "d10_5_a5_centered_state_cells.csv": centered,
        "d10_5_a5_construct_ablation.csv": comp,
        "d10_5_a5_centered_coverage.csv": coverage,
    }


def write_handoff(outputs: dict[str, Path], manifest: dict[str, Any]) -> Path:
    handoff = D10 / "d10_5_a2_a4_handoff.md"
    acceptance = pd.read_csv(outputs["d10_5_a5_a9_acceptance.csv"])
    branch = pd.read_csv(outputs["d10_5_branch_codes.csv"])
    dominance = pd.read_csv(outputs["d10_5_dominance_ncd.csv"])
    dom_summary = dominance[dominance["scope"].isin(["shared_axis_llama", "shared_axis_qwen"])]
    shared_count = int(dom_summary["opd_deepest_c"].fillna(False).sum()) if not dom_summary.empty else 0
    shared_total = int(dom_summary["opd_deepest_c"].notna().sum()) if not dom_summary.empty else 0
    ncd = dominance[dominance["scope"].eq("ncd_eps05_logtime_state_rank_delta")]
    ncd_pivot = ncd.pivot_table(index=["model", "layer"], columns="arm", values="ncd", aggfunc="mean").reset_index()
    lines = [
        "# D10.5/A2/A4 matched downstream handoff",
        "",
        "## Status",
        "",
        f"- status: `{manifest['status']}`",
        f"- created_utc: `{manifest['created_utc']}`",
        "- Llama track: `llama_d10_matched_numeric`",
        "- Qwen track: `qwen_d4_merged_state`",
        "- legacy Llama rows are not used in the D10.5 downstream tables.",
        "",
        "## Core Coverage",
        "",
        f"- integrated module rows: `{manifest['integrated_module_rows']}`",
        f"- integrated equal-seven rows: `{manifest['integrated_equal7_rows']}`",
        f"- integrated output rows: `{manifest['integrated_output_rows']}`",
        f"- A4 feature rows: `{manifest['a4_feature_rows']}`",
        f"- shared-axis early dominance: `{shared_count}/{shared_total}`",
        "",
        "## Branch Code",
        "",
        branch.to_markdown(index=False),
        "",
        "## NCD eps=.05",
        "",
        ncd_pivot.to_markdown(index=False),
        "",
        "## A5--A9 Acceptance",
        "",
        acceptance[["task", "status", "path"]].to_markdown(index=False),
        "",
        "## Output Files",
        "",
    ]
    for name, path in outputs.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    lines.append("No theory adjudication is included.")
    handoff.write_text("\n".join(lines), encoding="utf-8")
    return handoff


def run() -> dict[str, Any]:
    module, equal = load_state()
    output = load_outputs()
    joined = build_joined(equal, output)
    tables = d10_5_tables(module, equal, output, joined)
    tables |= a4_tables(module, joined)
    tables |= a5_centered_tables()
    tables["d10_5_a5_a9_acceptance.csv"] = a5_a9_acceptance()
    if len(tables["d10_5_a5_centered_state_cells.csv"]):
        tables["d10_5_a5_a9_acceptance.csv"].loc[
            tables["d10_5_a5_a9_acceptance.csv"]["task"].eq("A5_centered"),
            ["status", "path", "bytes", "sha256"],
        ] = [
            "partial_centered_cells_aggregated_from_stage4_formal_artifacts",
            str(D10 / "d10_5_a5_centered_state_cells.csv"),
            int(0),
            None,
        ]

    outputs: dict[str, Path] = {}
    for name, frame in tables.items():
        path = D10 / name
        atomic_csv(path, frame)
        outputs[name] = path
        mirror(path)

    if len(tables["d10_5_a5_centered_state_cells.csv"]):
        centered_path = outputs["d10_5_a5_centered_state_cells.csv"]
        acceptance = tables["d10_5_a5_a9_acceptance.csv"].copy()
        acceptance.loc[
            acceptance["task"].eq("A5_centered"),
            ["status", "path", "bytes", "sha256"],
        ] = [
            "partial_centered_cells_aggregated_from_stage4_formal_artifacts",
            str(centered_path),
            int(centered_path.stat().st_size),
            sha256(centered_path),
        ]
        tables["d10_5_a5_a9_acceptance.csv"] = acceptance
        atomic_csv(outputs["d10_5_a5_a9_acceptance.csv"], acceptance)
        mirror(outputs["d10_5_a5_a9_acceptance.csv"])

    manifest = {
        "schema_version": "cycle09_d10_5_a2_a4_manifest_v1",
        "status": "COMPLETE_D10_5_CORE_A2_A4_WITH_PARTIAL_A5_CENTERED_AGGREGATED",
        "created_utc": now(),
        "integrated_module_rows": int(len(tables["d10_5_integrated_state_module.csv"])),
        "integrated_equal7_rows": int(len(tables["d10_5_integrated_state_equal7.csv"])),
        "integrated_output_rows": int(len(tables["d10_5_integrated_outputs.csv"])),
        "a4_feature_rows": int(len(tables["d10_5_a4_feature_matrix.csv"])),
        "a5_centered_rows": int(len(tables["d10_5_a5_centered_state_cells.csv"])),
        "outputs": {
            name: {"path": str(path), "rows": int(len(tables[name])), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "caveats": [
            "A5 centered construct is aggregated where Stage4 centered cell JSON exists; it is not a full A5 all-arm/all-checkpoint centered grid.",
            "A4 full PABS/NSS masks are not asserted; pairwise T_SUB direction overlaps are summarized.",
            "Stepwise KL is unavailable from aggregate D10/D4 output artifacts; stepwise signed/absolute NLL is reconstructed.",
        ],
    }
    manifest_path = D10 / "d10_5_a2_a4_manifest.json"
    atomic_json(manifest_path, manifest)
    mirror(manifest_path)
    handoff = write_handoff(outputs, manifest)
    mirror(handoff)
    manifest["handoff"] = str(handoff)
    atomic_json(manifest_path, manifest)
    mirror(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("formal",), default="formal")
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
