#!/usr/bin/env python3
"""CPU-only D5 model comparison and D6/D7 raw-table delivery.

This script reads immutable Llama state/output tables, the new Qwen D4.1
merged-state tables, and D5's equal-seven energy output.  It writes raw
associations and grouped prediction tables only; it does not adjudicate any
Theory branch.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path("/root/LLM-output-density")
ROOT = Path("/root/autodl-tmp/cycle09_relative_functional_contraction")
FINAL = ROOT / "final"
MINI = REPO / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
ARMS = ("opd", "sft", "offkd", "seqkd")
TARGETS = ("cumulative_kl_base_to_current", "delta_nll_cumulative", "absolute_delta_nll_cumulative")
RNG = np.random.default_rng(20260726)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def corr(values: pd.DataFrame, x: str, y: str) -> dict[str, float | int]:
    local = values[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(local) < 3:
        return {"rows": int(len(local)), "pearson": np.nan, "spearman": np.nan, "kendall": np.nan}
    return {
        "rows": int(len(local)), "pearson": float(local[x].corr(local[y], method="pearson")),
        "spearman": float(local[x].corr(local[y], method="spearman")),
        "kendall": float(local[x].corr(local[y], method="kendall")),
    }


def grouped_rows(frame: pd.DataFrame, keys: list[str], label: str, x: str, targets: tuple[str, ...] = TARGETS) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        item = dict(zip(keys, values if isinstance(values, tuple) else (values,), strict=True))
        for target in targets:
            rows.append(item | {"association_scope": label, "metric": x, "target": target} | corr(group, x, target))
    return rows


def build_integrated() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    module_legacy = pd.read_csv(FINAL / "relative_functional_contraction_module_audit.csv")
    output_legacy = pd.read_csv(FINAL / "relative_contraction_matched_cumulative_outputs.csv")
    output_stepwise = pd.read_csv(FINAL / "relative_contraction_matched_stepwise_outputs.csv")
    q_module = pd.read_csv(FINAL / "qwen_d4_merged_state_module_audit.csv")
    q_output = pd.read_csv(FINAL / "qwen_d4_merged_state_outputs.csv")

    llama = module_legacy[(module_legacy["model"].eq("llama")) & module_legacy["arm"].isin(ARMS)].copy()
    llama["track"] = "legacy_llama"
    # Retain only the older Qwen outer layers for the D6 existing-three-layer audit.
    q_legacy = module_legacy[
        module_legacy["model"].eq("qwen") & module_legacy["arm"].eq("opd") & module_legacy["layer"].isin((9, 27))
    ].copy()
    q_legacy["track"] = "legacy_qwen_outer_layers"
    q_module = q_module[q_module["arm"].isin(ARMS)].copy()
    q_module["track"] = "d4_merged_state"
    state = pd.concat([llama, q_legacy, q_module], ignore_index=True, sort=False)
    state = state[state["checkpoint"].gt(0)].copy()
    state["attention_or_mlp"] = np.where(state["module"].str.startswith("self_attn"), "attention", "mlp")

    legacy_out = output_legacy[output_legacy["arm"].isin(ARMS)].copy()
    llama_out = legacy_out[legacy_out["model"].eq("llama")].copy()
    llama_out["track"] = "legacy_llama"
    q_outer_out = legacy_out[(legacy_out["model"].eq("qwen")) & legacy_out["arm"].eq("opd")].copy()
    q_outer_out["track"] = "legacy_qwen_outer_layers"
    q_output = q_output[q_output["arm"].isin(ARMS)].copy()
    q_output["track"] = "d4_merged_state"
    output = pd.concat([llama_out, q_outer_out, q_output], ignore_index=True, sort=False)
    output = output[output["checkpoint"].gt(0)].copy()

    cell_keys = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "track"]
    cell = state.groupby(cell_keys, dropna=False).agg(
        module_count=("module", "nunique"),
        c_epsilon=("relative_functional_contraction_module", "mean"),
        c_ratio_of_means=("absolute_contraction", "mean"),
    ).reset_index()
    base_mean = state.groupby(cell_keys, dropna=False)["state_rank_base"].mean().rename("base_rank_mean").reset_index()
    cell = cell.merge(base_mean, on=cell_keys, how="left")
    cell["c_ratio_of_means"] = cell["c_ratio_of_means"] / cell["base_rank_mean"]
    for group_name in ("attention", "mlp"):
        values = state[state["attention_or_mlp"].eq(group_name)].groupby(cell_keys, dropna=False)["relative_functional_contraction_module"].mean().rename(f"c_{group_name}").reset_index()
        cell = cell.merge(values, on=cell_keys, how="left")
    output_keys = ["model", "arm", "checkpoint", "probe_name", "track"]
    joined = cell.merge(output, on=output_keys, how="inner", suffixes=("", "_output"))
    return state, joined, output_stepwise


def d5_models(joined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    energy = pd.read_csv(FINAL / "d5_fairness_update_equal7.csv")
    keys = ["model", "arm", "checkpoint", "probe_name", "layer", "epsilon"]
    frame = joined.merge(energy, on=keys, how="inner", suffixes=("", "_d5"))
    frame = frame[(frame["module_count"].eq(7)) & frame["arm"].isin(ARMS)].copy()
    specs = {
        "Model-W": ["raw_update_energy_equal7"],
        "Model-C": ["c_epsilon"],
        "Model-WC": ["raw_update_energy_equal7", "c_epsilon"],
        "Model-WS": ["raw_update_energy_equal7", "whitened_update_energy_equal7"],
        "Model-WSC": ["raw_update_energy_equal7", "whitened_update_energy_equal7", "c_epsilon"],
    }
    results, predictions = [], []
    for (model, layer, epsilon), local in frame.groupby(["model", "layer", "epsilon"], sort=True):
        for target in TARGETS:
            for name, fields in specs.items():
                clean = local.dropna(subset=fields + [target]).copy()
                groups = clean["checkpoint"].to_numpy()
                if len(clean) < 12 or len(np.unique(groups)) < 3:
                    results.append({"model": model, "layer": layer, "epsilon": epsilon, "target": target, "feature_set": name, "status": "DEFERRED_INSUFFICIENT_GROUPS", "rows": len(clean), "checkpoint_groups": len(np.unique(groups))})
                    continue
                prediction = np.full(len(clean), np.nan)
                splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
                x, y = clean[fields].to_numpy(float), clean[target].to_numpy(float)
                for train, test in splitter.split(x, y, groups):
                    fit = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[train], y[train])
                    prediction[test] = fit.predict(x[test])
                results.append({
                    "model": model, "layer": layer, "epsilon": epsilon, "target": target, "feature_set": name,
                    "status": "complete", "rows": len(clean), "checkpoint_groups": len(np.unique(groups)),
                    "arms": ",".join(sorted(clean["arm"].unique())),
                    "checkpoints": ",".join(map(str, sorted(clean["checkpoint"].unique()))),
                    "probes": ",".join(sorted(clean["probe_name"].unique())),
                    "heldout_mae": float(mean_absolute_error(y, prediction)),
                    "heldout_r2": float(r2_score(y, prediction)),
                    "heldout_spearman": float(pd.Series(y).corr(pd.Series(prediction), method="spearman")),
                })
                for (_, item), actual, predicted in zip(clean.iterrows(), y, prediction, strict=True):
                    predictions.append({
                        "model": model, "layer": layer, "epsilon": epsilon, "target": target, "feature_set": name,
                        "evaluation_protocol": "leave_one_checkpoint_out", "arm": item["arm"], "checkpoint": item["checkpoint"],
                        "probe_name": item["probe_name"], "track": item["track"], "actual": actual, "predicted": predicted,
                    })
    return pd.DataFrame(results), pd.DataFrame(predictions)


def d6(state: pd.DataFrame, joined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    equal = joined[joined["module_count"].eq(7)].copy()
    correlations: list[dict[str, Any]] = []
    for metric in ("c_epsilon", "c_ratio_of_means", "c_attention", "c_mlp"):
        correlations += grouped_rows(equal, ["model", "arm", "layer", "epsilon", "track"], "model_arm_layer_epsilon", metric)
    module = state.merge(
        joined[["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "track", *TARGETS]],
        on=["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "track"], how="inner",
    )
    module_rows = grouped_rows(module, ["model", "arm", "layer", "epsilon", "module", "track"], "module", "relative_functional_contraction_module")
    within_arm = grouped_rows(equal, ["model", "arm", "checkpoint", "layer", "epsilon", "track"], "fixed_arm_checkpoint_across_domains", "c_epsilon")
    within_domain = grouped_rows(equal, ["model", "probe_name", "checkpoint", "layer", "epsilon", "track"], "fixed_domain_checkpoint_across_arms", "c_epsilon")
    demean = equal.copy()
    for column in ("c_epsilon", *TARGETS):
        demean[f"demean_{column}"] = demean[column] - demean.groupby(["model", "layer", "checkpoint", "track"])[column].transform("mean")
    demeaned = grouped_rows(demean, ["model", "layer", "epsilon", "track"], "checkpoint_demeaned_pooled", "demean_c_epsilon", tuple(f"demean_{t}" for t in TARGETS))
    return {
        "epsilon": pd.DataFrame(correlations), "module": pd.DataFrame(module_rows),
        "within_arm": pd.DataFrame(within_arm), "within_domain": pd.DataFrame(within_domain), "demeaned": pd.DataFrame(demeaned),
    }


def residual(values: pd.Series, steps: pd.Series) -> np.ndarray:
    y = values.to_numpy(float)
    x = np.column_stack([np.ones(len(y)), np.log1p(steps.to_numpy(float))])
    return y - x @ np.linalg.lstsq(x, y, rcond=None)[0]


def d7(joined: pd.DataFrame, stepwise: pd.DataFrame) -> dict[str, pd.DataFrame]:
    equal = joined[joined["module_count"].eq(7)].copy()
    signed = grouped_rows(equal, ["model", "arm", "layer", "epsilon", "track"], "model_arm_layer_epsilon", "c_epsilon", ("delta_nll_cumulative",))
    detrended_rows = []
    for values, group in equal.groupby(["model", "arm", "layer", "epsilon", "track"], sort=True):
        clean = group.dropna(subset=["c_epsilon", "delta_nll_cumulative"]).copy()
        item = dict(zip(["model", "arm", "layer", "epsilon", "track"], values, strict=True))
        if len(clean) >= 4 and clean["checkpoint"].nunique() >= 3:
            clean["c_residual"] = residual(clean["c_epsilon"], clean["checkpoint"])
            clean["nll_residual"] = residual(clean["delta_nll_cumulative"], clean["checkpoint"])
            detrended_rows.append(item | {"association_scope": "log1p_checkpoint_residualized"} | corr(clean, "c_residual", "nll_residual"))
        else:
            detrended_rows.append(item | {"association_scope": "log1p_checkpoint_residualized", "rows": len(clean), "pearson": np.nan, "spearman": np.nan, "kendall": np.nan})
    current = equal[(equal["track"].eq("legacy_llama"))].copy()
    current = current.sort_values(["model", "arm", "layer", "epsilon", "probe_name", "checkpoint"])
    current["delta_c"] = current.groupby(["model", "arm", "layer", "epsilon", "probe_name"])["c_epsilon"].diff().fillna(current["c_epsilon"])
    current["absolute_delta_c"] = current["delta_c"].abs()
    step = stepwise[stepwise["model"].eq("llama")].copy()
    step = step.merge(current[["model", "arm", "checkpoint", "probe_name", "layer", "epsilon", "delta_c", "absolute_delta_c"]], on=["model", "arm", "checkpoint", "probe_name"], how="inner")
    step_rows = []
    for metric in ("delta_c", "absolute_delta_c"):
        step_rows += grouped_rows(step, ["model", "arm", "layer", "epsilon"], "adjacent_checkpoint", metric, ("stepwise_kl_source_to_current", "delta_nll_stepwise", "absolute_delta_nll_stepwise"))
    grouped, predictions = d7_predictions(equal)
    interaction = d7_interaction(equal)
    return {
        "correlations": pd.DataFrame(signed), "detrended": pd.DataFrame(detrended_rows), "stepwise": pd.DataFrame(step_rows),
        "grouped": grouped, "predictions": predictions, "interaction": interaction,
    }


def d7_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, predictions = [], []
    for (model, layer, epsilon, track), local in frame.groupby(["model", "layer", "epsilon", "track"], sort=True):
        clean = local.dropna(subset=["c_epsilon", "delta_nll_cumulative"]).copy()
        groups = clean["checkpoint"].to_numpy()
        if len(clean) < 12 or len(np.unique(groups)) < 3:
            results.append({"model": model, "layer": layer, "epsilon": epsilon, "track": track, "status": "DEFERRED_INSUFFICIENT_GROUPS", "rows": len(clean)})
            continue
        x, y = clean[["c_epsilon"]].to_numpy(float), clean["delta_nll_cumulative"].to_numpy(float)
        pred = np.full(len(clean), np.nan)
        for train, test in GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(x, y, groups):
            pred[test] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[train], y[train]).predict(x[test])
        checkpoint_means = pd.DataFrame({"checkpoint": groups, "err": np.abs(y - pred)}).groupby("checkpoint")["err"].mean().to_numpy()
        draws = np.array([RNG.choice(checkpoint_means, size=len(checkpoint_means), replace=True).mean() for _ in range(256)])
        results.append({"model": model, "layer": layer, "epsilon": epsilon, "track": track, "status": "complete", "rows": len(clean), "checkpoint_groups": len(checkpoint_means), "heldout_mae": float(mean_absolute_error(y, pred)), "heldout_r2": float(r2_score(y, pred)), "grouped_bootstrap_ci_low": float(np.quantile(draws, .025)), "grouped_bootstrap_ci_high": float(np.quantile(draws, .975))})
        for (_, item), actual, estimate in zip(clean.iterrows(), y, pred, strict=True):
            predictions.append({"model": model, "layer": layer, "epsilon": epsilon, "track": track, "evaluation_protocol": "leave_one_checkpoint_out", "arm": item["arm"], "checkpoint": item["checkpoint"], "probe_name": item["probe_name"], "actual": actual, "predicted": estimate})
    return pd.DataFrame(results), pd.DataFrame(predictions)


def d7_interaction(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, layer, epsilon, track), local in frame.groupby(["model", "layer", "epsilon", "track"], sort=True):
        clean = local.dropna(subset=["c_epsilon", "delta_nll_cumulative"]).copy()
        if len(clean) < 16 or clean["arm"].nunique() < 2:
            rows.append({"model": model, "layer": layer, "epsilon": epsilon, "track": track, "status": "DEFERRED_INSUFFICIENT_COMMON_GRID", "rows": len(clean)})
            continue
        opd = clean["arm"].eq("opd").astype(float).to_numpy()
        domains = pd.get_dummies(clean["probe_name"], drop_first=True, dtype=float).to_numpy()
        x = np.column_stack([np.ones(len(clean)), clean["c_epsilon"], opd, clean["c_epsilon"].to_numpy() * opd, np.log1p(clean["checkpoint"]), domains])
        y = clean["delta_nll_cumulative"].to_numpy(float)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        groups = clean["checkpoint"].unique()
        draws = []
        for _ in range(256):
            sampled = RNG.choice(groups, size=len(groups), replace=True)
            boot = pd.concat([clean[clean["checkpoint"].eq(value)] for value in sampled], ignore_index=True)
            b_opd = boot["arm"].eq("opd").astype(float).to_numpy()
            b_dom = pd.get_dummies(boot["probe_name"], drop_first=True, dtype=float).reindex(columns=pd.get_dummies(clean["probe_name"], drop_first=True, dtype=float).columns, fill_value=0).to_numpy()
            b_x = np.column_stack([np.ones(len(boot)), boot["c_epsilon"], b_opd, boot["c_epsilon"].to_numpy() * b_opd, np.log1p(boot["checkpoint"]), b_dom])
            draws.append(np.linalg.lstsq(b_x, boot["delta_nll_cumulative"].to_numpy(float), rcond=None)[0][3])
        rows.append({"model": model, "layer": layer, "epsilon": epsilon, "track": track, "status": "complete", "rows": len(clean), "checkpoint_groups": len(groups), "opd_c_interaction": float(beta[3]), "grouped_bootstrap_ci_low": float(np.quantile(draws, .025)), "grouped_bootstrap_ci_high": float(np.quantile(draws, .975))})
    return pd.DataFrame(rows)


def run() -> dict[str, Any]:
    state, joined, stepwise = build_integrated()
    models, model_predictions = d5_models(joined)
    d6_tables = d6(state, joined)
    d7_tables = d7(joined, stepwise)
    outputs = {
        "relative_contraction_fair_common_grid_models.csv": models,
        "relative_contraction_fair_common_grid_predictions.csv": model_predictions,
        "relative_contraction_epsilon_layer_correlations.csv": d6_tables["epsilon"],
        "relative_contraction_module_correlations.csv": d6_tables["module"],
        "relative_contraction_within_arm_checkpoint_domain.csv": d6_tables["within_arm"],
        "relative_contraction_within_domain_checkpoint_arm.csv": d6_tables["within_domain"],
        "relative_contraction_demeaned_correlations.csv": d6_tables["demeaned"],
        "relative_contraction_signed_nll_correlations.csv": d7_tables["correlations"],
        "relative_contraction_signed_nll_detrended.csv": d7_tables["detrended"],
        "relative_contraction_signed_nll_stepwise.csv": d7_tables["stepwise"],
        "relative_contraction_signed_nll_grouped_models.csv": d7_tables["grouped"],
        "relative_contraction_signed_nll_interaction.csv": d7_tables["interaction"],
        "relative_contraction_signed_nll_predictions.csv": d7_tables["predictions"],
    }
    for name, frame in outputs.items():
        atomic_csv(FINAL / name, frame)
    manifest = {
        "schema_version": "cycle09_d5_d7_tables_v1", "status": "complete", "created_utc": now(),
        "state_rows": len(state), "joined_rows": len(joined),
        "outputs": {name: {"path": str(FINAL / name), "rows": len(frame)} for name, frame in outputs.items()},
        "qwen_main_track": "D4.1 merged-state L18; legacy Qwen L9/L27 retained only for existing-three-layer OPD robustness rows",
    }
    atomic_json(MINI / "d5_d7_tables_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("formal",), default="formal")
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False))


if __name__ == "__main__":
    main()
