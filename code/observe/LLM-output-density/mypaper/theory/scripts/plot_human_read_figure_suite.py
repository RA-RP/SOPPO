#!/usr/bin/env python3
"""Generate the complete explanatory figure suite for human_read-ch.md.

The figures are deliberately more expansive than a paper-ready figure set.
They separate:
  * the matched D10/D10.5 state/output panel,
  * intervention-only legacy tracks (alpha=.5 and frozenSelf0-KD), and
  * D11 weight-space comparison/audit products.

Run:
    python theory/scripts/plot_human_read_figure_suite.py
from /root/LLM-output-density/mypaper, or invoke it from any directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MINI = (
    ROOT
    / "local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion"
    / "run_01"
    / "mini"
)
OUT = ROOT / "theory" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

ARM_ORDER = ["opd", "sft", "offkd", "seqkd"]
ARM_LABEL = {"opd": "OPD", "sft": "SFT", "offkd": "off-KD", "seqkd": "seqKD"}
ARM_COLOR = {
    "opd": "#c43c35",
    "sft": "#3b6fb6",
    "offkd": "#2e8b57",
    "seqkd": "#8a5ab8",
    "alpha05": "#e28e2c",
    "frozen_self": "#6b7280",
}
PROBE_LABEL = {
    "E_general": "General",
    "E_math": "Math-held",
    "E_ood": "MMLU-Pro",
    "E_if": "IFEval",
    "S_math": "Math-CoT train",
    "E_math_hard_v2": "AIME25",
}
CORE = ["E_general", "E_math", "E_ood", "E_if"]
MODEL_LABEL = {"qwen": "Qwen3-4B · L18", "llama": "Llama-3.2-3B · L14"}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 210,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9dde3",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "legend.frameon": False,
        }
    )


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(MINI / name)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rank_corr(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average")))


def annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    title: str,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    fmt: str = ".1f",
    cbar_label: str | None = None,
) -> None:
    finite = values[np.isfinite(values)]
    span = max(abs(finite.min() - center), abs(finite.max() - center)) if finite.size else 1
    im = ax.imshow(values, cmap=cmap, vmin=center - span, vmax=center + span, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title)
    ax.grid(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                color = "white" if abs(v - center) > 0.62 * span else "#222222"
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8.5, color=color)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    if cbar_label:
        cb.set_label(cbar_label)


def plot_method_pipeline() -> None:
    fig = plt.figure(figsize=(15.5, 6.8))
    ax = fig.add_axes([0.02, 0.08, 0.70, 0.84])
    ax.set_axis_off()

    boxes = [
        (0.02, 0.58, 0.15, 0.19, "Frozen domain $D$", "fixed probe texts", "#e8f1fb"),
        (0.22, 0.58, 0.15, 0.19, "Activations $h$", "module input", "#e8f1fb"),
        (0.42, 0.58, 0.16, 0.19, "$\\Sigma_{D,t}$ and $S_{D,t}$", "$SS^\\top=\\mathbb{E}[hh^\\top]$", "#ecf7ef"),
        (0.63, 0.58, 0.14, 0.19, "$A_{D,t}=W_tS_{D,t}$", "functional state", "#fff2d9"),
        (0.82, 0.58, 0.15, 0.19, "SVD spectrum", "$\\sigma_1\\geq\\cdots\\geq\\sigma_q$", "#fde8e7"),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=ax.transAxes,
            fc=color,
            ec="#596273",
            lw=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.63 * h, title, transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.text(
            x + w / 2,
            y + 0.30 * h,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#4b5563",
        )
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i + 1][0]
        y = boxes[i][1] + boxes[i][3] / 2
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 0.008, y),
                (x2 - 0.008, y),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.25,
                color="#596273",
            )
        )

    lower = [
        (0.57, 0.17, 0.18, 0.19, "$r_{\\varepsilon,D,t}$", "minimum rank retaining $1-\\varepsilon$ energy", "#f3eafd"),
        (0.79, 0.17, 0.18, 0.19, "$\\Delta r$ and $c_\\varepsilon$", "trajectory and relative contraction", "#f3eafd"),
    ]
    for x, y, w, h, title, subtitle, color in lower:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            transform=ax.transAxes,
            fc=color,
            ec="#596273",
            lw=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + 0.64 * h, title, transform=ax.transAxes, ha="center", va="center", fontsize=12)
        ax.text(
            x + w / 2,
            y + 0.28 * h,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.3,
            color="#4b5563",
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.895, 0.57),
            (0.66, 0.37),
            transform=ax.transAxes,
            connectionstyle="arc3,rad=0.15",
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.25,
            color="#596273",
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (0.755, 0.265),
            (0.782, 0.265),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.25,
            color="#596273",
        )
    )
    ax.text(
        0.02,
        0.30,
        "$\\mathbb{E}\\|W_th-\\widetilde W h\\|_2^2=\\|(W_t-\\widetilde W)S_{D,t}\\|_F^2$",
        transform=ax.transAxes,
        fontsize=13,
        color="#263238",
    )
    ax.text(
        0.02,
        0.18,
        "Local optimum: truncated SVD of $WS$ minimizes expected layer-output error.\n"
        "Boundary: it is not a sufficient statistic for downstream nonlinear readout.",
        transform=ax.transAxes,
        fontsize=10,
        color="#374151",
        linespacing=1.45,
    )

    ax2 = fig.add_axes([0.76, 0.18, 0.22, 0.66])
    s = np.arange(1, 17)
    energy = np.array([31, 18, 12, 9, 7, 5, 4, 3, 2.5, 2, 1.7, 1.4, 1.1, 0.8, 0.6, 0.4])
    cumulative = np.cumsum(energy) / energy.sum()
    eps = 0.05
    r = int(np.argmax(cumulative >= 1 - eps) + 1)
    ax2.plot(s, cumulative, color="#c43c35", lw=2.6, marker="o", ms=4)
    ax2.axhline(1 - eps, color="#596273", ls="--", lw=1.2)
    ax2.axvline(r, color="#8a5ab8", ls="--", lw=1.2)
    ax2.scatter([r], [cumulative[r - 1]], s=80, color="#8a5ab8", zorder=4)
    ax2.text(r + 0.4, 0.80, f"$r_{{.05}}={r}$", color="#6f429c", fontsize=11)
    ax2.text(1.2, 0.955, "$1-\\varepsilon$", color="#374151", va="bottom")
    ax2.set_ylim(0.25, 1.015)
    ax2.set_xlim(1, 16)
    ax2.set_xlabel("retained singular directions $r$")
    ax2.set_ylabel("cumulative functional energy")
    ax2.set_title("Energy-threshold rank")
    fig.suptitle("Domain-conditioned functional rank: measurement pipeline and estimand", fontsize=16, y=0.98)
    save(fig, "hr_method_pipeline.png")


def matched_state() -> pd.DataFrame:
    d = read("d10_5_integrated_state_equal7.csv")
    return d[
        (d["epsilon"].eq(0.05))
        & d["arm"].isin(ARM_ORDER)
        & d["probe_name"].isin(CORE)
    ].copy()


def plot_matched_trajectories() -> None:
    d = matched_state()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), sharey=True)
    for ax, model in zip(axes, ["qwen", "llama"]):
        dm = d[d.model.eq(model)]
        steps = sorted(dm.checkpoint.unique())
        x = np.arange(len(steps) + 1)
        labels = [0] + steps
        for arm in ARM_ORDER:
            da = dm[dm.arm.eq(arm)]
            for probe in CORE:
                y = [0.0]
                vals = da[da.probe_name.eq(probe)].set_index("checkpoint")["state_rank_delta_mean"]
                y.extend([vals.get(step, np.nan) for step in steps])
                ax.plot(x, y, color=ARM_COLOR[arm], alpha=0.20, lw=1.05)
            means = da.groupby("checkpoint")["state_rank_delta_mean"].mean()
            y_mean = [0.0] + [means.get(step, np.nan) for step in steps]
            ax.plot(
                x,
                y_mean,
                color=ARM_COLOR[arm],
                lw=3,
                marker="o",
                ms=5,
                label=ARM_LABEL[arm],
            )
        ax.axhline(0, color="#60666f", ls="--", lw=1.1)
        ax.set_xticks(x, labels, rotation=42)
        ax.set_xlabel("checkpoint")
        ax.set_title(MODEL_LABEL[model])
        ax.text(
            0.02,
            0.04,
            "thin = individual core domains\nthick = equal-domain mean",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#4b5563",
        )
    axes[0].set_ylabel("$\\Delta r^{(7)}_{.05}$ relative to step 0")
    axes[1].legend(loc="upper right", ncol=2)
    fig.suptitle("Matched four-core trajectories: arm separation is shared, temporal shape is model-dependent", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "hr_matched_domain_trajectories.png")


def plot_dominance_ncd() -> None:
    d = read("d10_5_dominance_ncd.csv")
    fig = plt.figure(figsize=(15.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.0], wspace=0.35)
    for col, model in enumerate(["qwen", "llama"]):
        scope = "d4_four_core_qwen" if model == "qwen" else "d10_four_core_llama"
        x = d[d.scope.eq(scope)].copy()
        piv = x.pivot(index="probe_name", columns="checkpoint", values="rank_margin_nearest_offline")
        piv = piv.reindex(CORE)
        ax = fig.add_subplot(gs[0, col])
        annotated_heatmap(
            ax,
            piv.to_numpy(float),
            [PROBE_LABEL[p] for p in piv.index],
            [str(int(v)) for v in piv.columns],
            title=f"{MODEL_LABEL[model]}\nOPD margin to nearest offline arm",
            cmap="RdBu_r",
            center=0,
            fmt=".1f",
            cbar_label="directions; >0 means OPD is deeper",
        )
        ax.set_xlabel("checkpoint")
    ncd = d[d.scope.eq("ncd_eps05_logtime_state_rank_delta")].copy()
    ax = fig.add_subplot(gs[0, 2])
    width = 0.36
    xpos = np.arange(len(ARM_ORDER))
    for i, model in enumerate(["qwen", "llama"]):
        vals = (
            ncd[ncd.model.eq(model)]
            .set_index("arm")
            .reindex(ARM_ORDER)["ncd"]
            .to_numpy(float)
        )
        bars = ax.bar(
            xpos + (i - 0.5) * width,
            vals,
            width,
            color=["#7187a5", "#aeb9c8"][i],
            edgecolor="#303846",
            label=model.capitalize(),
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.18, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos, [ARM_LABEL[a] for a in ARM_ORDER], rotation=22)
    ax.set_ylabel("NCD through shared horizon $T=320$")
    ax.set_title("Negative-contraction dose")
    ax.legend()
    fig.suptitle("OPD early contraction dominance: 23/24 cells and the largest cross-horizon NCD", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "hr_dominance_ncd.png")


def plot_contraction_output() -> None:
    state = read("d10_5_integrated_state_equal7.csv")
    state = state[
        state.epsilon.eq(0.05)
        & state.arm.isin(ARM_ORDER)
        & state.probe_name.isin(CORE)
    ].copy()
    outputs = read("d10_5_integrated_outputs.csv")
    outputs = outputs[
        outputs.arm.isin(ARM_ORDER)
        & outputs.probe_name.isin(CORE)
    ].copy()
    d = state.merge(
        outputs[
            [
                "model",
                "arm",
                "checkpoint",
                "probe_name",
                "cumulative_kl_base_to_current",
                "delta_nll_cumulative",
                "absolute_delta_nll_cumulative",
            ]
        ],
        on=["model", "arm", "checkpoint", "probe_name"],
        how="inner",
    )
    d["c_epsilon"] = d["relative_functional_contraction_equal7"]
    corr = read("d10_5_checkpoint_demeaned_correlations.csv")
    corr = corr[corr.epsilon.eq(0.05)].copy()
    targets = [
        ("cumulative_kl_base_to_current", "cumulative KL"),
        ("absolute_delta_nll_cumulative", "absolute $\\Delta$NLL"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), gridspec_kw={"width_ratios": [1, 1, 0.82]})
    for row, model in enumerate(["llama", "qwen"]):
        dm = d[d.model.eq(model)]
        for col, (target, title) in enumerate(targets):
            ax = axes[row, col]
            rhos = []
            for arm in ARM_ORDER:
                da = dm[dm.arm.eq(arm)]
                ax.scatter(
                    100 * da.c_epsilon,
                    da[target],
                    s=30,
                    alpha=0.74,
                    color=ARM_COLOR[arm],
                    edgecolor="white",
                    linewidth=0.3,
                    label=ARM_LABEL[arm],
                )
                rhos.append(rank_corr(da.c_epsilon, da[target]))
            ax.axvline(0, color="#7a7f87", lw=0.8, ls="--")
            ax.set_xlabel("$100\\,c_{.05}$ (% baseline rank)")
            ax.set_ylabel(title)
            ax.set_title(f"{MODEL_LABEL[model]} · {title}\nwithin-arm $\\rho_s$ range {min(rhos):.2f}–{max(rhos):.2f}")
            if row == 0 and col == 0:
                ax.legend(ncol=2, fontsize=8)
        ax = axes[row, 2]
        vals = []
        for target, title in targets:
            target_name = (
                "demean_cumulative_kl_base_to_current"
                if target.startswith("cumulative")
                else "demean_absolute_delta_nll_cumulative"
            )
            z = corr[(corr.model.eq(model)) & (corr.target.eq(target_name))]
            vals.append(float(z.spearman.iloc[0]))
        bars = ax.bar(["KL", "|$\\Delta$NLL|"], vals, color=["#4c78a8", "#f28e2b"], width=0.62)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
        ax.axhline(0, color="#555", lw=0.9)
        ax.set_ylim(-0.1, 0.9)
        ax.set_ylabel("checkpoint-demeaned $\\rho_s$")
        ax.set_title(f"{model.capitalize()}\nremove model×checkpoint mean")
    fig.suptitle("Relative functional contraction tracks unsigned fixed-token output departure", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "hr_contraction_output_departure.png")


def plot_incremental_baselines() -> None:
    d = read("d11_same_cell_incremental_comparison.csv")
    pooled = d[d.model_scope.eq("pooled")].copy()
    feature_sets = ["W", "p_k", "C", "W_plus_C", "p_k_plus_C", "TPNT", "TPNT_plus_C"]
    labels = ["raw W", "$p_k$", "$C$", "W+$C$", "$p_k+C$", "TPNT", "TPNT+$C$"]
    targets = [
        ("cumulative_kl_base_to_current", "KL $R^2$"),
        ("absolute_delta_nll_cumulative", "|$\\Delta$NLL| $R^2$"),
        ("delta_nll_cumulative", "signed $\\Delta$NLL $R^2$"),
        ("is_opd", "OPD AUC"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4), sharey=False)
    colors = ["#9aa3ad", "#315f8c", "#c43c35", "#8297b0", "#6f4c9b", "#57a17b", "#b07aa1"]
    for ax, (target, title) in zip(axes, targets):
        analysis = "checkpoint_grouped_opd_vs_nonopd_macro" if target == "is_opd" else "checkpoint_grouped_regression"
        x = pooled[(pooled.analysis.eq(analysis)) & (pooled.target.eq(target))].set_index("feature_set")
        metric = "auc" if target == "is_opd" else "heldout_r2"
        vals = [float(x.loc[f, metric]) for f in feature_sets]
        bars = ax.bar(np.arange(len(vals)), vals, color=colors, width=0.76)
        ax.axhline(0 if metric == "heldout_r2" else 0.5, color="#5f6368", ls="--", lw=1)
        ax.set_xticks(np.arange(len(vals)), labels, rotation=50, ha="right")
        ax.set_title(title)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + (0.02 if val >= 0 else -0.045),
                f"{val:.2f}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7.5,
            )
        if metric == "auc":
            ax.set_ylim(0.42, 0.98)
        else:
            ax.set_ylim(min(-0.17, min(vals) - 0.05), max(vals) + 0.13)
    fig.suptitle("Same-cell held-out comparison: functional contraction is complementary to strong $p_k$, not universally superior", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "hr_incremental_weight_baselines.png")


def plot_exposure_interventions() -> None:
    formal = matched_state()
    alpha = read("qwen_alpha05_r_epsilon.csv")
    alpha = alpha[
        alpha.layer.eq(18)
        & alpha.epsilon.eq(0.05)
        & alpha.probe.isin(["E_ood", "E_if"])
        & alpha.step.isin([5, 20, 40, 80, 160])
    ]
    q = formal[
        formal.model.eq("qwen")
        & formal.arm.isin(["opd", "offkd"])
        & formal.probe_name.isin(["E_ood", "E_if"])
        & formal.checkpoint.isin([5, 20, 40, 80, 160])
    ]
    qmean = q.groupby(["arm", "checkpoint"], as_index=False)["state_rank_delta_mean"].mean()
    amean = alpha.groupby("step", as_index=False)["r_epsilon_delta"].mean()

    le = read("llama_early_320_r_epsilon.csv")
    lf = read("llama_frozen_self_r_epsilon.csv")
    filt = lambda z: z.layer.eq(14) & z.epsilon.eq(0.05) & z.step.isin([20, 40, 80, 160, 320])
    lopd = (
        le[filt(le) & le.arm.eq("opd")]
        .groupby(["step", "probe"], as_index=False)["delta_from_base"]
        .mean()
    )
    lfro = (
        lf[filt(lf) & lf.arm.eq("frozen_self")]
        .groupby(["step", "probe"], as_index=False)["delta_from_base"]
        .mean()
    )
    lm = lopd.merge(lfro, on=["step", "probe"], suffixes=("_opd", "_frozen"))
    lm["margin"] = lm.delta_from_base_frozen - lm.delta_from_base_opd

    fig = plt.figure(figsize=(16, 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    steps = [5, 20, 40, 80, 160]
    for arm in ["offkd", "opd"]:
        x = qmean[qmean.arm.eq(arm)].set_index("checkpoint").reindex(steps)
        ax.plot(steps, x.state_rank_delta_mean, marker="o", lw=2.6, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.plot(steps, amean.set_index("step").reindex(steps).r_epsilon_delta, marker="D", lw=2.6, color=ARM_COLOR["alpha05"], label="$\\alpha=.5$")
    ax.axhline(0, color="#666", ls="--", lw=1)
    ax.set_title(
        "Qwen intervention\nformal endpoints + legacy $\\alpha=.5$\nE_ood / E_if mean; ordering only",
        fontsize=10.5,
    )
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("$\\Delta r^{(7)}_{.05}$")
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    for name, col, color in [
        ("OPD", "delta_from_base_opd", ARM_COLOR["opd"]),
        ("frozenSelf0-KD", "delta_from_base_frozen", ARM_COLOR["frozen_self"]),
    ]:
        x = lm.groupby("step")[col].mean().reindex([20, 40, 80, 160, 320])
        ax.plot(x.index, x.values, marker="o", lw=2.7, color=color, label=name)
    ax.axhline(0, color="#666", ls="--", lw=1)
    ax.set_title("Llama current-refresh control\nsix-probe equal mean", fontsize=10.5)
    ax.set_xlabel("checkpoint")
    ax.set_ylabel("$\\Delta r^{(7)}_{.05}$")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    porder = ["S_math", "E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general"]
    piv = lm.pivot(index="probe", columns="step", values="margin").reindex(porder)
    annotated_heatmap(
        ax,
        piv.to_numpy(float),
        [PROBE_LABEL[p] for p in porder],
        [str(int(v)) for v in piv.columns],
        title="frozenSelf − OPD margin\npositive means current OPD is deeper",
        cmap="RdBu_r",
        center=0,
        fmt=".1f",
        cbar_label="directions",
    )
    ax.set_xlabel("checkpoint")
    ax.title.set_fontsize(10.5)
    fig.suptitle("On-policy exposure interventions: mixture movement and current-refresh separation", fontsize=15, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "hr_exposure_frozen_interventions.png")


def plot_signed_branch() -> None:
    raw = read("d10_5_output_correlations.csv")
    raw = raw[
        raw.epsilon.eq(0.05)
        & raw.target.eq("delta_nll_cumulative")
        & raw.arm.isin(ARM_ORDER)
    ]
    residual = read("d10_5_logstep_detrended_signed_nll.csv")
    residual = residual[residual.epsilon.eq(0.05) & residual.arm.isin(ARM_ORDER)]
    interaction = read("d10_5_signed_nll_interaction.csv")
    interaction = interaction[interaction.epsilon.eq(0.05)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    width = 0.36
    xpos = np.arange(len(ARM_ORDER))
    for ax, frame, title in [
        (axes[0], raw, "Raw cumulative alignment"),
        (axes[1], residual, "After removing log-step trend"),
    ]:
        for i, model in enumerate(["llama", "qwen"]):
            vals = frame[frame.model.eq(model)].set_index("arm").reindex(ARM_ORDER)["spearman"].to_numpy(float)
            ax.bar(
                xpos + (i - 0.5) * width,
                vals,
                width,
                color=["#92a8c6", "#e0a37b"][i],
                edgecolor="#343a40",
                label=model.capitalize(),
            )
        ax.axhline(0, color="#5f6368", lw=1)
        ax.set_xticks(xpos, [ARM_LABEL[a] for a in ARM_ORDER], rotation=22)
        ax.set_ylabel("Spearman $\\rho_s(c_{.05},\\Delta\\mathrm{NLL})$")
        ax.set_ylim(-0.35, 0.95)
        ax.set_title(title)
    axes[0].legend()

    ax = axes[2]
    for i, model in enumerate(["llama", "qwen"]):
        z = interaction[interaction.model.eq(model)].iloc[0]
        beta = z.opd_c_interaction
        lo = z.grouped_bootstrap_ci_low
        hi = z.grouped_bootstrap_ci_high
        ax.errorbar(
            [i],
            [beta],
            yerr=[[beta - lo], [hi - beta]],
            fmt="o",
            ms=8,
            capsize=5,
            lw=2,
            color=["#4c78a8", "#f28e2b"][i],
        )
        ax.text(i + 0.06, beta, f"{beta:.2f}\n[{lo:.2f}, {hi:.2f}]", va="center", fontsize=9)
    ax.axhline(0, color="#5f6368", lw=1, ls="--")
    ax.set_xticks([0, 1], ["Llama", "Qwen"])
    ax.set_xlim(-0.45, 1.55)
    ax.set_ylabel("OPD × $c_{.05}$ interaction")
    ax.set_title("Grouped-bootstrap interaction")
    fig.suptitle("Signed readout is a model-dependent branch, not a cross-model OPD-only law", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "hr_signed_readout_branch.png")


def terminal_behavior_gap() -> pd.DataFrame:
    rows = []
    # Qwen seqKD was delivered in block2, while the three-arm campaign holds
    # the matched off-KD endpoint. Extraction failures have their own audit.
    qseq = read("block2_final_g2_behavior.csv")
    qseq = qseq[(qseq.step.eq(624)) & qseq.arm.eq("seqkd")].iloc[0]
    qoff_all = pd.read_csv(MINI.parent / "offkd" / "three_arm_full_trajectory.csv")
    qoff = qoff_all[(qoff_all.step.eq(624)) & qoff_all.arm.eq("offkd")].iloc[0]
    qfail = read("S1_mmlupro_extract_audit.csv")
    qfail = qfail[qfail.step.eq(624)].set_index("arm")
    qmetrics = {
        "MATH acc": (qoff.math500_acc, qseq.math500_acc),
        "MATH cap-hit": (qoff.math500_trunc_rate, qseq.math500_trunc_rate),
        "MMLU strict": (qoff.mmlu_pro_exact_match, qseq.mmlu_pro_exact_match),
        "MMLU fail": (qfail.loc["offkd", "extract_fail_rate"], qfail.loc["seqkd", "extract_fail_rate"]),
        "IFEval instr.": (qoff.ifeval_instruction_strict, qseq.ifeval_instruction_strict),
    }
    for label, (off_value, seq_value) in qmetrics.items():
        rows.append(("Qwen", label, 100 * (seq_value - off_value)))
    l = read("llama_early_320_behavior.csv")
    l = l[(l.step.eq(320)) & l.arm.isin(["offkd", "seqkd"])]
    get = lambda arm, task, col: float(l[(l.arm.eq(arm)) & (l.task.eq(task))][col].iloc[0])
    lmetrics = {
        "MATH acc": ("math500", "accuracy"),
        "MATH cap-hit": ("math500", "cap_hit_rate"),
        "MMLU strict": ("mmlu_pro", "strict_accuracy"),
        "MMLU fail": ("mmlu_pro", "extract_failure_rate"),
        "IFEval instr.": ("ifeval", "instruction_strict_accuracy"),
    }
    for label, (task, col) in lmetrics.items():
        rows.append(("Llama", label, 100 * (get("seqkd", task, col) - get("offkd", task, col))))
    return pd.DataFrame(rows, columns=["model", "metric", "gap"])


def plot_support_readout() -> None:
    d = matched_state()
    fig = plt.figure(figsize=(16, 5.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.33)
    for col, model in enumerate(["qwen", "llama"]):
        x = (
            d[(d.model.eq(model)) & d.arm.isin(["offkd", "seqkd"])]
            .pivot(index=["checkpoint", "probe_name"], columns="arm", values="state_rank_delta_mean")
            .dropna()
        )
        ax = fig.add_subplot(gs[0, col])
        for probe in CORE:
            xp = x.xs(probe, level="probe_name")
            ax.scatter(xp.offkd, xp.seqkd, s=48, alpha=0.78, label=PROBE_LABEL[probe])
        lo = min(x.offkd.min(), x.seqkd.min()) - 1
        hi = max(x.offkd.max(), x.seqkd.max()) + 1
        ax.plot([lo, hi], [lo, hi], color="#61656b", ls="--", lw=1)
        pearson = float(x.offkd.corr(x.seqkd))
        mae = float((x.offkd - x.seqkd).abs().mean())
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("off-KD $\\Delta r^{(7)}_{.05}$")
        ax.set_ylabel("seqKD $\\Delta r^{(7)}_{.05}$")
        ax.set_title(f"{MODEL_LABEL[model]}\nfour-core path: Pearson={pearson:.3f}, MAE={mae:.2f}")
        if col == 0:
            ax.legend(fontsize=7.5)
    gaps = terminal_behavior_gap()
    ax = fig.add_subplot(gs[0, 2])
    metrics = gaps.metric.drop_duplicates().tolist()
    xpos = np.arange(len(metrics))
    width = 0.36
    for i, model in enumerate(["Qwen", "Llama"]):
        vals = gaps[gaps.model.eq(model)].set_index("metric").reindex(metrics).gap.to_numpy(float)
        ax.bar(
            xpos + (i - 0.5) * width,
            vals,
            width,
            label=model,
            color=["#92a8c6", "#e0a37b"][i],
            edgecolor="#343a40",
        )
    ax.axhline(0, color="#5f6368", lw=1)
    ax.set_xticks(xpos, metrics, rotation=42, ha="right")
    ax.set_ylabel("seqKD − off-KD terminal gap (percentage points)")
    ax.set_title("Matched support, different target\nbehavioral readout need not be locked")
    ax.legend()
    fig.suptitle("Support–readout separation: similar functional-rank paths are not behaviorally sufficient", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "hr_support_readout_separation.png")


def plot_rank_robustness() -> None:
    le = read("llama_early_320_r_epsilon.csv")
    l = le[
        le.arm.eq("opd")
        & le.step.eq(160)
        & le.probe.isin(CORE)
    ]
    lh = l.groupby(["layer", "epsilon"])["delta_from_base"].mean().unstack("epsilon").sort_index()

    q = read("C14_per_checkpoint_layer_sensitivity.csv")
    qprobes = ["E_general", "E_math_hard", "E_ood"]
    q = q[q.arm.eq("opd") & q.step.eq(160) & q.probe_family.isin(qprobes)]

    module = read("d10_5_integrated_state_module.csv")
    module = module[
        module.epsilon.eq(0.05)
        & module.arm.eq("opd")
        & module.checkpoint.eq(160)
        & module.probe_name.isin(CORE)
    ]
    mp = module.groupby(["model", "module"])["state_rank_delta"].mean().reset_index()
    module_order = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    short = ["q", "k", "v", "o", "gate", "up", "down"]

    fig = plt.figure(figsize=(16, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1, 1.15], wspace=0.34)
    ax = fig.add_subplot(gs[0, 0])
    annotated_heatmap(
        ax,
        lh.to_numpy(float),
        [f"L{v}" for v in lh.index],
        [f"$\\varepsilon$={v:g}" for v in lh.columns],
        title="Llama OPD @160\nfour-core mean",
        cmap="RdBu_r",
        center=0,
        fmt=".1f",
        cbar_label="$\\Delta r_\\varepsilon$",
    )

    ax = fig.add_subplot(gs[0, 1])
    for probe, marker in zip(qprobes, ["o", "s", "^"]):
        z = q[q.probe_family.eq(probe)].sort_values("layer")
        ax.plot(
            z.layer,
            z.r_epsilon_delta_mean,
            marker=marker,
            lw=1.8,
            label={"E_general": "General", "E_math_hard": "AIME24", "E_ood": "MMLU-Pro"}[probe],
        )
    mean = q.groupby("layer").r_epsilon_delta_mean.mean()
    ax.plot(mean.index, mean.values, color="#222", lw=3.0, marker="D", label="mean")
    ax.axhline(0, color="#666", ls="--", lw=1)
    ax.set_xticks(sorted(q.layer.unique()), [f"L{v}" for v in sorted(q.layer.unique())])
    ax.set_ylabel("$\\Delta r_{.05}$")
    ax.set_title("Qwen OPD @160\nlayer sensitivity")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    xpos = np.arange(len(module_order))
    for model in ["qwen", "llama"]:
        z = mp[mp.model.eq(model)].set_index("module").reindex(module_order)
        ax.plot(
            xpos,
            z.state_rank_delta,
            marker="o",
            lw=2.5,
            label=model.capitalize(),
            color={"qwen": "#4c78a8", "llama": "#f28e2b"}[model],
        )
    ax.axhline(0, color="#666", ls="--", lw=1)
    ax.set_xticks(xpos, short, rotation=30)
    ax.set_ylabel("mean core-domain $\\Delta r_{.05}$")
    ax.set_title("Headline layer @160\nmodule decomposition")
    ax.legend()
    fig.suptitle("Rank robustness audit: threshold, layer, and seven-module decompositions", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "hr_rank_robustness.png")


def plot_full_domain_small_multiples() -> None:
    d = matched_state()
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 14.0), sharey=True)
    for row, probe in enumerate(CORE):
        for col, model in enumerate(["qwen", "llama"]):
            ax = axes[row, col]
            z = d[(d.model.eq(model)) & d.probe_name.eq(probe)]
            steps = sorted(z.checkpoint.unique())
            xpos = np.arange(len(steps) + 1)
            for arm in ARM_ORDER:
                s = z[z.arm.eq(arm)].set_index("checkpoint").state_rank_delta_mean
                vals = [0.0] + [s.get(step, np.nan) for step in steps]
                ax.plot(
                    xpos,
                    vals,
                    marker="o",
                    ms=4,
                    lw=2.2,
                    color=ARM_COLOR[arm],
                    label=ARM_LABEL[arm],
                )
            ax.axhline(0, color="#666", ls="--", lw=0.9)
            ax.set_xticks(xpos, [0] + steps, rotation=38)
            if row == 0:
                ax.set_title(MODEL_LABEL[model])
            if col == 0:
                ax.set_ylabel(f"{PROBE_LABEL[probe]}\n$\\Delta r^{{(7)}}_{{.05}}$")
            if row == 3:
                ax.set_xlabel("checkpoint")
    axes[0, 1].legend(loc="upper right", ncol=2)
    fig.suptitle("Complete matched four-domain trajectories: no mean curve can hide a domain-specific exception", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "hr_full_domain_small_multiples.png")


def plot_weight_audit() -> None:
    e7 = read("d11_e7_spectrum_matched_null_summary.csv")
    e5 = read("d11_e5_layer_robustness_summary.csv")
    realnull = (
        e7.groupby(["model", "arm"])[["mean_real_overlap_lift", "mean_null_overlap_lift", "mean_z_tpnt"]]
        .mean()
        .reset_index()
    )
    pn = (
        e5.groupby(["model", "arm"])[["mean_pabs_joint_cos", "mean_nss_l1_top32"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    xpos = np.arange(len(ARM_ORDER))
    markers = {"qwen": "o", "llama": "s"}
    for model in ["qwen", "llama"]:
        z = realnull[realnull.model.eq(model)].set_index("arm").reindex(ARM_ORDER)
        axes[0].plot(
            xpos,
            z.mean_real_overlap_lift,
            marker=markers[model],
            lw=2.2,
            label=f"{model.capitalize()} real",
        )
        axes[0].plot(
            xpos,
            z.mean_null_overlap_lift,
            marker=markers[model],
            lw=1.5,
            ls="--",
            alpha=0.75,
            label=f"{model.capitalize()} null",
        )
    axes[0].set_ylabel("TPNT overlap lift")
    axes[0].set_title("TPNT real vs spectrum-matched null")
    axes[0].legend(fontsize=7.5, ncol=2)

    width = 0.36
    for i, model in enumerate(["qwen", "llama"]):
        z = pn[pn.model.eq(model)].set_index("arm").reindex(ARM_ORDER)
        axes[1].bar(
            xpos + (i - 0.5) * width,
            1e4 * (1 - z.mean_pabs_joint_cos.to_numpy(float)),
            width,
            label=model.capitalize(),
            color=["#92a8c6", "#e0a37b"][i],
            edgecolor="#343a40",
        )
        axes[2].bar(
            xpos + (i - 0.5) * width,
            1e5 * z.mean_nss_l1_top32.to_numpy(float),
            width,
            label=model.capitalize(),
            color=["#92a8c6", "#e0a37b"][i],
            edgecolor="#343a40",
        )
    axes[1].set_ylabel("$10^4(1-\\mathrm{PABS\\ cosine})$")
    axes[1].set_title("PABS: nearly identical joint subspaces")
    axes[2].set_ylabel("$10^5\\times$ NSS top-32 L1")
    axes[2].set_title("NSS: very small normalized spectrum shift")
    for ax in axes:
        ax.set_xticks(xpos, [ARM_LABEL[a] for a in ARM_ORDER], rotation=25)
    axes[1].legend()
    fig.suptitle("Weight-space audit: current TPNT/PABS/NSS signals are weakly training-specific in this LoRA setting", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "hr_weight_space_audit.png")


def plot_general_adjusted_v3() -> None:
    a = read("qwen_alpha05_r_epsilon.csv")
    a = a[a.layer.eq(18) & a.epsilon.eq(0.05) & a.step.isin([5, 20, 40, 80, 160])]
    mean = a.groupby(["probe", "step"])["r_epsilon_delta"].mean().unstack("step")
    general = mean.loc["E_general"]
    order = ["E_if", "E_math", "E_math_hard_v2", "E_ood", "S_math"]
    g = mean.loc[order].subtract(general, axis=1)

    # Frozen derived statistics from Appendix B.7. They were produced by the
    # trajectory-block bootstrap audit and currently have no standalone CSV.
    v3 = pd.DataFrame(
        {
            "domain": ["IFEval", "MMLU-Pro", "MATH", "IFEval", "MMLU-Pro", "MATH"],
            "model": ["Qwen", "Qwen", "Qwen", "Llama", "Llama", "Llama"],
            "rho": [0.574, 0.341, -0.491, 0.581, -0.023, 0.718],
            "lo": [0.374, -0.064, -0.709, 0.390, -0.436, 0.441],
            "hi": [0.736, 0.730, -0.334, 0.901, 0.192, 0.886],
        }
    )

    fig = plt.figure(figsize=(16, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 0.7], wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    annotated_heatmap(
        ax,
        g.to_numpy(float),
        [PROBE_LABEL[p] for p in order],
        [str(int(v)) for v in g.columns],
        title="Qwen $\\alpha=.5$: general-adjusted trajectory",
        cmap="RdBu_r",
        center=0,
        fmt=".1f",
        cbar_label="$G_D=\\Delta r_D-\\Delta r_{general}$",
    )
    ax.set_xlabel("checkpoint")

    ax = fig.add_subplot(gs[0, 1])
    domains = ["IFEval", "MMLU-Pro", "MATH"]
    xpos = np.arange(len(domains))
    for i, model in enumerate(["Qwen", "Llama"]):
        z = v3[v3.model.eq(model)].set_index("domain").reindex(domains)
        shift = (i - 0.5) * 0.22
        y = z.rho.to_numpy(float)
        lo = z.lo.to_numpy(float)
        hi = z.hi.to_numpy(float)
        ax.errorbar(
            xpos + shift,
            y,
            yerr=[y - lo, hi - y],
            fmt="o",
            ms=8,
            capsize=4,
            lw=2,
            label=model,
            color=["#4c78a8", "#f28e2b"][i],
        )
    ax.axhline(0, color="#666", lw=1, ls="--")
    ax.set_xticks(xpos, domains)
    ax.set_ylim(-0.9, 1.05)
    ax.set_ylabel("Spearman: $V^{(3)}$ vs behavior drawdown")
    ax.set_title("Recent reorganization load\ndomain-level stress test")
    ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    did = [-0.070, 0.294]
    bars = ax.bar(["Qwen", "Llama"], did, color=["#4c78a8", "#f28e2b"], width=0.62)
    ax.axhline(0, color="#666", lw=1)
    for bar, val in zip(bars, did):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.025 if val >= 0 else -0.055),
            f"{val:+.3f}",
            ha="center",
            va="bottom" if val >= 0 else "top",
        )
    ax.set_ylim(-0.18, 0.38)
    ax.set_ylabel("OPD − off-KD $V^{(3)}$ DiD")
    ax.set_title("No shared on-policy\ncausal direction")
    fig.suptitle("Secondary diagnostics: domain reallocation and recent functional reorganization", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "hr_general_adjusted_v3.png")


def main() -> None:
    setup_style()
    plot_method_pipeline()
    plot_matched_trajectories()
    plot_dominance_ncd()
    plot_contraction_output()
    plot_incremental_baselines()
    plot_exposure_interventions()
    plot_signed_branch()
    plot_support_readout()
    plot_rank_robustness()
    plot_full_domain_small_multiples()
    plot_weight_audit()
    plot_general_adjusted_v3()
    print(f"Generated 12 figures in {OUT}")


if __name__ == "__main__":
    main()
