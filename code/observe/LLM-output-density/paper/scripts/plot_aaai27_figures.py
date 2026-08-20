from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib import font_manager
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


REPO_ROOT = Path(__file__).resolve().parents[2]
MINI = REPO_ROOT / "paper_artifacts/cycle09/mini"
EQUAL5 = MINI / "equal5_non_qk"
FAT = MINI / "fat_outlink_round1_v2_link_equal5"
OUT_ZH = REPO_ROOT / "paper/zh/figures"
OUT_EN = REPO_ROOT / "paper/en/figures"

CORE_PROBES = ["E_general", "E_mathHeld", "E_mmluPro", "E_ifeval"]
PROBE_LABELS = {
    "E_general": "General",
    "E_mathHeld": "Held-out math",
    "E_mmluPro": "MMLU-Pro",
    "E_ifeval": "IFEval",
    "S_math": "Train support",
    "E_math": "Held-out math",
    "E_math_hard_v2": "AIME25",
    "E_ood": "MMLU-Pro",
    "E_if": "IFEval",
}
ARMS = ["offkd", "seqkd", "sft", "opd"]
ARM_LABELS = {
    "offkd": "off-KD",
    "seqkd": "seqKD",
    "sft": "SFT",
    "opd": "OPD",
    "alpha05": r"$\alpha=.5$",
    "frozen_self": "frozenSelf0-KD",
}
ARM_COLORS = {
    "offkd": "#0072B2",
    "seqkd": "#006B50",
    "sft": "#4D4D4D",
    "opd": "#9C4300",
    "alpha05": "#91476F",
    "frozen_self": "#8C564B",
}
ARM_MARKERS = {
    "offkd": "o",
    "seqkd": "s",
    "sft": "^",
    "opd": "D",
    "alpha05": "P",
    "frozen_self": "X",
}
LINE_STYLES = {
    "offkd": "-",
    "seqkd": "--",
    "sft": "-.",
    "opd": "-",
    "alpha05": ":",
    "frozen_self": "--",
}
NON_QK = {
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


def configure_style() -> None:
    for font_file in [
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf",
        "/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf",
    ]:
        if Path(font_file).is_file():
            font_manager.fontManager.addfont(font_file)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial"],
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.1,
            "patch.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_vector(fig: plt.Figure, name: str) -> None:
    """Save a PDF, then outline fonts to avoid Type 3/CID font failures."""
    temporary = OUT_ZH / f".{name}.fonted.pdf"
    destination = OUT_ZH / f"{name}.pdf"
    fig.savefig(temporary, format="pdf", facecolor="white")
    subprocess.run(
        [
            "gs",
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dCompatibilityLevel=1.5",
            "-dNoOutputFonts",
            "-sDEVICE=pdfwrite",
            f"-sOutputFile={destination}",
            str(temporary),
        ],
        check=True,
    )
    temporary.unlink()
    OUT_EN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(destination, OUT_EN / destination.name)
    plt.close(fig)


def log_positions(steps: list[int]) -> np.ndarray:
    return np.log1p(np.asarray(steps, dtype=float))


def add_step_axis(ax: plt.Axes, steps: list[int]) -> None:
    ax.set_xticks(log_positions(steps), [str(step) for step in steps])
    ax.set_xlabel("Checkpoint")


def functional_trajectories() -> pd.DataFrame:
    frame = pd.read_csv(EQUAL5 / "EQUAL5_functional_trajectories.csv")
    frame["checkpoint"] = pd.to_numeric(frame["checkpoint"], errors="coerce")
    return frame[
        frame["arm"].isin(ARMS)
        & frame["probe_name"].isin(CORE_PROBES)
        & frame["epsilon"].eq(0.05)
    ].copy()


def plot_arm_trajectory(
    ax: plt.Axes,
    frame: pd.DataFrame,
    model: str,
    steps: list[int],
    title: str,
) -> None:
    selected = frame[frame["model"].eq(model)]
    for arm in ARMS:
        arm_frame = selected[selected["arm"].eq(arm)]
        for probe in CORE_PROBES:
            series = (
                arm_frame[arm_frame["probe_name"].eq(probe)]
                .set_index("checkpoint")
                .reindex(steps)
            )
            values = 100 * series["c_equal5"].to_numpy(float)
            ax.plot(
                log_positions(steps),
                values,
                color=ARM_COLORS[arm],
                linestyle=LINE_STYLES[arm],
                alpha=0.22,
                linewidth=0.7,
            )
        mean = (
            arm_frame.groupby("checkpoint")["c_equal5"]
            .mean()
            .reindex(steps)
            .to_numpy(float)
        )
        ax.plot(
            log_positions(steps),
            100 * mean,
            color=ARM_COLORS[arm],
            linestyle=LINE_STYLES[arm],
            marker=ARM_MARKERS[arm],
            markersize=4.0,
            markeredgewidth=0.5,
            label=ARM_LABELS[arm],
        )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.set_title(title)
    ax.set_ylabel(r"Relative contraction $c_{.05}$ (%)")
    add_step_axis(ax, steps)
    if model == "qwen":
        shown = [5, 20, 80, 320, 624]
        ax.set_xticks(log_positions(shown), [str(step) for step in shown])


def figure_main_trajectories() -> None:
    trajectories = functional_trajectories()
    dominance = pd.read_csv(EQUAL5 / "EQUAL5_dominance_cells.csv")
    dominance = dominance[dominance["epsilon"].eq(0.05)].copy()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.0, 2.65),
        gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]},
        constrained_layout=True,
    )
    plot_arm_trajectory(
        axes[0],
        trajectories,
        "qwen",
        [5, 10, 20, 40, 80, 160, 320, 480, 624],
        "(a) Qwen",
    )
    plot_arm_trajectory(
        axes[1],
        trajectories,
        "llama",
        [5, 20, 40, 80, 160, 320],
        "(b) Llama",
    )
    axes[0].legend(
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=0.7,
    )

    ax = axes[2]
    model_style = {
        "qwen": ("#D55E00", "o", "Qwen"),
        "llama": ("#0072B2", "s", "Llama"),
    }
    for model, (color, marker, label) in model_style.items():
        subset = dominance[dominance["model"].eq(model)]
        for step_index, step in enumerate([20, 40, 80]):
            values = subset[subset["checkpoint"].eq(step)]["continuous_margin"].to_numpy(float)
            offsets = np.linspace(-0.11, 0.11, len(values))
            ax.scatter(
                np.full(len(values), step_index) + offsets,
                values,
                color=color,
                marker=marker,
                s=22,
                linewidths=0.5,
                edgecolors="white",
                label=label if step_index == 0 else None,
                zorder=3,
            )
            mean = float(np.mean(values))
            center = step_index + (-0.16 if model == "qwen" else 0.16)
            ax.plot([center - 0.08, center + 0.08], [mean, mean], color=color, linewidth=1.8)
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(range(3), ["20", "40", "80"])
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("OPD dominance margin\n(directions)")
    ax.set_title("(c) Early common window")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    save_vector(fig, "fig1_main_trajectories")


def module_relative(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    chosen = frame[frame["module"].isin(NON_QK)].copy()
    chosen["c_module"] = -chosen[value_column] / chosen["base_r_epsilon"]
    group_columns = [
        column
        for column in ["arm", "step", "probe", "epsilon", "layer"]
        if column in chosen.columns
    ]
    return chosen.groupby(group_columns, as_index=False)["c_module"].mean()


def module_delta(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    chosen = frame[frame["module"].isin(NON_QK)].copy()
    group_columns = [
        column
        for column in ["arm", "step", "probe", "epsilon", "layer"]
        if column in chosen.columns
    ]
    return chosen.groupby(group_columns, as_index=False)[value_column].mean()


def figure_support_controls() -> None:
    trajectories = functional_trajectories()
    alpha_raw = pd.read_csv(MINI / "qwen_alpha05_r_epsilon.csv")
    alpha = module_relative(alpha_raw, "r_epsilon_delta")
    alpha = alpha[
        alpha["epsilon"].eq(0.05)
        & alpha["layer"].eq(18)
        & alpha["probe"].isin(["E_ood", "E_if"])
    ]

    llama_opd_raw = pd.read_csv(MINI / "llama_early_320_r_epsilon.csv")
    llama_frozen_raw = pd.read_csv(MINI / "llama_frozen_self_r_epsilon.csv")
    llama_opd_c = module_relative(llama_opd_raw, "delta_from_base")
    llama_frozen_c = module_relative(llama_frozen_raw, "delta_from_base")
    llama_opd_d = module_delta(llama_opd_raw, "delta_from_base")
    llama_frozen_d = module_delta(llama_frozen_raw, "delta_from_base")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), constrained_layout=True)

    ax = axes[0, 0]
    steps = [5, 20, 40, 80, 160]
    qformal = trajectories[
        trajectories["model"].eq("qwen")
        & trajectories["arm"].isin(["offkd", "opd"])
        & trajectories["probe_name"].isin(["E_mmluPro", "E_ifeval"])
    ]
    for arm in ["offkd", "opd"]:
        values = (
            qformal[qformal["arm"].eq(arm)]
            .groupby("checkpoint")["c_equal5"]
            .mean()
            .reindex(steps)
        )
        ax.plot(
            log_positions(steps),
            100 * values,
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            linestyle=LINE_STYLES[arm],
            markersize=4.2,
            markeredgewidth=0.5,
            label=ARM_LABELS[arm],
        )
    alpha_values = alpha.groupby("step")["c_module"].mean().reindex(steps)
    ax.plot(
        log_positions(steps),
        100 * alpha_values,
        color=ARM_COLORS["alpha05"],
        marker=ARM_MARKERS["alpha05"],
        linestyle=LINE_STYLES["alpha05"],
        markersize=4.2,
        markeredgewidth=0.5,
        label=ARM_LABELS["alpha05"],
    )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.set_title("(a) Qwen exposure ordering")
    ax.set_ylabel(r"Relative contraction $c_{.05}$ (%)")
    add_step_axis(ax, steps)
    ax.legend(frameon=False, ncol=3, columnspacing=0.7, handlelength=1.5)

    ax = axes[0, 1]
    external = ["E_math", "E_math_hard_v2", "E_ood", "E_if", "E_general"]
    steps_refresh = [20, 40, 80, 160, 320]
    for frame, arm in [(llama_opd_c, "opd"), (llama_frozen_c, "frozen_self")]:
        selected = frame[
            frame["epsilon"].eq(0.05)
            & frame["layer"].eq(14)
            & frame["arm"].eq(arm)
            & frame["probe"].isin(external)
            & frame["step"].isin(steps_refresh)
        ]
        values = selected.groupby("step")["c_module"].mean().reindex(steps_refresh)
        ax.plot(
            log_positions(steps_refresh),
            100 * values,
            color=ARM_COLORS[arm],
            marker=ARM_MARKERS[arm],
            linestyle=LINE_STYLES[arm],
            markersize=4.2,
            markeredgewidth=0.5,
            label=ARM_LABELS[arm],
        )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.set_title("(b) Llama current-refresh control")
    ax.set_ylabel(r"Relative contraction $c_{.05}$ (%)")
    add_step_axis(ax, steps_refresh)
    ax.legend(frameon=False, handlelength=1.5)

    ax = axes[1, 0]
    probe_order = ["E_general", "E_math", "E_math_hard_v2", "E_ood", "E_if", "S_math"]
    filter_common = lambda frame: frame[
        frame["epsilon"].eq(0.05)
        & frame["layer"].eq(14)
        & frame["probe"].isin(probe_order)
        & frame["step"].isin(steps_refresh)
    ]
    opd_delta = filter_common(llama_opd_d)
    opd_delta = opd_delta[opd_delta["arm"].eq("opd")].rename(
        columns={"delta_from_base": "delta_opd"}
    )
    frozen_delta = filter_common(llama_frozen_d)
    frozen_delta = frozen_delta[frozen_delta["arm"].eq("frozen_self")].rename(
        columns={"delta_from_base": "delta_frozen"}
    )
    margins = opd_delta.merge(
        frozen_delta[["step", "probe", "delta_frozen"]],
        on=["step", "probe"],
        how="inner",
    )
    margins["margin"] = margins["delta_frozen"] - margins["delta_opd"]
    pivot = (
        margins.pivot(index="probe", columns="step", values="margin")
        .reindex(index=probe_order, columns=steps_refresh)
    )
    vmax = float(np.nanmax(np.abs(pivot.to_numpy(float))))
    image = ax.imshow(
        pivot.to_numpy(float),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
        aspect="auto",
    )
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = float(pivot.iloc[row, column])
            color = "white" if abs(value) > 0.58 * vmax else "black"
            ax.text(column, row, f"{value:.1f}", ha="center", va="center", color=color)
    ax.set_xticks(range(len(steps_refresh)), [str(step) for step in steps_refresh])
    ax.set_yticks(range(len(probe_order)), [PROBE_LABELS[p] for p in probe_order])
    ax.set_xlabel("Checkpoint")
    ax.set_title("(c) frozenSelf0-KD minus OPD margin")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Directions")

    ax = axes[1, 1]
    matched = trajectories[
        trajectories["arm"].isin(["offkd", "seqkd"])
        & trajectories["checkpoint"].gt(0)
    ]
    for model, color, marker in [
        ("qwen", "#D55E00", "o"),
        ("llama", "#0072B2", "s"),
    ]:
        paired = (
            matched[matched["model"].eq(model)]
            .pivot(
                index=["checkpoint", "probe_name"],
                columns="arm",
                values="delta_r_equal5",
            )
            .dropna()
        )
        pearson = float(paired["offkd"].corr(paired["seqkd"]))
        mae = float((paired["offkd"] - paired["seqkd"]).abs().mean())
        ax.scatter(
            paired["offkd"],
            paired["seqkd"],
            s=24,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.82,
            label=f"{model.capitalize()}: r={pearson:.3f}, MAE={mae:.2f}",
        )
    limits = ax.get_xlim()
    lower = min(limits[0], ax.get_ylim()[0])
    upper = max(limits[1], ax.get_ylim()[1])
    ax.plot([lower, upper], [lower, upper], color="#555555", linestyle="--", linewidth=0.7)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"off-KD $\Delta r_{.05}^{(5)}$")
    ax.set_ylabel(r"seqKD $\Delta r_{.05}^{(5)}$")
    ax.set_title("(d) Matched support, different target")
    ax.grid(color="#D9D9D9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.3)
    inset = inset_axes(ax, width="34%", height="34%", loc="lower right", borderpad=0.8)
    inset.bar(
        [0, 1],
        [68.2, 2.8],
        color=["#D55E00", "#0072B2"],
        edgecolor="#333333",
        linewidth=0.7,
    )
    inset.axhline(0, color="#555555", linewidth=0.7)
    inset.set_xticks([0, 1], ["Qwen", "Llama"], rotation=20)
    inset.set_ylabel(r"$\Delta$ cap-hit (pp)")
    inset.set_title("seqKD - off-KD", fontsize=9.0)
    inset.tick_params(labelsize=9.0)

    save_vector(fig, "fig2_support_controls")


def heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    rows: list[str],
    columns: list[str],
    title: str,
    *,
    cmap: str = "RdBu_r",
    vmin: float = -1,
    vmax: float = 1,
    fmt: str = ".2f",
) -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isnan(value):
                label = "-"
                color = "black"
            else:
                label = format(value, fmt)
                color = "white" if abs(value) > 0.58 * max(abs(vmin), abs(vmax)) else "black"
            ax.text(column, row, label, ha="center", va="center", color=color)
    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    return image


def figure_regional_output() -> None:
    correlations = pd.read_csv(FAT / "equal5_standalone_correlations.csv")
    correlations = correlations[
        correlations["feature"].eq("c_equal5") & correlations["epsilon"].eq(0.05)
    ]
    format_table = pd.read_csv(FAT / "equal5_format_realization_table.csv")

    fig = plt.figure(figsize=(7.0, 3.65), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.45, 0.85])

    ax = fig.add_subplot(grid[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    region_colors = {
        "P": "#D9D9D9",
        "F": "#56B4E9",
        "A": "#009E73",
        "C": "#E69F00",
        "B": "#CC79A7",
        "T": "#D55E00",
    }
    rows = [
        ("MMLU-Pro", [("P", 0.36), ("F", 0.26), ("A", 0.18), ("T", 0.16)]),
        ("MATH500", [("P", 0.30), ("C", 0.34), ("B", 0.20), ("T", 0.12)]),
    ]
    y_positions = [0.68, 0.28]
    for (label, regions), y in zip(rows, y_positions):
        ax.text(0.02, y + 0.17, label, ha="left", va="bottom", fontweight="bold")
        x = 0.02
        for region, width in regions:
            ax.add_patch(
                plt.Rectangle(
                    (x, y),
                    width,
                    0.15,
                    facecolor=region_colors[region],
                    edgecolor="#333333",
                    linewidth=0.7,
                )
            )
            ax.text(x + width / 2, y + 0.075, region, ha="center", va="center")
            x += width + 0.012
    ax.text(
        0.02,
        0.04,
        "P: prompt   F: format   A: option\n"
        "C: reasoning   B: boxed answer   T: termination",
        ha="left",
        va="bottom",
    )
    ax.set_title("(a) Token regions")

    row_order = [
        ("llama", "offkd"),
        ("llama", "opd"),
        ("llama", "seqkd"),
        ("llama", "sft"),
        ("qwen", "offkd"),
        ("qwen", "opd"),
        ("qwen", "seqkd"),
        ("qwen", "sft"),
    ]
    target_columns = [
        ("mmlu", "kl_a", "MMLU A"),
        ("mmlu", "kl_f", "MMLU F"),
        ("mmlu", "kl_t", "MMLU T"),
        ("math", "kl_b", "Math B"),
        ("math", "kl_t", "Math T"),
    ]
    matrix = np.full((len(row_order), len(target_columns)), np.nan)
    for row, (model, arm) in enumerate(row_order):
        for column, (domain, target, _) in enumerate(target_columns):
            selected = correlations[
                correlations["model"].eq(model)
                & correlations["arm"].eq(arm)
                & correlations["domain"].eq(domain)
                & correlations["target"].eq(target)
            ]
            if not selected.empty:
                matrix[row, column] = float(selected["spearman"].iloc[0])
    row_labels = [
        f"{'L' if model == 'llama' else 'Q'}-{ARM_LABELS[arm]}"
        for model, arm in row_order
    ]
    ax = fig.add_subplot(grid[0, 1])
    image = heatmap(
        ax,
        matrix,
        row_labels,
        [label for _, _, label in target_columns],
        r"(b) Within-arm $\rho_s(c_{.05},\mathrm{KL})$",
        vmin=-1,
        vmax=1,
    )
    ax.tick_params(axis="x", rotation=28)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025)
    colorbar.set_label(r"Spearman $\rho_s$")

    signed_matrix = np.full((len(row_order), 2), np.nan)
    for row, (model, arm) in enumerate(row_order):
        selected = format_table[
            format_table["model"].eq(model) & format_table["arm"].eq(arm)
        ]
        if not selected.empty:
            signed_matrix[row, 0] = float(
                selected["spearman_F_minus_A_vs_format_gap"].iloc[0]
            )
            signed_matrix[row, 1] = float(
                selected["spearman_delta_nll_f_vs_extract_failure"].iloc[0]
            )
    ax = fig.add_subplot(grid[0, 2])
    heatmap(
        ax,
        signed_matrix,
        row_labels,
        [r"$F-A$ vs gap", r"$\Delta F$ vs fail"],
        "(c) Readout boundary",
        vmin=-1,
        vmax=1,
    )
    ax.tick_params(axis="x", rotation=28)
    save_vector(fig, "fig3_regional_output")


def supplementary_full_domains() -> None:
    trajectories = functional_trajectories()
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 4.2), constrained_layout=True)
    model_steps = {
        "qwen": [5, 10, 20, 40, 80, 160, 320, 480, 624],
        "llama": [5, 20, 40, 80, 160, 320],
    }
    for row, model in enumerate(["qwen", "llama"]):
        for column, probe in enumerate(CORE_PROBES):
            ax = axes[row, column]
            steps = model_steps[model]
            selected = trajectories[
                trajectories["model"].eq(model)
                & trajectories["probe_name"].eq(probe)
            ]
            for arm in ARMS:
                series = (
                    selected[selected["arm"].eq(arm)]
                    .set_index("checkpoint")
                    .reindex(steps)
                )
                ax.plot(
                    log_positions(steps),
                    100 * series["c_equal5"],
                    color=ARM_COLORS[arm],
                    linestyle=LINE_STYLES[arm],
                    marker=ARM_MARKERS[arm],
                    markersize=3.2,
                    markeredgewidth=0.5,
                    label=ARM_LABELS[arm],
                )
            ax.axhline(0, color="#555555", linewidth=0.7)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
            ax.set_title(
                f"{'(a) ' if row == 0 and column == 0 else ''}"
                f"{model.capitalize()} / {PROBE_LABELS[probe]}"
            )
            tick_steps = [5, 40, 160, 624] if model == "qwen" else [5, 40, 160]
            ax.set_xticks(log_positions(tick_steps), [str(step) for step in tick_steps])
            if row == 1:
                ax.set_xlabel("Checkpoint")
            if column == 0:
                ax.set_ylabel(r"$c_{.05}$ (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=4,
        frameon=False,
    )
    save_vector(fig, "supp_fig_full_domain_trajectories")


def supplementary_robustness() -> None:
    dominance = pd.read_csv(EQUAL5 / "EQUAL5_dominance_cells.csv")
    ncd = pd.read_csv(EQUAL5 / "EQUAL5_ncd.csv")
    paired = pd.read_csv(EQUAL5 / "EQUAL5_equal7_paired_comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.0), constrained_layout=True)

    ax = axes[0]
    epsilons = [0.01, 0.025, 0.05, 0.1]
    positions = np.arange(len(epsilons))
    for offset, (model, color, marker) in zip(
        [-0.09, 0.09],
        [("qwen", "#9C4300", "o"), ("llama", "#0072B2", "s")],
    ):
        means = (
            dominance[dominance["model"].eq(model)]
            .groupby("epsilon")["continuous_margin"]
            .mean()
            .reindex(epsilons)
        )
        minima = (
            dominance[dominance["model"].eq(model)]
            .groupby("epsilon")["continuous_margin"]
            .min()
            .reindex(epsilons)
        )
        ax.errorbar(
            positions + offset,
            means,
            yerr=np.vstack([means - minima, np.zeros(len(means))]),
            fmt=marker,
            color=color,
            capsize=2.5,
            linewidth=1.0,
            markersize=4.5,
            label=model.capitalize(),
        )
    ax.axhline(0, color="#555555", linewidth=0.7)
    ax.set_xticks(positions, [".01", ".025", ".05", ".10"])
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("Mean margin; lower whisker=min")
    ax.set_title("(a) Threshold robustness")
    ax.set_ylim(-1.5, 16.5)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    ax = axes[1]
    chosen_ncd = ncd[ncd["epsilon"].eq(0.05)]
    x = np.arange(len(ARMS))
    width = 0.36
    for index, (model, color) in enumerate(
        [("qwen", "#9C4300"), ("llama", "#0072B2")]
    ):
        values = (
            chosen_ncd[chosen_ncd["model"].eq(model)]
            .set_index("arm")
            .reindex(ARMS)["equal5_non_qk"]
        )
        ax.bar(
            x + (index - 0.5) * width,
            values,
            width,
            color=color,
            edgecolor="#333333",
            label=model.capitalize(),
        )
    ax.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS], rotation=25)
    ax.set_ylabel("NCD (direction x log-step)")
    ax.set_title("(b) Common-horizon exposure")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)

    ax = axes[2]
    share_values = np.asarray([[12.5, 46.4], [5.8, 25.8]])
    x = np.arange(2)
    width = 0.34
    ax.bar(
        x - width / 2,
        share_values[:, 0],
        width,
        color="#999999",
        edgecolor="#333333",
        label="Raw direction share",
    )
    ax.bar(
        x + width / 2,
        share_values[:, 1],
        width,
        color="#91476F",
        edgecolor="#333333",
        label="Relative positive share",
    )
    ax.set_xticks(x, ["Qwen", "Llama"])
    ax.set_ylabel("q/k share (%)")
    ax.set_title("(c) Why q/k are audited separately")
    ax.set_ylim(0, 60)
    ax.legend(frameon=False, loc="upper center")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    _ = paired
    save_vector(fig, "supp_fig_robustness")


def main() -> None:
    configure_style()
    figure_main_trajectories()
    figure_support_controls()
    figure_regional_output()
    supplementary_full_domains()
    supplementary_robustness()


if __name__ == "__main__":
    main()
