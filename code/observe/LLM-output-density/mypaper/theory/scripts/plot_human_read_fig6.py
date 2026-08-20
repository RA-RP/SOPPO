"""Regenerate the matched four-core trajectory figure used by human_read-ch.md."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve()
THEORY = HERE.parents[1]
MYPAPER = HERE.parents[2]
SOURCE = (
    MYPAPER
    / "local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion"
    / "run_01"
    / "mini"
    / "d10_5_integrated_state_equal7.csv"
)
OUTPUT = THEORY / "figs" / "fig6_matched_core_trajectories.png"

PROBES = {"E_general", "E_math", "E_ood", "E_if"}
ARMS = ["opd", "sft", "offkd", "seqkd"]
LABELS = {"opd": "OPD", "sft": "SFT", "offkd": "off-KD", "seqkd": "seqKD"}
COLORS = {"opd": "#c43c39", "sft": "#3b6fb6", "offkd": "#2f8a59", "seqkd": "#8b5bb5"}


def main() -> None:
    frame = pd.read_csv(SOURCE)
    selected = frame[
        (frame["epsilon"] == 0.05)
        & frame["probe_name"].isin(PROBES)
        & frame["arm"].isin(ARMS)
        & (
            ((frame["model"] == "qwen") & (frame["layer"] == 18))
            | ((frame["model"] == "llama") & (frame["layer"] == 14))
        )
    ]
    means = (
        selected.groupby(["model", "arm", "checkpoint"], as_index=False)[
            "state_rank_delta_mean"
        ]
        .mean()
        .sort_values(["model", "arm", "checkpoint"])
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25), sharey=True, constrained_layout=True)
    for axis, model, layer in zip(axes, ["qwen", "llama"], [18, 14]):
        subset = means[means["model"] == model]
        ticks = sorted(subset["checkpoint"].unique())
        positions = {checkpoint: index for index, checkpoint in enumerate(ticks)}
        for arm in ARMS:
            arm_rows = subset[subset["arm"] == arm]
            axis.plot(
                arm_rows["checkpoint"].map(positions),
                arm_rows["state_rank_delta_mean"],
                marker="o",
                markersize=3.5,
                linewidth=1.8 if arm == "opd" else 1.35,
                color=COLORS[arm],
                label=LABELS[arm],
            )
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_xticks(range(len(ticks)))
        axis.set_xticklabels([str(int(value)) for value in ticks], rotation=35)
        axis.set_title(f"{'Qwen3-4B' if model == 'qwen' else 'Llama-3.2-3B'} · L{layer}")
        axis.set_xlabel("Checkpoint")
        axis.grid(axis="y", alpha=0.22, linewidth=0.6)
    axes[0].set_ylabel(r"Four-probe mean $\Delta r_{.05}^{(7)}$")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Matched four-core domain functional-rank trajectories", fontsize=11)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
