#!/usr/bin/env python3
"""Build the preregistered T3 E_ood ER-offset table from persisted spectra."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import cycle09_r4_common as c4

ARMS = ("opd", "sft", "offkd")
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
BACKFILL_STEPS = (80, 320, 480)
PROBE = "E_ood"
LAYER = 18
MINI = c4.MINI_ROOT
OUTPUT = MINI / "T3_eood_er_offset_three_arm.csv"
REPORT = MINI / "T3_eood_er_offset_three_arm.md"


def selected(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        (frame["task_id"] == PROBE)
        & (frame["track"] == "per_checkpoint")
        & (frame["layer"] == LAYER)
        & frame["module"].isin(c4.MODULES)
    ].copy()


def read_primary() -> pd.DataFrame:
    columns = [
        "arm", "step", "task_id", "track", "layer", "module", "effective_rank"
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        MINI / "R4_v2_spectra_all.csv", usecols=columns, chunksize=200_000
    ):
        keep = selected(chunk)
        keep = keep[
            keep["arm"].isin(("opd", "sft"))
            & keep["step"].isin(set(STEPS).difference(BACKFILL_STEPS))
        ]
        if not keep.empty:
            keep["source_kind"] = "R4_main_grid"
            chunks.append(keep)
    if not chunks:
        raise ValueError("no R4 main-grid E_ood rows found")
    return pd.concat(chunks, ignore_index=True)


def read_backfill(arm: str) -> pd.DataFrame:
    path = MINI / f"R4_v2_spectra_backfill_{arm}.csv"
    frame = selected(pd.read_csv(path))
    frame = frame[frame["step"].isin(BACKFILL_STEPS)].copy()
    frame["source_kind"] = "T3_geometry_backfill"
    return frame


def read_offkd() -> pd.DataFrame:
    frame = selected(pd.read_csv(MINI / "R4_v2_spectra_offkd.csv"))
    frame = frame[frame["step"].isin(STEPS)].copy()
    frame["source_kind"] = "offkd_geometry"
    return frame


def write_text_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    cells = pd.concat(
        [read_primary(), read_backfill("opd"), read_backfill("sft"), read_offkd()],
        ignore_index=True,
    )
    key = ["arm", "step", "module"]
    duplicates = cells.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate T3 cells: {cells.loc[duplicates, key].to_dict('records')[:5]}")
    expected_rows = len(ARMS) * len(STEPS) * len(c4.MODULES)
    if len(cells) != expected_rows:
        counts = cells.groupby(["arm", "step"]).size().to_dict()
        raise ValueError(
            f"T3 grid incomplete: expected {expected_rows}, got {len(cells)}; counts={counts}"
        )

    summary = (
        cells.groupby(["arm", "step"], as_index=False)
        .agg(
            er_mean_l18_7_modules=("effective_rank", "mean"),
            n_modules=("module", "nunique"),
            source_kind=("source_kind", lambda values: "|".join(sorted(set(values)))),
        )
    )
    base = summary[summary["step"] == 0].set_index("arm")[
        "er_mean_l18_7_modules"
    ]
    summary["er_offset_from_base"] = summary.apply(
        lambda row: row["er_mean_l18_7_modules"] - base.loc[row["arm"]], axis=1
    )
    summary["probe"] = PROBE
    summary["layer"] = LAYER
    summary["track"] = "per_checkpoint"
    summary = summary[
        [
            "arm", "step", "probe", "layer", "track", "n_modules",
            "er_mean_l18_7_modules", "er_offset_from_base", "source_kind",
        ]
    ].sort_values(
        ["arm", "step"],
        key=lambda series: series.map(
            {**{arm: index for index, arm in enumerate(ARMS)},
             **{step: index for index, step in enumerate(STEPS)}}
        ),
    )
    c4.write_csv_atomic(OUTPUT, summary.to_dict("records"), list(summary.columns))

    offsets = summary.pivot(index="arm", columns="step", values="er_offset_from_base")
    lines = [
        "# T3 E_ood L18 effective-rank offset from base",
        "",
        "| arm | " + " | ".join(str(step) for step in STEPS) + " |",
        "|---|" + "---:|" * len(STEPS),
    ]
    for arm in ARMS:
        values = [float(offsets.loc[arm, step]) for step in STEPS]
        lines.append(
            f"| {arm} | " + " | ".join(f"{value:+.6f}" for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "Raw reading: E_ood, L18, per-checkpoint whitening, mean over seven fixed modules.",
            f"Generated at {datetime.now(timezone.utc).isoformat()}.",
        ]
    )
    write_text_atomic(REPORT, "\n".join(lines) + "\n")
    c4.write_json_atomic(
        MINI / "T3_eood_er_offset_manifest.json",
        {
            "status": "complete",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arms": list(ARMS),
            "steps": list(STEPS),
            "probe": PROBE,
            "layer": LAYER,
            "track": "per_checkpoint",
            "aggregation": "mean_fixed_7_modules",
            "rows": len(summary),
            "cell_rows_validated": len(cells),
            "output": str(OUTPUT),
            "report": str(REPORT),
        },
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
