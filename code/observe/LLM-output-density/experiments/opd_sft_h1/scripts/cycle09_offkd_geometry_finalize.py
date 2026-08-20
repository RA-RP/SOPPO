#!/usr/bin/env python3
"""Validate off-KD geometry outputs and emit the preregistered raw tables.

This is mechanical: it checks schema/provenance/completeness, writes three compact
trajectory tables, and records the handin paths. It does not interpret the values
or evaluate the frozen Theory decision tree.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import cycle09_r4_common as c4

ARM = "offkd"
ARMS = ("opd", "sft", "offkd")
PRIMARY_STEPS = (0, 5, 10, 20, 40, 160, 624)
OPTIONAL_STEPS = (80, 320, 480)
OFFKD_STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
PROBES = ("legacy_S_math", "E_ood", "E_general", "E_math_hard", "S_bos")
MINI = c4.MINI_ROOT
CODE_EVOLUTION = c4.REPO / "mypaper/code/code_evolution.md"
REPORT = MINI / "offkd_geometry_raw_tables.md"
SECTION_MARKER = "## Cycle 09 off-KD control - Stage 3 geometry"


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def probe_family(task_id: Any) -> str:
    value = str(task_id)
    if value.startswith("S_bos__g"):
        return "S_bos"
    return value


def validate_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((MINI / "offkd_geometry_manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("status") == "complete", "manifest is not complete")
    require(tuple(manifest.get("steps", [])) == OFFKD_STEPS, "manifest step grid mismatch")
    require(manifest.get("base_forward_calls") == 0, "base forward calls must be zero")
    expected_rows = {"spectra": 2940, "m1": 5880, "m2": 3150, "theta": 3969}
    require(manifest.get("rows_expected") == expected_rows, "manifest row expectations mismatch")

    spectra = pd.read_csv(MINI / "R4_v2_spectra_offkd.csv")
    m1_all = pd.read_csv(MINI / "R4_m1_tail_ec.csv")
    m2_all = pd.read_csv(MINI / "R4_m2_output_drift.csv")
    theta_all = pd.read_csv(MINI / "R5_theta_reps.csv")
    m1 = m1_all[m1_all["arm"] == ARM].copy()
    m2 = m2_all[m2_all["arm"] == ARM].copy()
    theta = theta_all[theta_all["arm"] == ARM].copy()

    require(len(spectra) == expected_rows["spectra"], "off-KD spectra row count mismatch")
    require(len(m1) == expected_rows["m1"], "off-KD M1 row count mismatch")
    require(len(m2) == expected_rows["m2"], "off-KD M2 row count mismatch")
    require(len(theta) == expected_rows["theta"], "off-KD theta row count mismatch")
    require(set(spectra["arm"]) == {ARM}, "spectra contains another arm")
    require(set(theta["step"]) == set(OFFKD_STEPS[1:]), "theta step grid mismatch")
    require(
        set(theta["rank_rule"])
        == {"per-cell r_eps (05)", "per-cell r_eps (01)", "fixed k=64 control"},
        "theta rank-rule mismatch",
    )
    angle_columns = [
        "theta_u_max_deg", "theta_u_mean_deg", "theta_v_max_deg", "theta_v_mean_deg"
    ]
    require(np.isfinite(theta[angle_columns].to_numpy()).all(), "non-finite theta value")

    representation = m2[m2["module"] == "__representation__"]
    weight_m2 = m2[m2["module"].isin(c4.MODULES)]
    require(
        set(representation["reference"]) == {"paired_hidden_states"},
        "M2b reference schema mismatch",
    )
    require(
        set(representation["source_kind"]) == {"same_forward_text"},
        "M2b source schema mismatch",
    )
    require(
        set(weight_m2[weight_m2["step"] > 0]["source_kind"])
        == {"offkd_clean_fp32_ba"},
        "nonzero M2 rows are not clean adapter B@A",
    )
    require(
        set(weight_m2[weight_m2["step"] == 0]["source_kind"]) == {"base_identity"},
        "step-zero M2 provenance mismatch",
    )
    require(np.isfinite(m2["m2_output_drift"].to_numpy()).all(), "non-finite M2 value")
    sbos_zero = spectra[
        (spectra["step"] == 0) & spectra["task_id"].str.startswith("S_bos__g")
    ]
    require(
        set(sbos_zero["task_id"]) == {"S_bos__g3", "S_bos__g17", "S_bos__g31"},
        "S_bos seed baselines are incomplete",
    )
    require(
        sbos_zero.groupby("task_id").size().eq(42).all(),
        "S_bos seed baselines were merged or duplicated",
    )
    return m1_all, m2_all, spectra


def old_spectra_rows() -> pd.DataFrame:
    path = MINI / "R4_v2_spectra_all.csv"
    columns = [
        "arm", "step", "task_id", "track", "layer", "module", "effective_rank"
    ]
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=4096):
        wanted_probe = chunk["task_id"].isin(PROBES[:-1]) | chunk["task_id"].str.startswith(
            "S_bos__g"
        )
        mask = (
            chunk["arm"].isin(ARMS[:2])
            & chunk["step"].isin(PRIMARY_STEPS)
            & wanted_probe
            & (chunk["track"] == "per_checkpoint")
            & (chunk["layer"] == 18)
            & chunk["module"].isin(c4.MODULES)
        )
        if mask.any():
            selected.append(chunk.loc[mask])
    require(bool(selected), "no OPD/SFT spectra rows found")
    return pd.concat(selected, ignore_index=True)


def aggregate(
    frame: pd.DataFrame,
    *,
    value: str,
    task_column: str,
    filters: dict[str, Any],
    arms: tuple[str, ...] = ARMS,
    steps: tuple[int, ...] = PRIMARY_STEPS,
) -> pd.Series:
    selected = frame.copy()
    for column, wanted in filters.items():
        if isinstance(wanted, tuple):
            selected = selected[selected[column].isin(wanted)]
        elif isinstance(wanted, float):
            selected = selected[np.isclose(selected[column].astype(float), wanted)]
        else:
            selected = selected[selected[column] == wanted]
    selected = selected[
        selected["arm"].isin(arms)
        & selected["step"].isin(steps)
        & (selected["layer"] == 18)
        & selected["module"].isin(c4.MODULES)
    ].copy()
    selected["probe"] = selected[task_column].map(probe_family)
    selected = selected[selected["probe"].isin(PROBES)]

    counts = selected.groupby(["arm", "probe", "step"]).size()
    for arm in arms:
        for probe in PROBES:
            expected_count = 21 if probe == "S_bos" else 7
            for step in steps:
                key = (arm, probe, step)
                require(key in counts.index, f"missing metric cell {key}")
                require(
                    int(counts.loc[key]) == expected_count,
                    f"metric cell count mismatch {key}: {counts.loc[key]}",
                )
    result = selected.groupby(["arm", "probe", "step"])[value].mean()
    require(len(result) == len(arms) * len(PROBES) * len(steps), "aggregate grid mismatch")
    return result


def render_table(title: str, values: pd.Series, digits: int) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| probe | OPD [0,5,10,20,40,160,624] | SFT [0,5,10,20,40,160,624] | off-KD [0,5,10,20,40,160,624] |",
        "|---|---|---|---|",
    ]
    for probe in PROBES:
        trajectories = []
        for arm in ARMS:
            readings = [float(values.loc[(arm, probe, step)]) for step in PRIMARY_STEPS]
            trajectories.append("[" + ", ".join(f"{reading:.{digits}f}" for reading in readings) + "]")
        lines.append(f"| {probe} | {trajectories[0]} | {trajectories[1]} | {trajectories[2]} |")
    lines.append("")
    return lines


def render_optional_table(title: str, values: pd.Series, digits: int) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| probe | off-KD [80,320,480] |",
        "|---|---|",
    ]
    for probe in PROBES:
        readings = [float(values.loc[(ARM, probe, step)]) for step in OPTIONAL_STEPS]
        trajectory = "[" + ", ".join(
            f"{reading:.{digits}f}" for reading in readings
        ) + "]"
        lines.append(f"| {probe} | {trajectory} |")
    lines.append("")
    return lines


def append_code_evolution() -> None:
    content = CODE_EVOLUTION.read_text(encoding="utf-8")
    if SECTION_MARKER in content:
        return
    section = f"""

---

{SECTION_MARKER} (2026-07-16)

Added cycle09_offkd_geometry.py and watchdog_cycle09_offkd_geometry.sh to
measure the third arm on the frozen R4/R5 protocol. The measured off-KD grid is
{{0,5,10,20,40,80,160,320,480,624}}, with layers {{9,18,27}}, seven fixed modules, four
fixed corpora plus three independent S_bos seeds, no H probes, and zero base
forward calls. Base profiles come directly from cycle09_r4/scratch/references.

The clean update path is adapter B@A in fp32 only; merged-minus-base is absent.
Theta uses fp64 SVD plus fp64 QR and records r-epsilon 0.05, r-epsilon 0.01, and
fixed k=64 controls. Per-step atomic caches make the run resumable without retry.

The 80/320/480 adapters and merged models are numerical backfills from the recorded
landmarks and are labeled as such in the manifest.

Validated rows: spectra 2940, M1 5880, M2 3150, theta 3969. Full spectra are kept
separate in mini/R4_v2_spectra_offkd.csv; shared M1/M2/theta files contain the
idempotently appended offkd rows. Raw readings only, with no interpretation or
decision, are in mini/offkd_geometry_raw_tables.md.
"""
    write_text_atomic(CODE_EVOLUTION, content.rstrip() + section + "\n")


def main() -> None:
    m1, m2, offkd_spectra = validate_outputs()
    rank = aggregate(
        m1,
        value="r_epsilon_current",
        task_column="probe_family",
        filters={"track": "per_checkpoint", "epsilon": 0.05},
    )
    old_spectra = old_spectra_rows()
    offkd_er = offkd_spectra[
        (offkd_spectra["track"] == "per_checkpoint")
        & (offkd_spectra["layer"] == 18)
        & offkd_spectra["module"].isin(c4.MODULES)
    ]
    er = aggregate(
        pd.concat([old_spectra, offkd_er], ignore_index=True),
        value="effective_rank",
        task_column="task_id",
        filters={"track": "per_checkpoint"},
    )
    m2_x0 = aggregate(
        m2,
        value="m2_output_drift",
        task_column="task_id",
        filters={"reference": "X0_primary"},
    )
    rank_optional = aggregate(
        m1,
        value="r_epsilon_current",
        task_column="probe_family",
        filters={"track": "per_checkpoint", "epsilon": 0.05},
        arms=(ARM,),
        steps=OPTIONAL_STEPS,
    )
    er_optional = aggregate(
        offkd_er,
        value="effective_rank",
        task_column="task_id",
        filters={"track": "per_checkpoint"},
        arms=(ARM,),
        steps=OPTIONAL_STEPS,
    )
    m2_optional = aggregate(
        m2,
        value="m2_output_drift",
        task_column="task_id",
        filters={"reference": "X0_primary"},
        arms=(ARM,),
        steps=OPTIONAL_STEPS,
    )

    lines = [
        "# Cycle 09 off-KD geometry: raw three-arm tables",
        "",
        "L18; mean over the seven fixed modules. S_bos additionally averages its three generation seeds.",
        "Checkpoint order in every trajectory is [0,5,10,20,40,160,624].",
        "",
    ]
    lines.extend(render_table("r_epsilon (epsilon=0.05, per-checkpoint whitening)", rank, 6))
    lines.extend(render_table("Effective rank (per-checkpoint whitening)", er, 6))
    lines.extend(render_table("M2 output drift (X0 primary)", m2_x0, 9))
    lines.extend(render_optional_table(
        "Supplement: off-KD r_epsilon (epsilon=0.05)", rank_optional, 6
    ))
    lines.extend(render_optional_table(
        "Supplement: off-KD effective rank", er_optional, 6
    ))
    lines.extend(render_optional_table(
        "Supplement: off-KD M2 output drift (X0 primary)", m2_optional, 9
    ))
    write_text_atomic(REPORT, "\n".join(lines).rstrip() + "\n")
    append_code_evolution()
    print(f"[offkd-finalize] validated and wrote {REPORT}", flush=True)


if __name__ == "__main__":
    main()
