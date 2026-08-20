#!/usr/bin/env python3
"""Cycle 09 block 2 C2: assemble the preregistered dose-response table."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REPO = Path("/root/LLM-output-density")
MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
ARMS = ("opd", "sft", "offkd")
TASKS = {"E_ood": "ood", "E_general": "general", "E_math_hard": "math_hard"}
TRANSIENT_STEPS = (5, 10, 20, 40, 80)
MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
SPECTRA = (
    MINI / "R4_v2_spectra_all.csv",
    MINI / "R4_v2_spectra_offkd.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def eood_peaks() -> list[dict]:
    frame = pd.read_csv(MINI / "S1_transient_ci.csv")
    frame = frame[
        (frame["task_id"] == "E_ood")
        & (frame["metric"] == "er_offset_vs_base")
        & (frame["layer"] == 18)
        & (frame["module"] == "mean_fixed_7_modules")
        & (frame["step"].isin(TRANSIENT_STEPS))
    ]
    rows = []
    for arm in ARMS:
        column = f"{arm}_mean"
        low = f"{arm}_ci95_lo"
        high = f"{arm}_ci95_hi"
        if len(frame) != len(TRANSIENT_STEPS):
            raise RuntimeError(f"incomplete E_ood CI grid: {len(frame)}")
        selected = frame.loc[frame[column].abs().idxmax()]
        rows.append(
            {
                "arm": arm,
                "domain": "ood",
                "probe": "E_ood",
                "er_peak_step": int(selected["step"]),
                "er_peak_signed": float(selected[column]),
                "er_peak_absolute": abs(float(selected[column])),
                "er_peak_ci95_lo": float(selected[low]),
                "er_peak_ci95_hi": float(selected[high]),
                "er_peak_source_kind": "S1-3 sample-bootstrap CI mean",
                "er_peak_candidate_steps": ",".join(map(str, TRANSIENT_STEPS)),
                "er_peak_selection": "max absolute signed ER offset over steps 5,10,20,40,80",
            }
        )
    return rows


def static_point_rows() -> pd.DataFrame:
    wanted_tasks = {"E_general", "E_math_hard"}
    parts = []
    usecols = [
        "arm", "step", "task_id", "track", "layer", "module", "effective_rank"
    ]
    for path in SPECTRA:
        if not path.is_file():
            raise FileNotFoundError(path)
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=4096):
            keep = chunk[
                chunk["arm"].isin(ARMS)
                & chunk["task_id"].isin(wanted_tasks)
                & (chunk["track"] == "frozen_base")
                & (chunk["layer"] == 18)
                & chunk["module"].isin(MODULES)
                & chunk["step"].isin((0, *TRANSIENT_STEPS))
            ]
            if len(keep):
                parts.append(keep)
    if not parts:
        raise RuntimeError("no static E-domain spectra selected")
    frame = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["arm", "step", "task_id", "track", "layer", "module"]
    )
    counts = frame.groupby(["arm", "task_id", "step"]).size()
    if not (counts == len(MODULES)).all():
        raise RuntimeError(f"incomplete seven-module static cells: {counts.to_dict()}")
    observed_cells = set(zip(frame["arm"], frame["task_id"], frame["step"]))
    required_cells = {
        (arm, task, step)
        for arm in ARMS
        for task in wanted_tasks
        for step in (0, 5, 10, 20, 40)
    }
    if not required_cells.issubset(observed_cells):
        raise RuntimeError(f"missing required static cells: {sorted(required_cells-observed_cells)}")
    return frame


def static_peaks() -> list[dict]:
    frame = static_point_rows()
    means = (
        frame.groupby(["arm", "task_id", "step"], as_index=False)["effective_rank"]
        .mean()
        .rename(columns={"effective_rank": "er_mean_7_modules"})
    )
    baseline = means[means["step"] == 0][
        ["arm", "task_id", "er_mean_7_modules"]
    ].rename(columns={"er_mean_7_modules": "er_base"})
    means = means.merge(baseline, on=["arm", "task_id"], validate="many_to_one")
    means["er_offset"] = means["er_mean_7_modules"] - means["er_base"]
    means = means[means["step"].isin(TRANSIENT_STEPS)]
    rows = []
    for (arm, task), group in means.groupby(["arm", "task_id"], sort=True):
        candidate_steps = sorted(int(value) for value in group["step"].unique())
        selected = group.loc[group["er_offset"].abs().idxmax()]
        rows.append(
            {
                "arm": arm,
                "domain": TASKS[task],
                "probe": task,
                "er_peak_step": int(selected["step"]),
                "er_peak_signed": float(selected["er_offset"]),
                "er_peak_absolute": abs(float(selected["er_offset"])),
                "er_peak_ci95_lo": float("nan"),
                "er_peak_ci95_hi": float("nan"),
                "er_peak_source_kind": "R4 frozen-base point estimate; no sample CI",
                "er_peak_candidate_steps": ",".join(map(str, candidate_steps)),
                "er_peak_selection": "max absolute signed ER offset over available recorded transient steps",
            }
        )
    return rows


def behavior_damage() -> pd.DataFrame:
    audit = pd.read_csv(MINI / "S1_mmlupro_extract_audit.csv")
    ifeval = pd.read_csv(MINI / "S1_ifeval_breakdown.csv")
    categories = ("detectable_format", "startend", "length_constraints")
    records = []
    for arm in ARMS:
        m = audit[audit["arm"] == arm].set_index("step")
        if not {0, 40, 624}.issubset(m.index):
            raise RuntimeError(f"missing MMLU audit landmarks for {arm}")
        row = {
            "arm": arm,
            "mmlu_strict_fail_rate_0": float(m.loc[0, "extract_fail_rate"]),
            "mmlu_strict_fail_rate_40": float(m.loc[40, "extract_fail_rate"]),
            "mmlu_strict_fail_rate_624": float(m.loc[624, "extract_fail_rate"]),
            "mmlu_strict_fail_delta_624_minus_0": float(
                m.loc[624, "extract_fail_rate"] - m.loc[0, "extract_fail_rate"]
            ),
            "mmlu_collapse_depth_at_40_fail_40_minus_0": float(
                m.loc[40, "extract_fail_rate"] - m.loc[0, "extract_fail_rate"]
            ),
        }
        arm_if = ifeval[ifeval["arm"] == arm]
        for category in categories:
            values = arm_if[arm_if["instruction_category"] == category].set_index("step")
            if not {0, 624}.issubset(values.index):
                raise RuntimeError(f"missing IFEval {category} endpoints for {arm}")
            initial = float(values.loc[0, "pass_rate"])
            final = float(values.loc[624, "pass_rate"])
            prefix = "ifeval_" + category
            row[f"{prefix}_pass_rate_0"] = initial
            row[f"{prefix}_pass_rate_624"] = final
            row[f"{prefix}_change_624_minus_0"] = final - initial
            row[f"{prefix}_damage_0_minus_624"] = initial - final
        records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    peaks = pd.DataFrame([*eood_peaks(), *static_peaks()])
    expected = len(ARMS) * len(TASKS)
    if len(peaks) != expected:
        raise RuntimeError(f"peak rows={len(peaks)}/{expected}")
    output = peaks.merge(behavior_damage(), on="arm", validate="many_to_one")
    output = output.sort_values(["arm", "domain"], kind="stable").reset_index(drop=True)
    target = MINI / "C2_dose_response.csv"
    atomic_csv(target, output)
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "Cycle 09 block 2 C2",
        "rows": len(output),
        "arms": list(ARMS),
        "domains": list(TASKS.values()),
        "regression_performed": False,
        "formulas": {
            "x": "signed ER(step)-ER(step0), seven-module L18 mean; peak=max absolute over 5,10,20,40,80",
            "mmlu_endpoint": "strict_extract_fail_rate(624)-strict_extract_fail_rate(0)",
            "collapse_depth_at_40": "strict_extract_fail_rate(40)-strict_extract_fail_rate(0)",
            "ifeval_change": "category pass_rate(624)-category pass_rate(0)",
            "ifeval_damage": "category pass_rate(0)-category pass_rate(624)",
        },
        "sources": {
            str(path): sha256_file(path)
            for path in (
                MINI / "S1_transient_ci.csv",
                MINI / "R4_v2_spectra_all.csv",
                MINI / "R4_v2_spectra_offkd.csv",
                MINI / "S1_mmlupro_extract_audit.csv",
                MINI / "S1_ifeval_breakdown.csv",
            )
        },
        "output": str(target),
        "output_sha256": sha256_file(target),
    }
    atomic_json(MINI / "C2_dose_response_manifest.json", manifest)
    print(output.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
