#!/usr/bin/env python3
"""R2 domain-matched geometry/behavior event and observed-grid lead-lag tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import cycle09_block3_common as block3
import cycle09_stage3_common as s3


MINI = block3.MINI
ROOT = block3.RUN_ROOT / "domain_lead_lag"
EVENT_OUTPUT = ROOT / "domain_matched_events.csv"
LAG_OUTPUT = ROOT / "domain_matched_lead_lag.csv"
MANIFEST = ROOT / "domain_matched_lead_lag_manifest.json"
QWEN_STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
LLAMA_STEPS = (0, 5, 20, 40, 80, 160, 320, 624)
MODULES = tuple(s3.MODULES)


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def qwen_geometry() -> pd.DataFrame:
    rows = []
    supplement = pd.read_csv(
        require(MINI / "qwen_emath_emathhardv2_r_epsilon.csv")
    )
    for probe in ("E_math", "E_math_hard_v2"):
        selected = supplement[
            (supplement["probe"] == probe)
            & np.isclose(supplement["epsilon"], 0.05)
            & (supplement["layer"] == 18)
            & (supplement["track"] == "per_checkpoint")
        ]
        grouped = (
            selected.groupby(["arm", "step"], as_index=False)["r_epsilon_delta"]
            .mean()
            .rename(columns={"r_epsilon_delta": "geometry_value"})
        )
        grouped["probe"] = probe
        rows.append(grouped)

    r4 = pd.read_csv(require(MINI / "R4_m1_tail_ec.csv"))
    for probe, source_probe in (
        ("E_ood", "E_ood"),
        ("E_general", "E_general"),
        ("S_math", "legacy_S_math"),
    ):
        selected = r4[
            (r4["probe_family"] == source_probe)
            & np.isclose(r4["epsilon"], 0.05)
            & (r4["layer"] == 18)
            & (r4["track"] == "per_checkpoint")
        ]
        grouped = (
            selected.groupby(["arm", "step"], as_index=False)["r_epsilon_delta"]
            .mean()
            .rename(columns={"r_epsilon_delta": "geometry_value"})
        )
        grouped["probe"] = probe
        rows.append(grouped)

    eif = pd.read_csv(require(MINI / "C5_eif_m1_geometry.csv"))
    selected = eif[
        np.isclose(eif["epsilon"], 0.05)
        & (eif["layer"] == 18)
        & (eif["track"] == "per_checkpoint")
    ]
    grouped = (
        selected.groupby(["arm", "step"], as_index=False)["r_epsilon_delta"]
        .mean()
        .rename(columns={"r_epsilon_delta": "geometry_value"})
    )
    grouped["probe"] = "E_if"
    rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result["model_family"] = "qwen3_4b"
    result["n_modules"] = len(MODULES)
    return result


def llama_geometry() -> pd.DataFrame:
    frame = pd.read_csv(require(MINI / "llama_r_epsilon.csv"))
    selected = frame[
        np.isclose(frame["epsilon"], 0.05)
        & (frame["layer"] == 14)
        & (frame["track"] == "per_checkpoint")
    ].copy()
    selected["geometry_delta"] = pd.to_numeric(
        selected["delta_from_base"], errors="raise"
    )
    grouped = (
        selected.groupby(["arm", "step", "probe"], as_index=False)["geometry_delta"]
        .mean()
        .rename(columns={"geometry_delta": "geometry_value"})
    )
    base = grouped[grouped["step"] == 0].copy()
    if set(base["arm"]) == {"base"}:
        grouped = grouped[grouped["step"] != 0]
        grouped = pd.concat(
            [
                grouped,
                *[base.assign(arm=arm) for arm in s3.ARMS],
            ],
            ignore_index=True,
        )
    grouped["model_family"] = "llama3_2_3b"
    grouped["n_modules"] = len(MODULES)
    return grouped


def behavior_row(
    family: str,
    arm: str,
    step: int,
    metric: str,
    value: Any,
    probe: str,
    direction: str,
) -> dict[str, Any]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise RuntimeError(f"non-finite behavior value: {family}/{arm}/{step}/{metric}")
    return {
        "model_family": family,
        "arm": arm,
        "step": int(step),
        "behavior_metric": metric,
        "behavior_value": float(numeric),
        "matched_probe": probe,
        "behavior_event_direction": direction,
    }


def qwen_behavior() -> pd.DataFrame:
    rows = []
    trajectory = pd.read_csv(require(MINI / "three_arm_full_trajectory.csv"))
    for row in trajectory.to_dict("records"):
        for metric, column, probe, direction in (
            ("math500_accuracy", "math500_acc", "E_math", "min"),
            ("math500_cap_hit", "math500_trunc_rate", "E_math", "max"),
            ("math500_response_length", "math500_mean_response_len", "E_math", "max"),
            ("ifeval_prompt_strict", "ifeval_prompt_strict", "E_if", "min"),
            ("ifeval_instruction_strict", "ifeval_instruction_strict", "E_if", "min"),
        ):
            rows.append(
                behavior_row(
                    "qwen3_4b",
                    row["arm"],
                    row["step"],
                    metric,
                    row[column],
                    probe,
                    direction,
                )
            )

    flexible = pd.read_csv(require(MINI / "S1_mmlupro_flexible.csv"))
    for row in flexible.to_dict("records"):
        for metric, column, direction in (
            ("mmlu_pro_strict", "exact_match", "min"),
            ("mmlu_pro_flexible", "mmlu_pro_flexible", "min"),
            ("mmlu_pro_extract_failure", "strict_extract_fail_rate", "max"),
        ):
            rows.append(
                behavior_row(
                    "qwen3_4b",
                    row["arm"],
                    row["step"],
                    metric,
                    row[column],
                    "E_ood",
                    direction,
                )
            )

    categories = pd.read_csv(require(MINI / "S1_ifeval_breakdown.csv"))
    for row in categories.to_dict("records"):
        rows.append(
            behavior_row(
                "qwen3_4b",
                row["arm"],
                row["step"],
                f"ifeval_category:{row['instruction_category']}",
                row["pass_rate"],
                "E_if",
                "min",
            )
        )

    aime = pd.read_csv(require(MINI / "R3_id_completion.csv"))
    aime = aime[
        (aime["task"] == "aime24") & (aime["row_type"] == "seed_mean")
    ]
    for row in aime.to_dict("records"):
        rows.append(
            behavior_row(
                "qwen3_4b",
                row["arm"],
                row["step"],
                "aime24_accuracy",
                row["acc"],
                "E_math_hard_v2",
                "min",
            )
        )
    return pd.DataFrame(rows)


def llama_behavior() -> pd.DataFrame:
    rows = []
    summary = pd.read_csv(require(MINI / "llama_behavior_8ckpt.csv"))
    base_summary = summary[summary["step"] == 0]
    if set(base_summary["arm"]) == {"base"}:
        summary = pd.concat(
            [
                summary[summary["step"] != 0],
                *[base_summary.assign(arm=arm) for arm in s3.ARMS],
            ],
            ignore_index=True,
        )
    for row in summary.to_dict("records"):
        task = row["task"]
        if task == "math500":
            definitions = (
                ("math500_accuracy", "accuracy", "E_math", "min"),
                ("math500_cap_hit", "cap_hit_rate", "E_math", "max"),
                ("math500_response_length", "response_length_mean", "E_math", "max"),
            )
        elif task == "mmlu_pro":
            definitions = (
                ("mmlu_pro_strict", "strict_accuracy", "E_ood", "min"),
                ("mmlu_pro_flexible", "flexible_accuracy", "E_ood", "min"),
                ("mmlu_pro_extract_failure", "extract_failure_rate", "E_ood", "max"),
            )
        elif task == "ifeval":
            definitions = (
                ("ifeval_prompt_strict", "prompt_strict_accuracy", "E_if", "min"),
                (
                    "ifeval_instruction_strict",
                    "instruction_strict_accuracy",
                    "E_if",
                    "min",
                ),
            )
        else:
            raise RuntimeError(f"unexpected Llama behavior task: {task}")
        for metric, column, probe, direction in definitions:
            rows.append(
                behavior_row(
                    "llama3_2_3b",
                    row["arm"],
                    row["step"],
                    metric,
                    row[column],
                    probe,
                    direction,
                )
            )
    categories = pd.read_csv(require(MINI / "llama_ifeval_categories.csv"))
    base_categories = categories[categories["step"] == 0]
    if set(base_categories["arm"]) == {"base"}:
        categories = pd.concat(
            [
                categories[categories["step"] != 0],
                *[base_categories.assign(arm=arm) for arm in s3.ARMS],
            ],
            ignore_index=True,
        )
    category_column = next(
        (
            name
            for name in ("instruction_category", "category")
            if name in categories.columns
        ),
        None,
    )
    if category_column is None or "pass_rate" not in categories.columns:
        raise RuntimeError(
            f"unexpected Llama IFEval category schema: {categories.columns.tolist()}"
        )
    for row in categories.to_dict("records"):
        rows.append(
            behavior_row(
                "llama3_2_3b",
                row["arm"],
                row["step"],
                f"ifeval_category:{row[category_column]}",
                row["pass_rate"],
                "E_if",
                "min",
            )
        )
    return pd.DataFrame(rows)


def choose_event(frame: pd.DataFrame, column: str, direction: str) -> pd.Series:
    ordered = frame.sort_values("step", kind="stable")
    index = (
        ordered[column].astype(float).idxmin()
        if direction == "min"
        else ordered[column].astype(float).idxmax()
    )
    return ordered.loc[index]


def positive_geometry_event(frame: pd.DataFrame) -> tuple[bool, int | None, float | None]:
    ordered = frame.sort_values("step", kind="stable")
    row = ordered.loc[ordered["geometry_value"].astype(float).idxmax()]
    value = float(row["geometry_value"])
    return value > 0, (int(row["step"]) if value > 0 else None), (
        value if value > 0 else None
    )


def ordinal_relation(value: int) -> str:
    if value == 0:
        return "same_observed_checkpoint"
    if value == 1:
        return "geometry_leads_by_one_observed_interval"
    if value == -1:
        return "geometry_lags_by_one_observed_interval"
    if value > 1:
        return "geometry_leads_by_multiple_observed_intervals"
    return "geometry_lags_by_multiple_observed_intervals"


def build() -> dict[str, Any]:
    geometry = pd.concat([qwen_geometry(), llama_geometry()], ignore_index=True)
    behavior = pd.concat([qwen_behavior(), llama_behavior()], ignore_index=True)
    if geometry.duplicated(["model_family", "arm", "probe", "step"]).any():
        raise RuntimeError("duplicate geometry cells after seven-module aggregation")
    if behavior.duplicated(
        ["model_family", "arm", "behavior_metric", "step"]
    ).any():
        raise RuntimeError("duplicate behavior cells")

    event_rows = []
    for keys, frame in geometry.groupby(
        ["model_family", "arm", "probe"], sort=False
    ):
        defined, step, value = positive_geometry_event(frame)
        minimum = choose_event(frame, "geometry_value", "min")
        event_rows.extend(
            [
                {
                    "record_kind": "geometry",
                    "model_family": keys[0],
                    "arm": keys[1],
                    "matched_probe": keys[2],
                    "event": "positive_peak",
                    "event_direction": "max_positive_only",
                    "event_defined": defined,
                    "event_step": step if defined else "",
                    "event_value": value if defined else "",
                    "observed_grid": "|".join(map(str, sorted(frame["step"].unique()))),
                    "behavior_metric": "",
                },
                {
                    "record_kind": "geometry_audit",
                    "model_family": keys[0],
                    "arm": keys[1],
                    "matched_probe": keys[2],
                    "event": "signed_minimum",
                    "event_direction": "min",
                    "event_defined": True,
                    "event_step": int(minimum["step"]),
                    "event_value": float(minimum["geometry_value"]),
                    "observed_grid": "|".join(map(str, sorted(frame["step"].unique()))),
                    "behavior_metric": "",
                },
            ]
        )

    behavior_events = {}
    for keys, frame in behavior.groupby(
        ["model_family", "arm", "behavior_metric", "matched_probe"], sort=False
    ):
        directions = frame["behavior_event_direction"].unique()
        if len(directions) != 1:
            raise RuntimeError(f"behavior direction drift: {keys}")
        selected = choose_event(frame, "behavior_value", str(directions[0]))
        behavior_events[keys] = selected
        event_rows.append(
            {
                "record_kind": "behavior",
                "model_family": keys[0],
                "arm": keys[1],
                "matched_probe": keys[3],
                "event": "minimum" if directions[0] == "min" else "maximum",
                "event_direction": directions[0],
                "event_defined": True,
                "event_step": int(selected["step"]),
                "event_value": float(selected["behavior_value"]),
                "observed_grid": "|".join(map(str, sorted(frame["step"].unique()))),
                "behavior_metric": keys[2],
            }
        )

    lag_rows = []
    for keys, behavior_event in behavior_events.items():
        family, arm, metric, probe = keys
        behavior_frame = behavior[
            (behavior["model_family"] == family)
            & (behavior["arm"] == arm)
            & (behavior["behavior_metric"] == metric)
        ]
        geometry_frame = geometry[
            (geometry["model_family"] == family)
            & (geometry["arm"] == arm)
            & (geometry["probe"] == probe)
            & (geometry["step"].isin(behavior_frame["step"]))
        ]
        observed = sorted(
            set(behavior_frame["step"].astype(int)).intersection(
                geometry_frame["step"].astype(int)
            )
        )
        if not observed:
            raise RuntimeError(f"no matched observed grid: {keys}")
        behavior_matched = behavior_frame[behavior_frame["step"].isin(observed)]
        behavior_selected = choose_event(
            behavior_matched,
            "behavior_value",
            str(behavior_matched["behavior_event_direction"].iloc[0]),
        )
        defined, geometry_step, geometry_value = positive_geometry_event(
            geometry_frame[geometry_frame["step"].isin(observed)]
        )
        behavior_step = int(behavior_selected["step"])
        if defined and geometry_step is not None:
            index = {step: position for position, step in enumerate(observed)}
            ordinal = index[behavior_step] - index[geometry_step]
            raw = behavior_step - geometry_step
            relation = ordinal_relation(ordinal)
            cap_confounded = bool(
                family == "qwen3_4b"
                and min(behavior_step, geometry_step) <= 20
                and max(behavior_step, geometry_step) >= 40
            )
        else:
            ordinal = raw = ""
            relation = "geometry_positive_peak_undefined"
            cap_confounded = False
        lag_rows.append(
            {
                "model_family": family,
                "arm": arm,
                "behavior_metric": metric,
                "matched_probe": probe,
                "geometry_event": "positive_peak",
                "geometry_event_defined": defined,
                "geometry_step": geometry_step if defined else "",
                "geometry_value": geometry_value if defined else "",
                "behavior_event": (
                    "minimum"
                    if behavior_selected["behavior_event_direction"] == "min"
                    else "maximum"
                ),
                "behavior_step": behavior_step,
                "behavior_value": float(behavior_selected["behavior_value"]),
                "observed_grid": "|".join(map(str, observed)),
                "lag_checkpoint_ordinals": ordinal,
                "lag_optimizer_steps_audit": raw,
                "observed_interval_relation": relation,
                "positive_lag_means": "geometry_leads_behavior",
                "cap_boundary_confounded": cap_confounded,
                "inference_scope": "descriptive_time_order_only",
            }
        )

    block3.atomic_csv(EVENT_OUTPUT, event_rows)
    block3.atomic_csv(LAG_OUTPUT, lag_rows)
    block3.atomic_csv(MINI / EVENT_OUTPUT.name, event_rows)
    block3.atomic_csv(MINI / LAG_OUTPUT.name, lag_rows)
    sources = [
        MINI / "qwen_emath_emathhardv2_r_epsilon.csv",
        MINI / "R4_m1_tail_ec.csv",
        MINI / "C5_eif_m1_geometry.csv",
        MINI / "llama_r_epsilon.csv",
        MINI / "three_arm_full_trajectory.csv",
        MINI / "S1_mmlupro_flexible.csv",
        MINI / "S1_ifeval_breakdown.csv",
        MINI / "R3_id_completion.csv",
        MINI / "llama_behavior_8ckpt.csv",
        MINI / "llama_ifeval_categories.csv",
    ]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "task": "Cycle09 block3 R2 domain-matched probe-eval lead-lag",
        "pairing": {
            "MATH500": "E_math",
            "MMLU-Pro": "E_ood",
            "IFEval": "E_if",
            "AIME24_sparse_qwen_only": "E_math_hard_v2_AIME25",
            "E_general": "cross_domain_control_no_behavior_pair",
            "S_math": "support_anchor_no_behavior_pair",
        },
        "geometry_event": "positive peak of seven-module mean delta r_epsilon@0.05",
        "behavior_event": "minimum for scores/pass rates; maximum for cap/failure/length",
        "tie_rule": "earliest observed checkpoint",
        "lag": "behavior ordinal minus geometry ordinal on pair-specific observed grid",
        "qwen_cap_boundary": "crossing <=20 to >=40 is flagged confounded",
        "sources": [block3.artifact(require(path)) for path in sources],
        "outputs": [block3.artifact(EVENT_OUTPUT), block3.artifact(LAG_OUTPUT)],
        "created_utc": block3.utc_now(),
    }
    block3.atomic_json(MANIFEST, manifest)
    block3.atomic_json(MINI / MANIFEST.name, manifest)
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
