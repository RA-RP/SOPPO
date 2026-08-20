#!/usr/bin/env python3
"""C14/C15: read-only main-track backfill and cap-pilot statistic repair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cycle09_stage3_common as s3


R4_M1 = s3.MINI / "R4_m1_tail_ec.csv"
BEHAVIOR = s3.MINI / "three_arm_full_trajectory.csv"
OLD_CAP_TABLE = s3.MINI / "appendix_D_cap_pilot.csv"
CAP_SAMPLES = (
    s3.AUTODL
    / "cycle07_base_sft_trajectory/cap_pilot/math500_samples.jsonl"
)
C7_CONTRACT = s3.REPO / "mypaper/code/cycle09_c7_prospective_contract.json"
STATIC_PROBES = (
    "legacy_S_math",
    "E_ood",
    "E_general",
    "E_math_hard",
    "S_bos",
)
BEHAVIOR_METRICS = (
    "math500_acc",
    "math500_trunc_rate",
    "mmlu_pro_exact_match",
    "ifeval_prompt_strict",
    "ifeval_instruction_strict",
    "gpqa_diamond_acc",
    "truthfulqa_mc1_acc",
)
CAP = 24576


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def c14() -> None:
    contract = json.loads(C7_CONTRACT.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_c14_generation":
        raise RuntimeError(f"unexpected C7 contract status: {C7_CONTRACT}")
    frame = pd.read_csv(R4_M1)
    selected = frame[
        frame["probe_family"].isin(STATIC_PROBES)
        & (frame["track"] == "per_checkpoint")
        & np.isclose(frame["epsilon"], 0.05)
        & frame["layer"].isin(s3.LAYERS)
        & frame["module"].isin(s3.MODULES)
        & frame["arm"].isin(s3.ARMS)
        & frame["step"].isin(s3.STEPS)
    ].copy()
    grouped = []
    keys = ["arm", "step", "probe_family", "layer"]
    for key, cell in selected.groupby(keys, sort=False, dropna=False):
        probe = key[2]
        expected = len(s3.MODULES) * (3 if probe == "S_bos" else 1)
        if len(cell) != expected or set(cell["module"]) != set(s3.MODULES):
            raise RuntimeError(f"bad C14 source cell {key}: rows={len(cell)}")
        if probe == "S_bos" and cell["generation_seed"].nunique() != 3:
            raise RuntimeError(f"S_bos seed mismatch for {key}")
        grouped.append(
            {
                "arm": key[0],
                "step": int(key[1]),
                "probe_family": probe,
                "track": "per_checkpoint",
                "layer": int(key[3]),
                "epsilon": 0.05,
                "module_set": "|".join(s3.MODULES),
                "n_modules": len(s3.MODULES),
                "n_generation_seeds": 3 if probe == "S_bos" else 0,
                "n_module_seed_rows": len(cell),
                "r_epsilon_base_mean": float(cell["r_epsilon_base"].mean()),
                "r_epsilon_current_mean": float(cell["r_epsilon_current"].mean()),
                "r_epsilon_delta_mean": float(cell["r_epsilon_delta"].mean()),
            }
        )
    expected_rows = len(s3.ARMS) * len(s3.STEPS) * len(STATIC_PROBES) * len(s3.LAYERS)
    if len(grouped) != expected_rows:
        raise RuntimeError(f"C14 layer rows={len(grouped)} expected={expected_rows}")
    layer_output = s3.MINI / "C14_per_checkpoint_layer_sensitivity.csv"
    s3.atomic_csv(layer_output, grouped)

    layer = pd.DataFrame(grouped)
    primary = layer[
        (layer["probe_family"] == "E_ood") & (layer["layer"] == 18)
    ].copy()
    event_rows = []
    geometry_events: dict[str, list[dict]] = {}
    for arm in s3.ARMS:
        arm_rows = primary[primary["arm"] == arm].sort_values("step")
        if list(arm_rows["step"]) != list(s3.STEPS):
            raise RuntimeError(f"incomplete C14 E_ood trajectory for {arm}")
        nonzero = arm_rows[arm_rows["step"] > 0]
        positive = nonzero[nonzero["r_epsilon_delta_mean"] > 0]
        definitions = [
            (
                "first_positive_checkpoint",
                positive.iloc[0] if len(positive) else None,
            ),
            (
                "positive_peak",
                nonzero.loc[nonzero["r_epsilon_delta_mean"].idxmax()]
                if float(nonzero["r_epsilon_delta_mean"].max()) > 0
                else None,
            ),
            (
                "signed_minimum",
                arm_rows.loc[arm_rows["r_epsilon_delta_mean"].idxmin()],
            ),
        ]
        geometry_events[arm] = []
        for event, row in definitions:
            payload = {
                "arm": arm,
                "event": event,
                "defined": row is not None,
                "step": int(row["step"]) if row is not None else "",
                "r_epsilon_delta_mean": (
                    float(row["r_epsilon_delta_mean"]) if row is not None else ""
                ),
                "probe_family": "E_ood",
                "track": "per_checkpoint",
                "layer": 18,
                "epsilon": 0.05,
                "n_modules": 7,
            }
            event_rows.append(payload)
            geometry_events[arm].append(payload)
    geometry_output = s3.MINI / "C14_c7_geometry_events.csv"
    s3.atomic_csv(geometry_output, event_rows)

    behavior = pd.read_csv(BEHAVIOR)
    behavior = behavior[
        behavior["arm"].isin(s3.ARMS) & behavior["step"].isin(s3.STEPS)
    ].copy()
    if len(behavior) != len(s3.ARMS) * len(s3.STEPS):
        raise RuntimeError(f"behavior grid has {len(behavior)} rows")
    behavior_events = []
    for arm in s3.ARMS:
        arm_rows = behavior[behavior["arm"] == arm].sort_values("step")
        for metric in BEHAVIOR_METRICS:
            valid = arm_rows[pd.to_numeric(arm_rows[metric], errors="coerce").notna()]
            if valid.empty:
                raise RuntimeError(f"no finite behavior values: {arm}/{metric}")
            for event, index in (
                ("minimum", valid[metric].astype(float).idxmin()),
                ("maximum", valid[metric].astype(float).idxmax()),
            ):
                row = valid.loc[index]
                behavior_events.append(
                    {
                        "arm": arm,
                        "metric": metric,
                        "event": event,
                        "step": int(row["step"]),
                        "value": float(row[metric]),
                    }
                )
    behavior_output = s3.MINI / "C14_c7_behavior_extrema.csv"
    s3.atomic_csv(behavior_output, behavior_events)

    step_index = {step: index for index, step in enumerate(s3.STEPS)}
    lag_rows = []
    for geometry in event_rows:
        if not geometry["defined"]:
            continue
        for behavior_event in behavior_events:
            if behavior_event["arm"] != geometry["arm"]:
                continue
            lag_rows.append(
                {
                    "arm": geometry["arm"],
                    "geometry_event": geometry["event"],
                    "geometry_step": geometry["step"],
                    "behavior_metric": behavior_event["metric"],
                    "behavior_event": behavior_event["event"],
                    "behavior_step": behavior_event["step"],
                    "lag_checkpoint_index": (
                        step_index[behavior_event["step"]]
                        - step_index[int(geometry["step"])]
                    ),
                    "lag_optimizer_steps": (
                        behavior_event["step"] - int(geometry["step"])
                    ),
                    "interpretation_scope": "descriptive_time_order_only",
                }
            )
    lag_output = s3.MINI / "C14_c7_lead_lag_descriptive.csv"
    s3.atomic_csv(lag_output, lag_rows)

    manifest = s3.MINI / "C14_main_track_backfill_manifest.json"
    s3.atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "task": "C14",
            "contract": s3.artifact(C7_CONTRACT),
            "sources": [s3.artifact(R4_M1), s3.artifact(BEHAVIOR)],
            "aggregation": (
                "equal mean over fixed seven modules; S_bos additionally has "
                "three generation seeds with equal row weight"
            ),
            "outputs": [
                s3.artifact(layer_output),
                s3.artifact(geometry_output),
                s3.artifact(behavior_output),
                s3.artifact(lag_output),
            ],
            "old_frozen_base_outputs_overwritten": False,
            "inference": "descriptive only; no causal claim",
        },
    )
    print(f"[C14] layer_rows={len(grouped)} lag_rows={len(lag_rows)}")


def c15() -> None:
    rows = s3.read_jsonl(CAP_SAMPLES)
    if len(rows) != 60:
        raise RuntimeError(f"expected 60 cap-pilot samples, found {len(rows)}")
    sample_rows = []
    for index, row in enumerate(rows):
        response_tokens = int(row["resp_len"])
        finish = str(row["finish"])
        cap_hit = finish == "length" or response_tokens >= CAP
        sample_rows.append(
            {
                "sample_index": index,
                "generation_cap": CAP,
                "finish": finish,
                "response_tokens": response_tokens,
                "cap_hit": int(cap_hit),
                "correct": int(bool(row["ok"])),
                "has_boxed": int(bool(row["has_boxed"])),
            }
        )
    hits = [row for row in sample_rows if row["cap_hit"]]
    stopped = [row for row in sample_rows if not row["cap_hit"]]
    corrected = [
        {
            "run": "large_cap_pilot_step80_corrected",
            "arm": "sft",
            "step": 80,
            "generation_cap": CAP,
            "n": len(sample_rows),
            "accuracy": float(np.mean([row["correct"] for row in sample_rows])),
            "cap_hit_n": len(hits),
            "cap_hit_rate": len(hits) / len(sample_rows),
            "accuracy_given_cap_hit": (
                float(np.mean([row["correct"] for row in hits])) if hits else ""
            ),
            "stopped_n": len(stopped),
            "accuracy_given_stopped": (
                float(np.mean([row["correct"] for row in stopped]))
                if stopped
                else ""
            ),
            "cap_hit_definition": (
                "finish == length OR response_tokens >= generation_cap"
            ),
            "paired_with_formal_n500": False,
        }
    ]
    sample_output = s3.MINI / "C15_cap_pilot_samples_corrected.csv"
    corrected_output = s3.MINI / "C15_cap_pilot_corrected.csv"
    s3.atomic_csv(sample_output, sample_rows)
    s3.atomic_csv(corrected_output, corrected)

    old = pd.read_csv(OLD_CAP_TABLE)
    formal = old[old["run"] == "formal_step80"]
    if len(formal) != 1:
        raise RuntimeError("missing unique formal_step80 aggregate")
    formal_row = formal.iloc[0]
    comparison = [
        {
            "run": "formal_step80",
            "cap": int(formal_row["cap"]),
            "n": int(formal_row["n"]),
            "accuracy": float(formal_row["accuracy"]),
            "cap_hit_rate": float(formal_row["truncation_rate"]),
            "sample_set": "formal_N500",
            "paired": False,
        },
        {
            "run": corrected[0]["run"],
            "cap": CAP,
            "n": len(sample_rows),
            "accuracy": corrected[0]["accuracy"],
            "cap_hit_rate": corrected[0]["cap_hit_rate"],
            "sample_set": "pilot_N60",
            "paired": False,
        },
    ]
    comparison_output = s3.MINI / "C15_cap_unpaired_aggregate_comparison.csv"
    s3.atomic_csv(comparison_output, comparison)
    manifest = s3.MINI / "C15_cap_pilot_repair_manifest.json"
    s3.atomic_json(
        manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "task": "C15",
            "contract": s3.artifact(s3.CONTRACT),
            "sources": [s3.artifact(CAP_SAMPLES), s3.artifact(OLD_CAP_TABLE)],
            "outputs": [
                s3.artifact(sample_output),
                s3.artifact(corrected_output),
                s3.artifact(comparison_output),
            ],
            "claim_boundary": (
                "N=500 formal and N=60 pilot are unpaired; aggregate "
                "proximity is not a paired cap-robustness result"
            ),
        },
    )
    print(f"[C15] cap_hits={len(hits)}/{len(sample_rows)}")


def smoke() -> None:
    tiny = [
        {"finish": "length", "resp_len": 10},
        {"finish": "stop", "resp_len": 10},
        {"finish": "stop", "resp_len": 9},
    ]
    hits = [row["finish"] == "length" or row["resp_len"] >= 10 for row in tiny]
    if hits != [True, True, False]:
        raise RuntimeError(f"bad cap-hit logic: {hits}")
    if not R4_M1.is_file() or not CAP_SAMPLES.is_file():
        raise FileNotFoundError("C14/C15 smoke sources are missing")
    print(json.dumps({"status": "ok", "cap_hits": hits}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("c14", "c15", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    s3.assert_contract()
    if args.smoke:
        smoke()
        return
    if args.task in ("c14", "all"):
        c14()
    if args.task in ("c15", "all"):
        c15()


if __name__ == "__main__":
    main()
