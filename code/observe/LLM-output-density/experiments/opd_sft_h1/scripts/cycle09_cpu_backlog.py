#!/usr/bin/env python3
"""Close the Cycle 09 analysis-only backlog from immutable local artifacts.

This script never loads a model. It writes an analysis contract first, then
derives descriptive tables for C7/C9/C10/C12 and appendices D/E. It also emits
explicit readiness audits for C6 and C11 when the requested estimand is not
identified by the stored artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
MINI_DEFAULT = (
    REPO
    / "mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
CODE_EVOLUTION_DEFAULT = REPO / "mypaper/code/code_evolution.md"
AUTODL = Path("/root/autodl-tmp")

ARMS = ("opd", "sft", "offkd", "seqkd")
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
TRANSIENT_STEPS = (5, 10, 20, 40, 80)
LAYERS = (9, 18, 27)
MODULES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
STATIC_PROBES = (
    "legacy_S_math",
    "E_ood",
    "E_general",
    "E_math_hard",
    "S_bos",
)

START_MARKER = "<!-- cycle09-cpu-backlog-start -->"
END_MARKER = "<!-- cycle09-cpu-backlog-end -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract", "run"), default="run")
    parser.add_argument("--mini", type=Path, default=MINI_DEFAULT)
    parser.add_argument(
        "--code-evolution", type=Path, default=CODE_EVOLUTION_DEFAULT
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def require(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))


def probe_family(value: str) -> str:
    if value.startswith("S_bos__"):
        return "S_bos"
    return value


def source_paths(mini: Path) -> dict[str, Path]:
    return {
        "m1": mini / "R4_m1_tail_ec.csv",
        "m2": mini / "R4_m2_output_drift.csv",
        "theta": mini / "R5_theta_reps.csv",
        "trajectory": mini / "three_arm_full_trajectory.csv",
        "mmlu_audit": mini / "S1_mmlupro_extract_audit.csv",
        "ifeval": mini / "S1_ifeval_breakdown.csv",
        "emath_probe": AUTODL
        / "cycle09_r2/getslice/inputs/X_math_hard/x_probe.jsonl",
        "aime_eval": REPO / "Eval/tasks/data/aime24/train.jsonl",
        "probe_builder": REPO
        / "experiments/opd_sft_h1/scripts/cycle09_r2_build_battery.py",
        "formal_step80": AUTODL
        / "cycle07_base_sft_trajectory/eval/step_080/math500/"
        "step_080_samples.jsonl",
        "cap_pilot": AUTODL
        / "cycle07_base_sft_trajectory/cap_pilot/pilot_summary.json",
        "cap_pilot_samples": AUTODL
        / "cycle07_base_sft_trajectory/cap_pilot/math500_samples.jsonl",
        "s1_loglik_manifest": mini / "S1_mmlupro_loglik_manifest.json",
    }


def contract_payload(mini: Path) -> dict[str, Any]:
    sources = source_paths(mini)
    require(sources.values())
    return {
        "status": "posthoc_analysis_contract_for_existing_data",
        "scope": "Cycle 09 CPU backlog C6/C7/C9/C10/C11/C12 and appendices D/E",
        "claim_boundary": (
            "Descriptive closure only. This is not a prospective preregistration "
            "because the source results already existed."
        ),
        "source_files": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in sources.items()
        },
        "definitions": {
            "shared_grid": list(STEPS),
            "transient_window": list(TRANSIENT_STEPS),
            "module_aggregation": "arithmetic mean over the fixed seven modules",
            "seed_aggregation": "S_bos generation seeds are equally weighted",
            "C10": {
                "layer": 18,
                "track": "frozen_base",
                "epsilon": 0.05,
                "theta": "mean of per-module theta_u_max_deg; theta_v secondary",
                "movement": "mean per-module M2 X0_primary",
                "efficiency": "theta_deg / M2; denominator below 1e-4 flagged",
                "inference": "none; point-estimate table only",
            },
            "C9": {
                "scope": "OPD, SFT, off-KD at step 624; ten seeds x 30 AIME24",
                "completion_proxy": "presence of literal \\boxed in saved response",
                "finish_groups": ["stop", "length"],
                "warning": "boxed is an observable proxy, not a reasoning-completeness rubric",
            },
            "C7": {
                "geometry_event": (
                    "earliest positive maximum of L18 frozen-base r_epsilon_delta "
                    "within steps 5,10,20,40,80"
                ),
                "behavior_event": "earliest extremum within the same window",
                "lag": "behavior ordinal index minus geometry ordinal index",
                "inference": "none; no independence claim across probe rows",
            },
            "C6": {
                "rule": "do not fit mediation/regression with duplicated arm outcomes",
                "minimum_identification": (
                    "domain-matched geometry and output statistics plus more than four "
                    "independent conditions"
                ),
            },
            "C11": {
                "estimand": "model next-token entropy at an explicitly defined answer position",
                "rule": "generation text or sequence-level option LL is not token entropy",
            },
            "appendix_D": "retain cap schedule and report the unpaired cap pilot honestly",
            "appendix_E": (
                "L9/L18/L27 frozen-base r_epsilon_delta, epsilon 0.05, fixed modules"
            ),
        },
    }


def write_contract(mini: Path) -> Path:
    path = mini / "cpu_backlog_analysis_contract.json"
    payload = contract_payload(mini)
    payload["frozen_at_utc"] = utc_now()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old = {key: value for key, value in existing.items() if key != "frozen_at_utc"}
        new = {key: value for key, value in payload.items() if key != "frozen_at_utc"}
        if old != new:
            raise RuntimeError(f"Existing analysis contract differs: {path}")
        return path
    atomic_json(path, payload)
    return path


def run_c12(mini: Path, paths: dict[str, Path]) -> list[Path]:
    probes = read_jsonl(paths["emath_probe"])
    eval_rows = read_jsonl(paths["aime_eval"])
    if len(probes) != len(eval_rows):
        raise RuntimeError("E_math_hard and AIME24 row counts differ")

    rows = []
    for index, (probe, eval_row) in enumerate(zip(probes, eval_rows)):
        probe_text = str(probe["output"]["text"])
        eval_text = str(eval_row["Problem"])
        rows.append(
            {
                "row_index": index,
                "aime_id": eval_row.get("ID", ""),
                "probe_text_sha256": hashlib.sha256(
                    probe_text.encode("utf-8")
                ).hexdigest(),
                "eval_text_sha256": hashlib.sha256(
                    eval_text.encode("utf-8")
                ).hexdigest(),
                "exact_text_match": probe_text == eval_text,
            }
        )
    frame = pd.DataFrame(rows)
    csv_path = mini / "C12_emath_provenance.csv"
    atomic_csv(csv_path, frame)
    summary = {
        "status": "complete",
        "probe_rows": len(probes),
        "eval_rows": len(eval_rows),
        "exact_ordered_matches": int(frame["exact_text_match"].sum()),
        "all_exact_ordered_matches": bool(frame["exact_text_match"].all()),
        "probe_source": "Maxwell-Jia/aime_2024 train, Problem field",
        "eval_source": "Maxwell-Jia/aime_2024 train via Eval AIME24 data",
        "dependency_statement": (
            "E_math_hard is byte-for-byte the AIME24 evaluation question set. "
            "It is a same-domain probe, not an independent held-out probe."
        ),
        "source_hashes": {
            key: sha256(paths[key])
            for key in ("emath_probe", "aime_eval", "probe_builder")
        },
    }
    json_path = mini / "C12_emath_provenance.json"
    atomic_json(json_path, summary)
    return [csv_path, json_path]


def run_appendix_d(mini: Path, paths: dict[str, Path]) -> list[Path]:
    trajectory = pd.read_csv(paths["trajectory"])
    schedule = trajectory[
        [
            "arm",
            "step",
            "math500_n",
            "math500_cap",
            "math500_acc",
            "math500_trunc_rate",
            "math500_mean_response_len",
            "math500_source",
        ]
    ].copy()
    schedule["protocol_segment"] = np.where(
        schedule["step"] < 40, "early_cap_4096", "late_cap_16384"
    )
    schedule_path = mini / "appendix_D_cap_schedule.csv"
    atomic_csv(schedule_path, schedule)

    formal = read_jsonl(paths["formal_step80"])
    pilot = json.loads(paths["cap_pilot"].read_text(encoding="utf-8"))
    pilot_row = next(row for row in pilot["results"] if row["task"] == "math500")
    n_formal = len(formal)
    n_length = sum(str(row.get("finish")) == "length" for row in formal)
    n_length_correct = sum(
        str(row.get("finish")) == "length" and as_bool(row.get("ok"))
        for row in formal
    )
    n_stop = n_formal - n_length
    n_stop_correct = sum(
        str(row.get("finish")) != "length" and as_bool(row.get("ok"))
        for row in formal
    )
    comparison = pd.DataFrame(
        [
            {
                "run": "formal_step80",
                "arm": "sft",
                "step": 80,
                "cap": 4096,
                "n": n_formal,
                "accuracy": sum(as_bool(row.get("ok")) for row in formal) / n_formal,
                "truncation_rate": n_length / n_formal,
                "truncated_n": n_length,
                "truncated_correct_n": n_length_correct,
                "accuracy_given_truncated": n_length_correct / n_length,
                "stopped_n": n_stop,
                "stopped_correct_n": n_stop_correct,
                "accuracy_given_stopped": n_stop_correct / n_stop,
                "paired_sample_set": False,
                "source": str(paths["formal_step80"]),
            },
            {
                "run": "large_cap_pilot_step80",
                "arm": "sft",
                "step": 80,
                "cap": int(pilot_row["probe_cap"]),
                "n": int(pilot_row["n"]),
                "accuracy": float(pilot_row["acc_all"]),
                "truncation_rate": float(
                    pilot_row["trunc_pct_at_cap"][str(pilot_row["probe_cap"])]
                )
                / 100.0,
                "truncated_n": int(str(pilot_row["hit_probe_cap"]).split("/")[0]),
                "truncated_correct_n": np.nan,
                "accuracy_given_truncated": np.nan,
                "stopped_n": np.nan,
                "stopped_correct_n": np.nan,
                "accuracy_given_stopped": np.nan,
                "paired_sample_set": False,
                "source": str(paths["cap_pilot"]),
            },
        ]
    )
    comparison["comparison_caveat"] = (
        "The N=500 formal run and N=60 cap pilot are different sample sets; "
        "compare aggregate stability only."
    )
    comparison_path = mini / "appendix_D_cap_pilot.csv"
    atomic_csv(comparison_path, comparison)
    return [schedule_path, comparison_path]


def static_m1(mini: Path) -> pd.DataFrame:
    data = pd.read_csv(mini / "R4_m1_tail_ec.csv")
    data = data[
        data["arm"].isin(ARMS)
        & data["probe_family"].isin(STATIC_PROBES)
        & (data["track"] == "frozen_base")
        & np.isclose(data["epsilon"].astype(float), 0.05)
        & data["module"].isin(MODULES)
    ].copy()
    return data


def run_appendix_e(mini: Path) -> list[Path]:
    data = static_m1(mini)
    grouped = (
        data.groupby(["arm", "step", "probe_family", "layer"], as_index=False)
        .agg(
            r_epsilon_delta_mean=("r_epsilon_delta", "mean"),
            r_epsilon_delta_sd=("r_epsilon_delta", "std"),
            r_epsilon_base_mean=("r_epsilon_base", "mean"),
            r_epsilon_current_mean=("r_epsilon_current", "mean"),
            n_seed_module_cells=("module", "size"),
            n_modules=("module", "nunique"),
            n_task_ids=("task_id", "nunique"),
        )
        .sort_values(["arm", "probe_family", "step", "layer"])
    )
    expected = len(ARMS) * len(STEPS) * len(STATIC_PROBES) * len(LAYERS)
    if len(grouped) != expected:
        raise RuntimeError(f"Appendix E expected {expected} rows, got {len(grouped)}")
    long_path = mini / "appendix_E_layer_robustness.csv"
    atomic_csv(long_path, grouped)

    pivot = grouped.pivot_table(
        index=["arm", "step", "probe_family"],
        columns="layer",
        values="r_epsilon_delta_mean",
    ).reset_index()
    pivot.columns = [
        "arm",
        "step",
        "probe_family",
        "r_epsilon_delta_L9",
        "r_epsilon_delta_L18",
        "r_epsilon_delta_L27",
    ]
    values = pivot[
        ["r_epsilon_delta_L9", "r_epsilon_delta_L18", "r_epsilon_delta_L27"]
    ]
    pivot["same_sign_all_layers"] = (
        (values.ge(0).all(axis=1)) | (values.le(0).all(axis=1))
    )
    pivot["max_abs_layer_spread"] = values.max(axis=1) - values.min(axis=1)
    wide_path = mini / "appendix_E_layer_robustness_wide.csv"
    atomic_csv(wide_path, pivot)
    return [long_path, wide_path]


def run_c10(mini: Path) -> list[Path]:
    theta = pd.read_csv(mini / "R5_theta_reps.csv")
    theta["probe_family"] = theta["probe"].astype(str).map(probe_family)
    theta = theta[
        theta["arm"].isin(ARMS)
        & theta["probe_family"].isin(STATIC_PROBES)
        & (theta["track"] == "frozen_base")
        & (theta["layer"] == 18)
        & np.isclose(theta["epsilon"].astype(float), 0.05)
        & theta["module"].isin(MODULES)
    ].copy()
    theta_group = (
        theta.groupby(["arm", "step", "probe_family"], as_index=False)
        .agg(
            theta_u_max_deg=("theta_u_max_deg", "mean"),
            theta_u_mean_deg=("theta_u_mean_deg", "mean"),
            theta_v_max_deg=("theta_v_max_deg", "mean"),
            theta_v_mean_deg=("theta_v_mean_deg", "mean"),
            theta_n_seed_module_cells=("module", "size"),
            theta_n_modules=("module", "nunique"),
        )
    )

    m2 = pd.read_csv(mini / "R4_m2_output_drift.csv")
    m2["probe_family"] = m2["task_id"].astype(str).map(probe_family)
    m2 = m2[
        m2["arm"].isin(ARMS)
        & m2["probe_family"].isin(STATIC_PROBES)
        & (m2["layer"] == 18)
        & (m2["reference"] == "X0_primary")
        & m2["module"].isin(MODULES)
    ].copy()
    m2_group = (
        m2.groupby(["arm", "step", "probe_family"], as_index=False)
        .agg(
            m2_x0=("m2_output_drift", "mean"),
            m2_n_seed_module_cells=("module", "size"),
            m2_n_modules=("module", "nunique"),
        )
    )
    grid = pd.MultiIndex.from_product(
        [ARMS, STEPS[1:], STATIC_PROBES],
        names=["arm", "step", "probe_family"],
    ).to_frame(index=False)
    result = grid.merge(
        theta_group,
        on=["arm", "step", "probe_family"],
        how="left",
        validate="one_to_one",
    ).merge(
        m2_group,
        on=["arm", "step", "probe_family"],
        how="left",
        validate="one_to_one",
    )
    result["theta_available"] = result["theta_u_max_deg"].notna()
    result["m2_available"] = result["m2_x0"].notna()
    result["availability"] = np.select(
        [
            result["theta_available"] & result["m2_available"],
            ~result["theta_available"] & result["m2_available"],
            result["theta_available"] & ~result["m2_available"],
        ],
        ["complete", "missing_theta_source", "missing_m2_source"],
        default="missing_both_sources",
    )
    result["theta_u_max_per_m2"] = result["theta_u_max_deg"] / result["m2_x0"]
    result["theta_v_max_per_m2"] = result["theta_v_max_deg"] / result["m2_x0"]
    result["small_denominator_lt_1e4"] = result["m2_x0"] < 1e-4
    result["inference_status"] = np.where(
        result["availability"] == "complete",
        "point_estimate_only_no_four_arm_sample_ci",
        "not_computable_from_stored_sources",
    )
    result = result.sort_values(["probe_family", "step", "arm"])
    expected = len(ARMS) * (len(STEPS) - 1) * len(STATIC_PROBES)
    if len(result) != expected:
        raise RuntimeError(f"C10 expected {expected} rows, got {len(result)}")
    missing = result[result["availability"] != "complete"]
    expected_missing = {
        (arm, step, "S_bos")
        for arm in ("opd", "sft")
        for step in (5, 10, 20, 40, 160, 624)
    }
    observed_missing = set(
        missing[["arm", "step", "probe_family"]].itertuples(index=False, name=None)
    )
    if observed_missing != expected_missing:
        raise RuntimeError(
            "Unexpected C10 source gaps: "
            f"expected {sorted(expected_missing)}, got {sorted(observed_missing)}"
        )
    path = mini / "C10_rotation_efficiency_four_arm.csv"
    atomic_csv(path, result)
    eood_path = mini / "C10_rotation_efficiency_eood.csv"
    atomic_csv(eood_path, result[result["probe_family"] == "E_ood"].copy())
    return [path, eood_path]


def aime_sample_roots() -> dict[str, Path]:
    return {
        "opd": AUTODL / "cycle09_r3/id_completion/aime24/opd/step_624/cap_24576",
        "sft": AUTODL / "cycle09_r3/id_completion/aime24/sft/step_624/cap_24576",
        "offkd": AUTODL
        / "cycle09_offkd/eval/id_completion/aime24/offkd/step_624/cap_24576",
    }


def boxed_position(text: str) -> tuple[bool, float | None]:
    match = re.search(r"\\boxed\s*\{", text)
    if not match:
        return False, None
    return True, match.start() / max(len(text), 1)


def run_c9(mini: Path) -> list[Path]:
    roots = aime_sample_roots()
    require(roots.values())
    rows: list[dict[str, Any]] = []
    source_files: list[Path] = []
    for arm, root in roots.items():
        for seed in range(42, 52):
            path = root / f"seed_{seed}_samples.jsonl"
            require([path])
            source_files.append(path)
            samples = read_jsonl(path)
            if len(samples) != 30:
                raise RuntimeError(f"Expected 30 AIME rows: {path}")
            for sample_index, sample in enumerate(samples):
                text = str(sample.get("gen", ""))
                has_boxed, position = boxed_position(text)
                pred = str(sample.get("pred", "")).strip()
                rows.append(
                    {
                        "arm": arm,
                        "step": 624,
                        "seed": seed,
                        "sample_index": sample_index,
                        "problem_id": sample.get("id", sample_index),
                        "finish_reason": sample.get("finish", ""),
                        "response_tokens": int(sample.get("resp_len", 0)),
                        "correct": as_bool(sample.get("ok")),
                        "prediction_extractable": bool(pred),
                        "has_boxed_proxy": has_boxed,
                        "first_boxed_char_fraction": position,
                        "gold": sample.get("gold", ""),
                        "pred": pred,
                        "source_path": str(path),
                    }
                )
    samples_frame = pd.DataFrame(rows)
    samples_path = mini / "C9_aime24_finish_samples.csv"
    atomic_csv(samples_path, samples_frame)

    summary_rows = []
    for arm, group in samples_frame.groupby("arm", sort=False):
        length = group[group["finish_reason"] == "length"]
        stopped = group[group["finish_reason"] != "length"]
        seed_acc = group.groupby("seed")["correct"].mean()
        summary_rows.append(
            {
                "arm": arm,
                "step": 624,
                "n_seeds": group["seed"].nunique(),
                "n_samples": len(group),
                "accuracy": group["correct"].mean(),
                "seed_accuracy_mean": seed_acc.mean(),
                "seed_accuracy_sd": seed_acc.std(ddof=1),
                "truncation_rate": (group["finish_reason"] == "length").mean(),
                "truncated_n": len(length),
                "accuracy_given_truncated": length["correct"].mean(),
                "stopped_n": len(stopped),
                "accuracy_given_stopped": stopped["correct"].mean(),
                "truncated_but_correct_rate_all": (
                    (group["finish_reason"] == "length") & group["correct"]
                ).mean(),
                "boxed_proxy_rate": group["has_boxed_proxy"].mean(),
                "extractable_rate": group["prediction_extractable"].mean(),
                "mean_response_tokens": group["response_tokens"].mean(),
                "median_response_tokens": group["response_tokens"].median(),
                "completion_measure": "literal_boxed_proxy_not_reasoning_rubric",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = mini / "C9_aime24_finish_audit.csv"
    atomic_csv(summary_path, summary)

    seed_summary = (
        samples_frame.groupby(["arm", "seed"], as_index=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            truncation_rate=("finish_reason", lambda s: (s == "length").mean()),
            boxed_proxy_rate=("has_boxed_proxy", "mean"),
            mean_response_tokens=("response_tokens", "mean"),
        )
    )
    seed_path = mini / "C9_aime24_finish_by_seed.csv"
    atomic_csv(seed_path, seed_summary)

    provenance = {
        "status": "complete_observable_proxy",
        "scope": "OPD/SFT/off-KD step 624, seeds 42-51, 30 AIME24 prompts",
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_files
        ],
        "warning": (
            "The stored rows support finish reason, scorer correctness, extraction, "
            "and literal boxed-position proxies. They do not support a semantic "
            "step-by-step solution-completeness score."
        ),
    }
    provenance_path = mini / "C9_aime24_finish_manifest.json"
    atomic_json(provenance_path, provenance)
    return [samples_path, summary_path, seed_path, provenance_path]


def select_event(
    frame: pd.DataFrame, value: str, direction: str
) -> tuple[int, float]:
    ordered = frame.sort_values("step")
    target = ordered[value].max() if direction == "max" else ordered[value].min()
    row = ordered[np.isclose(ordered[value], target)].iloc[0]
    return int(row["step"]), float(row[value])


def behavior_events(mini: Path) -> pd.DataFrame:
    trajectory = pd.read_csv(mini / "three_arm_full_trajectory.csv")
    trajectory = trajectory[trajectory["step"].isin(TRANSIENT_STEPS)]
    mmlu = pd.read_csv(mini / "S1_mmlupro_extract_audit.csv")
    mmlu = mmlu[mmlu["step"].isin(TRANSIENT_STEPS)]
    ifeval = pd.read_csv(mini / "S1_ifeval_breakdown.csv")
    ifeval = ifeval[ifeval["step"].isin(TRANSIENT_STEPS)]

    specs = (
        ("math500_acc_min", trajectory, "math500_acc", "min", None),
        ("math500_length_max", trajectory, "math500_mean_response_len", "max", None),
        ("math500_truncation_max", trajectory, "math500_trunc_rate", "max", None),
        ("mmlu_strict_fail_max", mmlu, "extract_fail_rate", "max", None),
        (
            "ifeval_detectable_format_min",
            ifeval,
            "pass_rate",
            "min",
            "detectable_format",
        ),
        ("ifeval_startend_min", ifeval, "pass_rate", "min", "startend"),
        (
            "ifeval_length_constraints_min",
            ifeval,
            "pass_rate",
            "min",
            "length_constraints",
        ),
    )
    rows = []
    for arm in ARMS:
        for event_name, source, value, direction, category in specs:
            frame = source[source["arm"] == arm]
            if category is not None:
                frame = frame[frame["instruction_category"] == category]
            if set(frame["step"].astype(int)) != set(TRANSIENT_STEPS):
                raise RuntimeError(f"Incomplete behavior event grid: {arm} {event_name}")
            step, event_value = select_event(frame, value, direction)
            rows.append(
                {
                    "arm": arm,
                    "behavior_event": event_name,
                    "behavior_direction": direction,
                    "behavior_event_step": step,
                    "behavior_event_value": event_value,
                }
            )
    return pd.DataFrame(rows)


def geometry_events(mini: Path) -> pd.DataFrame:
    m1 = static_m1(mini)
    m1 = m1[(m1["layer"] == 18) & m1["step"].isin(TRANSIENT_STEPS)]
    means = (
        m1.groupby(["arm", "step", "probe_family"], as_index=False)
        .agg(r_epsilon_delta=("r_epsilon_delta", "mean"))
    )
    rows = []
    for arm in ARMS:
        for probe in STATIC_PROBES:
            frame = means[(means["arm"] == arm) & (means["probe_family"] == probe)]
            if set(frame["step"].astype(int)) != set(TRANSIENT_STEPS):
                raise RuntimeError(f"Incomplete geometry event grid: {arm} {probe}")
            peak_step, peak = select_event(frame, "r_epsilon_delta", "max")
            trough_step, trough = select_event(frame, "r_epsilon_delta", "min")
            rows.append(
                {
                    "arm": arm,
                    "probe_family": probe,
                    "positive_expansion_exists": peak > 0,
                    "geometry_positive_peak_step": peak_step if peak > 0 else np.nan,
                    "geometry_positive_peak_value": peak if peak > 0 else np.nan,
                    "signed_max_step": peak_step,
                    "signed_max_value": peak,
                    "signed_min_step": trough_step,
                    "signed_min_value": trough,
                }
            )
    return pd.DataFrame(rows)


def run_c7(mini: Path) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    geometry = geometry_events(mini)
    behavior = behavior_events(mini)
    rows = []
    step_index = {step: index for index, step in enumerate(TRANSIENT_STEPS)}
    for _, grow in geometry.iterrows():
        for _, brow in behavior[behavior["arm"] == grow["arm"]].iterrows():
            has_event = bool(grow["positive_expansion_exists"])
            gstep = (
                int(grow["geometry_positive_peak_step"])
                if has_event
                else None
            )
            bstep = int(brow["behavior_event_step"])
            rows.append(
                {
                    **grow.to_dict(),
                    **brow.to_dict(),
                    "lag_steps_raw": bstep - gstep if gstep is not None else np.nan,
                    "lag_checkpoint_ordinals": (
                        step_index[bstep] - step_index[gstep]
                        if gstep is not None
                        else np.nan
                    ),
                    "geometry_leads_behavior": (
                        bstep > gstep if gstep is not None else np.nan
                    ),
                    "inference_status": "descriptive_nonindependent_probe_rows",
                }
            )
    result = pd.DataFrame(rows)
    path = mini / "C7_lead_lag.csv"
    atomic_csv(path, result)

    valid = result[result["positive_expansion_exists"]].copy()
    summary = (
        valid.groupby("behavior_event", as_index=False)
        .agg(
            n_probe_arm_rows=("lag_checkpoint_ordinals", "size"),
            n_arms=("arm", "nunique"),
            geometry_leads_count=("geometry_leads_behavior", "sum"),
            median_lag_ordinals=("lag_checkpoint_ordinals", "median"),
            min_lag_ordinals=("lag_checkpoint_ordinals", "min"),
            max_lag_ordinals=("lag_checkpoint_ordinals", "max"),
        )
    )
    summary["n_probe_arm_rows_total"] = len(ARMS) * len(STATIC_PROBES)
    summary["n_probe_arm_rows_with_defined_geometry"] = summary[
        "n_probe_arm_rows"
    ]
    summary["n_probe_arm_rows_without_defined_geometry"] = (
        summary["n_probe_arm_rows_total"]
        - summary["n_probe_arm_rows_with_defined_geometry"]
    )
    summary["n_arms_total"] = len(ARMS)
    summary["n_arms_with_defined_geometry"] = summary["n_arms"]
    summary["independence_warning"] = (
        "probe-arm rows share arm-level behavior events; no inferential test"
    )
    summary_path = mini / "C7_lead_lag_summary.csv"
    atomic_csv(summary_path, summary)
    return [path, summary_path], geometry, behavior


def run_c6_readiness(
    mini: Path, geometry: pd.DataFrame, behavior: pd.DataFrame
) -> list[Path]:
    trajectory = pd.read_csv(mini / "three_arm_full_trajectory.csv")
    base = trajectory[trajectory["step"] == 0].set_index("arm")
    transient = trajectory[trajectory["step"].isin(TRANSIENT_STEPS)]
    ifeval = pd.read_csv(mini / "S1_ifeval_breakdown.csv")

    rows = []
    for _, grow in geometry[
        geometry["probe_family"].isin(("E_ood", "E_general", "E_math_hard"))
    ].iterrows():
        arm = str(grow["arm"])
        arm_traj = transient[transient["arm"] == arm]
        math_min = float(arm_traj["math500_acc"].min())
        math_base = float(base.loc[arm, "math500_acc"])
        format_rows = ifeval[
            (ifeval["arm"] == arm)
            & (ifeval["instruction_category"] == "detectable_format")
        ]
        format_base = float(format_rows[format_rows["step"] == 0]["pass_rate"].iloc[0])
        format_min = float(
            format_rows[format_rows["step"].isin(TRANSIENT_STEPS)]["pass_rate"].min()
        )
        gstep = grow["geometry_positive_peak_step"]
        matched = (
            arm_traj[arm_traj["step"] == int(gstep)].iloc[0]
            if not pd.isna(gstep)
            else None
        )
        rows.append(
            {
                "arm": arm,
                "probe_family": grow["probe_family"],
                "geometry_positive_peak_step": gstep,
                "geometry_positive_peak_value": grow["geometry_positive_peak_value"],
                "math500_dip_signed_min_minus_base": math_min - math_base,
                "math500_dip_damage_base_minus_min": math_base - math_min,
                "ifeval_detectable_format_damage_base_minus_min": (
                    format_base - format_min
                ),
                "math500_length_at_geometry_peak": (
                    float(matched["math500_mean_response_len"])
                    if matched is not None
                    else np.nan
                ),
                "math500_truncation_at_geometry_peak": (
                    float(matched["math500_trunc_rate"])
                    if matched is not None
                    else np.nan
                ),
                "distinct2_at_geometry_peak": np.nan,
                "independent_unit": arm,
                "outcomes_repeated_across_probe_rows": True,
            }
        )
    inputs = pd.DataFrame(rows)
    input_path = mini / "C6_mediation_inputs.csv"
    atomic_csv(input_path, inputs)

    audit = {
        "status": "not_identified_no_regression_run",
        "available_domain_rows": len(inputs),
        "independent_arm_units": int(inputs["independent_unit"].nunique()),
        "requested_predictors": [
            "geometry expansion",
            "same-checkpoint output length/distinct2/truncation",
        ],
        "blockers": [
            "C5 E_if geometry is not available, so IFEval lacks a domain-matched geometry cell.",
            "Only four independent arm conditions exist; a two-predictor mediation model is underidentified and unstable.",
            "Expanding to arm x probe creates repeated outcomes within arm and would be pseudoreplication.",
            "Four-arm domain-matched distinct-2 is not stored for the behavior outputs.",
        ],
        "completed_work": (
            "A raw alignment table was emitted. No regression, p-value, or causal "
            "mediation claim was produced."
        ),
        "dependency_order": "Run C5 and define independent replication before C6 inference.",
    }
    audit_path = mini / "C6_mediation_readiness.json"
    atomic_json(audit_path, audit)
    return [input_path, audit_path]


def run_c11_audit(mini: Path, paths: dict[str, Path]) -> list[Path]:
    aime_example = read_jsonl(next(iter(aime_sample_roots().values())) / "seed_42_samples.jsonl")[0]
    log_manifest = json.loads(paths["s1_loglik_manifest"].read_text(encoding="utf-8"))
    audit = {
        "status": "blocked_missing_requested_estimand",
        "requested_estimand": "model next-token entropy at the answer emission position",
        "required_fields": [
            "answer-position token index under a frozen tokenizer/prompt",
            "per-arm model logits or probability mass at that position",
            "a declared entropy convention for truncated top-k mass if full logits are absent",
        ],
        "saved_generation_fields": sorted(aime_example.keys()),
        "saved_generation_has_logits": False,
        "mmlupro_loglik_available": True,
        "mmlupro_loglik_scope": (
            "sequence-level conditional LL for complete answer-option strings; "
            "not a next-token vocabulary distribution"
        ),
        "teacher_rollout_scope": (
            "top-32 teacher logprobs exist for rollout training data, but they do not "
            "measure each trained arm at evaluation answer positions"
        ),
        "action_not_taken": (
            "No empirical text-frequency entropy or option-LL softmax entropy was "
            "substituted, because those are different constructs."
        ),
        "next_required_decision": (
            "Freeze task/prompt, answer-position rule, checkpoints, and full-vocabulary "
            "or top-k entropy convention, then run model forward passes."
        ),
        "loglik_manifest_sha256": sha256(paths["s1_loglik_manifest"]),
        "loglik_cell_count": len(log_manifest.get("cells", [])),
    }
    path = mini / "C11_answer_token_entropy_readiness.json"
    atomic_json(path, audit)
    return [path]


def table_markdown(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    view = frame[columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in view.iterrows()
    ]
    return [header, divider, *rows]


def write_handoff(mini: Path, artifacts: list[Path], contract: Path) -> Path:
    c9 = pd.read_csv(mini / "C9_aime24_finish_audit.csv")
    c10_full = pd.read_csv(mini / "C10_rotation_efficiency_four_arm.csv")
    c10_missing = c10_full[c10_full["availability"] != "complete"]
    c10 = pd.read_csv(mini / "C10_rotation_efficiency_eood.csv")
    c10 = c10[c10["step"].isin((5, 20, 40, 80, 624))]
    c7 = pd.read_csv(mini / "C7_lead_lag_summary.csv")
    c12 = json.loads((mini / "C12_emath_provenance.json").read_text())
    c6 = json.loads((mini / "C6_mediation_readiness.json").read_text())
    c11 = json.loads((mini / "C11_answer_token_entropy_readiness.json").read_text())

    lines = [
        "# Cycle 09 CPU Backlog Handoff",
        "",
        "Raw descriptive closure from stored artifacts. No model was loaded and no GPU inference was run.",
        "",
        f"Analysis contract: `{contract}` (`sha256={sha256(contract)}`).",
        "",
        "## Status",
        "",
        "| item | status |",
        "|---|---|",
        "| C12 E_math_hard provenance | complete |",
        "| Appendix D cap robustness table | complete |",
        "| Appendix E layer robustness table | complete |",
        "| C10 four-arm theta/M2 table | complete_grid_with_12_source_na |",
        "| C9 AIME24 finish audit | complete_observable_proxy |",
        "| C7 lead-lag table | complete_descriptive |",
        f"| C6 mediation | {c6['status']} |",
        f"| C11 answer-token entropy | {c11['status']} |",
        "",
        "## C12 Provenance",
        "",
        f"Exact ordered text matches: {c12['exact_ordered_matches']}/{c12['probe_rows']}.",
        "",
        "## C9 AIME24 Finish Audit",
        "",
        *table_markdown(
            c9,
            [
                "arm",
                "n_samples",
                "accuracy",
                "truncation_rate",
                "accuracy_given_truncated",
                "accuracy_given_stopped",
                "boxed_proxy_rate",
                "mean_response_tokens",
            ],
        ),
        "",
        "## C10 E_ood Rotation Efficiency",
        "",
        (
            f"Full grid rows: {len(c10_full)}; computable rows: "
            f"{(c10_full['availability'] == 'complete').sum()}; source-NA rows: "
            f"{len(c10_missing)}. The source-NA cells are OPD/SFT S_bos theta at "
            "steps 5/10/20/40/160/624; M2 is present."
        ),
        "",
        *table_markdown(
            c10,
            [
                "arm",
                "step",
                "theta_u_max_deg",
                "theta_v_max_deg",
                "m2_x0",
                "theta_u_max_per_m2",
                "small_denominator_lt_1e4",
            ],
        ),
        "",
        "## C7 Lead-Lag Summary",
        "",
        *table_markdown(
            c7,
            [
                "behavior_event",
                "n_probe_arm_rows_total",
                "n_probe_arm_rows_with_defined_geometry",
                "n_probe_arm_rows_without_defined_geometry",
                "n_arms_total",
                "n_arms_with_defined_geometry",
                "geometry_leads_count",
                "median_lag_ordinals",
            ],
        ),
        "",
        "## Non-identified Items",
        "",
        "C6: no regression was run because arm x probe rows repeat four arm-level outcomes, C5 E_if geometry is absent, and distinct-2 is incomplete.",
        "",
        "C11: saved text and sequence-level option LL do not contain the requested answer-position next-token distribution. No substitute entropy was reported.",
        "",
        "## Artifact Inventory",
        "",
        "| path | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for path in artifacts:
        lines.append(f"| `{path}` | {path.stat().st_size} | `{sha256(path)}` |")
    lines.append("")
    path = mini / "mini_cpu_backlog_theory_handoff.md"
    atomic_text(path, "\n".join(lines))
    return path


def update_code_evolution(path: Path, mini: Path, handoff: Path) -> None:
    block = "\n".join(
        [
            START_MARKER,
            "",
            "## Cycle 09 CPU-only backlog closure",
            "",
            "No model was loaded. C12 and appendices D/E completed mechanically; C9, C10, and C7 produced descriptive raw tables. C10 retains 12 unavailable OPD/SFT S_bos theta cells as source-NA values. C6 was left statistically unidentified rather than fitting a four-unit/pseudoreplicated mediation model. C11 was left blocked because saved generation text has no per-arm answer-position logits.",
            "",
            f"Raw handoff: `{handoff}`.",
            f"Analysis contract: `{mini / 'cpu_backlog_analysis_contract.json'}`.",
            "",
            END_MARKER,
        ]
    )
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if START_MARKER in text:
        start = text.index(START_MARKER)
        end = text.index(END_MARKER, start) + len(END_MARKER)
        updated = text[:start] + block + text[end:]
    else:
        updated = text.rstrip() + "\n\n---\n\n" + block + "\n"
    atomic_text(path, updated)


def artifact_record(path: Path) -> dict[str, Any]:
    rows = None
    if path.suffix == ".csv":
        rows = len(pd.read_csv(path))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "rows": rows,
        "sha256": sha256(path),
    }


def main() -> None:
    args = parse_args()
    args.mini.mkdir(parents=True, exist_ok=True)
    contract = write_contract(args.mini)
    if args.mode == "contract":
        print(json.dumps({"status": "contract_frozen", "path": str(contract)}))
        return

    paths = source_paths(args.mini)
    require(paths.values())
    artifacts: list[Path] = []
    artifacts.extend(run_c12(args.mini, paths))
    artifacts.extend(run_appendix_d(args.mini, paths))
    artifacts.extend(run_appendix_e(args.mini))
    artifacts.extend(run_c10(args.mini))
    artifacts.extend(run_c9(args.mini))
    c7_paths, geometry, behavior = run_c7(args.mini)
    artifacts.extend(c7_paths)
    artifacts.extend(run_c6_readiness(args.mini, geometry, behavior))
    artifacts.extend(run_c11_audit(args.mini, paths))

    handoff = write_handoff(args.mini, artifacts, contract)
    artifacts.append(handoff)
    update_code_evolution(args.code_evolution, args.mini, handoff)

    manifest = {
        "status": "complete_with_explicit_nonidentified_items",
        "completed_at_utc": utc_now(),
        "gpu_used": False,
        "contract": artifact_record(contract),
        "source_script": artifact_record(Path(__file__).resolve()),
        "task_status": {
            "C12": "complete",
            "appendix_D": "complete",
            "appendix_E": "complete",
            "C10": "complete_grid_with_12_source_na",
            "C9": "complete_observable_proxy",
            "C7": "complete_descriptive",
            "C6": "not_identified_no_regression_run",
            "C11": "blocked_missing_requested_estimand",
        },
        "artifacts": [artifact_record(path) for path in artifacts],
        "code_evolution": artifact_record(args.code_evolution),
    }
    manifest_path = args.mini / "cpu_backlog_completion_manifest.json"
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(manifest_path),
                "handoff": str(handoff),
                "artifacts": len(artifacts),
            }
        )
    )


if __name__ == "__main__":
    main()
