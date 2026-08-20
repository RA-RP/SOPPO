#!/usr/bin/env python3
"""Round 4 CPU summaries: M1, seed bands, and transient colocation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import cycle09_r4_common as c

R3_OOD = Path("/root/autodl-tmp/cycle09_r3/ood_expansion")
OPD_TRAJECTORY = (
    c.REPO
    / "mypaper/local_experiment_results/cycle_08_h_opd_vs_sft_comparison"
    / "run_01/trajectory_scores_unified.csv"
)
SFT_TRAJECTORY = (
    c.REPO
    / "mypaper/local_experiment_results/cycle_07_base_sft_trajectory"
    / "run_01/trajectory_scores.csv"
)
GRID = (0, 5, 10, 20, 40, 160, 624)
TAIL_RANKS = (32, 64, 128, 256)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def family(task_id: str, *, drop_seed: bool = False) -> str:
    value = re.sub(r"__step_\d{3}", "", task_id)
    if drop_seed:
        value = re.sub(r"__g\d+", "", value)
    return value


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def drift_core(current: list[float], base: list[float], rank: int) -> float:
    cur = np.asarray(current, dtype=np.float64)
    ref = np.asarray(base, dtype=np.float64)
    width = min(max(int(rank), 1), len(cur), len(ref))
    cur = cur[:width] / max(float(cur.sum()), 1e-30)
    ref = ref[:width] / max(float(ref.sum()), 1e-30)
    return float(np.sqrt(np.mean(np.square(np.log(cur + 1e-30) - np.log(ref + 1e-30)))))


def run_m1(args: argparse.Namespace) -> None:
    rows = read_csv(args.mini_root / "R4_v2_spectra_all.csv")
    if not rows:
        raise FileNotFoundError("R4_v2_spectra_all.csv is missing or empty")

    baselines: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        if int(row["step"]) != 0:
            continue
        key = (
            row["arm"],
            family(row["task_id"]),
            row["track"],
            int(row["layer"]),
            row["module"],
        )
        baselines[key] = json.loads(row["sigma_json"])

    output = []
    for row in rows:
        key = (
            row["arm"],
            family(row["task_id"]),
            row["track"],
            int(row["layer"]),
            row["module"],
        )
        base_sigma = baselines.get(key)
        if base_sigma is None:
            continue
        sigma = json.loads(row["sigma_json"])
        for epsilon in (0.05, 0.01):
            base_rank = c.functional_rank(base_sigma, epsilon)
            rank = c.functional_rank(sigma, epsilon)
            output.append(
                {
                    "arm": row["arm"],
                    "step": int(row["step"]),
                    "task_id": row["task_id"],
                    "probe_family": family(row["task_id"], drop_seed=True),
                    "probe_type": row["probe_type"],
                    "domain": row["domain"],
                    "generation_seed": row["generation_seed"],
                    "track": row["track"],
                    "layer": int(row["layer"]),
                    "module": row["module"],
                    "epsilon": epsilon,
                    "r_epsilon_base": base_rank,
                    "r_epsilon_current": rank,
                    "r_epsilon_delta": rank - base_rank,
                    "rank_reduced_vs_base": rank < base_rank,
                    "drift_core": drift_core(sigma, base_sigma, base_rank),
                    "core_rank_definition": "base_r_epsilon",
                    "ec_core_small_threshold": "not_numerically_preregistered",
                    **{
                        f"tail_energy_r{tail_rank}": c.tail_energy(sigma, tail_rank)
                        for tail_rank in TAIL_RANKS
                    },
                }
            )
    fields = list(output[0]) if output else []
    c.write_csv_atomic(args.mini_root / "R4_m1_tail_ec.csv", output, fields)

    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["generation_seed"] in ("", "None", None):
            continue
        key = (
            row["arm"],
            int(row["step"]),
            row["probe_type"],
            row["domain"],
            row["track"],
            int(row["layer"]),
            row["module"],
        )
        grouped[key].append(row)

    seed_rows = []
    for key, members in sorted(grouped.items()):
        values = {
            "effective_rank": [float(row["effective_rank"]) for row in members],
            "r_eps_005": [float(row["r_eps_005"]) for row in members],
            "r_eps_001": [float(row["r_eps_001"]) for row in members],
            "tail_energy_r32": [float(row["tail_energy_r32"]) for row in members],
        }
        base = {
            "arm": key[0],
            "step": key[1],
            "probe_type": key[2],
            "domain": key[3],
            "track": key[4],
            "layer": key[5],
            "module": key[6],
            "n_generation_seeds": len(members),
            "generation_seeds": ",".join(sorted(row["generation_seed"] for row in members)),
        }
        for name, vals in values.items():
            array = np.asarray(vals, dtype=np.float64)
            base[f"{name}_mean"] = float(array.mean())
            base[f"{name}_sd"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        seed_rows.append(base)
    c.write_csv_atomic(
        args.mini_root / "R4_v2_spectra_seed_summary.csv",
        seed_rows,
        list(seed_rows[0]) if seed_rows else [],
    )
    print(f"[M1] rows={len(output)} seed_summary={len(seed_rows)}", flush=True)


def local_extreme(values: dict[int, float], mode: str) -> tuple[int | None, float | None, float | None]:
    candidates = []
    steps = [step for step in GRID if step in values]
    for previous, step, following in zip(steps, steps[1:], steps[2:]):
        contrast = values[step] - 0.5 * (values[previous] + values[following])
        if mode == "low":
            contrast = -contrast
        candidates.append((contrast, step, values[step]))
    if not candidates:
        return None, None, None
    contrast, step, value = max(candidates)
    signed = contrast if mode == "high" else -contrast
    return step, value, signed


def spectral_series(rows: list[dict[str, str]], arm: str, kind: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["arm"] != arm or row["track"] != "per_checkpoint":
            continue
        if int(row["layer"]) != 18:
            continue
        if kind == "legacy" and row["task_id"] != "legacy_S_math":
            continue
        if kind == "h_ood" and not (
            row["probe_type"] == "H" and row["domain"] == "ood"
        ):
            continue
        grouped[int(row["step"])].append(float(row["effective_rank"]))
    return {step: float(np.mean(values)) for step, values in grouped.items()}


def trajectory_series(path: Path, column: str) -> dict[int, float]:
    result = {}
    for row in read_csv(path):
        step = int(row["step"])
        value = numeric(row.get(column))
        if step in GRID and value is not None:
            result[step] = value
    return result


def r3_ood_series(arm: str, column: str) -> dict[int, float]:
    csv_path = c.MINI_ROOT / "R3_ood_expansion.csv"
    result = {}
    for row in read_csv(csv_path):
        if row.get("arm") != arm:
            continue
        value = numeric(row.get(column))
        if value is not None and int(row["step"]) in GRID:
            result[int(row["step"])] = value
    return result


def sample_file(root: Path, arm: str, step: int, pattern: str) -> list[Path]:
    actual_arm = "opd" if step == 0 and not (root / arm / c.step_label(step)).exists() else arm
    return sorted((root / actual_arm / c.step_label(step)).rglob(pattern))


def parse_sample_scores(paths: list[Path], metric: str) -> dict[str, float]:
    scores = {}
    for path in paths:
        task_match = re.match(r"samples_(.+?)_\d{4}-\d{2}", path.name)
        task = task_match.group(1) if task_match else path.stem
        for row in c.read_jsonl(path):
            if metric == "ifeval":
                value = row.get("prompt_level_strict_acc")
            else:
                value = None
                for key, candidate in row.items():
                    if key.startswith("exact_match") and isinstance(candidate, (bool, int, float)):
                        value = candidate
                        break
            if value is None:
                continue
            scores[f"{task}:{row.get('doc_id')}"] = float(value)
    return scores


def paired_delta(
    target: dict[str, float],
    reference: dict[str, float],
    *,
    draws: int,
    seed: int,
) -> tuple[int, float | None, float | None, float | None]:
    keys = sorted(set(target).intersection(reference))
    if not keys:
        return 0, None, None, None
    deltas = np.asarray([target[key] - reference[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(draws, len(deltas)))
    means = deltas[indices].mean(axis=1)
    return (
        len(keys),
        float(deltas.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def run_colocation(args: argparse.Namespace) -> None:
    spectra = read_csv(args.mini_root / "R4_v2_spectra_all.csv")
    rows: list[dict[str, Any]] = []
    for arm in c.ARMS:
        legacy = spectral_series(spectra, arm, "legacy")
        h_ood = spectral_series(spectra, arm, "h_ood")
        trajectory = OPD_TRAJECTORY if arm == "opd" else SFT_TRAJECTORY
        id_math = trajectory_series(trajectory, "math500_acc")
        mmlu = trajectory_series(trajectory, "mmlu_pro_acc")
        ifeval = r3_ood_series(arm, "ifeval_prompt_strict")
        truthful = r3_ood_series(arm, "truthfulqa_mc1_acc")
        geo_step, geo_value, geo_contrast = local_extreme(legacy, "high")
        h_step, h_value, h_contrast = local_extreme(h_ood, "high")
        id_step, id_value, id_contrast = local_extreme(id_math, "low")
        if_step, if_value, if_contrast = local_extreme(ifeval, "low")
        mm_step, mm_value, mm_contrast = local_extreme(mmlu, "low")
        tq_step, tq_value, tq_contrast = local_extreme(truthful, "low")
        rows.append(
            {
                "row_type": "colocation_matrix",
                "arm": arm,
                "metric": "",
                "comparison": "",
                "step": "",
                "reference_step": "",
                "n_prompts": "",
                "point_delta": "",
                "ci95_lo": "",
                "ci95_hi": "",
                "geometry_l18_uptick_step": geo_step,
                "geometry_l18_value": geo_value,
                "geometry_l18_local_contrast": geo_contrast,
                "id_math500_low_step": id_step,
                "id_math500_value": id_value,
                "id_math500_local_contrast": id_contrast,
                "h_ood_transient_step": h_step,
                "h_ood_value": h_value,
                "h_ood_local_contrast": h_contrast,
                "ifeval_low_step": if_step,
                "ifeval_value": if_value,
                "ifeval_local_contrast": if_contrast,
                "mmlu_pro_low_step": mm_step,
                "mmlu_pro_value": mm_value,
                "mmlu_pro_local_contrast": mm_contrast,
                "truthfulqa_low_step": tq_step,
                "truthfulqa_value": tq_value,
                "truthfulqa_local_contrast": tq_contrast,
                "geometry_series_json": json.dumps(legacy, sort_keys=True),
                "h_ood_series_json": json.dumps(h_ood, sort_keys=True),
            }
        )

    behavior_specs = {
        "ifeval": {
            "root": R3_OOD,
            "pattern": "samples_ifeval_*.jsonl",
            "dips": {"opd": (10, 5, 20), "sft": (20, 10, 40)},
        },
        "mmlu_pro": {
            "root": args.run_root / "behavior/mmlu_pro",
            "pattern": "samples_mmlu_pro_*.jsonl",
            "dips": {"opd": (40, 20, 160), "sft": (40, 20, 160)},
        },
    }
    for metric, spec in behavior_specs.items():
        for arm in c.ARMS:
            dip, previous, following = spec["dips"][arm]
            target = parse_sample_scores(
                sample_file(spec["root"], arm, dip, spec["pattern"]), metric
            )
            for label, reference_step in (
                ("vs_base", 0),
                ("vs_previous", previous),
                ("vs_following", following),
            ):
                reference = parse_sample_scores(
                    sample_file(spec["root"], arm, reference_step, spec["pattern"]),
                    metric,
                )
                n, point, lo, hi = paired_delta(
                    target,
                    reference,
                    draws=args.behavior_draws,
                    seed=c.stable_seed(args.seed, metric, arm, label),
                )
                rows.append(
                    {
                        "row_type": "behavior_prompt_bootstrap",
                        "arm": arm,
                        "metric": metric,
                        "comparison": label,
                        "step": dip,
                        "reference_step": reference_step,
                        "n_prompts": n,
                        "point_delta": point,
                        "ci95_lo": lo,
                        "ci95_hi": hi,
                        "bootstrap_draws": args.behavior_draws,
                        "bootstrap_unit": "prompt",
                    }
                )

    fields = sorted({key for row in rows for key in row})
    c.write_csv_atomic(args.mini_root / "R4_transient_colocation.csv", rows, fields)
    print(f"[Colocation] rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", action="store_true")
    parser.add_argument("--colocation", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--run-root", type=Path, default=c.RUN_ROOT)
    parser.add_argument("--mini-root", type=Path, default=c.MINI_ROOT)
    parser.add_argument("--behavior-draws", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.all:
        args.m1 = args.colocation = True
    if not (args.m1 or args.colocation):
        parser.print_help()
        return
    if args.m1:
        run_m1(args)
    if args.colocation:
        run_colocation(args)


if __name__ == "__main__":
    main()
