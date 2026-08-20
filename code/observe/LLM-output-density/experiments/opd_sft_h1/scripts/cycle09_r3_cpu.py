#!/usr/bin/env python3
"""Cycle 09 Round 3 CPU artifacts.

R3-1: module-resolved L9/L18/L27 ER uptick and activation-theta readings.
R3-2: X-conditioned whitened ER preview from Round 2 xMat_X spectra.

Both commands are read-only with respect to Round 2 inputs and write only
Round 3 artifacts under the configured mini directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path("/root/LLM-output-density")
DEFAULT_MINI = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
DEFAULT_R2_RUN = Path("/root/autodl-tmp/cycle09_r2")

ARMS = ("opd", "sft")
STEPS = (0, 5, 10, 20, 40, 160, 624)
LAYERS = (9, 18, 27)
PROBES = ("X_math", "X_ood_knowledge", "X_general", "X_math_hard", "X_bos")
MODULES = (
    "mlp.down_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "self_attn.k_proj",
    "self_attn.o_proj",
    "self_attn.q_proj",
    "self_attn.v_proj",
)
DIP_STEP = {"opd": 5, "sft": 20}
NEIGHBORS = {5: (0, 10), 20: (10, 40)}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def effective_rank(values: list[float]) -> float:
    spectrum = np.asarray(values, dtype=np.float64)
    spectrum = spectrum[np.isfinite(spectrum) & (spectrum > 0)]
    total = float(spectrum.sum())
    if total <= 0.0:
        return math.nan
    probabilities = spectrum / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(math.exp(entropy))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def layer_spectra(path: Path, layer: int) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    data = read_json(path)
    value = data.get(f"layer_{layer}", {})
    return value if isinstance(value, dict) else {}


def x_spectrum_path(r2_run: Path, arm: str, step: int, layer: int, probe: str) -> Path:
    root = r2_run / "getslice" / "outputs" / arm / f"step_{step:03d}"
    full = root / "spectra" / probe / "X" / "xMat_X.json"
    if full.exists():
        return full
    return root / "landmark" / f"layer_{layer}" / probe / "X" / f"layer_{layer}" / "xMat_X.json"


def load_t5(mini: Path, er_probe: str) -> dict[tuple[str, int, str, int], float]:
    path = mini / "T5_full_layer_profile.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing Round 2 profile: {path}")

    values: dict[tuple[str, int, str, int], float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["probe"] != er_probe:
                continue
            values[
                (
                    row["arm"],
                    int(row["layer"]),
                    row["module"],
                    int(row["step"]),
                )
            ] = float(row["effective_rank"])
    return values


def load_theta(mini: Path, rank: int) -> dict[tuple[str, int, str, int], float]:
    path = mini / "T7_theta_r.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing Round 2 theta table: {path}")

    values: dict[tuple[str, int, str, int], float] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] != "vs_base" or int(row["r"]) != rank:
                continue
            if not row.get("theta_u"):
                continue
            values[
                (
                    row["arm"],
                    int(row["layer"]),
                    row["module"],
                    int(row["step_b"]),
                )
            ] = float(row["theta_u"])
    return values


def format_number(value: float | None, digits: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def run_r3_1(mini: Path, er_probe: str, theta_rank: int) -> None:
    er = load_t5(mini, er_probe)
    theta = load_theta(mini, theta_rank)
    rows: list[dict[str, Any]] = []

    for arm in ARMS:
        dip = DIP_STEP[arm]
        previous, following = NEIGHBORS[dip]
        for layer in LAYERS:
            for module in MODULES:
                er_previous = er.get((arm, layer, module, previous))
                er_dip = er.get((arm, layer, module, dip))
                er_following = er.get((arm, layer, module, following))
                uptick = None
                if None not in (er_previous, er_dip, er_following):
                    uptick = min(er_dip - er_previous, er_dip - er_following)

                theta_u = theta.get((arm, layer, module, dip))
                theta_angle = None
                if theta_u is not None:
                    theta_angle = math.degrees(math.acos(max(-1.0, min(1.0, theta_u))))

                rows.append(
                    {
                        "arm": arm,
                        "layer": layer,
                        "module": module,
                        "er_probe": er_probe,
                        "dip_step": dip,
                        "previous_step": previous,
                        "next_step": following,
                        "er_previous": format_number(er_previous),
                        "er_dip": format_number(er_dip),
                        "er_next": format_number(er_following),
                        "strict_local_uptick_er": format_number(uptick),
                        "theta_rank": theta_rank,
                        "theta_u_vs_base": format_number(theta_u),
                        "theta_u_angle_proxy_deg": format_number(theta_angle, 6),
                        "er_source": str(mini / "T5_full_layer_profile.csv"),
                        "theta_source": str(mini / "T7_theta_r.csv"),
                    }
                )

    fields = list(rows[0].keys())
    csv_path = mini / "R3_module_breakdown.csv"
    write_csv(csv_path, rows, fields)

    lines = [
        "# R3-1 — Module Breakdown",
        "",
        "Readings are copied from the Round 2 S-side ER table using its X_math-labelled rows; "
        "the table does not treat repeated probe labels as independent observations.",
        "",
        "| layer | arm | module | ER prev | ER dip | ER next | strict local uptick | theta_u vs base | theta angle proxy deg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {layer} | {arm} | {module} | {er_previous} | {er_dip} | {er_next} | "
            "{strict_local_uptick_er} | {theta_u_vs_base} | {theta_u_angle_proxy_deg} |".format(**row)
        )
    (mini / "R3_module_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[R3-1] wrote {csv_path} rows={len(rows)}", flush=True)


def run_r3_2(mini: Path, r2_run: Path) -> None:
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for arm in ARMS:
        for step in STEPS:
            for layer in LAYERS:
                for probe in PROBES:
                    path = x_spectrum_path(r2_run, arm, step, layer, probe)
                    spectra = layer_spectra(path, layer)
                    for module in MODULES:
                        values = spectra.get(module)
                        if values:
                            rows.append(
                                {
                                    "arm": arm,
                                    "step": step,
                                    "layer": layer,
                                    "probe": probe,
                                    "module": module,
                                    "x_whitened_effective_rank": format_number(effective_rank(values)),
                                    "n_singular_values": len(values),
                                    "source_path": str(path),
                                    "status": "ok",
                                }
                            )
                        else:
                            missing.append(
                                {
                                    "arm": arm,
                                    "step": step,
                                    "layer": layer,
                                    "probe": probe,
                                    "module": module,
                                    "source_path": str(path),
                                    "status": "missing",
                                }
                            )

    fields = [
        "arm",
        "step",
        "layer",
        "probe",
        "module",
        "x_whitened_effective_rank",
        "n_singular_values",
        "source_path",
        "status",
    ]
    csv_path = mini / "R3_xcond_whitened_er_preview.csv"
    write_csv(csv_path, rows, fields)
    write_csv(
        mini / "R3_xcond_whitened_er_preview_missing.csv",
        missing,
        ["arm", "step", "layer", "probe", "module", "source_path", "status"],
    )
    print(
        f"[R3-2] wrote {csv_path} rows={len(rows)} missing={len(missing)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-1", action="store_true", help="write module-resolved ER/theta table")
    parser.add_argument("--r3-2", action="store_true", help="write X-conditioned ER preview")
    parser.add_argument("--all", action="store_true", help="run R3-1 and R3-2")
    parser.add_argument("--mini-root", type=Path, default=DEFAULT_MINI)
    parser.add_argument("--r2-run-root", type=Path, default=DEFAULT_R2_RUN)
    parser.add_argument("--er-probe", default="X_math")
    parser.add_argument("--theta-rank", type=int, default=64)
    args = parser.parse_args()

    if not (args.r3_1 or args.r3_2 or args.all):
        parser.print_help()
        return
    args.mini_root.mkdir(parents=True, exist_ok=True)
    if args.r3_1 or args.all:
        run_r3_1(args.mini_root, args.er_probe, args.theta_rank)
    if args.r3_2 or args.all:
        run_r3_2(args.mini_root, args.r2_run_root)


if __name__ == "__main__":
    main()

