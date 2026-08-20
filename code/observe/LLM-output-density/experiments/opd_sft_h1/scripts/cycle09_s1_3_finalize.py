#!/usr/bin/env python3
"""Finalize S1-3 from completed per-cell ER bootstrap arrays.

This is deliberately CPU-only. It validates the 16 arrays produced by
cycle09_s1_3_transient_ci.py, reports same-step arm contrasts plus the
pre-registered offKD@20 - SFT@40 peak contrast, and appends rows to the shared
R5 bootstrap table without changing its schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import cycle09_r4_common as c4


TASK = "E_ood"
LAYER = 18
STEPS = (5, 10, 20, 40, 80)
ARMS = ("opd", "sft", "offkd")
DRAWS = 256
SEED = 42
CACHE = Path("/root/autodl-tmp/cycle09_s1_3/cache")
A4_CACHE = Path("/root/autodl-tmp/cycle09_r5/scratch/a4_cache")
FACTOR_ROOT = Path("/root/autodl-tmp/cycle09_r4/scratch/bootstrap_factors")
MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
SHARED = MINI / "R5_bootstrap_ci.csv"
SOURCE_KIND = "stage1_transient_er_bootstrap"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def interval(values: np.ndarray) -> tuple[float, float, float]:
    if values.shape != (DRAWS,) or not np.isfinite(values).all():
        raise ValueError(f"invalid interval input: shape={values.shape}")
    return (
        float(values.mean()),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0.0 or hi < 0.0)


def load_arrays() -> tuple[dict[tuple[str, int], np.ndarray], list[dict]]:
    cells: dict[tuple[str, int], np.ndarray] = {}
    provenance = []
    expected = [("opd", 0)] + [(arm, step) for arm in ARMS for step in STEPS]
    for arm, step in expected:
        path = CACHE / f"{arm}__{c4.step_label(step)}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = np.load(path, allow_pickle=False)
        if values.shape != (DRAWS, len(c4.MODULES)):
            raise ValueError(f"wrong cache shape {values.shape}: {path}")
        if values.dtype != np.float64 or not np.isfinite(values).all():
            raise ValueError(f"invalid cache dtype/values: {path}")
        cells[(arm, step)] = values
        provenance.append(
            {
                "arm": arm,
                "step": step,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return cells, provenance


def a4_parity(cells: dict[tuple[str, int], np.ndarray]) -> list[dict]:
    checks = []
    pairs = (("opd", 0), ("opd", 5), ("opd", 20), ("sft", 5), ("sft", 20))
    for arm, step in pairs:
        source = A4_CACHE / f"{TASK}__{arm}__{c4.step_label(step)}.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        with np.load(source, allow_pickle=False) as payload:
            reference = payload["er"]
        current = cells[(arm, step)]
        max_abs = float(np.max(np.abs(current - reference)))
        checks.append(
            {
                "arm": arm,
                "step": step,
                "source": str(source),
                "max_abs_difference": max_abs,
                "array_equal": bool(np.array_equal(current, reference)),
                "allclose_atol_1e-5": bool(
                    np.allclose(current, reference, rtol=1e-6, atol=1e-5)
                ),
            }
        )
        if not checks[-1]["allclose_atol_1e-5"]:
            raise RuntimeError(f"A4 parity failed for {arm}/{step}: {max_abs}")
    return checks


def series(
    offsets: dict[tuple[str, int], np.ndarray],
    arm: str,
    step: int,
    module_index: int | None,
) -> np.ndarray:
    values = offsets[(arm, step)]
    return values.mean(axis=1) if module_index is None else values[:, module_index]


def blank_row(columns: list[str]) -> dict:
    return {column: np.nan for column in columns}


def fill_interval(row: dict, prefix: str, values: np.ndarray) -> tuple[float, float, float]:
    mean, lo, hi = interval(values)
    row[f"{prefix}_mean"] = mean
    row[f"{prefix}_ci95_lo"] = lo
    row[f"{prefix}_ci95_hi"] = hi
    return mean, lo, hi


def make_rows(
    cells: dict[tuple[str, int], np.ndarray], columns: list[str]
) -> pd.DataFrame:
    base = cells[("opd", 0)]
    offsets = {(arm, step): cells[(arm, step)] - base for arm in ARMS for step in STEPS}
    rows = []
    module_specs = list(enumerate(c4.MODULES)) + [(None, "mean_fixed_7_modules")]

    for step in STEPS:
        for module_index, module in module_specs:
            row = blank_row(columns)
            row.update(
                {
                    "task_id": TASK,
                    "step": step,
                    "layer": LAYER,
                    "module": module,
                    "metric": "er_offset_vs_base",
                    "bootstrap_unit": "sample; windows nested",
                    "bootstrap_draws": DRAWS,
                    "quantity_definition": (
                        "ER(W_arm_step, gram_draw) - ER(W_base, gram_draw)"
                    ),
                    "source_kind": SOURCE_KIND,
                }
            )
            values = {arm: series(offsets, arm, step, module_index) for arm in ARMS}
            for arm in ARMS:
                fill_interval(row, arm, values[arm])

            comparisons = (
                ("opd_minus_sft", values["opd"] - values["sft"]),
                ("offkd_minus_opd", values["offkd"] - values["opd"]),
                ("offkd_minus_sft", values["offkd"] - values["sft"]),
            )
            for prefix, difference in comparisons:
                _, lo, hi = fill_interval(row, prefix, difference)
                row[f"{prefix}_ci_excludes_zero"] = excludes_zero(lo, hi)
            row["ci_excludes_zero"] = row["opd_minus_sft_ci_excludes_zero"]
            rows.append(row)

    for module_index, module in module_specs:
        row = blank_row(columns)
        row.update(
            {
                "task_id": "E_ood_cross_peak",
                "step": 20,
                "layer": LAYER,
                "module": module,
                "metric": "er_offset_peak_gap",
                "bootstrap_unit": "sample; windows nested",
                "bootstrap_draws": DRAWS,
                "quantity_definition": (
                    "offKD ER offset at step 20 minus SFT ER offset at step 40"
                ),
                "source_kind": SOURCE_KIND,
            }
        )
        offkd = series(offsets, "offkd", 20, module_index)
        sft = series(offsets, "sft", 40, module_index)
        fill_interval(row, "offkd", offkd)
        fill_interval(row, "sft", sft)
        _, lo, hi = fill_interval(row, "offkd_minus_sft", offkd - sft)
        row["offkd_minus_sft_ci_excludes_zero"] = excludes_zero(lo, hi)
        row["ci_excludes_zero"] = row["offkd_minus_sft_ci_excludes_zero"]
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    if not SHARED.is_file():
        raise FileNotFoundError(SHARED)
    shared = pd.read_csv(SHARED)
    columns = list(shared.columns)
    required = {
        "offkd_mean",
        "offkd_minus_opd_mean",
        "offkd_minus_sft_mean",
        "quantity_definition",
        "source_kind",
    }
    if not required.issubset(columns):
        raise RuntimeError(f"shared schema missing: {sorted(required - set(columns))}")

    cells, cache_provenance = load_arrays()
    parity = a4_parity(cells)
    rows = make_rows(cells, columns)
    if len(rows) != 48:
        raise RuntimeError(f"unexpected row count: {len(rows)}")

    retained = shared[shared["source_kind"].fillna("") != SOURCE_KIND].copy()
    combined = pd.concat([retained, rows], ignore_index=True)
    atomic_csv(rows, MINI / "S1_transient_ci.csv")
    atomic_csv(combined, SHARED)

    corpus = Path("/root/autodl-tmp/cycle09_r4/corpora/fixed/E_ood.jsonl")
    rng = np.random.default_rng(c4.stable_seed(SEED, TASK))
    draw_indices = rng.integers(0, 128, size=(DRAWS, 128))
    factor_files = []
    for arm, step in [("opd", 0)] + [(a, s) for a in ARMS for s in STEPS]:
        path = FACTOR_ROOT / arm / c4.step_label(step) / f"{TASK}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        factor_files.append(
            {
                "arm": arm,
                "step": step,
                "path": str(path),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    atomic_json(
        {
            "schema_version": 2,
            "task": "S1-3",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "probe": TASK,
            "steps": list(STEPS),
            "arms": list(ARMS),
            "layer": LAYER,
            "modules": list(c4.MODULES),
            "module_summary": "arithmetic mean of the fixed seven module ER values per draw",
            "draws": DRAWS,
            "seed": SEED,
            "stable_seed": c4.stable_seed(SEED, TASK),
            "n_samples": 128,
            "draw_indices_sha256": hashlib.sha256(draw_indices.tobytes()).hexdigest(),
            "bootstrap_unit": "sample; windows nested",
            "paired": True,
            "indices_shared_across_arms_and_steps": True,
            "peak_contrast": "offkd@20 minus sft@40",
            "corpus": {
                "path": str(corpus),
                "size_bytes": corpus.stat().st_size,
                "sha256": sha256_file(corpus),
            },
            "cache_arrays": cache_provenance,
            "a4_parity": parity,
            "factor_files": factor_files,
            "cache_reuse": {
                "opd@5": "R5 A4 ER array",
                "opd@20": "R5 A4 ER array",
                "sft@5": "R5 A4 ER array",
                "sft@20": "R5 A4 ER array",
                "all_other_cells": "computed by cycle09_s1_3_transient_ci.py",
            },
            "output_rows": len(rows),
            "shared_rows_before": len(shared),
            "shared_rows_after": len(combined),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MINI / "S1_transient_ci_manifest.json",
    )
    print(f"[S1-3 finalize] rows={len(rows)} shared={len(shared)}->{len(combined)}")
    print(
        rows[rows["module"] == "mean_fixed_7_modules"][
            [
                "task_id",
                "step",
                "metric",
                "opd_mean",
                "sft_mean",
                "offkd_mean",
                "opd_minus_sft_mean",
                "offkd_minus_opd_mean",
                "offkd_minus_sft_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
