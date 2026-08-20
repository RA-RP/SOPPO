#!/usr/bin/env python3
"""CPU lane for A3 bootstrap, A4 grouped diagnostics, A7 provenance, and A9 schema."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cycle09_block3_common as b3
import cycle09_stage4_state_displacement as s4

ROOT = s4.ROOT / "cpu"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row}) if rows else []
    with path.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path.with_suffix(".tmp"), path)


def rows_from_cells(tag: str) -> list[dict[str, Any]]:
    rows = []
    for p in s4.CELLS.rglob(f"*.{tag}.json"):
        data = s4.read_json(p, {})
        if data.get("status") == "complete":
            rows.extend(data.get("rows", []))
    return rows


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    """A3 sample-count inventory/bootstraps; skips cells lacking retained samples."""
    rows = []
    generator = np.random.default_rng(args.seed)
    for meta in s4.PROFILES.rglob(f"*.{args.tag}.json"):
        data = s4.read_json(meta, {})
        profile_path = Path(data.get("profile", ""))
        if not data.get("retains_per_sample") or not profile_path.is_file():
            continue
        bundle = torch.load(profile_path, map_location="cpu", weights_only=True)
        count = int(bundle["n_samples"])
        for n in sorted({min(count, x) for x in (8, 16, 32, 64, 128)}):
            if not n:
                continue
            for draw in range(args.draws):
                selected = generator.integers(0, count, size=n)
                rows.append({
                    "model": bundle["model"], "arm": bundle["arm"], "checkpoint": bundle["step"],
                    "probe_name": bundle["probe"], "sample_count": n, "draw": draw,
                    "selected_sample_sha256": b3.sha256_json(selected.tolist()),
                    "status": "resample_ready",
                })
    output = ROOT / "state_displacement_sample_count_bootstrap.csv"
    atomic_csv(output, rows)
    return {"status": "complete", "output": str(output), "rows": len(rows)}


def _finite_matrix(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values, keep = [], []
    for index, row in enumerate(rows):
        try:
            vector = [float(row[field]) for field in fields]
        except (KeyError, TypeError, ValueError):
            continue
        if all(np.isfinite(vector)):
            values.append(vector)
            keep.append(index)
    return np.asarray(values, dtype=np.float64), np.asarray(keep, dtype=np.int64)


def a4(args: argparse.Namespace) -> dict[str, Any]:
    """Strict same-cell baselines and checkpoint-grouped held-out arm discrimination."""
    rows = [
        row for row in rows_from_cells(args.tag)
        if row.get("epsilon") == 0.05 and row.get("arm") in ("opd", "offkd")
    ]
    output = ROOT / "strict_weight_activation_baseline_full_cells.csv"
    atomic_csv(output, rows)
    result_rows: list[dict[str, Any]] = []
    baseline_fields = ("weight_norm_fro", "weight_effective_rank")
    ours_fields = baseline_fields + (
        "state_rank",
        "displacement_norm_normalized",
        "displacement_rank_normalized",
    )
    for model in sorted({str(row.get("model")) for row in rows}):
        local = [row for row in rows if row.get("model") == model]
        groups = np.asarray([int(row["checkpoint"]) for row in local])
        labels = np.asarray([int(row["arm"] == "opd") for row in local])
        unique_groups = np.unique(groups)
        if len(unique_groups) < 3 or len(np.unique(labels)) != 2:
            result_rows.append({
                "model": model, "status": "DEFERRED_INSUFFICIENT_CHECKPOINT_GROUPS",
                "checkpoint_groups": len(unique_groups), "rows": len(local),
            })
            continue
        n_splits = min(5, len(unique_groups))
        splitter = GroupKFold(n_splits=n_splits)
        for feature_name, fields in (("strict_weight_only", baseline_fields), ("ours_plus_strict_weight", ours_fields)):
            matrix, keep = _finite_matrix(local, fields)
            if len(keep) < n_splits * 4:
                result_rows.append({
                    "model": model, "feature_set": feature_name,
                    "status": "DEFERRED_NONFINITE_FEATURES", "rows": int(len(keep)),
                })
                continue
            y, g = labels[keep], groups[keep]
            oof = np.full(len(y), np.nan, dtype=np.float64)
            for train, test in splitter.split(matrix, y, g):
                if len(np.unique(y[train])) != 2:
                    continue
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
                )
                clf.fit(matrix[train], y[train])
                oof[test] = clf.predict_proba(matrix[test])[:, 1]
            valid = np.isfinite(oof)
            if valid.sum() == 0 or len(np.unique(y[valid])) != 2:
                result_rows.append({
                    "model": model, "feature_set": feature_name,
                    "status": "DEFERRED_INVALID_GROUP_SPLIT", "rows": int(valid.sum()),
                })
                continue
            result_rows.append({
                "model": model, "feature_set": feature_name, "status": "complete",
                "folds": n_splits, "rows": int(valid.sum()), "checkpoint_groups": len(unique_groups),
                "heldout_log_loss": float(log_loss(y[valid], oof[valid], labels=[0, 1])),
                "heldout_auc": float(roc_auc_score(y[valid], oof[valid])),
                "heldout_balanced_accuracy": float(
                    balanced_accuracy_score(y[valid], (oof[valid] >= 0.5).astype(np.int64))
                ),
            })
    cv_output = ROOT / "incremental_arm_discrimination.csv"
    atomic_csv(cv_output, result_rows)
    payload = {
        "schema_version": "cycle09_stage4_a4_v2",
        "status": "complete",
        "task": "A4 strict baseline and checkpoint-grouped held-out arm discrimination",
        "input_rows": len(rows),
        "track_a": str(cv_output),
        "track_b": "DEFERRED_NONBLOCKING: domain-matched behavior joins are finalized after behavior manifests are audited",
        "output": str(output),
        "created_utc": b3.utc_now(),
    }
    atomic_json(ROOT / "incremental_information_cv_manifest.json", payload)
    return {"status": "complete", "output": str(output), "rows": len(rows)}


def a7(_: argparse.Namespace) -> dict[str, Any]:
    paths = [
        b3.REPO / "experiments/opd_sft_h1/scripts/cycle09_stage4_state_displacement.py",
        b3.REPO / "experiments/opd_sft_h1/scripts/cycle09_stage4_readout.py",
        b3.REPO / "experiments/opd_sft_h1/scripts/cycle09_r4_campaign.py",
        b3.REPO / "experiments/opd_sft_h1/scripts/cycle09_llama_behavior.py",
    ]
    rows = [{"path": str(p), "exists": p.is_file(), "sha256": b3.sha256_file(p) if p.is_file() else None}
            for p in paths]
    output = ROOT / "trainer_arm_implementation_audit.json"
    atomic_json(output, {"schema_version": 1, "status": "partial", "files": rows, "created_utc": b3.utc_now()})
    return {"status": "complete", "output": str(output), "rows": len(rows)}


def a9(_: argparse.Namespace) -> dict[str, Any]:
    fields = [
        "model", "arm", "checkpoint", "domain", "probe_name", "layer", "module", "epsilon",
        "support_ruler", "centered", "sample_count", "state_rank", "displacement_norm_raw",
        "displacement_norm_normalized", "displacement_rank", "matrix_cosine",
        "left_subspace_overlap", "right_subspace_overlap", "artifact_path",
    ]
    output = ROOT / "state_displacement_schema.json"
    atomic_json(output, {"schema_version": 1, "status": "complete", "fields": fields, "created_utc": b3.utc_now()})
    return {"status": "complete", "output": str(output), "rows": len(fields)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True, choices=("bootstrap", "a4", "a7", "a9"))
    p.add_argument("--tag", default="main")
    p.add_argument("--draws", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    value = {"bootstrap": bootstrap, "a4": a4, "a7": a7, "a9": a9}[args.phase](args)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
