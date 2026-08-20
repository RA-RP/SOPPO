#!/usr/bin/env python3
"""Validate N-2 outputs and reconstruct the off-KD spectra shard from caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cycle09_r4_common as c4
from cycle09_n2_h80_measure import SPECTRA_FIELDS


MINI = c4.MINI_ROOT
OFFKD = Path("/root/autodl-tmp/cycle09_offkd")
N2 = Path("/root/autodl-tmp/cycle09_n2")
STEPS10 = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
STEPS9 = tuple(step for step in STEPS10 if step)
H_STEPS = (0, 5, 10, 20, 40, 160, 624)
STATIC_TASKS = {
    "legacy_S_math",
    "E_ood",
    "E_general",
    "E_math_hard",
    "S_bos__g3",
    "S_bos__g17",
    "S_bos__g31",
}
X_TASKS = {"X_offkd_math"}


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_count(label: str, rows: list[dict[str, str]], expected: int) -> int:
    actual = len(rows)
    if actual != expected:
        raise ValueError(f"{label} row mismatch: expected={expected}, actual={actual}")
    return actual


def assert_steps(label: str, rows: list[dict[str, str]], expected: set[str]) -> None:
    actual = {str(row["step"]) for row in rows}
    if actual != expected:
        raise ValueError(f"{label} step mismatch: expected={expected}, actual={actual}")


def reconstruct_offkd_spectra() -> int:
    rows: list[dict[str, object]] = []
    sources = (
        (OFFKD / "geometry" / "checkpoints", STATIC_TASKS, 2940),
        (OFFKD / "geometry_xoffkd" / "checkpoints", X_TASKS, 420),
    )
    for root, expected_tasks, expected_rows in sources:
        source_rows = []
        for step in STEPS10:
            path = root / f"{c4.step_label(step)}.json"
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "complete" or int(payload.get("step", -1)) != step:
                raise ValueError(f"incomplete spectra cache: {path}")
            source_rows.extend(payload["spectra"])
        assert_count(str(root), source_rows, expected_rows)
        tasks = {str(row["task_id"]) for row in source_rows}
        if tasks != expected_tasks:
            raise ValueError(f"task mismatch for {root}: {tasks}")
        rows.extend(source_rows)

    key = lambda row: (
        row["arm"], str(row["step"]), row["task_id"], row["track"],
        str(row["layer"]), row["module"],
    )
    unique = {key(row): row for row in rows}
    if len(unique) != len(rows):
        raise ValueError(f"duplicate off-KD spectra cells: {len(rows) - len(unique)}")
    ordered = [unique[item] for item in sorted(unique)]
    for row in ordered:
        if set(row) != set(SPECTRA_FIELDS):
            raise ValueError(f"off-KD spectra schema mismatch: {set(row)}")
    c4.write_csv_atomic(MINI / "R4_v2_spectra_offkd.csv", ordered, list(SPECTRA_FIELDS))
    return len(ordered)


def select(
    filename: str,
    predicate: Callable[[dict[str, str]], bool],
) -> list[dict[str, str]]:
    return [row for row in read_csv(MINI / filename) if predicate(row)]


def validate_item2() -> dict[str, int]:
    def is_h80(row: dict[str, str], key: str = "task_id") -> bool:
        task_id = row.get(key, "")
        return (
            row.get("arm") == "opd"
            and row.get("step") == "80"
            and task_id.startswith(("H_opd_bos__", "H_opd_ood__"))
            and "__step_080__" in task_id
        )
    rows = {
        "spectra": select("R4_v2_spectra_backfill_opd.csv", is_h80),
        "m1": select("R4_m1_tail_ec.csv", is_h80),
        "m2": select("R4_m2_output_drift.csv", is_h80),
        "theta": select(
            "R5_theta_reps.csv", lambda row: is_h80(row, "probe")
        ),
    }
    expected = {"spectra": 252, "m1": 504, "m2": 270, "theta": 378}
    actual = {name: assert_count(f"item2 {name}", value, expected[name])
              for name, value in rows.items()}
    manifest_path = MINI / "h80_opd_geometry_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("item2 manifest is not complete")
    protocol = dict(manifest["protocol"])
    protocol["dW"] = "opd_top32_approx_merged_minus_base_fp32"
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache_root = N2 / "h80_opd_v2" / "checkpoints"
    for step in (0, 80):
        cache_path = cache_root / f"{c4.step_label(step)}.json"
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["protocol_fingerprint"] = fingerprint
        atomic_json(cache_path, payload)
    manifest["protocol"] = protocol
    manifest["protocol_fingerprint"] = fingerprint
    manifest["dW_track"] = "OPD top-32 approximation of merged(fp32)-base(fp32)"
    manifest["provenance_normalized_by"] = "cycle09_n2_finalize.py"
    atomic_json(manifest_path, manifest)
    return actual


def validate_item1() -> dict[str, int]:
    tasks = STATIC_TASKS | X_TASKS
    m1 = select(
        "R4_m1_tail_ec.csv",
        lambda row: row.get("arm") == "offkd" and row.get("task_id") in tasks,
    )
    m2 = select(
        "R4_m2_output_drift.csv",
        lambda row: row.get("arm") == "offkd" and row.get("task_id") in tasks,
    )
    theta = select(
        "R5_theta_reps.csv",
        lambda row: row.get("arm") == "offkd" and row.get("probe") in tasks,
    )
    assert_steps("item1 m1", m1, {str(step) for step in STEPS10})
    assert_steps("item1 m2", m2, {str(step) for step in STEPS10})
    assert_steps("item1 theta", theta, {str(step) for step in STEPS9})
    return {
        "spectra": reconstruct_offkd_spectra(),
        "m1": assert_count("item1 m1", m1, 6720),
        "m2": assert_count("item1 m2", m2, 3600),
        "theta": assert_count("item1 theta", theta, 4536),
    }


def validate_item3() -> int:
    rows = select(
        "R5_raw_er_fixed.csv", lambda row: row.get("arm") == "offkd"
    )
    assert_steps("item3 raw ER", rows, {str(step) for step in STEPS10})
    return assert_count("item3 raw ER", rows, 150)


def validate_item5() -> dict[str, int]:
    rho = select(
        "T4_rho_dualtrack.csv",
        lambda row: row.get("track", "").startswith("offkd_"),
    )
    theta = select("R3_theta_w.csv", lambda row: row.get("arm") == "offkd")
    assert_steps("item5 rho", rho, {str(step) for step in STEPS9})
    assert_steps("item5 theta", theta, {str(step) for step in STEPS9})
    return {
        "rho": assert_count("item5 rho", rho, 756),
        "theta_w": assert_count("item5 theta", theta, 945),
    }


def validate_item4() -> dict[str, int]:
    path = MINI / "offkd_h_geometry_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("steps") != list(H_STEPS):
        raise ValueError("item4 manifest is incomplete or has the wrong grid")
    return {key: int(value) for key, value in manifest["rows"].items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item2-only", action="store_true")
    args = parser.parse_args()
    if args.item2_only:
        print(json.dumps(validate_item2(), sort_keys=True))
        return

    results = {
        "item1_xoffkd_geometry": validate_item1(),
        "item2_opd_h80": validate_item2(),
        "item3_offkd_raw_er": validate_item3(),
        "item4_offkd_h": validate_item4(),
        "item5_offkd_weight_geometry": validate_item5(),
    }
    files = [
        "R4_v2_spectra_offkd.csv",
        "R4_v2_spectra_backfill_opd.csv",
        "R4_v2_spectra_h_offkd.csv",
        "R4_m1_tail_ec.csv",
        "R4_m2_output_drift.csv",
        "R5_theta_reps.csv",
        "R5_raw_er_fixed.csv",
        "T4_rho_dualtrack.csv",
        "R3_theta_w.csv",
    ]
    manifest = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "specification": "mypaper/theory/stage_plan_handoff.md N-2 plus user pull-forward",
        "results": results,
        "files": {
            name: {"path": str(MINI / name), "sha256": sha256(MINI / name)}
            for name in files
        },
        "shutdown": "disabled",
    }
    atomic_json(MINI / "n2_completion_manifest.json", manifest)
    atomic_json(
        N2 / "N2_STATUS.json",
        {
            "state": "complete",
            "detail": "all five pulled-forward N-2 items validated",
            "updated_at": manifest["created_at"],
            "manifest": str(MINI / "n2_completion_manifest.json"),
            "shutdown": "disabled",
        },
    )
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
