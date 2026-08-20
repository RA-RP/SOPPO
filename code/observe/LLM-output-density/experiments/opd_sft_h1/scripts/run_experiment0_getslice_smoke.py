from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.geometry_metrics import effective_rank, principal_angle_unavailable, spectral_gap
from opd_sft_h1.geometry_reader import read_geometry_rows
from opd_sft_h1.paths import ensure_dir, resolve_repo_path
from opd_sft_h1.registry import append_jsonl, validate_run_record


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _geometry_metric_rows(rows: list[dict[str, Any]], spectral_gap_k: int) -> list[dict[str, Any]]:
    metric_rows: list[dict[str, Any]] = []
    for row in rows:
        sigma = row["singular_values"]
        metric_rows.append(
            {
                "source": row["source"],
                "step": row["step"],
                "probe_distribution": row["probe_distribution"],
                "probe_source": row["probe_source"],
                "layer": row["layer"],
                "module": row["module"],
                "effective_rank": effective_rank(sigma),
                "spectral_gap_k": spectral_gap_k,
                "spectral_gap": spectral_gap(sigma, spectral_gap_k),
                **principal_angle_unavailable(),
                "singular_json_path": row["singular_json_path"],
            }
        )
    return metric_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--getslice-root", required=True)
    parser.add_argument("--output-root", default="/root/autodl-tmp/exp0609/experiment0_getslice_smoke")
    parser.add_argument("--spectral-gap-k", type=int, default=1)
    args = parser.parse_args()

    getslice_root = Path(args.getslice_root)
    output_root = resolve_repo_path(args.output_root)
    assert output_root is not None
    tables_dir = ensure_dir(output_root / "tables")
    registry_dir = ensure_dir(output_root / "registry")

    rows = read_geometry_rows(getslice_root)
    geometry_rows = [{**row, "singular_values": json.dumps(row["singular_values"])} for row in rows]
    metric_rows = _geometry_metric_rows(rows, args.spectral_gap_k)

    geometry_path = tables_dir / "geometry_long.csv"
    metrics_path = tables_dir / "geometry_metrics.csv"
    _write_csv(
        geometry_rows,
        geometry_path,
        [
            "source",
            "step",
            "probe_distribution",
            "probe_source",
            "layer",
            "module",
            "singular_values",
            "singular_json_path",
        ],
    )
    _write_csv(
        metric_rows,
        metrics_path,
        [
            "source",
            "step",
            "probe_distribution",
            "probe_source",
            "layer",
            "module",
            "effective_rank",
            "spectral_gap_k",
            "spectral_gap",
            "principal_angle",
            "principal_angle_status",
            "singular_json_path",
        ],
    )

    record = {
        "run_id": "experiment0_getslice_smoke",
        "trajectory_group_id": "opd_sft_h1",
        "method": "trl_opd_like",
        "role_label": "Experiment0-GetSlice-smoke",
        "parent_run_id": None,
        "start_checkpoint": None,
        "seed": None,
        "model": {},
        "data": {"getslice_root": str(getslice_root)},
        "training": {},
        "artifacts": {"geometry_long": str(geometry_path), "geometry_metrics": str(metrics_path)},
        "status": "completed" if rows else "completed_no_geometry_rows",
        "teacher_model": None,
        "teacher_mode": None,
        "lmbda": None,
        "beta": None,
        "loss_top_k": None,
        "use_vllm": False,
        "use_teacher_server": False,
        "teacher_model_server_url": None,
    }
    validate_run_record(record)
    append_jsonl(registry_dir / "run_registry.jsonl", record)

    print(json.dumps({"geometry_rows": len(rows), "geometry_long": str(geometry_path), "geometry_metrics": str(metrics_path)}))


if __name__ == "__main__":
    main()
