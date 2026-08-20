from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .ood_metrics import (
    general_ood_avg,
    general_ood_penalty,
    infer_score_scale,
    math500_gain,
    per_benchmark_drop,
    worst_ood_drop,
)
from .paths import ensure_dir
from .registry import validate_checkpoint_record


DEFAULT_OOD_BENCHMARKS = [
    "MMLU",
    "MMLU-STEM",
    "MMLU-Humanities",
    "MMLU-Social Sciences",
    "MMLU-Other",
    "TruthfulQA-MC1",
    "TruthfulQA-MC2",
    "WinoGrande",
    "IFEval prompt strict",
    "IFEval instruction strict",
]


def _safe_str(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _checkpoint_id(row: dict[str, Any]) -> str:
    source = _safe_str(row.get("Source"))
    size = _safe_str(row.get("DataSize"))
    end_time = _safe_str(row.get("EndTime"))
    parts = [part for part in (source, size, end_time) if part]
    return "__".join(parts)


def _select_baseline(df: pd.DataFrame, baseline_source: str | None, baseline_data_size: str | None) -> dict[str, Any]:
    candidate = df
    if baseline_source is not None and "Source" in df.columns:
        candidate = candidate[candidate["Source"].astype(str) == str(baseline_source)]
    if baseline_data_size is not None and "DataSize" in df.columns:
        candidate = candidate[candidate["DataSize"].astype(str) == str(baseline_data_size)]
    if candidate.empty:
        raise ValueError("No baseline row matched the requested baseline selector")
    return candidate.iloc[0].to_dict()


def _checkpoint_record(
    row: dict[str, Any],
    output_root: Path,
    method: str,
    trajectory_group_id: str | None,
    seed: int | None,
) -> dict[str, Any]:
    source = _safe_str(row.get("Source"))
    checkpoint_id = _checkpoint_id(row)
    record = {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": row.get("checkpoint_path"),
        "run_id": source or checkpoint_id,
        "trajectory_group_id": trajectory_group_id,
        "method": method,
        "role_label": source or method,
        "parent_run_id": None,
        "start_checkpoint": row.get("start_checkpoint"),
        "seed": seed,
        "model": {"source": source},
        "data": {"DataSize": _safe_str(row.get("DataSize")), "EndTime": _safe_str(row.get("EndTime"))},
        "training": {},
        "artifacts": {"eval_trajectory_csv": str(output_root / "tables" / "eval_trajectory.csv")},
        "status": "eval_ingested",
        "teacher_model": row.get("teacher_model"),
        "teacher_mode": row.get("teacher_mode"),
        "lmbda": row.get("lmbda"),
        "beta": row.get("beta"),
        "loss_top_k": row.get("loss_top_k"),
        "use_vllm": bool(row.get("use_vllm", False)),
        "use_teacher_server": bool(row.get("use_teacher_server", False)),
        "teacher_model_server_url": row.get("teacher_model_server_url"),
    }
    validate_checkpoint_record(record)
    return record


def ingest_target_metrics(
    csv_path: str | Path,
    output_root: str | Path,
    *,
    baseline_source: str | None = None,
    baseline_data_size: str | None = None,
    ood_benchmarks: list[str] | None = None,
    method: str = "trl_opd_like",
    trajectory_group_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Path]:
    csv_path = Path(csv_path)
    output_root = Path(output_root)
    tables_dir = ensure_dir(output_root / "tables")
    registry_dir = ensure_dir(output_root / "registry")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    baseline = _select_baseline(df, baseline_source, baseline_data_size)
    benchmarks = ood_benchmarks or DEFAULT_OOD_BENCHMARKS
    present_benchmarks = [benchmark for benchmark in benchmarks if benchmark in df.columns]

    eval_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []

    for _, series in df.iterrows():
        row = series.to_dict()
        checkpoint_id = _checkpoint_id(row)
        row["checkpoint_id"] = checkpoint_id
        row["math500_gain"] = math500_gain(row.get("MATH500"), baseline.get("MATH500"))
        row["general_ood_avg"] = general_ood_avg(row, present_benchmarks)
        row["general_ood_penalty_p2"] = general_ood_penalty(row, baseline, present_benchmarks, p=2)
        row["general_ood_penalty_p3"] = general_ood_penalty(row, baseline, present_benchmarks, p=3)
        row["general_ood_penalty"] = row["general_ood_penalty_p2"]
        row["worst_ood_drop"] = worst_ood_drop(row, baseline, present_benchmarks)
        row["score_scale"] = infer_score_scale(row, [*present_benchmarks, "MATH500"])
        eval_rows.append(row)

        drops = per_benchmark_drop(row, baseline, present_benchmarks)
        for benchmark, drop in drops.items():
            ood_rows.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "benchmark": benchmark,
                    "score": row.get(benchmark),
                    "baseline_score": baseline.get(benchmark),
                    "drop": drop,
                    "score_scale": row["score_scale"],
                }
            )

        checkpoint_records.append(_checkpoint_record(row, output_root, method, trajectory_group_id, seed))

    eval_path = tables_dir / "eval_trajectory.csv"
    penalty_path = tables_dir / "ood_penalty.csv"
    checkpoint_path = registry_dir / "checkpoints.jsonl"

    pd.DataFrame(eval_rows).to_csv(eval_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(ood_rows).to_csv(penalty_path, index=False, encoding="utf-8-sig")
    with checkpoint_path.open("w", encoding="utf-8") as f:
        for record in checkpoint_records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")

    return {
        "eval_trajectory": eval_path,
        "ood_penalty": penalty_path,
        "checkpoints": checkpoint_path,
    }
