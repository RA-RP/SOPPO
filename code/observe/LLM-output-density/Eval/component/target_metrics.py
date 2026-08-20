import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from Eval.component.result_files import iter_visible_json_files, split_model_size


def _metric_candidates(metric: str) -> List[str]:
    candidates = [metric]
    if "," in metric:
        candidates.append(metric.split(",", 1)[0])
    return list(dict.fromkeys(candidates))


def _lookup_metric(metric_map: Dict[str, Any], metric: str) -> Optional[Any]:
    for candidate in _metric_candidates(metric):
        value = metric_map.get(candidate)
        if isinstance(value, (int, float)):
            return value

    metric_prefix = metric.split(",", 1)[0]
    for key, value in metric_map.items():
        if isinstance(value, (int, float)) and str(key).split(",", 1)[0] == metric_prefix:
            return value
    return None


def _extract_value(data: Dict[str, Any], spec: Dict[str, Any]) -> Optional[Any]:
    field = str(spec.get("field", "results"))
    task_key = str(spec.get("task_key") or spec.get("task") or "")
    metric = str(spec.get("metric") or "")
    if not task_key or not metric:
        return None

    container = data.get(field)
    if not isinstance(container, dict):
        return None

    task_metrics = container.get(task_key)
    if isinstance(task_metrics, dict):
        return _lookup_metric(task_metrics, metric)

    # Some lm-eval outputs put group metrics under results or use aliases.
    for candidate_key, candidate_metrics in container.items():
        if str(candidate_key) == task_key and isinstance(candidate_metrics, dict):
            return _lookup_metric(candidate_metrics, metric)
    return None


def _candidate_roots(origin_root: Path, fix_root: Path, json_task: str) -> Iterable[Path]:
    fix_task_root = fix_root / json_task
    origin_task_root = origin_root / json_task
    if fix_task_root.exists():
        yield fix_task_root
    if origin_task_root.exists() and origin_task_root != fix_task_root:
        yield origin_task_root


def _iter_task_jsons(origin_root: Path, fix_root: Path, json_task: str):
    seen = set()
    for task_root in _candidate_roots(origin_root, fix_root, json_task):
        for file_path in iter_visible_json_files(str(task_root)):
            if file_path in seen:
                continue
            seen.add(file_path)
            source, size, suffix = split_model_size(file_path.stem)
            if not source or not size:
                print(f"⚠️ target_metrics 文件命名格式不符，跳过: {file_path}")
                continue
            yield file_path, source, size, suffix


def get_target_metrics_csv(
    *,
    origin_root: str | Path,
    fix_root: str | Path,
    output_root: str | Path,
    metric_specs: List[Dict[str, Any]],
    output_name: str = "target_metrics_results.csv",
) -> None:
    """Build one compact CSV from eval.target_metrics specs."""
    origin_root = Path(origin_root)
    fix_root = Path(fix_root)
    output_root = Path(output_root)
    os.makedirs(output_root, exist_ok=True)

    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    missing: List[Dict[str, Any]] = []

    for spec in metric_specs:
        record_name = str(spec.get("record_name") or spec.get("name") or "").strip()
        json_task = str(spec.get("json_task") or spec.get("task") or spec.get("task_key") or "").strip()
        if not record_name or not json_task:
            print(f"⚠️ target_metrics spec 缺少 record_name/json_task，跳过: {spec}")
            continue

        found_any_file = False
        for eval_file, source, size, suffix in _iter_task_jsons(origin_root, fix_root, json_task):
            found_any_file = True
            key = (source, size)
            row = rows.setdefault(key, {"Source": source, "DataSize": size, "EndTime": suffix})
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                missing.append({
                    "record_name": record_name,
                    "json_task": json_task,
                    "source": source,
                    "DataSize": size,
                    "reason": f"json_load_failed: {exc}",
                    "file": str(eval_file),
                })
                continue

            value = _extract_value(data, spec)
            if value is None:
                missing.append({
                    "record_name": record_name,
                    "json_task": json_task,
                    "field": spec.get("field", "results"),
                    "task_key": spec.get("task_key") or spec.get("task"),
                    "metric": spec.get("metric"),
                    "source": source,
                    "DataSize": size,
                    "reason": "metric_not_found",
                    "file": str(eval_file),
                })
                continue
            row[record_name] = value

        if not found_any_file:
            missing.append({
                "record_name": record_name,
                "json_task": json_task,
                "field": spec.get("field", "results"),
                "task_key": spec.get("task_key") or spec.get("task"),
                "metric": spec.get("metric"),
                "reason": "task_json_dir_or_files_missing",
            })

    if not rows:
        print("⚠️ target_metrics 没有生成任何行，CSV 未写入。")
        if missing:
            missing_path = output_root / "target_metrics_missing.csv"
            pd.DataFrame(missing).to_csv(missing_path, index=False, encoding="utf-8-sig")
            print(f"⚠️ target_metrics missing report saved to {missing_path}")
        return

    record_columns = [
        str(spec.get("record_name") or spec.get("name"))
        for spec in metric_specs
        if str(spec.get("record_name") or spec.get("name") or "").strip()
    ]
    df = pd.DataFrame(rows.values())
    for column in record_columns:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[["Source", "DataSize", "EndTime", *record_columns]]
    df = df.sort_values(by=["Source", "DataSize", "EndTime"])

    csv_path = output_root / output_name
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ target_metrics results saved to {csv_path}")

    if missing:
        missing_path = output_root / "target_metrics_missing.csv"
        pd.DataFrame(missing).to_csv(missing_path, index=False, encoding="utf-8-sig")
        print(f"⚠️ target_metrics missing report saved to {missing_path}")
