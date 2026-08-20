#!/usr/bin/env python3
# coding: utf-8
"""Shared utilities for offPolicyData pipelines."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PLACEHOLDER_OPEN = "__HEREDOC__"
PLACEHOLDER_CLOSE = "__APPEND__"
RANKING_EFFECTIVE_RANK = "effective_rank"
RANKING_TOPK_KL = "topk_KL"


@dataclass
class Record:
    global_id: int
    source: str
    source_line: int
    question: str
    answer: str
    raw: Dict[str, Any]


def strip_json_line_comments(text: str) -> str:
    output_lines = []
    for line in text.splitlines():
        chars = []
        in_string = False
        escape = False
        i = 0
        while i < len(line):
            ch = line[i]
            if in_string:
                chars.append(ch)
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                chars.append(ch)
                i += 1
                continue
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            chars.append(ch)
            i += 1
        output_lines.append("".join(chars))
    return "\n".join(output_lines)


def load_json_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(strip_json_line_comments(text))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def strip_unmatched_calc_tokens(text: str) -> str:
    parts: List[str] = []
    open_positions: List[int] = []
    idx = 0
    while idx < len(text):
        if text.startswith("<<", idx):
            open_positions.append(len(parts))
            parts.append("<<")
            idx += 2
            continue
        if text.startswith(">>", idx):
            if open_positions:
                open_positions.pop()
                parts.append(">>")
            idx += 2
            continue
        parts.append(text[idx])
        idx += 1
    for pos in open_positions:
        parts[pos] = ""
    return "".join(parts)


def clean_answer_text(answer: str) -> str:
    cleaned = str(answer or "").strip()
    cleaned = cleaned.replace(PLACEHOLDER_OPEN, "<<")
    cleaned = cleaned.replace(PLACEHOLDER_CLOSE, ">>")
    cleaned = strip_unmatched_calc_tokens(cleaned)
    return cleaned.strip()


def load_source_records(config: Dict[str, Any]) -> List[Record]:
    source_root = Path(config["source_root"])
    task_file = config.get("task_file", "gsm8k.jsonl")
    records: List[Record] = []
    for source in config.get("sources", []):
        path = source_root / source / task_file
        if not path.exists():
            raise FileNotFoundError(f"source file not found: {path}")
        kept = 0
        for line_no, raw in read_jsonl(path):
            question = str(raw.get("question", "")).strip()
            answer = clean_answer_text(str(raw.get("answer", "")).strip())
            if not question or not answer:
                continue
            records.append(
                Record(
                    global_id=len(records),
                    source=str(source),
                    source_line=line_no,
                    question=question,
                    answer=answer,
                    raw=raw,
                )
            )
            kept += 1
        print(f"[load] {source}: kept={kept} from {path}")
    if not records:
        raise ValueError("no usable QA records loaded")
    print(f"[load] total usable records: {len(records)}")
    return records


def run_root(config: Dict[str, Any]) -> Path:
    run_name = str(config.get("run_name") or "").strip()
    if not run_name or run_name == "auto":
        run_name = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    return Path(config["output_root"]) / run_name


def ensure_dir_for_stage(path: Path, overwrite: bool, allow_existing: bool = False) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    elif path.exists() and not allow_existing and any(path.iterdir()):
        raise FileExistsError(
            f"stage output already exists and is not empty: {path}. "
            "Use a new run_name or set overwrite_outputs=true."
        )
    path.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, records: Sequence[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            payload = {
                "global_id": record.global_id,
                "source": record.source,
                "source_line": record.source_line,
                "question": record.question,
                "answer": record.answer,
                "source_id": record.raw.get("source_id"),
                "source_index": record.raw.get("source_index"),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> List[Record]:
    records: List[Record] = []
    for _, raw in read_jsonl(path):
        records.append(
            Record(
                global_id=int(raw["global_id"]),
                source=str(raw["source"]),
                source_line=int(raw["source_line"]),
                question=str(raw["question"]),
                answer=str(raw["answer"]),
                raw=raw,
            )
        )
    return records


def write_dataset_dir(
    dataset_dir: Path,
    records: Sequence[Record],
    dataset_alias: str,
    task_file: str,
    train_file: str,
    extra_fields: Optional[Dict[int, Dict[str, Any]]] = None,
) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    task_path = dataset_dir / task_file
    train_path = dataset_dir / train_file
    with task_path.open("w", encoding="utf-8") as task_f, train_path.open("w", encoding="utf-8") as train_f:
        for record in records:
            extra = extra_fields.get(record.global_id, {}) if extra_fields else {}
            payload = dict(record.raw)
            payload.update(
                {
                    "question": record.question,
                    "answer": record.answer,
                    "source": record.source,
                    "source_line": record.source_line,
                    "global_id": record.global_id,
                    **extra,
                }
            )
            task_f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            train_f.write(
                json.dumps(
                    {"question": record.question, "answer": record.answer},
                    ensure_ascii=False,
                )
                + "\n"
            )
    save_json(
        dataset_dir / "dataset_info.json",
        {
            dataset_alias: {
                "file_name": train_file,
                "formatting": "alpaca",
                "columns": {"query": "question", "response": "answer"},
            }
        },
    )


def load_unit_records(unit_dir: Path, task_file: str = "gsm8k.jsonl") -> List[Record]:
    records: List[Record] = []
    task_path = unit_dir / task_file
    for _, raw in read_jsonl(task_path):
        records.append(
            Record(
                global_id=int(raw["global_id"]),
                source=str(raw["source"]),
                source_line=int(raw["source_line"]),
                question=str(raw["question"]),
                answer=str(raw["answer"]),
                raw=raw,
            )
        )
    return records


def discover_task_dirs(task_root: Path, prefix: str) -> List[str]:
    return sorted(path.name for path in task_root.iterdir() if path.is_dir() and path.name.startswith(prefix))


def write_getslice_config(
    config: Dict[str, Any],
    root: Path,
    task_root: Path,
    tasks: Sequence[str],
    s_nsamples: Optional[int] = None,
) -> Path:
    if not tasks:
        raise FileNotFoundError(f"no task dirs found under {task_root}")
    getslice_cfg = load_json_config(Path(config["getslice"]["base_config"]))
    model_override = config.get("getslice", {}).get("model") or config.get("embedding", {}).get("model_path")
    if model_override:
        getslice_cfg["model"] = resolve_local_path_string(config, model_override)
    getslice_cfg["s_jsonl_path"] = str(task_root)
    getslice_cfg["s_jsonl_file"] = config.get("task_file", "gsm8k.jsonl")
    getslice_cfg["tasks"] = list(tasks)
    getslice_cfg["save_path"] = str(root / config["getslice"].get("output_dir", "GetSliceOutput"))
    ranking = resolve_ranking_config(config)
    if ranking["metric"] == RANKING_TOPK_KL:
        getslice_cfg["mode"] = "split_whitened_svd"
        getslice_cfg["save_s_uv_path"] = None
        getslice_cfg["save_x_uv_path"] = None
        getslice_cfg.setdefault("save_x_json_path", "xMat_{task}.json")
        x_jsonl_path = config.get("getslice", {}).get("x_jsonl_path")
        if x_jsonl_path:
            getslice_cfg["x_jsonl_path"] = str(x_jsonl_path)
    else:
        getslice_cfg["mode"] = "s_only_svd"
    if s_nsamples is not None:
        getslice_cfg["s_nsamples"] = int(s_nsamples)
    out_path = root / config["getslice"].get("config_out", "getslice_config.json")
    save_json(out_path, getslice_cfg)
    print(f"[getslice_config] wrote {out_path}")
    print(f"[getslice_config] s_nsamples={getslice_cfg.get('s_nsamples')}")
    print(f"[getslice_config] ranking_metric={ranking['metric']}, mode={getslice_cfg.get('mode')}")
    return out_path


def run_getslice(config: Dict[str, Any], root: Path) -> None:
    cfg_path = root / config["getslice"].get("config_out", "getslice_config.json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing GetSlice config; run getslice_config first: {cfg_path}")
    script = Path(config["getslice"]["script"])
    cmd = [sys.executable, str(script), "--config", str(cfg_path)]
    print(f"[getslice] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(script.parent), check=True)


def resolve_local_path_string(config: Dict[str, Any], value: Any) -> str:
    raw = str(value)
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)
    repo_root = config.get("repo_root")
    if repo_root:
        repo_path = (Path(str(repo_root)).expanduser() / path).resolve()
        if raw.startswith(".") or repo_path.exists():
            return str(repo_path)
    return raw


def resolve_ranking_config(config: Dict[str, Any]) -> Dict[str, Any]:
    ranking_cfg = config.get("ranking", {}) or {}
    metric = normalize_ranking_metric(ranking_cfg.get("metric", RANKING_EFFECTIVE_RANK))
    direction = str(ranking_cfg.get("direction", "desc")).strip().lower()
    if direction not in {"asc", "desc"}:
        raise ValueError("ranking.direction must be 'asc' or 'desc'")

    topk_cfg = ranking_cfg.get("topk_kl", {}) or {}
    top_k = normalize_top_k(topk_cfg.get("top_k", 50))
    return {
        "metric": metric,
        "direction": direction,
        "top_k": top_k,
        "topk_label": topk_label(top_k),
    }


def normalize_ranking_metric(value: Any) -> str:
    metric = str(value or RANKING_EFFECTIVE_RANK).strip()
    normalized = metric.lower().replace("-", "_")
    if normalized == "effective_rank":
        return RANKING_EFFECTIVE_RANK
    if normalized in {"topk_kl", "topkkl", "top_k_kl"}:
        return RANKING_TOPK_KL
    raise ValueError(
        f"unsupported ranking.metric '{metric}'. "
        f"Choose one of: {RANKING_EFFECTIVE_RANK}, {RANKING_TOPK_KL}"
    )


def normalize_top_k(value: Any) -> int:
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"all", "full"}:
            return 0
        value = stripped
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ranking.topk_kl.top_k must be an integer, 0, or 'all'") from exc


def topk_label(top_k: int) -> str:
    top_k = int(top_k)
    if top_k == 0:
        return "all"
    if top_k > 0:
        return f"top{top_k}"
    return f"bottom{abs(top_k)}"


def effective_rank(singular_values: Sequence[float]) -> float:
    sv = np.abs(np.asarray(singular_values, dtype=np.float64))
    total = float(np.sum(sv))
    if total <= 0:
        return 0.0
    p = sv / total
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def topk_kl(source_singular_values: Sequence[float], x_singular_values: Sequence[float], top_k: int, eps: float = 1e-12) -> float:
    source = np.abs(np.asarray(source_singular_values, dtype=np.float64))
    reference = np.abs(np.asarray(x_singular_values, dtype=np.float64))
    n = min(len(source), len(reference))
    if n <= 0:
        return 0.0
    source = select_topk_values(source[:n], top_k)
    reference = select_topk_values(reference[:n], top_k)
    n = min(len(source), len(reference))
    if n <= 0:
        return 0.0
    source = source[:n]
    reference = reference[:n]
    if not np.isfinite(source).all() or not np.isfinite(reference).all():
        raise ValueError("topk_KL received NaN/Inf singular values")
    if float(np.sum(source)) <= 0 or float(np.sum(reference)) <= 0:
        return 0.0
    p = source + float(eps)
    q = reference + float(eps)
    p = p / np.sum(p)
    q = q / np.sum(q)
    value = float(np.sum(p * np.log(p / q)))
    if -1e-12 < value < 0:
        return 0.0
    return value


def select_topk_values(values: np.ndarray, top_k: int) -> np.ndarray:
    length = len(values)
    if length <= 0:
        return values
    top_k = int(top_k)
    if top_k == 0:
        return values
    if top_k > 0:
        return values[: min(top_k, length)]
    k = min(abs(top_k), length)
    return values[length - k :]


def parse_smat_unit_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("sMat_"):
        return stem[len("sMat_") :]
    return stem


def iter_smat_files(getslice_out: Path) -> List[Path]:
    return sorted(getslice_out.glob("**/sMat_*.json"))


def iter_xmat_files(getslice_out: Path) -> List[Path]:
    return sorted(getslice_out.glob("**/xMat_*.json"))


def load_x_singular_values(getslice_out: Path) -> Dict[str, Any]:
    files = iter_xmat_files(getslice_out)
    if not files:
        raise FileNotFoundError(
            f"ranking.metric=topk_KL requires xMat_*.json under {getslice_out}; "
            "run getslice_config/getslice with ranking.metric=topk_KL first"
        )
    preferred = [path for path in files if path.stem == "xMat_X"]
    path = preferred[0] if preferred else files[0]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested_singular_values(data: Dict[str, Any], layer_name: str, module_name: str) -> Optional[Sequence[float]]:
    layer = data.get(layer_name) if isinstance(data, dict) else None
    if not isinstance(layer, dict):
        return None
    values = layer.get(module_name)
    if values is None:
        return None
    return values


def stage_kl_from_smat(
    config: Dict[str, Any],
    root: Path,
    unit_id_key: str,
    unit_prefix: str,
    csv_name: str,
    modules_csv_name: str,
) -> None:
    getslice_out = root / config["getslice"].get("output_dir", "GetSliceOutput")
    files = iter_smat_files(getslice_out)
    if not files:
        raise FileNotFoundError(f"no sMat json files found under {getslice_out}; run getslice first")
    ranking = resolve_ranking_config(config)
    x_singular_values = load_x_singular_values(getslice_out) if ranking["metric"] == RANKING_TOPK_KL else None
    topk_metric_col = f"TopKKL_{ranking['topk_label']}"
    unit_topk_field = f"{unit_prefix}_topk_kl_{ranking['topk_label']}_mean"
    rows = []
    module_rows = []
    for path in files:
        unit_id = parse_smat_unit_id(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        effective_ranks = []
        topk_kls = []
        for layer_name, modules in data.items():
            if not str(layer_name).startswith("layer_") or not isinstance(modules, dict):
                continue
            for module_name, singular_values in modules.items():
                er_value = effective_rank(singular_values)
                effective_ranks.append(er_value)
                module_row = {
                    unit_id_key: unit_id,
                    "layer": layer_name,
                    "module": module_name,
                    "EffectiveRank": er_value,
                    "rank_metric": ranking["metric"],
                    "sMat_path": str(path),
                }
                if ranking["metric"] == RANKING_TOPK_KL:
                    reference_values = nested_singular_values(x_singular_values, layer_name, module_name)
                    if reference_values is None:
                        raise KeyError(
                            f"missing X singular values for {layer_name}.{module_name}; "
                            f"cannot compute ranking.metric={RANKING_TOPK_KL}"
                        )
                    kl_value = topk_kl(singular_values, reference_values, ranking["top_k"])
                    topk_kls.append(kl_value)
                    module_row[topk_metric_col] = kl_value
                module_rows.append(module_row)
        if effective_ranks:
            effective_rank_mean = float(np.mean(effective_ranks))
            effective_rank_std = float(np.std(effective_ranks))
            row = {
                unit_id_key: unit_id,
                "rank_metric": ranking["metric"],
                "rank_direction": ranking["direction"],
                f"{unit_prefix}_effective_rank_mean": effective_rank_mean,
                f"{unit_prefix}_effective_rank_std": effective_rank_std,
                f"{unit_prefix}_effective_rank_min": float(np.min(effective_ranks)),
                f"{unit_prefix}_effective_rank_max": float(np.max(effective_ranks)),
                "module_count": len(effective_ranks),
                "sMat_path": str(path),
            }
            if ranking["metric"] == RANKING_TOPK_KL:
                if not topk_kls:
                    raise ValueError(f"no topk_KL values parsed from {path}")
                rank_values = topk_kls
                row.update(
                    {
                        unit_topk_field: float(np.mean(topk_kls)),
                        f"{unit_prefix}_topk_kl_{ranking['topk_label']}_std": float(np.std(topk_kls)),
                        f"{unit_prefix}_topk_kl_{ranking['topk_label']}_min": float(np.min(topk_kls)),
                        f"{unit_prefix}_topk_kl_{ranking['topk_label']}_max": float(np.max(topk_kls)),
                    }
                )
            else:
                rank_values = effective_ranks
            row["rank_score_mean"] = float(np.mean(rank_values))
            row["rank_score_std"] = float(np.std(rank_values))
            row["rank_score"] = row["rank_score_mean"]
            rows.append(
                row
            )
    if not rows:
        raise ValueError(f"no KL ranking values parsed from {getslice_out}")
    rows.sort(key=lambda row: row["rank_score"], reverse=ranking["direction"] == "desc")
    for rank, row in enumerate(rows):
        row["kl_rank"] = rank
    with (root / csv_name).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            unit_id_key,
            "kl_rank",
            "rank_metric",
            "rank_direction",
            "rank_score",
            "rank_score_mean",
            "rank_score_std",
            f"{unit_prefix}_effective_rank_mean",
            f"{unit_prefix}_effective_rank_std",
            f"{unit_prefix}_effective_rank_min",
            f"{unit_prefix}_effective_rank_max",
            *(
                [
                    unit_topk_field,
                    f"{unit_prefix}_topk_kl_{ranking['topk_label']}_std",
                    f"{unit_prefix}_topk_kl_{ranking['topk_label']}_min",
                    f"{unit_prefix}_topk_kl_{ranking['topk_label']}_max",
                ]
                if ranking["metric"] == RANKING_TOPK_KL
                else []
            ),
            "module_count",
            "sMat_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (root / modules_csv_name).open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            unit_id_key,
            "layer",
            "module",
            "EffectiveRank",
            *([topk_metric_col] if ranking["metric"] == RANKING_TOPK_KL else []),
            "rank_metric",
            "sMat_path",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(module_rows)
    print(
        f"[kl] wrote {len(rows)} {unit_prefix} rows "
        f"(ranking_metric={ranking['metric']}, direction={ranking['direction']}) -> {root / csv_name}"
    )


def read_unit_kl(root: Path, csv_name: str, unit_id_key: str, unit_prefix: str) -> List[Dict[str, Any]]:
    path = root / csv_name
    if not path.exists():
        raise FileNotFoundError(f"missing KL csv; run kl first: {path}")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row["kl_rank"] = int(row["kl_rank"])
            for key, value in list(row.items()):
                if value in (None, ""):
                    continue
                if key in {"kl_rank", "module_count"}:
                    row[key] = int(float(value))
                elif key in {"rank_score", "rank_score_mean", "rank_score_std"}:
                    row[key] = float(value)
                elif key.startswith(f"{unit_prefix}_") and key.rsplit("_", 1)[-1] in {"mean", "std", "min", "max"}:
                    row[key] = float(value)
            if "rank_metric" not in row or not row["rank_metric"]:
                row["rank_metric"] = RANKING_TOPK_KL
            if "rank_direction" not in row or not row["rank_direction"]:
                row["rank_direction"] = "desc"
            if "rank_score" not in row or row["rank_score"] == "":
                row["rank_score"] = float(row[f"{unit_prefix}_effective_rank_mean"])
                row["rank_score_mean"] = row["rank_score"]
                row["rank_score_std"] = float(row.get(f"{unit_prefix}_effective_rank_std", 0.0))
            rows.append(row)
    rows.sort(key=lambda row: row["kl_rank"])
    return rows


def allocate_weighted_quotas(
    unit_ids: Sequence[str],
    capacities: Dict[str, int],
    weights: Dict[str, float],
    total: int,
    min_per_class: int,
) -> Dict[str, int]:
    quotas = {unit_id: 0 for unit_id in unit_ids}
    for unit_id in unit_ids:
        quotas[unit_id] = min(int(min_per_class), capacities[unit_id])
    used = sum(quotas.values())
    if used > total:
        raise ValueError("min_per_class allocation exceeds dataset_size")
    remaining = total - used
    while remaining > 0:
        eligible = [unit_id for unit_id in unit_ids if quotas[unit_id] < capacities[unit_id]]
        if not eligible:
            raise ValueError("not enough unit capacity to build requested dataset")
        weight_sum = sum(max(float(weights[unit_id]), 0.0) for unit_id in eligible)
        if weight_sum <= 0:
            local_weights = {unit_id: 1.0 for unit_id in eligible}
            weight_sum = float(len(eligible))
        else:
            local_weights = {unit_id: max(float(weights[unit_id]), 0.0) for unit_id in eligible}

        raw_add = {unit_id: remaining * local_weights[unit_id] / weight_sum for unit_id in eligible}
        add = {
            unit_id: min(int(math.floor(raw_add[unit_id])), capacities[unit_id] - quotas[unit_id])
            for unit_id in eligible
        }
        added = sum(add.values())
        if added == 0:
            order = sorted(
                eligible,
                key=lambda unit_id: (raw_add[unit_id] - math.floor(raw_add[unit_id]), local_weights[unit_id]),
                reverse=True,
            )
            for unit_id in order:
                if remaining <= 0:
                    break
                if quotas[unit_id] < capacities[unit_id]:
                    add[unit_id] += 1
                    added += 1
                    remaining -= 1
            for unit_id, value in add.items():
                quotas[unit_id] += value
            continue
        for unit_id, value in add.items():
            quotas[unit_id] += value
        remaining -= added
    return quotas


def rank_center(dataset_idx: int, num_datasets: int, num_units: int) -> float:
    if num_units <= 1 or num_datasets <= 1:
        return 0.0
    return dataset_idx * (num_units - 1) / (num_datasets - 1)


def active_class_metadata(capacities: Dict[str, int], dataset_size: int, agg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    capacity_values = [value for value in capacities.values() if value > 0]
    if not capacity_values:
        raise ValueError("no positive unit capacities")

    median_class_size = float(np.median(capacity_values))
    min_needed_classes = int(math.ceil(dataset_size / max(median_class_size, 1.0)))
    active_override = agg_cfg.get("active_classes")
    active_class_multiplier = float(agg_cfg.get("active_class_multiplier", 1.5))
    if active_override is None:
        active_classes = int(math.ceil(min_needed_classes * active_class_multiplier))
    else:
        active_classes = int(active_override)
    active_classes = max(1, min(len(capacities), active_classes))
    return {
        "median_class_size": median_class_size,
        "min_needed_classes": min_needed_classes,
        "active_class_multiplier": active_class_multiplier,
        "active_classes": active_classes,
    }


def sigma_from_active_classes(active_classes: int, center_rank: float, num_units: int, coverage_z: float) -> float:
    if num_units <= 1:
        return 1.0
    coverage_z = max(float(coverage_z), 1e-6)
    target_span = max(0.0, float(min(active_classes, num_units)) - 1.0)
    if target_span <= 0:
        return 1.0

    left_limit = float(center_rank)
    right_limit = float(num_units - 1) - float(center_rank)

    def covered_span(radius: float) -> float:
        return min(left_limit, radius) + min(right_limit, radius)

    max_span = covered_span(float(num_units - 1))
    target_span = min(target_span, max_span)
    lo = 0.0
    hi = float(num_units - 1)
    for _ in range(48):
        mid = (lo + hi) / 2.0
        if covered_span(mid) < target_span:
            lo = mid
        else:
            hi = mid
    return max(1.0, hi / coverage_z)


def gaussian_weights(
    unit_ids: Sequence[str],
    dataset_idx: int,
    num_datasets: int,
    sigma: float,
    capacities: Optional[Dict[str, int]] = None,
    capacity_alpha: float = 0.0,
) -> Dict[str, float]:
    count = len(unit_ids)
    if count == 1:
        return {unit_ids[0]: 1.0}
    mu = rank_center(dataset_idx, num_datasets, count)
    sigma = max(float(sigma), 1e-6)
    capacity_alpha = float(capacity_alpha)
    weights = {}
    for rank, unit_id in enumerate(unit_ids):
        gaussian = math.exp(-0.5 * ((rank - mu) / sigma) ** 2)
        if capacities is None or capacity_alpha == 0:
            capacity_weight = 1.0
        else:
            capacity_weight = max(float(capacities.get(unit_id, 0)), 0.0) ** capacity_alpha
        weights[unit_id] = gaussian * capacity_weight
    return weights


def compute_dataset_effective_rank_summary(quota_rows: Sequence[Dict[str, Any]], unit_effective_rank_field: str) -> Dict[str, Any]:
    used_rows = [row for row in quota_rows if int(row["used_count"]) > 0]
    if not used_rows:
        return {
            "dataset_effective_rank_weighted_mean": 0.0,
            "dataset_effective_rank_weighted_std": 0.0,
            "dataset_effective_rank_min": 0.0,
            "dataset_effective_rank_max": 0.0,
            "dataset_effective_rank_weighted_rank_mean": 0.0,
            "dataset_size": 0,
            "num_used_units": 0,
        }
    weights = np.asarray([int(row["used_count"]) for row in used_rows], dtype=np.float64)
    values = np.asarray([float(row[unit_effective_rank_field]) for row in used_rows], dtype=np.float64)
    ranks = np.asarray([float(row["kl_rank"]) for row in used_rows], dtype=np.float64)
    total = float(np.sum(weights))
    mean = float(np.sum(weights * values) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    return {
        "dataset_effective_rank_weighted_mean": mean,
        "dataset_effective_rank_weighted_std": float(np.sqrt(max(var, 0.0))),
        "dataset_effective_rank_min": float(np.min(values)),
        "dataset_effective_rank_max": float(np.max(values)),
        "dataset_effective_rank_weighted_rank_mean": float(np.sum(weights * ranks) / total),
        "dataset_size": int(total),
        "num_used_units": len(used_rows),
    }


def compute_dataset_score_summary(quota_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    used_rows = [row for row in quota_rows if int(row["used_count"]) > 0]
    if not used_rows:
        return {
            "dataset_rank_metric": "",
            "dataset_score_weighted_mean": 0.0,
            "dataset_score_weighted_std": 0.0,
            "dataset_score_min": 0.0,
            "dataset_score_max": 0.0,
            "dataset_score_weighted_rank_mean": 0.0,
        }
    weights = np.asarray([int(row["used_count"]) for row in used_rows], dtype=np.float64)
    values = np.asarray([float(row["rank_score"]) for row in used_rows], dtype=np.float64)
    ranks = np.asarray([float(row["kl_rank"]) for row in used_rows], dtype=np.float64)
    total = float(np.sum(weights))
    mean = float(np.sum(weights * values) / total)
    var = float(np.sum(weights * (values - mean) ** 2) / total)
    metrics = sorted({str(row.get("rank_metric", "")) for row in used_rows if row.get("rank_metric")})
    return {
        "dataset_rank_metric": metrics[0] if len(metrics) == 1 else ",".join(metrics),
        "dataset_score_weighted_mean": mean,
        "dataset_score_weighted_std": float(np.sqrt(max(var, 0.0))),
        "dataset_score_min": float(np.min(values)),
        "dataset_score_max": float(np.max(values)),
        "dataset_score_weighted_rank_mean": float(np.sum(weights * ranks) / total),
    }


def aggregate_ranked_datasets(
    config: Dict[str, Any],
    root: Path,
    rank_rows: Sequence[Dict[str, Any]],
    records_by_unit: Dict[str, List[Record]],
    unit_id_key: str,
    unit_prefix: str,
    ranked_dir_name: str,
    quota_csv_name: str,
    unit_size_field: str,
) -> None:
    agg_cfg = config.get("aggregation", {})
    ranking = resolve_ranking_config(config)
    row_metrics = {str(row.get("rank_metric") or RANKING_TOPK_KL) for row in rank_rows}
    row_directions = {str(row.get("rank_direction") or "desc") for row in rank_rows}
    if row_metrics and row_metrics != {ranking["metric"]}:
        raise ValueError(
            f"KL csv ranking metric {sorted(row_metrics)} does not match config ranking.metric={ranking['metric']}; "
            "rerun the kl stage with the current config"
        )
    if row_directions and row_directions != {ranking["direction"]}:
        raise ValueError(
            f"KL csv ranking direction {sorted(row_directions)} does not match config ranking.direction={ranking['direction']}; "
            "rerun the kl stage with the current config"
        )
    unit_effective_rank_field = f"{unit_prefix}_effective_rank_mean"
    unit_ids = [row[unit_id_key] for row in rank_rows]
    capacities = {unit_id: len(records_by_unit[unit_id]) for unit_id in unit_ids}
    num_datasets = int(agg_cfg.get("num_datasets", 1))
    dataset_size = int(agg_cfg.get("dataset_size", 3000))
    min_per_class = int(agg_cfg.get("min_per_class", 1))
    active_meta = active_class_metadata(capacities, dataset_size, agg_cfg)
    capacity_alpha = float(agg_cfg.get("capacity_alpha", 1.0))
    coverage_z = float(agg_cfg.get("sigma_coverage_z", 3.0))
    fixed_sigma = agg_cfg.get("rank_sigma")
    if sum(capacities.values()) < dataset_size:
        raise ValueError("total records are fewer than aggregation.dataset_size")
    if min_per_class > 0 and len(unit_ids) * min_per_class > dataset_size:
        raise ValueError("too many units for min_per_class and dataset_size")

    ranked_root = root / ranked_dir_name
    ensure_dir_for_stage(ranked_root, overwrite=bool(config.get("overwrite_outputs", False)))
    dataset_alias = config.get("dataset_alias", "gsm8k_math_train")
    task_file = config.get("task_file", "gsm8k.jsonl")
    train_file = config.get("train_file", "gsm8k-train.jsonl")
    rank_by_unit = {row[unit_id_key]: row for row in rank_rows}
    summary_rows = []

    for dataset_idx in range(num_datasets):
        dataset_id = f"dataset_{dataset_idx + 1:03d}"
        center_rank = rank_center(dataset_idx, num_datasets, len(unit_ids))
        if fixed_sigma is None:
            sigma = sigma_from_active_classes(
                active_classes=int(active_meta["active_classes"]),
                center_rank=center_rank,
                num_units=len(unit_ids),
                coverage_z=coverage_z,
            )
            sigma_mode = "auto_active_classes"
        else:
            sigma = float(fixed_sigma)
            sigma_mode = "fixed_rank_sigma"
        gaussian_only_weights = gaussian_weights(unit_ids, dataset_idx, num_datasets, float(sigma))
        weights = gaussian_weights(
            unit_ids,
            dataset_idx,
            num_datasets,
            float(sigma),
            capacities=capacities,
            capacity_alpha=capacity_alpha,
        )
        quotas = allocate_weighted_quotas(unit_ids, capacities, weights, dataset_size, min_per_class)
        selected: List[Record] = []
        extra_fields: Dict[int, Dict[str, Any]] = {}
        quota_rows = []
        for unit_id in unit_ids:
            rows = records_by_unit[unit_id]
            quota = quotas[unit_id]
            chosen = rows[:quota]
            selected.extend(chosen)
            rank_row = rank_by_unit[unit_id]
            unit_effective_rank_mean = float(rank_row[unit_effective_rank_field])
            rank_score = float(rank_row.get("rank_score", unit_effective_rank_mean))
            rank_metric = str(rank_row.get("rank_metric") or ranking["metric"])
            rank_score_mean = float(rank_row.get("rank_score_mean", rank_score))
            rank_score_std = float(rank_row.get("rank_score_std", 0.0))
            source_counts = Counter(r.source for r in chosen)
            quota_rows.append(
                {
                    "dataset_id": dataset_id,
                    unit_id_key: unit_id,
                    "kl_rank": int(rank_row["kl_rank"]),
                    "rank_metric": rank_metric,
                    "rank_score": rank_score,
                    "rank_score_mean": rank_score_mean,
                    "rank_score_std": rank_score_std,
                    unit_effective_rank_field: unit_effective_rank_mean,
                    unit_size_field: len(rows),
                    "gaussian_weight": gaussian_only_weights[unit_id],
                    "capacity_weight": max(float(capacities[unit_id]), 0.0) ** capacity_alpha,
                    "final_weight": weights[unit_id],
                    "quota": quota,
                    "used_count": len(chosen),
                    "score_contribution": len(chosen) * rank_score,
                    "effective_rank_contribution": len(chosen) * unit_effective_rank_mean,
                    "source_counts": json.dumps(dict(sorted(source_counts.items())), ensure_ascii=False),
                }
            )
            for record in chosen:
                extra_fields[record.global_id] = {
                    "offpolicy_dataset_id": dataset_id,
                    unit_id_key: unit_id,
                    f"{unit_prefix}_kl_rank": int(rank_row["kl_rank"]),
                    unit_effective_rank_field: unit_effective_rank_mean,
                    "rank_metric": rank_metric,
                    "rank_score": rank_score,
                }
        if len(selected) != dataset_size:
            raise RuntimeError(f"{dataset_id} selected {len(selected)} records, expected {dataset_size}")

        effective_rank_summary = compute_dataset_effective_rank_summary(quota_rows, unit_effective_rank_field)
        dataset_dir = ranked_root / dataset_id
        write_dataset_dir(dataset_dir, selected, dataset_alias, task_file, train_file, extra_fields)
        source_stats = Counter(r.source for r in selected)
        with (dataset_dir / "source_stats.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_id", "source", "count", "ratio"])
            writer.writeheader()
            for source, count in sorted(source_stats.items()):
                writer.writerow({"dataset_id": dataset_id, "source": source, "count": count, "ratio": count / dataset_size})
        with (dataset_dir / quota_csv_name).open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "dataset_id",
                unit_id_key,
                "kl_rank",
                "rank_metric",
                "rank_score",
                "rank_score_mean",
                "rank_score_std",
                unit_effective_rank_field,
                unit_size_field,
                "gaussian_weight",
                "capacity_weight",
                "final_weight",
                "quota",
                "used_count",
                "score_contribution",
                "effective_rank_contribution",
                "source_counts",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(quota_rows)
        save_json(
            dataset_dir / "ranking_meta.json",
            {
                "dataset_id": dataset_id,
                "dataset_size": dataset_size,
                "num_datasets": num_datasets,
                "rank_sigma": float(sigma),
                "rank_sigma_mode": sigma_mode,
                "sigma_coverage_z": coverage_z,
                "min_per_class": min_per_class,
                "capacity_alpha": capacity_alpha,
                "rank_metric": ranking["metric"],
                "rank_direction": ranking["direction"],
                "ranking_top_k": ranking["top_k"] if ranking["metric"] == RANKING_TOPK_KL else None,
                **active_meta,
                "center_rank": center_rank,
                "unit_order": unit_ids,
                f"{unit_prefix}_order": unit_ids,
                **compute_dataset_score_summary(quota_rows),
                **effective_rank_summary,
            },
        )
        score_summary = compute_dataset_score_summary(quota_rows)
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "rank_metric": ranking["metric"],
                "rank_direction": ranking["direction"],
                "ranking_top_k": ranking["top_k"] if ranking["metric"] == RANKING_TOPK_KL else "",
                "rank_sigma": float(sigma),
                "rank_sigma_mode": sigma_mode,
                "center_rank": center_rank,
                **score_summary,
                **effective_rank_summary,
            }
        )
        print(
            f"[aggregate] wrote {dataset_id}: {len(selected)} records, "
            f"dataset_score_weighted_mean={score_summary['dataset_score_weighted_mean']:.6f}, "
            f"dataset_effective_rank_weighted_mean={effective_rank_summary['dataset_effective_rank_weighted_mean']:.6f} -> {dataset_dir}"
        )

    with (root / "ranked_dataset_kl.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset_id",
            "rank_metric",
            "rank_direction",
            "ranking_top_k",
            "dataset_rank_metric",
            "dataset_score_weighted_mean",
            "dataset_score_weighted_std",
            "dataset_score_min",
            "dataset_score_max",
            "dataset_score_weighted_rank_mean",
            "dataset_effective_rank_weighted_mean",
            "dataset_effective_rank_weighted_std",
            "dataset_effective_rank_min",
            "dataset_effective_rank_max",
            "dataset_effective_rank_weighted_rank_mean",
            "dataset_size",
            "num_used_units",
            "rank_sigma",
            "rank_sigma_mode",
            "center_rank",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[aggregate] wrote dataset KL summary -> {root / 'ranked_dataset_kl.csv'}")
