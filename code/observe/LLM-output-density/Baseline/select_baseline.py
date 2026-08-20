#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build source-stratified NuminaMath baseline training subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_DATASET_NAME = "NuminaMath-1___5"
DEFAULT_INPUT_PATH = Path("/root/autodl-tmp/prepared/NuminaMath-1___5/train.parquet")
DEFAULT_OUTPUT_ROOT = Path("/root/autodl-tmp/baseline/NuminaMath-1___5")
DEFAULT_SIZES = [200, 300, 400, 600, 800, 1100, 1400, 1800]
DEFAULT_METHODS = ["random", "ppl_cond_middle", "diversity", "cfs"]
PPL_METHODS = {"ppl_cond_middle"}
DIVERSITY_METHODS = {"diversity"}
DEFAULT_DATASET_ALIAS = "sft_train"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        if value is None or value != value:
            return None
    except Exception:
        pass
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def load_config(path: str) -> Dict[str, Any]:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    with resolved.open("r", encoding="utf-8") as f:
        if resolved.suffix.lower() == ".json":
            loaded = json.load(f)
        else:
            loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint_file(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256_file(path),
    }


def stable_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_float_key(*parts: Any) -> float:
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF)


def resolve_path(value: Any, *, base: Path = REPO_ROOT) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return base / path


def normalize_methods(raw_methods: Any) -> List[str]:
    methods = raw_methods if isinstance(raw_methods, list) else DEFAULT_METHODS
    normalized: List[str] = []
    for method in methods:
        value = str(method).strip()
        if not value:
            continue
        if value == "ppl_cond":
            value = "ppl_cond_middle"
        if value == "ifd":
            raise ValueError("IFD baseline 已从默认实验矩阵移除，请使用 ppl_cond_middle/random/diversity/cfs")
        if value == "random_cfs":
            value = "cfs"
        if value not in {"random", "ppl_cond_middle", "diversity", "cfs"}:
            raise ValueError(f"不支持的 baseline method: {value}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("baseline.methods 不能为空")
    return normalized


def normalize_sizes(raw_sizes: Any) -> List[int]:
    sizes = raw_sizes if isinstance(raw_sizes, list) else DEFAULT_SIZES
    normalized: List[int] = []
    for size in sizes:
        value = int(size)
        if value <= 0:
            raise ValueError(f"baseline size 必须 > 0: {size}")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("baseline.sizes 不能为空")
    return normalized


def percent_label(value: float) -> str:
    return str(int(round(float(value) * 100)))


def method_dataset_name(
    dataset_name: str,
    method: str,
    cfs_ratio: Optional[float] = None,
    middle_fraction: Optional[float] = None,
    source_prefix: Optional[str] = None,
) -> str:
    prefix = str(source_prefix or dataset_name)
    suffix = method
    if method == "cfs":
        suffix = f"cfs{percent_label(float(cfs_ratio or 0.0))}"
    elif method == "ppl_cond_middle":
        suffix = f"ppl_mid{percent_label(float(middle_fraction if middle_fraction is not None else 0.70))}"
    return f"{prefix}-{suffix}"


def task_id_for(
    dataset_name: str,
    method: str,
    size: int,
    cfs_ratio: Optional[float] = None,
    middle_fraction: Optional[float] = None,
    source_prefix: Optional[str] = None,
) -> str:
    return f"{method_dataset_name(dataset_name, method, cfs_ratio, middle_fraction, source_prefix)}__{size}"


def load_train_frame(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"baseline input_path 不存在: {input_path}")
    if input_path.suffix.lower() != ".parquet":
        raise ValueError(f"baseline 当前只支持 NuminaMath parquet 输入: {input_path}")

    frame = pd.read_parquet(input_path)
    if "row_id" not in frame.columns:
        frame = frame.copy()
        frame.insert(0, "row_id", [str(i) for i in range(len(frame))])

    required = {"row_id", "problem", "solution", "source"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"baseline 输入缺少必要字段: {missing}")

    mask = (
        frame["problem"].notna()
        & frame["solution"].notna()
        & frame["problem"].astype(str).str.strip().ne("")
        & frame["solution"].astype(str).str.strip().ne("")
    )
    usable = frame.loc[mask].copy()
    usable["row_id"] = usable["row_id"].astype(str)
    usable["source"] = usable["source"].fillna("unknown").astype(str)
    if usable.empty:
        raise ValueError(f"baseline 输入没有可用 problem/solution 样本: {input_path}")
    return usable


def distribution(frame: pd.DataFrame, column: str = "source") -> Dict[str, int]:
    if frame.empty:
        return {}
    counts = frame[column].fillna("unknown").astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def allocate_stratified_counts(counts: pd.Series, total: int) -> Dict[str, int]:
    counts = counts[counts > 0].astype(int)
    available = int(counts.sum())
    if total > available:
        raise ValueError(f"请求样本数超过可用样本数: requested={total}, available={available}")
    if total <= 0:
        return {}

    raw = counts.astype(float) / float(available) * int(total)
    allocation = raw.apply(math.floor).astype(int).to_dict()
    remainders = (raw - raw.apply(math.floor)).to_dict()
    assigned = int(sum(allocation.values()))

    ordered_keys = sorted(
        allocation,
        key=lambda key: (remainders[key], int(counts[key]), str(key)),
        reverse=True,
    )
    idx = 0
    while assigned < total:
        key = ordered_keys[idx % len(ordered_keys)]
        if allocation[key] < int(counts[key]):
            allocation[key] += 1
            assigned += 1
        idx += 1

    return {str(key): int(value) for key, value in allocation.items() if int(value) > 0}


def select_from_sorted_frame(frame: pd.DataFrame, allocation: Dict[str, int], sort_columns: List[str]) -> pd.DataFrame:
    selected_frames: List[pd.DataFrame] = []
    for source, count in sorted(allocation.items()):
        if count <= 0:
            continue
        group = frame.loc[frame["source"].astype(str).eq(str(source))].sort_values(sort_columns, kind="mergesort")
        if len(group) < count:
            raise ValueError(f"source={source} 可用样本不足: requested={count}, available={len(group)}")
        selected_frames.append(group.head(count))
    if not selected_frames:
        raise ValueError("没有选择出任何样本")
    return pd.concat(selected_frames, ignore_index=True)


def add_random_key(frame: pd.DataFrame, seed: int, key_name: str = "_baseline_random_key") -> pd.DataFrame:
    result = frame.copy()
    result[key_name] = [stable_float_key(seed, row_id) for row_id in result["row_id"].astype(str)]
    return result


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                loaded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} 不是合法 JSONL: {exc}") from exc
            if isinstance(loaded, dict):
                records.append(loaded)
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def _coerce_nonempty_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def _normalize_existing_messages(value: Any) -> Optional[List[Dict[str, str]]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list):
        return None
    messages: List[Dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).strip()
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            return None
        text = _coerce_nonempty_text(content)
        if not text:
            return None
        messages.append({"role": role, "content": text})
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    return messages


def cfs_record_to_sharegpt(
    record: Dict[str, Any],
    *,
    index: int,
    source_path: Path,
    user_prompt: str,
) -> Optional[Dict[str, Any]]:
    messages = _normalize_existing_messages(record.get("messages") or record.get("message"))
    if messages is None:
        problem = _coerce_nonempty_text(record.get("problem") or record.get("question"))
        solution = _coerce_nonempty_text(record.get("solution") or record.get("answer"))
        if problem and solution:
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ]
        else:
            text = _coerce_nonempty_text(record.get("text") or record.get("raw") or record.get("output"))
            if not text:
                return None
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": text},
            ]

    metadata = {
        "source": "cfs",
        "cfs_id": str(record.get("id", index)),
        "cfs_source_path": str(source_path),
        "baseline_source": "cfs",
        "baseline_method": "cfs",
    }
    for key in ("model", "finish_reason", "created_at", "task", "source_id", "source_index"):
        if key in record:
            metadata[f"cfs_{key}"] = _json_safe(record[key])
    payload = {"messages": messages}
    payload.update(metadata)
    return payload


def _strip_json_line_comments(text: str) -> str:
    output_lines: List[str] = []
    for line in text.splitlines():
        chars: List[str] = []
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


def load_json_with_comments(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.loads(_strip_json_line_comments(f.read()))


def ensure_cfs_records_available(
    *,
    cfs_cfg: Dict[str, Any],
    target_count: int,
    output_dir: Path,
) -> Path:
    cfs_path = resolve_path(
        cfs_cfg.get("path", "/root/LLM-output-density/MyFunc/GetData/GetX/output/x_dataset.jsonl")
    )
    if cfs_path.exists() and len(read_jsonl(cfs_path)) >= target_count:
        return cfs_path

    if not bool(cfs_cfg.get("generate_if_missing", False)):
        available = len(read_jsonl(cfs_path)) if cfs_path.exists() else 0
        raise FileNotFoundError(
            f"CFS 数据不足: path={cfs_path}, required={target_count}, available={available}. "
            "可先运行 GetX，或设置 baseline.cfs.generate_if_missing=true。"
        )

    getx_config_path = resolve_path(
        cfs_cfg.get("getx_config_path", "MyFunc/GetData/GetX/config.json")
    )
    getx_script = resolve_path(cfs_cfg.get("getx_script", "MyFunc/GetData/GetX/get_x.py"))
    getx_dir = getx_script.parent
    if not getx_config_path.exists():
        raise FileNotFoundError(f"GetX config 不存在: {getx_config_path}")
    if not getx_script.exists():
        raise FileNotFoundError(f"GetX script 不存在: {getx_script}")

    generated_config = load_json_with_comments(getx_config_path)
    generated_output = output_dir / "generated_cfs.jsonl"
    generated_meta = output_dir / "generated_cfs_meta.json"
    generated_config["num_samples"] = target_count
    generated_config["output_jsonl_path"] = str(generated_output)
    generated_config["metadata_path"] = str(generated_meta)
    generated_config["resume"] = True
    generated_config.setdefault("overwrite", False)
    if cfs_cfg.get("getx_overrides"):
        generated_config.update(dict(cfs_cfg["getx_overrides"]))

    generated_config_path = output_dir / "getx_config.generated.json"
    save_json(generated_config_path, generated_config)
    cmd = [sys.executable, str(getx_script), "--config", str(generated_config_path)]
    print(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(getx_dir))
    if result.returncode != 0:
        raise RuntimeError(f"GetX 生成 CFS 数据失败，退出码: {result.returncode}")
    return generated_output


def select_cfs_jsonl(
    *,
    cfs_cfg: Dict[str, Any],
    target_count: int,
    seed: int,
    output_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    if target_count <= 0:
        write_jsonl(output_path, [])
        return {
            "path": str(output_path),
            "source_path": None,
            "requested_count": 0,
            "written_count": 0,
            "skipped_unusable": 0,
        }

    source_path = ensure_cfs_records_available(
        cfs_cfg=cfs_cfg,
        target_count=target_count,
        output_dir=output_dir,
    )
    source_records = read_jsonl(source_path)
    user_prompt = str(cfs_cfg.get("user_prompt", "Continue from the beginning of sequence."))
    converted: List[Tuple[float, Dict[str, Any]]] = []
    skipped = 0
    for idx, record in enumerate(source_records):
        normalized = cfs_record_to_sharegpt(
            record,
            index=idx,
            source_path=source_path,
            user_prompt=user_prompt,
        )
        if normalized is None:
            skipped += 1
            continue
        stable_id = record.get("id", idx)
        text_hint = (
            record.get("text")
            or record.get("raw")
            or record.get("question")
            or record.get("problem")
            or idx
        )
        key = stable_float_key(seed, stable_id, text_hint)
        converted.append((key, normalized))

    if len(converted) < target_count:
        raise ValueError(
            f"CFS 可用样本不足: requested={target_count}, usable={len(converted)}, "
            f"skipped_unusable={skipped}, source={source_path}"
        )

    selected = [record for _, record in sorted(converted, key=lambda item: item[0])[:target_count]]
    written = write_jsonl(output_path, selected)
    return {
        "path": str(output_path),
        "source_path": str(source_path),
        "requested_count": int(target_count),
        "written_count": int(written),
        "skipped_unusable": int(skipped),
        "source_fingerprint": _fingerprint_file(source_path),
    }


def _chat_prompt_text(tokenizer: Any, problem: str) -> str:
    messages = [{"role": "user", "content": problem}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"User: {problem}\nAssistant: "


def _tokenize_text(tokenizer: Any, text: str) -> List[int]:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return [int(item) for item in ids]


def _maybe_append_eos(ids: List[int], eos_token_id: Optional[int], add_eos: bool) -> List[int]:
    if not add_eos or eos_token_id is None:
        return ids
    if ids and ids[-1] == int(eos_token_id):
        return ids
    return ids + [int(eos_token_id)]


def _build_cond_sample(
    tokenizer: Any,
    problem: str,
    solution: str,
    *,
    max_length: int,
    add_eos: bool,
) -> Tuple[List[int], List[int], int]:
    prompt_ids = _tokenize_text(tokenizer, _chat_prompt_text(tokenizer, problem))
    solution_ids = _maybe_append_eos(
        _tokenize_text(tokenizer, solution),
        getattr(tokenizer, "eos_token_id", None),
        add_eos,
    )
    if not solution_ids:
        solution_ids = [int(getattr(tokenizer, "eos_token_id", 0) or 0)]

    if len(solution_ids) >= max_length:
        input_ids = solution_ids[:max_length]
        labels = input_ids[:]
    else:
        keep_prompt = max(0, max_length - len(solution_ids))
        prompt_ids = prompt_ids[-keep_prompt:] if keep_prompt else []
        input_ids = prompt_ids + solution_ids
        labels = [-100] * len(prompt_ids) + solution_ids
    return input_ids, labels, len(solution_ids)


def _build_alone_sample(
    tokenizer: Any,
    solution: str,
    *,
    max_length: int,
    add_eos: bool,
) -> Tuple[List[int], List[int], int]:
    solution_ids = _maybe_append_eos(
        _tokenize_text(tokenizer, solution),
        getattr(tokenizer, "eos_token_id", None),
        add_eos,
    )
    if not solution_ids:
        solution_ids = [int(getattr(tokenizer, "eos_token_id", 0) or 0)]
    input_ids = solution_ids[:max_length]
    labels = input_ids[:]
    return input_ids, labels, len(solution_ids)


def _score_batch(
    model: Any,
    tokenizer: Any,
    samples: List[Tuple[List[int], List[int], int]],
    *,
    device: str,
) -> List[Tuple[float, int]]:
    import torch
    import torch.nn.functional as F

    if not samples:
        return []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    max_len = max(len(input_ids) for input_ids, _, _ in samples)
    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    mask_rows: List[List[int]] = []
    for input_ids, labels, _ in samples:
        pad = max_len - len(input_ids)
        input_rows.append(input_ids + [int(pad_id)] * pad)
        label_rows.append(labels + [-100] * pad)
        mask_rows.append([1] * len(input_ids) + [0] * pad)

    input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
    label_tensor = torch.tensor(label_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = label_tensor[:, 1:].contiguous()
        valid_mask = shift_labels.ne(-100)
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            safe_labels.view(-1),
            reduction="none",
        ).view_as(safe_labels)
        denom = valid_mask.sum(dim=1).clamp_min(1)
        sample_losses = (losses * valid_mask).sum(dim=1) / denom

    return [(float(loss.item()), int(token_count)) for loss, (_, _, token_count) in zip(sample_losses, samples)]


def score_cache_paths(
    output_root: Path,
    model_name_or_path: str,
    input_path: Path,
    *,
    max_length: int,
    score_kind: str = "ppl_cond",
) -> Tuple[Path, Path]:
    model_label = Path(str(model_name_or_path)).name or "model"
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_label)
    input_hash = _fingerprint_file(input_path)["sha256"][:12]
    safe_kind = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(score_kind))
    root = output_root / "_ppl_cache" / f"{safe_model}_{input_hash}_{safe_kind}_len{int(max_length)}"
    return root / "scores.jsonl", root / "scores_meta.json"


def load_score_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    scores: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return scores
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = str(record.get("row_id", "")).strip()
            if row_id:
                scores[row_id] = record
    return scores


def _resolve_torch_dtype(raw_dtype: Any, device: str) -> Any:
    import torch

    dtype = str(raw_dtype or "auto").lower()
    if dtype == "auto":
        return torch.float16 if str(device).startswith("cuda") else torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"不支持的 baseline.ppl.dtype: {raw_dtype}")


def resolve_train_args_for_size(config: Dict[str, Any], size: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from Train.components.train_runner import DEFAULT_TRAIN_CONFIGS, load_train_args

    train_cfg = dict(config.get("train") or {})
    method = str(train_cfg.get("method", "lora")).lower()
    if method not in {"lora", "full"}:
        method = "lora"
    config_path = (
        train_cfg.get("config_path")
        or train_cfg.get("train_config")
        or train_cfg.get("template_path")
    )
    if not config_path:
        config_paths = train_cfg.get("config_paths") or train_cfg.get("method_configs") or {}
        if isinstance(config_paths, dict):
            config_path = config_paths.get(method)
    if not config_path:
        config_path = DEFAULT_TRAIN_CONFIGS[method]
    resolved_config_path = resolve_path(config_path)
    return load_train_args(
        resolved_config_path,
        method=method,
        max_samples=size,
    )


def resolve_train_cutoff_len(config: Dict[str, Any], size: int) -> int:
    train_args, _ = resolve_train_args_for_size(config, size)
    cutoff_len = int(train_args.get("cutoff_len", 1024))
    if cutoff_len <= 0:
        raise ValueError(f"训练 cutoff_len 必须 > 0，当前为: {cutoff_len}")
    return cutoff_len


def resolve_baseline_cutoff_len(config: Dict[str, Any], baseline_cfg: Dict[str, Any], size: int) -> int:
    for raw in (
        baseline_cfg.get("cutoff_len"),
        (baseline_cfg.get("ppl") or {}).get("cutoff_len"),
        (baseline_cfg.get("diversity") or {}).get("cutoff_len"),
    ):
        if raw not in (None, ""):
            cutoff_len = int(raw)
            if cutoff_len <= 0:
                raise ValueError(f"baseline cutoff_len 必须 > 0，当前为: {cutoff_len}")
            return cutoff_len
    return resolve_train_cutoff_len(config, size)


def compute_ppl_scores(
    frame: pd.DataFrame,
    *,
    config: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    input_path: Path,
    output_root: Path,
    max_length: int,
) -> Tuple[Dict[str, Dict[str, Any]], Path]:
    ppl_cfg = dict(baseline_cfg.get("ppl") or {})
    model_name_or_path = str(ppl_cfg.get("model_name_or_path") or config.get("base_model"))
    if not model_name_or_path:
        raise ValueError("PPL baseline 需要 baseline.ppl.model_name_or_path 或 base_model")
    model_path = resolve_path(model_name_or_path)
    model_ref = str(model_path if model_path.exists() else model_name_or_path)
    score_path = (
        resolve_path(ppl_cfg["score_path"])
        if ppl_cfg.get("score_path")
        else score_cache_paths(output_root, model_ref, input_path, max_length=max_length)[0]
    )
    meta_path = score_path.with_name("scores_meta.json")

    existing_scores = load_score_cache(score_path)
    row_ids = frame["row_id"].astype(str).tolist()
    missing_rows = frame.loc[~frame["row_id"].astype(str).isin(existing_scores.keys())].copy()
    if missing_rows.empty:
        return existing_scores, score_path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = str(ppl_cfg.get("device") or config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[PPL] CUDA 不可用，自动回退到 CPU")
        device = "cpu"
    dtype = _resolve_torch_dtype(ppl_cfg.get("dtype", "auto"), device)
    batch_size = max(1, int(ppl_cfg.get("batch_size", 1)))
    max_length = max(8, int(max_length))
    add_eos = bool(ppl_cfg.get("add_eos", True))
    log_every = max(1, int(ppl_cfg.get("log_every", 100)))

    print(f"[PPL] model={model_ref}")
    print(f"[PPL] cache={score_path}")
    print(f"[PPL] existing={len(existing_scores)} missing={len(missing_rows)} cutoff_len={max_length}")

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    score_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with score_path.open("a", encoding="utf-8") as out:
        records = missing_rows.to_dict(orient="records")
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            cond_samples = [
                _build_cond_sample(
                    tokenizer,
                    str(record.get("problem", "")),
                    str(record.get("solution", "")),
                    max_length=max_length,
                    add_eos=add_eos,
                )
                for record in batch_records
            ]
            cond_scores = _score_batch(model, tokenizer, cond_samples, device=device)

            for record, (loss_cond, cond_tokens) in zip(
                batch_records,
                cond_scores,
            ):
                ppl_cond = math.exp(min(loss_cond, 50.0))
                payload = {
                    "row_id": str(record["row_id"]),
                    "source": str(record.get("source", "unknown")),
                    "loss_cond": loss_cond,
                    "ppl_cond": ppl_cond,
                    "cond_token_count": cond_tokens,
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                existing_scores[payload["row_id"]] = payload
                processed += 1

            if processed % log_every == 0 or processed >= len(missing_rows):
                print(f"[PPL] scored {processed}/{len(missing_rows)} newly missing records")
                out.flush()

    save_json(
        meta_path,
        {
            "format_version": 1,
            "score_kind": "ppl_cond",
            "model_name_or_path": model_ref,
            "input_path": str(input_path),
            "input_fingerprint": _fingerprint_file(input_path),
            "record_count": len(row_ids),
            "scored_count": len(existing_scores),
            "cutoff_len": max_length,
            "batch_size": batch_size,
            "add_eos": add_eos,
            "updated_at": utc_now(),
        },
    )
    return existing_scores, score_path


def merge_scores(frame: pd.DataFrame, scores: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    score_rows = []
    for row_id, score in scores.items():
        score_rows.append(
            {
                "row_id": str(row_id),
                "loss_cond": score.get("loss_cond"),
                "ppl_cond": score.get("ppl_cond"),
            }
        )
    score_frame = pd.DataFrame(score_rows)
    merged = frame.merge(score_frame, on="row_id", how="left")
    missing = int(merged["ppl_cond"].isna().sum()) if "ppl_cond" in merged.columns else len(merged)
    if missing:
        raise ValueError(f"PPL cache 缺少 {missing} 条 row_id 分数，无法完成筛选")
    return merged


def _request_matches(meta_path: Path, request: Dict[str, Any]) -> bool:
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        return existing.get("request") == request
    except Exception:
        return False


def _write_selected_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = frame.drop(columns=[col for col in frame.columns if col.startswith("_baseline_")], errors="ignore")
    clean.to_parquet(path, index=False)


def _source_ratio_delta(original: Dict[str, int], selected: Dict[str, int]) -> Dict[str, float]:
    original_total = sum(original.values()) or 1
    selected_total = sum(selected.values()) or 1
    keys = sorted(set(original) | set(selected))
    return {
        key: (selected.get(key, 0) / selected_total) - (original.get(key, 0) / original_total)
        for key in keys
    }


def _train_epoch_override_scaled(
    config: Dict[str, Any],
    *,
    size: int,
    numerator_count: int,
    denominator_count: int,
    scale_rule: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if denominator_count <= 0 or numerator_count <= 0:
        return {}, {}
    try:
        train_args, train_source = resolve_train_args_for_size(config, size)
        base_epochs = train_args.get("num_train_epochs")
        if base_epochs is None:
            return {}, {"reason": "num_train_epochs not found", "train_config_source": train_source}
        scaled_epochs = max(1, int(round(float(base_epochs) * float(numerator_count) / float(denominator_count))))
        return (
            {"num_train_epochs": scaled_epochs},
            {
                "base_num_train_epochs": base_epochs,
                "scaled_num_train_epochs": scaled_epochs,
                "numerator_count": int(numerator_count),
                "denominator_count": int(denominator_count),
                "scale_rule": scale_rule,
                "train_config_source": train_source,
            },
        )
    except Exception as exc:
        return (
            {},
            {
                "error": repr(exc),
                "numerator_count": int(numerator_count),
                "denominator_count": int(denominator_count),
            },
        )


def middle_fraction_count(total: int, fraction: float) -> int:
    if not (0 < float(fraction) <= 1):
        raise ValueError(f"middle_fraction 必须在 (0, 1] 内，当前为: {fraction}")
    return max(1, min(int(total), int(round(float(total) * float(fraction)))))


def middle_window_bounds(total: int, fraction: float) -> Tuple[int, int]:
    count = middle_fraction_count(total, fraction)
    start = max(0, (int(total) - count) // 2)
    return start, start + count


def select_ppl_middle_frame(
    frame: pd.DataFrame,
    allocation: Dict[str, int],
    *,
    seed: int,
    middle_fraction: float,
    score_column: str = "ppl_cond",
) -> pd.DataFrame:
    selected_frames: List[pd.DataFrame] = []
    for source, count in sorted(allocation.items()):
        if count <= 0:
            continue
        group = frame.loc[frame["source"].astype(str).eq(str(source))].sort_values(
            [score_column, "row_id"],
            kind="mergesort",
        )
        if len(group) < count:
            raise ValueError(f"source={source} 可用样本不足: requested={count}, available={len(group)}")
        start, end = middle_window_bounds(len(group), middle_fraction)
        candidates = group.iloc[start:end].copy()
        if len(candidates) < count:
            candidates = group.copy()
        candidates["_baseline_middle_random_key"] = [
            stable_float_key(seed, "ppl_cond_middle", row_id)
            for row_id in candidates["row_id"].astype(str)
        ]
        selected_frames.append(
            candidates.sort_values(["_baseline_middle_random_key", "row_id"], kind="mergesort").head(count)
        )
    if not selected_frames:
        raise ValueError("没有选择出任何 PPL middle 样本")
    return pd.concat(selected_frames, ignore_index=True)


def diversity_cache_paths(
    output_root: Path,
    model_name_or_path: str,
    input_path: Path,
    *,
    cutoff_len: int,
    layer: Any,
) -> Tuple[Path, Path]:
    model_label = Path(str(model_name_or_path)).name or "model"
    safe_model = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_label)
    input_hash = _fingerprint_file(input_path)["sha256"][:12]
    layer_label = str(layer).replace("/", "_").replace(" ", "_")
    root = output_root / "_diversity_cache" / f"{safe_model}_{input_hash}_layer{layer_label}_len{int(cutoff_len)}"
    return root / "embeddings.npy", root / "embeddings_meta.json"


def _resolve_embedding_layer(value: Any) -> Any:
    if value in (None, ""):
        return 14
    if str(value).strip().lower() == "last":
        return "last"
    return int(value)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _sanitize_embedding_matrix(matrix: np.ndarray, *, context: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    bad_mask = ~np.isfinite(matrix)
    if bad_mask.any():
        bad_rows = int(np.any(bad_mask, axis=1).sum()) if matrix.ndim == 2 else 0
        print(
            f"[diversity][WARN] embedding contains non-finite values; "
            f"context={context}, bad_values={int(bad_mask.sum())}, bad_rows={bad_rows}. "
            "Replacing them with 0 before clustering."
        )
    return _normalize_rows(matrix)


def compute_diversity_embeddings(
    frame: pd.DataFrame,
    *,
    config: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    input_path: Path,
    output_root: Path,
    cutoff_len: int,
) -> Tuple[np.ndarray, Path]:
    div_cfg = dict(baseline_cfg.get("diversity") or {})
    model_name_or_path = str(div_cfg.get("model_name_or_path") or config.get("base_model"))
    if not model_name_or_path:
        raise ValueError("diversity baseline 需要 baseline.diversity.model_name_or_path 或 base_model")
    model_path = resolve_path(model_name_or_path)
    model_ref = str(model_path if model_path.exists() else model_name_or_path)
    layer = _resolve_embedding_layer(div_cfg.get("layer", 14))
    emb_path = (
        resolve_path(div_cfg["embedding_path"])
        if div_cfg.get("embedding_path")
        else diversity_cache_paths(output_root, model_ref, input_path, cutoff_len=cutoff_len, layer=layer)[0]
    )
    meta_path = emb_path.with_name("embeddings_meta.json")
    if emb_path.exists():
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != len(frame):
            raise ValueError(f"diversity embedding cache 行数不匹配: {emb_path}, cache={embeddings.shape[0]}, data={len(frame)}")
        embeddings = _sanitize_embedding_matrix(embeddings, context=f"cache:{emb_path}")
        return embeddings.astype(np.float32, copy=False), emb_path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = str(div_cfg.get("device") or config.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[diversity] CUDA 不可用，自动回退到 CPU")
        device = "cpu"
    dtype = _resolve_torch_dtype(div_cfg.get("dtype", "auto"), device)
    batch_size = max(1, int(div_cfg.get("batch_size", 8)))
    log_every = max(1, int(div_cfg.get("log_every", 100)))

    print(f"[diversity] model={model_ref}")
    print(f"[diversity] cache={emb_path}")
    print(f"[diversity] records={len(frame)} layer={layer} cutoff_len={cutoff_len}")

    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    rows = frame.to_dict(orient="records")
    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            texts = [
                f"Question: {str(record.get('problem', '')).strip()}\nAnswer: {str(record.get('solution', '')).strip()}"
                for record in batch
            ]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(cutoff_len),
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            if layer == "last":
                hidden = hidden_states[-1]
            else:
                idx = int(layer) + 1
                if idx <= 0 or idx >= len(hidden_states):
                    raise ValueError(f"diversity layer {layer} 超出模型层数范围: {len(hidden_states) - 1}")
                hidden = hidden_states[idx]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            if (start // batch_size + 1) % log_every == 0 or start + batch_size >= len(rows):
                print(f"[diversity] embedded {min(start + batch_size, len(rows))}/{len(rows)}")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    embeddings = _sanitize_embedding_matrix(embeddings, context=f"new:{emb_path}")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, embeddings)
    save_json(
        meta_path,
        {
            "format_version": 1,
            "model_name_or_path": model_ref,
            "input_path": str(input_path),
            "input_fingerprint": _fingerprint_file(input_path),
            "record_count": int(len(frame)),
            "cutoff_len": int(cutoff_len),
            "layer": layer,
            "pooling": "attention_mask_mean",
            "normalization": "sample_l2",
            "batch_size": int(batch_size),
            "updated_at": utc_now(),
        },
    )
    return embeddings, emb_path


def _safe_cache_token(value: Any) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    return token or "unknown"


def _diversity_cluster_request(
    *,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    input_path: Path,
    dataset_name: str,
    cutoff_len: int,
    seed: int,
) -> Dict[str, Any]:
    div_cfg = dict(baseline_cfg.get("diversity") or {})
    layer = _resolve_embedding_layer(div_cfg.get("layer", 14))
    target_cluster_size = int(div_cfg.get("target_cluster_size", 100))
    if target_cluster_size <= 0:
        raise ValueError(f"baseline.diversity.target_cluster_size 必须 > 0: {target_cluster_size}")
    return {
        "format_version": 2,
        "dataset_name": dataset_name,
        "source_prefix": baseline_cfg.get("source_prefix") or dataset_name,
        "input_fingerprint": _fingerprint_file(input_path),
        "row_count": int(len(frame)),
        "source_distribution": distribution(frame),
        "base_model": str(div_cfg.get("model_name_or_path") or config.get("base_model")),
        "cutoff_len": int(cutoff_len),
        "layer": layer,
        "target_cluster_size": target_cluster_size,
        "kmeans_batch_size": int(div_cfg.get("kmeans_batch_size", 2048)),
        "max_iter": int(div_cfg.get("max_iter", 100)),
        "n_init": int(div_cfg.get("n_init", 10)),
        "seed": int(seed),
    }


def diversity_cluster_cache_root(
    output_root: Path,
    dataset_name: str,
    request: Dict[str, Any],
) -> Path:
    return output_root / "_diversity_cluster_cache" / _safe_cache_token(dataset_name) / stable_hash(request)[:12]


def _cluster_distances(embeddings: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    assigned = centers[labels]
    diff = embeddings - assigned
    return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)


def ensure_diversity_cluster_cache(
    *,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    input_path: Path,
    output_root: Path,
    dataset_name: str,
    embeddings: np.ndarray,
    cutoff_len: int,
    seed: int,
) -> Path:
    """Build or reuse the shared per-NuminaMath_source diversity cluster cache."""
    if embeddings.shape[0] != len(frame):
        raise ValueError(f"diversity embeddings 行数不匹配: embeddings={embeddings.shape[0]}, frame={len(frame)}")
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError("diversity baseline 需要 scikit-learn，请先安装 requirements.txt") from exc

    request = _diversity_cluster_request(
        frame=frame,
        config=config,
        baseline_cfg=baseline_cfg,
        input_path=input_path,
        dataset_name=dataset_name,
        cutoff_len=cutoff_len,
        seed=seed,
    )
    cache_root = diversity_cluster_cache_root(output_root, dataset_name, request)
    manifest_path = cache_root / "cluster_manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("request") == request:
                return cache_root
        except Exception:
            pass

    div_cfg = dict(baseline_cfg.get("diversity") or {})
    target_cluster_size = int(request["target_cluster_size"])
    batch_size = int(request["kmeans_batch_size"])
    max_iter = int(request["max_iter"])
    n_init = int(request["n_init"])
    sources = frame["source"].astype(str).to_numpy()
    row_ids = frame["row_id"].astype(str).to_numpy()

    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    source_entries = []
    for source in sorted(set(sources.tolist())):
        positions = np.flatnonzero(sources == str(source)).astype(np.int64)
        source_embeddings = embeddings[positions]
        n_clusters = max(1, int(math.ceil(len(positions) / target_cluster_size)))
        n_clusters = min(n_clusters, len(positions))
        if n_clusters == 1:
            labels = np.zeros(len(positions), dtype=np.int32)
            centers = source_embeddings.mean(axis=0, keepdims=True).astype(np.float32)
            norm = np.linalg.norm(centers, axis=1, keepdims=True)
            centers = centers / np.clip(norm, 1e-12, None)
        else:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=batch_size,
                max_iter=max_iter,
                random_state=int(seed),
                n_init=n_init,
            )
            labels = kmeans.fit_predict(source_embeddings).astype(np.int32)
            centers = kmeans.cluster_centers_.astype(np.float32)
        distances = _cluster_distances(source_embeddings.astype(np.float32), centers.astype(np.float32), labels)
        source_token = _safe_cache_token(source)
        npz_name = f"{source_token}.npz"
        np.savez_compressed(
            cache_root / npz_name,
            positions=positions,
            labels=labels,
            centers=centers,
            distances=distances,
            row_ids=row_ids[positions].astype(str),
            n_clusters=np.asarray([n_clusters], dtype=np.int64),
            target_cluster_size=np.asarray([target_cluster_size], dtype=np.int64),
        )
        cluster_sizes = {
            f"cluster_{cid:04d}": int(np.sum(labels == cid))
            for cid in sorted(set(int(label) for label in labels))
        }
        source_entries.append(
            {
                "source": str(source),
                "source_token": source_token,
                "cache_file": npz_name,
                "available": int(len(positions)),
                "n_clusters": int(n_clusters),
                "target_cluster_size": int(target_cluster_size),
                "cluster_sizes": cluster_sizes,
            }
        )

    save_json(
        manifest_path,
        {
            "request": request,
            "cache_root": str(cache_root),
            "sources": source_entries,
            "created_at": utc_now(),
        },
    )
    print(f"[diversity] shared cluster cache ready: {cache_root}")
    return cache_root


def load_diversity_cluster_cache(cache_root: Path) -> Dict[str, Dict[str, Any]]:
    manifest_path = cache_root / "cluster_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"diversity cluster cache manifest 不存在: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    loaded: Dict[str, Dict[str, Any]] = {}
    for entry in manifest.get("sources", []):
        source = str(entry["source"])
        data = np.load(cache_root / entry["cache_file"], allow_pickle=False)
        loaded[source] = {
            "positions": data["positions"].astype(np.int64),
            "labels": data["labels"].astype(np.int32),
            "centers": data["centers"].astype(np.float32),
            "distances": data["distances"].astype(np.float32),
            "row_ids": data["row_ids"].astype(str),
            "meta": entry,
        }
    return loaded


def ensure_shared_diversity_cache_for_config(
    config: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    size: int,
) -> Path:
    input_path = resolve_path(baseline_cfg.get("input_path", DEFAULT_INPUT_PATH))
    output_root = resolve_path(baseline_cfg.get("output_root", DEFAULT_OUTPUT_ROOT))
    dataset_name = str(baseline_cfg.get("dataset_name") or input_path.parent.name or DEFAULT_DATASET_NAME)
    seed = int(baseline_cfg.get("seed", config.get("seed", 42)))
    frame = load_train_frame(input_path)
    cutoff_len = resolve_baseline_cutoff_len(config, baseline_cfg, int(size))
    embeddings, _ = compute_diversity_embeddings(
        frame,
        config=config,
        baseline_cfg=baseline_cfg,
        input_path=input_path,
        output_root=output_root,
        cutoff_len=cutoff_len,
    )
    return ensure_diversity_cluster_cache(
        frame=frame,
        config=config,
        baseline_cfg=baseline_cfg,
        input_path=input_path,
        output_root=output_root,
        dataset_name=dataset_name,
        embeddings=embeddings,
        cutoff_len=cutoff_len,
        seed=seed,
    )


def _round_robin_cluster_sample(
    positions: np.ndarray,
    labels: np.ndarray,
    *,
    count: int,
    seed: int,
    row_ids: Sequence[Any],
) -> List[int]:
    selected: List[int] = []
    seen = set()
    cluster_ids = sorted(set(int(label) for label in labels), key=lambda cid: stable_float_key(seed, "cluster", cid))
    for cid in cluster_ids:
        local = [idx for idx, label in enumerate(labels) if int(label) == cid]
        local.sort(key=lambda idx: (stable_float_key(seed, "diversity", row_ids[int(positions[idx])]), str(row_ids[int(positions[idx])])))
        for idx in local:
            pos = int(positions[idx])
            if pos not in seen:
                selected.append(pos)
                seen.add(pos)
                break
        if len(selected) >= count:
            return selected

    remaining = [int(pos) for pos in positions if int(pos) not in seen]
    remaining.sort(key=lambda pos: (stable_float_key(seed, "diversity-fill", row_ids[pos]), str(row_ids[pos])))
    selected.extend(remaining[: max(0, count - len(selected))])
    return selected[:count]


def select_diversity_frame(
    frame: pd.DataFrame,
    allocation: Dict[str, int],
    *,
    embeddings: np.ndarray,
    seed: int,
    diversity_cfg: Dict[str, Any],
    cluster_cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    if embeddings.shape[0] != len(frame):
        raise ValueError(f"diversity embeddings 行数不匹配: embeddings={embeddings.shape[0]}, frame={len(frame)}")
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as exc:
        raise ImportError("diversity baseline 需要 scikit-learn，请先安装 requirements.txt") from exc

    row_ids = frame["row_id"].astype(str).tolist()
    sources = frame["source"].astype(str).to_numpy()
    selected_positions: List[int] = []
    batch_size = int(diversity_cfg.get("kmeans_batch_size", 2048))
    max_iter = int(diversity_cfg.get("max_iter", 100))
    n_init = int(diversity_cfg.get("n_init", 10))
    target_cluster_size = int(diversity_cfg.get("target_cluster_size", 100))
    if target_cluster_size <= 0:
        raise ValueError(f"baseline.diversity.target_cluster_size 必须 > 0: {target_cluster_size}")
    cluster_cache = load_diversity_cluster_cache(cluster_cache_dir) if cluster_cache_dir is not None and (cluster_cache_dir / "cluster_manifest.json").exists() else {}

    for source, count in sorted(allocation.items()):
        positions = np.flatnonzero(sources == str(source))
        if len(positions) < count:
            raise ValueError(f"source={source} 可用样本不足: requested={count}, available={len(positions)}")
        if count <= 0:
            continue
        if len(positions) == count:
            selected_positions.extend(int(pos) for pos in positions)
            continue
        cached = cluster_cache.get(str(source))
        if cached is not None:
            cached_positions = cached["positions"].astype(np.int64)
            if set(cached_positions.tolist()) != set(positions.astype(np.int64).tolist()):
                raise ValueError(f"source={source} 的 cluster cache positions 与当前 frame 不匹配")
            order = {int(pos): idx for idx, pos in enumerate(cached_positions.tolist())}
            local_order = np.asarray([order[int(pos)] for pos in positions], dtype=np.int64)
            labels = cached["labels"][local_order].astype(np.int32)
        else:
            n_clusters = min(len(positions), max(1, int(math.ceil(len(positions) / target_cluster_size))))
            source_embeddings = embeddings[positions]
            if n_clusters == 1:
                labels = np.zeros(len(positions), dtype=np.int32)
            else:
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=batch_size,
                    max_iter=max_iter,
                    random_state=int(seed),
                    n_init=n_init,
                )
                labels = kmeans.fit_predict(source_embeddings).astype(np.int32)
        if cluster_cache_dir is not None and cached is None:
            print(f"[diversity][WARN] 未找到共享 cluster cache source={source}，已临时聚类但不会供 offPolicyData 复用")
        selected_positions.extend(
            _round_robin_cluster_sample(
                positions,
                labels,
                count=count,
                seed=seed,
                row_ids=row_ids,
            )
        )

    if not selected_positions:
        raise ValueError("没有选择出任何 diversity 样本")
    return frame.iloc[selected_positions].copy().reset_index(drop=True)

def build_one_baseline(
    *,
    frame: pd.DataFrame,
    config: Dict[str, Any],
    baseline_cfg: Dict[str, Any],
    input_path: Path,
    output_root: Path,
    dataset_name: str,
    method: str,
    size: int,
    seed: int,
    source_counts: pd.Series,
    scores_frame: Optional[pd.DataFrame],
    score_cache_path: Optional[Path],
    diversity_embeddings: Optional[np.ndarray],
    diversity_cache_path: Optional[Path],
    cutoff_len: int,
    dry_run: bool,
    overwrite: bool,
) -> Dict[str, Any]:
    cfs_cfg = dict(baseline_cfg.get("cfs") or {})
    ppl_cfg = dict(baseline_cfg.get("ppl") or {})
    diversity_cfg = dict(baseline_cfg.get("diversity") or {})
    cfs_ratio = float(cfs_cfg.get("ratio", baseline_cfg.get("cfs_ratio", 0.5)))
    if not (0.0 <= cfs_ratio <= 1.0):
        raise ValueError(f"baseline.cfs.ratio 必须在 [0, 1] 内，当前为: {cfs_ratio}")
    middle_fraction = float(ppl_cfg.get("middle_fraction", baseline_cfg.get("middle_fraction", 0.70)))
    nominal_size = int(size)
    source_prefix = str(baseline_cfg.get("source_prefix") or dataset_name)
    cfs_count = int(round(nominal_size * cfs_ratio)) if method == "cfs" else 0
    base_output_size = nominal_size
    train_sample_count = nominal_size + cfs_count
    if base_output_size <= 0:
        raise ValueError(f"baseline base_output_size 必须 > 0: method={method}, size={size}")
    if method == "cfs" and cfs_count <= 0:
        raise ValueError(f"cfs 至少需要 1 条 CFS 样本: size={size}, ratio={cfs_ratio}")
    dataset_for_task = method_dataset_name(dataset_name, method, cfs_ratio, middle_fraction, source_prefix)
    task_id = task_id_for(dataset_name, method, size, cfs_ratio, middle_fraction, source_prefix)
    task_root = output_root / task_id
    selected_path = task_root / "selected.parquet"
    cfs_path = task_root / "cfs.jsonl"
    meta_path = task_root / "selection_meta.json"

    allocation = allocate_stratified_counts(source_counts, base_output_size)
    request = {
        "format_version": 1,
        "dataset_name": dataset_name,
        "source_prefix": source_prefix,
        "method": method,
        "nominal_size": nominal_size,
        "base_output_size": int(base_output_size),
        "train_sample_count": int(train_sample_count),
        "seed": int(seed),
        "input_fingerprint": _fingerprint_file(input_path),
        "allocation": allocation,
        "cfs_ratio": cfs_ratio if method == "cfs" else None,
        "cfs_count": cfs_count if method == "cfs" else None,
        "middle_fraction": middle_fraction if method == "ppl_cond_middle" else None,
        "cutoff_len": int(cutoff_len) if method in PPL_METHODS or method in DIVERSITY_METHODS else None,
    }

    if not overwrite and not dry_run and selected_path.exists() and meta_path.exists() and _request_matches(meta_path, request):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "task_id": task_id,
            "dataset": dataset_for_task,
            "max_samples": int(size),
            "train_sample_count": int(meta.get("counts", {}).get("total_training_records", size)),
            "method": method,
            "input_paths": meta["outputs"]["prepare_input_paths"],
            "dataset_alias": meta["outputs"].get("dataset_alias", DEFAULT_DATASET_ALIAS),
            "train_args_overrides": meta.get("train_args_overrides", {}),
            "baseline_meta_path": str(meta_path),
            "baseline_training": meta.get("training", {}),
            "reused_existing": True,
            "meta": meta,
        }

    working_frame = add_random_key(frame, seed)
    if method in PPL_METHODS:
        if scores_frame is None:
            if dry_run:
                selected = select_from_sorted_frame(working_frame, allocation, ["_baseline_random_key", "row_id"])
                selected["baseline_score"] = None
            else:
                raise ValueError(f"{method} 需要 PPL scores，但 scores_frame=None")
        else:
            selected = select_ppl_middle_frame(
                scores_frame.copy(),
                allocation,
                seed=seed,
                middle_fraction=middle_fraction,
                score_column="ppl_cond",
            )
            selected["baseline_score"] = selected["ppl_cond"]
    elif method in DIVERSITY_METHODS:
        if diversity_embeddings is None:
            if dry_run:
                selected = select_from_sorted_frame(working_frame, allocation, ["_baseline_random_key", "row_id"])
                selected["baseline_score"] = selected["_baseline_random_key"]
            else:
                raise ValueError(f"{method} 需要 diversity embeddings，但 diversity_embeddings=None")
        else:
            shared_cluster_cache = ensure_diversity_cluster_cache(
                frame=frame,
                config=config,
                baseline_cfg=baseline_cfg,
                input_path=input_path,
                output_root=output_root,
                dataset_name=dataset_name,
                embeddings=diversity_embeddings,
                cutoff_len=cutoff_len,
                seed=seed,
            )
            selected = select_diversity_frame(
                frame,
                allocation,
                embeddings=diversity_embeddings,
                seed=seed,
                diversity_cfg=diversity_cfg,
                cluster_cache_dir=shared_cluster_cache,
            )
            selected["_baseline_random_key"] = [
                stable_float_key(seed, "diversity-selected", row_id)
                for row_id in selected["row_id"].astype(str)
            ]
            selected["baseline_score"] = selected["_baseline_random_key"]
    else:
        selected = select_from_sorted_frame(working_frame, allocation, ["_baseline_random_key", "row_id"])
        selected["baseline_score"] = selected["_baseline_random_key"]

    selected = selected.copy()
    selected["baseline_method"] = method
    selected["baseline_task_id"] = task_id
    selected["baseline_size"] = nominal_size
    selected["baseline_base_output_size"] = int(base_output_size)
    selected["baseline_selected_at"] = utc_now()

    if not dry_run and task_root.exists() and overwrite:
        shutil.rmtree(task_root)

    cfs_meta: Dict[str, Any] = {}
    prepare_input_paths = [str(selected_path)]
    train_args_overrides: Dict[str, Any] = {}
    training_meta: Dict[str, Any] = {}
    total_count = int(len(selected))
    if method == "ppl_cond_middle":
        training_meta.update(
            {
                "nominal_size": int(nominal_size),
                "actual_base_count": int(len(selected)),
                "middle_fraction": float(middle_fraction),
                "fair_sample_count": True,
            }
        )
    if method == "cfs":
        total_count += cfs_count
        cfs_meta = {
            "ratio": cfs_ratio,
            "base_count": int(len(selected)),
            "requested_count": cfs_count,
            "ratio_semantics": "extra_fraction_of_datasize",
        }
        if not dry_run:
            cfs_selected_meta = select_cfs_jsonl(
                cfs_cfg=cfs_cfg,
                target_count=cfs_count,
                seed=seed,
                output_path=cfs_path,
                output_dir=task_root,
            )
            cfs_meta.update(cfs_selected_meta)
        prepare_input_paths.append(str(cfs_path))
        training_meta.update(
            {
                "base_count": int(len(selected)),
                "cfs_count": int(cfs_count),
                "total_count": int(total_count),
                "cfs_ratio": float(cfs_ratio),
                "ratio_semantics": "extra_fraction_of_datasize",
                "data_size": int(nominal_size),
                "train_sample_count": int(total_count),
            }
        )
    expected_total = nominal_size + (cfs_count if method == "cfs" else 0)
    if total_count != expected_total:
        raise RuntimeError(
            f"baseline task {task_id} 训练样本数不匹配: total_count={total_count}, expected={expected_total}"
        )

    selected_dist = distribution(selected)
    original_dist = distribution(frame)
    meta = {
        "request": request,
        "format_version": 1,
        "stage": "baseline",
        "task_id": task_id,
        "dataset": dataset_for_task,
        "source_prefix": source_prefix,
        "method": method,
        "size": nominal_size,
        "nominal_size": nominal_size,
        "base_output_size": int(base_output_size),
        "train_sample_count": int(total_count),
        "seed": int(seed),
        "input_path": str(input_path),
        "outputs": {
            "selected_parquet": str(selected_path),
            "cfs_jsonl": str(cfs_path) if method == "cfs" else None,
            "prepare_input_paths": prepare_input_paths,
            "dataset_alias": str((config.get("prepare") or {}).get("dataset", {}).get("dataset_alias", DEFAULT_DATASET_ALIAS)),
        },
        "counts": {
            "input_records": int(len(frame)),
            "selected_base_records": int(len(selected)),
            "cfs_records": int(total_count - len(selected)),
            "total_training_records": int(total_count),
        },
        "allocation": allocation,
        "selection": {
            "middle_fraction": float(middle_fraction) if method == "ppl_cond_middle" else None,
            "cutoff_len": int(cutoff_len) if method in PPL_METHODS or method in DIVERSITY_METHODS else None,
            "ppl_score_column": "ppl_cond" if method in PPL_METHODS else None,
            "diversity_layer": diversity_cfg.get("layer", 14) if method in DIVERSITY_METHODS else None,
        },
        "source_distribution": {
            "original": original_dist,
            "selected_base": selected_dist,
            "ratio_delta": _source_ratio_delta(original_dist, selected_dist),
        },
        "score_cache": str(score_cache_path) if score_cache_path is not None and method in PPL_METHODS else None,
        "diversity_cache": str(diversity_cache_path) if diversity_cache_path is not None and method in DIVERSITY_METHODS else None,
        "diversity_cluster_cache": str(shared_cluster_cache) if method in DIVERSITY_METHODS and not dry_run and diversity_embeddings is not None else None,
        "cfs": cfs_meta,
        "train_args_overrides": train_args_overrides,
        "training": training_meta,
        "created_at": utc_now(),
    }

    if dry_run:
        print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    else:
        task_root.mkdir(parents=True, exist_ok=True)
        _write_selected_parquet(selected_path, selected)
        save_json(meta_path, meta)

    return {
        "task_id": task_id,
        "dataset": dataset_for_task,
        "max_samples": int(size),
        "train_sample_count": int(total_count),
        "method": method,
        "input_paths": prepare_input_paths,
        "dataset_alias": meta["outputs"]["dataset_alias"],
        "train_args_overrides": train_args_overrides,
        "baseline_meta_path": str(meta_path),
        "baseline_training": training_meta,
        "reused_existing": False,
        "meta": meta,
    }


def inject_baseline_tasks(config: Dict[str, Any], tasks: List[Dict[str, Any]]) -> None:
    prepare_cfg = config.setdefault("prepare", {}).setdefault("dataset", {})
    train_cfg = config.setdefault("train", {})
    eval_cfg = config.setdefault("eval", {})
    dataset_alias = str(prepare_cfg.get("dataset_alias", DEFAULT_DATASET_ALIAS))

    prepare_cfg["tasks"] = [
        {
            "task_id": task["task_id"],
            "input_paths": list(task["input_paths"]),
            "dataset_alias": task.get("dataset_alias", dataset_alias),
        }
        for task in tasks
    ]
    train_cfg["tasks"] = [
        {
            "dataset": task["dataset"],
            "max_samples": task["max_samples"],
            "train_sample_count": task.get("train_sample_count", task["max_samples"]),
            "data_size": task["max_samples"],
            "task_id": task["task_id"],
            "train_args_overrides": deepcopy(task.get("train_args_overrides") or {}),
            "baseline_meta_path": task.get("baseline_meta_path"),
            "baseline_training": deepcopy(task.get("baseline_training") or {}),
        }
        for task in tasks
    ]
    eval_cfg["tasks"] = [
        {
            "dataset": task["dataset"],
            "max_samples": task["max_samples"],
        }
        for task in tasks
    ]


def run_baseline(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    baseline_cfg = dict(config.get("baseline") or {})
    if not baseline_cfg.get("enable", True):
        print("[INFO] Baseline stage 已关闭，跳过数据筛选")
        return []

    input_path = resolve_path(baseline_cfg.get("input_path", DEFAULT_INPUT_PATH))
    output_root = resolve_path(baseline_cfg.get("output_root", DEFAULT_OUTPUT_ROOT))
    dataset_name = str(baseline_cfg.get("dataset_name") or input_path.parent.name or DEFAULT_DATASET_NAME)
    source_prefix = str(baseline_cfg.get("source_prefix") or dataset_name)
    methods = normalize_methods(baseline_cfg.get("methods"))
    sizes = normalize_sizes(baseline_cfg.get("sizes"))
    seed = int(baseline_cfg.get("seed", config.get("seed", 42)))
    dry_run = bool(baseline_cfg.get("dry_run", False))
    overwrite = bool(baseline_cfg.get("overwrite", False))

    print(f"[Baseline] input={input_path}")
    print(f"[Baseline] output_root={output_root}")
    print(f"[Baseline] dataset={dataset_name} source_prefix={source_prefix} methods={methods} sizes={sizes} seed={seed}")

    frame = load_train_frame(input_path)
    source_counts = frame["source"].value_counts().sort_index()
    cutoff_by_size = {int(size): resolve_baseline_cutoff_len(config, baseline_cfg, int(size)) for size in sizes}
    if any(method in PPL_METHODS or method in DIVERSITY_METHODS for method in methods):
        print(f"[Baseline] train cutoff_len by size: {cutoff_by_size}")

    score_frames_by_cutoff: Dict[int, pd.DataFrame] = {}
    score_paths_by_cutoff: Dict[int, Path] = {}
    diversity_by_cutoff: Dict[int, np.ndarray] = {}
    diversity_paths_by_cutoff: Dict[int, Path] = {}
    if dry_run and any(method in PPL_METHODS for method in methods) and not bool((baseline_cfg.get("ppl") or {}).get("score_in_dry_run", False)):
        print("[Baseline] dry_run=true，跳过 PPL 实际打分，仅验证配额和任务生成。")
    if dry_run and any(method in DIVERSITY_METHODS for method in methods) and not bool((baseline_cfg.get("diversity") or {}).get("embed_in_dry_run", False)):
        print("[Baseline] dry_run=true，跳过 diversity embedding，仅验证配额和任务生成。")

    tasks: List[Dict[str, Any]] = []
    for method in methods:
        for size in sizes:
            cutoff_len = cutoff_by_size[int(size)]
            scores_frame: Optional[pd.DataFrame] = None
            score_path: Optional[Path] = None
            diversity_embeddings: Optional[np.ndarray] = None
            diversity_cache_path: Optional[Path] = None
            if method in PPL_METHODS:
                should_score = not dry_run or bool((baseline_cfg.get("ppl") or {}).get("score_in_dry_run", False))
                if should_score:
                    if cutoff_len not in score_frames_by_cutoff:
                        scores, computed_score_path = compute_ppl_scores(
                            frame,
                            config=config,
                            baseline_cfg=baseline_cfg,
                            input_path=input_path,
                            output_root=output_root,
                            max_length=cutoff_len,
                        )
                        score_frames_by_cutoff[cutoff_len] = merge_scores(frame, scores)
                        score_paths_by_cutoff[cutoff_len] = computed_score_path
                        print(f"[Baseline] PPL cache ready: {computed_score_path}")
                    scores_frame = score_frames_by_cutoff[cutoff_len]
                    score_path = score_paths_by_cutoff[cutoff_len]
            if method in DIVERSITY_METHODS:
                should_embed = not dry_run or bool((baseline_cfg.get("diversity") or {}).get("embed_in_dry_run", False))
                if should_embed:
                    if cutoff_len not in diversity_by_cutoff:
                        embeddings, emb_path = compute_diversity_embeddings(
                            frame,
                            config=config,
                            baseline_cfg=baseline_cfg,
                            input_path=input_path,
                            output_root=output_root,
                            cutoff_len=cutoff_len,
                        )
                        diversity_by_cutoff[cutoff_len] = embeddings
                        diversity_paths_by_cutoff[cutoff_len] = emb_path
                        print(f"[Baseline] diversity cache ready: {emb_path}")
                    diversity_embeddings = diversity_by_cutoff[cutoff_len]
                    diversity_cache_path = diversity_paths_by_cutoff[cutoff_len]
            task = build_one_baseline(
                frame=frame,
                config=config,
                baseline_cfg=baseline_cfg,
                input_path=input_path,
                output_root=output_root,
                dataset_name=dataset_name,
                method=method,
                size=size,
                seed=seed,
                source_counts=source_counts,
                scores_frame=scores_frame,
                score_cache_path=score_path,
                diversity_embeddings=diversity_embeddings,
                diversity_cache_path=diversity_cache_path,
                cutoff_len=cutoff_len,
                dry_run=dry_run,
                overwrite=overwrite,
            )
            tasks.append(task)

    inject_baseline_tasks(config, tasks)
    summary = {
        "format_version": 1,
        "stage": "baseline",
        "input_path": str(input_path),
        "output_root": str(output_root),
        "dataset_name": dataset_name,
        "source_prefix": source_prefix,
        "methods": methods,
        "sizes": sizes,
        "cutoff_by_size": cutoff_by_size,
        "task_count": len(tasks),
        "tasks": [
            {
                "task_id": task["task_id"],
                "dataset": task["dataset"],
                "max_samples": task["max_samples"],
                "input_paths": task["input_paths"],
            }
            for task in tasks
        ],
        "created_at": utc_now(),
    }
    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    else:
        save_json(output_root / "baseline_tasks_meta.json", summary)
    print(f"[Baseline] generated tasks: {[task['task_id'] for task in tasks]}")
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Density.yaml", help="Pipeline config path")
    parser.add_argument("--methods", nargs="+", default=None, help="Override baseline methods")
    parser.add_argument("--sizes", nargs="+", type=int, default=None, help="Override baseline sizes")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print planned tasks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.setdefault("baseline", {})
    if args.methods is not None:
        config["baseline"]["methods"] = args.methods
    if args.sizes is not None:
        config["baseline"]["sizes"] = args.sizes
    if args.dry_run:
        config["baseline"]["dry_run"] = True
    if args.overwrite:
        config["baseline"]["overwrite"] = True
    run_baseline(config)


if __name__ == "__main__":
    main()
