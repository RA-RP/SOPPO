#!/usr/bin/env python3
# coding: utf-8
"""Embedding-cluster top-k KL pipeline."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .common import (
    Record,
    ensure_dir_for_stage,
    load_json_config,
    load_manifest,
    load_source_records,
    load_unit_records,
    read_unit_kl,
    resolve_ranking_config,
    run_getslice,
    save_json,
    stage_kl_from_smat,
    write_dataset_dir,
    write_getslice_config,
    write_manifest,
)

from Baseline.select_baseline import load_diversity_cluster_cache, load_train_frame


DEFAULT_CLUSTER_META = "cluster_meta.json"


def resolve_embedding_layer(config: Dict[str, Any], getslice_config: Dict[str, Any]) -> Any:
    layer = config.get("embedding", {}).get("layer", "getslice_target")
    if layer == "getslice_target":
        target = getslice_config.get("target_layer")
        if target is None or str(target).strip() == "":
            return "last"
        if isinstance(target, str) and target.startswith("layer_"):
            return int(target.split("_", 1)[1])
        return int(target)
    return layer


def compute_embeddings(config: Dict[str, Any], records: Sequence[Record], out_path: Path) -> np.ndarray:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emb_cfg = config.get("embedding", {})
    getslice_cfg = load_json_config(Path(config["getslice"]["base_config"]))
    layer = resolve_embedding_layer(config, getslice_cfg)
    model_path = emb_cfg.get("model_path") or getslice_cfg.get("model")
    requested_device = str(emb_cfg.get("device", "cuda"))
    device = requested_device if requested_device == "cpu" or torch.cuda.is_available() else "cpu"
    dtype_name = str(emb_cfg.get("dtype", "float16"))
    dtype = torch.float16 if dtype_name == "float16" and device != "cpu" else torch.float32
    batch_size = int(emb_cfg.get("batch_size", 8))
    max_length = int(emb_cfg.get("max_length", 512))

    print(f"[embed] model={model_path}, layer={layer}, device={device}, max_length={max_length}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    all_embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            texts = [f"Question: {r.question}\nAnswer: {r.answer}" for r in batch]
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            if layer == "last":
                hidden = hidden_states[-1]
            elif layer == "input":
                hidden = hidden_states[0]
            else:
                idx = int(layer) + 1
                if idx <= 0 or idx >= len(hidden_states):
                    raise ValueError(
                        f"embedding layer {layer} out of range for {len(hidden_states) - 1} layers"
                    )
                hidden = hidden_states[idx]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32))
            if (start // batch_size + 1) % 20 == 0:
                print(f"[embed] processed {min(start + batch_size, len(records))}/{len(records)}")

    embeddings = np.concatenate(all_embeddings, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    print(f"[embed] saved {embeddings.shape} -> {out_path}")
    return embeddings


def load_records_and_embeddings(config: Dict[str, Any], root: Path) -> Tuple[List[Record], np.ndarray]:
    manifest = root / "records_manifest.jsonl"
    emb_path = root / "embeddings.npy"
    if not manifest.exists():
        raise FileNotFoundError(f"missing manifest; run embed first: {manifest}")
    if not emb_path.exists():
        raise FileNotFoundError(f"missing embeddings; run embed first: {emb_path}")
    records = load_manifest(manifest)
    embeddings = np.load(emb_path)
    if len(records) != embeddings.shape[0]:
        raise ValueError("manifest and embeddings length mismatch")
    return records, embeddings


def stage_embed(config: Dict[str, Any], root: Path) -> None:
    ensure_dir_for_stage(root, overwrite=False, allow_existing=True)
    records = load_source_records(config)
    write_manifest(root / "records_manifest.jsonl", records)
    compute_embeddings(config, records, root / "embeddings.npy")


def balanced_assignments(embeddings: np.ndarray, centers: np.ndarray, capacities: List[int]) -> np.ndarray:
    diff = embeddings[:, None, :] - centers[None, :, :]
    distances = np.sum(diff * diff, axis=2)
    preferences = np.argsort(distances, axis=1)
    min_dist = np.min(distances, axis=1)
    order = np.argsort(min_dist)
    remaining = capacities[:]
    labels = np.full(embeddings.shape[0], -1, dtype=np.int32)
    for idx in order:
        for cluster_id in preferences[idx]:
            cluster_id = int(cluster_id)
            if remaining[cluster_id] > 0:
                labels[idx] = cluster_id
                remaining[cluster_id] -= 1
                break
        if labels[idx] < 0:
            raise RuntimeError("failed to assign sample to any cluster")
    return labels


def compute_cluster_distances(embeddings: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    assigned_centers = centers[labels]
    diff = embeddings - assigned_centers
    return np.sqrt(np.sum(diff * diff, axis=1))


def stage_cluster(config: Dict[str, Any], root: Path) -> None:
    from sklearn.cluster import MiniBatchKMeans

    records, embeddings = load_records_and_embeddings(config, root)
    cl_cfg = config.get("clustering", {})
    target_size = int(cl_cfg.get("target_cluster_size", 100))
    n = len(records)
    if target_size <= 0:
        raise ValueError("target_cluster_size must be > 0")
    n_clusters = max(1, math.ceil(n / target_size))
    print(f"[cluster] n={n}, target_cluster_size={target_size}, n_clusters={n_clusters}")

    if n_clusters == 1:
        labels = np.zeros(n, dtype=np.int32)
        centers = embeddings.mean(axis=0, keepdims=True)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(min=1e-12)
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=int(cl_cfg.get("batch_size", 1024)),
            max_iter=int(cl_cfg.get("max_iter", 100)),
            random_state=int(config.get("seed", 42)),
            n_init=10,
        )
        initial_labels = kmeans.fit_predict(embeddings)
        centers = kmeans.cluster_centers_.astype(np.float32)
        initial_counts = Counter(int(x) for x in initial_labels)
        low_capacity_cluster = min(range(n_clusters), key=lambda cid: initial_counts.get(cid, 0))
        capacities = [target_size for _ in range(n_clusters)]
        capacities[low_capacity_cluster] = n - target_size * (n_clusters - 1)
        if capacities[low_capacity_cluster] <= 0:
            capacities[low_capacity_cluster] = target_size
        labels = balanced_assignments(embeddings, centers, capacities)

    distances = compute_cluster_distances(embeddings, centers, labels)
    clusters_dir = root / "clusters"
    ensure_dir_for_stage(clusters_dir, overwrite=bool(config.get("overwrite_outputs", False)))

    dataset_alias = config.get("dataset_alias", "gsm8k_math_train")
    task_file = config.get("task_file", "gsm8k.jsonl")
    train_file = config.get("train_file", "gsm8k-train.jsonl")
    cluster_rows = []
    for cid in sorted(set(int(x) for x in labels)):
        indices = np.where(labels == cid)[0].tolist()
        indices.sort(key=lambda i: (float(distances[i]), records[i].global_id))
        cluster_records = [records[i] for i in indices]
        extra_fields = {
            records[i].global_id: {
                "cluster_id": f"cluster_{cid:04d}",
                "cluster_distance": float(distances[i]),
            }
            for i in indices
        }
        cluster_id = f"cluster_{cid:04d}"
        cluster_dir = clusters_dir / cluster_id
        write_dataset_dir(cluster_dir, cluster_records, dataset_alias, task_file, train_file, extra_fields)
        source_counts = Counter(r.source for r in cluster_records)
        save_json(
            cluster_dir / DEFAULT_CLUSTER_META,
            {
                "cluster_id": cluster_id,
                "size": len(cluster_records),
                "source_counts": dict(sorted(source_counts.items())),
                "mean_distance": float(np.mean([distances[i] for i in indices])),
                "min_distance": float(np.min([distances[i] for i in indices])),
                "max_distance": float(np.max([distances[i] for i in indices])),
            },
        )
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(cluster_records),
                "mean_distance": float(np.mean([distances[i] for i in indices])),
                **{f"source_{k}": v for k, v in sorted(source_counts.items())},
            }
        )

    with (root / "clusters.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in cluster_rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cluster_rows)
    np.save(root / "cluster_labels.npy", labels)
    np.save(root / "cluster_distances.npy", distances)
    np.save(root / "cluster_centers.npy", centers)
    print(f"[cluster] wrote {len(cluster_rows)} clusters -> {clusters_dir}")


def resolve_class_sample_size(config: Dict[str, Any]) -> int:
    value = config.get("kl_sample", {}).get("class_sample_size")
    if value is not None:
        return int(value)
    getslice_cfg = load_json_config(Path(config["getslice"]["base_config"]))
    return int(getslice_cfg.get("s_nsamples", 64))


def _discover_cluster_dirs(task_root: Path) -> List[str]:
    return sorted(
        path.name
        for path in task_root.iterdir()
        if path.is_dir() and (path.name.startswith("cluster_") or "__cluster_" in path.name)
    )


def _resolve_task_file_for_dirs(
    config: Dict[str, Any],
    task_root: Path,
    tasks: Sequence[str],
    *,
    fallback: str = "gsm8k.jsonl",
) -> str:
    configured = str(config.get("task_file", "dataset.jsonl"))
    if tasks and all((task_root / task / configured).exists() for task in tasks):
        return configured
    if tasks and all((task_root / task / fallback).exists() for task in tasks):
        print(f"[cluster_kl][WARN] using legacy task_file={fallback} under {task_root}")
        return fallback
    return configured


def _safe_token(value: Any) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)) or "unknown"


def _record_from_frame_row(global_id: int, row: Dict[str, Any]) -> Record:
    raw = dict(row)
    raw["question"] = str(row.get("problem") or row.get("question") or "").strip()
    raw["answer"] = str(row.get("solution") or row.get("answer") or "").strip()
    raw["source"] = str(row.get("source", "unknown"))
    raw["global_id"] = int(global_id)
    raw["source_line"] = int(global_id)
    return Record(
        global_id=int(global_id),
        source=str(row.get("source", "unknown")),
        source_line=int(global_id),
        question=raw["question"],
        answer=raw["answer"],
        raw=raw,
    )


def _load_base_records(config: Dict[str, Any]) -> List[Record]:
    base_dataset = config.get("base_dataset") or {}
    train_path = base_dataset.get("train_path")
    if not train_path:
        raise ValueError("offPolicyData cluster_kl 缺少 base_dataset.train_path")
    frame = load_train_frame(Path(str(train_path)).expanduser())
    rows = frame.to_dict(orient="records")
    return [_record_from_frame_row(idx, row) for idx, row in enumerate(rows)]


def _load_shared_cluster_cache(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cache_dir = config.get("cluster_cache_dir")
    if not cache_dir:
        raise ValueError("offPolicyData cluster_kl 需要 cluster_cache_dir，请先由 baseline diversity 生成共享聚类。")
    return load_diversity_cluster_cache(Path(str(cache_dir)).expanduser())


def _allocate_counts(counts: Dict[str, int], total: int) -> Dict[str, int]:
    counts = {str(k): int(v) for k, v in counts.items() if int(v) > 0}
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"请求样本数超过可用样本数: requested={total}, available={available}")
    raw = {key: total * value / available for key, value in counts.items()}
    allocation = {key: int(math.floor(value)) for key, value in raw.items()}
    assigned = sum(allocation.values())
    order = sorted(counts, key=lambda key: (raw[key] - math.floor(raw[key]), counts[key], key), reverse=True)
    idx = 0
    while assigned < total:
        key = order[idx % len(order)]
        if allocation[key] < counts[key]:
            allocation[key] += 1
            assigned += 1
        idx += 1
    return {key: value for key, value in allocation.items() if value > 0}


def stage_sample(config: Dict[str, Any], root: Path) -> None:
    records = _load_base_records(config)
    record_by_pos = {record.global_id: record for record in records}
    cluster_cache = _load_shared_cluster_cache(config)
    clusters_root = root / "clusters"
    sample_root = root / "kl_samples"
    clusters_index = root / "clusters_index.json"
    if (
        not bool(config.get("overwrite_outputs", False))
        and clusters_index.exists()
        and clusters_root.exists()
        and sample_root.exists()
        and any(sample_root.iterdir())
    ):
        print(f"[sample] reuse existing KL sample datasets -> {sample_root}")
        return
    ensure_dir_for_stage(clusters_root, overwrite=bool(config.get("overwrite_outputs", False)))
    ensure_dir_for_stage(sample_root, overwrite=bool(config.get("overwrite_outputs", False)))
    sample_size = resolve_class_sample_size(config)
    center_fraction = float(config.get("kl_sample", {}).get("center_fraction", 0.5))
    rng = random.Random(int(config.get("seed", 42)))
    dataset_alias = config.get("dataset_alias", "gsm8k_math_train")
    task_file = config.get("task_file", "gsm8k.jsonl")
    train_file = config.get("train_file", "gsm8k-train.jsonl")
    rows = []
    index_rows = []

    for source, cached in sorted(cluster_cache.items()):
        positions = cached["positions"].astype(np.int64)
        labels = cached["labels"].astype(np.int32)
        distances = cached["distances"].astype(np.float32)
        source_token = _safe_token(source)
        for cid in sorted(set(int(x) for x in labels)):
            cluster_id = f"{source_token}__cluster_{cid:04d}"
            local_indices = np.where(labels == cid)[0].tolist()
            local_indices.sort(key=lambda i: (float(distances[i]), int(positions[i])))
            cluster_records = [record_by_pos[int(positions[i])] for i in local_indices]
            extra_all = {
                record_by_pos[int(positions[i])].global_id: {
                    "cluster_id": cluster_id,
                    "cluster_source": str(source),
                    "cluster_distance": float(distances[i]),
                }
                for i in local_indices
            }
            write_dataset_dir(clusters_root / cluster_id, cluster_records, dataset_alias, task_file, train_file, extra_all)

            take = min(sample_size, len(local_indices))
            center_take = min(take, max(0, int(round(take * center_fraction))))
            selected = local_indices[:center_take]
            remaining = local_indices[center_take:]
            rng.shuffle(remaining)
            selected.extend(remaining[: take - center_take])
            selected_records = [record_by_pos[int(positions[i])] for i in selected]
            extra = {
                record_by_pos[int(positions[i])].global_id: {
                    "cluster_id": cluster_id,
                    "cluster_source": str(source),
                    "kl_sample": True,
                    "cluster_distance": float(distances[i]),
                }
                for i in selected
            }
            write_dataset_dir(sample_root / cluster_id, selected_records, dataset_alias, task_file, train_file, extra)
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "source": str(source),
                    "cluster_size": len(local_indices),
                    "sample_size": len(selected_records),
                    "center_selected": center_take,
                    "random_selected": len(selected_records) - center_take,
                }
            )
            index_rows.append(
                {
                    "cluster_id": cluster_id,
                    "source": str(source),
                    "cluster_size": len(local_indices),
                    "cluster_dir": str(clusters_root / cluster_id),
                    "sample_dir": str(sample_root / cluster_id),
                }
            )

    with (root / "kl_samples.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cluster_id", "source", "cluster_size", "sample_size", "center_selected", "random_selected"])
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        root / "clusters_index.json",
        {
            "cluster_cache_dir": str(config.get("cluster_cache_dir")),
            "base_dataset": config.get("base_dataset"),
            "clusters": index_rows,
            "source_cluster_counts": dict(sorted(Counter(row["source"] for row in index_rows).items())),
            "source_record_counts": dict(sorted(
                (source, int(sum(row["cluster_size"] for row in index_rows if row["source"] == source)))
                for source in {row["source"] for row in index_rows}
            )),
        },
    )
    print(f"[sample] wrote KL sample datasets -> {sample_root}")


def stage_getslice_config(config: Dict[str, Any], root: Path) -> Path:
    sample_root = root / "kl_samples"
    tasks = _discover_cluster_dirs(sample_root)
    cfg = dict(config)
    cfg["task_file"] = _resolve_task_file_for_dirs(config, sample_root, tasks)
    return write_getslice_config(cfg, root, sample_root, tasks)


def stage_run_getslice(config: Dict[str, Any], root: Path) -> None:
    if not (root / config["getslice"].get("config_out", "getslice_config.json")).exists():
        stage_getslice_config(config, root)
    run_getslice(config, root)


def stage_kl(config: Dict[str, Any], root: Path) -> None:
    csv_name = config.get("kl", {}).get("csv_name", "cluster_kl.csv")
    stage_kl_from_smat(
        config=config,
        root=root,
        unit_id_key="cluster_id",
        unit_prefix="cluster",
        csv_name=csv_name,
        modules_csv_name="cluster_kl_modules.csv",
    )


RankedRecordItem = Tuple[Record, Dict[str, Any]]


def _ranked_record_items(
    rows: Sequence[Dict[str, Any]],
    records_by_cluster: Dict[str, List[Record]],
) -> List[RankedRecordItem]:
    items: List[RankedRecordItem] = []
    for row in rows:
        cluster_id = row["cluster_id"]
        for record in records_by_cluster[cluster_id]:
            items.append((record, row))
    return items


def _split_ranked_record_items_into_bands(items: List[RankedRecordItem]) -> Dict[str, List[RankedRecordItem]]:
    if not items:
        return {"high": [], "mid": [], "low": []}
    parts = np.array_split(np.arange(len(items)), 3)
    band_items = {
        "high": [items[int(idx)] for idx in parts[0]],
        "mid": [items[int(idx)] for idx in parts[1]],
        "low": [items[int(idx)] for idx in parts[2]],
    }
    for band, band_values in list(band_items.items()):
        if not band_values:
            band_items[band] = list(items)
    return band_items


def _take_records_from_ranked_items(
    *,
    items: Sequence[RankedRecordItem],
    records_by_cluster: Dict[str, List[Record]],
    quota: int,
) -> Tuple[List[Record], List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    if len(items) < quota:
        raise ValueError(f"band capacity 不足: selected={len(items)}, requested={quota}")
    selected_items = list(items[:quota])
    selected = [record for record, _ in selected_items]
    used_by_cluster: Counter[str] = Counter()
    rank_rows_by_record: Dict[int, Dict[str, Any]] = {}
    rank_rows_by_cluster: Dict[str, Dict[str, Any]] = {}
    for record, row in selected_items:
        cluster_id = row["cluster_id"]
        used_by_cluster[cluster_id] += 1
        rank_rows_by_record[record.global_id] = row
        rank_rows_by_cluster.setdefault(cluster_id, row)

    quota_rows: List[Dict[str, Any]] = []
    for cluster_id, used_count in used_by_cluster.items():
        row = rank_rows_by_cluster[cluster_id]
        quota_rows.append({
            "cluster_id": cluster_id,
            "kl_rank": int(row["kl_rank"]),
            "rank_metric": row.get("rank_metric", ""),
            "rank_score": float(row.get("rank_score", 0.0)),
            "cluster_size": len(records_by_cluster[cluster_id]),
            "quota": int(used_count),
            "used_count": int(used_count),
        })
    return selected, quota_rows, rank_rows_by_record


def stage_aggregate(config: Dict[str, Any], root: Path) -> None:
    csv_name = config.get("kl", {}).get("csv_name", "cluster_kl.csv")
    kl_rows = read_unit_kl(root, csv_name, unit_id_key="cluster_id", unit_prefix="cluster")
    clusters_dir = root / "clusters"
    cluster_tasks = [row["cluster_id"] for row in kl_rows]
    input_task_file = _resolve_task_file_for_dirs(config, clusters_dir, cluster_tasks)
    output_task_file = config.get("task_file", "dataset.jsonl")
    train_file = config.get("train_file", "gsm8k-train.jsonl")
    dataset_alias = config.get("dataset_alias", "sft_train")
    agg_cfg = config.get("aggregation", {})
    dataset_size = int(agg_cfg.get("dataset_size", 0))
    if dataset_size <= 0:
        raise ValueError("aggregation.dataset_size 必须 > 0")
    bands = list(agg_cfg.get("bands") or ["high", "mid", "low"])
    if bands != ["high", "mid", "low"]:
        raise ValueError("cluster_kl 当前固定只支持 aggregation.bands = ['high', 'mid', 'low']")
    ranking = resolve_ranking_config(config)
    if ranking["metric"] != "topk_KL" or ranking["direction"] != "desc":
        raise ValueError("cluster_kl high/mid/low 固定要求 ranking.metric=topk_KL 且 direction=desc")

    cluster_index = load_json_config(root / "clusters_index.json")
    source_by_cluster = {row["cluster_id"]: row["source"] for row in cluster_index.get("clusters", [])}
    records_by_cluster = {
        row["cluster_id"]: load_unit_records(clusters_dir / row["cluster_id"], input_task_file)
        for row in kl_rows
    }
    source_record_counts: Dict[str, int] = {}
    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in kl_rows:
        source = str(source_by_cluster.get(row["cluster_id"], "unknown"))
        rows_by_source.setdefault(source, []).append(row)
        source_record_counts[source] = source_record_counts.get(source, 0) + len(records_by_cluster[row["cluster_id"]])

    for source, rows in rows_by_source.items():
        rows.sort(key=lambda row: (float(row.get("rank_score", 0.0)), -int(row["kl_rank"])), reverse=True)
    band_items_by_source = {
        source: _split_ranked_record_items_into_bands(_ranked_record_items(rows, records_by_cluster))
        for source, rows in rows_by_source.items()
    }

    allocation = _allocate_counts(source_record_counts, dataset_size)
    size_label = f"size{dataset_size}"
    ranked_root = root / "ranked_datasets" / size_label
    if (
        not bool(config.get("overwrite_outputs", False))
        and all((ranked_root / f"kl_{band}" / output_task_file).exists() for band in bands)
    ):
        print(f"[aggregate] reuse ranked datasets for {size_label} -> {ranked_root}")
        return
    ensure_dir_for_stage(ranked_root, overwrite=bool(config.get("overwrite_outputs", False)))
    summary_rows = []

    for band in bands:
        selected: List[Record] = []
        extra_fields: Dict[int, Dict[str, Any]] = {}
        quota_rows: List[Dict[str, Any]] = []
        for source, quota in sorted(allocation.items()):
            band_items = band_items_by_source[source][band]
            band_capacity = len(band_items)
            if band_capacity < quota:
                raise ValueError(
                    f"KL {band} band source={source} capacity 不足: capacity={band_capacity}, quota={quota}. "
                    "请增大训练池、减小 DataSize，或调小 target_cluster_size。"
                )
            source_selected, source_quota_rows, rank_rows_by_record = _take_records_from_ranked_items(
                items=band_items,
                records_by_cluster=records_by_cluster,
                quota=quota,
            )
            selected.extend(source_selected)
            for quota_row in source_quota_rows:
                quota_row["dataset_id"] = f"kl_{band}"
                quota_row["band"] = band
                quota_row["source"] = source
                quota_rows.append(quota_row)
            for record in source_selected:
                cluster_id = str(record.raw.get("cluster_id", ""))
                rank_row = rank_rows_by_record.get(record.global_id, {})
                extra_fields[record.global_id] = {
                    "offpolicy_dataset_id": f"kl_{band}",
                    "kl_band": band,
                    "cluster_id": cluster_id,
                    "cluster_kl_rank": int(rank_row.get("kl_rank", -1)),
                    "rank_metric": rank_row.get("rank_metric", ranking["metric"]),
                    "rank_score": float(rank_row.get("rank_score", 0.0)),
                }

        if len(selected) != dataset_size:
            raise RuntimeError(f"kl_{band} selected {len(selected)} records, expected {dataset_size}")
        dataset_dir = ranked_root / f"kl_{band}"
        write_dataset_dir(dataset_dir, selected, dataset_alias, output_task_file, train_file, extra_fields)

        source_stats = Counter(record.source for record in selected)
        with (dataset_dir / "source_stats.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset_id", "source", "count", "ratio"])
            writer.writeheader()
            for source, count in sorted(source_stats.items()):
                writer.writerow({"dataset_id": f"kl_{band}", "source": source, "count": count, "ratio": count / dataset_size})
        with (dataset_dir / "cluster_quota.csv").open("w", encoding="utf-8", newline="") as f:
            fieldnames = ["dataset_id", "band", "source", "cluster_id", "kl_rank", "rank_metric", "rank_score", "cluster_size", "quota", "used_count"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(quota_rows)
        score_values = [float(row["rank_score"]) for row in quota_rows for _ in range(int(row["used_count"]))]
        save_json(
            dataset_dir / "ranking_meta.json",
            {
                "dataset_id": f"kl_{band}",
                "band": band,
                "dataset_size": dataset_size,
                "rank_metric": ranking["metric"],
                "rank_direction": ranking["direction"],
                "ranking_top_k": ranking["top_k"],
                "source_allocation": allocation,
                "source_stats": dict(sorted(source_stats.items())),
                "score_mean": float(np.mean(score_values)) if score_values else 0.0,
                "score_min": float(np.min(score_values)) if score_values else 0.0,
                "score_max": float(np.max(score_values)) if score_values else 0.0,
            },
        )
        summary_rows.append(
            {
                "dataset_id": f"kl_{band}",
                "band": band,
                "rank_metric": ranking["metric"],
                "rank_direction": ranking["direction"],
                "ranking_top_k": ranking["top_k"],
                "dataset_size": dataset_size,
                "score_mean": float(np.mean(score_values)) if score_values else 0.0,
                "score_min": float(np.min(score_values)) if score_values else 0.0,
                "score_max": float(np.max(score_values)) if score_values else 0.0,
                "num_used_clusters": len([row for row in quota_rows if int(row["used_count"]) > 0]),
            }
        )
        print(f"[aggregate] wrote kl_{band}: {len(selected)} records -> {dataset_dir}")

    with (ranked_root / "ranked_dataset_kl.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["dataset_id", "band", "rank_metric", "rank_direction", "ranking_top_k", "dataset_size", "score_mean", "score_min", "score_max", "num_used_clusters"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[aggregate] wrote KL band summary -> {ranked_root / 'ranked_dataset_kl.csv'}")


STAGE_FUNCS = {
    "sample": stage_sample,
    "getslice_config": stage_getslice_config,
    "getslice": stage_run_getslice,
    "kl": stage_kl,
    "aggregate": stage_aggregate,
}

ALL_STAGES = ["sample", "getslice_config", "getslice", "kl", "aggregate"]
