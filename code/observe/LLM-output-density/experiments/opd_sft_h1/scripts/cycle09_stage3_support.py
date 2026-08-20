#!/usr/bin/env python3
"""T-SUPPORT statistics over an explicit frozen training-support manifest.

The manifest is mandatory. The script never searches arbitrary rollout files,
uses evaluation generations, or substitutes another arm/step. Reused offline
support is computed once and copied only when the manifest declares a shared
source group.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

import cycle09_stage3_followup_common as c


ROOT = c.scoped_run("H1_support")
OUTPUT = ROOT / "T_SUPPORT_stats.csv"
CLUSTERS = ROOT / "T_SUPPORT_near_duplicate_clusters.json"
MANIFEST = ROOT / "T_SUPPORT_manifest.json"
EOS_REASONS = {"stop", "eos", "stop_sequence", "data_eos"}
TRUNCATION_REASONS = {"length", "max_tokens", "max_length"}
_TOKENIZERS: dict[str, Any] = {}
_SOURCE_CACHE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
_METRIC_CACHE: dict[str, list[dict[str, Any]]] = {}


class HyperLogLog:
    """Small deterministic HLL for corpus-level distinct-n."""

    def __init__(self, precision: int = 16):
        self.precision = precision
        self.size = 1 << precision
        self.registers = bytearray(self.size)

    def add(self, value: bytes) -> None:
        digest = int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")
        index = digest >> (64 - self.precision)
        tail = (digest << self.precision) & ((1 << 64) - 1)
        rank = (64 - self.precision + 1) if tail == 0 else (64 - tail.bit_length() + 1)
        if rank > self.registers[index]:
            self.registers[index] = rank

    def estimate(self) -> float:
        size = float(self.size)
        alpha = 0.7213 / (1.0 + 1.079 / size)
        raw = alpha * size * size / sum(2.0 ** (-value) for value in self.registers)
        zeros = self.registers.count(0)
        if raw <= 2.5 * size and zeros:
            return size * math.log(size / zeros)
        return raw


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def lexical_tokens(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_text(value), flags=re.UNICODE)


def tokenizer(path: str) -> Any:
    if path not in _TOKENIZERS:
        from transformers import AutoTokenizer

        _TOKENIZERS[path] = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _TOKENIZERS[path]


def records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def response_text(row: dict[str, Any], cell: dict[str, Any]) -> str:
    requested = cell.get("response_field")
    fields = (requested,) if requested else (
        "output",
        "response",
        "generation",
        "generation_text",
        "completion",
        "text",
    )
    value = next((row.get(key) for key in fields if key and isinstance(row.get(key), str)), None)
    if value is None:
        token_field = str(cell.get("token_ids_field", "generation_token_ids"))
        token_ids = row.get(token_field)
        tokenizer_path = cell.get("tokenizer_path")
        if isinstance(token_ids, list) and tokenizer_path:
            value = tokenizer(str(tokenizer_path)).decode(token_ids, skip_special_tokens=False)
        elif isinstance(token_ids, list):
            value = " ".join(map(str, token_ids))
        else:
            raise KeyError("no recognized response text or token-id field")
    after = cell.get("text_after")
    if after:
        if str(after) not in value:
            raise RuntimeError(f"text_after delimiter absent in {cell['path']}: {after!r}")
        value = value.split(str(after), 1)[1]
    before = cell.get("text_before")
    if before and str(before) in value:
        value = value.split(str(before), 1)[0]
    return value


def response_tokens(row: dict[str, Any], text: str, cell: dict[str, Any]) -> list[str]:
    token_field = str(cell.get("token_ids_field", "generation_token_ids"))
    values = row.get(token_field)
    if isinstance(values, list):
        return [str(int(value)) for value in values]
    tokenizer_path = cell.get("tokenizer_path")
    if tokenizer_path:
        values = tokenizer(str(tokenizer_path)).encode(text, add_special_tokens=False)
        return [str(int(value)) for value in values]
    return lexical_tokens(text)


def eligible(row: dict[str, Any], cell: dict[str, Any]) -> bool:
    cap = cell.get("max_prompt_tokens")
    if cap is not None and row.get("n_prompt_tokens") is not None:
        return int(row["n_prompt_tokens"]) <= int(cap)
    return True


def encoded_ngram(items: list[str], index: int, n: int) -> bytes:
    return "\x1f".join(items[index : index + n]).encode("utf-8")


def simhash(items: list[str], feature_cap: int = 2048) -> int:
    count = max(1, len(items) - 3)
    indices: Iterable[int]
    if count <= feature_cap:
        indices = range(count)
    else:
        stride = count / feature_cap
        indices = (min(int(index * stride), count - 1) for index in range(feature_cap))
    votes = [0] * 64
    observed = 0
    for index in indices:
        feature = encoded_ngram(items, index, 4) if len(items) >= 4 else "\x1f".join(items).encode()
        digest = int.from_bytes(hashlib.blake2b(feature, digest_size=8).digest(), "big")
        observed += 1
        for bit in range(64):
            votes[bit] += 1 if digest & (1 << bit) else -1
    return sum((1 << bit) for bit, vote in enumerate(votes) if vote >= 0) if observed else 0


def near_duplicate_summary(normalized: list[str], hashes: list[int]) -> dict[str, Any]:
    parent = list(range(len(normalized)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    groups: dict[str, list[int]] = collections.defaultdict(list)
    for index, key in enumerate(normalized):
        groups[key].append(index)
    representatives: list[int] = []
    for group in groups.values():
        representatives.append(group[0])
        for index in group[1:]:
            union(group[0], index)

    buckets: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for index in representatives:
        for band in range(4):
            buckets[(band, (hashes[index] >> (16 * band)) & 0xFFFF)].append(index)
    compared: set[tuple[int, int]] = set()
    capped = 0
    for bucket in buckets.values():
        if len(bucket) > 200:
            capped += 1
            bucket = sorted(bucket, key=hashes.__getitem__)
            pairs = (
                (bucket[left], bucket[right])
                for left in range(len(bucket))
                for right in range(left + 1, min(left + 9, len(bucket)))
            )
        else:
            pairs = (
                (bucket[left], bucket[right])
                for left in range(len(bucket))
                for right in range(left + 1, len(bucket))
            )
        for left, right in pairs:
            pair = (min(left, right), max(left, right))
            if pair in compared:
                continue
            compared.add(pair)
            if (hashes[left] ^ hashes[right]).bit_count() <= 3:
                union(left, right)

    sizes = sorted(collections.Counter(find(index) for index in range(len(normalized))).values())
    probabilities = [size / len(normalized) for size in sizes]
    cluster_entropy = -sum(value * math.log(value) for value in probabilities if value)
    return {
        "method": "64-bit response-token 4-gram SimHash; 4x16-bit LSH; Hamming<=3",
        "simhash_feature_cap": 2048,
        "large_bucket_policy": "sorted-neighbor window=8 when bucket>200",
        "large_buckets_capped": capped,
        "candidate_pairs": len(compared),
        "cluster_count": len(sizes),
        "singleton_clusters": sum(size == 1 for size in sizes),
        "cluster_size_mean": statistics.fmean(sizes),
        "cluster_size_median": statistics.median(sizes),
        "cluster_size_p90": sizes[min(len(sizes) - 1, math.ceil(0.90 * len(sizes)) - 1)],
        "cluster_size_p95": sizes[min(len(sizes) - 1, math.ceil(0.95 * len(sizes)) - 1)],
        "cluster_size_max": max(sizes),
        "cluster_entropy": cluster_entropy,
        "effective_support_size": math.exp(cluster_entropy),
        "cluster_size_histogram": dict(sorted(collections.Counter(sizes).items())),
    }


def source_cache_key(cell: dict[str, Any]) -> str:
    fields = (
        "path",
        "response_field",
        "token_ids_field",
        "tokenizer_path",
        "text_after",
        "text_before",
        "max_prompt_tokens",
        "max_response_tokens",
    )
    return json.dumps({key: cell.get(key) for key in fields}, sort_keys=True)


def source_stats(cell: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    key = source_cache_key(cell)
    if key in _SOURCE_CACHE:
        return _SOURCE_CACHE[key]
    path = Path(cell["path"])
    rows = [row for row in records(path) if eligible(row, cell)]
    if not rows:
        raise RuntimeError(f"empty eligible support cell {path}")

    lengths: list[int] = []
    text_hashes: list[str] = []
    normalized_hashes: list[str] = []
    simhashes: list[int] = []
    finish_reasons: list[str] = []
    token_counts: collections.Counter[str] = collections.Counter()
    hll = {2: HyperLogLog(), 4: HyperLogLog()}
    total_ngrams = {2: 0, 4: 0}
    row_kl: list[float] = []
    row_loss: list[float] = []
    boxed = 0
    total_chars = 0

    for row in rows:
        text = response_text(row, cell)
        tokens = response_tokens(row, text, cell)
        length = int(row.get("response_token_length", row.get("n_tokens", len(tokens))))
        lengths.append(length)
        total_chars += len(text)
        boxed += int("\\boxed{" in text)
        finish_reasons.append(str(row.get("finish_reason", "")).lower())
        exact = hashlib.sha256(text.encode("utf-8")).hexdigest()
        normalized = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
        text_hashes.append(exact)
        normalized_hashes.append(normalized)
        simhashes.append(simhash(tokens))
        token_counts.update(tokens)
        for n in (2, 4):
            count = max(0, len(tokens) - n + 1)
            total_ngrams[n] += count
            for index in range(count):
                hll[n].add(encoded_ngram(tokens, index, n))
        for field, target in (("source_kl", row_kl), ("source_loss", row_loss)):
            try:
                target.append(float(row[field]))
            except (KeyError, TypeError, ValueError):
                pass

    token_total = sum(token_counts.values())
    token_entropy = -sum(
        (count / token_total) * math.log(count / token_total)
        for count in token_counts.values()
        if count
    )
    sequence_counts = collections.Counter(normalized_hashes)
    sequence_entropy = -sum(
        (count / len(rows)) * math.log(count / len(rows))
        for count in sequence_counts.values()
        if count
    )
    clusters = near_duplicate_summary(normalized_hashes, simhashes)
    cap = cell.get("max_response_tokens")
    truncation = [
        reason in TRUNCATION_REASONS or (cap is not None and length >= int(cap))
        for reason, length in zip(finish_reasons, lengths, strict=True)
    ]
    stats = {
        "n_sequences": len(rows),
        "response_tokens_total": sum(lengths),
        "response_tokens_mean": statistics.fmean(lengths),
        "response_tokens_median": statistics.median(lengths),
        "response_tokens_p90": sorted(lengths)[min(len(lengths) - 1, math.ceil(0.90 * len(lengths)) - 1)],
        "response_tokens_max": max(lengths),
        "response_chars_mean": total_chars / len(rows),
        "eos_rate": sum(reason in EOS_REASONS for reason in finish_reasons) / len(rows),
        "truncation_rate": sum(truncation) / len(rows),
        "cap_hit_rate": sum(bool(value) for value in truncation) / len(rows),
        "boxed_rate": boxed / len(rows),
        "exact_duplicate_rate": 1 - len(set(text_hashes)) / len(rows),
        "normalized_duplicate_rate": 1 - len(sequence_counts) / len(rows),
        "near_duplicate_cluster_rate": 1 - clusters["cluster_count"] / len(rows),
        "distinct_2": hll[2].estimate() / max(1, total_ngrams[2]),
        "distinct_4": hll[4].estimate() / max(1, total_ngrams[4]),
        "fourgram_repetition": 1 - hll[4].estimate() / max(1, total_ngrams[4]),
        "token_entropy": token_entropy,
        "sequence_entropy": sequence_entropy,
        "effective_exact_support_size": math.exp(sequence_entropy),
        "source_kl_mean": statistics.fmean(row_kl) if row_kl else math.nan,
        "source_loss_mean": statistics.fmean(row_loss) if row_loss else math.nan,
        "source_path": str(path),
    }
    _SOURCE_CACHE[key] = (stats, clusters)
    return stats, clusters


def metric_rows(path: Path) -> list[dict[str, Any]]:
    key = str(path)
    if key in _METRIC_CACHE:
        return _METRIC_CACHE[key]
    if path.suffix == ".csv":
        rows = pd.read_csv(path).to_dict("records")
    else:
        rows = records(path)
    _METRIC_CACHE[key] = rows
    return rows


def training_metric(cell: dict[str, Any]) -> tuple[float, str]:
    path_value = cell.get("metrics_path")
    if not path_value:
        return math.nan, ""
    path = Path(str(path_value))
    rows = [row for row in metric_rows(path) if int(row.get("step", -1)) == int(cell["step"])]
    if len(rows) != 1:
        return math.nan, ""
    row = rows[0]
    for key in ("actor/distillation/loss", "actor/loss", "loss", "source_kl", "source_loss"):
        try:
            return float(row[key]), key
        except (KeyError, TypeError, ValueError):
            continue
    return math.nan, ""


def cell_stats(cell: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stats, clusters = source_stats(cell)
    metric, metric_key = training_metric(cell)
    optional = {
        key: cell.get(key, "")
        for key in ("support_kind", "shared_source_group", "notes", "objective_kind")
    }
    row = {
        "family": cell["family"],
        "arm": cell["arm"],
        "step": int(cell["step"]),
        **optional,
        **stats,
        "training_objective_at_step": metric,
        "training_objective_key": metric_key,
        "metrics_path": str(cell.get("metrics_path", "")),
    }
    return row, clusters


def run(input_manifest: Path) -> dict[str, Any]:
    payload = c.read_json(input_manifest, {})
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError("support input manifest requires nonempty cells[]")
    required = {"family", "arm", "step", "path"}
    if any(required.difference(cell) for cell in cells):
        raise RuntimeError("every support cell needs family, arm, step, path")
    missing_paths = [cell["path"] for cell in cells if not Path(cell["path"]).is_file()]
    if missing_paths:
        raise FileNotFoundError(f"frozen support inputs missing: {missing_paths}")

    rows: list[dict[str, Any]] = []
    cluster_sources: dict[str, Any] = {}
    for cell in cells:
        row, clusters = cell_stats(cell)
        rows.append(row)
        cluster_sources[source_cache_key(cell)] = {
            "source_path": cell["path"],
            "shared_source_group": cell.get("shared_source_group"),
            **clusters,
        }

    groups: dict[str, int] = collections.Counter()
    for row, cell in zip(rows, cells, strict=True):
        group = cell.get("token_share_group")
        if group:
            groups[str(group)] += int(row["response_tokens_total"])
    for row, cell in zip(rows, cells, strict=True):
        group = cell.get("token_share_group")
        row["token_share_group"] = str(group or "")
        row["response_token_share"] = (
            row["response_tokens_total"] / groups[str(group)] if group else math.nan
        )

    c.atomic_csv(OUTPUT, rows)
    c.atomic_json(
        CLUSTERS,
        {
            "schema_version": 1,
            "status": "complete",
            "sources": list(cluster_sources.values()),
            "created_utc": c.utc_now(),
        },
    )
    status = "complete_with_declared_missing_cells" if payload.get("missing_cells") else "complete"
    llama_full = c.MINI / "llama_opd_support_stats.csv"
    alpha_sources = c.MINI / "qwen_alpha05_stage_b_support_stats.csv"
    result = {
        "schema_version": 2,
        "status": status,
        "task": "T-SUPPORT frozen training-support audit",
        "input_manifest": c.artifact(input_manifest),
        "output": c.artifact(OUTPUT),
        "near_duplicate_clusters": c.artifact(CLUSTERS),
        "cells": len(rows),
        "unique_physical_sources": len(_SOURCE_CACHE),
        "missing_cells": payload.get("missing_cells", []),
        "shared_source_groups": payload.get("shared_source_groups", {}),
        "llama_opd_full_support": c.artifact(llama_full),
        "alpha05_source_separated_support": c.artifact(alpha_sources),
        "prohibitions": [
            "no evaluation generations used as training support",
            "no silent arm or checkpoint substitution",
            "declared shared offline sources are not independent samples",
        ],
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("run",), required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().input_manifest), indent=2))
