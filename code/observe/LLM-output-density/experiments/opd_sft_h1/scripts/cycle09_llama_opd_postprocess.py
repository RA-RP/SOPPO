#!/usr/bin/env python3
"""Validate L1 rollout provenance and derive support/training statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import cycle09_block3_common as c


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
FLOAT = re.compile(r"^(?:np\.(?:float|int)\d*\()?([^()]*)\)?$")
METRIC_KEYS = (
    "actor/distillation/student_mass",
    "actor/distillation/teacher_mass",
    "actor/distillation/loss",
    "actor/loss",
    "actor/grad_norm",
    "actor/lr",
    "response_length/mean",
    "response_length/max",
    "response_length/min",
    "response_length/clip_ratio",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/update_actor",
    "timing_s/update_weights",
    "timing_s/save_checkpoint",
    "timing_s/step",
)


class HyperLogLog:
    """Small deterministic HLL used only for corpus-level distinct-n."""

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
        m = float(self.size)
        alpha = 0.7213 / (1.0 + 1.079 / m)
        raw = alpha * m * m / sum(2.0 ** (-register) for register in self.registers)
        zeros = self.registers.count(0)
        if raw <= 2.5 * m and zeros:
            return m * math.log(m / zeros)
        return raw


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).lower().split())


def tokens(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", normalize_text(value), flags=re.UNICODE)


def ngrams(items: list[str], n: int) -> Iterable[bytes]:
    for index in range(max(0, len(items) - n + 1)):
        yield "\x1f".join(items[index : index + n]).encode("utf-8")


def parse_number(value: str) -> float | None:
    value = value.strip().rstrip(",")
    match = FLOAT.match(value)
    if match:
        value = match.group(1)
    try:
        return float(value)
    except ValueError:
        return None


def parse_training_log(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    metrics: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI.sub("", raw).replace("\r", "")
            match = re.search(r"\bstep:(\d+)\s+-\s+", line)
            if not match:
                continue
            step = int(match.group(1))
            values: dict[str, float] = {}
            for part in line[match.end() :].split(" - "):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = key.strip()
                if key not in METRIC_KEYS:
                    continue
                parsed = parse_number(value)
                if parsed is not None:
                    values[key] = parsed
            if values:
                metrics[step] = values
    return metrics


def prompt_contract() -> tuple[dict[int, dict[str, Any]], list[int]]:
    rows = c.read_jsonl(c.L1_DATA / "llama_opd_prompt_map.jsonl")
    by_id = {int(row["prompt_id"]): row for row in rows}
    order = [int(row["prompt_id"]) for row in sorted(rows, key=lambda row: row["eligible_order"])]
    if len(rows) != 4999 or len(by_id) != len(rows):
        raise RuntimeError(f"prompt map contract drift: rows={len(rows)} unique={len(by_id)}")
    return by_id, order


def expected_prompt_ids(order: list[int], step: int) -> list[int]:
    per_epoch = len(order) // c.TRAIN_BATCH_SIZE
    offset = ((step - 1) % per_epoch) * c.TRAIN_BATCH_SIZE
    return order[offset : offset + c.TRAIN_BATCH_SIZE]


def raw_path(root: Path, step: int) -> Path:
    candidates = (root / f"{step}.jsonl", root / f"step_{step}.jsonl")
    return next((path for path in candidates if path.is_file()), candidates[0])


def validate_rollouts(
    *,
    raw_root: Path,
    total_steps: int,
    strict: bool,
    allowed_rollout_gaps: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    prompt_by_id, prompt_order = prompt_contract()
    all_rows: list[dict[str, Any]] = []
    inventory = []
    missing_steps = []
    for step in range(1, total_steps + 1):
        path = raw_path(raw_root, step)
        if not path.is_file():
            if step in allowed_rollout_gaps:
                missing_steps.append(step)
                inventory.append(
                    {
                        "step": step,
                        "path": str(path),
                        "status": "missing_terminal_rollout",
                    }
                )
                continue
            raise FileNotFoundError(path)
        rows = c.read_jsonl(path)
        if len(rows) != c.TRAIN_BATCH_SIZE:
            raise RuntimeError(f"rollout rows at step {step}: {len(rows)} != 16")
        expected = expected_prompt_ids(prompt_order, step)
        observed = []
        for batch_position, row in enumerate(rows):
            missing = [
                key
                for key in ("input", "output", "step", "response_token_length", "finish_reason")
                if key not in row
            ]
            if missing and strict:
                raise RuntimeError(f"rollout audit fields missing at step={step}: {missing}")
            prompt_id_recorded = "prompt_id" in row and row["prompt_id"] is not None
            prompt_id = int(row["prompt_id"]) if prompt_id_recorded else expected[batch_position]
            if prompt_id not in prompt_by_id:
                raise RuntimeError(f"unknown prompt id {prompt_id} at step={step}")
            observed.append(prompt_id)
            response_length = int(row.get("response_token_length", 0))
            if response_length <= 0 or response_length > c.MAX_RESPONSE_TOKENS:
                raise RuntimeError(f"invalid response token length at step={step}: {response_length}")
            output = str(row["output"])
            canonical = {
                "step": step,
                "epoch": 1 + (step - 1) // (len(prompt_order) // c.TRAIN_BATCH_SIZE),
                "batch_position": batch_position,
                "prompt_id": prompt_id,
                "prompt_id_provenance": (
                    "rollout_dump" if prompt_id_recorded else "deterministic_no_shuffle_batch_order"
                ),
                "source_row": int(prompt_by_id[prompt_id]["source_row"]),
                "eligible_order": int(prompt_by_id[prompt_id]["eligible_order"]),
                "prompt_text": str(row["input"]),
                "rollout_text": output,
                "response_token_length": response_length,
                "finish_reason": str(row.get("finish_reason", "unknown")),
                "cap_hit": response_length >= c.MAX_RESPONSE_TOKENS,
                "eos": str(row.get("finish_reason", "")) == "eos",
                "has_boxed": "\\boxed" in output,
            }
            all_rows.append(canonical)
        if strict and observed != expected:
            raise RuntimeError(
                f"prompt order mismatch at step={step}: observed={observed} expected={expected}"
            )
        inventory.append(
            {
                "step": step,
                "path": str(path),
                "rows": len(rows),
                "bytes": path.stat().st_size,
                "sha256": c.sha256_file(path),
            }
        )
    return all_rows, inventory, missing_steps


def simhash(value: str, feature_cap: int = 2048) -> int:
    items = tokens(value)
    features = list(ngrams(items, 4)) or [normalize_text(value).encode("utf-8")]
    if len(features) > feature_cap:
        stride = len(features) / feature_cap
        features = [features[min(int(index * stride), len(features) - 1)] for index in range(feature_cap)]
    packed = b"".join(hashlib.blake2b(feature, digest_size=8).digest() for feature in features)
    bit_matrix = np.unpackbits(
        np.frombuffer(packed, dtype=np.uint8).reshape(-1, 8),
        axis=1,
        bitorder="big",
    )
    positive = bit_matrix.sum(axis=0) * 2 >= len(features)
    result = 0
    for bit, selected in enumerate(positive):
        if selected:
            result |= 1 << bit
    return result


def near_duplicate_summary(texts: list[str]) -> dict[str, Any]:
    parent = list(range(len(texts)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    normalized_groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(texts):
        normalized_groups[normalize_text(value)].append(index)
    representatives = []
    for group in normalized_groups.values():
        representatives.append(group[0])
        for index in group[1:]:
            union(group[0], index)

    hashes = {index: simhash(texts[index]) for index in representatives}
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, value in hashes.items():
        for band in range(4):
            buckets[(band, (value >> (band * 16)) & 0xFFFF)].append(index)
    compared: set[tuple[int, int]] = set()
    capped_buckets = 0
    for bucket in buckets.values():
        if len(bucket) > 200:
            capped_buckets += 1
            bucket = sorted(bucket, key=hashes.get)
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
    sizes = sorted(Counter(find(index) for index in range(len(texts))).values())
    probabilities = [size / len(texts) for size in sizes]
    entropy = -sum(value * math.log(value) for value in probabilities if value)
    return {
        "near_duplicate_method": "64-bit token-4gram SimHash; 4x16-bit LSH; Hamming<=3",
        "simhash_feature_cap": 2048,
        "large_bucket_policy": "compare sorted-neighbor window=8 when bucket>200",
        "large_buckets_capped": capped_buckets,
        "candidate_pairs": len(compared),
        "cluster_count": len(sizes),
        "singleton_clusters": sum(size == 1 for size in sizes),
        "cluster_size_mean": statistics.fmean(sizes),
        "cluster_size_median": statistics.median(sizes),
        "cluster_size_p90": sorted(sizes)[min(len(sizes) - 1, math.ceil(0.90 * len(sizes)) - 1)],
        "cluster_size_p95": sorted(sizes)[min(len(sizes) - 1, math.ceil(0.95 * len(sizes)) - 1)],
        "cluster_size_max": max(sizes),
        "cluster_entropy": entropy,
        "effective_support_size": math.exp(entropy),
        "cluster_size_histogram": dict(sorted(Counter(sizes).items())),
    }


def support_row(scope: str, rows: list[dict[str, Any]], *, exact_ngrams: bool) -> dict[str, Any]:
    texts = [str(row["rollout_text"]) for row in rows]
    normalized = [normalize_text(value) for value in texts]
    unique_exact = len(set(texts))
    unique_normalized = len(set(normalized))
    total_ngrams = {2: 0, 4: 0}
    accumulators: dict[int, set[bytes] | HyperLogLog] = {
        n: set() if exact_ngrams else HyperLogLog() for n in (2, 4)
    }
    for value in texts:
        tokenized = tokens(value)
        for n in (2, 4):
            values = list(ngrams(tokenized, n))
            total_ngrams[n] += len(values)
            accumulator = accumulators[n]
            if isinstance(accumulator, set):
                accumulator.update(values)
            else:
                for item in values:
                    accumulator.add(item)
    unique_counts = {
        n: float(len(accumulator))
        if isinstance(accumulator, set)
        else accumulator.estimate()
        for n, accumulator in accumulators.items()
    }
    lengths = [int(row["response_token_length"]) for row in rows]
    return {
        "scope": scope,
        "n_responses": len(rows),
        "exact_duplicate_rate": (len(rows) - unique_exact) / len(rows),
        "normalized_duplicate_rate": (len(rows) - unique_normalized) / len(rows),
        "distinct_2": unique_counts[2] / max(total_ngrams[2], 1),
        "distinct_4": unique_counts[4] / max(total_ngrams[4], 1),
        "fourgram_repetition": 1.0 - unique_counts[4] / max(total_ngrams[4], 1),
        "distinct_method": "exact" if exact_ngrams else "HyperLogLog(p=16)",
        "response_length_mean": statistics.fmean(lengths),
        "response_length_median": statistics.median(lengths),
        "response_length_p90": sorted(lengths)[min(len(lengths) - 1, math.ceil(0.90 * len(lengths)) - 1)],
        "response_length_max": max(lengths),
        "cap_hit_rate": sum(bool(row["cap_hit"]) for row in rows) / len(rows),
        "eos_rate": sum(bool(row["eos"]) for row in rows) / len(rows),
        "boxed_rate": sum(bool(row["has_boxed"]) for row in rows) / len(rows),
    }


def build_support(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output = []
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)
    for step in sorted(by_step):
        output.append(support_row(f"step:{step}", by_step[step], exact_ngrams=True))
    output.append(support_row("overall", rows, exact_ngrams=False))
    return output, near_duplicate_summary([str(row["rollout_text"]) for row in rows])


def training_metric_rows(
    metrics: dict[int, dict[str, float]], total_steps: int, *, allowed_rollout_gaps: set[int]
) -> tuple[list[dict[str, Any]], list[int]]:
    missing = [step for step in range(1, total_steps + 1) if step not in metrics]
    if missing and not set(missing).issubset(allowed_rollout_gaps):
        raise RuntimeError(f"training log is missing metric rows: {missing[:20]}")
    rows = []
    for step in range(1, total_steps + 1):
        if step in missing:
            continue
        source = metrics[step]
        rows.append({"step": step, **{key: source.get(key, "") for key in METRIC_KEYS}})
    return rows, missing


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = c.L1_ROOT / "smoke" if args.smoke else c.L1_ROOT
    raw_root = root / "rollouts/raw"
    canonical_root = root / "rollouts/canonical"
    total_steps = args.total_steps or (2 if args.smoke else c.L1_FINAL_STEP)
    allowed_rollout_gaps = set(args.allow_rollout_gaps)
    if args.allow_terminal_rollout_gap:
        allowed_rollout_gaps.add(total_steps)
    rows, raw_inventory, missing_rollouts = validate_rollouts(
        raw_root=raw_root,
        total_steps=total_steps,
        strict=not args.allow_legacy_dump,
        allowed_rollout_gaps=allowed_rollout_gaps,
    )
    metrics = parse_training_log(args.log)
    metric_rows, missing_metrics = training_metric_rows(
        metrics,
        total_steps,
        allowed_rollout_gaps=allowed_rollout_gaps,
    )
    if missing_rollouts != missing_metrics:
        raise RuntimeError(
            "terminal rollout recovery requires matching raw and training-metric gaps: "
            f"raw={missing_rollouts} metrics={missing_metrics}"
        )
    terminal_gap = missing_rollouts

    canonical_root.mkdir(parents=True, exist_ok=True)
    canonical = canonical_root / "llama_opd_rollouts.jsonl"
    metrics_csv = canonical_root / "llama_opd_step_metrics.csv"
    support_csv = canonical_root / "llama_opd_support_stats.csv"
    cluster_json = canonical_root / "llama_opd_near_duplicate_clusters.json"
    inventory_csv = canonical_root / "llama_opd_checkpoint_inventory.csv"
    c.atomic_jsonl(canonical, rows)
    c.atomic_csv(metrics_csv, metric_rows)
    support_rows, cluster = build_support(rows)
    for row in support_rows:
        if row["scope"] == "overall":
            row.update({key: value for key, value in cluster.items() if key != "cluster_size_histogram"})
    support_fields = list(dict.fromkeys(key for row in support_rows for key in row))
    c.atomic_csv(support_csv, support_rows, support_fields)
    c.atomic_json(cluster_json, cluster)

    checkpoint_grid = tuple(step for step in c.TRAINING_CHECKPOINTS if step <= total_steps)
    checkpoints = c.checkpoint_inventory(checkpoint_grid) if not args.smoke else []
    if not args.smoke:
        missing = [row["step"] for row in checkpoints if not row["complete"]]
        if missing:
            raise RuntimeError(f"target checkpoints incomplete: {missing}")
        c.atomic_csv(inventory_csv, checkpoints)

    outputs = [canonical, metrics_csv, support_csv, cluster_json]
    if not args.smoke:
        outputs.append(inventory_csv)
    manifest = {
        "schema_version": 1,
        "status": "complete" if not terminal_gap else "complete_with_terminal_rollout_gap",
        "task": "Cycle09 block3 L1 Llama on-policy distillation",
        "created_utc": c.utc_now(),
        "mode": "smoke" if args.smoke else "formal",
        "student": str(c.LLAMA_STUDENT),
        "teacher": str(c.LLAMA_TEACHER),
        "prompt_manifest": c.artifact(c.L1_DATA / "prompt_manifest.json"),
        "training": {
            "seed": c.SEED,
            "shuffle": False,
            "batch_size": c.TRAIN_BATCH_SIZE,
            "epochs": 1 if args.smoke else 2,
            "steps": total_steps,
            "rollout": {"temperature": 0.6, "top_p": 0.9, "top_k": -1, "n": 1, "max_response_tokens": 10240},
            "distillation": "response-only token-mean forward KL; teacher raw top-32; no PG/reward",
            "lora": {"rank": 32, "alpha": 64, "targets": "all-linear"},
            "optimizer": {"name": "AdamW", "lr": 5e-5},
            "gpu_topology": "GPU0 student rollout+train; GPU1 teacher; disjoint verl pools",
            "save_frequency": 1 if args.smoke else 5,
            "registered_checkpoint_grid": list(checkpoint_grid),
            "full_two_epoch_schedule_steps": c.TOTAL_STEPS,
        },
        "rollout_rows": len(rows),
        "recorded_rollout_steps": sorted({int(row["step"]) for row in rows}),
        "recorded_training_metric_steps": [int(row["step"]) for row in metric_rows],
        "terminal_rollout_gap": (
            None
            if not terminal_gap
            else {
                "missing_steps": terminal_gap,
                "reason": "terminal rollout exceeded watchdog threshold after checkpoint persistence",
                "recovery_policy": "checkpoint retained; no synthetic rollout or metric rows were created",
            }
        ),
        "raw_rollout_inventory": raw_inventory,
        "training_log": c.artifact(args.log),
        "gpu_budget_ledger": c.artifact(c.BUDGET_LEDGER) if c.BUDGET_LEDGER.is_file() else None,
        "checkpoint_inventory": checkpoints,
        "support_protocol": {
            "normalization": "Unicode NFKC + lowercase + whitespace collapse",
            "near_duplicate": cluster["near_duplicate_method"],
            "distinct": "exact per batch; HyperLogLog(p=16) for overall corpus",
        },
        "outputs": [c.artifact(path) for path in outputs],
    }
    local_manifest = canonical_root / "llama_opd_training_manifest.json"
    c.atomic_json(local_manifest, manifest)
    if not args.smoke:
        c.atomic_csv(c.MINI / "llama_opd_support_stats.csv", support_rows, support_fields)
        c.atomic_csv(c.MINI / "llama_opd_checkpoint_inventory.csv", checkpoints)
        c.atomic_json(c.MINI / "llama_opd_training_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--total-steps", type=int, default=0)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--allow-legacy-dump", action="store_true")
    parser.add_argument("--allow-terminal-rollout-gap", action="store_true")
    parser.add_argument(
        "--allow-rollout-gaps",
        default="",
        type=lambda value: {
            int(item.strip()) for item in value.split(",") if item.strip()
        },
        help="explicit persisted rollout/metric gaps permitted during a resumed audit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
