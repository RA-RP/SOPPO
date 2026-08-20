#!/usr/bin/env python3
"""S1-7: token-level degradation audit for stored H/B1 generations.

The primary table has one row per (probe, arm, step). A seed-level companion table
keeps the three generation batches visible. This script reports readings only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


R4 = Path("/root/autodl-tmp/cycle09_r4/corpora/generated")
R5 = Path("/root/autodl-tmp/cycle09_r5/corpora/generated")
MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
SEEDS = (3, 17, 31)
H_DOMAINS = ("bos", "general", "ood")
H_ARMS = ("opd", "sft")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def ngram_ratio(token_ids: list[int], order: int, *, distinct: bool) -> float:
    total = len(token_ids) - order + 1
    if total <= 0:
        return float("nan")
    unique = len({tuple(token_ids[i : i + order]) for i in range(total)})
    ratio = unique / total
    return ratio if distinct else 1.0 - ratio


def expected_sources() -> list[dict]:
    sources: list[dict] = []
    for arm in H_ARMS:
        for step in STEPS:
            for domain in H_DOMAINS:
                for seed in SEEDS:
                    sources.append(
                        {
                            "probe": f"H_{domain}",
                            "arm": arm,
                            "step": step,
                            "generation_seed": seed,
                            "path": R4
                            / "H"
                            / arm
                            / f"step_{step:03d}"
                            / domain
                            / f"gen_seed_{seed}.jsonl",
                            "source_alias": "none",
                        }
                    )

    for step in STEPS:
        for seed in SEEDS:
            if step == 0:
                path = (
                    R4
                    / "X"
                    / "opd"
                    / "step_000"
                    / "math"
                    / f"gen_seed_{seed}.jsonl"
                )
                alias = "shared_base_from_R4_X_opd_step_000_math"
            else:
                path = (
                    R5
                    / "X"
                    / "sft"
                    / f"step_{step:03d}"
                    / "math"
                    / f"gen_seed_{seed}.jsonl"
                )
                alias = "none"
            sources.append(
                {
                    "probe": "B1_X_sft_math",
                    "arm": "sft",
                    "step": step,
                    "generation_seed": seed,
                    "path": path,
                    "source_alias": alias,
                }
            )
    return sources


def read_source(source: dict) -> tuple[list[dict], dict]:
    path = source["path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    rows: list[dict] = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            record = json.loads(raw)
            token_ids = record.get("generation_token_ids")
            if not isinstance(token_ids, list):
                raise ValueError(f"missing generation_token_ids: {path}:{line_number}")
            rows.append(
                {
                    "probe": source["probe"],
                    "arm": source["arm"],
                    "step": int(source["step"]),
                    "generation_seed": int(source["generation_seed"]),
                    "sample_id": str(record.get("sample_id", line_number)),
                    "length_tokens": len(token_ids),
                    "rep4": ngram_ratio(token_ids, 4, distinct=False),
                    "distinct2": ngram_ratio(token_ids, 2, distinct=True),
                    "truncated": int(record.get("finish_reason") == "length"),
                }
            )
    stat = path.stat()
    provenance = {
        "path": str(path),
        "probe": source["probe"],
        "arm": source["arm"],
        "step": int(source["step"]),
        "generation_seed": int(source["generation_seed"]),
        "source_alias": source["source_alias"],
        "n_samples": len(rows),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }
    return rows, provenance


def summarize(samples: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    keys = list(keys)
    records = []
    for values, group in samples.groupby(keys, sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        row = dict(zip(keys, values))
        row.update(
            {
                "n_samples": int(len(group)),
                "length_tokens_mean": float(group["length_tokens"].mean()),
                "length_tokens_median": float(group["length_tokens"].median()),
                "rep4_mean": float(group["rep4"].mean()),
                "distinct2_mean": float(group["distinct2"].mean()),
                "truncation_rate": float(group["truncated"].mean()),
            }
        )
        if "generation_seed" not in keys:
            row["generation_seed_batches"] = int(group["generation_seed"].nunique())
        records.append(row)
    return pd.DataFrame(records).sort_values(keys).reset_index(drop=True)


def main() -> None:
    sample_rows: list[dict] = []
    provenance: list[dict] = []
    for source in expected_sources():
        rows, source_provenance = read_source(source)
        sample_rows.extend(rows)
        provenance.append(source_provenance)

    samples = pd.DataFrame(sample_rows)
    primary = summarize(samples, ("probe", "arm", "step"))
    seed_level = summarize(samples, ("probe", "arm", "step", "generation_seed"))

    expected_primary_rows = len(H_DOMAINS) * len(H_ARMS) * len(STEPS) + len(STEPS)
    expected_seed_rows = expected_primary_rows * len(SEEDS)
    if len(primary) != expected_primary_rows or len(seed_level) != expected_seed_rows:
        raise RuntimeError(
            f"grid incomplete: primary={len(primary)}/{expected_primary_rows}, "
            f"seed={len(seed_level)}/{expected_seed_rows}"
        )
    if set(primary["step"]) != set(STEPS):
        raise RuntimeError(f"unexpected step grid: {sorted(primary['step'].unique())}")

    atomic_csv(primary, MINI / "S1_h_text_stats.csv")
    atomic_csv(seed_level, MINI / "S1_h_text_stats_by_seed.csv")
    atomic_json(
        {
            "schema_version": 2,
            "task": "S1-7",
            "scope_override": (
                "ten checkpoints per latest user decision; Stage 1 handoff text listed "
                "the older seven-checkpoint scope"
            ),
            "steps": list(STEPS),
            "generation_seeds": list(SEEDS),
            "primary_key": ["probe", "arm", "step"],
            "n_primary_rows": len(primary),
            "n_seed_rows": len(seed_level),
            "n_sample_rows": len(samples),
            "metric_definitions": {
                "length_tokens": "len(generation_token_ids)",
                "rep4": "1 - unique token-id 4-grams / token-id 4-grams, per sample then mean",
                "distinct2": "unique token-id 2-grams / token-id 2-grams, per sample then mean",
                "truncation": "finish_reason == 'length'",
            },
            "b1_step0_alias": "R4 X/opd/step_000/math is the shared base generation",
            "sources": provenance,
        },
        MINI / "S1_h_text_stats_manifest.json",
    )
    print(
        f"[S1-7] primary_rows={len(primary)} seed_rows={len(seed_level)} "
        f"samples={len(samples)} files={len(provenance)}"
    )
    print(primary.to_string(index=False))


if __name__ == "__main__":
    main()
