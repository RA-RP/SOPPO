#!/usr/bin/env python3
"""Fetch revision-pinned official sources for the Q1 domain-matched probes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

import cycle09_block3_common as c


MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
AIME25_REVISION = "563bb8404243c5f09de6ec262f2db674fe5bce9b"
MATH_FILES = (
    "algebra/test-00000-of-00001.parquet",
    "counting_and_probability/test-00000-of-00001.parquet",
    "geometry/test-00000-of-00001.parquet",
    "intermediate_algebra/test-00000-of-00001.parquet",
    "number_theory/test-00000-of-00001.parquet",
    "prealgebra/test-00000-of-00001.parquet",
    "precalculus/test-00000-of-00001.parquet",
)
SOURCE_ROOT = c.Q1_ROOT / "geometry_sources"
RAW_ROOT = SOURCE_ROOT / "raw"
MANIFEST = SOURCE_ROOT / "manifest.json"
MATH_OUTPUT = c.AUTODL / "dataset/hendrycks_math/test.jsonl"
AIME25_OUTPUT = c.AUTODL / "dataset/aime25/test.jsonl"


def download(url: str, target: Path) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            while chunk := response.read(8 << 20):
                handle.write(chunk)
    os.replace(temporary, target)


def math_url(relative: str) -> str:
    return (
        "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/"
        f"{MATH_REVISION}/{relative}"
    )


def aime25_url() -> str:
    return (
        "https://huggingface.co/datasets/math-ai/aime25/resolve/"
        f"{AIME25_REVISION}/test.jsonl"
    )


def prepare() -> dict:
    math_raw = []
    for relative in MATH_FILES:
        target = RAW_ROOT / "hendrycks_math" / relative
        download(math_url(relative), target)
        math_raw.append(target)
    aime_raw = RAW_ROOT / "aime25/test.jsonl"
    download(aime25_url(), aime_raw)

    math_rows = []
    for path in math_raw:
        math_rows.extend(pd.read_parquet(path).to_dict("records"))
    if len(math_rows) != 5000 or any(not str(row.get("problem", "")).strip() for row in math_rows):
        raise RuntimeError(f"unexpected Hendrycks MATH test inventory: {len(math_rows)} rows")
    aime_rows = c.read_jsonl(aime_raw)
    if len(aime_rows) != 30 or any(not str(row.get("problem", "")).strip() for row in aime_rows):
        raise RuntimeError(f"unexpected AIME25 inventory: {len(aime_rows)} rows")
    c.atomic_jsonl(MATH_OUTPUT, math_rows)
    c.atomic_jsonl(AIME25_OUTPUT, aime_rows)

    payload = {
        "schema_version": 1,
        "status": "complete",
        "task": "Q1 Stage-A fixed probe source acquisition",
        "sources": {
            "E_math": {
                "dataset": "EleutherAI/hendrycks_math",
                "revision": MATH_REVISION,
                "split": "official test; all seven subject shards",
                "urls": [math_url(relative) for relative in MATH_FILES],
                "raw_files": [c.artifact(path) for path in math_raw],
                "combined": c.artifact(MATH_OUTPUT),
                "rows_before_MATH500_and_training_dedup": len(math_rows),
            },
            "E_math_hard_v2": {
                "dataset": "math-ai/aime25",
                "revision": AIME25_REVISION,
                "split": "test",
                "url": aime25_url(),
                "raw": c.artifact(aime_raw),
                "combined": c.artifact(AIME25_OUTPUT),
                "rows_before_AIME24_and_training_dedup": len(aime_rows),
            },
        },
        "next_step": "cycle09_llama_probe_prepare.py --phase fixed performs frozen deduplication",
        "created_utc": c.utc_now(),
    }
    c.atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare",), default="prepare")
    args = parser.parse_args()
    print(json.dumps(prepare(), indent=2))


if __name__ == "__main__":
    main()
