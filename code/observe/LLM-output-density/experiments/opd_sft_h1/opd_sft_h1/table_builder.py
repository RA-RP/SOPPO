from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from .match_math500 import match_opd_to_sft
from .paths import ensure_dir


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path_obj.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path_obj


def load_csv_records(path: str | Path) -> list[dict[str, Any]]:
    return list(pd.read_csv(path, encoding="utf-8-sig").to_dict(orient="records"))


def build_matched_math500_pairs(
    opd_rows: Any,
    sft_rows: Any,
    output_path: str | Path,
    *,
    max_gap: float = 2.0,
) -> Path:
    matches = match_opd_to_sft(opd_rows, sft_rows, max_gap=max_gap)
    return write_csv(matches, output_path)
