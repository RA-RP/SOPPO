from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def _get(row: Mapping[str, Any] | Any, key: str) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    return getattr(row, key, None)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def math500_gain(checkpoint_math500: Any, baseline_math500: Any) -> float | None:
    checkpoint = _as_float(checkpoint_math500)
    baseline = _as_float(baseline_math500)
    if checkpoint is None or baseline is None:
        return None
    return checkpoint - baseline


def per_benchmark_drop(
    row: Mapping[str, Any] | Any,
    baseline_row: Mapping[str, Any] | Any,
    benchmarks: Iterable[str],
) -> dict[str, float | None]:
    drops: dict[str, float | None] = {}
    for benchmark in benchmarks:
        score = _as_float(_get(row, benchmark))
        baseline_score = _as_float(_get(baseline_row, benchmark))
        if score is None or baseline_score is None:
            drops[benchmark] = None
            continue
        drops[benchmark] = max(0.0, baseline_score - score)
    return drops


def general_ood_avg(row: Mapping[str, Any] | Any, benchmarks: Iterable[str]) -> float | None:
    values = [_as_float(_get(row, benchmark)) for benchmark in benchmarks]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def general_ood_penalty(
    row: Mapping[str, Any] | Any,
    baseline_row: Mapping[str, Any] | Any,
    benchmarks: Iterable[str],
    p: int | float,
) -> float | None:
    if p <= 0:
        raise ValueError("p must be positive")
    drops = [drop for drop in per_benchmark_drop(row, baseline_row, benchmarks).values() if drop is not None]
    if not drops:
        return None
    return sum(drop**p for drop in drops) ** (1.0 / p)


def worst_ood_drop(
    row: Mapping[str, Any] | Any,
    baseline_row: Mapping[str, Any] | Any,
    benchmarks: Iterable[str],
) -> float | None:
    drops = [drop for drop in per_benchmark_drop(row, baseline_row, benchmarks).values() if drop is not None]
    if not drops:
        return None
    return max(drops)


def infer_score_scale(row: Mapping[str, Any] | Any, benchmarks: Iterable[str]) -> str:
    values = [_as_float(_get(row, benchmark)) for benchmark in benchmarks]
    present = [value for value in values if value is not None]
    if not present:
        return "unknown"
    max_value = max(present)
    if max_value <= 1.5:
        return "0_1"
    if max_value <= 100.0:
        return "0_100"
    return "unknown"
