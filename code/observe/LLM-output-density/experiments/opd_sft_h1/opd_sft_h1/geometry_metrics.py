from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def _array(values: Sequence[float] | Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.abs(arr[np.isfinite(arr)])
    return arr


def effective_rank(sigma: Sequence[float] | Any) -> float:
    values = _array(sigma)
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0.0]
    return float(np.exp(-np.sum(probs * np.log(probs))))


def spectral_gap(sigma: Sequence[float] | Any, k: int) -> float | None:
    if k < 1:
        raise ValueError("k is 1-indexed and must be >= 1")
    values = np.sort(_array(sigma))[::-1]
    if len(values) <= k:
        return None
    return float(values[k - 1] - values[k])


def log_spectrum_drift(sigma_t: Sequence[float] | Any, sigma_0: Sequence[float] | Any, eps: float = 1e-12) -> float | None:
    current = _array(sigma_t)
    baseline = _array(sigma_0)
    length = min(len(current), len(baseline))
    if length == 0:
        return None
    diff = np.log(current[:length] + eps) - np.log(baseline[:length] + eps)
    return float(math.sqrt(np.mean(diff**2)))


def xs_log_spectrum_gap(sigma_x: Sequence[float] | Any, sigma_s: Sequence[float] | Any, eps: float = 1e-12) -> float | None:
    x_values = _array(sigma_x)
    s_values = _array(sigma_s)
    length = min(len(x_values), len(s_values))
    if length == 0:
        return None
    diff = np.log(x_values[:length] + eps) - np.log(s_values[:length] + eps)
    return float(np.mean(diff))


def principal_angle_unavailable() -> dict[str, Any]:
    return {"principal_angle": None, "principal_angle_status": "unavailable_no_uv"}
