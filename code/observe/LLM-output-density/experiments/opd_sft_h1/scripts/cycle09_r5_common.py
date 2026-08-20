#!/usr/bin/env python3
"""Shared definitions for Cycle 09 Round 5 (line A results / line B mechanism).

Numerics are inherited from Round 4 (`cycle09_r4_common` / `cycle09_r4_campaign`)
so that R5 readings are directly comparable with the v2 campaign: same windowing
v2, same hierarchical normalization, same SVD-LLM whitening (Cholesky jitter
1e-5), same r_eps / tail_energy / effective_rank definitions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cycle09_r4_common as c4  # noqa: E402

REPO = c4.REPO
R4_ROOT = c4.RUN_ROOT
RUN_ROOT = Path("/root/autodl-tmp/cycle09_r5")
MINI_ROOT = c4.MINI_ROOT

ARMS = c4.ARMS
STEPS = c4.STEPS
LAYERS = c4.LAYERS
MODULES = c4.MODULES
GENERATION_SEEDS = c4.GENERATION_SEEDS
WINDOW_SEED = c4.WINDOW_SEED
EPSILONS = (0.05, 0.01)
FIXED_RANK_CONTROL = 64  # k=64 robustness control demanded by the R5 handoff

# Cells whose grams are persisted (user ruling Q5-1): the diagonal (own text) and
# the first column (static X_0 reference). Everything else only needs spectra.
def keep_gram(weight_step: int, text_step: int) -> bool:
    return int(weight_step) == int(text_step) or int(text_step) == 0


def x_corpus_path(arm: str, text_step: int, seed: int) -> Path:
    """Text source X_j for the cross matrix.

    Column j=0 is the base model's own math rollouts (shared by both arms; the
    static reference). Columns j>=5 are the arm's own self-generation at step j:
    OPD = its training rollouts (already produced in R4), SFT = B1 (new).
    """
    if int(text_step) == 0:
        return c4.generated_corpus_path("X", "math", seed, "opd", 0, R4_ROOT)
    if arm == "opd":
        return c4.generated_corpus_path("X", "math", seed, "opd", int(text_step), R4_ROOT)
    if arm == "sft":
        return RUN_ROOT / "corpora/generated/X/sft" / c4.step_label(int(text_step)) / "math" / f"gen_seed_{seed}.jsonl"
    raise ValueError(f"unknown arm: {arm}")


def cell_task_id(arm: str, text_step: int, seed: int) -> str:
    source = "base" if int(text_step) == 0 else arm
    return f"Xcross__{source}__{c4.step_label(int(text_step))}__g{seed}"


def measurement_path(weight_arm: str, weight_step: int, task_id: str) -> Path:
    return (
        RUN_ROOT / "measurements" / weight_arm / c4.step_label(int(weight_step))
        / f"{task_id}.json"
    )


def gram_path(weight_arm: str, weight_step: int, task_id: str) -> Path:
    return (
        RUN_ROOT / "scratch/grams" / weight_arm / c4.step_label(int(weight_step))
        / f"{task_id}.pt"
    )


def spectral_gap(sigma: Any, k: int) -> float:
    """gamma_k = sigma_k - sigma_{k+1}  (1-indexed k, as in the A1 spec)."""
    values = np.asarray(list(sigma), dtype=np.float64)
    k = int(k)
    if k <= 0 or k >= values.size:
        return float("nan")
    return float(values[k - 1] - values[k])


def xs_log_spectrum_gap(sigma_x: Any, sigma_s: Any) -> float:
    """Mean |log sigma_X - log sigma_S| over the shared prefix (A5, R2 definition)."""
    x = np.asarray(list(sigma_x), dtype=np.float64)
    s = np.asarray(list(sigma_s), dtype=np.float64)
    width = min(x.size, s.size)
    if width == 0:
        return float("nan")
    x = np.clip(x[:width], 1e-12, None)
    s = np.clip(s[:width], 1e-12, None)
    return float(np.mean(np.abs(np.log(x) - np.log(s))))


def normalized_effective_rank(eigenvalues: Any, epsilon: float = 1e-12) -> float:
    """2605.30524 construct: erank~ = d^-1 * exp(-sum p_i log(p_i + eps)).

    Reported normalized (in [0, 1]) so it is directly comparable with their Fig 2
    (0.60-0.75), unlike our raw ER (1.1-3.8), which is three orders off (A6).
    """
    values = np.asarray(list(eigenvalues), dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    d = values.size
    if total <= 0 or d == 0:
        return 0.0
    p = values / total
    entropy = -float(np.sum(p * np.log(p + float(epsilon))))
    return float(np.exp(entropy) / d)


def top_eigen_share(eigenvalues: Any, k: int = 5) -> float:
    values = np.asarray(list(eigenvalues), dtype=np.float64)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    return float(values[: int(k)].sum() / total)


def principal_angles(u0: Any, ut: Any) -> tuple[float, float]:
    """Bjorck-Golub principal angles between two orthonormal bases.

    Returns (max angle deg, mean angle deg). Inputs: d x r column-orthonormal.
    """
    import torch

    # An fp32 SVD returns bases whose columns are only orthonormal to ~1e-3, which
    # pushes canonical cosines above 1 and (after clamping) floors every small angle
    # to exactly 0. The spanned subspace is still accurate, so re-orthonormalize the
    # truncated bases in float64 (QR) before taking angles; that restores a ~0.01 deg
    # resolution at fp32-SVD cost. Bjorck-Golub on the re-orthonormalized bases.
    q0 = torch.linalg.qr(u0.double(), mode="reduced")[0]
    qt = torch.linalg.qr(ut.double(), mode="reduced")[0]
    sv = torch.linalg.svdvals(q0.T @ qt).clamp(-1.0, 1.0)
    angles = torch.arccos(sv.clamp(0.0, 1.0)) * 180.0 / np.pi
    return float(angles.max()), float(angles.mean())
