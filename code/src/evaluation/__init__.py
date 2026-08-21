"""Evaluation metrics and command-line evaluation utilities."""

from .metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_calibration,
    compute_confidence_distribution,
)

__all__ = [
    "compute_accuracy",
    "compute_brier_score",
    "compute_calibration",
    "compute_confidence_distribution",
]
