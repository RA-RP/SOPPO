"""
Evaluation module for preference learning.
"""

from .metrics import compute_accuracy, compute_brier_score, compute_calibration
from .evaluator import evaluate_model, evaluate_and_save_predictions

__all__ = [
    'compute_accuracy',
    'compute_brier_score',
    'compute_calibration',
    'evaluate_model',
    'evaluate_and_save_predictions'
]
