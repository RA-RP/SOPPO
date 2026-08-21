"""
Evaluation metrics for preference learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor,
                     threshold: float = 0.5) -> float:
    """
    Compute preference accuracy.

    Acc = (1/|D|) × Σ_i 1[(p_i > threshold) = z_i]

    Args:
        predictions: Preference probabilities [N]
        labels: True labels [N]
        threshold: Decision threshold (default 0.5)

    Returns:
        Accuracy as float
    """
    preds = (predictions > threshold).long()
    correct = (preds == labels).sum().item()
    total = len(labels)
    return correct / total if total > 0 else 0.0


def compute_brier_score(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Brier score (lower is better).

    Brier = (1/|D|) × Σ_i (p_i - z_i)²

    Args:
        predictions: Preference probabilities [N]
        labels: True labels [N] (0 or 1)

    Returns:
        Brier score as float
    """
    labels_float = labels.float()
    brier = ((predictions - labels_float) ** 2).mean().item()
    return brier


def compute_calibration(predictions: torch.Tensor, labels: torch.Tensor,
                       n_bins: int = 10) -> Dict:
    """
    Compute calibration statistics (reliability diagram).

    Args:
        predictions: Preference probabilities [N]
        labels: True labels [N]
        n_bins: Number of calibration bins

    Returns:
        Calibration statistics dictionary
    """
    predictions_np = predictions.cpu().numpy().astype(np.float64)
    labels_np = labels.cpu().numpy()

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Compute calibration per bin
    calibration_data = []
    for index, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find samples in this bin
        upper_check = predictions_np <= bin_upper if index == n_bins - 1 else predictions_np < bin_upper
        in_bin = (predictions_np >= bin_lower) & upper_check
        prop_in_bin = in_bin.mean()

        if in_bin.sum() > 0:
            accuracy_in_bin = labels_np[in_bin].mean()
            avg_confidence_in_bin = predictions_np[in_bin].mean()
        else:
            accuracy_in_bin = 0.0
            avg_confidence_in_bin = (bin_lower + bin_upper) / 2

        calibration_data.append({
            'bin_lower': float(bin_lower),
            'bin_upper': float(bin_upper),
            'accuracy': float(accuracy_in_bin),
            'confidence': float(avg_confidence_in_bin),
            'proportion': float(prop_in_bin),
            'count': int(in_bin.sum())
        })

    # Compute Expected Calibration Error (ECE)
    ece = 0.0
    for data in calibration_data:
        if data['count'] > 0:
            ece += data['proportion'] * abs(data['accuracy'] - data['confidence'])

    return {
        'ece': ece,
        'bins': calibration_data
    }


def compute_confidence_distribution(predictions: torch.Tensor) -> Dict:
    """
    Compute confidence distribution statistics.

    Args:
        predictions: Preference probabilities [N]

    Returns:
        Confidence statistics
    """
    predictions_np = predictions.cpu().numpy().astype(np.float64)

    # Distance from 0.5
    distance_from_half = np.abs(predictions_np - 0.5)

    # Confidence levels
    confidence_50 = (distance_from_half > 0.1).mean()  # >60% or <40%
    confidence_70 = (distance_from_half > 0.2).mean()  # >70% or <30%
    confidence_90 = (distance_from_half > 0.4).mean()  # >90% or <10%

    # Entropy
    epsilon = np.finfo(np.float64).eps
    p_clipped = np.clip(predictions_np, epsilon, 1 - epsilon)
    entropy = -(p_clipped * np.log(p_clipped) +
               (1 - p_clipped) * np.log(1 - p_clipped))
    mean_entropy = entropy.mean()

    return {
        'confidence_50': float(confidence_50),
        'confidence_70': float(confidence_70),
        'confidence_90': float(confidence_90),
        'mean_entropy': float(mean_entropy),
        'mean_prediction': float(predictions_np.mean()),
        'std_prediction': float(predictions_np.std())
    }
