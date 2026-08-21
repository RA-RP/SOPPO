"""
Diagnostic utilities for monitoring training.
"""

import torch
import numpy as np
from typing import Dict, List
from collections import defaultdict
import json


class DiagnosticsTracker:
    """Track and aggregate diagnostic information during training."""

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, info: Dict):
        """Add diagnostic info from current step."""
        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.history[key].append(value)

    def get_recent(self, key: str, n: int = 10) -> List:
        """Get recent n values for a key."""
        if key not in self.history:
            return []
        return self.history[key][-n:]

    def get_mean(self, key: str, n: int = 10) -> float:
        """Get mean of recent n values."""
        recent = self.get_recent(key, n)
        if not recent:
            return 0.0
        return np.mean(recent)

    def get_statistics(self, key: str, n: int = None) -> Dict:
        """Get statistics for a key."""
        if key not in self.history:
            return {}

        values = self.history[key] if n is None else self.history[key][-n:]
        if not values:
            return {}

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values)
        }

    def save(self, filepath: str):
        """Save history to JSON."""
        with open(filepath, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def load(self, filepath: str):
        """Load history from JSON."""
        with open(filepath, 'r') as f:
            loaded = json.load(f)
            self.history = defaultdict(list, loaded)


def compute_responsibility_quality(p_i: torch.Tensor) -> Dict:
    """
    Compute responsibility quality metrics.

    Args:
        p_i: Preference probabilities [batch_size]

    Returns:
        Quality metrics
    """
    p_i_np = p_i.detach().cpu().numpy()

    # Sum of responsibilities
    sum_p = p_i_np.sum()
    sum_1_minus_p = (1 - p_i_np).sum()

    # Distribution statistics
    mean_p = p_i_np.mean()
    std_p = p_i_np.std()
    min_p = p_i_np.min()
    max_p = p_i_np.max()

    # Histogram (10 bins)
    hist, bin_edges = np.histogram(p_i_np, bins=10, range=(0, 1))

    # Check for degenerate cases
    is_collapsed = (std_p < 0.01)  # All p_i very close
    is_bimodal = (hist[0] + hist[-1]) > 0.8 * len(p_i_np)  # Most at extremes

    quality = {
        'sum_p': float(sum_p),
        'sum_1_minus_p': float(sum_1_minus_p),
        'mean': float(mean_p),
        'std': float(std_p),
        'min': float(min_p),
        'max': float(max_p),
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
        'is_collapsed': bool(is_collapsed),
        'is_bimodal': bool(is_bimodal)
    }

    return quality


def compute_prediction_distribution_stats(p_i: torch.Tensor) -> Dict:
    """
    Compute prediction distribution statistics.

    Args:
        p_i: Preference probabilities [batch_size]

    Returns:
        Distribution statistics
    """
    p_i_np = p_i.detach().cpu().numpy()

    # Entropy: H = -Σ [p log p + (1-p) log(1-p)]
    epsilon = 1e-10
    p_clipped = np.clip(p_i_np, epsilon, 1 - epsilon)
    entropy = -(p_clipped * np.log(p_clipped) +
               (1 - p_clipped) * np.log(1 - p_clipped))
    mean_entropy = entropy.mean()

    # Confidence: fraction with |p - 0.5| > threshold
    confidence_50 = np.mean(np.abs(p_i_np - 0.5) > 0.1)
    confidence_70 = np.mean(np.abs(p_i_np - 0.5) > 0.2)
    confidence_90 = np.mean(np.abs(p_i_np - 0.5) > 0.4)

    stats = {
        'mean_entropy': float(mean_entropy),
        'confidence_50': float(confidence_50),
        'confidence_70': float(confidence_70),
        'confidence_90': float(confidence_90)
    }

    return stats


def check_numerical_stability(tensors: Dict[str, torch.Tensor]) -> Dict:
    """
    Check for NaN and Inf in tensors.

    Args:
        tensors: Dictionary of named tensors to check

    Returns:
        Stability report
    """
    report = {}

    for name, tensor in tensors.items():
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        report[name] = {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'is_stable': not (has_nan or has_inf)
        }

        if has_nan or has_inf:
            print(f"WARNING: Numerical instability detected in {name}")
            print(f"  NaN: {has_nan}, Inf: {has_inf}")

    all_stable = all(info['is_stable'] for info in report.values())
    report['all_stable'] = all_stable

    return report


def log_training_step(
    step: int,
    total_loss: float,
    dpo_loss: float,
    aux_loss: float,
    lambda_t: float,
    learning_rate: float,
    diagnostics: Dict,
    log_file: str
):
    """
    Log training step information to JSONL file.

    Args:
        step: Current training step
        total_loss: Total loss value
        dpo_loss: DPO loss component
        aux_loss: Auxiliary loss (PE or SSPO hard pseudo-risk)
        lambda_t: Current lambda value
        learning_rate: Current learning rate
        diagnostics: Additional diagnostic info
        log_file: Path to JSONL log file
    """
    import jsonlines

    log_entry = {
        'step': step,
        'loss': {
            'total': total_loss,
            'dpo': dpo_loss,
            'aux': aux_loss,
            'lambda': lambda_t
        },
        'learning_rate': learning_rate,
        **diagnostics
    }

    with jsonlines.open(log_file, mode='a') as writer:
        writer.write(log_entry)
