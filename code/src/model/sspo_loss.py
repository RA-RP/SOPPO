"""Paper-faithful SSPO rewards, KDE thresholding, and shared objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F


def gather_equal_vector(local: torch.Tensor) -> torch.Tensor:
    """All-gather a same-sized detached vector from every training rank."""
    local = local.detach().float().reshape(-1)
    if not dist.is_initialized():
        return local
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return torch.cat(gathered)


def simpo_pair_losses(
    mean_logp_a: torch.Tensor,
    mean_logp_b: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pair SimPO/SSPO labeled loss with arbitrary randomized A/B order."""
    delta = float(beta) * (mean_logp_a - mean_logp_b)
    direction = labels.to(delta.dtype).mul(2).sub(1)
    oriented = direction * delta - float(margin)
    return -F.logsigmoid(oriented), delta


def pe_pair_probabilities(
    mean_logp_a: torch.Tensor,
    mean_logp_b: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Margin-free A-vs-B probability; a margin would create position bias."""
    return torch.sigmoid(float(beta) * (mean_logp_a - mean_logp_b))


def gamma_weight(step: int, gamma0: float, gamma_min: float, decay: float) -> float:
    return max(float(gamma_min), float(gamma0) * math.exp(-float(decay) * int(step)))


def objective_weights(method: Dict, step: int) -> Tuple[float, float]:
    if method["weighting"] == "exponential_gamma":
        gamma = gamma_weight(step, method["gamma0"], method["gamma_min"], method["gamma_decay"])
        return gamma, 1.0 - gamma
    if method["weighting"] == "normalized_fixed_lambda":
        value = float(method["fixed_lambda"])
        return 1.0 / (1.0 + value), value / (1.0 + value)
    raise ValueError(f"Unsupported joint weighting: {method['weighting']}")


def _scott_bandwidth(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    if values.numel() < 2:
        return values.new_tensor(epsilon)
    scale = values.std(unbiased=True)
    return (scale * values.numel() ** (-0.2)).clamp_min(epsilon)


def kde_bayes_threshold(
    winning: torch.Tensor,
    losing: torch.Tensor,
    prior: float,
    grid_points: int,
    epsilon: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Gaussian-KDE Bayes-risk minimizer on the paper's 200-point grid.

    The paper leaves bandwidth unspecified. v0.6 freezes Scott's rule separately
    for the winning and losing minibatch distributions.
    """
    winning = winning.detach().float().reshape(-1)
    losing = losing.detach().float().reshape(-1)
    if winning.numel() < 2 or losing.numel() < 2:
        raise ValueError("KDE threshold requires at least two winning and losing rewards")
    h_w = _scott_bandwidth(winning, epsilon)
    h_l = _scott_bandwidth(losing, epsilon)
    lower = torch.minimum(winning.min() - 3.0 * h_w, losing.min() - 3.0 * h_l)
    upper = torch.maximum(winning.max() + 3.0 * h_w, losing.max() + 3.0 * h_l)
    grid = torch.linspace(lower, upper, int(grid_points), device=winning.device)
    root_two = math.sqrt(2.0)
    cdf_w = (0.5 * (1.0 + torch.erf((grid[:, None] - winning[None, :]) / (h_w * root_two)))).mean(1)
    cdf_l = (0.5 * (1.0 + torch.erf((grid[:, None] - losing[None, :]) / (h_l * root_two)))).mean(1)
    risk = float(prior) * cdf_w + (1.0 - float(prior)) * (1.0 - cdf_l)
    index = int(torch.argmin(risk))
    threshold = grid[index]
    return threshold, {
        "kde_bandwidth_winning": float(h_w),
        "kde_bandwidth_losing": float(h_l),
        "kde_min_risk": float(risk[index]),
        "kde_grid_min": float(lower),
        "kde_grid_max": float(upper),
    }


@dataclass
class SSPOThresholdState:
    momentum: float = 0.95
    prior: float = 0.5
    epsilon: float = 1e-6
    grid_points: int = 200
    mean: float | None = None
    std: float | None = None
    threshold: float | None = None

    def update(
        self,
        all_rewards: torch.Tensor,
        winning_rewards: torch.Tensor,
        losing_rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """Update global-minibatch EMA statistics and normalized KDE threshold."""
        all_rewards = all_rewards.detach().float()
        batch_mean = float(all_rewards.mean())
        batch_std = max(float(all_rewards.std(unbiased=False)), self.epsilon)
        if self.mean is None:
            self.mean, self.std = batch_mean, batch_std
        else:
            self.mean = self.momentum * self.mean + (1.0 - self.momentum) * batch_mean
            self.std = self.momentum * self.std + (1.0 - self.momentum) * batch_std
        normalized_winning = (winning_rewards.detach().float() - self.mean) / self.std
        normalized_losing = (losing_rewards.detach().float() - self.mean) / self.std
        threshold_batch, kde = kde_bayes_threshold(
            normalized_winning,
            normalized_losing,
            self.prior,
            self.grid_points,
            self.epsilon,
        )
        value = float(threshold_batch)
        self.threshold = value if self.threshold is None else (
            self.momentum * self.threshold + (1.0 - self.momentum) * value
        )
        return {
            "reward_batch_mean": batch_mean,
            "reward_batch_std": batch_std,
            "reward_ema_mean": self.mean,
            "reward_ema_std": self.std,
            "threshold_batch": value,
            "threshold_ema": self.threshold,
            **kde,
        }

    def state_dict(self) -> Dict[str, float | None]:
        return {"mean": self.mean, "std": self.std, "threshold": self.threshold}


def hard_pseudo_response_losses(
    live_rewards: torch.Tensor,
    state: SSPOThresholdState,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if state.mean is None or state.std is None or state.threshold is None:
        raise ValueError("SSPO threshold state must be initialized before pseudo-labeling")
    normalized = (live_rewards - float(state.mean)) / float(state.std)
    labels = (normalized.detach() > float(state.threshold)).to(normalized.dtype)
    logits = normalized - float(state.threshold)
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    prior_weights = torch.where(labels.bool(), float(state.prior), 1.0 - float(state.prior))
    weighted = losses * prior_weights
    return weighted, {
        "pseudo_positive_rate": float(labels.mean().detach()),
        "normalized_reward_mean": float(normalized.mean().detach()),
        "unweighted_bce_mean": float(losses.mean().detach()),
    }
