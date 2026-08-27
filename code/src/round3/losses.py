"""Round3 losses, including the exact SSPO GitHub code-loss state machine.

The SSPO implementation intentionally mirrors MLAI-Yonsei/SSPO commit
2df9e9a: mean response log-probabilities, sequential chosen -> rejected ->
unpaired running-stat updates, the minimum normalized chosen value as the
threshold, and no KDE or threshold EMA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from ..model.pe_loss import PELoss


def dpo_objective(
    policy_a: torch.Tensor,
    policy_b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    labels: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reference-based DPO over response-token *total* log-probabilities."""
    delta = float(beta) * ((policy_a - reference_a) - (policy_b - reference_b))
    direction = labels.to(delta.dtype).mul(2).sub(1)
    per_pair = -F.logsigmoid(direction * delta)
    loss = per_pair.mean()
    return loss, {
        "loss_dpo": float(loss.detach()),
        "dpo_delta_mean": float(delta.detach().mean()),
        "dpo_accuracy": float(
            ((delta.detach() > 0) == labels.bool()).float().mean()
        ),
    }


def pe_objective(
    score_a: torch.Tensor,
    score_b: torch.Tensor,
    beta: float = 10.0,
    epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Exact 28-pair PE loss over one explicitly selected reward profile.

    For the legacy SimPO-reward arms the scores are raw mean response logp and
    ``beta=10``.  For the corrected DPO-reward arms they are total-response
    log-ratios ``log pi_theta - log pi_ref`` and ``beta=0.1``.
    """
    if score_a.numel() != 28 or score_b.numel() != 28:
        raise ValueError("Round3 PE requires exactly 28 logical candidate pairs")
    if score_a.shape != score_b.shape:
        raise ValueError("Round3 PE candidate score shapes must match")
    probabilities = torch.sigmoid(float(beta) * (score_a - score_b))
    loss, info = PELoss(
        epsilon=float(epsilon), distance="l1", detach_denominator=False
    ).to(probabilities.device)(probabilities)
    entropy = -(
        probabilities.clamp(1e-12, 1 - 1e-12)
        * probabilities.clamp(1e-12, 1 - 1e-12).log()
        + (1 - probabilities).clamp(1e-12, 1 - 1e-12)
        * (1 - probabilities).clamp(1e-12, 1 - 1e-12).log()
    ).mean()
    return loss, {
        **info,
        "probability_entropy": float(entropy.detach()),
        "probability_extreme_rate": float(
            ((probabilities.detach() < 0.05) | (probabilities.detach() > 0.95))
            .float()
            .mean()
        ),
    }


def rollout_anchor_statistics(
    mean_logp_a: torch.Tensor,
    mean_logp_b: torch.Tensor,
    rollout_is_a: torch.Tensor,
    beta: float = 10.0,
) -> Dict[str, float | int]:
    """Source-aligned, sample-free telemetry for SFT-anchor comparisons."""
    if (
        mean_logp_a.numel() != 28
        or mean_logp_b.numel() != 28
        or rollout_is_a.numel() != 28
    ):
        raise ValueError("Round3 rollout/SFT telemetry requires exactly 28 pairs")
    probability_a = torch.sigmoid(
        float(beta) * (mean_logp_a.detach() - mean_logp_b.detach())
    )
    rollout_probability = torch.where(
        rollout_is_a.to(device=probability_a.device, dtype=torch.bool),
        probability_a,
        1.0 - probability_a,
    )
    rollout_wins = int((rollout_probability > 0.5).sum().item())
    sft_wins = int((rollout_probability < 0.5).sum().item())
    ties = int((rollout_probability == 0.5).sum().item())
    return {
        "comparisons": 28,
        "rollout_hard_wins": rollout_wins,
        "sft_hard_wins": sft_wins,
        "ties": ties,
        "rollout_hard_win_rate": rollout_wins / 28.0,
        "rollout_soft_positive_mass": float(rollout_probability.sum().item()),
        "rollout_soft_win_probability_mean": float(rollout_probability.mean().item()),
    }


def joint_dpo_pe_objective(
    dpo_loss: torch.Tensor, pe_loss: torch.Tensor, lambda_pe: float = 0.1
) -> torch.Tensor:
    if float(lambda_pe) != 0.1:
        raise ValueError("Round3 dynamic PE fixes lambda_pe=0.1")
    return (dpo_loss + float(lambda_pe) * pe_loss) / (1.0 + float(lambda_pe))


@dataclass
class GitHubSSPOState:
    """Serializable running state from SSPO commit 2df9e9a lines 99-103."""

    momentum: float = 0.95
    epsilon: float = 1e-8
    clip_range: float = 5.0
    running_mean: float | None = None
    running_var: float | None = None

    def normalize(self, logps: torch.Tensor) -> torch.Tensor:
        if logps.numel() == 0:
            return logps
        batch_mean = logps.mean().detach()
        batch_var = logps.var(unbiased=False).detach() + float(self.epsilon)
        if self.running_mean is None or self.running_var is None:
            updated_mean = batch_mean
            updated_var = batch_var
        else:
            a = float(self.momentum)
            updated_mean = a * float(self.running_mean) + (1.0 - a) * batch_mean
            updated_var = a * float(self.running_var) + (1.0 - a) * batch_var
        self.running_mean = float(updated_mean)
        self.running_var = float(updated_var)
        return torch.clamp(
            (logps - updated_mean) / torch.sqrt(updated_var),
            -float(self.clip_range),
            float(self.clip_range),
        )

    def state_dict(self) -> Dict[str, float | None]:
        return {
            "momentum": float(self.momentum),
            "epsilon": float(self.epsilon),
            "clip_range": float(self.clip_range),
            "running_mean": self.running_mean,
            "running_var": self.running_var,
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any]) -> "GitHubSSPOState":
        required = {
            "momentum",
            "epsilon",
            "clip_range",
            "running_mean",
            "running_var",
        }
        if set(payload) != required:
            raise ValueError("SSPO running-state keys are missing or unexpected")
        state = cls(
            momentum=float(payload["momentum"]),
            epsilon=float(payload["epsilon"]),
            clip_range=float(payload["clip_range"]),
            running_mean=(
                None
                if payload["running_mean"] is None
                else float(payload["running_mean"])
            ),
            running_var=(
                None
                if payload["running_var"] is None
                else float(payload["running_var"])
            ),
        )
        if state.momentum != 0.95 or state.epsilon != 1e-8 or state.clip_range != 5.0:
            raise ValueError("SSPO running-state hyperparameters violate Round3")
        if (state.running_mean is None) != (state.running_var is None):
            raise ValueError("SSPO running mean/variance must both be set or both be None")
        if state.running_var is not None and state.running_var <= 0:
            raise ValueError("SSPO running variance must be positive")
        return state


def github_sspo_objective(
    chosen_mean_logps: torch.Tensor,
    rejected_mean_logps: torch.Tensor,
    unpaired_mean_logps: torch.Tensor,
    state: GitHubSSPOState,
    global_step: int,
    beta: float = 10.0,
    margin: float = 2.0,
    prior: float = 0.5,
    gamma_min: float = 0.125,
    gamma_decay: float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Reference-free SSPO code loss with source-faithful gradient paths.

    In particular, the chosen minimum is not detached in the upstream code, so
    the unlabeled risk can backpropagate through that threshold into the chosen
    response which attains the minimum.
    """
    if chosen_mean_logps.numel() != 4 or rejected_mean_logps.numel() != 4:
        raise ValueError("Round3 SSPO requires exactly four chosen/rejected pairs")
    if unpaired_mean_logps.numel() != 28:
        raise ValueError("Round3 SSPO requires exactly 28 unpaired singles")
    if int(global_step) < 0:
        raise ValueError("SSPO global_step must be non-negative")
    contract = (
        float(beta),
        float(margin),
        float(prior),
        float(gamma_min),
        float(gamma_decay),
    )
    if contract != (10.0, 2.0, 0.5, 0.125, 0.001):
        raise ValueError("SSPO parameters differ from the approved GitHub profile")

    # This call order is part of the scientific method, not an optimization.
    normalized_chosen = state.normalize(chosen_mean_logps)
    normalized_rejected = state.normalize(rejected_mean_logps)
    normalized_unpaired = state.normalize(unpaired_mean_logps)

    labeled_logits = float(beta) * (chosen_mean_logps - rejected_mean_logps) - float(margin)
    labeled_losses = -F.logsigmoid(labeled_logits)
    labeled_risk = labeled_losses.mean()

    threshold = normalized_chosen.min()
    difference = float(beta) * (normalized_unpaired - threshold)
    positive_risk = float(prior) * (-F.logsigmoid(difference))
    negative_risk = (1.0 - float(prior)) * (-F.logsigmoid(-difference))
    unlabeled_losses = torch.where(difference > 0, positive_risk, negative_risk)
    unlabeled_risk = unlabeled_losses.mean()
    gamma = max(float(gamma_min), math.exp(-float(gamma_decay) * int(global_step)))
    loss = gamma * labeled_risk + (1.0 - gamma) * unlabeled_risk
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite SSPO objective")
    return loss, {
        "loss_sspo": float(loss.detach()),
        "loss_labeled_simpo": float(labeled_risk.detach()),
        "loss_unpaired": float(unlabeled_risk.detach()),
        "gamma": float(gamma),
        "threshold_min_normalized_chosen": float(threshold.detach()),
        "pseudo_positive_rate": float((difference.detach() > 0).float().mean()),
        "running_mean": float(state.running_mean),
        "running_var": float(state.running_var),
        "chosen_reward_accuracy": float(
            (chosen_mean_logps.detach() > rejected_mean_logps.detach()).float().mean()
        ),
    }
