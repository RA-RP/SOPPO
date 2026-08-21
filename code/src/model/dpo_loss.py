"""Response-only scores and standard reference-based DPO."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sequence_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum next-token log-probabilities only over response tokens."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]
    shift_mask = loss_mask[:, 1:].to(shift_logits.dtype)
    token_logps = F.log_softmax(shift_logits, dim=-1).gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_logps * shift_mask).sum(dim=-1)


def response_token_count(loss_mask: torch.Tensor) -> torch.Tensor:
    return loss_mask[:, 1:].sum(dim=-1).clamp_min(1)


def compute_response_mean_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    return compute_sequence_logprob(logits, input_ids, loss_mask) / response_token_count(loss_mask)


def _model_pair_scores(model, batch: Dict[str, torch.Tensor], mean: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    values = []
    for side in ("a", "b"):
        outputs = model(
            input_ids=batch[f"input_ids_{side}"],
            attention_mask=batch[f"attention_mask_{side}"],
            use_cache=False,
            return_dict=True,
        )
        function = compute_response_mean_logprob if mean else compute_sequence_logprob
        values.append(
            function(
                outputs.logits,
                batch[f"input_ids_{side}"],
                batch[f"loss_mask_{side}"],
            )
        )
    return values[0], values[1]


def model_pair_logps(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return _model_pair_scores(model, batch, mean=False)


def model_pair_mean_logps(model, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    return _model_pair_scores(model, batch, mean=True)


def preference_delta(
    policy_a: torch.Tensor,
    policy_b: torch.Tensor,
    reference_a: torch.Tensor,
    reference_b: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return float(beta) * ((policy_a - reference_a) - (policy_b - reference_b))


class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = float(beta)

    def per_sample(
        self,
        policy_a: torch.Tensor,
        policy_b: torch.Tensor,
        reference_a: torch.Tensor,
        reference_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = preference_delta(policy_a, policy_b, reference_a, reference_b, self.beta)
        direction = labels.to(delta.dtype).mul(2).sub(1)
        return -F.logsigmoid(direction * delta), delta

    def forward(
        self,
        policy_a: torch.Tensor,
        policy_b: torch.Tensor,
        reference_a: torch.Tensor,
        reference_b: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        losses, delta = self.per_sample(
            policy_a, policy_b, reference_a, reference_b, labels
        )
        loss = losses.mean()
        probs_a = torch.sigmoid(delta)
        info = {
            "loss": float(loss.detach()),
            "accuracy": float(((probs_a > 0.5) == labels.bool()).float().mean().detach()),
            "p_a_mean": float(probs_a.mean().detach()),
            "delta_mean": float(delta.mean().detach()),
        }
        return loss, info
