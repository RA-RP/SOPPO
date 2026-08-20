"""
PE (Preference Encoding) structure loss for label-encoding supervision.

This implements the population-level structural supervision:
    L_PE = 0.5 * (||ê_+ - e_+||_1 + ||ê_- - e_-||_1)

where:
- ê_+ = Σ_i p_i q_i / (Σ_i p_i + ε)
- ê_- = Σ_i (1-p_i) q_i / (Σ_i (1-p_i) + ε)
- q_i = [p_i, 1-p_i]^T
- e_+ = [1, 0]^T (ideal encoding for preferred)
- e_- = [0, 1]^T (ideal encoding for dis-preferred)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class PELoss(nn.Module):
    """
    Preference Encoding (PE) structure loss.

    This loss enforces population-level structure on unlabeled preference pairs
    without assigning instance-level pseudo labels.
    """

    def __init__(
        self,
        epsilon: float = 1e-8,
        distance: str = 'l1',
        detach_denominator: bool = False
    ):
        """
        Args:
            epsilon: Numerical stability term for denominators
            distance: Distance metric ('l1' or 'l2')
            detach_denominator: Whether to stop gradient through denominators (emergency plan)
        """
        super().__init__()
        self.epsilon = epsilon
        self.distance = distance
        self.detach_denominator = detach_denominator

        # Ideal encodings
        self.register_buffer('e_plus', torch.tensor([1.0, 0.0]))
        self.register_buffer('e_minus', torch.tensor([0.0, 1.0]))

    def forward(self, p_i: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PE structure loss.

        Args:
            p_i: Preference probabilities [batch_size]
                 p_i = σ(Δ_i) where Δ_i = r_θ(x, y_a) - r_θ(x, y_b)

        Returns:
            loss: Scalar structure loss
            info: Diagnostic information including encoding estimates
        """
        batch_size = p_i.shape[0]

        # Construct q_i = [p_i, 1-p_i]^T for each sample
        # Shape: [batch_size, 2]
        q_i = torch.stack([p_i, 1 - p_i], dim=-1)

        # Compute responsibility-weighted sums
        # sum_p: scalar, Σ_i p_i
        # sum_1_minus_p: scalar, Σ_i (1-p_i)
        sum_p = p_i.sum()
        sum_1_minus_p = (1 - p_i).sum()

        # Add numerical stability term
        sum_p_stable = sum_p + self.epsilon
        sum_1_minus_p_stable = sum_1_minus_p + self.epsilon

        # Optionally detach denominators (emergency plan for training instability)
        if self.detach_denominator:
            sum_p_stable = sum_p_stable.detach()
            sum_1_minus_p_stable = sum_1_minus_p_stable.detach()

        # Estimate conditional encodings
        # ê_+ = Σ_i p_i q_i / (Σ_i p_i + ε)
        # Shape: [2]
        e_hat_plus = (p_i.unsqueeze(-1) * q_i).sum(dim=0) / sum_p_stable

        # ê_- = Σ_i (1-p_i) q_i / (Σ_i (1-p_i) + ε)
        # Shape: [2]
        e_hat_minus = ((1 - p_i).unsqueeze(-1) * q_i).sum(dim=0) / sum_1_minus_p_stable

        # Compute distance to ideal encodings
        if self.distance == 'l1':
            dist_plus = torch.abs(e_hat_plus - self.e_plus).sum()
            dist_minus = torch.abs(e_hat_minus - self.e_minus).sum()
        elif self.distance == 'l2':
            dist_plus = torch.sqrt(((e_hat_plus - self.e_plus) ** 2).sum())
            dist_minus = torch.sqrt(((e_hat_minus - self.e_minus) ** 2).sum())
        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        # PE structure loss
        loss = 0.5 * (dist_plus + dist_minus)

        # Diagnostic information
        info = {
            'loss': loss.item(),
            'e_hat_plus_0': e_hat_plus[0].item(),
            'e_hat_plus_1': e_hat_plus[1].item(),
            'e_hat_minus_0': e_hat_minus[0].item(),
            'e_hat_minus_1': e_hat_minus[1].item(),
            'dist_plus': dist_plus.item(),
            'dist_minus': dist_minus.item(),
            'sum_p': sum_p.item(),
            'sum_1_minus_p': sum_1_minus_p.item(),
            'p_i_mean': p_i.mean().item(),
            'p_i_std': p_i.std().item(),
            'p_i_min': p_i.min().item(),
            'p_i_max': p_i.max().item(),
        }

        return loss, info


def compute_pe_loss(
    policy_model,
    reference_model,
    input_ids_a: torch.Tensor,
    attention_mask_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    attention_mask_b: torch.Tensor,
    beta: float = 0.1,
    epsilon: float = 1e-8,
    distance: str = 'l1',
    detach_denominator: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute PE structure loss from models and unlabeled inputs.

    Args:
        policy_model: Policy model being trained
        reference_model: Frozen reference model
        input_ids_a: Input IDs for response_a
        attention_mask_a: Attention mask for response_a
        input_ids_b: Input IDs for response_b
        attention_mask_b: Attention mask for response_b
        beta: DPO temperature parameter
        epsilon: Numerical stability term
        distance: Distance metric
        detach_denominator: Whether to detach denominators

    Returns:
        loss: PE structure loss
        info: Diagnostic information
    """
    from .dpo_loss import compute_sequence_logprob

    # Get logits from policy model
    with torch.cuda.amp.autocast():
        policy_a_outputs = policy_model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            return_dict=True
        )
        policy_b_outputs = policy_model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            return_dict=True
        )

    # Get logits from reference model (no gradient)
    with torch.no_grad():
        reference_a_outputs = reference_model(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
            return_dict=True
        )
        reference_b_outputs = reference_model(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,
            return_dict=True
        )

    # Compute log probabilities
    policy_a_logps = compute_sequence_logprob(
        policy_a_outputs.logits, input_ids_a, attention_mask_a
    )
    policy_b_logps = compute_sequence_logprob(
        policy_b_outputs.logits, input_ids_b, attention_mask_b
    )
    reference_a_logps = compute_sequence_logprob(
        reference_a_outputs.logits, input_ids_a, attention_mask_a
    )
    reference_b_logps = compute_sequence_logprob(
        reference_b_outputs.logits, input_ids_b, attention_mask_b
    )

    # Compute implicit rewards
    reward_a = beta * (policy_a_logps - reference_a_logps)
    reward_b = beta * (policy_b_logps - reference_b_logps)

    # Compute Δ_i = r(x, y_a) - r(x, y_b)
    delta_i = reward_a - reward_b

    # Compute preference probabilities p_i = σ(Δ_i)
    p_i = torch.sigmoid(delta_i)

    # Compute PE structure loss
    pe_loss_fn = PELoss(
        epsilon=epsilon,
        distance=distance,
        detach_denominator=detach_denominator
    )
    loss, info = pe_loss_fn(p_i)

    return loss, info
