"""
Pseudo-target baseline: instance-level hard pseudo labels.

This baseline generates hard pseudo labels z̃_i = 1[p_i > 0.5] for unlabeled pairs
and treats them as ground truth for DPO training.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple


class PseudoTargetLoss(nn.Module):
    """
    Pseudo-target loss using hard instance-level labels.

    For each unlabeled pair (x_i, y_a, y_b):
    1. Compute p_i = σ(r_θ(x_i, y_a) - r_θ(x_i, y_b))
    2. Generate hard label: z̃_i = 1[p_i > 0.5]
    3. If z̃_i = 1, treat (y_w=y_a, y_l=y_b); else (y_w=y_b, y_l=y_a)
    4. Compute standard DPO loss with pseudo labels
    """

    def __init__(self, beta: float = 0.1, threshold: float = 0.5):
        """
        Args:
            beta: DPO temperature parameter
            threshold: Threshold for hard pseudo label (default 0.5)
        """
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(
        self,
        policy_a_logps: torch.Tensor,
        policy_b_logps: torch.Tensor,
        reference_a_logps: torch.Tensor,
        reference_b_logps: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute pseudo-target loss.

        Args:
            policy_a_logps: Log probs for response_a under policy
            policy_b_logps: Log probs for response_b under policy
            reference_a_logps: Log probs for response_a under reference
            reference_b_logps: Log probs for response_b under reference

        Returns:
            loss: Pseudo-target DPO loss
            info: Diagnostic information including pseudo label statistics
        """
        # Compute implicit rewards
        reward_a = self.beta * (policy_a_logps - reference_a_logps)
        reward_b = self.beta * (policy_b_logps - reference_b_logps)

        # Compute preference probabilities
        delta = reward_a - reward_b
        p_i = torch.sigmoid(delta)

        # Generate hard pseudo labels
        pseudo_labels = (p_i > self.threshold).float()  # 1 if a > b, 0 otherwise

        # Compute logits based on pseudo labels
        # If pseudo_label = 1: chosen=a, rejected=b, logit = reward_a - reward_b
        # If pseudo_label = 0: chosen=b, rejected=a, logit = reward_b - reward_a
        logits = torch.where(
            pseudo_labels > 0.5,
            reward_a - reward_b,  # a is chosen
            reward_b - reward_a   # b is chosen
        )

        # DPO loss: -log σ(logits)
        loss = -torch.nn.functional.logsigmoid(logits).mean()

        # Diagnostics
        probs = torch.sigmoid(logits)
        pseudo_label_ratio = pseudo_labels.mean()

        info = {
            'loss': loss.item(),
            'pseudo_label_ratio': pseudo_label_ratio.item(),  # ratio of a > b
            'p_i_mean': p_i.mean().item(),
            'p_i_std': p_i.std().item(),
            'p_i_min': p_i.min().item(),
            'p_i_max': p_i.max().item(),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'accuracy': (probs > 0.5).float().mean().item(),
        }

        return loss, info


def compute_pseudo_target_loss(
    policy_model,
    reference_model,
    input_ids_a: torch.Tensor,
    attention_mask_a: torch.Tensor,
    input_ids_b: torch.Tensor,
    attention_mask_b: torch.Tensor,
    beta: float = 0.1,
    threshold: float = 0.5
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute pseudo-target loss from models and unlabeled inputs.

    Args:
        policy_model: Policy model being trained
        reference_model: Frozen reference model
        input_ids_a: Input IDs for response_a
        attention_mask_a: Attention mask for response_a
        input_ids_b: Input IDs for response_b
        attention_mask_b: Attention mask for response_b
        beta: DPO temperature parameter
        threshold: Threshold for hard pseudo labels

    Returns:
        loss: Pseudo-target loss
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

    # Compute pseudo-target loss
    pt_loss_fn = PseudoTargetLoss(beta=beta, threshold=threshold)
    loss, info = pt_loss_fn(
        policy_a_logps,
        policy_b_logps,
        reference_a_logps,
        reference_b_logps
    )

    return loss, info
