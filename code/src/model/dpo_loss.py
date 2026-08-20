"""
Standard DPO (Direct Preference Optimization) loss.

Reference: Rafailov et al., "Direct Preference Optimization" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DPOLoss(nn.Module):
    """
    DPO loss for preference learning.

    L_DPO = -E[log σ(β * (log π_θ(y_w|x) - log π_ref(y_w|x)
                           - log π_θ(y_l|x) + log π_ref(y_l|x)))]

    where:
    - y_w: preferred (chosen) response
    - y_l: dis-preferred (rejected) response
    - β: temperature parameter controlling preference strength
    """

    def __init__(self, beta: float = 0.1):
        """
        Args:
            beta: Temperature parameter (default 0.1)
        """
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute DPO loss.

        Args:
            policy_chosen_logps: Log probabilities of chosen responses under policy
            policy_rejected_logps: Log probabilities of rejected responses under policy
            reference_chosen_logps: Log probabilities of chosen responses under reference
            reference_rejected_logps: Log probabilities of rejected responses under reference

        Returns:
            loss: Scalar loss
            info: Dictionary with diagnostic information
        """
        # Compute implicit rewards
        policy_chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        policy_rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        # Compute preference logits
        logits = policy_chosen_rewards - policy_rejected_rewards

        # DPO loss: -log σ(logits)
        loss = -F.logsigmoid(logits).mean()

        # Compute preference probabilities for diagnostics
        probs = torch.sigmoid(logits)

        info = {
            'loss': loss.item(),
            'logits_mean': logits.mean().item(),
            'logits_std': logits.std().item(),
            'probs_mean': probs.mean().item(),
            'probs_std': probs.std().item(),
            'accuracy': (probs > 0.5).float().mean().item(),
            'policy_chosen_rewards_mean': policy_chosen_rewards.mean().item(),
            'policy_rejected_rewards_mean': policy_rejected_rewards.mean().item(),
        }

        return loss, info


def compute_dpo_loss(
    policy_model,
    reference_model,
    input_ids_chosen: torch.Tensor,
    attention_mask_chosen: torch.Tensor,
    input_ids_rejected: torch.Tensor,
    attention_mask_rejected: torch.Tensor,
    beta: float = 0.1
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute DPO loss from models and inputs.

    Args:
        policy_model: Policy model being trained
        reference_model: Frozen reference model
        input_ids_chosen: Input IDs for chosen responses
        attention_mask_chosen: Attention mask for chosen responses
        input_ids_rejected: Input IDs for rejected responses
        attention_mask_rejected: Attention mask for rejected responses
        beta: Temperature parameter

    Returns:
        loss: Scalar loss
        info: Diagnostic information
    """
    # Get logits from policy model
    with torch.cuda.amp.autocast():
        policy_chosen_outputs = policy_model(
            input_ids=input_ids_chosen,
            attention_mask=attention_mask_chosen,
            return_dict=True
        )
        policy_rejected_outputs = policy_model(
            input_ids=input_ids_rejected,
            attention_mask=attention_mask_rejected,
            return_dict=True
        )

    # Get logits from reference model (no gradient)
    with torch.no_grad():
        reference_chosen_outputs = reference_model(
            input_ids=input_ids_chosen,
            attention_mask=attention_mask_chosen,
            return_dict=True
        )
        reference_rejected_outputs = reference_model(
            input_ids=input_ids_rejected,
            attention_mask=attention_mask_rejected,
            return_dict=True
        )

    # Compute log probabilities
    policy_chosen_logps = compute_sequence_logprob(
        policy_chosen_outputs.logits, input_ids_chosen, attention_mask_chosen
    )
    policy_rejected_logps = compute_sequence_logprob(
        policy_rejected_outputs.logits, input_ids_rejected, attention_mask_rejected
    )
    reference_chosen_logps = compute_sequence_logprob(
        reference_chosen_outputs.logits, input_ids_chosen, attention_mask_chosen
    )
    reference_rejected_logps = compute_sequence_logprob(
        reference_rejected_outputs.logits, input_ids_rejected, attention_mask_rejected
    )

    # Compute DPO loss
    dpo_loss_fn = DPOLoss(beta=beta)
    loss, info = dpo_loss_fn(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps
    )

    return loss, info


def compute_sequence_logprob(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probability of a sequence.

    Args:
        logits: Model logits [batch_size, seq_len, vocab_size]
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]

    Returns:
        log_probs: Log probability for each sequence [batch_size]
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Compute log probabilities
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for the actual tokens
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding tokens and sum
    masked_log_probs = token_log_probs * shift_mask
    sequence_log_probs = masked_log_probs.sum(dim=-1)

    return sequence_log_probs
