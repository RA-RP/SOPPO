"""Population-level preference encoding loss and exact distributed coefficients."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class PELoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8, distance: str = "l1", detach_denominator: bool = False):
        super().__init__()
        self.epsilon = float(epsilon)
        self.distance = distance
        self.detach_denominator = bool(detach_denominator)
        self.register_buffer("e_plus", torch.tensor([1.0, 0.0]))
        self.register_buffer("e_minus", torch.tensor([0.0, 1.0]))

    def forward(self, p_i: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        p_i = p_i.reshape(-1)
        if p_i.numel() < 2:
            raise ValueError("PE loss requires a population batch of at least two")
        q_i = torch.stack([p_i, 1.0 - p_i], dim=-1)
        sum_p = p_i.sum()
        sum_not_p = (1.0 - p_i).sum()
        denom_plus = sum_p + self.epsilon
        denom_minus = sum_not_p + self.epsilon
        if self.detach_denominator:
            denom_plus = denom_plus.detach()
            denom_minus = denom_minus.detach()
        e_hat_plus = (p_i[:, None] * q_i).sum(0) / denom_plus
        e_hat_minus = ((1.0 - p_i)[:, None] * q_i).sum(0) / denom_minus
        if self.distance == "l1":
            dist_plus = (e_hat_plus - self.e_plus).abs().sum()
            dist_minus = (e_hat_minus - self.e_minus).abs().sum()
        elif self.distance == "l2":
            dist_plus = torch.linalg.vector_norm(e_hat_plus - self.e_plus)
            dist_minus = torch.linalg.vector_norm(e_hat_minus - self.e_minus)
        else:
            raise ValueError(f"Unsupported PE distance: {self.distance}")
        loss = 0.5 * (dist_plus + dist_minus)
        info = {
            "loss": float(loss.detach()),
            "e_hat_plus": [float(value) for value in e_hat_plus.detach()],
            "e_hat_minus": [float(value) for value in e_hat_minus.detach()],
            "sum_p": float(sum_p.detach()),
            "sum_1_minus_p": float(sum_not_p.detach()),
            "p_mean": float(p_i.mean().detach()),
            "p_std": float(p_i.std(unbiased=False).detach()),
        }
        return loss, info


def exact_global_pe_coefficients(
    local_probabilities: torch.Tensor,
    epsilon: float,
    distance: str,
    detach_denominator: bool,
) -> Tuple[torch.Tensor, float, Dict]:
    """Return dL_PE/dp for this rank using the exact global optimizer batch.

    The first pass supplies detached probabilities for every local microbatch.
    All ranks gather equal-sized vectors, differentiate the dense population loss
    on a tiny leaf tensor, and take their own slice. The training second pass
    multiplies the local surrogate by ``world_size`` because DDP averages
    parameter gradients across ranks.
    """
    local = local_probabilities.detach().float().reshape(-1)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    if dist.is_initialized():
        dist.all_gather(gathered, local)
    else:
        gathered[0].copy_(local)
    global_leaf = torch.cat(gathered).detach().requires_grad_(True)
    loss, info = PELoss(epsilon, distance, detach_denominator).to(global_leaf.device)(global_leaf)
    gradient = torch.autograd.grad(loss, global_leaf, retain_graph=False)[0]
    start = rank * local.numel()
    local_gradient = gradient[start : start + local.numel()].detach()
    return local_gradient, float(loss.detach()), info


def pe_surrogate(
    live_probabilities: torch.Tensor,
    coefficients: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    if live_probabilities.numel() != coefficients.numel():
        raise ValueError("PE second-pass probability/coefficient size mismatch")
    return float(world_size) * (live_probabilities.reshape(-1) * coefficients).sum()
