import torch

from src.model.dpo_loss import DPOLoss, compute_response_mean_logprob, compute_sequence_logprob
from src.model.pe_loss import PELoss, exact_global_pe_coefficients, pe_surrogate
from src.model.sspo_loss import (
    SSPOThresholdState,
    gamma_weight,
    hard_pseudo_response_losses,
    objective_weights,
    pe_pair_probabilities,
    simpo_pair_losses,
)


def test_response_mask_excludes_prompt_tokens_and_mean_uses_response_length():
    logits = torch.zeros(1, 4, 5)
    ids = torch.tensor([[0, 1, 2, 3]])
    mask = torch.tensor([[0, 0, 1, 1]])
    summed = compute_sequence_logprob(logits, ids, mask)
    mean = compute_response_mean_logprob(logits, ids, mask)
    assert torch.allclose(summed, -2.0 * torch.log(torch.tensor([5.0])))
    assert torch.allclose(mean, -torch.log(torch.tensor([5.0])))


def test_dpo_respects_randomized_a_b_labels():
    objective = DPOLoss(beta=1.0)
    a = torch.tensor([2.0, 2.0])
    b = torch.tensor([0.0, 0.0])
    zeros = torch.zeros(2)
    loss, info = objective(a, b, zeros, zeros, torch.tensor([1, 0]))
    assert torch.isfinite(loss)
    assert info["accuracy"] == 0.5


def test_simpo_margin_and_margin_free_pe_probability_are_position_symmetric():
    a = torch.tensor([-1.0, -3.0])
    b = torch.tensor([-3.0, -1.0])
    losses, delta = simpo_pair_losses(a, b, torch.tensor([1, 0]), beta=10, margin=2)
    assert torch.allclose(losses[0], losses[1])
    probability = pe_pair_probabilities(a, b, beta=10)
    assert torch.allclose(probability[0], 1.0 - probability[1])
    assert torch.allclose(delta, torch.tensor([20.0, -20.0]))


def test_exp_and_normalized_static_objective_weights():
    exp = {"weighting": "exponential_gamma", "gamma0": 1.0, "gamma_min": 0.1, "gamma_decay": 0.01}
    assert objective_weights(exp, 0) == (1.0, 0.0)
    assert gamma_weight(10000, 1.0, 0.1, 0.01) == 0.1
    fixed = {"weighting": "normalized_fixed_lambda", "fixed_lambda": 1.0}
    assert objective_weights(fixed, 10) == (0.5, 0.5)


def test_scott_kde_state_separates_rewards_and_hard_loss_is_finite():
    state = SSPOThresholdState(momentum=0.95, prior=0.5, epsilon=1e-6, grid_points=200)
    winning = torch.tensor([1.5, 1.8, 2.0, 2.2])
    losing = torch.tensor([-2.1, -1.9, -1.7, -1.4])
    all_rewards = torch.cat([winning, losing, torch.tensor([-0.5, 0.5])])
    info = state.update(all_rewards, winning, losing)
    assert -1.0 < info["threshold_ema"] < 1.0
    live = torch.tensor([-2.0, 2.0], requires_grad=True)
    losses, hard = hard_pseudo_response_losses(live, state)
    assert torch.isfinite(losses).all()
    assert hard["pseudo_positive_rate"] == 0.5
    losses.mean().backward()
    assert torch.isfinite(live.grad).all()


def test_two_pass_pe_matches_dense_autograd():
    dense = torch.tensor([0.91, 0.77, 0.63, 0.42, 0.24, 0.08], requires_grad=True)
    loss, _ = PELoss(epsilon=1e-8)(dense)
    dense_gradient = torch.autograd.grad(loss, dense)[0]
    coefficients, reported_loss, _ = exact_global_pe_coefficients(dense.detach(), 1e-8, "l1", False)
    live = dense.detach().clone().requires_grad_(True)
    surrogate = pe_surrogate(live, coefficients, world_size=1)
    surrogate_gradient = torch.autograd.grad(surrogate, live)[0]
    assert abs(reported_loss - float(loss)) < 1e-7
    assert torch.allclose(surrogate_gradient, dense_gradient, atol=1e-6, rtol=1e-5)

    subbatched = dense.detach().clone().requires_grad_(True)
    for start in range(subbatched.numel()):
        pe_surrogate(
            subbatched[start : start + 1],
            coefficients[start : start + 1],
            world_size=1,
        ).backward()
    assert torch.allclose(subbatched.grad, dense_gradient, atol=1e-6, rtol=1e-5)


def test_half_probability_pe_is_finite_and_symmetric():
    values = torch.full((8,), 0.5, requires_grad=True)
    loss, _ = PELoss(epsilon=1e-8)(values)
    gradient = torch.autograd.grad(loss, values)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert torch.allclose(gradient, torch.zeros_like(gradient), atol=1e-7)
