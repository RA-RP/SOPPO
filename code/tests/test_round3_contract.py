"""CPU contract tests; execute only in the authorized Round3 server stage."""

import math
import copy

import torch

from src.round3.losses import (
    GitHubSSPOState,
    github_sspo_objective,
    joint_dpo_pe_objective,
    pe_objective,
    rollout_anchor_statistics,
)
from src.round3.queue_protocol import route_replica, rollout_seed
from src.round3.data import VIEW_COUNTS, _paired_record, _unpaired_record
from src.round3.rollout_worker import _round3_prompt_token_ids
from src.round3.trainer import _pair_vjp_surrogate


class _PromptTokenizerStub:
    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "prompt"}]
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "templated"

    def __call__(self, text, **kwargs):
        assert text == "templated"
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": list(range(1100))}


def test_rollout_prompt_ids_match_training_left_truncation_contract():
    effective, raw_count = _round3_prompt_token_ids(_PromptTokenizerStub(), "prompt")
    assert raw_count == 1100
    assert effective == list(range(76, 1100))


def test_malformed_source_rows_are_quarantined_without_text_or_exception():
    revision = "a" * 40
    pair = {
        "prompt_id": "pair-1",
        "prompt": "question",
        "chosen": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "rejected": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "   "},
        ],
    }
    record, audit = _paired_record(
        "HuggingFaceH4/ultrafeedback_binarized", revision, "test_prefs", pair, 374
    )
    assert record is None
    assert audit["reason_codes"] == ["empty_rejected"]
    assert "prompt" not in audit and "chosen" not in audit and "rejected" not in audit

    single = {
        "id": "single-1",
        "prompt": "   ",
        "messages": [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "answer"},
        ],
    }
    record, audit = _unpaired_record(
        "HuggingFaceH4/ultrachat_200k", revision, "train_sft", single, 0
    )
    assert record is None
    assert audit["reason_codes"] == ["empty_prompt", "message0_prompt_mismatch"]
    assert audit["canonical_prompt_sha256"] is None
    assert VIEW_COUNTS["test"] == 997


def test_github_sspo_sequential_initialization_and_no_threshold_ema():
    state = GitHubSSPOState()
    chosen = torch.tensor([-2.0, -1.0, 0.0, 1.0], requires_grad=True)
    rejected = torch.tensor([-3.0, -2.0, -1.0, 0.0], requires_grad=True)
    unpaired = torch.linspace(-3.0, 2.0, 28, requires_grad=True)
    loss, info = github_sspo_objective(chosen, rejected, unpaired, state, global_step=0)
    assert math.isfinite(float(loss))
    assert info["gamma"] == 1.0
    assert "threshold_ema" not in info
    assert state.running_mean is not None and state.running_var is not None
    loss.backward()
    assert chosen.grad is not None and rejected.grad is not None and unpaired.grad is not None


def test_sspo_state_roundtrip_is_fail_closed():
    state = GitHubSSPOState(running_mean=-1.0, running_var=0.25)
    restored = GitHubSSPOState.from_state_dict(state.state_dict())
    assert restored.state_dict() == state.state_dict()
    malformed = state.state_dict()
    del malformed["running_var"]
    try:
        GitHubSSPOState.from_state_dict(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing SSPO running state must fail closed")


def test_exact_28_pair_pe_and_normalized_joint_weight():
    score_a = torch.linspace(-2.0, 1.0, 28, requires_grad=True)
    score_b = torch.linspace(-1.5, 0.5, 28, requires_grad=True)
    pe, _ = pe_objective(score_a, score_b)
    joint = joint_dpo_pe_objective(torch.tensor(2.0), pe, 0.1)
    assert torch.allclose(joint, (torch.tensor(2.0) + 0.1 * pe) / 1.1)
    joint.backward()
    assert torch.isfinite(score_a.grad).all() and torch.isfinite(score_b.grad).all()


def test_dpo_reward_pe_is_exactly_half_when_policy_equals_reference():
    policy_a = torch.linspace(-300.0, -10.0, 28, requires_grad=True)
    policy_b = torch.linspace(-250.0, -20.0, 28, requires_grad=True)
    reference_a = policy_a.detach().clone()
    reference_b = policy_b.detach().clone()
    pe, info = pe_objective(
        policy_a - reference_a,
        policy_b - reference_b,
        beta=0.1,
    )
    assert info["p_mean"] == 0.5
    assert info["p_std"] == 0.0
    assert math.isfinite(float(pe))


def test_dpo_reward_pe_uses_total_reference_logratio_and_can_favor_b():
    policy_a = torch.full((28,), -120.0, requires_grad=True)
    policy_b = torch.full((28,), -90.0, requires_grad=True)
    reference_a = torch.full((28,), -100.0)
    reference_b = torch.full((28,), -100.0)
    pe, info = pe_objective(
        policy_a - reference_a,
        policy_b - reference_b,
        beta=0.1,
    )
    # A has the lower policy/reference log-ratio, so p(A>B) must be below 0.5.
    assert info["p_mean"] < 0.05
    pe.backward()
    assert torch.isfinite(policy_a.grad).all()
    assert torch.isfinite(policy_b.grad).all()


def test_pair_vjp_surrogate_uses_each_total_and_mean_coefficient_once():
    totals = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    means = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    total_coefficients = torch.tensor([0.1, 0.2, 0.3])
    mean_coefficients = torch.tensor([-0.4, -0.5, -0.6])
    surrogate = _pair_vjp_surrogate(
        totals, means, total_coefficients, mean_coefficients
    )
    surrogate.backward()
    assert torch.equal(totals.grad, total_coefficients)
    assert torch.equal(means.grad, mean_coefficients)


def test_pair_vjp_matches_direct_full_population_for_both_reward_profiles():
    features_a = torch.arange(56, dtype=torch.float32).reshape(28, 2) / 50.0
    features_b = torch.flip(features_a, dims=(0,)) + torch.tensor([0.2, -0.1])
    lengths_a = torch.arange(1, 29, dtype=torch.float32)
    lengths_b = torch.arange(29, 57, dtype=torch.float32)
    reference_a = torch.linspace(-1.0, 1.0, 28)
    reference_b = torch.linspace(0.5, -0.5, 28)

    for reward_type in ("simpo_mean_logp", "dpo_reference_logratio_total"):
        direct_weight = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        direct_total_a = features_a @ direct_weight
        direct_total_b = features_b @ direct_weight
        direct_mean_a = direct_total_a / lengths_a
        direct_mean_b = direct_total_b / lengths_b
        if reward_type == "simpo_mean_logp":
            direct_loss, _ = pe_objective(direct_mean_a, direct_mean_b, beta=10.0)
        else:
            direct_loss, _ = pe_objective(
                direct_total_a - reference_a,
                direct_total_b - reference_b,
                beta=0.1,
            )
        direct_loss.backward()
        expected_gradient = direct_weight.grad.detach().clone()

        vjp_weight = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        detached_total_a = (features_a @ vjp_weight).detach()
        detached_total_b = (features_b @ vjp_weight).detach()
        detached_mean_a = detached_total_a / lengths_a
        detached_mean_b = detached_total_b / lengths_b
        if reward_type == "simpo_mean_logp":
            leaf_a = detached_mean_a.requires_grad_(True)
            leaf_b = detached_mean_b.requires_grad_(True)
            leaf_loss, _ = pe_objective(leaf_a, leaf_b, beta=10.0)
            mean_coeff_a, mean_coeff_b = torch.autograd.grad(
                leaf_loss, (leaf_a, leaf_b)
            )
            total_coeff_a = torch.zeros_like(mean_coeff_a)
            total_coeff_b = torch.zeros_like(mean_coeff_b)
        else:
            leaf_a = detached_total_a.requires_grad_(True)
            leaf_b = detached_total_b.requires_grad_(True)
            leaf_loss, _ = pe_objective(
                leaf_a - reference_a,
                leaf_b - reference_b,
                beta=0.1,
            )
            total_coeff_a, total_coeff_b = torch.autograd.grad(
                leaf_loss, (leaf_a, leaf_b)
            )
            mean_coeff_a = torch.zeros_like(total_coeff_a)
            mean_coeff_b = torch.zeros_like(total_coeff_b)

        live_total_a = features_a @ vjp_weight
        live_total_b = features_b @ vjp_weight
        surrogate = _pair_vjp_surrogate(
            live_total_a,
            live_total_a / lengths_a,
            total_coeff_a,
            mean_coeff_a,
        ) + _pair_vjp_surrogate(
            live_total_b,
            live_total_b / lengths_b,
            total_coeff_b,
            mean_coeff_b,
        )
        surrogate.backward()
        assert torch.allclose(vjp_weight.grad, expected_gradient, atol=1e-6, rtol=1e-6)


def test_rollout_anchor_telemetry_is_source_aligned_after_ab_swaps():
    score_a = torch.tensor([1.0, -1.0] * 14)
    score_b = -score_a
    rollout_is_a = torch.tensor([True, False] * 14)
    info = rollout_anchor_statistics(score_a, score_b, rollout_is_a)
    assert info["comparisons"] == 28
    assert info["rollout_hard_wins"] == 28
    assert info["sft_hard_wins"] == 0
    assert info["ties"] == 0
    assert info["rollout_hard_win_rate"] == 1.0
    assert info["rollout_soft_win_probability_mean"] > 0.999


def test_rollout_route_and_seed_are_stable_and_draw_specific():
    values = [
        (route_replica("dpo_pe_rollout_only", 7, "sample", draw), rollout_seed(42, 7, "sample", draw))
        for draw in (0, 1)
    ]
    assert values == [
        (route_replica("dpo_pe_rollout_only", 7, "sample", draw), rollout_seed(42, 7, "sample", draw))
        for draw in (0, 1)
    ]
    assert values[0][1] != values[1][1]
    for method in (
        "dpo_pe_dpo_reward_sft_rollout",
        "dpo_pe_dpo_reward_rollout_only",
    ):
        assert route_replica(method, 7, "sample", 0) in {0, 1}
        assert rollout_seed(42, 7, "sample", 0) == rollout_seed(
            42, 7, "sample", 0
        )


def test_sspo_next_batch_loss_state_and_adam_update_roundtrip():
    parameter = torch.nn.Parameter(torch.linspace(-2.0, 2.0, 36))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.0)
    initial_state = GitHubSSPOState(running_mean=-1.0, running_var=0.5)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    parameter_state = parameter.detach().clone()
    running_state = copy.deepcopy(initial_state.state_dict())

    results = []
    for _ in range(2):
        candidate = torch.nn.Parameter(parameter_state.clone())
        candidate_optimizer = torch.optim.AdamW([candidate], lr=1e-3, weight_decay=0.0)
        candidate_optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        candidate_state = GitHubSSPOState.from_state_dict(copy.deepcopy(running_state))
        candidate_optimizer.zero_grad(set_to_none=True)
        loss, _ = github_sspo_objective(
            candidate[:4], candidate[4:8], candidate[8:], candidate_state, global_step=17
        )
        loss.backward()
        candidate_optimizer.step()
        results.append((float(loss), candidate.detach().clone(), candidate_state.state_dict()))
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]
