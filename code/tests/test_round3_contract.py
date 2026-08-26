"""CPU contract tests; execute only in the authorized Round3 server stage."""

import math
import copy

import torch

from src.round3.losses import GitHubSSPOState, github_sspo_objective, joint_dpo_pe_objective, pe_objective
from src.round3.queue_protocol import route_replica, rollout_seed
from src.round3.data import VIEW_COUNTS, _paired_record, _unpaired_record


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
