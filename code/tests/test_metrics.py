import math

import torch

from src.evaluation.metrics import compute_calibration, compute_confidence_distribution
from src.training.trainer import PatternBatchSampler, split_cpu_batch


def test_calibration_includes_probability_one_and_stays_finite():
    probabilities = torch.tensor([0.0, 0.5, 1.0])
    labels = torch.tensor([0, 1, 1])
    calibration = compute_calibration(probabilities, labels)
    assert sum(cell["count"] for cell in calibration["bins"]) == 3
    assert math.isfinite(calibration["ece"])
    confidence = compute_confidence_distribution(probabilities)
    assert math.isfinite(confidence["mean_entropy"])


def test_pattern_sampler_preserves_exact_joint_microbatch_shape():
    sampler = list(range(56))
    pattern = [3, 4, 3, 4, 3, 4, 3, 4]
    batches = list(PatternBatchSampler(sampler, pattern))
    assert [len(batch) for batch in batches] == pattern * 2
    assert sorted(index for batch in batches for index in batch) == sampler


def test_backward_subbatching_preserves_order_and_every_field():
    batch = {
        "sample_ids": ["a", "b", "c"],
        "input_ids_a": torch.arange(12).reshape(3, 4),
        "labels": torch.tensor([1, 0, 1]),
        "ref_logp_a": torch.tensor([-1.0, -2.0, -3.0]),
    }
    pieces = split_cpu_batch(batch, maximum_size=1)
    assert [piece["sample_ids"][0] for piece in pieces] == ["a", "b", "c"]
    assert torch.equal(torch.cat([piece["input_ids_a"] for piece in pieces]), batch["input_ids_a"])
    assert torch.equal(torch.cat([piece["labels"] for piece in pieces]), batch["labels"])
    assert torch.equal(torch.cat([piece["ref_logp_a"] for piece in pieces]), batch["ref_logp_a"])
