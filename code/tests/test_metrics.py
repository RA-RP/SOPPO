import math

import torch

from src.evaluation.metrics import compute_calibration, compute_confidence_distribution
from src.training.trainer import PatternBatchSampler


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
