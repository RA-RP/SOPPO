"""CPU regressions for the strict Q1 support-mixture implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict


REPO = Path("/root/LLM-output-density")
SCRIPTS = REPO / "experiments/opd_sft_h1/scripts"
VERL = Path("/root/autodl-tmp/verl")
for path in (str(SCRIPTS), str(VERL)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cycle09_q1_mixture_runtime import (  # noqa: E402
    FrozenExternalStore,
    build_external_teacher_tensors,
)
from verl.trainer.distillation.losses import _q1_source_equal_loss  # noqa: E402
from verl.utils import tensordict_utils as tu  # noqa: E402


def _data(source_ids: torch.Tensor, counts: dict[str, int]) -> TensorDict:
    data = TensorDict(
        {"support_source_id": source_ids.clone()},
        batch_size=[len(source_ids)],
    )
    tu.assign_non_tensor(data, dp_size=1, **counts)
    return data


def test_source_equal_loss_is_microbatch_invariant() -> None:
    source_ids = torch.tensor([0] * 8 + [1] * 8, dtype=torch.int64)
    response_mask = torch.zeros((16, 6), dtype=torch.bool)
    for row in range(16):
        response_mask[row, : 1 + (row % 6)] = True
    losses = torch.arange(16 * 6, dtype=torch.float32).reshape(16, 6) / 10
    external_tokens = int(response_mask[:8].sum())
    self_tokens = int(response_mask[8:].sum())
    counts = {
        "q1_external_sample_count": 8,
        "q1_self_sample_count": 8,
        "q1_external_token_count": external_tokens,
        "q1_self_token_count": self_tokens,
    }

    full, metrics = _q1_source_equal_loss(
        losses, response_mask, _data(source_ids, counts)
    )
    parts = []
    for selection in (slice(0, 3), slice(3, 11), slice(11, 16)):
        part, _ = _q1_source_equal_loss(
            losses[selection],
            response_mask[selection],
            _data(source_ids[selection], counts),
        )
        parts.append(part)
    expected = 0.5 * losses[:8].masked_select(response_mask[:8]).mean()
    expected += 0.5 * losses[8:].masked_select(response_mask[8:]).mean()
    assert torch.allclose(full, expected)
    assert torch.allclose(sum(parts), full)
    assert metrics["distillation/source_external_sample_share"].aggregate() == 0.5
    assert metrics["distillation/source_self_sample_share"].aggregate() == 0.5


def test_source_equal_loss_differs_from_pooled_token_mean() -> None:
    source_ids = torch.tensor([0] * 8 + [1] * 8)
    response_mask = torch.zeros((16, 10), dtype=torch.bool)
    response_mask[:8, :1] = True
    response_mask[8:, :] = True
    losses = torch.zeros((16, 10))
    losses[:8, 0] = 10.0
    counts = {
        "q1_external_sample_count": 8,
        "q1_self_sample_count": 8,
        "q1_external_token_count": 8,
        "q1_self_token_count": 80,
    }
    strict, _ = _q1_source_equal_loss(
        losses, response_mask, _data(source_ids, counts)
    )
    pooled = losses.masked_select(response_mask).mean()
    assert torch.isclose(strict, torch.tensor(5.0))
    assert not torch.isclose(strict, pooled)


def test_external_teacher_alignment() -> None:
    prompt = np.asarray([11, 12, 13], dtype=np.int32)
    response = np.asarray([21, 22], dtype=np.int32)
    topk_ids = np.arange(8, dtype=np.int32).reshape(2, 4)
    topk_logprobs = -np.arange(8, dtype=np.float32).reshape(2, 4)
    teacher_ids, teacher_logprobs = build_external_teacher_tensors(
        prompt, response, topk_ids, topk_logprobs
    )
    assert teacher_ids.shape == teacher_logprobs.shape == (5, 4)
    assert torch.equal(teacher_ids[2:4], torch.from_numpy(topk_ids))
    assert torch.equal(teacher_logprobs[2:4], torch.from_numpy(topk_logprobs))
    assert not teacher_ids[:2].any()
    assert not teacher_ids[4].any()


def test_frozen_store_mmap_roundtrip(tmp_path: Path) -> None:
    arrays = {
        "prompt_ids.npy": np.asarray([1, 2, 3], dtype=np.int32),
        "prompt_offsets.npy": np.asarray([0, 2, 3], dtype=np.int64),
        "response_ids.npy": np.asarray([4, 5, 6], dtype=np.int32),
        "response_offsets.npy": np.asarray([0, 1, 3], dtype=np.int64),
        "row_offsets.npy": np.asarray([[0, 1], [1, 3]], dtype=np.int64),
        "top32_ids.npy": np.arange(12, dtype=np.int32).reshape(3, 4),
        "top32_logprob.npy": -np.arange(12, dtype=np.float32).reshape(3, 4),
    }
    for name, value in arrays.items():
        np.save(tmp_path / name, value, allow_pickle=False)
    store = FrozenExternalStore(tmp_path)
    prompt, response, ids, logprobs = store.sample(1)
    assert prompt.tolist() == [3]
    assert response.tolist() == [5, 6]
    assert ids.shape == logprobs.shape == (2, 4)
    assert isinstance(store.top32_ids, np.memmap)
