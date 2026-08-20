#!/usr/bin/env python3
"""Runtime dataset and frozen-external agent loop for the Q1 alpha=.5 arm."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)
from verl.utils.dataset.rl_dataset import RLHFDataset


@lru_cache(maxsize=4)
def _load_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


class FrozenExternalStore:
    """Memory-map frozen rollout tokens and RAW top-k rows in each agent worker."""

    REQUIRED = (
        "prompt_ids.npy",
        "prompt_offsets.npy",
        "response_ids.npy",
        "response_offsets.npy",
        "top32_ids.npy",
        "top32_logprob.npy",
        "row_offsets.npy",
    )

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        missing = [name for name in self.REQUIRED if not (self.root / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete Q1 frozen store {self.root}: {missing}")
        for name in self.REQUIRED:
            setattr(self, name.removesuffix(".npy"), _load_array(str(self.root / name)))
        if self.row_offsets.ndim == 2 and self.row_offsets.shape[1] == 2:
            source_starts = self.row_offsets[:, 0]
            source_ends = self.row_offsets[:, 1]
        elif self.row_offsets.ndim == 1:
            source_starts = self.row_offsets[:-1]
            source_ends = self.row_offsets[1:]
        else:
            raise RuntimeError(f"unsupported frozen row_offsets shape: {self.row_offsets.shape}")
        if not (
            np.array_equal(self.response_offsets[:-1], source_starts)
            and np.array_equal(self.response_offsets[1:], source_ends)
        ):
            raise RuntimeError("response offsets do not match frozen top-32 row offsets")
        if self.top32_ids.shape != self.top32_logprob.shape:
            raise RuntimeError("frozen top-32 id/logprob shape mismatch")
        if int(source_ends[-1]) != int(self.top32_ids.shape[0]):
            raise RuntimeError("frozen top-32 row count mismatch")

    def sample(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        index = int(index)
        if index < 0 or index + 1 >= len(self.prompt_offsets):
            raise IndexError(f"external record index out of range: {index}")
        p0, p1 = map(int, self.prompt_offsets[index : index + 2])
        r0, r1 = map(int, self.response_offsets[index : index + 2])
        return (
            self.prompt_ids[p0:p1],
            self.response_ids[r0:r1],
            self.top32_ids[r0:r1],
            self.top32_logprob[r0:r1],
        )


class Q1MixtureDataset(RLHFDataset):
    """RLHFDataset that keeps source identity as a collated tensor."""

    def __getitem__(self, item):
        row = super().__getitem__(item)
        source_id = int(row.pop("support_source_id"))
        if source_id not in (0, 1):
            raise ValueError(f"invalid support_source_id={source_id}")
        row["support_source_id"] = torch.tensor(source_id, dtype=torch.int64)
        return row


def build_external_teacher_tensors(
    prompt_ids: np.ndarray,
    response_ids: np.ndarray,
    response_topk_ids: np.ndarray,
    response_topk_logprobs: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align frozen response distributions to VERL's next-token sequence rows."""

    prompt_len = int(len(prompt_ids))
    response_len = int(len(response_ids))
    if prompt_len <= 0 or response_len <= 0:
        raise ValueError(f"empty frozen sequence: prompt={prompt_len}, response={response_len}")
    if response_topk_ids.shape != response_topk_logprobs.shape:
        raise ValueError("top-k id/logprob shape mismatch")
    if response_topk_ids.ndim != 2 or response_topk_ids.shape[0] != response_len:
        raise ValueError("top-k rows must align one-for-one with response tokens")

    total_len = prompt_len + response_len
    topk = int(response_topk_ids.shape[1])
    teacher_ids = torch.zeros((total_len, topk), dtype=torch.int32)
    teacher_logprobs = torch.zeros((total_len, topk), dtype=torch.float32)
    start = prompt_len - 1
    end = start + response_len
    teacher_ids[start:end] = torch.from_numpy(
        np.asarray(response_topk_ids, dtype=np.int32).copy()
    )
    teacher_logprobs[start:end] = torch.from_numpy(
        np.asarray(response_topk_logprobs, dtype=np.float32).copy()
    )
    return teacher_ids, teacher_logprobs


class FrozenExternalAgentLoop(AgentLoopBase):
    """Return frozen off-KD text and top-32 distributions without a live teacher call."""

    def __init__(self, *args, frozen_root: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = FrozenExternalStore(frozen_root)

    async def run(
        self,
        sampling_params: dict[str, Any],
        external_record_index: int,
        **kwargs,
    ):
        del sampling_params
        # Unlike a live vLLM rollout, this frozen trajectory has no server
        # response carrying a model-version tag.  Preserve the trainer's
        # enqueue step so v1's off-policy staleness metrics remain well-defined.
        global_steps = int(kwargs.get("global_steps", 0))
        prompt, response, topk_ids, topk_logprobs = self.store.sample(int(external_record_index))
        if len(prompt) > self.rollout_config.prompt_length:
            raise RuntimeError(f"frozen prompt exceeds cap: {len(prompt)}")
        if len(response) > self.rollout_config.response_length:
            response = response[: self.rollout_config.response_length]
            topk_ids = topk_ids[: self.rollout_config.response_length]
            topk_logprobs = topk_logprobs[: self.rollout_config.response_length]
        teacher_ids, teacher_logprobs = build_external_teacher_tensors(
            prompt, response, topk_ids, topk_logprobs
        )
        return AgentLoopOutput(
            prompt_ids=np.asarray(prompt, dtype=np.int64).tolist(),
            response_ids=np.asarray(response, dtype=np.int64).tolist(),
            response_mask=[1] * len(response),
            num_turns=2,
            metrics=AgentLoopMetrics(generate_sequences=0.0, num_preempted=0),
            extra_fields={
                "teacher_ids": teacher_ids,
                "teacher_logprobs": teacher_logprobs,
                "turn_scores": [],
                "tool_rewards": [],
                "q1_frozen_external": True,
                "min_global_steps": global_steps,
                "max_global_steps": global_steps,
            },
        )
