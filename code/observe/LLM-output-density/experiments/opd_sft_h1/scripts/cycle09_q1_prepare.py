#!/usr/bin/env python3
"""Prepare the exact Q1 8-self/8-external schedule and mmap frozen teacher data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import cycle09_block3_common as c


FORMAL_PARQUET = c.Q1_DATA / "qwen_alpha05_schedule_624.parquet"
STAGE_B_PARQUET = c.Q1_DATA / "qwen_alpha05_schedule_320.parquet"
SMOKE_PARQUET = c.Q1_DATA / "qwen_alpha05_schedule_smoke32.parquet"
MANIFEST = c.Q1_DATA / "qwen_alpha05_prepare_manifest.json"
SOURCE_ROLLOUT = c.QWEN_OFFKD_ROOT / "rollout/teacher_rollout.jsonl"
SOURCE_TOP32 = c.QWEN_OFFKD_ROOT / "rollout/teacher_top32_logprob.npz"
FROZEN_MEMBERS = ("top32_ids.npy", "top32_logprob.npy", "row_offsets.npy")
USABLE_PROMPTS = 4992
BATCH_SIZE = 16
STAGE_A_STEPS = 160
STAGE_B_FINAL_STEP = 320


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSON at {path}:{line_number}") from exc


def _messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in list(value)
    ]


def _extract_npy_members(npz_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(npz_path) as archive:
        names = set(archive.namelist())
        missing = set(FROZEN_MEMBERS) - names
        if missing:
            raise RuntimeError(f"frozen top-32 archive missing members: {sorted(missing)}")
        for name in FROZEN_MEMBERS:
            target = destination / name
            if target.is_file():
                continue
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                with archive.open(name) as source, temporary.open("wb") as sink:
                    shutil.copyfileobj(source, sink, length=16 << 20)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)


def _scan_rollout(path: Path) -> tuple[list[dict[str, int]], dict[int, int]]:
    metadata: list[dict[str, int]] = []
    by_prompt_id: dict[int, int] = {}
    for ordinal, row in enumerate(_jsonl(path)):
        prompt_ids = row["prompt_token_ids"]
        response_ids = row["generation_token_ids"]
        prompt_id = int(row["index"])
        start = int(row["logprob_row_start"])
        end = int(row["logprob_row_end"])
        if end - start != len(response_ids):
            raise RuntimeError(f"frozen response/top-32 mismatch at record {ordinal}")
        if int(row["n_prompt_tokens"]) != len(prompt_ids) or int(row["n_tokens"]) != len(response_ids):
            raise RuntimeError(f"frozen token-count metadata mismatch at record {ordinal}")
        if prompt_id in by_prompt_id:
            raise RuntimeError(f"duplicate frozen prompt index: {prompt_id}")
        by_prompt_id[prompt_id] = ordinal
        metadata.append(
            {
                "prompt_length": len(prompt_ids),
                "response_length": len(response_ids),
                "row_start": start,
                "row_end": end,
            }
        )
    if len(metadata) != 5000:
        raise RuntimeError(f"expected 5000 frozen rollout rows, found {len(metadata)}")
    return metadata, by_prompt_id


def _write_flat_tokens(path: Path, metadata: list[dict[str, int]]) -> None:
    prompt_offsets = np.zeros(len(metadata) + 1, dtype=np.int64)
    response_offsets = np.zeros(len(metadata) + 1, dtype=np.int64)
    for index, row in enumerate(metadata):
        prompt_offsets[index + 1] = prompt_offsets[index] + row["prompt_length"]
        response_offsets[index + 1] = response_offsets[index] + row["response_length"]

    outputs = {
        "prompt_offsets.npy": prompt_offsets,
        "response_offsets.npy": response_offsets,
    }
    for name, value in outputs.items():
        target = path / name
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
        os.replace(temporary, target)

    prompt_target = path / "prompt_ids.npy"
    response_target = path / "response_ids.npy"
    if prompt_target.is_file() and response_target.is_file():
        return
    prompt_tmp = prompt_target.with_name(f".{prompt_target.name}.{os.getpid()}.tmp")
    response_tmp = response_target.with_name(f".{response_target.name}.{os.getpid()}.tmp")
    prompt_mm = np.lib.format.open_memmap(
        prompt_tmp, mode="w+", dtype=np.int32, shape=(int(prompt_offsets[-1]),)
    )
    response_mm = np.lib.format.open_memmap(
        response_tmp, mode="w+", dtype=np.int32, shape=(int(response_offsets[-1]),)
    )
    try:
        for index, row in enumerate(_jsonl(SOURCE_ROLLOUT)):
            p0, p1 = map(int, prompt_offsets[index : index + 2])
            r0, r1 = map(int, response_offsets[index : index + 2])
            prompt_mm[p0:p1] = np.asarray(row["prompt_token_ids"], dtype=np.int32)
            response_mm[r0:r1] = np.asarray(row["generation_token_ids"], dtype=np.int32)
        prompt_mm.flush()
        response_mm.flush()
        del prompt_mm, response_mm
        os.replace(prompt_tmp, prompt_target)
        os.replace(response_tmp, response_target)
    finally:
        prompt_tmp.unlink(missing_ok=True)
        response_tmp.unlink(missing_ok=True)


def _verify_frozen_store(metadata: list[dict[str, int]]) -> dict[str, Any]:
    arrays = {
        name: np.load(c.Q1_FROZEN / name, mmap_mode="r", allow_pickle=False)
        for name in (
            "prompt_ids.npy",
            "prompt_offsets.npy",
            "response_ids.npy",
            "response_offsets.npy",
            *FROZEN_MEMBERS,
        )
    }
    row_offsets = arrays["row_offsets.npy"]
    if row_offsets.ndim == 2 and row_offsets.shape[1] == 2:
        source_starts = row_offsets[:, 0]
        source_ends = row_offsets[:, 1]
    elif row_offsets.ndim == 1:
        source_starts = row_offsets[:-1]
        source_ends = row_offsets[1:]
    else:
        raise RuntimeError(f"unsupported frozen row_offsets shape: {row_offsets.shape}")
    if not (
        np.array_equal(arrays["response_offsets.npy"][:-1], source_starts)
        and np.array_equal(arrays["response_offsets.npy"][1:], source_ends)
    ):
        raise RuntimeError("frozen response offsets differ from source top-32 offsets")
    if arrays["top32_ids.npy"].shape != arrays["top32_logprob.npy"].shape:
        raise RuntimeError("frozen top-32 arrays have different shapes")
    if arrays["top32_ids.npy"].shape[1] != 32:
        raise RuntimeError(f"expected RAW top-32, found shape {arrays['top32_ids.npy'].shape}")
    if int(arrays["prompt_offsets.npy"][-1]) != len(arrays["prompt_ids.npy"]):
        raise RuntimeError("flat prompt token count mismatch")
    if int(arrays["response_offsets.npy"][-1]) != len(arrays["response_ids.npy"]):
        raise RuntimeError("flat response token count mismatch")
    if len(arrays["prompt_offsets.npy"]) != len(metadata) + 1:
        raise RuntimeError("frozen record count mismatch")
    return {
        name: {
            "path": str(c.Q1_FROZEN / name),
            "bytes": (c.Q1_FROZEN / name).stat().st_size,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for name, value in arrays.items()
    }


def _prompt_id(row: pd.Series, fallback: int) -> int:
    extra = dict(row["extra_info"])
    return int(extra.get("index", fallback))


def _build_schedule(by_prompt_id: dict[int, int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        c.QWEN_STUDENT, local_files_only=True, trust_remote_code=True
    )
    source = pd.read_parquet(c.SOURCE_PROMPTS)
    if len(source) != 5000:
        raise RuntimeError(f"Qwen prompt source row drift: {len(source)}")
    eligible_indices: list[int] = []
    prompt_ids: list[int] = []
    rejected: list[dict[str, int]] = []
    for source_row, row in source.iterrows():
        messages = _messages(row["prompt"])
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        prompt_id = _prompt_id(row, int(source_row))
        if len(ids) > c.MAX_PROMPT_TOKENS:
            rejected.append(
                {
                    "source_row": int(source_row),
                    "prompt_id": prompt_id,
                    "prompt_tokens": len(ids),
                }
            )
            continue
        if prompt_id not in by_prompt_id:
            raise RuntimeError(f"eligible prompt {prompt_id} missing frozen off-KD rollout")
        eligible_indices.append(int(source_row))
        prompt_ids.append(prompt_id)
    if len(eligible_indices) != 4999:
        raise RuntimeError(f"expected 4999 eligible Qwen prompts, found {len(eligible_indices)}")

    selected = source.iloc[eligible_indices[:USABLE_PROMPTS]].copy().reset_index(drop=True)
    selected_prompt_ids = prompt_ids[:USABLE_PROMPTS]
    epochs: list[pd.DataFrame] = []
    for logical_epoch in (0, 1):
        frame = selected.copy()
        slot = np.arange(USABLE_PROMPTS, dtype=np.int64) % BATCH_SIZE
        self_mask = slot < (BATCH_SIZE // 2)
        if logical_epoch == 1:
            self_mask = ~self_mask
        source_ids = self_mask.astype(np.int64)
        frame["support_source_id"] = source_ids
        frame["support_source"] = np.where(self_mask, "self", "external")
        frame["agent_name"] = np.where(
            self_mask, "single_turn_agent", "q1_frozen_external"
        )
        frame["external_record_index"] = [
            by_prompt_id[prompt_id] for prompt_id in selected_prompt_ids
        ]
        frame["logical_epoch"] = logical_epoch
        frame["logical_step"] = logical_epoch * (USABLE_PROMPTS // BATCH_SIZE) + (
            np.arange(USABLE_PROMPTS, dtype=np.int64) // BATCH_SIZE
        )
        frame["batch_slot"] = slot
        frame["prompt_identity"] = selected_prompt_ids
        epochs.append(frame)
    schedule = pd.concat(epochs, ignore_index=True)
    for start in range(0, len(schedule), BATCH_SIZE):
        batch = schedule.iloc[start : start + BATCH_SIZE]
        counts = batch["support_source_id"].value_counts().to_dict()
        if counts != {0: 8, 1: 8}:
            raise RuntimeError(f"non-8/8 schedule at physical row {start}: {counts}")
    for prompt_identity, rows in schedule.groupby("prompt_identity", sort=False):
        if rows["support_source_id"].tolist() != [1, 0] and rows["support_source_id"].tolist() != [0, 1]:
            raise RuntimeError(f"source assignment did not swap for prompt {prompt_identity}")
    return schedule, {
        "source_rows": len(source),
        "eligible_rows": len(eligible_indices),
        "used_prompts_per_epoch": USABLE_PROMPTS,
        "unused_eligible_tail_per_epoch": len(eligible_indices) - USABLE_PROMPTS,
        "rejected": rejected,
        "physical_rows": len(schedule),
        "updates": len(schedule) // BATCH_SIZE,
    }


def _build_stage_b_schedule(schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep Stage A intact, then replay its prompts once with support sources swapped."""
    rows_per_stage = STAGE_A_STEPS * BATCH_SIZE
    first_epoch_rows = USABLE_PROMPTS
    stage_a = schedule.iloc[:rows_per_stage].copy()
    stage_b = schedule.iloc[first_epoch_rows : first_epoch_rows + rows_per_stage].copy()
    # Keep log metadata aligned with the resumed optimizer step, not the source epoch.
    stage_a["logical_step"] = np.arange(STAGE_A_STEPS, dtype=np.int64).repeat(BATCH_SIZE)
    stage_b["logical_step"] = np.arange(
        STAGE_A_STEPS, STAGE_B_FINAL_STEP, dtype=np.int64
    ).repeat(BATCH_SIZE)
    paired = pd.concat((stage_a, stage_b), ignore_index=True)

    if len(paired) // BATCH_SIZE != STAGE_B_FINAL_STEP:
        raise RuntimeError("Q1 Stage B schedule must reserve exactly 320 updates")
    if not np.array_equal(
        stage_a["prompt_identity"].to_numpy(), stage_b["prompt_identity"].to_numpy()
    ):
        raise RuntimeError("Q1 Stage B prompts do not exactly replay Stage A prompts")
    if not np.array_equal(
        stage_a["support_source_id"].to_numpy(),
        1 - stage_b["support_source_id"].to_numpy(),
    ):
        raise RuntimeError("Q1 Stage B did not swap every self/external assignment")
    for start in range(0, len(paired), BATCH_SIZE):
        counts = paired.iloc[start : start + BATCH_SIZE]["support_source_id"].value_counts().to_dict()
        if counts != {0: 8, 1: 8}:
            raise RuntimeError(f"non-8/8 Stage B schedule at physical row {start}: {counts}")
    return paired, {
        "physical_rows": len(paired),
        "updates": len(paired) // BATCH_SIZE,
        "stage_a_updates": STAGE_A_STEPS,
        "stage_b_updates": STAGE_A_STEPS,
        "prompt_contract": "Stage A prompts replayed in the same order with support sources swapped",
    }


def prepare(*, validate_only: bool = False) -> dict[str, Any]:
    for source in (c.SOURCE_PROMPTS, SOURCE_ROLLOUT, SOURCE_TOP32):
        if not source.is_file():
            raise FileNotFoundError(source)
    if not c.model_check(c.QWEN_STUDENT)["complete"]:
        raise FileNotFoundError(f"incomplete Qwen student: {c.QWEN_STUDENT}")

    metadata, by_prompt_id = _scan_rollout(SOURCE_ROLLOUT)
    if not validate_only:
        c.Q1_DATA.mkdir(parents=True, exist_ok=True)
        c.Q1_FROZEN.mkdir(parents=True, exist_ok=True)
        _extract_npy_members(SOURCE_TOP32, c.Q1_FROZEN)
        _write_flat_tokens(c.Q1_FROZEN, metadata)
    frozen = _verify_frozen_store(metadata) if c.Q1_FROZEN.is_dir() else {}
    schedule, schedule_meta = _build_schedule(by_prompt_id)
    if len(schedule) // BATCH_SIZE != 624:
        raise RuntimeError("Q1 schedule must reserve exactly 624 updates")
    stage_b_schedule, stage_b_meta = _build_stage_b_schedule(schedule)

    if not validate_only:
        schedule.to_parquet(FORMAL_PARQUET, index=False)
        stage_b_schedule.to_parquet(STAGE_B_PARQUET, index=False)
        schedule.iloc[:32].to_parquet(SMOKE_PARQUET, index=False)
    payload = {
        "schema_version": 1,
        "status": "validated" if validate_only else "complete",
        "created_utc": c.utc_now(),
        "arm": "qwen_alpha05_support_mixture",
        "source_semantics": {"external": 0, "self": 1},
        "loss_contract": ".5*mean_token(self)+.5*mean_token(external)",
        "batch_contract": "ordered physical batches; exactly 8 external + 8 self",
        "epoch_contract": "same 4992 prompts; source assignment swapped in logical epoch 2",
        "stage_a_hard_stop": STAGE_A_STEPS,
        "stage_b_requires_new_go": True,
        "stage_b_final_stop": STAGE_B_FINAL_STEP,
        "stage_b_contract": "reuse each Stage A prompt once with swapped self/external supports",
        "source_prompt": c.artifact(c.SOURCE_PROMPTS),
        "source_rollout": c.artifact(SOURCE_ROLLOUT),
        "source_top32": {
            "path": str(SOURCE_TOP32),
            "bytes": SOURCE_TOP32.stat().st_size,
        },
        "frozen_store": frozen,
        "schedule": schedule_meta,
        "stage_b_schedule": stage_b_meta,
        "outputs": (
            [
                c.artifact(FORMAL_PARQUET),
                c.artifact(STAGE_B_PARQUET),
                c.artifact(SMOKE_PARQUET),
            ]
            if not validate_only
            else []
        ),
    }
    if not validate_only:
        c.atomic_json(MANIFEST, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(validate_only=args.validate_only), indent=2))


if __name__ == "__main__":
    main()
