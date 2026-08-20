#!/usr/bin/env python3
"""Freeze the ordered 4,999-prompt Llama OPD input and its audit map."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c


FORMAL_PARQUET = c.L1_DATA / "llama_opd_prompts_4999.parquet"
SMOKE_PARQUET = c.L1_DATA / "llama_opd_prompts_smoke32.parquet"
PROMPT_MAP = c.L1_DATA / "llama_opd_prompt_map.jsonl"
MANIFEST = c.L1_DATA / "prompt_manifest.json"


def _messages(value: Any) -> list[dict[str, str]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    rows = list(value)
    return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prepare(*, validate_only: bool = False) -> dict[str, Any]:
    if not c.SOURCE_PROMPTS.is_file():
        raise FileNotFoundError(c.SOURCE_PROMPTS)
    if not c.model_check(c.LLAMA_STUDENT)["complete"]:
        raise FileNotFoundError(f"incomplete student model: {c.LLAMA_STUDENT}")

    tokenizer = c.load_llama_tokenizer()
    template_contract = c.llama_template_contract()
    frame = pd.read_parquet(c.SOURCE_PROMPTS)
    required = {"data_source", "prompt", "ability", "reward_model", "extra_info"}
    if len(frame) != 5000 or not required.issubset(frame.columns):
        raise RuntimeError(
            f"source prompt contract drift: rows={len(frame)} columns={frame.columns.tolist()}"
        )

    keep: list[int] = []
    map_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for source_row, row in frame.iterrows():
        messages = _messages(row["prompt"])
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise RuntimeError(f"unexpected prompt schema at source row {source_row}")
        extra = dict(row["extra_info"])
        prompt_id = int(extra.get("index", source_row))
        if prompt_id in seen_ids:
            raise RuntimeError(f"duplicate source prompt id: {prompt_id}")
        seen_ids.add(prompt_id)
        token_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if len(token_ids) > c.MAX_PROMPT_TOKENS:
            rejected.append(
                {
                    "source_row": int(source_row),
                    "prompt_id": prompt_id,
                    "prompt_tokens": len(token_ids),
                }
            )
            continue
        eligible_order = len(keep)
        keep.append(int(source_row))
        map_rows.append(
            {
                "prompt_id": prompt_id,
                "source_row": int(source_row),
                "eligible_order": eligible_order,
                "prompt_tokens": len(token_ids),
                "raw_user_sha256": text_hash(messages[0]["content"]),
                "formatted_prompt_sha256": text_hash(formatted),
                "formatted_prompt": formatted,
            }
        )

    if len(keep) != 4999:
        raise RuntimeError(
            f"Llama eligible prompt count={len(keep)}, expected=4999; rejected={rejected}"
        )

    eligible = frame.iloc[keep].copy().reset_index(drop=True)
    updated_extra = []
    for order, value in enumerate(eligible["extra_info"]):
        extra = dict(value)
        extra.update(
            {
                "prompt_id": int(map_rows[order]["prompt_id"]),
                "source_row": int(map_rows[order]["source_row"]),
                "eligible_order": order,
            }
        )
        updated_extra.append(extra)
    eligible["extra_info"] = updated_extra

    if not validate_only:
        c.L1_DATA.mkdir(parents=True, exist_ok=True)
        eligible.to_parquet(FORMAL_PARQUET, index=False)
        eligible.iloc[:32].to_parquet(SMOKE_PARQUET, index=False)
        c.atomic_jsonl(PROMPT_MAP, map_rows)

    outputs = []
    runtime_model = None
    if not validate_only:
        runtime_model = c.ensure_llama_runtime_model()
        outputs = [c.artifact(path) for path in (FORMAL_PARQUET, SMOKE_PARQUET, PROMPT_MAP)]
    payload = {
        "schema_version": 1,
        "status": "validated" if validate_only else "complete",
        "created_utc": c.utc_now(),
        "source": c.artifact(c.SOURCE_PROMPTS),
        "student_model": str(c.LLAMA_STUDENT),
        "student_runtime_model": runtime_model,
        "student_tokenizer_config": c.artifact(c.LLAMA_STUDENT / "tokenizer_config.json"),
        "chat_template": {
            "source": template_contract["chat_template_source"],
            "sha256": template_contract["chat_template_sha256"],
            "text": template_contract["chat_template"],
            "student_tokenizer_sha256": template_contract["student_tokenizer_sha256"],
            "teacher_tokenizer_sha256": template_contract["teacher_tokenizer_sha256"],
        },
        "selection": "source order; retain prompt chat-template length <=1024",
        "source_rows": len(frame),
        "eligible_rows": len(eligible),
        "rejected_rows": rejected,
        "shuffle": False,
        "effective_batch_size": c.TRAIN_BATCH_SIZE,
        "drop_last_each_epoch": True,
        "updates_per_epoch": len(eligible) // c.TRAIN_BATCH_SIZE,
        "unused_tail_per_epoch": len(eligible) % c.TRAIN_BATCH_SIZE,
        "total_epochs": 2,
        "total_updates_available": 2 * (len(eligible) // c.TRAIN_BATCH_SIZE),
        "stage_a_stop_step": c.L1_STAGE_A_FINAL_STEP,
        "stage_b_stop_step": c.L1_STAGE_B_FINAL_STEP,
        "stage_b_requires_explicit_go": True,
        "outputs": outputs,
    }
    if not 1 <= c.L1_STAGE_B_FINAL_STEP <= payload["total_updates_available"]:
        raise RuntimeError(
            "requested L1 Stage-B stop step is outside the available two-epoch schedule: "
            f"{c.L1_STAGE_B_FINAL_STEP} not in [1, {payload['total_updates_available']}]"
        )
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
