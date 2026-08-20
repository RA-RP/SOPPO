# coding: utf8
import json
import os
import random
from typing import Dict, List

import torch


MAX_WARN_EXAMPLES = 5


def _require_mode(mode: str):
    if mode not in ("x", "s"):
        raise ValueError(f"mode must be 'x' or 's', got: {mode}")


def _to_clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_x_text_strict(record: dict, line_no: int, jsonl_path: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{jsonl_path}:{line_no} must be a JSON object for X records")

    if "output" not in record:
        raise KeyError(f"{jsonl_path}:{line_no} missing required key: output")
    if not isinstance(record["output"], dict):
        raise TypeError(f"{jsonl_path}:{line_no} key 'output' must be an object")

    output_obj = record["output"]
    if "text" not in output_obj:
        raise KeyError(f"{jsonl_path}:{line_no} missing required key: output.text")

    return _to_clean_text(output_obj["text"])


def _extract_s_text_strict(record: dict, line_no: int, jsonl_path: str) -> str:
    if not isinstance(record, dict):
        raise TypeError(f"{jsonl_path}:{line_no} must be a JSON object for S records")

    if "question" not in record:
        raise KeyError(f"{jsonl_path}:{line_no} missing required key: question")
    if "answer" not in record:
        raise KeyError(f"{jsonl_path}:{line_no} missing required key: answer")

    question = _to_clean_text(record["question"])
    answer = _to_clean_text(record["answer"])
    return f"{question} {answer}".strip()


def load_texts_from_jsonl(jsonl_path, mode="x"):
    _require_mode(mode)

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"jsonl file not found: {jsonl_path}")

    texts = []
    skipped_mismatch = 0
    skipped_empty = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except Exception as e:
                raise ValueError(f"{jsonl_path}:{line_no} is not valid JSON: {e}") from e

            try:
                if mode == "x":
                    text = _extract_x_text_strict(record, line_no=line_no, jsonl_path=jsonl_path)
                else:
                    text = _extract_s_text_strict(record, line_no=line_no, jsonl_path=jsonl_path)
            except (KeyError, TypeError, ValueError) as e:
                skipped_mismatch += 1
                if skipped_mismatch <= MAX_WARN_EXAMPLES:
                    print(f"[Warn] Skip line due to strict field mismatch: {e}")
                continue

            if text:
                texts.append(text)
            else:
                skipped_empty += 1
                if skipped_empty <= MAX_WARN_EXAMPLES:
                    print(f"[Warn] Empty strict field text at {jsonl_path}:{line_no}")

    if skipped_mismatch > MAX_WARN_EXAMPLES:
        print(f"[Warn] ... and {skipped_mismatch - MAX_WARN_EXAMPLES} more strict-mismatch lines skipped.")
    if skipped_empty > MAX_WARN_EXAMPLES:
        print(f"[Warn] ... and {skipped_empty - MAX_WARN_EXAMPLES} more empty-text lines skipped.")

    if skipped_mismatch > 0 or skipped_empty > 0:
        print(
            "[Info] Strict read summary "
            f"({mode}): kept={len(texts)}, "
            f"skipped_mismatch={skipped_mismatch}, skipped_empty={skipped_empty}"
        )

    if len(texts) == 0:
        raise ValueError(f"No usable text found in jsonl: {jsonl_path}")
    return texts


def get_token_data_from_jsonl(
    jsonl_path,
    tokenizer,
    nsamples,
    seqlen=2048,
    seed=3,
    batch_size=1,
    cache_file=None,
    mode="x",
):
    """
    Build calibration batches from an existing JSONL dataset with strict fields.
    X mode requires: output.text
    S mode requires: question, answer
    Returns: List[{"input_ids": Tensor[B, L], "attention_mask": Tensor[B, L]}]
    """
    _require_mode(mode)

    if cache_file is not None and os.path.exists(cache_file):
        return torch.load(cache_file)

    texts = load_texts_from_jsonl(jsonl_path, mode=mode)
    tot_text = "\n\n".join(texts)
    if len(tot_text) <= int(seqlen) + 1:
        raise ValueError(
            f"Insufficient text length in {jsonl_path}. "
            f"Need more than seqlen={seqlen} characters after concat."
        )

    random.seed(seed)
    nsamples = int(nsamples)
    batch_size = int(batch_size)
    if nsamples <= 0:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # Keep step1-style flush behavior.
    nsamples += 1

    traindataset: List[Dict[str, torch.Tensor]] = []
    inp = None
    produced_batches = 0

    for s in range(nsamples):
        i = random.randint(0, len(tot_text) - seqlen - 1)
        j = i + seqlen * 10
        trainenc = tokenizer(tot_text[i:j], return_tensors="pt")

        if trainenc.input_ids.shape[1] < seqlen:
            continue

        if s % batch_size == 0:
            if s != 0 and inp is not None:
                attention_mask = torch.ones_like(inp)
                traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
                produced_batches += 1
            inp = trainenc.input_ids[:, :seqlen]
        else:
            inp = torch.cat((inp, trainenc.input_ids[:, :seqlen]), dim=0)

    if produced_batches == 0:
        raise ValueError(
            f"Failed to build token batches from {jsonl_path}. "
            "Try smaller seqlen or larger dataset."
        )

    if cache_file is not None:
        cache_dir = os.path.dirname(cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        torch.save(traindataset, cache_file)

    return traindataset
