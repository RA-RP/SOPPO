#!/usr/bin/env python3
"""Real-base equivalence smoke for Stage-3 selective logit projection."""

from __future__ import annotations

import gc
import json
import math

import torch
import torch.nn.functional as F

import cycle09_c11_answer_entropy as c11
import cycle09_c8_training_ppl as c8
import cycle09_stage3_common as s3


OUTPUT = s3.RUN_ROOT / "supervisor/model_smoke.json"
LOGIT_TOLERANCE = 0.125
METRIC_TOLERANCE = 5e-4


@torch.inference_mode()
def compare_c11(model, row: dict, device: str) -> dict:
    ids = torch.tensor(row["input_token_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention = torch.ones_like(ids)
    full_output = model(input_ids=ids, attention_mask=attention, use_cache=False)
    full_logits = full_output.logits[0, -1].float()
    del full_output

    decoder_output = model.model(
        input_ids=ids,
        attention_mask=attention,
        use_cache=False,
        return_dict=True,
    )
    selective_logits = model.get_output_embeddings()(
        decoder_output.last_hidden_state[0, -1]
    ).float()
    del decoder_output
    max_abs = float((full_logits - selective_logits).abs().max().cpu())

    full_log_probs = torch.log_softmax(full_logits, dim=-1)
    full_entropy = float(
        (-(full_log_probs.exp() * full_log_probs).sum()).cpu()
    )
    scored = c11.score_batch(model, [row], device, pad_token_id=0)[0]
    entropy_abs = abs(full_entropy - scored["full_vocab_entropy_nats"])
    del ids, attention, full_logits, selective_logits, full_log_probs
    return {
        "sample_id": row["sample_id"],
        "context_tokens": len(row["input_token_ids"]),
        "full_vs_selective_logit_max_abs": max_abs,
        "full_vs_scored_entropy_abs": entropy_abs,
        "full_entropy_nats": full_entropy,
        "selective_entropy_nats": scored["full_vocab_entropy_nats"],
    }


@torch.inference_mode()
def compare_c8(model, raw: dict, device: str) -> dict:
    scored_tokens = min(4, int(raw["response_tokens_scored"]))
    prompt_tokens = int(raw["prompt_tokens"])
    row = {
        **raw,
        "input_token_ids": raw["input_token_ids"][: prompt_tokens + scored_tokens],
        "response_tokens_original": scored_tokens,
        "response_tokens_scored": scored_tokens,
        "truncated": 0,
    }
    ids = torch.tensor(row["input_token_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention = torch.ones_like(ids)
    left = prompt_tokens - 1
    right = left + scored_tokens
    targets = ids[0, left + 1 : right + 1]

    full_output = model(input_ids=ids, attention_mask=attention, use_cache=False)
    full_logits = full_output.logits[0, left:right].float()
    del full_output
    decoder_output = model.model(
        input_ids=ids,
        attention_mask=attention,
        use_cache=False,
        return_dict=True,
    )
    selective_logits = model.get_output_embeddings()(
        decoder_output.last_hidden_state[0, left:right]
    ).float()
    del decoder_output
    max_abs = float((full_logits - selective_logits).abs().max().cpu())
    full_nll = float(F.cross_entropy(full_logits, targets, reduction="sum").cpu())
    scored = c8.score_batch(
        model,
        [row],
        device,
        pad_token_id=0,
        loss_chunk_tokens=2,
    )[0]
    nll_abs = abs(full_nll - scored["nll_sum"])
    del ids, attention, targets, full_logits, selective_logits
    return {
        "sample_id": row["sample_id"],
        "prompt_tokens": prompt_tokens,
        "response_tokens_scored": scored_tokens,
        "full_vs_selective_logit_max_abs": max_abs,
        "full_vs_scored_nll_abs": nll_abs,
        "full_nll_sum": full_nll,
        "selective_nll_sum": scored["nll_sum"],
    }


def main() -> None:
    s3.assert_contract()
    model_path = s3.require_model("opd", 0)
    c8.prepare_corpora()
    c11.prepare_corpus()
    c8_rows = [
        row
        for name in c8.CORPORA
        for row in s3.read_jsonl(c8.corpus_path(name))
    ]
    c8_row = min(c8_rows, key=lambda row: int(row["prompt_tokens"]))
    c11_row = min(
        s3.read_jsonl(c11.CORPUS),
        key=lambda row: len(row["input_token_ids"]),
    )

    device = "cuda"
    model = c8.load_model(model_path, device)
    try:
        if not hasattr(model, "model") or model.get_output_embeddings() is None:
            raise RuntimeError("base model does not expose decoder/output head")
        c8_result = compare_c8(model, c8_row, device)
        c11_result = compare_c11(model, c11_row, device)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    logit_values = (
        c8_result["full_vs_selective_logit_max_abs"],
        c11_result["full_vs_selective_logit_max_abs"],
    )
    metric_values = (
        c8_result["full_vs_scored_nll_abs"],
        c11_result["full_vs_scored_entropy_abs"],
    )
    if not (
        all(
            math.isfinite(value) and value <= LOGIT_TOLERANCE
            for value in logit_values
        )
        and all(
            math.isfinite(value) and value <= METRIC_TOLERANCE
            for value in metric_values
        )
    ):
        raise RuntimeError(
            f"selective output-head equivalence failed: C8={c8_result}, C11={c11_result}"
        )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": s3.utc_now(),
        "model_path": str(model_path),
        "model_integrity": s3.model_integrity(model_path),
        "logit_max_abs_tolerance": LOGIT_TOLERANCE,
        "metric_abs_tolerance": METRIC_TOLERANCE,
        "c8": c8_result,
        "c11": c11_result,
    }
    s3.atomic_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
