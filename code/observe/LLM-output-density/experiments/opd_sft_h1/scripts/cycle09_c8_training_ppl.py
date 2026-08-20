#!/usr/bin/env python3
"""C8: four-arm checkpoint PPL on three frozen training-text corpora."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import cycle09_s1_5_train_corpus_ppl as s15
import cycle09_stage3_common as s3


TASK = "C8"
ROOT = s3.RUN_ROOT / "c8_training_ppl"
CORPUS_ROOT = ROOT / "corpora"
CORPUS_MANIFEST = CORPUS_ROOT / "manifest.json"
CELL_ROOT = ROOT / "cells"
CORPORA = (
    "X_OPD_reconstructed",
    "X_SFT_dataset",
    "X_teacher",
)
N_SAMPLES = 500
SEED = 42
MAX_TOTAL_TOKENS = 10240
PROTOCOL_VERSION = "cycle09-c8-response-ppl-v1"


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def corpus_path(name: str) -> Path:
    return CORPUS_ROOT / f"{name}.jsonl"


def truncate_sample(sample: dict[str, Any]) -> dict[str, Any]:
    prompt = [int(token) for token in sample["prompt_token_ids"]]
    response = [int(token) for token in sample["response_token_ids"]]
    if not prompt or len(prompt) >= MAX_TOTAL_TOKENS:
        raise ValueError(
            f"invalid prompt length for {sample['sample_id']}: {len(prompt)}"
        )
    keep = min(len(response), MAX_TOTAL_TOKENS - len(prompt))
    if keep < 1:
        raise ValueError(f"empty response after truncation: {sample['sample_id']}")
    return {
        "sample_id": str(sample["sample_id"]),
        "source_index": int(sample["source_index"]),
        "input_token_ids": prompt + response[:keep],
        "prompt_tokens": len(prompt),
        "response_tokens_original": len(response),
        "response_tokens_scored": keep,
        "truncated": int(keep < len(response)),
    }


def prepare_corpora() -> dict[str, Any]:
    sources = {
        "X_OPD_reconstructed": s3.OPD_RECONSTRUCTION,
        "X_SFT_dataset": s3.SFT_TRAIN,
        "X_teacher": s3.TEACHER_TRAIN,
    }
    for name, path in sources.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"C8 source missing for {name}: {path}")
    source_hashes = {name: s3.sha256_file(path) for name, path in sources.items()}
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "n_samples_per_corpus": N_SAMPLES,
        "selection_seed": SEED,
        "selection_indices_sha256": s3.sha256_json(s15.selected_indices(N_SAMPLES)),
        "source_sha256": source_hashes,
    }
    if CORPUS_MANIFEST.is_file():
        old = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
        if all(old.get(key) == value for key, value in expected.items()):
            for name in CORPORA:
                if not corpus_path(name).is_file():
                    raise FileNotFoundError(corpus_path(name))
            return old
        raise RuntimeError(f"incompatible existing C8 corpus manifest: {CORPUS_MANIFEST}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(s3.BASE_MODEL), trust_remote_code=True
    )
    loaded: dict[str, list[dict[str, Any]]] = {}
    loaded["X_OPD_reconstructed"], opd_provenance = s15.load_rollout_samples(
        s3.OPD_RECONSTRUCTION,
        "opd",
        "deterministic Cycle08 step-0 reconstruction; not an online archive",
        N_SAMPLES,
    )
    loaded["X_SFT_dataset"], sft_provenance = s15.load_sft_samples(
        tokenizer, N_SAMPLES
    )
    loaded["X_teacher"], teacher_provenance = s15.load_rollout_samples(
        s3.TEACHER_TRAIN,
        "teacher",
        "static Qwen3-8B teacher rollout shared by off-KD and seqKD",
        N_SAMPLES,
    )
    provenance = {
        "X_OPD_reconstructed": opd_provenance,
        "X_SFT_dataset": sft_provenance,
        "X_teacher": teacher_provenance,
    }
    inventory = {}
    for name in CORPORA:
        rows = [truncate_sample(row) for row in loaded[name]]
        if len(rows) != N_SAMPLES or len({row["sample_id"] for row in rows}) != N_SAMPLES:
            raise RuntimeError(f"bad C8 prepared corpus {name}")
        s3.atomic_jsonl(corpus_path(name), rows)
        inventory[name] = {
            "path": str(corpus_path(name)),
            "sha256": s3.sha256_file(corpus_path(name)),
            "n_samples": len(rows),
            "n_response_tokens_scored": sum(
                row["response_tokens_scored"] for row in rows
            ),
            "truncated_n": sum(row["truncated"] for row in rows),
        }
    payload = {
        "schema_version": 1,
        **expected,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "score_scope": "response tokens only conditioned on complete prompt",
        "aggregation": "token-weighted NLL then exp",
        "sources": provenance,
        "corpora": inventory,
    }
    s3.atomic_json(CORPUS_MANIFEST, payload)
    return payload


def load_corpora() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    manifest = prepare_corpora()
    corpora = {name: s3.read_jsonl(corpus_path(name)) for name in CORPORA}
    for name, rows in corpora.items():
        if len(rows) != N_SAMPLES:
            raise RuntimeError(f"C8 {name} rows={len(rows)}")
    return corpora, manifest


def make_batches(
    samples: list[dict[str, Any]], token_budget: int, max_batch_size: int
) -> list[list[dict[str, Any]]]:
    ordered = sorted(samples, key=lambda row: len(row["input_token_ids"]))
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for sample in ordered:
        length = len(sample["input_token_ids"])
        proposed = len(current) + 1
        if current and (
            proposed > max_batch_size or proposed * length > token_budget
        ):
            batches.append(current)
            current = []
        current.append(sample)
    if current:
        batches.append(current)
    return batches


@torch.inference_mode()
def score_batch(
    model,
    samples: list[dict[str, Any]],
    device: str,
    pad_token_id: int,
    loss_chunk_tokens: int,
) -> list[dict[str, Any]]:
    maximum = max(len(row["input_token_ids"]) for row in samples)
    input_ids = torch.full(
        (len(samples), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    for index, row in enumerate(samples):
        ids = torch.tensor(row["input_token_ids"], dtype=torch.long, device=device)
        input_ids[index, : len(ids)] = ids
        attention[index, : len(ids)] = 1
    output_head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    optimized = hasattr(model, "model") and output_head is not None
    if optimized:
        decoder_output = model.model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = decoder_output.last_hidden_state
        del decoder_output
        logits = None
    else:
        output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        logits = output.logits
        hidden_states = None
    results = []
    for index, row in enumerate(samples):
        length = len(row["input_token_ids"])
        start = int(row["prompt_tokens"]) - 1
        stop = length - 1
        if stop - start != int(row["response_tokens_scored"]):
            raise RuntimeError(f"C8 token boundary mismatch: {row['sample_id']}")
        nll_sum = 0.0
        for left in range(start, stop, loss_chunk_tokens):
            right = min(stop, left + loss_chunk_tokens)
            targets = input_ids[index, left + 1 : right + 1]
            selected_logits = (
                output_head(hidden_states[index, left:right]).float()
                if optimized
                else logits[index, left:right].float()
            )
            value = F.cross_entropy(selected_logits, targets, reduction="sum")
            nll_sum += float(value.cpu())
            del value, targets, selected_logits
        if not math.isfinite(nll_sum):
            raise FloatingPointError(f"non-finite C8 NLL: {row['sample_id']}")
        results.append(
            {
                "sample_id": row["sample_id"],
                "source_index": row["source_index"],
                "prompt_tokens": row["prompt_tokens"],
                "response_tokens_original": row["response_tokens_original"],
                "response_tokens_scored": row["response_tokens_scored"],
                "truncated": row["truncated"],
                "nll_sum": nll_sum,
                "mean_nll": nll_sum / row["response_tokens_scored"],
            }
        )
    if optimized:
        del hidden_states
    else:
        del output, logits
    del input_ids, attention
    torch.cuda.empty_cache()
    return results


def cell_path(arm: str, step: int) -> Path:
    label = "base" if step == 0 else arm
    return CELL_ROOT / label / s3.step_label(step) / "result.json"


def protocol_id(manifest: dict[str, Any]) -> str:
    return s3.sha256_json(
        {
            "version": PROTOCOL_VERSION,
            "corpora": manifest["corpora"],
            "max_total_tokens": MAX_TOTAL_TOKENS,
            "aggregation": "token-weighted NLL then exp",
            "logit_projection": (
                "decoder hidden states followed by the original output head only "
                "at scored response positions"
            ),
        }
    )


def cell_complete(path: Path, expected_protocol: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            payload.get("status") == "complete"
            and payload.get("protocol_id") == expected_protocol
            and len(payload.get("rows", [])) == len(CORPORA) * N_SAMPLES
        )
    except (OSError, json.JSONDecodeError):
        return False


def load_model(path: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    return model


def run_cell(
    arm: str,
    step: int,
    corpora: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    output_path = cell_path(arm, step)
    expected_protocol = protocol_id(manifest)
    if cell_complete(output_path, expected_protocol):
        print(f"[C8 cached] {arm}/{step}", flush=True)
        return
    with lock(output_path.with_suffix(".lock")):
        if cell_complete(output_path, expected_protocol):
            return
        existing: dict[str, dict[str, Any]] = {}
        if output_path.is_file():
            old = json.loads(output_path.read_text(encoding="utf-8"))
            if old.get("protocol_id") != expected_protocol:
                raise RuntimeError(f"incompatible C8 cell cache: {output_path}")
            existing = {
                f"{row['corpus']}::{row['sample_id']}": row
                for row in old.get("rows", [])
            }
        model_path = s3.require_model(arm, step)
        tokenizer = AutoTokenizer.from_pretrained(
            str(s3.BASE_MODEL), trust_remote_code=True
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError(f"no pad/eos token for {model_path}")
        print(
            f"[C8 load] {arm}/{step} cached={len(existing)}/{len(CORPORA) * N_SAMPLES}",
            flush=True,
        )
        model = load_model(model_path, args.device)
        try:
            batches_since_save = 0
            for corpus in CORPORA:
                pending = [
                    row
                    for row in corpora[corpus]
                    if f"{corpus}::{row['sample_id']}" not in existing
                ]
                batches = make_batches(
                    pending, args.token_budget, args.max_batch_size
                )
                for batch_index, batch in enumerate(batches, start=1):
                    scored = score_batch(
                        model,
                        batch,
                        args.device,
                        int(pad_token_id),
                        args.loss_chunk_tokens,
                    )
                    for row in scored:
                        result = {"corpus": corpus, **row}
                        existing[f"{corpus}::{row['sample_id']}"] = result
                    batches_since_save += 1
                    if batches_since_save >= args.save_every_batches:
                        s3.atomic_json(
                            output_path,
                            {
                                "schema_version": 1,
                                "status": "partial",
                                "task": TASK,
                                "arm": "base" if step == 0 else arm,
                                "step": step,
                                "protocol_id": expected_protocol,
                                "model_path": str(model_path),
                                "rows": list(existing.values()),
                            },
                        )
                        batches_since_save = 0
                    print(
                        f"[C8] {arm}/{step} {corpus} batch "
                        f"{batch_index}/{len(batches)} total={len(existing)}",
                        flush=True,
                    )
            ordered = [
                existing[f"{corpus}::{sample['sample_id']}"]
                for corpus in CORPORA
                for sample in corpora[corpus]
            ]
            if len(ordered) != len(CORPORA) * N_SAMPLES:
                raise RuntimeError(f"incomplete C8 cell {arm}/{step}")
            s3.atomic_json(
                output_path,
                {
                    "schema_version": 1,
                    "status": "complete",
                    "task": TASK,
                    "arm": "base" if step == 0 else arm,
                    "step": step,
                    "protocol_id": expected_protocol,
                    "model_path": str(model_path),
                    "model_integrity": s3.model_integrity(model_path),
                    "rows": ordered,
                },
            )
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()


def run_cells(args: argparse.Namespace) -> None:
    corpora, manifest = load_corpora()
    cells = []
    if 0 in args.steps:
        cells.append((args.arms[0], 0))
    cells.extend(
        (arm, step)
        for arm in args.arms
        for step in args.steps
        if step != 0
    )
    for arm, step in cells:
        run_cell(arm, step, corpora, manifest, args)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for corpus in CORPORA:
        cell = [row for row in rows if row["corpus"] == corpus]
        if len(cell) != N_SAMPLES:
            raise RuntimeError(f"C8 summary {corpus} rows={len(cell)}")
        nll_sum = sum(float(row["nll_sum"]) for row in cell)
        tokens = sum(int(row["response_tokens_scored"]) for row in cell)
        mean_nll = nll_sum / tokens
        output.append(
            {
                "corpus": corpus,
                "ppl": math.exp(mean_nll),
                "mean_nll": mean_nll,
                "nll_sum": nll_sum,
                "n_samples": len(cell),
                "n_scored_tokens": tokens,
                "prompt_tokens_mean": float(
                    np.mean([row["prompt_tokens"] for row in cell])
                ),
                "response_tokens_original_mean": float(
                    np.mean([row["response_tokens_original"] for row in cell])
                ),
                "response_tokens_scored_mean": float(
                    np.mean([row["response_tokens_scored"] for row in cell])
                ),
                "truncation_rate": float(
                    np.mean([row["truncated"] for row in cell])
                ),
            }
        )
    return output


def finalize() -> None:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    expected_protocol = protocol_id(manifest)
    summary_rows = []
    sample_rows = []
    inventory = []
    base_payload = json.loads(
        cell_path("opd", 0).read_text(encoding="utf-8")
    )
    if not cell_complete(cell_path("opd", 0), expected_protocol):
        raise RuntimeError("C8 base cell is incomplete")
    for arm in s3.ARMS:
        for step in s3.STEPS:
            path = cell_path(arm, step)
            if step == 0:
                payload = base_payload
                path = cell_path("opd", 0)
            else:
                if not cell_complete(path, expected_protocol):
                    raise RuntimeError(f"incomplete C8 cell {arm}/{step}")
                payload = json.loads(path.read_text(encoding="utf-8"))
            for row in summarize(payload["rows"]):
                summary_rows.append({"arm": arm, "step": step, **row})
            for row in payload["rows"]:
                sample_rows.append({"arm": arm, "step": step, **row})
            inventory.append(
                {
                    "arm": arm,
                    "step": step,
                    "cell_path": str(path),
                    "cell_sha256": s3.sha256_file(path),
                }
            )
    if len(summary_rows) != len(s3.ARMS) * len(s3.STEPS) * len(CORPORA):
        raise RuntimeError(f"C8 final summary rows={len(summary_rows)}")
    summary_output = s3.MINI / "C8_training_corpus_ppl.csv"
    samples_output = s3.MINI / "C8_training_corpus_ppl_samples.csv"
    inventory_output = s3.MINI / "C8_training_corpus_ppl_inventory.csv"
    s3.atomic_csv(summary_output, summary_rows)
    s3.atomic_csv(samples_output, sample_rows)
    s3.atomic_csv(inventory_output, inventory)
    output_manifest = s3.MINI / "C8_training_corpus_ppl_manifest.json"
    s3.atomic_json(
        output_manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "task": TASK,
            "contract": s3.artifact(s3.CONTRACT),
            "corpus_manifest": s3.artifact(CORPUS_MANIFEST),
            "protocol_id": expected_protocol,
            "arms": list(s3.ARMS),
            "steps": list(s3.STEPS),
            "outputs": [
                s3.artifact(summary_output),
                s3.artifact(samples_output),
                s3.artifact(inventory_output),
            ],
            "warning": (
                "X_OPD_reconstructed is a deterministic step-0 reconstruction "
                "because original online OPD responses were not archived"
            ),
        },
    )
    print(f"[C8 finalized] summary_rows={len(summary_rows)}", flush=True)


def smoke() -> None:
    fake = [
        {"sample_id": str(index), "input_token_ids": list(range(length))}
        for index, length in enumerate((3, 4, 7, 9, 12))
    ]
    batches = make_batches(fake, token_budget=18, max_batch_size=3)
    if any(len(batch) * max(len(row["input_token_ids"]) for row in batch) > 18 for batch in batches):
        raise RuntimeError(f"bad C8 batching: {batches}")
    toy = [
        {"corpus": "x", "nll_sum": 2.0, "response_tokens_scored": 2},
        {"corpus": "x", "nll_sum": 6.0, "response_tokens_scored": 3},
    ]
    value = sum(row["nll_sum"] for row in toy) / sum(
        row["response_tokens_scored"] for row in toy
    )
    if not np.isclose(value, 1.6):
        raise RuntimeError(f"bad token-weighted aggregation: {value}")

    class UniformModel:
        def __call__(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            logits = torch.zeros(
                (*input_ids.shape, 7), dtype=torch.float32, device=input_ids.device
            )
            return type("Output", (), {"logits": logits})()

    scored = score_batch(
        UniformModel(),
        [
            {
                "sample_id": "shift_test",
                "source_index": 0,
                "input_token_ids": [1, 2, 3, 4, 5],
                "prompt_tokens": 2,
                "response_tokens_original": 3,
                "response_tokens_scored": 3,
                "truncated": 0,
            }
        ],
        "cpu",
        pad_token_id=0,
        loss_chunk_tokens=2,
    )[0]
    if not np.isclose(scored["nll_sum"], 3 * math.log(7), atol=1e-6):
        raise RuntimeError(f"bad causal shift/NLL: {scored}")
    print(
        json.dumps(
            {
                "status": "ok",
                "batch_sizes": [len(x) for x in batches],
                "shift_test_nll": scored["nll_sum"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("prepare", "cells", "finalize", "all"), default="all"
    )
    parser.add_argument("--arms", default="all")
    parser.add_argument("--steps", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token-budget", type=int, default=24576)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--loss-chunk-tokens", type=int, default=128)
    parser.add_argument("--save-every-batches", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    s3.assert_contract()
    if args.smoke:
        smoke()
        return
    args.arms = s3.parse_names(args.arms, s3.ARMS)
    args.steps = s3.parse_ints(args.steps, s3.STEPS)
    if args.phase in ("prepare", "all"):
        prepare_corpora()
    if args.phase in ("cells", "all"):
        run_cells(args)
    if args.phase in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
