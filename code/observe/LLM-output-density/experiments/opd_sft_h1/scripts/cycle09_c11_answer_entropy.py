#!/usr/bin/env python3
"""C11: full-vocabulary next-token entropy on frozen MMLU-Pro answer positions."""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import cycle09_s1_8_mmlupro_loglik as s18
import cycle09_stage3_common as s3


TASK = "C11"
ROOT = s3.RUN_ROOT / "c11_answer_entropy"
CORPUS = ROOT / "corpus/mmlupro_answer_positions.jsonl"
CORPUS_MANIFEST = ROOT / "corpus/manifest.json"
CELL_ROOT = ROOT / "cells"
PROMPT_OUTPUT = s3.MINI / "C11_mmlupro_answer_position_prompt_template.txt"
N_QUESTIONS = 1400
N_CATEGORIES = 14
PROTOCOL_VERSION = "cycle09-c11-full-vocab-answer-entropy-v1"


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def ensure_source() -> Path:
    if not s3.MMLUPRO_INPUT.is_file():
        print("[C11 prepare] rebuilding frozen S1-8 input", flush=True)
        s18.prepare_input()
    if not s3.MMLUPRO_INPUT.is_file() or s3.MMLUPRO_INPUT.stat().st_size == 0:
        raise FileNotFoundError(s3.MMLUPRO_INPUT)
    return s3.MMLUPRO_INPUT


def prepare_corpus() -> dict[str, Any]:
    source = ensure_source()
    task_utils = s18.load_task_utils()
    template_text = task_utils.PROMPT_TEMPLATE + " "
    source_hash = s3.sha256_file(source)
    expected = {
        "protocol_version": PROTOCOL_VERSION,
        "source_sha256": source_hash,
        "n_questions": N_QUESTIONS,
        "prompt_template": template_text,
        "answer_position": "next token after the explicitly tokenized terminal space",
        "candidate_first_token_convention": (
            "tokenizer.encode(option, add_special_tokens=False)[0] after the "
            "explicit-space context state"
        ),
    }
    if CORPUS_MANIFEST.is_file() and CORPUS.is_file():
        old = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
        if all(old.get(key) == value for key, value in expected.items()):
            if not PROMPT_OUTPUT.is_file():
                s3.atomic_text(PROMPT_OUTPUT, template_text + "\n")
            return old
        raise RuntimeError(f"incompatible existing C11 corpus: {CORPUS}")

    source_rows = s3.read_jsonl(source)
    if len(source_rows) != N_QUESTIONS:
        raise RuntimeError(f"expected {N_QUESTIONS} MMLU-Pro rows, found {len(source_rows)}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(s3.BASE_MODEL), trust_remote_code=True
    )
    prepared = []
    for row in source_rows:
        context = task_utils.doc_to_text(row) + " "
        if not context.endswith("Answer: "):
            raise ValueError(f"bad C11 terminal context for {row['question_id']}")
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        if not context_ids:
            raise ValueError(f"empty C11 context for {row['question_id']}")
        options = task_utils.doc_to_choice(row)
        option_token_ids = [
            tokenizer.encode(option, add_special_tokens=False) for option in options
        ]
        if not 3 <= len(options) <= 10 or any(not ids for ids in option_token_ids):
            raise ValueError(f"bad C11 options for {row['question_id']}")
        answer_index = int(row["answer_index"])
        if not 0 <= answer_index < len(options):
            raise ValueError(f"bad C11 answer index for {row['question_id']}")
        prepared.append(
            {
                "sample_id": f"mmlupro_{int(row['question_id'])}",
                "question_id": int(row["question_id"]),
                "category": str(row["category"]),
                "context_text": context,
                "input_token_ids": [int(token) for token in context_ids],
                "option_first_token_ids": [int(ids[0]) for ids in option_token_ids],
                "answer_index": answer_index,
            }
        )
    category_counts = Counter(row["category"] for row in prepared)
    option_counts = Counter(len(row["option_first_token_ids"]) for row in prepared)
    if len(category_counts) != N_CATEGORIES or set(category_counts.values()) != {100}:
        raise RuntimeError(f"C11 category balance mismatch: {category_counts}")
    if len({row["question_id"] for row in prepared}) != N_QUESTIONS:
        raise RuntimeError("C11 question IDs are not unique")
    s3.atomic_jsonl(CORPUS, prepared)
    s3.atomic_text(PROMPT_OUTPUT, template_text + "\n")
    payload = {
        "schema_version": 1,
        **expected,
        "source_path": str(source),
        "corpus_path": str(CORPUS),
        "corpus_sha256": s3.sha256_file(CORPUS),
        "question_ids_sha256": s3.sha256_json(
            [row["question_id"] for row in prepared]
        ),
        "categories": dict(sorted(category_counts.items())),
        "option_count_distribution": {
            str(count): frequency for count, frequency in sorted(option_counts.items())
        },
        "primary": "full-vocabulary Shannon entropy in nats",
        "secondary": [
            "effective vocabulary exp(entropy)",
            "gold option first-token probability",
            "probability mass on all available unique option first tokens",
            "entropy renormalized over unique option first tokens",
        ],
    }
    s3.atomic_json(CORPUS_MANIFEST, payload)
    return payload


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
) -> list[dict[str, Any]]:
    maximum = max(len(row["input_token_ids"]) for row in samples)
    input_ids = torch.full(
        (len(samples), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention = torch.zeros_like(input_ids)
    lengths = []
    for index, row in enumerate(samples):
        ids = torch.tensor(row["input_token_ids"], dtype=torch.long, device=device)
        input_ids[index, : len(ids)] = ids
        attention[index, : len(ids)] = 1
        lengths.append(len(ids))
    indices = torch.arange(len(samples), device=device)
    positions = torch.tensor(lengths, dtype=torch.long, device=device) - 1
    output_head = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    if hasattr(model, "model") and output_head is not None:
        decoder_output = model.model(
            input_ids=input_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = decoder_output.last_hidden_state[indices, positions]
        logits = output_head(last_hidden).float()
        del decoder_output, last_hidden
    else:
        output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        logits = output.logits[indices, positions].float()
        del output
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropies = -(probs * log_probs).sum(dim=-1)
    if not torch.isfinite(entropies).all():
        raise FloatingPointError("non-finite C11 full-vocabulary entropy")
    entropy_values = entropies.cpu().numpy()
    results = []
    for index, row in enumerate(samples):
        candidate_ids = sorted(set(int(token) for token in row["option_first_token_ids"]))
        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        candidate_probs = probs[index, candidate_tensor]
        candidate_mass = candidate_probs.sum()
        if float(candidate_mass) <= 0:
            raise FloatingPointError(f"zero C11 candidate mass: {row['sample_id']}")
        normalized = candidate_probs / candidate_mass
        restricted_entropy = -(normalized * normalized.log()).sum()
        gold_id = int(row["option_first_token_ids"][int(row["answer_index"])])
        entropy = float(entropy_values[index])
        result = {
            "sample_id": row["sample_id"],
            "question_id": row["question_id"],
            "category": row["category"],
            "context_tokens": lengths[index],
            "vocab_size": logits.shape[-1],
            "full_vocab_entropy_nats": entropy,
            "effective_vocabulary": math.exp(entropy),
            "gold_option_first_token_id": gold_id,
            "gold_option_first_token_probability": float(probs[index, gold_id].cpu()),
            "option_first_token_unique_n": len(candidate_ids),
            "option_first_token_mass": float(candidate_mass.cpu()),
            "option_first_token_restricted_entropy_nats": float(
                restricted_entropy.cpu()
            ),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in result.items()
            if key
            in {
                "full_vocab_entropy_nats",
                "effective_vocabulary",
                "gold_option_first_token_probability",
                "option_first_token_mass",
                "option_first_token_restricted_entropy_nats",
            }
        ):
            raise FloatingPointError(f"non-finite C11 row: {row['sample_id']}")
        results.append(result)
        del candidate_tensor, candidate_probs, candidate_mass, normalized
    del logits, log_probs, probs, entropies, input_ids, attention, indices, positions
    torch.cuda.empty_cache()
    return results


def cell_path(arm: str, step: int) -> Path:
    label = "base" if step == 0 else arm
    return CELL_ROOT / label / s3.step_label(step) / "result.json"


def protocol_id(manifest: dict[str, Any]) -> str:
    return s3.sha256_json(
        {
            "version": PROTOCOL_VERSION,
            "corpus_sha256": manifest["corpus_sha256"],
            "answer_position": manifest["answer_position"],
            "candidate_first_token_convention": manifest[
                "candidate_first_token_convention"
            ],
            "numeric": (
                "bf16 decoder/output-head logits projected only at the final "
                "context position -> float32 log_softmax and entropy"
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
            and len(payload.get("rows", [])) == N_QUESTIONS
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
    samples: list[dict[str, Any]],
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path = cell_path(arm, step)
    expected_protocol = protocol_id(manifest)
    if cell_complete(path, expected_protocol):
        print(f"[C11 cached] {arm}/{step}", flush=True)
        return
    with lock(path.with_suffix(".lock")):
        if cell_complete(path, expected_protocol):
            return
        completed: dict[str, dict[str, Any]] = {}
        if path.is_file():
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("protocol_id") != expected_protocol:
                raise RuntimeError(f"incompatible C11 cell cache: {path}")
            completed = {row["sample_id"]: row for row in old.get("rows", [])}
        model_path = s3.require_model(arm, step)
        tokenizer = AutoTokenizer.from_pretrained(
            str(s3.BASE_MODEL), trust_remote_code=True
        )
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("C11 tokenizer has no pad/eos token")
        pending = [row for row in samples if row["sample_id"] not in completed]
        batches = make_batches(pending, args.token_budget, args.max_batch_size)
        print(
            f"[C11 load] {arm}/{step} cached={len(completed)}/{N_QUESTIONS} "
            f"batches={len(batches)}",
            flush=True,
        )
        model = load_model(model_path, args.device)
        try:
            since_save = 0
            for batch_index, batch in enumerate(batches, start=1):
                for row in score_batch(
                    model, batch, args.device, int(pad_token_id)
                ):
                    completed[row["sample_id"]] = row
                since_save += 1
                if since_save >= args.save_every_batches:
                    s3.atomic_json(
                        path,
                        {
                            "schema_version": 1,
                            "status": "partial",
                            "task": TASK,
                            "arm": "base" if step == 0 else arm,
                            "step": step,
                            "protocol_id": expected_protocol,
                            "model_path": str(model_path),
                            "rows": list(completed.values()),
                        },
                    )
                    since_save = 0
                print(
                    f"[C11] {arm}/{step} batch={batch_index}/{len(batches)} "
                    f"samples={len(completed)}/{N_QUESTIONS}",
                    flush=True,
                )
            ordered = [completed[row["sample_id"]] for row in samples]
            if len(ordered) != N_QUESTIONS:
                raise RuntimeError(f"incomplete C11 cell {arm}/{step}")
            s3.atomic_json(
                path,
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
    manifest = prepare_corpus()
    samples = s3.read_jsonl(CORPUS)
    if len(samples) != N_QUESTIONS:
        raise RuntimeError(f"C11 prepared rows={len(samples)}")
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
        run_cell(arm, step, samples, manifest, args)


SUMMARY_METRICS = (
    "full_vocab_entropy_nats",
    "effective_vocabulary",
    "gold_option_first_token_probability",
    "option_first_token_mass",
    "option_first_token_restricted_entropy_nats",
)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"n_questions": len(rows)}
    for metric in SUMMARY_METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_std"] = float(values.std(ddof=1))
        output[f"{metric}_median"] = float(np.median(values))
        output[f"{metric}_p05"] = float(np.quantile(values, 0.05))
        output[f"{metric}_p95"] = float(np.quantile(values, 0.95))
    output["context_tokens_mean"] = float(
        np.mean([row["context_tokens"] for row in rows])
    )
    output["option_first_token_unique_n_mean"] = float(
        np.mean([row["option_first_token_unique_n"] for row in rows])
    )
    return output


def finalize() -> None:
    manifest = json.loads(CORPUS_MANIFEST.read_text(encoding="utf-8"))
    expected_protocol = protocol_id(manifest)
    base_path = cell_path("opd", 0)
    if not cell_complete(base_path, expected_protocol):
        raise RuntimeError("C11 base cell is incomplete")
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    summary_rows = []
    category_rows = []
    sample_rows = []
    inventory = []
    for arm in s3.ARMS:
        for step in s3.STEPS:
            path = base_path if step == 0 else cell_path(arm, step)
            if not cell_complete(path, expected_protocol):
                raise RuntimeError(f"incomplete C11 cell {arm}/{step}")
            payload = (
                base_payload
                if step == 0
                else json.loads(path.read_text(encoding="utf-8"))
            )
            rows = payload["rows"]
            summary_rows.append({"arm": arm, "step": step, **summarize(rows)})
            categories = sorted({row["category"] for row in rows})
            if len(categories) != N_CATEGORIES:
                raise RuntimeError(f"C11 categories missing for {arm}/{step}")
            for category in categories:
                subset = [row for row in rows if row["category"] == category]
                if len(subset) != 100:
                    raise RuntimeError(
                        f"C11 category rows {arm}/{step}/{category}={len(subset)}"
                    )
                category_rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "category": category,
                        **summarize(subset),
                    }
                )
            sample_rows.extend({"arm": arm, "step": step, **row} for row in rows)
            inventory.append(
                {
                    "arm": arm,
                    "step": step,
                    "cell_path": str(path),
                    "cell_sha256": s3.sha256_file(path),
                }
            )
    expected_summary = len(s3.ARMS) * len(s3.STEPS)
    if len(summary_rows) != expected_summary:
        raise RuntimeError(f"C11 summary rows={len(summary_rows)}")
    summary_output = s3.MINI / "C11_mmlupro_answer_token_entropy.csv"
    category_output = s3.MINI / "C11_mmlupro_answer_token_entropy_by_category.csv"
    samples_output = s3.MINI / "C11_mmlupro_answer_token_entropy_samples.csv"
    inventory_output = s3.MINI / "C11_mmlupro_answer_token_entropy_inventory.csv"
    s3.atomic_csv(summary_output, summary_rows)
    s3.atomic_csv(category_output, category_rows)
    s3.atomic_csv(samples_output, sample_rows)
    s3.atomic_csv(inventory_output, inventory)
    output_manifest = s3.MINI / "C11_mmlupro_answer_token_entropy_manifest.json"
    s3.atomic_json(
        output_manifest,
        {
            "schema_version": 1,
            "status": "complete",
            "task": TASK,
            "contract": s3.artifact(s3.CONTRACT),
            "corpus_manifest": s3.artifact(CORPUS_MANIFEST),
            "prompt_template": s3.artifact(PROMPT_OUTPUT),
            "protocol_id": expected_protocol,
            "numeric": (
                "bf16 decoder/output-head logits projected only at the final "
                "context position; float32 log_softmax and entropy"
            ),
            "outputs": [
                s3.artifact(summary_output),
                s3.artifact(category_output),
                s3.artifact(samples_output),
                s3.artifact(inventory_output),
            ],
            "no_substitute": (
                "sequence-level option LL and empirical response frequency are "
                "not used as token entropy"
            ),
        },
    )
    print(f"[C11 finalized] summary_rows={len(summary_rows)}", flush=True)


def smoke() -> None:
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = float((-(log_probs.exp() * log_probs).sum()).item())
    if not np.isclose(entropy, math.log(2.0), atol=1e-7):
        raise RuntimeError(f"bad C11 entropy: {entropy}")
    fake = [
        {"sample_id": str(index), "input_token_ids": list(range(length))}
        for index, length in enumerate((3, 5, 8, 11))
    ]
    batches = make_batches(fake, token_budget=16, max_batch_size=3)
    if any(len(batch) * max(len(row["input_token_ids"]) for row in batch) > 16 for batch in batches):
        raise RuntimeError(f"bad C11 batching: {batches}")

    class UniformModel:
        def __call__(self, input_ids, attention_mask, use_cache):
            del attention_mask, use_cache
            values = torch.zeros(
                (*input_ids.shape, 4), dtype=torch.float32, device=input_ids.device
            )
            return type("Output", (), {"logits": values})()

    scored = score_batch(
        UniformModel(),
        [
            {
                "sample_id": "short",
                "question_id": 1,
                "category": "test",
                "input_token_ids": [1, 2],
                "option_first_token_ids": [0, 1, 1],
                "answer_index": 1,
            },
            {
                "sample_id": "long",
                "question_id": 2,
                "category": "test",
                "input_token_ids": [1, 2, 3],
                "option_first_token_ids": [2, 3, 3],
                "answer_index": 0,
            },
        ],
        "cpu",
        pad_token_id=0,
    )
    for row in scored:
        if not (
            np.isclose(row["full_vocab_entropy_nats"], math.log(4), atol=1e-6)
            and np.isclose(row["gold_option_first_token_probability"], 0.25)
            and np.isclose(row["option_first_token_mass"], 0.5)
            and np.isclose(
                row["option_first_token_restricted_entropy_nats"],
                math.log(2),
                atol=1e-6,
            )
        ):
            raise RuntimeError(f"bad C11 answer-position scoring: {row}")
    print(
        json.dumps(
            {
                "status": "ok",
                "binary_entropy_nats": entropy,
                "batch_sizes": [len(batch) for batch in batches],
                "answer_position_rows": len(scored),
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
    parser.add_argument("--token-budget", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--save-every-batches", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    s3.assert_contract()
    if args.smoke:
        smoke()
        return
    args.arms = s3.parse_names(args.arms, s3.ARMS)
    args.steps = s3.parse_ints(args.steps, s3.STEPS)
    if args.phase in ("prepare", "all"):
        prepare_corpus()
    if args.phase in ("cells", "all"):
        run_cells(args)
    if args.phase in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
