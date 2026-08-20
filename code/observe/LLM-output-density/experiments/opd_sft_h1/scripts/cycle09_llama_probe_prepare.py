#!/usr/bin/env python3
"""Freeze the six Llama probe corpora with cross-eval/training deduplication."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

import cycle09_block3_common as c
import cycle09_r4_common as c4


ROOT = c.RUN_ROOT / "llama_geometry"
CORPUS_ROOT = ROOT / "corpora"
MANIFEST = CORPUS_ROOT / "probe_manifest.json"
MATH_CAP = 32
AIME25_CAP = 30
S_MATH_CAP = 32
INSTRUCTION = "\nPlease reason step by step and put the final answer in \\boxed{}."

DEFAULT_MATH_CANDIDATES = (
    c.AUTODL / "dataset/hendrycks_math/test.jsonl",
    c.AUTODL / "dataset/MATH/test.jsonl",
    c.AUTODL / "prepared/hendrycks_math_test.jsonl",
)
DEFAULT_AIME25_CANDIDATES = (
    c.AUTODL / "dataset/aime25/test.parquet",
    c.AUTODL / "dataset/aime25/test.jsonl",
    c.AUTODL / "dataset/math-ai/aime25/test.parquet",
    c.AUTODL / "prepared/aime25.jsonl",
)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"\\(?:left|right)", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if path.suffix == ".jsonl":
        return c.read_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("test", "validation", "train", "data"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported JSON record container: {path}")


def text_from(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "Problem", "prompt", "text", "raw_problem"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("raw_problem", "problem", "question"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if str(message.get("role")) == "user":
                return str(message.get("content", "")).strip()
    output = row.get("output")
    if isinstance(output, dict) and output.get("text"):
        return str(output["text"]).strip()
    if isinstance(row.get("generation_text"), str):
        return str(row["generation_text"]).strip()
    raise KeyError("record has no recognized prompt text")


def find_source(explicit: Path | None, candidates: tuple[Path, ...], name: str) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing {name} source; pass --{name.replace('_', '-')} explicitly; "
        f"checked={list(map(str, candidates))}"
    )


def source_prompt_fingerprints() -> set[str]:
    frame = pd.read_parquet(c.SOURCE_PROMPTS)
    values = set()
    for prompt in frame["prompt"]:
        messages = prompt.tolist() if hasattr(prompt, "tolist") else list(prompt)
        text = str(messages[0]["content"])
        for suffix in (
            "\nPlease reason step by step, and put your final answer within \\boxed{}.",
            INSTRUCTION,
        ):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
        values.add(normalize(text))
    return values


def local_eval_fingerprints(path: Path) -> set[str]:
    return {normalize(text_from(row)) for row in read_records(path)}


def fixed_row(tokenizer: Any, sample_id: str, probe: str, text: str, source: str) -> dict[str, Any]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return {
        "sample_id": sample_id,
        "probe_type": "E",
        "domain": probe,
        "source_kind": source,
        "prompt_text": "",
        "generation_text": text,
        "prompt_token_ids": [],
        "generation_token_ids": list(map(int, ids)),
        "full_token_ids": list(map(int, ids)),
        "eligible_start": 0,
        "eligible_end": len(ids),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def write_probe(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows or any(row["eligible_end"] <= row["eligible_start"] for row in rows):
        raise RuntimeError(f"invalid probe rows for {path}")
    c.atomic_jsonl(path, rows)


def frozen_external(tokenizer: Any, probe: str, source: Path, expected: int) -> Path:
    target = CORPUS_ROOT / f"{probe}.jsonl"
    texts = [text_from(row) for row in read_records(source)[:expected]]
    if len(texts) != expected or len({normalize(text) for text in texts}) != expected:
        raise RuntimeError(f"{probe} source row/uniqueness drift: {source}")
    rows = [
        fixed_row(tokenizer, f"{probe}_{index:03d}", probe, text, f"retokenized:{source}")
        for index, text in enumerate(texts)
    ]
    write_probe(target, rows)
    return target


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = c.load_llama_tokenizer()
    math_source = find_source(args.math_source, DEFAULT_MATH_CANDIDATES, "math_source")
    aime25_source = find_source(args.aime25_source, DEFAULT_AIME25_CANDIDATES, "aime25_source")
    training = source_prompt_fingerprints()
    math500_path = c.REPO / "Eval/tasks/data/hendrycks_math500/test.jsonl"
    aime24_path = c.REPO / "Eval/tasks/data/aime24/train.jsonl"
    math500 = local_eval_fingerprints(math500_path)
    aime24 = local_eval_fingerprints(aime24_path)

    math_candidates = []
    seen = set()
    removed_math500 = removed_training = removed_duplicate = 0
    for row in read_records(math_source):
        text = text_from(row)
        fingerprint = normalize(text)
        if fingerprint in math500:
            removed_math500 += 1
        elif fingerprint in training:
            removed_training += 1
        elif fingerprint in seen:
            removed_duplicate += 1
        else:
            seen.add(fingerprint)
            math_candidates.append(text)
    random.Random(c.SEED).shuffle(math_candidates)
    if len(math_candidates) < MATH_CAP:
        raise RuntimeError(f"E_math has only {len(math_candidates)} eligible rows")
    math_texts = math_candidates[:MATH_CAP]
    e_math = CORPUS_ROOT / "E_math.jsonl"
    write_probe(
        e_math,
        [
            fixed_row(tokenizer, f"E_math_{index:03d}", "math", text, str(math_source))
            for index, text in enumerate(math_texts)
        ],
    )

    hard_texts = []
    hard_seen = set()
    removed_aime24 = removed_hard_training = 0
    for row in read_records(aime25_source):
        text = text_from(row)
        fingerprint = normalize(text)
        if fingerprint in aime24:
            removed_aime24 += 1
        elif fingerprint in training:
            removed_hard_training += 1
        elif fingerprint not in hard_seen:
            hard_seen.add(fingerprint)
            hard_texts.append(text)
    if len(hard_texts) != AIME25_CAP:
        raise RuntimeError(
            f"E_math_hard_v2 requires exactly 30 AIME25 rows after dedup; got={len(hard_texts)}"
        )
    e_hard = CORPUS_ROOT / "E_math_hard_v2.jsonl"
    write_probe(
        e_hard,
        [
            fixed_row(tokenizer, f"E_math_hard_v2_{index:03d}", "math_hard_v2", text, str(aime25_source))
            for index, text in enumerate(hard_texts)
        ],
    )

    r4_fixed = c.AUTODL / "cycle09_r4/corpora/fixed"
    e_ood = frozen_external(tokenizer, "E_ood", r4_fixed / "E_ood.jsonl", 128)
    e_general = frozen_external(tokenizer, "E_general", r4_fixed / "E_general.jsonl", 128)
    ifeval_path = c.REPO / "Eval/tasks/data/ifeval/train.jsonl"
    e_if = frozen_external(tokenizer, "E_if", ifeval_path, 541)

    outputs = [e_math, e_hard, e_ood, e_if, e_general]
    manifest = {
        "schema_version": 1,
        "status": "fixed_probes_complete",
        "created_utc": c.utc_now(),
        "tokenizer": str(c.LLAMA_STUDENT),
        "window_protocol": {
            "window_seed": c4.WINDOW_SEED,
            "window_tokens": c4.WINDOW_TOKENS,
            "window_k": c4.WINDOW_K,
            "normalization": "window token mean -> sample window mean -> equal sample mean",
        },
        "sources": {
            "E_math": str(math_source),
            "E_math_hard_v2": str(aime25_source),
            "E_ood": str(r4_fixed / "E_ood.jsonl"),
            "E_if": str(ifeval_path),
            "E_general": str(r4_fixed / "E_general.jsonl"),
        },
        "dedup": {
            "normalization": "NFKC/lower/alphanumeric-only",
            "E_math_removed_math500": removed_math500,
            "E_math_removed_training": removed_training,
            "E_math_removed_duplicates": removed_duplicate,
            "E_math_hard_v2_removed_aime24": removed_aime24,
            "E_math_hard_v2_removed_training": removed_hard_training,
        },
        "row_counts": {path.stem: len(c.read_jsonl(path)) for path in outputs},
        "outputs": [c.artifact(path) for path in outputs],
        "S_math": "pending_generation",
    }
    c.atomic_json(MANIFEST, manifest)
    return manifest


def generate_s_math(args: argparse.Namespace) -> dict[str, Any]:
    manifest = c.read_json(MANIFEST, {})
    if manifest.get("status") not in {"fixed_probes_complete", "complete"}:
        raise RuntimeError("fixed probes must be prepared before S_math")
    target = CORPUS_ROOT / "S_math.jsonl"
    if target.is_file() and len(c.read_jsonl(target)) == S_MATH_CAP:
        manifest["status"] = "complete"
        manifest["S_math"] = {
            "source": str(c.LLAMA_STUDENT),
            "n": S_MATH_CAP,
            "temperature": 0.6,
            "top_p": 0.9,
            "max_new_tokens": 1024,
            "seed_rule": "sha256 stable_seed(42,S_math,row)",
            "output": c.artifact(target),
            "cached": True,
        }
        manifest["updated_utc"] = c.utc_now()
        c.atomic_json(MANIFEST, manifest)
        return manifest
    from vllm import LLM, SamplingParams

    c.ensure_llama_runtime_model()
    tokenizer = c.load_llama_tokenizer()
    source = c.read_jsonl(CORPUS_ROOT / "E_math.jsonl")[:S_MATH_CAP]
    texts = [row["generation_text"] + INSTRUCTION for row in source]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        for text in texts
    ]
    llm = LLM(
        model=str(c.LLAMA_STUDENT_RUNTIME),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        trust_remote_code=True,
    )
    try:
        requests = llm.generate(
            prompts,
            [
                SamplingParams(
                    temperature=0.6,
                    top_p=0.9,
                    max_tokens=1024,
                    seed=c4.stable_seed(c.SEED, "S_math", index),
                )
                for index in range(S_MATH_CAP)
            ],
        )
        rows = []
        for index, request in enumerate(requests):
            output = request.outputs[0]
            prompt_ids = list(map(int, request.prompt_token_ids))
            generation_ids = list(map(int, output.token_ids))
            rows.append(
                {
                    "sample_id": f"S_math_{index:03d}",
                    "probe_type": "S",
                    "domain": "math",
                    "source_kind": "llama_base_common_support_anchor",
                    "generation_seed": c4.stable_seed(c.SEED, "S_math", index),
                    "prompt_text": texts[index],
                    "formatted_prompt": prompts[index],
                    "generation_text": output.text,
                    "prompt_token_ids": prompt_ids,
                    "generation_token_ids": generation_ids,
                    "full_token_ids": prompt_ids + generation_ids,
                    "eligible_start": len(prompt_ids),
                    "eligible_end": len(prompt_ids) + len(generation_ids),
                    "finish_reason": output.finish_reason,
                }
            )
        write_probe(target, rows)
    finally:
        del llm
    manifest["status"] = "complete"
    manifest["S_math"] = {
        "source": str(c.LLAMA_STUDENT),
        "n": S_MATH_CAP,
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 1024,
        "seed_rule": "sha256 stable_seed(42,S_math,row)",
        "output": c.artifact(target),
    }
    manifest["updated_utc"] = c.utc_now()
    c.atomic_json(MANIFEST, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fixed", "s_math", "all"), default="all")
    parser.add_argument("--math-source", type=Path)
    parser.add_argument("--aime25-source", type=Path)
    parser.add_argument("--gpu-mem", type=float, default=0.75)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = prepare(arguments) if arguments.phase in {"fixed", "all"} else {}
    if arguments.phase in {"s_math", "all"}:
        result = generate_s_math(arguments)
    print(json.dumps(result, indent=2))
