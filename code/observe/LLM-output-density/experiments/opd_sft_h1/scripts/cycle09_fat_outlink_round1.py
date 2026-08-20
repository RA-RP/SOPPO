#!/usr/bin/env python3
"""FAT-R1 teacher-forced output-link decomposition.

This runner implements only the confirmed first round in
``stage_plan_handoff.md``: MMLU-Pro and MATH500 span-level NLL/KL under the
formal deployed checkpoint states.  It never trains, never rolls out, and never
stores full sequence-by-vocabulary tensors.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


REPO = Path("/root/LLM-output-density")
SIDE = REPO / "experiments/opd_sft_h1"
SCRIPTS = SIDE / "scripts"
for item in (SIDE, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import cycle09_block3_common as b3  # noqa: E402
import cycle09_llama_model_export as lexport  # noqa: E402
import cycle09_offkd_eval as offkd_eval  # noqa: E402
import cycle09_stage3_common as qstage  # noqa: E402
from scripts.run_opd_minimal_closure import merge_lora_adapter  # noqa: E402


TASK = "FAT-R1-v2"
SCHEMA = "cycle09_fat_outlink_round1_v2"
AUTODL = Path("/root/autodl-tmp")
MINI = (
    REPO
    / "mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
)
OUT = MINI / "fat_outlink_round1_v2"
SCRATCH = AUTODL / "cycle09_fat_outlink_round1_v2"
BASE_CACHE = SCRATCH / "base_cache"
TEMP_MERGED = SCRATCH / "tmp_merged"
LOG = SCRATCH / "logs"

MMLU_PATH = AUTODL / "cycle09_s1/mmlupro_loglik/input/mmlupro_1400.jsonl"
MATH_PATH = REPO / "Eval/tasks/data/hendrycks_math500/test.jsonl"
EQUAL5_MANIFEST = MINI / "equal5_non_qk/EQUAL5_manifest.json"

QWEN_STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
LLAMA_STEPS = (0, 5, 20, 40, 80, 160, 320)
ARMS = ("opd", "sft", "offkd", "seqkd")
MODELS = ("qwen", "llama")
DOMAINS = ("mmlu", "math")
KL_SPANS = ("f_pre", "f_post", "f", "a", "b", "t")
NLL_SPANS = ("p", "c", "f_pre", "f_post", "f", "a", "b", "t")
LETTERS = "ABCDEFGHIJ"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(value)
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) if rows else []
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


@contextmanager
def file_lock(path: Path):
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def append_code_evolution(text: str) -> None:
    path = REPO / "mypaper/code/code_evolution.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text.strip() + "\n")


def step_label(step: int) -> str:
    return f"step_{int(step):03d}"


def checkpoint_label(step: int) -> str:
    return f"checkpoint-{int(step):06d}"


def qwen_adapter_path(arm: str, step: int) -> Path:
    if arm == "sft":
        return AUTODL / "cycle07_base_sft_trajectory/checkpoints" / step_label(step)
    if arm == "offkd":
        return offkd_eval.adapter_path(AUTODL / "cycle09_offkd", step)
    if arm == "seqkd":
        return AUTODL / "cycle09_seqkd/checkpoints" / checkpoint_label(step)
    raise ValueError(f"no adapter path for {arm}")


def adapter_complete(path: Path, require_complete: bool) -> bool:
    ok = (path / "adapter_config.json").is_file() and (path / "adapter_model.safetensors").is_file()
    return ok and ((path / "complete.json").is_file() if require_complete else True)


def model_complete_qwen(path: Path) -> bool:
    return bool(qstage.model_integrity(path).get("complete"))


def model_complete_llama(path: Path) -> bool:
    return bool(b3.model_check(path).get("complete"))


def direct_model_path(model: str, arm: str, step: int) -> Path:
    if model == "qwen":
        if step == 0:
            return qstage.BASE_MODEL
        return qstage.model_path(arm, step)
    if model == "llama":
        return lexport.merged_target(arm, step)
    raise ValueError(model)


def expected_states() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, steps in (("qwen", QWEN_STEPS), ("llama", LLAMA_STEPS)):
        rows.append({"model": model, "arm": "base", "step": 0, "is_shared_base": True})
        for arm in ARMS:
            for step in steps:
                if step:
                    rows.append({"model": model, "arm": arm, "step": step, "is_shared_base": False})
    return rows


def checkpoint_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in expected_states():
        model, arm, step = row["model"], row["arm"], int(row["step"])
        path = direct_model_path(model, "opd" if arm == "base" else arm, step)
        status = "MISSING"
        source_type = "direct_merged"
        source_path = path
        adapter = ""
        if model == "llama":
            status = "READY_DIRECT_MERGED" if model_complete_llama(path) else "BLOCKED_MISSING_MERGED"
        elif step == 0:
            status = "READY_DIRECT_BASE" if model_complete_qwen(path) else "BLOCKED_MISSING_BASE"
        elif arm == "opd":
            status = "READY_DIRECT_MERGED" if model_complete_qwen(path) else "BLOCKED_MISSING_MERGED"
        else:
            adapter_path = qwen_adapter_path(arm, step)
            need_complete = arm in ("offkd", "seqkd")
            adapter = str(adapter_path)
            source_type = "ephemeral_bf16_merge_from_adapter"
            source_path = adapter_path
            status = "READY_EPHEMERAL_BF16_MERGE" if adapter_complete(adapter_path, need_complete) else "BLOCKED_MISSING_ADAPTER"
        rows.append({
            **row,
            "model_path": str(path),
            "source_path": str(source_path),
            "adapter_path": adapter,
            "source_type": source_type,
            "status": status,
        })
    return rows


@contextmanager
def materialized_qwen(arm: str, step: int):
    if step == 0 or arm == "opd":
        yield direct_model_path("qwen", arm if step else "opd", step)
        return
    adapter = qwen_adapter_path(arm, step)
    target = TEMP_MERGED / "qwen" / arm / step_label(step)
    try:
        shutil.rmtree(target, ignore_errors=True)
        merge_lora_adapter(qstage.BASE_MODEL, adapter, target)
        yield target
    finally:
        shutil.rmtree(target, ignore_errors=True)


@contextmanager
def materialized_model(model: str, arm: str, step: int):
    if model == "qwen":
        with materialized_qwen(arm, step) as path:
            yield path
    else:
        yield direct_model_path("llama", arm, step)


def load_tokenizer(model: str, path: Path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path: Path, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.eval().to(device)
    return model


def unload_model(model: Any) -> None:
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_answer(value: str) -> str:
    return "".join(str(value).strip().split())


def find_boxed_spans(solution: str) -> list[tuple[int, int, int, int]]:
    spans = []
    marker = r"\boxed{"
    start = 0
    while True:
        idx = solution.find(marker, start)
        if idx < 0:
            break
        open_idx = idx + len(marker) - 1
        depth = 0
        close_idx = -1
        for pos in range(open_idx, len(solution)):
            char = solution[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    close_idx = pos
                    break
        if close_idx >= 0:
            spans.append((idx, idx + len(marker), open_idx + 1, close_idx))
            start = close_idx + 1
        else:
            start = idx + len(marker)
    return spans


@dataclass
class PreparedSample:
    domain: str
    sample_id: str
    category: str
    text: str
    spans: dict[str, list[tuple[int, int]]]
    metadata: dict[str, Any]


def load_mmlu() -> list[dict[str, Any]]:
    rows = read_jsonl(MMLU_PATH)
    if len(rows) != 1400:
        raise RuntimeError(f"MMLU-Pro row count drift: {len(rows)}")
    ids = [str(row["question_id"]) for row in rows]
    if len(set(ids)) != 1400:
        raise RuntimeError("MMLU-Pro question_id is not unique")
    counts = Counter(row["category"] for row in rows)
    bad = {key: value for key, value in counts.items() if value != 100}
    if len(counts) != 14 or bad:
        raise RuntimeError(f"MMLU-Pro category parity drift: counts={dict(counts)}")
    return rows


def load_math() -> list[dict[str, Any]]:
    rows = read_jsonl(MATH_PATH)
    ids = [str(row["unique_id"]) for row in rows]
    if len(rows) != 500 or len(set(ids)) != 500:
        raise RuntimeError(f"MATH500 unique_id drift rows={len(rows)} unique={len(set(ids))}")
    return rows


def prepare_mmlu(rows: list[dict[str, Any]]) -> list[PreparedSample]:
    prepared = []
    for row in rows:
        options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(row["options"]))
        p = f"Question:\n{row['question']}\nOptions:\n{options}\nAnswer:\n"
        fpre = "The answer is ("
        answer = str(row["answer"])
        fpost = ")."
        text = p + fpre + answer + fpost
        p0 = 0
        fpre0 = len(p)
        a0 = fpre0 + len(fpre)
        fpost0 = a0 + len(answer)
        prepared.append(PreparedSample(
            domain="mmlu",
            sample_id=str(row["question_id"]),
            category=str(row["category"]),
            text=text,
            spans={
                "p": [(p0, len(p))],
                "f_pre": [(fpre0, a0)],
                "a": [(a0, fpost0)],
                "f_post": [(fpost0, len(text))],
                "c": [],
            },
            metadata={
                "question_id": str(row["question_id"]),
                "category": str(row["category"]),
                "answer": answer,
                "answer_index": int(row["answer_index"]),
                "source_doc_hash": row.get("source_doc_hash", ""),
            },
        ))
    return prepared


def prepare_math(rows: list[dict[str, Any]]) -> tuple[list[PreparedSample], list[dict[str, Any]]]:
    prepared = []
    parse_rows = []
    for row in rows:
        solution = str(row["solution"])
        boxes = find_boxed_spans(solution)
        status = "ok" if boxes else "missing_boxed"
        answer_consistent = False
        parsed_answer = ""
        if boxes:
            box_start, fpre_end, answer_start, answer_end = boxes[-1]
            close_idx = answer_end
            parsed_answer = solution[answer_start:answer_end]
            answer_consistent = normalize_answer(parsed_answer) == normalize_answer(row.get("answer", ""))
            prompt = f"Problem:\n{row['problem']}\nSolution:\n"
            offset = len(prompt)
            text = prompt + solution
            box_abs_start = offset + box_start
            box_abs_end = offset + close_idx + 1
            prepared.append(PreparedSample(
                domain="math",
                sample_id=str(row["unique_id"]),
                category=f"{row.get('subject', '')}|level_{row.get('level', '')}",
                text=text,
                spans={
                    "p": [(0, offset)],
                    "c": [(offset, len(text))],
                    "b": [(box_abs_start, box_abs_end)],
                    "f_pre": [],
                    "a": [],
                    "f_post": [],
                },
                metadata={
                    "unique_id": str(row["unique_id"]),
                    "subject": row.get("subject", ""),
                    "level": row.get("level", ""),
                    "answer": row.get("answer", ""),
                    "parsed_answer": parsed_answer,
                    "n_boxed_occurrences": len(boxes),
                    "answer_consistent": answer_consistent,
                    "box_char_start": box_abs_start,
                    "box_char_end": box_abs_end,
                    "box_solution_char_start": box_start,
                    "box_solution_char_end": close_idx + 1,
                    "span_protocol": "token_closed_final_box_v2",
                    "format_answer_separable": False,
                },
            ))
        parse_rows.append({
            "domain": "math",
            "sample_id": str(row["unique_id"]),
            "status": status,
            "n_boxed_occurrences": len(boxes),
            "parsed_answer": parsed_answer,
            "answer": row.get("answer", ""),
            "answer_consistent": answer_consistent,
        })
    if len(prepared) != 500:
        raise RuntimeError(f"MATH500 boxed parse blocked: prepared={len(prepared)}")
    return prepared, parse_rows


def token_hash(tok: Any) -> str:
    vocab = tok.get_vocab()
    payload = {
        "name_or_path": getattr(tok, "name_or_path", ""),
        "vocab_size": len(vocab),
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "bos_token_id": tok.bos_token_id,
        "unk_token_id": tok.unk_token_id,
        "special_tokens_map": tok.special_tokens_map,
    }
    return sha256_json(payload)


def encode_with_regions(tok: Any, sample: PreparedSample) -> dict[str, Any]:
    enc = tok(sample.text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    ids.append(int(tok.eos_token_id))
    offsets.append((len(sample.text), len(sample.text)))
    region_positions: dict[str, list[int]] = {key: [] for key in ("p", "c", "f_pre", "a", "f_post", "b", "t")}
    crossings: list[dict[str, Any]] = []
    box_audit: dict[str, Any] = {}
    if sample.domain == "math":
        prompt_end = sample.spans["p"][0][1]
        box_start = int(sample.metadata["box_char_start"])
        box_end = int(sample.metadata["box_char_end"])
        raw_solution_tokens = []
        box_tokens = []
        prompt_crossings = []
        for idx, (start, end) in enumerate(offsets[:-1]):
            if start == end:
                continue
            if start < prompt_end and end <= prompt_end:
                region_positions["p"].append(idx)
            elif start >= prompt_end:
                raw_solution_tokens.append(idx)
            elif start < prompt_end < end:
                prompt_crossings.append(idx)
            if end > box_start and start < box_end:
                box_tokens.append(idx)
        box_set = set(box_tokens)
        region_positions["b"] = sorted(box_tokens)
        region_positions["c"] = [idx for idx in raw_solution_tokens if idx not in box_set]
        if prompt_crossings:
            for idx in prompt_crossings:
                start, end = offsets[idx]
                crossings.append({
                    "sample_id": sample.sample_id,
                    "region": "p_solution_boundary",
                    "token_index": idx,
                    "token_start": start,
                    "token_end": end,
                    "span_start": prompt_end,
                    "span_end": prompt_end,
                    "token_text": sample.text[start:end],
                })
        if box_tokens:
            first, last = min(box_tokens), max(box_tokens)
            left_start, left_end = offsets[first]
            right_start, right_end = offsets[last]
            covered = set(region_positions["b"]) | set(region_positions["c"])
            disjoint = not (set(region_positions["b"]) & set(region_positions["c"]))
            box_audit = {
                "box_char_start": box_start,
                "box_char_end": box_end,
                "box_token_first": first,
                "box_token_last": last,
                "n_tokens_b": len(region_positions["b"]),
                "n_tokens_c": len(region_positions["c"]),
                "left_spill_chars": max(0, box_start - left_start),
                "right_spill_chars": max(0, right_end - box_end),
                "left_boundary_token_text": sample.text[left_start:left_end],
                "right_boundary_token_text": sample.text[right_start:right_end],
                "c_intersect_b_empty": disjoint,
                "c_union_b_covers_raw_solution_tokens": covered == set(raw_solution_tokens),
                "span_protocol": "token_closed_final_box_v2",
                "format_answer_separable": False,
            }
    else:
        for idx, (start, end) in enumerate(offsets[:-1]):
            for region, spans in sample.spans.items():
                for span_start, span_end in spans:
                    if end <= span_start or start >= span_end:
                        continue
                    if start >= span_start and end <= span_end:
                        region_positions[region].append(idx)
                    else:
                        crossings.append({
                            "sample_id": sample.sample_id,
                            "region": region,
                            "token_index": idx,
                            "token_start": start,
                            "token_end": end,
                            "span_start": span_start,
                            "span_end": span_end,
                            "token_text": sample.text[start:end],
                        })
    region_positions["t"].append(len(ids) - 1)
    region_positions["f"] = sorted(region_positions["f_pre"] + region_positions["f_post"])
    return {
        "input_ids": ids,
        "offsets": offsets,
        "regions": region_positions,
        "crossings": crossings,
        "box_audit": box_audit,
        "n_chars": len(sample.text),
    }


def build_token_manifest(limit_mmlu: int = 0, limit_math: int = 0) -> dict[str, Any]:
    mmlu = prepare_mmlu(load_mmlu())
    math_samples, math_parse = prepare_math(load_math())
    if limit_mmlu:
        mmlu = mmlu[:limit_mmlu]
    if limit_math:
        math_samples = math_samples[:limit_math]
    rows = {"mmlu": mmlu, "math": math_samples}
    model_tokenizers = {}
    mask_rows = []
    sample_rows = []
    manifests: dict[str, Any] = {
        "schema_version": SCHEMA,
        "task": TASK,
        "created_utc": utc_now(),
        "domains": {},
        "models": {},
    }
    for model in MODELS:
        base_path = direct_model_path(model, "opd", 0)
        tok = load_tokenizer(model, base_path)
        model_tokenizers[model] = tok
        manifests["models"][model] = {
            "tokenizer_path": str(base_path),
            "tokenizer_hash": token_hash(tok),
            "eos_token_id": tok.eos_token_id,
            "eos_token": tok.eos_token,
            "pad_token_id": tok.pad_token_id,
            "padding_side": tok.padding_side,
            "chat_template_used": False,
            "template_state": "plain_completion_teacher_forcing_no_chat_template",
        }
        for domain, samples in rows.items():
            for sample in samples:
                encoded = encode_with_regions(tok, sample)
                for crossing in encoded["crossings"]:
                    mask_rows.append({
                        "model": model,
                        "domain": domain,
                        "sample_id": sample.sample_id,
                        "boundary_crossing": True,
                        **crossing,
                    })
                if domain == "math":
                    mask_rows.append({
                        "model": model,
                        "domain": domain,
                        "sample_id": sample.sample_id,
                        "boundary_crossing": False,
                        **encoded["box_audit"],
                    })
                nonempty_ok = bool(encoded["regions"]["f_pre"] and encoded["regions"]["a"] and encoded["regions"]["f_post"] and encoded["regions"]["t"])
                if domain == "math":
                    nonempty_ok = bool(
                        encoded["regions"]["b"]
                        and encoded["regions"]["c"]
                        and encoded["regions"]["t"]
                        and encoded["box_audit"].get("c_intersect_b_empty")
                        and encoded["box_audit"].get("c_union_b_covers_raw_solution_tokens")
                    )
                sample_rows.append({
                    "model": model,
                    "domain": domain,
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "n_input_tokens": len(encoded["input_ids"]),
                    "n_tokens_p": len(encoded["regions"]["p"]),
                    "n_tokens_c": len(encoded["regions"]["c"]),
                    "n_tokens_f_pre": len(encoded["regions"]["f_pre"]),
                    "n_tokens_a": len(encoded["regions"]["a"]),
                    "n_tokens_f_post": len(encoded["regions"]["f_post"]),
                    "n_tokens_b": len(encoded["regions"]["b"]),
                    "n_tokens_t": len(encoded["regions"]["t"]),
                    "boundary_crossing": bool(encoded["crossings"]),
                    "span_nonempty_ok": nonempty_ok,
                    "text_sha256": hashlib.sha256(sample.text.encode("utf-8")).hexdigest(),
                    **{key: encoded["box_audit"].get(key, "") for key in (
                        "box_char_start", "box_char_end", "box_token_first", "box_token_last",
                        "left_spill_chars", "right_spill_chars", "span_protocol",
                        "format_answer_separable",
                    )},
                })
    for domain, samples in rows.items():
        manifests["domains"][domain] = {
            "sample_count": len(samples),
            "sample_ids": [sample.sample_id for sample in samples],
            "sample_ids_sha256": sha256_json([sample.sample_id for sample in samples]),
            "template_hash": sha256_json({
                "mmlu": "Question:\\n{question}\\nOptions:\\nA. ...\\nAnswer:\\nThe answer is ({gold_letter}).<EOS>",
                "math": "Problem:\\n{problem}\\nSolution:\\n{raw_solution}<EOS>",
                "math_direct_no_cot": "explicitly_removed",
            }.get(domain)),
        }
    atomic_csv(OUT / "fat_r1_v2_mask_audit.csv", mask_rows)
    atomic_csv(OUT / "fat_r1_v2_sample_manifest_index.csv", sample_rows)
    atomic_json(OUT / "fat_r1_v2_sample_manifest.json", {
        **manifests,
        "math_parse_audit": math_parse,
        "sample_manifest_index": str(OUT / "fat_r1_v2_sample_manifest_index.csv"),
        "sample_manifest_index_sha256": sha256_file(OUT / "fat_r1_v2_sample_manifest_index.csv"),
        "mask_audit": str(OUT / "fat_r1_v2_mask_audit.csv"),
        "mask_audit_sha256": sha256_file(OUT / "fat_r1_v2_mask_audit.csv"),
    })
    atomic_json(OUT / "fat_r1_v2_template_and_tokenizer_manifest.json", manifests)
    hard_crossings = [row for row in mask_rows if row.get("boundary_crossing")]
    if hard_crossings:
        raise RuntimeError(f"FAILED_TOKEN_BOUNDARY_PROTOCOL: {len(hard_crossings)} boundary crossings")
    bad = [row for row in sample_rows if not row["span_nonempty_ok"]]
    if bad:
        raise RuntimeError(f"FAILED_EMPTY_SPAN_PROTOCOL: {bad[:5]}")
    return {"samples": rows, "tokenizers": model_tokenizers, "manifest": manifests, "sample_rows": sample_rows}


def select_smoke_samples(samples: dict[str, list[PreparedSample]]) -> dict[str, list[PreparedSample]]:
    mmlu = samples["mmlu"]
    seen = set()
    picked = []
    for sample in mmlu:
        if sample.category not in seen:
            picked.append(sample)
            seen.add(sample.category)
        if len(seen) == 14:
            break
    return {"mmlu": picked, "math": samples["math"][:20]}


def make_batches(encoded: list[dict[str, Any]], max_batch_tokens: int) -> list[list[dict[str, Any]]]:
    ordered = sorted(encoded, key=lambda row: row["length"])
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_max = 0
    for row in ordered:
        projected_max = max(current_max, row["length"])
        projected = projected_max * (len(current) + 1)
        if current and projected > max_batch_tokens:
            batches.append(current)
            current = []
            current_max = 0
        current.append(row)
        current_max = max(current_max, row["length"])
    if current:
        batches.append(current)
    return batches


def encoded_records(tok: Any, samples: list[PreparedSample]) -> list[dict[str, Any]]:
    records = []
    for sample in samples:
        encoded = encode_with_regions(tok, sample)
        target_positions = sorted({pos for span in NLL_SPANS for pos in encoded["regions"].get(span, []) if pos > 0})
        kl_source_spans = ("b", "t") if sample.domain == "math" else ("f_pre", "f_post", "a", "t")
        kl_positions = sorted({pos for span in kl_source_spans for pos in encoded["regions"].get(span, []) if pos > 0})
        records.append({
            "sample": sample,
            "input_ids": encoded["input_ids"],
            "regions": encoded["regions"],
            "length": len(encoded["input_ids"]),
            "target_positions": target_positions,
            "kl_positions": kl_positions,
            "predict_positions": sorted(set(pos - 1 for pos in target_positions)),
        })
    return records


def pad_batch(batch: list[dict[str, Any]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(row["length"] for row in batch)
    ids = torch.full((len(batch), max_len), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, row in enumerate(batch):
        values = torch.tensor(row["input_ids"], dtype=torch.long)
        ids[i, : len(values)] = values
        mask[i, : len(values)] = 1
    return ids.to(device), mask.to(device)


@torch.no_grad()
def forward_selected(
    model_obj: Any,
    tok: Any,
    records: list[dict[str, Any]],
    device: str,
    max_batch_tokens: int,
    keep_logits: bool,
) -> list[dict[str, Any]]:
    rows = []
    for batch in make_batches(records, max_batch_tokens):
        input_ids, attention = pad_batch(batch, tok.pad_token_id, device)
        union_keep = sorted({pos for row in batch for pos in row["predict_positions"]})
        keep_tensor = torch.tensor(union_keep, dtype=torch.long, device=device)
        outputs = model_obj(input_ids=input_ids, attention_mask=attention, use_cache=False, logits_to_keep=keep_tensor)
        logits = outputs.logits.detach().float()
        keep_index = {pos: idx for idx, pos in enumerate(union_keep)}
        for bi, record in enumerate(batch):
            sample = record["sample"]
            token_rows = []
            for target_pos in record["target_positions"]:
                logit = logits[bi, keep_index[target_pos - 1]]
                logp = torch.log_softmax(logit, dim=-1)
                target_id = int(record["input_ids"][target_pos])
                token_row = {
                    "sample_id": sample.sample_id,
                    "target_pos": target_pos,
                    "target_id": target_id,
                    "nll": float(-logp[target_id].item()),
                }
                if keep_logits and target_pos in record["kl_positions"]:
                    token_row["logits"] = logit.to("cpu", dtype=torch.bfloat16).contiguous()
                token_rows.append(token_row)
            rows.append({
                "sample_id": sample.sample_id,
                "metadata": sample.metadata,
                "category": sample.category,
                "regions": record["regions"],
                "target_token_rows": token_rows,
            })
        del logits, outputs, input_ids, attention
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def base_cache_path(model: str, domain: str) -> Path:
    return BASE_CACHE / model / f"{domain}.pt"


def base_cache_meta(model: str, domain: str) -> Path:
    return base_cache_path(model, domain).with_suffix(".json")


def ensure_base_cache(model: str, tok: Any, samples: list[PreparedSample], device: str, max_batch_tokens: int) -> dict[str, Any]:
    path = base_cache_path(model, samples[0].domain)
    meta = base_cache_meta(model, samples[0].domain)
    expected_sample_ids = [sample.sample_id for sample in samples]
    expected_hash = sha256_json(expected_sample_ids)
    with file_lock(path):
        cached = read_json(meta, {})
        if cached.get("status") == "complete" and cached.get("sample_ids_sha256") == expected_hash and path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)
        with materialized_model(model, "opd", 0) as model_path:
            model_obj = load_model(model_path, device)
            try:
                rows = forward_selected(
                    model_obj,
                    tok,
                    encoded_records(tok, samples),
                    device,
                    max_batch_tokens=max_batch_tokens,
                    keep_logits=True,
                )
            finally:
                unload_model(model_obj)
        payload = {
            "schema_version": SCHEMA,
            "status": "complete",
            "task": TASK,
            "model": model,
            "domain": samples[0].domain,
            "sample_ids": expected_sample_ids,
            "sample_ids_sha256": expected_hash,
            "logit_storage_dtype": "bf16_selected_span_logits_only",
            "log_softmax_metric_dtype": "fp32",
            "kl_direction": "base_to_checkpoint_D_KL_p0_parallel_pt",
            "records": rows,
            "created_utc": utc_now(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        atomic_json(meta, {
            key: payload[key] for key in (
                "schema_version", "status", "task", "model", "domain", "sample_ids_sha256",
                "logit_storage_dtype", "log_softmax_metric_dtype", "kl_direction", "created_utc",
            )
        } | {"artifact": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        return payload


def span_for_position(regions: dict[str, list[int]], pos: int) -> str:
    for span in ("b", "f_pre", "a", "f_post", "t", "c", "p"):
        if pos in set(regions.get(span, [])):
            return span
    return "unknown"


def base_maps(cache: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    result = {}
    for record in cache["records"]:
        mapping = {}
        for token_row in record["target_token_rows"]:
            mapping[int(token_row["target_pos"])] = token_row
        result[str(record["sample_id"])] = mapping
    return result


def sample_metrics(
    domain: str,
    model: str,
    arm: str,
    step: int,
    current_rows: list[dict[str, Any]],
    base_cache: dict[str, Any],
    tokenizer_hash: str,
    sample_manifest_hash: str,
    template_hash: str,
    model_path_hash: str,
) -> list[dict[str, Any]]:
    base_by_sample = base_maps(base_cache)
    rows = []
    for current in current_rows:
        sample_id = str(current["sample_id"])
        base_tokens = base_by_sample[sample_id]
        per_span_nll: dict[str, list[float]] = defaultdict(list)
        per_span_delta: dict[str, list[float]] = defaultdict(list)
        per_span_kl: dict[str, list[float]] = defaultdict(list)
        regions = current["regions"]
        for token_row in current["target_token_rows"]:
            pos = int(token_row["target_pos"])
            span = span_for_position(regions, pos)
            base_row = base_tokens[pos]
            nllt = float(token_row["nll"])
            nll0 = float(base_row["nll"])
            per_span_nll[span].append(nllt)
            per_span_delta[span].append(nllt - nll0)
            if span in ("f_pre", "a", "f_post", "b", "t"):
                current_logits = token_row.get("logits")
                if current_logits is None:
                    raise RuntimeError("current logits missing for KL span")
                base_logits = base_row["logits"].float()
                logp0 = torch.log_softmax(base_logits, dim=-1)
                logpt = torch.log_softmax(current_logits.float(), dim=-1)
                kl = float((logp0.exp() * (logp0 - logpt)).sum().item())
                if kl < -1e-5:
                    raise RuntimeError(f"negative KL beyond tolerance {kl} {model}/{arm}/{step}/{sample_id}/{span}")
                per_span_kl[span].append(max(0.0, kl))
        for combined, parts in (("f", ("f_pre", "f_post")), ("c", ("c",))):
            if combined == "f":
                per_span_nll[combined] = per_span_nll["f_pre"] + per_span_nll["f_post"]
                per_span_delta[combined] = per_span_delta["f_pre"] + per_span_delta["f_post"]
                per_span_kl[combined] = per_span_kl["f_pre"] + per_span_kl["f_post"]
        row = {
            "model": model,
            "arm": "base" if step == 0 else arm,
            "checkpoint": step,
            "domain": domain,
            "sample_id": sample_id,
            "category": current["category"],
            "is_shared_base": step == 0,
            "sample_manifest_hash": sample_manifest_hash,
            "template_hash": template_hash,
            "tokenizer_hash": tokenizer_hash,
            "base_cache_hash": base_cache.get("sample_ids_sha256", ""),
            "model_path_hash": model_path_hash,
        }
        for span in ("p", "c", "f_pre", "a", "f_post", "f", "b", "t"):
            row[f"n_tokens_{span}"] = len(regions.get(span, []))
        for span in NLL_SPANS:
            values = per_span_nll.get(span, [])
            deltas = per_span_delta.get(span, [])
            row[f"nll_{span}"] = float(np.mean(values)) if values else math.nan
            row[f"delta_nll_{span}"] = float(np.mean(deltas)) if deltas else math.nan
            row[f"abs_delta_nll_{span}"] = abs(row[f"delta_nll_{span}"]) if not math.isnan(row[f"delta_nll_{span}"]) else math.nan
        for span in KL_SPANS:
            values = per_span_kl.get(span, [])
            row[f"kl_{span}"] = float(np.mean(values)) if values else math.nan
        if domain == "mmlu":
            row["question_id"] = sample_id
            row["strict_exact"] = math.nan
            row["flexible_exact"] = math.nan
            row["strict_extract_fail"] = math.nan
            row["response_length"] = math.nan
            row["truncated"] = math.nan
        else:
            row["unique_id"] = sample_id
            row["answer_consistent"] = bool(current["metadata"].get("answer_consistent", False))
            row["n_boxed_occurrences"] = current["metadata"].get("n_boxed_occurrences", math.nan)
            row["subject"] = current["metadata"].get("subject", "")
            row["level"] = current["metadata"].get("level", "")
            row["box_char_start"] = current["metadata"].get("box_char_start", math.nan)
            row["box_char_end"] = current["metadata"].get("box_char_end", math.nan)
            row["span_protocol"] = current["metadata"].get("span_protocol", "")
            row["format_answer_separable"] = current["metadata"].get("format_answer_separable", False)
        rows.append(row)
    return rows


def run_cell(
    model: str,
    arm: str,
    step: int,
    tok: Any,
    samples_by_domain: dict[str, list[PreparedSample]],
    device: str,
    max_batch_tokens: int,
    sample_manifest_hashes: dict[str, str],
    template_hashes: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    report_arm = "base" if step == 0 else arm
    out: dict[str, list[dict[str, Any]]] = {}
    with materialized_model(model, arm, step) as path:
        model_hash = sha256_json({"path": str(path), "config": sha256_file(path / "config.json") if (path / "config.json").is_file() else ""})
        model_obj = load_model(path, device)
        try:
            for domain, samples in samples_by_domain.items():
                cache = ensure_base_cache(model, tok, samples, device, max_batch_tokens)
                current = forward_selected(
                    model_obj,
                    tok,
                    encoded_records(tok, samples),
                    device,
                    max_batch_tokens=max_batch_tokens,
                    keep_logits=True,
                )
                out[domain] = sample_metrics(
                    domain,
                    model,
                    report_arm,
                    step,
                    current,
                    cache,
                    tokenizer_hash=token_hash(tok),
                    sample_manifest_hash=sample_manifest_hashes[domain],
                    template_hash=template_hashes[domain],
                    model_path_hash=model_hash,
                )
        finally:
            unload_model(model_obj)
    return out


def cell_status_path(model: str, arm: str, step: int) -> Path:
    return SCRATCH / "cells" / model / arm / step_label(step) / "status.json"


def cell_sample_path(model: str, arm: str, step: int, domain: str) -> Path:
    return SCRATCH / "cells" / model / arm / step_label(step) / f"{domain}_samples.csv"


def write_cell_outputs(model: str, arm: str, step: int, rows_by_domain: dict[str, list[dict[str, Any]]]) -> None:
    for domain, rows in rows_by_domain.items():
        atomic_csv(cell_sample_path(model, "base" if step == 0 else arm, step, domain), rows)
    atomic_json(cell_status_path(model, "base" if step == 0 else arm, step), {
        "schema_version": SCHEMA,
        "status": "complete",
        "model": model,
        "arm": "base" if step == 0 else arm,
        "checkpoint": step,
        "domains": {domain: len(rows) for domain, rows in rows_by_domain.items()},
        "created_utc": utc_now(),
    })


def cell_complete(model: str, arm: str, step: int) -> bool:
    report_arm = "base" if step == 0 else arm
    status = read_json(cell_status_path(model, report_arm, step), {})
    return status.get("status") == "complete" and all(cell_sample_path(model, report_arm, step, d).is_file() for d in DOMAINS)


def states_for_run(models: Iterable[str], smoke: bool) -> list[tuple[str, str, int]]:
    states = []
    for model in models:
        steps = QWEN_STEPS if model == "qwen" else LLAMA_STEPS
        if smoke:
            terminal = 624 if model == "qwen" else 320
            states.extend([(model, "base", 0), (model, "opd", terminal), (model, "seqkd", terminal)])
        else:
            states.append((model, "base", 0))
            for arm in ARMS:
                for step in steps:
                    if step:
                        states.append((model, arm, step))
    seen = set()
    unique = []
    for state in states:
        key = (state[0], "base" if state[2] == 0 else state[1], state[2])
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def run_forward(args: argparse.Namespace, smoke: bool = False) -> None:
    bundle = build_token_manifest()
    samples = bundle["samples"]
    if smoke:
        samples = select_smoke_samples(samples)
    template_hashes = bundle["manifest"]["domains"]
    sample_manifest_hashes = {
        domain: sha256_json([sample.sample_id for sample in domain_samples])
        for domain, domain_samples in samples.items()
    }
    models = MODELS if args.models == "all" else tuple(item.strip() for item in args.models.split(",") if item.strip())
    started = time.time()
    runtime_rows = []
    for model in models:
        tok = load_tokenizer(model, direct_model_path(model, "opd", 0))
        for model_name, arm, step in states_for_run([model], smoke):
            if cell_complete(model_name, arm, step):
                continue
            t0 = time.time()
            print(f"[FAT] forward {model_name}/{arm}/{step} smoke={smoke}", flush=True)
            with file_lock(cell_status_path(model_name, arm, step)):
                if cell_complete(model_name, arm, step):
                    continue
                rows = run_cell(
                    model_name,
                    arm,
                    step,
                    tok,
                    samples,
                    args.device,
                    args.max_batch_tokens,
                    sample_manifest_hashes,
                    {domain: str(meta["template_hash"]) for domain, meta in template_hashes.items()},
                )
                write_cell_outputs(model_name, arm, step, rows)
            runtime_rows.append({
                "created_utc": utc_now(),
                "phase": "smoke" if smoke else "formal",
                "model": model_name,
                "arm": arm,
                "checkpoint": step,
                "wall_seconds": round(time.time() - t0, 3),
                "device": args.device,
                "max_batch_tokens": args.max_batch_tokens,
                "cuda_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 3) if torch.cuda.is_available() else 0,
            })
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    existing = []
    ledger = OUT / "fat_r1_v2_runtime_and_gpu_ledger.csv"
    if ledger.is_file():
        existing = pd.read_csv(ledger).to_dict("records")
    atomic_csv(ledger, existing + runtime_rows)
    print(f"[FAT] {'smoke' if smoke else 'formal'} complete wall={time.time()-started:.1f}s", flush=True)


def collect_sample_rows(domain: str) -> pd.DataFrame:
    paths = sorted((SCRATCH / "cells").glob(f"*/*/step_*/*{domain}_samples.csv"))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_cells(samples: pd.DataFrame, domain: str) -> pd.DataFrame:
    if samples.empty:
        return samples
    group_cols = ["model", "arm", "checkpoint", "domain"]
    if domain == "mmlu":
        numeric_cols = [col for col in samples.columns if col.startswith(("nll_", "delta_nll_", "abs_delta_nll_", "kl_", "n_tokens_"))]
        macro = samples.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
        cat = samples.groupby(group_cols + ["category"], dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
        cat_macro = cat.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
        macro["aggregation"] = "sample_macro"
        cat_macro["aggregation"] = "category_macro_14"
        return pd.concat([macro, cat_macro], ignore_index=True)
    numeric_cols = [col for col in samples.columns if col.startswith(("nll_", "delta_nll_", "abs_delta_nll_", "kl_", "n_tokens_"))]
    result = samples.groupby(group_cols, dropna=False)[numeric_cols].mean(numeric_only=True).reset_index()
    result["aggregation"] = "sample_macro"
    return result


def load_behavior_join() -> pd.DataFrame:
    frames = []
    qwen = MINI / "block2_final_g2_behavior.csv"
    if qwen.is_file():
        df = pd.read_csv(qwen)
        df.insert(0, "model", "qwen")
        frames.append(df)
    llama = MINI / "llama_early_320_behavior.csv"
    if llama.is_file():
        df = pd.read_csv(llama)
        df.insert(0, "model", "llama")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(mmlu: pd.DataFrame, math_df: pd.DataFrame, draws: int = 1024) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for domain, frame in (("mmlu", mmlu), ("math", math_df)):
        if frame.empty:
            continue
        metrics = [col for col in frame.columns if col.startswith(("delta_nll_", "abs_delta_nll_", "kl_"))]
        for (model, arm, step), group in frame.groupby(["model", "arm", "checkpoint"], dropna=False):
            n = len(group)
            if not n:
                continue
            values = group[metrics].to_numpy(dtype=float)
            for mi, metric in enumerate(metrics):
                col = values[:, mi]
                if np.all(np.isnan(col)):
                    continue
                means = []
                for _ in range(draws):
                    idx = rng.integers(0, n, size=n)
                    means.append(float(np.nanmean(col[idx])))
                rows.append({
                    "domain": domain,
                    "model": model,
                    "arm": arm,
                    "checkpoint": step,
                    "metric": metric,
                    "n": n,
                    "bootstrap_seed": 42,
                    "draws": draws,
                    "mean": float(np.nanmean(col)),
                    "ci_low": float(np.percentile(means, 2.5)),
                    "ci_high": float(np.percentile(means, 97.5)),
                })
    return pd.DataFrame(rows)


def region_contrasts(mmlu_cells: pd.DataFrame, math_cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not mmlu_cells.empty:
        for row in mmlu_cells.to_dict("records"):
            rows.append({
                "model": row.get("model"),
                "arm": row.get("arm"),
                "checkpoint": row.get("checkpoint"),
                "domain": "mmlu",
                "aggregation": row.get("aggregation"),
                "delta_nll_f_minus_a": row.get("delta_nll_f", math.nan) - row.get("delta_nll_a", math.nan),
                "delta_nll_f_minus_p": row.get("delta_nll_f", math.nan) - row.get("delta_nll_p", math.nan),
                "kl_f_minus_a": row.get("kl_f", math.nan) - row.get("kl_a", math.nan),
                "delta_nll_b_minus_c": math.nan,
                "delta_nll_b_minus_p": math.nan,
                "kl_b_minus_c": math.nan,
                "notes": "mechanical MMLU contrast; kl_p is NA in round1",
            })
    if not math_cells.empty:
        for row in math_cells.to_dict("records"):
            rows.append({
                "model": row.get("model"),
                "arm": row.get("arm"),
                "checkpoint": row.get("checkpoint"),
                "domain": "math",
                "aggregation": row.get("aggregation"),
                "delta_nll_f_minus_a": math.nan,
                "delta_nll_f_minus_p": math.nan,
                "kl_f_minus_a": math.nan,
                "delta_nll_b_minus_c": row.get("delta_nll_b", math.nan) - row.get("delta_nll_c", math.nan),
                "delta_nll_b_minus_p": row.get("delta_nll_b", math.nan) - row.get("delta_nll_p", math.nan),
                "kl_b_minus_c": math.nan,
                "notes": "mechanical MATH token-closed B contrast; KL_C is NA in round1",
            })
    return pd.DataFrame(rows)


def finalize() -> None:
    mmlu = collect_sample_rows("mmlu")
    math_df = collect_sample_rows("math")
    atomic_csv(OUT / "fat_r1_v2_mmlu_samples.csv", mmlu.to_dict("records") if not mmlu.empty else [])
    atomic_csv(OUT / "fat_r1_v2_math_samples.csv", math_df.to_dict("records") if not math_df.empty else [])
    mmlu_cells = aggregate_cells(mmlu, "mmlu")
    math_cells = aggregate_cells(math_df, "math")
    atomic_csv(OUT / "fat_r1_v2_mmlu_cells.csv", mmlu_cells.to_dict("records") if not mmlu_cells.empty else [])
    atomic_csv(OUT / "fat_r1_v2_math_cells.csv", math_cells.to_dict("records") if not math_cells.empty else [])
    contrasts = region_contrasts(mmlu_cells, math_cells)
    atomic_csv(OUT / "fat_r1_v2_region_contrasts.csv", contrasts.to_dict("records") if not contrasts.empty else [])
    behavior = load_behavior_join()
    atomic_csv(OUT / "fat_r1_v2_behavior_join.csv", behavior.to_dict("records") if not behavior.empty else [])
    boot = bootstrap_ci(mmlu, math_df)
    atomic_csv(OUT / "fat_r1_v2_bootstrap_ci.csv", boot.to_dict("records") if not boot.empty else [])
    inventory = checkpoint_inventory()
    atomic_csv(OUT / "fat_r1_v2_checkpoint_inventory.csv", inventory)
    base_cache_rows = []
    for path in sorted(BASE_CACHE.glob("*/*.json")):
        payload = read_json(path, {})
        base_cache_rows.append({"path": str(path), **payload})
    atomic_json(OUT / "fat_r1_v2_base_cache_manifest.json", {"schema_version": SCHEMA, "rows": base_cache_rows})
    statuses = []
    for model, arm, step in states_for_run(MODELS, smoke=False):
        status = read_json(cell_status_path(model, arm, step), {})
        statuses.append({
            "model": model,
            "arm": arm,
            "checkpoint": step,
            "status": status.get("status", "missing"),
            "mmlu_rows": status.get("domains", {}).get("mmlu", 0),
            "math_rows": status.get("domains", {}).get("math", 0),
        })
    atomic_csv(OUT / "fat_r1_v2_task_status.csv", statuses)
    outputs = {
        name: {
            "path": str(OUT / name),
            "exists": (OUT / name).is_file(),
            "sha256": sha256_file(OUT / name) if (OUT / name).is_file() else None,
            "rows": (sum(1 for _ in (OUT / name).open(encoding="utf-8")) - 1) if (OUT / name).is_file() and name.endswith(".csv") else None,
        }
        for name in [
            "fat_r1_v2_task_status.csv",
            "fat_r1_v2_sample_manifest.json",
            "fat_r1_v2_checkpoint_inventory.csv",
            "fat_r1_v2_template_and_tokenizer_manifest.json",
            "fat_r1_v2_mask_audit.csv",
            "fat_r1_v2_mmlu_samples.csv",
            "fat_r1_v2_mmlu_cells.csv",
            "fat_r1_v2_math_samples.csv",
            "fat_r1_v2_math_cells.csv",
            "fat_r1_v2_region_contrasts.csv",
            "fat_r1_v2_behavior_join.csv",
            "fat_r1_v2_bootstrap_ci.csv",
            "fat_r1_v2_base_cache_manifest.json",
            "fat_r1_v2_runtime_and_gpu_ledger.csv",
        ]
    }
    payload = {
        "schema_version": SCHEMA,
        "task": TASK,
        "status": "complete" if all(row["status"] == "complete" for row in statuses) else "partial",
        "created_utc": utc_now(),
        "authorized_scope": ["FAT-R1-S0", "FAT-R1-M1", "FAT-R1-M2"],
        "explicitly_removed": ["MATH_DIRECT_NO_COT"],
        "numeric_protocol": {
            "forward_dtype": "bf16",
            "log_softmax_nll_kl_dtype": "fp32",
            "kl_direction": "base_to_checkpoint_D_KL_p0_parallel_pt",
            "full_vocab_kl": True,
            "stored_logits": "selected F/A/T span logits only; no full sequence-vocab tensors",
            "checkpoint_load": "one checkpoint load processes MMLU and MATH consecutively",
        },
        "input_sha256": {
            "mmlupro_1400": sha256_file(MMLU_PATH),
            "math500": sha256_file(MATH_PATH),
            "equal5_manifest_registered_only": sha256_file(EQUAL5_MANIFEST) if EQUAL5_MANIFEST.is_file() else None,
        },
        "outputs": outputs,
    }
    atomic_json(OUT / "fat_r1_v2_manifest.json", payload)
    handoff = [
        "# FAT-R1 handoff",
        "",
        f"- created_utc: `{payload['created_utc']}`",
        f"- status: `{payload['status']}`",
        "- scope: `FAT-R1-S0`, `FAT-R1-M1`, `FAT-R1-M2` only",
        "- guard: zero training; zero rollout; teacher-forcing forward only",
        "- numeric: BF16 forward; FP32 log_softmax/NLL/KL; exact full-vocabulary KL direction `D_KL(p0 || pt)`",
        "- explicitly removed: `MATH_DIRECT_NO_COT`",
        "",
        "## Row counts",
        "",
        "| file | rows | sha256 |",
        "|---|---:|---|",
    ]
    for name, info in outputs.items():
        handoff.append(f"| `{name}` | {info['rows'] if info['rows'] is not None else 'NA'} | `{info['sha256']}` |")
    handoff.append("")
    handoff.append("## Completion status")
    handoff.append("")
    handoff.append("| model | arm | checkpoint | status | mmlu_rows | math_rows |")
    handoff.append("|---|---|---:|---|---:|---:|")
    for row in statuses:
        handoff.append(f"| {row['model']} | {row['arm']} | {row['checkpoint']} | {row['status']} | {row['mmlu_rows']} | {row['math_rows']} |")
    atomic_text(OUT / "fat_r1_v2_handoff.md", "\n".join(handoff) + "\n")
    payload["outputs"]["fat_r1_v2_handoff.md"] = {
        "path": str(OUT / "fat_r1_v2_handoff.md"),
        "sha256": sha256_file(OUT / "fat_r1_v2_handoff.md"),
    }
    atomic_json(OUT / "fat_r1_v2_manifest.json", payload)
    append_code_evolution(
        f"""
## {utc_now()} FAT-R1 output-link correction return

- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_outlink_round1.py`
- Output root: `{OUT}`
- Manifest: `{OUT / 'fat_r1_v2_manifest.json'}`
- Scope: confirmed first round only (`FAT-R1-S0`, `FAT-R1-M1`, `FAT-R1-M2`); no rollout/training/free generation.
- Numeric protocol: BF16 checkpoint forward, FP32 log-softmax/NLL/KL, exact full-vocabulary KL `D_KL(p0 || pt)`.
"""
    )


def write_blocked_preflight(error: str) -> None:
    inventory = checkpoint_inventory()
    atomic_csv(OUT / "fat_r1_v2_checkpoint_inventory.csv", inventory)
    mask_path = OUT / "fat_r1_v2_mask_audit.csv"
    sample_index = OUT / "fat_r1_v2_sample_manifest_index.csv"
    mask_rows = pd.read_csv(mask_path) if mask_path.is_file() else pd.DataFrame()
    sample_rows = pd.read_csv(sample_index) if sample_index.is_file() else pd.DataFrame()
    summary_rows = []
    if not mask_rows.empty:
        grouped = (
            mask_rows.groupby(["model", "domain", "region"], dropna=False)
            .size()
            .reset_index(name="boundary_crossing_rows")
        )
        summary_rows = grouped.to_dict("records")
    status_rows = [
        {
            "task": "FAT-R1-S0",
            "status": "BLOCKED_TOKEN_BOUNDARY_PROTOCOL",
            "blocked_reason": error,
            "boundary_crossing_rows": int(len(mask_rows)),
            "checkpoint_blocked_count": int(sum(not str(row["status"]).startswith("READY") for row in inventory)),
            "created_utc": utc_now(),
        },
        {
            "task": "FAT-R1-M1",
            "status": "NOT_STARTED_DUE_S0_GATE",
            "blocked_reason": "S0 did not pass; formal forward not launched",
            "boundary_crossing_rows": int(len(mask_rows)),
            "checkpoint_blocked_count": 0,
            "created_utc": utc_now(),
        },
        {
            "task": "FAT-R1-M2",
            "status": "NOT_STARTED_DUE_S0_GATE",
            "blocked_reason": "MATH token span boundaries cross formatter/answer regions",
            "boundary_crossing_rows": int(len(mask_rows)),
            "checkpoint_blocked_count": 0,
            "created_utc": utc_now(),
        },
    ]
    atomic_csv(OUT / "fat_r1_v2_task_status.csv", status_rows)
    manifest = {
        "schema_version": SCHEMA,
        "task": TASK,
        "status": "blocked_at_S0",
        "created_utc": utc_now(),
        "blocked_reason": error,
        "boundary_crossing_total_rows": int(len(mask_rows)),
        "boundary_crossing_summary": summary_rows,
        "checkpoint_inventory_sha256": sha256_file(OUT / "fat_r1_v2_checkpoint_inventory.csv"),
        "sample_manifest_sha256": sha256_file(OUT / "fat_r1_v2_sample_manifest.json") if (OUT / "fat_r1_v2_sample_manifest.json").is_file() else None,
        "mask_audit_sha256": sha256_file(mask_path) if mask_path.is_file() else None,
        "sample_index_sha256": sha256_file(sample_index) if sample_index.is_file() else None,
        "numeric_protocol": {
            "forward": "not launched",
            "reason": "S0 token boundary gate failed before smoke/formal M1/M2",
        },
        "guard": {
            "zero_training": True,
            "zero_rollout": True,
            "zero_free_generation": True,
            "math_direct_no_cot": "explicitly_removed_not_constructed",
        },
        "row_counts": {
            "fat_r1_v2_task_status.csv": len(status_rows),
            "fat_r1_v2_checkpoint_inventory.csv": len(inventory),
            "fat_r1_v2_mask_audit.csv": int(len(mask_rows)),
            "fat_r1_v2_sample_manifest_index.csv": int(len(sample_rows)),
        },
    }
    atomic_json(OUT / "fat_r1_v2_manifest.json", manifest)
    handoff = [
        "# FAT-R1 S0 blocked handoff",
        "",
        f"- created_utc: `{manifest['created_utc']}`",
        "- status: `BLOCKED_TOKEN_BOUNDARY_PROTOCOL`",
        "- formal M1/M2 forward: `not started`",
        "- reason: tokenizer offset audit found MATH span boundary crossings around `\\boxed{...}`; protocol forbids silently reassigning or changing template.",
        "- MMLU: no boundary-crossing rows observed in this S0 audit.",
        "- MATH direct/no-CoT: not constructed.",
        "",
        "## Boundary summary",
        "",
        "| model | domain | region | rows |",
        "|---|---|---|---:|",
    ]
    for row in summary_rows:
        handoff.append(
            f"| {row['model']} | {row['domain']} | {row['region']} | {row['boundary_crossing_rows']} |"
        )
    handoff.extend([
        "",
        "## Files",
        "",
        "| file | rows | sha256 |",
        "|---|---:|---|",
    ])
    for name, count in manifest["row_counts"].items():
        path = OUT / name
        handoff.append(f"| `{name}` | {count} | `{sha256_file(path) if path.is_file() else None}` |")
    atomic_text(OUT / "fat_r1_v2_handoff.md", "\n".join(handoff) + "\n")
    append_code_evolution(
        f"""
## {utc_now()} FAT-R1 S0 boundary gate

- Script: `experiments/opd_sft_h1/scripts/cycle09_fat_outlink_round1.py`
- Output root: `{OUT}`
- Status: `BLOCKED_TOKEN_BOUNDARY_PROTOCOL`.
- Finding: MATH `\\boxed{{...}}` span offset audit produced {len(mask_rows)} boundary-crossing rows; formal M1/M2 forward was not launched.
"""
    )


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    inventory = checkpoint_inventory()
    atomic_csv(OUT / "fat_r1_v2_checkpoint_inventory.csv", inventory)
    blocked = [row for row in inventory if not str(row["status"]).startswith("READY")]
    try:
        build_token_manifest()
    except RuntimeError as error:
        if "FAILED_TOKEN_BOUNDARY_PROTOCOL" in str(error) or "FAILED_EMPTY_SPAN_PROTOCOL" in str(error):
            write_blocked_preflight(str(error))
        raise
    status_rows = [{
        "task": "FAT-R1-S0",
        "status": "blocked" if blocked else "complete",
        "blocked_count": len(blocked),
        "created_utc": utc_now(),
    }]
    atomic_csv(OUT / "fat_r1_v2_task_status.csv", status_rows)
    if blocked:
        raise RuntimeError(f"checkpoint inventory blocked: {blocked[:5]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "smoke", "formal", "finalize", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--models", default="all", help="all,qwen,llama or comma list")
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    if args.mode in ("preflight", "all"):
        preflight()
    if args.mode in ("smoke", "all"):
        run_forward(args, smoke=True)
    if args.mode in ("formal", "all"):
        run_forward(args, smoke=False)
    if args.mode in ("finalize", "all"):
        finalize()


if __name__ == "__main__":
    main()
