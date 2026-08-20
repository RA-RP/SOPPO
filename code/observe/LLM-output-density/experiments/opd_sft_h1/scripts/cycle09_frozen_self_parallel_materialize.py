#!/usr/bin/env python3
"""Two-GPU, data-parallel materializer for the H5 frozenSelf0-KD control.

The H5 support must use one step-0 Llama student sample followed by RAW teacher
top-32 log-probabilities over exactly that sampled token sequence.  A single
vLLM engine made this entirely GPU0-bound.  This script splits the immutable
prompt order into two contiguous shards, runs one independent one-GPU engine
per shard, then restores the original order before producing the normal frozen
store.  It deliberately does *not* use tensor parallelism: for autoregressive
rollout, data parallelism gives the useful wall-clock reduction and preserves
an unambiguous prompt-to-output provenance record.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import cycle09_block3_common as b3
import cycle09_stage3_followup_common as c


FORMAL_ROOT = c.scoped_run("H5_frozen_self")
PROMPTS = b3.L1_DATA / "llama_opd_prompts_4999.parquet"
STUDENT = b3.LLAMA_STUDENT
TEACHER = b3.LLAMA_TEACHER
PYTHON = c.DENSITY_PYTHON

TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = -1
SEED = 42
MAX_TOKENS = 10240
MAX_MODEL_LEN = 11265
TOPK = 32
N_SHARDS = 2


def artifact_root(smoke: bool) -> Path:
    root = FORMAL_ROOT / "parallel_smoke" if smoke else FORMAL_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    c.atomic_json(path, value)


def load_records(limit: int) -> list[dict[str, Any]]:
    frame = pd.read_parquet(PROMPTS)
    if limit:
        frame = frame.iloc[:limit]
    records: list[dict[str, Any]] = []
    for order, (source_index, row) in enumerate(frame.iterrows()):
        messages = row["prompt"]
        messages = messages.tolist() if hasattr(messages, "tolist") else list(messages)
        records.append(
            {
                "order": order,
                "index": int(source_index),
                "messages": [dict(message) for message in messages],
                "data_source": str(row["data_source"]),
                "ground_truth": str(row["reward_model"]["ground_truth"]),
            }
        )
    if not records:
        raise RuntimeError(f"no prompts found in {PROMPTS}")
    return records


def shard_bounds(total: int, shard: int) -> tuple[int, int]:
    if not 0 <= shard < N_SHARDS:
        raise ValueError(f"invalid shard={shard}")
    start = total * shard // N_SHARDS
    end = total * (shard + 1) // N_SHARDS
    return start, end


def shard_root(root: Path, shard: int) -> Path:
    path = root / "rollout_shards" / f"shard_{shard:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    c.atomic_jsonl(path, records)


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024**2)
    raise RuntimeError("MemAvailable is unavailable")


def completed_pass1(path: Path, expected_orders: list[int]) -> bool:
    manifest = c.read_json(path / "pass1_manifest.json", {})
    raw = path / "teacher_rollout_pass1.jsonl"
    if manifest.get("status") != "complete" or not raw.is_file():
        return False
    observed = [int(row["order"]) for row in read_jsonl(raw)]
    return observed == expected_orders


def pass1(root: Path, shard: int, limit: int) -> dict[str, Any]:
    all_rows = load_records(limit)
    start, end = shard_bounds(len(all_rows), shard)
    rows = all_rows[start:end]
    work = shard_root(root, shard)
    raw = work / "teacher_rollout_pass1.jsonl"
    expected_orders = list(range(start, end))
    if completed_pass1(work, expected_orders):
        return c.read_json(work / "pass1_manifest.json", {})

    from vllm import LLM, SamplingParams

    tokenizer = b3.load_llama_tokenizer(STUDENT)
    rendered = [
        tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    llm = LLM(
        model=str(STUDENT),
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=MAX_MODEL_LEN,
        seed=SEED,
        enforce_eager=False,
        max_logprobs=TOPK,
    )
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        n=1,
        max_tokens=MAX_TOKENS,
        seed=SEED,
    )
    started = time.time()
    outputs = llm.generate(rendered, sampling)
    seconds = time.time() - started
    persisted: list[dict[str, Any]] = []
    for row, prompt, output in zip(rows, rendered, outputs):
        completion = output.outputs[0]
        persisted.append(
            {
                "order": row["order"],
                "index": row["index"],
                "data_source": row["data_source"],
                "ground_truth": row["ground_truth"],
                "prompt": prompt,
                "generation": completion.text,
                "prompt_token_ids": list(output.prompt_token_ids),
                "generation_token_ids": list(completion.token_ids),
                "finish_reason": completion.finish_reason,
                "n_prompt_tokens": len(output.prompt_token_ids),
                "n_tokens": len(completion.token_ids),
                "has_boxed": "\\boxed{" in completion.text,
            }
        )
    atomic_jsonl(raw, persisted)
    del outputs, llm
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "pass": "pass1_step0_student_sampling",
        "shard": shard,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "order_start": start,
        "order_end_exclusive": end,
        "n_records": len(persisted),
        "sampling": {
            "model": str(STUDENT),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
        },
        "raw": c.artifact(raw),
        "seconds": seconds,
        "created_utc": c.utc_now(),
    }
    atomic_json(work / "pass1_manifest.json", manifest)
    return manifest


def completed_pass2(work: Path, records: list[dict[str, Any]]) -> bool:
    stream = work / "pass2_stream"
    progress = c.read_json(stream / "progress.json", {})
    expected_tokens = sum(int(record["n_tokens"]) for record in records)
    try:
        ids = np.load(stream / "top32_ids.npy", mmap_mode="r")
        logprobs = np.load(stream / "top32_logprob.npy", mmap_mode="r")
        offsets = np.load(stream / "row_offsets.npy", mmap_mode="r")
    except (FileNotFoundError, ValueError):
        return False
    return (
        progress.get("status") == "complete"
        and ids.shape == (expected_tokens, TOPK)
        and logprobs.shape == ids.shape
        and offsets.shape == (len(records), 2)
        and (not len(records) or int(offsets[-1, 1]) == expected_tokens)
    )


def pass2(root: Path, shard: int, limit: int) -> dict[str, Any]:
    all_rows = load_records(limit)
    start, end = shard_bounds(len(all_rows), shard)
    work = shard_root(root, shard)
    raw = work / "teacher_rollout_pass1.jsonl"
    if not raw.is_file():
        raise FileNotFoundError(f"pass2 requires completed pass1: {raw}")
    records = read_jsonl(raw)
    if [int(record["order"]) for record in records] != list(range(start, end)):
        raise RuntimeError(f"shard={shard} pass1 order mismatch")
    if completed_pass2(work, records):
        return c.read_json(work / "pass2_manifest.json", {})

    stream = work / "pass2_stream"
    stream.mkdir(parents=True, exist_ok=True)
    ids_path = stream / "top32_ids.npy"
    logprobs_path = stream / "top32_logprob.npy"
    offsets_path = stream / "row_offsets.npy"
    progress_path = stream / "progress.json"
    total_tokens = sum(int(record["n_tokens"]) for record in records)
    source_sha256 = c.sha256_file(raw)

    progress = c.read_json(progress_path, {})
    expected = {
        "source_sha256": source_sha256,
        "n_records": len(records),
        "total_tokens": total_tokens,
        "topk": TOPK,
    }
    if progress:
        mismatched = {key: (progress.get(key), value) for key, value in expected.items() if progress.get(key) != value}
        if mismatched:
            raise RuntimeError(f"shard={shard} pass2 resume mismatch: {mismatched}")
        top_ids = np.lib.format.open_memmap(ids_path, mode="r+")
        top_logprobs = np.lib.format.open_memmap(logprobs_path, mode="r+")
        offsets = np.lib.format.open_memmap(offsets_path, mode="r+")
        next_record = int(progress["next_record"])
        cursor = int(progress["cursor"])
        elapsed_seconds = float(progress.get("elapsed_seconds", 0.0))
    else:
        top_ids = np.lib.format.open_memmap(ids_path, mode="w+", dtype=np.int32, shape=(total_tokens, TOPK))
        top_logprobs = np.lib.format.open_memmap(logprobs_path, mode="w+", dtype=np.float32, shape=(total_tokens, TOPK))
        offsets = np.lib.format.open_memmap(offsets_path, mode="w+", dtype=np.int64, shape=(len(records), 2))
        next_record = 0
        cursor = 0
        elapsed_seconds = 0.0
        progress = expected | {
            "schema_version": 1,
            "status": "running",
            "next_record": next_record,
            "cursor": cursor,
            "elapsed_seconds": elapsed_seconds,
            "record_batch": 4,
        }
        atomic_json(progress_path, progress)

    expected_cursor = sum(int(record["n_tokens"]) for record in records[:next_record])
    if cursor != expected_cursor:
        raise RuntimeError(f"shard={shard} pass2 cursor={cursor}, expected={expected_cursor}")

    from vllm import LLM, SamplingParams

    llm = None
    if next_record < len(records):
        llm = LLM(
            model=str(TEACHER),
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            max_model_len=MAX_MODEL_LEN,
            seed=SEED,
            enforce_eager=True,
            max_logprobs=TOPK,
            max_num_batched_tokens=2048,
            max_num_seqs=1,
        )
        forward = SamplingParams(temperature=1.0, max_tokens=1, prompt_logprobs=TOPK)
        while next_record < len(records):
            batch_end = min(next_record + 4, len(records))
            batch = records[next_record:batch_end]
            sequences = [{"prompt_token_ids": record["prompt_token_ids"] + record["generation_token_ids"]} for record in batch]
            started = time.time()
            outputs = llm.generate(sequences, forward, use_tqdm=False)
            elapsed_seconds += time.time() - started
            for output_row, (record, output) in enumerate(zip(batch, outputs), start=next_record):
                token_start = int(record["n_prompt_tokens"])
                token_end = token_start + int(record["n_tokens"])
                prompt_logprobs = output.prompt_logprobs
                if token_end > len(prompt_logprobs):
                    raise RuntimeError(f"shard={shard} row={output_row} incomplete teacher sequence")
                next_cursor = cursor + int(record["n_tokens"])
                ids_view = top_ids[cursor:next_cursor]
                lp_view = top_logprobs[cursor:next_cursor]
                ids_view.fill(-1)
                lp_view.fill(np.nan)
                for position in range(token_start, token_end):
                    entry = prompt_logprobs[position]
                    if not entry:
                        continue
                    ranked = sorted(entry.items(), key=lambda item: item[1].logprob, reverse=True)
                    for slot, (token_id, info) in enumerate(ranked[:TOPK]):
                        ids_view[position - token_start, slot] = int(token_id)
                        lp_view[position - token_start, slot] = float(info.logprob)
                if np.any(ids_view < 0) or not np.isfinite(lp_view).all():
                    raise RuntimeError(f"shard={shard} row={output_row} incomplete RAW top-{TOPK}")
                offsets[output_row] = (cursor, next_cursor)
                cursor = next_cursor
            top_ids.flush()
            top_logprobs.flush()
            offsets.flush()
            next_record = batch_end
            progress.update({
                "next_record": next_record,
                "cursor": cursor,
                "elapsed_seconds": elapsed_seconds,
                "available_ram_gib_after_gc": available_ram_gib(),
                "updated_at_unix": time.time(),
            })
            atomic_json(progress_path, progress)
            print(f"[frozen-self shard {shard}] pass2 {next_record}/{len(records)} records, {cursor}/{total_tokens} tokens", flush=True)
            del outputs, sequences, batch
            gc.collect()
            if available_ram_gib() < 128.0:
                raise MemoryError(f"shard={shard} pass2 host-RAM guard tripped")

    if cursor != total_tokens or next_record != len(records):
        raise RuntimeError(f"shard={shard} pass2 incomplete")
    progress.update({"status": "complete", "completed_at_unix": time.time()})
    atomic_json(progress_path, progress)
    if llm is not None:
        del llm
    del top_ids, top_logprobs, offsets
    gc.collect()
    torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "pass": "pass2_teacher_raw_top32",
        "shard": shard,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "order_start": start,
        "order_end_exclusive": end,
        "n_records": len(records),
        "total_tokens": total_tokens,
        "teacher": str(TEACHER),
        "stream": [c.artifact(stream / name) for name in ("top32_ids.npy", "top32_logprob.npy", "row_offsets.npy", "progress.json")],
        "elapsed_seconds": elapsed_seconds,
        "created_utc": c.utc_now(),
    }
    atomic_json(work / "pass2_manifest.json", manifest)
    return manifest


def worker(args: argparse.Namespace) -> None:
    root = artifact_root(args.smoke)
    if args.worker_pass == "pass1":
        value = pass1(root, args.shard, args.limit)
    else:
        value = pass2(root, args.shard, args.limit)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def launch_workers(root: Path, worker_pass: str, limit: int, smoke: bool) -> None:
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    for shard in range(N_SHARDS):
        log_path = logs / f"parallel_{worker_pass}_gpu{shard}.log"
        log = log_path.open("ab", buffering=0)
        command = [str(PYTHON), str(Path(__file__).resolve()), "--worker", "--worker-pass", worker_pass, "--shard", str(shard), "--limit", str(limit)]
        if smoke:
            command.append("--smoke")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard)
        process = subprocess.Popen(command, cwd=c.REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((shard, process, log))
    failures: list[str] = []
    for shard, process, log in processes:
        result = process.wait()
        log.close()
        if result:
            failures.append(f"shard={shard} pass={worker_pass} rc={result}; log={logs / f'parallel_{worker_pass}_gpu{shard}.log'}")
    if failures:
        raise RuntimeError("; ".join(failures))


def merge(root: Path, limit: int) -> dict[str, Any]:
    rollout = root / "rollout"
    stream = rollout / "pass2_stream"
    rollout.mkdir(parents=True, exist_ok=True)
    stream.mkdir(parents=True, exist_ok=True)
    all_rows = load_records(limit)
    expected_orders = list(range(len(all_rows)))
    sources: dict[int, dict[str, Any]] = {}
    ordered: dict[int, tuple[int, dict[str, Any], int]] = {}
    for shard in range(N_SHARDS):
        work = shard_root(root, shard)
        records = read_jsonl(work / "teacher_rollout_pass1.jsonl")
        if not completed_pass2(work, records):
            raise RuntimeError(f"cannot merge incomplete shard={shard}")
        ids = np.load(work / "pass2_stream/top32_ids.npy", mmap_mode="r")
        logprobs = np.load(work / "pass2_stream/top32_logprob.npy", mmap_mode="r")
        offsets = np.load(work / "pass2_stream/row_offsets.npy", mmap_mode="r")
        sources[shard] = {"records": records, "ids": ids, "logprobs": logprobs, "offsets": offsets}
        for local_row, record in enumerate(records):
            order = int(record["order"])
            if order in ordered:
                raise RuntimeError(f"duplicate original order={order}")
            ordered[order] = (shard, record, local_row)
    if sorted(ordered) != expected_orders:
        raise RuntimeError("shard merge does not cover the immutable prompt order")

    raw_records = [ordered[order][1] for order in expected_orders]
    raw_path = rollout / "teacher_rollout_pass1.jsonl"
    atomic_jsonl(raw_path, raw_records)
    total_tokens = sum(int(record["n_tokens"]) for record in raw_records)
    ids_temp = stream / "top32_ids.partial.npy"
    lps_temp = stream / "top32_logprob.partial.npy"
    offsets_temp = stream / "row_offsets.partial.npy"
    for path in (ids_temp, lps_temp, offsets_temp):
        if path.exists():
            path.unlink()
    ids_out = np.lib.format.open_memmap(ids_temp, mode="w+", dtype=np.int32, shape=(total_tokens, TOPK))
    lps_out = np.lib.format.open_memmap(lps_temp, mode="w+", dtype=np.float32, shape=(total_tokens, TOPK))
    offsets_out = np.lib.format.open_memmap(offsets_temp, mode="w+", dtype=np.int64, shape=(len(raw_records), 2))
    cursor = 0
    output_records: list[dict[str, Any]] = []
    for output_row, order in enumerate(expected_orders):
        shard, record, local_row = ordered[order]
        source = sources[shard]
        source_start, source_end = (int(value) for value in source["offsets"][local_row])
        count = source_end - source_start
        if count != int(record["n_tokens"]):
            raise RuntimeError(f"shard={shard} order={order} token-count mismatch")
        end = cursor + count
        ids_out[cursor:end] = source["ids"][source_start:source_end]
        lps_out[cursor:end] = source["logprobs"][source_start:source_end]
        offsets_out[output_row] = (cursor, end)
        output_record = dict(record)
        output_record["logprob_row_start"] = cursor
        output_record["logprob_row_end"] = end
        output_records.append(output_record)
        cursor = end
    if cursor != total_tokens:
        raise RuntimeError(f"merge cursor={cursor}, expected={total_tokens}")
    ids_out.flush()
    lps_out.flush()
    offsets_out.flush()
    del ids_out, lps_out, offsets_out
    os.replace(ids_temp, stream / "top32_ids.npy")
    os.replace(lps_temp, stream / "top32_logprob.npy")
    os.replace(offsets_temp, stream / "row_offsets.npy")
    atomic_jsonl(rollout / "teacher_rollout.jsonl", output_records)

    lengths = np.asarray([record["n_tokens"] for record in raw_records], dtype=np.int64)
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "stage": "frozenSelf0-KD two-GPU data-parallel rollout",
        "n_prompts": len(raw_records),
        "prompt_order": "original parquet file order, restored after contiguous two-way sharding",
        "sampling_pass1": {
            "model": str(STUDENT),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
        },
        "logprob_pass2": {
            "model": str(TEACHER),
            "convention": "RAW temperature=1.0 prompt_logprobs over exact pass1 token IDs",
            "topk": TOPK,
        },
        "data_parallel": {
            "strategy": "two independent one-GPU vLLM engines; contiguous fixed prompt shards",
            "n_shards": N_SHARDS,
            "merge": "strict original order",
            "shards": [
                {
                    "shard": shard,
                    "cuda_visible_devices": str(shard),
                    "pass1": c.artifact(shard_root(root, shard) / "pass1_manifest.json"),
                    "pass2": c.artifact(shard_root(root, shard) / "pass2_manifest.json"),
                }
                for shard in range(N_SHARDS)
            ],
        },
        "outputs": [c.artifact(path) for path in (raw_path, rollout / "teacher_rollout.jsonl", stream / "top32_ids.npy", stream / "top32_logprob.npy", stream / "row_offsets.npy")],
        "truncation_rate": sum(record["finish_reason"] == "length" for record in raw_records) / len(raw_records),
        "length_stats": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        },
        "created_utc": c.utc_now(),
    }
    atomic_json(stream / "progress.json", {"status": "complete", "n_records": len(raw_records), "total_tokens": total_tokens, "topk": TOPK})
    atomic_json(rollout / "rollout_manifest.json", manifest)
    return manifest


def materialize(args: argparse.Namespace) -> None:
    root = artifact_root(args.smoke)
    if args.phase in ("all", "pass1"):
        launch_workers(root, "pass1", args.limit, args.smoke)
    if args.phase in ("all", "pass2"):
        launch_workers(root, "pass2", args.limit, args.smoke)
    if args.phase in ("all", "merge"):
        merged = merge(root, args.limit)
    else:
        merged = {"status": "workers_complete"}
    if not args.smoke and args.phase in ("all", "merge"):
        from cycle09_frozen_self_materialize import convert

        support = convert(root / "rollout")
        merged["frozen_support"] = support
    print(json.dumps(merged, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "pass1", "pass2", "merge"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-pass", choices=("pass1", "pass2"))
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    if args.smoke:
        args.limit = 8
    if args.limit <= 0:
        args.limit = len(load_records(0))
    if args.worker:
        if args.worker_pass is None or args.shard is None:
            parser.error("--worker requires --worker-pass and --shard")
        worker(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
