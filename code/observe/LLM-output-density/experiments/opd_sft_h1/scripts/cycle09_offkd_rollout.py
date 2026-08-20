#!/usr/bin/env python3
"""off-KD control arm — Stage 1: teacher rollout (Qwen3-8B).

off-KD = the off-policy version of OPD: same teacher, same loss, same LoRA, same grid.
The ONLY variable is where the training input comes from (OPD = the student's own
rollouts; off-KD = the teacher's pre-generated static responses). Every sampling
parameter is therefore pinned to Cycle 08's real training config
(outputs/2026-07-02/10-40-01/.hydra/config.yaml), not the smoke run.

TWO PASSES — this is the load-bearing detail (blocking check, QA_cycle09.md):
  pass 1  sampling:  temperature=0.6, top_p=0.9, top_k=-1, n=1, seed=42, max_tokens=10240
  pass 2  forward:   prompt_logprobs=32 at temperature=1.0 over the EXACT token sequence
                     from pass 1 (prompt_token_ids + generation_token_ids concatenated,
                     never regenerated), which is what verl's teacher server does
                     (experimental/teacher_loop/teacher_manager.py:40-55) and what
                     forward_kl_topk consumes (student side is a plain F.log_softmax).

Taking logprobs from pass 1 instead would hand the loss *processed* (temperature/top-p
scaled) logprobs -> wrong KL -> the experiment is void.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO = Path("/root/LLM-output-density")
EXP_ROOT = Path("/root/autodl-tmp/cycle09_offkd")
COPYBACK = (
    REPO
    / "mypaper/local_experiment_results"
    / "cycle_09_aaai_competitiveness_completion/run_01/offkd"
)
TEACHER = Path("/root/autodl-tmp/model/Qwen/Qwen3-8B")
# Pass 1 may be overridden by the frozenSelf0-KD materializer; pass 2 remains TEACHER.
SAMPLING_MODEL = TEACHER
PROMPTS = Path("/root/autodl-tmp/cycle08_opd_trajectory/data/opd_prompts_5k.parquet")

TOKENIZER_LOADER = None
# Pinned to Cycle 08's real training run (2026-07-02/10-40-01), verified in-config:
#   rollout.temperature 0.6 / top_p 0.9 / top_k -1 / seed 42 / do_sample true / dtype bf16
#   data.max_response_length 10240 / data.max_prompt_length 1024 / max_model_len 11265
TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = -1
SEED = 42
MAX_TOKENS = 10240
MAX_PROMPT_TOKENS = 1024
MAX_MODEL_LEN = 11265
TOPK_LOGPROBS = 32
RUN_LABEL = "offkd"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024**2)
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def load_prompts(limit: int | None) -> list[dict[str, Any]]:
    frame = pd.read_parquet(PROMPTS)
    rows = []
    for index, row in frame.iterrows():  # file order == OPD training order
        messages = row["prompt"]
        messages = messages.tolist() if hasattr(messages, "tolist") else list(messages)
        rows.append(
            {
                "index": int(index),
                "messages": [dict(m) for m in messages],
                "data_source": str(row["data_source"]),
                "ground_truth": str(row["reward_model"]["ground_truth"]),
            }
        )
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stage", choices=("pass1", "pass2", "all"), default="all")
    # pass 1 wants a big KV cache; pass 2 must leave room for the [T, vocab] prompt-logprob
    # buffer (11k tokens x 152k vocab x 4B = 6.8 GB), so it runs at a lower utilization.
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--gpu-mem-pass2", type=float, default=0.45)
    parser.add_argument("--pass2-batch-tokens", type=int, default=2048)
    parser.add_argument(
        "--pass2-record-batch",
        type=int,
        default=4,
        help="Maximum records whose Python prompt-logprob objects may coexist.",
    )
    parser.add_argument(
        "--pass2-min-available-ram-gib",
        type=float,
        default=128.0,
        help="Stop after a flushed batch if host available RAM falls below this guard.",
    )
    parser.add_argument(
        "--pass1-minutes",
        type=float,
        default=0.0,
        help="Preserve measured pass-1 time when resuming with --stage pass2.",
    )
    parser.add_argument("--out", type=Path, default=EXP_ROOT / "rollout")
    args = parser.parse_args()

    if args.smoke:
        args.limit = 8
        args.out = EXP_ROOT / "smoke/rollout"

    args.out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.limit or None)
    print(f"[offkd] prompts={len(prompts)} (file order, opd_prompts_5k.parquet)", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = TOKENIZER_LOADER(SAMPLING_MODEL) if TOKENIZER_LOADER else AutoTokenizer.from_pretrained(str(SAMPLING_MODEL))
    raw_path = args.out / "teacher_rollout_pass1.jsonl"
    gen_seconds = args.pass1_minutes * 60.0

    # ---------------------------------------------------------------- pass 1
    if args.stage in ("pass1", "all"):
      llm = LLM(
        model=str(SAMPLING_MODEL),
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=MAX_MODEL_LEN,
        seed=SEED,
        enforce_eager=False,
        max_logprobs=TOPK_LOGPROBS,
      )
      rendered = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        for row in prompts
      ]
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
      gen_seconds = time.time() - started
      print(f"[offkd] pass1 generation done in {gen_seconds / 60:.1f} min", flush=True)

      records = []
      for row, text, output in zip(prompts, rendered, outputs):
        completion = output.outputs[0]
        prompt_ids = list(output.prompt_token_ids)
        gen_ids = list(completion.token_ids)
        records.append(
            {
                "index": row["index"],
                "data_source": row["data_source"],
                "ground_truth": row["ground_truth"],
                "prompt": text,
                "generation": completion.text,
                "prompt_token_ids": prompt_ids,
                "generation_token_ids": gen_ids,
                "finish_reason": completion.finish_reason,
                "n_prompt_tokens": len(prompt_ids),
                "n_tokens": len(gen_ids),
                "has_boxed": "\\boxed{" in completion.text,
            }
        )
      # persist pass-1 BEFORE touching pass 2: a pass-2 crash must never cost the
      # generation hours (the first smoke died exactly there, on an OOM).
      with open(raw_path, "w", encoding="utf-8") as handle:
          for record in records:
              handle.write(json.dumps(record, ensure_ascii=False) + "\n")
      print(f"[offkd] pass1 persisted -> {raw_path}", flush=True)
      del llm
      gc.collect()
      torch.cuda.empty_cache()

    if args.stage == "pass1":
        return

    # ---------------------------------------------------------------- pass 2
    # RAW top-32 logprobs: forward pass over the EXACT pass-1 token sequence at
    # temperature=1.0 (verl teacher convention). No regeneration.
    # Runs in its own engine at lower utilization: prompt_logprobs materializes a
    # [tokens, vocab] buffer (11k x 152k x 4B = 6.8 GB) that OOMs under pass-1's KV cache.
    records = [json.loads(l) for l in open(raw_path, encoding="utf-8")]
    if args.pass2_record_batch < 1:
        raise ValueError("--pass2-record-batch must be positive")
    total_tokens = sum(int(record["n_tokens"]) for record in records)
    stream_dir = args.out / "pass2_stream"
    stream_dir.mkdir(parents=True, exist_ok=True)
    progress_path = stream_dir / "progress.json"
    ids_path = stream_dir / "top32_ids.npy"
    logprobs_path = stream_dir / "top32_logprob.npy"
    offsets_path = stream_dir / "row_offsets.npy"
    source_sha256 = file_sha256(raw_path)

    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {
            "source_sha256": source_sha256,
            "n_records": len(records),
            "total_tokens": total_tokens,
            "topk": TOPK_LOGPROBS,
        }
        mismatched = {
            key: (progress.get(key), value)
            for key, value in expected.items()
            if progress.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"pass2 progress/source mismatch: {mismatched}")
        top_ids = np.lib.format.open_memmap(ids_path, mode="r+")
        top_logprobs = np.lib.format.open_memmap(logprobs_path, mode="r+")
        offsets = np.lib.format.open_memmap(offsets_path, mode="r+")
        next_record = int(progress["next_record"])
        cursor = int(progress["cursor"])
        lp_seconds = float(progress["elapsed_seconds"])
        expected_cursor = sum(int(record["n_tokens"]) for record in records[:next_record])
        if cursor != expected_cursor:
            raise RuntimeError(
                f"pass2 progress cursor={cursor}, expected={expected_cursor}"
            )
        print(
            f"[offkd] pass2 resume record={next_record}/{len(records)} "
            f"cursor={cursor}/{total_tokens}",
            flush=True,
        )
    else:
        top_ids = np.lib.format.open_memmap(
            ids_path,
            mode="w+",
            dtype=np.int32,
            shape=(total_tokens, TOPK_LOGPROBS),
        )
        top_logprobs = np.lib.format.open_memmap(
            logprobs_path,
            mode="w+",
            dtype=np.float32,
            shape=(total_tokens, TOPK_LOGPROBS),
        )
        offsets = np.lib.format.open_memmap(
            offsets_path,
            mode="w+",
            dtype=np.int64,
            shape=(len(records), 2),
        )
        next_record = 0
        cursor = 0
        lp_seconds = 0.0
        progress = {
            "schema_version": 1,
            "status": "running",
            "source_sha256": source_sha256,
            "n_records": len(records),
            "total_tokens": total_tokens,
            "topk": TOPK_LOGPROBS,
            "next_record": next_record,
            "cursor": cursor,
            "elapsed_seconds": lp_seconds,
            "record_batch": args.pass2_record_batch,
        }
        atomic_json(progress_path, progress)

    if top_ids.shape != (total_tokens, TOPK_LOGPROBS):
        raise RuntimeError(f"unexpected ids memmap shape: {top_ids.shape}")
    if top_logprobs.shape != top_ids.shape:
        raise RuntimeError(f"unexpected logprob memmap shape: {top_logprobs.shape}")
    if offsets.shape != (len(records), 2):
        raise RuntimeError(f"unexpected offsets memmap shape: {offsets.shape}")

    llm = None
    if next_record < len(records):
        llm = LLM(
            model=str(TEACHER),
            dtype="bfloat16",
            gpu_memory_utilization=args.gpu_mem_pass2,
            max_model_len=MAX_MODEL_LEN,
            seed=SEED,
            enforce_eager=True,
            max_logprobs=TOPK_LOGPROBS,
            max_num_batched_tokens=args.pass2_batch_tokens,
            max_num_seqs=1,
        )
        forward = SamplingParams(
            temperature=1.0, max_tokens=1, prompt_logprobs=TOPK_LOGPROBS
        )

        while next_record < len(records):
            batch_end = min(next_record + args.pass2_record_batch, len(records))
            batch_records = records[next_record:batch_end]
            sequences = [
                {
                    "prompt_token_ids": (
                        record["prompt_token_ids"] + record["generation_token_ids"]
                    )
                }
                for record in batch_records
            ]
            started = time.time()
            logprob_outputs = llm.generate(sequences, forward, use_tqdm=False)
            lp_seconds += time.time() - started

            for row, (record, output) in enumerate(
                zip(batch_records, logprob_outputs), start=next_record
            ):
                token_start = int(record["n_prompt_tokens"])
                token_end = token_start + int(record["n_tokens"])
                prompt_logprobs = output.prompt_logprobs
                if token_end > len(prompt_logprobs):
                    raise RuntimeError(
                        f"row={row} prompt_logprobs={len(prompt_logprobs)} "
                        f"needed={token_end}"
                    )
                next_cursor = cursor + int(record["n_tokens"])
                ids_view = top_ids[cursor:next_cursor]
                logprobs_view = top_logprobs[cursor:next_cursor]
                ids_view.fill(-1)
                logprobs_view.fill(np.nan)
                for position in range(token_start, token_end):
                    entry = prompt_logprobs[position]
                    if not entry:
                        continue
                    ranked = sorted(
                        entry.items(), key=lambda item: item[1].logprob, reverse=True
                    )
                    for slot, (token_id, info) in enumerate(
                        ranked[:TOPK_LOGPROBS]
                    ):
                        ids_view[position - token_start, slot] = int(token_id)
                        logprobs_view[position - token_start, slot] = float(
                            info.logprob
                        )
                if np.any(ids_view < 0) or not np.isfinite(logprobs_view).all():
                    raise RuntimeError(f"incomplete top-32 extraction at row={row}")
                offsets[row] = (cursor, next_cursor)
                cursor = next_cursor

            top_ids.flush()
            top_logprobs.flush()
            offsets.flush()
            next_record = batch_end
            del output, prompt_logprobs, ids_view, logprobs_view
            del logprob_outputs, sequences, batch_records
            gc.collect()
            free_ram = available_ram_gib()
            progress.update(
                {
                    "next_record": next_record,
                    "cursor": cursor,
                    "elapsed_seconds": lp_seconds,
                    "available_ram_gib_after_gc": free_ram,
                    "updated_at_unix": time.time(),
                }
            )
            atomic_json(progress_path, progress)
            print(
                f"[offkd] pass2 streamed records={next_record}/{len(records)} "
                f"tokens={cursor}/{total_tokens} free_ram={free_ram:.1f}GiB",
                flush=True,
            )
            if free_ram < args.pass2_min_available_ram_gib:
                raise MemoryError(
                    f"host RAM guard tripped at record={next_record}: "
                    f"{free_ram:.1f} GiB available < "
                    f"{args.pass2_min_available_ram_gib:.1f} GiB"
                )

    if cursor != total_tokens or next_record != len(records):
        raise RuntimeError(
            f"pass2 incomplete: records={next_record}/{len(records)}, "
            f"tokens={cursor}/{total_tokens}"
        )
    progress["status"] = "stream_complete"
    progress["completed_at_unix"] = time.time()
    atomic_json(progress_path, progress)
    print(f"[offkd] pass2 teacher logprobs done in {lp_seconds / 60:.1f} min", flush=True)
    if llm is not None:
        del llm
        llm = None
        gc.collect()
        torch.cuda.empty_cache()

    with open(args.out / "teacher_rollout.jsonl", "w", encoding="utf-8") as handle:
        for record, (start, end) in zip(records, offsets):
            start, end = int(start), int(end)
            record["logprob_row_start"] = start
            record["logprob_row_end"] = end
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    np.savez(
        args.out / "teacher_top32_logprob.npz",
        top32_ids=top_ids,
        top32_logprob=top_logprobs,
        row_offsets=np.asarray(offsets),
    )

    lengths = np.array([r["n_tokens"] for r in records])
    truncated = sum(1 for r in records if r["finish_reason"] == "length")
    manifest = {
        "schema_version": 1,
        "stage": f"{RUN_LABEL} stage 1: teacher rollout",
        "arm": RUN_LABEL,
        "teacher_model": str(TEACHER),
        "prompt_file": str(PROMPTS),
        "prompt_order": "file order (== OPD training order); seed-42 5k sample of Math-CoT-20k",
        "n_prompts": len(records),
        "sampling_pass1": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "n": 1,
            "seed": SEED,
            "max_tokens": MAX_TOKENS,
            "dtype": "bfloat16",
            "engine": "vllm",
        },
        "logprob_pass2": {
            "convention": "RAW (temperature=1.0, prompt_logprobs) — verl teacher convention",
            "topk": TOPK_LOGPROBS,
            "input": "exact pass-1 token sequence (prompt_token_ids + generation_token_ids)",
            "why": (
                "forward_kl_topk consumes F.log_softmax(student_logits) on the student side; "
                "verl's teacher server forces temperature=1.0 prompt_logprobs "
                "(teacher_manager.py:40-55). Sampling-time logprobs would be processed -> wrong KL."
            ),
        },
        "config_provenance": (
            "Cycle 08 real training run outputs/2026-07-02/10-40-01/.hydra/config.yaml "
            "(verified: max_response_length=10240, max_prompt_length=1024, max_model_len=11265); "
            "the run cited in the handoff (2026-06-30/16-32-23) is the 16-prompt smoke"
        ),
        "truncation_rate": truncated / max(len(records), 1),
        "n_truncated": truncated,
        "truncated_kept": True,
        "length_stats": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "max": int(lengths.max()),
        },
        "has_boxed_rate": float(np.mean([r["has_boxed"] for r in records])),
        "timing_minutes": {"pass1_generation": gen_seconds / 60, "pass2_logprobs": lp_seconds / 60},
    }
    (args.out / "rollout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"[offkd] truncation_rate={manifest['truncation_rate']:.4f} "
        f"len mean/median/p90={lengths.mean():.0f}/{np.median(lengths):.0f}/"
        f"{np.percentile(lengths, 90):.0f} boxed={manifest['has_boxed_rate']:.3f}",
        flush=True,
    )

    del llm, top_ids, top_logprobs, offsets
    gc.collect()
    torch.cuda.empty_cache()

    if not args.smoke:
        COPYBACK.mkdir(parents=True, exist_ok=True)
        for name in ("teacher_rollout.jsonl", "rollout_manifest.json"):
            (COPYBACK / name).write_bytes((args.out / name).read_bytes())
        print(f"[offkd] copyback -> {COPYBACK}", flush=True)


if __name__ == "__main__":
    main()
