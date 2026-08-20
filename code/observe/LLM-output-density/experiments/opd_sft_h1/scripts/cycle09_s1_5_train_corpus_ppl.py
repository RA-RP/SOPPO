#!/usr/bin/env python3
"""S1-5: base-model conditional PPL on OPD, SFT, and off-KD training texts.

Cycle 08 did not persist its online student responses. The prepare stage therefore
creates a clearly labelled, deterministic step-0 reconstruction on 500 frozen OPD
prompts with the Cycle 08 sampling parameters. It is never labelled as an original
training archive. PPL scores response tokens only, conditioned on the prompt, with
one common right-truncation rule for all three corpora.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import cycle09_r4_common as c4


PROMPT_PARQUET = Path(
    "/root/autodl-tmp/cycle08_opd_trajectory/data/opd_prompts_5k.parquet"
)
SFT_JSONL = Path(
    "/root/autodl-tmp/cycle07_base_sft_trajectory/data_prep/train_5k.jsonl"
)
OFFKD_JSONL = Path("/root/autodl-tmp/cycle09_offkd/rollout/teacher_rollout.jsonl")
RUN_ROOT = Path("/root/autodl-tmp/cycle09_s1")
MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)

N_SAMPLES = 500
SEED = 42
MAX_TOTAL_TOKENS = 10240
MAX_PROMPT_TOKENS = 1024
MAX_GENERATION_TOKENS = 10240
TEMPERATURE = 0.6
TOP_P = 0.9
TOP_K = -1
GEN_BATCH_SIZE = 16
PROTOCOL_VERSION = "s1-5-conditional-response-ppl-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def atomic_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def selected_indices(limit: int) -> list[int]:
    rng = np.random.default_rng(SEED)
    return [int(index) for index in rng.permutation(5000)[:limit]]


def load_prompt_rows(indices: list[int]) -> list[dict]:
    frame = pd.read_parquet(PROMPT_PARQUET)
    if len(frame) != 5000:
        raise RuntimeError(f"expected 5000 OPD prompts, found {len(frame)}")
    rows = []
    for index in indices:
        row = frame.iloc[index]
        messages = row["prompt"]
        messages = messages.tolist() if hasattr(messages, "tolist") else list(messages)
        rows.append(
            {
                "index": index,
                "messages": [dict(message) for message in messages],
                "data_source": str(row["data_source"]),
            }
        )
    return rows


def opd_root(smoke: bool) -> Path:
    return RUN_ROOT / ("s1_5_opd_rollout_smoke" if smoke else "s1_5_opd_rollout")


def prepare_opd(smoke: bool, gpu_memory: float) -> None:
    limit = 8 if smoke else N_SAMPLES
    indices = selected_indices(limit)
    rows = load_prompt_rows(indices)
    root = opd_root(smoke)
    records_dir = root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    pending = []
    complete = {}
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        target = records_dir / f"{row['index']:05d}.json"
        if target.is_file():
            payload = json.loads(target.read_text(encoding="utf-8"))
            if (
                payload.get("index") == row["index"]
                and payload.get("prompt_sha256") == prompt_hash
                and payload.get("sampling_protocol") == "cycle08_step0_reconstruction_v1"
            ):
                complete[row["index"]] = payload
                continue
            raise RuntimeError(f"incompatible OPD reconstruction cache: {target}")
        pending.append((row, rendered, prompt_hash, target))

    print(
        f"[S1-5 prepare] requested={limit} cached={len(complete)} pending={len(pending)}",
        flush=True,
    )
    started = time.monotonic()
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=str(c4.BASE_MODEL),
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory,
            max_model_len=11265,
            seed=SEED,
            enforce_eager=False,
        )
        try:
            for start in range(0, len(pending), GEN_BATCH_SIZE):
                batch = pending[start : start + GEN_BATCH_SIZE]
                prompts = [item[1] for item in batch]
                sampling = [
                    SamplingParams(
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        top_k=TOP_K,
                        n=1,
                        max_tokens=MAX_GENERATION_TOKENS,
                        seed=c4.stable_seed(SEED, "S1-5-opd", item[0]["index"]),
                    )
                    for item in batch
                ]
                outputs = llm.generate(prompts, sampling)
                for item, output in zip(batch, outputs):
                    row, rendered, prompt_hash, target = item
                    completion = output.outputs[0]
                    prompt_ids = list(output.prompt_token_ids)
                    generation_ids = list(completion.token_ids)
                    if len(prompt_ids) > MAX_PROMPT_TOKENS:
                        raise RuntimeError(
                            f"OPD prompt {row['index']} has {len(prompt_ids)} tokens"
                        )
                    payload = {
                        "index": row["index"],
                        "data_source": row["data_source"],
                        "prompt": rendered,
                        "generation": completion.text,
                        "prompt_token_ids": prompt_ids,
                        "generation_token_ids": generation_ids,
                        "finish_reason": completion.finish_reason,
                        "n_prompt_tokens": len(prompt_ids),
                        "n_tokens": len(generation_ids),
                        "per_request_seed": c4.stable_seed(
                            SEED, "S1-5-opd", row["index"]
                        ),
                        "prompt_sha256": prompt_hash,
                        "sampling_protocol": "cycle08_step0_reconstruction_v1",
                    }
                    atomic_json(payload, target)
                    complete[row["index"]] = payload
                print(
                    f"[S1-5 prepare] {min(start + len(batch), len(pending))}/"
                    f"{len(pending)} newly generated",
                    flush=True,
                )
        finally:
            del llm
            gc.collect()
            torch.cuda.empty_cache()

    ordered = [complete[index] for index in indices]
    if len(ordered) != limit:
        raise RuntimeError(f"incomplete OPD reconstruction: {len(ordered)}/{limit}")
    output = root / "opd_step0_reconstructed_rollout.jsonl"
    atomic_jsonl(ordered, output)
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-5 OPD source preparation",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_status": "deterministic_reconstruction_not_original_training_archive",
            "reason": (
                "Cycle 08 verl configuration had rollout_data_dir=None and no student "
                "rollout archive remains on disk"
            ),
            "base_model": str(c4.BASE_MODEL),
            "prompt_source": {
                "path": str(PROMPT_PARQUET),
                "sha256": sha256_file(PROMPT_PARQUET),
                "n_total": 5000,
            },
            "selection_seed": SEED,
            "selected_indices": indices,
            "selected_indices_sha256": sha256_json(indices),
            "n_samples": limit,
            "sampling": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "max_tokens": MAX_GENERATION_TOKENS,
                "base_seed": SEED,
                "per_request_seed": "stable_seed(42, 'S1-5-opd', source_index)",
            },
            "elapsed_seconds_this_invocation": time.monotonic() - started,
            "output": str(output),
            "output_sha256": sha256_file(output),
        },
        root / "manifest.json",
    )
    print(f"[S1-5 prepare] complete -> {output}", flush=True)


def split_sft_text(text: str) -> tuple[str, str]:
    marker = "<|im_start|>assistant\n"
    if text.count(marker) != 1:
        raise ValueError("SFT row does not contain one assistant marker")
    prompt, response = text.split(marker, 1)
    prompt += marker
    suffix = "<|im_end|>\n"
    if response.endswith(suffix):
        response = response[:-1]
    elif response.endswith("<|im_end|>"):
        pass
    else:
        raise ValueError("SFT row lacks the assistant end marker")
    return prompt, response


def load_sft_samples(tokenizer, limit: int) -> tuple[list[dict], dict]:
    all_rows = [
        json.loads(line)["text"]
        for line in SFT_JSONL.open(encoding="utf-8")
        if line.strip()
    ]
    if len(all_rows) != 5000:
        raise RuntimeError(f"expected 5000 SFT rows, found {len(all_rows)}")
    indices = selected_indices(limit)
    samples = []
    for index in indices:
        prompt, response = split_sft_text(all_rows[index])
        samples.append(
            {
                "sample_id": f"sft_{index:05d}",
                "source_index": index,
                "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False),
                "response_token_ids": tokenizer.encode(response, add_special_tokens=False),
            }
        )
    return samples, {
        "path": str(SFT_JSONL),
        "sha256": sha256_file(SFT_JSONL),
        "source_kind": "Cycle07 Math-CoT-20k dataset CoT",
        "selection_seed": SEED,
        "selected_indices_sha256": sha256_json(indices),
    }


def load_indexed_jsonl(path: Path, indices: list[int]) -> dict[int, dict]:
    wanted = set(indices)
    found = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            index = int(record["index"])
            if index in wanted:
                found[index] = record
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"{path} missing selected indices: {sorted(missing)[:10]}")
    return found


def load_rollout_samples(
    path: Path, prefix: str, source_kind: str, limit: int
) -> tuple[list[dict], dict]:
    indices = selected_indices(limit)
    found = load_indexed_jsonl(path, indices)
    samples = []
    for index in indices:
        record = found[index]
        prompt_ids = record.get("prompt_token_ids")
        response_ids = record.get("generation_token_ids")
        if not isinstance(prompt_ids, list) or not isinstance(response_ids, list):
            raise ValueError(f"missing token IDs in {path}, index={index}")
        samples.append(
            {
                "sample_id": f"{prefix}_{index:05d}",
                "source_index": index,
                "prompt_token_ids": [int(token) for token in prompt_ids],
                "response_token_ids": [int(token) for token in response_ids],
            }
        )
    return samples, {
        "path": str(path),
        "sha256": sha256_file(path),
        "source_kind": source_kind,
        "selection_seed": SEED,
        "selected_indices_sha256": sha256_json(indices),
    }


def truncate_sample(sample: dict) -> dict:
    prompt = sample["prompt_token_ids"]
    response = sample["response_token_ids"]
    if not prompt or len(prompt) >= MAX_TOTAL_TOKENS:
        raise ValueError(f"invalid prompt length for {sample['sample_id']}: {len(prompt)}")
    keep = min(len(response), MAX_TOTAL_TOKENS - len(prompt))
    if keep < 1:
        raise ValueError(f"empty response after truncation: {sample['sample_id']}")
    return {
        **sample,
        "input_token_ids": prompt + response[:keep],
        "prompt_tokens": len(prompt),
        "response_tokens_original": len(response),
        "response_tokens_scored": keep,
        "truncated": int(keep < len(response)),
    }


@torch.inference_mode()
def score_sample(model, sample: dict, device: str) -> dict:
    ids = torch.tensor(
        sample["input_token_ids"], dtype=torch.long, device=device
    ).unsqueeze(0)
    labels = ids.clone()
    labels[:, : sample["prompt_tokens"]] = -100
    output = model(input_ids=ids, labels=labels, use_cache=False)
    loss = float(output.loss.detach().float().cpu())
    if not math.isfinite(loss):
        raise FloatingPointError(f"non-finite loss for {sample['sample_id']}")
    result = {
        "sample_id": sample["sample_id"],
        "source_index": sample["source_index"],
        "prompt_tokens": sample["prompt_tokens"],
        "response_tokens_original": sample["response_tokens_original"],
        "response_tokens_scored": sample["response_tokens_scored"],
        "truncated": sample["truncated"],
        "mean_nll": loss,
        "nll_sum": loss * sample["response_tokens_scored"],
    }
    del ids, labels, output
    return result


def score_corpus(
    *,
    model,
    corpus_id: str,
    samples: list[dict],
    cache_root: Path,
    protocol_id: str,
    device: str,
) -> list[dict]:
    path = cache_root / f"{corpus_id}.json"
    completed = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_id") != protocol_id:
            raise RuntimeError(f"incompatible PPL cache: {path}")
        completed = {row["sample_id"]: row for row in payload.get("rows", [])}
    for index, raw in enumerate(samples, start=1):
        sample = truncate_sample(raw)
        if sample["sample_id"] not in completed:
            completed[sample["sample_id"]] = score_sample(model, sample, device)
        if index % 10 == 0 or index == len(samples):
            ordered = [
                completed[item["sample_id"]]
                for item in samples
                if item["sample_id"] in completed
            ]
            atomic_json(
                {
                    "protocol_id": protocol_id,
                    "corpus_id": corpus_id,
                    "rows": ordered,
                },
                path,
            )
            print(
                f"[S1-5 PPL] {corpus_id} {len(ordered)}/{len(samples)}",
                flush=True,
            )
    ordered = [completed[sample["sample_id"]] for sample in samples]
    if len(ordered) != len(samples):
        raise RuntimeError(f"incomplete PPL cache for {corpus_id}")
    return ordered


def run_ppl(smoke: bool, device: str) -> None:
    limit = 8 if smoke else N_SAMPLES
    opd_path = opd_root(smoke) / "opd_step0_reconstructed_rollout.jsonl"
    if not opd_path.is_file():
        raise FileNotFoundError(f"run --stage prepare first: {opd_path}")
    tokenizer = AutoTokenizer.from_pretrained(str(c4.BASE_MODEL))
    corpora = {}
    provenance = {}
    corpora["X_OPD"], provenance["X_OPD"] = load_rollout_samples(
        opd_path,
        "opd",
        "deterministic base/step-0 student rollout reconstruction",
        limit,
    )
    corpora["X_SFT"], provenance["X_SFT"] = load_sft_samples(tokenizer, limit)
    corpora["X_offKD"], provenance["X_offKD"] = load_rollout_samples(
        OFFKD_JSONL,
        "offkd",
        "formal static Qwen3-8B teacher rollout used by off-KD training",
        limit,
    )
    protocol = {
        "version": PROTOCOL_VERSION,
        "mode": "smoke" if smoke else "formal",
        "n_samples_per_corpus": limit,
        "selection_seed": SEED,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "truncation": "keep full prompt; right-truncate response to max_total_tokens",
        "terminal_token_policy": (
            "preserve each corpus's original generated/training <|im_end|> token "
            "when present; remove only the SFT JSONL trailing newline"
        ),
        "score_scope": "response tokens only, conditioned on the complete prompt",
        "aggregation": "token-weighted NLL, then exp",
        "dtype": "bfloat16",
        "attention": "sdpa",
        "sources": provenance,
    }
    protocol_id = sha256_json(protocol)
    cache_root = (
        RUN_ROOT
        / ("s1_5_ppl_cache_smoke" if smoke else "s1_5_ppl_cache")
        / protocol_id
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(c4.BASE_MODEL),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    all_results = {}
    try:
        for corpus_id in ("X_OPD", "X_SFT", "X_offKD"):
            all_results[corpus_id] = score_corpus(
                model=model,
                corpus_id=corpus_id,
                samples=corpora[corpus_id],
                cache_root=cache_root,
                protocol_id=protocol_id,
                device=device,
            )
            torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    sample_rows = []
    summary_rows = []
    for corpus_id, rows in all_results.items():
        for row in rows:
            sample_rows.append({"corpus": corpus_id, **row})
        nll_sum = sum(row["nll_sum"] for row in rows)
        tokens = sum(row["response_tokens_scored"] for row in rows)
        mean_nll = nll_sum / tokens
        summary_rows.append(
            {
                "corpus": corpus_id,
                "ppl": math.exp(mean_nll),
                "mean_nll": mean_nll,
                "nll_sum": nll_sum,
                "n_samples": len(rows),
                "n_scored_tokens": tokens,
                "prompt_tokens_mean": float(
                    np.mean([row["prompt_tokens"] for row in rows])
                ),
                "response_tokens_original_mean": float(
                    np.mean([row["response_tokens_original"] for row in rows])
                ),
                "response_tokens_scored_mean": float(
                    np.mean([row["response_tokens_scored"] for row in rows])
                ),
                "truncation_rate": float(np.mean([row["truncated"] for row in rows])),
            }
        )
    summary = pd.DataFrame(summary_rows)
    samples_frame = pd.DataFrame(sample_rows)
    if smoke:
        print(summary.to_string(index=False))
        return
    atomic_csv(summary, MINI / "S1_train_corpus_base_ppl.csv")
    atomic_csv(samples_frame, MINI / "S1_train_corpus_base_ppl_samples.csv")
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-5",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol,
            "protocol_id": protocol_id,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "base_model": str(c4.BASE_MODEL),
            "n_summary_rows": len(summary),
            "n_sample_rows": len(samples_frame),
            "cache_root": str(cache_root),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MINI / "S1_train_corpus_base_ppl_manifest.json",
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "ppl", "all"), default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory", type=float, default=0.82)
    args = parser.parse_args()
    if args.stage in ("prepare", "all"):
        prepare_opd(args.smoke, args.gpu_memory)
    if args.stage in ("ppl", "all"):
        run_ppl(args.smoke, args.device)


if __name__ == "__main__":
    main()
