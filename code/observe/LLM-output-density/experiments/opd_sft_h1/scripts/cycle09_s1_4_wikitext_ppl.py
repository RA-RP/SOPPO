#!/usr/bin/env python3
"""S1-4: fixed-token wikitext-family PPL trajectory for three arms x ten steps."""

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
from transformers import AutoModelForCausalLM

import cycle09_r4_common as c4


CORPUS = Path("/root/autodl-tmp/cycle09_r4/corpora/fixed/E_general.jsonl")
OFFKD = Path("/root/autodl-tmp/cycle09_offkd/_merged_models")
RUN_ROOT = Path("/root/autodl-tmp/cycle09_s1")
MINI = Path(
    "/root/LLM-output-density/mypaper/local_experiment_results/"
    "cycle_09_aaai_competitiveness_completion/run_01/mini"
)
STEPS = (0, 5, 10, 20, 40, 80, 160, 320, 480, 624)
ARMS = ("opd", "sft", "offkd")
SEED = 42
PROTOCOL_VERSION = "s1-4-fixed-e-general-token-ids-v1"


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


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        tmp = Path(handle.name)
        frame.to_csv(handle, index=False)
    os.replace(tmp, path)


def model_path(arm: str, step: int) -> Path:
    if step == 0:
        return c4.BASE_MODEL
    if arm == "offkd":
        path = OFFKD / c4.step_label(step)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(path)
        return path
    return c4.model_path(arm, step)


def load_corpus(smoke: bool) -> tuple[list[dict], dict]:
    rows = []
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            token_ids = record.get("full_token_ids")
            if not isinstance(token_ids, list) or len(token_ids) < 2:
                raise ValueError(f"invalid full_token_ids for {record.get('sample_id')}")
            if record.get("eligible_start") != 0:
                raise ValueError("E_general fixed corpus unexpectedly contains a prompt region")
            rows.append(
                {
                    "sample_id": str(record["sample_id"]),
                    "token_ids": [int(token) for token in token_ids],
                }
            )
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 E_general rows, found {len(rows)}")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(rows))
    rows = [rows[int(index)] for index in order]
    if smoke:
        rows = rows[:8]
    provenance = {
        "path": str(CORPUS),
        "sha256": sha256_file(CORPUS),
        "size_bytes": CORPUS.stat().st_size,
        "source_family": "wikitext; exact frozen E_general corpus",
        "tokenization": "reuse full_token_ids stored by Round 4; no retokenization",
        "slice_seed": SEED,
        "n_documents": len(rows),
        "sample_order_sha256": sha256_json([row["sample_id"] for row in rows]),
        "min_tokens": min(len(row["token_ids"]) for row in rows),
        "max_tokens": max(len(row["token_ids"]) for row in rows),
        "total_input_tokens": sum(len(row["token_ids"]) for row in rows),
        "score_scope": "all next-token targets within each document; no cross-document targets",
    }
    return rows, provenance


def batches(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@torch.inference_mode()
def evaluate_model(
    path: Path, rows: list[dict], device: str, batch_size: int
) -> dict:
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    model.config.use_cache = False
    nll_sum = 0.0
    n_targets = 0
    try:
        for batch_index, batch in enumerate(batches(rows, batch_size), start=1):
            max_length = max(len(row["token_ids"]) for row in batch)
            input_ids = torch.zeros(
                (len(batch), max_length), dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            labels = torch.full_like(input_ids, -100)
            targets = 0
            for row_index, row in enumerate(batch):
                ids = torch.tensor(row["token_ids"], dtype=torch.long, device=device)
                length = ids.numel()
                input_ids[row_index, :length] = ids
                attention_mask[row_index, :length] = 1
                labels[row_index, :length] = ids
                targets += length - 1
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            loss = float(output.loss.detach().float().cpu())
            if not math.isfinite(loss):
                raise FloatingPointError(f"non-finite PPL loss for {path}")
            nll_sum += loss * targets
            n_targets += targets
            del input_ids, attention_mask, labels, output
            if batch_index % 4 == 0:
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    mean_nll = nll_sum / n_targets
    return {
        "n_documents": len(rows),
        "n_scored_tokens": n_targets,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "ppl": math.exp(mean_nll),
        "elapsed_seconds": time.monotonic() - started,
    }


def cache_path(root: Path, arm: str, step: int) -> Path:
    return root / f"{arm}__{c4.step_label(step)}.json"


def get_cell(
    *,
    cache_root: Path,
    arm: str,
    step: int,
    path: Path,
    rows: list[dict],
    corpus_provenance: dict,
    protocol_id: str,
    device: str,
    batch_size: int,
) -> dict:
    target = cache_path(cache_root, arm, step)
    expected = {
        "protocol_id": protocol_id,
        "corpus_sha256": corpus_provenance["sha256"],
        "sample_order_sha256": corpus_provenance["sample_order_sha256"],
        "model_path": str(path),
    }
    if target.is_file():
        cached = json.loads(target.read_text(encoding="utf-8"))
        if all(cached.get(key) == value for key, value in expected.items()):
            if cached.get("n_documents") == len(rows) and cached.get("ppl", 0) > 0:
                print(f"[S1-4 cached] {arm}/{c4.step_label(step)}", flush=True)
                return cached
        raise RuntimeError(f"incompatible existing cache: {target}")

    print(f"[S1-4] {arm}/{c4.step_label(step)} model={path}", flush=True)
    result = evaluate_model(path, rows, device, batch_size)
    payload = {
        **expected,
        **result,
        "arm": arm,
        "step": step,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(payload, target)
    print(
        f"[S1-4] {arm}/{c4.step_label(step)} ppl={payload['ppl']:.8f} "
        f"tokens={payload['n_scored_tokens']} sec={payload['elapsed_seconds']:.1f}",
        flush=True,
    )
    return payload


def checkpoint_provenance(arm: str, step: int) -> str:
    if arm == "offkd" and step in (80, 320, 480):
        return "numerical_backfill_from_landmark"
    if step == 0:
        return "shared_base"
    return "stored_checkpoint"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    rows, corpus_provenance = load_corpus(args.smoke)
    mode = "smoke" if args.smoke else "formal"
    protocol_payload = {
        "version": PROTOCOL_VERSION,
        "mode": mode,
        "corpus_sha256": corpus_provenance["sha256"],
        "sample_order_sha256": corpus_provenance["sample_order_sha256"],
        "n_documents": len(rows),
        "dtype": "bfloat16",
        "attention": "sdpa",
        "batch_size": args.batch_size,
    }
    protocol_id = sha256_json(protocol_payload)
    cache_root = RUN_ROOT / ("s1_4_cache_smoke" if args.smoke else "s1_4_cache")

    unique_cells = [("base", 0, c4.BASE_MODEL)]
    if not args.smoke:
        unique_cells.extend(
            (arm, step, model_path(arm, step))
            for arm in ARMS
            for step in STEPS
            if step != 0
        )
    results = {}
    for arm, step, path in unique_cells:
        results[(arm, step)] = get_cell(
            cache_root=cache_root,
            arm=arm,
            step=step,
            path=path,
            rows=rows,
            corpus_provenance=corpus_provenance,
            protocol_id=protocol_id,
            device=args.device,
            batch_size=args.batch_size,
        )

    if args.smoke:
        print(json.dumps(results[("base", 0)], indent=2, sort_keys=True))
        return

    output_rows = []
    base_ppl = results[("base", 0)]["ppl"]
    for arm in ARMS:
        for step in STEPS:
            result = results[("base", 0)] if step == 0 else results[(arm, step)]
            output_rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "ppl": result["ppl"],
                    "mean_nll": result["mean_nll"],
                    "nll_sum": result["nll_sum"],
                    "n_documents": result["n_documents"],
                    "n_scored_tokens": result["n_scored_tokens"],
                    "ppl_relative_change_vs_base": result["ppl"] / base_ppl - 1.0,
                    "model_path": (
                        str(c4.BASE_MODEL) if step == 0 else str(model_path(arm, step))
                    ),
                    "checkpoint_provenance": checkpoint_provenance(arm, step),
                }
            )
    frame = pd.DataFrame(output_rows).sort_values(["arm", "step"]).reset_index(drop=True)
    if len(frame) != 30:
        raise RuntimeError(f"incomplete S1-4 output: {len(frame)}")
    atomic_csv(frame, MINI / "S1_wikitext_ppl.csv")
    atomic_json(
        {
            "schema_version": 1,
            "task": "S1-4",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": protocol_payload,
            "protocol_id": protocol_id,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "corpus": corpus_provenance,
            "arms": list(ARMS),
            "steps": list(STEPS),
            "base_computed_once_and_aliased": True,
            "n_unique_model_evaluations": len(unique_cells),
            "n_output_rows": len(frame),
            "cache_root": str(cache_root),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        MINI / "S1_wikitext_ppl_manifest.json",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
