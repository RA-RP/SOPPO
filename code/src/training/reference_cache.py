"""Distributed, immutable reference log-probability cache builder (server only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import jsonlines
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler

from ..data.dataset import (
    PreferenceCollator,
    PreferenceDataset,
    create_dataloader,
    data_file_sha256,
)
from ..model.dpo_loss import model_pair_logps
from ..model.model_utils import DTYPES, load_policy_model, load_tokenizer


def synchronized_rank_zero_check(rank: int, message: str | None) -> None:
    payload = [message if rank == 0 else None]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(payload[0])


def initialize() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rank, local_rank, world_size = initialize()
    output = Path(args.output).resolve()
    output_error = None
    if rank == 0 and (output.exists() or output.with_suffix(".manifest.json").exists()):
        output_error = f"Refuse to overwrite reference cache: {output}"
    synchronized_rank_zero_check(rank, output_error)

    tokenizer = load_tokenizer(args.model)
    dataset = PreferenceDataset(
        args.input,
        tokenizer,
        max_length=args.max_length,
        enable_thinking=False,
        limit=args.limit,
    )
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False) if world_size > 1 else None
    loader = create_dataloader(
        dataset,
        args.batch_size,
        PreferenceCollator(tokenizer.pad_token_id),
        shuffle=False,
        sampler=sampler,
    )
    model = load_policy_model(
        args.model,
        args.model_manifest,
        dtype_name=args.dtype,
        gradient_checkpointing=False,
    ).to(local_rank)
    model.eval()
    shard = output.with_suffix(output.suffix + f".rank{rank}.tmp")
    shard.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(shard, "w") as writer, torch.inference_mode():
        for batch in loader:
            device_batch = {
                key: value.to(local_rank) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            with torch.autocast("cuda", dtype=DTYPES[args.dtype]):
                logp_a, logp_b = model_pair_logps(model, device_batch)
            for sample_id, value_a, value_b in zip(batch["sample_ids"], logp_a, logp_b):
                writer.write(
                    {
                        "sample_id": sample_id,
                        "ref_logp_a": float(value_a),
                        "ref_logp_b": float(value_b),
                    }
                )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        rows = {}
        for shard_rank in range(world_size):
            current = output.with_suffix(output.suffix + f".rank{shard_rank}.tmp")
            with jsonlines.open(current) as reader:
                for row in reader:
                    rows[row["sample_id"]] = row
        if len(rows) != len(dataset):
            raise ValueError(f"Reference cache row mismatch: {len(rows)} != {len(dataset)}")
        with jsonlines.open(output, "w") as writer:
            for sample_id in sorted(rows):
                writer.write(rows[sample_id])
        for shard_rank in range(world_size):
            output.with_suffix(output.suffix + f".rank{shard_rank}.tmp").unlink()
        manifest_bytes = Path(args.model_manifest).read_bytes()
        manifest = {
            "schema_version": 1,
            "model_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "input_sha256": data_file_sha256(args.input),
            "rows": len(rows),
            "max_length": args.max_length,
            "enable_thinking": False,
            "response_only": True,
            "cache_sha256": data_file_sha256(output),
        }
        output.with_suffix(".manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Reference cache complete: {output}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
