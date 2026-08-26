"""Precompute frozen Qwen3-1.7B total-response log-probs for Round3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import jsonlines
import torch

from ..model.dpo_loss import compute_sequence_logprob
from ..model.model_utils import DTYPES, load_policy_model, load_tokenizer
from .config import load_round3_config, validate_round3_config
from .data import PairCollator, PairDataset, TOKENIZATION_CONTRACT, file_sha256
from .queue_protocol import canonical_json


def _move(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _score_file(
    model,
    tokenizer,
    input_path: Path,
    output_path: Path,
    require_labels: bool,
    batch_size: int,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 reference cache: {output_path}")
    dataset = PairDataset(input_path, tokenizer, require_labels=require_labels)
    collator = PairCollator(tokenizer.pad_token_id)
    rows: List[Dict[str, Any]] = []
    device = torch.device("cuda:0")
    with torch.inference_mode():
        for start in range(0, len(dataset), int(batch_size)):
            examples = [dataset[index] for index in range(start, min(len(dataset), start + int(batch_size)))]
            batch = _move(collator(examples), device)
            values = []
            for side in ("a", "b"):
                with torch.autocast("cuda", dtype=dtype):
                    outputs = model(
                        input_ids=batch[f"input_ids_{side}"],
                        attention_mask=batch[f"attention_mask_{side}"],
                        use_cache=False,
                        return_dict=True,
                    )
                    values.append(
                        compute_sequence_logprob(
                            outputs.logits,
                            batch[f"input_ids_{side}"],
                            batch[f"loss_mask_{side}"],
                        ).float()
                    )
            if not torch.isfinite(values[0]).all() or not torch.isfinite(values[1]).all():
                raise FloatingPointError("Non-finite Round3 frozen-reference score")
            for sample_id, value_a, value_b in zip(batch["sample_ids"], values[0], values[1]):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "ref_logp_a": float(value_a),
                        "ref_logp_b": float(value_b),
                    }
                )
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(rows)
    return {
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output": str(output_path),
        "output_sha256": file_sha256(output_path),
        "rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    if args.batch_size < 1:
        raise ValueError("Reference-cache batch size must be positive")
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Round3 reference cache requires CUDA_VISIBLE_DEVICES=0")
    data_dir = Path(config["data"]["data_dir"]).resolve()
    output_dir = Path(config["data"]["reference_cache_dir"]).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refuse to reuse Round3 reference-cache directory: {output_dir}")
    output_dir.mkdir(parents=True)
    tokenizer = load_tokenizer(config["model"]["name_or_path"])
    dtype = DTYPES[config["model"]["torch_dtype"]]
    model = load_policy_model(
        config["model"]["name_or_path"],
        config["model"]["manifest_path"],
        config["model"]["torch_dtype"],
        config["model"]["attention_implementation"],
        gradient_checkpointing=False,
    ).cuda().eval()
    specs = (
        ("paired_train_8k.jsonl", "paired_train_8k.reference.jsonl", True),
        ("paired_train_1k.jsonl", "paired_train_1k.reference.jsonl", True),
        ("validation_1k.jsonl", "validation_1k.reference.jsonl", True),
        ("test.public.jsonl", "test.reference.jsonl", False),
    )
    files = {}
    for input_name, output_name, labels in specs:
        files[output_name] = _score_file(
            model,
            tokenizer,
            data_dir / input_name,
            output_dir / output_name,
            labels,
            args.batch_size,
            dtype,
        )
    manifest = {
        "schema_version": "round3.reference_cache_manifest.v1",
        "model": config["model"]["repo_id"],
        "model_path": str(Path(config["model"]["name_or_path"]).resolve()),
        "model_manifest": str(Path(config["model"]["manifest_path"]).resolve()),
        "model_manifest_sha256": file_sha256(config["model"]["manifest_path"]),
        "tokenization_contract": TOKENIZATION_CONTRACT,
        "git_commit": config["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest(),
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
