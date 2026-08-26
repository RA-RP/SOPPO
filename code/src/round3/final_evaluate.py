"""Selected-checkpoint-only Round3 independent 1,000-pair final evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import jsonlines
import numpy as np
import torch

from ..model.dpo_loss import compute_sequence_logprob, response_token_count
from ..model.model_utils import DTYPES, load_adapter_for_inference, load_policy_model, load_tokenizer
from .config import load_round3_config, validate_round3_config
from .data import PairCollator, PairDataset, file_sha256
from .queue_protocol import canonical_json


QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)


def _reference_cache(path: Path) -> Dict[str, Dict[str, float]]:
    records = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in records:
                raise ValueError(f"Duplicate final-test reference ID: {sample_id}")
            records[sample_id] = {
                "ref_logp_a": float(row["ref_logp_a"]),
                "ref_logp_b": float(row["ref_logp_b"]),
            }
    return records


def _private_labels(path: Path, expected_ids: Sequence[str]) -> Dict[str, int]:
    labels = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            if row["sample_id"] in labels or int(row["label"]) not in {0, 1}:
                raise ValueError("Malformed private Round3 test label")
            labels[row["sample_id"]] = int(row["label"])
    if set(labels) != set(expected_ids) or len(labels) != len(expected_ids):
        raise ValueError("Private labels do not exactly match Round3 public test IDs")
    return labels


def _move(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _ece(probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    bins = []
    total = len(probabilities)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1]
            if index == 14
            else probabilities < edges[index + 1]
        )
        count = int(mask.sum())
        accuracy = float(labels[mask].mean()) if count else 0.0
        confidence = float(probabilities[mask].mean()) if count else 0.0
        contribution = (count / total) * abs(accuracy - confidence) if count else 0.0
        ece += contribution
        bins.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "right_closed": index == 14,
                "count": count,
                "accuracy": accuracy,
                "mean_probability": confidence,
                "contribution": contribution,
            }
        )
    return {"ece_15": float(ece), "bins": bins}


def _metrics(probabilities: Sequence[float], labels: Sequence[int]) -> Dict[str, Any]:
    p = np.asarray(probabilities, dtype=np.float64)
    z = np.asarray(labels, dtype=np.int64)
    if len(p) != 1000 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Round3 final probabilities are incomplete or invalid")
    tie = p == 0.5
    accuracy_credit = np.where(tie, 0.5, ((p > 0.5) == z).astype(np.float64))
    clamped = np.clip(p, 1e-12, 1 - 1e-12)
    clamp_count = int(np.sum(clamped != p))
    nll = -np.mean(z * np.log(clamped) + (1 - z) * np.log(1 - clamped))
    confidence = np.maximum(p, 1 - p)
    predicted_a_rate = float(np.mean(p > 0.5))
    ece = _ece(p, z)
    return {
        "samples": len(p),
        "accuracy_tie_half_credit": float(accuracy_credit.mean()),
        "exact_tie_count": int(tie.sum()),
        "nll_report_clamp_1e_12": float(nll),
        "nll_clamp_count": clamp_count,
        "brier": float(np.mean((p - z) ** 2)),
        **ece,
        "probability_mean": float(p.mean()),
        "probability_std": float(p.std()),
        "probability_quantiles": {str(value): float(np.quantile(p, value)) for value in QUANTILES},
        "probability_near_zero_le_0.01": float(np.mean(p <= 0.01)),
        "probability_near_half_abs_le_0.01": float(np.mean(np.abs(p - 0.5) <= 0.01)),
        "probability_near_one_ge_0.99": float(np.mean(p >= 0.99)),
        "sum_probability_a": float(p.sum()),
        "sum_probability_b": float((1 - p).sum()),
        "confidence_mean": float(confidence.mean()),
        "confidence_std": float(confidence.std()),
        "confidence_quantiles": {str(value): float(np.quantile(confidence, value)) for value in QUANTILES},
        "confidence_ge_0.6": float(np.mean(confidence >= 0.6)),
        "confidence_ge_0.7": float(np.mean(confidence >= 0.7)),
        "confidence_ge_0.9": float(np.mean(confidence >= 0.9)),
        "confidence_ge_0.99": float(np.mean(confidence >= 0.99)),
        "collapse_extreme_probability_rate": float(np.mean((p <= 0.01) | (p >= 0.99))),
        "collapse_majority_side_rate": max(predicted_a_rate, 1.0 - predicted_a_rate),
        "predicted_a_rate": predicted_a_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint")
    group.add_argument("--frozen-base", action="store_true")
    parser.add_argument("--best-json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    config = load_round3_config(args.config)
    validate_round3_config(config)
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("Round3 final evaluation requires CUDA_VISIBLE_DEVICES=0")
    if args.batch_size < 1:
        raise ValueError("Round3 final evaluation batch size must be positive")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 final evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    data_dir = Path(config["data"]["data_dir"]).resolve()
    cache_dir = Path(config["data"]["reference_cache_dir"]).resolve()
    public_path = data_dir / "test_1k.public.jsonl"
    reference = _reference_cache(cache_dir / "test_1k.reference.jsonl")
    tokenizer = load_tokenizer(config["model"]["name_or_path"])
    dataset = PairDataset(public_path, tokenizer, require_labels=False, reference_cache=reference)
    if len(dataset) != 1000:
        raise ValueError("Round3 independent test must contain exactly 1,000 pairs")
    labels_by_id = _private_labels(
        data_dir / "test_1k.private_labels.jsonl",
        [row["sample_id"] for row in dataset.rows],
    )
    selected = None
    if args.frozen_base:
        method_id = "frozen_base"
        model = load_policy_model(
            config["model"]["name_or_path"],
            config["model"]["manifest_path"],
            config["model"]["torch_dtype"],
            config["model"]["attention_implementation"],
            gradient_checkpointing=False,
        )
    else:
        if not args.best_json:
            raise ValueError("Selected-checkpoint evaluation requires --best-json")
        selected = json.loads(Path(args.best_json).read_text(encoding="utf-8"))
        if Path(selected["checkpoint"]).resolve() != Path(args.checkpoint).resolve():
            raise ValueError("Round3 final checkpoint differs from common selection best.json")
        if selected.get("method_id") != config["method"]["name"]:
            raise ValueError("Round3 final checkpoint method differs from resolved config")
        checkpoint = Path(args.checkpoint).resolve()
        metadata = json.loads((checkpoint / "checkpoint_meta.json").read_text(encoding="utf-8"))
        complete = json.loads((checkpoint / "COMPLETE.json").read_text(encoding="utf-8"))
        selected_step = int(selected.get("checkpoint_step", -1))
        if complete != metadata:
            raise ValueError("Round3 selected checkpoint COMPLETE marker/metadata mismatch")
        if (
            selected_step not in range(25, 251, 25)
            or int(metadata.get("optimizer_step", -1)) != selected_step
            or metadata.get("method_id") != config["method"]["name"]
            or metadata.get("git_commit") != config["provenance"]["git_commit"]
            or metadata.get("config_sha256")
            != hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
            or metadata.get("adapter_sha256")
            != file_sha256(checkpoint / "adapter_model.safetensors")
            or metadata.get("training_state_sha256")
            != file_sha256(checkpoint / "training_state.pt")
        ):
            raise ValueError("Round3 selected durable checkpoint provenance mismatch")
        verified = json.loads(
            (Path(args.best_json).resolve().parent / "selection_verified.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            verified.get("status") != "verified"
            or int(verified.get("checkpoint_step", -1)) != selected_step
            or Path(verified.get("checkpoint", "")).resolve() != checkpoint
        ):
            raise ValueError("Round3 final checkpoint lacks matching independent selection evidence")
        method_id = config["method"]["name"]
        model = load_adapter_for_inference(
            str(checkpoint),
            config["model"]["name_or_path"],
            config["model"]["manifest_path"],
            config["model"]["torch_dtype"],
        )
    model.cuda().eval()
    collator = PairCollator(tokenizer.pad_token_id)
    dtype = DTYPES[config["model"]["torch_dtype"]]
    ref_probabilities: List[float] = []
    raw_probabilities: List[float] = []
    labels: List[int] = []
    sample_ids: List[str] = []
    truncation = {
        "prompt_tokens_removed_a": 0,
        "prompt_tokens_removed_b": 0,
        "response_tokens_removed_a": 0,
        "response_tokens_removed_b": 0,
    }
    with torch.inference_mode():
        for start in range(0, len(dataset), args.batch_size):
            examples = [dataset[index] for index in range(start, min(len(dataset), start + args.batch_size))]
            for key in truncation:
                truncation[key] += sum(int(row[key]) for row in examples)
            batch = _move(collator(examples))
            totals, means = [], []
            for side in ("a", "b"):
                with torch.autocast("cuda", dtype=dtype):
                    outputs = model(
                        input_ids=batch[f"input_ids_{side}"],
                        attention_mask=batch[f"attention_mask_{side}"],
                        use_cache=False,
                        return_dict=True,
                    )
                    total = compute_sequence_logprob(
                        outputs.logits,
                        batch[f"input_ids_{side}"],
                        batch[f"loss_mask_{side}"],
                    ).float()
                    totals.append(total)
                    means.append(total / response_token_count(batch[f"loss_mask_{side}"]))
            if args.frozen_base:
                ref_probability = torch.full_like(totals[0], 0.5)
            else:
                ref_delta = 0.1 * (
                    (totals[0] - batch["ref_logp_a"])
                    - (totals[1] - batch["ref_logp_b"])
                )
                ref_probability = torch.sigmoid(ref_delta)
            raw_probability = torch.sigmoid(10.0 * (means[0] - means[1]))
            if not torch.isfinite(ref_probability).all() or not torch.isfinite(raw_probability).all():
                raise FloatingPointError("Non-finite Round3 final-test probability")
            for index, sample_id in enumerate(batch["sample_ids"]):
                sample_ids.append(sample_id)
                labels.append(labels_by_id[sample_id])
                ref_probabilities.append(float(ref_probability[index]))
                raw_probabilities.append(float(raw_probability[index]))
    heads = {
        "dpo_reference_delta_beta_0.1": _metrics(ref_probabilities, labels),
        "raw_mean_logp_delta_beta_10": _metrics(raw_probabilities, labels),
    }
    result = {
        "schema_version": "round3.final_metrics.v1",
        "method_id": method_id,
        "selected_step": int(selected["checkpoint_step"]) if selected else None,
        "best_eval_selection_loss": float(selected["eval_selection_loss"]) if selected else None,
        "test_view": "independent_test_1k",
        "truncation": truncation,
        "heads": heads,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with jsonlines.open(output_dir / "predictions.private.jsonl", "w") as writer:
        for sample_id, label, probability_ref, probability_raw in zip(
            sample_ids, labels, ref_probabilities, raw_probabilities
        ):
            writer.write(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "dpo_reference_delta_beta_0.1": probability_ref,
                    "raw_mean_logp_delta_beta_10": probability_raw,
                }
            )
    (output_dir / "complete.json").write_text(
        json.dumps({"status": "succeeded", "metrics": "metrics.json"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
