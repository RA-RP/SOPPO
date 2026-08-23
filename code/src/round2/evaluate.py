"""Independent one-GPU evaluation of a selected round2 TP-LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict

import jsonlines
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import canonical_json
from ..data.dataset import PreferenceCollator, PreferenceDataset
from ..evaluation.metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_calibration,
    compute_confidence_distribution,
)
from ..model.model_manifest import verify_manifest
from .config import load_round2_config, validate_round2_config
from .queue_protocol import file_sha256
from .tp_trainer import DTYPES, _move_batch, _response_mean_logp


def _read_private_labels(path: Path, public_ids: list[str]) -> Dict[str, int]:
    labels: Dict[str, int] = {}
    with jsonlines.open(path) as reader:
        for row in reader:
            sample_id = row.get("sample_id")
            label = row.get("label")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("Private test labels contain a malformed sample_id")
            if sample_id in labels:
                raise ValueError(f"Duplicate private test-label ID: {sample_id}")
            if int(label) not in {0, 1}:
                raise ValueError(f"Private test label must be binary: {sample_id}")
            labels[sample_id] = int(label)
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("Public test inputs contain duplicate sample IDs")
    if set(labels) != set(public_ids):
        raise ValueError("Private test-label IDs do not exactly match public test inputs")
    return labels


def _verify_checkpoint(checkpoint: Path, config: Dict[str, Any]) -> None:
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "checkpoint_meta.json",
        "READY.json",
        "run_config.yaml",
    )
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Selected round2 adapter is incomplete: {name}")
    metadata = json.loads((checkpoint / "checkpoint_meta.json").read_text())
    ready = json.loads((checkpoint / "READY.json").read_text())
    expected_hash = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if metadata.get("config_sha256") != expected_hash:
        raise ValueError("Selected adapter was produced by a different resolved config")
    if ready.get("config_sha256") != expected_hash or ready.get("step") != metadata.get(
        "step"
    ):
        raise ValueError("Selected adapter READY metadata is inconsistent")
    if Path(metadata.get("base_model", "")).resolve() != Path(
        config["model"]["name_or_path"]
    ).resolve():
        raise ValueError("Selected adapter/base model mismatch")
    if Path(metadata.get("model_manifest", "")).resolve() != Path(
        config["model"]["manifest_path"]
    ).resolve():
        raise ValueError("Selected adapter/model manifest mismatch")
    if ready.get("adapter_sha256") != file_sha256(
        checkpoint / "adapter_model.safetensors"
    ):
        raise ValueError("Selected adapter checksum mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_round2_config(args.config)
    validate_round2_config(config)

    expected_visible = str(config["evaluation"]["gpu_id"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_visible:
        raise RuntimeError(
            "Evaluation CUDA_VISIBLE_DEVICES differs from resolved config: "
            f"actual={os.environ.get('CUDA_VISIBLE_DEVICES')}, expected={expected_visible}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Round2 evaluation requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    output_dir = Path(config["output"]["run_dir"]).resolve()
    evaluation_dir = output_dir / "evaluation"
    if evaluation_dir.exists():
        raise FileExistsError(f"Refuse to overwrite round2 evaluation: {evaluation_dir}")
    best = json.loads((output_dir / "best.json").read_text())
    checkpoint = Path(best["policy_checkpoint"]).resolve()
    _verify_checkpoint(checkpoint, config)

    model_path = Path(config["model"]["name_or_path"]).resolve()
    verify_manifest(model_path, Path(config["model"]["manifest_path"]).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=DTYPES[config["model"]["torch_dtype"]],
        attn_implementation=config["model"]["attention_implementation"],
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    policy = PeftModel.from_pretrained(
        base,
        checkpoint,
        is_trainable=False,
        local_files_only=True,
    ).to(device).eval()

    data_dir = Path(config["data"]["data_dir"]).resolve()
    dataset = PreferenceDataset(
        str(data_dir / "test_inputs.jsonl"),
        tokenizer,
        max_length=int(config["model"]["max_seq_len"]),
        require_labels=False,
        enable_thinking=False,
    )
    if len(dataset) != int(config["data"]["test_samples"]):
        raise ValueError(
            "Round2 public test count changed: "
            f"actual={len(dataset)}, expected={config['data']['test_samples']}"
        )
    public_ids = [row["sample_id"] for row in dataset.samples]
    labels_by_id = _read_private_labels(
        data_dir / "private_labels" / "test_labels.jsonl", public_ids
    )
    collator = PreferenceCollator(tokenizer.pad_token_id)
    beta = float(config["training"]["simpo_beta"])
    predictions = []
    labels = []
    dtype = DTYPES[config["model"]["torch_dtype"]]
    with torch.inference_mode():
        for index in range(len(dataset)):
            cpu_batch = collator([dataset[index]])
            batch = _move_batch(cpu_batch, device)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                score_a = _response_mean_logp(policy, batch, "a")
                score_b = _response_mean_logp(policy, batch, "b")
            probability = float(torch.sigmoid(beta * (score_a.float() - score_b.float())))
            if not math.isfinite(probability):
                raise FloatingPointError("Round2 evaluation produced a non-finite probability")
            predictions.append(probability)
            labels.append(labels_by_id[cpu_batch["sample_ids"][0]])

    prediction_tensor = torch.tensor(predictions)
    label_tensor = torch.tensor(labels)
    calibration = compute_calibration(prediction_tensor, label_tensor)
    metrics = {
        "method": config["method"]["name"],
        "git_commit": config["provenance"]["git_commit"],
        "config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "accuracy": compute_accuracy(prediction_tensor, label_tensor),
        "brier": compute_brier_score(prediction_tensor, label_tensor),
        "ece": calibration["ece"],
        "samples": len(labels),
        "score_type": config["evaluation"]["score_type"],
        "simpo_beta": beta,
        "selected_checkpoint": str(checkpoint),
        "selected_validation_accuracy": float(best["val_accuracy"]),
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        **compute_confidence_distribution(prediction_tensor),
    }
    evaluation_dir.mkdir(parents=True)
    (evaluation_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (evaluation_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with jsonlines.open(evaluation_dir / "predictions.private.jsonl", "w") as writer:
        for sample_id, probability, label in zip(public_ids, predictions, labels):
            writer.write(
                {"sample_id": sample_id, "probability": probability, "label": label}
            )
    (evaluation_dir / "complete.json").write_text(
        json.dumps({"status": "succeeded", "metrics": "metrics.json"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
