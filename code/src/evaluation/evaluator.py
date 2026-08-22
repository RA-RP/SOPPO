"""Independent server-only evaluation of PEFT adapters with private labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonlines
import torch
import yaml

from ..config import validate_config
from ..data.dataset import PreferenceCollator, PreferenceDataset, create_dataloader
from ..model.dpo_loss import model_pair_logps, model_pair_mean_logps, preference_delta
from ..model.model_utils import DTYPES, load_adapter_for_inference, load_tokenizer
from ..training.trainer import DPO_METHODS, cache_for, verify_cache_contract
from .metrics import compute_accuracy, compute_brier_score, compute_calibration, compute_confidence_distribution


def checkpoint_method(checkpoint: Path) -> tuple[str, dict]:
    config_path = checkpoint / "run_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint run config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    return config["method"]["name"], config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--test-inputs", required=True)
    parser.add_argument("--private-labels", required=True)
    parser.add_argument("--reference-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite evaluation: {output}")

    method, run_config = checkpoint_method(checkpoint)
    test_inputs = Path(args.test_inputs).resolve()
    cache_file = None
    if method in DPO_METHODS:
        if not args.reference_cache:
            raise ValueError(f"{method} evaluation requires a reference cache")
        cache_file = cache_for(args.reference_cache, test_inputs)
        verify_cache_contract(
            cache_file,
            test_inputs,
            Path(args.model_manifest),
            args.max_length,
            False,
        )

    tokenizer = load_tokenizer(args.base_model)
    dataset = PreferenceDataset(
        str(test_inputs),
        tokenizer,
        max_length=args.max_length,
        reference_cache_path=str(cache_file) if cache_file else None,
        require_labels=False,
        enable_thinking=False,
    )
    loader = create_dataloader(
        dataset,
        args.batch_size,
        PreferenceCollator(tokenizer.pad_token_id),
        shuffle=False,
        num_workers=2,
    )
    private = {}
    with jsonlines.open(args.private_labels) as reader:
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in private:
                raise ValueError(f"Duplicate private test-label ID: {sample_id}")
            private[sample_id] = int(row["label"])
    public_id_list = [row["sample_id"] for row in dataset.samples]
    public_ids = set(public_id_list)
    if len(public_ids) != len(public_id_list):
        raise ValueError("Duplicate sample IDs in public test inputs")
    if set(private) != public_ids:
        raise ValueError("Private test-label IDs do not exactly match public test inputs")

    model = load_adapter_for_inference(
        str(checkpoint),
        args.base_model,
        args.model_manifest,
        args.dtype,
    ).cuda().eval()
    predictions = []
    labels = []
    sample_ids = []
    dtype = DTYPES[args.dtype]
    dpo_beta = float(run_config["training"]["dpo_beta"])
    simpo_beta = float(run_config["training"]["simpo_beta"])
    with torch.inference_mode():
        for cpu_batch in loader:
            batch = {
                key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in cpu_batch.items()
            }
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if method in DPO_METHODS:
                    policy_a, policy_b = model_pair_logps(model, batch)
                    delta = preference_delta(
                        policy_a,
                        policy_b,
                        batch["ref_logp_a"],
                        batch["ref_logp_b"],
                        dpo_beta,
                    )
                else:
                    mean_a, mean_b = model_pair_mean_logps(model, batch)
                    delta = simpo_beta * (mean_a - mean_b)
            probs = torch.sigmoid(delta.float()).cpu()
            if not torch.isfinite(probs).all():
                raise FloatingPointError("Non-finite test predictions")
            for sample_id, probability in zip(cpu_batch["sample_ids"], probs):
                sample_ids.append(sample_id)
                predictions.append(float(probability))
                labels.append(private[sample_id])

    prediction_tensor = torch.tensor(predictions)
    label_tensor = torch.tensor(labels)
    calibration = compute_calibration(prediction_tensor, label_tensor)
    metrics = {
        "method": method,
        "accuracy": compute_accuracy(prediction_tensor, label_tensor),
        "brier": compute_brier_score(prediction_tensor, label_tensor),
        "ece": calibration["ece"],
        "samples": len(labels),
        "score_type": "dpo_reference_delta" if method in DPO_METHODS else "simpo_mean_logp_delta",
        "dpo_beta": dpo_beta if method in DPO_METHODS else None,
        "simpo_beta": simpo_beta if method not in DPO_METHODS else None,
        **compute_confidence_distribution(prediction_tensor),
    }
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with jsonlines.open(output / "predictions.private.jsonl", "w") as writer:
        for sample_id, probability, label in zip(sample_ids, predictions, labels):
            writer.write({"sample_id": sample_id, "probability": probability, "label": label})
    (output / "complete.json").write_text(
        json.dumps({"status": "succeeded", "metrics": "metrics.json"}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
