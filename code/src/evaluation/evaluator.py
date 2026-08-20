"""
Model evaluation on test set with label isolation.
"""

import torch
import jsonlines
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

from .metrics import (
    compute_accuracy,
    compute_brier_score,
    compute_calibration,
    compute_confidence_distribution
)


def evaluate_model(
    policy_model,
    reference_model,
    dataloader,
    private_labels_path: str,
    beta: float = 0.1,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on test set.

    Args:
        policy_model: Trained policy model
        reference_model: Reference model
        dataloader: DataLoader for test inputs
        private_labels_path: Path to private test labels
        beta: DPO temperature
        device: Device

    Returns:
        Evaluation metrics dictionary
    """
    policy_model.eval()
    reference_model.eval()

    # Load private labels
    private_labels = {}
    with jsonlines.open(private_labels_path) as reader:
        for obj in reader:
            private_labels[obj['sample_id']] = obj['label']

    all_predictions = []
    all_labels = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            sample_ids = batch['sample_ids']
            input_ids_a = batch['input_ids_a'].to(device)
            attention_mask_a = batch['attention_mask_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            attention_mask_b = batch['attention_mask_b'].to(device)

            # Compute preference probabilities
            from ..model.dpo_loss import compute_sequence_logprob

            policy_a_outputs = policy_model(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
                return_dict=True
            )
            policy_b_outputs = policy_model(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                return_dict=True
            )
            reference_a_outputs = reference_model(
                input_ids=input_ids_a,
                attention_mask=attention_mask_a,
                return_dict=True
            )
            reference_b_outputs = reference_model(
                input_ids=input_ids_b,
                attention_mask=attention_mask_b,
                return_dict=True
            )

            policy_a_logps = compute_sequence_logprob(
                policy_a_outputs.logits, input_ids_a, attention_mask_a
            )
            policy_b_logps = compute_sequence_logprob(
                policy_b_outputs.logits, input_ids_b, attention_mask_b
            )
            reference_a_logps = compute_sequence_logprob(
                reference_a_outputs.logits, input_ids_a, attention_mask_a
            )
            reference_b_logps = compute_sequence_logprob(
                reference_b_outputs.logits, input_ids_b, attention_mask_b
            )

            # Compute p_i = σ(r_θ(x, y_a) - r_θ(x, y_b))
            reward_a = beta * (policy_a_logps - reference_a_logps)
            reward_b = beta * (policy_b_logps - reference_b_logps)
            delta = reward_a - reward_b
            probs = torch.sigmoid(delta)

            # Match with private labels
            for i, sample_id in enumerate(sample_ids):
                if sample_id in private_labels:
                    all_predictions.append(probs[i].item())
                    all_labels.append(private_labels[sample_id])
                    all_sample_ids.append(sample_id)
                else:
                    print(f"WARNING: Sample {sample_id} not found in private labels")

    # Convert to tensors
    predictions = torch.tensor(all_predictions)
    labels = torch.tensor(all_labels)

    # Compute metrics
    accuracy = compute_accuracy(predictions, labels)
    brier = compute_brier_score(predictions, labels)
    calibration = compute_calibration(predictions, labels)
    confidence_dist = compute_confidence_distribution(predictions)

    metrics = {
        'accuracy': accuracy,
        'brier': brier,
        'ece': calibration['ece'],
        'total_samples': len(labels),
        **confidence_dist
    }

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE: {calibration['ece']:.4f}")
    print(f"Confidence >60%: {confidence_dist['confidence_50']:.4f}")
    print(f"Mean Entropy: {confidence_dist['mean_entropy']:.4f}")

    return metrics, {
        'predictions': all_predictions,
        'labels': all_labels,
        'sample_ids': all_sample_ids,
        'calibration': calibration
    }


def evaluate_and_save_predictions(
    policy_model,
    reference_model,
    dataloader,
    private_labels_path: str,
    output_dir: str,
    beta: float = 0.1,
    device: str = "cuda"
):
    """
    Evaluate model and save predictions to file.

    Args:
        policy_model: Trained policy model
        reference_model: Reference model
        dataloader: DataLoader for test inputs
        private_labels_path: Path to private test labels
        output_dir: Directory to save results
        beta: DPO temperature
        device: Device
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Evaluate
    metrics, details = evaluate_model(
        policy_model,
        reference_model,
        dataloader,
        private_labels_path,
        beta,
        device
    )

    # Save metrics
    import json
    with open(output_path / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions (SERVER ONLY - DO NOT RETURN TO LOCAL)
    with jsonlines.open(output_path / "test_predictions.jsonl", "w") as writer:
        for sample_id, pred, label in zip(
            details['sample_ids'],
            details['predictions'],
            details['labels']
        ):
            writer.write({
                'sample_id': sample_id,
                'prediction': pred,
                'label': label
            })

    # Save calibration data
    with open(output_path / "calibration.json", "w") as f:
        json.dump(details['calibration'], f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("- test_metrics.json (can be returned)")
    print("- test_predictions.jsonl (SERVER ONLY)")
    print("- calibration.json (can be returned)")
