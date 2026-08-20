"""
UltraFeedback dataset preparation with label hiding and position randomization.

Server-only script. DO NOT run locally.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import jsonlines
from datasets import load_dataset
from tqdm import tqdm


def prepare_ultrafeedback_dataset(
    output_dir: str,
    total_samples: int = 10000,
    labeled_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dataset_name: str = "openbmb/UltraFeedback",
    max_seq_len: int = 2048
):
    """
    Download and prepare UltraFeedback dataset with label hiding.

    Args:
        output_dir: Server directory to save processed data
        total_samples: Total samples to sample (default 10k)
        labeled_ratio: Ratio of labeled training data (default 0.1)
        test_ratio: Ratio of test data (default 0.1)
        seed: Random seed
        dataset_name: HuggingFace dataset name
        max_seq_len: Maximum sequence length for truncation check

    Output structure:
        <output_dir>/
        ├── labeled_train.jsonl     # 900 samples
        ├── labeled_val.jsonl       # 100 samples (from 1k labeled)
        ├── unlabeled_train.jsonl   # 8k samples (position randomized)
        ├── test_inputs.jsonl       # 1k samples (position randomized)
        ├── private_labels/
        │   ├── unlabeled_labels.jsonl
        │   └── test_labels.jsonl
        ├── manifest_private.json   # Server only
        └── manifest_public.json    # Can be returned to local
    """
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    private_labels_path = output_path / "private_labels"
    private_labels_path.mkdir(exist_ok=True)

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    # Step 1: Sample total_samples randomly
    print(f"Sampling {total_samples} samples...")
    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    sampled_indices = all_indices[:total_samples]
    sampled_data = dataset.select(sampled_indices)

    # Step 2: Deduplicate by prompt
    print("Deduplicating by prompt...")
    prompt_to_samples = defaultdict(list)
    for idx, sample in enumerate(tqdm(sampled_data)):
        prompt = sample['instruction']
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        prompt_to_samples[prompt_hash].append((idx, sample))

    # Keep first sample for each prompt
    deduplicated = []
    for prompt_hash, samples in prompt_to_samples.items():
        deduplicated.append(samples[0])

    print(f"After deduplication: {len(deduplicated)} samples")

    # Step 3: Split into labeled, unlabeled, test
    # Ensure no prompt appears in multiple splits
    random.shuffle(deduplicated)

    n_labeled = int(len(deduplicated) * labeled_ratio)
    n_test = int(len(deduplicated) * test_ratio)
    n_unlabeled = len(deduplicated) - n_labeled - n_test

    labeled_samples = deduplicated[:n_labeled]
    unlabeled_samples = deduplicated[n_labeled:n_labeled + n_unlabeled]
    test_samples = deduplicated[n_labeled + n_unlabeled:]

    # Further split labeled into train/val (90/10)
    n_labeled_val = int(len(labeled_samples) * 0.1)
    labeled_val = labeled_samples[:n_labeled_val]
    labeled_train = labeled_samples[n_labeled_val:]

    print(f"Split sizes - Labeled train: {len(labeled_train)}, "
          f"Labeled val: {len(labeled_val)}, "
          f"Unlabeled: {len(unlabeled_samples)}, "
          f"Test: {len(test_samples)}")

    # Step 4: Process and save data
    truncation_stats = {"labeled": 0, "unlabeled": 0, "test": 0}
    position_stats = {"unlabeled_original": 0, "unlabeled_swapped": 0,
                      "test_original": 0, "test_swapped": 0}

    def process_sample(idx_sample_tuple, hide_label=False, randomize_position=False):
        """Process a single sample."""
        idx, sample = idx_sample_tuple

        prompt = sample['instruction']
        completions = sample.get('completions', [])

        # Find chosen and rejected based on ratings or annotations
        # Assuming UltraFeedback format with scores
        if len(completions) < 2:
            return None

        # Sort by score descending
        sorted_completions = sorted(completions,
                                     key=lambda x: x.get('overall_score', 0),
                                     reverse=True)
        chosen = sorted_completions[0]['response']
        rejected = sorted_completions[-1]['response']

        # Check truncation (approximate, without tokenization)
        approx_len = len(prompt) + len(chosen) + len(rejected)
        is_truncated = approx_len > max_seq_len * 4  # rough char estimate

        if randomize_position:
            # Randomly swap chosen/rejected positions
            swap = random.random() < 0.5
            if swap:
                response_a, response_b = rejected, chosen
                true_label = 0  # response_b (position 1) is preferred
                position_key = "swapped"
            else:
                response_a, response_b = chosen, rejected
                true_label = 1  # response_a (position 0) is preferred
                position_key = "original"
        else:
            response_a, response_b = chosen, rejected
            true_label = 1
            position_key = None

        result = {
            "sample_id": f"uf_{idx}",
            "prompt": prompt,
            "response_a": response_a,
            "response_b": response_b,
            "is_truncated": is_truncated
        }

        if not hide_label:
            result["label"] = true_label

        label_info = {
            "sample_id": f"uf_{idx}",
            "label": true_label,
            "original_chosen": chosen,
            "original_rejected": rejected
        }

        return result, label_info, is_truncated, position_key

    # Process labeled training
    print("Processing labeled training set...")
    with jsonlines.open(output_path / "labeled_train.jsonl", "w") as f:
        for item in tqdm(labeled_train):
            processed = process_sample(item, hide_label=False, randomize_position=False)
            if processed:
                data, _, is_trunc, _ = processed
                f.write(data)
                if is_trunc:
                    truncation_stats["labeled"] += 1

    # Process labeled validation
    print("Processing labeled validation set...")
    with jsonlines.open(output_path / "labeled_val.jsonl", "w") as f:
        for item in tqdm(labeled_val):
            processed = process_sample(item, hide_label=False, randomize_position=False)
            if processed:
                data, _, is_trunc, _ = processed
                f.write(data)
                if is_trunc:
                    truncation_stats["labeled"] += 1

    # Process unlabeled (hide labels, randomize positions)
    print("Processing unlabeled set...")
    unlabeled_labels = []
    with jsonlines.open(output_path / "unlabeled_train.jsonl", "w") as f:
        for item in tqdm(unlabeled_samples):
            processed = process_sample(item, hide_label=True, randomize_position=True)
            if processed:
                data, label_info, is_trunc, pos_key = processed
                f.write(data)
                unlabeled_labels.append(label_info)
                if is_trunc:
                    truncation_stats["unlabeled"] += 1
                if pos_key:
                    position_stats[f"unlabeled_{pos_key}"] += 1

    # Save unlabeled private labels
    with jsonlines.open(private_labels_path / "unlabeled_labels.jsonl", "w") as f:
        for label in unlabeled_labels:
            f.write(label)

    # Process test (hide labels, randomize positions)
    print("Processing test set...")
    test_labels = []
    with jsonlines.open(output_path / "test_inputs.jsonl", "w") as f:
        for item in tqdm(test_samples):
            processed = process_sample(item, hide_label=True, randomize_position=True)
            if processed:
                data, label_info, is_trunc, pos_key = processed
                f.write(data)
                test_labels.append(label_info)
                if is_trunc:
                    truncation_stats["test"] += 1
                if pos_key:
                    position_stats[f"test_{pos_key}"] += 1

    # Save test private labels
    with jsonlines.open(private_labels_path / "test_labels.jsonl", "w") as f:
        for label in test_labels:
            f.write(label)

    # Step 5: Compute audit statistics
    print("Computing audit statistics...")

    # Position randomization check
    position_audit = {
        "unlabeled": {
            "total": len(unlabeled_samples),
            "original_position": position_stats["unlabeled_original"],
            "swapped_position": position_stats["unlabeled_swapped"],
            "ratio": position_stats["unlabeled_swapped"] / len(unlabeled_samples) if unlabeled_samples else 0
        },
        "test": {
            "total": len(test_samples),
            "original_position": position_stats["test_original"],
            "swapped_position": position_stats["test_swapped"],
            "ratio": position_stats["test_swapped"] / len(test_samples) if test_samples else 0
        }
    }

    # Truncation stats
    truncation_audit = {
        "labeled_truncated": truncation_stats["labeled"],
        "labeled_total": len(labeled_train) + len(labeled_val),
        "unlabeled_truncated": truncation_stats["unlabeled"],
        "unlabeled_total": len(unlabeled_samples),
        "test_truncated": truncation_stats["test"],
        "test_total": len(test_samples)
    }

    # Compute SHA-256 checksums
    checksums = {}
    for filename in ["labeled_train.jsonl", "labeled_val.jsonl",
                     "unlabeled_train.jsonl", "test_inputs.jsonl"]:
        filepath = output_path / filename
        with open(filepath, "rb") as f:
            checksums[filename] = hashlib.sha256(f.read()).hexdigest()

    # Save manifests
    manifest_private = {
        "dataset": dataset_name,
        "total_samples_requested": total_samples,
        "total_samples_after_dedup": len(deduplicated),
        "labeled_train": len(labeled_train),
        "labeled_val": len(labeled_val),
        "unlabeled_train": len(unlabeled_samples),
        "test": len(test_samples),
        "seed": seed,
        "position_audit": position_audit,
        "truncation_audit": truncation_audit,
        "checksums": checksums,
        "max_seq_len_check": max_seq_len
    }

    manifest_public = {
        "dataset": dataset_name,
        "labeled_train": len(labeled_train),
        "labeled_val": len(labeled_val),
        "unlabeled_train": len(unlabeled_samples),
        "test": len(test_samples),
        "position_randomization_ratio": {
            "unlabeled": position_audit["unlabeled"]["ratio"],
            "test": position_audit["test"]["ratio"]
        },
        "truncation_ratio": {
            "labeled": truncation_stats["labeled"] / (len(labeled_train) + len(labeled_val)),
            "unlabeled": truncation_stats["unlabeled"] / len(unlabeled_samples) if unlabeled_samples else 0,
            "test": truncation_stats["test"] / len(test_samples) if test_samples else 0
        },
        "checksums": checksums
    }

    with open(output_path / "manifest_private.json", "w") as f:
        json.dump(manifest_private, f, indent=2)

    with open(output_path / "manifest_public.json", "w") as f:
        json.dump(manifest_public, f, indent=2)

    print("\n=== Data Preparation Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Labeled train: {len(labeled_train)}")
    print(f"Labeled val: {len(labeled_val)}")
    print(f"Unlabeled: {len(unlabeled_samples)}")
    print(f"Test: {len(test_samples)}")
    print(f"\nPosition randomization (should be ~0.50):")
    print(f"  Unlabeled: {position_audit['unlabeled']['ratio']:.3f}")
    print(f"  Test: {position_audit['test']['ratio']:.3f}")
    print(f"\nTruncation ratio (should be <0.05):")
    print(f"  Labeled: {manifest_public['truncation_ratio']['labeled']:.3f}")
    print(f"  Unlabeled: {manifest_public['truncation_ratio']['unlabeled']:.3f}")
    print(f"  Test: {manifest_public['truncation_ratio']['test']:.3f}")

    # Audit checks
    warnings = []
    if abs(position_audit["unlabeled"]["ratio"] - 0.5) > 0.05:
        warnings.append(f"WARNING: Unlabeled position ratio {position_audit['unlabeled']['ratio']:.3f} deviates from 0.5")
    if abs(position_audit["test"]["ratio"] - 0.5) > 0.05:
        warnings.append(f"WARNING: Test position ratio {position_audit['test']['ratio']:.3f} deviates from 0.5")
    if manifest_public["truncation_ratio"]["labeled"] > 0.05:
        warnings.append(f"WARNING: Labeled truncation ratio {manifest_public['truncation_ratio']['labeled']:.3f} exceeds 0.05")

    if warnings:
        print("\n=== AUDIT WARNINGS ===")
        for w in warnings:
            print(w)
        return False

    print("\n=== All audit checks passed ===")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Server output directory")
    parser.add_argument("--total_samples", type=int, default=10000)
    parser.add_argument("--labeled_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_name", default="openbmb/UltraFeedback")
    parser.add_argument("--max_seq_len", type=int, default=2048)

    args = parser.parse_args()

    success = prepare_ultrafeedback_dataset(
        output_dir=args.output_dir,
        total_samples=args.total_samples,
        labeled_ratio=args.labeled_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        dataset_name=args.dataset_name,
        max_seq_len=args.max_seq_len
    )

    exit(0 if success else 1)
