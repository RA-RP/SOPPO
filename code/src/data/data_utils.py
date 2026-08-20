"""
Data utility functions for auditing and statistics.
"""

import json
import jsonlines
from pathlib import Path
from typing import Dict, List
import numpy as np


def audit_position_randomization(data_path: str, private_labels_path: str) -> Dict:
    """
    Audit position randomization by comparing public data and private labels.

    Args:
        data_path: Path to public JSONL (unlabeled_train.jsonl or test_inputs.jsonl)
        private_labels_path: Path to private labels JSONL

    Returns:
        Audit report with position statistics
    """
    # Load public data
    public_samples = {}
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            public_samples[obj['sample_id']] = obj

    # Load private labels
    private_labels = {}
    with jsonlines.open(private_labels_path) as reader:
        for obj in reader:
            private_labels[obj['sample_id']] = obj

    # Check position randomization
    position_original = 0
    position_swapped = 0

    for sample_id, public_data in public_samples.items():
        if sample_id not in private_labels:
            print(f"WARNING: {sample_id} in public but not in private labels")
            continue

        private_data = private_labels[sample_id]

        # Check if positions match original
        response_a_is_chosen = (public_data['response_a'] == private_data['original_chosen'])

        if response_a_is_chosen:
            position_original += 1
        else:
            position_swapped += 1

    total = position_original + position_swapped
    ratio = position_swapped / total if total > 0 else 0

    audit = {
        'total_samples': total,
        'position_original': position_original,
        'position_swapped': position_swapped,
        'swap_ratio': ratio,
        'expected_ratio': 0.5,
        'deviation': abs(ratio - 0.5),
        'passes_check': abs(ratio - 0.5) < 0.05
    }

    return audit


def compute_data_statistics(data_path: str, tokenizer=None) -> Dict:
    """
    Compute statistics for a dataset.

    Args:
        data_path: Path to JSONL file
        tokenizer: Optional tokenizer for length statistics

    Returns:
        Statistics dictionary
    """
    samples = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            samples.append(obj)

    stats = {
        'total_samples': len(samples),
        'has_labels': 'label' in samples[0] if samples else False,
    }

    if tokenizer:
        # Compute token length statistics
        lengths_a = []
        lengths_b = []

        for sample in samples:
            prompt = sample['prompt']
            response_a = sample['response_a']
            response_b = sample['response_b']

            len_a = len(tokenizer(prompt + response_a)['input_ids'])
            len_b = len(tokenizer(prompt + response_b)['input_ids'])

            lengths_a.append(len_a)
            lengths_b.append(len_b)

        stats['length_stats'] = {
            'response_a': {
                'mean': np.mean(lengths_a),
                'std': np.std(lengths_a),
                'min': np.min(lengths_a),
                'max': np.max(lengths_a),
                'median': np.median(lengths_a)
            },
            'response_b': {
                'mean': np.mean(lengths_b),
                'std': np.std(lengths_b),
                'min': np.min(lengths_b),
                'max': np.max(lengths_b),
                'median': np.median(lengths_b)
            }
        }

    return stats


def verify_label_isolation(unlabeled_path: str, private_labels_dir: str) -> bool:
    """
    Verify that unlabeled dataset does not contain labels.

    Args:
        unlabeled_path: Path to unlabeled JSONL
        private_labels_dir: Directory containing private labels

    Returns:
        True if isolation is verified, False otherwise
    """
    # Check unlabeled file has no labels
    with jsonlines.open(unlabeled_path) as reader:
        for obj in reader:
            if 'label' in obj:
                print(f"ERROR: Label found in unlabeled data: {obj['sample_id']}")
                return False
            if 'original_chosen' in obj or 'original_rejected' in obj:
                print(f"ERROR: Original label info leaked in unlabeled data: {obj['sample_id']}")
                return False

    # Check private labels exist and are accessible
    private_labels_path = Path(private_labels_dir) / "unlabeled_labels.jsonl"
    if not private_labels_path.exists():
        print(f"ERROR: Private labels not found at {private_labels_path}")
        return False

    # Verify private labels contain labels
    with jsonlines.open(private_labels_path) as reader:
        for obj in reader:
            if 'label' not in obj:
                print(f"ERROR: Label missing in private labels: {obj['sample_id']}")
                return False

    print("Label isolation verified successfully")
    return True


def check_cross_split_leakage(labeled_path: str, unlabeled_path: str,
                              test_path: str) -> Dict:
    """
    Check for prompt leakage across splits.

    Args:
        labeled_path: Path to labeled JSONL
        unlabeled_path: Path to unlabeled JSONL
        test_path: Path to test JSONL

    Returns:
        Leakage report
    """
    import hashlib

    def load_prompt_hashes(path):
        hashes = set()
        with jsonlines.open(path) as reader:
            for obj in reader:
                prompt_hash = hashlib.sha256(obj['prompt'].encode()).hexdigest()
                hashes.add(prompt_hash)
        return hashes

    labeled_hashes = load_prompt_hashes(labeled_path)
    unlabeled_hashes = load_prompt_hashes(unlabeled_path)
    test_hashes = load_prompt_hashes(test_path)

    # Check intersections
    labeled_unlabeled = labeled_hashes & unlabeled_hashes
    labeled_test = labeled_hashes & test_hashes
    unlabeled_test = unlabeled_hashes & test_hashes

    report = {
        'labeled_size': len(labeled_hashes),
        'unlabeled_size': len(unlabeled_hashes),
        'test_size': len(test_hashes),
        'labeled_unlabeled_overlap': len(labeled_unlabeled),
        'labeled_test_overlap': len(labeled_test),
        'unlabeled_test_overlap': len(unlabeled_test),
        'no_leakage': (len(labeled_unlabeled) == 0 and
                      len(labeled_test) == 0 and
                      len(unlabeled_test) == 0)
    }

    return report
