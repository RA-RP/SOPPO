"""
PyTorch Dataset classes for preference learning.
"""

import jsonlines
from pathlib import Path
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset


class PreferenceDataset(Dataset):
    """Base dataset for preference pairs."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        """
        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load data
        self.samples = []
        with jsonlines.open(self.data_path) as reader:
            for obj in reader:
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt = sample['prompt']
        response_a = sample['response_a']
        response_b = sample['response_b']

        # Tokenize prompt + response_a
        text_a = prompt + response_a
        encoding_a = self.tokenizer(
            text_a,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        # Tokenize prompt + response_b
        text_b = prompt + response_b
        encoding_b = self.tokenizer(
            text_b,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        result = {
            'sample_id': sample['sample_id'],
            'input_ids_a': encoding_a['input_ids'].squeeze(0),
            'attention_mask_a': encoding_a['attention_mask'].squeeze(0),
            'input_ids_b': encoding_b['input_ids'].squeeze(0),
            'attention_mask_b': encoding_b['attention_mask'].squeeze(0),
        }

        # Add label if present
        if 'label' in sample:
            result['label'] = torch.tensor(sample['label'], dtype=torch.long)

        return result


class LabeledDataset(PreferenceDataset):
    """Dataset for labeled preference pairs (D_L)."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        super().__init__(data_path, tokenizer, max_length)

        # Verify all samples have labels
        for sample in self.samples:
            assert 'label' in sample, f"Sample {sample['sample_id']} missing label"


class UnlabeledDataset(PreferenceDataset):
    """Dataset for unlabeled preference pairs (D_U).

    Labels are hidden during training. Access to private labels is forbidden.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048,
                 verify_no_labels: bool = True):
        super().__init__(data_path, tokenizer, max_length)

        # Verify no labels are present
        if verify_no_labels:
            for sample in self.samples:
                assert 'label' not in sample, \
                    f"Unlabeled dataset should not contain labels: {sample['sample_id']}"


class TestDataset(PreferenceDataset):
    """Dataset for test set evaluation.

    Loads inputs without labels. True labels loaded separately by evaluation script.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        super().__init__(data_path, tokenizer, max_length)


def create_dataloader(dataset, batch_size: int, shuffle: bool = True,
                     num_workers: int = 0, collate_fn=None):
    """Create DataLoader with standard settings."""
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


def preference_collate_fn(batch: List[Dict]) -> Dict:
    """Collate function for preference datasets."""

    result = {
        'sample_ids': [item['sample_id'] for item in batch],
        'input_ids_a': torch.stack([item['input_ids_a'] for item in batch]),
        'attention_mask_a': torch.stack([item['attention_mask_a'] for item in batch]),
        'input_ids_b': torch.stack([item['input_ids_b'] for item in batch]),
        'attention_mask_b': torch.stack([item['attention_mask_b'] for item in batch]),
    }

    # Add labels if present
    if 'label' in batch[0]:
        result['labels'] = torch.stack([item['label'] for item in batch])

    return result
