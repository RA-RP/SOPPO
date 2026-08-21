"""Qwen3-aware preference datasets with dynamic padding and response masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
from torch.utils.data import DataLoader, Dataset


TOKENIZATION_CONTRACT = "qwen3_chat_prompt_plus_separate_response_v1"


def load_reference_cache(path: Optional[str]) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Reference cache not found: {cache_path}")
    records: Dict[str, Dict[str, float]] = {}
    with jsonlines.open(cache_path) as reader:
        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in records:
                raise ValueError(f"Duplicate sample in reference cache: {sample_id}")
            records[sample_id] = {
                "ref_logp_a": float(row["ref_logp_a"]),
                "ref_logp_b": float(row["ref_logp_b"]),
            }
    return records


class PreferenceDataset(Dataset):
    """Preference pairs encoded with Qwen3's chat template."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        reference_cache_path: Optional[str] = None,
        require_labels: Optional[bool] = None,
        enable_thinking: bool = False,
        limit: Optional[int] = None,
    ):
        self.data_path = Path(data_path).resolve()
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_thinking = bool(enable_thinking)
        self.reference_cache = load_reference_cache(reference_cache_path)
        self.samples = []
        with jsonlines.open(self.data_path) as reader:
            for row in reader:
                if limit is not None and len(self.samples) >= limit:
                    break
                self.samples.append(row)
        if not self.samples:
            raise ValueError(f"Dataset is empty: {self.data_path}")
        seen_ids = set()
        for row in self.samples:
            required = ("sample_id", "prompt", "response_a", "response_b")
            if any(not isinstance(row.get(key), str) or not row[key] for key in required):
                raise ValueError(f"Malformed preference row in {self.data_path}: {row.get('sample_id')}")
            if row["sample_id"] in seen_ids:
                raise ValueError(f"Duplicate sample_id in {self.data_path}: {row['sample_id']}")
            seen_ids.add(row["sample_id"])
            has_label = "label" in row
            if require_labels is True and not has_label:
                raise ValueError(f"Labeled dataset sample has no label: {row.get('sample_id')}")
            if has_label and int(row["label"]) not in {0, 1}:
                raise ValueError(f"Preference label must be 0/1: {row['sample_id']}")
            if require_labels is False:
                forbidden = {"label", "original_chosen", "original_rejected"} & set(row)
                if forbidden:
                    raise ValueError(
                        f"Label isolation failure for {row.get('sample_id')}: {sorted(forbidden)}"
                    )
            if self.reference_cache and row["sample_id"] not in self.reference_cache:
                raise ValueError(f"Reference cache misses sample: {row['sample_id']}")

    def __len__(self) -> int:
        return len(self.samples)

    def _chat_ids(self, prompt: str, response: str) -> Dict[str, List[int]]:
        # Qwen3 inserts the disabled-thinking marker only when constructing a
        # generation prompt. Build the response sequence from that exact text so
        # the response boundary cannot silently diverge from the training input.
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        eos = self.tokenizer.eos_token or ""
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        # Tokenizing prompt_text and prompt_text+response independently is not
        # prefix-stable for Qwen3: its tokenizer may merge bytes across the
        # assistant boundary.  Encode the two regions independently and join
        # token IDs so the response-only loss boundary is exact by construction.
        response_ids = self.tokenizer(response + eos, add_special_tokens=False)["input_ids"]
        if not response_ids:
            raise ValueError("Response tokenization is empty")
        full_ids = prompt_ids + response_ids
        response_start = len(prompt_ids)
        if len(full_ids) > self.max_length:
            removed = len(full_ids) - self.max_length
            full_ids = full_ids[removed:]
            response_start = max(0, response_start - removed)
        loss_mask = [0] * response_start + [1] * (len(full_ids) - response_start)
        if sum(loss_mask[1:]) == 0:
            raise ValueError("Response was fully truncated; increase model.max_seq_len")
        return {"input_ids": full_ids, "loss_mask": loss_mask}

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        encoded_a = self._chat_ids(sample["prompt"], sample["response_a"])
        encoded_b = self._chat_ids(sample["prompt"], sample["response_b"])
        result = {
            "sample_id": sample["sample_id"],
            "input_ids_a": encoded_a["input_ids"],
            "loss_mask_a": encoded_a["loss_mask"],
            "input_ids_b": encoded_b["input_ids"],
            "loss_mask_b": encoded_b["loss_mask"],
        }
        if "label" in sample:
            result["label"] = int(sample["label"])
        if self.reference_cache:
            result.update(self.reference_cache[sample["sample_id"]])
        return result


class PreferenceCollator:
    """Right-pad each side to the longest sequence in the current microbatch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def _pad(self, rows: List[List[int]], value: int) -> torch.Tensor:
        width = max(len(row) for row in rows)
        return torch.tensor([row + [value] * (width - len(row)) for row in rows], dtype=torch.long)

    def __call__(self, batch: List[Dict]) -> Dict:
        result = {"sample_ids": [row["sample_id"] for row in batch]}
        for side in ("a", "b"):
            ids = [row[f"input_ids_{side}"] for row in batch]
            masks = [row[f"loss_mask_{side}"] for row in batch]
            result[f"input_ids_{side}"] = self._pad(ids, self.pad_token_id)
            result[f"attention_mask_{side}"] = self._pad([[1] * len(row) for row in ids], 0)
            result[f"loss_mask_{side}"] = self._pad(masks, 0)
        if "label" in batch[0]:
            result["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        if "ref_logp_a" in batch[0]:
            for key in ("ref_logp_a", "ref_logp_b"):
                result[key] = torch.tensor([row[key] for row in batch], dtype=torch.float32)
        return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    collate_fn,
    shuffle: bool = True,
    num_workers: int = 0,
    sampler=None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )


def data_file_sha256(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
