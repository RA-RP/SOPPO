"""Round3 deterministic dual-source preparation and capped tokenization.

This module is an execution-plane entrypoint. It must only be run on an
authorized server because it loads datasets and writes sample-level artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import jsonlines
import torch
from torch.utils.data import Dataset


SEED = 42
NAMESPACES = {
    "validation": "round3-paired-validation-v3",
    "test": "round3-paired-independent-test-v3",
    "paired_train": "round3-uf-paired-train-8k-v1",
    "unpaired_train": "round3-uc-unpaired-train-7k-v1",
    "ab_swap": "round3-paired-ab-swap-v1",
}
VIEW_COUNTS = {
    "paired_train_8k": 8000,
    "paired_train_1k": 1000,
    "unpaired_train_7k": 7000,
    "validation": 1000,
    "test": 1000,
}
TOKENIZATION_CONTRACT = "round3_qwen3_native_nonthinking_prompt1024_completion1024_v1"


def canonical_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Canonical text input must be a string")
    normalized = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("Canonical text must not be empty")
    return normalized


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_key(namespace: str, sample_id: str) -> str:
    return sha256_text(f"{namespace}\0{SEED}\0{sample_id}")


def _message(messages: Any, role: str) -> str:
    if not isinstance(messages, list):
        raise ValueError("Conversation field must be a message list")
    for item in messages:
        if isinstance(item, dict) and item.get("role") == role:
            return canonical_text(item.get("content"))
    raise ValueError(f"Conversation is missing a non-empty {role} message")


def _source_id(
    repo: str, revision: str, split: str, row: Dict[str, Any], index: int
) -> Tuple[str, str, Dict[str, Any]]:
    candidate = row.get("prompt_id") or row.get("id")
    prompt_id = str(candidate) if candidate is not None else ""
    source_id = f"{repo}:{revision}:{split}:{prompt_id}:{index}"
    sample_id = sha256_text("\0".join((repo, revision, split, prompt_id, str(index))))
    return source_id, sample_id, {
        "dataset_id": repo,
        "resolved_revision": revision,
        "split": split,
        "prompt_id": prompt_id,
        "source_row_index": int(index),
    }


def _paired_record(repo: str, revision: str, split: str, row: Dict[str, Any], index: int) -> Dict[str, Any]:
    if row.get("prompt_id") is None or not str(row.get("prompt_id")):
        raise ValueError("UltraFeedback row is missing prompt_id")
    prompt = canonical_text(row.get("prompt"))
    chosen_prompt = _message(row.get("chosen"), "user")
    rejected_prompt = _message(row.get("rejected"), "user")
    if chosen_prompt != prompt or rejected_prompt != prompt:
        raise ValueError("UltraFeedback chosen/rejected user content differs from prompt")
    chosen = _message(row.get("chosen"), "assistant")
    rejected = _message(row.get("rejected"), "assistant")
    source_id, sample_id, source_provenance = _source_id(repo, revision, split, row, index)
    swap = int(deterministic_key(NAMESPACES["ab_swap"], sample_id)[-1], 16) & 1
    if swap:
        response_a, response_b, label = rejected, chosen, 0
    else:
        response_a, response_b, label = chosen, rejected, 1
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "source_provenance": source_provenance,
        "canonical_prompt_sha256": sha256_text(prompt),
        "prompt": prompt,
        "response_a": response_a,
        "response_b": response_b,
        "label": label,
    }


def _unpaired_record(repo: str, revision: str, split: str, row: Dict[str, Any], index: int) -> Dict[str, Any]:
    prompt = canonical_text(row.get("prompt"))
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("UltraChat train_sft row requires at least user/assistant messages")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError("UltraChat messages[0:2] must be user then assistant")
    message_prompt = canonical_text(messages[0].get("content"))
    if message_prompt != prompt:
        raise ValueError("UltraChat messages[0] differs from prompt")
    response = canonical_text(messages[1].get("content"))
    source_id, sample_id, source_provenance = _source_id(repo, revision, split, row, index)
    return {
        "sample_id": sample_id,
        "source_id": source_id,
        "source_provenance": source_provenance,
        "canonical_prompt_sha256": sha256_text(prompt),
        "prompt": prompt,
        "response": response,
    }


def _deduplicate(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    by_prompt: Dict[str, Dict[str, Any]] = {}
    removed = 0
    for record in records:
        prompt_hash = record["canonical_prompt_sha256"]
        incumbent = by_prompt.get(prompt_hash)
        if incumbent is None or record["sample_id"] < incumbent["sample_id"]:
            removed += int(incumbent is not None)
            by_prompt[prompt_hash] = record
        else:
            removed += 1
    return list(by_prompt.values()), removed


def _select(records: Sequence[Dict[str, Any]], namespace: str, count: int) -> List[Dict[str, Any]]:
    ranked = sorted(records, key=lambda row: (deterministic_key(namespace, row["sample_id"]), row["sample_id"]))
    if len(ranked) < int(count):
        raise ValueError(f"Insufficient eligible rows for {namespace}: {len(ranked)} < {count}")
    return ranked[: int(count)]


def _public_pair(record: Dict[str, Any], include_label: bool) -> Dict[str, Any]:
    keys = ("sample_id", "prompt", "response_a", "response_b")
    output = {key: record[key] for key in keys}
    if include_label:
        output["label"] = int(record["label"])
    return output


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 data artifact: {path}")
    with jsonlines.open(path, "w") as writer:
        for row in rows:
            writer.write(row)


def _cache_file_manifest(dataset) -> List[Dict[str, Any]]:
    output = []
    for item in dataset.cache_files:
        path = Path(item["filename"]).resolve()
        output.append({"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return sorted(output, key=lambda row: row["path"])


def _snapshot_parquet_manifest(snapshot_path: str | Path) -> List[Dict[str, Any]]:
    root = Path(snapshot_path).resolve()
    files = []
    for path in sorted(root.rglob("*.parquet")):
        files.append(
            {
                "relative_path": str(path.relative_to(root)),
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    if not files:
        raise FileNotFoundError(f"Dataset snapshot has no source parquet files: {root}")
    return files


def prepare_round3_data(
    output_dir: Path,
    ultrafeedback_revision: str,
    ultrachat_revision: str,
) -> Dict[str, Any]:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    if output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite Round3 data directory: {output_dir}")
    for revision, name in (
        (ultrafeedback_revision, "ultrafeedback_revision"),
        (ultrachat_revision, "ultrachat_revision"),
    ):
        if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
            raise ValueError(f"{name} must be a full lowercase commit SHA")
    output_dir.mkdir(parents=True)
    uf_repo = "HuggingFaceH4/ultrafeedback_binarized"
    uc_repo = "HuggingFaceH4/ultrachat_200k"
    uf_snapshot = snapshot_download(
        repo_id=uf_repo,
        repo_type="dataset",
        revision=ultrafeedback_revision,
        allow_patterns=["*.parquet", "**/*.parquet"],
    )
    uc_snapshot = snapshot_download(
        repo_id=uc_repo,
        repo_type="dataset",
        revision=ultrachat_revision,
        allow_patterns=["*.parquet", "**/*.parquet"],
    )
    train_prefs = load_dataset(uf_repo, split="train_prefs", revision=ultrafeedback_revision)
    test_prefs = load_dataset(uf_repo, split="test_prefs", revision=ultrafeedback_revision)
    train_sft = load_dataset(uc_repo, split="train_sft", revision=ultrachat_revision)

    paired_test, test_duplicates = _deduplicate(
        _paired_record(uf_repo, ultrafeedback_revision, "test_prefs", dict(row), index)
        for index, row in enumerate(test_prefs)
    )
    validation = _select(paired_test, NAMESPACES["validation"], 1000)
    validation_prompts = {row["canonical_prompt_sha256"] for row in validation}
    test_candidates = [row for row in paired_test if row["canonical_prompt_sha256"] not in validation_prompts]
    independent_test = _select(test_candidates, NAMESPACES["test"], 1000)
    heldout_prompts = validation_prompts | {row["canonical_prompt_sha256"] for row in independent_test}

    paired_train_all, train_duplicates = _deduplicate(
        _paired_record(uf_repo, ultrafeedback_revision, "train_prefs", dict(row), index)
        for index, row in enumerate(train_prefs)
    )
    train_overlap_removed = sum(row["canonical_prompt_sha256"] in heldout_prompts for row in paired_train_all)
    paired_train_candidates = [row for row in paired_train_all if row["canonical_prompt_sha256"] not in heldout_prompts]
    paired_master = _select(paired_train_candidates, NAMESPACES["paired_train"], 8000)
    paired_limited = paired_master[:1000]
    all_paired_prompts = heldout_prompts | {row["canonical_prompt_sha256"] for row in paired_master}

    unpaired_all, unpaired_duplicates = _deduplicate(
        _unpaired_record(uc_repo, ultrachat_revision, "train_sft", dict(row), index)
        for index, row in enumerate(train_sft)
    )
    unpaired_overlap_removed = sum(row["canonical_prompt_sha256"] in all_paired_prompts for row in unpaired_all)
    unpaired_candidates = [row for row in unpaired_all if row["canonical_prompt_sha256"] not in all_paired_prompts]
    unpaired = _select(unpaired_candidates, NAMESPACES["unpaired_train"], 7000)

    _write_jsonl(output_dir / "paired_train_8k.jsonl", (_public_pair(row, True) for row in paired_master))
    _write_jsonl(output_dir / "paired_train_1k.jsonl", (_public_pair(row, True) for row in paired_limited))
    _write_jsonl(
        output_dir / "unpaired_train_7k.jsonl",
        ({key: row[key] for key in ("sample_id", "prompt", "response")} for row in unpaired),
    )
    _write_jsonl(output_dir / "validation_1k.jsonl", (_public_pair(row, True) for row in validation))
    _write_jsonl(output_dir / "test_1k.public.jsonl", (_public_pair(row, False) for row in independent_test))
    _write_jsonl(
        output_dir / "test_1k.private_labels.jsonl",
        ({"sample_id": row["sample_id"], "label": int(row["label"])} for row in independent_test),
    )
    _write_jsonl(
        output_dir / "source_manifest.jsonl",
        (
            {
                "view": view,
                **{
                    key: row[key]
                    for key in ("sample_id", "source_id", "canonical_prompt_sha256")
                },
                **row["source_provenance"],
            }
            for view, rows in (
                ("paired_train_8k", paired_master),
                ("paired_train_1k", paired_limited),
                ("unpaired_train_7k", unpaired),
                ("validation", validation),
                ("test", independent_test),
            )
            for row in rows
        ),
    )
    files = {}
    for path in sorted(output_dir.glob("*.jsonl")):
        files[path.name] = {"rows": sum(1 for _ in path.open(encoding="utf-8")), "bytes": path.stat().st_size, "sha256": file_sha256(path)}
    manifest = {
        "schema_version": "round3.data_manifest.v1",
        "seed": SEED,
        "namespaces": NAMESPACES,
        "repositories": {
            "ultrafeedback": {"repo": uf_repo, "revision": ultrafeedback_revision},
            "ultrachat": {"repo": uc_repo, "revision": ultrachat_revision},
        },
        "source_cache_files": {
            "train_prefs": _cache_file_manifest(train_prefs),
            "test_prefs": _cache_file_manifest(test_prefs),
            "train_sft": _cache_file_manifest(train_sft),
        },
        "source_parquet_files": {
            "ultrafeedback": _snapshot_parquet_manifest(uf_snapshot),
            "ultrachat": _snapshot_parquet_manifest(uc_snapshot),
        },
        "exclusions": {
            "test_duplicate_prompts": test_duplicates,
            "train_duplicate_prompts": train_duplicates,
            "train_heldout_prompt_overlap": train_overlap_removed,
            "unpaired_duplicate_prompts": unpaired_duplicates,
            "unpaired_paired_prompt_overlap": unpaired_overlap_removed,
        },
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


class Round3TextEncoder:
    """Qwen3-native prompt/response encoding with independent 1024-token caps."""

    def __init__(self, tokenizer, max_prompt_length: int = 1024, max_completion_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.max_completion_length = int(max_completion_length)
        if (self.max_prompt_length, self.max_completion_length) != (1024, 1024):
            raise ValueError("Round3 encoder requires prompt/completion caps of 1024")

    def encode(self, prompt: str, response: str) -> Dict[str, Any]:
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids_full = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids_full = self.tokenizer(
            response + (self.tokenizer.eos_token or ""), add_special_tokens=False
        )["input_ids"]
        if not response_ids_full:
            raise ValueError("Round3 response tokenization is empty")
        prompt_ids = prompt_ids_full[-self.max_prompt_length :]
        response_ids = response_ids_full[: self.max_completion_length]
        input_ids = prompt_ids + response_ids
        if len(input_ids) > 2048:
            raise ValueError("Round3 encoded sequence exceeds 2048")
        return {
            "input_ids": input_ids,
            "loss_mask": [0] * len(prompt_ids) + [1] * len(response_ids),
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "prompt_tokens_removed": len(prompt_ids_full) - len(prompt_ids),
            "response_tokens_removed": len(response_ids_full) - len(response_ids),
        }


class PairDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, require_labels: bool, reference_cache: Dict[str, Dict[str, float]] | None = None):
        self.path = Path(path).resolve()
        self.encoder = Round3TextEncoder(tokenizer)
        self.reference_cache = reference_cache or {}
        self.rows = []
        with jsonlines.open(self.path) as reader:
            for row in reader:
                forbidden = {"chosen", "rejected", "original_chosen", "original_rejected"} & set(row)
                if forbidden:
                    raise ValueError(f"Non-canonical preference fields in {self.path}: {sorted(forbidden)}")
                has_label = "label" in row
                if require_labels != has_label:
                    raise ValueError(f"Label presence mismatch for {row.get('sample_id')}")
                if has_label and int(row["label"]) not in {0, 1}:
                    raise ValueError("Pair label must be 0/1")
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Empty Round3 pair dataset: {self.path}")
        ids = [row["sample_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {self.path}")
        if self.reference_cache and set(ids) != set(self.reference_cache):
            raise ValueError("Reference cache IDs do not exactly match pair dataset")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        source = self.rows[index]
        output: Dict[str, Any] = {"sample_id": source["sample_id"]}
        for side in ("a", "b"):
            encoded = self.encoder.encode(source["prompt"], source[f"response_{side}"])
            for key, value in encoded.items():
                output[f"{key}_{side}"] = value
        if "label" in source:
            output["label"] = int(source["label"])
        if self.reference_cache:
            output.update(self.reference_cache[source["sample_id"]])
        return output


class SingleDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer):
        self.path = Path(path).resolve()
        self.encoder = Round3TextEncoder(tokenizer)
        with jsonlines.open(self.path) as reader:
            self.rows = list(reader)
        if not self.rows or len({row["sample_id"] for row in self.rows}) != len(self.rows):
            raise ValueError("Round3 single dataset is empty or has duplicate IDs")
        forbidden = {"label", "response_a", "response_b", "chosen", "rejected"}
        for row in self.rows:
            if forbidden & set(row):
                raise ValueError("Round3 unpaired single leaked pair/label fields")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        source = self.rows[index]
        encoded = self.encoder.encode(source["prompt"], source["response"])
        return {"sample_id": source["sample_id"], **encoded}


def _pad(rows: Sequence[Sequence[int]], value: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    return torch.tensor([list(row) + [value] * (width - len(row)) for row in rows], dtype=torch.long)


class PairCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        output: Dict[str, Any] = {"sample_ids": [row["sample_id"] for row in batch]}
        for side in ("a", "b"):
            ids = [row[f"input_ids_{side}"] for row in batch]
            output[f"input_ids_{side}"] = _pad(ids, self.pad_token_id)
            output[f"attention_mask_{side}"] = _pad([[1] * len(value) for value in ids], 0)
            output[f"loss_mask_{side}"] = _pad([row[f"loss_mask_{side}"] for row in batch], 0)
        if "label" in batch[0]:
            output["labels"] = torch.tensor([row["label"] for row in batch], dtype=torch.long)
        for key in ("ref_logp_a", "ref_logp_b"):
            if key in batch[0]:
                output[key] = torch.tensor([row[key] for row in batch], dtype=torch.float32)
        output["truncation"] = {
            key: sum(int(row[key]) for row in batch)
            for key in (
                "prompt_tokens_removed_a",
                "prompt_tokens_removed_b",
                "response_tokens_removed_a",
                "response_tokens_removed_b",
            )
        }
        return output


class SingleCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        ids = [row["input_ids"] for row in batch]
        return {
            "sample_ids": [row["sample_id"] for row in batch],
            "input_ids": _pad(ids, self.pad_token_id),
            "attention_mask": _pad([[1] * len(value) for value in ids], 0),
            "loss_mask": _pad([row["loss_mask"] for row in batch], 0),
            "truncation": {
                "prompt_tokens_removed": sum(int(row["prompt_tokens_removed"]) for row in batch),
                "response_tokens_removed": sum(int(row["response_tokens_removed"]) for row in batch),
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ultrafeedback-revision", required=True)
    parser.add_argument("--ultrachat-revision", required=True)
    args = parser.parse_args()
    manifest = prepare_round3_data(
        Path(args.output_dir).resolve(),
        args.ultrafeedback_revision,
        args.ultrachat_revision,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
