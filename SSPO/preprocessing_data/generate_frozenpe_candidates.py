"""Generate the fixed Qwen3 candidate used by FrozenPE unlabeled rows.

This script is intentionally separate from training: candidate B is generated
once from a pinned initialization, written to JSON, and never refreshed during
optimization. Run it on the authorized server execution environment.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_REVISION = "b9352fbb8ce704292730cf54b3b1dceb2a808738"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed FrozenPE candidate pairs.")
    parser.add_argument("--input_file", default="./data/ultra_combined_fb0.1_ch0.1.json")
    parser.add_argument("--output_file", default="./data/ultra_combined_fb0.1_ch0.1_frozenpe.json")
    parser.add_argument("--dataset_info", default="./data/dataset_info.json")
    parser.add_argument("--dataset_name", default="ultra_combined_fb0.1_ch0.1_frozenpe")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=DEFAULT_REVISION)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--fail_on_drop",
        action="store_true",
        help="Fail instead of silently changing the candidate population when a generation is empty or duplicated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def should_swap(instruction: str, seed: int) -> bool:
    key = f"{seed}\0{instruction}".encode("utf-8")
    return hashlib.sha256(key).digest()[0] % 2 == 1


def build_prompt(tokenizer, instruction: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def model_load_kwargs(model_name_or_path: str, model_revision: str, cache_dir: str | None) -> dict:
    kwargs = {"cache_dir": cache_dir, "trust_remote_code": True}
    if not os.path.isdir(model_name_or_path):
        kwargs["revision"] = model_revision
    return kwargs


def validate_row(row: Dict[str, object], index: int) -> str:
    instruction = row.get("instruction")
    chosen = row.get("chosen")
    rejected = row.get("rejected")
    unlabeled = row.get("unlabeled")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"Row {index} has an empty or non-string instruction.")
    if not all(isinstance(value, str) for value in [chosen, rejected, unlabeled]):
        raise ValueError(f"Row {index} must contain string chosen/rejected/unlabeled fields.")

    is_labeled = bool(chosen.strip() and rejected.strip()) and not unlabeled.strip()
    is_unlabeled = bool(unlabeled.strip()) and not chosen.strip() and not rejected.strip()
    if not (is_labeled or is_unlabeled):
        raise ValueError(f"Row {index} is neither an exclusive labeled row nor an exclusive unlabeled row.")
    return "labeled" if is_labeled else "unlabeled"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    output_path = Path(args.output_file).resolve()
    dataset_info_path = Path(args.dataset_info).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")

    for target in [output_path, manifest_path]:
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite explicitly.")
    if args.batch_size <= 0 or args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and token limits must be positive.")

    with input_path.open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("FrozenPE input must be a JSON list.")

    row_types = [validate_row(row, index) for index, row in enumerate(rows)]
    unlabeled_indices = [index for index, row_type in enumerate(row_types) if row_type == "unlabeled"]
    if not unlabeled_indices or len(unlabeled_indices) == len(rows):
        raise ValueError("FrozenPE input must contain both labeled and unlabeled rows.")

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        **model_load_kwargs(args.model_name_or_path, args.model_revision, args.cache_dir),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map="auto",
        **model_load_kwargs(args.model_name_or_path, args.model_revision, args.cache_dir),
    )
    model.eval()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    generated_by_index: Dict[int, str] = {}
    for start in tqdm(range(0, len(unlabeled_indices), args.batch_size), desc="Generating fixed candidate B"):
        batch_indices = unlabeled_indices[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, rows[index]["instruction"]) for index in batch_indices]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for index, text in zip(batch_indices, texts):
            generated_by_index[index] = text.strip()

    output_rows: List[Dict[str, object]] = []
    dropped_empty = 0
    dropped_duplicate = 0
    swapped = 0
    for index, (row, row_type) in enumerate(zip(rows, row_types)):
        output_row = dict(row)
        if row_type == "labeled":
            output_row["unlabeled_b"] = ""
            output_row["candidate_a_source"] = ""
            output_row["candidate_b_source"] = ""
            output_rows.append(output_row)
            continue

        original = row["unlabeled"].strip()
        generated_text = generated_by_index[index]
        if not generated_text:
            dropped_empty += 1
            continue
        if generated_text == original:
            dropped_duplicate += 1
            continue

        if should_swap(row["instruction"], args.seed):
            output_row["unlabeled"] = generated_text
            output_row["unlabeled_b"] = original
            output_row["candidate_a_source"] = "qwen3_initial"
            output_row["candidate_b_source"] = "ultrachat_original"
            swapped += 1
        else:
            output_row["unlabeled"] = original
            output_row["unlabeled_b"] = generated_text
            output_row["candidate_a_source"] = "ultrachat_original"
            output_row["candidate_b_source"] = "qwen3_initial"
        output_rows.append(output_row)

    if args.fail_on_drop and (dropped_empty or dropped_duplicate):
        raise RuntimeError(
            "Candidate generation would change the frozen smoke population: "
            f"empty={dropped_empty}, duplicate={dropped_duplicate}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as file:
        json.dump(output_rows, file, ensure_ascii=False, indent=2)
    os.replace(temporary_output, output_path)

    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as file:
            dataset_info = json.load(file)
    else:
        dataset_info = {}
    dataset_info[args.dataset_name] = {
        "file_name": os.path.relpath(output_path, dataset_info_path.parent),
        "ranking": True,
        "columns": {
            "prompt": "instruction",
            "chosen": "chosen",
            "rejected": "rejected",
            "unlabeled": "unlabeled",
            "unlabeled_b": "unlabeled_b",
        },
    }
    temporary_dataset_info = dataset_info_path.with_suffix(dataset_info_path.suffix + ".tmp")
    with temporary_dataset_info.open("w", encoding="utf-8") as file:
        json.dump(dataset_info, file, ensure_ascii=False, indent=2)
    os.replace(temporary_dataset_info, dataset_info_path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "dataset_name": args.dataset_name,
        "model_name_or_path": args.model_name_or_path,
        "model_revision": args.model_revision,
        "generation": {
            "do_sample": False,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
            "seed": args.seed,
            "dtype": args.dtype,
        },
        "counts": {
            "input_total": len(rows),
            "input_labeled": row_types.count("labeled"),
            "input_unlabeled": row_types.count("unlabeled"),
            "output_total": len(output_rows),
            "output_unlabeled": len(unlabeled_indices) - dropped_empty - dropped_duplicate,
            "swapped_candidate_order": swapped,
            "dropped_empty_generation": dropped_empty,
            "dropped_exact_duplicate": dropped_duplicate,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))
    print(f"FrozenPE dataset: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
