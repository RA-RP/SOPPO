"""Generate AlpacaEval 2.0 model outputs for a StaticPE checkpoint."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_REVISION = "b9352fbb8ce704292730cf54b3b1dceb2a808738"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AlpacaEval 2.0 outputs with Qwen3 non-thinking mode.")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--model_revision", default=DEFAULT_REVISION)
    parser.add_argument("--adapter_name_or_path", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--generator_name", default="qwen3-1.7b-staticpe-lambda0.1")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(tokenizer, instruction: str) -> str:
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_file).resolve()
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for target in [output_path, manifest_path]:
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {target}; pass --overwrite explicitly.")
    if args.batch_size <= 0 or args.max_prompt_length <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and token limits must be positive.")

    eval_dataset = load_dataset(
        "tatsu-lab/alpaca_eval",
        "alpaca_eval",
        split="eval",
        cache_dir=args.cache_dir,
    )
    instructions = list(eval_dataset["instruction"])

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
    )
    if args.adapter_name_or_path:
        model = PeftModel.from_pretrained(model, args.adapter_name_or_path)
    model.eval()

    outputs = []
    for start in tqdm(range(0, len(instructions), args.batch_size), desc="Generating AlpacaEval outputs"):
        batch_instructions = instructions[start : start + args.batch_size]
        prompts = [build_prompt(tokenizer, instruction) for instruction in batch_instructions]
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
        batch_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for instruction, output in zip(batch_instructions, batch_outputs):
            output = output.strip()
            if not output:
                raise RuntimeError(f"Generated an empty response for instruction: {instruction!r}")
            outputs.append({
                "instruction": instruction,
                "output": output,
                "generator": args.generator_name,
            })

    if len(outputs) != len(instructions):
        raise RuntimeError(f"Expected {len(instructions)} outputs, but generated {len(outputs)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as file:
        json.dump(outputs, file, ensure_ascii=False, indent=2)
    os.replace(temporary_output, output_path)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "tatsu-lab/alpaca_eval",
        "dataset_config": "alpaca_eval",
        "split": "eval",
        "num_outputs": len(outputs),
        "model_name_or_path": args.model_name_or_path,
        "model_revision": args.model_revision,
        "adapter_name_or_path": args.adapter_name_or_path,
        "generator_name": args.generator_name,
        "generation": {
            "do_sample": False,
            "max_prompt_length": args.max_prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "enable_thinking": False,
            "dtype": args.dtype,
        },
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"AlpacaEval outputs: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
