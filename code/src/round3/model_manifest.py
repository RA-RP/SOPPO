"""Create the immutable Qwen3-1.7B Round3 model manifest on the server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..model.model_manifest import sha256_file, verify_manifest


def build(model_dir: Path, resolved_revision: str) -> dict:
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Qwen3-1.7B config/tokenizer files are incomplete")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3":
        raise ValueError("Round3 model must have model_type=qwen3")
    if not isinstance(tokenizer_config.get("chat_template"), str) or not tokenizer_config[
        "chat_template"
    ].strip():
        raise ValueError("Round3 post-trained Qwen3 must provide a non-empty chat template")
    if not list(model_dir.glob("*.safetensors")):
        raise FileNotFoundError("Qwen3-1.7B safetensors weights are missing")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    files = {}
    for path in sorted(item for item in model_dir.iterdir() if item.is_file()):
        if path.name == "model_manifest.json":
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": "round3.model_manifest.v1",
        "repo_id": "Qwen/Qwen3-1.7B",
        "resolved_revision": resolved_revision,
        "model_type": "qwen3",
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "special_tokens": {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        },
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--resolved-revision")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    manifest = model_dir / "model_manifest.json"
    if args.write:
        if not args.resolved_revision:
            raise ValueError("Writing a Round3 model manifest requires --resolved-revision")
        if manifest.exists():
            raise FileExistsError(f"Refuse to overwrite Round3 model manifest: {manifest}")
        temporary = manifest.with_suffix(".json.partial")
        temporary.write_text(
            json.dumps(build(model_dir, args.resolved_revision), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest)
    verify_manifest(model_dir, manifest)
    print(f"Verified Round3 model manifest: {manifest}")


if __name__ == "__main__":
    main()
