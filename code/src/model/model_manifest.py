"""Create and verify the immutable local Qwen3 model manifest (server only)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


EXPECTED_REPO = "Qwen/Qwen3-4B"
EXPECTED_MODEL_TYPE = "qwen3"
EXPECTED_LAYERS = 36


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(model_dir: Path) -> Dict:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ValueError(f"Expected model_type={EXPECTED_MODEL_TYPE}, got {config.get('model_type')}")
    if int(config.get("num_hidden_layers", -1)) != EXPECTED_LAYERS:
        raise ValueError(
            f"Expected {EXPECTED_LAYERS} hidden layers, got {config.get('num_hidden_layers')}"
        )
    for name in ("config.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"Missing required model file: {name}")
    if not list(model_dir.glob("*.safetensors")):
        raise FileNotFoundError("No safetensors model weights found")

    files = {}
    for path in sorted(p for p in model_dir.iterdir() if p.is_file()):
        if path.name == "model_manifest.json":
            continue
        files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema_version": 1,
        "repo_id": EXPECTED_REPO,
        "model_type": EXPECTED_MODEL_TYPE,
        "num_hidden_layers": EXPECTED_LAYERS,
        "files": files,
    }


def verify_manifest(model_dir: Path, manifest_path: Path) -> None:
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recorded != build_manifest(model_dir):
        raise ValueError("Model manifest mismatch; refuse mutable or incomplete weights")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    manifest_path = model_dir / "model_manifest.json"
    if args.write:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(build_manifest(model_dir), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
        print(f"Wrote immutable model manifest: {manifest_path}")
    else:
        verify_manifest(model_dir, manifest_path)
        print(f"Verified model manifest: {manifest_path}")


if __name__ == "__main__":
    main()
