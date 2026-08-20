#!/usr/bin/env python3
"""Download Cycle 09 block-2 Llama models from ModelScope.

The downloader uses the exact ModelScope repositories selected for this run,
keeps all large files on autodl-tmp, resumes partial downloads, and validates
that the result is a complete non-quantized safetensors checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/root/autodl-tmp/model/Meta/modelscope")
DEFAULT_MANIFEST = Path(
    "/root/autodl-tmp/cycle09_block2/modelscope_download_manifest.json"
)
MODEL_SPECS = {
    "base": {
        "model_id": "llm-research/llama-3.2-3b",
        "directory": "Llama-3.2-3B",
        "role": "G4 preflight and G6 student/base",
        "minimum_weight_gib": 5.5,
    },
    "teacher": {
        "model_id": "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        "directory": "Meta-Llama-3.1-8B-Instruct",
        "role": "G5 rollout teacher",
        "minimum_weight_gib": 14.0,
    },
}
ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors",
    "model.safetensors.index.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "LICENSE*",
    "README.md",
]
MINIMUM_FREE_GIB = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the two Cycle 09 Llama models from ModelScope."
    )
    parser.add_argument(
        "--models",
        default="base,teacher",
        help="Comma-separated roles: base,teacher (default: both, base first).",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional ModelScope revision; omitted uses the repository default.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--hash-weights",
        action="store_true",
        help="Also SHA256 every weight shard after download (slower).",
    )
    args = parser.parse_args()
    roles = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = sorted(set(roles) - set(MODEL_SPECS))
    if not roles or unknown:
        parser.error(f"invalid --models value; unknown roles: {unknown}")
    if args.max_workers < 1 or args.retries < 1:
        parser.error("--max-workers and --retries must be positive")
    args.models = roles
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def download_with_retries(
    snapshot_download: Any,
    *,
    model_id: str,
    local_dir: Path,
    revision: str | None,
    max_workers: int,
    retries: int,
) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            return snapshot_download(
                model_id=model_id,
                revision=revision,
                local_dir=str(local_dir),
                allow_patterns=ALLOW_PATTERNS,
                max_workers=max_workers,
            )
        except Exception:
            if attempt == retries:
                raise
            delay = 10 * attempt
            print(
                f"[download] {model_id} attempt {attempt}/{retries} failed; "
                f"retrying in {delay}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def expected_weight_shards(local_dir: Path) -> set[str]:
    index_path = local_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return {path.name for path in local_dir.glob("model*.safetensors")}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid safetensors index: {index_path}")
    return {str(name) for name in weight_map.values()}


def validate_local(
    local_dir: Path, *, minimum_weight_gib: float, hash_weights: bool
) -> dict[str, Any]:
    errors: list[str] = []
    config_path = local_dir / "config.json"
    if not config_path.is_file():
        return {"status": "fail", "errors": ["missing config.json"]}
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if "quantization_config" in config:
        errors.append("quantization_config present; full BF16/FP16 weights required")
    architectures = config.get("architectures", [])
    if architectures and "LlamaForCausalLM" not in architectures:
        errors.append(f"unexpected architecture: {architectures}")
    if config.get("model_type") != "llama":
        errors.append(f"unexpected model_type: {config.get('model_type')}")

    expected = expected_weight_shards(local_dir)
    missing = sorted(name for name in expected if not (local_dir / name).is_file())
    if missing:
        errors.append(f"missing {len(missing)} weight shards")
    weight_paths = sorted(local_dir / name for name in expected if (local_dir / name).is_file())
    weight_bytes = sum(path.stat().st_size for path in weight_paths)
    if weight_bytes < int(minimum_weight_gib * 2**30):
        errors.append(
            f"weight bytes {weight_bytes / 2**30:.2f} GiB below "
            f"minimum {minimum_weight_gib:.2f} GiB"
        )
    if not any((local_dir / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        errors.append("missing tokenizer.json/tokenizer.model")

    files = []
    for path in sorted(item for item in local_dir.rglob("*") if item.is_file()):
        record: dict[str, Any] = {
            "path": str(path.relative_to(local_dir)),
            "size": path.stat().st_size,
        }
        if path.name in {"config.json", "model.safetensors.index.json"}:
            record["sha256"] = sha256_file(path)
        elif hash_weights and path.suffix == ".safetensors":
            record["sha256"] = sha256_file(path)
        files.append(record)

    return {
        "status": "pass" if not errors else "fail",
        "model_type": config.get("model_type"),
        "architectures": architectures,
        "torch_dtype": config.get("torch_dtype"),
        "quantized": "quantization_config" in config,
        "expected_weight_shards": sorted(expected),
        "missing_weight_shards": missing,
        "weight_bytes": weight_bytes,
        "files": files,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MODELSCOPE_CACHE", "/root/autodl-tmp/modelscope_cache")

    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "ModelScope is missing. Install it in the density environment first."
        ) from error

    args.output_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_root).free
    required_free = int(MINIMUM_FREE_GIB * 2**30)
    if free_bytes < required_free:
        raise SystemExit(
            f"Insufficient disk: need at least {MINIMUM_FREE_GIB:.1f} GiB free, "
            f"found {free_bytes / 2**30:.1f} GiB"
        )

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "downloading",
        "source": "ModelScope",
        "started_at": utc_now(),
        "output_root": str(args.output_root),
        "free_bytes_before": free_bytes,
        "requested_revision": args.revision,
        "allow_patterns": ALLOW_PATTERNS,
        "models": {},
    }
    atomic_json(args.manifest, manifest)

    try:
        for role in args.models:
            spec = MODEL_SPECS[role]
            local_dir = args.output_root / spec["directory"]
            record: dict[str, Any] = {
                "role": spec["role"],
                "model_id": spec["model_id"],
                "revision": args.revision,
                "local_dir": str(local_dir),
                "status": "downloading",
                "started_at": utc_now(),
            }
            manifest["models"][role] = record
            atomic_json(args.manifest, manifest)
            print(
                f"[download] {role}: {spec['model_id']} -> {local_dir}", flush=True
            )
            resolved_path = download_with_retries(
                snapshot_download,
                model_id=spec["model_id"],
                local_dir=local_dir,
                revision=args.revision,
                max_workers=args.max_workers,
                retries=args.retries,
            )
            validation = validate_local(
                local_dir,
                minimum_weight_gib=float(spec["minimum_weight_gib"]),
                hash_weights=args.hash_weights,
            )
            record.update(
                {
                    "status": "complete" if validation["status"] == "pass" else "fail",
                    "resolved_path": str(resolved_path),
                    "completed_at": utc_now(),
                    "validation": validation,
                }
            )
            atomic_json(args.manifest, manifest)
            if validation["status"] != "pass":
                raise RuntimeError(
                    f"validation failed for {spec['model_id']}: "
                    + "; ".join(validation["errors"])
                )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error_type"] = type(error).__name__
        manifest["error"] = str(error)
        manifest["failed_at"] = utc_now()
        atomic_json(args.manifest, manifest)
        raise

    manifest["status"] = "complete"
    manifest["completed_at"] = utc_now()
    manifest["free_bytes_after"] = shutil.disk_usage(args.output_root).free
    atomic_json(args.manifest, manifest)
    print(f"[complete] manifest: {args.manifest}", flush=True)


if __name__ == "__main__":
    main()
