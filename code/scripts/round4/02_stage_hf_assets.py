#!/usr/bin/env python3
"""Freeze and stage the public Hugging Face assets needed by Round4.

Run only on the authorized, networked staging server. Each asset is downloaded
at an immutable Hub commit and receives a payload manifest for later transfer
verification on the A100 host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download


MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "b9352fbb8ce704292730cf54b3b1dceb2a808738"
ULTRAFEEDBACK_ID = "HuggingFaceH4/ultrafeedback_binarized"
ULTRACHAT_ID = "HuggingFaceH4/ultrachat_200k"
MANIFEST_NAME = "ROUND4_ASSET_MANIFEST.json"
STATE_NAME = ".ROUND4_DOWNLOAD_STATE.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-base", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-target", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--index-output", required=True)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--ultrafeedback-revision", default="main")
    parser.add_argument("--ultrachat-revision", default="main")
    parser.add_argument("--max-workers", type=int, default=8)
    return parser.parse_args()


def canonical_below(path: str, base: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise ValueError(f"{label} must stay below server base") from error
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)


def payload_files(target: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    excluded = {MANIFEST_NAME, STATE_NAME}
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Transfer payload must not contain symlinks: {path}")
        if not path.is_file() or path.name in excluded:
            continue
        relative = path.relative_to(target).as_posix()
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if not records:
        raise RuntimeError(f"Asset has no payload files: {target}")
    return records


def verify_complete(target: Path, expected_repo: str, expected_sha: str) -> dict[str, Any] | None:
    manifest_path = target / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("repo_id") != expected_repo or manifest.get("resolved_revision") != expected_sha:
        raise RuntimeError(f"Existing manifest does not match requested source: {target}")
    expected = manifest.get("payload_files")
    if not isinstance(expected, list) or expected != payload_files(target):
        raise RuntimeError(f"Existing asset payload failed checksum verification: {target}")
    return manifest


def prepare_resumable_target(target: Path, repo_id: str, repo_type: str, resolved_sha: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    state_path = target / STATE_NAME
    state = {"repo_id": repo_id, "repo_type": repo_type, "resolved_revision": resolved_sha}
    existing_entries = [path for path in target.iterdir() if path.name not in {STATE_NAME, MANIFEST_NAME}]
    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as file:
            if json.load(file) != state:
                raise RuntimeError(f"Partial target belongs to a different source: {target}")
    elif existing_entries:
        raise RuntimeError(f"Refusing untracked non-empty asset target: {target}")
    else:
        atomic_json(state_path, state)


def stage_asset(
    *,
    api: HfApi,
    repo_id: str,
    repo_type: str,
    requested_revision: str,
    target: Path,
    cache_dir: Path,
    max_workers: int,
) -> dict[str, Any]:
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=requested_revision)
    resolved_sha = info.sha
    complete = verify_complete(target, repo_id, resolved_sha) if target.exists() else None
    if complete is not None:
        print(f"Verified existing {repo_type} asset: {repo_id}@{resolved_sha}")
        return complete

    prepare_resumable_target(target, repo_id, repo_type, resolved_sha)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=resolved_sha,
        cache_dir=str(cache_dir),
        local_dir=str(target),
        local_dir_use_symlinks=False,
        max_workers=max_workers,
    )

    # Hub local-dir metadata is useful only while downloading and may contain
    # staging-machine cache details. The immutable payload is self-manifested.
    hub_metadata = target / ".cache" / "huggingface"
    if hub_metadata.exists():
        shutil.rmtree(hub_metadata)
    cache_parent = target / ".cache"
    if cache_parent.exists() and not any(cache_parent.iterdir()):
        cache_parent.rmdir()

    manifest = {
        "schema": "round4-hf-asset-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_id": repo_id,
        "repo_type": repo_type,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_sha,
        "payload_files": payload_files(target),
    }
    manifest["payload_file_count"] = len(manifest["payload_files"])
    manifest["payload_bytes"] = sum(record["bytes"] for record in manifest["payload_files"])
    atomic_json(target / MANIFEST_NAME, manifest)
    (target / STATE_NAME).unlink(missing_ok=True)
    verify_complete(target, repo_id, resolved_sha)
    print(f"Staged {repo_type} asset: {repo_id}@{resolved_sha}")
    return manifest


def locked_revisions(index_output: Path) -> dict[tuple[str, str], str]:
    if not index_output.is_file():
        return {}
    with index_output.open("r", encoding="utf-8") as file:
        index = json.load(file)
    locked: dict[tuple[str, str], str] = {}
    for asset in index.get("assets", []):
        locked[(asset["repo_type"], asset["repo_id"])] = asset["resolved_revision"]
    return locked


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")

    server_base = Path(args.server_base).expanduser().resolve(strict=True)
    data_root = canonical_below(args.data_root, server_base, "data root")
    model_target = canonical_below(args.model_target, server_base, "model target")
    cache_dir = canonical_below(args.cache_dir, server_base, "cache directory")
    index_output = canonical_below(args.index_output, server_base, "index output")
    data_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    prior_locks = locked_revisions(index_output)
    requested = [
        ("model", MODEL_ID, prior_locks.get(("model", MODEL_ID), args.model_revision), model_target),
        (
            "dataset",
            ULTRAFEEDBACK_ID,
            prior_locks.get(("dataset", ULTRAFEEDBACK_ID), args.ultrafeedback_revision),
            None,
        ),
        (
            "dataset",
            ULTRACHAT_ID,
            prior_locks.get(("dataset", ULTRACHAT_ID), args.ultrachat_revision),
            None,
        ),
    ]

    api = HfApi()
    manifests: list[dict[str, Any]] = []
    asset_entries: list[dict[str, Any]] = []
    for repo_type, repo_id, revision, fixed_target in requested:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        resolved_sha = info.sha
        target = fixed_target or data_root / f"{repo_id.rsplit('/', 1)[-1]}-{resolved_sha[:12]}"
        manifest = stage_asset(
            api=api,
            repo_id=repo_id,
            repo_type=repo_type,
            requested_revision=resolved_sha,
            target=target,
            cache_dir=cache_dir,
            max_workers=args.max_workers,
        )
        manifests.append(manifest)
        asset_entries.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "resolved_revision": manifest["resolved_revision"],
                "target_relative_to_server_base": target.relative_to(server_base).as_posix(),
                "manifest_sha256": sha256_file(target / MANIFEST_NAME),
                "payload_file_count": manifest["payload_file_count"],
                "payload_bytes": manifest["payload_bytes"],
            }
        )

    index = {
        "schema": "round4-hf-asset-index-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets": asset_entries,
        "total_payload_bytes": sum(item["payload_bytes"] for item in manifests),
    }
    atomic_json(index_output, index)
    print(f"Asset index: {index_output}")


if __name__ == "__main__":
    main()
