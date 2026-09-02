#!/usr/bin/env python3
"""Materialize a verified local-dir asset as an offline Hugging Face Hub snapshot.

AlpacaEval 0.6.2 internally calls ``hf_hub_download`` for two LC metric CSVs.
This bridge exposes the already transferred, immutable dataset snapshot through
the Hub cache layout without making a network request or duplicating large
files when source and cache share a filesystem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(snapshot: Path, records: list[dict]) -> None:
    expected = {record["path"] for record in records}
    observed = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*") if path.is_file()}
    if observed != expected:
        raise RuntimeError("Offline Hub snapshot file set does not match the frozen asset manifest")
    for record in records:
        path = snapshot / record["path"]
        if path.is_symlink() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"Offline Hub snapshot failed verification: {record['path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--hf-home", required=True)
    parser.add_argument("--expected-repo-id", required=True)
    args = parser.parse_args()

    asset = Path(args.asset_dir).expanduser().resolve(strict=True)
    hf_home = Path(args.hf_home).expanduser().resolve()
    manifest = json.loads((asset / "ROUND4_ASSET_MANIFEST.json").read_text(encoding="utf-8"))
    if manifest.get("repo_id") != args.expected_repo_id or manifest.get("repo_type") != "dataset":
        raise RuntimeError("Frozen asset identity does not match the requested dataset")
    revision = manifest.get("resolved_revision")
    if not isinstance(revision, str) or len(revision) != 40:
        raise RuntimeError("Frozen asset has no full resolved revision")
    records = manifest.get("payload_files")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Frozen asset has no payload records")
    for record in records:
        source = asset / record["path"]
        if source.is_symlink() or source.stat().st_size != record["bytes"] or sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"Frozen source asset failed verification: {record['path']}")

    repo_cache_name = f"datasets--{args.expected_repo_id.replace('/', '--')}"
    repo_cache = hf_home / "hub" / repo_cache_name
    snapshot = repo_cache / "snapshots" / revision
    if snapshot.exists():
        verify_snapshot(snapshot, records)
    else:
        partial = snapshot.with_name(f".{revision}.partial")
        if partial.exists():
            raise FileExistsError(f"Refusing to reuse partial Hub cache snapshot: {partial}")
        partial.mkdir(parents=True)
        for record in records:
            source = asset / record["path"]
            target = partial / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        verify_snapshot(partial, records)
        os.replace(partial, snapshot)

    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    main_ref = refs / "main"
    temporary = refs / ".main.tmp"
    temporary.write_text(revision + "\n", encoding="utf-8")
    os.replace(temporary, main_ref)
    if main_ref.read_text(encoding="utf-8").strip() != revision:
        raise RuntimeError("Failed to bind the offline Hub main ref")
    print(snapshot)


if __name__ == "__main__":
    main()
