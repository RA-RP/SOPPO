#!/usr/bin/env python3
"""Create and verify an immutable, Git-free source snapshot for one Slurm DAG."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def entries(root: Path) -> tuple[dict, dict]:
    files = {}
    symlinks = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks[relative] = os.readlink(path)
        elif path.is_file():
            mode = path.stat().st_mode
            files[relative] = {
                "sha256": sha256_file(path),
                "executable": bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)),
            }
    return files, symlinks


def require_snapshot_root(root: Path, manifest: Path) -> None:
    if not root.is_dir():
        raise SystemExit(f"Snapshot root is missing: {root}")
    if os.path.lexists(root / ".git"):
        raise SystemExit(f"Snapshot must not contain Git metadata: {root / '.git'}")
    try:
        manifest.relative_to(root)
    except ValueError:
        return
    raise SystemExit("Snapshot manifest must live outside the snapshotted source root")


def create(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    manifest = Path(args.manifest).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        raise SystemExit(f"Invalid source commit: {args.commit}")
    require_snapshot_root(root, manifest)
    if manifest.exists():
        raise SystemExit(f"Refuse to overwrite source manifest: {manifest}")
    files, symlinks = entries(root)
    if not files:
        raise SystemExit(f"Source snapshot is empty: {root}")
    payload = {
        "schema_version": 1,
        "git_commit": args.commit,
        "files": files,
        "symlinks": symlinks,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest)
    print(sha256_file(manifest))


def verify(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    manifest = Path(args.manifest).resolve()
    require_snapshot_root(root, manifest)
    if not manifest.is_file():
        raise SystemExit(f"Source manifest is missing: {manifest}")
    actual_manifest_sha256 = sha256_file(manifest)
    if actual_manifest_sha256 != args.manifest_sha256:
        raise SystemExit(
            "Source manifest checksum mismatch: "
            f"expected={args.manifest_sha256}, actual={actual_manifest_sha256}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise SystemExit("Unsupported source manifest schema")
    if payload.get("git_commit") != args.commit:
        raise SystemExit(
            "Source snapshot commit mismatch: "
            f"expected={args.commit}, manifest={payload.get('git_commit')}"
        )
    actual_files, actual_symlinks = entries(root)
    if payload.get("files") != actual_files or payload.get("symlinks") != actual_symlinks:
        expected_paths = set(payload.get("files", {})) | set(payload.get("symlinks", {}))
        actual_paths = set(actual_files) | set(actual_symlinks)
        missing = sorted(expected_paths - actual_paths)[:5]
        extra = sorted(actual_paths - expected_paths)[:5]
        changed = sorted(
            path
            for path in expected_paths & actual_paths
            if payload.get("files", {}).get(path) != actual_files.get(path)
            or payload.get("symlinks", {}).get(path) != actual_symlinks.get(path)
        )[:5]
        raise SystemExit(
            "Source snapshot content mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    print(f"Verified source snapshot: {root} @ {args.commit[:12]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--root", required=True)
    create_parser.add_argument("--manifest", required=True)
    create_parser.add_argument("--commit", required=True)
    create_parser.set_defaults(func=create)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", required=True)
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--manifest-sha256", required=True)
    verify_parser.add_argument("--commit", required=True)
    verify_parser.set_defaults(func=verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
