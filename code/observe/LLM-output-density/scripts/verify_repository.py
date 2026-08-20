#!/usr/bin/env python3
"""Check the repository boundary and curated artifact manifest."""

from __future__ import annotations

import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 50 * 1024 * 1024
FORBIDDEN_DIRS = {"__pycache__", ".git", "local_experiment_results", "runs"}
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".gguf", ".safetensors"}
SECRET_PATTERNS = {
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "GitHub token": re.compile(rb"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in result.stdout.split(b"\0") if item]


def main() -> int:
    errors: list[str] = []
    files = repository_files()

    for path in files:
        relative = path.relative_to(ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"oversized file: {relative} ({path.stat().st_size} bytes)")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"model/checkpoint artifact: {relative}")
        if any(part in FORBIDDEN_DIRS for part in relative.parts):
            errors.append(f"forbidden generated directory: {relative}")

        if path.stat().st_size <= 8 * 1024 * 1024:
            data = path.read_bytes()
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(data):
                    errors.append(f"{label} pattern: {relative}")

    manifest_path = ROOT / "paper_artifacts" / "MANIFEST.csv"
    if not manifest_path.exists():
        errors.append("missing paper_artifacts/MANIFEST.csv")
    else:
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                artifact = ROOT / "paper_artifacts" / row["relative_path"]
                if not artifact.is_file():
                    errors.append(f"missing curated artifact: {row['relative_path']}")
                    continue
                actual_size = artifact.stat().st_size
                if actual_size != int(row["size_bytes"]):
                    errors.append(f"artifact size mismatch: {row['relative_path']}")
                if sha256(artifact) != row["sha256"]:
                    errors.append(f"artifact hash mismatch: {row['relative_path']}")

    if errors:
        print("Repository verification failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Repository verification passed ({len(files)} files checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
