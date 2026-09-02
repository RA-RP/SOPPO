#!/usr/bin/env python3
"""Resolve exactly one staged asset directory by its manifest repo_id."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--required-file", default=None)
    args = parser.parse_args()

    root = Path(args.search_root).expanduser().resolve(strict=True)
    matches: list[Path] = []
    for manifest_path in root.rglob("ROUND4_ASSET_MANIFEST.json"):
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if manifest.get("repo_id") == args.repo_id:
            matches.append(manifest_path.parent.resolve())
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen asset for {args.repo_id}, found {len(matches)} below {root}")
    target = matches[0]
    if args.required_file:
        required = target / args.required_file
        if not required.is_file() or required.stat().st_size == 0:
            raise FileNotFoundError(f"Frozen asset is missing {args.required_file}: {target}")
    print(target)


if __name__ == "__main__":
    main()
