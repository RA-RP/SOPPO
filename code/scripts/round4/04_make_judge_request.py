#!/usr/bin/env python3
"""Create an aggregate-only A100→4090 judge request manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", required=True)
    parser.add_argument("--reference-outputs", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--outputs-filename", default="alpacaeval_smoke_outputs.json")
    args = parser.parse_args()

    export_root = Path(args.export_root).resolve(strict=True)
    reference = Path(args.reference_outputs).resolve(strict=True)
    if not args.code_commit or len(args.code_commit) != 40:
        raise ValueError("--code-commit must be a full SHA")
    methods: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        output = export_root / method / args.outputs_filename
        manifest = output.with_suffix(output.suffix + ".manifest.json")
        rows = read_json(output)
        generated = read_json(manifest)
        if not isinstance(rows, list) or generated.get("num_outputs") != len(rows):
            raise RuntimeError(f"invalid generated output contract for {method}")
        if generated.get("output_sha256") != sha256_file(output):
            raise RuntimeError(f"output checksum mismatch for {method}")
        methods[method] = {
            "relative_output_path": str(output.relative_to(export_root)),
            "output_sha256": sha256_file(output),
            "row_count": len(rows),
            "generation_manifest_sha256": sha256_file(manifest),
        }

    request = {
        "schema": "round4-judge-request-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "code_commit": args.code_commit,
        "reference_outputs_sha256": sha256_file(reference),
        "methods": methods,
    }
    destination = Path(args.output_file).resolve() if args.output_file else export_root / "JUDGE_REQUEST.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite judge request: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(destination)


if __name__ == "__main__":
    main()
