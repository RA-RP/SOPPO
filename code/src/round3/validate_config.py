"""Validate and optionally materialize one fully resolved Round3 config."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .config import load_round3_config, validate_round3_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--write-resolved")
    args = parser.parse_args()
    config = load_round3_config(args.config, args.override)
    validate_round3_config(config)
    if args.write_resolved:
        output = Path(args.write_resolved).resolve()
        if output.exists():
            raise FileExistsError(f"Refuse to overwrite resolved Round3 config: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            yaml.safe_dump(config, sort_keys=True, allow_unicode=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()

