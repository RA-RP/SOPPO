"""Round2 configuration command-line helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import save_config
from .config import load_round2_config, validate_round2_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--write-resolved")
    args = parser.parse_args()

    config = load_round2_config(args.config, args.override)
    validate_round2_config(config)
    if args.write_resolved:
        output = Path(args.write_resolved).resolve()
        if output.exists():
            raise FileExistsError(f"Refuse to overwrite resolved config: {output}")
        save_config(config, output)
        print(f"Resolved config: {output}")
    else:
        print(json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
