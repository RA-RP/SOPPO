from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SIDECAR_ROOT = Path(__file__).resolve().parents[1]
if str(SIDECAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIDECAR_ROOT))

from opd_sft_h1.trl_runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/opd_sft_h1/configs/trl_first_minimal.yaml")
    args = parser.parse_args()
    result = run_from_config(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
