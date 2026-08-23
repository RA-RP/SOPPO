#!/usr/bin/env bash
# Round2 preflight validates GPU disjointness and immutable output paths.
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
CONFIG_PATH="$ROUND2_CONFIG_DIR/$CONFIG_NAME"
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: round2 config not found: $CONFIG_PATH" >&2
    exit 1
fi

if [[ ! -x "$ROUND2_PYTHON" ]]; then
    echo "ERROR: round2 Python is unavailable: $ROUND2_PYTHON" >&2
    exit 1
fi

TRAIN_GPU_IDS="$($ROUND2_PYTHON - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
import yaml
path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
print(config["megatron"]["gpu_ids"])
PY
)"
ROLLOUT_GPU_IDS="$($ROUND2_PYTHON - "$CONFIG_PATH" <<'PY'
from pathlib import Path
import sys
import yaml
path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
print(config["rollout"]["gpu_ids"])
PY
)"

IFS=',' read -r -a TRAIN_ARRAY <<< "$TRAIN_GPU_IDS"
IFS=',' read -r -a ROLLOUT_ARRAY <<< "$ROLLOUT_GPU_IDS"
for gpu in "${TRAIN_ARRAY[@]}"; do
    for rgpu in "${ROLLOUT_ARRAY[@]}"; do
        if [[ "$gpu" == "$rgpu" ]]; then
            echo "ERROR: training and rollout GPUs must be disjoint ($gpu)" >&2
            exit 1
        fi
    done
done

RUN_DIR="$ROUND2_RUN_ROOT"
if [[ -e "$RUN_DIR" ]]; then
    echo "ERROR: Refuse to reuse existing round2 run directory: $RUN_DIR" >&2
    exit 1
fi

echo "Round2 config: $CONFIG_PATH"
echo "Training GPUs: $TRAIN_GPU_IDS"
echo "Rollout GPUs:  $ROLLOUT_GPU_IDS"
echo "Run root:      $RUN_DIR"
