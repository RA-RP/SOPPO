#!/usr/bin/env bash
# Derive/revalidate the immutable 24k response_a anchor from public MVP data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

[[ -x "$ROUND2_TRAIN_PYTHON" ]] || {
    echo "ERROR: Round2 training environment is missing; run 00_setup_envs.sh first" >&2
    exit 1
}

DATA_DIR="${SOPPO_DATA_DIR:-$DATA_ROOT/ultrafeedback/mvp-v0.5-30k}"
SOURCE="$DATA_DIR/unlabeled_train.jsonl"
ANCHOR_DIR="${SOPPO_ROUND2_SFT_ANCHOR_DIR:-$DATA_ROOT/round2/mvp-v0.5-30k/sft_anchor_response_a}"
[[ -f "$SOURCE" ]] || {
    echo "ERROR: frozen MVP unlabeled data is missing: $SOURCE" >&2
    exit 1
}

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND2_TRAIN_PYTHON" -m src.round2.prepare_sft_anchor \
    --unlabeled "$SOURCE" \
    --output-dir "$ANCHOR_DIR" \
    --expected-rows 24000

echo "Use SOPPO_ROUND2_SFT_DATA_FILE=$ANCHOR_DIR/sft_anchor.jsonl"
