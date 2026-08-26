#!/usr/bin/env bash
# Materialize the approved 8K/1K/7K/1K/997 dual-source views on the server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
UF_REVISION="$(round3_resolved_source_sha ultrafeedback)"
UC_REVISION="$(round3_resolved_source_sha ultrachat)"
[[ -x "$ROUND3_TRAIN_PYTHON" ]] || { echo "ERROR: Round3 train environment is missing" >&2; exit 1; }
[[ ! -e "$ROUND3_DATA_DIR" ]] || {
    echo "ERROR: Round3 data directory already exists; verify it rather than overwrite: $ROUND3_DATA_DIR" >&2
    exit 1
}
mkdir -p "$(dirname "$ROUND3_DATA_DIR")"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND3_TRAIN_PYTHON" -m src.round3.data \
    --output-dir "$ROUND3_DATA_DIR" \
    --ultrafeedback-revision "$UF_REVISION" \
    --ultrachat-revision "$UC_REVISION"
