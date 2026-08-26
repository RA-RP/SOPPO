#!/usr/bin/env bash
# Run CPU/config/data-free tests in the authorized Round3 train environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
[[ -x "$ROUND3_TRAIN_PYTHON" ]] || { echo "ERROR: Round3 train environment is missing" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
"$ROUND3_TRAIN_PYTHON" -m pytest -q "$CODE_ROOT/tests/test_round3_contract.py"
"$ROUND3_TRAIN_PYTHON" -m pip check
"$ROUND3_ROLLOUT_PYTHON" -m pip check

