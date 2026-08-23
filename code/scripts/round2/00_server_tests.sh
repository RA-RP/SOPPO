#!/usr/bin/env bash
# Run the full regression suite in the isolated round2 training environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

[[ -x "$ROUND2_TRAIN_PYTHON" ]] || {
    echo "ERROR: Round2 training Python is missing: $ROUND2_TRAIN_PYTHON" >&2
    exit 1
}
TEST_DIR="$ROUND2_RUN_ROOT/server_tests"
[[ ! -e "$TEST_DIR" ]] || {
    echo "ERROR: Refuse to overwrite round2 test evidence: $TEST_DIR" >&2
    exit 1
}
mkdir -p "$TEST_DIR"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

"$ROUND2_TRAIN_PYTHON" -m pytest "$CODE_ROOT/tests" -vv \
    > "$TEST_DIR/pytest.log" 2>&1
GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
"$ROUND2_TRAIN_PYTHON" - "$TEST_DIR/complete.json" "$GIT_COMMIT" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "succeeded",
            "suite": "code/tests",
            "git_commit": sys.argv[2],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
PY

echo "Round2 server tests passed: $TEST_DIR"
