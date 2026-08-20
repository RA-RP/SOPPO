#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/root/LLM-output-density
RUN_ROOT=/root/autodl-tmp/cycle09_t1
LOG_ROOT="$RUN_ROOT/logs"
STATUS="$RUN_ROOT/T1_STATUS.txt"
LOG="$LOG_ROOT/formal.log"
PYTHON=/root/miniconda3/envs/density/bin/python
SCRIPT="$REPO/experiments/opd_sft_h1/scripts/cycle09_t1_bootstrap.py"

mkdir -p "$LOG_ROOT"

finish() {
    code=$?
    if [[ $code -eq 0 ]]; then
        state=COMPLETED
    else
        state=FAILED
    fi
    printf '%s\nexit_code=%s\nupdated_utc=%s\n' \
        "$state" "$code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"
    exit "$code"
}
trap finish EXIT

printf 'RUNNING\npid=%s\nstarted_utc=%s\n' \
    "$$" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS"

cd "$REPO"
"$PYTHON" "$SCRIPT" --mode all --svd-mode values 2>&1 | tee -a "$LOG"
