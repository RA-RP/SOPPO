#!/usr/bin/env bash
# Run one round2 method with an explicit training -> rollout handoff.
# A rollout may be launched concurrently only when it targets an existing,
# immutable checkpoint supplied through SOPPO_ROUND2_ROLLOUT_CHECKPOINT.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

CONFIG_NAME="${1:-soppo_pe_sft_rollout_exp.yaml}"
CONFIG_PATH="$ROUND2_CONFIG_DIR/$CONFIG_NAME"
RUN_DIR="$ROUND2_RUN_ROOT/$CONFIG_NAME"
RESOLVED="$RUN_DIR/config.resolved.yaml"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

[[ -f "$RESOLVED" ]] || {
    echo "ERROR: resolved config is missing; run 01_resolve_config.sh first" >&2
    exit 1
}
[[ -x "$ROUND2_PYTHON" ]] || { echo "ERROR: Python missing: $ROUND2_PYTHON" >&2; exit 1; }
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

TRAIN_LOG="$LOG_DIR/megatron.log"
ROLLOUT_LOG="$LOG_DIR/rollout.log"
TRAIN_PID_FILE="$RUN_DIR/megatron.pid"
ROLLOUT_PID_FILE="$RUN_DIR/rollout.pid"
STATUS_FILE="$RUN_DIR/status.json"

if [[ -e "$TRAIN_PID_FILE" || -e "$ROLLOUT_PID_FILE" || -e "$STATUS_FILE" ]]; then
    echo "ERROR: Refuse to reuse round2 attempt: $RUN_DIR" >&2
    exit 1
fi

cat > "$STATUS_FILE" <<EOF
{
  "state": "training",
  "method_config": "$CONFIG_NAME",
  "backend": "megatron",
  "training_gpu_ids": "${SOPPO_ROUND2_TRAIN_GPU_IDS:-0,1}",
  "rollout_gpu_ids": "${SOPPO_ROUND2_ROLLOUT_GPU_IDS:-2}"
}
EOF

"$ROUND2_PYTHON" -m src.round2.run_megatron --config "$RESOLVED" > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
printf '%s\n' "$TRAIN_PID" > "$TRAIN_PID_FILE"

if [[ -n "${SOPPO_ROUND2_ROLLOUT_CHECKPOINT:-}" ]]; then
    CHECKPOINT="$SOPPO_ROUND2_ROLLOUT_CHECKPOINT"
    [[ -d "$CHECKPOINT" ]] || { echo "ERROR: rollout checkpoint is missing: $CHECKPOINT" >&2; kill "$TRAIN_PID" 2>/dev/null || true; exit 1; }
    echo "Starting rollout against an existing immutable checkpoint: $CHECKPOINT"
    "$ROUND2_PYTHON" -m src.round2.run_rollout \
        --config "$RESOLVED" \
        --override "rollout.policy_checkpoint=$CHECKPOINT" > "$ROLLOUT_LOG" 2>&1 &
    ROLLOUT_PID=$!
    printf '%s\n' "$ROLLOUT_PID" > "$ROLLOUT_PID_FILE"
fi

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
if (( TRAIN_STATUS != 0 )); then
    python3 - "$STATUS_FILE" "$TRAIN_STATUS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.update(state="failed", failed_stage="megatron", exit_code=int(sys.argv[2]))
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
    exit "$TRAIN_STATUS"
fi

if [[ -n "${ROLLOUT_PID:-}" ]]; then
    set +e
    wait "$ROLLOUT_PID"
    ROLLOUT_STATUS=$?
    set -e
    if (( ROLLOUT_STATUS != 0 )); then
        python3 - "$STATUS_FILE" "$ROLLOUT_STATUS" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.update(state="failed", failed_stage="rollout", exit_code=int(sys.argv[2]))
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
        exit "$ROLLOUT_STATUS"
    fi
else
    echo "Training completed; starting rollout from final checkpoint."
    "$ROUND2_PYTHON" -m src.round2.run_rollout --config "$RESOLVED" > "$ROLLOUT_LOG" 2>&1
fi

python3 - "$STATUS_FILE" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text())
data.update(state="completed", exit_code=0)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
rm -f "$TRAIN_PID_FILE" "$ROLLOUT_PID_FILE"
echo "Round2 method completed: $CONFIG_NAME"
