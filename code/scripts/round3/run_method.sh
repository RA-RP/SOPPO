#!/usr/bin/env bash
# Run one isolated Round3 method; dynamic methods own exactly two worker groups.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round3_env.sh"
round3_require_experiment_id
METHOD="${1:?usage: run_method.sh METHOD [strong_smoke|formal]}"
MODE="${2:-formal}"
CONFIG="$ROUND3_RUN_ROOT/resolved/$MODE/$METHOD.yaml"
[[ -f "$CONFIG" ]] || { echo "ERROR: resolved Round3 config missing: $CONFIG" >&2; exit 1; }
RUN_DIR="$(round3_resolved_value "$CONFIG" output.run_dir)"
ROLLOUT_ENABLED="$(round3_resolved_value "$CONFIG" rollout.enabled)"
CONTROL_DIR="$ROUND3_RUN_ROOT/control/$MODE/$METHOD"
STATUS="$CONTROL_DIR/status.json"
PREFLIGHT="$CONTROL_DIR/preflight.json"
LOG_DIR="$ROUND3_LOG_ROOT/$MODE/$METHOD"
[[ ! -e "$STATUS" ]] || { echo "ERROR: refuse to reuse Round3 method attempt: $STATUS" >&2; exit 1; }
mkdir -p "$CONTROL_DIR" "$LOG_DIR"
export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"

update_status() {
    local state="$1"
    local stage="$2"
    local code="${3:-0}"
    "$ROUND3_TRAIN_PYTHON" - "$STATUS" "$state" "$stage" "$code" "$METHOD" "$MODE" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
value = {
    "state": sys.argv[2], "stage": sys.argv[3], "exit_code": int(sys.argv[4]),
    "method_id": sys.argv[5], "execution_mode": sys.argv[6],
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
partial = path.with_suffix(".json.partial")
partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
partial.replace(path)
PY
}

worker_pids=()
worker_pgids=()
worker_starttimes=()
worker_group_is_ours() {
    local index="$1"
    local pid="${worker_pids[$index]}"
    local pgid="${worker_pgids[$index]}"
    local starttime="${worker_starttimes[$index]}"
    [[ -r "/proc/$pid/stat" ]] || return 1
    [[ "$(awk '{print $22}' "/proc/$pid/stat")" == "$starttime" ]] || return 1
    [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')" == "$pgid" ]] || return 1
}
cleanup_workers() {
    if (( ${#worker_pids[@]} == 0 )); then
        return 0
    fi
    mkdir -p "$RUN_DIR/rollouts"
    : > "$RUN_DIR/rollouts/STOP"
    for _ in $(seq 1 60); do
        local alive=0
        for index in "${!worker_pids[@]}"; do
            worker_group_is_ours "$index" && alive=1
        done
        (( alive == 0 )) && break
        sleep 1
    done
    for index in "${!worker_pids[@]}"; do
        if worker_group_is_ours "$index"; then
            kill -TERM -- "-${worker_pgids[$index]}" 2>/dev/null || true
        fi
    done
    for _ in $(seq 1 10); do
        local alive=0
        for index in "${!worker_pids[@]}"; do
            worker_group_is_ours "$index" && alive=1
        done
        (( alive == 0 )) && break
        sleep 1
    done
    for index in "${!worker_pids[@]}"; do
        if worker_group_is_ours "$index"; then
            kill -KILL -- "-${worker_pgids[$index]}" 2>/dev/null || true
        fi
    done
    for pid in "${worker_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup_workers EXIT
trap 'update_status stopped signal 143 || true; exit 143' INT TERM

update_status running preflight
preflight_args=(--config "$CONFIG" --repo-root "$SOPPO_ROOT" --output "$PREFLIGHT")
if [[ "$MODE" == "formal" ]]; then
    preflight_args+=(--global-storage-evidence "$ROUND3_RUN_ROOT/formal_storage_gate.json")
fi
CUDA_VISIBLE_DEVICES=0,1,2 "$ROUND3_TRAIN_PYTHON" -m src.round3.preflight "${preflight_args[@]}"

if [[ "$ROLLOUT_ENABLED" == "true" ]]; then
    update_status running rollout_startup
    for replica in 0 1; do
        gpu="$((replica + 1))"
        setsid env CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn \
            "$ROUND3_ROLLOUT_PYTHON" -m src.round3.rollout_worker \
            --config "$CONFIG" --replica-id "$replica" \
            > "$LOG_DIR/rollout_replica_${replica}.log" 2>&1 &
        pid=$!
        pgid="$(ps -o pgid= -p "$pid" | tr -d '[:space:]')"
        [[ "$pgid" == "$pid" ]] || { echo "ERROR: replica $replica lacks an isolated process group" >&2; exit 1; }
        starttime="$(awk '{print $22}' "/proc/$pid/stat")"
        worker_pids+=("$pid")
        worker_pgids+=("$pgid")
        worker_starttimes+=("$starttime")
        printf '%s\n' "$pid" > "$CONTROL_DIR/replica_${replica}.pid"
        printf '%s\n' "$pgid" > "$CONTROL_DIR/replica_${replica}.pgid"
        printf '%s\n' "$starttime" > "$CONTROL_DIR/replica_${replica}.starttime"
    done
    for _ in $(seq 1 600); do
        [[ -f "$RUN_DIR/rollouts/replica_0.ready.json" && -f "$RUN_DIR/rollouts/replica_1.ready.json" ]] && break
        for pid in "${worker_pids[@]}"; do
            kill -0 "$pid" 2>/dev/null || { echo "ERROR: Round3 rollout replica exited before ACK readiness" >&2; exit 1; }
        done
        sleep 1
    done
    [[ -f "$RUN_DIR/rollouts/replica_0.ready.json" && -f "$RUN_DIR/rollouts/replica_1.ready.json" ]] || {
        echo "ERROR: Round3 rollout replicas did not become ready" >&2
        exit 1
    }
fi

update_status running training
set +e
CUDA_VISIBLE_DEVICES=0 "$ROUND3_TRAIN_PYTHON" -m src.round3.trainer \
    --config "$CONFIG" > "$LOG_DIR/train.log" 2>&1
train_status=$?
set -e
if (( train_status != 0 )); then
    update_status failed training "$train_status"
    exit "$train_status"
fi
cleanup_workers
worker_pids=()
worker_pgids=()
worker_starttimes=()
update_status completed training 0
trap - EXIT INT TERM
