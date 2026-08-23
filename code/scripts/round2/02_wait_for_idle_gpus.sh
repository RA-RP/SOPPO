#!/usr/bin/env bash
# Wait without preemption until all resolved Round2 GPUs are stably idle.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/round2_env.sh"

POLL_SECONDS="${SOPPO_ROUND2_GPU_POLL_SECONDS:-30}"
IDLE_CONFIRMATIONS="${SOPPO_ROUND2_GPU_IDLE_CONFIRMATIONS:-3}"
MAX_USED_MEMORY_MIB="${SOPPO_ROUND2_GPU_IDLE_MAX_USED_MIB:-1024}"
MAX_UTILIZATION_PERCENT="${SOPPO_ROUND2_GPU_IDLE_MAX_UTIL_PERCENT:-5}"
WAIT_TIMEOUT_SECONDS="${SOPPO_ROUND2_GPU_WAIT_TIMEOUT_SECONDS:-0}"
MIN_TOTAL_MEMORY_MIB=23552
STATUS_FILE="$ROUND2_RUN_ROOT/gpu_wait.json"

require_nonnegative_integer() {
    local name="$1"
    local value="$2"
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $name must be a non-negative integer; got: $value" >&2
        exit 1
    fi
}

require_positive_integer() {
    local name="$1"
    local value="$2"
    require_nonnegative_integer "$name" "$value"
    if (( value == 0 )); then
        echo "ERROR: $name must be greater than zero" >&2
        exit 1
    fi
}

trim_whitespace() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

require_positive_integer SOPPO_ROUND2_GPU_POLL_SECONDS "$POLL_SECONDS"
require_positive_integer SOPPO_ROUND2_GPU_IDLE_CONFIRMATIONS "$IDLE_CONFIRMATIONS"
require_nonnegative_integer SOPPO_ROUND2_GPU_IDLE_MAX_USED_MIB "$MAX_USED_MEMORY_MIB"
require_nonnegative_integer SOPPO_ROUND2_GPU_IDLE_MAX_UTIL_PERCENT "$MAX_UTILIZATION_PERCENT"
require_nonnegative_integer SOPPO_ROUND2_GPU_WAIT_TIMEOUT_SECONDS "$WAIT_TIMEOUT_SECONDS"
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "ERROR: nvidia-smi is required for the Round2 GPU wait gate" >&2
    exit 1
}
[[ -x "$ROUND2_TRAIN_PYTHON" ]] || {
    echo "ERROR: Round2 training Python is missing: $ROUND2_TRAIN_PYTHON" >&2
    exit 1
}

METHODS=(
    soppo_pe_sft_rollout_exp
    soppo_pe_rollout_only_exp
)
FIRST_RESOLVED="$ROUND2_RUN_ROOT/${METHODS[0]}/config.resolved.yaml"
SECOND_RESOLVED="$ROUND2_RUN_ROOT/${METHODS[1]}/config.resolved.yaml"
for resolved in "$FIRST_RESOLVED" "$SECOND_RESOLVED"; do
    [[ -f "$resolved" ]] || {
        echo "ERROR: resolved Round2 config is missing: $resolved" >&2
        exit 1
    }
done

TRAIN_GPU_IDS="$(round2_resolved_value "$FIRST_RESOLVED" tensor_parallel.gpu_ids)"
ROLLOUT_GPU_IDS="$(round2_resolved_value "$FIRST_RESOLVED" rollout.gpu_ids)"
SECOND_TRAIN_GPU_IDS="$(round2_resolved_value "$SECOND_RESOLVED" tensor_parallel.gpu_ids)"
SECOND_ROLLOUT_GPU_IDS="$(round2_resolved_value "$SECOND_RESOLVED" rollout.gpu_ids)"
if [[ "$TRAIN_GPU_IDS" != "$SECOND_TRAIN_GPU_IDS" \
    || "$ROLLOUT_GPU_IDS" != "$SECOND_ROLLOUT_GPU_IDS" ]]; then
    echo "ERROR: both Round2 methods must resolve to the same GPU assignment" >&2
    exit 1
fi
for gpu_list in "$TRAIN_GPU_IDS" "$ROLLOUT_GPU_IDS"; do
    if [[ ! "$gpu_list" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "ERROR: invalid resolved GPU list: $gpu_list" >&2
        exit 1
    fi
done

IFS=',' read -r -a TRAIN_GPU_ARRAY <<< "$TRAIN_GPU_IDS"
IFS=',' read -r -a ROLLOUT_GPU_ARRAY <<< "$ROLLOUT_GPU_IDS"
TARGET_GPU_IDS=("${TRAIN_GPU_ARRAY[@]}" "${ROLLOUT_GPU_ARRAY[@]}")
declare -A SEEN_GPU_IDS=()
for gpu_id in "${TARGET_GPU_IDS[@]}"; do
    if [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
        echo "ERROR: training and rollout GPU assignments overlap at GPU $gpu_id" >&2
        exit 1
    fi
    SEEN_GPU_IDS[$gpu_id]=1
done
if (( ${#TARGET_GPU_IDS[@]} != 3 )); then
    echo "ERROR: Round2 wait gate requires exactly three unique target GPUs" >&2
    exit 1
fi
TARGET_GPU_CSV="$(IFS=','; printf '%s' "${TARGET_GPU_IDS[*]}")"

EXPECTED_GIT_COMMIT="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
    echo "ERROR: Round2 GPU wait requires a clean SOPPO checkout" >&2
    exit 1
fi
RESOLVED_GIT_COMMIT="$(round2_resolved_value "$FIRST_RESOLVED" provenance.git_commit)"
if [[ "$EXPECTED_GIT_COMMIT" != "$RESOLVED_GIT_COMMIT" ]]; then
    echo "ERROR: checkout commit differs from the resolved Round2 commit" >&2
    exit 1
fi

STARTED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
START_EPOCH="$(date +%s)"
REQUIRED_IDLE_SECONDS=$(( POLL_SECONDS * IDLE_CONFIRMATIONS ))

write_status() {
    local state="$1"
    local consecutive_idle_checks="$2"
    local observed_idle_seconds="$3"
    local elapsed_seconds="$4"
    local reason="$5"
    local snapshot="$6"
    "$ROUND2_TRAIN_PYTHON" - \
        "$STATUS_FILE" "$state" "$STARTED_AT" "$EXPECTED_GIT_COMMIT" \
        "$TARGET_GPU_CSV" "$TRAIN_GPU_IDS" "$ROLLOUT_GPU_IDS" \
        "$POLL_SECONDS" "$IDLE_CONFIRMATIONS" "$REQUIRED_IDLE_SECONDS" \
        "$MAX_USED_MEMORY_MIB" "$MAX_UTILIZATION_PERCENT" \
        "$WAIT_TIMEOUT_SECONDS" "$consecutive_idle_checks" \
        "$observed_idle_seconds" "$elapsed_seconds" "$reason" "$snapshot" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
snapshot = []
for line in sys.argv[18].splitlines():
    if not line:
        continue
    gpu_id, name, total, used, utilization, pids = line.split("|", 5)
    snapshot.append(
        {
            "gpu_id": int(gpu_id),
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "utilization_percent": int(utilization),
            "compute_pids": [int(value) for value in pids.split(",") if value],
        }
    )
payload = {
    "state": sys.argv[2],
    "started_at": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": sys.argv[4],
    "target_gpu_ids": [int(value) for value in sys.argv[5].split(",")],
    "training_gpu_ids": sys.argv[6],
    "rollout_gpu_ids": sys.argv[7],
    "poll_seconds": int(sys.argv[8]),
    "required_idle_confirmations": int(sys.argv[9]),
    "required_idle_seconds": int(sys.argv[10]),
    "max_used_memory_mib": int(sys.argv[11]),
    "max_utilization_percent": int(sys.argv[12]),
    "timeout_seconds": int(sys.argv[13]),
    "consecutive_idle_checks": int(sys.argv[14]),
    "observed_idle_seconds": int(sys.argv[15]),
    "elapsed_seconds": int(sys.argv[16]),
    "reason": sys.argv[17],
    "snapshot": snapshot,
}
temporary = path.with_suffix(".json.partial")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
}

fail_wait_gate() {
    local reason="$1"
    local elapsed_seconds="$2"
    local snapshot="${3:-}"
    write_status failed 0 0 "$elapsed_seconds" "$reason" "$snapshot" || true
    echo "ERROR: $reason" >&2
    exit 1
}

echo "Waiting for exclusive Round2 GPUs: $TARGET_GPU_CSV"
echo "Idle contract: no compute PID, used memory <= ${MAX_USED_MEMORY_MIB} MiB, utilization <= ${MAX_UTILIZATION_PERCENT}%"
echo "Stability window: ${REQUIRED_IDLE_SECONDS}s (${IDLE_CONFIRMATIONS} x ${POLL_SECONDS}s); timeout: ${WAIT_TIMEOUT_SECONDS}s (0=infinite)"
echo "This gate only observes GPUs; it never sends a signal to another process."

consecutive_idle_checks=0
idle_since_epoch=0
snapshot=""
while true; do
    now_epoch="$(date +%s)"
    elapsed_seconds=$(( now_epoch - START_EPOCH ))
    actual_commit="$(git -C "$SOPPO_ROOT" rev-parse HEAD)"
    if [[ "$actual_commit" != "$EXPECTED_GIT_COMMIT" ]]; then
        fail_wait_gate "SOPPO commit changed while waiting for GPUs" "$elapsed_seconds" "$snapshot"
    fi
    if [[ -n "$(git -C "$SOPPO_ROOT" status --porcelain)" ]]; then
        fail_wait_gate "SOPPO checkout became dirty while waiting for GPUs" "$elapsed_seconds" "$snapshot"
    fi

    all_idle=true
    reasons=()
    snapshot=""
    for gpu_id in "${TARGET_GPU_IDS[@]}"; do
        if ! gpu_row="$(nvidia-smi -i "$gpu_id" \
            --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null)"; then
            fail_wait_gate "nvidia-smi could not query target GPU $gpu_id" "$elapsed_seconds" "$snapshot"
        fi
        IFS=',' read -r reported_id gpu_name total_mib used_mib utilization <<< "$gpu_row"
        reported_id="$(trim_whitespace "$reported_id")"
        gpu_name="$(trim_whitespace "$gpu_name")"
        total_mib="$(trim_whitespace "$total_mib")"
        used_mib="$(trim_whitespace "$used_mib")"
        utilization="$(trim_whitespace "$utilization")"
        if [[ ! "$reported_id" =~ ^[0-9]+$ \
            || ! "$total_mib" =~ ^[0-9]+$ \
            || ! "$used_mib" =~ ^[0-9]+$ \
            || ! "$utilization" =~ ^[0-9]+$ ]]; then
            fail_wait_gate "unexpected nvidia-smi metrics for GPU $gpu_id: $gpu_row" "$elapsed_seconds" "$snapshot"
        fi
        if [[ "$reported_id" != "$gpu_id" ]]; then
            fail_wait_gate "nvidia-smi returned GPU $reported_id while GPU $gpu_id was requested" "$elapsed_seconds" "$snapshot"
        fi
        if [[ "$gpu_name" != *"RTX 4090"* || total_mib -lt MIN_TOTAL_MEMORY_MIB ]]; then
            fail_wait_gate "GPU $gpu_id does not satisfy the RTX 4090 / 23 GiB hardware contract" "$elapsed_seconds" "$snapshot"
        fi

        if ! process_rows="$(nvidia-smi -i "$gpu_id" \
            --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
            fail_wait_gate "nvidia-smi could not query compute processes on GPU $gpu_id" "$elapsed_seconds" "$snapshot"
        fi
        active_pids="$(printf '%s\n' "$process_rows" \
            | awk '$1 ~ /^[0-9]+$/ {print $1}' \
            | paste -sd, -)"
        snapshot+="${gpu_id}|${gpu_name}|${total_mib}|${used_mib}|${utilization}|${active_pids}"$'\n'

        if [[ -n "$active_pids" ]]; then
            all_idle=false
            reasons+=("gpu${gpu_id}:compute_pid=${active_pids}")
        fi
        if (( used_mib > MAX_USED_MEMORY_MIB )); then
            all_idle=false
            reasons+=("gpu${gpu_id}:used=${used_mib}MiB")
        fi
        if (( utilization > MAX_UTILIZATION_PERCENT )); then
            all_idle=false
            reasons+=("gpu${gpu_id}:util=${utilization}%")
        fi
    done

    now_epoch="$(date +%s)"
    elapsed_seconds=$(( now_epoch - START_EPOCH ))
    if [[ "$all_idle" == true ]]; then
        if (( idle_since_epoch == 0 )); then
            idle_since_epoch="$now_epoch"
        fi
        consecutive_idle_checks=$(( consecutive_idle_checks + 1 ))
        observed_idle_seconds=$(( now_epoch - idle_since_epoch ))
        reason="all target GPUs currently satisfy the idle contract"
        if (( consecutive_idle_checks >= IDLE_CONFIRMATIONS \
            && observed_idle_seconds >= REQUIRED_IDLE_SECONDS )); then
            write_status ready "$consecutive_idle_checks" "$observed_idle_seconds" \
                "$elapsed_seconds" "$reason" "$snapshot"
            echo "GPU wait gate passed after ${elapsed_seconds}s; stable idle window=${observed_idle_seconds}s"
            exit 0
        fi
    else
        consecutive_idle_checks=0
        idle_since_epoch=0
        observed_idle_seconds=0
        reason="busy: $(IFS=';'; printf '%s' "${reasons[*]}")"
    fi

    write_status waiting "$consecutive_idle_checks" "$observed_idle_seconds" \
        "$elapsed_seconds" "$reason" "$snapshot"
    printf '[%s] waiting: %s; stable_idle=%ss/%ss\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$reason" \
        "$observed_idle_seconds" "$REQUIRED_IDLE_SECONDS"
    if (( WAIT_TIMEOUT_SECONDS > 0 && elapsed_seconds >= WAIT_TIMEOUT_SECONDS )); then
        fail_wait_gate "GPU wait timed out after ${elapsed_seconds}s" "$elapsed_seconds" "$snapshot"
    fi
    sleep "$POLL_SECONDS"
done
