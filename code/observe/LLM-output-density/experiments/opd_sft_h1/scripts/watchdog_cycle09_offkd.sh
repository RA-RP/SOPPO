#!/usr/bin/env bash
# Detached serial runner for formal pass2 -> validation -> smoke -> off-KD train.
# It never retries a failed stage and intentionally does not power off the host.

set -u
set -o pipefail

SCRIPT_DIR="/root/LLM-output-density/experiments/opd_sft_h1/scripts"
EXP_ROOT="/root/autodl-tmp/cycle09_offkd"
ROLLOUT_DIR="${EXP_ROOT}/rollout"
LOG_DIR="${EXP_ROOT}/logs"
STATUS="${EXP_ROOT}/watchdog_status.json"
MASTER_LOG="${LOG_DIR}/watchdog.log"
ROLLOUT_PY="/root/miniconda3/envs/density/bin/python"
TRAIN_PY="/root/autodl-tmp/envs/verl/bin/python"
ROLLOUT_SCRIPT="${SCRIPT_DIR}/cycle09_offkd_rollout.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/cycle09_offkd_train.py"
ACTIVE_PID=""
ACTIVE_STAGE="initializing"

mkdir -p "${LOG_DIR}"
touch "${MASTER_LOG}"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_status() {
  local state="$1"
  local detail="${2:-}"
  printf '{"updated_at":"%s","state":"%s","stage":"%s","detail":"%s","pid":%s}\n' \
    "$(timestamp)" "${state}" "${ACTIVE_STAGE}" "${detail}" "$$" > "${STATUS}"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "${MASTER_LOG}"
}

stop_active() {
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}" 2>/dev/null || true
  fi
}

on_signal() {
  stop_active
  write_status "interrupted" "watchdog received a termination signal"
  log "interrupted during ${ACTIVE_STAGE}"
  exit 130
}
trap on_signal INT TERM HUP

run_stage() {
  local stage="$1"
  local duration="$2"
  shift 2
  local stage_log="${LOG_DIR}/${stage}.log"
  local last_size=-1
  local stable_checks=0

  ACTIVE_STAGE="${stage}"
  write_status "running" "timeout=${duration}"
  log "START ${stage}; timeout=${duration}; log=${stage_log}"
  timeout --signal=TERM --kill-after=5m "${duration}" "$@" > "${stage_log}" 2>&1 &
  ACTIVE_PID=$!

  while kill -0 "${ACTIVE_PID}" 2>/dev/null; do
    sleep 60
    local current_size
    current_size=$(stat -c %s "${stage_log}" 2>/dev/null || printf '0')
    if [[ "${current_size}" == "${last_size}" ]]; then
      stable_checks=$((stable_checks + 1))
    else
      stable_checks=0
    fi
    last_size="${current_size}"
    local gpu
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 || printf 'unavailable')
    local ram_available
    ram_available=$(awk '/MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)
    log "MONITOR ${stage}; log_bytes=${current_size}; unchanged_checks=${stable_checks}; gpu=${gpu}; ram_available=${ram_available}GiB"
    if (( stable_checks == 5 )); then
      log "ADVISORY ${stage}: stage log has not changed for five checks; process remains alive"
    fi
  done

  wait "${ACTIVE_PID}"
  local rc=$?
  ACTIVE_PID=""
  if (( rc != 0 )); then
    write_status "failed" "exit_code=${rc}; no automatic retry"
    log "FAIL ${stage}; exit_code=${rc}; tail follows"
    tail -n 80 "${stage_log}" | tee -a "${MASTER_LOG}"
    exit "${rc}"
  fi
  write_status "stage_complete" "exit_code=0"
  log "DONE ${stage}"
}

log "Cycle09 off-KD detached workflow started (no host auto-shutdown)"

if [[ -s "${ROLLOUT_DIR}/teacher_rollout.jsonl" \
   && -s "${ROLLOUT_DIR}/teacher_top32_logprob.npz" \
   && -s "${ROLLOUT_DIR}/rollout_manifest.json" ]]; then
  log "SKIP pass2 generation: all three formal artifacts already exist"
else
  run_stage "formal_pass2" "6h" \
    env CUDA_VISIBLE_DEVICES=0 "${ROLLOUT_PY}" "${ROLLOUT_SCRIPT}" \
      --stage pass2 \
      --pass1-minutes 352.0 \
      --pass2-record-batch 4 \
      --pass2-min-available-ram-gib 128 \
      --out "${ROLLOUT_DIR}"
fi

run_stage "formal_pass2_validate" "30m" \
  env CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${TRAIN_SCRIPT}" --validate-only

if [[ -f "${EXP_ROOT}/smoke/train_pass2/training_manifest.json" ]] \
   && rg -q '"status": "complete"' "${EXP_ROOT}/smoke/train_pass2/training_manifest.json"; then
  log "SKIP train smoke: completed smoke manifest exists"
else
  run_stage "longest_sequence_train_smoke" "1h" \
    env CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${TRAIN_SCRIPT}" --smoke --resume auto
fi

if [[ -f "${EXP_ROOT}/checkpoints/training_manifest.json" ]] \
   && rg -q '"status": "complete"' "${EXP_ROOT}/checkpoints/training_manifest.json"; then
  log "SKIP formal training: completed manifest exists"
else
  run_stage "formal_offkd_train" "36h" \
    env CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${TRAIN_SCRIPT}" --resume auto
fi

ACTIVE_STAGE="all_complete"
write_status "complete" "formal pass2, validation, smoke, and training completed"
log "ALL COMPLETE"
