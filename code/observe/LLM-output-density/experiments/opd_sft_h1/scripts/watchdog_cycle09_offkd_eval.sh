#!/usr/bin/env bash
# Detached continuation: checkpoint backfill -> three-arm ten-point evaluation.
# It never retries a failed stage and intentionally does not power off the host.

set -u
set -o pipefail

SCRIPT_DIR="/root/LLM-output-density/experiments/opd_sft_h1/scripts"
EXP_ROOT="/root/autodl-tmp/cycle09_offkd"
TRAIN_MANIFEST="${EXP_ROOT}/checkpoints/training_manifest.json"
FINAL_COMPLETE="${EXP_ROOT}/checkpoints/checkpoint-000624/complete.json"
TRAIN_WATCHDOG_STATUS="${EXP_ROOT}/watchdog_status.json"
EVAL_ROOT="${EXP_ROOT}/eval"
LOG_DIR="${EXP_ROOT}/logs"
STATUS="${EXP_ROOT}/eval_watchdog_status.json"
MASTER_LOG="${LOG_DIR}/eval_watchdog.log"
PY="/root/miniconda3/envs/density/bin/python"
TRAIN_PY="/root/autodl-tmp/envs/verl/bin/python"
EVAL_SCRIPT="${SCRIPT_DIR}/cycle09_offkd_eval.py"
TRAIN_SCRIPT="${SCRIPT_DIR}/cycle09_offkd_train.py"
BACKFILL_VALIDATE="${SCRIPT_DIR}/cycle09_offkd_backfill_validate.py"
R3_OOD_SCRIPT="${SCRIPT_DIR}/cycle09_r3_ood.py"
THREE_ARM_AGG="${SCRIPT_DIR}/cycle09_three_arm_eval_aggregate.py"
R3_ROOT="/root/autodl-tmp/cycle09_r3"
MINI_ROOT="/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini"
FULL_MANIFEST="/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/offkd/three_arm_full_trajectory_manifest.json"
FULL_GRID="0,5,10,20,40,80,160,320,480,624"
ACTIVE_PID=""
ACTIVE_STAGE="waiting_for_training"
WAIT_STARTED=$(date +%s)
WAIT_TIMEOUT_SECONDS=$((8 * 60 * 60))

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
  write_status "interrupted" "eval watchdog received a termination signal"
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
    tail -n 100 "${stage_log}" | tee -a "${MASTER_LOG}"
    exit "${rc}"
  fi
  write_status "stage_complete" "exit_code=0"
  log "DONE ${stage}"
}

training_complete() {
  [[ -s "${TRAIN_MANIFEST}" ]] \
    && [[ -s "${FINAL_COMPLETE}" ]] \
    && rg -q '"status": "complete"' "${TRAIN_MANIFEST}" \
    && [[ -s "${TRAIN_WATCHDOG_STATUS}" ]] \
    && rg -q '"state":"complete"' "${TRAIN_WATCHDOG_STATUS}"
}

training_failed() {
  [[ -s "${TRAIN_WATCHDOG_STATUS}" ]] \
    && rg -q '"state":"(failed|interrupted)"' "${TRAIN_WATCHDOG_STATUS}"
}

all_complete() {
  [[ -s "${FULL_MANIFEST}" ]] \
    && rg -q '"status": "complete"' "${FULL_MANIFEST}"
}

log "Cycle09 off-KD eval continuation started (no retry, no host auto-shutdown)"

if all_complete; then
  ACTIVE_STAGE="all_complete"
  write_status "complete" "three-arm full trajectory was already complete"
  log "SKIP: three-arm full trajectory manifest is already complete"
  exit 0
fi

while ! training_complete; do
  if training_failed; then
    write_status "failed" "training watchdog failed before eval handoff"
    log "FAIL: training watchdog failed before eval handoff"
    exit 1
  fi
  now=$(date +%s)
  if (( now - WAIT_STARTED > WAIT_TIMEOUT_SECONDS )); then
    write_status "failed" "training wait exceeded 8h; no automatic retry"
    log "FAIL: training wait exceeded 8h"
    exit 124
  fi
  progress=$(
    "${PY}" -c \
      "import json,pathlib; p=pathlib.Path('${TRAIN_MANIFEST}'); d=json.load(open(p)) if p.exists() else {}; print(str(d.get('completed_steps','?'))+'/'+str(d.get('total_steps','624')))" \
      2>/dev/null || printf '?/624'
  )
  write_status "waiting" "training_progress=${progress}"
  log "WAIT training_progress=${progress}; final_complete=$([[ -s "${FINAL_COMPLETE}" ]] && printf yes || printf no)"
  sleep 60
done

log "HANDOFF training complete and checkpoint-000624 is complete"

if [[ -s "${EXP_ROOT}/checkpoint_backfill/from_040/checkpoint-000080/complete.json" ]]; then
  log "SKIP backfill 80: complete checkpoint exists"
else
  run_stage "offkd_backfill_080" "2h" \
    env CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${TRAIN_SCRIPT}" \
      --output-dir "${EXP_ROOT}/checkpoint_backfill/from_040" \
      --resume "${EXP_ROOT}/checkpoints/checkpoint-000040" \
      --max-steps 80 \
      --checkpoint-grid "${FULL_GRID}" \
      --no-copyback
fi

if [[ -s "${EXP_ROOT}/checkpoint_backfill/from_160/checkpoint-000320/complete.json" \
   && -s "${EXP_ROOT}/checkpoint_backfill/from_160/checkpoint-000480/complete.json" ]]; then
  log "SKIP backfill 320/480: complete checkpoints exist"
else
  run_stage "offkd_backfill_320_480" "8h" \
    env CUDA_VISIBLE_DEVICES=0 "${TRAIN_PY}" "${TRAIN_SCRIPT}" \
      --output-dir "${EXP_ROOT}/checkpoint_backfill/from_160" \
      --resume "${EXP_ROOT}/checkpoints/checkpoint-000160" \
      --max-steps 480 \
      --checkpoint-grid "${FULL_GRID}" \
      --no-copyback

fi

run_stage "offkd_backfill_validate" "30m" \
  env CUDA_VISIBLE_DEVICES=0 "${PY}" "${BACKFILL_VALIDATE}"

run_stage "baseline_ood_full_grid" "3h" \
  env CUDA_VISIBLE_DEVICES=0 "${PY}" "${R3_OOD_SCRIPT}" \
    --all \
    --run-root "${R3_ROOT}" \
    --mini-root "${MINI_ROOT}" \
    --arms "opd,sft" \
    --steps "${FULL_GRID}"

run_stage "offkd_eval_preflight" "30m" \
  env CUDA_VISIBLE_DEVICES=0 "${PY}" "${EVAL_SCRIPT}" --dry-run

if [[ -s "${EXP_ROOT}/smoke_eval/evaluation_manifest.json" ]] \
   && rg -q '"status": "complete"' "${EXP_ROOT}/smoke_eval/evaluation_manifest.json"; then
  log "SKIP smoke: completed smoke eval manifest exists"
else
  run_stage "offkd_eval_smoke" "1h" \
    env CUDA_VISIBLE_DEVICES=0 "${PY}" "${EVAL_SCRIPT}" --smoke
fi

run_stage "offkd_eval_formal" "36h" \
  env CUDA_VISIBLE_DEVICES=0 "${PY}" "${EVAL_SCRIPT}"

run_stage "three_arm_full_trajectory_aggregate" "30m" \
  env CUDA_VISIBLE_DEVICES=0 "${PY}" "${THREE_ARM_AGG}"

ACTIVE_STAGE="all_complete"
write_status "complete" "backfill, validation, baseline gap-fill, offKD eval, and three-arm aggregate completed"
log "ALL COMPLETE"
