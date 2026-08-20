#!/usr/bin/env bash
# Detached fail-stop runner for confirmed Stage-1 emergency tasks S1-8/S1-9.
set -Eeuo pipefail

REPO=/root/LLM-output-density
RUN=/root/autodl-tmp/cycle09_s1
LOGS=$RUN/logs
SCRIPTS=$REPO/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
STATUS=$RUN/S1_89_STATUS.txt
CHAIN_LOG=$LOGS/s1_89_chain.log
CURRENT_STAGE=startup

mkdir -p "$LOGS"
cd "$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

log() {
  printf '[s1-89 %s] %s\n' "$(date -u '+%F %T UTC')" "$*" | tee -a "$CHAIN_LOG"
}

status() {
  local state=$1
  local detail=$2
  local tmp=$STATUS.tmp.$$
  printf 'stage1 emergency S1-8/S1-9: %s\ndetail: %s\nupdated: %s\nshutdown: disabled\nnight_block: not_started\nn2_resume: disabled\n' \
    "$state" "$detail" "$(date -u '+%F %T UTC')" > "$tmp"
  mv "$tmp" "$STATUS"
}

fail() {
  local rc=$?
  trap - ERR
  log "FAILED stage=$CURRENT_STAGE rc=$rc"
  status FAILED "$CURRENT_STAGE rc=$rc; inspect $CHAIN_LOG"
  exit "$rc"
}
trap fail ERR

CURRENT_STAGE="S1-8 formal MMLU-Pro conditional LL grid"
status RUNNING "$CURRENT_STAGE"
log "START $CURRENT_STAGE"
"$PY" "$SCRIPTS/cycle09_s1_8_mmlupro_loglik.py" >> "$LOGS/s1_8_mmlupro_loglik.log" 2>&1
log "DONE $CURRENT_STAGE"

CURRENT_STAGE="S1-8/S1-9 validation and Theory handoff"
status RUNNING "$CURRENT_STAGE"
log "START $CURRENT_STAGE"
"$PY" "$SCRIPTS/cycle09_s1_89_finalize.py" >> "$LOGS/s1_89_finalize.log" 2>&1
log "DONE $CURRENT_STAGE"

CURRENT_STAGE=complete
status COMPLETE "S1-8/S1-9 validated and handed back; machine left running"
log "COMPLETE confirmed S1-8/S1-9; no night block, no N2 resume, no shutdown"
