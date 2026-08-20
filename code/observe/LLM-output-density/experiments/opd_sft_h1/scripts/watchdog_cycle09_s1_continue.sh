#!/usr/bin/env bash
# Continue confirmed Stage 1 after the already-running S1-3 job.
# Fail-stop, no retry, no night block, no data sync, and no machine shutdown.
set -Eeuo pipefail

REPO=/root/LLM-output-density
RUN=/root/autodl-tmp/cycle09_s1
LOGS=$RUN/logs
SCRIPTS=$REPO/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
OLD_STATUS=$RUN/S1_STATUS.txt
STATUS=$RUN/S1_FULL_STATUS.txt
CHAIN_LOG=$LOGS/continuation.log
CURRENT_STAGE=startup

mkdir -p "$LOGS"
cd "$REPO"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

log() {
  printf '[s1-full %s] %s\n' "$(date -u '+%F %T UTC')" "$*" | tee -a "$CHAIN_LOG"
}

status() {
  local state=$1
  local detail=$2
  local tmp=$STATUS.tmp.$$
  printf 'stage1 full chain: %s\ndetail: %s\nupdated: %s\nshutdown: disabled\nnight_block: not_started\ndata_sync: not_started\n' \
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

run_step() {
  local name=$1
  local logfile=$2
  shift 2
  CURRENT_STAGE=$name
  status RUNNING "$name"
  log "START $name"
  "$@" >> "$LOGS/$logfile" 2>&1
  log "DONE $name"
}

status RUNNING "waiting for existing S1-3 process"
log "waiting for existing cycle09_s1_3_transient_ci.py --all"
while pgrep -f '[c]ycle09_s1_3_transient_ci.py --all' >/dev/null; do
  sleep 30
done

CURRENT_STAGE="verify existing S1-3 exit"
for _ in $(seq 1 12); do
  if grep -q '^stage1 chain: DONE' "$OLD_STATUS" 2>/dev/null; then
    break
  fi
  if grep -q '^stage1 chain: FAILED' "$OLD_STATUS" 2>/dev/null; then
    log "existing S1-3 watcher reported FAILED"
    status FAILED "existing S1-3 watcher reported failure"
    exit 1
  fi
  sleep 5
done
grep -q '^stage1 chain: DONE' "$OLD_STATUS"
log "existing S1-3 watcher reported DONE"

run_step "S1-3 finalizer" s1_3_finalize.log \
  "$PY" "$SCRIPTS/cycle09_s1_3_finalize.py"

run_step "S1-4 smoke" s1_4_smoke.log \
  "$PY" "$SCRIPTS/cycle09_s1_4_wikitext_ppl.py" --smoke
run_step "S1-4 formal" s1_4_formal.log \
  "$PY" "$SCRIPTS/cycle09_s1_4_wikitext_ppl.py"

run_step "S1-5 smoke rollout preparation" s1_5_smoke_prepare.log \
  "$PY" "$SCRIPTS/cycle09_s1_5_train_corpus_ppl.py" --stage prepare --smoke
run_step "S1-5 smoke PPL" s1_5_smoke_ppl.log \
  "$PY" "$SCRIPTS/cycle09_s1_5_train_corpus_ppl.py" --stage ppl --smoke
run_step "S1-5 formal rollout preparation" s1_5_formal_prepare.log \
  "$PY" "$SCRIPTS/cycle09_s1_5_train_corpus_ppl.py" --stage prepare
run_step "S1-5 formal PPL" s1_5_formal_ppl.log \
  "$PY" "$SCRIPTS/cycle09_s1_5_train_corpus_ppl.py" --stage ppl

run_step "S1-6 smoke" s1_6_smoke.log \
  "$PY" "$SCRIPTS/cycle09_s1_6_direction.py" --smoke
run_step "S1-6 formal" s1_6_formal.log \
  "$PY" "$SCRIPTS/cycle09_s1_6_direction.py"

run_step "S1-1/2 MMLU-Pro log smoke" s1_mmlupro_smoke.log \
  "$PY" "$SCRIPTS/cycle09_s1_mmlupro_logs.py" --smoke
run_step "S1-1/2 MMLU-Pro formal log grid" s1_mmlupro_formal.log \
  "$PY" "$SCRIPTS/cycle09_s1_mmlupro_logs.py"
run_step "S1-1/2 offline extraction audit" s1_12_audit.log \
  "$PY" "$SCRIPTS/cycle09_s1_12_mmlupro.py"

run_step "machine migration inventory refresh" s1_inventory.log \
  "$PY" "$SCRIPTS/cycle09_s1_inventory.py"
run_step "Stage 1 validation and Theory handoff" s1_handoff.log \
  "$PY" "$SCRIPTS/cycle09_s1_handoff.py"

CURRENT_STAGE=complete
status COMPLETE "all seven Stage 1 tasks validated and handed back; machine left running"
log "COMPLETE all confirmed Stage 1 tasks; no night block, no sync, no shutdown"
