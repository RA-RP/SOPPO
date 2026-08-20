#!/usr/bin/env bash
# Cycle 09 R5 — A4 only (B line and A line already completed 13:18 / 17:29).
# Sequential, no concurrency. Auto-shutdown on completion OR failure.
set -uo pipefail
RUN_ROOT=/root/autodl-tmp/cycle09_r5
SCRIPTS=/root/LLM-output-density/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
MINI=/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG=$RUN_ROOT/logs/r5_watchdog.log
STATUS=$RUN_ROOT/R5_STATUS.txt
ABORT=$RUN_ROOT/ABORT_SHUTDOWN
GRACE=120
log(){ printf '[r5-a4 %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
status(){ printf 'cycle09 round5 pipeline: %s\ndetail    : %s\nupdated_at: %s\n' "$1" "$2" "$(date '+%F %T')" > "$STATUS"; }
rm -f "$ABORT"
log "restart after OOM: running A4 only (gram-cache leak fixed; ~47GB freed)"
status RUNNING r5_a4_bootstrap
$PY $SCRIPTS/cycle09_r5_bootstrap.py --all >> $RUN_ROOT/logs/r5_a4_bootstrap.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$MINI/R5_bootstrap_ci.csv" ]; then status FAILED r5_a4_bootstrap; log "FAILED rc=$rc"
else status COMPLETED "all Cycle 09 Round 5 stages completed"; log "COMPLETED"; fi
log "AutoDL shutdown in ${GRACE}s. Cancel: touch $ABORT"
r=$GRACE; while [ $r -gt 0 ]; do [ -f "$ABORT" ] && { log "ABORT found; staying up"; exit $rc; }; sleep 1; r=$((r-1)); done
log "shutdown now"; sync; shutdown
