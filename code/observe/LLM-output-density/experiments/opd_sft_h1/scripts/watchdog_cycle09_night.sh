#!/usr/bin/env bash
# Night chain: A4 bootstrap (resume from per-cell cache) -> off-KD teacher rollout -> shutdown.
# A4 runs FIRST (user needs its CI table in the morning). Sequential, never concurrent.
set -uo pipefail
R5=/root/autodl-tmp/cycle09_r5
OFFKD=/root/autodl-tmp/cycle09_offkd
SCRIPTS=/root/LLM-output-density/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
MINI=/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG=$R5/logs/night_chain.log
ABORT=$R5/ABORT_SHUTDOWN
GRACE=180
mkdir -p $R5/logs $OFFKD
log(){ printf '[night %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
status(){ printf 'night chain: %s\nupdated_at: %s\n' "$1" "$(date '+%F %T')" > $R5/NIGHT_STATUS.txt; }

rm -f "$ABORT"
log "=== chain start: A4 (resume) -> off-KD rollout -> shutdown ==="

status "RUNNING a4_bootstrap (resume)"
log "stage 1/2: A4 bootstrap (resuming from $(ls $R5/scratch/a4_cache 2>/dev/null | wc -l) cached cells)"
$PY $SCRIPTS/cycle09_r5_bootstrap.py --all >> $R5/logs/r5_a4_bootstrap.log 2>&1
rc=$?
if [ $rc -eq 0 ] && [ -s "$MINI/R5_bootstrap_ci.csv" ]; then log "A4 COMPLETED"; status "A4 done"
else log "A4 FAILED rc=$rc (continuing to rollout anyway; A4 cells are checkpointed)"; status "A4 FAILED"; fi

status "RUNNING offkd_rollout"
log "stage 2/2: off-KD teacher rollout (5000 prompts, two passes)"
$PY $SCRIPTS/cycle09_offkd_rollout.py >> $OFFKD/rollout.log 2>&1
rc2=$?
if [ $rc2 -eq 0 ]; then log "rollout COMPLETED"; status "ALL COMPLETED"
else log "rollout FAILED rc=$rc2 (pass1 jsonl is persisted; rerun with --stage pass2)"; status "rollout FAILED"; fi

log "AutoDL shutdown in ${GRACE}s. Cancel: touch $ABORT"
r=$GRACE; while [ $r -gt 0 ]; do [ -f "$ABORT" ] && { log "ABORT found; staying up"; exit 0; }; sleep 1; r=$((r-1)); done
log "shutdown now"; sync; shutdown
