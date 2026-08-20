#!/usr/bin/env bash
# Unified-cap MATH re-test watchdog.
#
# Runs cap_unified_retest.py (re-evaluate the truncation-affected generative math
# tasks at the UNIFIED long cap, BOTH arms), then POWERS OFF autodl on completion
# OR crash — same stop-billing behaviour as watchdog_cycle08.sh.
#
# Scope is passed via RETEST_ARGS env (default = the doc's agreed quick check).
# Detached via `setsid nohup`; survives vscode/ssh close.
# Cancel the auto-poweroff within the grace window:
#   touch /root/autodl-tmp/cap_unified_retest/ABORT_SHUTDOWN
set -uo pipefail

SCRIPTS=/root/LLM-output-density/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
OUT=/root/autodl-tmp/cap_unified_retest
RUN_LOG=$OUT/cap_unified_retest.log
WD_LOG=$OUT/watchdog_cap_retest.log
STATUS=$OUT/RETEST_STATUS.txt
ABORT=$OUT/ABORT_SHUTDOWN
GRACE=${GRACE:-90}
RETEST_ARGS=${RETEST_ARGS:-}

mkdir -p "$OUT"
rm -f "$ABORT"
log(){ echo "[cap-retest-wd $(date +'%F %T')] $*" | tee -a "$WD_LOG"; }

log "=============================================================="
log "starting cap_unified_retest.py  args=[$RETEST_ARGS]"
log "run log -> $RUN_LOG ; watchdog log -> $WD_LOG"
cd "$SCRIPTS" || { log "FATAL cannot cd $SCRIPTS"; exit 1; }

"$PY" cap_unified_retest.py $RETEST_ARGS >> "$RUN_LOG" 2>&1
RC=$?

# --- supplements (CPU): aggregate OPD+SFT geometry/OverlapLift summaries + comparison ---
log "running CPU supplements (geometry/overlap summaries + OPD-vs-SFT)"
"$PY" cycle08_supplements.py >> "$RUN_LOG" 2>&1 || log "supplements FAILED (non-fatal)"
# --- random-LoRA A08 reference (GPU): only if the builder script is present ---
if [ -f "$SCRIPTS/cycle08_random_lora_ref.py" ]; then
    log "running random-LoRA A08 reference"
    "$PY" cycle08_random_lora_ref.py >> "$RUN_LOG" 2>&1 || log "random-LoRA ref FAILED (non-fatal)"
else
    log "cycle08_random_lora_ref.py absent -> random-LoRA A08 deferred to a later session"
fi

if [ "$RC" -eq 0 ]; then
    MSG="COMPLETED OK (exit 0) — unified-cap re-test done"
else
    MSG="STOPPED WITH ERROR (exit $RC) — inspect $RUN_LOG"
fi
{
  echo "cap unified re-test finished: $MSG"
  echo "finished at : $(date +'%F %T')"
  echo "exit code   : $RC"
  echo "args        : $RETEST_ARGS"
  echo "outputs     : $OUT/cap_unified_trajectory.csv , $OUT/opd_vs_sft_unified.md"
} > "$STATUS"
log "$MSG"
log "will POWER OFF autodl in ${GRACE}s. To cancel: touch $ABORT"

for ((i=GRACE; i>0; i--)); do
    if [ -f "$ABORT" ]; then
        log "ABORT flag found -> NOT powering off. Watchdog exits; box stays up."
        exit 0
    fi
    sleep 1
done
log "grace elapsed -> calling shutdown NOW (stops GPU billing)"
shutdown
