#!/usr/bin/env bash
# Cycle 08 OPD run watchdog.
#
# Runs the FULL pipeline (run_cycle08.py: resume training from the latest verl
# checkpoint -> convert -> 2-GPU eval -> 2-GPU geometry -> aggregate), then
# POWERS OFF the autodl container whether the pipeline COMPLETES or CRASHES, so
# GPU billing stops the moment there is nothing left to compute.
#
# Autodl `shutdown` = kills supervisord -> container 关机 (stops GPU billing;
# both system disk and /root/autodl-tmp persist, so all logs/checkpoints/
# deliverables survive and the run can be inspected/resumed after reboot).
#
# Detached via `setsid nohup` at launch, so closing vscode / ssh does not kill it.
#
# Abort the auto-shutdown (e.g. you woke up and want to keep the box):
#   touch /root/autodl-tmp/cycle08_opd_trajectory/ABORT_SHUTDOWN
# during the grace window -> watchdog exits WITHOUT powering off.
set -uo pipefail

RUN_ROOT=/root/autodl-tmp/cycle08_opd_trajectory
SCRIPTS=/root/LLM-output-density/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
RUN_LOG=$RUN_ROOT/run_cycle08_full.log
WD_LOG=$RUN_ROOT/watchdog.log
STATUS=$RUN_ROOT/RUN_STATUS.txt
ABORT=$RUN_ROOT/ABORT_SHUTDOWN
GRACE=${GRACE:-90}   # seconds before poweroff; abortable via ABORT file

log(){ echo "[watchdog $(date +'%F %T')] $*" | tee -a "$WD_LOG"; }

rm -f "$ABORT"   # clear any stale abort flag from a previous run
log "=============================================================="
log "starting run_cycle08.py (real run, resumes from latest checkpoint)"
log "resume from step: $(cat "$RUN_ROOT/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null)"
log "pipeline log -> $RUN_LOG ; watchdog log -> $WD_LOG"

cd "$SCRIPTS" || { log "FATAL cannot cd $SCRIPTS"; exit 1; }

# Run the whole pipeline; capture its exit code.
"$PY" run_cycle08.py >> "$RUN_LOG" 2>&1
RC=$?

if [ "$RC" -eq 0 ]; then
    MSG="COMPLETED OK (exit 0) — deliverables produced"
else
    MSG="STOPPED WITH ERROR (exit $RC) — inspect $RUN_LOG (tail it after reboot)"
fi
{
  echo "cycle08 OPD run finished: $MSG"
  echo "finished at : $(date +'%F %T')"
  echo "exit code   : $RC"
  echo "last ckpt   : $(cat "$RUN_ROOT/checkpoints/latest_checkpointed_iteration.txt" 2>/dev/null)"
  echo "run log     : $RUN_LOG"
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
