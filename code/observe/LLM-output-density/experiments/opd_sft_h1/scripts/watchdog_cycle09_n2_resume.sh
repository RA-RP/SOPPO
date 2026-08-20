#!/usr/bin/env bash
# Resume N-2 from the failed OPD H@80 cell, then run off-KD H seven-step work.
# Fail-stop: COMPLETE is written only after the strict finalizer passes.
set -Eeuo pipefail

ROOT=/root/autodl-tmp/cycle09_n2
SCRIPTS=/root/LLM-output-density/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
LOG=$ROOT/logs/resume_chain.log
STATUS=$ROOT/N2_RESUME_STATUS.txt

mkdir -p "$ROOT/logs"

log() {
    printf '[n2-resume %s] %s\n' "$(date -u '+%F %T UTC')" "$*" | tee -a "$LOG"
}

status() {
    printf 'N2 resume chain: %s\ndetail: %s\nupdated: %s\nshutdown: disabled\n' \
        "$1" "$2" "$(date -u '+%F %T UTC')" > "$STATUS"
}

on_error() {
    local rc=$?
    status FAILED "rc=$rc command=$BASH_COMMAND"
    log "FAILED rc=$rc command=$BASH_COMMAND"
    exit "$rc"
}
trap on_error ERR

status RUNNING "item2 corrected OPD H@80 measurement"
log "START item2 corrected OPD H@80"
"$PY" "$SCRIPTS/cycle09_n2_h80_measure.py" \
    --arm opd \
    --probe-step 80 \
    --domains bos,ood \
    --steps 0,80 \
    --work-root "$ROOT/h80_opd_v2" \
    >> "$ROOT/logs/item2_h80_resume_v4.log" 2>&1
log "DONE item2 corrected OPD H@80"

log "VALIDATE item2 OPD H@80"
"$PY" "$SCRIPTS/cycle09_n2_finalize.py" --item2-only \
    >> "$ROOT/logs/item2_validate.log" 2>&1
log "DONE item2 OPD H@80"

status RUNNING "item4 off-KD H seven-step generation and geometry"
log "START item4 off-KD H seven-step"
"$PY" "$SCRIPTS/cycle09_n2_offkd_h.py" \
    --python "$PY" >> "$ROOT/logs/item4_offkd_h_v2.log" 2>&1
log "DONE item4 off-KD H seven-step"

status RUNNING "strict final validation and spectra reconstruction"
log "START N-2 final validation"
"$PY" "$SCRIPTS/cycle09_n2_finalize.py" \
    >> "$ROOT/logs/finalize.log" 2>&1
log "DONE N-2 final validation"

status COMPLETE "all pulled-forward N-2 items validated"
log "N-2 compute chain COMPLETE (shutdown disabled)"
