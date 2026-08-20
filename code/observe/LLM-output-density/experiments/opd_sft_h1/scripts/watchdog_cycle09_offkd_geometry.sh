#!/usr/bin/env bash
# Detached, single-run watchdog for the off-KD geometry main grid.
# It never retries and never powers off the host.
set -uo pipefail

ROOT=/root/autodl-tmp/cycle09_offkd/geometry
REPO=/root/LLM-output-density
SCRIPT=$REPO/experiments/opd_sft_h1/scripts/cycle09_offkd_geometry.py
PY=/root/miniconda3/envs/density/bin/python
MINI=$REPO/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG_DIR=$ROOT/logs
LOG=$LOG_DIR/formal.log
WATCHDOG_LOG=$LOG_DIR/watchdog.log
STATUS=$ROOT/OFFKD_GEOMETRY_STATUS.txt

mkdir -p "$LOG_DIR"

log() {
    printf '[offkd-geometry %s] %s\n' "$(date -u '+%F %T UTC')" "$*" | tee -a "$WATCHDOG_LOG"
}

write_status() {
    {
        printf 'cycle09 offkd geometry: %s\n' "$1"
        printf 'detail    : %s\n' "$2"
        printf 'updated_at: %s\n' "$(date -u '+%F %T UTC')"
        printf 'formal_log: %s\n' "$LOG"
        printf 'watchdog  : %s\n' "$WATCHDOG_LOG"
    } > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
}

main() {
    log "starting main grid 0,5,10,20,40,160,624"
    write_status "RUNNING" "main-grid geometry"

    "$PY" "$SCRIPT" >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        log "formal run failed with rc=$rc"
        write_status "FAILED" "formal run rc=$rc; inspect $LOG"
        exit "$rc"
    fi

    if [ ! -s "$MINI/offkd_geometry_manifest.json" ] ||
       [ ! -s "$MINI/R4_v2_spectra_offkd.csv" ]; then
        log "formal command exited zero but required outputs are missing"
        write_status "FAILED" "required output missing"
        exit 1
    fi

    log "formal run completed"
    write_status "COMPLETED" "main-grid geometry and internal row validation complete"
}

main "$@"
