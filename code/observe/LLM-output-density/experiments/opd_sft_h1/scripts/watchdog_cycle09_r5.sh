#!/usr/bin/env bash
# Cycle 09 Round 5 autonomous single-GPU pipeline watchdog.
# Sequential (no concurrency, no memory contention), each stage logged, key
# artifacts validated, then AutoDL powers off on completion OR failure.
# Cancel the auto-shutdown within the grace window: touch <RUN_ROOT>/ABORT_SHUTDOWN
set -uo pipefail

RUN_ROOT=/root/autodl-tmp/cycle09_r5
REPO=/root/LLM-output-density
SCRIPTS=$REPO/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
MINI=$REPO/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG_DIR=$RUN_ROOT/logs
WATCHDOG_LOG=$LOG_DIR/r5_watchdog.log
STATUS=$RUN_ROOT/R5_STATUS.txt
ABORT=$RUN_ROOT/ABORT_SHUTDOWN
GRACE=120

mkdir -p "$LOG_DIR"

log() { printf '[r5-watchdog %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCHDOG_LOG"; }

write_status() {
    { printf 'cycle09 round5 pipeline: %s\n' "$1"
      printf 'detail    : %s\n' "$2"
      printf 'updated_at: %s\n' "$(date '+%F %T')"
      printf 'watchdog  : %s\n' "$WATCHDOG_LOG"; } > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
}

require_artifact() {
    if [ ! -s "$1" ]; then log "missing/empty required artifact: $1"; return 1; fi
}

run_stage() {
    local name="$1"; shift
    local stage_log=$LOG_DIR/$name.log
    log "starting $name; log -> $stage_log"
    write_status "RUNNING" "$name"
    "$@" >> "$stage_log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then log "$name FAILED rc=$rc (see $stage_log)"; return "$rc"; fi
    log "$name completed"
}

finish() {
    local rc="$1" state="$2" detail="$3"
    write_status "$state" "$detail"
    log "$state: $detail"
    log "AutoDL shutdown in ${GRACE}s. Cancel within grace: touch $ABORT"
    local r=$GRACE
    while [ "$r" -gt 0 ]; do
        if [ -f "$ABORT" ]; then
            write_status "$state" "$detail; shutdown CANCELLED"
            log "ABORT_SHUTDOWN found; leaving instance up."; exit "$rc"
        fi
        sleep 1; r=$((r - 1))
    done
    log "grace elapsed; AutoDL shutdown now."; sync; shutdown
    exit "$rc"
}

main() {
    rm -f "$ABORT"
    log "=============================================================="
    log "starting autonomous Cycle 09 Round 5 pipeline"
    log "sequence: B (selfgen -> 7x7 cross matrix -> mismatch/prereg) -> A1/A5/A6/A2/A3 -> A4 bootstrap"

    # Line B (GPU main line; the possible mechanism explanation)
    if ! run_stage "r5_b_line" "$PY" "$SCRIPTS/cycle09_r5_bline.py" --all; then
        finish 1 "FAILED" "r5_b_line"; fi
    require_artifact "$MINI/R5_cross_matrix.csv"     || finish 1 "FAILED" "cross matrix missing"
    require_artifact "$MINI/R5_mismatch.csv"         || finish 1 "FAILED" "mismatch missing"
    require_artifact "$MINI/R5_b4_prereg_readout.csv" || finish 1 "FAILED" "B4 readout missing"

    # Line A (results line): gamma / theta / e_keep diagnosis / raw fix
    if ! run_stage "r5_a_line" "$PY" "$SCRIPTS/cycle09_r5_aline.py" --all; then
        finish 1 "FAILED" "r5_a_line"; fi
    require_artifact "$MINI/R5_gamma_reps.csv" || finish 1 "FAILED" "gamma missing"
    require_artifact "$MINI/R5_theta_reps.csv" || finish 1 "FAILED" "theta missing"

    # A4 sample-level bootstrap (the statistical load-bearing beam)
    if ! run_stage "r5_a4_bootstrap" "$PY" "$SCRIPTS/cycle09_r5_bootstrap.py" --all; then
        finish 1 "FAILED" "r5_a4_bootstrap"; fi
    require_artifact "$MINI/R5_bootstrap_ci.csv" || finish 1 "FAILED" "bootstrap CI missing"

    finish 0 "COMPLETED" "all Cycle 09 Round 5 stages completed"
}
main "$@"
