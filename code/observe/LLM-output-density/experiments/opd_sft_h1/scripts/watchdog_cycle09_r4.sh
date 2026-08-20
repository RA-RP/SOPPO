#!/usr/bin/env bash
# Cycle 09 Round 4 autonomous single-GPU pipeline watchdog (v2 window campaign).
# Runs every zero-training R4 stage SEQUENTIALLY (no concurrency), validates key
# artifacts, then powers off the AutoDL instance on completion OR failure so
# billing stops overnight. Detached via `setsid nohup` — no VS Code needed.
# Cancel the auto-shutdown within the grace window:  touch <RUN_ROOT>/ABORT_SHUTDOWN
set -uo pipefail

RUN_ROOT=/root/autodl-tmp/cycle09_r4
REPO=/root/LLM-output-density
SCRIPTS=$REPO/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
MINI=$REPO/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG_DIR=$RUN_ROOT/logs
WATCHDOG_LOG=$LOG_DIR/r4_campaign_watchdog.log
STATUS=$RUN_ROOT/R4_CAMPAIGN_STATUS.txt
ABORT=$RUN_ROOT/ABORT_SHUTDOWN
GRACE=120

mkdir -p "$LOG_DIR"

log() { printf '[r4-watchdog %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCHDOG_LOG"; }

write_status() {
    { printf 'cycle09 round4 pipeline: %s\n' "$1"
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
    log "starting autonomous Cycle 09 Round 4 pipeline (v2, 3 layers, full quality)"
    log "sequence: campaign --all -> bootstrap --all -> behavior --all -> postprocess --all"

    # R4-1: generation + windowing + profiling + M2/M3 (GPU, the long stage)
    if ! run_stage "r4_1_campaign" "$PY" "$SCRIPTS/cycle09_r4_campaign.py" --all; then
        finish 1 "FAILED" "r4_1_campaign"; fi
    require_artifact "$MINI/R4_v2_spectra_all.csv" || finish 1 "FAILED" "campaign spectra missing"
    require_artifact "$MINI/R4_v2_manifest.json"   || finish 1 "FAILED" "campaign manifest missing"

    # R4-3.1/3.2: sample-level bootstrap (L18 re-derivation + discriminability)
    if ! run_stage "r4_3_bootstrap" "$PY" "$SCRIPTS/cycle09_r4_bootstrap.py" --all; then
        finish 1 "FAILED" "r4_3_bootstrap"; fi

    # R4-3.3 inputs: behavioral evals (IFEval / MMLU-Pro prompt-level)
    if ! run_stage "r4_3_behavior" "$PY" "$SCRIPTS/cycle09_r4_behavior.py" --all; then
        finish 1 "FAILED" "r4_3_behavior"; fi

    # R4-2 (M1) + R4-3.3 (colocation, needs behavior)
    if ! run_stage "r4_2_postprocess" "$PY" "$SCRIPTS/cycle09_r4_postprocess.py" --all; then
        finish 1 "FAILED" "r4_2_postprocess"; fi
    require_artifact "$MINI/R4_m1_tail_ec.csv" || finish 1 "FAILED" "M1 output missing"

    finish 0 "COMPLETED" "all Cycle 09 Round 4 formal stages completed"
}
main "$@"
