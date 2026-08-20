#!/usr/bin/env bash
# Cycle 09 Round 3 autonomous single-GPU campaign watchdog.
#
# The existing R3-4 collection remains in its own screen session. This watchdog
# waits for it, validates the complete artifact matrix, then runs every
# remaining approved zero-training stage sequentially. It is intentionally a
# shell-level watchdog because AutoDL's bare shutdown command stops billing.
set -uo pipefail

RUN_ROOT=/root/autodl-tmp/cycle09_r3
REPO=/root/LLM-output-density
SCRIPTS=$REPO/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
MINI=$REPO/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG_DIR=$RUN_ROOT/logs
WATCHDOG_LOG=$LOG_DIR/r3_campaign_watchdog.log
STATUS=$RUN_ROOT/R3_CAMPAIGN_STATUS.txt
ABORT=$RUN_ROOT/ABORT_SHUTDOWN
R3_4_LOG=$LOG_DIR/r3_4_getslice_formal.log
R3_4_SCREEN=cycle09_r3_4
POLL_SECONDS=60
GRACE=120
EXPECTED_SPECTRA=84
EXPECTED_FACTORS=8064
EXPECTED_RAW=84

log() {
    printf '[watchdog %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$WATCHDOG_LOG"
}

write_status() {
    local state="$1"
    local detail="$2"
    {
        printf 'cycle09 round3 campaign: %s\n' "$state"
        printf 'detail    : %s\n' "$detail"
        printf 'updated_at: %s\n' "$(date '+%F %T')"
        printf 'watchdog  : %s\n' "$WATCHDOG_LOG"
    } > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
}

screen_alive() {
    screen -ls 2>/dev/null | grep -Fq ".$R3_4_SCREEN"
}

wait_for_r3_4() {
    if ! screen_alive; then
        log "R3-4 screen is already absent; validating its final artifacts."
        return 0
    fi
    log "waiting for active R3-4 screen session $R3_4_SCREEN"
    while screen_alive; do
        write_status "WAITING_R3_4" "R3-4 GetSlice collection is still active"
        sleep "$POLL_SECONDS"
    done
    log "R3-4 screen exited; validating final artifacts."
}

count_files() {
    find "$1" -type f 2>/dev/null | wc -l | tr -d '[:space:]'
}

verify_r3_4() {
    local spectra
    local factors
    local raw

    if ! grep -Fq "[Summary] X ER rows=1470 anisotropy rows=252" "$R3_4_LOG"; then
        log "R3-4 log has no successful final summary."
        return 1
    fi
    spectra=$(count_files "$RUN_ROOT/spectra")
    factors=$(count_files "$RUN_ROOT/factors")
    raw=$(count_files "$RUN_ROOT/raw_geometry")
    if [ "$spectra" != "$EXPECTED_SPECTRA" ] || [ "$factors" != "$EXPECTED_FACTORS" ] || [ "$raw" != "$EXPECTED_RAW" ]; then
        log "R3-4 artifact count mismatch: spectra=$spectra/$EXPECTED_SPECTRA factors=$factors/$EXPECTED_FACTORS raw=$raw/$EXPECTED_RAW"
        return 1
    fi
    require_artifact "$MINI/R3_getslice_manifest.json" || return 1
    require_artifact "$MINI/R3_xcond_whitened_er.csv" || return 1
    require_artifact "$MINI/R3_anisotropy.csv" || return 1
    log "R3-4 validation passed: spectra=$spectra factors=$factors raw=$raw"
}

require_artifact() {
    if [ ! -s "$1" ]; then
        log "missing or empty required artifact: $1"
        return 1
    fi
}

run_stage() {
    local name="$1"
    shift
    local stage_log=$LOG_DIR/$name.log
    log "starting $name; log -> $stage_log"
    write_status "RUNNING" "$name"
    "$@" >> "$stage_log" 2>&1
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        log "$name failed with exit code $rc"
        return "$rc"
    fi
    log "$name completed successfully"
}

finish() {
    local rc="$1"
    local state="$2"
    local detail="$3"
    write_status "$state" "$detail"
    log "$state: $detail"
    log "AutoDL shutdown in $GRACE seconds. Cancel only during this grace window with: touch $ABORT"
    local remaining=$GRACE
    while [ "$remaining" -gt 0 ]; do
        if [ -f "$ABORT" ]; then
            write_status "$state" "$detail; automatic shutdown cancelled"
            log "ABORT_SHUTDOWN found; leaving the instance running."
            exit "$rc"
        fi
        sleep 1
        remaining=$((remaining - 1))
    done
    log "grace elapsed; calling AutoDL shutdown now."
    sync
    shutdown
    exit "$rc"
}

main() {
    mkdir -p "$LOG_DIR"
    rm -f "$ABORT"
    log "=============================================================="
    log "starting autonomous Cycle 09 Round 3 campaign"
    log "current R3-4 log -> $R3_4_LOG"
    log "future stages run sequentially on one GPU; no VS Code session is needed"

    wait_for_r3_4
    if ! verify_r3_4; then
        finish 1 "FAILED" "R3-4 did not finish with a complete formal artifact matrix"
    fi

    if ! run_stage "r3_4_sample_bootstrap" "$PY" "$SCRIPTS/cycle09_r3_sample_bootstrap.py" --validate-factors --bootstrap --draws 256; then
        finish 1 "FAILED" "r3_4_sample_bootstrap"
    fi
    if ! require_artifact "$MINI/R3_er_sample_bands.csv" || ! require_artifact "$MINI/R3_er_sample_bands_draws.json"; then
        finish 1 "FAILED" "r3_4_sample_bootstrap outputs"
    fi

    if ! run_stage "r3_3_theta_w" "$PY" "$SCRIPTS/cycle09_r3_theta_w.py"; then
        finish 1 "FAILED" "r3_3_theta_w"
    fi
    if ! require_artifact "$MINI/R3_theta_w.csv"; then
        finish 1 "FAILED" "r3_3_theta_w output"
    fi

    if ! run_stage "r3_6_ood" "$PY" "$SCRIPTS/cycle09_r3_ood.py" --all; then
        finish 1 "FAILED" "r3_6_ood"
    fi
    if ! require_artifact "$MINI/R3_ood_expansion.csv"; then
        finish 1 "FAILED" "r3_6_ood output"
    fi

    if ! run_stage "r3_5_sxh" "$PY" "$SCRIPTS/cycle09_r3_sxh.py" --all; then
        finish 1 "FAILED" "r3_5_sxh"
    fi
    if ! require_artifact "$MINI/R3_sxh_er.csv" || ! require_artifact "$MINI/R3_sxh_generation_manifest.json"; then
        finish 1 "FAILED" "r3_5_sxh outputs"
    fi

    if ! run_stage "r3_7_id" "$PY" "$SCRIPTS/cycle09_r3_id.py" --all; then
        finish 1 "FAILED" "r3_7_id"
    fi
    if ! require_artifact "$MINI/R3_numina_cap_pilot.csv" || ! require_artifact "$MINI/R3_numina_cap_selection.json" || ! require_artifact "$MINI/R3_id_completion.csv"; then
        finish 1 "FAILED" "r3_7_id outputs"
    fi

    finish 0 "COMPLETED" "all queued Cycle 09 Round 3 formal stages completed"
}

main "$@"
