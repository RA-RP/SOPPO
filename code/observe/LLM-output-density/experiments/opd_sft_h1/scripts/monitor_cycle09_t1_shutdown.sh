#!/usr/bin/env bash
# Independent completion monitor for an already-running Cycle09 T1 job.
# It powers off only after a validated COMPLETED state. A failed run leaves the
# instance online for diagnosis. During the grace period, cancellation is:
#   touch /root/autodl-tmp/cycle09_t1/ABORT_SHUTDOWN
set -uo pipefail

RUN_ROOT=/root/autodl-tmp/cycle09_t1
REPO=/root/LLM-output-density
MINI=$REPO/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
T1_STATUS=$RUN_ROOT/T1_STATUS.txt
MONITOR_STATUS=$RUN_ROOT/SHUTDOWN_MONITOR_STATUS.txt
LOG=$RUN_ROOT/logs/shutdown_monitor.log
ABORT=$RUN_ROOT/ABORT_SHUTDOWN
GRACE=${GRACE:-120}
POLL=${POLL:-15}

mkdir -p "$RUN_ROOT/logs"

log() {
    printf '[t1-shutdown %s] %s\n' "$(date -u '+%F %T UTC')" "$*" | tee -a "$LOG"
}

write_status() {
    {
        printf '%s\n' "$1"
        printf 'detail=%s\n' "$2"
        printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf 'monitor_pid=%s\n' "$$"
    } > "$MONITOR_STATUS.tmp"
    mv "$MONITOR_STATUS.tmp" "$MONITOR_STATUS"
}

required_artifacts_complete() {
    [ -s "$MINI/R5_t1_bootstrap_ci.csv" ] &&
    [ -s "$MINI/T1_bootstrap_manifest.json" ] &&
    [ -s "$MINI/R5_bootstrap_ci.csv" ] &&
    [ -s "$REPO/mypaper/code/code_evolution.md" ]
}

grace_then_shutdown() {
    write_status "SHUTDOWN_PENDING" "T1 complete; grace_seconds=$GRACE"
    log "T1 complete and artifacts present; AutoDL shutdown in ${GRACE}s."
    log "Cancel during grace with: touch $ABORT"
    remaining=$GRACE
    while [ "$remaining" -gt 0 ]; do
        if [ -f "$ABORT" ]; then
            write_status "CANCELLED" "ABORT_SHUTDOWN found"
            log "ABORT_SHUTDOWN found; leaving the instance online."
            exit 0
        fi
        sleep 1
        remaining=$((remaining - 1))
    done
    write_status "SHUTDOWN_REQUESTED" "grace elapsed"
    log "Grace elapsed; syncing filesystems and calling AutoDL shutdown now."
    sync
    shutdown
}

main() {
    rm -f "$ABORT"
    write_status "ARMED" "waiting for T1_STATUS=COMPLETED"
    log "Monitor armed; poll=${POLL}s grace=${GRACE}s."
    while true; do
        if [ ! -s "$T1_STATUS" ]; then
            sleep "$POLL"
            continue
        fi
        state=$(head -n 1 "$T1_STATUS" | tr -d '\r')
        case "$state" in
            COMPLETED)
                if required_artifacts_complete; then
                    grace_then_shutdown
                    exit 0
                fi
                write_status "BLOCKED" "COMPLETED but required artifacts are missing"
                log "T1 says COMPLETED, but required artifacts are missing; no shutdown."
                exit 1
                ;;
            FAILED)
                write_status "STOPPED_ON_FAILURE" "T1 failed; instance intentionally left online"
                log "T1 failed; leaving the instance online for diagnosis."
                exit 1
                ;;
            RUNNING)
                ;;
            *)
                log "Unrecognized T1 status '$state'; continuing to monitor."
                ;;
        esac
        sleep "$POLL"
    done
}

main "$@"
