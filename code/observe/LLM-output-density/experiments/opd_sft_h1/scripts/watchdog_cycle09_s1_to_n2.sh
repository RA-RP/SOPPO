#!/usr/bin/env bash
# Bridge the confirmed S1-8/S1-9 chain directly into the interrupted N-2 work.
set -Eeuo pipefail

S1_PID=${S1_PID:-221064}
S1_STATUS=/root/autodl-tmp/cycle09_s1/S1_89_STATUS.txt
N2_ROOT=/root/autodl-tmp/cycle09_n2
N2_RUNNER=/root/LLM-output-density/experiments/opd_sft_h1/scripts/watchdog_cycle09_n2_resume.sh
STATUS=$N2_ROOT/S1_TO_N2_STATUS.txt
LOG=$N2_ROOT/logs/s1_to_n2_bridge.log

mkdir -p "$N2_ROOT/logs"

record() {
    local state=$1
    local detail=$2
    local tmp=$STATUS.tmp.$$
    printf 'S1 to N2 bridge: %s\ndetail: %s\nupdated: %s\nshutdown: disabled\nnight_block: not_started\n' \
        "$state" "$detail" "$(date -u '+%F %T UTC')" > "$tmp"
    mv "$tmp" "$STATUS"
    printf '[s1-to-n2 %s] %s: %s\n' "$(date -u '+%F %T UTC')" "$state" "$detail" >> "$LOG"
}

record WAITING "S1-8/S1-9 PID=$S1_PID"
while ! grep -q '^stage1 emergency S1-8/S1-9: COMPLETE$' "$S1_STATUS" 2>/dev/null; do
    if grep -q '^stage1 emergency S1-8/S1-9: FAILED$' "$S1_STATUS" 2>/dev/null; then
        record FAILED "S1-8/S1-9 reported failure; N2 not started"
        exit 1
    fi
    if ! kill -0 "$S1_PID" 2>/dev/null; then
        record FAILED "S1 runner exited without COMPLETE; N2 not started"
        exit 1
    fi
    sleep 2
done

if pgrep -f '[c]ycle09_n2_h80_measure.py|[c]ycle09_n2_offkd_h.py|[w]atchdog_cycle09_n2_resume.sh' >/dev/null; then
    record FAILED "an N2 worker is already active; refusing duplicate launch"
    exit 1
fi

record STARTING "S1 complete; exec corrected N2 resume chain"
exec /bin/bash "$N2_RUNNER"
