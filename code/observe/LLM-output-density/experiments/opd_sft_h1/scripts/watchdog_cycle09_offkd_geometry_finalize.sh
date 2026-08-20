#!/usr/bin/env bash
# Wait for the seven-point run, add three backfill points, then finalize once.
set -uo pipefail

ROOT=/root/autodl-tmp/cycle09_offkd/geometry
REPO=/root/LLM-output-density
PY=/root/miniconda3/envs/density/bin/python
GEOMETRY=$REPO/experiments/opd_sft_h1/scripts/cycle09_offkd_geometry.py
FINALIZER=$REPO/experiments/opd_sft_h1/scripts/cycle09_offkd_geometry_finalize.py
FORMAL_STATUS=$ROOT/OFFKD_GEOMETRY_STATUS.txt
STATUS=$ROOT/OFFKD_GEOMETRY_HANDIN_STATUS.txt
LOG=$ROOT/logs/finalize.log
EXTEND_LOG=$ROOT/logs/optional_80_320_480.log

write_status() {
    {
        printf 'cycle09 offkd geometry handin: %s\n' "$1"
        printf 'detail    : %s\n' "$2"
        printf 'updated_at: %s\n' "$(date -u '+%F %T UTC')"
        printf 'log       : %s\n' "$LOG"
        printf 'extend_log: %s\n' "$EXTEND_LOG"
    } > "$STATUS.tmp"
    mv "$STATUS.tmp" "$STATUS"
}

write_status "WAITING" "formal geometry"
while true; do
    if [ -s "$FORMAL_STATUS" ] &&
       grep -q '^cycle09 offkd geometry: COMPLETED$' "$FORMAL_STATUS"; then
        break
    fi
    if [ -s "$FORMAL_STATUS" ] &&
       grep -q '^cycle09 offkd geometry: FAILED$' "$FORMAL_STATUS"; then
        write_status "BLOCKED" "formal geometry failed; no handin written"
        exit 1
    fi
    sleep 60
done

write_status "RUNNING" "measuring optional checkpoints 80,320,480"
"$PY" "$GEOMETRY" --steps 0,5,10,20,40,80,160,320,480,624 >> "$EXTEND_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    write_status "FAILED" "optional geometry rc=$rc; inspect $EXTEND_LOG"
    exit "$rc"
fi

write_status "RUNNING" "validating and writing raw tables"
"$PY" "$FINALIZER" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
    write_status "FAILED" "finalizer rc=$rc; inspect $LOG"
    exit "$rc"
fi
write_status "COMPLETED" "raw tables and code_evolution handin written"
