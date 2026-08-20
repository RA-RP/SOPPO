#!/usr/bin/env bash
# T3: OPD/SFT geometry backfill at {80,320,480}. Sequential, arm by arm.
# Safety: the three shared csvs are backed up before the first append; after each arm we
# assert the arm's existing seven-point rows survived (the off-KD script's append wiped a
# whole arm, which would have destroyed them here).
set -uo pipefail
ROOT=/root/autodl-tmp/cycle09_t3_backfill
SCRIPTS=/root/LLM-output-density/experiments/opd_sft_h1/scripts
PY=/root/miniconda3/envs/density/bin/python
MINI=/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini
LOG=$ROOT/logs/t3_watchdog.log
STATUS=$ROOT/T3_STATUS.txt
mkdir -p $ROOT/logs $ROOT/backup
log(){ printf '[t3 %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }
status(){ printf 'cycle09 T3 backfill: %s\ndetail    : %s\nupdated_at: %s\n' "$1" "$2" "$(date '+%F %T')" > $STATUS; }

# one-time backup of the shared csvs
for f in R4_m1_tail_ec.csv R4_m2_output_drift.csv R5_theta_reps.csv; do
    [ -f "$ROOT/backup/$f" ] || cp "$MINI/$f" "$ROOT/backup/$f"
done
log "backed up 3 shared csvs -> $ROOT/backup"

check_rows(){  # arm -> assert its 7 main-grid steps still present
    $PY - "$1" <<'PYEOF'
import sys, pandas as pd
arm = sys.argv[1]
MINI = "/root/LLM-output-density/mypaper/local_experiment_results/cycle_09_aaai_competitiveness_completion/run_01/mini/"
d = pd.read_csv(MINI + "R4_m1_tail_ec.csv")
have = sorted(d[d.arm == arm].step.unique())
need = [0, 5, 10, 20, 40, 160, 624]
missing = [s for s in need if s not in have]
if missing:
    raise SystemExit(f"FATAL: {arm} lost main-grid steps {missing}; have={have}")
print(f"OK {arm}: steps={have}")
PYEOF
}

for arm in sft opd; do
    status RUNNING "$arm 80,320,480"
    log "starting arm=$arm steps=80,320,480"
    $PY $SCRIPTS/cycle09_t3_backfill_geometry.py --arm $arm --steps 80,320,480 \
        >> $ROOT/logs/t3_$arm.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then log "arm=$arm FAILED rc=$rc"; status FAILED "$arm rc=$rc"; exit $rc; fi
    if ! check_rows $arm >> "$LOG" 2>&1; then
        log "arm=$arm ROW-LOSS DETECTED; restoring backup"
        cp $ROOT/backup/*.csv $MINI/
        status FAILED "$arm row loss; backup restored"; exit 1
    fi
    log "arm=$arm completed and row-check passed"
done
status COMPLETED "both arms backfilled"
log "T3 COMPLETED"
