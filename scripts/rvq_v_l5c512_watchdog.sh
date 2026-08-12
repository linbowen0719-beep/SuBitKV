#!/usr/bin/env bash
set -u

cd /home/bowen/Quant-VideoGen || exit 1

LOG_DIR=/data/bowen/rvq_diag/rvq_hparam_fast/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/v_l5c512_watchdog_$(date +%Y%m%d_%H%M%S).log"
FINAL_SUMMARY=/data/bowen/rvq_diag/rvq_hparam_fast/sweep_v_l5c512_summary.json

echo "[watchdog] start $(date)" >> "$LOG"

runner_active() {
  pgrep -af "run_rvq_hparam_fast_sweep.py --output-dir /data/bowen/rvq_diag" \
    | grep -E "l5c512" >/dev/null 2>&1
}

while true; do
  if [[ -f "$FINAL_SUMMARY" ]]; then
    echo "[watchdog] final V L5C512 summary exists; exit $(date)" >> "$LOG"
    exit 0
  fi

  if runner_active; then
    sleep 60
    continue
  fi

  echo "[watchdog] V L5C512 runner missing; resume $(date)" >> "$LOG"
  /home/bowen/anaconda3/envs/hw8/bin/python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
    --output-dir /data/bowen/rvq_diag \
    --variants l5c512 \
    --tensors V \
    --gpu-groups '2,3' \
    --max-parallel 1 \
    --worker-python /home/bowen/anaconda3/envs/hw8/bin/python \
    --dump-python /home/bowen/anaconda3/envs/hw8/bin/python \
    --eval-vector-sample 4096 \
    --query-sample 128 \
    --key-sample 2048 \
    --topk 32 \
    --eval-batch-size 8192 >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog] resume command exited rc=$rc $(date)" >> "$LOG"
  sleep 30
done
