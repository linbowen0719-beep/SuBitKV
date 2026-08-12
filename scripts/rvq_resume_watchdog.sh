#!/usr/bin/env bash
set -u

cd /home/bowen/Quant-VideoGen || exit 1

LOG_DIR=/data/bowen/rvq_diag/rvq_hparam_fast/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/resume_watchdog_$(date +%Y%m%d_%H%M%S).log"
RUNNER_PATTERN="run_rvq_hparam_fast_sweep.py --output-dir /data/bowen/rvq_diag --variants l6c256,l8c256,l5c512"
FINAL_SUMMARY=/data/bowen/rvq_diag/rvq_hparam_fast/sweep_v_l5c512_summary.json

echo "[watchdog] start $(date)" >> "$LOG"

while true; do
  if [[ -f "$FINAL_SUMMARY" ]]; then
    echo "[watchdog] final summary exists; exit $(date)" >> "$LOG"
    exit 0
  fi

  if pgrep -f "$RUNNER_PATTERN" >/dev/null 2>&1; then
    sleep 60
    continue
  fi

  echo "[watchdog] runner missing; resume $(date)" >> "$LOG"
  /home/bowen/anaconda3/envs/hw8/bin/python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
    --output-dir /data/bowen/rvq_diag \
    --variants l6c256,l8c256,l5c512 \
    --tensors K,V \
    --gpu-groups '0,1;2,3;4,5;6,7' \
    --max-parallel 4 \
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
