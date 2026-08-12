#!/usr/bin/env bash
set -euo pipefail

cd /home/bowen/Quant-VideoGen

LOG_DIR=/data/bowen/rvq_diag/rvq_hparam_fast/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/extra_after_precision_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "[extra-after-precision] wait for current precision sweep $(date)" >> "$LOG"
while pgrep -af 'run_rvq_hparam_fast_sweep.py .*--variants l6c512,l4c1024,l5c1024' >/dev/null; do
  sleep 60
done

echo "[extra-after-precision] start GPU4-7 $(date)" >> "$LOG"
/home/bowen/anaconda3/envs/hw8/bin/python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
  --output-dir /data/bowen/rvq_diag \
  --variants l8c512,l6c1024,l8c1024 \
  --tensors K,V \
  --gpu-groups '4,5;6,7' \
  --max-parallel 2 \
  --worker-python /home/bowen/anaconda3/envs/hw8/bin/python \
  --dump-python /home/bowen/anaconda3/envs/hw8/bin/python \
  --eval-vector-sample 4096 \
  --query-sample 128 \
  --key-sample 2048 \
  --topk 32 \
  --eval-batch-size 8192 >> "$LOG" 2>&1
echo "[extra-after-precision] done rc=$? $(date)" >> "$LOG"
