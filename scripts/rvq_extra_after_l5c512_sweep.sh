#!/usr/bin/env bash
set -euo pipefail

cd /home/bowen/Quant-VideoGen

LOG_DIR=/data/bowen/rvq_diag/rvq_hparam_fast/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/extra_after_l5c512_sweep_$(date +%Y%m%d_%H%M%S).log"
READY=/data/bowen/rvq_diag/rvq_hparam_fast/sweep_v_l5c512_summary.json

echo "[extra-after-l5c512] wait for $READY $(date)" >> "$LOG"
while [[ ! -f "$READY" ]]; do
  sleep 60
done

echo "[extra-after-l5c512] start GPU2,3 $(date)" >> "$LOG"
/home/bowen/anaconda3/envs/hw8/bin/python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
  --output-dir /data/bowen/rvq_diag \
  --variants l7c256,l7c512 \
  --tensors K,V \
  --gpu-groups '2,3' \
  --max-parallel 1 \
  --worker-python /home/bowen/anaconda3/envs/hw8/bin/python \
  --dump-python /home/bowen/anaconda3/envs/hw8/bin/python \
  --eval-vector-sample 4096 \
  --query-sample 128 \
  --key-sample 2048 \
  --topk 32 \
  --eval-batch-size 8192 >> "$LOG" 2>&1
echo "[extra-after-l5c512] done rc=$? $(date)" >> "$LOG"
