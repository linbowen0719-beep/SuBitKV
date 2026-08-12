#!/usr/bin/env bash
set -euo pipefail

cd /home/bowen/Quant-VideoGen

LOG_DIR=/data/bowen/rvq_diag/rvq_hparam_fast/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/extra_precision_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "[extra-precision] start $(date)" >> "$LOG"
/home/bowen/anaconda3/envs/hw8/bin/python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
  --output-dir /data/bowen/rvq_diag \
  --variants l6c512,l4c1024,l5c1024 \
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
echo "[extra-precision] done rc=$? $(date)" >> "$LOG"
