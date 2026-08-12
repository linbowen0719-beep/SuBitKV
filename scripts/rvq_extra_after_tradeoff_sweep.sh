#!/usr/bin/env bash
set -euo pipefail

cd /home/bowen/Quant-VideoGen

LOG_DIR=/data/bowen/rvq_diag/rvq_hparam_fast/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/extra_after_tradeoff_sweep_$(date +%Y%m%d_%H%M%S).log"

echo "[extra-after-tradeoff] wait for current tradeoff sweep $(date)" >> "$LOG"
while pgrep -af 'run_rvq_hparam_fast_sweep.py .*--variants l2c512,l3c512,l3c1024,l6c128' >/dev/null; do
  sleep 60
done

echo "[extra-after-tradeoff] start GPU1 $(date)" >> "$LOG"
/home/bowen/anaconda3/envs/hw8/bin/python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
  --output-dir /data/bowen/rvq_diag \
  --variants l1c512,l1c1024,l2c1024,l4c64,l5c128,l8c64 \
  --tensors K,V \
  --gpu-groups '1' \
  --max-parallel 1 \
  --worker-python /home/bowen/anaconda3/envs/hw8/bin/python \
  --dump-python /home/bowen/anaconda3/envs/hw8/bin/python \
  --eval-vector-sample 4096 \
  --query-sample 128 \
  --key-sample 2048 \
  --topk 32 \
  --eval-batch-size 8192 >> "$LOG" 2>&1
echo "[extra-after-tradeoff] done rc=$? $(date)" >> "$LOG"
