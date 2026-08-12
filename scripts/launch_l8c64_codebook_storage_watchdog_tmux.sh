#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-l8c64_codebook_storage_watchdog}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
ROOT=${ROOT:-/data/bowen/t2v_kv_bench_moviegen_20train_10test}
PY=${PY:-/home/bowen/anaconda3/envs/hw8/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_USED_MB=${MAX_USED_MB:-6000}
MAX_UTIL=${MAX_UTIL:-20}
POLL_SECONDS=${POLL_SECONDS:-60}

mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PY' scripts/run_l8c64_codebook_storage_watchdog.py \
    --output-dir '$ROOT' \
    --python '$PY' \
    --gpus '$GPUS' \
    --max-used-mb '$MAX_USED_MB' \
    --max-util '$MAX_UTIL' \
    --poll-seconds '$POLL_SECONDS' \
    |& tee '$ROOT/logs/l8c64_codebook_storage_watchdog_main.log'"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Log: $ROOT/logs/l8c64_codebook_storage_watchdog_main.log"
