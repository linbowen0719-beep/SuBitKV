#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-rvq21_retrain_watchdog}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
ROOT=${ROOT:-/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk}
PY=${PY:-/home/bowen/anaconda3/envs/hw8/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_USED_MB=${MAX_USED_MB:-6000}
MAX_UTIL=${MAX_UTIL:-20}
POLL_SECONDS=${POLL_SECONDS:-60}
RUN_ID=${RUN_ID:-rvq21_retrained_$(date -u +%Y%m%dT%H%M%SZ)}
VBENCH_FLAG=${VBENCH_FLAG:---run-vbench}

mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PY' scripts/run_21chunk_rvq_retrain_watchdog.py \
    --output-dir '$ROOT' \
    --python '$PY' \
    --gpus '$GPUS' \
    --max-used-mb '$MAX_USED_MB' \
    --max-util '$MAX_UTIL' \
    --poll-seconds '$POLL_SECONDS' \
    --benchmark-run-id '$RUN_ID' \
    $VBENCH_FLAG \
    |& tee '$ROOT/logs/rvq21_retrain_watchdog_main.log'"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Log: $ROOT/logs/rvq21_retrain_watchdog_main.log"
echo "Benchmark run id: $RUN_ID"
