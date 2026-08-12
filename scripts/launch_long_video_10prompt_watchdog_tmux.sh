#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-rvq21_10prompt_watchdog}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
PY=${PY:-/home/bowen/anaconda3/envs/hw8/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_USED_MB=${MAX_USED_MB:-6000}
MAX_UTIL=${MAX_UTIL:-20}
POLL_SECONDS=${POLL_SECONDS:-60}
RUN_ID=${RUN_ID:-rvq21_retrained_10prompt_$(date -u +%Y%m%dT%H%M%SZ)}
VBENCH_FLAG=${VBENCH_FLAG:---run-vbench}
CONFIG=${CONFIG:-experiments/long_video/l8c64_int4_21chunk_retrained_10prompts.yaml}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/long_video}

mkdir -p "$REPO/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PY' scripts/run_long_video_10prompt_watchdog.py \
    --config '$CONFIG' \
    --output-root '$OUTPUT_ROOT' \
    --run-id '$RUN_ID' \
    --python '$PY' \
    --gpus '$GPUS' \
    --max-used-mb '$MAX_USED_MB' \
    --max-util '$MAX_UTIL' \
    --poll-seconds '$POLL_SECONDS' \
    $VBENCH_FLAG \
    |& tee 'logs/${RUN_ID}.log'"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Run id: $RUN_ID"
echo "Log: $REPO/logs/${RUN_ID}.log"
