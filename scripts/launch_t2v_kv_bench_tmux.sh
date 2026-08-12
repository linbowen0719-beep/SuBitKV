#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-t2v_kv_bench}"
REPO_ROOT="${REPO_ROOT:-/home/bowen/Quant-VideoGen}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/bowen/t2v_kv_bench_moviegen_20train_10test}"
PYTHON_BIN="${PYTHON_BIN:-/home/bowen/anaconda3/envs/hw8/bin/python}"

mkdir -p "${OUTPUT_DIR}/logs"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  exit 0
fi

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${REPO_ROOT}' && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '${PYTHON_BIN}' scripts/run_t2v_kv_bench.py \
   --gpus 2,3,4,5,6,7 \
   --output-dir '${OUTPUT_DIR}' \
   |& tee '${OUTPUT_DIR}/logs/main.log'"

echo "Started tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
echo "Log: ${OUTPUT_DIR}/logs/main.log"
