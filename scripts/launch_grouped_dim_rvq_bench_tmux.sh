#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-grouped_dim_rvq_bench}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
ROOT=${ROOT:-/data/bowen/t2v_kv_bench_moviegen_20train_10test}
PYTHON=${PYTHON:-/home/bowen/anaconda3/envs/hw8/bin/python}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

mkdir -p "$ROOT/logs/grouped_dim_rvq"

tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$PYTHON' scripts/run_grouped_dim_rvq_bench.py \
    --output-dir '$ROOT' \
    --python '$PYTHON' \
    --gpus '$GPUS' \
    --configs l8c64,l8c256 \
    --modes direct,hadamard \
    --train-max-used-mb 24000 \
    --sample-size 12000 \
    --kmeans-max-iters 20 \
    --seed 42 \
    --num-prompts 10 \
    |& tee '$ROOT/logs/grouped_dim_rvq/main.log'"

echo "started tmux session: $SESSION"
echo "log: $ROOT/logs/grouped_dim_rvq/main.log"
