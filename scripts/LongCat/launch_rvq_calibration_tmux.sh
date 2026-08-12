#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/bowen/Quant-VideoGen}
RVQ_ROOT=${RVQ_ROOT:-/data/bowen/longcat_rvq_l8c64}
PROMPT_FILE=${PROMPT_FILE:-/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_train.txt}
SESSION=${SESSION:-longcat_rvq_calibration}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
CALIBRATION_CHUNK=${CALIBRATION_CHUNK:-8}
CALIBRATION_SAMPLES=${CALIBRATION_SAMPLES:-2500}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}

if [[ -x /home/bowen/anaconda3/envs/HunyuanVideo/bin/python ]]; then
  PY=${PY:-/home/bowen/anaconda3/envs/HunyuanVideo/bin/python}
else
  PY=${PY:-$(command -v python)}
fi

cd "$REPO"
"$PY" scripts/LongCat/prepare_rvq_codebooks.py \
  --stage init \
  --root "$RVQ_ROOT" \
  --prompt-file "$PROMPT_FILE"

NUM_PROMPTS=$(awk 'NF {count++} END {print count+0}' "$RVQ_ROOT/prompts_train.txt")
SESSION="$SESSION" \
ROOT="$RVQ_ROOT/calibration_runs" \
PY="$PY" \
PROMPT_FILE="$RVQ_ROOT/prompts_train.txt" \
NUM_PROMPTS="$NUM_PROMPTS" \
METHODS=bf16_reference \
GPUS="$GPUS" \
MAX_ATTEMPTS="$MAX_ATTEMPTS" \
MAX_CONTINUATION_CHUNKS="$CALIBRATION_CHUNK" \
AUTO_MAX_CHUNKS=0 \
VBENCH_FLAG= \
RVQ_CALIBRATION_DIR="$RVQ_ROOT" \
RVQ_CALIBRATION_SPLIT=train \
RVQ_CALIBRATION_CHUNK="$CALIBRATION_CHUNK" \
RVQ_CALIBRATION_SAMPLES="$CALIBRATION_SAMPLES" \
bash scripts/LongCat/launch_chunk5_watchdog_tmux.sh
