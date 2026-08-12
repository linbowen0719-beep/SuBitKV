#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-longcat_chunk5_watchdog}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
ROOT=${ROOT:-/data/bowen/longcat_chunk5}
if [[ -z "${PY:-}" ]]; then
  if [[ -x /home/bowen/anaconda3/envs/HunyuanVideo/bin/python ]]; then
    PY=/home/bowen/anaconda3/envs/HunyuanVideo/bin/python
  else
    PY=$(command -v python)
  fi
fi
LONGCAT_REPO=${LONGCAT_REPO:-$REPO/experiments/LongCat}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/bowen/LongCat-Video/weights/LongCat-Video}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
EMPTY_MEM_MB=${EMPTY_MEM_MB:-100}
EMPTY_UTIL=${EMPTY_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-60}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
STALL_SECONDS=${STALL_SECONDS:-1800}
SEED=${SEED:-123}
PROMPT_FILE=${PROMPT_FILE:-}
PROMPT_START=${PROMPT_START:-0}
NUM_PROMPTS=${NUM_PROMPTS:-1}
MAX_CONTINUATION_CHUNKS=${MAX_CONTINUATION_CHUNKS:-16}
AUTO_MAX_CHUNKS=${AUTO_MAX_CHUNKS:-1}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
METHODS=${METHODS:-bf16_reference,first_recent_l8c64_int4,qvg_int4,qvg_int2}
VBENCH_FLAG=${VBENCH_FLAG---run-vbench}
RVQ_CALIBRATION_DIR=${RVQ_CALIBRATION_DIR:-}
RVQ_CALIBRATION_SPLIT=${RVQ_CALIBRATION_SPLIT:-train}
RVQ_CALIBRATION_CHUNK=${RVQ_CALIBRATION_CHUNK:-8}
RVQ_CALIBRATION_SAMPLES=${RVQ_CALIBRATION_SAMPLES:-2048}

mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

cmd=(
  "$PY" scripts/LongCat/watch_chunk5_experiment.py
  --output-dir "$ROOT"
  --python "$PY"
  --longcat-repo "$LONGCAT_REPO"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --gpus "$GPUS"
  --empty-mem-mb "$EMPTY_MEM_MB"
  --empty-util "$EMPTY_UTIL"
  --poll-seconds "$POLL_SECONDS"
  --max-attempts "$MAX_ATTEMPTS"
  --stall-seconds "$STALL_SECONDS"
  --seed "$SEED"
  --prompt-start "$PROMPT_START"
  --num-prompts "$NUM_PROMPTS"
  --max-continuation-chunks "$MAX_CONTINUATION_CHUNKS"
  --num-inference-steps "$NUM_INFERENCE_STEPS"
  --methods "$METHODS"
)

if [[ -n "$PROMPT_FILE" ]]; then
  cmd+=(--prompt-file "$PROMPT_FILE")
fi

if [[ -n "$RVQ_CALIBRATION_DIR" ]]; then
  cmd+=(
    --rvq-calibration-dir "$RVQ_CALIBRATION_DIR"
    --rvq-calibration-split "$RVQ_CALIBRATION_SPLIT"
    --rvq-calibration-chunk "$RVQ_CALIBRATION_CHUNK"
    --rvq-calibration-samples "$RVQ_CALIBRATION_SAMPLES"
  )
fi

case "${AUTO_MAX_CHUNKS,,}" in
  1|true|yes|on) cmd+=(--auto-max-chunks) ;;
esac

if [[ -n "$VBENCH_FLAG" ]]; then
  read -r -a vbench_args <<< "$VBENCH_FLAG"
  cmd+=("${vbench_args[@]}")
fi

printf -v quoted_cmd '%q ' "${cmd[@]}"
printf -v quoted_repo '%q' "$REPO"
printf -v quoted_log '%q' "$ROOT/logs/chunk5_watchdog_main.log"

tmux new-session -d -s "$SESSION" \
  "cd $quoted_repo && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $quoted_cmd |& tee -a $quoted_log"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
echo "Log: $ROOT/logs/chunk5_watchdog_main.log"
echo "Output root: $ROOT"
