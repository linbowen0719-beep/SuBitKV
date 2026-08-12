#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON="${PYTHON:-/home/bowen/QuantVid/quantvid/bin/python}"
MODEL_PATH="${MODEL_PATH:-/data/bowen/hunyuan8b/HunyuanVideo-1.5}"
ACTION_CKPT="${ACTION_CKPT:-/data/bowen/hf_cache_hunyuan8b/models--tencent--HY-WorldPlay/snapshots/f4c29235647707b571479a69b569e4166f9f5bf8/ar_distilled_action_model/model.safetensors}"
IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/assets/hyworld.png}"
OUTPUT_PATH="${OUTPUT_PATH:-$REPO_ROOT/results/hunyuan8b_hyvideo_bf16_61f}"

PROMPT="${PROMPT:-A first-person exploration video moving steadily through a detailed natural scene, maintaining geometric consistency and smooth camera motion.}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"
POSE="${POSE:-w-15}"
VIDEO_LENGTH="${VIDEO_LENGTH:-61}"
SEED="${SEED:-123}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
NUM_GPUS="${NUM_GPUS:-1}"
MASTER_PORT="${MASTER_PORT:-29573}"

RESOLUTION="${RESOLUTION:-480p}"
ASPECT_RATIO="${ASPECT_RATIO:-16:9}"
DTYPE="${DTYPE:-bf16}"
SR="${SR:-false}"
REWRITE="${REWRITE:-false}"
FEW_STEP="${FEW_STEP:-true}"
OFFLOADING="${OFFLOADING:-true}"
GROUP_OFFLOADING="${GROUP_OFFLOADING:-false}"
USE_SAGEATTN="${USE_SAGEATTN:-false}"
USE_FP8_GEMM="${USE_FP8_GEMM:-false}"
USE_VAE_PARALLEL="${USE_VAE_PARALLEL:-false}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-false}"
FP8_QUANT_TYPE="${FP8_QUANT_TYPE:-fp8-per-block}"
INCLUDE_PATTERNS="${INCLUDE_PATTERNS:-double_blocks}"
SAGE_BLOCKS_RANGE="${SAGE_BLOCKS_RANGE:-0-53}"

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
export CUDA_VISIBLE_DEVICES PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$REPO_ROOT/experiments/HY-WorldPlay${PYTHONPATH:+:$PYTHONPATH}"

cmd=(
  "$PYTHON" -m torch.distributed.run
  "--nproc_per_node=$NUM_GPUS"
  "--master_port=$MASTER_PORT"
  "$REPO_ROOT/experiments/HY-WorldPlay/hyvideo/generate.py"
  --prompt "$PROMPT"
  --negative_prompt "$NEGATIVE_PROMPT"
  --image_path "$IMAGE_PATH"
  --resolution "$RESOLUTION"
  --aspect_ratio "$ASPECT_RATIO"
  --video_length "$VIDEO_LENGTH"
  --seed "$SEED"
  --rewrite "$REWRITE"
  --sr "$SR"
  --pose "$POSE"
  --output_path "$OUTPUT_PATH"
  --model_path "$MODEL_PATH"
  --action_ckpt "$ACTION_CKPT"
  --few_step "$FEW_STEP"
  --num_inference_steps "$NUM_INFERENCE_STEPS"
  --model_type ar
  --dtype "$DTYPE"
  --offloading "$OFFLOADING"
  --group_offloading "$GROUP_OFFLOADING"
  --use_sageattn "$USE_SAGEATTN"
  --use_fp8_gemm "$USE_FP8_GEMM"
  --use_vae_parallel "$USE_VAE_PARALLEL"
  --enable_torch_compile "$ENABLE_TORCH_COMPILE"
  --quant_type "$FP8_QUANT_TYPE"
  --include_patterns "$INCLUDE_PATTERNS"
  --sage_blocks_range "$SAGE_BLOCKS_RANGE"
  "$@"
)

mkdir -p "$OUTPUT_PATH"
cd "$REPO_ROOT"

printf 'Running HunyuanVideo-1.5 / HY-WorldPlay via hyvideo:\n'
printf '  output: %s/gen.mp4\n' "$OUTPUT_PATH"
printf '  command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
