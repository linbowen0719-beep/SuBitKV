#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROMPT="${PROMPT:-A paved pathway leads towards a stone arch bridge spanning a calm body of water.  Lush green trees and foliage line the path and the far bank of the water. A traditional-style pavilion with a tiered, reddish-brown roof sits on the far shore. The water reflects the surrounding greenery and the sky.  The scene is bathed in soft, natural light, creating a tranquil and serene atmosphere. The pathway is composed of large, rectangular stones, and the bridge is constructed of light gray stone.  The overall composition emphasizes the peaceful and harmonious nature of the landscape.}"

IMAGE_PATH="${IMAGE_PATH:-assets/hyworld.png}"
POSE="${POSE:-w-8,s-8,a-8,d-8,up-8,down-8}"
QVG_CKPT_ROOT="${QVG_CKPT_ROOT:-$REPO_ROOT/ckpts}"
QVG_CACHE_ROOT="${QVG_CACHE_ROOT:-$REPO_ROOT/.cache}"
HYWORLDPLAY_CKPT_ROOT="${HYWORLDPLAY_CKPT_ROOT:-$QVG_CKPT_ROOT/HY-WorldPlay}"
export HF_HOME="${HF_HOME:-$QVG_CACHE_ROOT/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
: "${PYTORCH_ALLOC_CONF:=expandable_segments:True}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"
: "${HYWORLDPLAY_OFFLOAD_FOR_DECODE:=1}"
: "${HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE:=1}"
: "${HYWORLDPLAY_ENABLE_VAE_TILING:=0}"
export PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF HYWORLDPLAY_OFFLOAD_FOR_DECODE HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE HYWORLDPLAY_ENABLE_VAE_TILING

MEMORY_FRAMES="${MEMORY_FRAMES:-48}"
TEMPORAL_CONTEXT_SIZE="${TEMPORAL_CONTEXT_SIZE:-44}"
PRED_LATENT_SIZE="${PRED_LATENT_SIZE:-4}"
NUM_CHUNK="${NUM_CHUNK:-12}"
NUM_FRAMES="${NUM_FRAMES:-961}"

#########################################################
# Quantization Configuration
#########################################################
quant_type="${QUANT_TYPE:-triton-nstages-kmeans-int2}"
# quant_type="triton-nstages-kmeans-int4"
cache_num_k_centroids="${CACHE_NUM_K_CENTROIDS:-256}"
cache_num_v_centroids="${CACHE_NUM_V_CENTROIDS:-256}"
kmeans_max_iters="${KMEANS_MAX_ITERS:-2}"
quant_block_size="${QUANT_BLOCK_SIZE:-64}"
num_prq_stages="${NUM_PRQ_STAGES:-1}"

quant_dir=${quant_type}_${quant_block_size}/kc_${cache_num_k_centroids}_vc_${cache_num_v_centroids}_nstages_${num_prq_stages}
output_folder="${OUTPUT_FOLDER:-results/hyworldplay/${quant_dir}}"
SAVE_FULL_PRECISION_KV_CACHE="${SAVE_FULL_PRECISION_KV_CACHE:-0}"
KV_CACHE_SAVE_DIR="${KV_CACHE_SAVE_DIR:-$output_folder/full_precision_kv_cache}"
KV_CACHE_SAVE_MODE="${KV_CACHE_SAVE_MODE:-new}"

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/experiments/HY-WorldPlay${PYTHONPATH:+:$PYTHONPATH}"
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib/stubs${LIBRARY_PATH:+:$LIBRARY_PATH}

kv_cache_save_args=()
if [[ "$SAVE_FULL_PRECISION_KV_CACHE" == "1" || "$SAVE_FULL_PRECISION_KV_CACHE" == "true" ]]; then
  kv_cache_save_args+=(
    --save_full_precision_kv_cache
    --kv_cache_save_dir "$KV_CACHE_SAVE_DIR"
    --kv_cache_save_mode "$KV_CACHE_SAVE_MODE"
  )
fi

torchrun --nproc_per_node="$NPROC_PER_NODE" --standalone "$REPO_ROOT/experiments/HY-WorldPlay/wan/generate.py" \
  --num_frames "$NUM_FRAMES" \
  --input "$PROMPT" \
  --image_path "$IMAGE_PATH" \
  --num_chunk "$NUM_CHUNK" \
  --pose "$POSE" \
  --ar_model_path "$HYWORLDPLAY_CKPT_ROOT/wan_transformer" \
  --ckpt_path "$HYWORLDPLAY_CKPT_ROOT/wan_distilled_model/model.pt" \
  --offload_text_encoder \
  --out "$output_folder" \
  --memory_frames $MEMORY_FRAMES \
  --temporal_context_size $TEMPORAL_CONTEXT_SIZE \
  --pred_latent_size $PRED_LATENT_SIZE \
  --quant_type $quant_type \
  --quant_block_size $quant_block_size \
  --cache_num_k_centroids $cache_num_k_centroids \
  --cache_num_v_centroids $cache_num_v_centroids \
  --kmeans_max_iters $kmeans_max_iters \
  --num_prq_stages $num_prq_stages \
  "${kv_cache_save_args[@]}"
