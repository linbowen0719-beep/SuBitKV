#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-no_scale_variants_11chunk}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
PY=${PY:-/home/bowen/anaconda3/envs/hw8/bin/python}
DATA=${DATA:-/data/bowen/t2v_kv_bench_moviegen_20train_10test}
PROMPT_SOURCE=${PROMPT_SOURCE:-/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt}
CONFIG=${CONFIG:-$REPO/experiments/long_video/no_scale_variants_11chunk_10prompts.yaml}
RUN_ID=${RUN_ID:-no_scale_variants_11chunk_10prompt_20260707}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
LOG=${LOG:-$REPO/logs/no_scale_variants_11chunk_10prompt_20260707_master.log}
LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
SETTINGS=full_precision_cpu_offload,nonorm_l8c64_int4,first_recent_nonorm_l8c64_int4,nonorm_l8c64_online_kv_block_scale,first_recent_nonorm_l8c64_online_kv_block_scale,nonorm_l8c64_offline_kv_block_scale,first_recent_nonorm_l8c64_offline_kv_block_scale,nonorm_k_token_v_channel_l8c64,first_recent_nonorm_k_token_v_channel_l8c64,nonorm_progressive_token_l4_l16_avg7,first_recent_nonorm_progressive_token_l4_l16_avg7,nonorm_progressive_layer_l4_l16_avg7,first_recent_nonorm_progressive_layer_l4_l16_avg7,nonorm_progressive_block_l4_l16_avg7,first_recent_nonorm_progressive_block_l4_l16_avg7

count_codebooks() {
  local dir=$1
  if [[ -d "$dir" ]]; then
    find "$dir" -maxdepth 1 -name 'layer_*_head_*.pt' | wc -l
  else
    echo 0
  fi
}

pick_free_gpu() {
  while true; do
    local status
    status=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true)
    while IFS=',' read -r idx mem util; do
      idx=$(echo "$idx" | xargs)
      mem=$(echo "$mem" | xargs)
      util=$(echo "$util" | xargs)
      if [[ -n "$idx" && "$mem" -le 1000 && "$util" -le 5 ]]; then
        echo "$idx"
        return 0
      fi
    done <<< "$status"
    echo "[wait] no free GPU under 1000 MiB and 5% util; sleeping 60s" >&2
    sleep 60
  done
}

free_gpu_csv() {
  while true; do
    local status out
    status=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true)
    out=$(awk -F',' '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if (($2 + 0) <= 1000 && ($3 + 0) <= 5) {
          if (s == "") s = $1; else s = s "," $1;
        }
      }
      END { print s }
    ' <<< "$status")
    if [[ -n "$out" ]]; then
      echo "$out"
      return 0
    fi
    echo "[wait] no free GPU list under 1000 MiB and 5% util; sleeping 60s" >&2
    sleep 60
  done
}

run_on_free_gpu() {
  local gpu
  gpu=$(pick_free_gpu)
  echo "[gpu] using $gpu for $*"
  CUDA_VISIBLE_DEVICES="$gpu" "$@"
}

run_body() {
  cd "$REPO"
  mkdir -p "$(dirname "$LOG")" "$DATA/calibration" "$DATA/metrics"
  echo "[start] $(date -Is)"
  echo "[repo] $REPO"
  echo "[config] $CONFIG"

  if [[ $(count_codebooks "$DATA/nonorm_k_l16c64_codebooks") -lt 720 ]]; then
    ACTIVE_GPUS=$(free_gpu_csv)
    echo "[prereq] training K no-norm L16C64 codebooks"
    echo "[gpus] $ACTIVE_GPUS"
    "$PY" scripts/rvq/diag/run_rvq_diag.py \
      --prompt_source "$PROMPT_SOURCE" \
      --output_dir "$DATA" \
      --train_num 20 --test_num 10 --seed 42 \
      --first_stage_layers "$LAYERS" \
      --gpus "$ACTIVE_GPUS" \
      --n_spatial 880 --levels 16 --codebook_size 64 \
      --codebook_tensor K \
      --codebook_subdir nonorm_k_l16c64_codebooks \
      --metrics_subdir nonorm_k_l16c64_metrics \
      --indices_subdir nonorm_k_l16c64_indices \
      --stage_prefix nonorm_k_l16c64_ \
      --sample_size 50000 --kmeans_max_iters 20 \
      --input_norm_mode none --skip_dump --skip_eval \
      --eval_vector_sample 2048 --query_sample 96 --key_sample 1024 --topk 32 \
      --eval_batch_size 8192 --num_frames 173 --num_chunk 11 --image_path none --pose w-96 \
      --memory_frames 80 --temporal_context_size 76 --pred_latent_size 4 --offload_text_encoder
  else
    echo "[reuse] K no-norm L16C64 codebooks already complete"
  fi

  if [[ $(count_codebooks "$DATA/nonorm_v_l16c64_codebooks") -lt 720 ]]; then
    ACTIVE_GPUS=$(free_gpu_csv)
    echo "[prereq] training V no-norm L16C64 codebooks"
    echo "[gpus] $ACTIVE_GPUS"
    "$PY" scripts/rvq/diag/run_rvq_diag.py \
      --prompt_source "$PROMPT_SOURCE" \
      --output_dir "$DATA" \
      --train_num 20 --test_num 10 --seed 42 \
      --first_stage_layers "$LAYERS" \
      --gpus "$ACTIVE_GPUS" \
      --n_spatial 880 --levels 16 --codebook_size 64 \
      --codebook_tensor V \
      --codebook_subdir nonorm_v_l16c64_codebooks \
      --metrics_subdir nonorm_v_l16c64_metrics \
      --indices_subdir nonorm_v_l16c64_indices \
      --stage_prefix nonorm_v_l16c64_ \
      --sample_size 50000 --kmeans_max_iters 20 \
      --input_norm_mode none --skip_dump --skip_eval \
      --eval_vector_sample 2048 --query_sample 96 --key_sample 1024 --topk 32 \
      --eval_batch_size 8192 --num_frames 173 --num_chunk 11 --image_path none --pose w-96 \
      --memory_frames 80 --temporal_context_size 76 --pred_latent_size 4 --offload_text_encoder
  else
    echo "[reuse] V no-norm L16C64 codebooks already complete"
  fi

  if [[ $(count_codebooks "$DATA/grouped_direct_nonorm_v_l8c64_codebooks") -lt 720 ]]; then
    ACTIVE_GPUS=$(free_gpu_csv)
    echo "[prereq] training grouped/channel no-norm L8C64 codebooks"
    echo "[gpus] $ACTIVE_GPUS"
    "$PY" scripts/run_grouped_dim_rvq_bench.py \
      --output-dir "$DATA" \
      --configs l8c64 \
      --modes direct_nonorm \
      --tensors V \
      --gpus "$ACTIVE_GPUS" \
      --max-used-mb 1000 \
      --train-max-used-mb 1000 \
      --max-util 5 \
      --poll-seconds 60 \
      --sample-size 12000 \
      --kmeans-max-iters 20 \
      --max-retries 10 \
      --skip-offline --skip-video --skip-quality --skip-vbench
  else
    echo "[reuse] grouped/channel no-norm L8C64 V codebooks already complete"
  fi

  CALIB=$DATA/calibration/nonorm_l8c64_kv_block_scale_train4f_q96.pt
  if [[ ! -f "$CALIB" ]]; then
    echo "[prereq] building offline KV block calibration table"
    run_on_free_gpu "$PY" scripts/build_rvq_calibration_table.py \
      --dump-root "$DATA/kv_dump" \
      --split-json "$DATA/prompt_split.json" \
      --split train \
      --prompt-limit 20 \
      --layers all \
      --frame-limit 4 \
      --block-size 880 \
      --query-sample 96 \
      --batch-size 2048 \
      --seed 42 \
      --k-codebook-dir "$DATA/nonorm_k_l8c64_codebooks" \
      --v-codebook-dir "$DATA/nonorm_v_l8c64_codebooks" \
      --codebook-storage-dtype int4 \
      --scale-storage-dtype uint4 \
      --compute-dtype bf16 \
      --output "$CALIB" \
      --device cuda:0
  else
    echo "[reuse] calibration table already exists: $CALIB"
  fi

  THRESH_JSON=$DATA/metrics/nonorm_progressive_l4_l16_avg7_threshold.json
  if [[ ! -f "$THRESH_JSON" ]]; then
    echo "[prereq] tuning progressive L4-L16 threshold on train split"
    run_on_free_gpu "$PY" scripts/tune_progressive_rvq_threshold.py \
      --dump-root "$DATA/kv_dump" \
      --k-codebook-dir "$DATA/nonorm_k_l16c64_codebooks" \
      --v-codebook-dir "$DATA/nonorm_v_l16c64_codebooks" \
      --split train \
      --prompts 0,1,2,3 \
      --layers "$LAYERS" \
      --token-sample 1024 \
      --min-depth 4 \
      --max-depth 16 \
      --target-average-depth 7 \
      --batch-size 1024 \
      --seed 42 \
      --compute-dtype bf16 \
      --codebook-storage-dtype int4 \
      --output-json "$THRESH_JSON" \
      --output-csv "$DATA/metrics/nonorm_progressive_l4_l16_avg7_threshold.csv" \
      --device cuda:0
  else
    echo "[reuse] progressive threshold report already exists: $THRESH_JSON"
  fi

  "$PY" scripts/update_progressive_threshold_config.py \
    --config "$CONFIG" \
    --threshold-json "$THRESH_JSON"

  THRESH=$("$PY" - <<PY
import json
from pathlib import Path
print(json.loads(Path("$THRESH_JSON").read_text())["threshold"])
PY
)
  for granularity in token layer block; do
    echo "[microbench] progressive granularity=$granularity threshold=$THRESH"
    MB_OUT="$REPO/results/rvq_throughput/nonorm_progressive_l4_l16_avg7"
    MB_CSV="$MB_OUT/nonorm_progressive_${granularity}_l4_l16_avg7_throughput.csv"
    if [[ -f "$MB_CSV" ]]; then
      echo "[reuse] microbench already exists: $MB_CSV"
      continue
    fi
    run_on_free_gpu "$PY" scripts/benchmark_rvq_throughput.py \
      --root "$DATA" \
      --split train \
      --prompt-id 0 \
      --layers 0,15,29 \
      --tensors K,V \
      --chunk-start-frame 0 \
      --chunk-frames 4 \
      --k-codebook-dir "$DATA/nonorm_k_l16c64_codebooks" \
      --v-codebook-dir "$DATA/nonorm_v_l16c64_codebooks" \
      --compute-dtype bf16 \
      --codebook-storage-dtype int4 \
      --scale-storage-dtype uint4 \
      --progressive \
      --progressive-min-depth 4 \
      --progressive-max-depth 16 \
      --progressive-error-threshold "$THRESH" \
      --progressive-granularity "$granularity" \
      --progressive-block-size 880 \
      --batch-size 2048 \
      --warmup 2 \
      --repeat 5 \
      --out-dir "$MB_OUT" \
      --tag "nonorm_progressive_${granularity}_l4_l16_avg7" \
      --device cuda:0
  done

  SMOKE_DONE="$REPO/results/long_video/$RUN_ID.smoke_done"
  if [[ -f "$SMOKE_DONE" ]]; then
    echo "[reuse] smoke already completed: $(cat "$SMOKE_DONE")"
  else
    SMOKE_ID=no_scale_variants_11chunk_smoke_$(date +%Y%m%dT%H%M%S)
    SMOKE_GPU=$(pick_free_gpu)
    echo "[smoke] run_id=$SMOKE_ID"
    echo "[gpu] using $SMOKE_GPU for smoke benchmark_long_video"
    "$PY" scripts/benchmark_long_video.py \
      --config "$CONFIG" \
      --output-root "$REPO/results/long_video" \
      --run-id "$SMOKE_ID" \
      --settings "$SETTINGS" \
      --smoke \
      --execute \
      --allow-full-run \
      --skip-quality \
      --skip_vbench \
      --gpus "$SMOKE_GPU"
    echo "$SMOKE_ID" > "$SMOKE_DONE"
  fi

  FORMAL_START_EPOCH=$(date +%s)
  FINAL_TABLE="$REPO/results/long_video/$RUN_ID/final_14row_table.csv"
  if [[ -f "$FINAL_TABLE" ]]; then
    echo "[reuse] final table already exists: $FINAL_TABLE"
  else
    echo "[formal] launching 10-prompt watchdog run_id=$RUN_ID"
    "$PY" scripts/run_long_video_10prompt_watchdog.py \
      --config "$CONFIG" \
      --output-root "$REPO/results/long_video" \
      --run-id "$RUN_ID" \
      --settings "$SETTINGS" \
      --python "$PY" \
      --gpus "$GPUS" \
      --max-used-mb 1000 \
      --max-util 5 \
      --poll-seconds 60 \
      --max-retries 10 \
      --run-vbench

    "$PY" scripts/summarize_no_scale_variants_11chunk.py \
      --run-dir "$REPO/results/long_video/$RUN_ID" \
      --formal-start-epoch "$FORMAL_START_EPOCH"
  fi

  echo "[done] $(date -Is)"
}

if [[ "${1:-}" == "--inner" ]]; then
  run_body 2>&1 | tee -a "$LOG"
else
  mkdir -p "$(dirname "$LOG")"
  tmux new-session -d -s "$SESSION" "cd '$REPO' && bash '$0' --inner"
  echo "[tmux] session=$SESSION"
  echo "[log] $LOG"
fi
