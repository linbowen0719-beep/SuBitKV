#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bowen/Quant-VideoGen"
PY="/home/bowen/anaconda3/envs/hw8/bin/python"
PROMPTS="/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt"
OUT="/data/bowen/rvq_diag_t2v_moviegen_l4c256"
LAYERS="$(seq -s, 0 29)"
GPUS="0,1,2,3,4,5,6,7"

mkdir -p "$OUT"
cd "$ROOT"

echo "[queue] $(date --iso-8601=seconds) waiting for rvq_lossless_overnight to finish"
while tmux has-session -t rvq_lossless_overnight 2>/dev/null; do
  sleep 300
done
echo "[queue] $(date --iso-8601=seconds) overnight session finished"

if [[ ! -f /data/bowen/rvq_diag/lossless_overnight/train_summary.md ]]; then
  echo "[queue] warning: train_summary.md not found; continuing because tmux session ended"
fi

COMMON_ARGS=(
  --prompt_source "$PROMPTS"
  --output_dir "$OUT"
  --train_num 20
  --test_num 5
  --seed 42
  --first_stage_layers "$LAYERS"
  --gpus "$GPUS"
  --n_spatial 880
  --levels 4
  --codebook_size 256
  --sample_size 50000
  --kmeans_max_iters 20
  --eval_splits train,test
  --num_frames 173
  --num_chunk 11
  --num_inference_steps 50
  --pose "w-96"
  --memory_frames 80
  --temporal_context_size 76
  --pred_latent_size 4
  --image_path none
  --offload_text_encoder
  --dump_python "$PY"
  --worker_python "$PY"
)

echo "[run] $(date --iso-8601=seconds) T2V MovieGen L4C256 K dump/train/eval"
"$PY" scripts/rvq/diag/run_rvq_diag.py \
  "${COMMON_ARGS[@]}" \
  --codebook_tensor K \
  --codebook_subdir codebooks \
  --metrics_subdir metrics \
  --indices_subdir indices

echo "[run] $(date --iso-8601=seconds) T2V MovieGen L4C256 V train/eval using existing dump"
"$PY" scripts/rvq/diag/run_rvq_diag.py \
  "${COMMON_ARGS[@]}" \
  --codebook_tensor V \
  --codebook_subdir v_codebooks \
  --metrics_subdir v_metrics \
  --indices_subdir v_indices \
  --stage_prefix v_ \
  --skip_dump

echo "[summary] $(date --iso-8601=seconds)"
"$PY" scripts/summarize_t2v_moviegen_l4c256.py --output-dir "$OUT"
echo "[done] $(date --iso-8601=seconds)"
