#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/bowen/Quant-VideoGen}
RVQ_ROOT=${RVQ_ROOT:-/data/bowen/longcat_rvq_l8c64}
FORMAL_ROOT=${FORMAL_ROOT:-/data/bowen/longcat_chunk5}
TRAIN_PROMPTS=${TRAIN_PROMPTS:-/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_train.txt}
TEST_PROMPTS=${TEST_PROMPTS:-/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_test.txt}
CALIBRATION_SESSION=${CALIBRATION_SESSION:-longcat_rvq_calibration}
FORMAL_SESSION=${FORMAL_SESSION:-longcat_rvq_formal}
BASELINE_SESSION=${BASELINE_SESSION:-longcat_chunk5_watchdog_agent}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SEED=${SEED:-123}
LONGCAT_REPO=${LONGCAT_REPO:-$REPO/experiments/LongCat}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/bowen/LongCat-Video/weights/LongCat-Video}
SMOKE_MAX_ATTEMPTS=${SMOKE_MAX_ATTEMPTS:-3}
EMPTY_MEM_MB=${EMPTY_MEM_MB:-100}
EMPTY_UTIL=${EMPTY_UTIL:-5}
POLL_SECONDS=${POLL_SECONDS:-60}

if [[ -x /home/bowen/anaconda3/envs/HunyuanVideo/bin/python ]]; then
  PY=${PY:-/home/bowen/anaconda3/envs/HunyuanVideo/bin/python}
else
  PY=${PY:-$(command -v python)}
fi

wait_for_empty_gpu() {
  local allowed=",${GPUS},"
  local index memory utilization
  while true; do
    while IFS=, read -r index memory utilization; do
      index=${index//[[:space:]]/}
      memory=${memory//[[:space:]]/}
      utilization=${utilization//[[:space:]]/}
      if [[ "$allowed" == *",${index},"* ]] && \
         (( memory <= EMPTY_MEM_MB && utilization <= EMPTY_UTIL )); then
        if flock -n "/tmp/subitkv_longcat_gpu_${index}.lock" -c true; then
          printf '%s\n' "$index"
          return 0
        fi
      fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    echo "[wait] no empty GPU for RVQ smoke; retrying in ${POLL_SECONDS}s" >&2
    sleep "$POLL_SECONDS"
  done
}

wait_for_bf16_references() {
  local expected=$1
  local missing prompt_index status
  while true; do
    missing=0
    for prompt_index in $(seq 0 $((expected - 1))); do
      status="$FORMAL_ROOT/bf16_reference/prompt_$(printf '%04d' "$prompt_index")_seed_${SEED}/status.json"
      if [[ ! -f "$status" ]] || ! "$PY" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "complete" else 1)' "$status"; then
        missing=$((missing + 1))
      fi
    done
    if (( missing == 0 )); then
      return 0
    fi
    echo "[wait] missing BF16 references: ${missing}/${expected}"
    sleep "$POLL_SECONDS"
  done
}

smoke_tag() {
  local inputs=(
    scripts/LongCat/run_chunk5_experiment.py
    scripts/LongCat/verify_rvq_smoke.py
    experiments/LongCat/longcat_video/modules/utils.py
    subitkv/compress.py
    subitkv/uncompress.py
    subitkv/rvq.py
    "$RVQ_ROOT/k_codebooks/manifest.json"
    "$RVQ_ROOT/v_codebooks/manifest.json"
  )
  local file
  for file in "${inputs[@]}"; do
    [[ -f "$file" ]] || { echo "Missing RVQ smoke signature input: $file" >&2; return 1; }
  done
  sha256sum "${inputs[@]}" | sha256sum | cut -c1-16
}

run_rvq_smoke() {
  local method=$1
  local chunks=$2
  local smoke_root=$3
  local run_dir="$smoke_root/$method/prompt_0000_seed_${SEED}"
  local log_dir="$smoke_root/logs"
  local attempt gpu log_path lease_fd

  mkdir -p "$log_dir"
  if "$PY" scripts/LongCat/verify_rvq_smoke.py \
      --run-dir "$run_dir" --method "$method" --expected-chunks "$chunks" >/dev/null 2>&1; then
    echo "[smoke] reuse verified run: $run_dir"
    return 0
  fi

  for attempt in $(seq 1 "$SMOKE_MAX_ATTEMPTS"); do
    while true; do
      gpu=$(wait_for_empty_gpu)
      exec {lease_fd}>"/tmp/subitkv_longcat_gpu_${gpu}.lock"
      if flock -n "$lease_fd"; then
        break
      fi
      exec {lease_fd}>&-
      echo "[wait] GPU $gpu was leased before RVQ smoke launch; retrying" >&2
      sleep "$POLL_SECONDS"
    done
    log_path="$log_dir/${method}_attempt${attempt}_gpu${gpu}.log"
    echo "[smoke] method=$method chunks=$chunks attempt=$attempt gpu=$gpu"
    if CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$PY" scripts/LongCat/run_chunk5_experiment.py \
        --longcat-repo "$LONGCAT_REPO" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --output-dir "$smoke_root" \
        --method "$method" \
        --prompt-file "$TEST_PROMPTS" \
        --prompt-index 0 \
        --seed "$SEED" \
        --max-continuation-chunks "$chunks" \
        --num-inference-steps 50 \
        --k-rvq-codebook-dir "$RVQ_ROOT/k_codebooks" \
        --v-rvq-codebook-dir "$RVQ_ROOT/v_codebooks" \
        --reference-dir "$FORMAL_ROOT/bf16_reference/prompt_0000_seed_${SEED}" \
        --reuse-reference-seed \
        --no-save-npy |& tee -a "$log_path"; then
      if "$PY" scripts/LongCat/verify_rvq_smoke.py \
          --run-dir "$run_dir" --method "$method" --expected-chunks "$chunks"; then
        echo "[smoke] passed method=$method attempt=$attempt"
        flock -u "$lease_fd"
        exec {lease_fd}>&-
        return 0
      fi
      echo "[smoke] verification failed method=$method attempt=$attempt" >&2
    else
      echo "[smoke] runner failed method=$method attempt=$attempt" >&2
    fi
    flock -u "$lease_fd"
    exec {lease_fd}>&-
  done
  echo "[smoke] exhausted attempts method=$method" >&2
  return 1
}

cd "$REPO"
while tmux has-session -t "$CALIBRATION_SESSION" 2>/dev/null; do
  echo "[wait] calibration watchdog is still running"
  sleep 60
done

wait_for_bf16_references 10

while tmux has-session -t "$BASELINE_SESSION" 2>/dev/null; do
  echo "[wait] baseline/QVG watchdog is still running"
  sleep "$POLL_SECONDS"
done

"$PY" scripts/LongCat/prepare_rvq_codebooks.py \
  --stage train \
  --root "$RVQ_ROOT" \
  --prompt-file "$TRAIN_PROMPTS" \
  --gpus auto

"$PY" scripts/LongCat/prepare_rvq_codebooks.py \
  --stage verify \
  --root "$RVQ_ROOT" \
  --prompt-file "$TRAIN_PROMPTS"

SMOKE_ROOT="$RVQ_ROOT/smoke_runs/$(smoke_tag)"
echo "[smoke] root=$SMOKE_ROOT"
run_rvq_smoke rvq_l8c64_int4 1 "$SMOKE_ROOT"
run_rvq_smoke first_recent_rvq_l8c64_int4 3 "$SMOKE_ROOT"

SESSION="$FORMAL_SESSION" \
ROOT="$FORMAL_ROOT" \
PY="$PY" \
PROMPT_FILE="$TEST_PROMPTS" \
PROMPT_START=0 \
NUM_PROMPTS=10 \
METHODS=bf16_reference,first_recent_l8c64_int4,rvq_l8c64_int4,first_recent_rvq_l8c64_int4,qvg_int4,qvg_int2 \
GPUS="$GPUS" \
MAX_ATTEMPTS=3 \
EMPTY_MEM_MB="$EMPTY_MEM_MB" \
EMPTY_UTIL="$EMPTY_UTIL" \
POLL_SECONDS="$POLL_SECONDS" \
MAX_CONTINUATION_CHUNKS=16 \
AUTO_MAX_CHUNKS=1 \
VBENCH_FLAG=--run-vbench \
bash scripts/LongCat/launch_chunk5_watchdog_tmux.sh
