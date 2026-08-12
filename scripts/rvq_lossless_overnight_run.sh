#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/bowen/Quant-VideoGen"
PY="/home/bowen/anaconda3/envs/hw8/bin/python"
OUT="/data/bowen/rvq_diag/lossless_overnight"
LOGDIR="${OUT}/logs"
mkdir -p "${LOGDIR}"
cd "${ROOT}"

configs=(l8c1024 l8c512 l8c256 l7c256 l6c256 l4c256 l8c64 l6c128)
gpus=(0 1 2 3 4 5 6 7)

run_batch() {
  local split="$1"
  local tensor="$2"
  for i in "${!configs[@]}"; do
    cfg="${configs[$i]}"
    gpu="${gpus[$i]}"
    echo "[launch] split=${split} tensor=${tensor} config=${cfg} gpu=${gpu}"
    "${PY}" scripts/rvq_lossless_overnight_explore.py \
      --config "${cfg}" \
      --tensor "${tensor}" \
      --split "${split}" \
      --gpu "${gpu}" \
      --out-dir "${OUT}" \
      --include-centroid-order \
      > "${LOGDIR}/${split}_${cfg}_${tensor}.log" 2>&1 &
  done
  wait
}

echo "[start] $(date --iso-8601=seconds)"
run_batch test K
run_batch test V
"${PY}" scripts/summarize_lossless_overnight.py --out-dir "${OUT}" --split test \
  > "${LOGDIR}/test_summary.log" 2>&1

# Optional train split gives a second view of the same codecs. It is longer
# because train has 20 prompts, so run it after the immediately useful test
# report exists.
run_batch train K
run_batch train V
"${PY}" scripts/summarize_lossless_overnight.py --out-dir "${OUT}" --split train \
  > "${LOGDIR}/train_summary.log" 2>&1

echo "[done] $(date --iso-8601=seconds)"
