#!/bin/bash
set -uo pipefail
ulimit -c 0

RANK="${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
CODE_ROOT=/home/czh5/zhenghui/bowen/ICLR2026_RVQ-subbit-hy
DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/polaris.env"
module use /soft/modulefiles
module load cudatoolkit-standalone/12.8.1
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export PATH="$(dirname "${QVG_VBENCH}"):${PATH}"

PY="${QVG_VENV}/bin/python"
RUN="${QVG_OUTPUT_ROOT}/self_forcing_rvq_subbit_20260904d"
REF="${QVG_OUTPUT_ROOT}/moviegen-1003-polaris/self_forcing/bf16"
EVAL="${RUN}/evaluation_subbit"
METHODS=adaptive_temporal_avg0,adaptive_temporal_avg0p2_alt,adaptive_temporal_avg0p5

pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" \
    "${CODE_ROOT}/scripts/analysis/evaluate_self_forcing_cross_model_quality.py" \
    --run-root "${RUN}" --reference-root "${REF}" --methods "${METHODS}" \
    --device cuda:0 --shard-index "$((RANK * 4 + gpu))" --num-shards 40 \
    --prompt-count 1003 --seed 0 \
    >> "${EVAL}/quality_rank${RANK}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "${pid}" || rc=1; done
[ "${rc}" -eq 0 ] || exit "${rc}"

per_rank=101
start=$((RANK * per_rank))
stop=$((start + per_rank))
[ "${stop}" -gt 1003 ] && stop=1003
[ "${start}" -ge 1003 ] && exit 0
"${PY}" "${CODE_ROOT}/scripts/analysis/run_cross_model_vbench.py" \
  --run-root "${RUN}" --output-root "${EVAL}/vbench/shard_${RANK}" \
  --methods "${METHODS}" --gpus 0,1,2,3 --prompt-start "${start}" \
  --prompt-count "$((stop - start))" --seed 0 \
  --dimensions subject_consistency,background_consistency,imaging_quality,aesthetic_quality
