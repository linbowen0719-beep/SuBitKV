#!/bin/bash
# One rank owns dataset_row % 100 and runs the three requested sub-bit arms.
set -uo pipefail
ulimit -c 0

RANK="${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
SHARDS="${QVG_NUM_SHARDS:-10}"
METHODS="${QVG_METHODS:-ours_rvq_avg0,ours_rvq_avg0p2_alt,ours_rvq_avg0p5}"
METHOD_TAG="${METHODS//,/__}"
DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/hy_i2v.env"
export QVG_REPO_ROOT=/home/czh5/zhenghui/bowen/ICLR2026_RVQ-subbit-hy

module use /soft/modulefiles
module load cudatoolkit-standalone/12.8.1
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export PYTHONPATH="${QVG_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY="${http_proxy}" HTTPS_PROXY="${https_proxy}"

OUT="${QVG_OUTPUT_ROOT}/hyworldplay-moviegen-i2v-20-973/${QVG_RUN_ID}"
"${QVG_PYTHON}" "${QVG_REPO_ROOT}/scripts/main/hyworldplay_i2v.py" \
  --run-id "${QVG_RUN_ID}" --gpus 0,1,2,3 --shard-index "${RANK}" --num-shards "${SHARDS}" \
  --resume --execute generate --split test \
  --methods "${METHODS}"
rc=$?
printf '%s\n' "${rc}" > "${OUT}/.status_subbit_${METHOD_TAG}_rank${RANK}"
exit "${rc}"
