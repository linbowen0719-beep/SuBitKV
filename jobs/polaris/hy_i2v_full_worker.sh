#!/bin/bash
# Per-rank worker for the FULL 973-row run: 100 ranks, one per node, rank i
# drives its node's 4 GPUs over rows where dataset_row % 100 == i.  Unlike the
# 10% worker there is no nesting trick and no idle rank: shard i of 100 IS the
# coverage, ranks 0-72 get 10 rows, 73-99 get 9.
set -uo pipefail
ulimit -c 0

RANK="${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
export DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/hy_i2v.env"

module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/12.8.1
module load gcc-native/14
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14

export PYTHONPATH="${SUBITKV_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${DATA_ROOT}/cache/torch"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY="${http_proxy}" HTTPS_PROXY="${https_proxy}"
export PATH="$(dirname "${SUBITKV_VBENCH}"):${PATH}"

STAGE="${SUBITKV_STAGE:?}"
RUN_ID="${SUBITKV_RUN_ID:-hy-main-r1}"
OUT="${SUBITKV_OUTPUT_ROOT}/hyworldplay-moviegen-i2v-20-973/${RUN_ID}"
mkdir -p "${OUT}"

log() { printf '[rank %s | %s | %s] %s\n' "${RANK}" "$(hostname -s)" "$(date +%H:%M:%S)" "$*"; }
cd "${SUBITKV_REPO_ROOT}"

log "${STAGE}  shard ${RANK}/100"
"${SUBITKV_PYTHON}" scripts/main/hyworldplay_i2v.py \
  --run-id "${RUN_ID}" --gpus 0,1,2,3 \
  --shard-index "${RANK}" --num-shards 100 \
  --resume --execute "${STAGE}" --split test --methods all
rc=$?
log "${STAGE} rc=${rc}"
echo "${rc}" > "${OUT}/.status_full_${STAGE}_rank${RANK}"
exit 0
