#!/bin/bash
set -uo pipefail
ulimit -c 0

RANK="${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/hy_i2v.env"
export SUBITKV_REPO_ROOT=/home/czh5/zhenghui/bowen/ICLR2026_RVQ-subbit-hy

module use /soft/modulefiles
module load cudatoolkit-standalone/12.8.1
module load gcc-native/14
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14
export PYTHONPATH="${SUBITKV_REPO_ROOT}:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HOME="${DATA_ROOT}/cache/huggingface" TORCH_HOME="${DATA_ROOT}/cache/torch"
export PATH="$(dirname "${SUBITKV_VBENCH}"):${PATH}"

"${SUBITKV_PYTHON}" "${SUBITKV_REPO_ROOT}/scripts/main/hyworldplay_i2v.py" \
  --run-id "${SUBITKV_RUN_ID}" --gpus 0,1,2,3 --shard-index "${RANK}" --num-shards 10 \
  --resume --execute evaluate --split test \
  --methods ours_rvq_avg0,ours_rvq_avg0p2_alt,ours_rvq_avg0p5
