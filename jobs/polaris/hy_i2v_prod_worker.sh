#!/bin/bash
# Per-rank worker for the prod-queue stages.  One rank per node, each driving
# that node's 4 GPUs through its own slice of the agreed 10% test set.
set -uo pipefail
ulimit -c 0

RANK="${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/hy_i2v.env"

module use /soft/modulefiles
module load conda
module load cudatoolkit-standalone/12.8.1
# Triton JIT-compiles a C helper at first kernel launch and picks up $CC; the
# default PrgEnv-nvidia hands it nvc, which rejects -Wno-psabi.  gcc-native is
# what the debug-queue script already loads and what build_env validated.
module load gcc-native/14
export CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14

export PYTHONPATH="${SUBITKV_REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${DATA_ROOT}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
# LPIPS pulls a 528 MB VGG from download.pytorch.org on first use; without this
# every node re-downloads it into its own home cache.
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

# The agreed pilot is dataset_row % 10 == 0.  Splitting those 99 rows over ten
# nodes needs a lower bound the CLI does not have, but nesting the modulus does
# it exactly: {r : r%100 in {0,10,...,90}} == {r : r%10 == 0}, verified against
# the frozen split (99 rows either way, 9-11 per rank).
SHARD=$(( RANK * 10 ))
NSHARDS=100

log() { printf '[rank %s | %s | %s] %s\n' "${RANK}" "$(hostname -s)" "$(date +%H:%M:%S)" "$*"; }
cd "${SUBITKV_REPO_ROOT}"

if [ "${RANK}" -ge 10 ]; then log "idle"; exit 0; fi
log "${STAGE}  shard ${SHARD}/${NSHARDS}"
"${SUBITKV_PYTHON}" scripts/main/hyworldplay_i2v.py \
  --run-id "${RUN_ID}" --gpus 0,1,2,3 \
  --shard-index "${SHARD}" --num-shards "${NSHARDS}" \
  --resume --execute "${STAGE}" --split test --methods all
rc=$?
log "${STAGE} rc=${rc}"
echo "${rc}" > "${OUT}/.status_${STAGE}_rank${RANK}"
exit 0
