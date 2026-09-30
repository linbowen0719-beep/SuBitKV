#!/bin/bash
set -uo pipefail
ulimit -c 0

RANK="${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-0}}}"
CODE_ROOT=/home/czh5/zhenghui/bowen/ICLR2026_RVQ-subbit-hy
DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/polaris.env"
module use /soft/modulefiles
module load cudatoolkit-standalone/12.8.1
export CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDAHOSTCXX=/usr/bin/g++
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY="${http_proxy}" HTTPS_PROXY="${https_proxy}"

PY="${SUBITKV_VENV}/bin/python"
BUNDLE="${SUBITKV_DATA_ROOT}/paper_main_polaris_v1"
OUT="${SUBITKV_OUTPUT_ROOT}/self_forcing_rvq_subbit_20260904d"
SHARDS="${SUBITKV_NUM_SHARDS:-10}"
PER_RANK=$(((1003 + SHARDS - 1) / SHARDS))
START=$((RANK * PER_RANK))
STOP=$((START + PER_RANK))
[ "${STOP}" -gt 1003 ] && STOP=1003
if [ "${START}" -ge 1003 ]; then exit 0; fi
K_HASH=$("${PY}" -c "import json; print(json.load(open('${BUNDLE}/manifest.json'))['hashes']['self_forcing/nonorm_k_l16c64_codebooks'])")
V_HASH=$("${PY}" -c "import json; print(json.load(open('${BUNDLE}/manifest.json'))['hashes']['self_forcing/nonorm_v_l16c64_codebooks'])")
"${PY}" "${CODE_ROOT}/scripts/analysis/run_self_forcing_cross_model_jobs.py" \
  --output-root "${OUT}" --prompt-start "${START}" --prompt-count "$((STOP - START))" --gpus 0,1,2,3 \
  --methods adaptive_temporal_avg0,adaptive_temporal_avg0p2_alt,adaptive_temporal_avg0p5 \
  --prompt-file "${CODE_ROOT}/assets/MovieGenVideoBench.txt" --seed 0 \
  --model-config "${CODE_ROOT}/experiments/Self-Forcing/configs/self_forcing_dmd.yaml" \
  --checkpoint "${SUBITKV_CKPT_ROOT}/Self-Forcing/self_forcing_dmd.pt" \
  --k-codebooks "${BUNDLE}/self_forcing/nonorm_k_l16c64_codebooks" \
  --v-codebooks "${BUNDLE}/self_forcing/nonorm_v_l16c64_codebooks" \
  --k-codebook-hash "${K_HASH}" --v-codebook-hash "${V_HASH}" \
  --routing "${BUNDLE}/self_forcing/routing.json" --tree "${BUNDLE}/shared/tree_params.pt" \
  --v-params "${BUNDLE}/shared/progressive_v_params.pt" --profile "${BUNDLE}/shared/temporal_profile.json"
rc=$?
printf '%s\n' "${rc}" > "${OUT}/.status_subbit_rank${RANK}"
exit "${rc}"
