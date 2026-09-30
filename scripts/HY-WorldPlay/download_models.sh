#!/bin/bash
set -euo pipefail

BASE=https://huggingface.co/tencent/HY-WorldPlay/resolve/main
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBITKV_CKPT_ROOT="${SUBITKV_CKPT_ROOT:-$REPO_ROOT/ckpts}"
HYWORLDPLAY_CKPT_ROOT="${HYWORLDPLAY_CKPT_ROOT:-$SUBITKV_CKPT_ROOT/HY-WorldPlay}"

mkdir -p "$HYWORLDPLAY_CKPT_ROOT/wan_transformer" "$HYWORLDPLAY_CKPT_ROOT/wan_distilled_model"

wget -c -O "$HYWORLDPLAY_CKPT_ROOT/wan_transformer/config.json" "$BASE/wan_transformer/config.json"
wget -c -O "$HYWORLDPLAY_CKPT_ROOT/wan_transformer/diffusion_pytorch_model.safetensors" "$BASE/wan_transformer/diffusion_pytorch_model.safetensors"
wget -c -O "$HYWORLDPLAY_CKPT_ROOT/wan_distilled_model/model.pt" "$BASE/wan_distilled_model/model.pt"
