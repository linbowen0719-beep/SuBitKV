#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QVG_CKPT_ROOT="${QVG_CKPT_ROOT:-$REPO_ROOT/ckpts}"
SELF_FORCING_CKPT_ROOT="$QVG_CKPT_ROOT/Self-Forcing"

# Download Wan2.1-T2V-1.3B base model
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir "$SELF_FORCING_CKPT_ROOT/Wan2.1-T2V-1.3B"

# Download Self-Forcing DMD checkpoint
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir "$SELF_FORCING_CKPT_ROOT"
mv "$SELF_FORCING_CKPT_ROOT/checkpoints/self_forcing_dmd.pt" "$SELF_FORCING_CKPT_ROOT/self_forcing_dmd.pt"
rmdir "$SELF_FORCING_CKPT_ROOT/checkpoints" 2>/dev/null || true
