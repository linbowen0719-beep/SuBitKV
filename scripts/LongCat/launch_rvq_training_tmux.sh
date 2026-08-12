#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/bowen/Quant-VideoGen}
RVQ_ROOT=${RVQ_ROOT:-/data/bowen/longcat_rvq_l8c64}
PROMPT_FILE=${PROMPT_FILE:-/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_train.txt}
SESSION=${SESSION:-longcat_rvq_training}
GPUS=${GPUS:-auto}
LOG=${LOG:-$RVQ_ROOT/logs/longcat_rvq_training.log}

if [[ -x /home/bowen/anaconda3/envs/HunyuanVideo/bin/python ]]; then
  PY=${PY:-/home/bowen/anaconda3/envs/HunyuanVideo/bin/python}
else
  PY=${PY:-$(command -v python)}
fi

mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi

printf -v quoted_repo '%q' "$REPO"
printf -v quoted_py '%q' "$PY"
printf -v quoted_root '%q' "$RVQ_ROOT"
printf -v quoted_prompts '%q' "$PROMPT_FILE"
printf -v quoted_gpus '%q' "$GPUS"
printf -v quoted_log '%q' "$LOG"

tmux new-session -d -s "$SESSION" \
  "cd $quoted_repo && ( $quoted_py scripts/LongCat/prepare_rvq_codebooks.py --stage init --root $quoted_root --prompt-file $quoted_prompts && $quoted_py scripts/LongCat/prepare_rvq_codebooks.py --stage train --root $quoted_root --prompt-file $quoted_prompts --gpus $quoted_gpus && $quoted_py scripts/LongCat/prepare_rvq_codebooks.py --stage verify --root $quoted_root --prompt-file $quoted_prompts ) |& tee -a $quoted_log"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
