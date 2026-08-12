#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/bowen/Quant-VideoGen}
RVQ_ROOT=${RVQ_ROOT:-/data/bowen/longcat_rvq_l8c64}
SESSION=${SESSION:-longcat_rvq_pipeline_supervisor}
LOG=${LOG:-$RVQ_ROOT/logs/longcat_rvq_pipeline_supervisor.log}
RETRY_SECONDS=${RETRY_SECONDS:-60}

mkdir -p "$(dirname "$LOG")"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi

printf -v quoted_repo '%q' "$REPO"
printf -v quoted_log '%q' "$LOG"
tmux new-session -d -s "$SESSION" \
  "cd $quoted_repo && while true; do bash scripts/LongCat/supervise_rvq_pipeline.sh && break; status=\$?; echo \"[restart] RVQ supervisor rc=\$status; retrying in ${RETRY_SECONDS}s\"; sleep ${RETRY_SECONDS}; done |& tee -a $quoted_log"

echo "Started tmux session: $SESSION"
echo "Log: $LOG"
