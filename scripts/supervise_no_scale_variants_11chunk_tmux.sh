#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-no_scale_variants_11chunk}
SUPERVISOR_SESSION=${SUPERVISOR_SESSION:-no_scale_variants_11chunk_supervisor}
REPO=${REPO:-/home/bowen/Quant-VideoGen}
RUN_ID=${RUN_ID:-no_scale_variants_11chunk_10prompt_20260707}
LOG=${SUPERVISOR_LOG:-$REPO/logs/no_scale_variants_11chunk_10prompt_20260707_supervisor.log}
FINAL_TABLE="$REPO/results/long_video/$RUN_ID/final_14row_table.csv"

run_loop() {
  cd "$REPO"
  mkdir -p "$(dirname "$LOG")"
  echo "[supervisor-start] $(date -Is) session=$SESSION run_id=$RUN_ID" | tee -a "$LOG"
  while true; do
    if [[ -s "$FINAL_TABLE" ]]; then
      echo "[supervisor-done] $(date -Is) final_table=$FINAL_TABLE" | tee -a "$LOG"
      exit 0
    fi
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "[supervisor-alive] $(date -Is) worker_session=$SESSION" | tee -a "$LOG"
    else
      echo "[supervisor-restart] $(date -Is) worker_session=$SESSION final_missing=$FINAL_TABLE" | tee -a "$LOG"
      bash "$REPO/scripts/launch_no_scale_variants_11chunk_tmux.sh" | tee -a "$LOG"
    fi
    sleep 120
  done
}

if [[ "${1:-}" == "--inner" ]]; then
  run_loop
else
  if tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null; then
    echo "[supervisor] already running: $SUPERVISOR_SESSION"
  else
    tmux new-session -d -s "$SUPERVISOR_SESSION" "cd '$REPO' && bash '$0' --inner"
    echo "[supervisor] session=$SUPERVISOR_SESSION"
  fi
  echo "[supervisor-log] $LOG"
fi
