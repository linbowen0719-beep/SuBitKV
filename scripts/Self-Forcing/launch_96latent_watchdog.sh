#!/usr/bin/env bash
set -euo pipefail

repo=/home/bowen/Quant-VideoGen
python=/home/bowen/QuantVid/quantvid/bin/python
output=/data/bowen/self_forcing_96latent_moviegen
rvq=/data/bowen/self_forcing_96latent_rvq_l8c64
session=self_forcing_96latent

mkdir -p "$output"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session"
  exit 0
fi

tmux new-session -d -s "$session" \
  "cd '$repo' && exec '$python' scripts/Self-Forcing/watch_96latent_experiment.py \
    --output-root '$output' \
    --rvq-root '$rvq' \
    --python '$python' \
    --gpus 0,1,2,3,4,5,6,7 \
    --empty-mem-mib 100 \
    --empty-util 5 \
    --poll-seconds 30 \
    --max-attempts 3 2>&1 | tee -a '$output/watchdog.log'"

echo "started tmux session: $session"
