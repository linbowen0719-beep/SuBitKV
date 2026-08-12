#!/usr/bin/env bash
set -euo pipefail

repo=/home/bowen/Quant-VideoGen
python=/home/bowen/QuantVid/quantvid/bin/python
output=/data/bowen/self_forcing_96latent_moviegen
worker=self_forcing_96latent

mkdir -p "$output"
while [[ ! -e "$output/experiment.done" ]]; do
  if ! tmux has-session -t "$worker" 2>/dev/null; then
    bash "$repo/scripts/Self-Forcing/launch_96latent_watchdog.sh"
  fi
  sleep 30
done

cd "$repo"
"$python" scripts/Self-Forcing/watch_96latent_experiment.py \
  --output-root "$output" \
  --rvq-root /data/bowen/self_forcing_96latent_rvq_l8c64 \
  --python "$python" \
  --aggregate-only \
  >>"$output/final_aggregate.log" 2>&1
