#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/bowen/t2v_kv_bench_moviegen_20train_10test
REPO=/home/bowen/Quant-VideoGen
TMP="$ROOT/tmp/native_int4_perchunk_shards"
PY=/home/bowen/anaconda3/envs/hw8/bin/python

rm -rf "$TMP"
mkdir -p "$TMP"
cd "$REPO"

COMMON=(
  --prompt-file "$ROOT/prompts_test.txt"
  --num-prompts 2
  --seed 42
  --setting int4
  --num_frames 173
  --num_chunk 11
  --num_inference_steps 50
  --image_path none
  --pose w-96
  --memory_frames 80
  --temporal_context_size 76
  --pred_latent_size 4
  --offload_text_encoder
  --kv_logical_head_dim 256
  --kv_quant_factor 1
)

for spec in 0:0 1:2 2:4 3:6 4:8; do
  gpu="${spec%%:*}"
  start="${spec##*:}"
  port=$((29700 + gpu))
  out="$TMP/part_$start"
  mkdir -p "$out"
  echo "[start] gpu=$gpu port=$port start=$start out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" MASTER_PORT="$port" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" scripts/eval/run_small_eval.py --prompt-start "$start" --output-dir "$out" "${COMMON[@]}" \
    > "$TMP/part_$start.log" 2>&1 &
done

wait

"$PY" - <<'PY'
import csv
import shutil
from pathlib import Path

root = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
tmp = root / "tmp" / "native_int4_perchunk_shards"
target = root / "videos" / "native_int4_perchunk"
target.mkdir(parents=True, exist_ok=True)

rows = []
parts = [p for p in tmp.glob("part_*") if p.is_dir()]
for part in sorted(parts, key=lambda p: int(p.name.split("_")[1])):
    metrics = part / "metrics.csv"
    if not metrics.exists():
        raise FileNotFoundError(metrics)
    with metrics.open(newline="", encoding="utf-8") as f:
        rows.extend(csv.DictReader(f))
    for mp4 in part.glob("*.mp4"):
        shutil.copy2(mp4, target / mp4.name)
    cfg = part / "config.json"
    if cfg.exists():
        shutil.copy2(cfg, target / "config.json")

rows.sort(key=lambda r: int(r["prompt_id"]))
if len(rows) != 10:
    raise RuntimeError(f"expected 10 rows, got {len(rows)}")
for row in rows:
    row["video_path"] = str(target / f"{row['prompt_id']}-{row['seed']}.mp4")

with (target / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"[merged] {target / 'metrics.csv'}")
PY
