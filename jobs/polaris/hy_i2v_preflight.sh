#!/bin/bash
# Every on-disk dependency the HY i2v pipeline loads, checked before any GPU
# stage is submitted.  Grown one item per failure so far — Glyph, then
# SigLIP — each of which cost a queue round-trip and a capture attempt.
DATA_ROOT=/lus/grand/projects/SDR/chengming/zhenghui/ICLR2026_RVQ_polaris
. "${DATA_ROOT}/hy_i2v.env"
M="${QVG_HY8B_MODEL}"; rc=0
for p in "$M/config.json" "$M/scheduler/scheduler_config.json" \
         "$M/transformer/480p_i2v/config.json" "$M/transformer/480p_i2v/diffusion_pytorch_model.safetensors" \
         "$M/vae/config.json" "$M/vae/diffusion_pytorch_model.safetensors" \
         "$M/text_encoder/llm/config.json" "$M/text_encoder/llm/model.safetensors.index.json" \
         "$M/text_encoder/byt5-small/config.json" \
         "$M/text_encoder/Glyph-SDXL-v2/checkpoints/byt5_model.pt" \
         "$M/text_encoder/Glyph-SDXL-v2/assets/color_idx.json" \
         "$M/text_encoder/Glyph-SDXL-v2/assets/multilingual_10-lang_idx.json" \
         "$M/vision_encoder/siglip/image_encoder/model.safetensors" \
         "$M/vision_encoder/siglip/feature_extractor/preprocessor_config.json" \
         "${QVG_HY_ACTION_CKPT}" \
         "${QVG_MOVIEGEN_ROOT}/data/test-00000-of-00001.parquet"; do
  [ -s "$p" ] && echo "  ok   ${p#$DATA_ROOT/}" || { echo "  MISS ${p#$DATA_ROOT/}"; rc=1; }
done
n=$(ls "${QVG_MOVIEGEN_ROOT}"/data/test_with_generations-*.parquet 2>/dev/null|wc -l); [ "$n" -eq 33 ] || { echo "  MISS moviegen shards $n/33"; rc=1; }
# attempt budget: quarantine (never delete) any capture prompt that has burned all 3
A="${QVG_DATA_ROOT}/hyworldplay-moviegen-i2v-20-973/artifacts/capture_runs"
for p in "$A"/prompt_*; do [ -d "$p" ] || continue
  [ "$(ls -d "$p"/attempt_* 2>/dev/null|wc -l)" -ge 3 ] && [ -z "$(ls -d "$p"/attempt_*/capture/kv_dump/train/* 2>/dev/null)" ] && { mkdir -p "$A/_quarantine"; mv "$p" "$A/_quarantine/$(basename $p).$(date +%s)"; echo "  quarantined $(basename $p) (3 failed attempts)"; }
done
[ $rc -eq 0 ] && echo "PREFLIGHT OK" || echo "PREFLIGHT FAILED"; exit $rc
