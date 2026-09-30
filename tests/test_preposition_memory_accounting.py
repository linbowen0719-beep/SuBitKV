from __future__ import annotations

from types import SimpleNamespace

import torch

from subitkv.kv_cache import ChunkedKVCache
from scripts.eval.run_small_eval import cache_metrics_from_runner


def runner(cache_mode: str):
    k = ChunkedKVCache(1, 32, 2, 128, 1, torch.bfloat16,
                       torch.device("cpu"))
    v = ChunkedKVCache(1, 32, 2, 128, 1, torch.bfloat16,
                       torch.device("cpu"))
    k.store_quantized(0, 32, {"packed": torch.empty(100, dtype=torch.uint8)})
    v.store_quantized(0, 32, {"packed": torch.empty(120, dtype=torch.uint8)})
    layer = {"k": k, "v": v, "cache_mode": cache_mode}
    return SimpleNamespace(pipe=SimpleNamespace(_kv_cache=[layer]))


def test_prerope_payload_is_physical_but_bf16_denominator_is_logical() -> None:
    pre = cache_metrics_from_runner(
        runner("pre_kv_reconstruct"), logical_head_dim=256, reference_frames=1
    )
    legacy_logical = cache_metrics_from_runner(
        runner("final"), logical_head_dim=256, reference_frames=1
    )
    assert pre["physical_kv_payload_bytes"] == 220
    assert legacy_logical["physical_kv_payload_bytes"] == 440
    assert pre["compression_ratio"] == 2 * legacy_logical["compression_ratio"]
