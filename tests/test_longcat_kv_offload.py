from pathlib import Path
from types import SimpleNamespace
import sys

import torch


LONGCAT_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "LongCat"
sys.path.insert(0, str(LONGCAT_ROOT))

from longcat_video.modules.utils import _onload_kv_cache  # noqa: E402
from scripts.LongCat.run_chunk5_experiment import (  # noqa: E402
    dump_rvq_calibration,
    fixed_rvq_codebook_accounting,
)


def test_onload_nested_quantized_cache_does_not_mutate_cpu_storage():
    k_data = torch.ones(2, dtype=torch.float32)
    v_data = torch.zeros(2, dtype=torch.float32)
    k_cache = {"data": k_data, "nested": [{"scale": torch.ones(1)}]}
    v_cache = {"data": v_data, "nested": [{"scale": torch.ones(1)}]}

    k_loaded, v_loaded = _onload_kv_cache(
        (k_cache, v_cache), torch.device("meta")
    )

    assert k_loaded is not k_cache
    assert v_loaded is not v_cache
    assert k_loaded["data"].device.type == "meta"
    assert v_loaded["nested"][0]["scale"].device.type == "meta"
    assert k_cache["data"] is k_data
    assert v_cache["data"] is v_data
    assert k_cache["nested"][0]["scale"].device.type == "cpu"
    assert v_cache["nested"][0]["scale"].device.type == "cpu"


def test_dump_longcat_rvq_calibration_is_sampled_and_idempotent(tmp_path):
    generator = torch.Generator(device="cpu").manual_seed(17)
    kv_cache = {
        layer: (
            torch.randn(1, 32, 20, 128, generator=generator, dtype=torch.bfloat16),
            torch.randn(1, 32, 20, 128, generator=generator, dtype=torch.bfloat16),
        )
        for layer in range(2)
    }
    args = SimpleNamespace(
        rvq_calibration_dir=tmp_path,
        rvq_calibration_split="train",
        rvq_calibration_chunk=8,
        rvq_calibration_samples=7,
        prompt_index=3,
        seed=123,
    )

    dump_rvq_calibration(kv_cache, args=args, chunk_index=8)
    first_path = tmp_path / "kv_dump" / "train" / "3" / "layer_0.pt"
    before = first_path.stat().st_mtime_ns
    payload = torch.load(first_path, map_location="cpu", weights_only=False)
    dump_rvq_calibration(kv_cache, args=args, chunk_index=8)

    assert payload["K"].shape == (1, 32, 7, 128)
    assert payload["V"].shape == (1, 32, 7, 128)
    assert payload["sample_indices"].shape == (7,)
    assert first_path.stat().st_mtime_ns == before


def test_longcat_l8c64_int4_codebook_storage_accounting(monkeypatch):
    monkeypatch.delenv("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", raising=False)
    accounting = fixed_rvq_codebook_accounting("rvq_l8c64_int4")

    assert accounting["shared_codebook_stored_bytes"] == 99 * 1024**2
    assert accounting["shared_codebook_runtime_resident_bytes"] == 489 * 1024**2
    assert accounting["runtime_cache_enabled"] is True
    assert fixed_rvq_codebook_accounting("qvg_int4")["shared_codebook_stored_bytes"] == 0
