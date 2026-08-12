from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_wan5b_p1_exact_replay", ROOT / "scripts" / "analyze_wan5b_p1_exact_replay.py"
)
assert SPEC is not None and SPEC.loader is not None
exact = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exact)


def test_uint4_pack_round_trip() -> None:
    values = torch.arange(37, dtype=torch.uint8).remainder(16).reshape(1, 37)
    packed, shape = exact.pack_uint4(values)
    restored = exact.unpack_uint4(packed, shape)
    torch.testing.assert_close(restored, values, rtol=0, atol=0)
    assert packed.numel() == 19


def test_int4_runtime_matches_packed_payload() -> None:
    generator = torch.Generator().manual_seed(13)
    codebook = torch.randn(3, 8, 16, generator=generator)
    packed, runtime = exact.quantize_int4_codebooks([codebook])
    record = packed[0]
    signed = exact.unpack_uint4(record["q_packed"], record["q_shape"]).to(torch.int16) - 8
    expected = signed.to(torch.bfloat16) * record["scale"].to(torch.bfloat16)
    torch.testing.assert_close(runtime[0], expected, rtol=0, atol=0)
    assert int(signed.min()) >= -7 and int(signed.max()) <= 7


def test_batched_history_reconstruction_preserves_input() -> None:
    generator = torch.Generator().manual_seed(19)
    x = torch.randn(1, 2, 29, 7, generator=generator).to(torch.bfloat16)
    before = x.clone()
    training = x[0].float()
    codebooks = exact.fit_fp32_rvq(training, [8, 4], iterations=2, seed=29)
    _packed, runtime = exact.quantize_int4_codebooks(codebooks)
    reconstructed = exact.reconstruct_history(x, runtime, torch.device("cpu"), 11)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert reconstructed.shape == x.shape


def test_validate_capture_accepts_lean_post_validation_schema() -> None:
    raw = {
        key: torch.zeros(1)
        for key in {
            "query_pre", "value_pre", "query_post_rope", "query_post_prope",
            "query_rope_cos", "query_rope_sin", "key_rope_cos", "key_rope_sin",
            "query_viewmats", "query_Ks", "key_viewmats", "key_Ks",
            "query_position_indices", "key_position_indices",
            "post_validation_offsets", "key_post_rope_validation",
            "value_post_rope_validation", "key_post_prope_validation",
            "value_post_prope_validation",
        }
    }
    raw["key_pre"] = torch.zeros(1, 1, 4, 2)
    raw["key_indices"] = torch.arange(4)
    raw["metadata"] = {
        "alignment": "exact_same_call_same_indices", "key_sample": 4,
        "key_len": 4, "lean_reference_reconstructed": True,
    }
    metadata = exact.validate_capture(raw)
    assert metadata["lean_reference_reconstructed"] is True
