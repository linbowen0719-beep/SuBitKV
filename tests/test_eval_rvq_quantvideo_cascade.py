import argparse
import importlib.util
from pathlib import Path
import sys
import types

import torch


PATH = Path(__file__).resolve().parents[1] / "scripts" / "eval_rvq_quantvideo_cascade.py"
sys.path.insert(0, str(PATH.parent))

# The base test environment intentionally does not load the CUDA/Triton
# QuantVideo kernels.  These stubs let us unit-test the orchestration helpers;
# the real codec path is covered by the GPU smoke run.
compress_stub = types.ModuleType("quant_videogen.compress")
compress_stub.compress_kv_cache = None
compress_stub.get_quantize_fn = None
functions_stub = types.ModuleType("quant_videogen.functions")
functions_stub.triton_prq_dequantize_tensor = None
config_stub = types.ModuleType("quant_videogen.sim.quant.quantize_config")
config_stub.QuantizeConfig = object
sys.modules["quant_videogen.compress"] = compress_stub
sys.modules["quant_videogen.functions"] = functions_stub
sys.modules["quant_videogen.sim.quant.quantize_config"] = config_stub

SPEC = importlib.util.spec_from_file_location("eval_rvq_quantvideo_cascade", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_inference_span_sampling_is_aligned_and_deterministic():
    args = argparse.Namespace(key_sample=7040, key_sampling="inference_span")
    first = MOD.key_indices(38720, args, torch.Generator().manual_seed(42))
    second = MOD.key_indices(38720, args, torch.Generator().manual_seed(42))
    torch.testing.assert_close(first, second)
    assert first.numel() == 7040
    assert int(first[0]) % 7040 == 0
    assert int(first[-1]) - int(first[0]) + 1 == 7040


def test_residual_mode_adds_quantized_residual_to_rvq_base(monkeypatch):
    def identity_qv(k, v, bits, args):
        assert bits == 4
        return k, v, 123

    monkeypatch.setattr(MOD, "qv_pair", identity_qv)
    args = argparse.Namespace()
    k = torch.randn(19, 128)
    v = torch.randn(19, 128)
    kb = torch.randn(19, 128)
    vb = torch.randn(19, 128)
    kh, vh, payload = MOD.qv_residual_pair(
        k, v, kb, vb, 4, args, torch.device("cpu")
    )
    expected_k = kb.bfloat16().reshape(1, 1, 19, 128) + (
        k.bfloat16() - kb.bfloat16()
    ).reshape(1, 1, 19, 128)
    expected_v = vb.bfloat16().reshape(1, 1, 19, 128) + (
        v.bfloat16() - vb.bfloat16()
    ).reshape(1, 1, 19, 128)
    torch.testing.assert_close(kh, expected_k, atol=0, rtol=0)
    torch.testing.assert_close(vh, expected_v, atol=0, rtol=0)
    assert payload == 123


def test_scalar_payload_counts_packed_values_and_fp16_metadata():
    x = torch.randn(10, 128)
    _, vector_sym = MOD.scalar_quantize(x, bits=4, group_size=128, affine=False)
    _, group_sym = MOD.scalar_quantize(x, bits=4, group_size=32, affine=False)
    _, group_affine = MOD.scalar_quantize(x, bits=4, group_size=32, affine=True)
    assert vector_sym == (10 * 128 * 4 + 10 * 16) // 8
    assert group_sym == (10 * 128 * 4 + 10 * 4 * 16) // 8
    assert group_affine == (10 * 128 * 4 + 10 * 4 * 2 * 16) // 8


def test_scalar_residual_adds_correction_to_bf16_base():
    torch.manual_seed(7)
    k = torch.randn(9, 128)
    v = torch.randn(9, 128)
    kb = k + 0.01 * torch.randn_like(k)
    vb = v + 0.01 * torch.randn_like(v)
    kh, vh, payload = MOD.scalar_residual_pair(
        k, v, kb, vb, bits=4, group_size=32, affine=True, device=torch.device("cpu")
    )
    assert kh.dtype == torch.bfloat16
    assert vh.dtype == torch.bfloat16
    assert payload > 0
    assert torch.mean((kh.float() - k) ** 2) < torch.mean((kb.bfloat16().float() - k) ** 2)
    assert torch.mean((vh.float() - v) ** 2) < torch.mean((vb.bfloat16().float() - v) ** 2)


def test_gaussian_lloyd_max_int2_matches_reference_values():
    actual = torch.tensor(MOD.gaussian_lloyd_max_values(2))
    expected = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)


def test_normalized_fwht_is_orthonormal_and_self_inverse():
    torch.manual_seed(11)
    x = torch.randn(7, 64)
    transformed = MOD.normalized_fwht(x)
    reconstructed = MOD.normalized_fwht(transformed)
    torch.testing.assert_close(reconstructed, x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        transformed.square().sum(-1), x.square().sum(-1), atol=1e-4, rtol=1e-5
    )


def test_higgs_lite_payload_refinement_and_residual_addback():
    torch.manual_seed(17)
    x = torch.randn(10, 128).bfloat16()
    plain, payload32 = MOD.higgs_quantize(x, 2, 32, 0, 123)
    refined, refined_payload32 = MOD.higgs_quantize(x, 2, 32, 1, 123)
    _, payload64 = MOD.higgs_quantize(x, 2, 64, 1, 123)
    assert payload32 == (x.numel() * 2 + 10 * 4 * 16 + 7) // 8
    assert payload64 == (x.numel() * 2 + 10 * 2 * 16 + 7) // 8
    assert refined_payload32 == payload32
    assert torch.mean((refined.float() - x.float()) ** 2) <= torch.mean(
        (plain.float() - x.float()) ** 2
    )

    base = (x.float() + 0.1 * torch.randn_like(x.float())).bfloat16()
    corrected, _, _ = MOD.higgs_residual_pair(
        x,
        x,
        base,
        base,
        bits=2,
        group_size=32,
        refine_iters=1,
        seed=123,
        device=torch.device("cpu"),
    )
    assert torch.mean((corrected.float() - x.float()) ** 2) < torch.mean(
        (base.float() - x.float()) ** 2
    )


def test_paired_bootstrap_improvement_is_deterministic_and_paired():
    baseline = {0: 2.0, 1: 4.0, 2: 8.0}
    candidate = {0: 1.0, 1: 2.0, 2: 4.0}
    first = MOD.paired_improvement_stats(candidate, baseline, seed=42, samples=1000)
    second = MOD.paired_improvement_stats(candidate, baseline, seed=42, samples=1000)
    assert first == second
    assert first["QKV_MSE_improvement_vs_rvq_pct"] == 50.0
    assert first["paired_prompt_wins"] == 3
    assert abs(first["QKV_MSE_improvement_bootstrap_ci95_low"] - 50.0) < 1e-12
    assert abs(first["QKV_MSE_improvement_bootstrap_ci95_high"] - 50.0) < 1e-12
    direct = MOD.paired_improvement_stats(
        candidate, baseline, seed=42, samples=1000, baseline_label="direct"
    )
    assert direct["QKV_MSE_improvement_vs_direct_pct"] == 50.0
    assert "QKV_MSE_improvement_vs_rvq_pct" not in direct
