import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_fixed_budget_rvq_screen.py"
SPEC = importlib.util.spec_from_file_location("fixed_budget_screen", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_grouped_candidates_respect_registered_index_budgets():
    assert MOD.GROUP_METHODS["g4l2_direct"].index_bits == MOD.BASE_BITS["l8"] == 48
    assert MOD.GROUP_METHODS["g4l4_direct"].index_bits == MOD.BASE_BITS["l16"] == 96


def test_sparse_rescue_budget_includes_route_and_scale():
    for budget in ("l8", "l16"):
        for bits in (2, 4):
            depth, fraction, payload = MOD.rescue_fraction(budget, bits)
            assert depth * 6 + 1 + fraction * payload <= MOD.BASE_BITS[budget] + 1e-6


def test_kv_asymmetric_stage_allocations_are_pair_rate_matched():
    for budget, k_depth, v_depth in (("l8", 4, 12), ("l8", 6, 10), ("l16", 8, 24), ("l16", 12, 20)):
        assert (k_depth + v_depth) * 6 == 2 * MOD.BASE_BITS[budget]


def test_hadamard_group_roundtrip_and_common_quantizer():
    x = torch.randn(17, 128)
    method = MOD.GROUP_METHODS["g4l2_hadamard"]
    perm = MOD.permutation(42)
    restored = MOD.from_groups(MOD.to_groups(x, method, perm), method, perm)
    torch.testing.assert_close(restored, x, atol=2e-5, rtol=2e-5)
    for bits in (2, 4):
        for affine in (False, True):
            out = MOD.affine_quant(x, bits, 32, affine)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()


def test_rvq_reconstruct_does_not_mutate_reference():
    x = torch.randn(23, 128)
    reference = x.clone()
    codebooks = torch.randn(2, 64, 128)
    MOD.rvq_reconstruct(x, codebooks, torch.device("cpu"))
    torch.testing.assert_close(x, reference, atol=0, rtol=0)


def test_int4_codebook_quantization_keeps_zero_centroids_finite():
    codebooks = torch.zeros(3, 8, 128)
    out = MOD.quantize_codebook_int4(codebooks)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, codebooks)
