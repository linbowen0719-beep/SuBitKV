import math

import torch

from quant_videogen.higgs import (
    gaussian_lloyd_max_values,
    higgs_dequantize_tensor,
    higgs_quantize_tensor,
    normalized_fwht,
    pack_lowbit_indices,
    unpack_lowbit_indices,
)


def test_int2_table_matches_reference():
    actual = torch.tensor(gaussian_lloyd_max_values(2))
    expected = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104])
    torch.testing.assert_close(actual, expected, atol=5e-4, rtol=0)


def test_fwht_is_orthonormal_and_self_inverse():
    torch.manual_seed(3)
    x = torch.randn(5, 64)
    z = normalized_fwht(x)
    torch.testing.assert_close(normalized_fwht(z), x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(z.square().sum(-1), x.square().sum(-1), atol=1e-4, rtol=1e-5)


def test_lowbit_pack_round_trip_int2_and_int4():
    for bits in (2, 4):
        values = torch.arange(131, dtype=torch.uint8) % (1 << bits)
        packed = pack_lowbit_indices(values, bits)
        restored = unpack_lowbit_indices(packed, bits, values.numel())
        torch.testing.assert_close(restored, values)
        assert packed.numel() == math.ceil(values.numel() * bits / 8)


def test_higgs_payload_and_round_trip_quality():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 11, 128, dtype=torch.bfloat16)
    for bits in (2, 4):
        state = higgs_quantize_tensor(
            x, bits=bits, group_size=32, refine_iters=1, seed=20260722
        )
        reconstructed = higgs_dequantize_tensor(state, torch.bfloat16)
        assert reconstructed.shape == x.shape
        assert reconstructed.dtype == torch.bfloat16
        groups = x.numel() // 32
        assert state["packed_indices"].numel() == math.ceil(x.numel() * bits / 8)
        assert state["scale"].numel() == groups
        assert state["scale"].element_size() == 2
        mse = torch.mean((reconstructed.float() - x.float()).square()).item()
        assert math.isfinite(mse)
        assert mse < (0.25 if bits == 2 else 0.02)


def test_scale_refinement_does_not_worsen_same_tensor():
    torch.manual_seed(11)
    x = torch.randn(257, 128, dtype=torch.bfloat16)
    plain = higgs_dequantize_tensor(
        higgs_quantize_tensor(x, bits=2, group_size=32, refine_iters=0, seed=91)
    )
    refined = higgs_dequantize_tensor(
        higgs_quantize_tensor(x, bits=2, group_size=32, refine_iters=1, seed=91)
    )
    plain_mse = torch.mean((plain.float() - x.float()).square())
    refined_mse = torch.mean((refined.float() - x.float()).square())
    assert refined_mse <= plain_mse
