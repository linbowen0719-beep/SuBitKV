from __future__ import annotations

import pytest
import torch

from subitkv import progressive_sign_adaptive_absmax_scalar as progressive
from subitkv import sign_adaptive_absmax_scalar as leaf
from subitkv.kernels.triton.rvq import rvq_decode_bhsd_triton


@pytest.mark.parametrize(
    "order",
    (
        (3, 2, 1, 0),
        (3, 1, 2, 0),
        (0, 1, 2, 3),
    ),
)
def test_b4_is_bit_exact_to_production_leaf_for_any_bit_order(
    order: tuple[int, ...],
) -> None:
    generator = torch.Generator().manual_seed(20260730)
    source = torch.randn(1, 4, 17, 128, generator=generator).to(torch.bfloat16)
    params = progressive.build_midpoint_params(bit_order=order)
    state = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, params=params, use_fast=False
    )
    restored = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
        state
    )
    leaf_state = leaf.encode_sign_adaptive_absmax_scalar_residual(
        source, use_fast=False
    )
    leaf_restored = leaf.decode_sign_adaptive_absmax_scalar_residual(
        leaf_state, use_fast=False
    )
    assert torch.equal(restored, leaf_restored)
    assert progressive.payload_nbytes(state) == leaf.payload_nbytes(leaf_state)


def test_every_prefix_is_physical_and_nested() -> None:
    source = torch.linspace(-3, 3, 2 * 3 * 128).reshape(1, 2, 3, 128)
    state = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, max_bits=8, use_fast=False
    )
    previous = None
    for bits in (8, 4, 3, 2, 1):
        prefix = progressive.truncate_progressive_sign_adaptive_absmax_state(
            state, bits
        )
        decoded = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
            prefix, torch.float32
        )
        assert torch.isfinite(decoded).all()
        expected = source.numel() * bits // 8 + source.numel() // 128
        assert progressive.payload_nbytes(prefix) == expected
        if previous is not None:
            assert all(
                torch.equal(left, right)
                for left, right in zip(prefix["planes"], previous["planes"])
            )
        previous = prefix


def test_b8_prefix_preserves_production_b4_leaf() -> None:
    generator = torch.Generator().manual_seed(20260731)
    source = torch.randn(1, 4, 17, 128, generator=generator).to(torch.bfloat16)
    state = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, max_bits=8, use_fast=False
    )
    b4 = progressive.truncate_progressive_sign_adaptive_absmax_state(state, 4)
    restored = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(b4)
    leaf_state = leaf.encode_sign_adaptive_absmax_scalar_residual(
        source, use_fast=False
    )
    leaf_restored = leaf.decode_sign_adaptive_absmax_scalar_residual(
        leaf_state, use_fast=False
    )
    assert torch.equal(restored, leaf_restored)
    assert progressive.payload_nbytes(b4) == leaf.payload_nbytes(leaf_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_b8_fast_encode_and_all_prefix_decodes_match_eager() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    source = torch.randn(1, 4, 73, 128, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    eager = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, max_bits=8, use_fast=False
    )
    fast = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, max_bits=8, use_fast=True
    )
    assert torch.equal(eager["scales"], fast["scales"])
    assert all(
        torch.equal(left, right) for left, right in zip(eager["planes"], fast["planes"])
    )
    for bits in (1, 2, 3, 4, 5, 6, 7, 8):
        prefix = progressive.truncate_progressive_sign_adaptive_absmax_state(fast, bits)
        eager_decoded = (
            progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                prefix, use_fast=False
            )
        )
        fast_decoded = (
            progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                prefix, use_fast=True
            )
        )
        assert torch.equal(eager_decoded, fast_decoded)
        base = torch.randn_like(fast_decoded)
        expected_sum = (base + fast_decoded).to(torch.bfloat16)
        base_ptr = base.data_ptr()
        fused_sum = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
            prefix, use_fast=True, _base=base
        )
        assert fused_sum.data_ptr() == base_ptr
        assert torch.equal(fused_sum, expected_sum)
        larger = torch.zeros(
            fast_decoded.shape[0],
            fast_decoded.shape[1],
            fast_decoded.shape[2] + 4,
            fast_decoded.shape[3],
            device=fast_decoded.device,
            dtype=fast_decoded.dtype,
        )
        larger[:, :, 2:-2].copy_(torch.randn_like(fast_decoded))
        expected_strided = (larger[:, :, 2:-2].clone() + fast_decoded).to(
            torch.bfloat16
        )
        actual_strided = (
            progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                prefix,
                use_fast=True,
                _base=larger,
                _base_token_offset=2,
            )
        )
        assert torch.equal(actual_strided, expected_strided)
        assert torch.equal(larger[:, :, 2:-2], expected_strided)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_direct_shallow_fast_encode_matches_b8_prefix_bit_exactly() -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260804)
    source = torch.randn(1, 4, 73, 128, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    full = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, max_bits=8, use_fast=True
    )
    for bits in (1, 2, 3, 4):
        direct = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
            source, max_bits=bits, use_fast=True
        )
        prefix = progressive.truncate_progressive_sign_adaptive_absmax_state(full, bits)
        assert torch.equal(
            direct["scales"].view(torch.uint8),
            prefix["scales"].view(torch.uint8),
        )
        assert all(
            torch.equal(left, right)
            for left, right in zip(direct["planes"], prefix["planes"])
        )
        assert torch.equal(
            progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                direct, use_fast=True
            ),
            progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
                prefix, use_fast=True
            ),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", (1, 2, 3, 4, 8))
def test_fused_dense_rvq_progressive_restore_is_bit_exact(bits: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    source = torch.randn(1, 3, 73, 128, device="cuda", generator=generator).to(
        torch.bfloat16
    )
    full = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, max_bits=8, use_fast=True
    )
    state = progressive.truncate_progressive_sign_adaptive_absmax_state(full, bits)
    levels, centroids = 16, 64
    indices = torch.randint(
        centroids,
        (levels, *source.shape[:3]),
        device="cuda",
        dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    codebooks = (
        (
            torch.randn(
                levels,
                source.shape[1],
                centroids,
                128,
                device="cuda",
                generator=generator,
            )
            * 0.02
        )
        .to(torch.bfloat16)
        .contiguous()
    )
    base = rvq_decode_bhsd_triton(
        indices, codebooks, None, torch.bfloat16, apply_scale=False
    )
    residual = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
        state, torch.bfloat16, use_fast=True
    )
    expected = (base + residual).to(torch.bfloat16)
    fused = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
        state,
        torch.bfloat16,
        use_fast=True,
        _fused_rvq=(indices, codebooks),
    )
    assert torch.equal(fused, expected)

    destination = torch.zeros(
        1, 3, source.shape[2] + 4, 128, device="cuda", dtype=torch.bfloat16
    )
    actual = progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(
        state,
        torch.bfloat16,
        use_fast=True,
        _base=destination,
        _base_token_offset=2,
        _fused_rvq=(indices, codebooks),
    )
    assert torch.equal(actual, expected)
    assert torch.equal(destination[:, :, 2:-2], expected)


def test_corrupt_or_nonprefix_state_is_rejected() -> None:
    source = torch.randn(1, 2, 3, 128)
    state = progressive.encode_progressive_sign_adaptive_absmax_scalar_residual(
        source, use_fast=False
    )
    broken = dict(state)
    broken["planes"] = tuple(state["planes"][:2])
    with pytest.raises(ValueError, match="bitplanes"):
        progressive.decode_progressive_sign_adaptive_absmax_scalar_residual(broken)
    with pytest.raises(ValueError, match="target bits"):
        progressive.truncate_progressive_sign_adaptive_absmax_state(state, 0)
