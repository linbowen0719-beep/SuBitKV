from __future__ import annotations

import pytest
import torch

from quant_videogen.kernels.triton.rvq import (
    rvq_decode_bhsd_triton,
    rvq_encode_bhsd_staged_triton,
    rvq_encode_bhsd_triton,
)

try:
    import triton  # noqa: F401
except Exception:  # pragma: no cover - optional dependency/runtime varies by host
    _TRITON_AVAILABLE = False
else:
    _TRITON_AVAILABLE = True


def _torch_bf16_encode_reference(
    tensor: torch.Tensor,
    codebooks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror the standard RMS RVQ fallback's BF16 assignment arithmetic."""
    bsz, heads, seq_len, dim = tensor.shape
    scale = torch.sqrt(torch.mean(tensor.float().square(), dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = (tensor.float() / scale).to(torch.bfloat16)
    all_ids = []

    for level in range(codebooks.shape[0]):
        residual_hsd = residual.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
        dots = torch.bmm(
            residual_hsd,
            codebooks[level].transpose(1, 2),
        ).float()
        residual_sq = residual_hsd.float().square().sum(dim=-1, keepdim=True)
        codebook_sq = codebooks[level].float().square().sum(dim=-1)
        distance = residual_sq + codebook_sq[:, None, :] - 2.0 * dots
        ids = torch.argmin(distance, dim=-1)
        all_ids.append(ids.reshape(heads, bsz, seq_len).permute(1, 0, 2))

        assigned = torch.gather(
            codebooks[level],
            1,
            ids[..., None].expand(-1, -1, dim),
        )
        residual = (residual_hsd - assigned).reshape(heads, bsz, seq_len, dim)
        residual = residual.permute(1, 0, 2, 3).contiguous()

    return torch.stack(all_ids), scale


def _torch_bf16_decode_reference(
    indices: torch.Tensor,
    codebooks: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    _levels, bsz, heads, seq_len = indices.shape
    dim = codebooks.shape[-1]
    accumulated = torch.zeros(
        (bsz, heads, seq_len, dim),
        device=indices.device,
        dtype=torch.float32,
    )
    for level in range(indices.shape[0]):
        per_head = []
        for head in range(heads):
            per_head.append(codebooks[level, head][indices[level, :, head].long()])
        assigned = torch.stack(per_head, dim=1)
        accumulated = (accumulated + assigned.float()).to(torch.bfloat16).float()
    return (accumulated * scale).to(torch.bfloat16)


def _torch_bf16_no_norm_encode_reference(
    tensor: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    residual = tensor.to(torch.bfloat16)
    all_ids = []
    bsz, heads, seq_len, dim = tensor.shape
    for level in range(codebooks.shape[0]):
        residual_hsd = residual.permute(1, 0, 2, 3).reshape(heads, bsz * seq_len, dim)
        dots = torch.bmm(residual_hsd, codebooks[level].transpose(1, 2)).float()
        distance = (
            residual_hsd.float().square().sum(dim=-1, keepdim=True)
            + codebooks[level].float().square().sum(dim=-1)[:, None, :]
            - 2.0 * dots
        )
        ids = torch.argmin(distance, dim=-1)
        all_ids.append(ids.reshape(heads, bsz, seq_len).permute(1, 0, 2))
        assigned = torch.gather(codebooks[level], 1, ids[..., None].expand(-1, -1, dim))
        residual = (residual_hsd - assigned).reshape(heads, bsz, seq_len, dim)
        residual = residual.permute(1, 0, 2, 3).contiguous()
    return torch.stack(all_ids)


@pytest.mark.parametrize("centroids", [64, 256, 512, 1024])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ Triton kernels require CUDA and Triton",
)
def test_tiled_centroid_rvq_encode_matches_torch_bf16_fallback(centroids: int):
    generator = torch.Generator(device="cuda").manual_seed(1000 + centroids)
    tensor = torch.randn(
        (1, 1, 17, 128),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    codebooks = (
        torch.randn(
            (2, 1, centroids, 128),
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.125
    ).contiguous()
    codebook_sq = codebooks.float().square().sum(dim=-1).contiguous()
    index_dtype = torch.uint8 if centroids <= 256 else torch.int16

    expected_ids, expected_scale = _torch_bf16_encode_reference(tensor, codebooks)
    actual_ids, actual_scale = rvq_encode_bhsd_triton(
        tensor,
        codebooks,
        codebook_sq,
        num_levels=2,
        index_dtype=index_dtype,
        block_tokens=16,
    )

    assert actual_ids.dtype == index_dtype
    torch.testing.assert_close(actual_ids.long(), expected_ids.long(), atol=0, rtol=0)
    torch.testing.assert_close(actual_scale, expected_scale, atol=2e-7, rtol=2e-7)

    actual_reconstruction = rvq_decode_bhsd_triton(
        actual_ids,
        codebooks,
        actual_scale,
        output_dtype=torch.bfloat16,
        block_tokens=16,
    )
    expected_reconstruction = _torch_bf16_decode_reference(
        expected_ids,
        codebooks,
        actual_scale,
    )
    torch.testing.assert_close(actual_reconstruction, expected_reconstruction, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ Triton kernels require CUDA and Triton",
)
def test_no_norm_rvq_encode_decode_matches_torch_bf16_fallback():
    generator = torch.Generator(device="cuda").manual_seed(20260715)
    tensor = torch.randn((1, 2, 19, 128), device="cuda", dtype=torch.bfloat16, generator=generator).contiguous()
    codebooks = (torch.randn((3, 2, 64, 128), device="cuda", dtype=torch.bfloat16, generator=generator) * 0.125).contiguous()
    codebook_sq = codebooks.float().square().sum(dim=-1).contiguous()
    expected_ids = _torch_bf16_no_norm_encode_reference(tensor, codebooks)
    actual_ids, unused_scale = rvq_encode_bhsd_triton(
        tensor,
        codebooks,
        codebook_sq,
        num_levels=3,
        index_dtype=torch.uint8,
        block_tokens=16,
        normalize=False,
    )
    assert unused_scale.numel() == 1
    torch.testing.assert_close(actual_ids.long(), expected_ids.long(), atol=0, rtol=0)
    actual = rvq_decode_bhsd_triton(
        actual_ids,
        codebooks,
        None,
        output_dtype=torch.bfloat16,
        block_tokens=16,
        apply_scale=False,
    )
    expected = _torch_bf16_decode_reference(
        expected_ids,
        codebooks,
        torch.ones((1, 2, 19, 1), device="cuda"),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    source = torch.randn(
        actual.shape,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    fused_residual = rvq_decode_bhsd_triton(
        actual_ids,
        codebooks,
        None,
        output_dtype=torch.bfloat16,
        block_tokens=16,
        apply_scale=False,
        residual_source=source,
    )
    torch.testing.assert_close(
        fused_residual,
        (source - actual).contiguous(),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ Triton kernels require CUDA and Triton",
)
def test_staged_no_norm_encode_is_exact_and_does_not_mutate_input():
    generator = torch.Generator(device="cuda").manual_seed(20260803)
    tensor = torch.randn(
        (1, 2, 37, 128), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    original = tensor.clone()
    codebooks = (
        torch.randn(
            (6, 2, 64, 128), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.125
    ).contiguous()
    codebook_sq = codebooks.float().square().sum(dim=-1).contiguous()
    expected = _torch_bf16_no_norm_encode_reference(tensor, codebooks)

    actual, unused_scale, _stage_mse = rvq_encode_bhsd_staged_triton(
        tensor,
        codebooks,
        codebook_sq,
        num_levels=6,
        index_dtype=torch.uint8,
        block_tokens=16,
        normalize=False,
    )

    assert unused_scale.numel() == 1
    assert torch.equal(tensor, original)
    torch.testing.assert_close(actual.long(), expected.long(), atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ Triton kernels require CUDA and Triton",
)
def test_packed_uint4_scale_decode_matches_materialized_scale():
    generator = torch.Generator(device="cuda").manual_seed(20260731)
    indices = torch.randint(
        0, 64, (7, 1, 4, 19), device="cuda", dtype=torch.uint8,
        generator=generator,
    ).contiguous()
    codebooks = (
        torch.randn(
            (7, 4, 64, 128), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.125
    ).contiguous()
    q = torch.randint(
        0, 16, (1, 4, 19, 1), device="cuda", dtype=torch.uint8,
        generator=generator,
    )
    flat = q.reshape(-1)
    packed = (flat[0::2] | (flat[1::2] << 4)).contiguous()
    scale_min = torch.rand(
        (1, 4, 1, 1), device="cuda", dtype=torch.float16,
        generator=generator,
    ).contiguous()
    scale_step = (
        torch.rand(
            (1, 4, 1, 1), device="cuda", dtype=torch.float16,
            generator=generator,
        )
        * 0.1
        + 0.01
    ).contiguous()
    scale = (q.float() * scale_step.float() + scale_min.float()).contiguous()
    expected = rvq_decode_bhsd_triton(
        indices, codebooks, scale, torch.bfloat16, block_tokens=16,
    )
    actual = rvq_decode_bhsd_triton(
        indices,
        codebooks,
        None,
        torch.bfloat16,
        block_tokens=16,
        packed_uint4_scale=(packed, scale_min, scale_step),
    )
    assert torch.equal(actual, expected)
    larger = torch.full(
        (1, 4, 31, 128),
        7.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    strided = rvq_decode_bhsd_triton(
        indices,
        codebooks,
        None,
        torch.bfloat16,
        block_tokens=16,
        packed_uint4_scale=(packed, scale_min, scale_step),
        out=larger,
        out_token_offset=5,
    )
    assert torch.equal(strided, expected)
    assert torch.all(larger[:, :, :5] == 7)
    assert torch.all(larger[:, :, 24:] == 7)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _TRITON_AVAILABLE,
    reason="RVQ Triton kernels require CUDA and Triton",
)
def test_tracked_stage_mse_matches_encoded_bf16_residuals():
    generator = torch.Generator(device="cuda").manual_seed(20260801)
    tensor = torch.randn(
        (1, 2, 37, 128), device="cuda", dtype=torch.bfloat16,
        generator=generator,
    ).contiguous()
    codebooks = (
        torch.randn(
            (6, 2, 64, 128), device="cuda", dtype=torch.bfloat16,
            generator=generator,
        )
        * 0.125
    ).contiguous()
    codebook_sq = codebooks.float().square().sum(dim=-1).contiguous()
    indices, scale, tracked = rvq_encode_bhsd_triton(
        tensor,
        codebooks,
        codebook_sq,
        num_levels=6,
        index_dtype=torch.uint8,
        block_tokens=16,
        track_stage_mse=True,
    )
    residual = (tensor.float() / scale).to(torch.bfloat16)
    expected = [residual.float().square().mean()]
    for level in range(6):
        assigned = torch.stack(
            [
                codebooks[level, head][indices[level, :, head].long()]
                for head in range(2)
            ],
            dim=1,
        )
        residual = (residual - assigned).to(torch.bfloat16)
        expected.append(residual.float().square().mean())
    torch.testing.assert_close(
        tracked,
        torch.stack(expected),
        rtol=2e-6,
        atol=2e-7,
    )
