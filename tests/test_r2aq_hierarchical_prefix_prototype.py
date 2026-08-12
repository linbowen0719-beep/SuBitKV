from __future__ import annotations

import pytest
import torch

import quant_videogen.r2aq as codec
from quant_videogen.kernels.triton.r2aq_hierarchical_prefix_prototype import (
    build_fine_prefix,
    decode_hierarchical_prefix,
)


def test_fine_prefix_exact_local_counts_and_physical_bytes() -> None:
    shape = (1, 1, 256, 128)
    groups = 512
    selected = torch.zeros(groups, dtype=torch.uint8)
    selected[0:256:3] = 1
    selected[256:512:5] = 1
    raw = selected << 7
    for fine_span, expected_bytes in ((64, 8), (32, 16)):
        prefix = build_fine_prefix(
            raw, shape=shape, fine_span=fine_span
        )
        assert prefix.dtype == torch.uint8
        assert prefix.numel() == expected_bytes
        subtiles = 256 // fine_span
        for tile in range(2):
            source = selected[tile * 256 : (tile + 1) * 256]
            for subtile in range(subtiles):
                expected = int(source[: subtile * fine_span].sum())
                assert int(prefix[0, tile, subtile]) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("fine_span", [32, 64])
def test_hierarchical_prefix_prototype_is_bit_exact(
    monkeypatch: pytest.MonkeyPatch, fine_span: int
) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batch_heads, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens, device=flattened.device)
            .remainder(n_clusters)
            .expand(batch_heads, -1)
            .contiguous()
        )
        centers = torch.linspace(
            -1.0,
            1.0,
            batch_heads * n_clusters * dim,
            device=flattened.device,
            dtype=flattened.dtype,
        ).reshape(batch_heads, n_clusters, dim)
        sizes = torch.zeros(
            (batch_heads, n_clusters),
            device=flattened.device,
            dtype=torch.int64,
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)
    source = torch.randn(
        (1, 2, 512, 128),
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(20260724),
    )
    state = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        bits=4,
        n_clusters=256,
        selection_fraction=0.434091,
    )
    fine = build_fine_prefix(
        state["suffix"]["scale_raw"],
        shape=tuple(state["shape"]),
        fine_span=fine_span,
    )
    expected = codec.decode_r2aq_residual(
        state, torch.bfloat16, use_fast_path=False
    )
    for _ in range(4):
        actual = decode_hierarchical_prefix(
            state, fine, fine_span=fine_span
        )
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
