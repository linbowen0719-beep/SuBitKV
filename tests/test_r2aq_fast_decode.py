from __future__ import annotations

import math

import pytest
import torch

import quant_videogen.r2aq as codec
from quant_videogen.kernels.triton.r2aq import decode_r2aq_g64


DIM = 128
GROUP = 64
TILE_GROUPS = 256
CLUSTERS = 256


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    flat = bits.contiguous().view(-1).to(torch.uint8)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat(
            [flat, torch.zeros(pad, dtype=torch.uint8, device=flat.device)]
        )
    weights = (1 << torch.arange(8, device=flat.device)).to(torch.uint8)
    return (flat.view(-1, 8) * weights).sum(1).to(torch.uint8).contiguous()


def _pack_int4_low_first(values: torch.Tensor) -> torch.Tensor:
    flat = values.contiguous().view(-1).to(torch.uint8)
    assert flat.numel() % 2 == 0
    return (
        (flat[0::2] & 0x0F) | ((flat[1::2] & 0x0F) << 4)
    ).contiguous()


def _make_case(
    *,
    device: torch.device,
    bits: int,
    tokens: int = 257,
) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
    shape = (1, 2, tokens, DIM)
    elements = math.prod(shape)
    groups = elements // GROUP
    groups_per_batch_head = tokens * (DIM // GROUP)
    generator = torch.Generator(device=device).manual_seed(1200 + bits)

    all_bits = torch.randint(
        0,
        2,
        (elements, 5),
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    # Exercise both selector values and both sides of the 256-group tile
    # boundary.  Two heads use different masks with the same selected count,
    # which also catches a missing per-head rank reset.
    group_ids = torch.arange(groups_per_batch_head, device=device)
    first = ((group_ids * 17 + group_ids.square()) % 37) < 16
    first[-3:] = torch.tensor([True, False, True], device=device)
    second = first.roll(29)
    second[torch.nonzero(~second, as_tuple=False)[0, 0]] = True
    selected_mask_2d = torch.stack((first, second))
    selected_mask = selected_mask_2d.reshape(-1)
    assert int(second.sum().item()) == int(first.sum().item()) + 1

    positive_scale = (
        torch.rand(groups, generator=generator, device=device)
        .mul_(1.75)
        .add_(2.0**-8)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    assert not bool((positive_scale & 0x80).any())
    scale_raw = (
        positive_scale | (selected_mask.to(torch.uint8) << 7)
    ).contiguous()

    grouped_bits = all_bits.view(groups, GROUP, 5)
    segments = [_pack_bits(grouped_bits[:, :, 0])]
    for depth in range(1, bits):
        segments.append(_pack_bits(grouped_bits[:, :, depth][selected_mask]))
        segments.append(_pack_bits(grouped_bits[:, :, depth][~selected_mask]))
    segments.append(_pack_bits(grouped_bits[:, :, bits][selected_mask]))

    starts = torch.arange(
        0, groups_per_batch_head, TILE_GROUPS, device=device
    )
    prefix = torch.stack(
        [
            torch.stack(
                [row[: int(start)].sum() for start in starts] + [row.sum()]
            )
            for row in selected_mask_2d
        ]
    ).reshape(-1)
    selected_count = int(selected_mask.sum().item())
    prefix_dtype = (
        torch.int16 if groups_per_batch_head < 65536 else torch.int32
    )
    prefix = prefix.to(prefix_dtype).contiguous()

    base_centroids = torch.linspace(
        -1.0, 1.0, 1 << bits, dtype=torch.float32, device=device
    )
    refined_centroids = torch.linspace(
        -1.0,
        1.0,
        1 << (bits + 1),
        dtype=torch.float32,
        device=device,
    )

    predictor_q = torch.randint(
        -7,
        8,
        (shape[0], shape[1], CLUSTERS, DIM),
        dtype=torch.int16,
        generator=generator,
        device=device,
    )
    # QuantVideo's wire representation stores signed q as q+7; nibble 15 is
    # intentionally unused by the symmetric [-7,7] alphabet.
    predictor_packed = _pack_int4_low_first(
        (predictor_q + 7).to(torch.uint8)
    )
    predictor_scale_raw = (
        torch.rand(
            (shape[0], shape[1], CLUSTERS, 1),
            generator=generator,
            device=device,
        )
        .mul_(0.25)
        .add_(2.0**-8)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .contiguous()
    )
    cluster_ids = torch.randint(
        0,
        CLUSTERS,
        (shape[0], shape[1], tokens),
        dtype=torch.uint8,
        generator=generator,
        device=device,
    ).contiguous()

    scale = (scale_raw & 0x7F).view(torch.float8_e4m3fn).float()
    dense_code = torch.zeros(elements, dtype=torch.int64, device=device)
    for depth in range(bits):
        dense_code = (dense_code << 1) | all_bits[:, depth].long()
    base = base_centroids[dense_code]
    refined_code = (dense_code << 1) | all_bits[:, bits].long()
    refined = refined_centroids[refined_code]
    suffix = torch.where(
        selected_mask.repeat_interleave(GROUP),
        refined,
        base,
    ) * scale.repeat_interleave(GROUP)
    gather_index = cluster_ids.long().unsqueeze(-1).expand(
        *cluster_ids.shape, DIM
    )
    predictor_float = torch.gather(
        predictor_q.float()
        * predictor_scale_raw.view(torch.float8_e4m3fn).float(),
        2,
        gather_index,
    )
    suffix_float = suffix.view(shape)

    kwargs = {
        "segments": tuple(segments),
        "suffix_scale_raw": scale_raw,
        "selector_prefix_offsets": prefix,
        "base_centroids": base_centroids,
        "refined_centroids": refined_centroids,
        "predictor_q_packed": predictor_packed,
        "predictor_scale_raw": predictor_scale_raw,
        "cluster_ids": cluster_ids,
        "shape": shape,
        "nominal_bits": bits,
        "selected_group_count": selected_count,
    }
    return kwargs, (predictor_float, suffix_float)


def test_fast_decode_is_fail_closed_without_cuda() -> None:
    with pytest.raises(ValueError, match="requires CUDA"):
        decode_r2aq_g64(
            segments=(
                torch.zeros(16, dtype=torch.uint8),
                torch.zeros(8, dtype=torch.uint8),
            ),
            suffix_scale_raw=torch.zeros(2, dtype=torch.uint8),
            selector_prefix_offsets=torch.zeros(2, dtype=torch.int16),
            base_centroids=torch.zeros(2),
            refined_centroids=torch.zeros(4),
            predictor_q_packed=torch.zeros(
                CLUSTERS * DIM // 2, dtype=torch.uint8
            ),
            predictor_scale_raw=torch.zeros(CLUSTERS, dtype=torch.uint8),
            cluster_ids=torch.zeros(1, 1, 1, dtype=torch.uint8),
            shape=(1, 1, 1, DIM),
            nominal_bits=1,
            selected_group_count=1,
            output_dtype=torch.bfloat16,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_fast_decode_matches_eager_and_sparse_rank(bits: int) -> None:
    kwargs, (predictor, suffix) = _make_case(
        device=torch.device("cuda"), bits=bits
    )
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        actual = decode_r2aq_g64(**kwargs, output_dtype=dtype)
        expected = (predictor + suffix).to(dtype)
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_decode_rank_workspace_is_repeatable_across_allocator_reuse() -> None:
    """The transient selector-rank workspace must survive repeated restores."""

    kwargs, (predictor, suffix) = _make_case(
        device=torch.device("cuda"), bits=4, tokens=512
    )
    expected = (predictor + suffix).to(torch.bfloat16)
    for _ in range(12):
        actual = decode_r2aq_g64(
            **kwargs, output_dtype=torch.bfloat16
        )
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        del actual


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fast_decode_rejects_malformed_sparse_plane() -> None:
    kwargs, _ = _make_case(device=torch.device("cuda"), bits=2, tokens=3)
    kwargs["segments"] = (*kwargs["segments"][:-1], kwargs["segments"][-1][:-1])
    with pytest.raises(ValueError, match=r"segments\[3\]"):
        decode_r2aq_g64(**kwargs, output_dtype=torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_codec_opt_in_fast_decode_matches_eager_for_all_prefixes(
    monkeypatch: pytest.MonkeyPatch,
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
            -2.0,
            2.0,
            batch_heads * n_clusters * dim,
            device=flattened.device,
            dtype=torch.float32,
        ).reshape(batch_heads, n_clusters, dim).to(flattened.dtype)
        sizes = torch.zeros(
            (batch_heads, n_clusters),
            device=flattened.device,
            dtype=torch.int64,
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)
    source = torch.randn(
        (1, 2, 257, DIM),
        generator=torch.Generator(device="cuda").manual_seed(20260724),
        device="cuda",
        dtype=torch.bfloat16,
    )
    full = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        max_nominal_bits=4,
        n_clusters=CLUSTERS,
        selection_fraction=0.4,
    )
    for bits in (1, 2, 3, 4):
        state = codec.truncate_r2aq_state(full, nominal_bits=bits)
        payload_before = codec.payload_nbytes(state)
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            eager = codec.decode_r2aq_residual(
                state, dtype, use_fast_path=False
            )
            fused = codec.decode_r2aq_residual(
                state, dtype, use_fast_path=True
            )
            torch.testing.assert_close(
                fused, eager, atol=0.0, rtol=0.0
            )
        assert codec.payload_nbytes(state) == payload_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_coherent_tail_cuda_eager_and_terminal_fast_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batch_heads, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens, device=flattened.device)
            .remainder(n_clusters)
            .expand(batch_heads, -1)
            .contiguous()
        )
        centers = torch.zeros(
            batch_heads,
            n_clusters,
            dim,
            device=flattened.device,
            dtype=flattened.dtype,
        )
        sizes = torch.zeros(
            batch_heads,
            n_clusters,
            device=flattened.device,
            dtype=torch.int64,
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)
    source = torch.randn(
        (1, 1, 512, DIM),
        generator=torch.Generator(device="cuda").manual_seed(91),
        device="cuda",
        dtype=torch.bfloat16,
    )
    utility = torch.ones(
        (1, 2, 5, 2), device="cuda", dtype=torch.float16
    )
    common = {
        "tensor_name": "V",
        "max_nominal_bits": 4,
        "n_clusters": CLUSTERS,
        "selection_fraction": 0.434091,
        "utility_weights": utility,
        "layer_idx": 0,
        "age_bin": 0,
    }
    control = codec.encode_r2aq_residual(
        source, ablation="full_tree", **common
    )
    state = codec.encode_r2aq_residual(
        source, ablation="coherent_tail", **common
    )
    assert codec.payload_nbytes(state) == codec.payload_nbytes(control)
    eager = codec.decode_r2aq_residual(
        state, torch.bfloat16, use_fast_path=False
    )
    assert eager.is_cuda and eager.shape == source.shape
    with pytest.raises(ValueError, match="coherent_tail"):
        codec.decode_r2aq_residual(
            state, torch.bfloat16, use_fast_path=True
        )
    for bits in (1, 2, 3):
        prefix = codec.truncate_r2aq_state(state, nominal_bits=bits)
        control_prefix = codec.truncate_r2aq_state(
            control, nominal_bits=bits
        )
        assert "coherent_tail" not in prefix
        assert codec.payload_nbytes(prefix) == codec.payload_nbytes(
            control_prefix
        )
        torch.testing.assert_close(
            codec.decode_r2aq_residual(
                prefix, torch.bfloat16, use_fast_path=True
            ),
            codec.decode_r2aq_residual(control_prefix, torch.bfloat16),
            atol=0,
            rtol=0,
        )
    g128 = codec.encode_r2aq_residual(
        source,
        ablation="coherent_tail_g128",
        coherent_tail_group_fraction=1.0,
        **common,
    )
    assert g128["coherent_tail"]["group_size"] == 128
    assert codec.payload_nbytes(g128) == codec.payload_nbytes(control)
    assert codec.decode_r2aq_residual(
        g128, torch.bfloat16, use_fast_path=False
    ).is_cuda
    with pytest.raises(ValueError, match="coherent_tail"):
        codec.decode_r2aq_residual(
            g128, torch.bfloat16, use_fast_path=True
        )
    for bits in (1, 2, 3):
        prefix = codec.truncate_r2aq_state(g128, nominal_bits=bits)
        control_prefix = codec.truncate_r2aq_state(
            control, nominal_bits=bits
        )
        assert codec.payload_nbytes(prefix) == codec.payload_nbytes(
            control_prefix
        )
        torch.testing.assert_close(
            codec.decode_r2aq_residual(
                prefix, torch.bfloat16, use_fast_path=True
            ),
            codec.decode_r2aq_residual(control_prefix, torch.bfloat16),
            atol=0,
            rtol=0,
        )


def test_codec_fast_decode_is_explicit_and_fail_closed_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batch_heads, tokens, dim = flattened.shape
        ids = torch.arange(tokens).remainder(n_clusters).expand(
            batch_heads, -1
        ).contiguous()
        centers = torch.zeros(
            batch_heads, n_clusters, dim, dtype=flattened.dtype
        )
        sizes = torch.zeros(batch_heads, n_clusters, dtype=torch.int64)
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)
    state = codec.encode_r2aq_residual(
        torch.randn(1, 1, 256, DIM, dtype=torch.bfloat16),
        tensor_name="V",
        bits=2,
        n_clusters=CLUSTERS,
        selection_fraction=0.25,
    )
    eager = codec.decode_r2aq_residual(state, torch.bfloat16)
    explicit_eager = codec.decode_r2aq_residual(
        state, torch.bfloat16, use_fast_path=False
    )
    torch.testing.assert_close(eager, explicit_eager, atol=0, rtol=0)
    with pytest.raises(ValueError, match="unavailable"):
        codec.decode_r2aq_residual(
            state, torch.bfloat16, use_fast_path=True
        )
