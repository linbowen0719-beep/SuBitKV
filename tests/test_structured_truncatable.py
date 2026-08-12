from __future__ import annotations

import pytest
import torch

import quant_videogen.structured_truncatable as codec


def _gaussian_params() -> dict:
    thresholds = []
    centroids = []
    for depth in range(4):
        nodes = 1 << depth
        thresholds.append(torch.linspace(-1.5, 1.5, nodes))
        centroids.append(torch.linspace(-2.0, 2.0, 2 * nodes))
    values = {"thresholds": thresholds, "centroids": centroids}
    return {
        "schema_version": "truncatable_residual_params.v1",
        "schemes": {"gaussian_tree": {"K": values, "V": values}},
    }


def _install_predictor(monkeypatch) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batches, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens, device=flattened.device)
            .remainder(n_clusters)
            .expand(batches, -1)
            .contiguous()
        )
        centers = torch.zeros(
            batches,
            n_clusters,
            dim,
            dtype=flattened.dtype,
            device=flattened.device,
        )
        sizes = torch.zeros(
            batches, n_clusters, dtype=torch.int64, device=flattened.device
        )
        sizes.scatter_add_(
            1,
            ids,
            torch.ones_like(ids, dtype=torch.int64),
        )
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)


def test_structured_prefix_round_trip_and_physical_rate(monkeypatch) -> None:
    _install_predictor(monkeypatch)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(
        (1, 1, 256, 128), generator=generator, dtype=torch.bfloat16
    )
    full = codec.encode_structured_residual(
        x,
        bits=4,
        tensor_name="K",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=256,
        kmeans_max_iters=2,
        hadamard=False,
    )

    previous_error = None
    for bits in (1, 2, 3, 4):
        state = codec.truncate_structured_state(full, bits)
        reconstructed = codec.decode_structured_residual(
            state, torch.bfloat16
        )
        assert reconstructed.shape == x.shape
        assert reconstructed.dtype == x.dtype
        error = (reconstructed.float() - x.float()).square().mean().item()
        if previous_error is not None:
            assert error <= previous_error
        previous_error = error

        expected_rate = (
            bits
            + 16.0 / 128.0
            + 8.0 / 128.0
            + 256.0 * 128.0 * 16.0 / float(256 * 128)
        )
        assert codec.physical_bits_per_value(state) == expected_rate


def test_structured_decode_adds_selected_centroid(monkeypatch) -> None:
    _install_predictor(monkeypatch)
    x = torch.zeros((1, 1, 256, 128), dtype=torch.float32)
    state = codec.encode_structured_residual(
        x,
        bits=2,
        tensor_name="V",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=256,
    )
    state["centroids"][0, 0, :, 0] = torch.arange(256)
    reconstructed = codec.decode_structured_residual(state, torch.float32)
    torch.testing.assert_close(
        reconstructed[0, 0, :, 0],
        torch.arange(256, dtype=torch.float32)
        + reconstructed[0, 0, 0, 0],
    )


def test_fast_decode_selection_preserves_cpu_fallback(monkeypatch) -> None:
    _install_predictor(monkeypatch)
    x = torch.randn(
        (1, 1, 16, 128),
        generator=torch.Generator().manual_seed(19),
        dtype=torch.bfloat16,
    )
    state = codec.encode_structured_residual(
        x,
        bits=2,
        tensor_name="K",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=16,
        hadamard=True,
    )
    automatic = codec.decode_structured_residual(state, torch.bfloat16)
    eager = codec.decode_structured_residual(
        state, torch.bfloat16, use_fast_path=False
    )
    torch.testing.assert_close(automatic, eager, atol=0.0, rtol=0.0)
    with pytest.raises(ValueError, match="not on CUDA"):
        codec.decode_structured_residual(
            state, torch.bfloat16, use_fast_path=True
        )

    # Default dispatch must remain eager even for a state that a future
    # eligibility rule considers fast-path compatible.  This prevents worker
    # processes imported at different times from mixing implementations.
    monkeypatch.setattr(
        codec,
        "_fast_decode_unsupported_reason",
        lambda *_args, **_kwargs: None,
    )

    def forbidden_fast_path(*_args, **_kwargs):
        raise AssertionError("default decode unexpectedly selected fused CUDA")

    monkeypatch.setattr(
        codec, "_decode_structured_residual_fast", forbidden_fast_path
    )
    default_after_opt_in_probe = codec.decode_structured_residual(
        state, torch.bfloat16
    )
    torch.testing.assert_close(
        default_after_opt_in_probe, eager, atol=0.0, rtol=0.0
    )


def _move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    return value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cuda_decode_matches_eager_for_every_prefix(monkeypatch) -> None:
    _install_predictor(monkeypatch)
    generator = torch.Generator().manual_seed(29)
    x = torch.randn((1, 1, 17, 128), generator=generator, dtype=torch.bfloat16)
    full = codec.encode_structured_residual(
        x,
        bits=4,
        tensor_name="V",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=256,
        hadamard=True,
        transform_seed=137,
        scale_refine_iters=2,
    )
    # Exercise the fused centroid add as well as the suffix transform.
    full["centroids"].normal_(generator=torch.Generator().manual_seed(31))
    full = _move_tensors(full, torch.device("cuda"))

    for bits in (1, 2, 3, 4):
        state = codec.truncate_structured_state(full, bits)
        payload_before = codec.payload_nbytes(state)
        planes_before = tuple(
            plane.clone() for plane in state["suffix"]["planes"]
        )
        scale_before = state["suffix"]["scale"].clone()
        for output_dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            eager = codec.decode_structured_residual(
                state, output_dtype, use_fast_path=False
            )
            fused = codec.decode_structured_residual(
                state, output_dtype, use_fast_path=True
            )
            torch.testing.assert_close(fused, eager, atol=0.0, rtol=0.0)
        assert codec.payload_nbytes(state) == payload_before
        assert all(
            torch.equal(before, after)
            for before, after in zip(planes_before, state["suffix"]["planes"])
        )
        assert torch.equal(scale_before, state["suffix"]["scale"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_cuda_decode_real_cache_shape_is_exact_and_fail_closed(
    monkeypatch,
) -> None:
    _install_predictor(monkeypatch)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260723)
    x = torch.randn(
        (1, 24, 7040, 128),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    full = codec.encode_structured_residual(
        x,
        bits=4,
        tensor_name="K",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=256,
        kmeans_max_iters=2,
        hadamard=True,
        transform_seed=20260722,
        scale_refine_iters=1,
    )
    full["centroids"].normal_(
        generator=torch.Generator(device=device).manual_seed(31)
    )

    for bits in (1, 2, 3, 4):
        state = codec.truncate_structured_state(full, bits)
        payload_before = codec.payload_nbytes(state)
        for output_dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            eager = codec.decode_structured_residual(
                state, output_dtype, use_fast_path=False
            )
            fused = codec.decode_structured_residual(
                state, output_dtype, use_fast_path=True
            )
            assert torch.equal(fused, eager)
        assert codec.payload_nbytes(state) == payload_before

    corrupted = codec.truncate_structured_state(full, 2)
    corrupted["suffix"]["plane_count"] -= 1
    with pytest.raises(ValueError, match="plane_count"):
        codec.decode_structured_residual(
            corrupted, torch.bfloat16, use_fast_path=True
        )
