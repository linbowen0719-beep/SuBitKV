from __future__ import annotations

import copy

import pytest
import torch

from quant_videogen.truncatable_residual import (
    decode_residual,
    encode_residual,
    groupwise_hadamard,
    payload_nbytes,
    truncate_state,
)


SCHEMES = (
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)


@pytest.fixture(scope="module")
def codec_params() -> dict:
    thresholds = []
    gaussian_centroids = []
    for depth in range(4):
        nodes = 1 << depth
        thresholds.append(torch.linspace(-1.5, 1.5, nodes))
        gaussian_centroids.append(torch.linspace(-2.0, 2.0, 2 * nodes))

    # A deterministic, ordered 4-D binary tree.  Its size is representative
    # of the real T=16 table while the test input itself remains tiny.
    vq_centroids = []
    for depth in range(1, 17):
        nodes = 1 << depth
        coordinate = (
            (torch.arange(nodes, dtype=torch.float32) + 0.5)
            .div(float(nodes))
            .mul(4.0)
            .sub(2.0)
        )
        vq_centroids.append(coordinate[:, None].expand(-1, 4).contiguous())

    per_scheme = {
        "gaussian_tree": {
            "thresholds": thresholds,
            "centroids": gaussian_centroids,
        },
        "additive_binary": {
            "alpha": torch.tensor([0.8, 0.4, 0.2, 0.1]),
        },
        "hierarchical_vq": {
            "centroids": vq_centroids,
        },
    }
    return {
        "schema_version": "truncatable_residual_params.v1",
        "schemes": {
            scheme: {"K": values, "V": values}
            for scheme, values in per_scheme.items()
        },
    }


def _params_for(scheme: str, codec_params: dict) -> dict | None:
    return None if scheme == "nested_affine" else codec_params


@pytest.mark.parametrize("scheme", SCHEMES)
def test_encode_once_prefix_matches_direct_prefix(
    scheme: str,
    codec_params: dict,
) -> None:
    generator = torch.Generator().manual_seed(123)
    x = torch.randn((1, 2, 64), generator=generator, dtype=torch.float32)
    params = _params_for(scheme, codec_params)
    full = encode_residual(x, scheme, 4, "K", params_path=params)

    for bits in (1, 2, 3, 4):
        truncated = truncate_state(full, bits)
        direct = encode_residual(x, scheme, bits, "K", params_path=params)

        assert truncated["stored_bits"] == bits
        assert len(truncated["planes"]) == bits * (
            4 if scheme == "hierarchical_vq" else 1
        )
        assert len(direct["planes"]) == len(truncated["planes"])
        for direct_plane, truncated_plane in zip(
            direct["planes"], truncated["planes"]
        ):
            assert torch.equal(direct_plane, truncated_plane)

        direct_hat = decode_residual(direct, torch.float32)
        truncated_hat = decode_residual(truncated, torch.float32)
        torch.testing.assert_close(
            truncated_hat, direct_hat, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize(
    ("scheme", "expected_bytes"),
    (
        ("nested_affine", (64, 96, 128, 160)),
        ("gaussian_tree", (40, 72, 104, 136)),
        ("additive_binary", (40, 72, 104, 136)),
        ("hierarchical_vq", (40, 72, 104, 136)),
    ),
)
def test_payload_bytes_match_dense_bitplane_theory(
    scheme: str,
    expected_bytes: tuple[int, ...],
    codec_params: dict,
) -> None:
    # 256 values: G32 affine has 8 groups and two FP16 parameters/group;
    # G64 codecs have 4 groups and one FP16 scale/group.
    x = torch.linspace(-1.0, 1.0, 256, dtype=torch.float32).reshape(2, 128)
    params = _params_for(scheme, codec_params)
    measured = []
    for bits in (1, 2, 3, 4):
        state = encode_residual(x, scheme, bits, "V", params_path=params)
        measured.append(payload_nbytes(state))
    assert tuple(measured) == expected_bytes
    assert measured == sorted(measured)


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("hadamard", (False, True))
def test_shape_dtype_finiteness_and_hadamard(
    scheme: str,
    hadamard: bool,
    codec_params: dict,
) -> None:
    x = torch.randn((2, 3, 128), dtype=torch.bfloat16)
    state = encode_residual(
        x,
        scheme,
        3,
        "V",
        params_path=_params_for(scheme, codec_params),
        hadamard=hadamard,
        seed=77,
    )
    reconstructed = decode_residual(state, torch.bfloat16)
    assert reconstructed.shape == x.shape
    assert reconstructed.dtype == torch.bfloat16
    assert reconstructed.is_contiguous()
    assert bool(torch.isfinite(reconstructed.float()).all())


def test_groupwise_hadamard_is_orthonormal_and_self_inverse() -> None:
    x = torch.randn((3, 128), dtype=torch.float32)
    transformed = groupwise_hadamard(x, 32, seed=91)
    restored = groupwise_hadamard(transformed, 32, seed=91, inverse=True)
    torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(
        transformed.square().sum(),
        x.square().sum(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_explicit_group_size_override_preserves_default_behavior(
    codec_params: dict,
) -> None:
    x = torch.randn((2, 128), dtype=torch.float32)
    default = encode_residual(
        x, "gaussian_tree", 3, "K", params_path=codec_params
    )
    explicit = encode_residual(
        x,
        "gaussian_tree",
        3,
        "K",
        params_path=codec_params,
        group_size=64,
    )
    assert default["group_size"] == explicit["group_size"] == 64
    for left, right in zip(default["planes"], explicit["planes"]):
        assert torch.equal(left, right)

    widened = encode_residual(
        x,
        "gaussian_tree",
        3,
        "K",
        params_path=codec_params,
        group_size=128,
    )
    assert widened["group_size"] == 128
    assert widened["scale"].numel() * 2 == default["scale"].numel()


def test_nested_affine_zero_range_is_exact_and_finite() -> None:
    x = torch.full((2, 128), 1.25, dtype=torch.float32)
    for bits in (1, 2, 3, 4):
        state = encode_residual(x, "nested_affine", bits, "K")
        reconstructed = decode_residual(state, torch.float32)
        torch.testing.assert_close(reconstructed, x, rtol=0.0, atol=0.0)
        assert bool(torch.isfinite(reconstructed).all())


def test_truncation_is_physical_and_does_not_mutate_source(
    codec_params: dict,
) -> None:
    x = torch.randn((2, 128), dtype=torch.float32)
    full = encode_residual(
        x, "additive_binary", 4, "K", params_path=codec_params
    )
    source_planes = tuple(full["planes"])
    truncated = truncate_state(full, 2)

    assert len(full["planes"]) == 4
    assert len(truncated["planes"]) == 2
    assert tuple(full["planes"]) == source_planes
    for retained, source in zip(truncated["planes"], source_planes[:2]):
        assert retained.data_ptr() != source.data_ptr()
        assert torch.equal(retained, source)
    discarded_ptrs = {plane.data_ptr() for plane in source_planes[2:]}
    assert not any(
        plane.data_ptr() in discarded_ptrs for plane in truncated["planes"]
    )


def test_parameter_path_and_direct_mapping_are_equivalent(
    tmp_path,
    codec_params: dict,
) -> None:
    gaussian_only = {
        "gaussian_tree": copy.deepcopy(
            codec_params["schemes"]["gaussian_tree"]
        )
    }
    path = tmp_path / "codec_params.pt"
    torch.save(gaussian_only, path)
    x = torch.randn((1, 128), dtype=torch.float32)

    from_path = encode_residual(
        x, "gaussian_tree", 3, "K", params_path=path
    )
    from_mapping = encode_residual(
        x, "gaussian_tree", 3, "K", params_path=gaussian_only
    )
    assert "params_path" in from_path
    assert "shared_params" not in from_path
    assert "shared_params" in from_mapping
    for path_plane, mapping_plane in zip(
        from_path["planes"], from_mapping["planes"]
    ):
        assert torch.equal(path_plane, mapping_plane)
    torch.testing.assert_close(
        decode_residual(from_path, torch.float32),
        decode_residual(from_mapping, torch.float32),
        rtol=0.0,
        atol=0.0,
    )


def test_invalid_inputs_and_corrupt_states(codec_params: dict) -> None:
    x = torch.randn((1, 128), dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported residual scheme"):
        encode_residual(x, "unknown", 4, "K")
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        encode_residual(x, "nested_affine", 0, "K")
    with pytest.raises(ValueError, match=r"\[1, 4\]"):
        encode_residual(x, "nested_affine", 5, "K")
    with pytest.raises(ValueError, match="tensor_name"):
        encode_residual(x, "nested_affine", 4, "Q")
    with pytest.raises(ValueError, match="divisible"):
        encode_residual(torch.randn(1, 96), "gaussian_tree", 4, "K", codec_params)
    with pytest.raises(TypeError, match="floating point"):
        encode_residual(torch.zeros(1, 128, dtype=torch.int32), "nested_affine", 4, "K")
    with pytest.raises(ValueError, match="requires params_path"):
        encode_residual(x, "gaussian_tree", 4, "K")

    bad_alpha = copy.deepcopy(codec_params)
    bad_alpha["schemes"]["additive_binary"]["K"] = {
        "alpha": torch.tensor([0.8, 0.9, 0.2, 0.1])
    }
    with pytest.raises(ValueError, match="strictly decreasing"):
        encode_residual(x, "additive_binary", 4, "K", bad_alpha)

    full = encode_residual(x, "nested_affine", 3, "K")
    with pytest.raises(ValueError, match="cannot increase"):
        truncate_state(full, 4)
    corrupt = dict(full)
    corrupt["planes"] = tuple(full["planes"][:-1])
    with pytest.raises(ValueError, match="requires 3 planes"):
        decode_residual(corrupt, torch.float32)
    with pytest.raises(TypeError, match="floating point"):
        decode_residual(full, torch.int32)
