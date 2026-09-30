from __future__ import annotations

import pytest
import torch

from subitkv import centered_absmax_scalar
from subitkv import static_equalized_midrange as codec
from subitkv.truncatable_residual import groupwise_hadamard


def params(heads: int, device: torch.device | str = "cpu") -> dict:
    scale = torch.ones(30, heads, 128, device=device)
    scale[:, :, :32] = 0.25
    scale[:, :, 32:64] = 0.5
    scale[:, :, 96:] = 2.0
    return codec.make_params(scale)


def test_k_matches_explicit_static_equalization() -> None:
    torch.manual_seed(7)
    tensor = torch.randn(1, 2, 19, 128, dtype=torch.bfloat16)
    table = params(2)
    state = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="K",
        layer_idx=3,
        params=table,
    )
    decoded = codec.decode_static_equalized_midrange_residual(
        state, torch.float32, use_fast=False
    )
    scale = table["k_channel_scale"][3].float()[None, :, None, :]
    suffix = centered_absmax_scalar.encode_centered_absmax_scalar_residual(
        tensor.float() / scale,
        bits=4,
        group_size=128,
        center_mode="midrange",
    )
    expected = (
        centered_absmax_scalar.decode_centered_absmax_scalar_residual(
            suffix, torch.float32, use_fast=False
        )
        * scale
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert codec.payload_nbytes(state) * 8 / tensor.numel() == pytest.approx(
        4.125
    )


def test_v_is_uniform_midrange() -> None:
    torch.manual_seed(11)
    tensor = torch.randn(1, 2, 17, 128, dtype=torch.bfloat16)
    state = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="V",
        layer_idx=8,
        params=params(2),
    )
    decoded = codec.decode_static_equalized_midrange_residual(
        state, torch.float32, use_fast=False
    )
    uniform = centered_absmax_scalar.encode_centered_absmax_scalar_residual(
        tensor,
        bits=4,
        group_size=128,
        center_mode="midrange",
    )
    expected = centered_absmax_scalar.decode_centered_absmax_scalar_residual(
        uniform, torch.float32, use_fast=False
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_shared_table_is_counted_once() -> None:
    tensor = torch.randn(1, 2, 5, 128, dtype=torch.bfloat16)
    table = params(2)
    left = codec.encode_static_equalized_midrange_residual(
        tensor, tensor_name="K", layer_idx=0, params=table
    )
    right = codec.encode_static_equalized_midrange_residual(
        tensor, tensor_name="V", layer_idx=0, params=table
    )
    seen: set[str] = set()
    expected = 30 * 2 * 128
    assert codec.shared_payload_nbytes(left, seen) == expected
    assert codec.shared_payload_nbytes(right, seen) == 0


def test_train_routed_hadamard_matches_explicit_transform() -> None:
    torch.manual_seed(13)
    tensor = torch.randn(1, 2, 11, 128, dtype=torch.bfloat16)
    scale = params(2)["k_channel_scale"].float()
    mask = torch.zeros(30, dtype=torch.bool)
    mask[29] = True
    table = codec.make_params(scale, hadamard_layers=mask)
    state = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="K",
        layer_idx=29,
        params=table,
    )
    decoded = codec.decode_static_equalized_midrange_residual(
        state, torch.float32, use_fast=False
    )
    layer_scale = table["k_channel_scale"][29].float()[None, :, None, :]
    normalized = tensor.float() / layer_scale
    rotated = groupwise_hadamard(
        normalized, 128, seed=codec.HADAMARD_SEED
    )
    suffix = centered_absmax_scalar.encode_centered_absmax_scalar_residual(
        rotated,
        bits=4,
        group_size=128,
        center_mode="midrange",
    )
    expected = (
        groupwise_hadamard(
            centered_absmax_scalar.decode_centered_absmax_scalar_residual(
                suffix, torch.float32, use_fast=False
            ),
            128,
            seed=codec.HADAMARD_SEED,
            inverse=True,
        )
        * layer_scale
    )
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    assert state["hadamard"] is True
    assert table["shared_table_bytes"] == scale.numel() + 4

    unselected = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="K",
        layer_idx=28,
        params=table,
    )
    value = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="V",
        layer_idx=29,
        params=table,
    )
    assert unselected["hadamard"] is False
    assert value["hadamard"] is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fast_and_eager_restore_match() -> None:
    device = torch.device("cuda")
    torch.manual_seed(17)
    tensor = torch.randn(
        1, 2, 257, 128, device=device, dtype=torch.bfloat16
    )
    state = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="K",
        layer_idx=4,
        params=params(2, device),
    )
    eager = codec.decode_static_equalized_midrange_residual(
        state, torch.bfloat16, use_fast=False
    )
    fast = codec.decode_static_equalized_midrange_residual(
        state, torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fast, eager, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hadamard_fast_and_eager_restore_match() -> None:
    device = torch.device("cuda")
    torch.manual_seed(19)
    tensor = torch.randn(
        1, 2, 257, 128, device=device, dtype=torch.bfloat16
    )
    mask = torch.zeros(30, dtype=torch.bool, device=device)
    mask[29] = True
    state = codec.encode_static_equalized_midrange_residual(
        tensor,
        tensor_name="K",
        layer_idx=29,
        params=codec.make_params(
            params(2, device)["k_channel_scale"].float(),
            hadamard_layers=mask,
        ),
    )
    eager = codec.decode_static_equalized_midrange_residual(
        state, torch.bfloat16, use_fast=False
    )
    fast = codec.decode_static_equalized_midrange_residual(
        state, torch.bfloat16, use_fast=True
    )
    torch.testing.assert_close(fast, eager, rtol=0, atol=0)
