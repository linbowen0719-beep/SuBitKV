from __future__ import annotations

import copy

import pytest
import torch

from subitkv import routed_gaussian as codec


def _routing() -> dict[str, object]:
    offsets = [
        [-1, 0, 0, 1],
        [1, 0, 0, -1],
    ]
    return {
        "schema_version": codec.ROUTING_SCHEMA,
        "num_layers": 2,
        "num_heads": 4,
        "offsets": {"K": offsets, "V": list(reversed(offsets))},
    }


def test_fixed_tree_shapes_and_ordering() -> None:
    tree = codec.build_fixed_gaussian_tree(dtype=torch.float32)
    assert [item.numel() for item in tree["thresholds"]] == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    assert [item.numel() for item in tree["reconstruction"]] == [
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    for values in tree.values():
        for item in values:
            assert bool(torch.isfinite(item).all())
            assert bool(torch.all(item[1:] > item[:-1]))


def test_routing_budget_and_b1_exception() -> None:
    audit = codec.validate_routing_table(_routing())
    assert audit["counts"]["K"] == {"low": 2, "normal": 4, "high": 2}
    invalid = copy.deepcopy(_routing())
    invalid["offsets"]["K"][0][0] = 0
    with pytest.raises(ValueError, match="equal low/high"):
        codec.validate_routing_table(invalid)

    x = torch.randn(1, 4, 8, 128)
    full = codec.encode_routed_gaussian_residual(
        x,
        tensor_name="K",
        layer_idx=0,
        routing_table=_routing(),
        bits=4,
    )
    expected = {
        1: (1, 1, 1, 1),
        2: (1, 2, 2, 3),
        3: (2, 3, 3, 4),
        4: (3, 4, 4, 5),
    }
    for bits, depths in expected.items():
        prefix = codec.truncate_routed_gaussian_state(full, bits)
        assert prefix["routed_depths"] == depths
        assert sum(depths) / len(depths) == bits


def test_prefix_matches_direct_and_payload_is_exact() -> None:
    generator = torch.Generator().manual_seed(17)
    x = torch.randn((1, 4, 16, 128), generator=generator)
    full = codec.encode_routed_gaussian_residual(
        x,
        tensor_name="V",
        layer_idx=1,
        routing_table=_routing(),
        bits=4,
        retain_terminal_code=True,
    )
    values = x.numel()
    scale_bytes = values // 128
    for bits in range(1, 5):
        prefix = codec.truncate_routed_gaussian_state(full, bits)
        direct = codec.encode_routed_gaussian_residual(
            x,
            tensor_name="V",
            layer_idx=1,
            routing_table=_routing(),
            bits=bits,
        )
        assert codec.payload_nbytes(prefix) == scale_bytes + values * bits // 8
        assert codec.payload_nbytes(prefix) == codec.payload_nbytes(direct)
        for left, right in zip(prefix["round_segments"], direct["round_segments"]):
            assert torch.equal(left, right)
        torch.testing.assert_close(
            codec.decode_routed_gaussian_residual(prefix, torch.float32),
            codec.decode_routed_gaussian_residual(direct, torch.float32),
            rtol=0.0,
            atol=0.0,
        )


def test_b8_truncates_bit_exactly_to_historical_b4() -> None:
    generator = torch.Generator().manual_seed(20260731)
    x = torch.randn((1, 4, 16, 128), generator=generator)
    kwargs = {
        "tensor_name": "K",
        "layer_idx": 0,
        "routing_table": _routing(),
        "group_size": 128,
        "hadamard_group_size": 128,
    }
    direct_b4 = codec.encode_routed_gaussian_residual(x, bits=4, **kwargs)
    full_b8 = codec.encode_routed_gaussian_residual(x, bits=8, **kwargs)
    prefix_b4 = codec.truncate_routed_gaussian_state(full_b8, 4)
    assert torch.equal(direct_b4["scale_raw"], prefix_b4["scale_raw"])
    assert all(
        torch.equal(left, right)
        for left, right in zip(direct_b4["round_segments"], prefix_b4["round_segments"])
    )
    assert torch.equal(
        codec.decode_routed_gaussian_residual(direct_b4),
        codec.decode_routed_gaussian_residual(prefix_b4),
    )


def test_b4_physical_rate_and_shared_charge_once() -> None:
    x = torch.randn(1, 4, 8, 128)
    k = codec.encode_routed_gaussian_residual(
        x, tensor_name="K", layer_idx=0, routing_table=_routing(), bits=4
    )
    v = codec.encode_routed_gaussian_residual(
        x, tensor_name="V", layer_idx=0, routing_table=_routing(), bits=4
    )
    assert k["diagnostics"]["physical_bits_per_value"] == 4.0625
    seen: set[str] = set()
    first = codec.shared_payload_nbytes(k, seen)
    second = codec.shared_payload_nbytes(v, seen)
    assert first > second > 0
    assert codec.shared_payload_nbytes(k, seen) == 0


def test_corrupt_route_hash_is_rejected() -> None:
    x = torch.randn(1, 4, 8, 128)
    state = codec.encode_routed_gaussian_residual(
        x, tensor_name="K", layer_idx=0, routing_table=_routing(), bits=2
    )
    state["routing_table_packed"] = state["routing_table_packed"].clone()
    state["routing_table_packed"][0] ^= 1
    with pytest.raises(ValueError):
        codec.decode_routed_gaussian_residual(state)


def test_joint_kv_budget_can_transfer_one_bit_from_v_to_k() -> None:
    routing = _routing()
    routing["budget_scope"] = "joint_kv"
    routing["offsets"]["K"] = [[1, 0, 0, 0], [0, 0, 0, 0]]
    routing["offsets"]["V"] = [[-1, 0, 0, 0], [0, 0, 0, 0]]
    audit = codec.validate_routing_table(routing)
    assert audit["budget_scope"] == "joint_kv"
    assert audit["counts"]["K"]["high"] == audit["counts"]["V"]["low"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_g128_encode_matches_eager() -> None:
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(20260730)
    x = torch.randn(
        (1, 4, 16, 128),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    kwargs = {
        "tensor_name": "K",
        "layer_idx": 0,
        "routing_table": _routing(),
        "bits": 4,
        "group_size": 128,
        "hadamard_group_size": 128,
    }
    eager = codec.encode_routed_gaussian_residual(x, **kwargs, use_fast=False)
    fast = codec.encode_routed_gaussian_residual(x, **kwargs, use_fast=True)
    assert torch.equal(fast["scale_raw"], eager["scale_raw"])
    for actual, expected in zip(fast["round_segments"], eager["round_segments"]):
        assert torch.equal(actual, expected)
    assert codec.payload_nbytes(fast) == codec.payload_nbytes(eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("bits", (1, 2, 3, 4, 8))
def test_fused_g128_decode_matches_eager(bits: int) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(20260725 + bits)
    x = torch.randn(
        (1, 4, 16, 128),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    state = codec.encode_routed_gaussian_residual(
        x,
        tensor_name="K",
        layer_idx=0,
        routing_table=_routing(),
        bits=bits,
        group_size=128,
        hadamard_group_size=128,
    )
    eager = codec.decode_routed_gaussian_residual(state, torch.bfloat16, use_fast=False)
    fused = codec.decode_routed_gaussian_residual(state, torch.bfloat16, use_fast=True)
    torch.testing.assert_close(fused, eager, rtol=0.0, atol=0.0)
