from __future__ import annotations

import copy

import pytest
import torch

import subitkv.r2aq as codec
import subitkv.sfbr_r2aq as sfbr


@pytest.fixture(autouse=True)
def _cpu_predictor(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_predictor(
        tensor: torch.Tensor, *, n_clusters: int, max_iters: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        batch, tokens, dim = tensor.shape
        ids = (
            torch.arange(tokens, device=tensor.device)
            .remainder(n_clusters)
            .expand(batch, -1)
            .contiguous()
        )
        centroids = torch.zeros(
            batch, n_clusters, dim, device=tensor.device, dtype=tensor.dtype
        )
        sizes = torch.zeros(
            batch, n_clusters, device=tensor.device, dtype=torch.int32
        )
        for cluster in range(n_clusters):
            mask = ids.eq(cluster)
            sizes[:, cluster] = mask.sum(1)
            centroids[:, cluster] = (
                tensor * mask[..., None]
            ).sum(1) / sizes[:, cluster, None].clamp_min(1)
        return ids, centroids, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)


def _routing() -> dict[str, object]:
    return {
        "schema_version": sfbr.ROUTING_SCHEMA,
        "num_layers": 2,
        "num_heads": 4,
        "offsets": {
            "K": [[-1, 0, 0, 1], [1, 0, 0, -1]],
            "V": [[0, -1, 1, 0], [0, 1, -1, 0]],
        },
    }


def _tensor() -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260725)
    return torch.randn(1, 4, 8, 128, generator=generator, dtype=torch.float32)


def _encode(bits: int = 4, *, tensor_name: str = "K", layer_idx: int = 0):
    return codec.encode_r2aq_residual(
        _tensor(),
        allocation_mode="static_functional",
        routing_table=_routing(),
        tensor_name=tensor_name,
        layer_idx=layer_idx,
        bits=bits,
        n_clusters=4,
        kmeans_max_iters=1,
        seed=17,
    )


def test_routing_validation_and_packing_budget() -> None:
    audit = sfbr.validate_routing_table(_routing())
    assert audit["counts"]["K"] == {"low": 2, "normal": 4, "high": 2}
    assert audit["counts"]["V"] == {"low": 2, "normal": 4, "high": 2}
    assert audit["packed_bytes"] == 4

    invalid = copy.deepcopy(_routing())
    invalid["offsets"]["K"][0][0] = 0
    with pytest.raises(ValueError, match="equal -1/\\+1"):
        sfbr.validate_routing_table(invalid)


@pytest.mark.parametrize("bits", (1, 2, 3, 4))
def test_single_stream_prefix_matches_direct_encode(bits: int) -> None:
    full = _encode(4)
    prefix = codec.truncate_r2aq_state(full, nominal_bits=bits)
    direct = _encode(bits)

    assert prefix["routed_depths"] == tuple(bits + item for item in (-1, 0, 0, 1))
    assert prefix["routing_table_id"] == direct["routing_table_id"]
    assert prefix["suffix"]["round_counts"] == direct["suffix"]["round_counts"]
    for actual, expected in zip(
        prefix["suffix"]["round_segments"], direct["suffix"]["round_segments"]
    ):
        assert torch.equal(actual, expected)
    assert codec.payload_nbytes(prefix) == codec.payload_nbytes(direct)
    assert torch.equal(
        codec.decode_r2aq_residual(prefix, torch.float32),
        codec.decode_r2aq_residual(direct, torch.float32),
    )


def test_round_planes_have_exact_global_average_budget() -> None:
    states = [
        _encode(4, tensor_name=tensor_name, layer_idx=layer_idx)
        for tensor_name in ("K", "V")
        for layer_idx in range(2)
    ]
    values = sum(torch.tensor(state["shape"]).prod().item() for state in states)
    plane_bits = sum(
        sum(int(value) for value in state["suffix"]["round_counts"])
        for state in states
    )
    assert plane_bits == 4 * values
    assert all(
        codec.payload_breakdown(state)["selector_payload_bytes"] == 0
        for state in states
    )


def test_shared_routing_is_charged_once_and_corruption_is_rejected() -> None:
    first = _encode(4, tensor_name="K", layer_idx=0)
    second = _encode(4, tensor_name="V", layer_idx=1)
    seen: set[str] = set()
    first_shared = codec.shared_payload_nbytes(first, seen)
    second_shared = codec.shared_payload_nbytes(second, seen)
    assert first_shared > 0
    # Reconstruction/threshold tables are K/V-specific; routing is the only
    # table de-duplicated between these two states.
    assert second_shared == first_shared - int(first["routing_table_packed"].numel())

    broken = dict(first)
    broken["routing_table_packed"] = first["routing_table_packed"].clone()
    broken["routing_table_packed"][0] ^= 1
    with pytest.raises(ValueError):
        codec.decode_r2aq_residual(broken)


def test_invalid_output_dtype_and_explicit_fast_path_fail_cleanly() -> None:
    state = _encode(2)
    with pytest.raises(TypeError, match="floating torch dtype"):
        codec.decode_r2aq_residual(state, torch.int32)
    with pytest.raises(ValueError, match="fused SFBR"):
        codec.decode_r2aq_residual(state, use_fast_path=True)


def test_calibration_depth_decode_is_explicit_and_not_payload() -> None:
    tensor = _tensor()
    state = sfbr.encode_sfbr_r2aq_residual(
        tensor,
        tensor_name="K",
        layer_idx=0,
        routing_table=_routing(),
        bits=4,
        n_clusters=4,
        kmeans_max_iters=1,
        seed=17,
        retain_calibration_code=True,
    )
    before = sfbr.payload_nbytes(state, include_shared=True)
    assert state["calibration"]["terminal_code"].numel() == tensor.numel()
    assert sfbr.payload_nbytes(state, include_shared=True) == before
    routed = sfbr.decode_sfbr_calibration_depths(
        state, state["routed_depths"], torch.float32
    )
    assert torch.equal(routed, sfbr.decode_sfbr_r2aq_residual(state, torch.float32))
    with pytest.raises(ValueError, match="does not retain"):
        sfbr.decode_sfbr_calibration_depths(_encode(4), (3, 4, 4, 5))
