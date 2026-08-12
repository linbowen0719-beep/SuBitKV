from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from quant_videogen import progressive_sign_adaptive_absmax_scalar as progressive
from quant_videogen import compress as compress_mod
from quant_videogen import temporal_adaptive_residual as temporal


def test_b16_is_bit_exact_and_v_prefixes_physically_shrink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    generator = torch.Generator().manual_seed(20260730)
    source = torch.randn(1, 2, 9, 128, generator=generator).to(torch.bfloat16)
    base_value = torch.zeros_like(source)
    base = {
        "quant_method": "rvq",
        "indices": torch.arange(8, dtype=torch.uint8),
    }
    state = temporal.create_b16_state(
        source, base, tensor_name="V", layer_idx=3
    )
    assert state["base"] is base
    assert torch.equal(temporal.decode_state(state), source)

    params_path = tmp_path / "progressive_v.pt"
    torch.save(
        progressive.build_midpoint_params(method="midpoint_msb"),
        params_path,
    )
    config = SimpleNamespace(
        progressive_v_absmax_params_path=str(params_path),
        progressive_v_absmax_method="midpoint_msb",
        progressive_v_absmax_group_size=128,
    )
    def fake_rvq_decode(packed, dtype, **kwargs):
        residual_source = kwargs.get("_residual_source")
        if residual_source is not None:
            return (
                residual_source.to(dtype) - base_value.to(dtype)
            ).contiguous()
        return base_value.to(dtype)

    monkeypatch.setattr(
        temporal, "rvq_dequantize_tensor_cache", fake_rvq_decode
    )
    b8 = temporal.transition_state(state, 8, quant_config=config)
    direct_b4 = temporal.transition_state(state, 4, quant_config=config)
    b4 = temporal.transition_state(b8, 4, quant_config=config)
    b3 = temporal.transition_state(b4, 3, quant_config=config)
    b3_hot_path = temporal.transition_state(
        b4, 3, quant_config=config, verify_identity=False
    )
    b2 = temporal.transition_state(b3, 2, quant_config=config)
    b1 = temporal.transition_state(b2, 1, quant_config=config)

    assert b8["base"] is base
    assert b3_hot_path["base"] is base
    assert b3_hot_path["rvq_identity_hash"] == b3["rvq_identity_hash"]
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            b3_hot_path["residual"]["planes"], b3["residual"]["planes"]
        )
    )
    assert b4["base"] is base
    assert direct_b4["bits"] == 4
    assert direct_b4["diagnostics"]["materialized_direct_target_bits"] == 4
    assert torch.equal(
        direct_b4["residual"]["scales"].view(torch.uint8),
        b4["residual"]["scales"].view(torch.uint8),
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            direct_b4["residual"]["planes"], b4["residual"]["planes"]
        )
    )
    assert b4["rvq_identity_hash"] == state["rvq_identity_hash"]
    assert b2["rvq_identity_hash"] == state["rvq_identity_hash"]
    assert b1["rvq_identity_hash"] == state["rvq_identity_hash"]
    assert b8["bf16_bypass"] is None
    assert b8["local_payload_bytes"] > b4["local_payload_bytes"]
    assert b4["local_payload_bytes"] > b3["local_payload_bytes"]
    assert b3["local_payload_bytes"] > b2["local_payload_bytes"]
    assert b2["local_payload_bytes"] > b1["local_payload_bytes"]
    assert len(b8["residual"]["planes"]) == 8
    assert len(b4["residual"]["planes"]) == 4
    assert len(b3["residual"]["planes"]) == 3
    assert len(b2["residual"]["planes"]) == 2
    assert len(b1["residual"]["planes"]) == 1
    assert torch.isfinite(temporal.decode_state(b1).float()).all()


def test_fast_rvq_identity_guard_detects_inplace_mutation() -> None:
    source = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    indices = torch.arange(8, dtype=torch.uint8)
    state = temporal.create_b16_state(
        source,
        {"quant_method": "rvq", "indices": indices},
        tensor_name="K",
        layer_idx=0,
    )
    assert temporal.rvq_identity_is_valid(state)
    indices[0].add_(1)
    assert not temporal.rvq_identity_is_valid(state)


def test_fast_admission_identity_avoids_portable_hash_until_audit() -> None:
    source = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    indices = torch.arange(8, dtype=torch.uint8)
    state = temporal.create_b16_state(
        source,
        {"quant_method": "rvq", "indices": indices},
        tensor_name="K",
        layer_idx=0,
        portable_identity=False,
    )
    assert state["rvq_identity_portable"] is False
    assert state["rvq_identity_hash"].startswith("runtime:")
    assert "rvq_portable_identity_hash" not in state
    assert temporal.rvq_identity_is_valid(state)
    assert temporal.rvq_identity_is_valid(state, full_audit=True)
    assert "rvq_portable_identity_hash" in state
    indices[0].add_(1)
    assert not temporal.rvq_identity_is_valid(state)
    assert not temporal.rvq_identity_is_valid(state, full_audit=True)


def test_temporal_admission_encodes_directly_to_scheduler_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.zeros(1, 1, 4, 128, dtype=torch.bfloat16)
    base = {"quant_method": "rvq", "indices": torch.zeros(1, dtype=torch.uint8)}
    calls: list[tuple[int, int]] = []

    monkeypatch.setattr(
        compress_mod,
        "create_b16_state",
        lambda *_args, **_kwargs: {"bits": 16, "base": base},
    )

    def fake_transition(
        state, target_bits, *, quant_config, verify_identity=True
    ):
        calls.append((int(state["bits"]), int(target_bits)))
        assert verify_identity is False
        return {**state, "bits": int(target_bits), "bf16_bypass": None}

    monkeypatch.setattr(compress_mod, "transition_state", fake_transition)
    config = SimpleNamespace(
        temporal_adaptive_residual=True,
        temporal_adaptive_admission_target_bits=3,
    )
    result = compress_mod._quantize_rvq_residual(
        source,
        base,
        tensor_name="K",
        quant_type="unused",
        quant_config=config,
        quantize_fn=lambda _tensor: None,
        num_centroids=256,
        layer_idx=7,
    )

    assert result["bits"] == 3
    assert result["bf16_bypass"] is None
    assert calls == [(16, 3)]


@pytest.mark.parametrize(("tensor_name", "expected"), (("K", 4), ("V", 3)))
def test_temporal_admission_honors_tensor_specific_target(
    monkeypatch: pytest.MonkeyPatch,
    tensor_name: str,
    expected: int,
) -> None:
    source = torch.zeros(1, 1, 4, 128, dtype=torch.bfloat16)
    base = {"quant_method": "rvq", "indices": torch.zeros(1, dtype=torch.uint8)}
    monkeypatch.setattr(
        compress_mod,
        "create_b16_state",
        lambda *_args, **_kwargs: {"bits": 16, "base": base},
    )
    monkeypatch.setattr(
        compress_mod,
        "transition_state",
        lambda state, target_bits, **_kwargs: {
            **state,
            "bits": int(target_bits),
        },
    )
    config = SimpleNamespace(
        temporal_adaptive_residual=True,
        temporal_adaptive_admission_target_bits=2,
        temporal_adaptive_k_admission_target_bits=4,
        temporal_adaptive_v_admission_target_bits=3,
    )
    result = compress_mod._quantize_rvq_residual(
        source,
        base,
        tensor_name=tensor_name,
        quant_type="unused",
        quant_config=config,
        quantize_fn=lambda _tensor: None,
        num_centroids=256,
        layer_idx=7,
    )
    assert result["bits"] == expected


def test_scheduler_has_ten_monotone_rows_and_exact_terminal_budget() -> None:
    profile = temporal.DistortionProfile(
        distortions={
            16: 0.0, 8: 0.2, 4: 1.0, 3: 1.8, 2: 3.2, 1: 7.0
        },
        age_weights=(1.0, 0.8, 0.6, 0.4, 0.25, 0.16, 0.1),
        first_anchor_weight=1.7,
    )
    schedules: list[tuple[int, ...]] = []
    current: list[int] = []
    for _step in range(10):
        current.append(16)
        current = list(
            temporal.schedule_precisions(
                current, profile, capacity_bit_chunks=40
            )
        )
        schedules.append(tuple(current))

    assert len(schedules) == 10
    assert sum(schedules[-1]) == 40
    assert all(sum(row) <= 40 for row in schedules)
    for previous, present in zip(schedules, schedules[1:]):
        assert all(
            temporal.ALLOWED_BITS.index(new)
            >= temporal.ALLOWED_BITS.index(old)
            for old, new in zip(previous, present)
        )
    assert all(
        value in temporal.ALLOWED_BITS for row in schedules for value in row
    )


def test_profile_rejects_nonmonotone_distortion() -> None:
    raw = {
        "schema_version": temporal.PROFILE_SCHEMA,
        "distortions": {
            "16": 0.0,
            "8": 0.2,
            "4": 1.0,
            "3": 0.5,
            "2": 2.0,
            "1": 4.0,
        },
        "age_weights": [1.0],
    }
    with pytest.raises(ValueError, match="D16<=D8<=D4<=D3<=D2<=D1"):
        temporal.load_profile(raw)


def test_force_recent_b16_uses_b2_b3_at_exact_average_four() -> None:
    profile = temporal.DistortionProfile(
        distortions={
            16: 0.0,
            8: 0.2,
            4: 1.0,
            3: 4.046730820153158,
            2: 18.6456943551374,
            1: 80.0,
        },
        age_weights=(
            1.9103240546221123,
            1.1573130208157982,
            1.0323978418061701,
            0.8908389,
            0.7948952,
            0.738077,
        ),
        first_anchor_weight=0.9862643754062901,
    )
    schedules: list[tuple[int, ...]] = []
    current: list[int] = []
    for _step in range(10):
        current.append(16)
        current = list(
            temporal.schedule_precisions(
                current,
                profile,
                capacity_bit_chunks=40,
                allocation_mode="force_recent_b16",
            )
        )
        schedules.append(tuple(current))

    terminal = schedules[-1]
    assert terminal[-1] == 16
    assert sum(terminal) == 40
    assert 2 in terminal
    assert 3 in terminal
    assert all(sum(row) <= 40 for row in schedules)
    for previous, present in zip(schedules, schedules[1:]):
        assert all(
            temporal.ALLOWED_BITS.index(new)
            >= temporal.ALLOWED_BITS.index(old)
            for old, new in zip(previous, present)
        )


def test_frozen_joint_schedule_is_followed_and_validated() -> None:
    rows = (
        (16,),
        (16, 16),
        (4, 16, 16),
        (4, 4, 16, 16),
        (4, 4, 4, 4, 16),
        (4, 4, 4, 4, 4, 16),
        (4, 4, 4, 4, 4, 4, 16),
        (4, 3, 3, 3, 3, 4, 4, 16),
        (3, 3, 3, 3, 3, 3, 3, 3, 16),
        (3, 3, 2, 2, 2, 3, 3, 3, 3, 16),
    )
    profile = temporal.DistortionProfile(
        distortions={
            16: 0.0, 8: 0.2, 4: 1.0, 3: 4.0, 2: 18.0, 1: 70.0
        },
        age_weights=(1.0,),
        joint_schedule=rows,
    )
    current: list[int] = []
    observed = []
    for _step in range(10):
        current.append(16)
        current = list(
            temporal.schedule_precisions(
                current,
                profile,
                capacity_bit_chunks=40,
                allocation_mode="frozen_joint_schedule",
            )
        )
        observed.append(tuple(current))
    assert tuple(observed) == rows
    assert sum(observed[-1]) == 40
    assert 2 in observed[-1] and 3 in observed[-1]


@pytest.mark.parametrize(
    ("capacity", "rows"),
    [
        (
            10,
            (
                (4,),
                (4, 4),
                (3, 3, 4),
                (2, 2, 3, 3),
                (2, 2, 2, 2, 2),
                (1, 1, 2, 2, 2, 2),
                (1, 1, 1, 1, 2, 2, 2),
                (1, 1, 1, 1, 1, 1, 2, 2),
                (1, 1, 1, 1, 1, 1, 1, 1, 2),
                (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
            ),
        ),
        (
            20,
            (
                (16,),
                (4, 16),
                (4, 4, 4),
                (4, 4, 4, 4),
                (4, 4, 4, 4, 4),
                (3, 3, 3, 3, 4, 4),
                (3, 2, 3, 3, 3, 3, 3),
                (3, 2, 2, 2, 2, 3, 3, 3),
                (2, 2, 2, 2, 2, 2, 2, 3, 3),
                (2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
            ),
        ),
        (
            30,
            (
                (16,),
                (4, 16),
                (4, 4, 16),
                (4, 4, 4, 16),
                (4, 4, 4, 4, 4),
                (4, 4, 4, 4, 4, 4),
                (4, 4, 4, 4, 4, 4, 4),
                (4, 3, 3, 4, 4, 4, 4, 4),
                (3, 3, 3, 3, 3, 3, 4, 4, 4),
                (3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
            ),
        ),
    ],
)
def test_frozen_joint_average_two_and_three(
    capacity: int,
    rows: tuple[tuple[int, ...], ...],
) -> None:
    profile = temporal.DistortionProfile(
        distortions={
            16: 0.0, 8: 0.2, 4: 1.0, 3: 4.0, 2: 18.0, 1: 70.0
        },
        age_weights=(1.0,),
        joint_schedule=rows,
        joint_schedule_capacity_bit_chunks=capacity,
    )
    current: list[int] = []
    for expected in rows:
        current.append(16)
        current = list(
            temporal.schedule_precisions(
                current,
                profile,
                capacity_bit_chunks=capacity,
                allocation_mode="frozen_joint_schedule",
            )
        )
        assert tuple(current) == expected
