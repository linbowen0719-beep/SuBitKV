from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import subitkv.hyworldplay8b_cache_codec as codec
from subitkv.hyworldplay8b_cache_codec import (
    AVG2_PROFILE_SCHEMA,
    TEMPORAL_PROFILE_SCHEMA,
    CODEC_QUANT_TYPES,
    _advanced_shared_bytes,
    _tensor_payload_breakdown,
    choose_post_truncation_bits,
    decode_vision_kv,
    append_encoded_vision_kv,
    encode_vision_kv,
    load_avg2_profile,
    make_codec_settings,
    tensor_tree_nbytes,
    tensor_tree_to,
)


def test_codec_settings_are_fixed_and_rvq_requires_codebooks() -> None:
    assert set(CODEC_QUANT_TYPES) == {
        "rvq_full", "rvq_ours_int2",
        "rvq_ours_avg1", "rvq_ours_avg2", "rvq_ours_avg3", "rvq_ours_avg4",
        "rvq_ours_avg0", "rvq_ours_avg0p2_alt", "rvq_ours_avg0p5",
        "kivi", "kivi_int2", "kivi_int4",
        "turboquant", "turboquant_int2", "turboquant_int4",
        "quantvideo", "quantvideo_int2", "quantvideo_int4",
    }
    assert make_codec_settings("kivi").quant_config.kivi_group_size == 32
    assert make_codec_settings("kivi_int2").quant_type == "kivi-all-int2"
    assert make_codec_settings("turboquant_int4").quant_type == "turboquant-prod-k4-v4"
    assert make_codec_settings("quantvideo_int2").quant_type == "triton-nstages-kmeans-int2"
    assert make_codec_settings("quantvideo").quant_config.kmeans_max_iters == 2
    try:
        make_codec_settings("rvq_full")
    except ValueError as error:
        assert "codebook" in str(error)
    else:
        raise AssertionError("rvq_full accepted missing codebooks")


def test_relative_stop_fast_approximate_has_independent_opt_out(
    tmp_path, monkeypatch
) -> None:
    required = []
    for name in ("routing.json", "tree.pt", "v.pt", "profile.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        required.append(path)

    def build():
        return make_codec_settings(
            "rvq_ours_avg1",
            k_rvq_codebook_dir=str(tmp_path / "k"),
            v_rvq_codebook_dir=str(tmp_path / "v"),
            routed_gaussian_routing_path=str(required[0]),
            routed_gaussian_tree_params_path=str(required[1]),
            progressive_v_absmax_params_path=str(required[2]),
            temporal_profile_path=str(required[3]),
        )

    monkeypatch.delenv(
        "HYWORLDPLAY_RVQ_RELATIVE_STOP_FAST_APPROXIMATE", raising=False
    )
    assert build().quant_config.rvq_relative_stop_fast_approximate is False
    monkeypatch.setenv("HYWORLDPLAY_RVQ_RELATIVE_STOP_FAST_APPROXIMATE", "1")
    assert build().quant_config.rvq_relative_stop_fast_approximate is True


def test_monolithic_exact_relative_stop_is_opt_in_and_train_routed(
    tmp_path, monkeypatch
) -> None:
    required = []
    for name in ("routing.json", "tree.pt", "v.pt", "profile.json"):
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        required.append(path)

    def build():
        return make_codec_settings(
            "rvq_ours_avg1",
            k_rvq_codebook_dir=str(tmp_path / "k"),
            v_rvq_codebook_dir=str(tmp_path / "v"),
            routed_gaussian_routing_path=str(required[0]),
            routed_gaussian_tree_params_path=str(required[1]),
            progressive_v_absmax_params_path=str(required[2]),
            temporal_profile_path=str(required[3]),
        ).quant_config

    monkeypatch.delenv(
        "HYWORLDPLAY_RVQ_RELATIVE_STOP_MONOLITHIC_EXACT", raising=False
    )
    assert build().rvq_relative_stop_monolithic_exact is False
    monkeypatch.setenv(
        "HYWORLDPLAY_RVQ_RELATIVE_STOP_MONOLITHIC_EXACT", "1"
    )
    config = build()
    assert config.rvq_relative_stop_monolithic_exact is True
    assert config.rvq_relative_stop_exact_staged_k_layers == (
        1, 2, 3, 5, 11, 13, 35, 36, 41, 42, 43, 44
    )
    assert config.rvq_relative_stop_exact_staged_v_layers == (
        4, 9, 14, 18, 19, 41
    )


def _avg2_rows_k() -> list[list[int]]:
    return [
        [16],
        [4, 8],
        [4, 4, 4],
        [4, 4, 4, 4],
        [4, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [3, 2, 3, 3, 3, 3, 3],
        [3, 2, 2, 2, 2, 3, 3, 3],
        [2, 2, 2, 2, 2, 2, 2, 3, 3],
        [2] * 10,
    ]


def _avg2_rows_v() -> list[list[int]]:
    return [
        [16],
        [8, 8],
        [4, 8, 8],
        [4, 4, 4, 8],
        [4, 4, 4, 4, 4],
        [3, 3, 3, 3, 4, 4],
        [3, 2, 3, 3, 3, 3, 3],
        [3, 2, 2, 2, 2, 3, 3, 3],
        [2, 2, 2, 2, 2, 2, 2, 3, 3],
        [2] * 10,
    ]


def _avg2_profile_payload() -> dict:
    tensor = lambda rows: {
        "capacity_bit_chunks": 20,
        "distortions": {"8": 0.1, "4": 0.2, "3": 0.3, "2": 0.4, "1": 0.5},
        "age_weights": [1.0] * 10,
        "joint_schedule": rows,
    }
    return {
        "schema_version": AVG2_PROFILE_SCHEMA,
        "allowed_bits": [16, 8, 4, 3, 2, 1],
        "horizon": 10,
        "tensor_profiles": {"K": tensor(_avg2_rows_k()), "V": tensor(_avg2_rows_v())},
    }


def test_avg2_profile_has_independent_monotone_k_v_capacity_20() -> None:
    profile = load_avg2_profile(_avg2_profile_payload())
    assert profile.schedule("K", 1) == (16,)
    assert profile.schedule("V", 1) == (16,)
    assert profile.schedule("K", 2) == (4, 8)
    assert profile.schedule("V", 2) == (8, 8)
    assert sum(profile.schedule("K", 10)) == 20
    assert sum(profile.schedule("V", 10)) == 20


def test_terminal_avg2_budget_uses_shared_model_wide_ownership() -> None:
    independent = [
        {
            "kv_shared_budget": False,
            "selected_k_bit_sum": 20,
            "selected_v_bit_sum": 20,
        }
        for _ in range(2)
    ]
    assert codec.validate_avg2_terminal_budget(
        independent, expected_layers=2
    )["ownership"] == "per_layer_per_tensor_independent"

    shared = [
        {
            "kv_shared_budget": True,
            "selected_k_bit_sum": 23,
            "selected_v_bit_sum": 17,
        },
        {
            "kv_shared_budget": True,
            "selected_k_bit_sum": 19,
            "selected_v_bit_sum": 21,
        },
    ]
    audit = codec.validate_avg2_terminal_budget(shared, expected_layers=2)
    assert audit == {
        "ownership": "model_wide_kv_shared",
        "K_bit_sum": 42,
        "V_bit_sum": 38,
        "combined_bit_sum": 80,
        "nominal_residual_avg_bits": 2.0,
        "K_capacity_bit_chunks": 20,
        "V_capacity_bit_chunks": 20,
        "K_terminal_bit_chunks": 20,
        "V_terminal_bit_chunks": 20,
        "profile_horizon": 10,
    }

    shared[1]["selected_v_bit_sum"] = 20
    with pytest.raises(RuntimeError, match="combined nominal budget"):
        codec.validate_avg2_terminal_budget(shared, expected_layers=2)


def test_avg2_profile_accepts_legacy_declaration_without_b16() -> None:
    raw = _avg2_profile_payload()
    raw["allowed_bits"] = [8, 4, 3, 2, 1]
    profile = load_avg2_profile(raw)
    assert profile.schedule("K", 1) == (16,)


def test_avg2_profile_rejects_precision_upgrade() -> None:
    raw = _avg2_profile_payload()
    raw["tensor_profiles"]["K"]["joint_schedule"][2][0] = 8
    with pytest.raises(ValueError, match="precision upgrade"):
        load_avg2_profile(raw)


def _avg1_profile_payload() -> dict:
    k_rows = [
        [8], [4, 4], [3, 3, 4], [2, 2, 3, 3], [2] * 5,
        [2, 1, 1, 2, 2, 2], [1, 1, 1, 1, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 2],
        [1] * 10,
    ]
    v_rows = [
        [8], [4, 4], [3, 3, 4], [2, 2, 3, 3], [1, 2, 2, 2, 3],
        [1, 1, 1, 2, 2, 3], [1, 1, 1, 1, 2, 2, 2],
        [1, 1, 1, 1, 1, 1, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 2],
        [1] * 10,
    ]
    tensor = lambda rows: {
        "capacity_bit_chunks": 10,
        "distortions": {"8": 0.1, "4": 0.2, "3": 0.3, "2": 0.4, "1": 0.5},
        "age_weights": [1.0] * 10,
        "joint_schedule": rows,
    }
    return {
        "schema_version": TEMPORAL_PROFILE_SCHEMA,
        "allowed_bits": [16, 8, 4, 3, 2, 1],
        "horizon": 10,
        "tensor_profiles": {"K": tensor(k_rows), "V": tensor(v_rows)},
    }


def test_avg1_profile_has_independent_capacity_10_and_terminal_audit() -> None:
    profile = load_avg2_profile(_avg1_profile_payload())
    assert profile.capacity("K") == profile.capacity("V") == 10
    assert profile.schedule("K", 1) == (8,)
    assert profile.schedule("V", 5) == (1, 2, 2, 2, 3)
    assert profile.schedule("K", 10) == (1,) * 10
    rows = [
        {
            "kv_shared_budget": False,
            "selected_k_bit_sum": 10,
            "selected_v_bit_sum": 10,
            "nominal_capacity_k_bit_chunks": 10,
            "nominal_capacity_v_bit_chunks": 10,
        }
        for _ in range(2)
    ]
    audit = codec.validate_avg2_terminal_budget(rows, expected_layers=2)
    assert audit["K_bit_sum"] == audit["V_bit_sum"] == 20
    assert audit["nominal_residual_avg_bits"] == 1.0


def test_avg2_encoder_uses_independent_frozen_schedules(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_avg2_profile_payload()), encoding="utf-8")
    config = SimpleNamespace()
    settings = codec.CodecSettings(
        name="rvq_ours_avg2",
        quant_type="unused",
        quant_config=config,
        frame_spatial_tokens=1,
        latent_frames_per_chunk=4,
        temporal_profile_path=str(profile_path),
    )
    observed: list[tuple[int, int, int]] = []

    monkeypatch.setattr(codec, "get_quantize_fn", lambda *_args: None)
    monkeypatch.setattr(codec, "_advanced_state_bytes", lambda _state: 10)
    monkeypatch.setattr(codec, "_advanced_shared_bytes", lambda _state: 0)

    def fake_compress(k, v, *_args, layer_idx, span_idx, **_kwargs):
        k_bits = int(config.temporal_adaptive_k_admission_target_bits)
        v_bits = int(config.temporal_adaptive_v_admission_target_bits)
        observed.append((span_idx, k_bits, v_bits))
        shape = tuple(k.shape)
        return {"bits": k_bits, "shape": shape}, {"bits": v_bits, "shape": shape}

    monkeypatch.setattr(codec, "compress_kv_cache", fake_compress)
    k = torch.zeros(2, 16, 8, 128, dtype=torch.bfloat16)
    v = torch.ones_like(k)
    packed_k, packed_v = codec._encode_ours_avg2(k, v, layer_idx=0, settings=settings)
    assert observed == [(0, 4, 8), (1, 8, 8)]
    accounting = packed_k["qvg_accounting"]
    assert accounting["selected_k_bits"] == [4, 8]
    assert accounting["selected_v_bits"] == [8, 8]
    assert accounting["terminal_k_bit_sum"] == 20
    assert accounting["terminal_v_bit_sum"] == 20
    assert packed_k["qvg_accounting"] is packed_v["qvg_accounting"]


def test_post_truncation_allocator_never_exceeds_physical_budget() -> None:
    sizes = [
        {0: 10, 1: 14, 2: 18, 3: 22, 4: 26},
        {0: 11, 1: 15, 2: 19, 3: 23, 4: 27},
    ]
    quality = [
        {0: 0.0, 1: 8.0, 2: 12.0, 3: 14.0, 4: 15.0},
        {0: 0.0, 1: 4.0, 2: 7.0, 3: 9.0, 4: 10.0},
    ]
    selected = choose_post_truncation_bits(sizes, quality, budget_bytes=40)
    assert sum(sizes[index][bits] for index, bits in enumerate(selected)) <= 40
    assert selected[0] >= selected[1]


def test_post_truncation_allocator_rejects_base_over_budget() -> None:
    try:
        choose_post_truncation_bits(
            [{0: 21, 1: 22, 2: 23, 3: 24, 4: 25}],
            [{0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}],
            budget_bytes=20,
        )
    except ValueError as error:
        assert "base-only" in str(error)
    else:
        raise AssertionError("allocator accepted a base larger than its budget")


def test_advanced_shared_metadata_is_charged_separately() -> None:
    assert _advanced_shared_bytes({"residual": {"shared_table_bytes": 321}}) == 321
    assert _advanced_shared_bytes({"residual": None}) == 0


def test_rvq_base_physical_payload_is_split_by_component() -> None:
    payload = {
        "indices": torch.zeros(3, 5, dtype=torch.uint8),
        "scale_q_packed": torch.zeros(7, dtype=torch.uint8),
        "scale_min": torch.zeros(2, dtype=torch.float16),
        "relative_stop_depth": torch.zeros(1, dtype=torch.uint8),
        "other": torch.zeros(4, dtype=torch.int32),
        "metadata": {"levels": 16},
    }
    result = _tensor_payload_breakdown(payload)
    assert result == {
        "indices_bytes": 15,
        "scales_bytes": 11,
        "early_stop_depth_bytes": 1,
        "other_tensor_bytes": 16,
        "total_bytes": 43,
    }


def test_tensor_tree_accounting_deduplicates_shared_storage() -> None:
    tensor = torch.ones(8, dtype=torch.float32)
    tree = {"a": tensor, "b": [tensor, {"ignored": "metadata"}]}
    assert tensor_tree_nbytes(tree) == tensor.untyped_storage().nbytes()
    moved = tensor_tree_to(tree, torch.device("cpu"))
    assert moved["a"].device.type == "cpu"
    assert moved["a"] is moved["b"][0]


def test_direct_codec_roundtrip_cpu() -> None:
    generator = torch.Generator().manual_seed(123)
    k = torch.randn(1, 2, 33, 128, generator=generator).to(torch.bfloat16)
    v = torch.randn(1, 2, 33, 128, generator=generator).to(torch.bfloat16)
    for name in ("kivi", "turboquant"):
        packed_k, packed_v = encode_vision_kv(
            k, v, layer_idx=0, settings=make_codec_settings(name)
        )
        k_hat, v_hat = decode_vision_kv(
            packed_k, packed_v, device=torch.device("cpu")
        )
        assert k_hat.shape == k.shape
        assert v_hat.shape == v.shape
        assert k_hat.dtype == torch.bfloat16


def test_fixed_codec_append_preserves_old_packed_segment() -> None:
    generator = torch.Generator().manual_seed(123)
    old_k = torch.randn(1, 2, 4, 128, generator=generator).to(torch.bfloat16)
    old_v = torch.randn(1, 2, 4, 128, generator=generator).to(torch.bfloat16)
    new_k = torch.randn(1, 2, 4, 128, generator=generator).to(torch.bfloat16)
    new_v = torch.randn(1, 2, 4, 128, generator=generator).to(torch.bfloat16)
    settings = codec.CodecSettings(
        name="kivi",
        quant_type=CODEC_QUANT_TYPES["kivi"],
        quant_config=make_codec_settings("kivi").quant_config,
        frame_spatial_tokens=1,
        latent_frames_per_chunk=4,
    )
    packed_k, packed_v = encode_vision_kv(
        old_k, old_v, layer_idx=0, settings=settings
    )
    original_k_tensors = {
        key: value
        for key, value in packed_k.items()
        if torch.is_tensor(value)
    }
    appended_k, appended_v = append_encoded_vision_kv(
        packed_k,
        packed_v,
        new_k,
        new_v,
        layer_idx=0,
        settings=settings,
    )
    first = appended_k["segments"][0]["cache"]
    assert first is packed_k
    assert all(first[key] is value for key, value in original_k_tensors.items())
    assert appended_k["qvg_accounting"]["old_spans_reencoded"] is False
    assert appended_k["qvg_accounting"]["new_spans_encoded"] == 1
    k_hat, v_hat = decode_vision_kv(
        appended_k, appended_v, device=torch.device("cpu")
    )
    assert k_hat.shape == (1, 2, 8, 128)
    assert v_hat.shape == (1, 2, 8, 128)
    # The experimental temporal fast backend must fall back cleanly for
    # non-temporal segmented codecs such as KIVI.
    import subitkv.uncompress as restore

    k_fast, v_fast = restore.uncompress_kv_cache(
        appended_k, appended_v, segmented_backend="preallocated"
    )
    torch.testing.assert_close(k_fast, k_hat)
    torch.testing.assert_close(v_fast, v_hat)


def test_fixed_codec_append_accepts_native_state_without_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QuantVideo PRQ payloads derive shape and have no top-level shape key."""

    settings = codec.CodecSettings(
        name="quantvideo",
        quant_type=CODEC_QUANT_TYPES["quantvideo"],
        quant_config=make_codec_settings("quantvideo").quant_config,
        frame_spatial_tokens=1,
        latent_frames_per_chunk=4,
    )
    old_k = {"quant_method": "native", "payload": torch.zeros(4)}
    old_v = {"quant_method": "native", "payload": torch.ones(4)}
    new_k_state = {"quant_method": "native", "payload": torch.full((4,), 2.0)}
    new_v_state = {"quant_method": "native", "payload": torch.full((4,), 3.0)}
    monkeypatch.setattr(codec, "get_quantize_fn", lambda *_args: None)
    monkeypatch.setattr(
        codec,
        "compress_kv_cache",
        lambda *_args, **_kwargs: (new_k_state, new_v_state),
    )
    new_k = torch.zeros(1, 2, 4, 128, dtype=torch.bfloat16)
    new_v = torch.ones_like(new_k)

    appended_k, appended_v = append_encoded_vision_kv(
        old_k,
        old_v,
        new_k,
        new_v,
        layer_idx=0,
        settings=settings,
    )

    assert [(row["start"], row["end"]) for row in appended_k["segments"]] == [
        (0, 4),
        (4, 8),
    ]
    assert appended_k["segments"][0]["cache"] is old_k
    assert appended_v["segments"][0]["cache"] is old_v


def test_avg2_append_transitions_old_and_encodes_only_new_span(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_avg2_profile_payload()), encoding="utf-8")
    config = SimpleNamespace()
    settings = codec.CodecSettings(
        name="rvq_ours_avg2",
        quant_type="unused",
        quant_config=config,
        frame_spatial_tokens=1,
        latent_frames_per_chunk=4,
        temporal_profile_path=str(profile_path),
    )
    old_shape = (2, 16, 4, 128)
    old_k_state = {"bits": 16, "shape": old_shape, "identity": "old-k"}
    old_v_state = {"bits": 16, "shape": old_shape, "identity": "old-v"}
    old_k = codec._segmented([old_k_state], torch.bfloat16, {})
    old_v = codec._segmented([old_v_state], torch.bfloat16, {})
    transitions = []
    admissions = []

    def fake_transition(state, bits, **_kwargs):
        transitions.append((state["identity"], bits))
        return {**state, "bits": bits}

    def fake_compress(k, v, *_args, layer_idx, span_idx, **_kwargs):
        admissions.append((layer_idx, span_idx))
        shape = tuple(k.shape)
        return (
            {"bits": config.temporal_adaptive_k_admission_target_bits, "shape": shape, "identity": "new-k"},
            {"bits": config.temporal_adaptive_v_admission_target_bits, "shape": shape, "identity": "new-v"},
        )

    monkeypatch.setattr(codec, "get_quantize_fn", lambda *_args: None)
    monkeypatch.setattr(codec, "transition_state", fake_transition)
    monkeypatch.setattr(codec, "compress_kv_cache", fake_compress)
    monkeypatch.setattr(codec, "_advanced_state_bytes", lambda _state: 10)
    monkeypatch.setattr(codec, "_advanced_shared_bytes", lambda _state: 0)
    new_k = torch.zeros(old_shape, dtype=torch.bfloat16)
    new_v = torch.ones_like(new_k)
    appended_k, appended_v = append_encoded_vision_kv(
        old_k,
        old_v,
        new_k,
        new_v,
        layer_idx=7,
        settings=settings,
    )
    assert admissions == [(7, 1)]
    assert transitions == [("old-k", 4), ("old-v", 8)]
    assert [row["cache"]["bits"] for row in appended_k["segments"]] == [4, 8]
    assert [row["cache"]["bits"] for row in appended_v["segments"]] == [8, 8]
    accounting = appended_k["qvg_accounting"]
    assert accounting["append_only_history"] is True
    assert accounting["old_spans_reencoded"] is False
    assert accounting["rvq_indices_may_be_reencoded"] is False


def test_avg2_append_reuses_unchanged_span_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_avg2_profile_payload()), encoding="utf-8")
    config = SimpleNamespace()
    settings = codec.CodecSettings(
        name="rvq_ours_avg2",
        quant_type="unused",
        quant_config=config,
        frame_spatial_tokens=1,
        latent_frames_per_chunk=4,
        temporal_profile_path=str(profile_path),
    )
    shape = (2, 16, 4, 128)
    k0, k1 = {"bits": 4, "shape": shape}, {"bits": 8, "shape": shape}
    v0, v1 = {"bits": 8, "shape": shape}, {"bits": 8, "shape": shape}
    transitions: list[tuple[int, int]] = []

    def fake_transition(state, bits, **_kwargs):
        transitions.append((id(state), int(bits)))
        return {**state, "bits": int(bits)}

    def fake_compress(k, v, *_args, **_kwargs):
        return (
            {"bits": int(config.temporal_adaptive_k_admission_target_bits), "shape": tuple(k.shape)},
            {"bits": int(config.temporal_adaptive_v_admission_target_bits), "shape": tuple(v.shape)},
        )

    monkeypatch.setattr(codec, "get_quantize_fn", lambda *_args: None)
    monkeypatch.setattr(codec, "transition_state", fake_transition)
    monkeypatch.setattr(codec, "compress_kv_cache", fake_compress)
    monkeypatch.setattr(codec, "_advanced_state_bytes", lambda _state: 10)
    monkeypatch.setattr(codec, "_advanced_shared_bytes", lambda _state: 0)
    new = torch.zeros(shape, dtype=torch.bfloat16)
    appended_k, appended_v = append_encoded_vision_kv(
        codec._segmented([k0, k1], torch.bfloat16, {}),
        codec._segmented([v0, v1], torch.bfloat16, {}),
        new,
        new,
        layer_idx=0,
        settings=settings,
    )
    assert appended_k["segments"][0]["cache"] is k0
    assert appended_v["segments"][1]["cache"] is v1
    assert transitions == [(id(k1), 4), (id(v0), 4)]


def test_preallocated_segmented_temporal_decode_matches_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subitkv.uncompress as restore

    states = [
        {
            "quant_method": restore.TEMPORAL_ADAPTIVE_RESIDUAL_METHOD,
            "shape": (1, 2, 3, 4),
            "value": torch.full((1, 2, 3, 4), float(index)),
        }
        for index in range(3)
    ]
    segmented = codec._segmented(states, torch.bfloat16, {})
    monkeypatch.setattr(
        restore,
        "decode_temporal_adaptive_residual_state",
        lambda state, output_dtype=torch.bfloat16: state["value"].to(output_dtype),
    )
    batch_calls: list[tuple[int, int, int]] = []

    def fake_batch(
        items, *, leading_tokens=0, trailing_tokens=0, **_kwargs
    ):
        batch_calls.append(
            (len(items), int(leading_tokens), int(trailing_tokens))
        )
        history = torch.cat([item["value"] for item in items], dim=2).to(
            torch.bfloat16
        )
        if not leading_tokens and not trailing_tokens:
            return history
        output = torch.empty(
            1,
            2,
            int(leading_tokens) + int(history.shape[2]) + int(trailing_tokens),
            4,
            dtype=torch.bfloat16,
        )
        output[
            :, :, int(leading_tokens) : int(leading_tokens) + history.shape[2]
        ].copy_(history)
        return output

    monkeypatch.setattr(
        restore, "uncompress_temporal_adaptive_residual_states", fake_batch
    )
    legacy = restore.uncompress_single_kv_cache(
        segmented, segmented_backend="legacy"
    )
    preallocated = restore.uncompress_single_kv_cache(
        segmented, segmented_backend="preallocated"
    )
    reserved = restore.uncompress_single_kv_cache(
        segmented,
        segmented_backend="preallocated",
        leading_tokens=2,
        trailing_tokens=4,
    )
    torch.testing.assert_close(preallocated, legacy, rtol=0, atol=0)
    torch.testing.assert_close(reserved[:, :, 2:11], legacy, rtol=0, atol=0)
    assert reserved.shape == (1, 2, 15, 4)
    assert batch_calls == [(3, 0, 0), (3, 2, 4)]
