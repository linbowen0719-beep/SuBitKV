from __future__ import annotations

import copy

import pytest
import torch

import subitkv.r2aq as codec


def _install_predictor(monkeypatch: pytest.MonkeyPatch, *, nonzero: bool = False) -> None:
    def fake_predictor(flattened, *, n_clusters, max_iters):
        batch_heads, tokens, dim = flattened.shape
        ids = (
            torch.arange(tokens, device=flattened.device)
            .remainder(n_clusters)
            .expand(batch_heads, -1)
            .contiguous()
        )
        centers = torch.zeros(
            (batch_heads, n_clusters, dim),
            device=flattened.device,
            dtype=flattened.dtype,
        )
        if nonzero:
            values = torch.linspace(
                -2.1,
                2.3,
                n_clusters * dim,
                device=flattened.device,
                dtype=torch.float32,
            ).reshape(n_clusters, dim)
            centers[:] = values.to(flattened.dtype)
        sizes = torch.zeros(
            (batch_heads, n_clusters),
            device=flattened.device,
            dtype=torch.int64,
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(codec, "_online_vector_predictor", fake_predictor)


def test_production_rate_matches_qv_after_counted_selector_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn(
        (1, 1, 7040, 128),
        generator=torch.Generator().manual_seed(7),
        dtype=torch.bfloat16,
    )
    full = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        max_nominal_bits=4,
        n_clusters=256,
        predictor_group_size=128,
        suffix_group_size=64,
        selection_fraction=0.434091,
    )

    assert full["suffix"]["selected_group_count"] == 6098
    assert full["suffix"]["selector_prefix_offsets"].shape == (56,)
    assert full["suffix"]["selector_prefix_offsets"].dtype == torch.int16
    for bits in (1, 2, 3, 4):
        state = codec.truncate_r2aq_state(full, nominal_bits=bits)
        assert codec.physical_bits_per_value(state) == pytest.approx(
            bits + 0.7693181818181818
        )
        breakdown = codec.payload_breakdown(state)
        assert breakdown["selector_payload_bytes"] == 112
        assert sum(
            value
            for key, value in breakdown.items()
            if key != "shared_table_bytes"
        ) == codec.payload_nbytes(state)


def test_coherent_tail_is_v_only_b4_rate_matched_and_physically_truncatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn(
        (1, 1, 7040, 128),
        generator=torch.Generator().manual_seed(19),
        dtype=torch.bfloat16,
    )
    utility = torch.ones((30, 2, 5, 2), dtype=torch.float16)
    common = {
        "max_nominal_bits": 4,
        "n_clusters": 256,
        "selection_fraction": 0.434091,
        "utility_weights": utility,
        "layer_idx": 0,
        "age_bin": 0,
    }
    control = codec.encode_r2aq_residual(
        source,
        tensor_name="V",
        ablation="full_tree",
        **common,
    )
    state = codec.encode_r2aq_residual(
        source,
        tensor_name="V",
        ablation="coherent_tail",
        coherent_tail_group_fraction=0.55,
        **common,
    )
    assert codec.payload_nbytes(state) == codec.payload_nbytes(control)
    assert (
        state["suffix"]["selected_group_count"]
        == control["suffix"]["selected_group_count"]
    )
    torch.testing.assert_close(
        state["suffix"]["scale_raw"],
        control["suffix"]["scale_raw"],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        state["suffix"]["selector_prefix_offsets"],
        control["suffix"]["selector_prefix_offsets"],
        atol=0,
        rtol=0,
    )
    tail = state["coherent_tail"]
    assert tail["applies_at_nominal_bits"] == 4
    assert tail["selected_group_count"] == 280
    assert tail["group_ids"].dtype == torch.int16
    assert bool(
        (
            tail["group_ids"][1:].to(torch.int64)
            > tail["group_ids"][:-1].to(torch.int64)
        ).all()
    )
    breakdown = codec.payload_breakdown(state)
    assert breakdown["tail_value_payload_bytes"] == 280 * 32
    assert breakdown["tail_scale_payload_bytes"] == 280
    assert breakdown["tail_id_payload_bytes"] == 280 * 2
    terminal = state["suffix"][
        "terminal_refinement_selected_count"
    ]
    assert (
        state["suffix"]["selected_group_count"] - terminal
    ) * 8 == 280 * 35
    assert sum(
        value
        for key, value in breakdown.items()
        if key != "shared_table_bytes"
    ) == codec.payload_nbytes(state)

    for bits in (1, 2, 3):
        prefix = codec.truncate_r2aq_state(state, nominal_bits=bits)
        control_prefix = codec.truncate_r2aq_state(
            control, nominal_bits=bits
        )
        assert "coherent_tail" not in prefix
        assert (
            "terminal_refinement_selected_count"
            not in prefix["suffix"]
        )
        assert codec.payload_breakdown(prefix)[
            "tail_value_payload_bytes"
        ] == 0
        assert codec.payload_nbytes(prefix) == codec.payload_nbytes(
            control_prefix
        )
        for name in (
            "scale_raw",
            "selector_prefix_offsets",
            "padding",
        ):
            torch.testing.assert_close(
                prefix["suffix"][name],
                control_prefix["suffix"][name],
                atol=0,
                rtol=0,
            )
        for actual, expected in zip(
            prefix["suffix"]["segments"],
            control_prefix["suffix"]["segments"],
        ):
            torch.testing.assert_close(
                actual, expected, atol=0, rtol=0
            )
        torch.testing.assert_close(
            codec.decode_r2aq_residual(prefix),
            codec.decode_r2aq_residual(control_prefix),
            atol=0,
            rtol=0,
        )
    b4 = codec.truncate_r2aq_state(state, nominal_bits=4)
    assert b4["coherent_tail"]["group_ids"].data_ptr() != tail[
        "group_ids"
    ].data_ptr()
    assert codec.decode_r2aq_residual(b4).shape == source.shape
    with pytest.raises(ValueError, match="coherent_tail"):
        codec.decode_r2aq_residual(b4, use_fast_path=True)


def test_coherent_tail_wire_corruption_and_direct_lowbit_encode_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn((1, 1, 512, 128), dtype=torch.bfloat16)
    utility = torch.ones((1, 2, 5, 2), dtype=torch.float16)
    kwargs = {
        "tensor_name": "V",
        "n_clusters": 256,
        "ablation": "coherent_tail",
        "utility_weights": utility,
        "layer_idx": 0,
        "age_bin": 0,
        "selection_fraction": 0.434091,
    }
    with pytest.raises(ValueError, match="encode B4 then truncate"):
        codec.encode_r2aq_residual(source, bits=3, **kwargs)
    state = codec.encode_r2aq_residual(source, bits=4, **kwargs)

    missing = copy.deepcopy(state)
    missing.pop("coherent_tail")
    with pytest.raises(ValueError, match="requires a tail mapping"):
        codec.decode_r2aq_residual(missing)

    duplicate = copy.deepcopy(state)
    if duplicate["coherent_tail"]["group_ids"].numel() > 1:
        duplicate["coherent_tail"]["group_ids"][1] = duplicate[
            "coherent_tail"
        ]["group_ids"][0]
        with pytest.raises(ValueError, match="unique sorted"):
            codec.decode_r2aq_residual(duplicate)

    nan_scale = copy.deepcopy(state)
    nan_scale["coherent_tail"]["scale_raw"][0] = 0x7F
    with pytest.raises(ValueError, match="positive finite FP8"):
        codec.decode_r2aq_residual(nan_scale)

    invalid_prefix = codec.truncate_r2aq_state(state, nominal_bits=3)
    invalid_prefix["coherent_tail"] = copy.deepcopy(state["coherent_tail"])
    with pytest.raises(ValueError, match="only valid for V at nominal B4"):
        codec.decode_r2aq_residual(invalid_prefix)

    invalid_terminal = copy.deepcopy(state)
    invalid_terminal["suffix"][
        "terminal_refinement_selected_count"
    ] += 1
    with pytest.raises(
        ValueError, match="segment 7|exactly replace terminal"
    ):
        codec.decode_r2aq_residual(invalid_terminal)


def test_coherent_tail_g128_is_optional_prefix_safe_equal_rate_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn(
        (1, 1, 7040, 128),
        generator=torch.Generator().manual_seed(117),
        dtype=torch.bfloat16,
    )
    utility = torch.ones((30, 2, 5, 2), dtype=torch.float16)
    common = {
        "tensor_name": "V",
        "max_nominal_bits": 4,
        "n_clusters": 256,
        "selection_fraction": 0.434091,
        "utility_weights": utility,
        "layer_idx": 0,
        "age_bin": 0,
    }
    control = codec.encode_r2aq_residual(
        source, ablation="full_tree", **common
    )
    state = codec.encode_r2aq_residual(
        source,
        ablation="coherent_tail_g128",
        coherent_tail_group_fraction=1.0,
        **common,
    )
    tail = state["coherent_tail"]
    assert tail["group_size"] == 128
    assert tail["channel_groups"] == 1
    assert tail["selected_group_count"] == 256
    breakdown = codec.payload_breakdown(state)
    assert breakdown["tail_value_payload_bytes"] == 256 * 64
    assert breakdown["tail_scale_payload_bytes"] == 256
    assert breakdown["tail_id_payload_bytes"] == 256 * 2
    terminal = state["suffix"][
        "terminal_refinement_selected_count"
    ]
    assert (
        state["suffix"]["selected_group_count"] - terminal
    ) * 8 == 256 * 67
    assert codec.payload_nbytes(state) == codec.payload_nbytes(control)
    for bits in (1, 2, 3):
        prefix = codec.truncate_r2aq_state(state, nominal_bits=bits)
        expected = codec.truncate_r2aq_state(
            control, nominal_bits=bits
        )
        assert codec.payload_nbytes(prefix) == codec.payload_nbytes(expected)
        torch.testing.assert_close(
            codec.decode_r2aq_residual(prefix),
            codec.decode_r2aq_residual(expected),
            atol=0,
            rtol=0,
        )
    assert codec.decode_r2aq_residual(state).shape == source.shape
    with pytest.raises(ValueError, match="coherent_tail"):
        codec.decode_r2aq_residual(state, use_fast_path=True)


def test_prefix_decode_is_monotone_and_truncation_owns_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn(
        (1, 1, 256, 128),
        generator=torch.Generator().manual_seed(11),
        dtype=torch.bfloat16,
    )
    full = codec.encode_r2aq_residual(
        source,
        tensor_name="V",
        max_nominal_bits=4,
        n_clusters=256,
    )
    previous = None
    for bits in (1, 2, 3, 4):
        state = codec.truncate_r2aq_state(full, nominal_bits=bits)
        decoded = codec.decode_r2aq_residual(state, torch.bfloat16)
        error = float((decoded.float() - source.float()).square().mean())
        if previous is not None:
            assert error <= previous
        previous = error
        assert decoded.shape == source.shape
        assert decoded.dtype == source.dtype
        for retained, original in zip(
            state["suffix"]["segments"], full["suffix"]["segments"]
        ):
            assert retained.data_ptr() != original.data_ptr()


def test_suffix_is_recomputed_from_decoded_int4_centroids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch, nonzero=True)
    source = torch.zeros((1, 1, 256, 128), dtype=torch.bfloat16)
    state = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        max_nominal_bits=4,
        n_clusters=256,
    )
    decoded_centroids = codec._decode_predictor(state)
    predictor = codec._gather_centroids(
        decoded_centroids, state["predictor"]["cluster_ids"]
    )
    expected_max = (-predictor).float().reshape(-1, 64).abs().amax(dim=-1)
    stored_max = codec._from_positive_e4m3_raw(
        state["suffix"]["scale_raw"]
    ).reshape(-1)
    expected_fp8 = codec._from_positive_e4m3_raw(
        codec._to_positive_e4m3_raw(expected_max)
    )
    torch.testing.assert_close(stored_max, expected_fp8, atol=0, rtol=0)


def test_selector_uses_only_fp8_sign_and_decode_clears_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn((1, 1, 128, 128), dtype=torch.bfloat16)
    state = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        max_nominal_bits=2,
        n_clusters=64,
        selection_fraction=0.25,
    )
    raw = state["suffix"]["scale_raw"]
    selector = raw.bitwise_and(0x80).ne(0)
    magnitude = codec._from_positive_e4m3_raw(raw)
    cleared = codec._from_positive_e4m3_raw(raw.bitwise_and(0x7F))
    torch.testing.assert_close(magnitude, cleared, atol=0, rtol=0)
    assert int(selector.sum()) == int(
        state["suffix"]["selected_group_count"]
    )


def test_corrupted_segment_and_selector_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16)
    state = codec.encode_r2aq_residual(
        source,
        tensor_name="V",
        max_nominal_bits=2,
        n_clusters=64,
    )
    bad_segment = copy.copy(state)
    bad_segment["suffix"] = dict(state["suffix"])
    bad_segment["suffix"]["segments"] = tuple(
        state["suffix"]["segments"][:-1]
    )
    with pytest.raises(ValueError, match="segment layout"):
        codec.decode_r2aq_residual(bad_segment)

    bad_selector = copy.copy(state)
    bad_selector["suffix"] = dict(state["suffix"])
    bad_selector["suffix"]["scale_raw"] = state["suffix"]["scale_raw"].clone()
    bad_selector["suffix"]["scale_raw"].reshape(-1)[0] ^= 0x80
    with pytest.raises(ValueError, match="selector flags"):
        codec.decode_r2aq_residual(bad_selector)


def test_qv_anchor_levels_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_predictor(monkeypatch)
    tree = codec._default_tree()
    reconstruction = list(tree["reconstruction"])
    reconstruction[1] = reconstruction[1].clone()
    reconstruction[1][0] = -0.9
    params = {
        "trees": {
            "K": {
                "thresholds": tree["thresholds"],
                "reconstruction": reconstruction,
            }
        }
    }
    with pytest.raises(ValueError, match="depth 2"):
        codec.encode_r2aq_residual(
            torch.zeros((1, 1, 32, 128), dtype=torch.bfloat16),
            tensor_name="K",
            bits=2,
            n_clusters=32,
            tree_params=params,
        )


def test_independent_b1_b4_wire_matches_b4_physical_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch, nonzero=True)
    source = torch.randn(
        (1, 2, 256, 128),
        generator=torch.Generator().manual_seed(1234),
        dtype=torch.bfloat16,
    )
    full = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        bits=4,
        n_clusters=256,
        selection_fraction=0.25,
    )
    for bits in (1, 2, 3, 4):
        independent = codec.encode_r2aq_residual(
            source,
            tensor_name="K",
            bits=bits,
            n_clusters=256,
            selection_fraction=0.25,
        )
        prefix = codec.truncate_r2aq_state(full, bits=bits)
        for name in ("packed_centroids", "scale_raw", "cluster_ids"):
            torch.testing.assert_close(
                independent["predictor"][name],
                prefix["predictor"][name],
                atol=0,
                rtol=0,
            )
        for name in ("scale_raw", "selector_prefix_offsets", "padding"):
            torch.testing.assert_close(
                independent["suffix"][name],
                prefix["suffix"][name],
                atol=0,
                rtol=0,
            )
        assert (
            independent["suffix"]["segment_names"]
            == prefix["suffix"]["segment_names"]
        )
        assert (
            independent["suffix"]["segment_counts"]
            == prefix["suffix"]["segment_counts"]
        )
        for direct, retained in zip(
            independent["suffix"]["segments"],
            prefix["suffix"]["segments"],
        ):
            torch.testing.assert_close(direct, retained, atol=0, rtol=0)
        assert codec.payload_nbytes(independent) == codec.payload_nbytes(prefix)
        torch.testing.assert_close(
            codec.decode_r2aq_residual(independent),
            codec.decode_r2aq_residual(prefix),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "corruption",
    ("initial", "tile", "terminal", "nonmonotonic"),
)
def test_selector_prefix_offsets_corruption_fails_before_fast_launch(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    _install_predictor(monkeypatch)
    state = codec.encode_r2aq_residual(
        torch.randn((1, 2, 512, 128), dtype=torch.bfloat16),
        tensor_name="V",
        bits=3,
        n_clusters=256,
        selection_fraction=0.25,
    )
    bad = copy.deepcopy(state)
    offsets = bad["suffix"]["selector_prefix_offsets"]
    groups_per_row = (
        bad["suffix"]["scale_raw"].numel() // 2
    )
    entries = (
        groups_per_row + codec._SELECTOR_TILE_GROUPS - 1
    ) // codec._SELECTOR_TILE_GROUPS + 1
    if corruption == "initial":
        offsets[0] = 1
    elif corruption == "tile":
        offsets[1] += 1
    elif corruption == "terminal":
        offsets[entries - 1] += 1
    else:
        offsets[1] = offsets[2] + 1
    with pytest.raises(ValueError, match="selector_prefix_offsets"):
        codec.decode_r2aq_residual(
            bad,
            torch.bfloat16,
            use_fast_path=True,
        )


def test_wire_alias_dtype_and_fp8_nan_corruption_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    state = codec.encode_r2aq_residual(
        torch.randn((1, 1, 256, 128), dtype=torch.bfloat16),
        tensor_name="K",
        bits=2,
        n_clusters=256,
        selection_fraction=0.25,
    )

    alias = copy.deepcopy(state)
    alias["segments"] = alias["suffix"]["segments"]
    with pytest.raises(ValueError, match="top-level wire aliases"):
        codec.payload_nbytes(alias)

    wrong_dtype = copy.deepcopy(state)
    wrong_dtype["predictor"]["cluster_ids"] = wrong_dtype["predictor"][
        "cluster_ids"
    ].to(torch.int64)
    with pytest.raises(ValueError, match="cluster_ids"):
        codec.decode_r2aq_residual(wrong_dtype)

    nan_scale = copy.deepcopy(state)
    nan_scale["suffix"]["scale_raw"].reshape(-1)[0] = 0x7F
    with pytest.raises(ValueError, match="NaN codes"):
        codec.decode_r2aq_residual(nan_scale)


def test_global_shared_tables_charge_k_v_reconstruction_and_utility_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    utility = torch.ones((30, 2, 5, 2), dtype=torch.float16)
    source = torch.randn((1, 1, 128, 128), dtype=torch.bfloat16)
    states = [
        codec.encode_r2aq_residual(
            source,
            tensor_name=tensor_name,
            bits=2,
            n_clusters=64,
            ablation="utility_waterfill",
            utility_weights=utility,
            layer_idx=0,
            age_bin=0,
            selection_fraction=0.25,
        )
        for tensor_name in ("K", "V")
    ]
    assert codec.payload_breakdown(states[0])["shared_table_bytes"] == 1386
    assert codec.payload_breakdown(states[1])["shared_table_bytes"] == 1386
    seen: set[str] = set()
    assert codec.shared_payload_nbytes(states[0], seen) == 1386
    assert codec.shared_payload_nbytes(states[1], seen) == 186
    assert len(seen) == 5
    assert sum(
        codec.shared_payload_nbytes(state, set())
        for state in states
    ) == 2772


def test_shared_owner_reserve_is_exact_for_every_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn(
        (1, 1, 7040, 128),
        generator=torch.Generator().manual_seed(112),
        dtype=torch.bfloat16,
    )
    utility = torch.ones((30, 2, 5, 2), dtype=torch.float16)
    states = {
        tensor_name: codec.encode_r2aq_residual(
            source,
            tensor_name=tensor_name,
            bits=4,
            n_clusters=256,
            ablation="full_tree",
            utility_weights=utility,
            layer_idx=0,
            age_bin=0,
            shared_budget_reserve_bytes=(
                codec.owner_shared_budget_nbytes(
                    tensor_name, utility_weights=utility
                )
            ),
        )
        for tensor_name in ("K", "V")
    }
    assert states["K"]["shared_budget_reserve_bytes"] == 1386
    assert states["V"]["shared_budget_reserve_bytes"] == 186
    seen: set[str] = set()
    shared = {
        name: codec.shared_payload_nbytes(states[name], seen)
        for name in ("K", "V")
    }
    assert shared == {"K": 1386, "V": 186}
    values = source.numel()
    qv_side_bytes = 256 * 128 * 2 + 7040 + values // 64
    for bits in (1, 2, 3, 4):
        for name in ("K", "V"):
            prefix = codec.truncate_r2aq_state(states[name], bits)
            expected = qv_side_bytes + bits * values // 8
            assert codec.payload_nbytes(prefix) + shared[name] == expected
            assert prefix["local_payload_target_bytes"] == expected - shared[name]


@pytest.mark.parametrize("ablation", ("coherent_tail", "coherent_tail_g128"))
def test_tail_shared_reserve_preserves_all_smaller_prefixes(
    monkeypatch: pytest.MonkeyPatch,
    ablation: str,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn(
        (1, 1, 7040, 128),
        generator=torch.Generator().manual_seed(913),
        dtype=torch.bfloat16,
    )
    utility = torch.ones((30, 2, 5, 2), dtype=torch.float16)
    reserve = codec.owner_shared_budget_nbytes("V", utility_weights=utility)
    common = {
        "tensor_name": "V",
        "bits": 4,
        "n_clusters": 256,
        "utility_weights": utility,
        "layer_idx": 0,
        "age_bin": 0,
        "shared_budget_reserve_bytes": reserve,
    }
    control = codec.encode_r2aq_residual(
        source, ablation="full_tree", **common
    )
    state = codec.encode_r2aq_residual(
        source,
        ablation=ablation,
        coherent_tail_group_fraction=1.0,
        **common,
    )
    for bits in (1, 2, 3, 4):
        prefix = codec.truncate_r2aq_state(state, bits)
        expected = codec.truncate_r2aq_state(control, bits)
        assert codec.payload_nbytes(prefix) == codec.payload_nbytes(expected)
        if bits < 4:
            torch.testing.assert_close(
                codec.decode_r2aq_residual(prefix),
                codec.decode_r2aq_residual(expected),
                atol=0,
                rtol=0,
            )


def test_padding_and_shared_budget_metadata_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn((1, 1, 512, 128), dtype=torch.bfloat16)
    state = codec.encode_r2aq_residual(
        source,
        tensor_name="K",
        bits=4,
        n_clusters=256,
    )
    assert state["suffix"]["padding"].numel() > 0

    missing_padding = copy.deepcopy(state)
    missing_padding["suffix"]["padding"] = torch.empty(0, dtype=torch.uint8)
    with pytest.raises(ValueError, match="local payload"):
        codec.payload_nbytes(missing_padding)

    forged_target = copy.deepcopy(state)
    forged_target["local_payload_target_bytes"] -= 1
    with pytest.raises(ValueError, match="local_payload_target_bytes"):
        codec.payload_nbytes(forged_target)

    forged_threshold_ref = copy.deepcopy(state)
    refs = [dict(ref) for ref in forged_threshold_ref["shared_table_refs"]]
    threshold = next(ref for ref in refs if ref["kind"] == "thresholds")
    threshold["nbytes"] -= 2
    forged_threshold_ref["shared_table_refs"] = tuple(refs)
    forged_threshold_ref["shared_table_bytes"] -= 2
    with pytest.raises(ValueError, match="threshold shared-table"):
        codec.payload_nbytes(forged_threshold_ref)


def test_selection_modes_and_utility_context_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn((1, 1, 64, 128), dtype=torch.bfloat16)
    utility = torch.ones((1, 2, 5, 2), dtype=torch.float16)
    with pytest.raises(ValueError, match="forbids utility"):
        codec.encode_r2aq_residual(
            source,
            tensor_name="K",
            bits=2,
            n_clusters=64,
            selection_mode="scale",
            utility_weights=utility,
        )
    with pytest.raises(ValueError, match="requires utility"):
        codec.encode_r2aq_residual(
            source,
            tensor_name="K",
            bits=2,
            n_clusters=64,
            selection_mode="scale_utility",
        )
    with pytest.raises(ValueError, match="explicit layer"):
        codec.encode_r2aq_residual(
            source,
            tensor_name="K",
            bits=2,
            n_clusters=64,
            selection_mode="scale_utility",
            utility_weights=utility,
            age_bin=0,
        )
    with pytest.raises(ValueError, match="explicit age"):
        codec.encode_r2aq_residual(
            source,
            tensor_name="K",
            bits=2,
            n_clusters=64,
            selection_mode="scale_utility",
            utility_weights=utility,
            layer_idx=0,
        )


def test_predictor_suffix_energy_diagnostics_are_scalar_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_predictor(monkeypatch, nonzero=True)
    state = codec.encode_r2aq_residual(
        torch.randn((1, 1, 64, 128), dtype=torch.bfloat16),
        tensor_name="V",
        bits=2,
        n_clusters=64,
    )
    diagnostics = state["diagnostics"]
    bf16_energy = diagnostics["bf16_predictor_suffix_energy"]
    int4_energy = diagnostics["int4_predictor_suffix_energy"]
    assert diagnostics["predictor_suffix_energy_relative_increase"] == (
        pytest.approx(int4_energy / bf16_energy - 1.0)
    )
    assert not any(torch.is_tensor(value) for value in diagnostics.values())


def test_selector_offset_width_uses_per_row_capacity_not_global_selection():
    rows = 24
    groups_per_row = 14080
    matrix = torch.zeros((rows, groups_per_row), dtype=torch.bool)
    matrix[:, :3000] = True
    assert int(matrix.sum()) > 65535
    offsets = codec._selector_offsets(matrix.reshape(-1), rows)
    assert offsets.dtype == torch.int16
    expected_entries = (
        groups_per_row + codec._SELECTOR_TILE_GROUPS - 1
    ) // codec._SELECTOR_TILE_GROUPS + 1
    assert offsets.numel() == rows * expected_entries
    assert codec._unsigned_offsets_nbytes(
        rows * groups_per_row, rows
    ) == offsets.numel() * 2
