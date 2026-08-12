from __future__ import annotations

import pytest
import torch

import quant_videogen.structured_truncatable as structured
from scripts import screen_structured_temporal_prefix_exact_attention as screen


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
        for cluster in range(n_clusters):
            mask = ids == cluster
            denominator = mask.sum(dim=1).clamp_min(1).to(flattened.dtype)
            centers[:, cluster] = (
                flattened * mask.unsqueeze(-1)
            ).sum(dim=1) / denominator.unsqueeze(-1)
        sizes = torch.zeros(
            batches, n_clusters, dtype=torch.int64, device=flattened.device
        )
        sizes.scatter_add_(1, ids, torch.ones_like(ids, dtype=torch.int64))
        return ids, centers, sizes, max_iters

    monkeypatch.setattr(
        structured, "_online_vector_predictor", fake_predictor
    )


def test_terminal_axis_is_exactly_five_7040_token_spans() -> None:
    spans = screen.span_slices(35_200)
    assert [(value.start, value.stop) for value in spans] == [
        (0, 7040),
        (7040, 14080),
        (14080, 21120),
        (21120, 28160),
        (28160, 35200),
    ]
    with pytest.raises(ValueError, match="5x7040=35200"):
        screen.span_slices(31_680)


def test_registered_schedules_are_strictly_byte_equal_uniform_b2() -> None:
    rows = screen.schedule_payload_audit()
    assert {row["method"] for row in rows} == set(screen.STRUCTURED_METHODS)
    assert all(row["suffix_plane_units"] == 10 for row in rows)
    assert all(row["byte_difference"] == 0 for row in rows)
    assert all(row["exact_equal_uniform_B2"] for row in rows)
    strict = next(row for row in rows if row["method"] == screen.STRICT_METHOD)
    assert strict["B0_scale_retained"]


def test_explicit_b0_is_predictor_only_but_physically_retains_scale(
    monkeypatch,
) -> None:
    _install_predictor(monkeypatch)
    generator = torch.Generator().manual_seed(117)
    source = torch.randn(
        (1, 2, 8, 128), generator=generator, dtype=torch.bfloat16
    )
    full = structured.encode_structured_residual(
        source,
        bits=4,
        tensor_name="K",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=4,
        hadamard=True,
        scale_refine_iters=1,
    )
    b0 = screen.truncate_temporal_structured_state(full, 0)
    assert b0["schema_version"] == screen.B0_SCHEMA
    assert b0["b0_scale_retained"]
    assert b0["scale"].dtype == torch.float16
    assert b0["planes"] == ()
    expected = screen.predictor_from_state(full).to(torch.bfloat16)
    decoded = screen.decode_temporal_structured_state(b0, torch.bfloat16)
    torch.testing.assert_close(decoded, expected, atol=0.0, rtol=0.0)

    payload = screen.temporal_structured_payload_breakdown(b0)
    assert payload["suffix_scale_bytes"] == b0["scale"].numel() * 2
    assert payload["suffix_plane_bytes"] == 0
    assert payload["total_bytes"] == (
        b0["centroids"].numel() * b0["centroids"].element_size()
        + b0["cluster_ids"].numel() * b0["cluster_ids"].element_size()
        + b0["scale"].numel() * b0["scale"].element_size()
    )


def test_positive_prefix_uses_real_structured_payload_accounting(
    monkeypatch,
) -> None:
    _install_predictor(monkeypatch)
    source = torch.randn((1, 2, 9, 128), dtype=torch.bfloat16)
    full = structured.encode_structured_residual(
        source,
        bits=4,
        tensor_name="V",
        params_path=_gaussian_params(),
        group_size=128,
        n_clusters=4,
        hadamard=True,
        scale_refine_iters=1,
    )
    b2 = screen.truncate_temporal_structured_state(full, 2)
    breakdown = screen.temporal_structured_payload_breakdown(b2)
    assert breakdown["total_bytes"] == structured.payload_nbytes(b2["state"])
    assert breakdown["suffix_plane_bytes"] == 2 * ((source.numel() + 7) // 8)


def test_tail_helpers_use_upper_empirical_tail_and_absolute_age() -> None:
    assert screen.empirical_cvar([1.0, 2.0, 3.0, 4.0, 5.0], 0.8) == 5.0
    assert screen.empirical_cvar([1.0, 2.0, 3.0, 4.0], 0.5) == 3.5
    rows = [
        {"historical_signed_age_center_shift_tokens": -3.0},
        {"historical_signed_age_center_shift_tokens": 2.0},
    ]
    values = screen.transformed_metric_values(
        rows, "historical_signed_age_center_shift_tokens"
    )
    assert values.tolist() == [3.0, 2.0]


def test_asymmetric_kv_schedules_keep_each_tensor_at_ten_planes(
    monkeypatch,
) -> None:
    candidate = "K_uniform_V_temporal_test"
    monkeypatch.setattr(
        screen,
        "STRUCTURED_METHODS",
        (screen.UNIFORM_METHOD, candidate),
    )
    monkeypatch.setattr(
        screen,
        "SCHEDULES",
        {
            screen.UNIFORM_METHOD: (2, 2, 2, 2, 2),
            candidate: (4, 1, 1, 1, 3),
        },
    )
    monkeypatch.setattr(
        screen,
        "TENSOR_SCHEDULES",
        {
            screen.UNIFORM_METHOD: {
                "K": (2, 2, 2, 2, 2),
                "V": (2, 2, 2, 2, 2),
            },
            candidate: {
                "K": (2, 2, 2, 2, 2),
                "V": (4, 1, 1, 1, 3),
            },
        },
    )
    assert screen.method_schedule(candidate, "K") == (2, 2, 2, 2, 2)
    assert screen.method_schedule(candidate, "V") == (4, 1, 1, 1, 3)
    rows = screen.schedule_payload_audit()
    candidate_rows = [row for row in rows if row["method"] == candidate]
    assert {row["tensor"] for row in candidate_rows} == {"K", "V"}
    assert all(row["suffix_plane_units"] == 10 for row in candidate_rows)
    assert all(row["byte_difference"] == 0 for row in candidate_rows)
