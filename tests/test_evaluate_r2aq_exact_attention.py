from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import torch.nn.functional as F

from scripts import evaluate_r2aq_exact_attention as gate


def test_method_grid_and_metric_provenance_are_explicit() -> None:
    args = gate.parse_args(
        ["create", "--ablations", "scale_waterfill,full_tree"]
    )
    points = gate.method_points(args)
    assert points[:3] == (
        ("l8c64_int4_rvq_base", 0),
        ("rvq_plus_quantvideo", 2),
        ("rvq_plus_quantvideo", 4),
    )
    assert points[3:] == tuple(
        (f"r2aq_{ablation}", bits)
        for ablation in ("scale_waterfill", "full_tree")
        for bits in (1, 2, 3, 4)
    )
    config = gate.experiment_config(args, [])
    provenance = config["metric_provenance"]
    assert provenance["exact_capture_replay"] is True
    assert provenance["offline_proxy_used"] is False
    assert provenance["attention_scope"] == gate.EXACT_ATTENTION_SCOPE
    assert "offline" in provenance["offline_proxy_scope_not_reported"]
    storage = config["history_storage_protocol"]
    assert storage["physical_quant_span_tokens"] == 7_040
    assert storage["generation_chunk_tokens"] == 3_520
    assert "remains BF16" in storage["pending_history"]


def test_worker_activates_explicit_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[torch.device] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", calls.append)
    monkeypatch.setattr(gate, "initialize", lambda _args: ([], "a" * 64, None))
    monkeypatch.setattr(gate, "write_run_info", lambda *_args: None)
    args = SimpleNamespace(
        device="cuda:6",
        shard_id=0,
        num_shards=1,
    )
    gate.cmd_worker(args)
    assert calls == [torch.device("cuda:6")]


def test_selection_fraction_defaults_to_exact_reclaimed_byte_budget() -> None:
    args = gate.parse_args(["create"])
    assert args.selection_fraction is None
    assert "coherent_tail" not in args.ablations
    coherent = gate.parse_args(
        ["create", "--ablations", "coherent_tail"]
    )
    assert coherent.ablations == ("coherent_tail",)
    assert coherent.coherent_tail_group_fraction == pytest.approx(0.55)
    with pytest.raises(SystemExit):
        gate.parse_args(["create", "--payload-tolerance-bytes", "1"])
    coherent_g128 = gate.parse_args(
        ["create", "--ablations", "coherent_tail_g128"]
    )
    assert coherent_g128.ablations == ("coherent_tail_g128",)
    assert gate.parse_args(
        ["create", "--selection-fraction", "auto"]
    ).selection_fraction is None
    assert gate.parse_args(
        ["create", "--selection-fraction", "0.25"]
    ).selection_fraction == pytest.approx(0.25)


def test_capture_root_override_is_resolved(tmp_path) -> None:
    args = gate.parse_args(
        [
            "create",
            "--capture-root",
            str(tmp_path),
            "--ablations",
            "scale_waterfill",
        ]
    )
    assert args.capture_root == tmp_path.resolve()
    with pytest.raises(SystemExit):
        gate.parse_args(["create", "--selection-fraction", "1.1"])


def test_grid_only_exact_gate_has_uniform_and_routed_b4_points() -> None:
    args = gate.parse_args(
        [
            "create",
            "--ablations",
            "",
            "--grid-routing",
            "/tmp/routing.json",
            "--grid-tree-params",
            "/tmp/tree.pt",
        ]
    )
    assert gate.method_points(args)[-2:] == (
        ("uniform_grid_gaussian", 4),
        ("propagation_routed_grid_gaussian", 4),
    )
    config = gate.experiment_config(args, [])
    assert config["grid_gaussian"]["clustering"] is False
    assert config["grid_gaussian"]["bits"] == 4


def test_frame_only_exact_gate_is_pre_position_and_cluster_free() -> None:
    args = gate.parse_args(
        [
            "create",
            "--ablations",
            "",
            "--grid-routing",
            "/tmp/routing.json",
            "--grid-tree-params",
            "/tmp/tree.pt",
            "--gaussian-predictor",
            "frame",
        ]
    )
    assert gate.method_points(args)[-2:] == (
        ("uniform_frame_mean_gaussian", 4),
        ("propagation_routed_frame_mean_gaussian", 4),
    )
    config = gate.experiment_config(args, [])
    gaussian = config["grid_gaussian"]
    assert gaussian["predictor_mode"] == "frame"
    assert gaussian["predictor"] == "FP8 frame mean"
    assert gaussian["clustering"] is False
    assert gaussian["cache_representation"] == "pre_rope_pre_prope"
    assert gaussian["positional_replay_role"] == "evaluation_only_after_decode"
    assert gaussian["transform_seed"] == args.codec_seed
    assert "global_fixed" in gaussian["transform_seed_scope"]


def test_age_bins_follow_oldest_to_newest_history_axis() -> None:
    assert gate.token_age_bins(10, 5, torch.device("cpu")).tolist() == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]


def test_runtime_history_partition_matches_factor2_wan_schedule() -> None:
    plan = gate.runtime_history_partition(31_680)
    assert plan["generation_chunk_tokens"] == 3_520
    assert plan["physical_quant_span_tokens"] == 7_040
    assert plan["quantized_spans"] == (
        (0, 7_040),
        (7_040, 14_080),
        (14_080, 21_120),
        (21_120, 28_160),
    )
    assert plan["pending_bf16_range"] == (28_160, 31_680)
    assert plan["pending_bf16_tokens"] == 3_520

    even = gate.runtime_history_partition(14_080)
    assert even["quantized_spans"] == ((0, 7_040), (7_040, 14_080))
    assert even["pending_bf16_range"] is None
    assert even["pending_bf16_tokens"] == 0

    with pytest.raises(ValueError, match="generation-chunk aligned"):
        gate.runtime_history_partition(31_679)


def test_tail_statistics_use_empirical_upper_tail() -> None:
    stats = gate.tail_statistics(torch.tensor([0.0, 1.0, 4.0, 9.0]))
    assert stats["attention_output_squared_error_mean"] == 3.5
    assert stats["attention_output_squared_error_P999"] >= stats[
        "attention_output_squared_error_P99"
    ]
    assert stats["attention_output_squared_error_CVaR99"] == 9.0


def test_exact_output_tail_replay_matches_direct_sdpa() -> None:
    generator = torch.Generator().manual_seed(17)
    query = torch.randn(1, 2, 3, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    value = torch.randn(1, 2, 5, 8, generator=generator)
    candidate_key = key + 0.01 * torch.randn(
        key.shape, generator=generator
    )
    candidate_value = value + 0.02 * torch.randn(
        value.shape, generator=generator
    )
    reference_output = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0
    )
    reference = {
        "batches": [
            {
                "start": 0,
                "end": 3,
                "output": reference_output,
            }
        ]
    }
    metrics: dict[str, list[dict[str, float]]] = {}
    candidates = {
        "K_only": (candidate_key, value),
        "V_only": (key, candidate_value),
        "K_and_V": (candidate_key, candidate_value),
    }
    for intervention, (candidate_k, candidate_v) in candidates.items():
        output = F.scaled_dot_product_attention(
            query, candidate_k, candidate_v, dropout_p=0.0
        )
        error = (output.float() - reference_output.float()).square()
        metrics[intervention] = [
            {
                "attention_output_mse": float(
                    error[:, head].mean()
                )
            }
            for head in range(2)
        ]

    gate.add_exact_output_tails(
        metrics,
        query=query,
        key_reference=key,
        key_candidate=candidate_key,
        value_reference=value,
        value_candidate=candidate_value,
        output_matrix=None,
        evidence=None,
        reference=reference,
    )
    for intervention in gate.INTERVENTIONS:
        for row in metrics[intervention]:
            assert set(gate.TAIL_METRICS) <= set(row)
            assert row["attention_output_squared_error_mean"] == pytest.approx(
                row["attention_output_mse"], rel=2e-5
            )
            assert row["attention_output_squared_error_CVaR99"] >= row[
                "attention_output_squared_error_P99"
            ]


def test_strict_payload_charges_shared_tables_once() -> None:
    class FakeCodec:
        @staticmethod
        def payload_nbytes(state: Mapping[str, Any]) -> int:
            return int(state["local"])

        @staticmethod
        def payload_breakdown(state: Mapping[str, Any]) -> dict[str, int]:
            return {
                "centroid_payload_bytes": int(state["local"]) - 10,
                "label_payload_bytes": 10,
                "shared_table_bytes": 999,
            }

        @staticmethod
        def shared_payload_nbytes(
            state: Mapping[str, Any], seen: set[str]
        ) -> int:
            table = str(state["table"])
            if table in seen:
                return 0
            seen.add(table)
            return int(state["shared"])

    totals, breakdowns = gate.strict_r2aq_pair_payload(
        FakeCodec,
        {
            "K": {"local": 100, "shared": 12, "table": "global"},
            "V": {"local": 120, "shared": 12, "table": "global"},
        },
    )
    assert totals == {"K": 112, "V": 120}
    assert breakdowns["K"]["shared_table_bytes"] == 12
    assert breakdowns["V"]["shared_table_bytes"] == 0


def test_span_payload_deduplicates_tables_and_charges_pending_bf16() -> None:
    class FakeCodec:
        @staticmethod
        def payload_nbytes(state: Mapping[str, Any]) -> int:
            return int(state["local"])

        @staticmethod
        def payload_breakdown(state: Mapping[str, Any]) -> dict[str, int]:
            return {
                "centroid_payload_bytes": int(state["local"]) - 10,
                "label_payload_bytes": 10,
                "shared_table_bytes": 999,
            }

        @staticmethod
        def shared_payload_nbytes(
            state: Mapping[str, Any], seen: set[str]
        ) -> int:
            table = str(state["table"])
            if table in seen:
                return 0
            seen.add(table)
            return int(state["shared"])

    totals, breakdowns = gate.strict_r2aq_span_payload(
        FakeCodec,
        {
            "K": [
                {"local": 100, "shared": 12, "table": "global"},
                {"local": 110, "shared": 12, "table": "global"},
            ],
            "V": [
                {"local": 120, "shared": 12, "table": "global"},
                {"local": 130, "shared": 12, "table": "global"},
            ],
        },
        retained_bf16_bytes={"K": 40, "V": 60},
    )
    assert totals == {"K": 262, "V": 310}
    assert breakdowns["K"]["shared_table_bytes"] == 12
    assert breakdowns["V"]["shared_table_bytes"] == 0
    assert breakdowns["K"]["retained_bf16_bytes"] == 40
    assert breakdowns["V"]["retained_bf16_bytes"] == 60


def _coverage_row(
    *,
    task: Mapping[str, Any],
    half: str,
    intervention: str,
    head: int,
    method: str = "l8c64_int4_rvq_base",
    bits: int = 0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": gate.SCHEMA,
        "config_hash": "a" * 64,
        "task_index": task["task_index"],
        "prompt_id": task["prompt_id"],
        "layer": task["layer"],
        "method": method,
        "bits": bits,
        "attention_half": half,
        "intervention": intervention,
        "head": head,
        "attention_scope": gate.EXACT_ATTENTION_SCOPE,
        "exact_capture_replay": True,
        "offline_proxy_used": False,
        "rvq_payload_K_bytes": 1,
        "residual_payload_K_bytes": 1,
        "total_payload_K_bytes": 2,
        "source_BF16_K_bytes": 4,
        "joint_compression_ratio_K_vs_BF16": 2.0,
        "rvq_payload_V_bytes": 1,
        "residual_payload_V_bytes": 1,
        "total_payload_V_bytes": 2,
        "source_BF16_V_bytes": 4,
        "joint_compression_ratio_V_vs_BF16": 2.0,
    }
    row.update(
        {
            metric: 1.0
            for metric in (*gate.protocol.AGGREGATE_METRICS, *gate.TAIL_METRICS)
        }
    )
    row.update(
        {
            "rvq_payload_K_bytes": 1,
            "residual_payload_K_bytes": 1,
            "total_payload_K_bytes": 2,
            "source_BF16_K_bytes": 4,
            "joint_compression_ratio_K_vs_BF16": 2.0,
            "rvq_payload_V_bytes": 1,
            "residual_payload_V_bytes": 1,
            "total_payload_V_bytes": 2,
            "source_BF16_V_bytes": 4,
            "joint_compression_ratio_V_vs_BF16": 2.0,
        }
    )
    return row


def test_coverage_rejects_offline_rows() -> None:
    task = {"task_index": 0, "prompt_id": 0, "layer": 10}
    rows = [
        _coverage_row(
            task=task,
            half=half,
            intervention=intervention,
            head=head,
        )
        for half in gate.ATTENTION_HALVES
        for intervention in gate.INTERVENTIONS
        for head in range(gate.replay.p0.HEADS)
    ]
    coverage = gate.validate_rows(
        rows,
        tasks=[task],
        config_hash="a" * 64,
        expected_method_points=(("l8c64_int4_rvq_base", 0),),
    )
    assert coverage["complete"] is True
    assert coverage["offline_proxy_used"] is False
    rows[0]["offline_proxy_used"] = True
    with pytest.raises(RuntimeError, match="offline_proxy_used"):
        gate.validate_rows(
            rows,
            tasks=[task],
            config_hash="a" * 64,
            expected_method_points=(("l8c64_int4_rvq_base", 0),),
        )


def test_strict_rate_match_validation_uses_physical_components() -> None:
    task = {"task_index": 0, "prompt_id": 0, "layer": 10}
    rows: list[dict[str, Any]] = []
    for method in ("rvq_plus_quantvideo", "r2aq_scale_waterfill"):
        for half in gate.ATTENTION_HALVES:
            for intervention in gate.INTERVENTIONS:
                for head in range(gate.replay.p0.HEADS):
                    row = _coverage_row(
                        task=task,
                        half=half,
                        intervention=intervention,
                        head=head,
                        method=method,
                        bits=2,
                    )
                    if method.startswith("r2aq_"):
                        row.update(
                            {
                                "strict_payload_accounted": True,
                                "strict_rate_match_vs_QV": True,
                                "residual_payload_delta_K_bytes_vs_QV": 0,
                                "residual_payload_delta_V_bytes_vs_QV": 0,
                            }
                        )
                        for tensor_name in ("K", "V"):
                            for component in gate.PAYLOAD_COMPONENTS:
                                row[
                                    f"r2aq_{component}_{tensor_name}"
                                ] = 0
                            row[
                                f"r2aq_centroid_payload_bytes_{tensor_name}"
                            ] = 1
                    rows.append(row)
    coverage = gate.validate_rows(
        rows,
        tasks=[task],
        config_hash="a" * 64,
        expected_method_points=(
            ("rvq_plus_quantvideo", 2),
            ("r2aq_scale_waterfill", 2),
        ),
        require_rate_match=True,
    )
    assert coverage["complete"] is True
    assert coverage["global_strict_budget_checks"] == [
        {
            "method": "r2aq_scale_waterfill",
            "bits": 2,
            "K_delta_bytes": 0,
            "V_delta_bytes": 0,
            "KV_delta_bytes": 0,
            "shared_K_bytes": 0,
            "shared_V_bytes": 0,
            "reserve_K_bytes": 0,
            "reserve_V_bytes": 0,
        }
    ]

    wrong_owner = [dict(row) for row in rows]
    for row in wrong_owner:
        if row["method"] == "r2aq_scale_waterfill":
            row["r2aq_shared_budget_reserve_K_bytes"] = 1
    with pytest.raises(RuntimeError, match="shared owner/reserve"):
        gate.validate_rows(
            wrong_owner,
            tasks=[task],
            config_hash="a" * 64,
            expected_method_points=(
                ("rvq_plus_quantvideo", 2),
                ("r2aq_scale_waterfill", 2),
            ),
            require_rate_match=True,
        )

    broken = next(
        row for row in rows if row["method"] == "r2aq_scale_waterfill"
    )
    broken["residual_payload_delta_K_bytes_vs_QV"] = 1
    with pytest.raises(RuntimeError, match="reported K payload delta"):
        gate.validate_rows(
            rows,
            tasks=[task],
            config_hash="a" * 64,
            expected_method_points=(
                ("rvq_plus_quantvideo", 2),
                ("r2aq_scale_waterfill", 2),
            ),
        )


def test_run_info_is_complete_and_create_only(tmp_path) -> None:
    args = SimpleNamespace(
        command="create",
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
        utility_weights=None,
        tree_params=None,
        seed=42,
        codec_seed=7,
        shard_id=0,
        num_shards=1,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    task = {
        "task_index": 0,
        "prompt_id": 3,
        "prompt": "a test prompt",
        "layer": 10,
    }
    path = gate.write_run_info(
        args, [task], "b" * 64, config_path
    )
    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    for key in (
        "git_commit",
        "config_path",
        "config_hash",
        "codebook_path",
        "codebook_hash",
        "hostname",
        "gpu_type",
        "seed",
        "prompt_list",
        "timestamp",
    ):
        assert key in payload
    assert gate.write_run_info(
        args, [task], "b" * 64, config_path
    ) == path
    with pytest.raises(RuntimeError, match="incompatible"):
        gate.write_run_info(args, [task], "c" * 64, config_path)
