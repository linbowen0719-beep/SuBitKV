from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import analyze_r2aq_exact_formal as analysis


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _four_shard_source(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    config: dict[str, object] = {"execution": {"num_shards": 4}}
    _write_json(root / "config.json", config)
    config_hash = analysis.sha256_file(root / "config.json")
    aggregate: dict[str, object] = {
        "config_hash": config_hash,
        "task_indices": list(range(8)),
    }
    for shard_id in range(4):
        _write_json(
            root
            / "run_info"
            / f"worker_shard_{shard_id:03d}_of_004.json",
            {
                "shard_id": shard_id,
                "num_shards": 4,
                "config_hash": config_hash,
                "task_indices": [shard_id, shard_id + 4],
            },
        )
    return config, aggregate


def test_discovers_all_four_contiguous_worker_shards(tmp_path: Path) -> None:
    config, aggregate = _four_shard_source(tmp_path)
    paths = analysis.discover_worker_run_info(tmp_path, config, aggregate)
    assert [path.name for path in paths] == [
        f"worker_shard_{shard_id:03d}_of_004.json" for shard_id in range(4)
    ]


def test_rejects_missing_worker_shard(tmp_path: Path) -> None:
    config, aggregate = _four_shard_source(tmp_path)
    (tmp_path / "run_info" / "worker_shard_002_of_004.json").unlink()
    with pytest.raises(ValueError, match="not contiguous"):
        analysis.discover_worker_run_info(tmp_path, config, aggregate)


def test_rejects_worker_config_hash_mismatch(tmp_path: Path) -> None:
    config, aggregate = _four_shard_source(tmp_path)
    path = tmp_path / "run_info" / "worker_shard_003_of_004.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["config_hash"] = "wrong"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="config_hash differs"):
        analysis.discover_worker_run_info(tmp_path, config, aggregate)


def test_strict_payload_uses_each_methods_unique_shared_owner() -> None:
    rows = []
    methods = {
        "rvq_plus_quantvideo": {
            "shared": (0, 0),
            "reserve": (0, 0),
            "local": (4000, 4000),
        },
        "r2aq_full_tree": {
            "shared": (1386, 186),
            "reserve": (1386, 186),
            "local": (2614, 3814),
        },
        "r2aq_scale_waterfill": {
            "shared": (186, 186),
            "reserve": (186, 186),
            "local": (3814, 3814),
        },
        "r2aq_int4_predictor_qv": {
            "shared": (186, 186),
            "reserve": (0, 0),
            "local": (3000, 3000),
        },
    }
    for method, spec in methods.items():
        for task_index in range(2):
            owner = method != "rvq_plus_quantvideo" and task_index == 0
            shared_k, shared_v = spec["shared"] if owner else (0, 0)
            reserve_k, reserve_v = spec["reserve"] if owner else (0, 0)
            local_k_total, local_v_total = spec["local"]
            local_k = local_k_total // 2
            local_v = local_v_total // 2
            rows.append(
                {
                    "task_index": task_index,
                    "prompt_id": task_index,
                    "layer": 0,
                    "method": method,
                    "ablation": method,
                    "bits": 2,
                    "source_BF16_K_bytes": 8000,
                    "source_BF16_V_bytes": 8000,
                    "rvq_payload_K_bytes": 0,
                    "rvq_payload_V_bytes": 0,
                    "residual_payload_K_bytes": local_k + shared_k,
                    "residual_payload_V_bytes": local_v + shared_v,
                    "r2aq_shared_table_bytes_K": shared_k,
                    "r2aq_shared_table_bytes_V": shared_v,
                    "total_encode_K_seconds": 1,
                    "total_encode_V_seconds": 1,
                    "total_decode_K_seconds": 1,
                    "total_decode_V_seconds": 1,
                    "physical_quant_span_count": 1,
                    "r2aq_shared_budget_reserve_K_bytes": reserve_k,
                    "r2aq_shared_budget_reserve_V_bytes": reserve_v,
                    "shared_budget_owner": owner,
                }
            )
    result = analysis.payload_runtime_table(pd.DataFrame(rows)).set_index("method")
    assert result.loc["r2aq_full_tree", "audited_fixed_shared_KV_bytes"] == 1572
    assert result.loc["r2aq_scale_waterfill", "audited_fixed_shared_KV_bytes"] == 372
    assert (
        result.loc["r2aq_int4_predictor_qv", "audited_fixed_shared_KV_bytes"]
        == 372
    )
    assert bool(result.loc["r2aq_full_tree", "audited_total_payload_exact_vs_QV"])
    assert bool(
        result.loc["r2aq_scale_waterfill", "audited_total_payload_exact_vs_QV"]
    )
    assert not bool(
        result.loc[
            "r2aq_int4_predictor_qv", "audited_total_payload_exact_vs_QV"
        ]
    )


def _prefix_frame() -> pd.DataFrame:
    rows = []
    for bits in analysis.PREFIX_SAFETY_BITS:
        for method in (
            analysis.PREFIX_REFERENCE_METHOD,
            analysis.PREFIX_CANDIDATE_METHOD,
        ):
            row = {
                "task_index": 0,
                "prompt_id": 0,
                "layer": 0,
                "head": 0,
                "bits": bits,
                "intervention": "K_and_V",
                "attention_half": "camera_prope",
                "method": method,
                "residual_payload_K_bytes": 100,
                "residual_payload_V_bytes": 100,
                "total_payload_K_bytes": 200,
                "total_payload_V_bytes": 200,
                "r2aq_shared_budget_reserve_K_bytes": 10,
                "r2aq_shared_budget_reserve_V_bytes": 10,
            }
            row.update({metric: 0.25 for metric in analysis.METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def test_prefix_safety_is_raw_equality_without_qv_or_bootstrap() -> None:
    detail, gate = analysis.prefix_safety_gate(_prefix_frame())
    assert gate["status"] == "PASS"
    assert gate["passes_all"] is True
    assert gate["qv_baseline_used"] is False
    assert gate["bootstrap_used"] is False
    assert set(gate["per_bit"]) == {"B1", "B3"}
    assert all(bool(value) for value in detail["exact_equal"])
    assert max(float(value or 0) for value in detail["max_abs_diff"]) == 0


def test_prefix_safety_detects_one_raw_metric_difference() -> None:
    frame = _prefix_frame()
    mask = (
        (frame["method"] == analysis.PREFIX_CANDIDATE_METHOD)
        & (frame["bits"] == 3)
    )
    frame.loc[mask, "historical_V_mse"] = 0.5
    detail, gate = analysis.prefix_safety_gate(frame)
    assert gate["status"] == "FAIL"
    assert gate["per_bit"]["B1"]["passes_all"] is True
    assert gate["per_bit"]["B3"]["passes_all"] is False
    row = detail[
        (detail["bits"] == 3) & (detail["field"] == "historical_V_mse")
    ].iloc[0]
    assert row["differing_rows"] == 1
    assert row["max_abs_diff"] == pytest.approx(0.25)


def test_exact_history_rate_separates_pending_bf16_and_codebook_scope() -> None:
    source_k = 31_680 * 2
    full_payload = 50_000
    payload = pd.DataFrame(
        [
            {
                "task_count": 1,
                "physical_quant_span_count_total": 4,
                "source_BF16_K_MiB": source_k / 2**20,
                "source_BF16_V_MiB": source_k / 2**20,
                "audited_benchmark_total_payload_KV_MiB": full_payload / 2**20,
            }
        ]
    )
    config = {
        "history_storage_protocol": {
            "physical_quant_span_tokens": 7040,
            "generation_chunk_tokens": 3520,
        },
        "rvq": {"heads": 1, "dimension": 1},
        "payload_accounting": {
            "RVQ_shared_codebooks": (
                "excluded equally from all methods; hashes are recorded"
            )
        },
    }
    row = analysis.annotate_history_rate_scope(payload, config).iloc[0]
    assert row["encoded_physical_span_count_per_task"] == 4
    assert row["encoded_history_tokens_per_task"] == 28_160
    assert row["pending_BF16_tokens_per_task"] == 3_520
    assert row["full_history_compression_ratio_vs_BF16"] == pytest.approx(
        (2 * source_k) / full_payload
    )
    assert row["encoded_only_compression_ratio_vs_BF16"] == pytest.approx(
        (28_160 * 2 * 2) / (full_payload - 3_520 * 2 * 2)
    )
