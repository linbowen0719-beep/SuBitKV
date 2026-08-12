from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pytest

from scripts import build_r2aq_results_report as report


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _training(root: Path) -> Path:
    root.mkdir(parents=True)
    rows = [
        {
            "tensor": tensor,
            "depth": depth,
            "MSE": 0.1 / depth,
            "attention_weighted_MSE": 0.08 / depth,
            "squared_error_CVaR99": 0.2 / depth,
            "fixed_qv_anchor": depth in (2, 4),
        }
        for tensor in ("K", "V")
        for depth in range(1, 6)
    ]
    _json(
        root / "params_summary.json",
        {
            "schema_version": "r2aq_params_summary.v1",
            "prompt_ids": list(range(20)),
            "cv_macro": rows,
        },
    )
    (root / "params.pt").write_bytes(b"params")
    _json(root / "utility_weights.json", {"weights": [1]})
    _json(root / "config.json", {})
    _csv(root / "five_fold_prompt_cv.csv", [{"fold": 0}])
    artifacts = {
        "params": str(root / "params.pt"),
        "summary": str(root / "params_summary.json"),
        "cv": str(root / "five_fold_prompt_cv.csv"),
        "utility": str(root / "utility_weights.json"),
        "config": str(root / "config.json"),
    }
    _json(
        root / "manifest.json",
        {
            "schema_version": "r2aq_params_manifest.v1",
            "artifacts": artifacts,
            "sha256": {
                key: report.sha256_file(Path(path)) for key, path in artifacts.items()
            },
            "shared_parameter_bytes": 1448,
        },
    )
    _json(
        root / "run_info.json",
        {
            "formal_evaluation_run": False,
            "seed": 42,
            "prompt_list": list(range(20)),
        },
    )
    return root


def _static(source: Path, analysis: Path) -> tuple[Path, Path]:
    (source / "aggregate").mkdir(parents=True)
    (source / "run_info").mkdir()
    _json(
        source / "aggregate" / "summary.json",
        {
            "coverage": {
                "split": "train",
                "prompts": list(range(20)),
                "units": 40,
            }
        },
    )
    _csv(source / "aggregate" / "summary.csv", [{"method": "x"}])
    _csv(source / "aggregate" / "prompt_macro.csv", [{"prompt_id": 0}])
    _json(source / "run_info" / "aggregate_config.json", {})
    _json(
        source / "run_info" / "aggregate.json",
        {"tree_params_hash": "a", "utility_weights_hash": "b"},
    )
    analysis.mkdir()
    config = {
        "schema_version": "r2aq_static_formal_analysis_config.v1",
        "run_root": str(source.resolve()),
        "pairing_unit": "prompt",
        "shared_table_accounting": "global once",
        "source_baseline": "K+V BF16",
    }
    _json(analysis / "config.json", config)
    quality = []
    methods = (
        "rvq_qv_b2",
        "rvq_qv_b4",
        "r2aq_full_tree_b2",
        "r2aq_full_tree_b4",
        "r2aq_utility_waterfill_b4",
    )
    for method in methods:
        bits = 2 if "b2" in method else 4
        quality.append(
            {
                "method": method,
                "bits": bits,
                "prompts": 20,
                "units": 40,
                "K_MSE": 0.1,
                "V_MSE": 0.2,
                "QKV_output_MSE": 0.01,
                "QKV_output_squared_error_CVaR99": 0.02,
            }
        )
    _csv(analysis / "quality_attention.csv", quality)
    paired = []
    for method in ("r2aq_full_tree_b2", "r2aq_utility_waterfill_b4"):
        paired.append(
            {
                "method": method,
                "metric": "QKV_output_MSE",
                "paired_relative_improvement_mean": 0.2,
                "paired_relative_improvement_CI_low": 0.1,
                "paired_relative_improvement_CI_high": 0.3,
                "prompt_win_rate": 0.9,
            }
        )
    _csv(analysis / "paired_relative_improvement.csv", paired)
    payload = []
    for method in methods:
        payload.append(
            {
                "method": method,
                "source_BF16_bytes": 1000,
                "residual_local_bytes": 200,
                "shared_table_bytes_global_once": 10,
                "rvq_bytes": 100,
                "joint_payload_bytes": 310,
                "joint_payload_MiB": 310 / 2**20,
                "joint_compression_ratio_vs_sampled_KV_BF16": 1000 / 310,
            }
        )
    _csv(analysis / "payload.csv", payload)
    _csv(analysis / "runtime.csv", [{"method": "rvq_qv_b2"}])
    _csv(
        analysis / "predictor_diagnostics.csv",
        [
            {
                "method": "r2aq_full_tree_b2",
                "K_codec_bf16_predictor_suffix_energy": 1.0,
                "K_codec_int4_predictor_suffix_energy": 1.01,
                "K_codec_predictor_suffix_energy_relative_increase": 0.01,
                "V_codec_bf16_predictor_suffix_energy": 2.0,
                "V_codec_int4_predictor_suffix_energy": 2.01,
                "V_codec_predictor_suffix_energy_relative_increase": 0.005,
            }
        ],
    )
    _json(
        analysis / "run_info.json",
        {"config_hash": report.sha256_file(analysis / "config.json")},
    )
    artifact_names = (
        "config.json",
        "run_info.json",
        "quality_attention.csv",
        "paired_relative_improvement.csv",
        "payload.csv",
        "runtime.csv",
        "predictor_diagnostics.csv",
    )
    _json(
        analysis / "evidence_manifest.json",
        {
            "schema_version": "r2aq_static_formal_evidence_manifest.v1",
            "artifacts": {
                name: {
                    "path": str((analysis / name).resolve()),
                    "sha256": report.sha256_file(analysis / name),
                }
                for name in artifact_names
            },
        },
    )
    return source, analysis


def _exact(source: Path, analysis: Path) -> tuple[Path, Path]:
    (source / "run_info").mkdir(parents=True)
    _json(
        source / "config.json",
        {"history_storage_protocol": {"physical_quant_span_tokens": 7040}},
    )
    _json(
        source / "run_info" / "aggregate.json",
        {
            "exact_capture_replay": True,
            "offline_proxy_used": False,
            "attention_scope": report.EXACT_SCOPE,
            "config_hash": report.sha256_file(source / "config.json"),
        },
    )
    analysis.mkdir()
    _json(
        analysis / "analysis_config.json",
        {"source": str(source.resolve())},
    )
    validation = {
        "prompts": list(range(10)),
        "layers": [0, 10, 20, 29],
        "heads": list(range(24)),
        "bits": [2, 4],
    }
    _json(
        analysis / "run_info.json",
        {
            "schema_version": "r2aq_exact_formal_analysis.v1",
            "config_hash": report.sha256_file(analysis / "analysis_config.json"),
            "validation": validation,
        },
    )
    variants = {}
    for variant, passed, improvement, wins in (
        ("r2aq_full_tree.B2", True, 20.0, 1.0),
        ("r2aq_full_tree.B4", False, 4.0, 0.6),
        ("r2aq_utility_waterfill.B4", False, -1.0, 0.4),
    ):
        variants[variant] = {
            "method": variant.split(".")[0],
            "bits": int(variant[-1]),
            "benchmark_payload_delta_KV_bytes_vs_QV": 1448,
            "global_shared_table_KV_bytes": 1448,
            "V_only_camera_CVaR99_gate": {
                "passes_all": passed,
                "qv_mean": 1.0,
                "candidate_mean": 0.8,
                "relative_improvement_of_means_pct": improvement,
                "paired_bootstrap_ci95": [0.1 if passed else -0.1, 0.3],
                "prompt_win_rate": wins,
                "mean_not_worse_than_qv": True,
                "bootstrap_ci_lower_gt_zero": passed,
                "at_least_80pct_prompt_wins": passed,
            },
            "selected_exact_metrics": {
                "V_only.camera_prope.attention_output_squared_error_CVaR99": {
                    "qv_mean": 1.0,
                    "candidate_mean": 0.8,
                    "relative_improvement_of_means_pct": improvement,
                    "paired_bootstrap_ci95": [0.1, 0.3],
                    "prompt_win_rate": wins,
                }
            },
        }
    _json(
        analysis / "variant_gates.json",
        {
            "schema_version": "r2aq_exact_formal_variant_gates.v1",
            "variants": variants,
        },
    )
    _csv(analysis / "paired_bootstrap_10000.csv", [{"method": "x"}])
    payload_rows = []
    for method, bits, shared_k, shared_v in (
        ("rvq_plus_quantvideo", 2, 0, 0),
        ("r2aq_full_tree", 2, 1324, 124),
        ("rvq_plus_quantvideo", 4, 0, 0),
        ("r2aq_full_tree", 4, 1324, 124),
        ("r2aq_utility_waterfill", 4, 1324, 124),
    ):
        k_mib = 1 + shared_k / 2**20
        v_mib = 1 + shared_v / 2**20
        payload_rows.append(
            {
                "method": method,
                "bits": bits,
                "source_BF16_K_MiB": 10,
                "source_BF16_V_MiB": 10,
                "global_shared_table_K_bytes": shared_k,
                "global_shared_table_V_bytes": shared_v,
                "global_shared_table_KV_bytes": shared_k + shared_v,
                "benchmark_total_payload_K_MiB": k_mib,
                "benchmark_total_payload_V_MiB": v_mib,
                "benchmark_total_payload_KV_MiB": k_mib + v_mib,
                "benchmark_compression_KV_vs_BF16": 20 / (k_mib + v_mib),
                "benchmark_payload_delta_KV_bytes_vs_QV": shared_k + shared_v,
            }
        )
    _csv(analysis / "payload_runtime.csv", payload_rows)
    generated = (
        "analysis_config.json",
        "run_info.json",
        "variant_gates.json",
        "paired_bootstrap_10000.csv",
        "payload_runtime.csv",
    )
    _json(
        analysis / "evidence_manifest.json",
        {
            "schema_version": "r2aq_exact_formal_evidence_manifest.v1",
            "generated_files": [
                {
                    "path": str((analysis / name).resolve()),
                    "sha256": report.sha256_file(analysis / name),
                }
                for name in generated
            ],
        },
    )
    return source, analysis


def _prefixsafe_staged(root: Path) -> Path:
    root.mkdir(parents=True)
    _json(
        root / "config.json",
        {"schema_version": "r2aq_coherent_tail_staged.analysis.config.v1"},
    )
    focus = {}
    paired = []
    for half in ("standard_rope", "camera_prope"):
        for intervention in ("K_only", "V_only", "K_and_V"):
            for metric in (
                "attention_output_squared_error_mean",
                "attention_output_squared_error_CVaR99",
            ):
                key = f"{half}.{intervention}.{metric}"
                row = {
                    "attention_half": half,
                    "candidate_mean": 0.5,
                    "candidate_over_qv": 0.5,
                    "improvement_fraction": 0.5,
                    "intervention": intervention,
                    "metric": metric,
                    "n_prompts": 3,
                    "paired_bootstrap_improvement_ci95_high": 0.6,
                    "paired_bootstrap_improvement_ci95_low": 0.4,
                    "per_prompt_candidate_over_qv": "0.5,0.5,0.5",
                    "prompt_ids": "0,4,9",
                    "prompt_win_rate": 1.0,
                    "qv_mean": 1.0,
                }
                focus[key] = row
                paired.append(row)
    _json(
        root / "summary.json",
        {
            "schema_version": "r2aq_coherent_tail_staged.summary.v1",
            "descriptive_only_small_n": True,
            "n_prompts": 3,
            "layers": [0, 10, 20, 29],
            "heads_per_layer": 24,
            "method": "r2aq_coherent_tail_g128",
            "reference": "rvq_plus_quantvideo",
            "focus": focus,
        },
    )
    _csv(root / "paired_prompt_bootstrap.csv", paired)
    _csv(
        root / "paired_unit_metrics.csv",
        [
            {
                "prompt_id": prompt,
                "layer": layer,
                "intervention": intervention,
                "attention_half": half,
            }
            for prompt in (0, 4, 9)
            for layer in (0, 10, 20, 29)
            for intervention in ("K_only", "V_only", "K_and_V")
            for half in ("standard_rope", "camera_prope")
        ],
    )
    payload = []
    for prompt in (0, 4, 9):
        for layer in (0, 10, 20, 29):
            for bits in (2, 4):
                payload.append(
                    {
                        "prompt_id": prompt,
                        "layer": layer,
                        "bits": bits,
                        "local_delta_K_bytes_vs_QV": 0,
                        "local_delta_V_bytes_vs_QV": 0,
                        "shared_K_bytes": 1324,
                        "shared_V_bytes": 124,
                        "strict_rate_match_vs_QV": True,
                        "tail_G128_group_count": 24576 if bits == 4 else "",
                        "tail_value_bytes": 1572864 if bits == 4 else "",
                        "tail_scale_bytes": 24576 if bits == 4 else "",
                        "tail_id_bytes": 49152 if bits == 4 else "",
                        "selected_P5_groups_before_tail": 585408 if bits == 4 else "",
                        "terminal_P5_group_count": 379584 if bits == 4 else "",
                        "physical_span_count": 4 if bits == 4 else "",
                    }
                )
    _csv(root / "payload_audit.csv", payload)
    output_names = (
        "config.json",
        "summary.json",
        "paired_prompt_bootstrap.csv",
        "paired_unit_metrics.csv",
        "payload_audit.csv",
    )
    _json(
        root / "run_info.json",
        {
            "schema_version": "r2aq_coherent_tail_staged.analysis.run_info.v1",
            "config_hash": report.sha256_file(root / "config.json"),
            "output_hashes": {
                name: report.sha256_file(root / name) for name in output_names
            },
            "source_hashes": {},
        },
    )
    return root


def _prefix_safety(root: Path) -> Path:
    root.mkdir(parents=True)
    _json(root / "analysis_config.json", {"source": "/tmp/strict-exact-source"})
    validation = {
        "prompts": list(range(10)),
        "layers": [0, 10, 20, 29],
        "heads": list(range(24)),
        "bits": [2, 4],
    }
    _json(
        root / "run_info.json",
        {
            "schema_version": "r2aq_exact_formal_analysis.v1",
            "config_hash": report.sha256_file(root / "analysis_config.json"),
            "validation": validation,
        },
    )
    rows = []
    per_bit = {}
    for bits in (1, 3):
        rows.extend(
            [
                {
                    "bits": bits,
                    "comparison_scope": "raw_exact_metric",
                    "field": "historical_K_mse",
                    "rows_compared": 5760,
                    "differing_rows": 0,
                    "exact_equal": True,
                    "max_abs_diff": 0,
                },
                {
                    "bits": bits,
                    "comparison_scope": "task_physical_payload",
                    "field": "total_payload_K_bytes",
                    "rows_compared": 40,
                    "differing_rows": 0,
                    "exact_equal": True,
                    "max_abs_diff": 0,
                },
            ]
        )
        per_bit[f"B{bits}"] = {
            "bits": bits,
            "raw_metric_rows_compared": 5760,
            "task_payload_rows_compared": 40,
            "metric_field_count": 9,
            "payload_field_count": 40,
            "metrics_exact_equal": True,
            "payload_exact_equal": True,
            "max_metric_abs_diff": 0,
            "max_numeric_payload_abs_diff": 0,
            "passes_all": True,
        }
    _csv(root / "prefix_safety_detail.csv", rows)
    _json(
        root / "prefix_safety_gate.json",
        {
            "schema_version": "r2aq_exact_prefix_safety_gate.v1",
            "status": "PASS",
            "passes_all": True,
            "candidate": "r2aq_coherent_tail_g128",
            "reference": "r2aq_full_tree",
            "bits": [1, 3],
            "comparison_unit": "raw exact row",
            "qv_baseline_used": False,
            "bootstrap_used": False,
            "per_bit": per_bit,
        },
    )
    generated = (
        "analysis_config.json",
        "run_info.json",
        "prefix_safety_detail.csv",
        "prefix_safety_gate.json",
    )
    _json(
        root / "evidence_manifest.json",
        {
            "schema_version": "r2aq_exact_formal_evidence_manifest.v1",
            "generated_files": [
                {
                    "path": str((root / name).resolve()),
                    "sha256": report.sha256_file(root / name),
                }
                for name in generated
            ],
        },
    )
    return root


def _coherent_video_3p(root: Path) -> Path:
    root.mkdir(parents=True)
    config = {
        "schema_version": "r2aq_coherent_tail_video_gate.config.v1",
        "scope": (
            "three-prompt staged development gate; not the formal ten-prompt "
            "benchmark"
        ),
        "variant": "l8c64_rvq_r2aq_coherent_tail_g128_f100_b4",
        "prompt_stages": [
            {"stage": 0, "prompt_id": 0},
            {"stage": 1, "prompt_id": 4},
            {"stage": 2, "prompt_id": 9},
        ],
        "generation": {"actual_solver_updates": 4},
        "similarity": {"protocol_id": report.SIMILARITY_PROTOCOL},
        "vbench": {"authorized": False},
    }
    _json(root / "config.json", config)
    config_hash = report.sha256_file(root / "config.json")
    _json(
        root / "run_info.json",
        {
            "schema_version": "r2aq_coherent_tail_video_gate.run_info.v1",
            "config_hash": config_hash,
            "formal_evaluation_run": False,
            "vbench_run": False,
        },
    )
    similarity = root / "similarity" / "final"
    similarity.mkdir(parents=True)
    similarity_rows = [
        {
            "prompt_id": prompt_id,
            "prompt": f"prompt {prompt_id}",
            "setting": config["variant"],
            "mse": 0.01,
            "ssim": 0.8,
            "lpips": 0.2,
            "psnr": 20,
        }
        for prompt_id in (0, 4, 9)
    ]
    _csv(similarity / "per_prompt.csv", similarity_rows)
    _json(
        similarity / "summary.json",
        {
            "schema_version": "frame_aligned_video_similarity.v2",
            "metric_protocol": {"protocol_id": report.SIMILARITY_PROTOCOL},
        },
    )
    for stage, prompt_ids in enumerate(([0], [0, 4], [0, 4, 9])):
        final = stage == 2
        passed = True
        gate = {
            "schema_version": "r2aq_coherent_tail_video_stage_gate.v1",
            "stage": stage,
            "prompt_ids": list(prompt_ids),
            "baseline": "l8c64_rvq_residual_qv_int4",
            "metric_protocol": report.SIMILARITY_PROTOCOL,
            "composite_improvement_pct": 12.0,
            "prompt_bootstrap_ci_low_pct": 1.0,
            "prompt_bootstrap_ci_high_pct": 15.0,
            "per_prompt": [
                {
                    "prompt_id": prompt_id,
                    "composite_improvement_pct": 12.0,
                    "won": True,
                }
                for prompt_id in prompt_ids
            ],
            "all_observed_prompts_win": True,
            "payload_gate_passed": True,
            "passed": passed,
            "next_stage_authorized": passed and not final,
            "final_three_prompt_gate_passed": passed and final,
            "formal_ten_prompt_claim_authorized": False,
            "vbench_authorized": False,
            "similarity_path": str(similarity.resolve()),
            "similarity_hashes": {
                "csv": report.sha256_file(similarity / "per_prompt.csv"),
                "json": report.sha256_file(similarity / "summary.json"),
            },
        }
        _json(root / "gates" / f"stage_{stage}.json", gate)
    runtime_rows = [
        {
            "prompt_id": prompt_id,
            "compression_ratio_including_shared_codebook": 3.07,
            "e2e_time_s": 10,
            "denoise_time_s": 8,
            "quant_time_s": 1,
            "peak_gpu_mem_mb": 100,
        }
        for prompt_id in (0, 4, 9)
    ]
    runtime_path = root / "runtime" / "stage_2_through_prompt_9.csv"
    _csv(runtime_path, runtime_rows)
    _json(
        root / "runtime" / "stage_2_through_prompt_9.run_info.json",
        {
            "schema_version": (
                "r2aq_coherent_tail_video_runtime_aggregate.run_info.v1"
            ),
            "stage": 2,
            "config_hash": config_hash,
            "runtime_path": str(runtime_path.resolve()),
            "runtime_hash": report.sha256_file(runtime_path),
            "payload_audits": [
                {"prompt_id": prompt_id, "passed": True}
                for prompt_id in (0, 4, 9)
            ],
        },
    )
    return root


def _hierarchical(root: Path) -> Path:
    root.mkdir(parents=True)
    config = {
        "schema_version": "r2aq_hierarchical_prefix_benchmark.v1",
        "prototype_only": True,
        "canonical_wire_modified": False,
        "fine_prefix_accounting": {
            "G32_bytes_per_tensor": 10560,
            "G64_bytes_per_tensor": 5280,
        },
    }
    _json(root / "config.json", config)
    _json(
        root / "run_info.json",
        {
            "config_hash": report.sha256_file(root / "config.json"),
            "gpu_type": "test",
        },
    )
    _json(
        root / "summary.json",
        {"acceptance": {"all_rows_pass": False}},
    )
    rows = []
    for bits in (2, 4):
        rows.append(
            {
                "bits": bits,
                "method": "qv",
                "fine_prefix_bytes_per_tensor": 0,
                "full_median_ms_mean_KV": 1.0,
                "full_delta_percent_vs_QV": 0,
                "full_restore_le_10pct": True,
            }
        )
        for method, extra in (("hierarchical_g32", 10560), ("hierarchical_g64", 5280)):
            rows.append(
                {
                    "bits": bits,
                    "method": method,
                    "fine_prefix_bytes_per_tensor": extra,
                    "full_median_ms_mean_KV": 1.6,
                    "full_delta_percent_vs_QV": 60,
                    "full_restore_le_10pct": False,
                }
            )
    _csv(root / "aggregate_kv.csv", rows)
    artifact_names = (
        "config.json",
        "run_info.json",
        "summary.json",
        "aggregate_kv.csv",
    )
    _json(
        root / "artifact_manifest.json",
        {
            "schema_version": "r2aq_hierarchical_prefix_artifacts.v1",
            "artifacts": {
                name: report.sha256_file(root / name) for name in artifact_names
            },
        },
    )
    return root


def test_training_hashes_are_revalidated(tmp_path: Path) -> None:
    root = _training(tmp_path / "training")
    section = report.load_training_section(root)
    assert section["status"] == report.COMPLETE
    assert section["metadata"]["shared_parameter_bytes"] == 1448
    assert section["metadata"]["deployment_fixed_shared_KV_bytes"] == 1572
    (root / "params.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        report.load_training_section(root)


def test_static_formal_keeps_proxy_scope_and_physical_payload(tmp_path: Path) -> None:
    source, analysis = _static(tmp_path / "source", tmp_path / "analysis")
    section = report.load_static_section(source, analysis)
    assert section["status"] == report.COMPLETE
    assert section["scope"] == report.STATIC_SCOPE
    assert section["metadata"]["proxy_gates"]["B2_full_tree"]["status"] == report.PASS
    assert "不能替代exact attention" in section["reason"]
    predictor = section["tables"]["predictor"][0]
    assert predictor[
        "K_codec_predictor_suffix_energy_global_ratio_of_means_increase"
    ] == pytest.approx(0.01)
    assert predictor[
        "V_codec_predictor_suffix_energy_prompt_ratio_mean_increase"
    ] == pytest.approx(0.005)

    rows = read_rows(analysis / "payload.csv")
    rows[0]["joint_payload_bytes"] = "311"
    _csv(analysis / "payload.csv", rows)
    with pytest.raises(ValueError, match="double-count/missing"):
        report._validate_formal_static_payload(rows[0])


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_exact_gate_parser_preserves_b2_pass_and_b4_fail(tmp_path: Path) -> None:
    source, analysis = _exact(tmp_path / "source", tmp_path / "analysis")
    section = report.load_exact_section(source, analysis)
    gates = {row["variant"]: row["status"] for row in section["tables"]["gates"]}
    assert gates["r2aq_full_tree.B2"] == report.PASS
    assert gates["r2aq_full_tree.B4"] == report.FAIL
    assert gates["r2aq_utility_waterfill.B4"] == report.FAIL
    b2 = next(
        row
        for row in section["tables"]["gates"]
        if row["variant"] == "r2aq_full_tree.B2"
    )
    assert b2["audited_fixed_shared_KV_bytes"] == 1572
    assert b2["audited_payload_delta_KV_bytes_vs_QV"] == 1572
    assert b2["rate_class"] == "near-rate_not_strict"


def test_quality_gate_requires_ci_and_tail_non_regression() -> None:
    status, improvement, _ = report.quality_gate(
        evidence_complete=True,
        candidate=0.8,
        baseline=1.0,
        improvement_ci_low=0.01,
        candidate_tail=0.9,
        baseline_tail=1.0,
    )
    assert status == report.PASS
    assert improvement == pytest.approx(0.2)
    status, _, _ = report.quality_gate(
        evidence_complete=True,
        candidate=0.8,
        baseline=1.0,
        improvement_ci_low=-0.01,
    )
    assert status == report.FAIL


def test_prefixsafe_staged_is_descriptive_not_formal(tmp_path: Path) -> None:
    section = report.load_prefixsafe_staged_section(
        _prefixsafe_staged(tmp_path / "staged")
    )
    assert section["status"] == report.COMPLETE
    assert section["metadata"]["descriptive_gate"] == report.DESCRIPTIVE_PASS
    assert section["metadata"]["formal_eligible"] is False
    assert section["metadata"]["audited_rate_class"] == "near-rate_not_strict"
    assert section["metadata"]["tail_accounting"]["tail_bytes_per_task"] == 1646592
    assert all(
        row["audited_total_payload_exact_vs_QV"] is False
        for row in section["tables"]["payload"]
    )
    formal = report.load_strict_formal_section(tmp_path / "staged")
    assert formal["status"] == report.PENDING


def test_b1_b3_prefix_safety_uses_raw_identity_without_qv_ci(
    tmp_path: Path,
) -> None:
    section = report.load_prefix_safety_section(
        _prefix_safety(tmp_path / "prefix-safety")
    )
    assert section["status"] == report.COMPLETE
    assert section["metadata"]["gate"] == report.PASS
    assert section["metadata"]["no_qv_baseline"] is True
    assert section["metadata"]["no_bootstrap_or_ci"] is True
    assert section["metadata"]["per_bit"]["B1"]["raw_metric_rows_compared"] == 5760


def test_coherent_video_three_prompt_stays_descriptive(
    tmp_path: Path,
) -> None:
    section = report.load_coherent_video_3p_section(
        _coherent_video_3p(tmp_path / "video-3p")
    )
    assert section["status"] == report.COMPLETE
    assert section["metadata"]["development_gate"] == report.DESCRIPTIVE_PASS
    assert section["metadata"]["formal_10prompt_gate_eligible"] is False
    assert section["metadata"]["vbench_authorized"] is False
    assert {int(row["prompt_id"]) for row in section["rows"]} == {0, 4, 9}


def test_hierarchical_prefix_is_a_failed_non_equal_rate_prototype(
    tmp_path: Path,
) -> None:
    section = report.load_hierarchical_prefix_section(
        _hierarchical(tmp_path / "hierarchical")
    )
    assert section["status"] == report.COMPLETE
    assert section["metadata"]["gate"] == report.FAIL
    assert section["metadata"]["canonical_wire_modified"] is False


def test_few_step_protocol_and_byte_identity(tmp_path: Path) -> None:
    protocol, _ = report._few_step_source_evidence()
    assert protocol["effective_denoising_updates_per_generation_chunk"] == 4
    assert protocol["requested_cli_value_controls_effective_update_count"] is False
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"same-video")
    second.write_bytes(b"same-video")
    audit = report._byte_identical_video_audit(first, second)
    assert audit["byte_identical"] is True
    second.write_bytes(b"different-video")
    with pytest.raises(ValueError, match="video bytes differ"):
        report._byte_identical_video_audit(first, second)


def test_build_is_create_only_for_bundle_but_report_is_updateable(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    report_path = tmp_path / "docs" / "report.md"
    report_path.parent.mkdir()
    report_path.write_text("old report", encoding="utf-8")
    args = SimpleNamespace(
        data_root=tmp_path,
        output_dir=tmp_path / "immutable",
        report=report_path,
        training_dir=missing / "training",
        static_source_dir=missing / "static-source",
        static_analysis_dir=missing / "static-analysis",
        exact_source_dir=missing / "exact-source",
        exact_analysis_dir=missing / "exact-analysis",
        decode_dir=missing / "decode",
        video_prompt0_dir=missing / "prompt0",
        qv_b2_similarity_json=missing / "qv2.json",
        qv_b4_similarity_json=missing / "qv4.json",
        coherent_tail_dir=missing / "tail",
        prefixsafe_staged_dir=missing / "prefixsafe",
        hierarchical_prefix_dir=missing / "hierarchical",
        strict_rate_dir=missing / "strict-rate",
        strict_formal_dir=missing / "strict-formal",
        strict_video_dir=missing / "strict-video",
        three_prompt_similarity_dir=missing / "three",
        similarity_dir=missing / "formal",
        vbench_dir=missing / "vbench",
    )
    paths = report.build(args)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["no_experiments_run"] is True
    assert "VBench未获授权" in report_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        report.build(args)
