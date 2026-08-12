from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/summarize_wan5b_h7_tiered_video_evidence.py"
AGGREGATOR = REPO_ROOT / "scripts/aggregate_wan5b_attention_report.py"
PROMPTS = ["synthetic prompt zero", "synthetic prompt one", "synthetic prompt two"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic_run(
    root: Path,
    summarizer,
    *,
    missing_audit: bool = False,
    tiered_depth: float = 6.0,
    uniform_depth: float | str = 6.0,
    bad_codebook_hash: bool = False,
) -> Path:
    run_dir = root / "h7_run"
    run_dir.mkdir()
    manifest = json.loads(summarizer.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    baseline = manifest["baseline"]
    config = run_dir / "experiment_config.json"
    config.write_text(
        json.dumps(
            {
                "baseline": summarizer.BASELINE_ID,
                "codebooks": {
                    tensor: {
                        "path": baseline[f"{tensor.lower()}_codebook_path"],
                        "hash": baseline[f"{tensor.lower()}_codebook_hash"],
                        "files": 720,
                        "input_norm_mode": "none",
                        "shape": [8, 64, 128],
                        "tensor": tensor,
                    }
                    for tensor in ("K", "V")
                },
            }
        ),
        encoding="utf-8",
    )
    config_hash = summarizer.sha256_file(config)

    gate_path = run_dir / "offline_h7_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "schema_version": "wan5b_report_evidence.v1",
                "hypothesis_id": "H7",
                "experiment_id": "h7_matched_rate_offline_gate",
                "baseline_id": summarizer.BASELINE_ID,
                "status": "complete",
                "verdict": {"status": "supported"},
                "interpretation": {
                    "video_smoke_authorized": True,
                    "applied_rule_ids": ["H7-R1", "H7-R2", "H7-R3"],
                },
            }
        ),
        encoding="utf-8",
    )
    gate = {
        "provided": True,
        "execution_authorized": True,
        "path": str(gate_path),
        "sha256": summarizer.sha256_file(gate_path),
        "verdict": {"status": "supported"},
        "applied_rule_ids": ["H7-R1", "H7-R2", "H7-R3"],
    }
    codebook_paths = {
        "K": baseline["k_codebook_path"],
        "V": baseline["v_codebook_path"],
    }
    codebook_hashes = {
        "K": "bad" if bad_codebook_hash else baseline["k_codebook_hash"],
        "V": baseline["v_codebook_hash"],
    }
    schedule = [8, 4, 4, 6, 6, 6, 6, 6, 6, 8]
    accounting = summarizer.launcher.terminal_index_payload_accounting(10)
    policy = {
        "canonical_name": summarizer.launcher.H7_TIERED_POLICY,
        "boundary_depth": 8,
        "middle_depth": 6,
        "old_depth": 4,
        "oldest_middle_chunks": 2,
        "recent_chunks": 1,
        "generation_frames": 4,
        "tokens_per_frame": 880,
        "mutation": "encode once at L8; later clone only indices[:target_depth]",
    }
    uniform_quality = {
        "psnr": [20.0, 21.0, 22.0],
        "ssim": [0.70, 0.71, 0.72],
        "lpips": [0.30, 0.29, 0.28],
    }
    tiered_quality = {
        "psnr": [20.4, 21.3, 22.2],
        "ssim": [0.72, 0.73, 0.74],
        "lpips": [0.27, 0.26, 0.25],
    }
    for setting in summarizer.SETTING_NAMES:
        setting_dir = run_dir / setting
        setting_dir.mkdir()
        tiered = setting == summarizer.TIERED_NAME
        command = (
            "python scripts/eval/run_small_eval.py --num_frames 173 --num_chunk 11 "
            "--kv_quant_factor 1 --rvq_codebook_storage_dtype int4 "
            "--setting krvq-vrvq-i4 --rvq_num_levels "
            + ("8 --kv_quant_tiered_depth_policy" if tiered else "6")
        )
        run_info = {
            "git_commit": "0123456789abcdef",
            "config_path": str(config),
            "config_hash": config_hash,
            "codebook_path": codebook_paths,
            "codebook_hash": codebook_hashes,
            "hostname": "synthetic-host",
            "gpu_type": "Synthetic GPU",
            "seed": 42,
            "prompt_list": PROMPTS,
            "timestamp": "2026-07-15T00:00:00+00:00",
            "model": "Wan2.2-TI2V-5B / HY-WorldPlay",
            "baseline": summarizer.BASELINE_ID,
            "input_norm_mode": "none",
            "per_token_rms_scale_stored": False,
            "codebook_storage_dtype": "int4",
            "setting": setting,
            "command": command,
            "tiered_policy": policy if tiered else None,
            "rvq_num_levels": 8 if tiered else 6,
            "offline_gate": gate,
            "terminal_index_payload_accounting": accounting,
        }
        (setting_dir / "run_info.json").write_text(json.dumps(run_info), encoding="utf-8")
        rows = [
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": 42,
                "setting": "krvq-vrvq-i4",
                "video_path": f"{setting_dir}/{prompt_id}-42.mp4",
                "e2e_time_s": 100 + prompt_id + int(tiered),
                "denoise_time_s": 90 + prompt_id,
                "quant_time_s": 5 + int(tiered),
                "peak_gpu_mem_mb": 30000,
                "kv_cache_mem_mb": 903.0 if tiered else 900.0,
                "compression_ratio": 27.0 if tiered else 27.1,
                "rvq_actual_avg_depth": tiered_depth if tiered else uniform_depth,
                "rvq_k_attention_kl": "",
                "rvq_v_output_mse": "",
            }
            for prompt_id, prompt in enumerate(PROMPTS)
        ]
        write_csv(setting_dir / "metrics.csv", rows)
        values = tiered_quality if tiered else uniform_quality
        quality_rows = [
            {
                "prompt_id": prompt_id,
                "prompt": PROMPTS[prompt_id],
                "setting": setting,
                "psnr": values["psnr"][prompt_id],
                "ssim": values["ssim"][prompt_id],
                "lpips": values["lpips"][prompt_id],
            }
            for prompt_id in range(3)
        ]
        (setting_dir / "quality.json").write_text(
            json.dumps(
                {
                    "setting": setting,
                    "num_videos": 3,
                    "psnr_mean": sum(row["psnr"] for row in quality_rows) / 3,
                    "ssim_mean": sum(row["ssim"] for row in quality_rows) / 3,
                    "lpips_mean": sum(row["lpips"] for row in quality_rows) / 3,
                    "videos": quality_rows,
                }
            ),
            encoding="utf-8",
        )
        write_csv(setting_dir / "quality.csv", quality_rows)
        if tiered and not missing_audit:
            cache_objects = 60
            physical = cache_objects * 24 * 4 * 880 * sum(schedule)
            storage = {
                "schedule": schedule,
                "average_depth": 6.0,
                "cache_objects": cache_objects,
                "span_count": cache_objects * len(schedule),
                "index_count": physical,
                "actual_uint8_index_payload_bytes": physical,
                "logical_c64_6bit_index_payload_bytes": physical * 0.75,
                "includes_codebooks": False,
                "includes_container_overhead": False,
                "per_token_rms_scale_present": False,
            }
            for prompt_id, prompt in enumerate(PROMPTS):
                audit = {
                    "schema_version": 1,
                    "policy": summarizer.launcher.H7_TIERED_POLICY,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": 42,
                    "num_generation_chunks": 11,
                    "num_cache_chunks": 10,
                    "runtime_codebook_storage": "int4",
                    "input_norm_mode_required": "none",
                    "pipeline_audit": {
                        "final": {
                            "schedule": schedule,
                            "average_depth": 6.0,
                            "storage": storage,
                        }
                    },
                }
                (setting_dir / f"{prompt_id}-42-tiered_depth_audit.json").write_text(
                    json.dumps(audit), encoding="utf-8"
                )
    return run_dir


def test_complete_record_is_aggregator_compatible_and_reports_deltas(tmp_path: Path) -> None:
    summarizer = load_module(SCRIPT, "h7_video_summarizer_complete")
    run_dir = synthetic_run(tmp_path, summarizer)
    output = tmp_path / "canonical" / "h7.json"
    assert summarizer.main([str(run_dir), "--output", str(output)]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["run_info"]["input_norm_mode"] == "none"
    assert record["run_info"]["codebook_storage_dtype"] == "int4"
    assert record["run_info"]["per_token_rms_scale_stored"] is False
    assert record["experiment_id"] == "h7_three_prompt_tiered_video_smoke"
    assert record["baseline_id"] == "nonorm_l8c64_int4"
    assert record["status"] == "complete"
    assert record["completeness"]["offline_gate_authorized"] is True
    assert record["completeness"]["actual_index_rate_matched"] is True
    assert record["prediction_verdict"]["status"] == "supported"
    assert record["verdict"]["status"] == "partially_supported"
    assert record["actual_storage_accounting"]["matched"] is True
    assert record["actual_storage_accounting"]["runtime_codebook_payload_bytes"] == 48_660_480
    assert record["actual_storage_accounting"]["default_dequantized_acceleration_cache"] == {
        "implementation_default": "HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE=1 when unset",
        "bf16_runtime_codebooks_bytes": 188_743_680,
        "fp32_codebook_squared_norm_bytes": 2_949_120,
        "total_bytes": 191_692_800,
        "mib": 182.8125,
        "equal_for_both_settings": True,
        "included_in_preregistered_int4_payload": False,
    }
    assert record["actual_storage_accounting"]["rows"][0][
        "actual_preregistered_payload_bytes"
    ] == 352_788_480
    assert record["matched_rate_validation"]["per_prompt"][0][
        "kv_cache_mib_delta_tiered_minus_uniform"
    ] == 3.0

    aggregator = load_module(AGGREGATOR, "h7_video_aggregator")
    manifest = json.loads(aggregator.DEFAULT_HYPOTHESES.read_text(encoding="utf-8"))
    accepted, rejected, warnings = aggregator.load_external_evidence([output], manifest["baseline"])
    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert accepted[0]["verdict_eligible"] is True
    with pytest.raises(FileExistsError):
        summarizer.main([str(run_dir), "--output", str(output)])


def test_missing_tiered_runtime_audits_stays_partial(tmp_path: Path) -> None:
    summarizer = load_module(SCRIPT, "h7_video_summarizer_missing")
    run_dir = synthetic_run(tmp_path, summarizer, missing_audit=True)
    record = summarizer.build_record(run_dir, summarizer.DEFAULT_MANIFEST)
    assert record["status"] == "partial"
    assert record["matched_rate_validation"]["passed"] is False
    assert record["prediction_verdict"]["status"] == "inconclusive"
    assert any("tiered_depth_audit" in issue for issue in record["completeness"]["issues"])


def test_reported_average_depth_mismatch_stays_partial(tmp_path: Path) -> None:
    summarizer = load_module(SCRIPT, "h7_video_summarizer_depth")
    run_dir = synthetic_run(tmp_path, summarizer, tiered_depth=6.1)
    record = summarizer.build_record(run_dir, summarizer.DEFAULT_MANIFEST)
    assert record["status"] == "partial"
    assert record["completeness"]["actual_index_rate_matched"] is False


def test_fixed_l6_empty_depth_cell_uses_validated_command_and_runtime_geometry(
    tmp_path: Path,
) -> None:
    summarizer = load_module(SCRIPT, "h7_video_summarizer_fixed_depth_empty")
    run_dir = synthetic_run(tmp_path, summarizer, uniform_depth="")
    record = summarizer.build_record(run_dir, summarizer.DEFAULT_MANIFEST)
    assert record["status"] == "complete"
    rate = record["matched_rate_validation"]
    assert rate["passed"] is True
    first = rate["per_prompt"][0]
    assert first["uniform_avg_depth"] == 6.0
    assert first["uniform_avg_depth_logged"] is None
    assert first["uniform_depth_source"] == "fixed_l6_command_and_runtime_geometry"
    assert (
        first["uniform_actual_uint8_index_payload_bytes"]
        == first["tiered_actual_uint8_index_payload_bytes"]
        == 304_128_000
    )


def test_preregistered_codebook_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    summarizer = load_module(SCRIPT, "h7_video_summarizer_hash")
    run_dir = synthetic_run(tmp_path, summarizer, bad_codebook_hash=True)
    with pytest.raises(ValueError, match="preregistered baseline"):
        summarizer.build_record(run_dir, summarizer.DEFAULT_MANIFEST)
