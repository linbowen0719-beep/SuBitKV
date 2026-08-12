from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/summarize_wan5b_h5_video_evidence.py"
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


def write_synthetic_run(
    root: Path,
    summarizer,
    *,
    quality_count: int = 3,
    dynamic_kv_mib: float = 940.693359375,
    wrong_k_hash: bool = False,
) -> Path:
    run_dir = root / "h5_run"
    run_dir.mkdir()
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps({"baseline": summarizer.BASELINE_ID, "prompts": PROMPTS}),
        encoding="utf-8",
    )
    config_hash = summarizer.sha256_file(config_path)
    manifest = json.loads(summarizer.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    baseline = manifest["baseline"]
    codebook_paths = {
        "K": baseline["k_codebook_path"],
        "V": baseline["v_codebook_path"],
    }
    codebook_hashes = {
        "K": "bad-hash" if wrong_k_hash else baseline["k_codebook_hash"],
        "V": baseline["v_codebook_hash"],
    }
    static_quality = {
        "psnr": [20.0, 21.0, 22.0],
        "ssim": [0.70, 0.71, 0.72],
        "lpips": [0.30, 0.29, 0.28],
    }
    dynamic_quality = {
        "psnr": [20.5, 21.5, 22.5],
        "ssim": [0.72, 0.73, 0.74],
        "lpips": [0.27, 0.26, 0.25],
    }
    for setting in summarizer.SETTING_NAMES:
        setting_dir = run_dir / setting
        setting_dir.mkdir()
        dynamic = setting == summarizer.DYNAMIC_NAME
        policy = (
            "--kv_quant_recent_frames 1"
            if dynamic
            else "--kv_quant_skip_last_frames 1"
        )
        command = (
            "python scripts/eval/run_small_eval.py --rvq_num_levels 8 "
            "--rvq_codebook_storage_dtype int4 --kv_quant_skip_first_frames 1 "
            + policy
        )
        run_info = {
            "git_commit": "0123456789abcdef",
            "config_path": str(config_path),
            "config_hash": config_hash,
            "codebook_path": codebook_paths,
            "codebook_hash": codebook_hashes,
            "hostname": "synthetic-host",
            "gpu_type": "Synthetic GPU",
            "seed": 42,
            "prompt_list": PROMPTS,
            "timestamp": "2026-07-14T00:00:00+00:00",
            "model": "Wan2.2-TI2V-5B / HY-WorldPlay",
            "baseline": summarizer.BASELINE_ID,
            "input_norm_mode": "none",
            "per_token_rms_scale_stored": False,
            "setting": setting,
            "command": command,
        }
        (setting_dir / "run_info.json").write_text(
            json.dumps(run_info), encoding="utf-8"
        )
        kv_mib = dynamic_kv_mib if dynamic else 940.693359375
        metric_rows = [
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seed": 42,
                "setting": "kvrvq-i4",
                "video_path": f"{setting_dir}/{prompt_id}-42.mp4",
                "e2e_time_s": 100.0 + prompt_id + int(dynamic),
                "kv_cache_mem_mb": kv_mib,
                "compression_ratio": 26.3103802672148,
            }
            for prompt_id, prompt in enumerate(PROMPTS)
        ]
        write_csv(setting_dir / "metrics.csv", metric_rows)
        quality_values = dynamic_quality if dynamic else static_quality
        quality_rows = [
            {
                "prompt_id": prompt_id,
                "prompt": PROMPTS[prompt_id],
                "setting": setting,
                "psnr": quality_values["psnr"][prompt_id],
                "ssim": quality_values["ssim"][prompt_id],
                "lpips": quality_values["lpips"][prompt_id],
            }
            for prompt_id in range(quality_count)
        ]
        quality = {
            "setting": setting,
            "num_videos": quality_count,
            "psnr_mean": sum(row["psnr"] for row in quality_rows) / quality_count,
            "ssim_mean": sum(row["ssim"] for row in quality_rows) / quality_count,
            "lpips_mean": sum(row["lpips"] for row in quality_rows) / quality_count,
            "videos": quality_rows,
        }
        (setting_dir / "quality.json").write_text(
            json.dumps(quality), encoding="utf-8"
        )
        write_csv(setting_dir / "quality.csv", quality_rows)
    return run_dir


def test_complete_record_is_canonical_supported_trend_and_never_overwrites(
    tmp_path: Path,
) -> None:
    summarizer = load_module(SCRIPT, "h5_video_summarizer_complete")
    run_dir = write_synthetic_run(tmp_path, summarizer)
    output = tmp_path / "canonical" / "h5.json"
    source_before = (run_dir / summarizer.STATIC_NAME / "quality.json").read_bytes()

    assert summarizer.main([str(run_dir), "--output", str(output)]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["schema_version"] == "wan5b_report_evidence.v1"
    assert record["experiment_id"] == "h5_three_prompt_no_norm_video_smoke"
    assert record["baseline_id"] == "nonorm_l8c64_int4"
    assert record["status"] == "complete"
    assert record["completeness"] == {
        "required_prompts_per_setting": 3,
        "metrics_complete": True,
        "quality_complete": True,
        "prompt_identity_complete": True,
        "terminal_budget_matched": True,
        "issues": [],
    }
    assert record["prediction_verdict"]["status"] == "supported"
    assert record["verdict"]["status"] == "partially_supported"
    assert record["terminal_budget_validation"]["passed"] is True
    assert len(record["tables"][0]["rows"]) == 6
    assert "terminal-only" in record["terminal_only_budget_caveat"]
    assert (run_dir / summarizer.STATIC_NAME / "quality.json").read_bytes() == source_before

    aggregator = load_module(AGGREGATOR, "h5_video_aggregator_complete")
    manifest = json.loads(aggregator.DEFAULT_HYPOTHESES.read_text(encoding="utf-8"))
    accepted, rejected, warnings = aggregator.load_external_evidence(
        [output], manifest["baseline"]
    )
    assert rejected == []
    assert warnings == []
    assert len(accepted) == 1
    assert accepted[0]["verdict_eligible"] is True

    with pytest.raises(FileExistsError):
        summarizer.main([str(run_dir), "--output", str(output)])
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_missing_one_quality_result_stays_partial(tmp_path: Path) -> None:
    summarizer = load_module(SCRIPT, "h5_video_summarizer_partial")
    run_dir = write_synthetic_run(tmp_path, summarizer, quality_count=2)

    record = summarizer.build_record(run_dir, summarizer.DEFAULT_MANIFEST)

    assert record["status"] == "partial"
    assert record["completeness"]["quality_complete"] is False
    assert record["prediction_verdict"]["status"] == "inconclusive"
    assert any("quality rows, expected 3" in issue for issue in record["completeness"]["issues"])


def test_actual_terminal_budget_mismatch_stays_partial(tmp_path: Path) -> None:
    summarizer = load_module(SCRIPT, "h5_video_summarizer_budget")
    run_dir = write_synthetic_run(tmp_path, summarizer, dynamic_kv_mib=941.0)

    record = summarizer.build_record(run_dir, summarizer.DEFAULT_MANIFEST)

    assert record["status"] == "partial"
    assert record["completeness"]["quality_complete"] is True
    assert record["completeness"]["terminal_budget_matched"] is False
    assert record["terminal_budget_validation"]["per_prompt"][0]["matched"] is False
    assert record["prediction_verdict"]["status"] == "inconclusive"


def test_preregistered_codebook_hash_mismatch_is_rejected_before_output(
    tmp_path: Path,
) -> None:
    summarizer = load_module(SCRIPT, "h5_video_summarizer_hash")
    run_dir = write_synthetic_run(tmp_path, summarizer, wrong_k_hash=True)
    output = tmp_path / "h5.json"

    with pytest.raises(ValueError, match="preregistered baseline"):
        summarizer.main([str(run_dir), "--output", str(output)])
    assert not output.exists()
