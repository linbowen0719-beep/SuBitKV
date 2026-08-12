from __future__ import annotations

import json

import pytest

from scripts import run_wan5b_functional_rvq_video_bench as bench


def test_vbench_prompt_values_are_normalized_to_published_aggregate(tmp_path) -> None:
    cfg = {"output_dir": str(tmp_path)}
    result_dir = (
        tmp_path / "video_benchmark" / "vbench" / "candidate" / "imaging_quality"
    )
    result_dir.mkdir(parents=True)
    (result_dir / "results_eval_results.json").write_text(
        json.dumps(
            {
                "imaging_quality": [
                    0.75,
                    [
                        {"video_path": "/tmp/0-42.mp4", "video_results": 70.0},
                        {"video_path": "/tmp/1-42.mp4", "video_results": 80.0},
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    values = bench.parse_vbench_per_prompt(cfg, "candidate", "imaging_quality")
    assert values == pytest.approx({0: 0.70, 1: 0.80})


def test_vbench_unit_scale_is_left_unchanged(tmp_path) -> None:
    cfg = {"output_dir": str(tmp_path)}
    result_dir = (
        tmp_path / "video_benchmark" / "vbench" / "candidate" / "subject_consistency"
    )
    result_dir.mkdir(parents=True)
    (result_dir / "results_eval_results.json").write_text(
        json.dumps(
            {
                "subject_consistency": [
                    0.75,
                    [
                        {"video_path": "/tmp/0-42.mp4", "video_results": 0.70},
                        {"video_path": "/tmp/1-42.mp4", "video_results": 0.80},
                    ],
                ]
            }
        ),
        encoding="utf-8",
    )
    values = bench.parse_vbench_per_prompt(cfg, "candidate", "subject_consistency")
    assert values == pytest.approx({0: 0.70, 1: 0.80})


def test_retry_suffix_preserves_failed_attempt_artifacts(tmp_path) -> None:
    paths = [
        tmp_path / "metrics_prompt_3.csv",
        tmp_path / "config_prompt_3.json",
        tmp_path / "video_candidate_prompt3.log",
    ]
    assert bench.available_attempt_suffix(paths) == ""
    paths[0].write_text("partial\n", encoding="utf-8")
    assert bench.available_attempt_suffix(paths) == "_retry1"
    paths[2].with_name("video_candidate_prompt3_retry1.log").write_text(
        "failed\n", encoding="utf-8"
    )
    assert bench.available_attempt_suffix(paths) == "_retry2"
