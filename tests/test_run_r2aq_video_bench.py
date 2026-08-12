from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import run_r2aq_video_bench as runner


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_video_command_binds_physical_span_protocol_and_full_tree(
    tmp_path: Path,
) -> None:
    command = runner.video_command(
        tmp_path, "b2_full_tree", 3, tmp_path / "attempt", smoke=False
    )
    assert _arg(command, "--rvq_residual_codec") == "r2aq"
    assert _arg(command, "--rvq_residual_codec_span_schedule") == "r,r,r,r,r"
    assert _arg(command, "--r2aq_residual_bits") == "2"
    assert _arg(command, "--r2aq_selection_mode") == "scale_utility"
    assert _arg(command, "--r2aq_tree_params_path") == str(runner.PARAMS)
    assert _arg(command, "--r2aq_utility_weights_path") == str(runner.UTILITY)
    assert _arg(command, "--kv_logical_head_dim") == "256"
    assert _arg(command, "--kv_quant_factor") == "2"
    assert "--r2aq_fast_decode" in command


def test_utility_b4_does_not_load_learned_tree(tmp_path: Path) -> None:
    command = runner.video_command(
        tmp_path, "b4_utility", 0, tmp_path / "attempt", smoke=True
    )
    assert _arg(command, "--num_inference_steps") == "2"
    assert _arg(command, "--num_chunk") == "11"
    assert _arg(command, "--r2aq_residual_bits") == "4"
    assert "--r2aq_utility_weights_path" in command
    assert "--r2aq_tree_params_path" not in command


def test_smoke_attempts_cannot_satisfy_formal_completion(tmp_path: Path) -> None:
    assert runner.attempt_root(
        tmp_path, "b2_full_tree", 0, smoke=True
    ) != runner.attempt_root(
        tmp_path, "b2_full_tree", 0, smoke=False
    )


def test_shards_are_disjoint_and_complete(tmp_path: Path) -> None:
    jobs = [
        {"task_index": index, "variant": "b2_full_tree", "prompt_id": index}
        for index in range(10)
    ]
    (tmp_path / "jobs.json").write_text(
        json.dumps({"jobs": jobs}), encoding="utf-8"
    )
    shards = [
        runner.selected_jobs(tmp_path, shard, 4) for shard in range(4)
    ]
    indices = [int(job["task_index"]) for shard in shards for job in shard]
    assert sorted(indices) == list(range(10))
    assert len(indices) == len(set(indices))


def test_vbench_is_fail_closed_before_gate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="blocked"):
        runner.vbench_worker(
            tmp_path,
            ("b2_full_tree",),
            0,
            1,
            0,
            dry_run=True,
            exact_gate_json=None,
        )


def test_parse_variants_rejects_duplicate_and_unknown() -> None:
    with pytest.raises(ValueError):
        runner.parse_variants("b2_full_tree,b2_full_tree")
    with pytest.raises(ValueError):
        runner.parse_variants("mystery")


def test_config_preregisters_rate_and_similarity_gate() -> None:
    payload = runner.config_payload(("b2_full_tree", "b4_utility"))
    protocol = payload["cache_protocol"]
    assert protocol["physical_quantized_span_tokens"] == 7040
    assert protocol["pending_bf16_tokens"] == 3520
    gate = payload["similarity"]["gate"]
    assert gate["composite_improvement_pct"] == 10.0
    assert gate["minimum_prompt_win_rate"] == 0.8
    assert payload["vbench"]["requires_external_exact_attention_gate"] is True


def test_bootstrap_composite_ci_uses_macro_metric_ratios() -> None:
    baseline = {
        index: {"mse": 2.0, "ssim": 0.8, "lpips": 0.4, "psnr": 10.0}
        for index in range(10)
    }
    candidate = {
        index: {"mse": 1.0, "ssim": 0.9, "lpips": 0.2, "psnr": 13.0}
        for index in range(10)
    }
    low, high = runner.bootstrap_composite_ci(candidate, baseline)
    assert low == pytest.approx(50.0)
    assert high == pytest.approx(50.0)


def test_create_only_csv_does_not_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    runner.write_csv_create_only(path, [{"prompt_id": 0, "value": 1.0}])
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    assert rows == [{"prompt_id": "0", "value": "1.0"}]
