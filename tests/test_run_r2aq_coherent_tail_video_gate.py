from __future__ import annotations

import csv
from pathlib import Path
import sys

import pytest

from scripts.eval import run_small_eval
from scripts import run_r2aq_coherent_tail_video_gate as runner


def _arg(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_video_command_is_preregistered_g128_f100_and_mixed_decode(
    tmp_path: Path,
) -> None:
    command = runner.video_command(
        tmp_path, 4, tmp_path / "attempt", smoke=False
    )
    assert _arg(command, "--num_inference_steps") == "50"
    assert _arg(command, "--num_chunk") == "11"
    assert _arg(command, "--rvq_residual_codec_span_schedule") == (
        "r,r,r,r,r"
    )
    assert _arg(command, "--r2aq_residual_bits") == "4"
    assert _arg(command, "--r2aq_ablation") == "coherent_tail_g128"
    assert _arg(
        command, "--r2aq_coherent_tail_group_fraction"
    ) == "1.0"
    assert _arg(command, "--r2aq_coherent_tail_group_size") == "128"
    assert _arg(command, "--r2aq_shared_budget_owner_layer") == "0"
    assert _arg(command, "--r2aq_shared_budget_owner_span") == "0"
    assert _arg(command, "--r2aq_payload_logical_multiplier") == "2"
    assert "--r2aq_fast_decode" in command
    assert "--r2aq_coherent_tail_eager_decode" in command
    assert "--r2aq_selection_fraction" not in command


def test_video_command_roundtrips_small_eval_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command = runner.video_command(
        tmp_path, 0, tmp_path / "attempt", smoke=True
    )
    monkeypatch.setattr(sys, "argv", ["run_small_eval.py", *command[2:]])
    args = run_small_eval.parse_args()
    run_small_eval.validate_r2aq_codec_options(args)
    fragment = run_small_eval.rvq_residual_quant_config(args)
    assert fragment["r2aq_ablation"] == "coherent_tail_g128"
    assert fragment["r2aq_coherent_tail_group_fraction"] == 1.0
    assert fragment["r2aq_shared_budget_owner_layer"] == 0
    assert fragment["r2aq_shared_budget_owner_span"] == 0
    assert fragment["r2aq_payload_logical_multiplier"] == 2


def test_config_reports_actual_fixed_solver_updates_and_forbids_vbench() -> None:
    payload = runner.config_payload()
    generation = payload["generation"]
    assert generation["cli_num_inference_steps_requested"] == 50
    assert generation["actual_solver_updates"] == 4
    assert payload["vbench"]["authorized"] is False
    protocol = payload["cache_protocol"]
    assert protocol["physical_quantized_span_tokens"] == 7040
    assert protocol["pending_bf16_tokens"] == 3520


def _candidate_and_baseline() -> tuple[dict[str, str], dict[str, str]]:
    baseline = {
        "k_payload_mib": "3978.808594",
        "v_payload_mib": "3978.808594",
        "residual_k_payload_mib": "3688.769531",
        "residual_v_payload_mib": "3688.769531",
        "shared_codebook_mib": "91.406250",
    }
    qv_k = runner._nearest_bytes_from_legacy_mib(
        baseline["k_payload_mib"]
    )
    qv_v = runner._nearest_bytes_from_legacy_mib(
        baseline["v_payload_mib"]
    )
    qv_res_k = runner._nearest_bytes_from_legacy_mib(
        baseline["residual_k_payload_mib"]
    )
    qv_res_v = runner._nearest_bytes_from_legacy_mib(
        baseline["residual_v_payload_mib"]
    )
    qv_shared = runner._nearest_bytes_from_legacy_mib(
        baseline["shared_codebook_mib"]
    )
    r2aq_shared = 1572
    local_reserve = r2aq_shared // 2
    candidate = {
        "prompt_id": "0",
        "k_payload_bytes": str(qv_k - 1386),
        "v_payload_bytes": str(qv_v - 186),
        "kv_payload_bytes": str(qv_k + qv_v - r2aq_shared),
        "residual_k_payload_bytes": str(qv_res_k - 1386),
        "residual_v_payload_bytes": str(qv_res_v - 186),
        "residual_kv_payload_bytes": str(
            qv_res_k + qv_res_v - r2aq_shared
        ),
        "r2aq_shared_table_bytes": str(r2aq_shared),
        "global_shared_bytes": str(qv_shared + r2aq_shared),
        "physical_kv_payload_bytes": str(qv_k + qv_v + qv_shared),
        "r2aq_shared_budget_owner_states": "2",
        "r2aq_shared_budget_reserved_local_bytes": str(local_reserve),
        "r2aq_shared_budget_reserved_logical_bytes": str(r2aq_shared),
        "r2aq_shared_budget_exactly_repaid": "True",
    }
    return candidate, baseline


def test_strict_payload_gate_counts_shared_once_and_fails_one_byte_drift() -> None:
    candidate, baseline = _candidate_and_baseline()
    audit = runner.strict_payload_audit(candidate, baseline)
    assert audit["passed"] is True
    candidate["physical_kv_payload_bytes"] = str(
        int(candidate["physical_kv_payload_bytes"]) + 1
    )
    bad = runner.strict_payload_audit(candidate, baseline)
    assert bad["passed"] is False
    assert bad["checks"]["total_physical_kv_exactly_matches_qv4"] is False


def test_stage_progression_and_launch_are_fail_closed(tmp_path: Path) -> None:
    assert runner.previous_stage_authorized(tmp_path, 0) is True
    assert runner.previous_stage_authorized(tmp_path, 1) is False
    with pytest.raises(RuntimeError, match="launch-authorized"):
        runner.worker(
            tmp_path,
            0,
            0,
            smoke=True,
            dry_run=False,
            launch_authorized=False,
        )


def test_create_only_csv_does_not_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = [{"prompt_id": 0, "value": 1.0}]
    runner.write_csv_create_only(path, rows)
    runner.write_csv_create_only(path, rows)
    assert list(csv.DictReader(path.open(newline="", encoding="utf-8"))) == [
        {"prompt_id": "0", "value": "1.0"}
    ]


def test_macro_composite_and_bootstrap_use_only_staged_prompts() -> None:
    baseline = {
        prompt_id: {
            "mse": 2.0,
            "ssim": 0.8,
            "lpips": 0.4,
            "psnr": 10.0,
        }
        for prompt_id in runner.PROMPT_IDS
    }
    candidate = {
        prompt_id: {
            "mse": 1.0,
            "ssim": 0.9,
            "lpips": 0.2,
            "psnr": 13.0,
        }
        for prompt_id in runner.PROMPT_IDS
    }
    improvement, _ = runner._macro_composite(
        candidate, baseline, runner.PROMPT_IDS
    )
    low, high = runner.bootstrap_ci(
        candidate, baseline, runner.PROMPT_IDS, seed=1
    )
    assert improvement == pytest.approx(50.0)
    assert low == pytest.approx(50.0)
    assert high == pytest.approx(50.0)
