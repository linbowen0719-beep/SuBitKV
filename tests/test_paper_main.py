from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


REPO = Path(__file__).parents[1]


def load_module(name: str, path: Path):
    sys.path.insert(0, str(REPO / "scripts/main"))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate = load_module(
    "qvg_paper_aggregate", REPO / "scripts/main/aggregate_paper_main.py"
)


def portable_environment(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "SUBITKV_CKPT_ROOT": str(tmp_path / "ckpts"),
        "SUBITKV_DATA_ROOT": str(tmp_path / "data"),
        "SUBITKV_OUTPUT_ROOT": str(tmp_path / "outputs"),
        "SUBITKV_CODEBOOK_ROOT": str(tmp_path / "codebooks"),
        "SUBITKV_CACHE_ROOT": str(tmp_path / "cache"),
    })
    return env


def test_frozen_protocol_has_11_rows_per_model() -> None:
    config_path = REPO / "configs/main/moviegen-1003.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prompt = REPO / "assets/MovieGenVideoBench.txt"

    assert len(payload["methods"]["wan2_2_5b"]) == 11
    assert len(payload["methods"]["self_forcing"]) == 11
    assert len(prompt.read_text(encoding="utf-8").splitlines()) == 1003
    train_prompts = REPO / "assets/moviegen_train20.txt"
    assert len(train_prompts.read_text(encoding="utf-8").splitlines()) == 20
    assert payload["protocol"]["prompt_sha256"] == (
        "72921e5b91ed14665d6cac50fca51f04dff356baddeffbab324ac9280e8ec05a"
    )
    assert "turboquant_prod_k4_v4" in payload["methods"]["wan2_2_5b"]
    assert "turboquant_prod_k2_v2" in payload["methods"]["self_forcing"]
    assert payload["protocol"]["bf16_kv_cpu_offload"] is True
    assert payload["rvq_training"]["prompt_sha256"] == (
        "e7a4ad6f324a3671b4205f8c2c9ff3c487370d8ea7089836556ec4b3b3b6cdc0"
    )


def test_wan_direct_codec_ratio_uses_physical_d128_accounting() -> None:
    assert aggregate.table_ratio("quantvideo_prerope_int4", 5.99) == 2.995
    assert aggregate.table_ratio("kivi_all_int2", 10.6664) == 5.3332
    assert aggregate.table_ratio("turboquant_prod_k2_v2", 14.6282) == 7.3141
    assert aggregate.table_ratio("adaptive_temporal_int8_avg2", 6.1756) == 6.1756
    assert aggregate.table_ratio("bf16", 1.0) == 1.0


def test_paper_plan_is_read_only_and_complete(tmp_path: Path) -> None:
    env = portable_environment(tmp_path)
    result = subprocess.run(
        [
            sys.executable, "scripts/run_paper_main.py", "plan",
            "--run-id", "unit-main", "--gpus", "0,1", "--stage", "generate",
        ],
        cwd=REPO, env=env, text=True, capture_output=True, check=True,
    )
    payload = json.loads(result.stdout)
    names = [item["name"] for item in payload["commands"]]
    assert names == ["wan:create", "wan:gate", "wan:videos", "self:generate"]
    assert payload["mode"] == "dry-run"
    assert not (tmp_path / "outputs").exists()
    self_command = next(item for item in payload["commands"] if item["name"] == "self:generate")
    argv = self_command["argv"]
    for required in (
        "--k-codebooks", "--v-codebooks", "--routing", "--tree",
        "--v-params", "--profile", "--prompt-file", "--checkpoint",
    ):
        assert required in argv


def test_paper_doctor_reports_missing_assets(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_paper_main.py", "doctor"],
        cwd=REPO, env=portable_environment(tmp_path), text=True, capture_output=True,
    )
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert any("missing bundle asset" in error for error in payload["errors"])
    assert any("Self-Forcing DMD checkpoint" in error for error in payload["errors"])


def test_supported_paper_files_have_no_personal_absolute_paths() -> None:
    paths = [
        REPO / "configs/main/moviegen-1003.yaml",
        REPO / "scripts/run_paper_main.py",
        REPO / "scripts/main/paper_main_common.py",
        REPO / "scripts/main/export_main_table_bundle.py",
        REPO / "scripts/main/aggregate_paper_main.py",
        REPO / "scripts/main/train_rvq.py",
        REPO / "scripts/main/run_moviegen.py",
        REPO / "scripts/main/run_benchmark.py",
        REPO / "scripts/run_adaptive_rvq_best_residual_video_bench.py",
        REPO / "scripts/Self-Forcing/run_progressive_temporal_experiment.py",
        REPO / "scripts/analysis/run_self_forcing_cross_model_jobs.py",
        REPO / "scripts/analysis/evaluate_self_forcing_cross_model_quality.py",
        REPO / "scripts/analysis/run_cross_model_vbench.py",
        REPO / "scripts/eval/run_small_eval.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/home/bowen" not in text, path
        assert "/data/bowen" not in text, path


def test_bf16_offload_is_explicit_in_both_integrations() -> None:
    wan = (REPO / "scripts/run_adaptive_rvq_best_residual_video_bench.py").read_text()
    self_forcing = (
        REPO / "scripts/Self-Forcing/run_progressive_temporal_experiment.py"
    ).read_text()
    assert "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD" in wan
    assert 'common["kv_cache_cpu_offload"] = args.method == "bf16"' in self_forcing
