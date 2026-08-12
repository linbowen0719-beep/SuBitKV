from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wan5b_tensor_hybrid_video_gate.py"
SPEC = importlib.util.spec_from_file_location("tensor_hybrid_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(tmp_path: Path, prompt_id: int = 4) -> SimpleNamespace:
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    params = tmp_path / "params.pt"
    params.write_bytes(b"params")
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(
        "\n".join(f"prompt {index}" for index in range(10)) + "\n",
        encoding="utf-8",
    )
    return SimpleNamespace(
        python=python,
        prompt_file=prompt_file,
        prompt_id=prompt_id,
        gpu=0,
        output_root=tmp_path / "outputs",
        codec_params=params,
        k_codebook_dir=tmp_path / "k",
        v_codebook_dir=tmp_path / "v",
        preregistration=tmp_path / "preregistration.json",
    )


def _arg_value(command: list[str], option: str) -> str:
    position = command.index(option)
    return command[position + 1]


def test_generation_command_is_the_frozen_single_prompt_hybrid(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    prompt_dir = tmp_path / "candidate"
    command = MODULE.generation_command(args, prompt_dir)

    assert _arg_value(command, "--prompt-start") == "4"
    assert _arg_value(command, "--num-prompts") == "1"
    assert _arg_value(command, "--num_chunk") == "11"
    assert _arg_value(command, "--num_inference_steps") == "50"
    assert _arg_value(command, "--rvq_residual_codec") == "quantvideo"
    assert (
        _arg_value(command, "--rvq_k_residual_codec")
        == "structured_truncatable"
    )
    assert _arg_value(command, "--rvq_v_residual_codec") == "quantvideo"
    assert _arg_value(command, "--truncatable_residual_bits") == "2"
    assert _arg_value(command, "--structured_residual_k_bits") == "2"
    assert "--structured_residual_fast_decode" in command
    assert "--kv_quant_structured_temporal_policy" not in command


def test_registered_prompt_choices_reject_unregistered_prompt() -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--prompt-id", "3"])


def test_validate_preregistration_checks_frozen_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prereg = tmp_path / "preregistration.json"
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    payload = {
        "schema_version": "tensor_hybrid_video_dev_gate.preregistration.v1",
        "source_sha256": {"source.py": MODULE.sha256_file(source)},
    }
    prereg.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        MODULE, "PREREGISTRATION_SHA256", MODULE.sha256_file(prereg)
    )
    monkeypatch.setattr(MODULE, "REPO_ROOT", tmp_path)

    assert MODULE.validate_preregistration(prereg) == payload
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no longer match"):
        MODULE.validate_preregistration(prereg)


def test_audit_outputs_accepts_exact_protocol_and_payload(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    prompt_dir = tmp_path / "candidate"
    prompt_dir.mkdir()
    runtime_args = {
        **MODULE.FROZEN_CODEC_PARAMS,
        "prompt_start": 4,
        "num_prompts": 1,
        "seed": MODULE.SEED,
        "num_frames": MODULE.NUM_FRAMES,
        "num_chunk": MODULE.GENERATION_CHUNKS,
        "num_inference_steps": MODULE.CLI_STEPS,
        "output_dir": str(prompt_dir),
        "truncatable_residual_params_path": str(args.codec_params),
        "k_rvq_codebook_dir": str(args.k_codebook_dir),
        "v_rvq_codebook_dir": str(args.v_codebook_dir),
        "kv_quant_structured_temporal_policy": False,
    }
    (prompt_dir / "config.json").write_text(
        json.dumps(
            {
                "args": runtime_args,
                "resolved_quant_type": "triton-nstages-kmeans-int2",
            }
        ),
        encoding="utf-8",
    )
    chunks = "\n".join(
        f"Generate time for chunk  {index} is 1.0"
        for index in range(MODULE.GENERATION_CHUNKS)
    )
    noise = "\n".join(
        ["Calculating noise prediction"] * MODULE.EXPECTED_NOISE_PREDICTIONS
    )
    (prompt_dir / "generation.log").write_text(
        chunks + "\n" + noise + "\n", encoding="utf-8"
    )
    (prompt_dir / "4-42.mp4").write_bytes(b"video")
    with (prompt_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "prompt_id",
                "seed",
                "kv_cache_mem_mb",
                "residual_kv_payload_mib",
                "compression_ratio",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": 4,
                "seed": 42,
                "kv_cache_mem_mb": "4955.273438",
                "residual_kv_payload_mib": "4283.789062",
                "compression_ratio": "4.994679",
            }
        )

    audit = MODULE.audit_outputs(args, prompt_dir)
    assert audit["status"] == "PASS"
    assert audit["sampler"]["noise_prediction_count"] == 44
    assert audit["payload"][
        "matches_frozen_qv2_budget_at_reporting_precision"
    ]


def test_execute_prompt_directory_is_create_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, prompt_id=0)
    args.output_root.mkdir()
    prompt_dir = args.output_root / f"{MODULE.SETTING_NAME}_p0"
    prompt_dir.mkdir()

    monkeypatch.setattr(MODULE, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(
        MODULE,
        "validate_preregistration",
        lambda _path: {
            "schema_version": (
                "tensor_hybrid_video_dev_gate.preregistration.v1"
            )
        },
    )
    monkeypatch.setattr(
        MODULE,
        "audit_codebook_dir",
        lambda path, tensor: {
            "path": str(path),
            "hash": tensor,
            "files": 720,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "source_provenance",
        lambda: {
            "worktree_dirty": True,
            "source_sha256": {},
            "execution_source_file_count": 0,
            "execution_source_manifest_sha256": "x",
            "git_diff_sha256_tracked_sources": "y",
        },
    )
    args.execute = True
    with pytest.raises(FileExistsError):
        MODULE.main([])
