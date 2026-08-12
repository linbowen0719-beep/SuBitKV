from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wan5b_k_layer_selective_video_dev.py"
SPEC = importlib.util.spec_from_file_location("k_layer_video_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _args(
    tmp_path: Path,
    *,
    candidate: str = "B_mid20",
    prompt_id: int = 4,
) -> SimpleNamespace:
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
        execute=False,
        candidate=candidate,
        prompt_id=prompt_id,
        gpu=0,
        output_root=tmp_path / "outputs",
        prompt_file=prompt_file,
        python=python,
        k_codebook_dir=tmp_path / "k",
        v_codebook_dir=tmp_path / "v",
        codec_params=params,
        design_dir=tmp_path / "design",
    )


def _arg_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    (
        ("A_mid10", "8-15"),
        ("B_mid20", "16-23"),
        ("C_mid10_mid20", "8-23"),
    ),
)
def test_generation_command_uses_only_registered_layer_dispatch(
    tmp_path: Path,
    candidate: str,
    expected: str,
) -> None:
    args = _args(tmp_path, candidate=candidate)
    command = MODULE.generation_command(args, tmp_path / "output")
    assert _arg_value(command, "--prompt-start") == "4"
    assert _arg_value(command, "--num-prompts") == "1"
    assert _arg_value(command, "--num_chunk") == "11"
    assert _arg_value(command, "--num_inference_steps") == "50"
    assert _arg_value(command, "--rvq_residual_codec") == "quantvideo"
    assert _arg_value(command, "--rvq_v_residual_codec") == "quantvideo"
    assert _arg_value(command, "--rvq_k_structured_layers") == expected
    assert "--rvq_k_residual_codec" not in command
    assert _arg_value(command, "--truncatable_residual_bits") == "2"
    assert _arg_value(command, "--structured_residual_k_bits") == "2"
    assert "--structured_residual_fast_decode" in command
    assert "--kv_quant_structured_temporal_policy" not in command


def test_cli_rejects_unregistered_candidate_and_prompt() -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--candidate", "posthoc"])
    with pytest.raises(SystemExit):
        MODULE.parse_args(["--prompt-id", "1"])


def test_repository_design_amendment_and_source_freeze_validate() -> None:
    protocol, amendment = MODULE.validate_frozen_design(MODULE.DESIGN_DIR)
    assert protocol["development"]["prompt_ids"] == [0, 4, 9]
    assert (
        amendment["correction"]["source_sha256"]
        == MODULE.FROZEN_SOURCE_SHA256
    )


def test_validate_frozen_design_detects_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    design_dir = tmp_path / "design"
    design_dir.mkdir()
    names = tuple(MODULE.FROZEN_DESIGN_SHA256)
    for name in names:
        (design_dir / name).write_text("placeholder", encoding="utf-8")
    protocol = {
        "schema_version": "k_layer_selective_structured_dev_design.v1",
        "candidates": [
            {"id": key, "structured_K_layers": list(value)}
            for key, value in MODULE.CANDIDATE_LAYERS.items()
        ],
        "development": {"prompt_ids": [0, 4, 9]},
    }
    (design_dir / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    source_root = tmp_path / "source_root"
    source_root.mkdir()
    source_hashes = {}
    for relative in MODULE.FROZEN_SOURCE_SHA256:
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        source_hashes[relative] = MODULE.sha256_file(path)
    amendment = {
        "schema_version": (
            "k_layer_selective_structured_dev_design.runtime_amendment.v1"
        ),
        "parent_design": {},
        "correction": {
            "source_sha256": source_hashes,
            "runtime_codebook_hash": MODULE.FROZEN_CODEBOOK_HASH,
        },
    }
    amendment_path = design_dir / MODULE.AMENDMENT_PATH.name
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    design_hashes = {
        name: MODULE.sha256_file(design_dir / name) for name in names
    }
    amendment["parent_design"] = {
        "protocol_sha256": design_hashes["protocol.json"],
        "candidate_registry_sha256": design_hashes["candidate_registry.csv"],
    }
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")

    monkeypatch.setattr(MODULE, "REPO_ROOT", source_root)
    monkeypatch.setattr(MODULE, "FROZEN_SOURCE_SHA256", source_hashes)
    monkeypatch.setattr(MODULE, "FROZEN_DESIGN_SHA256", design_hashes)
    monkeypatch.setattr(
        MODULE, "AMENDMENT_SHA256", MODULE.sha256_file(amendment_path)
    )
    MODULE.validate_frozen_design(design_dir)

    first = source_root / next(iter(source_hashes))
    first.write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source changed"):
        MODULE.validate_frozen_design(design_dir)


def _write_valid_outputs(
    args: SimpleNamespace,
    prompt_dir: Path,
) -> None:
    runtime_args = {
        **MODULE.FROZEN_CODEC_PARAMS,
        "rvq_k_structured_layers": MODULE.layer_spec(args.candidate),
        "prompt_start": args.prompt_id,
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
    (prompt_dir / f"{args.prompt_id}-42.mp4").write_bytes(b"video")
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
                "prompt_id": args.prompt_id,
                "seed": 42,
                "kv_cache_mem_mb": "4955.273438",
                "residual_kv_payload_mib": "4283.789062",
                "compression_ratio": "4.994679",
            }
        )


def test_audit_outputs_checks_sampler_payload_and_hashes(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    prompt_dir = tmp_path / "candidate"
    prompt_dir.mkdir()
    _write_valid_outputs(args, prompt_dir)
    audit = MODULE.audit_outputs(args, prompt_dir)
    assert audit["status"] == "PASS"
    assert audit["sampler"]["noise_prediction_count"] == 44
    assert audit["sampler"]["generation_chunks"] == list(range(11))
    assert audit["payload"]["terminal_residual_payload_MiB"] == 4283.789062
    assert len(audit["outputs"]["video"]["sha256"]) == 64
    assert len(audit["outputs"]["generation_log"]["sha256"]) == 64


def test_audit_outputs_rejects_error_signature(tmp_path: Path) -> None:
    args = _args(tmp_path)
    prompt_dir = tmp_path / "candidate"
    prompt_dir.mkdir()
    _write_valid_outputs(args, prompt_dir)
    with (prompt_dir / "generation.log").open("a", encoding="utf-8") as handle:
        handle.write("RuntimeError: injected\n")
    with pytest.raises(RuntimeError, match="contains errors"):
        MODULE.audit_outputs(args, prompt_dir)


def test_execute_directory_is_create_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, candidate="A_mid10", prompt_id=0)
    args.execute = True
    args.output_root.mkdir()
    prompt_dir = args.output_root / "k_layer_A_mid10_p0"
    prompt_dir.mkdir()
    monkeypatch.setattr(MODULE, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(
        MODULE,
        "validate_frozen_design",
        lambda _path: (
            {
                "schema_version": (
                    "k_layer_selective_structured_dev_design.v1"
                )
            },
            {
                "schema_version": (
                    "k_layer_selective_structured_dev_design."
                    "runtime_amendment.v1"
                )
            },
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "audit_codebook_dir",
        lambda path, tensor: {
            "path": str(path),
            "hash": MODULE.FROZEN_CODEBOOK_HASH[tensor],
            "files": 720,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "source_provenance",
        lambda: {
            "worktree_dirty": True,
            "frozen_source_sha256": MODULE.FROZEN_SOURCE_SHA256,
            "runner_sha256": "x",
        },
    )
    with pytest.raises(FileExistsError):
        MODULE.main([])
