from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_tensor_hybrid_video_gate.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_tensor_hybrid_video_gate", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": gate.SIMILARITY_PROTOCOL,
        "decoded_color_space": "RGB",
        "decoded_float_input_range": [0.0, 1.0],
        "frame_0_included": False,
        "per_video_aggregation": "arithmetic mean over aligned decoded frames",
        "dataset_aggregation": "arithmetic mean over prompt-level metrics",
        "psnr": {
            "data_range": 1.0,
            "aggregation": "mean of per-frame dB values",
        },
        "ssim": {
            "implementation": "torch_avg_pool_11x11_rgb",
            "data_range": 1.0,
        },
        "lpips": {
            "package": "lpips",
            "package_version": "0.1.4",
            "network": "vgg",
            "model_version": "0.1",
            "normalize_input": True,
            "effective_input_range": [-1.0, 1.0],
        },
    }


def _row(
    prompt_id: int,
    *,
    scale: float,
    kind: str,
    reference: Path,
    video: Path,
) -> dict[str, object]:
    mse = (0.10 + 0.001 * prompt_id) * scale
    ssim_loss = (0.20 + 0.001 * prompt_id) * scale
    lpips = (0.20 + 0.001 * prompt_id) * scale
    return {
        "prompt_id": prompt_id,
        "prompt": f"prompt {prompt_id}",
        "setting": kind,
        "reference_video": str(reference.resolve()),
        "test_video": str(video.resolve()),
        "frames": 173,
        "compared_frames": 172,
        "mse": mse,
        "psnr": 20.0 - 10.0 * (scale - 0.8),
        "ssim": 1.0 - ssim_loss,
        "lpips": lpips,
    }


def _similarity(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": gate.SIMILARITY_SCHEMA,
        "metric_protocol": _protocol(),
        "setting": rows[0]["setting"],
        "reference": "bf16",
        "test": rows[0]["setting"],
        "num_videos": len(rows),
        "psnr_mean": sum(float(row["psnr"]) for row in rows) / len(rows),
        "ssim_mean": sum(float(row["ssim"]) for row in rows) / len(rows),
        "lpips_mean": sum(float(row["lpips"]) for row in rows) / len(rows),
        "videos": rows,
    }


def _write_similarity_run(
    root: Path, row: dict[str, object]
) -> tuple[Path, Path]:
    prompt_id = int(row["prompt_id"])
    output_json = root / f"p{prompt_id}" / "similarity.json"
    output_csv = output_json.with_suffix(".csv")
    _json(output_json, _similarity([row]))
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    config = output_json.parent / "similarity_config.json"
    _json(config, {"prompt_id": prompt_id, "protocol": gate.SIMILARITY_PROTOCOL})
    _json(
        output_json.parent / "run_info.json",
        {
            "schema_version": gate.SIMILARITY_RUN_INFO_SCHEMA,
            "git_commit": "fixture",
            "config_path": str(config.resolve()),
            "config_hash": _sha(config),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": "fixture",
            "gpu_type": "fixture",
            "seed": 42,
            "prompt_list": [
                {"prompt_id": prompt_id, "prompt": row["prompt"]}
            ],
            "timestamp": "2026-07-24T01:00:00Z",
            "output_json": str(output_json.resolve()),
            "output_json_sha256": _sha(output_json),
            "output_csv": str(output_csv.resolve()),
            "output_csv_sha256": _sha(output_csv),
        },
    )
    return output_json, output_csv


def _log(path: Path, noise: int = 44) -> None:
    lines = ["Number of inference steps: 50"]
    remaining = noise
    for chunk in range(11):
        count = min(4, remaining)
        lines.extend(["Calculating noise prediction"] * count)
        remaining -= count
        lines.append(f"Generate time for chunk {chunk} is 1.0 seconds")
    lines.extend(["Calculating noise prediction"] * remaining)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture(
    tmp_path: Path, *, candidate_scale: float = 0.80
) -> dict[str, object]:
    actual_prereg = json.loads(
        gate.DEFAULT_PREREGISTRATION.read_text(encoding="utf-8")
    )
    prereg_dir = tmp_path / "frozen"
    prereg = prereg_dir / "preregistration.json"

    references: dict[int, Path] = {}
    candidate_videos: dict[int, Path] = {}
    for prompt_id in gate.EXPECTED_PROMPT_IDS:
        reference = tmp_path / "reference" / f"{prompt_id}-42.mp4"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(f"reference-{prompt_id}".encode())
        references[prompt_id] = reference

    comparator_records: dict[str, dict[str, str]] = {}
    for comparator_id, scale in (
        ("uniform_structured_b2", 1.0),
        ("rvq_quantvideo_int2", 0.98),
    ):
        rows = []
        for prompt_id in range(10):
            reference = references.get(prompt_id)
            if reference is None:
                reference = tmp_path / "reference" / f"{prompt_id}-42.mp4"
                reference.write_bytes(f"reference-{prompt_id}".encode())
            video = tmp_path / "baseline_video" / comparator_id / f"{prompt_id}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"baseline")
            rows.append(
                _row(
                    prompt_id,
                    scale=scale,
                    kind=comparator_id,
                    reference=reference,
                    video=video,
                )
            )
        path = tmp_path / "baselines" / f"{comparator_id}.json"
        _json(path, _similarity(rows))
        comparator_records[comparator_id] = {
            "label": comparator_id,
            "similarity_json": str(path.resolve()),
            "sha256": _sha(path),
        }
    actual_prereg["comparators"] = comparator_records
    _json(prereg, actual_prereg)
    _json(
        prereg_dir / "run_info.json",
        {
            "schema_version": gate.PREREGISTRATION_RUN_INFO_SCHEMA,
            "config_path": str(prereg.resolve()),
            "config_hash": _sha(prereg),
            "seed": 42,
        },
    )

    actual_amendment = json.loads(
        (
            gate.DEFAULT_GATE_ROOT / "preregistration_amendment_001.json"
        ).read_text(encoding="utf-8")
    )
    amendment = prereg_dir / "preregistration_amendment_001.json"
    actual_amendment["parent_preregistration"] = {
        "path": str(prereg.resolve()),
        "sha256": _sha(prereg),
    }
    _json(amendment, actual_amendment)
    _json(
        prereg_dir / "amendment_001_run_info.json",
        {
            "schema_version": gate.AMENDMENT_RUN_INFO_SCHEMA,
            "config_path": str(amendment.resolve()),
            "config_hash": _sha(amendment),
            "codebook_hash": gate._load_json(amendment)["correction"][
                "runtime_codebook_hash"
            ],
        },
    )
    amended_hashes = actual_amendment["correction"]["runtime_codebook_hash"]

    roots: list[Path] = []
    similarity_jsons: list[Path] = []
    similarity_csvs: list[Path] = []
    for prompt_id in gate.EXPECTED_PROMPT_IDS:
        root = tmp_path / "runtime" / f"p{prompt_id}"
        root.mkdir(parents=True)
        config = root / "config.json"
        _json(config, {"args": {**gate.EXPECTED_CONFIG}})
        log = root / "generation.log"
        _log(log)
        video = root / f"{prompt_id}-42.mp4"
        video.write_bytes(f"candidate-{prompt_id}".encode())
        candidate_videos[prompt_id] = video
        _json(
            root / "run_info.json",
            {
                "schema_version": gate.RUNTIME_RUN_INFO_SCHEMA,
                "git_commit": "fixture",
                "config_path": str(config.resolve()),
                "config_hash": _sha(config),
                "codebook_path": {"K": "/k", "V": "/v"},
                "codebook_hash": amended_hashes,
                "hostname": "fixture",
                "gpu_type": "fixture",
                "seed": 42,
                "prompt_list": [
                    {"prompt_id": prompt_id, "prompt": f"prompt {prompt_id}"}
                ],
                "timestamp": "2026-07-24T01:00:00Z",
                "completion_timestamp": "2026-07-24T01:01:00Z",
                "source_sha256": actual_prereg["source_sha256"],
                "preregistration_amendment_path": str(amendment.resolve()),
                "preregistration_amendment_hash": _sha(amendment),
                "generation_log": {
                    "path": str(log.resolve()),
                    "sha256": _sha(log),
                },
                "video_path": {
                    "path": str(video.resolve()),
                    "sha256": _sha(video),
                },
                "terminal_payload": {
                    "joint_including_shared_rvq_codebooks_MiB": (
                        gate.EXPECTED_JOINT_MIB
                    ),
                    "residual_MiB": gate.EXPECTED_RESIDUAL_MIB,
                },
            },
        )
        row = _row(
            prompt_id,
            scale=candidate_scale,
            kind="tensor_hybrid",
            reference=references[prompt_id],
            video=video,
        )
        sim_json, sim_csv = _write_similarity_run(tmp_path / "similarity", row)
        roots.append(root)
        similarity_jsons.append(sim_json)
        similarity_csvs.append(sim_csv)
    return {
        "preregistration": prereg,
        "amendment": amendment,
        "roots": roots,
        "similarity_jsons": similarity_jsons,
        "similarity_csvs": similarity_csvs,
        "output": tmp_path / "output",
    }


def _args(fixture: dict[str, object]) -> argparse.Namespace:
    return argparse.Namespace(
        preregistration=fixture["preregistration"],
        amendment=fixture["amendment"],
        candidate_run_root=fixture["roots"],
        candidate_similarity_json=fixture["similarity_jsons"],
        candidate_similarity_csv=fixture["similarity_csvs"],
        output_dir=fixture["output"],
    )


def _runtime_file(
    fixture: dict[str, object], prompt_index: int, name: str
) -> Path:
    return Path(fixture["roots"][prompt_index]) / name


def test_pass_writes_create_only_complete_report(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    assert gate.main(_args(fixture)) == 0
    output = Path(fixture["output"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["promotion_gate"]["overall_pass"]
    assert summary["promotion_gate"]["decision"] == "PROMOTE_10PROMPT"
    assert summary["runtime_gate"]["validated_before_candidate_quality_read"]
    assert all(
        comparison["composite_improvement_percent"] >= 10.0
        for comparison in summary["comparisons"].values()
    )
    expected = {
        "summary.json",
        "aggregate_comparison.csv",
        "prompt_comparison.csv",
        "runtime_validation.csv",
        "evidence_manifest.json",
        "README_zh.md",
        "run_info.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    before = {path.name: _sha(path) for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="create-only"):
        gate.main(_args(fixture))
    assert before == {path.name: _sha(path) for path in output.iterdir()}


def test_below_ten_percent_stops_without_vbench(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, candidate_scale=0.94)
    gate.main(_args(fixture))
    summary = json.loads(
        (Path(fixture["output"]) / "summary.json").read_text()
    )
    assert not summary["promotion_gate"]["overall_pass"]
    assert summary["promotion_gate"]["decision"] == "STOP_NO_VBENCH"
    assert not summary["promotion_gate"]["vbench_run_by_this_evaluator"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("config", "runtime_config.rvq_k_residual_codec mismatch"),
        ("source", "source_sha256 mismatch"),
        ("payload", "terminal joint payload"),
        ("amendment", "preregistration_amendment_hash mismatch"),
    ],
)
def test_runtime_tampering_rejected_before_output(
    tmp_path: Path, mutation: str, message: str
) -> None:
    fixture = _fixture(tmp_path)
    root = Path(fixture["roots"][0])
    run_info_path = root / "run_info.json"
    run_info = json.loads(run_info_path.read_text())
    if mutation == "config":
        config_path = root / "config.json"
        config = json.loads(config_path.read_text())
        config["args"]["rvq_k_residual_codec"] = "quantvideo"
        _json(config_path, config)
        run_info["config_hash"] = _sha(config_path)
    elif mutation == "source":
        run_info["source_sha256"]["quant_videogen/compress.py"] = "0" * 64
    elif mutation == "payload":
        run_info["terminal_payload"][
            "joint_including_shared_rvq_codebooks_MiB"
        ] += 0.01
    elif mutation == "amendment":
        run_info["preregistration_amendment_hash"] = "0" * 64
    else:
        raise AssertionError(mutation)
    _json(run_info_path, run_info)
    with pytest.raises(ValueError, match=message):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_effective_step_log_tamper_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    log = _runtime_file(fixture, 1, "generation.log")
    _log(log, noise=43)
    run_info_path = _runtime_file(fixture, 1, "run_info.json")
    run_info = json.loads(run_info_path.read_text())
    run_info["generation_log"]["sha256"] = _sha(log)
    _json(run_info_path, run_info)
    with pytest.raises(ValueError, match="noise_prediction_count mismatch"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_duplicate_runtime_prompt_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run_info_path = _runtime_file(fixture, 2, "run_info.json")
    run_info = json.loads(run_info_path.read_text())
    run_info["prompt_list"] = [{"prompt_id": 4, "prompt": "prompt 4"}]
    _json(run_info_path, run_info)
    with pytest.raises(ValueError, match="complete and unique"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_candidate_similarity_json_csv_tamper_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    similarity = Path(fixture["similarity_jsons"][0])
    payload = json.loads(similarity.read_text())
    payload["videos"][0]["mse"] *= 2.0
    _json(similarity, payload)
    with pytest.raises(ValueError, match="prompt 0.mse mismatch"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()


def test_runtime_failure_precedes_missing_candidate_quality(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    Path(fixture["similarity_jsons"][0]).unlink()
    run_info_path = _runtime_file(fixture, 0, "run_info.json")
    run_info = json.loads(run_info_path.read_text())
    run_info["terminal_payload"][
        "joint_including_shared_rvq_codebooks_MiB"
    ] += 1.0
    _json(run_info_path, run_info)
    with pytest.raises(ValueError, match="terminal joint payload"):
        gate.main(_args(fixture))
    assert not Path(fixture["output"]).exists()
