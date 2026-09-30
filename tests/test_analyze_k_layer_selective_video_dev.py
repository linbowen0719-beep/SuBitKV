from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "analyze_k_layer_selective_video_dev.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_k_layer_selective_video_dev", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


LAYERS = {
    "A_mid10": list(range(8, 16)),
    "B_mid20": list(range(16, 24)),
    "C_mid10_mid20": list(range(8, 24)),
}


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol() -> dict[str, object]:
    return {
        "protocol_id": gate.SIMILARITY_PROTOCOL,
        "frame_0_included": False,
        "lpips": {"normalize_input": True},
    }


def _similarity_row(
    prompt_id: int,
    *,
    scale: float,
    video: Path,
    reference: Path,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "prompt": f"prompt {prompt_id}",
        "setting": "fixture",
        "reference_video": str(reference.resolve()),
        "test_video": str(video.resolve()),
        "frames": 173,
        "compared_frames": 172,
        "mse": (0.10 + prompt_id * 0.001) * scale,
        "psnr": 20.0,
        "ssim": 1.0 - (0.20 + prompt_id * 0.001) * scale,
        "lpips": (0.30 + prompt_id * 0.001) * scale,
    }


def _similarity_payload(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": gate.SIMILARITY_SCHEMA,
        "metric_protocol": _protocol(),
        "videos": rows,
    }


def _write_candidate_similarity(
    root: Path,
    candidate: str,
    prompt_id: int,
    scale: float,
    video: Path,
    reference: Path,
) -> Path:
    path = root / candidate / f"p{prompt_id}" / "similarity.json"
    row = _similarity_row(
        prompt_id,
        scale=scale,
        video=video,
        reference=reference,
    )
    _json(path, _similarity_payload([row]))
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    _json(
        path.parent / "run_info.json",
        {
            "output_json": str(path.resolve()),
            "output_json_sha256": _sha(path),
            "output_csv": str(csv_path.resolve()),
            "output_csv_sha256": _sha(csv_path),
        },
    )
    return path


def _write_runtime(
    root: Path,
    candidate: str,
    prompt_id: int,
    design: Path,
    *,
    layers: list[int] | None = None,
    noise_calls: int = 44,
) -> Path:
    run_root = root / candidate / f"p{prompt_id}"
    run_root.mkdir(parents=True)
    video = run_root / f"{prompt_id}-42.mp4"
    video.write_bytes(f"{candidate}-{prompt_id}".encode())
    config_path = run_root / "config.json"
    args = dict(gate.EXPECTED_CONFIG)
    args.update(
        {
            "prompt_start": prompt_id,
            "structured_residual_v_bits": 2,
            "rvq_k_structured_layers": ",".join(
                str(layer) for layer in (layers if layers is not None else LAYERS[candidate])
            ),
        }
    )
    _json(config_path, {"args": args})
    log_path = run_root / "generation.log"
    lines = ["Number of inference steps: 50"]
    lines.extend(["Calculating noise prediction"] * noise_calls)
    lines.extend(
        f"Generate time for chunk {chunk} is 1.0 seconds" for chunk in range(11)
    )
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    source_hashes = {"subitkv/compress.py": "a" * 64}
    _json(
        run_root / "run_info.json",
        {
            "git_commit": "fixture",
            "candidate_id": candidate,
            "structured_K_layers": LAYERS[candidate],
            "config_path": str(config_path.resolve()),
            "config_hash": _sha(config_path),
            "seed": 42,
            "prompt_list": [
                {"prompt_id": prompt_id, "prompt": f"prompt {prompt_id}"}
            ],
            "generation_log": {
                "path": str(log_path.resolve()),
                "sha256": _sha(log_path),
            },
            "video_path": {
                "path": str(video.resolve()),
                "sha256": _sha(video),
            },
            "terminal_payload": {
                "joint_including_shared_rvq_codebooks_MiB": gate.EXPECTED_JOINT_MIB,
                "residual_MiB": gate.EXPECTED_RESIDUAL_MIB,
            },
            "source_provenance": {
                "frozen_source_sha256": source_hashes,
                "frozen_source_manifest_sha256": "b" * 64,
            },
            "frozen_design": {
                "path": str(design.resolve()),
                "artifact_sha256": {
                    "protocol.json": _sha(design / "protocol.json"),
                    "candidate_registry.csv": _sha(
                        design / "candidate_registry.csv"
                    ),
                },
            },
            "runtime_amendment": {
                "path": str(
                    (design / "runtime_amendment_001.json").resolve()
                ),
                "sha256": _sha(design / "runtime_amendment_001.json"),
            },
        },
    )
    return run_root


def _write_design(root: Path, baseline_prereg: Path) -> Path:
    design = root / "design"
    design.mkdir()
    candidates = []
    registry_rows = []
    for candidate in gate.EXPECTED_CANDIDATES:
        structured = LAYERS[candidate]
        qv2 = sorted(set(range(30)) - set(structured))
        candidates.append(
            {
                "id": candidate,
                "structured_K_layers": structured,
                "qv2_K_layers": qv2,
            }
        )
        registry_rows.append(
            {
                "candidate_id": candidate,
                "structured_K_layers": ",".join(map(str, structured)),
                "qv2_K_layers": ",".join(map(str, qv2)),
                "residual_bits_per_scalar": 2,
            }
        )
    protocol = {
        "schema_version": gate.DESIGN_SCHEMA,
        "status": "DESIGN_ONLY_NO_CANDIDATE_RUNS",
        "candidates": candidates,
        "development": {
            "prompt_ids": list(gate.EXPECTED_PROMPTS),
            "comparators": [
                "rvq_quantvideo_int2",
                "uniform_structured_b2",
            ],
            "if_none_eligible": "STOP_NO_HELDOUT_NO_VBENCH",
        },
    }
    protocol_path = design / "protocol.json"
    _json(protocol_path, protocol)
    registry_path = design / "candidate_registry.csv"
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry_rows[0]))
        writer.writeheader()
        writer.writerows(registry_rows)
    _json(
        design / "evidence_manifest.json",
        {
            "schema_version": gate.EVIDENCE_SCHEMA,
            "design_artifacts": {
                "protocol.json": _sha(protocol_path),
                "candidate_registry.csv": _sha(registry_path),
            },
            "authoritative_sources": [
                {
                    "path": str(baseline_prereg.resolve()),
                    "sha256": _sha(baseline_prereg),
                    "use": "Existing baseline paths and hashes",
                }
            ],
        },
    )
    _json(
        design / "runtime_amendment_001.json",
        {
            "schema_version": (
                "k_layer_selective_structured_dev_design.runtime_amendment.v1"
            ),
            "status": "SOURCE_AND_RUNTIME_FREEZE_BEFORE_CANDIDATE_EXECUTION",
            "parent_design": {
                "protocol_sha256": _sha(protocol_path),
                "candidate_registry_sha256": _sha(registry_path),
            },
            "correction": {
                "source_sha256": {
                    "subitkv/compress.py": "a" * 64
                }
            },
        },
    )
    return design


def _fixture(
    tmp_path: Path,
    *,
    scales: dict[str, float] | None = None,
) -> tuple[argparse.Namespace, dict[tuple[str, int], Path]]:
    scales = scales or {candidate: 0.95 for candidate in gate.EXPECTED_CANDIDATES}
    references: dict[int, Path] = {}
    for prompt_id in range(10):
        reference = tmp_path / "references" / f"{prompt_id}.mp4"
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(f"reference-{prompt_id}".encode())
        references[prompt_id] = reference

    comparator_records = {}
    for comparator in ("rvq_quantvideo_int2", "uniform_structured_b2"):
        rows = []
        for prompt_id in range(10):
            video = tmp_path / "baseline_videos" / comparator / f"{prompt_id}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"baseline")
            rows.append(
                _similarity_row(
                    prompt_id,
                    scale=1.0,
                    video=video,
                    reference=references[prompt_id],
                )
            )
        path = tmp_path / "baselines" / f"{comparator}.json"
        _json(path, _similarity_payload(rows))
        comparator_records[comparator] = {
            "similarity_json": str(path.resolve()),
            "sha256": _sha(path),
        }
    prereg = tmp_path / "baseline_preregistration.json"
    _json(prereg, {"comparators": comparator_records})
    design = _write_design(tmp_path, prereg)

    runtimes: list[str] = []
    similarities: list[str] = []
    runtime_by_key: dict[tuple[str, int], Path] = {}
    for candidate in gate.EXPECTED_CANDIDATES:
        for prompt_id in gate.EXPECTED_PROMPTS:
            runtime = _write_runtime(
                tmp_path / "runtime", candidate, prompt_id, design
            )
            runtime_by_key[(candidate, prompt_id)] = runtime
            video = runtime / f"{prompt_id}-42.mp4"
            similarity = _write_candidate_similarity(
                tmp_path / "similarity",
                candidate,
                prompt_id,
                scales[candidate],
                video,
                references[prompt_id],
            )
            runtimes.append(f"{candidate}={runtime}")
            similarities.append(f"{candidate}={similarity}")
    return (
        argparse.Namespace(
            design_dir=design,
            runtime=runtimes,
            similarity=similarities,
            output_dir=tmp_path / "evaluation",
        ),
        runtime_by_key,
    )


def test_no_candidate_stops_before_heldout_or_vbench(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    assert gate.main(args) == 0
    summary = json.loads(
        (args.output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["decision"] == "STOP_NO_HELDOUT_NO_VBENCH"
    assert summary["winner"] is None
    assert summary["runtime_audit_count"] == 9
    assert summary["similarity_count"] == 9


def test_winner_uses_frozen_maximin_order(tmp_path: Path) -> None:
    args, _ = _fixture(
        tmp_path,
        scales={
            "A_mid10": 0.80,
            "B_mid20": 0.75,
            "C_mid10_mid20": 0.70,
        },
    )
    gate.main(args)
    summary = json.loads(
        (args.output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["decision"] == "PROMOTE_HELDOUT_P158"
    assert summary["winner"] == "C_mid10_mid20"
    assert summary["ranking"][0]["eligible"] is True


def test_runtime_failure_prevents_any_candidate_quality_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, runtime_by_key = _fixture(tmp_path)
    bad_root = runtime_by_key[("C_mid10_mid20", 9)]
    log = bad_root / "generation.log"
    log.write_text(
        log.read_text(encoding="utf-8").replace(
            "Calculating noise prediction\n", "", 1
        ),
        encoding="utf-8",
    )
    run_info_path = bad_root / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["generation_log"]["sha256"] = _sha(log)
    _json(run_info_path, run_info)

    def forbidden(_path: Path) -> tuple[dict[str, object], dict[str, object]]:
        raise AssertionError("quality was opened before full runtime audit")

    monkeypatch.setattr(gate, "_read_candidate_similarity", forbidden)
    with pytest.raises(ValueError, match="noise calls"):
        gate.main(args)
    assert not args.output_dir.exists()


def test_wrong_candidate_layer_set_fails_closed(tmp_path: Path) -> None:
    args, runtime_by_key = _fixture(tmp_path)
    root = runtime_by_key[("A_mid10", 0)]
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["args"]["rvq_k_structured_layers"] = "8-14"
    _json(config_path, config)
    run_info_path = root / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["config_hash"] = _sha(config_path)
    _json(run_info_path, run_info)
    with pytest.raises(ValueError, match="layer set mismatch"):
        gate.main(args)
    assert not args.output_dir.exists()


def test_incomplete_runtime_grid_fails_before_quality(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    args.runtime.pop()
    with pytest.raises(ValueError, match="expected 9 entries"):
        gate.main(args)
    assert not args.output_dir.exists()


def test_output_is_create_only(tmp_path: Path) -> None:
    args, _ = _fixture(tmp_path)
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError, match="create-only"):
        gate.main(args)
