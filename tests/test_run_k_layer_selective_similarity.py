from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_k_layer_selective_similarity.py"
SPEC = importlib.util.spec_from_file_location(
    "run_k_layer_selective_similarity", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
similarity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(similarity)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_runtime_grid(root: Path) -> list[tuple[str, Path]]:
    mappings: list[tuple[str, Path]] = []
    for candidate in similarity.EXPECTED_CANDIDATES:
        for prompt_id in similarity.EXPECTED_PROMPTS:
            unit = root / f"k_layer_{candidate}_p{prompt_id}"
            unit.mkdir(parents=True)
            config = unit / "config.json"
            metrics = unit / "metrics.csv"
            log = unit / "generation.log"
            video = unit / f"{prompt_id}-42.mp4"
            _json(config, {"fixture": True})
            metrics.write_text(
                "prompt_id,seed,video_path\n"
                f"{prompt_id},42,{video}\n",
                encoding="utf-8",
            )
            log.write_text("fixture log\n", encoding="utf-8")
            video.write_bytes(f"{candidate}-{prompt_id}".encode())
            outputs = {
                "config": {"path": str(config.resolve()), "sha256": _sha(config)},
                "metrics": {
                    "path": str(metrics.resolve()),
                    "sha256": _sha(metrics),
                },
                "generation_log": {
                    "path": str(log.resolve()),
                    "sha256": _sha(log),
                },
                "video": {"path": str(video.resolve()), "sha256": _sha(video)},
            }
            run_info = {
                "schema_version": similarity.GENERATION_RUN_INFO_SCHEMA,
                "candidate_id": candidate,
                "seed": 42,
                "prompt_list": [
                    {
                        "prompt_id": prompt_id,
                        "prompt": f"prompt {prompt_id}",
                    }
                ],
                "config_path": str(config.resolve()),
                "config_hash": _sha(config),
                "metrics": outputs["metrics"],
                "generation_log": outputs["generation_log"],
                "video_path": outputs["video"],
                "codebook_path": {"K": "/fixture/k", "V": "/fixture/v"},
                "codebook_hash": {"K": "a" * 64, "V": "b" * 64},
            }
            gate = {
                "schema_version": similarity.GENERATION_AUDIT_SCHEMA,
                "status": "PASS",
                "candidate_id": candidate,
                "prompt_id": prompt_id,
                "outputs": outputs,
            }
            _json(unit / "run_info.json", run_info)
            _json(unit / "gate_audit.json", gate)
            mappings.append((candidate, unit))
    return mappings


def _args(tmp_path: Path, runtime_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        execute=True,
        audit_only=False,
        candidate="A_mid10",
        prompt_id=0,
        gpu=0,
        device="cpu",
        runtime_root=runtime_root,
        runtime=None,
        design_dir=tmp_path / "design",
        output_root=tmp_path / "similarity",
    )


def _fake_full_audit(
    mappings: list[tuple[str, Path]],
    _design: dict[str, Any],
    metadata_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del mappings
    return [
        {
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "video_path": row["video_path"],
            "video_sha256": row["declared_video_sha256"],
        }
        for row in metadata_rows
    ]


def test_incomplete_grid_fails_before_any_candidate_video_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = _write_runtime_grid(tmp_path / "runtime")
    (mappings[-1][1] / "run_info.json").unlink()
    original = similarity.sha256_file

    def guarded(path: Path) -> str:
        if path.suffix == ".mp4":
            raise AssertionError("candidate video was opened before full gate")
        return original(path)

    monkeypatch.setattr(similarity, "sha256_file", guarded)
    with pytest.raises(RuntimeError, match="runtime gate incomplete"):
        similarity.metadata_runtime_gate(mappings)


def test_nonpass_audit_fails_before_any_candidate_video_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mappings = _write_runtime_grid(tmp_path / "runtime")
    bad = mappings[-1][1] / "gate_audit.json"
    payload = json.loads(bad.read_text(encoding="utf-8"))
    payload["status"] = "FAIL"
    _json(bad, payload)
    original = similarity.sha256_file

    def guarded(path: Path) -> str:
        if path.suffix == ".mp4":
            raise AssertionError("candidate video was opened before full gate")
        return original(path)

    monkeypatch.setattr(similarity, "sha256_file", guarded)
    with pytest.raises(RuntimeError, match="not PASS"):
        similarity.metadata_runtime_gate(mappings)


def test_metadata_gate_never_hashes_video(tmp_path: Path) -> None:
    mappings = _write_runtime_grid(tmp_path / "runtime")
    original = similarity.sha256_file
    opened: list[Path] = []

    def guarded(path: Path) -> str:
        opened.append(path)
        if path.suffix == ".mp4":
            raise AssertionError("metadata gate opened candidate video")
        return original(path)

    similarity.sha256_file = guarded
    try:
        rows = similarity.metadata_runtime_gate(mappings)
    finally:
        similarity.sha256_file = original
    assert len(rows) == 9
    assert not any(path.suffix == ".mp4" for path in opened)
    assert {row["gate_status"] for row in rows} == {"PASS"}


def test_execute_publishes_evaluator_compatible_hashed_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    _write_runtime_grid(runtime_root)
    args = _args(tmp_path, runtime_root)
    protocol_path = tmp_path / "design" / "protocol.json"
    _json(protocol_path, {"fixture": True})
    design = {
        "protocol_path": protocol_path,
        "candidates": {},
        "amendment_path": tmp_path / "amendment.json",
        "amendment_sha256": "a" * 64,
        "frozen_source_sha256": {},
        "baseline_prereg": {},
    }
    reference = tmp_path / "reference.mp4"
    reference.write_bytes(b"reference")

    monkeypatch.setattr(
        similarity.dev_gate, "_validate_design", lambda _path: design
    )
    monkeypatch.setattr(similarity, "full_runtime_audit", _fake_full_audit)
    monkeypatch.setattr(
        similarity,
        "resolve_reference",
        lambda _design, _prompt_id: {
            "path": str(reference.resolve()),
            "sha256": _sha(reference),
            "bytes": reference.stat().st_size,
            "confirmed_by_comparators": [
                "rvq_quantvideo_int2",
                "uniform_structured_b2",
            ],
        },
    )
    monkeypatch.setattr(
        similarity,
        "evaluate_video",
        lambda *_args, **_kwargs: (
            {
                "frames": 173,
                "compared_frames": 172,
                "mse": 0.012,
                "psnr": 23.0,
                "ssim": 0.75,
                "lpips": 0.21,
            },
            {
                "protocol_id": similarity.SIMILARITY_PROTOCOL,
                "frame_0_included": False,
                "lpips": {"normalize_input": True},
            },
            "cpu",
            "CPU",
        ),
    )
    monkeypatch.setattr(similarity, "_git_commit", lambda: "fixture-commit")

    assert similarity.main(args) == 0
    output = (
        args.output_root
        / "A_mid10"
        / "p0"
        / "similarity.json"
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    row = payload["videos"][0]
    assert row["mse"] == pytest.approx(0.012)
    assert row["psnr"] == pytest.approx(23.0)
    assert row["ssim"] == pytest.approx(0.75)
    assert row["lpips"] == pytest.approx(0.21)
    assert row["normalized_lpips_v2"] == pytest.approx(0.21)
    assert row["reference_video_sha256"] == _sha(reference)
    assert row["test_video_sha256"] == _sha(
        runtime_root / "k_layer_A_mid10_p0" / "0-42.mp4"
    )
    assert payload["runtime_gate"]["metadata_pass_count"] == 9

    # This is the same reader used by the strict final gate's
    # repeated ``--similarity CANDIDATE=JSON`` interface.
    typed, _evidence = similarity.dev_gate._read_candidate_similarity(output)
    assert typed["prompt_id"] == 0
    assert typed["lpips"] == pytest.approx(0.21)

    with pytest.raises(FileExistsError, match="create-only"):
        similarity.main(args)


def test_frozen_reference_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "0-42.mp4"
    reference.write_bytes(b"frozen-reference")
    comparator_records: dict[str, dict[str, str]] = {}
    for comparator in ("rvq_quantvideo_int2", "uniform_structured_b2"):
        path = tmp_path / f"{comparator}.json"
        _json(
            path,
            {
                "schema_version": similarity.SIMILARITY_SCHEMA,
                "metric_protocol": {
                    "protocol_id": similarity.SIMILARITY_PROTOCOL,
                    "frame_0_included": False,
                    "lpips": {"normalize_input": True},
                },
                "videos": [
                    {
                        "prompt_id": 0,
                        "reference_video": str(reference.resolve()),
                    }
                ],
            },
        )
        comparator_records[comparator] = {
            "similarity_json": str(path.resolve())
        }
    design = {"baseline_prereg": {"comparators": comparator_records}}
    monkeypatch.setitem(
        similarity.BF16_REFERENCES,
        0,
        {
            "path": reference,
            "sha256": _sha(reference),
            "bytes": reference.stat().st_size,
        },
    )
    resolved = similarity.resolve_reference(design, 0)
    assert resolved["sha256"] == _sha(reference)

    reference.write_bytes(b"tampered-reference")
    with pytest.raises(ValueError, match="size changed|hash changed"):
        similarity.resolve_reference(design, 0)
