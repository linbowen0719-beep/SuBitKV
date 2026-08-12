from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
import torch

from scripts.eval import compare_video_quality


class RecordingLpips:
    version = "0.1"

    def __init__(self) -> None:
        self.normalize: bool | None = None

    def __call__(
        self,
        reference: torch.Tensor,
        test: torch.Tensor,
        *,
        normalize: bool,
    ) -> torch.Tensor:
        self.normalize = normalize
        return torch.mean((reference - test) ** 2)


def test_lpips_normalization_flag_is_forwarded_explicitly() -> None:
    model = RecordingLpips()
    reference = torch.zeros(1, 3, 2, 2)
    test = torch.ones(1, 3, 2, 2)

    value = compare_video_quality.calculate_lpips(
        model,
        reference,
        test,
        normalize_input=True,
    )

    assert value.item() == 1.0
    assert model.normalize is True


def test_metric_protocol_distinguishes_strict_and_legacy_lpips() -> None:
    model = RecordingLpips()
    strict = compare_video_quality.metric_protocol(
        model,
        include_first_frame=False,
        lpips_normalize_input=True,
    )
    legacy = compare_video_quality.metric_protocol(
        model,
        include_first_frame=False,
        lpips_normalize_input=False,
    )

    assert (
        strict["protocol_id"]
        == compare_video_quality.NORMALIZED_LPIPS_PROTOCOL_ID
    )
    assert strict["lpips"]["normalize_input"] is True
    assert strict["lpips"]["effective_input_range"] == [-1.0, 1.0]
    assert (
        legacy["protocol_id"]
        == compare_video_quality.LEGACY_LPIPS_PROTOCOL_ID
    )
    assert legacy["lpips"]["normalize_input"] is False
    assert legacy["lpips"]["effective_input_range"] == [0.0, 1.0]


class FakeLpipsModule(torch.nn.Module):
    version = "test"

    def forward(
        self,
        reference: torch.Tensor,
        test: torch.Tensor,
        *,
        normalize: bool,
    ) -> torch.Tensor:
        del normalize
        return torch.mean((reference - test) ** 2)


def _write_metrics_csv(path: Path) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["prompt_id", "seed", "prompt", "video_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "prompt_id": 4,
                "seed": 42,
                "prompt": "a deterministic test prompt",
                "video_path": "candidate/4-42.mp4",
            }
        )


def _fake_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        reference_dir=str(tmp_path / "reference"),
        test_dir=str(tmp_path / "candidate"),
        metrics_csv=str(tmp_path / "metrics.csv"),
        setting="candidate",
        output_json=str(tmp_path / "similarity" / "quality.json"),
        output_csv=str(tmp_path / "similarity" / "quality.csv"),
        run_info_json=str(tmp_path / "similarity" / "run_info.json"),
        device="cpu",
        include_first_frame=False,
        lpips_normalize_input=True,
    )


def test_main_writes_reproducible_create_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _fake_args(tmp_path)
    _write_metrics_csv(Path(args.metrics_csv))
    monkeypatch.setattr(compare_video_quality, "parse_args", lambda: args)
    monkeypatch.setattr(
        compare_video_quality.lpips,
        "LPIPS",
        lambda *, net: FakeLpipsModule(),
    )
    monkeypatch.setattr(
        compare_video_quality,
        "compare_pair",
        lambda *positional, **keyword: {
            "frames": 173,
            "compared_frames": 172,
            "mse": 0.125,
            "psnr": 9.0309,
            "ssim": 0.75,
            "lpips": 0.25,
        },
    )
    monkeypatch.setattr(
        compare_video_quality,
        "git_commit",
        lambda: "a" * 40,
    )

    compare_video_quality.main()

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    run_info_json = Path(args.run_info_json)
    config_json = run_info_json.parent / "experiment_config.json"
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    config = json.loads(config_json.read_text(encoding="utf-8"))
    run_info = json.loads(run_info_json.read_text(encoding="utf-8"))

    assert summary["schema_version"] == "frame_aligned_video_similarity.v2"
    assert summary["metric_protocol"]["protocol_id"] == (
        compare_video_quality.NORMALIZED_LPIPS_PROTOCOL_ID
    )
    assert summary["videos"][0]["mse"] == pytest.approx(0.125)
    assert output_csv.is_file()

    assert config["schema_version"] == "video_similarity.config.v1"
    assert config["metrics_csv_sha256"] == compare_video_quality.sha256_file(
        Path(args.metrics_csv)
    )
    assert config["evaluator_sha256"] == compare_video_quality.sha256_file(
        Path(compare_video_quality.__file__).resolve()
    )
    assert run_info["schema_version"] == "video_similarity.run_info.v1"
    assert run_info["git_commit"] == "a" * 40
    assert run_info["config_path"] == str(config_json.resolve())
    assert run_info["config_hash"] == compare_video_quality.sha256_file(
        config_json
    )
    assert run_info["codebook_path"] is None
    assert run_info["codebook_hash"] is None
    assert run_info["seed"] == 42
    assert run_info["prompt_list"] == [
        {
            "prompt_id": 4,
            "prompt": "a deterministic test prompt",
        }
    ]
    assert run_info["output_json_sha256"] == (
        compare_video_quality.sha256_file(output_json)
    )
    assert run_info["output_csv_sha256"] == (
        compare_video_quality.sha256_file(output_csv)
    )


def test_create_only_preflight_runs_before_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    args = _fake_args(tmp_path)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True)
    output_json.write_text("existing\n", encoding="utf-8")
    monkeypatch.setattr(compare_video_quality, "parse_args", lambda: args)

    def fail_if_initialized(*args: object, **kwargs: object) -> None:
        raise AssertionError("LPIPS must not initialize after preflight failure")

    monkeypatch.setattr(
        compare_video_quality.lpips,
        "LPIPS",
        fail_if_initialized,
    )

    with pytest.raises(FileExistsError, match="create-only"):
        compare_video_quality.main()


def test_create_only_preflight_rejects_path_aliases(tmp_path: Path) -> None:
    target = tmp_path / "outputs" / "quality.json"
    alias = tmp_path / "outputs" / ".." / "outputs" / "quality.json"

    with pytest.raises(ValueError, match="must be distinct"):
        compare_video_quality._preflight_create_only_outputs([target, alias])
