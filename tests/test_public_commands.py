"""Offline calibration, one-video inference and table generation stay separate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from subitkv.pipeline.common import REPO, digest, load_config, write_json


@pytest.mark.parametrize("script,extra,expected", [
    ("calibrate.py", [], "offline"),
    ("generate_table.py", [], "table"),
    ("infer.py", ["--prompt", "A frog detective"], "single"),
    ("prepare_moviegen_frames.py", [], "frames"),
])
def test_public_command_dry_run_is_read_only(tmp_path, script, extra, expected):
    result = subprocess.run([sys.executable, str(REPO / "scripts" / script), *extra],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["mode"] == "dry-run"
    assert not list(tmp_path.iterdir())
    if expected == "offline":
        assert [c[0] for c in plan["commands"]] == ["download-models", "collect", "calibrate"]
    elif expected == "table":
        assert {c[0] for c in plan["commands"]} == {"infer", "metrics"}
        assert not plan["fresh_calibration"]
    elif expected == "single":
        assert plan["videos"] == 1 and not plan["performs_calibration"]
    else:
        assert plan["split_counts"] == {"calibration": 20, "test": 10, "validation": 973}


@pytest.mark.parametrize("method,selected", [("subitkv", 0.5), ("bf16", None), ("kivi_int2", "kivi_int2")])
def test_single_generates_exactly_one_video_and_resumes(tmp_path, monkeypatch, method, selected):
    from subitkv.pipeline import single

    calls, validations = [], []
    monkeypatch.setattr(single, "validate_calibration", lambda run, need_bundle: validations.append(need_bundle))
    monkeypatch.setattr(single, "require_runtime", lambda: None)
    def generate(run, index, prompt, split, output, *, budget):
        calls.append((index, prompt, split, budget, run.config["model_config"]["seed"]))
        assert "eval_set" not in run.config["workload"]
        path = output / "sample.mp4"
        path.write_bytes(b"one generated video")
        record = dict(video=str(path), video_sha256=digest(path))
        write_json(output / "record.json", record)
        return record
    monkeypatch.setattr(single.adapters, "generate", generate)
    argv = ["--prompt", "A frog detective", "--method", method, "--seed", "123",
            "--output-dir", str(tmp_path / "video"), "--execute"]
    single.main(argv)
    single.main(argv)
    assert calls == [(0, "A frog detective", "custom", selected, 123)]
    assert validations == [method == "subitkv"] * 2
    assert len(list((tmp_path / "video").rglob("*.mp4"))) == 1


def test_custom_hy_image_is_passed_to_native_model(tmp_path, monkeypatch):
    from subitkv.pipeline import adapters
    from subitkv.pipeline.workflow import Run

    image = tmp_path / "custom.png"
    image.write_bytes(b"conditioning identity")
    cfg = load_config(REPO / "configs/subitkv.yaml", "hyworldplay_8b", "full")
    cfg["runtime"]["root"] = str(tmp_path / "runs")
    cfg["data"]["single_conditioning_image"] = str(image)
    cfg["workload"].pop("eval_set")
    run = Run(cfg, "single", True)
    commands = []
    def process(argv, env, output):
        commands.append([str(x) for x in argv])
        video = output / "sample.mp4"
        video.write_bytes(b"fixture video")
        return video, 1.0
    monkeypatch.setattr(adapters, "process", process)
    monkeypatch.setattr(adapters, "runtime_metrics", lambda *args: {})
    record = adapters.generate(run, 0, "custom prompt", "custom", run.root / "video", budget=0.5)
    command = commands[0]
    assert command[command.index("--image_path") + 1] == str(image)
    assert "dataset_row" not in record


def test_hy_calibration_requires_prepared_images_before_download(monkeypatch):
    from subitkv.pipeline import experiment

    monkeypatch.setattr(experiment, "run_stage", lambda *args: pytest.fail("Must not download models before image preflight"))
    with pytest.raises(SystemExit) as error:
        experiment.offline_main(["--model", "hyworldplay_8b", "--execute"])
    assert error.value.code == 2
