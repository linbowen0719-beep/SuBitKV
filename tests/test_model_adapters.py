"""Check release adapters against their native parser surfaces without weights."""

import ast
import json
from pathlib import Path

import pytest

from subitkv.pipeline.common import MODELS, REPO, load_config
from subitkv.pipeline.workflow import Run


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("budget", [
    None, 0, 0.5, 4, "kivi_int2", "kivi_int4", "turboquant_int2",
    "turboquant_int4", "quantvideo_int2", "quantvideo_int4",
])
def test_adapter_launch_is_self_contained(tmp_path, monkeypatch, model, budget):
    from subitkv.pipeline import adapters

    config = load_config(REPO / "configs/subitkv.yaml", model, "smoke")
    config["runtime"]["root"] = str(tmp_path / "runs")
    config["data"]["conditioning_images"] = str(tmp_path / "images")
    image = tmp_path / "images/eval/0000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"input-identity-fixture")
    run = Run(config, "adapter", False)
    commands = []

    def process(argv, env, destination):
        commands.append(([str(v) for v in argv], env))
        video = destination / "sample.mp4"
        video.write_bytes(b"video-identity-fixture")
        return video, 0.1

    monkeypatch.setattr(adapters, "process", process)
    monkeypatch.setattr(adapters, "runtime_metrics", lambda *args: {})
    record = adapters.generate(
        run, 0, "A frog detective", "eval", run.root / "video", budget=budget
    )
    argv, env = commands[0]
    assert record["method"] == ("bf16" if budget is None else adapters.label(budget))
    assert all("scripts/eval" not in v and "/experiments/" not in v for v in argv)
    if model == "wan2_2_5b":
        job = json.loads(Path(argv[-1]).read_text())
        assert (job["quant_config"] is None) == (budget is None)
    else:
        native = next(Path(v) for v in argv if v.endswith(".py"))
        tree = ast.parse(native.read_text())
        supported = {
            arg.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }
        start = argv.index(str(native)) + 1
        flags = {v for v in argv[start:] if v.startswith("--")}
        assert flags <= supported
        assert native.is_relative_to(REPO / "subitkv/_vendor")
        if model == "hyworldplay_8b":
            codec = argv[argv.index("--kv_cache_codec") + 1]
            expected = (
                "none"
                if budget is None
                else budget if isinstance(budget, str)
                else "rvq_ours_"
                + adapters.label(budget).replace("avg0p2", "avg0p2_alt")
            )
            assert codec == expected


def test_baselines_need_no_calibration_assets():
    pytest.importorskip("torch")
    from subitkv.hyworldplay8b_cache_codec import make_codec_settings

    from subitkv.baselines import BASELINE_QUANT_TYPES

    for name, kind in BASELINE_QUANT_TYPES.items():
        settings = make_codec_settings(name)
        assert settings.quant_type == kind
        assert not settings.quant_config.k_rvq
        assert not settings.quant_config.temporal_adaptive_residual
    with pytest.raises(ValueError, match="unsupported"):
        make_codec_settings("rvq_full")
