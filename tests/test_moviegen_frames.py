"""Exercise real first-frame decoding and frozen row matching without downloads."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from subitkv.pipeline.common import REPO
from subitkv.pipeline import moviegen_frames as frames


def test_hub_download_uses_pinned_revision_and_cached_lock(tmp_path, monkeypatch):
    calls = []
    class Api:
        def repo_info(self, **kwargs):
            calls.append(("info", kwargs))
            return SimpleNamespace(sha="a" * 40)
        def list_repo_files(self, **kwargs):
            calls.append(("list", kwargs))
            return [frames.PROMPTS, "data/test_with_generations-00000-of-00001.parquet", "README.md"]
    def download(**kwargs):
        calls.append(("download", kwargs))
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture parquet placeholder")
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api, hf_hub_download=download))
    first = frames.dataset_files(tmp_path, "main", False)
    second = frames.dataset_files(tmp_path, "main", False)
    assert first == second
    assert sum(name == "info" for name, _ in calls) == 1
    assert sum(name == "list" for name, _ in calls) == 1
    assert all(call["revision"] == "a" * 40 and call["repo_type"] == "dataset" for name, call in calls if name == "download")


def local_video_dataset(tmp_path, *, wrong_prompt=False):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg unavailable")
    video = tmp_path / "red-then-blue.mp4"
    pixels = bytes([255, 0, 0]) * 16 * 16 + bytes([0, 0, 255]) * 16 * 16
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "16x16", "-r", "1", "-i", "pipe:0", "-c:v", "mpeg4", str(video)],
                   input=pixels, check=True)
    prompts = (REPO / "assets/MovieGenVideoBench.txt").read_text().splitlines()
    root = tmp_path / "dataset"
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.table({"prompt": prompts}), root / frames.PROMPTS)
    actual = list(prompts)
    if wrong_prompt:
        actual[223] = "Wrong dataset row"
    pq.write_table(pa.table({"prompt": actual, "video": [video.read_bytes()] * 1003}),
                   root / "data/test_with_generations-00000-of-00001.parquet")
    return root, ffmpeg, video.read_bytes()


def test_extracts_actual_frame_zero_matches_rows_and_resumes(tmp_path, monkeypatch):
    root, ffmpeg, video = local_video_dataset(tmp_path)
    output = tmp_path / "images"
    result = frames.prepare(root, output, ["test", "validation"], [0], local_only=True, ffmpeg=ffmpeg)
    assert {r["dataset_row"] for r in result["records"]} == {223, 0}
    assert all(r["source_video_sha256"] == hashlib.sha256(video).hexdigest() for r in result["records"])
    image = output / "test/0000.png"
    raw = subprocess.check_output([ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(image),
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    assert sum(raw[0::3]) / 256 > 200 and sum(raw[2::3]) / 256 < 40
    monkeypatch.setattr(frames, "extract_first_frame", lambda *args: pytest.fail("Completed image must not be decoded again"))
    again = frames.prepare(root, output, ["test", "validation"], [0], local_only=True, ffmpeg=ffmpeg)
    assert again == result
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        frames.prepare(root, output, ["test"], [0], local_only=True, ffmpeg=ffmpeg)


def test_rejects_generated_video_prompt_mismatch(tmp_path):
    root, ffmpeg, _ = local_video_dataset(tmp_path, wrong_prompt=True)
    output = tmp_path / "images"
    with pytest.raises(ValueError, match="prompt mismatch"):
        frames.prepare(root, output, ["test"], [0], local_only=True, ffmpeg=ffmpeg)
    assert not list(output.glob("test/*.png"))
