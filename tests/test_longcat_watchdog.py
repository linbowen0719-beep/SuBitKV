from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path
import time

from scripts.LongCat.watch_chunk5_experiment import (
    completed,
    foreign_compute_pids,
    inherited_process_stalled,
    is_process_descendant,
    parse_running_video_job,
    RunningVideoJob,
    vbench_input_dir,
    vbench_prompt_json,
)


def args_for(tmp_path):
    return SimpleNamespace(output_dir=tmp_path, seed=123)


def write_completed(tmp_path, *, method="qvg_int4", prompt_index=2, chunks=3):
    directory = tmp_path / method / f"prompt_{prompt_index:04d}_seed_123"
    directory.mkdir(parents=True)
    total_frames = 17 + 20 * chunks
    (directory / "status.json").write_text(
        json.dumps({"status": "complete", "chunks": chunks, "total_frames": total_frames}),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "method": method,
                "prompt_index": prompt_index,
                "num_continuation_chunks": chunks,
                "total_frames": total_frames,
            }
        ),
        encoding="utf-8",
    )
    (directory / "video.mp4").write_bytes(b"video")
    return directory


def test_completed_accepts_consistent_artifacts(tmp_path):
    write_completed(tmp_path)
    assert completed(args_for(tmp_path), "qvg_int4", 2)


def test_completed_rejects_inconsistent_frame_count(tmp_path):
    directory = write_completed(tmp_path)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    metrics["total_frames"] += 1
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    assert not completed(args_for(tmp_path), "qvg_int4", 2)


def test_completed_rejects_empty_video(tmp_path):
    directory = write_completed(tmp_path)
    (directory / "video.mp4").write_bytes(b"")
    assert not completed(args_for(tmp_path), "qvg_int4", 2)


def test_is_process_descendant_follows_parent_chain():
    parents = {300: 200, 200: 100, 100: 1}

    assert is_process_descendant(300, 100, parents.get)
    assert is_process_descendant(100, 100, parents.get)
    assert not is_process_descendant(300, 101, parents.get)


def test_foreign_compute_pids_excludes_job_process_tree():
    parents = {301: 300, 300: 100, 900: 1}
    inventory = {1: {"compute_pids": {100, 301, 900}}}

    assert foreign_compute_pids(inventory, 1, 100, parents.get) == {900}


def test_parse_running_video_job_reserves_offloaded_runner_gpu(tmp_path):
    command = [
        "/env/bin/python",
        str((Path(__file__).parents[1] / "scripts/LongCat/run_chunk5_experiment.py").resolve()),
        "--output-dir",
        str(tmp_path),
        "--method",
        "qvg_int4",
        "--prompt-index",
        "3",
        "--seed",
        "123",
    ]

    job = parse_running_video_job(500, command, {"CUDA_VISIBLE_DEVICES": "2"})

    assert job is not None
    assert job.name == "video:qvg_int4:prompt0003"
    assert job.gpu == 2
    assert job.output_dir == tmp_path.resolve()


def test_inherited_process_stalled_uses_latest_attempt_log(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    old_log = logs / "video_qvg_int4_prompt0003_attempt1_gpu2.log"
    current_log = logs / "video_qvg_int4_prompt0003_attempt1_gpu2_retry.log"
    old_log.write_text("old", encoding="utf-8")
    current_log.write_text("current", encoding="utf-8")
    os.utime(old_log, (time.time() - 100, time.time() - 100))
    running = RunningVideoJob(500, 2, tmp_path, "qvg_int4", 3, 123)
    args = SimpleNamespace(output_dir=tmp_path, stall_seconds=30)

    assert not inherited_process_stalled(args, running)


def test_vbench_inputs_flatten_prompt_video_directories(tmp_path):
    args = SimpleNamespace(output_dir=tmp_path, seed=123, prompt_start=0, num_prompts=2)
    for prompt_index in range(2):
        directory = tmp_path / "qvg_int4" / f"prompt_{prompt_index:04d}_seed_123"
        directory.mkdir(parents=True)
        (directory / "video.mp4").write_bytes(b"video")
        (directory / "run_info.json").write_text(
            json.dumps({"prompt_list": [f"prompt {prompt_index}"]}), encoding="utf-8"
        )

    input_dir = vbench_input_dir(args, "qvg_int4")
    prompt_map = json.loads(vbench_prompt_json(args, "qvg_int4").read_text(encoding="utf-8"))

    assert sorted(path.name for path in input_dir.glob("*.mp4")) == [
        "prompt_0000.mp4",
        "prompt_0001.mp4",
    ]
    assert all(path.is_symlink() for path in input_dir.glob("*.mp4"))
    assert prompt_map == {"prompt_0000.mp4": "prompt 0", "prompt_0001.mp4": "prompt 1"}
