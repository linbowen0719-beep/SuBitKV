"""Wan2.2 launch adapter and single-video inference worker.

The workflow writes one job; the worker loads the upstream model in an isolated
process so model imports and GPU allocations do not leak between stages.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import sys
import time
from pathlib import Path

from subitkv.pipeline.common import REPO, write_json

DEFAULT_NEGATIVE_PROMPT = (
    "\u8272\u8c03\u8273\u4e3d,\u8fc7\u66dd,\u9759\u6001,"
    "\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05,\u5b57\u5e55,\u98ce\u683c,"
    "\u4f5c\u54c1,\u753b\u4f5c,\u753b\u9762,\u9759\u6b62,"
    "\u6574\u4f53\u53d1\u7070,\u6700\u5dee\u8d28\u91cf,\u4f4e\u8d28\u91cf,"
    "JPEG\u538b\u7f29\u6b8b\u7559,\u4e11\u964b\u7684,\u6b8b\u7f3a\u7684,"
    "\u591a\u4f59\u7684\u624b\u6307,\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8,"
    "\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8,\u7578\u5f62\u7684,"
    "\u6bc1\u5bb9\u7684,\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53,"
    "\u624b\u6307\u878d\u5408,\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762,"
    "\u6742\u4e71\u7684\u80cc\u666f,\u4e09\u6761\u817f,"
    "\u80cc\u666f\u4eba\u5f88\u591a,\u5012\u7740\u8d70"
)


def build_command(
    run,
    prompt_id,
    prompt,
    split,
    destination,
    prompt_file,
    env,
    *,
    budget=None,
    capture=None,
):
    from subitkv.pipeline.adapters import quant_config

    c, m = run.config, run.config["model_config"]
    job = dict(
        model=m,
        prompt=prompt,
        destination=str(destination),
        base=str(run.models / m["base"]),
        transformer=str(run.models / m["transformer"]),
        checkpoint=str(run.models / m["checkpoint"]),
        quant_config=quant_config(c, run.bundle, budget)
        if budget is not None
        else None,
        levels=c["calibration"]["levels"],
    )
    job_path = destination / "job.json"
    write_json(job_path, job)
    env["HYWORLDPLAY_KV_CACHE_MODE"] = "pre_kv_reconstruct"
    env["HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD"] = "1" if budget is None else "0"
    if capture:
        env.update(
            HYWORLDPLAY_RVQ_DIAG_DUMP="1",
            HYWORLDPLAY_RVQ_DIAG_DUMP_DIR=str(capture),
            HYWORLDPLAY_RVQ_DIAG_LAYERS=",".join(map(str, range(m["layers"]))),
            HYWORLDPLAY_RVQ_DIAG_POST_CHUNK_CACHE="1",
            HYWORLDPLAY_RVQ_DIAG_SPLIT="train",
            HYWORLDPLAY_RVQ_DIAG_PROMPT_ID=str(prompt_id),
            HYWORLDPLAY_RVQ_DIAG_PROMPT=prompt,
        )
    return [sys.executable, "-m", "subitkv.models.wan", "--job", job_path], env


def run_job(job):
    import torch
    from diffusers.utils import export_to_video

    from .wan_accounting import cache_metrics_from_runner, parse_logged_metrics

    sys.path.insert(0, str(REPO / "subitkv/_vendor/HY-WorldPlay/wan"))
    sys.path.insert(0, str(REPO / "subitkv/_vendor/HY-WorldPlay"))
    os.environ.setdefault("HYWORLDPLAY_OFFLOAD_FOR_DECODE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ["HYWORLDPLAY_KV_CACHE_MODE"] = "pre_kv_reconstruct"
    os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = str(job["levels"])
    from wan.generate import WanRunner

    m = job["model"]
    output = Path(job["destination"])
    output.mkdir(parents=True, exist_ok=True)
    runner = WanRunner(
        model_id=job["base"],
        ckpt_path=job["checkpoint"],
        ar_model_path=job["transformer"],
        offload_text_encoder=True,
    )
    inputs = dict(
        prompt=job["prompt"],
        negative_prompt=m.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
        num_frames=m["frames"],
        num_inference_steps=m["steps"],
        guidance_scale=1,
        height=m.get("height", 704),
        width=m.get("width", 1280),
        image_path=None,
        use_memory=True,
        context_window_length=m["memory_frames"],
        memory_frames=m["memory_frames"],
        temporal_context_size=m["temporal_context_size"],
        pred_latent_size=m["pred_latent_size"],
        offload_text_encoder=True,
        quant_config=job["quant_config"],
        kv_quant_factor=1,
        seed=m["seed"],
        pose=m["pose"],
        num_chunk=m["chunks"],
        skip_decode=False,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    from subitkv.memory import reset_workspace_stats, workspace_stats

    reset_workspace_stats()
    log = io.StringIO()
    started = time.perf_counter()
    try:
        with (
            contextlib.redirect_stdout(Tee(sys.stdout, log)),
            contextlib.redirect_stderr(Tee(sys.stderr, log)),
        ):
            result = runner.predict(inputs)
        if result["video"] is None:
            raise RuntimeError("Wan did not produce a decoded video")
        video = output / f"0-{m['seed']}.mp4"
        export_to_video(result["video"][0], str(video), fps=m.get("fps", 16))
        torch.cuda.synchronize()
        row = dict(
            e2e_time_s=time.perf_counter() - started,
            peak_gpu_mem_mb=torch.cuda.max_memory_allocated() / 1024**2,
            video_path=str(video),
        )
        row.update(parse_logged_metrics(log.getvalue()))
        row.update(cache_metrics_from_runner(runner))
        row.update(workspace_stats())
        row["peak_gpu_mem_mb"] = max(row["peak_gpu_mem_mb"], row["codec_process_peak_mib"])
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        audit = getattr(runner.pipe, "_temporal_adaptive_audit", None)
        if audit is not None:
            write_json(output / "cache_audit.json", audit)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class Tee(io.TextIOBase):
    def __init__(self, stream, capture):
        self.stream, self.capture = stream, capture

    def write(self, text):
        self.stream.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    run_job(json.loads(args.job.read_text()))


if __name__ == "__main__":
    main()
