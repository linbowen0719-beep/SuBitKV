#!/usr/bin/env python3
"""Run L8C64 INT4 frame-reserve ablations.

Settings:
- RVQ-L8C64 INT4 per-chunk, first 1 + last 1 latent frame kept BF16
- RVQ-L8C64 INT4 per-chunk, first 2 + last 2 latent frames kept BF16
- RVQ-L8C64 INT4 per-chunk, first 3 + last 3 latent frames kept BF16
- With --reserve-mode recent_frames, replace static last-frame retention with
  dynamic recent-frame retention at every online quantization step.

Outputs:
- videos/rvq_l8c64_int4_perchunk_skip_first_last_{n}frames/
- metrics/frame_reserve_runtime_memory.csv
- metrics/frame_reserve_vs_bf16.csv
- metrics/frame_reserve_vbench.csv
- metrics/frame_reserve_ablation_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")

LAYERS = 30
HEADS = 24
TOKENS_PER_FRAME = 880
LOGICAL_DIM = 256
BF16_BYTES = 2
CACHED_FRAMES = 40
RVQ_LEVELS = 8
RVQ_CODEBOOK_SIZE = 64

VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]

VIDEO_METRIC_COLUMNS = [
    "prompt_id",
    "prompt",
    "seed",
    "setting",
    "video_path",
    "e2e_time_s",
    "denoise_time_s",
    "quant_time_s",
    "peak_gpu_mem_mb",
    "kv_cache_mem_mb",
    "compression_ratio",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-file", type=Path, default=None)
    p.add_argument("--reference-dir", type=Path, default=None)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video-workers", type=int, default=3)
    p.add_argument(
        "--reserve-mode",
        choices=["last_frames", "recent_frames", "sparse_first_recent_chunks"],
        default="last_frames",
        help=(
            "last_frames keeps a static final quantizable frame range. "
            "recent_frames keeps the newest N cached latent frames BF16 at every "
            "online quantization step. sparse_first_recent_chunks drops middle "
            "KV entries entirely and keeps only first/recent BF16 chunks."
        ),
    )
    p.add_argument(
        "--pairs",
        default="",
        help=(
            "Optional comma-separated first:recent frame pairs for "
            "--reserve-mode recent_frames, e.g. '1:4,1:3,3:1'. "
            "If omitted, runs symmetric 1:1,2:2,3:3 or last-frame defaults."
        ),
    )
    p.add_argument(
        "--anchor-frame-stride",
        type=int,
        default=0,
        help=(
            "If >0, keep every Nth latent cache frame BF16 in addition to the "
            "first/recent frame rules."
        ),
    )
    p.add_argument(
        "--anchor-frame-offset",
        type=int,
        default=0,
        help="Frame offset used with --anchor-frame-stride.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    return p.parse_args()


def output_stem(args: argparse.Namespace) -> str:
    if args.reserve_mode == "sparse_first_recent_chunks":
        return "sparse_kv_first_recent"
    if args.reserve_mode == "recent_frames" and args.pairs.strip() and args.anchor_frame_stride > 0:
        return "recent_frame_pair_anchor_reserve"
    if args.reserve_mode == "recent_frames" and args.pairs.strip():
        return "recent_frame_pair_reserve"
    return "recent_frame_reserve" if args.reserve_mode == "recent_frames" else "frame_reserve"


def parse_pairs(pairs: str) -> list[tuple[int, int]]:
    parsed: list[tuple[int, int]] = []
    for raw in pairs.split(","):
        item = raw.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid pair {item!r}; expected first:recent")
        first_s, recent_s = item.split(":", 1)
        first = int(first_s)
        recent = int(recent_s)
        if first < 0 or recent < 0:
            raise ValueError(f"Invalid pair {item!r}; values must be non-negative")
        parsed.append((first, recent))
    return parsed


def settings(
    root: Path,
    reserve_mode: str = "last_frames",
    pairs: str = "",
    anchor_frame_stride: int = 0,
    anchor_frame_offset: int = 0,
) -> list[dict[str, Any]]:
    k_dir = root / "sweep_k_l8c64_codebooks"
    v_dir = root / "sweep_v_l8c64_codebooks"
    common = [
        "--setting",
        "kvrvq-i4",
        "--k_rvq_codebook_dir",
        str(k_dir),
        "--v_rvq_codebook_dir",
        str(v_dir),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    specs = []
    if reserve_mode in {"recent_frames", "sparse_first_recent_chunks"} and pairs.strip():
        frame_pairs = parse_pairs(pairs)
    else:
        frame_pairs = [(n, n) for n in (1, 2, 3)]
    for first_frames, recent_frames in frame_pairs:
        if reserve_mode == "sparse_first_recent_chunks":
            label = (
                "Sparse-KV BF16 first/recent only "
                f"first{first_frames}+recent{recent_frames} frames"
            )
            dir_name = f"sparse_kv_bf16_first{first_frames}_recent{recent_frames}frames"
            extra = [
                "--setting",
                "bf16",
                "--kv_logical_head_dim",
                "256",
                "--kv_reference_frames",
                str(CACHED_FRAMES),
                "--kv_sparse_policy",
                "first_recent_chunks",
                "--kv_sparse_first_frames",
                str(first_frames),
                "--kv_sparse_recent_frames",
                str(recent_frames),
            ]
        elif reserve_mode == "recent_frames":
            anchor_label = ""
            anchor_suffix = ""
            anchor_extra: list[str] = []
            if anchor_frame_stride > 0:
                anchor_label = (
                    f" + anchor every{anchor_frame_stride}"
                    f"off{anchor_frame_offset} frames"
                )
                anchor_suffix = (
                    f"_anchor{anchor_frame_stride}"
                    f"off{anchor_frame_offset}frames"
                )
                anchor_extra = [
                    "--kv_quant_anchor_frame_stride",
                    str(anchor_frame_stride),
                    "--kv_quant_anchor_frame_offset",
                    str(anchor_frame_offset),
                ]
            label = (
                f"RVQ-L8C64 INT4 per-chunk skip "
                f"first{first_frames}+recent{recent_frames} frames"
                f"{anchor_label}"
            )
            dir_name = (
                "rvq_l8c64_int4_perchunk_skip_"
                f"first{first_frames}_recent{recent_frames}frames"
                f"{anchor_suffix}"
            )
            extra = [
                *common,
                "--kv_quant_skip_first_frames",
                str(first_frames),
                "--kv_quant_recent_frames",
                str(recent_frames),
                *anchor_extra,
            ]
        else:
            n = first_frames
            label = f"RVQ-L8C64 INT4 per-chunk skip first{n}+last{n} frames"
            dir_name = f"rvq_l8c64_int4_perchunk_skip_first_last_{n}frames"
            extra = [
                *common,
                "--kv_quant_skip_first_frames",
                str(n),
                "--kv_quant_skip_last_frames",
                str(n),
            ]
        specs.append(
            {
                "label": label,
                "dir": dir_name,
                "extra": extra,
                "retained_bf16_frames": retained_frame_count(
                    first_frames,
                    recent_frames if reserve_mode in {"recent_frames", "sparse_first_recent_chunks"} else 0,
                    anchor_frame_stride if reserve_mode == "recent_frames" else 0,
                    anchor_frame_offset,
                )
                if reserve_mode in {"recent_frames", "sparse_first_recent_chunks"}
                else first_frames + recent_frames,
                "reserved_each_side": first_frames if first_frames == recent_frames else "",
                "first_frames": first_frames,
                "recent_frames": recent_frames if reserve_mode in {"recent_frames", "sparse_first_recent_chunks"} else "",
                "anchor_frame_stride": anchor_frame_stride if reserve_mode == "recent_frames" else 0,
                "anchor_frame_offset": anchor_frame_offset if reserve_mode == "recent_frames" else 0,
            }
        )
    return specs


def retained_frame_count(
    first_frames: int,
    recent_frames: int,
    anchor_frame_stride: int = 0,
    anchor_frame_offset: int = 0,
    cached_frames: int = CACHED_FRAMES,
) -> int:
    retained = set(range(0, min(max(0, first_frames), cached_frames)))
    if recent_frames > 0:
        retained.update(range(max(0, cached_frames - recent_frames), cached_frames))
    if anchor_frame_stride > 0:
        offset = max(0, anchor_frame_offset) % anchor_frame_stride
        retained.update(range(offset, cached_frames, anchor_frame_stride))
    return len(retained)


def as_float(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def full_kv_mib(frames: int = CACHED_FRAMES) -> float:
    bytes_ = 2 * LAYERS * HEADS * frames * TOKENS_PER_FRAME * LOGICAL_DIM * BF16_BYTES
    return bytes_ / 1024**2


def rvq_l8c64_int4_mib(compressed_frames: int, retained_bf16_frames: int) -> float:
    vectors = LAYERS * HEADS * compressed_frames * TOKENS_PER_FRAME
    index_bits = math.ceil(math.log2(RVQ_CODEBOOK_SIZE))
    one_index = vectors * RVQ_LEVELS * index_bits / 8.0
    one_scale = vectors / 2.0
    one_codebook = LAYERS * HEADS * RVQ_LEVELS * RVQ_CODEBOOK_SIZE * LOGICAL_DIM / 2.0
    one_codebook_meta = LAYERS * HEADS * RVQ_LEVELS * RVQ_CODEBOOK_SIZE * BF16_BYTES
    compressed = 4 * one_index + 4 * one_scale + 2 * one_codebook + 2 * one_codebook_meta
    retained = full_kv_mib(retained_bf16_frames) * 1024**2
    return (compressed + retained) / 1024**2


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_video_cmd(
    args: argparse.Namespace,
    spec: dict[str, Any],
    prompt_start: int | None = None,
    num_prompts: int | None = None,
    metrics_name: str | None = None,
    config_name: str | None = None,
) -> list[str]:
    prompt_file = args.prompt_file or (args.output_dir / "prompts_test.txt")
    out_dir = args.output_dir / "videos" / spec["dir"]
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(prompt_file),
        "--num-prompts",
        str(args.num_prompts if num_prompts is None else num_prompts),
        "--output-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--num_frames",
        "173",
        "--num_chunk",
        "11",
        "--num_inference_steps",
        "50",
        "--image_path",
        "none",
        "--pose",
        "w-96",
        "--memory_frames",
        "80",
        "--temporal_context_size",
        "76",
        "--pred_latent_size",
        "4",
        "--offload_text_encoder",
        *spec["extra"],
    ]
    if prompt_start is not None:
        cmd.extend(["--prompt-start", str(prompt_start)])
    if metrics_name is not None:
        cmd.extend(["--metrics-name", metrics_name])
    if config_name is not None:
        cmd.extend(["--config-name", config_name])
    return cmd


def collect_prompt_metric_rows(
    out_dir: Path,
    include_final: bool = True,
) -> dict[int, dict[str, str]]:
    rows_by_prompt: dict[int, dict[str, str]] = {}
    metric_paths = sorted(out_dir.glob("metrics_prompt_*.csv"))
    if include_final:
        metric_paths.insert(0, out_dir / "metrics.csv")
    for path in metric_paths:
        if not path.exists():
            continue
        try:
            rows = read_metrics_rows(path)
        except Exception:
            continue
        for row in rows:
            try:
                prompt_id = int(row["prompt_id"])
            except Exception:
                continue
            video_path = Path(row.get("video_path", ""))
            if not video_path.exists():
                continue
            rows_by_prompt[prompt_id] = row
    return rows_by_prompt


def write_metrics_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "null") for key in VIDEO_METRIC_COLUMNS})


def run_sharded_video(
    args: argparse.Namespace,
    spec: dict[str, Any],
    gpus: list[int],
    stem: str,
) -> None:
    out_dir = args.output_dir / "videos" / spec["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {} if args.overwrite else collect_prompt_metric_rows(out_dir, include_final=True)
    jobs = []
    for prompt_id in range(args.num_prompts):
        if prompt_id in existing:
            print(f"[video-prompt-skip] {spec['label']} prompt={prompt_id}", flush=True)
            continue
        metrics_name = f"metrics_prompt_{prompt_id}.csv"
        config_name = f"config_prompt_{prompt_id}.json"
        jobs.append(
            {
                "name": f"video {spec['label']} prompt={prompt_id}",
                "cmd": build_video_cmd(
                    args,
                    spec,
                    prompt_start=prompt_id,
                    num_prompts=1,
                    metrics_name=metrics_name,
                    config_name=config_name,
                ),
                "log": args.output_dir
                / "logs"
                / f"{stem}_video_{spec['dir']}_prompt{prompt_id}.log",
            }
        )
    if jobs:
        run_process_queue(jobs, gpus)

    merged = collect_prompt_metric_rows(out_dir, include_final=not args.overwrite)
    missing = [idx for idx in range(args.num_prompts) if idx not in merged]
    if missing:
        raise RuntimeError(f"Missing prompt metrics for {spec['label']}: {missing}")
    write_metrics_csv(out_dir / "metrics.csv", [merged[idx] for idx in range(args.num_prompts)])


def run_process_queue(jobs: list[dict[str, Any]], gpus: list[int], poll_s: int = 30) -> None:
    active: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    next_idx = 0
    while next_idx < len(jobs) or active:
        busy = {job["gpu"] for job in active}
        while next_idx < len(jobs):
            gpu = next((item for item in gpus if item not in busy), None)
            if gpu is None:
                break
            job = jobs[next_idx]
            next_idx += 1
            log_path = job["log"]
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(job["cmd"]) + "\n")
            log.flush()
            env = os.environ.copy()
            env.update(job.get("env", {}))
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            env.setdefault("MASTER_PORT", str(29650 + int(gpu)))
            print(f"[start] gpu={gpu} {job['name']} log={log_path}", flush=True)
            proc = subprocess.Popen(
                job["cmd"],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active.append({"proc": proc, "log": log, "log_path": log_path, "gpu": gpu, "name": job["name"]})
            busy.add(gpu)

        time.sleep(poll_s)
        still = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            if rc == 0:
                print(f"[done] {job['name']}", flush=True)
            else:
                print(f"[failed] {job['name']} rc={rc} log={job['log_path']}", flush=True)
                failures.append({"name": job["name"], "rc": rc, "log": str(job["log_path"])})
        active = still

    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))


def run_videos(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_video:
        return
    jobs = []
    stem = output_stem(args)
    worker_gpus = gpus[: max(1, min(len(gpus), args.video_workers))]
    for spec in settings(
        args.output_dir,
        args.reserve_mode,
        args.pairs,
        args.anchor_frame_stride,
        args.anchor_frame_offset,
    ):
        out_dir = args.output_dir / "videos" / spec["dir"]
        if metrics_complete(out_dir / "metrics.csv", args.num_prompts) and not args.overwrite:
            print(f"[video-skip] {spec['label']}", flush=True)
            continue
        if args.video_workers > 1:
            run_sharded_video(args, spec, worker_gpus, stem)
            continue
        jobs.append(
            {
                "name": f"video {spec['label']}",
                "cmd": build_video_cmd(args, spec),
                "log": args.output_dir / "logs" / f"{stem}_video_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, worker_gpus)


def run_quality(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_quality:
        return
    reference_dir = args.reference_dir or (args.output_dir / "videos" / "bf16")
    jobs = []
    stem = output_stem(args)
    quality_root = args.output_dir / f"vs_bf16_{stem}"
    for spec in settings(
        args.output_dir,
        args.reserve_mode,
        args.pairs,
        args.anchor_frame_stride,
        args.anchor_frame_offset,
    ):
        out_dir = args.output_dir / "videos" / spec["dir"]
        metrics_csv = out_dir / "metrics.csv"
        out_json = quality_root / f"{spec['dir']}.json"
        out_csv = quality_root / f"{spec['dir']}.csv"
        if out_json.exists() and out_csv.exists() and not args.overwrite:
            print(f"[quality-skip] {spec['label']}", flush=True)
            continue
        jobs.append(
            {
                "name": f"quality {spec['label']}",
                "cmd": [
                    str(args.python),
                    str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                    "--reference-dir",
                    str(reference_dir),
                    "--test-dir",
                    str(out_dir),
                    "--metrics-csv",
                    str(metrics_csv),
                    "--setting",
                    spec["label"],
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                    "--device",
                    "cuda",
                ],
                "log": args.output_dir / "logs" / f"{stem}_quality_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, gpus, poll_s=10)


def run_vbench(args: argparse.Namespace, gpus: list[int]) -> None:
    if args.skip_vbench:
        return
    if not args.vbench_bin.exists():
        raise FileNotFoundError(args.vbench_bin)
    jobs = []
    expected_results: list[Path] = []
    stem = output_stem(args)
    vbench_root = args.output_dir / f"vbench_{stem}"
    for spec in settings(
        args.output_dir,
        args.reserve_mode,
        args.pairs,
        args.anchor_frame_stride,
        args.anchor_frame_offset,
    ):
        out_dir = args.output_dir / "videos" / spec["dir"]
        metrics_csv = out_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        prompt_json = vbench_root / f"prompts_{spec['dir']}.json"
        prompt_json.parent.mkdir(parents=True, exist_ok=True)
        prompt_map = {}
        for row in read_metrics_rows(metrics_csv):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = (REPO_ROOT / video_path).resolve()
            prompt_map[str(video_path)] = row["prompt"]
        prompt_json.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        for dim in VBENCH_DIMS:
            result_dir = vbench_root / spec["dir"] / dim
            sentinel = result_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                print(f"[vbench-skip] {spec['label']} {dim}", flush=True)
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            expected_results.append(result_dir)
            jobs.append(
                {
                    "name": f"vbench {spec['label']} {dim}",
                    "cmd": [
                        str(args.vbench_bin),
                        "evaluate",
                        "--ngpus",
                        "1",
                        "--output_path",
                        str(result_dir),
                        "--videos_path",
                        str(out_dir),
                        "--dimension",
                        dim,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(prompt_json),
                    ],
                    "log": args.output_dir / "logs" / f"{stem}_vbench_{spec['dir']}_{dim}.log",
                    "env": {"PYTHONUNBUFFERED": "1"},
                }
            )
    run_process_queue(jobs, gpus, poll_s=15)
    missing = []
    for result_dir in expected_results:
        if list(result_dir.glob("*_eval_results.json")):
            (result_dir / "vbench.done").touch()
        else:
            missing.append(str(result_dir))
    if missing:
        raise RuntimeError("VBench completed without result JSON:\n" + "\n".join(missing))


def parse_vbench_result(label: str, result_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"setting": label}
    for dim in VBENCH_DIMS:
        row[dim] = None
        dim_root = result_root / dim
        for path in sorted(dim_root.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            stack = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    value = item.get(dim)
                    if isinstance(value, (int, float)):
                        row[dim] = value
                    elif isinstance(value, dict):
                        for key in ("score", "mean", "value"):
                            if isinstance(value.get(key), (int, float)):
                                row[dim] = value[key]
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        row[dim] = value[0]
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row.keys())) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> None:
    runtime_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    vbench_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    full_mib = full_kv_mib(CACHED_FRAMES)
    stem = output_stem(args)
    quality_root = args.output_dir / f"vs_bf16_{stem}"
    vbench_root = args.output_dir / f"vbench_{stem}"
    for spec in settings(
        args.output_dir,
        args.reserve_mode,
        args.pairs,
        args.anchor_frame_stride,
        args.anchor_frame_offset,
    ):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        rows = read_metrics_rows(metrics_csv)

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = rows[1:] if drop0 and len(rows) > 1 else rows
            return [v for v in (as_float(row.get(key)) for row in src) if v is not None]

        retained = int(spec["retained_bf16_frames"])
        if args.reserve_mode == "sparse_first_recent_chunks":
            kv_mib = mean(vals("kv_cache_mem_mb")) or full_kv_mib(retained)
            compression_ratio = mean(vals("compression_ratio"))
            if compression_ratio is None and kv_mib:
                compression_ratio = full_mib / kv_mib
        else:
            kv_mib = rvq_l8c64_int4_mib(CACHED_FRAMES - retained, retained)
            compression_ratio = full_mib / kv_mib
        runtime = {
            "setting": spec["label"],
            "reserved_each_side_frames": spec["reserved_each_side"],
            "first_frames": spec["first_frames"],
            "recent_frames": spec["recent_frames"],
            "retained_bf16_frames": retained,
            "videos": len(rows),
            "e2e_mean_s": mean(vals("e2e_time_s")),
            "e2e_drop0_s": mean(vals("e2e_time_s", True)),
            "denoise_mean_s": mean(vals("denoise_time_s")),
            "quant_drop0_s": mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": kv_mib,
            "compression_ratio": compression_ratio,
        }
        runtime_rows.append(runtime)

        quality_json = quality_root / f"{spec['dir']}.json"
        quality = {"setting": spec["label"], "psnr": None, "ssim": None, "lpips": None}
        if quality_json.exists():
            payload = json.loads(quality_json.read_text(encoding="utf-8"))
            quality = {
                "setting": spec["label"],
                "psnr": payload.get("psnr_mean"),
                "ssim": payload.get("ssim_mean"),
                "lpips": payload.get("lpips_mean"),
            }
        quality_rows.append(quality)
        vbench = parse_vbench_result(spec["label"], vbench_root / spec["dir"])
        vbench_rows.append(vbench)
        summary_rows.append(
            {
                **runtime,
                **{k: v for k, v in quality.items() if k != "setting"},
                **{k: v for k, v in vbench.items() if k != "setting"},
            }
        )
    write_csv(args.output_dir / "metrics" / f"{stem}_runtime_memory.csv", runtime_rows)
    write_csv(args.output_dir / "metrics" / f"{stem}_vs_bf16.csv", quality_rows)
    write_csv(args.output_dir / "metrics" / f"{stem}_vbench.csv", vbench_rows)
    write_csv(args.output_dir / "metrics" / f"{stem}_ablation_summary.csv", summary_rows)


def main() -> None:
    args = parse_args()
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus must not be empty")
    run_videos(args, gpus)
    run_quality(args, gpus)
    run_vbench(args, gpus)
    aggregate(args)
    stem = output_stem(args)
    print(f"[summary] {args.output_dir / 'metrics' / f'{stem}_ablation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
