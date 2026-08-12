#!/usr/bin/env python3
"""Train and benchmark temporal-residual L8C64 RVQ.

Definition:
For each layer/head/spatial token sequence x[t], train RVQ on
delta[0] = x[0], delta[t] = x[t] - x[t-1]. Runtime encode/decode uses the same
delta space and reconstructs by cumulative sum along the frame axis.

Example:
tmux new-session -d -s temporal_delta_l8c64 \
  'cd /home/bowen/Quant-VideoGen && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   /home/bowen/anaconda3/envs/hw8/bin/python scripts/run_temporal_residual_rvq_bench.py \
     --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
     --gpus 1,3,4,5 \
     |& tee /data/bowen/t2v_kv_bench_moviegen_20train_10test/logs/temporal_delta_l8c64_main.log'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
RVQ_DIAG_DIR = REPO_ROOT / "scripts" / "rvq" / "diag"
if str(RVQ_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIAG_DIR))

from run_rvq_diag import (  # noqa: E402
    train_kmeans_faiss,
    train_kmeans_quantvideo,
    train_kmeans_sklearn,
)

DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH = Path("/home/bowen/anaconda3/bin/vbench")

LAYERS = 30
HEADS = 24
LEVELS = 8
CODEBOOK_SIZE = 64
TOKENS_PER_FRAME = 880
LOGICAL_DIM = 256
BF16_BYTES = 2
CACHED_FRAMES = 40
CHUNK_SIZE = 4
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
    p.add_argument("--gpus", default="1,3,4,5")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--video-workers", type=int, default=4)
    p.add_argument("--wait-for-gpus", action="store_true")
    p.add_argument("--wait-min-free-mb", type=int, default=36000)
    p.add_argument("--wait-poll-s", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--worker", choices=["train-shard"], default=None)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def parse_gpus(raw: str) -> list[int]:
    values = [int(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--gpus must not be empty")
    return values


def temporal_dir(root: Path, tensor: str) -> Path:
    return root / f"temporal_delta_{tensor.lower()}_l8c64_codebooks"


def codebook_path(root: Path, tensor: str, layer: int, head: int) -> Path:
    return temporal_dir(root, tensor) / f"layer_{layer}_head_{head}.pt"


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads((root / "prompt_split.json").read_text(encoding="utf-8"))
    return list(payload[split])


def layer_file(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def temporal_delta(data: torch.Tensor) -> torch.Tensor:
    # data: [T, S, D] for one head.
    delta = torch.empty_like(data, dtype=torch.float32)
    delta[0].copy_(data[0].float())
    if data.shape[0] > 1:
        delta[1:].copy_(data[1:].float() - data[:-1].float())
    return delta


def sample_temporal_delta_vectors(
    root: Path,
    tensor: str,
    layer: int,
    head: int,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    items = split_items(root, "train")
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    samples: list[torch.Tensor] = []
    remaining = sample_size
    for item in items:
        if remaining <= 0:
            break
        payload = torch.load(
            layer_file(root, "train", int(item["prompt_id"]), layer),
            map_location="cpu",
            weights_only=False,
        )
        data = payload[tensor][:, head].float()  # [T, 880, D]
        delta = temporal_delta(data).reshape(-1, data.shape[-1]).contiguous()
        gen = torch.Generator(device="cpu").manual_seed(
            seed + layer * 100000 + head * 1000 + int(item["prompt_id"])
        )
        take = min(per_prompt, delta.shape[0], remaining)
        ids = torch.randperm(delta.shape[0], generator=gen)[:take]
        samples.append(delta[ids])
        remaining -= take
    if not samples:
        raise RuntimeError(f"No samples for tensor={tensor} layer={layer} head={head}")
    return torch.cat(samples, dim=0).contiguous()


def train_kmeans_auto(
    x: torch.Tensor,
    k: int,
    max_iters: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    errors = []
    try:
        return train_kmeans_quantvideo(x, k, max_iters, device)
    except Exception as exc:
        errors.append(f"QuantVideo KMeans failed: {exc}")
    try:
        return train_kmeans_faiss(x, k, max_iters, 0)
    except Exception as exc:
        errors.append(f"FAISS GPU KMeans failed: {exc}")
    try:
        return train_kmeans_sklearn(x, k, max_iters, seed)
    except Exception as exc:
        errors.append(f"sklearn MiniBatchKMeans failed: {exc}")
        raise RuntimeError("; ".join(errors)) from exc


def train_one(args: argparse.Namespace, tensor: str, layer: int, head: int) -> None:
    out_path = codebook_path(args.output_dir, tensor, layer, head)
    if out_path.exists() and not args.overwrite:
        print(f"[train-skip] tensor={tensor} layer={layer} head={head}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = sample_temporal_delta_vectors(args.output_dir, tensor, layer, head, args.sample_size, args.seed)
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = (x / scale).contiguous()

    codebooks = []
    residual_mse = []
    backends = []
    print(
        f"[train] temporal-delta tensor={tensor} layer={layer} head={head} samples={x.shape[0]} "
        "backend priority: QuantVideo KMeans / FAISS GPU KMeans / sklearn MiniBatchKMeans",
        flush=True,
    )
    for level in range(LEVELS):
        centroids, ids, backend = train_kmeans_auto(
            residual,
            CODEBOOK_SIZE,
            args.kmeans_max_iters,
            args.seed + layer * 100000 + head * 1000 + level,
            device,
        )
        assigned = centroids[ids].float()
        residual = (residual.float() - assigned).contiguous()
        mse = float((residual * residual).mean().item())
        print(
            f"[train] temporal-delta tensor={tensor} layer={layer} head={head} "
            f"level={level + 1} backend={backend} residual_mse={mse:.8g}",
            flush=True,
        )
        codebooks.append(centroids.float().cpu())
        residual_mse.append(mse)
        backends.append(backend)

    payload = {
        "codebooks": torch.stack(codebooks, dim=0).float(),
        "temporal_residual_mode": "delta",
        "temporal_frame_seq_length": TOKENS_PER_FRAME,
        "config": {
            "layer": layer,
            "head": head,
            "levels": LEVELS,
            "codebook_size": CODEBOOK_SIZE,
            "sample_size": int(x.shape[0]),
            "kmeans_max_iters": args.kmeans_max_iters,
            "backend_priority": ["QuantVideo KMeans", "FAISS GPU KMeans", "sklearn MiniBatchKMeans"],
            "backends": backends,
            "residual_mse": residual_mse,
            "tensor": tensor,
            "temporal_residual_mode": "delta",
            "temporal_frame_seq_length": TOKENS_PER_FRAME,
            "temporal_first_frame": "absolute",
            "temporal_delta": "x[t]-x[t-1] per spatial token",
            "normalization": "per temporal-delta vector RMS scale, then greedy RVQ in normalized delta space",
            "key_type": "pre_rope" if tensor == "K" else None,
            "value_type": "pre_rope" if tensor == "V" else None,
        },
    }
    torch.save(payload, out_path)
    print(f"[train-done] temporal-delta tensor={tensor} layer={layer} head={head}", flush=True)


def train_shard(args: argparse.Namespace) -> None:
    tasks = [(tensor, layer, head) for tensor in ("K", "V") for layer in range(LAYERS) for head in range(HEADS)]
    for tensor, layer, head in tasks[args.shard_id :: max(1, args.num_shards)]:
        train_one(args, tensor, layer, head)


def gpu_free_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    result = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            result[int(parts[0])] = int(parts[1])
    return result


def wait_for_gpus(gpu_ids: list[int], min_free_mb: int, poll_s: int) -> None:
    while True:
        free = gpu_free_mib()
        missing = [gpu for gpu in gpu_ids if free.get(gpu, 0) < min_free_mb]
        if not missing:
            print(f"[gpu-ready] {gpu_ids}", flush=True)
            return
        print(
            f"[gpu-wait] need >= {min_free_mb} MiB on {gpu_ids}; "
            f"current={{{', '.join(f'{g}: {free.get(g, 0)}' for g in gpu_ids)}}}",
            flush=True,
        )
        time.sleep(poll_s)


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
            env.setdefault("MASTER_PORT", str(29700 + int(gpu)))
            print(f"[start] gpu={gpu} {job['name']} log={log_path}", flush=True)
            proc = subprocess.Popen(job["cmd"], cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
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


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) if rows else ["status"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def spawn_training(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_train:
        return
    pending = [
        (tensor, layer, head)
        for tensor in ("K", "V")
        for layer in range(LAYERS)
        for head in range(HEADS)
        if args.overwrite or not codebook_path(args.output_dir, tensor, layer, head).exists()
    ]
    if not pending:
        print("[train] all temporal-delta codebooks already exist", flush=True)
        return
    jobs = []
    n = len(gpu_ids)
    for shard_id, gpu in enumerate(gpu_ids):
        jobs.append(
            {
                "name": f"train-shard-{shard_id}",
                "cmd": [
                    str(args.python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "train-shard",
                    "--output-dir",
                    str(args.output_dir),
                    "--sample-size",
                    str(args.sample_size),
                    "--kmeans-max-iters",
                    str(args.kmeans_max_iters),
                    "--seed",
                    str(args.seed),
                    "--shard-id",
                    str(shard_id),
                    "--num-shards",
                    str(n),
                ],
                "log": args.output_dir / "logs" / f"temporal_delta_train_gpu{gpu}.log",
            }
        )
    start = time.perf_counter()
    run_process_queue(jobs, gpu_ids, poll_s=30)
    elapsed = time.perf_counter() - start
    append_csv(
        args.output_dir / "metrics" / "temporal_delta_l8c64_training_time.csv",
        [
            {
                "phase": "train_KV_temporal_delta_l8c64",
                "wall_time_s": elapsed,
                "gpus": ",".join(str(g) for g in gpu_ids),
                "sample_size": args.sample_size,
                "kmeans_max_iters": args.kmeans_max_iters,
            }
        ],
    )


def settings(root: Path) -> list[dict[str, Any]]:
    common = [
        "--setting",
        "kvrvq-i4",
        "--k_rvq_codebook_dir",
        str(temporal_dir(root, "K")),
        "--v_rvq_codebook_dir",
        str(temporal_dir(root, "V")),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    return [
        {
            "label": "TemporalDelta-RVQ-L8C64-INT4-perchunk",
            "dir": "temporal_delta_rvq_l8c64_int4_perchunk",
            "extra": common,
        },
        {
            "label": "TemporalDelta-RVQ-L8C64-INT4-perchunk-first1-recent1",
            "dir": "temporal_delta_rvq_l8c64_int4_perchunk_first1_recent1",
            "extra": [
                *common,
                "--kv_quant_skip_chunk_policy",
                "first",
                "--kv_quant_recent_policy",
                "nearest1",
            ],
        },
    ]


def build_video_cmd(
    args: argparse.Namespace,
    spec: dict[str, Any],
    prompt_start: int | None = None,
    num_prompts: int | None = None,
    metrics_name: str | None = None,
    config_name: str | None = None,
) -> list[str]:
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        str(args.num_prompts if num_prompts is None else num_prompts),
        "--output-dir",
        str(args.output_dir / "videos" / spec["dir"]),
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


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = read_metrics_rows(path)
    except Exception:
        return False
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def collect_prompt_metric_rows(out_dir: Path, include_final: bool = True) -> dict[int, dict[str, str]]:
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
            if not Path(row.get("video_path", "")).exists():
                continue
            rows_by_prompt[prompt_id] = row
    return rows_by_prompt


def write_video_metrics(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "null") for key in VIDEO_METRIC_COLUMNS})


def run_sharded_video(args: argparse.Namespace, spec: dict[str, Any], gpu_ids: list[int]) -> None:
    out_dir = args.output_dir / "videos" / spec["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {} if args.overwrite else collect_prompt_metric_rows(out_dir, include_final=True)
    jobs = []
    for prompt_id in range(args.num_prompts):
        if prompt_id in existing:
            print(f"[video-prompt-skip] {spec['label']} prompt={prompt_id}", flush=True)
            continue
        jobs.append(
            {
                "name": f"video {spec['label']} prompt={prompt_id}",
                "cmd": build_video_cmd(
                    args,
                    spec,
                    prompt_start=prompt_id,
                    num_prompts=1,
                    metrics_name=f"metrics_prompt_{prompt_id}.csv",
                    config_name=f"config_prompt_{prompt_id}.json",
                ),
                "log": args.output_dir / "logs" / f"temporal_delta_video_{spec['dir']}_prompt{prompt_id}.log",
            }
        )
    if jobs:
        run_process_queue(jobs, gpu_ids, poll_s=30)
    merged = collect_prompt_metric_rows(out_dir, include_final=not args.overwrite)
    missing = [idx for idx in range(args.num_prompts) if idx not in merged]
    if missing:
        raise RuntimeError(f"Missing prompt metrics for {spec['label']}: {missing}")
    write_video_metrics(out_dir / "metrics.csv", [merged[idx] for idx in range(args.num_prompts)])


def run_videos(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_video:
        return
    worker_gpus = gpu_ids[: max(1, min(len(gpu_ids), args.video_workers))]
    for spec in settings(args.output_dir):
        metrics_path = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if metrics_complete(metrics_path, args.num_prompts) and not args.overwrite:
            print(f"[video-skip] {spec['label']}", flush=True)
            continue
        run_sharded_video(args, spec, worker_gpus)


def run_quality(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_quality:
        return
    reference_dir = args.output_dir / "videos" / "bf16"
    jobs = []
    for spec in settings(args.output_dir):
        out_json = args.output_dir / "vs_bf16_temporal_delta" / f"{spec['dir']}.json"
        out_csv = args.output_dir / "vs_bf16_temporal_delta" / f"{spec['dir']}.csv"
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
                    str(args.output_dir / "videos" / spec["dir"]),
                    "--metrics-csv",
                    str(args.output_dir / "videos" / spec["dir"] / "metrics.csv"),
                    "--setting",
                    spec["label"],
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                    "--device",
                    "cuda",
                ],
                "log": args.output_dir / "logs" / f"temporal_delta_quality_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, gpu_ids, poll_s=10)


def run_vbench(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_vbench:
        return
    jobs = []
    expected = []
    for spec in settings(args.output_dir):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        prompt_json = args.output_dir / "vbench_temporal_delta" / f"prompts_{spec['dir']}.json"
        prompt_json.parent.mkdir(parents=True, exist_ok=True)
        prompt_map = {}
        for row in read_metrics_rows(metrics_csv):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = (REPO_ROOT / video_path).resolve()
            prompt_map[str(video_path)] = row["prompt"]
        prompt_json.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        for dim in VBENCH_DIMS:
            result_dir = args.output_dir / "vbench_temporal_delta" / spec["dir"] / dim
            sentinel = result_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                print(f"[vbench-skip] {spec['label']} {dim}", flush=True)
                continue
            result_dir.mkdir(parents=True, exist_ok=True)
            expected.append(result_dir)
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
                        str(args.output_dir / "videos" / spec["dir"]),
                        "--dimension",
                        dim,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(prompt_json),
                    ],
                    "log": args.output_dir / "logs" / f"temporal_delta_vbench_{spec['dir']}_{dim}.log",
                    "env": {"PYTHONUNBUFFERED": "1"},
                }
            )
    run_process_queue(jobs, gpu_ids, poll_s=15)
    for result_dir in expected:
        if list(result_dir.glob("*_eval_results.json")):
            (result_dir / "vbench.done").touch()


def as_float(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_vbench_result(label: str, result_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"setting": label}
    for dim in VBENCH_DIMS:
        row[dim] = None
        for path in sorted((result_root / dim).rglob("*.json")):
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


def aggregate(args: argparse.Namespace) -> None:
    runtime_rows = []
    quality_rows = []
    vbench_rows = []
    summary_rows = []
    for spec in settings(args.output_dir):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        rows = read_metrics_rows(metrics_csv) if metrics_csv.exists() else []

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = rows[1:] if drop0 and len(rows) > 1 else rows
            return [v for v in (as_float(row.get(key)) for row in src) if v is not None]

        runtime = {
            "setting": spec["label"],
            "videos": len(rows),
            "e2e_mean_s": mean(vals("e2e_time_s")),
            "e2e_drop0_s": mean(vals("e2e_time_s", True)),
            "denoise_mean_s": mean(vals("denoise_time_s")),
            "quant_drop0_s": mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
            "compression_ratio": mean(vals("compression_ratio")),
        }
        runtime_rows.append(runtime)

        quality_json = args.output_dir / "vs_bf16_temporal_delta" / f"{spec['dir']}.json"
        quality = {"setting": spec["label"], "psnr": None, "ssim": None, "lpips": None}
        if quality_json.exists():
            payload = json.loads(quality_json.read_text(encoding="utf-8"))
            quality.update(
                {
                    "psnr": payload.get("psnr_mean"),
                    "ssim": payload.get("ssim_mean"),
                    "lpips": payload.get("lpips_mean"),
                }
            )
        quality_rows.append(quality)
        vbench = parse_vbench_result(spec["label"], args.output_dir / "vbench_temporal_delta" / spec["dir"])
        vbench_rows.append(vbench)
        summary_rows.append(
            {
                **runtime,
                **{k: v for k, v in quality.items() if k != "setting"},
                **{k: v for k, v in vbench.items() if k != "setting"},
            }
        )

    write_csv(args.output_dir / "metrics" / "temporal_delta_l8c64_runtime.csv", runtime_rows)
    write_csv(args.output_dir / "metrics" / "temporal_delta_l8c64_vs_bf16.csv", quality_rows)
    write_csv(args.output_dir / "metrics" / "temporal_delta_l8c64_vbench.csv", vbench_rows)
    write_csv(args.output_dir / "metrics" / "temporal_delta_l8c64_summary.csv", summary_rows)
    report = args.output_dir / "reports" / "temporal_delta_l8c64_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Temporal-Residual L8C64 INT4 RVQ",
                "",
                "- Data: MovieGen T2V split under this output dir, train 20 / test 10, seed 42.",
                "- Training space: per layer/head/spatial-token temporal delta.",
                "- Delta definition: `delta[0]=x[0]`, `delta[t]=x[t]-x[t-1]`.",
                "- Runtime: encode delta with standard RVQ and decode with cumulative sum along frame axis.",
                "- Storage: same L8C64 INT4 codebook/index/scale format as normal RVQ.",
                "",
                "See `metrics/temporal_delta_l8c64_summary.csv`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.worker == "train-shard":
        train_shard(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reports").mkdir(parents=True, exist_ok=True)
    gpu_ids = parse_gpus(args.gpus)
    if args.wait_for_gpus:
        wait_for_gpus(gpu_ids, args.wait_min_free_mb, args.wait_poll_s)
    spawn_training(args, gpu_ids)
    run_videos(args, gpu_ids)
    run_quality(args, gpu_ids)
    run_vbench(args, gpu_ids)
    aggregate(args)
    print(f"[summary] {args.output_dir / 'metrics' / 'temporal_delta_l8c64_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
