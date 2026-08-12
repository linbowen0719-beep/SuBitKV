#!/usr/bin/env python3
"""Train and evaluate per-level residual-normalized L8C64 RVQ.

This experiment keeps the runtime RVQ format compatible with ``kvrvq-i4`` but
changes the codebook training objective: before every RVQ level, the current
residual is normalized by dimension-wise mean/std. Two deployment modes are
trained from the same normalized residual codebooks:

1. dynamic: runtime computes residual mean/std from the current cache span and
   stores those stats in the packed cache.
2. static: runtime uses train-split residual mean/std stored in the codebook.

Example:
tmux new-session -d -s residnorm_l8c64 \
  'cd /home/bowen/Quant-VideoGen && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   /home/bowen/anaconda3/envs/hw8/bin/python scripts/run_residnorm_rvq_bench.py \
     --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
     --gpus 4,5,6,7 \
     --wait-for-gpus \
     |& tee /data/bowen/t2v_kv_bench_moviegen_20train_10test/logs/residnorm_l8c64_main.log'
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

from run_rvq_diag import (
    nearest_torch,
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
LOGICAL_DIM = 256
BF16_BYTES = 2
CACHED_FRAMES = 40
TOKENS_PER_FRAME = 880
CHUNK_SIZE = 4
VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
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


def dynamic_dir(root: Path, tensor: str) -> Path:
    return root / f"residnorm_dynamic_{tensor.lower()}_l8c64_codebooks"


def static_dir(root: Path, tensor: str) -> Path:
    return root / f"residnorm_static_{tensor.lower()}_l8c64_codebooks"


def codebook_path(root: Path, tensor: str, mode: str, layer: int, head: int) -> Path:
    base = dynamic_dir(root, tensor) if mode == "dynamic" else static_dir(root, tensor)
    return base / f"layer_{layer}_head_{head}.pt"


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads((root / "prompt_split.json").read_text(encoding="utf-8"))
    return list(payload[split])


def layer_file(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def sample_tensor_vectors(
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
        payload = torch.load(layer_file(root, "train", int(item["prompt_id"]), layer), map_location="cpu", weights_only=False)
        data = payload[tensor][:, head].reshape(-1, payload[tensor].shape[-1]).float()
        gen = torch.Generator(device="cpu").manual_seed(seed + layer * 100000 + head * 1000 + int(item["prompt_id"]))
        take = min(per_prompt, data.shape[0], remaining)
        ids = torch.randperm(data.shape[0], generator=gen)[:take]
        samples.append(data[ids])
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
    out_dynamic = codebook_path(args.output_dir, tensor, "dynamic", layer, head)
    out_static = codebook_path(args.output_dir, tensor, "static", layer, head)
    if out_dynamic.exists() and out_static.exists() and not args.overwrite:
        print(f"[train-skip] tensor={tensor} layer={layer} head={head}", flush=True)
        return

    out_dynamic.parent.mkdir(parents=True, exist_ok=True)
    out_static.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = sample_tensor_vectors(args.output_dir, tensor, layer, head, args.sample_size, args.seed)
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = (x / scale).contiguous()

    codebooks = []
    means = []
    stds = []
    residual_mse = []
    backends = []
    print(
        f"[train] tensor={tensor} layer={layer} head={head} samples={x.shape[0]} "
        "backend priority: QuantVideo KMeans / FAISS GPU KMeans / sklearn MiniBatchKMeans",
        flush=True,
    )
    for level in range(LEVELS):
        mean = residual.float().mean(dim=0, keepdim=True)
        std = torch.sqrt(torch.mean((residual.float() - mean) ** 2, dim=0, keepdim=True)).clamp_min(1e-6)
        fit_residual = ((residual.float() - mean) / std).contiguous()
        centroids, ids, backend = train_kmeans_auto(
            fit_residual,
            CODEBOOK_SIZE,
            args.kmeans_max_iters,
            args.seed + layer * 100000 + head * 1000 + level,
            device,
        )
        assigned = centroids[ids].float() * std + mean
        residual = (residual.float() - assigned).contiguous()
        mse = float((residual * residual).mean().item())
        print(
            f"[train] tensor={tensor} layer={layer} head={head} level={level + 1} "
            f"backend={backend} residual_mse={mse:.8g}",
            flush=True,
        )
        codebooks.append(centroids.float().cpu())
        means.append(mean.squeeze(0).float().cpu())
        stds.append(std.squeeze(0).float().cpu())
        residual_mse.append(mse)
        backends.append(backend)

    common_config = {
        "layer": layer,
        "head": head,
        "levels": LEVELS,
        "codebook_size": CODEBOOK_SIZE,
        "sample_size": int(x.shape[0]),
        "kmeans_max_iters": args.kmeans_max_iters,
        "backend_priority": ["QuantVideo KMeans", "FAISS GPU KMeans", "sklearn MiniBatchKMeans"],
        "backends": backends,
        "residual_mse": residual_mse,
        "normalization": "initial per-vector RMS, then per-level residual mean/std",
        "tensor": tensor,
        "key_type": "pre_rope" if tensor == "K" else None,
        "value_type": "pre_rope" if tensor == "V" else None,
    }
    cb = torch.stack(codebooks, dim=0).float()
    dynamic_payload = {
        "codebooks": cb,
        "residual_norm_mode": "dynamic",
        "config": {**common_config, "residual_norm_mode": "dynamic"},
    }
    static_payload = {
        "codebooks": cb,
        "residual_norm_mode": "static",
        "residual_mean": torch.stack(means, dim=0).half(),
        "residual_std": torch.stack(stds, dim=0).half(),
        "config": {**common_config, "residual_norm_mode": "static", "residual_stat_source": "train_split"},
    }
    torch.save(dynamic_payload, out_dynamic)
    torch.save(static_payload, out_static)
    print(f"[train-done] tensor={tensor} layer={layer} head={head}", flush=True)


def train_shard(args: argparse.Namespace) -> None:
    tasks = [(tensor, layer, head) for tensor in ("K", "V") for layer in range(LAYERS) for head in range(HEADS)]
    assigned = tasks[args.shard_id :: max(1, args.num_shards)]
    for tensor, layer, head in assigned:
        train_one(args, tensor, layer, head)


def gpu_free_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    result: dict[int, int] = {}
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
            print(f"[gpu-ready] {gpu_ids} free={{{', '.join(f'{g}: {free.get(g)}' for g in gpu_ids)}}}", flush=True)
            return
        print(
            f"[gpu-wait] need >= {min_free_mb} MiB free on {gpu_ids}; "
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
            env.setdefault("MASTER_PORT", str(29600 + int(gpu)))
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


def spawn_training(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_train:
        return
    pending = [
        (tensor, layer, head)
        for tensor in ("K", "V")
        for layer in range(LAYERS)
        for head in range(HEADS)
        if args.overwrite
        or not codebook_path(args.output_dir, tensor, "dynamic", layer, head).exists()
        or not codebook_path(args.output_dir, tensor, "static", layer, head).exists()
    ]
    if not pending:
        print("[train] all residual-normalized codebooks already exist", flush=True)
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
                "log": args.output_dir / "logs" / f"residnorm_train_gpu{gpu}.log",
            }
        )
    start = time.perf_counter()
    run_process_queue(jobs, gpu_ids, poll_s=30)
    elapsed = time.perf_counter() - start
    append_csv(
        args.output_dir / "metrics" / "residnorm_l8c64_training_time.csv",
        [
            {
                "phase": "train_KV_residual_norm_l8c64",
                "wall_time_s": elapsed,
                "gpus": ",".join(str(g) for g in gpu_ids),
                "sample_size": args.sample_size,
                "kmeans_max_iters": args.kmeans_max_iters,
            }
        ],
    )


def settings(root: Path) -> list[dict[str, Any]]:
    base = [
        "--setting",
        "kvrvq-i4",
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        "256",
        "--kv_quant_factor",
        "1",
    ]
    out = []
    for mode in ("dynamic", "static"):
        k_dir = dynamic_dir(root, "K") if mode == "dynamic" else static_dir(root, "K")
        v_dir = dynamic_dir(root, "V") if mode == "dynamic" else static_dir(root, "V")
        common = [
            *base,
            "--k_rvq_codebook_dir",
            str(k_dir),
            "--v_rvq_codebook_dir",
            str(v_dir),
        ]
        title = "Dynamic" if mode == "dynamic" else "Static"
        out.append(
            {
                "label": f"ResidNorm-{title}-RVQ-L8C64-INT4-perchunk",
                "dir": f"residnorm_{mode}_rvq_l8c64_int4_perchunk",
                "extra": common,
                "mode": mode,
                "retained_bf16_frames": 0,
            }
        )
        out.append(
            {
                "label": f"ResidNorm-{title}-RVQ-L8C64-INT4-perchunk-first1-recent1",
                "dir": f"residnorm_{mode}_rvq_l8c64_int4_perchunk_first1_recent1",
                "extra": [*common, "--kv_quant_skip_chunk_policy", "first", "--kv_quant_recent_policy", "nearest1"],
                "mode": mode,
                "retained_bf16_frames": 2 * CHUNK_SIZE,
            }
        )
    return out


def build_video_cmd(args: argparse.Namespace, spec: dict[str, Any]) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        str(args.num_prompts),
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


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    except Exception:
        return False
    return len(rows) == expected_rows and all(Path(row["video_path"]).exists() for row in rows)


def run_videos(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_video:
        return
    jobs = []
    for spec in settings(args.output_dir):
        metrics_path = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if metrics_complete(metrics_path, args.num_prompts) and not args.overwrite:
            print(f"[video-skip] {spec['label']}", flush=True)
            continue
        jobs.append(
            {
                "name": f"video {spec['label']}",
                "cmd": build_video_cmd(args, spec),
                "log": args.output_dir / "logs" / f"residnorm_video_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, gpu_ids, poll_s=30)


def run_quality(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_quality:
        return
    reference_dir = args.output_dir / "videos" / "bf16"
    jobs = []
    for spec in settings(args.output_dir):
        out_json = args.output_dir / "vs_bf16_residnorm" / f"{spec['dir']}.json"
        out_csv = args.output_dir / "vs_bf16_residnorm" / f"{spec['dir']}.csv"
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
                "log": args.output_dir / "logs" / f"residnorm_quality_{spec['dir']}.log",
            }
        )
    run_process_queue(jobs, gpu_ids, poll_s=10)


def read_metrics_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_vbench(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_vbench:
        return
    jobs = []
    expected = []
    for spec in settings(args.output_dir):
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        prompt_json = args.output_dir / "vbench_residnorm" / f"prompts_{spec['dir']}.json"
        prompt_json.parent.mkdir(parents=True, exist_ok=True)
        prompt_map = {}
        for row in read_metrics_rows(metrics_csv):
            video_path = Path(row["video_path"])
            if not video_path.is_absolute():
                video_path = (REPO_ROOT / video_path).resolve()
            prompt_map[str(video_path)] = row["prompt"]
        prompt_json.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
        for dim in VBENCH_DIMS:
            result_dir = args.output_dir / "vbench_residnorm" / spec["dir"] / dim
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
                    "log": args.output_dir / "logs" / f"residnorm_vbench_{spec['dir']}_{dim}.log",
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


def full_kv_mib(frames: int = CACHED_FRAMES) -> float:
    bytes_ = 2 * LAYERS * HEADS * frames * TOKENS_PER_FRAME * LOGICAL_DIM * BF16_BYTES
    return bytes_ / 1024**2


def rvq_l8c64_int4_mib(compressed_frames: int, retained_bf16_frames: int) -> float:
    vectors = LAYERS * HEADS * compressed_frames * TOKENS_PER_FRAME
    index_bits = math.ceil(math.log2(CODEBOOK_SIZE))
    one_index = vectors * LEVELS * index_bits / 8.0
    one_scale = vectors / 2.0
    one_codebook = LAYERS * HEADS * LEVELS * CODEBOOK_SIZE * LOGICAL_DIM / 2.0
    one_codebook_meta = LAYERS * HEADS * LEVELS * CODEBOOK_SIZE * BF16_BYTES
    compressed = 4 * one_index + 4 * one_scale + 2 * one_codebook + 2 * one_codebook_meta
    retained = full_kv_mib(retained_bf16_frames) * 1024**2
    return (compressed + retained) / 1024**2


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(dict.fromkeys(k for row in rows for k in row.keys())) if rows else ["status"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def aggregate(args: argparse.Namespace) -> None:
    runtime_rows = []
    quality_rows = []
    vbench_rows = []
    summary_rows = []
    specs = settings(args.output_dir)
    measured_perchunk: dict[str, float] = {}
    baseline_perchunk_mib = rvq_l8c64_int4_mib(CACHED_FRAMES, 0)
    full_mib = full_kv_mib(CACHED_FRAMES)
    for spec in specs:
        if int(spec.get("retained_bf16_frames", 0)) != 0:
            continue
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        if not metrics_csv.exists():
            continue
        rows = read_metrics_rows(metrics_csv)
        vals = [as_float(row.get("kv_cache_mem_mb")) for row in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            measured_perchunk[str(spec.get("mode", ""))] = sum(vals) / len(vals)
    for spec in specs:
        metrics_csv = args.output_dir / "videos" / spec["dir"] / "metrics.csv"
        rows = read_metrics_rows(metrics_csv) if metrics_csv.exists() else []

        def vals(key: str, drop0: bool = False) -> list[float]:
            src = rows[1:] if drop0 and len(rows) > 1 else rows
            return [v for v in (as_float(row.get(key)) for row in src) if v is not None]

        measured_kv = mean(vals("kv_cache_mem_mb"))
        measured_ratio = mean(vals("compression_ratio"))
        retained_frames = int(spec.get("retained_bf16_frames", 0))
        mode = str(spec.get("mode", ""))
        extra_metadata = measured_perchunk.get(mode, baseline_perchunk_mib) - baseline_perchunk_mib
        paper_kv = rvq_l8c64_int4_mib(CACHED_FRAMES - retained_frames, retained_frames) + extra_metadata
        runtime = {
            "setting": spec["label"],
            "videos": len(rows),
            "e2e_mean_s": mean(vals("e2e_time_s")),
            "e2e_drop0_s": mean(vals("e2e_time_s", True)),
            "denoise_mean_s": mean(vals("denoise_time_s")),
            "quant_drop0_s": mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": paper_kv,
            "compression_ratio": full_mib / paper_kv if paper_kv else None,
            "measured_kv_cache_mem_mb": measured_kv,
            "measured_compression_ratio": measured_ratio,
            "retained_bf16_frames": retained_frames,
        }
        runtime_rows.append(runtime)

        quality_json = args.output_dir / "vs_bf16_residnorm" / f"{spec['dir']}.json"
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
        vbench = parse_vbench_result(spec["label"], args.output_dir / "vbench_residnorm" / spec["dir"])
        vbench_rows.append(vbench)
        summary_rows.append({**runtime, **{k: v for k, v in quality.items() if k != "setting"}, **{k: v for k, v in vbench.items() if k != "setting"}})

    write_csv(args.output_dir / "metrics" / "residnorm_l8c64_runtime.csv", runtime_rows)
    write_csv(args.output_dir / "metrics" / "residnorm_l8c64_vs_bf16.csv", quality_rows)
    write_csv(args.output_dir / "metrics" / "residnorm_l8c64_vbench.csv", vbench_rows)
    write_csv(args.output_dir / "metrics" / "residnorm_l8c64_summary.csv", summary_rows)
    report_lines = [
        "# Residual-Normalized L8C64 INT4 RVQ",
        "",
        "- Base config: T2V MovieGen 20 train / 10 test, seed 42, num_frames=173, num_chunk=11.",
        "- Codebook: L8C64, runtime INT4 codebooks + UINT4 initial RMS scales.",
        "- Dynamic: per-level residual mean/std computed from the current cache span.",
        "- Static: per-level residual mean/std from train split stored in codebook payload.",
        "",
        "See `metrics/residnorm_l8c64_summary.csv` for runtime, PSNR/SSIM/LPIPS, and VBench.",
    ]
    report_path = args.output_dir / "reports" / "residnorm_l8c64_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


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
    print(f"[summary] {args.output_dir / 'metrics' / 'residnorm_l8c64_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
