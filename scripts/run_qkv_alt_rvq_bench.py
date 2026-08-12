#!/usr/bin/env python3
"""Train/evaluate L8C64 INT4 QKV-aware RVQ.

This runner is intentionally self-contained and resumable. It trains standard
runtime RVQ codebooks, so generation still goes through run_small_eval.py and
quant_videogen.rvq.rvq_encode_tensor_cache.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH_BIN = Path("/home/bowen/anaconda3/bin/vbench")

LAYERS = 30
HEADS = 24
LOGICAL_KV_DIM = 256
TOKENS_PER_FRAME = 880
DEFAULT_NUM_PROMPTS = 10
VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]


@dataclass(frozen=True)
class Setting:
    label: str
    method: str
    video_subdir: str
    k_dir: Path
    v_dir: Path
    skip_first: bool = False
    recent1: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--variant", default="l8c64", choices=["l8c64"])
    p.add_argument("--levels", type=int, default=8)
    p.add_argument("--codebook-size", type=int, default=64)
    p.add_argument("--storage", default="int4", choices=["int4"])
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--objective-query-sample", type=int, default=128)
    p.add_argument("--query-sample", type=int, default=128)
    p.add_argument("--key-sample", type=int, default=2048)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=8192)
    p.add_argument("--rvq-batch-size", type=int, default=4096)
    p.add_argument("--kv-quant-factor", type=int, default=1)
    p.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH_BIN)
    p.add_argument("--min-free-gpu-mb", type=int, default=35000)
    p.add_argument("--gpu-poll-s", type=int, default=60)
    p.add_argument("--video-workers", type=int, default=8)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-offline-eval", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-vs-bf16", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    # Worker-only args.
    p.add_argument("--worker", choices=["train", "offline-eval"], default="")
    p.add_argument("--tensor", choices=["K", "V"], default="K")
    p.add_argument("--task-values", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-codebook-dir", type=Path, default=None)
    p.add_argument("--current-k-dir", type=Path, default=None)
    p.add_argument("--current-v-dir", type=Path, default=None)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--prompt-ids", default="")
    p.add_argument("--setting-name", default="")
    p.add_argument("--metrics-rows-out", type=Path, default=None)
    return p.parse_args()


def parse_gpus(raw: str) -> list[int]:
    gpus = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def clean_name(value: str) -> str:
    return value.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def mkdirs(root: Path) -> None:
    for rel in [
        "metrics",
        "reports",
        "logs/qkv_alt",
        "status/qkv_alt",
        "videos",
        "tmp/qkv_alt",
        "vs_bf16_qkv_alt",
        "vbench_qkv_alt",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def baseline_k_dir(root: Path) -> Path:
    return root / "sweep_k_l8c64_codebooks"


def baseline_v_dir(root: Path) -> Path:
    return root / "sweep_v_l8c64_codebooks"


def round_k_dir(root: Path, round_idx: int) -> Path:
    return root / f"qkvalt_l8c64_round{round_idx}_k_codebooks"


def round_v_dir(root: Path, round_idx: int) -> Path:
    return root / f"qkvalt_l8c64_round{round_idx}_v_codebooks"


def final_k_dir(root: Path) -> Path:
    return root / "qkvalt_l8c64_k_codebooks"


def final_v_dir(root: Path) -> Path:
    return root / "qkvalt_l8c64_v_codebooks"


def codebook_path(cb_dir: Path, layer: int, head: int) -> Path:
    return cb_dir / f"layer_{layer}_head_{head}.pt"


def all_tasks() -> list[int]:
    return [layer * 100 + head for layer in range(LAYERS) for head in range(HEADS)]


def codebook_complete(cb_dir: Path) -> bool:
    return all(codebook_path(cb_dir, layer, head).exists() for layer in range(LAYERS) for head in range(HEADS))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    path = root / "prompt_split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload[split]


def layer_file(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def sample_tensor_vectors_with_indices(
    root: Path,
    layer: int,
    head: int,
    tensor_name: str,
    sample_size: int,
    seed: int,
) -> tuple[torch.Tensor, list[tuple[dict[str, Any], torch.Tensor]]]:
    items = split_items(root, "train")
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    samples: list[torch.Tensor] = []
    records: list[tuple[dict[str, Any], torch.Tensor]] = []
    for item in items:
        payload = torch.load(layer_file(root, "train", int(item["prompt_id"]), layer), map_location="cpu", weights_only=False)
        tensor = payload[tensor_name][:, head].reshape(-1, payload[tensor_name].shape[-1])
        gen = torch.Generator(device="cpu").manual_seed(seed + layer * 100000 + head * 1000 + int(item["prompt_id"]))
        take = min(per_prompt, tensor.shape[0])
        indices = torch.randperm(tensor.shape[0], generator=gen)[:take]
        samples.append(tensor[indices].float())
        records.append((item, indices))
    x = torch.cat(samples, dim=0)
    if x.shape[0] > sample_size:
        # Keep order stable with records by trimming from the tail. The default
        # 20-train/50000 setup lands exactly at sample_size.
        keep = sample_size
        x = x[:keep]
        trimmed: list[tuple[dict[str, Any], torch.Tensor]] = []
        remaining = keep
        for item, indices in records:
            if remaining <= 0:
                break
            take = min(remaining, indices.numel())
            trimmed.append((item, indices[:take]))
            remaining -= take
        records = trimmed
    return x.contiguous(), records


def nearest_torch(x: torch.Tensor, centroids: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    ids = []
    c = centroids.float()
    c_sq = (c * c).sum(dim=1)
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].float()
        dist = (xb * xb).sum(dim=1, keepdim=True) + c_sq[None, :] - 2.0 * xb @ c.T
        ids.append(torch.argmin(dist, dim=1).cpu())
    return torch.cat(ids, dim=0).long()


def train_kmeans_weighted_torch(
    x: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    max_iters: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    x_cpu = x.detach().cpu().float().contiguous()
    x_dev = x_cpu.to(device=device, dtype=torch.float32)
    w_dev = weights.detach().to(device=device, dtype=torch.float32).reshape(-1).clamp_min(1e-8)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    init = torch.randperm(x_cpu.shape[0], generator=gen)[:k]
    if init.numel() < k:
        init = torch.randint(0, x_cpu.shape[0], (k,), generator=gen)
    centroids = x_dev[init.to(device)].clone()
    ids = torch.zeros(x_cpu.shape[0], dtype=torch.long)
    for _ in range(max(1, max_iters)):
        ids = nearest_torch(x_dev, centroids, batch_size=8192)
        ids_dev = ids.to(device)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros((k, 1), device=device, dtype=torch.float32)
        sums.index_add_(0, ids_dev, x_dev * w_dev[:, None])
        counts.index_add_(0, ids_dev, w_dev[:, None])
        empty = counts.squeeze(1) <= 0
        centroids = sums / counts.clamp_min(1e-8)
        if empty.any():
            repl = torch.randperm(x_cpu.shape[0], generator=gen)[: int(empty.sum().item())]
            centroids[empty] = x_dev[repl.to(device)]
    ids = nearest_torch(x_dev, centroids, batch_size=8192)
    return centroids.detach().cpu().float(), ids.long(), "torch Weighted KMeans"


def weighted_centroids_from_ids(
    x: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    k: int,
    fallback_fit_centroids: torch.Tensor,
    transform: torch.Tensor | None,
) -> torch.Tensor:
    x_f = x.float()
    ids_l = ids.long()
    w = weights.detach().cpu().float().reshape(-1, 1).clamp_min(1e-8)
    sums = torch.zeros((k, x_f.shape[1]), dtype=torch.float32)
    counts = torch.zeros((k, 1), dtype=torch.float32)
    sums.index_add_(0, ids_l.cpu(), x_f.cpu() * w)
    counts.index_add_(0, ids_l.cpu(), w)
    centroids = sums / counts.clamp_min(1e-8)
    empty = counts.squeeze(1) <= 0
    if empty.any():
        fallback = fallback_fit_centroids.float().cpu()
        if transform is not None:
            fallback = fallback / transform.float().cpu().clamp_min(1e-6)
        centroids[empty] = fallback[empty]
    return centroids.contiguous()


def load_codebook(cb_dir: Path, layer: int, head: int) -> dict[str, Any]:
    path = codebook_path(cb_dir, layer, head)
    if not path.exists():
        raise FileNotFoundError(f"Missing codebook: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


@torch.no_grad()
def rvq_encode_reconstruct(x: torch.Tensor, payload: dict[str, Any], device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    codebooks = payload["codebooks"] if isinstance(payload, dict) else payload
    flat = x.reshape(-1, x.shape[-1]).float().to(device)
    scale = torch.sqrt(torch.mean(flat * flat, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = flat / scale
    recon_norm = torch.zeros_like(residual)
    all_ids = []
    for level in range(codebooks.shape[0]):
        centroids = codebooks[level].to(device=device, dtype=torch.float32)
        ids_cpu = nearest_torch(residual, centroids, batch_size=batch_size)
        ids = ids_cpu.to(device)
        recon_norm += centroids[ids]
        residual = residual - centroids[ids]
        all_ids.append(ids_cpu.to(torch.uint8 if codebooks.shape[1] <= 256 else torch.int16))
    recon = recon_norm * scale
    idx = torch.stack(all_ids, dim=0).reshape(codebooks.shape[0], *x.shape[:-1]).cpu()
    return idx, recon.reshape_as(x).detach().cpu()


@torch.no_grad()
def reconstruct_head_tensor(
    tensor: torch.Tensor,
    cb_dir: Path,
    layer: int,
    head: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    payload = load_codebook(cb_dir, layer, head)
    _ids, recon = rvq_encode_reconstruct(tensor.contiguous(), payload, device, batch_size)
    return recon.float()


@torch.no_grad()
def qkv_k_metric_and_weights(
    args: argparse.Namespace,
    layer: int,
    head: int,
    records: list[tuple[dict[str, Any], torch.Tensor]],
    current_k_dir: Path,
    current_v_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    diag_sum: torch.Tensor | None = None
    weight_chunks: list[torch.Tensor] = []
    count = 0
    for item, indices in records:
        payload = torch.load(layer_file(args.output_dir, "train", int(item["prompt_id"]), layer), map_location="cpu", weights_only=False)
        q = payload["Q"][:, head].float().reshape(-1, payload["Q"].shape[-1])
        k = payload["K"][:, head].float().contiguous()
        v = payload["V"][:, head].float().contiguous()
        khat = reconstruct_head_tensor(k, current_k_dir, layer, head, device, args.eval_batch_size).reshape(-1, k.shape[-1]).to(device)
        vhat = reconstruct_head_tensor(v, current_v_dir, layer, head, device, args.eval_batch_size).reshape(-1, v.shape[-1]).to(device)

        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + int(item["prompt_id"]) * 1000000 + layer * 10000 + head * 100 + 421
        )
        take = min(int(args.objective_query_sample), q.shape[0])
        q_idx = torch.randperm(q.shape[0], generator=gen)[:take]
        q_sample = q[q_idx].to(device)
        logits_hat = q_sample @ khat.T / math.sqrt(k.shape[-1])
        attn_hat = F.softmax(logits_hat.float(), dim=-1)
        ohat = attn_hat @ vhat.float()

        idx = indices.long().to(device)
        selected_v = vhat.index_select(0, idx)
        selected_attn = attn_hat.index_select(1, idx)
        contrast = ((selected_v[None, :, :].float() - ohat[:, None, :].float()) ** 2).sum(dim=-1)
        q_sq = q_sample.float() ** 2
        metric = ((selected_attn.float() ** 2)[:, :, None] * contrast[:, :, None] * q_sq[:, None, :]).mean(dim=0)
        k_selected = k.reshape(-1, k.shape[-1]).index_select(0, indices.long()).float().to(device)
        scale_sq = torch.mean(k_selected * k_selected, dim=-1, keepdim=True).clamp_min(1e-8)
        metric = metric * scale_sq
        metric_cpu = metric.detach().cpu().float().clamp_min(1e-12)
        diag_sum = metric_cpu.sum(dim=0) if diag_sum is None else diag_sum + metric_cpu.sum(dim=0)
        weight_chunks.append(metric_cpu.mean(dim=1))
        count += metric_cpu.shape[0]
        del payload, q, k, v, khat, vhat, q_sample, logits_hat, attn_hat, ohat, metric
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if diag_sum is None or count == 0:
        raise RuntimeError("No samples for qkv_k metric")
    diag = (diag_sum / max(1, count)).clamp_min(1e-10)
    diag = diag / diag.mean().clamp_min(1e-10)
    transform = torch.sqrt(diag).float().contiguous()
    weights = torch.cat(weight_chunks, dim=0).float().clamp_min(1e-10)
    weights = weights / weights.mean().clamp_min(1e-10)
    return transform, weights.contiguous()


@torch.no_grad()
def qkv_v_weights(
    args: argparse.Namespace,
    layer: int,
    head: int,
    records: list[tuple[dict[str, Any], torch.Tensor]],
    current_k_dir: Path,
    device: torch.device,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for item, indices in records:
        payload = torch.load(layer_file(args.output_dir, "train", int(item["prompt_id"]), layer), map_location="cpu", weights_only=False)
        q = payload["Q"][:, head].float().reshape(-1, payload["Q"].shape[-1])
        k = payload["K"][:, head].float().contiguous()
        khat = reconstruct_head_tensor(k, current_k_dir, layer, head, device, args.eval_batch_size).reshape(-1, k.shape[-1]).to(device)
        gen = torch.Generator(device="cpu").manual_seed(
            args.seed + int(item["prompt_id"]) * 1000000 + layer * 10000 + head * 100 + 313
        )
        take = min(int(args.objective_query_sample), q.shape[0])
        q_idx = torch.randperm(q.shape[0], generator=gen)[:take]
        q_sample = q[q_idx].to(device)
        logits = q_sample @ khat.T / math.sqrt(k.shape[-1])
        attn = F.softmax(logits.float(), dim=-1)
        w_all = (attn * attn).mean(dim=0).detach().cpu()
        selected = w_all[indices.long()].float()
        selected = selected / selected.mean().clamp_min(1e-10)
        chunks.append(selected)
        del payload, q, k, khat, q_sample, logits, attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    weights = torch.cat(chunks, dim=0).float().clamp_min(1e-10)
    return (weights / weights.mean().clamp_min(1e-10)).contiguous()


def train_one_codebook(args: argparse.Namespace, layer: int, head: int) -> None:
    out_path = codebook_path(args.out_codebook_dir, layer, head)
    if out_path.exists() and not args.overwrite:
        print(f"[train-skip] tensor={args.tensor} layer={layer} head={head}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x, records = sample_tensor_vectors_with_indices(
        args.output_dir,
        layer,
        head,
        args.tensor,
        args.sample_size,
        args.seed,
    )
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = (x / scale).contiguous()

    metric_transform = None
    if args.tensor == "K":
        metric_transform, sample_weights = qkv_k_metric_and_weights(
            args,
            layer,
            head,
            records,
            args.current_k_dir,
            args.current_v_dir,
            device,
        )
        objective = "qkv_k"
    else:
        sample_weights = qkv_v_weights(args, layer, head, records, args.current_k_dir, device)
        objective = "qkv_v"

    codebooks = []
    residual_mse = []
    backends = []
    print(
        f"[train] tensor={args.tensor} objective={objective} layer={layer} head={head} "
        f"samples={residual.shape[0]} levels={args.levels} codebook={args.codebook_size}",
        flush=True,
    )
    for level in range(args.levels):
        if metric_transform is not None:
            fit_residual = residual.float() * metric_transform.float()[None, :]
            transform = metric_transform
        else:
            fit_residual = residual.float()
            transform = None
        try:
            fit_centroids, ids, backend = train_kmeans_weighted_torch(
                fit_residual,
                sample_weights,
                args.codebook_size,
                args.kmeans_max_iters,
                args.seed + layer * 100000 + head * 1000 + level,
                device,
            )
        except Exception as exc:
            print(f"[train] weighted torch KMeans failed: {exc}", flush=True)
            raise
        centroids = weighted_centroids_from_ids(
            residual,
            ids,
            sample_weights,
            args.codebook_size,
            fit_centroids,
            transform,
        )
        assigned = centroids[ids].float()
        residual = (residual.float() - assigned).contiguous()
        mse = float((residual * residual).mean().item())
        codebooks.append(centroids)
        residual_mse.append(mse)
        backends.append(backend)
        print(f"[train] tensor={args.tensor} layer={layer} head={head} level={level + 1} residual_mse={mse:.8g}", flush=True)

    payload = {
        "codebooks": torch.stack(codebooks, dim=0).float(),
        "config": {
            "layer": layer,
            "head": head,
            "levels": args.levels,
            "codebook_size": args.codebook_size,
            "sample_size": int(x.shape[0]),
            "kmeans_max_iters": args.kmeans_max_iters,
            "backends": backends,
            "residual_mse": residual_mse,
            "normalization": "per-vector RMS",
            "train_objective": objective,
            "runtime_format": "standard_rvq_no_extra_assignment_codebooks",
            "current_k_dir": str(args.current_k_dir),
            "current_v_dir": str(args.current_v_dir),
            "tensor": args.tensor,
            "key_type": "pre_rope" if args.tensor == "K" else None,
            "value_type": "pre_prope" if args.tensor == "V" else None,
            "qkv_metric_transform_diag": metric_transform.float().cpu().tolist() if metric_transform is not None else None,
        },
    }
    torch.save(payload, out_path)
    print(f"[train-save] {out_path}", flush=True)


def train_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tasks = [int(x) for x in args.task_values.split(",") if x.strip()]
    for task in tasks:
        train_one_codebook(args, task // 100, task % 100)


def gpu_free_memory_mb() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    free: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                free[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return free


def wait_for_launch_gpu(gpu_ids: list[int], busy: set[int], min_free_mb: int, poll_s: int) -> int:
    while True:
        free = gpu_free_memory_mb()
        for gpu in gpu_ids:
            if gpu in busy:
                continue
            if not free or free.get(gpu, 0) >= min_free_mb:
                return gpu
        print(f"[gpu-wait] no GPU with >= {min_free_mb} MiB free; sleeping {poll_s}s", flush=True)
        time.sleep(poll_s)


def launch_train_stage(
    args: argparse.Namespace,
    tensor: str,
    out_dir: Path,
    current_k_dir: Path,
    current_v_dir: Path,
    gpu_ids: list[int],
    stage_name: str,
) -> float:
    if codebook_complete(out_dir) and not args.overwrite:
        print(f"[train-stage-skip] {stage_name} {out_dir}", flush=True)
        return 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in all_tasks() if args.overwrite or not codebook_path(out_dir, task // 100, task % 100).exists()]
    if not tasks:
        return 0.0
    procs = []
    start = time.perf_counter()
    for i, gpu in enumerate(gpu_ids):
        assigned = tasks[i::len(gpu_ids)]
        if not assigned:
            continue
        log_path = args.output_dir / "logs" / "qkv_alt" / f"{stage_name}_gpu{gpu}.log"
        cmd = [
            str(args.worker_python),
            str(Path(__file__).resolve()),
            "--worker",
            "train",
            "--output-dir",
            str(args.output_dir),
            "--levels",
            str(args.levels),
            "--codebook-size",
            str(args.codebook_size),
            "--sample-size",
            str(args.sample_size),
            "--kmeans-max-iters",
            str(args.kmeans_max_iters),
            "--objective-query-sample",
            str(args.objective_query_sample),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--seed",
            str(args.seed),
            "--tensor",
            tensor,
            "--out-codebook-dir",
            str(out_dir),
            "--current-k-dir",
            str(current_k_dir),
            "--current-v-dir",
            str(current_v_dir),
            "--gpu",
            str(gpu),
            "--task-values",
            ",".join(str(x) for x in assigned),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log = log_path.open("w", encoding="utf-8")
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        print(f"[train-launch] {stage_name} gpu={gpu} tasks={len(assigned)}", flush=True)
        procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
    failures = []
    for proc, log, log_path in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failures.append((rc, log_path))
    elapsed = time.perf_counter() - start
    if failures:
        msg = "\n".join(f"rc={rc} log={path}" for rc, path in failures)
        raise RuntimeError(f"Training stage {stage_name} failed:\n{msg}")
    return elapsed


def copy_codebook_tree(src: Path, dst: Path, overwrite: bool = False) -> None:
    if codebook_complete(dst) and not overwrite:
        return
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.glob("layer_*_head_*.pt"):
        target = dst / path.name
        if overwrite or not target.exists():
            shutil.copy2(path, target)


def train_alternating(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_train:
        return
    train_time_path = args.output_dir / "metrics" / "qkv_alt_training_time.csv"
    fields = ["phase", "round", "tensor", "wall_time_s", "gpus", "out_dir", "current_k_dir", "current_v_dir"]
    current_k = baseline_k_dir(args.output_dir)
    current_v = baseline_v_dir(args.output_dir)
    if not codebook_complete(current_k) or not codebook_complete(current_v):
        raise FileNotFoundError("Missing baseline L8C64 K/V codebooks")
    for round_idx in range(1, args.rounds + 1):
        out_k = round_k_dir(args.output_dir, round_idx)
        elapsed_k = launch_train_stage(args, "K", out_k, current_k, current_v, gpu_ids, f"round{round_idx}_k")
        append_csv(
            train_time_path,
            {
                "phase": "train",
                "round": round_idx,
                "tensor": "K",
                "wall_time_s": elapsed_k,
                "gpus": ",".join(str(g) for g in gpu_ids),
                "out_dir": str(out_k),
                "current_k_dir": str(current_k),
                "current_v_dir": str(current_v),
            },
            fields,
        )
        out_v = round_v_dir(args.output_dir, round_idx)
        elapsed_v = launch_train_stage(args, "V", out_v, out_k, current_v, gpu_ids, f"round{round_idx}_v")
        append_csv(
            train_time_path,
            {
                "phase": "train",
                "round": round_idx,
                "tensor": "V",
                "wall_time_s": elapsed_v,
                "gpus": ",".join(str(g) for g in gpu_ids),
                "out_dir": str(out_v),
                "current_k_dir": str(out_k),
                "current_v_dir": str(current_v),
            },
            fields,
        )
        current_k, current_v = out_k, out_v
    copy_codebook_tree(current_k, final_k_dir(args.output_dir), args.overwrite)
    copy_codebook_tree(current_v, final_v_dir(args.output_dir), args.overwrite)


def codebook_storage(storage: str) -> tuple[str, str]:
    if storage == "int4":
        return "int4", "uint4"
    return "bf16", "bf16"


@torch.no_grad()
def reconstruct_layer(target: torch.Tensor, layer: int, cb_dir: Path, storage: str, device: torch.device) -> torch.Tensor:
    from quant_videogen.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache

    codebook_dtype, scale_dtype = codebook_storage(storage)
    packed = rvq_encode_tensor_cache(
        target.to(device=device),
        layer_idx=layer,
        codebook_dir=cb_dir,
        batch_size=4096,
        compute_dtype="bf16",
        codebook_storage_dtype=codebook_dtype,
        scale_storage_dtype=scale_dtype,
    )
    return rvq_dequantize_tensor_cache(packed, target.dtype).detach().cpu()


def topk_overlap(logits: torch.Tensor, logits_hat: torch.Tensor, topk: int) -> float:
    k = min(int(topk), logits.shape[-1])
    a = torch.topk(logits, k=k, dim=-1).indices
    b = torch.topk(logits_hat, k=k, dim=-1).indices
    return float((a[:, :, None] == b[:, None, :]).any(dim=-1).float().mean().item())


@torch.no_grad()
def offline_eval_one(
    args: argparse.Namespace,
    method: str,
    setting: str,
    k_dir: Path,
    v_dir: Path,
    split: str,
    prompt_id: int,
    layer: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = torch.load(layer_file(args.output_dir, split, prompt_id, layer), map_location="cpu", weights_only=False)
    q = payload["Q"].float()
    k = payload["K"].float()
    v = payload["V"].float()
    k_hat = reconstruct_layer(k, layer, k_dir, args.storage, device).float()
    v_hat = reconstruct_layer(v, layer, v_dir, args.storage, device).float()

    k_err = k - k_hat
    v_err = v - v_hat
    k_mse = (k_err * k_err).mean(dim=(0, 2, 3))
    v_mse = (v_err * v_err).mean(dim=(0, 2, 3))
    k_cos = F.cosine_similarity(k, k_hat, dim=-1).mean(dim=(0, 2))
    v_cos = F.cosine_similarity(v, v_hat, dim=-1).mean(dim=(0, 2))

    n_tokens = k.shape[0] * k.shape[2]
    rows = []
    for head in range(HEADS):
        gen = torch.Generator(device="cpu").manual_seed(args.seed + prompt_id * 1000000 + layer * 10000 + head * 100)
        q_take = min(args.query_sample, n_tokens)
        k_take = min(args.key_sample, n_tokens)
        q_idx = torch.randperm(n_tokens, generator=gen)[:q_take]
        kv_idx = torch.randperm(n_tokens, generator=gen)[:k_take]
        q_full = q[:, head].reshape(-1, q.shape[-1]).to(device)
        k_full = k[:, head].reshape(-1, k.shape[-1]).to(device)
        kh_full = k_hat[:, head].reshape(-1, k.shape[-1]).to(device)
        v_full = v[:, head].reshape(-1, v.shape[-1]).to(device)
        vh_full = v_hat[:, head].reshape(-1, v.shape[-1]).to(device)
        q_s = q_full.index_select(0, q_idx.to(device))
        k_s = k_full.index_select(0, kv_idx.to(device))
        kh_s = kh_full.index_select(0, kv_idx.to(device))
        v_s = v_full.index_select(0, kv_idx.to(device))
        vh_s = vh_full.index_select(0, kv_idx.to(device))
        logits = q_s @ k_s.T / math.sqrt(k_s.shape[-1])
        logits_hat = q_s @ kh_s.T / math.sqrt(k_s.shape[-1])
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_q = F.log_softmax(logits_hat.float(), dim=-1)
        p = log_p.exp()
        p_hat = log_q.exp()
        av = p @ v_s.float()
        av_vhat = p @ vh_s.float()
        qkv = p_hat @ vh_s.float()
        rows.append(
            {
                "setting": setting,
                "method": method,
                "split": split,
                "prompt_id": prompt_id,
                "layer": layer,
                "head": head,
                "k_reconstruction_mse": float(k_mse[head].item()),
                "k_reconstruction_cosine": float(k_cos[head].item()),
                "v_reconstruction_mse": float(v_mse[head].item()),
                "v_reconstruction_cosine": float(v_cos[head].item()),
                "qk_logits_mse": float(((logits - logits_hat) ** 2).mean().item()),
                "attention_kl": float((p * (log_p - log_q)).sum(dim=-1).mean().item()),
                "topk_overlap": topk_overlap(logits, logits_hat, args.topk),
                "attn_output_mse": float(((av - av_vhat) ** 2).mean().item()),
                "qkv_output_mse": float(((av - qkv) ** 2).mean().item()),
                "qkv_output_cosine": float(F.cosine_similarity(av, qkv, dim=-1).mean().item()),
            }
        )
    return rows


def offline_eval_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tasks = [int(x) for x in args.task_values.split(",") if x.strip()]
    prompt_ids = [int(x) for x in args.prompt_ids.split(",") if x.strip()]
    rows = []
    if args.setting_name == "baseline":
        method = "baseline"
        setting = "RVQ-L8C64-INT4"
        k_dir = baseline_k_dir(args.output_dir)
        v_dir = baseline_v_dir(args.output_dir)
    else:
        method = "qkv_alt"
        setting = "QKVAlt-RVQ-L8C64-INT4"
        k_dir = final_k_dir(args.output_dir)
        v_dir = final_v_dir(args.output_dir)
    for layer in tasks:
        for prompt_id in prompt_ids:
            rows.extend(offline_eval_one(args, method, setting, k_dir, v_dir, args.split, prompt_id, layer, device))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.metrics_rows_out is None:
        raise ValueError("--metrics-rows-out is required for offline eval worker")
    write_csv(args.metrics_rows_out, rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "k_reconstruction_mse",
        "k_reconstruction_cosine",
        "v_reconstruction_mse",
        "v_reconstruction_cosine",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
        "attn_output_mse",
        "qkv_output_mse",
        "qkv_output_cosine",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["setting"], row["method"], row["split"]), []).append(row)
    out = []
    for (setting, method, split), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "setting": setting,
            "method": method,
            "split": split,
            "n": len(group),
        }
        for key in keys:
            vals = [float(r[key]) for r in group if r.get(key) not in (None, "", "null")]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / len(vals)
                result[f"{key}_mean"] = mean
                result[f"{key}_std"] = math.sqrt(var)
        out.append(result)
    return out


def train_test_gap(agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["setting"], r["method"], r["split"]): r for r in agg}
    metric_keys = [k[:-5] for k in agg[0] if k.endswith("_mean")] if agg else []
    rows = []
    for setting, method, _split in sorted({(r["setting"], r["method"], r["split"]) for r in agg}):
        train = by_key.get((setting, method, "train"))
        test = by_key.get((setting, method, "test"))
        if not train or not test:
            continue
        row = {"setting": setting, "method": method}
        for key in metric_keys:
            tr = train.get(f"{key}_mean")
            te = test.get(f"{key}_mean")
            if tr is not None and te is not None:
                row[f"train_{key}"] = tr
                row[f"test_{key}"] = te
                row[f"gap_{key}"] = float(te) - float(tr)
        rows.append(row)
    return rows


def launch_offline_eval(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_offline_eval:
        return
    out = args.output_dir / "metrics" / "qkv_alt_reconstruction.csv"
    rows_out = args.output_dir / "metrics" / "qkv_alt_reconstruction_rows.csv"
    if out.exists() and rows_out.exists() and not args.overwrite:
        print(f"[offline-skip] {out}", flush=True)
        return
    tmp = args.output_dir / "tmp" / "qkv_alt" / "offline_rows"
    if tmp.exists() and args.overwrite:
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    procs = []
    for setting_name in ("baseline", "qkv_alt"):
        for split in ("train", "test"):
            prompt_ids = [int(item["prompt_id"]) for item in split_items(args.output_dir, split)]
            for i, gpu in enumerate(gpu_ids):
                layers = list(range(LAYERS))[i::len(gpu_ids)]
                if not layers:
                    continue
                part = tmp / f"{setting_name}_{split}_gpu{gpu}.csv"
                if part.exists() and not args.overwrite:
                    continue
                log_path = args.output_dir / "logs" / "qkv_alt" / f"offline_{setting_name}_{split}_gpu{gpu}.log"
                cmd = [
                    str(args.worker_python),
                    str(Path(__file__).resolve()),
                    "--worker",
                    "offline-eval",
                    "--output-dir",
                    str(args.output_dir),
                    "--storage",
                    args.storage,
                    "--query-sample",
                    str(args.query_sample),
                    "--key-sample",
                    str(args.key_sample),
                    "--topk",
                    str(args.topk),
                    "--seed",
                    str(args.seed),
                    "--split",
                    split,
                    "--prompt-ids",
                    ",".join(str(x) for x in prompt_ids),
                    "--setting-name",
                    setting_name,
                    "--gpu",
                    str(gpu),
                    "--task-values",
                    ",".join(str(x) for x in layers),
                    "--metrics-rows-out",
                    str(part),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                log = log_path.open("w", encoding="utf-8")
                log.write("[cmd] " + " ".join(cmd) + "\n")
                log.flush()
                print(f"[offline-launch] setting={setting_name} split={split} gpu={gpu}", flush=True)
                procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
    failures = []
    for proc, log, log_path in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failures.append((rc, log_path))
    if failures:
        raise RuntimeError("offline eval failed:\n" + "\n".join(f"rc={rc} log={path}" for rc, path in failures))
    rows = []
    for part in sorted(tmp.glob("*.csv")):
        with part.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    write_csv(rows_out, rows)
    agg = aggregate_rows(rows)
    write_csv(out, agg)
    write_csv(args.output_dir / "metrics" / "qkv_alt_train_test_gap.csv", train_test_gap(agg))
    worst = sorted(rows, key=lambda r: float(r["qkv_output_mse"]), reverse=True)[:20]
    write_csv(args.output_dir / "metrics" / "qkv_alt_worst_qkv.csv", worst)


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return False
    return len(rows) == expected_rows and all(Path(r["video_path"]).exists() for r in rows if r.get("video_path"))


def qkv_settings(args: argparse.Namespace) -> list[Setting]:
    return [
        Setting(
            "QKVAlt-RVQ-L8C64-INT4-perchunk",
            "qkv_alt",
            "qkvalt_rvq_l8c64_int4_perchunk",
            final_k_dir(args.output_dir),
            final_v_dir(args.output_dir),
        ),
        Setting(
            "QKVAlt-RVQ-L8C64-INT4-perchunk-first1-recent1",
            "qkv_alt",
            "qkvalt_rvq_l8c64_int4_perchunk_first1_recent1",
            final_k_dir(args.output_dir),
            final_v_dir(args.output_dir),
            skip_first=True,
            recent1=True,
        ),
    ]


def baseline_settings(args: argparse.Namespace) -> list[Setting]:
    return [
        Setting(
            "RVQ-L8C64-INT4-perchunk",
            "baseline",
            "rvq_l8c64_int4_perchunk",
            baseline_k_dir(args.output_dir),
            baseline_v_dir(args.output_dir),
        ),
        Setting(
            "RVQ-L8C64-INT4-perchunk-first1-recent1",
            "baseline",
            "rvq_l8c64_int4_perchunk_skip_first_nearest_chunk",
            baseline_k_dir(args.output_dir),
            baseline_v_dir(args.output_dir),
            skip_first=True,
            recent1=True,
        ),
    ]


def build_video_cmd(args: argparse.Namespace, setting: Setting, out_dir: Path, prompt_start: int, num_prompts: int) -> list[str]:
    cmd = [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--prompt-start",
        str(prompt_start),
        "--num-prompts",
        str(num_prompts),
        "--output-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--setting",
        "kvrvq-i4",
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
        "--k_rvq_codebook_dir",
        str(setting.k_dir),
        "--v_rvq_codebook_dir",
        str(setting.v_dir),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        str(LOGICAL_KV_DIM),
        "--kv_quant_factor",
        str(args.kv_quant_factor),
    ]
    if setting.skip_first:
        cmd.extend(["--kv_quant_skip_chunk_policy", "first"])
    if setting.recent1:
        cmd.extend(["--kv_quant_recent_policy", "nearest1"])
    return cmd


def merge_shards(root: Path, setting: Setting, shard_dirs: list[Path]) -> None:
    final_dir = root / "videos" / setting.video_subdir
    final_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in shard_dirs:
        metrics = shard / "metrics.csv"
        if not metrics.exists():
            raise FileNotFoundError(metrics)
        with metrics.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
        for mp4 in shard.glob("*.mp4"):
            shutil.copy2(mp4, final_dir / mp4.name)
        cfg = shard / "config.json"
        if cfg.exists() and not (final_dir / "config.json").exists():
            shutil.copy2(cfg, final_dir / "config.json")
    rows.sort(key=lambda r: int(r["prompt_id"]))
    for row in rows:
        row["video_path"] = str(final_dir / f"{row['prompt_id']}-{row['seed']}.mp4")
    write_csv(final_dir / "metrics.csv", rows)


def run_videos(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_video:
        return
    pending: list[tuple[Setting, int, int, Path]] = []
    for setting in qkv_settings(args):
        final_metrics = args.output_dir / "videos" / setting.video_subdir / "metrics.csv"
        if metrics_complete(final_metrics, args.num_prompts) and not args.overwrite:
            print(f"[video-skip] {setting.label}", flush=True)
            continue
        shard_root = args.output_dir / "tmp" / "qkv_alt" / "video_shards" / setting.video_subdir
        if shard_root.exists() and args.overwrite:
            shutil.rmtree(shard_root)
        shard_root.mkdir(parents=True, exist_ok=True)
        workers = min(max(1, args.video_workers), len(gpu_ids), args.num_prompts)
        chunk = math.ceil(args.num_prompts / workers)
        for start in range(0, args.num_prompts, chunk):
            n = min(chunk, args.num_prompts - start)
            shard_dir = shard_root / f"part_{start}"
            if metrics_complete(shard_dir / "metrics.csv", n) and not args.overwrite:
                continue
            pending.append((setting, start, n, shard_dir))
    active: list[dict[str, Any]] = []
    failures = []
    next_idx = 0
    while next_idx < len(pending) or active:
        busy = {job["gpu"] for job in active}
        while next_idx < len(pending) and len(active) < args.video_workers:
            gpu = wait_for_launch_gpu(gpu_ids, busy, args.min_free_gpu_mb, args.gpu_poll_s)
            setting, start, n, shard_dir = pending[next_idx]
            next_idx += 1
            shard_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "logs" / "qkv_alt" / f"video_{setting.video_subdir}_start{start}_gpu{gpu}.log"
            cmd = build_video_cmd(args, setting, shard_dir, start, n)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["MASTER_PORT"] = str(29900 + gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[video-launch] {setting.label} start={start} n={n} gpu={gpu}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "log_path": log_path, "gpu": gpu, "setting": setting, "start": start})
            busy.add(gpu)
        time.sleep(20)
        still = []
        for job in active:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["log"].close()
            if rc != 0:
                failures.append({"setting": job["setting"].label, "start": job["start"], "rc": rc, "log": str(job["log_path"])})
            else:
                print(f"[video-done] {job['setting'].label} start={job['start']}", flush=True)
        active = still
    if failures:
        path = args.output_dir / "metrics" / "qkv_alt_video_failures.json"
        path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"video generation failed; see {path}")
    for setting in qkv_settings(args):
        shard_root = args.output_dir / "tmp" / "qkv_alt" / "video_shards" / setting.video_subdir
        if shard_root.exists():
            shards = sorted([p for p in shard_root.glob("part_*") if p.is_dir()], key=lambda p: int(p.name.split("_")[1]))
            if shards:
                merge_shards(args.output_dir, setting, shards)


def numeric(value: str | None) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_runtime(args: argparse.Namespace) -> None:
    rows = []
    for setting in baseline_settings(args) + qkv_settings(args):
        path = args.output_dir / "videos" / setting.video_subdir / "metrics.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        def vals(key: str, drop0: bool = False) -> list[float]:
            src = items[1:] if drop0 and len(items) > 1 else items
            return [v for v in (numeric(row.get(key)) for row in src) if v is not None]
        rows.append(
            {
                "setting": setting.label,
                "method": setting.method,
                "e2e_mean_s": mean(vals("e2e_time_s")),
                "e2e_drop0_s": mean(vals("e2e_time_s", True)),
                "denoise_mean_s": mean(vals("denoise_time_s")),
                "quant_drop0_s": mean(vals("quant_time_s", True)),
                "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
                "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
                "compression_ratio": mean(vals("compression_ratio")),
                "memory_note": "same RVQ format as baseline" if setting.method == "qkv_alt" else "",
            }
        )
    write_csv(args.output_dir / "metrics" / "qkv_alt_runtime.csv", rows)


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=run_env, stdout=log, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; log={log_path}")


def run_vs_bf16(args: argparse.Namespace, gpu_ids: list[int]) -> list[dict[str, Any]]:
    if args.skip_vs_bf16:
        return []
    rows = []
    bf16_dir = args.output_dir / "videos" / "bf16"
    for setting in baseline_settings(args) + qkv_settings(args):
        video_dir = args.output_dir / "videos" / setting.video_subdir
        metrics_csv = video_dir / "metrics.csv"
        if not metrics_csv.exists() or not bf16_dir.exists():
            continue
        out_json = args.output_dir / "vs_bf16_qkv_alt" / f"{setting.video_subdir}.json"
        out_csv = args.output_dir / "vs_bf16_qkv_alt" / f"{setting.video_subdir}.csv"
        if not out_json.exists() or args.overwrite:
            run_cmd(
                [
                    str(args.worker_python),
                    str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                    "--reference-dir",
                    str(bf16_dir),
                    "--test-dir",
                    str(video_dir),
                    "--metrics-csv",
                    str(metrics_csv),
                    "--setting",
                    setting.label,
                    "--output-json",
                    str(out_json),
                    "--output-csv",
                    str(out_csv),
                    "--device",
                    "cuda",
                ],
                args.output_dir / "logs" / "qkv_alt" / f"vs_bf16_{setting.video_subdir}.log",
                env={"CUDA_VISIBLE_DEVICES": str(gpu_ids[0])},
            )
        if out_json.exists():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            rows.append(
                {
                    "setting": setting.label,
                    "method": setting.method,
                    "psnr": payload.get("psnr_mean"),
                    "ssim": payload.get("ssim_mean"),
                    "lpips": payload.get("lpips_mean"),
                }
            )
    write_csv(args.output_dir / "metrics" / "qkv_alt_vs_bf16.csv", rows)
    return rows


def parse_vbench_result(result_dir: Path) -> dict[str, Any]:
    row = {dim: None for dim in VBENCH_DIMS}
    for path in sorted(result_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stack: list[Any] = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for dim in VBENCH_DIMS:
                    value = item.get(dim)
                    if isinstance(value, (int, float)):
                        row[dim] = value
                    elif isinstance(value, dict):
                        for key in ("score", "mean", "value"):
                            if isinstance(value.get(key), (int, float)):
                                row[dim] = value[key]
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return row


def run_vbench(args: argparse.Namespace, gpu_ids: list[int]) -> list[dict[str, Any]]:
    if args.skip_vbench or not args.vbench_bin.exists():
        return []
    rows = []
    for setting in baseline_settings(args) + qkv_settings(args):
        video_dir = args.output_dir / "videos" / setting.video_subdir
        if not (video_dir / "metrics.csv").exists():
            continue
        result_dir = args.output_dir / "vbench_qkv_alt" / setting.video_subdir
        result_dir.mkdir(parents=True, exist_ok=True)
        for dim in VBENCH_DIMS:
            dim_dir = result_dir / dim
            sentinel = dim_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                continue
            dim_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_cmd(
                    [
                        str(args.vbench_bin),
                        "evaluate",
                        "--ngpus",
                        str(min(4, len(gpu_ids))),
                        "--output_path",
                        str(dim_dir),
                        "--videos_path",
                        str(video_dir),
                        "--dimension",
                        dim,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(args.output_dir / "prompts_test.txt"),
                    ],
                    args.output_dir / "logs" / "qkv_alt" / f"vbench_{setting.video_subdir}_{dim}.log",
                    env={"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)},
                )
                sentinel.write_text("ok\n", encoding="utf-8")
            except Exception as exc:
                (dim_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                print(f"[vbench-failed] {setting.label} {dim}: {exc}", flush=True)
        parsed = parse_vbench_result(result_dir)
        parsed.update({"setting": setting.label, "method": setting.method})
        rows.append(parsed)
    write_csv(args.output_dir / "metrics" / "qkv_alt_vbench.csv", rows)
    return rows


def write_report(args: argparse.Namespace) -> None:
    report = args.output_dir / "reports" / "qkv_alt_l8c64_summary.md"
    lines = [
        "# QKVAlt L8C64 INT4 RVQ Benchmark",
        "",
        "## Setup",
        "",
        "- Split: MovieGen T2V 20 train / 10 test, seed 42.",
        "- Variant: L8C64, runtime codebook storage INT4, scale storage UINT4.",
        "- Runtime format: standard RVQ cache, no extra assignment codebooks.",
        "- Quantization schedule: per-chunk; second QKVAlt setting keeps first chunk and nearest cached chunk BF16.",
        "",
    ]
    for title, rel in [
        ("Offline Reconstruction / Attention", "metrics/qkv_alt_reconstruction.csv"),
        ("Train-Test Gap", "metrics/qkv_alt_train_test_gap.csv"),
        ("Runtime / Memory", "metrics/qkv_alt_runtime.csv"),
        ("PSNR / SSIM / LPIPS", "metrics/qkv_alt_vs_bf16.csv"),
        ("VBench", "metrics/qkv_alt_vbench.csv"),
    ]:
        path = args.output_dir / rel
        lines.extend([f"## {title}", ""])
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                headers = list(rows[0].keys())
                lines.append("| " + " | ".join(headers) + " |")
                lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    vals = []
                    for h in headers:
                        value = row.get(h, "")
                        try:
                            vals.append(f"{float(value):.4f}")
                        except (TypeError, ValueError):
                            vals.append(str(value))
                    lines.append("| " + " | ".join(vals) + " |")
            else:
                lines.append("_No rows._")
        else:
            lines.append(f"_Missing `{rel}`._")
        lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")


def smoke_check(args: argparse.Namespace) -> None:
    if args.skip_smoke:
        return
    needed = [
        baseline_k_dir(args.output_dir),
        baseline_v_dir(args.output_dir),
        args.output_dir / "kv_dump" / "train" / "0" / "layer_0.pt",
        args.output_dir / "prompts_test.txt",
        args.output_dir / "videos" / "bf16",
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError("Smoke check missing required inputs:\n" + "\n".join(missing))
    print("[smoke] required inputs exist", flush=True)


def main() -> None:
    args = parse_args()
    if args.worker == "train":
        train_worker(args)
        return
    if args.worker == "offline-eval":
        offline_eval_worker(args)
        return

    mkdirs(args.output_dir)
    gpu_ids = parse_gpus(args.gpus)
    smoke_check(args)
    train_alternating(args, gpu_ids)
    if not codebook_complete(final_k_dir(args.output_dir)) or not codebook_complete(final_v_dir(args.output_dir)):
        raise RuntimeError("Final QKVAlt K/V codebooks are incomplete")
    launch_offline_eval(args, gpu_ids)
    run_videos(args, gpu_ids)
    aggregate_runtime(args)
    run_vs_bf16(args, gpu_ids)
    run_vbench(args, gpu_ids)
    write_report(args)
    print(f"[done] report={args.output_dir / 'reports' / 'qkv_alt_l8c64_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
