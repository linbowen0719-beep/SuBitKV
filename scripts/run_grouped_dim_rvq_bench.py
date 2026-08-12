#!/usr/bin/env python3
"""Grouped-dimension RVQ benchmark for MovieGen T2V KV cache.

This trains deployable grouped RVQ codebooks and benchmarks them with the same
MovieGen 20-train / 10-test split used by the current T2V KV experiments.

Default benchmark settings:
  - L8C64 INT4 direct grouped RVQ
  - L8C64 INT4 Hadamard + grouped RVQ
  - L8C256 INT4 direct grouped RVQ
  - L8C256 INT4 Hadamard + grouped RVQ

Each 128-D internal K/V vector is split into 4 deterministic random groups of
32 dimensions.  Each group has its own RMS scale and its own RVQ assignments.
Runtime codebook storage is INT4 and scale storage is UINT4.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
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
INTERNAL_DIM = 128
LOGICAL_KV_DIM = 256
GROUPS = 4
GROUP_DIM = 32
VBENCH_DIMS = ["subject_consistency", "background_consistency", "imaging_quality", "aesthetic_quality"]


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int


@dataclass(frozen=True)
class Setting:
    variant: Variant
    mode: str

    @property
    def dirname(self) -> str:
        return f"grouped_{self.mode}_{self.variant.name}_int4_perchunk"

    @property
    def label(self) -> str:
        if self.mode == "hadamard":
            mode = "HadamardGrouped"
        elif self.mode == "hadamard_nonorm":
            mode = "NoNormHadamardGrouped"
        elif self.mode == "direct_nonorm":
            mode = "NoNormGrouped"
        else:
            mode = "Grouped"
        return f"{mode}-{self.variant.name.upper()}-INT4-perchunk"


def transform_mode(mode: str) -> str:
    return "hadamard" if str(mode).startswith("hadamard") else "direct"


def input_norm_mode(mode: str) -> str:
    return "none" if str(mode).endswith("_nonorm") else "rms"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH)
    p.add_argument("--configs", default="l8c64,l8c256")
    p.add_argument("--modes", default="direct,hadamard")
    p.add_argument("--tensors", default="K,V", help="Comma-separated tensors to train: K,V")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--max-used-mb", type=int, default=6000)
    p.add_argument("--train-max-used-mb", type=int, default=24000)
    p.add_argument("--max-util", type=int, default=20)
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--sample-size", type=int, default=12000)
    p.add_argument("--kmeans-max-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--num-layers", type=int, default=LAYERS)
    p.add_argument("--num-heads", type=int, default=HEADS)
    p.add_argument("--recon-sample", type=int, default=2048)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--master-port-base", type=int, default=30040)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-offline", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--worker", choices=["", "train", "eval"], default="")
    p.add_argument("--task-file", type=Path, default=None)
    return p.parse_args()


def parse_gpus(raw: str) -> list[int]:
    gpus = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def parse_variants(raw: str) -> list[Variant]:
    known = {
        "l8c64": Variant("l8c64", 8, 64),
        "l8c256": Variant("l8c256", 8, 256),
    }
    out = []
    for item in raw.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in known:
            raise ValueError(f"unknown config {key}")
        out.append(known[key])
    if not out:
        raise ValueError("--configs produced no variants")
    return out


def parse_modes(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        mode = item.strip().lower()
        if not mode:
            continue
        if mode not in {"direct", "hadamard", "direct_nonorm", "hadamard_nonorm"}:
            raise ValueError(f"unknown grouped mode {mode}")
        out.append(mode)
    if not out:
        raise ValueError("--modes produced no modes")
    return out


def parse_tensors(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        tensor = item.strip().upper()
        if not tensor:
            continue
        if tensor not in {"K", "V"}:
            raise ValueError(f"unknown tensor {tensor}")
        out.append(tensor)
    if not out:
        raise ValueError("--tensors produced no tensors")
    return out


def settings_from_args(args: argparse.Namespace) -> list[Setting]:
    return [Setting(v, mode) for v in parse_variants(args.configs) for mode in parse_modes(args.modes)]


def mkdirs(root: Path) -> None:
    for rel in [
        "logs/grouped_dim_rvq",
        "status/grouped_dim_rvq",
        "metrics/grouped_dim_rvq_rows",
        "videos_grouped_dim_rvq",
        "vs_bf16_grouped_dim_rvq",
        "vbench_grouped_dim_rvq",
        "reports",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    with (root / "prompt_split.json").open("r", encoding="utf-8") as f:
        return json.load(f)[split]


def layer_path(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def codebook_subdir(setting: Setting, tensor: str) -> str:
    return f"grouped_{setting.mode}_{tensor.lower()}_{setting.variant.name}_codebooks"


def codebook_path(root: Path, setting: Setting, tensor: str, layer: int, head: int) -> Path:
    return root / codebook_subdir(setting, tensor) / f"layer_{layer}_head_{head}.pt"


def video_dir(root: Path, setting: Setting) -> Path:
    return root / "videos_grouped_dim_rvq" / setting.dirname


def dim_permutation(seed: int, dim: int = INTERNAL_DIM) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(dim, generator=gen).long()


def hadamard_matrix(dim: int) -> torch.Tensor:
    if dim <= 0 or (dim & (dim - 1)):
        raise ValueError(f"Hadamard dimension must be a power of two, got {dim}")
    h = torch.tensor([[1.0]])
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return (h / math.sqrt(dim)).contiguous()


def sample_vectors(
    root: Path,
    split: str,
    prompt_ids: list[int],
    layer: int,
    head: int,
    tensor: str,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    chunks = []
    per_prompt = max(1, math.ceil(sample_size / max(1, len(prompt_ids))))
    for pid in prompt_ids:
        payload = torch.load(layer_path(root, split, pid, layer), map_location="cpu", weights_only=False)
        x = payload[tensor][:, head].reshape(-1, payload[tensor].shape[-1]).float()
        gen = torch.Generator(device="cpu").manual_seed(seed + pid * 1000003 + layer * 1009 + head)
        take = min(per_prompt, x.shape[0])
        chunks.append(x[torch.randperm(x.shape[0], generator=gen)[:take]])
    x = torch.cat(chunks, dim=0).contiguous()
    if x.shape[0] > sample_size:
        gen = torch.Generator(device="cpu").manual_seed(seed + layer * 100000 + head)
        x = x[torch.randperm(x.shape[0], generator=gen)[:sample_size]]
    return x.contiguous()


def to_group_space(x: torch.Tensor, mode: str, perm: torch.Tensor) -> torch.Tensor:
    y = x.float()
    if mode == "hadamard":
        y = y @ hadamard_matrix(y.shape[-1])
    y = y.index_select(-1, perm)
    return y.reshape(y.shape[0], GROUPS, GROUP_DIM).contiguous()


def from_group_space(grouped: torch.Tensor, mode: str, perm: torch.Tensor) -> torch.Tensor:
    y_perm = grouped.reshape(grouped.shape[0], INTERNAL_DIM).float()
    y = torch.empty_like(y_perm)
    y.index_copy_(1, perm, y_perm)
    if mode == "hadamard":
        y = y @ hadamard_matrix(y.shape[-1])
    return y.contiguous()


def normalize_groups(grouped: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.sqrt(torch.mean(grouped.float() * grouped.float(), dim=-1, keepdim=True)).clamp_min(1e-6)
    return (grouped.float() / scale).contiguous(), scale


def train_kmeans_auto(
    x: torch.Tensor,
    k: int,
    max_iters: int,
    seed: int,
    device: torch.device,
    gpu: int,
) -> tuple[torch.Tensor, torch.Tensor, str, list[str]]:
    errors = []
    try:
        c, ids, backend = train_kmeans_quantvideo(x, k, max_iters, device)
        return c, ids, backend, errors
    except Exception as exc:
        errors.append(f"QuantVideo KMeans failed: {exc}")
    try:
        c, ids, backend = train_kmeans_faiss(x, k, max_iters, gpu)
        return c, ids, backend, errors
    except Exception as exc:
        errors.append(f"FAISS GPU KMeans failed: {exc}")
    c, ids, backend = train_kmeans_sklearn(x, k, max_iters, seed)
    return c, ids, backend, errors


def train_one(args: argparse.Namespace, task: dict[str, Any], device: torch.device, gpu: int) -> dict[str, Any]:
    setting = Setting(Variant(task["variant"], int(task["levels"]), int(task["codebook_size"])), task["mode"])
    tensor = task["tensor"]
    layer = int(task["layer"])
    head = int(task["head"])
    out = codebook_path(args.output_dir, setting, tensor, layer, head)
    if out.exists() and not args.overwrite:
        return {"status": "exists", **task, "path": str(out)}
    out.parent.mkdir(parents=True, exist_ok=True)

    train_ids = [int(x["prompt_id"]) for x in split_items(args.output_dir, "train")]
    seed = args.seed + layer * 100000 + head * 1000 + (0 if tensor == "K" else 97)
    x = sample_vectors(args.output_dir, "train", train_ids, layer, head, tensor, args.sample_size, seed)
    sample_manifest_sha256 = hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()
    perm = dim_permutation(args.seed, x.shape[-1])
    runtime_transform = transform_mode(setting.mode)
    norm_mode = input_norm_mode(setting.mode)
    grouped_raw = to_group_space(x, runtime_transform, perm)
    if norm_mode == "none":
        grouped = grouped_raw.float().contiguous()
    else:
        grouped, _scale = normalize_groups(grouped_raw)
    codebooks = []
    backends: list[list[str]] = []
    residual_mse: list[list[float]] = []
    start_time = time.perf_counter()
    print(
        f"[train] {setting.dirname} tensor={tensor} layer={layer} head={head} "
        f"samples={x.shape[0]} backend priority: QuantVideo KMeans / FAISS GPU KMeans / sklearn MiniBatchKMeans",
        flush=True,
    )
    for group in range(GROUPS):
        residual = grouped[:, group, :].contiguous()
        group_codebooks = []
        group_backends = []
        group_mse = []
        for level in range(setting.variant.levels):
            centroids, ids, backend, errors = train_kmeans_auto(
                residual,
                setting.variant.codebook_size,
                args.kmeans_max_iters,
                seed + group * 100 + level,
                device,
                gpu,
            )
            for msg in errors:
                print(f"[train] {msg}", flush=True)
            assigned = centroids[ids.long()].float()
            residual = (residual.float() - assigned).contiguous()
            mse = float((residual * residual).mean().item())
            group_codebooks.append(centroids.float())
            group_backends.append(backend)
            group_mse.append(mse)
            print(
                f"[train] {setting.dirname} {tensor} L{layer:02d} H{head:02d} "
                f"G{group} level={level + 1} backend={backend} residual_mse={mse:.8g}",
                flush=True,
            )
        codebooks.append(torch.stack(group_codebooks, dim=0))
        backends.append(group_backends)
        residual_mse.append(group_mse)
    # [L, G, C, Dg], so load_layer_codebooks_from_dir stacks heads as [L,H,G,C,Dg].
    cb = torch.stack(codebooks, dim=1).contiguous()
    payload = {
        "codebooks": cb.float(),
        "rvq_format": "grouped_dim",
        "group_transform": "hadamard" if runtime_transform == "hadamard" else "none",
        "input_norm_mode": norm_mode,
        "dim_permutation": perm,
        "num_groups": GROUPS,
        "group_dim": GROUP_DIM,
        "original_dim": int(x.shape[-1]),
        "codebook_size": setting.variant.codebook_size,
        "config": {
            "rvq_format": "grouped_dim",
            "variant": setting.variant.name,
            "mode": setting.mode,
            "tensor": tensor,
            "layer": layer,
            "head": head,
            "levels": setting.variant.levels,
            "codebook_size": setting.variant.codebook_size,
            "num_groups": GROUPS,
            "group_dim": GROUP_DIM,
            "original_dim": int(x.shape[-1]),
            "normalization": "none" if norm_mode == "none" else "per-group RMS after transform/permutation",
            "input_norm_mode": norm_mode,
            "group_transform": "hadamard" if runtime_transform == "hadamard" else "none",
            "dim_permutation": perm.tolist(),
            "sample_size": int(x.shape[0]),
            "sample_seed_mode": "grouped_dim_v1",
            "sample_manifest_sha256": sample_manifest_sha256,
            "kmeans_max_iters": args.kmeans_max_iters,
            "backend_priority": ["QuantVideo KMeans", "FAISS GPU KMeans", "sklearn MiniBatchKMeans"],
            "backends": backends,
            "residual_mse": residual_mse,
        },
    }
    torch.save(payload, out)
    return {"status": "trained", "wall_time_s": time.perf_counter() - start_time, **task, "path": str(out)}


def quantize_codebooks_int4(codebooks: torch.Tensor) -> torch.Tensor:
    max_abs = codebooks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.round(codebooks.float() / (max_abs / 7.0)).clamp(-7, 7)
    return (q * (max_abs / 7.0)).contiguous()


def nearest_ids(x: torch.Tensor, centroids: torch.Tensor, device: torch.device, batch_size: int = 8192) -> torch.Tensor:
    ids = []
    c = centroids.float().to(device)
    c_sq = (c * c).sum(dim=1)
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].float().to(device)
        dist = (xb * xb).sum(dim=1, keepdim=True) + c_sq[None, :] - 2.0 * (xb @ c.T)
        ids.append(torch.argmin(dist, dim=1).cpu())
    return torch.cat(ids, dim=0)


def reconstruct_grouped(x: torch.Tensor, payload: dict[str, Any], mode: str, device: torch.device) -> torch.Tensor:
    perm = payload["dim_permutation"].long()
    runtime_transform = transform_mode(mode)
    norm_mode = str(payload.get("input_norm_mode", payload.get("config", {}).get("input_norm_mode", input_norm_mode(mode)))).lower()
    grouped_raw = to_group_space(x.reshape(-1, INTERNAL_DIM).float(), runtime_transform, perm)
    if norm_mode in {"none", "raw"}:
        grouped = grouped_raw.float().contiguous()
        scale = None
    else:
        grouped, scale = normalize_groups(grouped_raw)
    cb = quantize_codebooks_int4(payload["codebooks"].float())
    recon_groups = []
    for group in range(GROUPS):
        residual = grouped[:, group, :].contiguous()
        recon = torch.zeros_like(residual)
        for level in range(cb.shape[0]):
            centroids = cb[level, group]
            ids = nearest_ids(residual, centroids, device)
            assigned = centroids[ids.long()].float()
            recon += assigned
            residual = residual - assigned
        recon_groups.append(recon)
    recon_grouped = torch.stack(recon_groups, dim=1)
    if scale is not None:
        recon_grouped = recon_grouped * scale
    out = from_group_space(recon_grouped, runtime_transform, perm)
    return out.reshape_as(x).contiguous()


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, a.shape[-1])
    ai = torch.topk(a, k=k, dim=-1).indices
    bi = torch.topk(b, k=k, dim=-1).indices
    overlap = (ai[:, :, None] == bi[:, None, :]).any(dim=-1).float().mean(dim=-1)
    return float(overlap.mean().item())


def eval_one(args: argparse.Namespace, task: dict[str, Any], device: torch.device) -> dict[str, Any]:
    setting = Setting(Variant(task["variant"], int(task["levels"]), int(task["codebook_size"])), task["mode"])
    prompt_id = int(task["prompt_id"])
    layer = int(task["layer"])
    head = int(task["head"])
    out = args.output_dir / "metrics" / "grouped_dim_rvq_rows" / f"{setting.dirname}_p{prompt_id}_l{layer}_h{head}.json"
    if out.exists() and not args.overwrite:
        return json.loads(out.read_text(encoding="utf-8"))
    payload = torch.load(layer_path(args.output_dir, "test", prompt_id, layer), map_location="cpu", weights_only=False)
    k = payload["K"][:, head].reshape(-1, INTERNAL_DIM).float()
    v = payload["V"][:, head].reshape(-1, INTERNAL_DIM).float()
    q = payload["Q"][:, head].reshape(-1, INTERNAL_DIM).float()
    k_cb = torch.load(codebook_path(args.output_dir, setting, "K", layer, head), map_location="cpu", weights_only=False)
    v_cb = torch.load(codebook_path(args.output_dir, setting, "V", layer, head), map_location="cpu", weights_only=False)

    total = k.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed + prompt_id * 1000000 + layer * 10000 + head)
    recon_idx = torch.randperm(total, generator=gen)[: min(args.recon_sample, total)]
    q_idx = torch.randperm(total, generator=gen)[: min(args.query_sample, total)]
    key_idx = torch.randperm(total, generator=gen)[: min(args.key_sample, total)]
    kh_rec = reconstruct_grouped(k[recon_idx], k_cb, setting.mode, device)
    vh_rec = reconstruct_grouped(v[recon_idx], v_cb, setting.mode, device)
    k_ref = k[recon_idx]
    v_ref = v[recon_idx]

    q_s = q[q_idx].to(device)
    k_s = k[key_idx].to(device)
    v_s = v[key_idx].to(device)
    kh_s = reconstruct_grouped(k[key_idx], k_cb, setting.mode, device).to(device)
    vh_s = reconstruct_grouped(v[key_idx], v_cb, setting.mode, device).to(device)
    logits = q_s @ k_s.T / math.sqrt(INTERNAL_DIM)
    logits_hat = q_s @ kh_s.T / math.sqrt(INTERNAL_DIM)
    p = F.softmax(logits.float(), dim=-1)
    ph = F.softmax(logits_hat.float(), dim=-1)
    out_ref = p @ v_s.float()
    out_hat = ph @ vh_s.float()
    row = {
        "setting": setting.label,
        "variant": setting.variant.name,
        "mode": setting.mode,
        "prompt_id": prompt_id,
        "layer": layer,
        "head": head,
        "K_MSE": float(((k_ref - kh_rec) ** 2).mean().item()),
        "K_cos": float(F.cosine_similarity(k_ref, kh_rec, dim=-1).mean().item()),
        "V_MSE": float(((v_ref - vh_rec) ** 2).mean().item()),
        "V_cos": float(F.cosine_similarity(v_ref, vh_rec, dim=-1).mean().item()),
        "QK_MSE": float(((logits - logits_hat) ** 2).mean().item()),
        "attention_KL": float((p * (torch.log(p.clamp_min(1e-8)) - torch.log(ph.clamp_min(1e-8)))).sum(dim=-1).mean().item()),
        "topk_overlap": topk_overlap(logits, logits_hat, args.topk),
        "QKV_MSE": float(((out_ref - out_hat) ** 2).mean().item()),
        "QKV_cos": float(F.cosine_similarity(out_ref, out_hat, dim=-1).mean().item()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fields:
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def worker_main(args: argparse.Namespace) -> None:
    tasks = json.loads(args.task_file.read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    rows = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    gpu = int(visible) if visible else 0
    for task in tasks:
        try:
            if args.worker == "train":
                rows.append(train_one(args, task, device, gpu))
            else:
                rows.append(eval_one(args, task, device))
        except Exception as exc:
            rows.append({"status": "failed", "error": repr(exc), **task})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(args.task_file.with_suffix(".results.csv"), rows)


def gpu_status() -> dict[int, tuple[int, int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    status = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            status[int(parts[0])] = (int(parts[1]), int(parts[2]))
    return status


def free_gpus(args: argparse.Namespace, allowed: list[int], max_used_mb: int | None = None) -> list[int]:
    status = gpu_status()
    limit = int(args.max_used_mb if max_used_mb is None else max_used_mb)
    return [
        g
        for g in allowed
        if g in status and status[g][0] <= limit and status[g][1] <= args.max_util
    ]


def launch_sharded(args: argparse.Namespace, mode: str, tasks: list[dict[str, Any]], gpus: list[int]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    task_dir = args.output_dir / "status" / "grouped_dim_rvq" / f"{mode}_{int(time.time())}"
    task_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for i, gpu in enumerate(gpus):
        shard = tasks[i:: len(gpus)]
        if not shard:
            continue
        task_file = task_dir / f"{mode}_gpu{gpu}.json"
        task_file.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        log_path = args.output_dir / "logs" / "grouped_dim_rvq" / f"{mode}_gpu{gpu}_{task_dir.name}.log"
        cmd = [
            str(args.python),
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--task-file",
            str(task_file),
            "--output-dir",
            str(args.output_dir),
            "--sample-size",
            str(args.sample_size),
            "--kmeans-max-iters",
            str(args.kmeans_max_iters),
            "--seed",
            str(args.seed),
            "--recon-sample",
            str(args.recon_sample),
            "--query-sample",
            str(args.query_sample),
            "--key-sample",
            str(args.key_sample),
            "--topk",
            str(args.topk),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        pending.append({"cmd": cmd, "task_file": task_file, "log_path": log_path, "tasks": len(shard), "name": f"{mode}:{task_file.stem}"})
    rows = []
    failures = []
    active: list[dict[str, Any]] = []
    while pending or active:
        busy = {int(x["gpu"]) for x in active}
        for gpu in free_gpus(args, gpus, max_used_mb=args.train_max_used_mb):
            if not pending or gpu in busy:
                continue
            job = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["MASTER_PORT"] = str(args.master_port_base + gpu)
            log = job["log_path"].open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(job["cmd"]) + "\n")
            log.flush()
            print(f"[{mode}] gpu={gpu} tasks={job['tasks']} log={job['log_path']}", flush=True)
            proc = subprocess.Popen(job["cmd"], cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "gpu": gpu, **job})
            busy.add(gpu)
        if not active and pending:
            print(
                f"[{mode}-wait] pending_shards={len(pending)} "
                f"max_used_mb={args.train_max_used_mb} gpu_status={gpu_status()}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
            continue
        time.sleep(30)
        keep = []
        for item in active:
            rc = item["proc"].poll()
            if rc is None:
                keep.append(item)
                continue
            item["log"].close()
            result_path = item["task_file"].with_suffix(".results.csv")
            if result_path.exists():
                with result_path.open(newline="", encoding="utf-8") as f:
                    rows.extend(csv.DictReader(f))
            if rc != 0:
                failures.append({"mode": mode, "rc": rc, "log": str(item["log_path"]), "task_file": str(item["task_file"])})
                print(f"[{mode}-fail] gpu={item['gpu']} rc={rc} log={item['log_path']}", flush=True)
            else:
                print(f"[{mode}-done] gpu={item['gpu']} {item['name']}", flush=True)
        active = keep
    if failures:
        fail_path = args.output_dir / "metrics" / f"grouped_dim_rvq_{mode}_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"[warn] failures written to {fail_path}", flush=True)
    return rows


def train_tasks(args: argparse.Namespace, settings: list[Setting]) -> list[dict[str, Any]]:
    tasks = []
    tensors = parse_tensors(args.tensors)
    for setting in settings:
        for tensor in tensors:
            for layer in range(args.num_layers):
                for head in range(args.num_heads):
                    path = codebook_path(args.output_dir, setting, tensor, layer, head)
                    if path.exists() and not args.overwrite:
                        continue
                    tasks.append(
                        {
                            "variant": setting.variant.name,
                            "levels": setting.variant.levels,
                            "codebook_size": setting.variant.codebook_size,
                            "mode": setting.mode,
                            "tensor": tensor,
                            "layer": layer,
                            "head": head,
                        }
                    )
    return tasks


def eval_tasks(args: argparse.Namespace, settings: list[Setting]) -> list[dict[str, Any]]:
    test_ids = [int(x["prompt_id"]) for x in split_items(args.output_dir, "test")][: args.num_prompts]
    tasks = []
    for setting in settings:
        for pid in test_ids:
            for layer in range(args.num_layers):
                for head in range(args.num_heads):
                    out = args.output_dir / "metrics" / "grouped_dim_rvq_rows" / f"{setting.dirname}_p{pid}_l{layer}_h{head}.json"
                    if out.exists() and not args.overwrite:
                        continue
                    tasks.append(
                        {
                            "variant": setting.variant.name,
                            "levels": setting.variant.levels,
                            "codebook_size": setting.variant.codebook_size,
                            "mode": setting.mode,
                            "prompt_id": pid,
                            "layer": layer,
                            "head": head,
                        }
                    )
    return tasks


def aggregate_offline(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in (root / "metrics" / "grouped_dim_rvq_rows").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") != "failed":
            rows.append(row)
    write_csv(root / "metrics" / "grouped_dim_rvq_offline_rows.csv", rows)
    keys = ["K_MSE", "K_cos", "V_MSE", "V_cos", "QK_MSE", "attention_KL", "topk_overlap", "QKV_MSE", "QKV_cos"]
    summaries = []
    groups = sorted({r["setting"] for r in rows if "setting" in r})
    for name in groups:
        group = [r for r in rows if r.get("setting") == name]
        out = {"setting": name, "rows": len(group)}
        for key in keys:
            vals = [float(r[key]) for r in group if r.get(key) not in (None, "", "nan")]
            out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
        summaries.append(out)
    write_csv(root / "metrics" / "grouped_dim_rvq_offline_summary.csv", summaries)
    return summaries


def metrics_complete(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f)) >= expected_rows
    except Exception:
        return False


def build_video_cmd(args: argparse.Namespace, setting: Setting) -> list[str]:
    return [
        str(args.python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        str(args.num_prompts),
        "--output-dir",
        str(video_dir(args.output_dir, setting)),
        "--seed",
        str(args.seed),
        "--setting",
        "kvrvq",
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
        str(args.output_dir / codebook_subdir(setting, "K")),
        "--v_rvq_codebook_dir",
        str(args.output_dir / codebook_subdir(setting, "V")),
        "--rvq_codebook_storage_dtype",
        "int4",
        "--rvq_scale_storage_dtype",
        "uint4",
        "--kv_logical_head_dim",
        str(LOGICAL_KV_DIM),
        "--kv_quant_factor",
        "1",
    ]


def run_jobs_waiting_for_gpus(args: argparse.Namespace, jobs: list[tuple[str, list[str], Path]], gpus: list[int]) -> None:
    pending = list(jobs)
    active: list[dict[str, Any]] = []
    retries: dict[str, int] = {}
    while pending or active:
        busy = {x["gpu"] for x in active}
        for gpu in free_gpus(args, gpus):
            if gpu in busy or not pending:
                continue
            name, cmd, log_path = pending.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["MASTER_PORT"] = str(args.master_port_base + 100 + gpu)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[launch] gpu={gpu} {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"name": name, "cmd": cmd, "log": log, "log_path": log_path, "proc": proc, "gpu": gpu})
            busy.add(gpu)
        if not active and pending:
            print(f"[wait] pending={len(pending)} gpu_status={gpu_status()}", flush=True)
            time.sleep(args.poll_seconds)
            continue
        time.sleep(30)
        keep = []
        for item in active:
            rc = item["proc"].poll()
            if rc is None:
                keep.append(item)
                continue
            item["log"].close()
            if rc != 0:
                count = retries.get(item["name"], 0) + 1
                retries[item["name"]] = count
                print(f"[fail] {item['name']} rc={rc} log={item['log_path']} retry={count}", flush=True)
                if count <= args.max_retries:
                    pending.append((item["name"], item["cmd"], item["log_path"]))
            else:
                print(f"[done] gpu={item['gpu']} {item['name']}", flush=True)
        active = keep


def run_videos(args: argparse.Namespace, settings: list[Setting], gpus: list[int]) -> None:
    if args.skip_video:
        return
    jobs = []
    for setting in settings:
        out = video_dir(args.output_dir, setting)
        if metrics_complete(out / "metrics.csv", args.num_prompts) and not args.overwrite:
            print(f"[video-skip] {setting.label}", flush=True)
            continue
        jobs.append((f"video:{setting.dirname}", build_video_cmd(args, setting), args.output_dir / "logs" / "grouped_dim_rvq" / f"video_{setting.dirname}.log"))
    run_jobs_waiting_for_gpus(args, jobs, gpus)


def run_quality(args: argparse.Namespace, settings: list[Setting], gpus: list[int]) -> None:
    if args.skip_quality:
        return
    jobs = []
    out_root = args.output_dir / "vs_bf16_grouped_dim_rvq"
    for setting in settings:
        out_json = out_root / f"{setting.dirname}.json"
        out_csv = out_root / f"{setting.dirname}.csv"
        if out_json.exists() and out_csv.exists() and not args.overwrite:
            continue
        if not metrics_complete(video_dir(args.output_dir, setting) / "metrics.csv", args.num_prompts):
            continue
        cmd = [
            str(args.python),
            str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
            "--reference-dir",
            str(args.output_dir / "videos" / "bf16"),
            "--test-dir",
            str(video_dir(args.output_dir, setting)),
            "--metrics-csv",
            str(video_dir(args.output_dir, setting) / "metrics.csv"),
            "--setting",
            setting.label,
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
            "--device",
            "cuda",
        ]
        jobs.append((f"quality:{setting.dirname}", cmd, args.output_dir / "logs" / "grouped_dim_rvq" / f"quality_{setting.dirname}.log"))
    run_jobs_waiting_for_gpus(args, jobs, gpus)


def prompt_json_path(root: Path, setting: Setting) -> Path:
    return root / "vbench_grouped_dim_rvq" / f"prompts_{setting.dirname}.json"


def write_prompt_json(root: Path, setting: Setting) -> None:
    rows = read_csv(video_dir(root, setting) / "metrics.csv")
    prompt_map = {}
    for row in rows:
        path = Path(row["video_path"])
        if not path.is_absolute():
            path = (REPO_ROOT / path).resolve()
        prompt_map[str(path)] = row["prompt"]
    out = prompt_json_path(root, setting)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")


def vbench_complete(root: Path, setting: Setting, dim: str) -> bool:
    result_dir = root / "vbench_grouped_dim_rvq" / setting.dirname / dim
    return bool(list(result_dir.glob("*_eval_results.json")))


def run_vbench(args: argparse.Namespace, settings: list[Setting], gpus: list[int]) -> None:
    if args.skip_vbench:
        return
    jobs = []
    for setting in settings:
        if not metrics_complete(video_dir(args.output_dir, setting) / "metrics.csv", args.num_prompts):
            continue
        write_prompt_json(args.output_dir, setting)
        for dim in VBENCH_DIMS:
            if vbench_complete(args.output_dir, setting, dim) and not args.overwrite:
                continue
            result_dir = args.output_dir / "vbench_grouped_dim_rvq" / setting.dirname / dim
            cmd = [
                str(args.vbench_bin),
                "evaluate",
                "--ngpus",
                "1",
                "--output_path",
                str(result_dir),
                "--videos_path",
                str(video_dir(args.output_dir, setting)),
                "--dimension",
                dim,
                "--mode",
                "custom_input",
                "--prompt_file",
                str(prompt_json_path(args.output_dir, setting)),
            ]
            jobs.append((f"vbench:{setting.dirname}:{dim}", cmd, args.output_dir / "logs" / "grouped_dim_rvq" / f"vbench_{setting.dirname}_{dim}.log"))
    run_jobs_waiting_for_gpus(args, jobs, gpus)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_vbench(root: Path, setting: Setting) -> dict[str, float | None]:
    out = {dim: None for dim in VBENCH_DIMS}
    for dim in VBENCH_DIMS:
        files = sorted((root / "vbench_grouped_dim_rvq" / setting.dirname / dim).glob("*_eval_results.json"))
        if not files:
            continue
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
        stack: list[Any] = [payload]
        while stack and out[dim] is None:
            item = stack.pop()
            if isinstance(item, dict):
                value = item.get(dim)
                if isinstance(value, (int, float)):
                    out[dim] = float(value)
                elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                    out[dim] = float(value[0])
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return out


def aggregate_runtime_video(args: argparse.Namespace, settings: list[Setting]) -> list[dict[str, Any]]:
    rows = []
    for setting in settings:
        metrics = read_csv(video_dir(args.output_dir, setting) / "metrics.csv")
        if not metrics:
            continue
        def vals(key: str, drop0: bool = False) -> list[float]:
            src = metrics[1:] if drop0 and len(metrics) > 1 else metrics
            return [v for v in (as_float(r.get(key)) for r in src) if v is not None]
        row: dict[str, Any] = {
            "setting": setting.label,
            "videos": len(metrics),
            "e2e_mean_s": mean(vals("e2e_time_s")),
            "e2e_drop0_s": mean(vals("e2e_time_s", True)),
            "denoise_mean_s": mean(vals("denoise_time_s")),
            "quant_drop0_s": mean(vals("quant_time_s", True)),
            "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
            "kv_cache_mem_mb": mean(vals("kv_cache_mem_mb")),
            "compression_ratio": mean(vals("compression_ratio")),
            "psnr": None,
            "ssim": None,
            "lpips": None,
        }
        qjson = args.output_dir / "vs_bf16_grouped_dim_rvq" / f"{setting.dirname}.json"
        if qjson.exists():
            payload = json.loads(qjson.read_text(encoding="utf-8"))
            row["psnr"] = payload.get("psnr_mean")
            row["ssim"] = payload.get("ssim_mean")
            row["lpips"] = payload.get("lpips_mean")
        row.update(parse_vbench(args.output_dir, setting))
        rows.append(row)
    write_csv(args.output_dir / "metrics" / "grouped_dim_rvq_runtime_video.csv", rows)
    return rows


def make_report(args: argparse.Namespace, settings: list[Setting]) -> Path:
    offline = aggregate_offline(args.output_dir)
    runtime = aggregate_runtime_video(args, settings)
    path = args.output_dir / "reports" / "grouped_dim_rvq_summary.md"
    lines = [
        "# Grouped-Dimension RVQ Summary",
        "",
        f"Command: `{' '.join(sys.argv)}`",
        "",
        "## Offline Cache Metrics",
        "",
    ]
    if offline:
        keys = ["setting", "rows", "K_MSE_mean", "K_cos_mean", "V_MSE_mean", "V_cos_mean", "QK_MSE_mean", "attention_KL_mean", "topk_overlap_mean", "QKV_MSE_mean", "QKV_cos_mean"]
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in offline:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    lines.extend(["", "## Runtime / Video Metrics", ""])
    if runtime:
        keys = ["setting", "videos", "e2e_mean_s", "denoise_mean_s", "quant_drop0_s", "kv_cache_mem_mb", "compression_ratio", "psnr", "ssim", "lpips", *VBENCH_DIMS]
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for row in runtime:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return
    mkdirs(args.output_dir)
    gpus = parse_gpus(args.gpus)
    settings = settings_from_args(args)
    config = {
        "command": " ".join(sys.argv),
        "output_dir": str(args.output_dir),
        "settings": [
            {
                "label": s.label,
                "dirname": s.dirname,
                "levels": s.variant.levels,
                "codebook_size": s.variant.codebook_size,
                "mode": s.mode,
                "groups": GROUPS,
                "group_dim": GROUP_DIM,
                "storage": "int4_codebook_uint4_scale",
            }
            for s in settings
        ],
        "tensors": parse_tensors(args.tensors),
        "seed": args.seed,
        "sample_size": args.sample_size,
        "sample_seed_mode": "grouped_dim_v1",
        "kmeans_max_iters": args.kmeans_max_iters,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "backend_priority": ["QuantVideo KMeans", "FAISS GPU KMeans", "sklearn MiniBatchKMeans"],
    }
    (args.output_dir / "grouped_dim_rvq_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[settings] {', '.join(s.label for s in settings)}", flush=True)
    print(f"[gpus] {gpus}", flush=True)

    if not args.skip_train:
        tasks = train_tasks(args, settings)
        print(f"[train] pending={len(tasks)}", flush=True)
        launch_sharded(args, "train", tasks, gpus)
    if not args.skip_offline:
        tasks = eval_tasks(args, settings)
        print(f"[offline] pending={len(tasks)}", flush=True)
        launch_sharded(args, "eval", tasks, gpus)
        aggregate_offline(args.output_dir)
    run_videos(args, settings, gpus)
    run_quality(args, settings, gpus)
    run_vbench(args, settings, gpus)
    report = make_report(args, settings)
    print(f"[complete] report={report}", flush=True)


if __name__ == "__main__":
    main()
