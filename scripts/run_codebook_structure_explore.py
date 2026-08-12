#!/usr/bin/env python3
"""Explore structured codebooks for T2V KV-cache compression.

This is an orchestration script for the MovieGen T2V 20/10 split under
`/data/bowen/t2v_kv_bench_moviegen_20train_10test`. It intentionally keeps new
artifacts separate from the existing RVQ sweep so partial runs are resumable.

The script has two tiers:
- offline screening over all 30 layers x 24 heads on the first few test prompts;
- optional 10-video generation for the best runtime-compatible candidates.

Runtime-compatible candidates are represented as ordinary RVQ codebook payloads
and can be consumed by run_small_eval.py. PQ/hybrid candidates are offline-only.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PROMPT_SOURCE = Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/MovieGenVideoBench.txt")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH_BIN = Path("/home/bowen/anaconda3/bin/vbench")
LAYERS = 30
HEADS = 24
D = 128
LOGICAL_KV_DIM = 256
N_SPATIAL = 880
VBENCH_DIMS = ["subject_consistency", "background_consistency", "imaging_quality", "aesthetic_quality"]


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int
    storage: str = "int4"


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    runtime: bool
    train_k: bool = True
    train_v: bool = True
    uses_assignment_codebooks: bool = False
    offline_only_reason: str = ""


VARIANTS: dict[str, tuple[int, int]] = {
    "l8c64": (8, 64),
    "l4c256": (4, 256),
    "l8c1024": (8, 1024),
}


CANDIDATES: dict[str, Candidate] = {
    "baseline": Candidate("baseline", "baseline", True, train_k=False, train_v=False),
    "querycov_existing": Candidate(
        "querycov_existing",
        "transform_rvq",
        True,
        train_k=False,
        train_v=False,
        uses_assignment_codebooks=False,
    ),
    "shared_stage": Candidate("shared_stage", "shared_codebook", True),
    "tail_shared": Candidate("tail_shared", "shared_codebook", True),
    "real_token_stage1": Candidate("real_token_stage1", "temporal_real_token_dictionary", True),
    "hadamard_rvq": Candidate(
        "hadamard_rvq",
        "transform_rvq",
        True,
        uses_assignment_codebooks=True,
    ),
    "diagq_rvq_k": Candidate(
        "diagq_rvq_k",
        "transform_rvq",
        True,
        train_v=False,
        uses_assignment_codebooks=True,
    ),
    "pq_contiguous": Candidate(
        "pq_contiguous",
        "pq",
        False,
        offline_only_reason="PQ runtime compressor is not implemented in run_small_eval.py.",
    ),
    "hadamard_pq": Candidate(
        "hadamard_pq",
        "pq",
        False,
        offline_only_reason="PQ runtime compressor is not implemented in run_small_eval.py.",
    ),
    "rvq2_int8_residual": Candidate(
        "rvq2_int8_residual",
        "hybrid_residual",
        False,
        train_k=False,
        train_v=False,
        offline_only_reason="Hybrid residual runtime codec is not implemented; memory is estimated offline.",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--prompt-source", type=Path, default=DEFAULT_PROMPT_SOURCE)
    p.add_argument("--configs", type=str, default="l8c64:int4,l4c256:int4")
    p.add_argument(
        "--candidates",
        type=str,
        default=(
            "baseline,querycov_existing,shared_stage,tail_shared,real_token_stage1,"
            "hadamard_rvq,diagq_rvq_k,pq_contiguous,hadamard_pq,rvq2_int8_residual"
        ),
    )
    p.add_argument("--gpus", type=str, default="auto")
    p.add_argument("--min-free-gpu-mb", type=int, default=30000)
    p.add_argument("--max-hours", type=float, default=9.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-size", type=int, default=12000)
    p.add_argument("--kmeans-iters", type=int, default=8)
    p.add_argument("--screen-prompts", type=int, default=3)
    p.add_argument("--recon-sample", type=int, default=2048)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--key-sample", type=int, default=1024)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--video-topk", type=int, default=1, help="Non-baseline runtime candidates per variant to run as videos.")
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--vbench-bin", type=Path, default=DEFAULT_VBENCH_BIN)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-offline", action="store_true")
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-vs-bf16", action="store_true")
    p.add_argument("--skip-vbench", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--worker", choices=["train", "eval"], default="")
    p.add_argument("--task-file", type=Path, default=None)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def parse_variants(raw: str) -> list[Variant]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, storage = item.split(":", 1)
        else:
            name, storage = item, "int4"
        name = name.lower().strip()
        if name not in VARIANTS:
            raise ValueError(f"unknown config {name}")
        levels, c = VARIANTS[name]
        out.append(Variant(name, levels, c, storage.lower().strip()))
    if not out:
        raise ValueError("--configs produced no variants")
    return out


def parse_candidates(raw: str) -> list[Candidate]:
    out = []
    for name in raw.split(","):
        name = name.strip().lower()
        if not name:
            continue
        if name not in CANDIDATES:
            raise ValueError(f"unknown candidate {name}")
        out.append(CANDIDATES[name])
    return out


def clean(name: str) -> str:
    return name.lower().replace("+", "plus").replace("/", "_").replace("-", "_").replace(" ", "_")


def mkdirs(root: Path) -> None:
    for rel in [
        "logs/codebook_explore",
        "status/codebook_explore",
        "metrics/codebook_explore_rows",
        "videos_codebook_explore",
        "vs_bf16_codebook_explore",
        "vbench_codebook_explore",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def variant_from_task(task: dict[str, Any]) -> Variant:
    return Variant(str(task["variant"]), int(task["levels"]), int(task["codebook_size"]), str(task["storage"]))


def candidate_from_name(name: str) -> Candidate:
    return CANDIDATES[str(name)]


def baseline_k_subdir(v: Variant) -> str:
    return "codebooks" if v.name == "l4c256" else f"sweep_k_{v.name}_codebooks"


def baseline_v_subdir(v: Variant) -> str:
    return "v_codebooks" if v.name == "l4c256" else f"sweep_v_{v.name}_codebooks"


def qk_existing_k_subdir(v: Variant) -> str:
    return f"qkrvq_{v.name}_codebooks"


def codebook_subdir(v: Variant, c: Candidate, tensor: str) -> str:
    if c.name == "baseline":
        return baseline_k_subdir(v) if tensor == "K" else baseline_v_subdir(v)
    if c.name == "querycov_existing":
        return qk_existing_k_subdir(v) if tensor == "K" else baseline_v_subdir(v)
    if c.name == "diagq_rvq_k" and tensor == "V":
        return baseline_v_subdir(v)
    return f"codebook_explore_{c.name}_{v.name}_{tensor.lower()}_codebooks"


def pq_subdir(v: Variant, c: Candidate, tensor: str) -> str:
    return f"codebook_explore_{c.name}_{v.name}_{tensor.lower()}_pq"


def codebook_path(root: Path, subdir: str, layer: int, head: int) -> Path:
    return root / subdir / f"layer_{layer}_head_{head}.pt"


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    with (root / "prompt_split.json").open("r", encoding="utf-8") as f:
        return json.load(f)[split]


def layer_path(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def idle_gpus(min_free_mb: int) -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return [0]
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, free, util = map(int, parts[:3])
        if free >= min_free_mb and util <= 15:
            gpus.append(idx)
    return gpus


def parse_gpus(raw: str, min_free_mb: int) -> list[int]:
    if raw.strip().lower() == "auto":
        found = idle_gpus(min_free_mb)
        if found:
            return found
        raise RuntimeError(f"no idle GPU found with >= {min_free_mb} MB free")
    return [int(x) for x in raw.split(",") if x.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_cmd(
    cmd: list[str],
    log_path: Path,
    *,
    env: dict[str, str] | None = None,
    cwd: Path = REPO_ROOT,
    sentinel: Path | None = None,
    overwrite: bool = False,
) -> float:
    if sentinel is not None and sentinel.exists() and not overwrite:
        print(f"[skip] {sentinel}", flush=True)
        return 0.0
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + " ".join(str(x) for x in cmd) + "\n")
        log.flush()
        rc = subprocess.call([str(x) for x in cmd], cwd=str(cwd), env=run_env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; log={log_path}")
    if sentinel is not None:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text(json.dumps({"elapsed_s": elapsed, "cmd": cmd}, indent=2), encoding="utf-8")
    return elapsed


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
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for pid in prompt_ids:
        payload = torch.load(layer_path(root, split, pid, layer), map_location="cpu", weights_only=False)
        x = payload[tensor][:, head].reshape(-1, D).float()
        take = min(per_prompt, x.shape[0])
        idx = torch.randperm(x.shape[0], generator=gen)[:take]
        chunks.append(x[idx])
    x = torch.cat(chunks, dim=0)
    if x.shape[0] > sample_size:
        idx = torch.randperm(x.shape[0], generator=gen)[:sample_size]
        x = x[idx]
    return x.contiguous()


def normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True)).clamp_min(1e-6)
    return (x.float() / scale).contiguous(), scale


def nearest_torch(x: torch.Tensor, centroids: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    ids = []
    c = centroids.float()
    c_sq = (c * c).sum(dim=1)
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].float()
        dist = (xb * xb).sum(dim=1, keepdim=True) + c_sq[None, :] - 2.0 * (xb @ c.T)
        ids.append(torch.argmin(dist, dim=1).cpu())
    return torch.cat(ids, dim=0)


def kmeans_torch(
    x_cpu: torch.Tensor,
    k: int,
    iters: int,
    seed: int,
    device: torch.device,
    batch_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_cpu.float().to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    if x.shape[0] < k:
        repeats = math.ceil(k / x.shape[0])
        x = x.repeat((repeats, 1))[:k]
    perm = torch.randperm(x.shape[0], generator=gen, device=device)[:k]
    centroids = x[perm].clone()
    ids = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    for _ in range(max(1, iters)):
        ids_chunks = []
        c_sq = (centroids * centroids).sum(dim=1)
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size]
            dist = (xb * xb).sum(dim=1, keepdim=True) + c_sq[None, :] - 2.0 * (xb @ centroids.T)
            ids_chunks.append(torch.argmin(dist, dim=1))
        ids = torch.cat(ids_chunks, dim=0)
        sums = torch.zeros_like(centroids)
        counts = torch.bincount(ids, minlength=k).float().to(device).clamp_min(0.0)
        sums.index_add_(0, ids, x)
        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            repl = torch.randperm(x.shape[0], generator=gen, device=device)[: int((~nonempty).sum().item())]
            centroids[~nonempty] = x[repl]
    return centroids.detach().cpu().contiguous(), ids.detach().cpu()


def hadamard_matrix(dim: int) -> torch.Tensor:
    if dim & (dim - 1):
        raise ValueError("Hadamard dimension must be a power of two")
    h = torch.tensor([[1.0]])
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(dim)


def metric_diag_from_queries(root: Path, layer: int, head: int, sample_size: int, seed: int) -> torch.Tensor:
    q = sample_vectors(root, "train", [int(x["prompt_id"]) for x in split_items(root, "train")], layer, head, "Q", sample_size, seed)
    qn, _ = normalize(q)
    var = (qn * qn).mean(dim=0)
    var = var / var.mean().clamp_min(1e-8)
    return torch.sqrt(var.clamp_min(1e-6)).contiguous()


def rvq_encode_norm(
    x_norm: torch.Tensor,
    codebooks: torch.Tensor,
    *,
    assignment_codebooks: torch.Tensor | None = None,
    assignment_transform: torch.Tensor | None = None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = x_norm.float().to(device)
    recon = torch.zeros_like(residual)
    ids_all = []
    base = codebooks.float().to(device)
    assign = assignment_codebooks.float().to(device) if assignment_codebooks is not None else None
    transform = assignment_transform.float().to(device) if assignment_transform is not None else None
    for level in range(base.shape[0]):
        distance_centroids = assign[level] if assign is not None else base[level]
        rb = residual
        if transform is not None and assign is not None:
            rb_dist = rb * transform if transform.ndim == 1 else rb @ transform
        else:
            rb_dist = rb
        ids = nearest_torch(rb_dist.detach().cpu(), distance_centroids.detach().cpu()).to(device)
        chosen = base[level][ids]
        recon += chosen
        residual = residual - chosen
        ids_all.append(ids.detach().cpu())
    return torch.stack(ids_all, dim=0), recon.detach().cpu()


def load_rvq_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "codebooks" not in payload:
        payload = {"codebooks": payload}
    return payload


def train_rvq_independent(
    x_norm: torch.Tensor,
    levels: int,
    k: int,
    iters: int,
    seed: int,
    device: torch.device,
    *,
    fit_transform: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    residual = x_norm.float().contiguous()
    codebooks = []
    assignment_codebooks = []
    for level in range(levels):
        fit_residual = residual @ fit_transform if fit_transform is not None and fit_transform.ndim == 2 else residual
        if fit_transform is not None and fit_transform.ndim == 1:
            fit_residual = residual * fit_transform
        fit_centroids, ids = kmeans_torch(fit_residual, k, iters, seed + level, device)
        if fit_transform is None:
            centroids = fit_centroids
        elif fit_transform.ndim == 1:
            centroids = fit_centroids / fit_transform.clamp_min(1e-6)
        else:
            centroids = fit_centroids @ torch.linalg.pinv(fit_transform)
        residual = residual - centroids[ids.long()]
        codebooks.append(centroids.float())
        if fit_transform is not None:
            assignment_codebooks.append(fit_centroids.float())
    assign = torch.stack(assignment_codebooks, dim=0) if assignment_codebooks else None
    return torch.stack(codebooks, dim=0).contiguous(), assign


def train_shared_stage(x_norm: torch.Tensor, levels: int, k: int, iters: int, seed: int, device: torch.device) -> torch.Tensor:
    c, _ = kmeans_torch(x_norm, k, iters, seed, device)
    return c[None, :, :].repeat(levels, 1, 1).contiguous()


def train_tail_shared(
    x_norm: torch.Tensor,
    baseline_codebooks: torch.Tensor,
    levels: int,
    k: int,
    iters: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    keep = min(2, levels)
    _ids, recon2 = rvq_encode_norm(x_norm, baseline_codebooks[:keep], device=device)
    residual = x_norm - recon2
    tail, _ = kmeans_torch(residual, k, iters, seed, device)
    out = [baseline_codebooks[i].float() for i in range(keep)]
    while len(out) < levels:
        out.append(tail.float())
    return torch.stack(out, dim=0).contiguous()


def train_real_token_stage1(
    x_norm: torch.Tensor,
    baseline_codebooks: torch.Tensor,
    levels: int,
    k: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if x_norm.shape[0] < k:
        reps = math.ceil(k / x_norm.shape[0])
        pool = x_norm.repeat((reps, 1))[:k]
    else:
        pool = x_norm[torch.randperm(x_norm.shape[0], generator=gen)[:k]]
    out = [pool.float()]
    for level in range(1, levels):
        out.append(baseline_codebooks[level].float())
    return torch.stack(out, dim=0).contiguous()


def train_pq(
    x_norm: torch.Tensor,
    m: int,
    k: int,
    iters: int,
    seed: int,
    device: torch.device,
    *,
    transform: torch.Tensor | None = None,
) -> dict[str, Any]:
    y = x_norm @ transform if transform is not None else x_norm
    subdim = D // m
    codebooks = []
    for part in range(m):
        sub = y[:, part * subdim : (part + 1) * subdim].contiguous()
        cb, _ = kmeans_torch(sub, k, iters, seed + part, device)
        codebooks.append(cb)
    return {
        "method": "pq",
        "codebooks": torch.stack(codebooks, dim=0).contiguous(),
        "m": int(m),
        "codebook_size": int(k),
        "transform": transform.float().contiguous() if transform is not None else None,
    }


def pq_reconstruct(x: torch.Tensor, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    x_norm, scale = normalize(x.reshape(-1, D).float())
    transform = payload.get("transform")
    y = x_norm @ transform if torch.is_tensor(transform) else x_norm
    codebooks = payload["codebooks"].float()
    m = int(payload["m"])
    subdim = D // m
    parts = []
    for part in range(m):
        sub = y[:, part * subdim : (part + 1) * subdim].contiguous()
        cb = codebooks[part]
        ids = nearest_torch(sub.to(device), cb.to(device)).cpu()
        parts.append(cb[ids.long()])
    yh = torch.cat(parts, dim=-1)
    if torch.is_tensor(transform):
        yh = yh @ transform.T
    return (yh * scale).reshape_as(x).contiguous()


def train_one_task(args: argparse.Namespace, task: dict[str, Any], device: torch.device) -> dict[str, Any]:
    root = args.output_dir
    v = variant_from_task(task)
    c = candidate_from_name(task["candidate"])
    tensor = task["tensor"]
    layer = int(task["layer"])
    head = int(task["head"])
    seed = args.seed + layer * 10000 + head * 100 + (0 if tensor == "K" else 11)

    if c.name in {"baseline", "querycov_existing"} or (c.name == "diagq_rvq_k" and tensor == "V"):
        return {"status": "skip_existing", **task}

    if c.family == "pq":
        out = codebook_path(root, pq_subdir(v, c, tensor), layer, head)
    else:
        out = codebook_path(root, codebook_subdir(v, c, tensor), layer, head)
    if out.exists() and not args.overwrite:
        return {"status": "exists", **task}
    out.parent.mkdir(parents=True, exist_ok=True)

    train_ids = [int(x["prompt_id"]) for x in split_items(root, "train")]
    x = sample_vectors(root, "train", train_ids, layer, head, tensor, args.sample_size, seed)
    x_norm, _scale = normalize(x)
    elapsed_start = time.perf_counter()

    baseline_payload = None
    baseline_cb = None
    if c.name in {"tail_shared", "real_token_stage1"}:
        base_subdir = baseline_k_subdir(v) if tensor == "K" else baseline_v_subdir(v)
        baseline_payload = load_rvq_payload(codebook_path(root, base_subdir, layer, head))
        baseline_cb = baseline_payload["codebooks"].float()

    payload: dict[str, Any]
    if c.name == "shared_stage":
        codebooks = train_shared_stage(x_norm, v.levels, v.codebook_size, args.kmeans_iters, seed, device)
        payload = {"codebooks": codebooks}
    elif c.name == "tail_shared":
        assert baseline_cb is not None
        codebooks = train_tail_shared(x_norm, baseline_cb, v.levels, v.codebook_size, args.kmeans_iters, seed, device)
        payload = {"codebooks": codebooks}
    elif c.name == "real_token_stage1":
        assert baseline_cb is not None
        codebooks = train_real_token_stage1(x_norm, baseline_cb, v.levels, v.codebook_size, seed)
        payload = {"codebooks": codebooks}
    elif c.name == "hadamard_rvq":
        h = hadamard_matrix(D)
        codebooks, assignment = train_rvq_independent(
            x_norm,
            v.levels,
            v.codebook_size,
            args.kmeans_iters,
            seed,
            device,
            fit_transform=h,
        )
        payload = {"codebooks": codebooks, "assignment_codebooks": assignment, "assignment_transform": h}
    elif c.name == "diagq_rvq_k":
        if tensor != "K":
            return {"status": "skip_v_baseline", **task}
        diag = metric_diag_from_queries(root, layer, head, args.sample_size, seed)
        codebooks, assignment = train_rvq_independent(
            x_norm,
            v.levels,
            v.codebook_size,
            args.kmeans_iters,
            seed,
            device,
            fit_transform=diag,
        )
        payload = {"codebooks": codebooks, "assignment_codebooks": assignment, "assignment_transform": diag}
    elif c.name in {"pq_contiguous", "hadamard_pq"}:
        transform = hadamard_matrix(D) if c.name == "hadamard_pq" else None
        payload = train_pq(
            x_norm,
            v.levels,
            v.codebook_size,
            args.kmeans_iters,
            seed,
            device,
            transform=transform,
        )
    else:
        raise ValueError(f"unsupported train candidate {c.name}")

    payload["metadata"] = {
        "candidate": c.name,
        "variant": v.name,
        "tensor": tensor,
        "layer": layer,
        "head": head,
        "levels": v.levels,
        "codebook_size": v.codebook_size,
        "sample_size": int(x.shape[0]),
        "kmeans_iters": args.kmeans_iters,
        "seed": seed,
    }
    torch.save(payload, out)
    return {"status": "trained", "wall_time_s": time.perf_counter() - elapsed_start, **task, "path": str(out)}


def train_worker(args: argparse.Namespace) -> None:
    torch.cuda.set_device(0) if torch.cuda.is_available() else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tasks = json.loads(args.task_file.read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        try:
            rows.append(train_one_task(args, task, device))
        except RuntimeError as exc:
            rows.append({"status": "failed", "error": str(exc), **task})
            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({"status": "failed", "error": repr(exc), **task})
            torch.cuda.empty_cache()
    write_csv(args.task_file.with_suffix(".results.csv"), rows)


def rvq_reconstruct(x: torch.Tensor, payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    x_flat = x.reshape(-1, D).float()
    x_norm, scale = normalize(x_flat)
    codebooks = payload["codebooks"].float()
    assign = payload.get("assignment_codebooks")
    transform = payload.get("assignment_transform")
    _ids, recon_norm = rvq_encode_norm(
        x_norm,
        codebooks,
        assignment_codebooks=assign if torch.is_tensor(assign) else None,
        assignment_transform=transform if torch.is_tensor(transform) else None,
        device=device,
    )
    return (recon_norm * scale).reshape_as(x).contiguous()


def hybrid_rvq2_int8_residual(x: torch.Tensor, baseline_payload: dict[str, Any], device: torch.device) -> torch.Tensor:
    x_flat = x.reshape(-1, D).float()
    x_norm, scale = normalize(x_flat)
    base = baseline_payload["codebooks"].float()[:2]
    _ids, recon2 = rvq_encode_norm(x_norm, base, device=device)
    residual = x_norm - recon2
    max_abs = residual.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    q = torch.round(residual / (max_abs / 127.0)).clamp(-127, 127)
    residual_hat = q * (max_abs / 127.0)
    return ((recon2 + residual_hat) * scale).reshape_as(x).contiguous()


def load_candidate_payload(root: Path, v: Variant, c: Candidate, tensor: str, layer: int, head: int) -> dict[str, Any]:
    if c.family == "pq":
        return load_rvq_payload(codebook_path(root, pq_subdir(v, c, tensor), layer, head))
    if c.name == "rvq2_int8_residual":
        base_subdir = baseline_k_subdir(v) if tensor == "K" else baseline_v_subdir(v)
        return load_rvq_payload(codebook_path(root, base_subdir, layer, head))
    return load_rvq_payload(codebook_path(root, codebook_subdir(v, c, tensor), layer, head))


def reconstruct_candidate(
    x: torch.Tensor,
    root: Path,
    v: Variant,
    c: Candidate,
    tensor: str,
    layer: int,
    head: int,
    device: torch.device,
) -> torch.Tensor:
    payload = load_candidate_payload(root, v, c, tensor, layer, head)
    if c.family == "pq":
        return pq_reconstruct(x, payload, device)
    if c.name == "rvq2_int8_residual":
        return hybrid_rvq2_int8_residual(x, payload, device)
    return rvq_reconstruct(x, payload, device)


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, a.shape[-1])
    ai = torch.topk(a, k=k, dim=-1).indices
    bi = torch.topk(b, k=k, dim=-1).indices
    overlap = (ai[:, :, None] == bi[:, None, :]).any(dim=-1).float().mean(dim=-1)
    return float(overlap.mean().item())


def eval_one_task(args: argparse.Namespace, task: dict[str, Any], device: torch.device) -> dict[str, Any]:
    root = args.output_dir
    v = variant_from_task(task)
    c = candidate_from_name(task["candidate"])
    layer = int(task["layer"])
    head = int(task["head"])
    pid = int(task["prompt_id"])
    out = root / "metrics" / "codebook_explore_rows" / f"{v.name}_{c.name}_p{pid}_l{layer}_h{head}.json"
    if out.exists() and not args.overwrite:
        return json.loads(out.read_text(encoding="utf-8"))
    payload = torch.load(layer_path(root, "test", pid, layer), map_location="cpu", weights_only=False)
    k = payload["K"][:, head].reshape(-1, D).float()
    v_tensor = payload["V"][:, head].reshape(-1, D).float()
    q = payload["Q"][:, head].reshape(-1, D).float()
    total = k.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed + pid * 1000000 + layer * 10000 + head * 100)
    recon_idx = torch.randperm(total, generator=gen)[: min(args.recon_sample, total)]
    q_idx = torch.randperm(total, generator=gen)[: min(args.query_sample, total)]
    key_idx = torch.randperm(total, generator=gen)[: min(args.key_sample, total)]

    kh_rec = reconstruct_candidate(k[recon_idx], root, v, c, "K", layer, head, device)
    vh_rec = reconstruct_candidate(v_tensor[recon_idx], root, v, c, "V", layer, head, device)
    k_ref = k[recon_idx]
    v_ref = v_tensor[recon_idx]
    k_err = k_ref - kh_rec
    v_err = v_ref - vh_rec
    k_mse = float((k_err * k_err).mean().item())
    v_mse = float((v_err * v_err).mean().item())
    k_cos = float(F.cosine_similarity(k_ref, kh_rec, dim=-1).mean().item())
    v_cos = float(F.cosine_similarity(v_ref, vh_rec, dim=-1).mean().item())

    q_s = q[q_idx].to(device)
    k_s = k[key_idx].to(device)
    v_s = v_tensor[key_idx].to(device)
    kh_s = reconstruct_candidate(k[key_idx], root, v, c, "K", layer, head, device).to(device)
    vh_s = reconstruct_candidate(v_tensor[key_idx], root, v, c, "V", layer, head, device).to(device)
    logits = q_s @ k_s.T / math.sqrt(D)
    logits_hat = q_s @ kh_s.T / math.sqrt(D)
    p = F.softmax(logits.float(), dim=-1)
    ph = F.softmax(logits_hat.float(), dim=-1)
    out_ref = p @ v_s.float()
    out_hat = ph @ vh_s.float()
    row = {
        "variant": v.name,
        "candidate": c.name,
        "family": c.family,
        "runtime_compatible": c.runtime,
        "prompt_id": pid,
        "layer": layer,
        "head": head,
        "K_MSE": k_mse,
        "K_cos": k_cos,
        "V_MSE": v_mse,
        "V_cos": v_cos,
        "QK_MSE": float(((logits - logits_hat) ** 2).mean().item()),
        "attention_KL": float((p * (torch.log(p.clamp_min(1e-8)) - torch.log(ph.clamp_min(1e-8)))).sum(dim=-1).mean().item()),
        "topk_overlap": topk_overlap(logits, logits_hat, args.topk),
        "QKV_MSE": float(((out_ref - out_hat) ** 2).mean().item()),
        "QKV_cos": float(F.cosine_similarity(out_ref, out_hat, dim=-1).mean().item()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def eval_worker(args: argparse.Namespace) -> None:
    torch.cuda.set_device(0) if torch.cuda.is_available() else None
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tasks = json.loads(args.task_file.read_text(encoding="utf-8"))
    rows = []
    for task in tasks:
        try:
            rows.append(eval_one_task(args, task, device))
        except RuntimeError as exc:
            rows.append({"status": "failed", "error": str(exc), **task})
            torch.cuda.empty_cache()
        except Exception as exc:
            rows.append({"status": "failed", "error": repr(exc), **task})
            torch.cuda.empty_cache()
    write_csv(args.task_file.with_suffix(".results.csv"), rows)


def launch_sharded(args: argparse.Namespace, mode: str, tasks: list[dict[str, Any]], gpus: list[int]) -> list[dict[str, Any]]:
    if not tasks:
        return []
    task_dir = args.output_dir / "status" / "codebook_explore" / f"{mode}_tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, gpu in enumerate(gpus):
        shard = tasks[i:: len(gpus)]
        if not shard:
            continue
        task_file = task_dir / f"{mode}_gpu{gpu}_{int(time.time())}.json"
        task_file.write_text(json.dumps(shard, indent=2), encoding="utf-8")
        log_path = args.output_dir / "logs" / "codebook_explore" / f"{mode}_gpu{gpu}_{task_file.stem}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["MASTER_PORT"] = str(30400 + gpu)
        log = log_path.open("w", encoding="utf-8")
        cmd = [
            str(args.worker_python),
            str(Path(__file__).resolve()),
            "--worker",
            mode,
            "--task-file",
            str(task_file),
            "--gpu",
            str(gpu),
            "--output-dir",
            str(args.output_dir),
            "--sample-size",
            str(args.sample_size),
            "--kmeans-iters",
            str(args.kmeans_iters),
            "--recon-sample",
            str(args.recon_sample),
            "--query-sample",
            str(args.query_sample),
            "--key-sample",
            str(args.key_sample),
            "--topk",
            str(args.topk),
            "--seed",
            str(args.seed),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        print(f"[{mode}] gpu={gpu} tasks={len(shard)} log={log_path}", flush=True)
        procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, task_file, log_path))
    failures = []
    result_rows = []
    for proc, log, task_file, log_path in procs:
        rc = proc.wait()
        log.close()
        result_path = task_file.with_suffix(".results.csv")
        if result_path.exists():
            with result_path.open(newline="", encoding="utf-8") as f:
                result_rows.extend(csv.DictReader(f))
        if rc != 0:
            failures.append({"mode": mode, "rc": rc, "log": str(log_path)})
    if failures:
        fail_path = args.output_dir / "metrics" / "codebook_explore_failures.json"
        fail_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"[warn] {mode} failures: {fail_path}", flush=True)
    return result_rows


def training_tasks(args: argparse.Namespace, variants: list[Variant], candidates: list[Candidate]) -> list[dict[str, Any]]:
    tasks = []
    for v in variants:
        for c in candidates:
            if c.name in {"baseline", "querycov_existing", "rvq2_int8_residual"}:
                continue
            for tensor in ("K", "V"):
                if tensor == "K" and not c.train_k:
                    continue
                if tensor == "V" and not c.train_v:
                    continue
                for layer in range(LAYERS):
                    for head in range(HEADS):
                        if c.family == "pq":
                            path = codebook_path(args.output_dir, pq_subdir(v, c, tensor), layer, head)
                        else:
                            path = codebook_path(args.output_dir, codebook_subdir(v, c, tensor), layer, head)
                        if path.exists() and not args.overwrite:
                            continue
                        tasks.append(
                            {
                                "variant": v.name,
                                "levels": v.levels,
                                "codebook_size": v.codebook_size,
                                "storage": v.storage,
                                "candidate": c.name,
                                "tensor": tensor,
                                "layer": layer,
                                "head": head,
                            }
                        )
    return tasks


def eval_tasks(args: argparse.Namespace, variants: list[Variant], candidates: list[Candidate]) -> list[dict[str, Any]]:
    test_ids = [int(x["prompt_id"]) for x in split_items(args.output_dir, "test")][: args.screen_prompts]
    tasks = []
    for v in variants:
        for c in candidates:
            for pid in test_ids:
                for layer in range(LAYERS):
                    for head in range(HEADS):
                        out = args.output_dir / "metrics" / "codebook_explore_rows" / f"{v.name}_{c.name}_p{pid}_l{layer}_h{head}.json"
                        if out.exists() and not args.overwrite:
                            continue
                        tasks.append(
                            {
                                "variant": v.name,
                                "levels": v.levels,
                                "codebook_size": v.codebook_size,
                                "storage": v.storage,
                                "candidate": c.name,
                                "prompt_id": pid,
                                "layer": layer,
                                "head": head,
                            }
                        )
    return tasks


def aggregate_offline(root: Path, variants: list[Variant], candidates: list[Candidate]) -> list[dict[str, Any]]:
    rows = []
    for path in (root / "metrics" / "codebook_explore_rows").glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if row.get("status") == "failed":
            continue
        rows.append(row)
    write_csv(root / "metrics" / "codebook_explore_offline_rows.csv", rows)
    summaries = []
    keys = ["K_MSE", "K_cos", "V_MSE", "V_cos", "QK_MSE", "attention_KL", "topk_overlap", "QKV_MSE", "QKV_cos"]
    for v in variants:
        for c in candidates:
            group = [r for r in rows if r.get("variant") == v.name and r.get("candidate") == c.name and "QKV_MSE" in r]
            if not group:
                continue
            out: dict[str, Any] = {
                "variant": v.name,
                "candidate": c.name,
                "family": c.family,
                "runtime_compatible": c.runtime,
                "offline_only_reason": c.offline_only_reason,
                "rows": len(group),
            }
            for key in keys:
                vals = [float(r[key]) for r in group if r.get(key) not in (None, "", "nan")]
                out[f"{key}_mean"] = sum(vals) / len(vals) if vals else None
            summaries.append(out)
    summaries.sort(key=lambda r: (r["variant"], float(r.get("QKV_MSE_mean") or 1e30)))
    write_csv(root / "metrics" / "codebook_explore_offline.csv", summaries)
    return summaries


def setting_for_storage(storage: str) -> str:
    if storage == "int4":
        return "kvrvq-i4"
    if storage == "int8":
        return "kvrvq-i8"
    return "kvrvq"


def storage_args(storage: str) -> list[str]:
    if storage == "int4":
        return ["--rvq_codebook_storage_dtype", "int4", "--rvq_scale_storage_dtype", "uint4"]
    if storage == "int8":
        return ["--rvq_codebook_storage_dtype", "int8", "--rvq_scale_storage_dtype", "uint8"]
    return ["--rvq_codebook_storage_dtype", "bf16", "--rvq_scale_storage_dtype", "bf16"]


def label(v: Variant, c: Candidate) -> str:
    return f"CodebookExplore-{c.name}-{v.name.upper()}-{v.storage.upper()}-perchunk"


def video_dir(root: Path, v: Variant, c: Candidate) -> Path:
    return root / "videos_codebook_explore" / clean(label(v, c))


def build_video_cmd(args: argparse.Namespace, v: Variant, c: Candidate) -> list[str]:
    return [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(args.output_dir / "prompts_test.txt"),
        "--num-prompts",
        "10",
        "--output-dir",
        str(video_dir(args.output_dir, v, c)),
        "--seed",
        str(args.seed),
        "--setting",
        setting_for_storage(v.storage),
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
        str(args.output_dir / codebook_subdir(v, c, "K")),
        "--v_rvq_codebook_dir",
        str(args.output_dir / codebook_subdir(v, c, "V")),
        "--kv_logical_head_dim",
        str(LOGICAL_KV_DIM),
        "--kv_quant_factor",
        "1",
        *storage_args(v.storage),
    ]


def metrics_complete(path: Path, expected_rows: int = 10) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as f:
            return sum(1 for _ in csv.DictReader(f)) >= expected_rows
    except Exception:
        return False


def select_video_jobs(variants: list[Variant], candidates: list[Candidate], summaries: list[dict[str, Any]], topk: int) -> list[tuple[Variant, Candidate]]:
    cand_map = {c.name: c for c in candidates}
    jobs = []
    for v in variants:
        # Always include matched baseline.
        if "baseline" in cand_map:
            jobs.append((v, cand_map["baseline"]))
        runtime_rows = [
            r
            for r in summaries
            if r["variant"] == v.name
            and r["candidate"] != "baseline"
            and cand_map[r["candidate"]].runtime
            and r.get("QKV_MSE_mean") is not None
        ]
        runtime_rows.sort(key=lambda r: float(r["QKV_MSE_mean"]))
        for row in runtime_rows[:topk]:
            jobs.append((v, cand_map[row["candidate"]]))
    seen = set()
    unique = []
    for v, c in jobs:
        key = (v.name, c.name)
        if key not in seen:
            unique.append((v, c))
            seen.add(key)
    return unique


def run_videos(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]], gpus: list[int]) -> list[dict[str, Any]]:
    if args.skip_video:
        return []
    failures = []
    pending = []
    for v, c in jobs:
        out = video_dir(args.output_dir, v, c)
        if metrics_complete(out / "metrics.csv") and not args.overwrite:
            print(f"[video-skip] {label(v, c)}", flush=True)
        else:
            pending.append((v, c))
    active: list[dict[str, Any]] = []
    next_i = 0
    while next_i < len(pending) or active:
        busy = {x["gpu"] for x in active}
        free = idle_gpus(args.min_free_gpu_mb)
        for gpu in gpus:
            if next_i >= len(pending):
                break
            if gpu in busy or gpu not in free:
                continue
            v, c = pending[next_i]
            next_i += 1
            out = video_dir(args.output_dir, v, c)
            out.mkdir(parents=True, exist_ok=True)
            name = label(v, c)
            log_path = args.output_dir / "logs" / "codebook_explore" / f"video_{clean(name)}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["MASTER_PORT"] = str(30600 + gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            cmd = build_video_cmd(args, v, c)
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[video] gpu={gpu} {name}", flush=True)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append({"proc": proc, "log": log, "gpu": gpu, "name": name, "log_path": log_path})
            busy.add(gpu)
        if not active and next_i < len(pending):
            print(f"[video-wait] no idle GPU in {gpus}", flush=True)
            time.sleep(60)
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
                failures.append({"setting": item["name"], "rc": rc, "log": str(item["log_path"])})
            else:
                print(f"[video-done] {item['name']}", flush=True)
        active = keep
    if failures:
        path = args.output_dir / "metrics" / "codebook_explore_failures.json"
        path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"[warn] video failures written to {path}", flush=True)
    return aggregate_runtime(args.output_dir, jobs)


def numeric(v: Any) -> float | None:
    if v in (None, "", "null"):
        return None
    try:
        return float(v)
    except Exception:
        return None


def avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def aggregate_runtime(root: Path, jobs: list[tuple[Variant, Candidate]]) -> list[dict[str, Any]]:
    rows = []
    for v, c in jobs:
        path = video_dir(root, v, c) / "metrics.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            items = list(csv.DictReader(f))
        def vals(key: str, drop0: bool = False) -> list[float]:
            src = items[1:] if drop0 and len(items) > 1 else items
            return [x for x in (numeric(row.get(key)) for row in src) if x is not None]
        rows.append(
            {
                "setting": label(v, c),
                "variant": v.name,
                "candidate": c.name,
                "e2e_mean_s": avg(vals("e2e_time_s")),
                "e2e_drop0_s": avg(vals("e2e_time_s", True)),
                "denoise_mean_s": avg(vals("denoise_time_s")),
                "quant_drop0_s": avg(vals("quant_time_s", True)),
                "peak_gpu_mem_mb": max(vals("peak_gpu_mem_mb") or [float("nan")]),
                "kv_cache_mem_mb": avg(vals("kv_cache_mem_mb")),
                "compression_ratio": avg(vals("compression_ratio")),
                "memory_note": "run_small_eval accounting; assignment-codebook extras are added in report when present",
            }
        )
    write_csv(root / "metrics" / "codebook_explore_runtime.csv", rows)
    return rows


def run_vs_bf16(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]], gpus: list[int]) -> list[dict[str, Any]]:
    if args.skip_vs_bf16:
        return []
    rows = []
    bf16 = args.output_dir / "videos" / "bf16"
    for v, c in jobs:
        name = label(v, c)
        test_dir = video_dir(args.output_dir, v, c)
        metrics_csv = test_dir / "metrics.csv"
        if not metrics_csv.exists():
            continue
        out_json = args.output_dir / "vs_bf16_codebook_explore" / f"{clean(name)}.json"
        out_csv = args.output_dir / "vs_bf16_codebook_explore" / f"{clean(name)}.csv"
        if not out_json.exists() or args.overwrite:
            try:
                run_cmd(
                    [
                        str(args.worker_python),
                        str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
                        "--reference-dir",
                        str(bf16),
                        "--test-dir",
                        str(test_dir),
                        "--metrics-csv",
                        str(metrics_csv),
                        "--setting",
                        name,
                        "--output-json",
                        str(out_json),
                        "--output-csv",
                        str(out_csv),
                        "--device",
                        "cuda",
                    ],
                    args.output_dir / "logs" / "codebook_explore" / f"vs_bf16_{clean(name)}.log",
                    env={"CUDA_VISIBLE_DEVICES": str(gpus[0])},
                    overwrite=True,
                )
            except Exception as exc:
                rows.append({"setting": name, "variant": v.name, "candidate": c.name, "error": str(exc)})
                continue
        if out_json.exists():
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            rows.append(
                {
                    "setting": name,
                    "variant": v.name,
                    "candidate": c.name,
                    "psnr": payload.get("psnr_mean"),
                    "ssim": payload.get("ssim_mean"),
                    "lpips": payload.get("lpips_mean"),
                }
            )
    write_csv(args.output_dir / "metrics" / "codebook_explore_video.csv", rows)
    return rows


def write_vbench_prompts(root: Path, v: Variant, c: Candidate) -> Path:
    mpath = video_dir(root, v, c) / "metrics.csv"
    with mpath.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = root / "vbench_codebook_explore" / f"prompts_{clean(label(v, c))}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({r["video_path"]: r["prompt"] for r in rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def parse_vbench_result(result_dir: Path) -> dict[str, Any]:
    row = {dim: None for dim in VBENCH_DIMS}
    for path in result_dir.rglob("*_eval_results.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for dim in VBENCH_DIMS:
            value = payload.get(dim)
            if isinstance(value, list) and value:
                row[dim] = value[0]
            elif isinstance(value, (int, float)):
                row[dim] = value
    return row


def run_vbench(args: argparse.Namespace, jobs: list[tuple[Variant, Candidate]], gpus: list[int]) -> list[dict[str, Any]]:
    if args.skip_vbench or not args.vbench_bin.exists():
        return []
    rows = []
    for v, c in jobs:
        mpath = video_dir(args.output_dir, v, c) / "metrics.csv"
        if not mpath.exists():
            continue
        name = label(v, c)
        prompt_json = write_vbench_prompts(args.output_dir, v, c)
        result_dir = args.output_dir / "vbench_codebook_explore" / clean(name)
        for dim in VBENCH_DIMS:
            dim_dir = result_dir / dim
            sentinel = dim_dir / "vbench.done"
            if sentinel.exists() and not args.overwrite:
                continue
            try:
                run_cmd(
                    [
                        str(args.vbench_bin),
                        "evaluate",
                        "--ngpus",
                        str(min(4, len(gpus))),
                        "--output_path",
                        str(dim_dir),
                        "--videos_path",
                        str(video_dir(args.output_dir, v, c)),
                        "--dimension",
                        dim,
                        "--mode",
                        "custom_input",
                        "--prompt_file",
                        str(prompt_json),
                    ],
                    args.output_dir / "logs" / "codebook_explore" / f"vbench_{clean(name)}_{dim}.log",
                    env={"CUDA_VISIBLE_DEVICES": ",".join(str(x) for x in gpus[: min(4, len(gpus))])},
                    sentinel=sentinel,
                    overwrite=args.overwrite,
                )
            except Exception as exc:
                (dim_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                print(f"[vbench-failed] {name} {dim}: {exc}", flush=True)
        parsed = parse_vbench_result(result_dir)
        parsed.update({"setting": name, "variant": v.name, "candidate": c.name})
        rows.append(parsed)
    write_csv(args.output_dir / "metrics" / "codebook_explore_vbench.csv", rows)
    return rows


def training_time_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in (root / "status" / "codebook_explore" / "train_tasks").glob("*.results.csv"):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "trained":
                    rows.append(row)
    write_csv(root / "metrics" / "codebook_explore_training_time.csv", rows)
    return rows


def make_report(
    args: argparse.Namespace,
    variants: list[Variant],
    candidates: list[Candidate],
    offline: list[dict[str, Any]],
    runtime: list[dict[str, Any]],
    video: list[dict[str, Any]],
    vbench: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    command: str,
) -> Path:
    runtime_by = {(r.get("variant"), r.get("candidate")): r for r in runtime}
    video_by = {(r.get("variant"), r.get("candidate")): r for r in video}
    vbench_by = {(r.get("variant"), r.get("candidate")): r for r in vbench}
    offline_by = {(r.get("variant"), r.get("candidate")): r for r in offline}
    train_sum: dict[tuple[str, str], float] = {}
    for row in train_rows:
        key = (row.get("variant", ""), row.get("candidate", ""))
        train_sum[key] = train_sum.get(key, 0.0) + float(row.get("wall_time_s") or 0.0)

    lines = [
        "# Codebook Structure Exploration",
        "",
        f"- Output root: `{args.output_dir}`",
        f"- Command: `{command}`",
        f"- Prompt split: 20 train / 10 test, seed {args.seed}",
        f"- Offline screening prompts: {args.screen_prompts}",
        f"- Top-k overlap k: {args.topk}",
        "",
        "## Summary Table",
        "",
        "| setting | method | PSNR | SSIM | LPIPS | subject | background | image | aesthetic | KV MB | ratio | e2e | denoise | quant | train time s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in variants:
        for c in candidates:
            key = (v.name, c.name)
            if key not in runtime_by and key not in video_by:
                continue
            rt = runtime_by.get(key, {})
            vi = video_by.get(key, {})
            vb = vbench_by.get(key, {})
            train_t = train_sum.get(key, None)
            lines.append(
                "| "
                + " | ".join(
                    [
                        label(v, c),
                        c.family,
                        fmt(vi.get("psnr")),
                        fmt(vi.get("ssim")),
                        fmt(vi.get("lpips")),
                        fmt(vb.get("subject_consistency")),
                        fmt(vb.get("background_consistency")),
                        fmt(vb.get("imaging_quality")),
                        fmt(vb.get("aesthetic_quality")),
                        fmt(rt.get("kv_cache_mem_mb")),
                        fmt(rt.get("compression_ratio")),
                        fmt(rt.get("e2e_mean_s")),
                        fmt(rt.get("denoise_mean_s")),
                        fmt(rt.get("quant_drop0_s")),
                        fmt(train_t),
                    ]
                )
                + " |"
            )
    lines += [
        "",
        "## Offline Diagnostic Table",
        "",
        "| config | candidate | family | runtime | K MSE | K cos | V MSE | V cos | QK MSE | attn KL | top-k | QKV MSE | QKV cos |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in offline:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    str(row["candidate"]),
                    str(row["family"]),
                    str(row["runtime_compatible"]),
                    fmt(row.get("K_MSE_mean")),
                    fmt(row.get("K_cos_mean")),
                    fmt(row.get("V_MSE_mean")),
                    fmt(row.get("V_cos_mean")),
                    fmt(row.get("QK_MSE_mean")),
                    fmt(row.get("attention_KL_mean")),
                    fmt(row.get("topk_overlap_mean")),
                    fmt(row.get("QKV_MSE_mean")),
                    fmt(row.get("QKV_cos_mean")),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Interpretation Notes",
        "",
        "- `baseline` is the matched existing RVQ codebook for the same L/C setting.",
        "- `querycov_existing` is the previous QK-objective K codebook with baseline V.",
        "- `hadamard_rvq` and `diagq_rvq_k` store assignment-space centroids/transform in addition to reconstruction centroids; this improves assignment flexibility but increases codebook metadata unless compressed separately.",
        "- `pq_contiguous` and `hadamard_pq` are offline-only in this run; they are useful for judging whether an OPQ/PQ-style runtime is worth implementing.",
        "- `rvq2_int8_residual` is also offline-only and intentionally reports a hybrid residual upper-bound, not a deployable memory-neutral codec.",
        "",
        "## Pareto Analysis",
        "",
    ]
    for v in variants:
        base = offline_by.get((v.name, "baseline"))
        if not base:
            continue
        base_qkv = float(base["QKV_MSE_mean"])
        better = [
            r
            for r in offline
            if r["variant"] == v.name and r["candidate"] != "baseline" and float(r["QKV_MSE_mean"]) < base_qkv
        ]
        better.sort(key=lambda r: float(r["QKV_MSE_mean"]))
        if better:
            names = ", ".join(f"{r['candidate']} ({float(r['QKV_MSE_mean']):.4f})" for r in better[:5])
            lines.append(f"- `{v.name}`: offline QKV improves over baseline for {names}.")
        else:
            lines.append(f"- `{v.name}`: no candidate beat baseline on offline QKV MSE in completed rows.")
    lines += [
        "",
        "## Recommended Next Step",
        "",
        "Use the completed video rows, if any, to choose the follow-up. If only offline rows are available, prioritize candidates that improve QKV MSE without collapsing K/V cosine and are runtime-compatible.",
        "",
        "## Files",
        "",
        "- `metrics/codebook_explore_offline.csv`",
        "- `metrics/codebook_explore_video.csv`",
        "- `metrics/codebook_explore_runtime.csv`",
        "- `metrics/codebook_explore_training_time.csv`",
        "- `metrics/codebook_explore_failures.json` if failures occurred",
    ]
    path = args.output_dir / "metrics" / "codebook_explore_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "null"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> None:
    args = parse_args()
    if args.worker == "train":
        train_worker(args)
        return
    if args.worker == "eval":
        eval_worker(args)
        return

    start = time.time()
    mkdirs(args.output_dir)
    variants = parse_variants(args.configs)
    candidates = parse_candidates(args.candidates)
    gpus = parse_gpus(args.gpus, args.min_free_gpu_mb)
    command = " ".join(sys.argv)
    (args.output_dir / "codebook_explore_config.json").write_text(
        json.dumps(
            {
                "command": command,
                "output_dir": str(args.output_dir),
                "prompt_source": str(args.prompt_source),
                "variants": [v.__dict__ for v in variants],
                "candidates": [c.__dict__ for c in candidates],
                "gpus": gpus,
                "seed": args.seed,
                "sample_size": args.sample_size,
                "kmeans_iters": args.kmeans_iters,
                "screen_prompts": args.screen_prompts,
                "topk": args.topk,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[gpus] {gpus}", flush=True)

    train_rows: list[dict[str, Any]] = []
    if not args.skip_train:
        tasks = training_tasks(args, variants, candidates)
        print(f"[train] pending tasks={len(tasks)}", flush=True)
        train_rows = launch_sharded(args, "train", tasks, gpus)
        training_time_rows(args.output_dir)
    else:
        train_rows = training_time_rows(args.output_dir)

    offline: list[dict[str, Any]] = []
    if not args.skip_offline:
        tasks = eval_tasks(args, variants, candidates)
        print(f"[offline] pending tasks={len(tasks)}", flush=True)
        launch_sharded(args, "eval", tasks, gpus)
        offline = aggregate_offline(args.output_dir, variants, candidates)
    else:
        offline = aggregate_offline(args.output_dir, variants, candidates)

    jobs = select_video_jobs(variants, candidates, offline, args.video_topk)
    print("[video-jobs] " + ", ".join(label(v, c) for v, c in jobs), flush=True)
    runtime = run_videos(args, jobs, gpus)
    video = run_vs_bf16(args, jobs, gpus)

    elapsed = time.time() - start
    remaining = args.max_hours * 3600 - elapsed
    if remaining > 1800:
        vbench = run_vbench(args, jobs, gpus)
    else:
        print(f"[vbench-skip] only {remaining/60:.1f} minutes left in budget", flush=True)
        vbench = []

    train_rows = training_time_rows(args.output_dir)
    offline = aggregate_offline(args.output_dir, variants, candidates)
    runtime = aggregate_runtime(args.output_dir, jobs)
    report = make_report(args, variants, candidates, offline, runtime, video, vbench, train_rows, command)
    print(f"[complete] report={report}", flush=True)


if __name__ == "__main__":
    main()
