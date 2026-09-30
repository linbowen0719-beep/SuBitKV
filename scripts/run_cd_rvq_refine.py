#!/usr/bin/env python3
"""Quick coordinate-descent refinement validation for L8C64 RVQ.

This script validates Experiment 2 only:

1. Load existing greedy L8C64 K/V RVQ codebooks.
2. Re-encode sampled train vectors once to get fixed 8-level assignments.
3. Refine codebooks by coordinate descent with assignments frozen.
4. Evaluate every pass through the same INT4 codebook-dequant path used at
   runtime, without generating videos.

Example:

python scripts/run_cd_rvq_refine.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --gpus 4,5,6,7 \
  --passes 5
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
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")

LAYERS = 30
HEADS = 24
LEVELS = 8
CODEBOOK_SIZE = 64


METRIC_KEYS = [
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--passes", type=int, default=5)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--assign-batch-size", type=int, default=8192)
    p.add_argument("--query-sample", type=int, default=128)
    p.add_argument("--key-sample", type=int, default=2048)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--skip-refine", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    # Worker-only arguments.
    p.add_argument("--worker", choices=["refine", "eval"], default="")
    p.add_argument("--tensor", choices=["K", "V"], default="K")
    p.add_argument("--task-values", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--metrics-rows-out", type=Path, default=None)
    p.add_argument("--stage-rows-out", type=Path, default=None)
    p.add_argument("--head-values", default="")
    return p.parse_args()


def parse_gpus(raw: str) -> list[int]:
    gpus = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def mkdirs(root: Path) -> None:
    for rel in [
        "metrics",
        "plots",
        "reports",
        "logs/cd_rvq_refine",
        "tmp/cd_rvq_refine/refine",
        "tmp/cd_rvq_refine/eval",
        "status/cd_rvq_refine",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads((root / "prompt_split.json").read_text(encoding="utf-8"))
    return payload[split]


def layer_file(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def baseline_dir(root: Path, tensor: str) -> Path:
    lower = tensor.lower()
    return root / f"sweep_{lower}_l8c64_codebooks"


def refined_dir(root: Path, tensor: str, pass_idx: int) -> Path:
    lower = tensor.lower()
    return root / f"cd_refine_{lower}_l8c64_pass{pass_idx}_codebooks"


def codebook_path(cb_dir: Path, layer: int, head: int) -> Path:
    return cb_dir / f"layer_{layer}_head_{head}.pt"


def all_tasks() -> list[int]:
    return [layer * 100 + head for layer in range(LAYERS) for head in range(HEADS)]


def codebook_complete(cb_dir: Path) -> bool:
    return all(codebook_path(cb_dir, layer, head).exists() for layer in range(LAYERS) for head in range(HEADS))


def marker_path(root: Path, tensor: str, task: int) -> Path:
    return root / "status" / "cd_rvq_refine" / f"{tensor.lower()}_{task}.done"


def load_payload(cb_dir: Path, layer: int, head: int) -> dict[str, Any]:
    path = codebook_path(cb_dir, layer, head)
    if not path.exists():
        raise FileNotFoundError(f"Missing codebook: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "codebooks" not in payload:
        return {"codebooks": payload, "config": {}}
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nearest_torch(x: torch.Tensor, centroids: torch.Tensor, batch_size: int) -> torch.Tensor:
    ids: list[torch.Tensor] = []
    c = centroids.float()
    c_sq = (c * c).sum(dim=1)
    for start in range(0, x.shape[0], batch_size):
        xb = x[start : start + batch_size].float()
        dist = (xb * xb).sum(dim=1, keepdim=True) + c_sq[None, :] - 2.0 * xb @ c.T
        ids.append(torch.argmin(dist, dim=1).cpu())
    return torch.cat(ids, dim=0).long()


def int4_dequant_codebooks(codebooks: torch.Tensor) -> torch.Tensor:
    """Match subitkv.rvq per-vector symmetric INT4 codebook storage."""
    max_abs = codebooks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / 7.0
    q = torch.round(codebooks.float() / scale).clamp(-7, 7)
    return (q * scale).float().contiguous()


@torch.no_grad()
def encode_reconstruct_vectors(
    x: torch.Tensor,
    codebooks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Greedy RVQ reconstruct sampled vectors with pre-dequantized codebooks."""
    x_dev = x.float().to(device)
    scale = torch.sqrt(torch.mean(x_dev * x_dev, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x_dev / scale
    recon_norm = torch.zeros_like(residual)
    for level in range(codebooks.shape[0]):
        centroids = codebooks[level].to(device=device, dtype=torch.float32)
        ids_cpu = nearest_torch(residual, centroids, batch_size)
        ids = ids_cpu.to(device)
        assigned = centroids[ids]
        recon_norm += assigned
        residual -= assigned
    return (recon_norm * scale).detach().cpu()


def sample_train_vectors(
    root: Path,
    layer: int,
    head: int,
    tensor_name: str,
    sample_size: int,
    seed: int,
) -> torch.Tensor:
    items = split_items(root, "train")
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    chunks: list[torch.Tensor] = []
    remaining = sample_size
    for item in items:
        if remaining <= 0:
            break
        prompt_id = int(item["prompt_id"])
        payload = torch.load(layer_file(root, "train", prompt_id, layer), map_location="cpu", weights_only=False)
        tensor = payload[tensor_name][:, head].reshape(-1, payload[tensor_name].shape[-1])
        gen = torch.Generator(device="cpu").manual_seed(seed + layer * 100000 + head * 1000 + prompt_id)
        take = min(per_prompt, remaining, tensor.shape[0])
        indices = torch.randperm(tensor.shape[0], generator=gen)[:take]
        chunks.append(tensor[indices].float())
        remaining -= take
    if not chunks:
        raise RuntimeError(f"No train samples for tensor={tensor_name} layer={layer} head={head}")
    return torch.cat(chunks, dim=0).contiguous()


@torch.no_grad()
def fixed_assignments(
    x_norm: torch.Tensor,
    codebooks: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = x_norm.to(device=device, dtype=torch.float32)
    ids: list[torch.Tensor] = []
    contribs: list[torch.Tensor] = []
    for level in range(codebooks.shape[0]):
        centroids = codebooks[level].to(device=device, dtype=torch.float32)
        level_ids = nearest_torch(residual, centroids, batch_size)
        assigned = centroids[level_ids.to(device)].detach().cpu()
        residual = residual - assigned.to(device)
        ids.append(level_ids.to(torch.long))
        contribs.append(assigned.float())
    return torch.stack(ids, dim=0), torch.stack(contribs, dim=0)


def save_codebook(
    root: Path,
    tensor: str,
    pass_idx: int,
    layer: int,
    head: int,
    codebooks: torch.Tensor,
    base_config: dict[str, Any],
    sample_size: int,
) -> None:
    out_dir = refined_dir(root, tensor, pass_idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = dict(base_config)
    config.update(
        {
            "layer": layer,
            "head": head,
            "levels": int(codebooks.shape[0]),
            "codebook_size": int(codebooks.shape[1]),
            "sample_size": int(sample_size),
            "normalization": "per-vector RMS",
            "tensor": tensor,
            "refinement": "coordinate_descent_fixed_assignments",
            "refinement_pass": int(pass_idx),
            "runtime_eval_storage": "int4",
        }
    )
    torch.save({"codebooks": codebooks.cpu().float().contiguous(), "config": config}, codebook_path(out_dir, layer, head))


def residual_stage_rows(
    root: Path,
    tensor: str,
    pass_idx: int,
    layer: int,
    head: int,
    x_norm: torch.Tensor,
    ids: torch.Tensor,
    codebooks: torch.Tensor,
) -> list[dict[str, Any]]:
    cb_i4 = int4_dequant_codebooks(codebooks)
    residual = x_norm.float().clone()
    rows: list[dict[str, Any]] = []
    for level in range(cb_i4.shape[0]):
        assigned = cb_i4[level][ids[level].long()]
        residual -= assigned.float()
        rows.append(
            {
                "tensor": tensor,
                "pass": pass_idx,
                "layer": layer,
                "head": head,
                "level": level + 1,
                "residual_mse": float((residual * residual).mean().item()),
            }
        )
    return rows


def refine_one(args: argparse.Namespace, tensor: str, layer: int, head: int) -> list[dict[str, Any]]:
    done = marker_path(args.output_dir, tensor, layer * 100 + head)
    if done.exists() and not args.overwrite:
        return []
    base = load_payload(baseline_dir(args.output_dir, tensor), layer, head)
    base_codebooks = base["codebooks"].float().contiguous()
    if base_codebooks.shape[:2] != (LEVELS, CODEBOOK_SIZE):
        raise ValueError(f"Expected L8C64 codebook, got {tuple(base_codebooks.shape)}")
    x = sample_train_vectors(args.output_dir, layer, head, tensor, args.sample_size, args.seed)
    scale = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True)).clamp_min(1e-6)
    x_norm = (x / scale).float().contiguous()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ids, contribs = fixed_assignments(x_norm, base_codebooks, device, args.assign_batch_size)
    codebooks = base_codebooks.clone()
    stage_rows: list[dict[str, Any]] = residual_stage_rows(args.output_dir, tensor, 0, layer, head, x_norm, ids, codebooks)
    recon = contribs.sum(dim=0)
    base_config = base.get("config", {}) if isinstance(base, dict) else {}

    for pass_idx in range(1, args.passes + 1):
        for level in range(LEVELS):
            without_level = recon - contribs[level]
            target = x_norm - without_level
            level_ids = ids[level].long()
            sums = torch.zeros_like(codebooks[level])
            counts = torch.zeros((CODEBOOK_SIZE, 1), dtype=torch.float32)
            sums.index_add_(0, level_ids, target.float())
            counts.index_add_(0, level_ids, torch.ones((target.shape[0], 1), dtype=torch.float32))
            updated = sums / counts.clamp_min(1.0)
            empty = counts.squeeze(1) <= 0
            if empty.any():
                updated[empty] = codebooks[level][empty]
            codebooks[level] = updated
            new_contrib = codebooks[level][level_ids].float()
            recon = without_level + new_contrib
            contribs[level] = new_contrib
        save_codebook(args.output_dir, tensor, pass_idx, layer, head, codebooks, base_config, x.shape[0])
        stage_rows.extend(residual_stage_rows(args.output_dir, tensor, pass_idx, layer, head, x_norm, ids, codebooks))
        mse = float(((x_norm - recon) ** 2).mean().item())
        print(
            f"[refine] tensor={tensor} layer={layer} head={head} pass={pass_idx} "
            f"train_norm_mse={mse:.8g}",
            flush=True,
        )
    done.write_text("done\n", encoding="utf-8")
    return stage_rows


def refine_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tasks = [int(x) for x in args.task_values.split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        rows.extend(refine_one(args, args.tensor, task // 100, task % 100))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.stage_rows_out is not None:
        write_csv(args.stage_rows_out, rows)


def codebook_for_pass(root: Path, tensor: str, pass_idx: int) -> Path:
    return baseline_dir(root, tensor) if pass_idx == 0 else refined_dir(root, tensor, pass_idx)


def load_layer_codebooks_all(root: Path, tensor: str, pass_idx: int, layer: int, device: torch.device) -> torch.Tensor:
    """Load all 24 heads as [L, H, C, D] after INT4 dequantization."""
    heads = []
    cb_dir = codebook_for_pass(root, tensor, pass_idx)
    for head in range(HEADS):
        payload = load_payload(cb_dir, layer, head)
        heads.append(payload["codebooks"].float())
    codebooks = torch.stack(heads, dim=1).contiguous()
    return int4_dequant_codebooks(codebooks).to(device=device, dtype=torch.float32)


@torch.no_grad()
def encode_reconstruct_heads(
    x: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    """Vectorized greedy RVQ for [H, N, D] with [L, H, C, D]."""
    x_f = x.float()
    scale = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x_f / scale
    recon_norm = torch.zeros_like(residual)
    for level in range(codebooks.shape[0]):
        centroids = codebooks[level].float()  # [H, C, D]
        centroid_sq = (centroids * centroids).sum(dim=-1)  # [H, C]
        rb_sq = (residual * residual).sum(dim=-1, keepdim=True)
        dots = torch.bmm(residual, centroids.transpose(1, 2))
        dist = rb_sq + centroid_sq[:, None, :] - 2.0 * dots
        ids = torch.argmin(dist, dim=-1)
        assigned = torch.gather(centroids, 1, ids[..., None].expand(-1, -1, centroids.shape[-1]))
        recon_norm += assigned
        residual -= assigned
    return recon_norm * scale


@torch.no_grad()
def eval_one_layer(
    args: argparse.Namespace,
    payload: dict[str, Any],
    split: str,
    prompt_id: int,
    layer: int,
    pass_idx: int,
    device: torch.device,
    head_values: list[int],
    k_codebooks: torch.Tensor,
    v_codebooks: torch.Tensor,
) -> list[dict[str, Any]]:
    q_all = payload["Q"].permute(1, 0, 2, 3).reshape(HEADS, -1, payload["Q"].shape[-1]).float()
    k_all = payload["K"].permute(1, 0, 2, 3).reshape(HEADS, -1, payload["K"].shape[-1]).float()
    v_all = payload["V"].permute(1, 0, 2, 3).reshape(HEADS, -1, payload["V"].shape[-1]).float()
    n_tokens = k_all.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed + prompt_id * 1000000 + layer * 10000 + 17)
    q_take = min(args.query_sample, n_tokens)
    k_take = min(args.key_sample, n_tokens)
    q_idx = torch.randperm(n_tokens, generator=gen)[:q_take].to(device)
    kv_idx = torch.randperm(n_tokens, generator=gen)[:k_take].to(device)

    heads = torch.tensor(head_values, dtype=torch.long)
    q_s = q_all.index_select(0, heads).to(device).index_select(1, q_idx)
    k_s = k_all.index_select(0, heads).to(device).index_select(1, kv_idx)
    v_s = v_all.index_select(0, heads).to(device).index_select(1, kv_idx)
    k_cb = k_codebooks.index_select(1, heads.to(device))
    v_cb = v_codebooks.index_select(1, heads.to(device))
    kh_s = encode_reconstruct_heads(k_s, k_cb)
    vh_s = encode_reconstruct_heads(v_s, v_cb)

    k_mse = ((k_s - kh_s) ** 2).mean(dim=(1, 2))
    v_mse = ((v_s - vh_s) ** 2).mean(dim=(1, 2))
    k_cos = F.cosine_similarity(k_s, kh_s, dim=-1).mean(dim=1)
    v_cos = F.cosine_similarity(v_s, vh_s, dim=-1).mean(dim=1)

    logits = torch.bmm(q_s, k_s.transpose(1, 2)) / math.sqrt(k_s.shape[-1])
    logits_hat = torch.bmm(q_s, kh_s.transpose(1, 2)) / math.sqrt(k_s.shape[-1])
    log_p = F.log_softmax(logits.float(), dim=-1)
    log_q = F.log_softmax(logits_hat.float(), dim=-1)
    p = log_p.exp()
    p_hat = log_q.exp()
    av = torch.bmm(p, v_s.float())
    av_vhat = torch.bmm(p, vh_s.float())
    qkv = torch.bmm(p_hat, vh_s.float())
    topk = min(int(args.topk), logits.shape[-1])
    top_a = torch.topk(logits, k=topk, dim=-1).indices
    top_b = torch.topk(logits_hat, k=topk, dim=-1).indices
    overlap = (top_a[..., :, None] == top_b[..., None, :]).any(dim=-1).float().mean(dim=(1, 2))
    qk_mse = ((logits - logits_hat) ** 2).mean(dim=(1, 2))
    kl = (p * (log_p - log_q)).sum(dim=-1).mean(dim=1)
    av_mse = ((av - av_vhat) ** 2).mean(dim=(1, 2))
    qkv_mse = ((av - qkv) ** 2).mean(dim=(1, 2))
    qkv_cos = F.cosine_similarity(av, qkv, dim=-1).mean(dim=1)

    rows: list[dict[str, Any]] = []
    for idx, head in enumerate(head_values):
        rows.append(
            {
                "setting": "RVQ-L8C64-INT4" if pass_idx == 0 else f"CD-RVQ-L8C64-INT4-pass{pass_idx}",
                "method": "baseline" if pass_idx == 0 else "coordinate_descent_fixed_assignments",
                "pass": pass_idx,
                "split": split,
                "prompt_id": prompt_id,
                "layer": layer,
                "head": head,
                "k_reconstruction_mse": float(k_mse[idx].item()),
                "k_reconstruction_cosine": float(k_cos[idx].item()),
                "v_reconstruction_mse": float(v_mse[idx].item()),
                "v_reconstruction_cosine": float(v_cos[idx].item()),
                "qk_logits_mse": float(qk_mse[idx].item()),
                "attention_kl": float(kl[idx].item()),
                "topk_overlap": float(overlap[idx].item()),
                "attn_output_mse": float(av_mse[idx].item()),
                "qkv_output_mse": float(qkv_mse[idx].item()),
                "qkv_output_cosine": float(qkv_cos[idx].item()),
                "query_sample": q_take,
                "key_sample": k_take,
            }
        )
    return rows


def eval_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tasks = [int(x) for x in args.task_values.split(",") if x.strip()]
    heads = [int(x) for x in args.head_values.split(",") if x.strip()] if args.head_values.strip() else list(range(HEADS))
    prompt_ids = [int(item["prompt_id"]) for item in split_items(args.output_dir, args.split)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    for layer in tasks:
        codebooks_by_pass = [
            (
                load_layer_codebooks_all(args.output_dir, "K", pass_idx, layer, device),
                load_layer_codebooks_all(args.output_dir, "V", pass_idx, layer, device),
            )
            for pass_idx in range(args.passes + 1)
        ]
        for prompt_id in prompt_ids:
            payload = torch.load(layer_file(args.output_dir, args.split, prompt_id, layer), map_location="cpu", weights_only=False)
            for pass_idx, (k_codebooks, v_codebooks) in enumerate(codebooks_by_pass):
                rows.extend(
                    eval_one_layer(
                        args,
                        payload,
                        args.split,
                        prompt_id,
                        layer,
                        pass_idx,
                        device,
                        heads,
                        k_codebooks,
                        v_codebooks,
                    )
                )
            print(f"[eval] split={args.split} gpu={args.gpu} layer={layer} prompt={prompt_id}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if args.metrics_rows_out is None:
        raise ValueError("--metrics-rows-out is required for eval worker")
    write_csv(args.metrics_rows_out, rows)


def launch_refine(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_refine:
        return
    tmp = args.output_dir / "tmp" / "cd_rvq_refine" / "refine"
    if tmp.exists() and args.overwrite:
        shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
    for tensor in ("K", "V"):
        missing = [task for task in all_tasks() if args.overwrite or not marker_path(args.output_dir, tensor, task).exists()]
        if not missing:
            print(f"[refine-skip] tensor={tensor}", flush=True)
            continue
        procs = []
        for i, gpu in enumerate(gpu_ids):
            tasks = missing[i::len(gpu_ids)]
            if not tasks:
                continue
            part = tmp / f"stage_{tensor.lower()}_gpu{gpu}.csv"
            log_path = args.output_dir / "logs" / "cd_rvq_refine" / f"refine_{tensor.lower()}_gpu{gpu}.log"
            cmd = [
                str(args.worker_python),
                str(Path(__file__).resolve()),
                "--worker",
                "refine",
                "--output-dir",
                str(args.output_dir),
                "--passes",
                str(args.passes),
                "--sample-size",
                str(args.sample_size),
                "--seed",
                str(args.seed),
                "--assign-batch-size",
                str(args.assign_batch_size),
                "--tensor",
                tensor,
                "--gpu",
                str(gpu),
                "--task-values",
                ",".join(str(x) for x in tasks),
                "--stage-rows-out",
                str(part),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[refine-launch] tensor={tensor} gpu={gpu} tasks={len(tasks)}", flush=True)
            procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        failures = []
        for proc, log, log_path in procs:
            rc = proc.wait()
            log.close()
            if rc != 0:
                failures.append((rc, log_path))
        if failures:
            raise RuntimeError("refine failed:\n" + "\n".join(f"rc={rc} log={path}" for rc, path in failures))

    stage_rows: list[dict[str, Any]] = []
    for part in sorted(tmp.glob("stage_*.csv")):
        stage_rows.extend(read_csv_rows(part))
    write_csv(args.output_dir / "metrics" / "cd_rvq_refine_stage_gain.csv", stage_rows)
    write_csv(args.output_dir / "metrics" / "cd_rvq_refine_stage_gain_summary.csv", aggregate_stage_gain(stage_rows))


def launch_eval(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_eval:
        return
    rows_path = args.output_dir / "metrics" / "cd_rvq_refine_rows.csv"
    summary_path = args.output_dir / "metrics" / "cd_rvq_refine_summary.csv"
    if rows_path.exists() and summary_path.exists() and not args.overwrite:
        print(f"[eval-skip] {summary_path}", flush=True)
        return
    tmp = args.output_dir / "tmp" / "cd_rvq_refine" / "eval"
    if tmp.exists() and args.overwrite:
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        procs = []
        for i, gpu in enumerate(gpu_ids):
            layers = list(range(LAYERS))[i::len(gpu_ids)]
            if not layers:
                continue
            part = tmp / f"rows_{split}_gpu{gpu}.csv"
            if part.exists() and not args.overwrite:
                continue
            log_path = args.output_dir / "logs" / "cd_rvq_refine" / f"eval_{split}_gpu{gpu}.log"
            cmd = [
                str(args.worker_python),
                str(Path(__file__).resolve()),
                "--worker",
                "eval",
                "--output-dir",
                str(args.output_dir),
                "--passes",
                str(args.passes),
                "--seed",
                str(args.seed),
                "--assign-batch-size",
                str(args.assign_batch_size),
                "--query-sample",
                str(args.query_sample),
                "--key-sample",
                str(args.key_sample),
                "--topk",
                str(args.topk),
                "--split",
                split,
                "--gpu",
                str(gpu),
                "--task-values",
                ",".join(str(x) for x in layers),
                "--metrics-rows-out",
                str(part),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[eval-launch] split={split} gpu={gpu} layers={len(layers)}", flush=True)
            procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        failures = []
        for proc, log, log_path in procs:
            rc = proc.wait()
            log.close()
            if rc != 0:
                failures.append((rc, log_path))
        if failures:
            raise RuntimeError("eval failed:\n" + "\n".join(f"rc={rc} log={path}" for rc, path in failures))
    rows: list[dict[str, str]] = []
    for part in sorted(tmp.glob("rows_*.csv")):
        rows.extend(read_csv_rows(part))
    write_csv(rows_path, rows)
    summary = aggregate_rows(rows)
    write_csv(summary_path, summary)
    write_csv(args.output_dir / "metrics" / "cd_rvq_refine_train_test_gap.csv", train_test_gap(summary))
    make_plot(args.output_dir, summary)
    write_report(args.output_dir, summary)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((int(row["pass"]), str(row["split"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (pass_idx, split), group in sorted(groups.items()):
        result: dict[str, Any] = {
            "pass": pass_idx,
            "setting": "RVQ-L8C64-INT4" if pass_idx == 0 else f"CD-RVQ-L8C64-INT4-pass{pass_idx}",
            "method": "baseline" if pass_idx == 0 else "coordinate_descent_fixed_assignments",
            "split": split,
            "n_rows": len(group),
        }
        for key in METRIC_KEYS:
            vals = [float(r[key]) for r in group if r.get(key) not in ("", None, "null")]
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            result[f"{key}_mean"] = mean
            result[f"{key}_std"] = math.sqrt(var)
        out.append(result)
    return out


def train_test_gap(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(int(r["pass"]), str(r["split"])): r for r in summary}
    rows: list[dict[str, Any]] = []
    for pass_idx in sorted({int(r["pass"]) for r in summary}):
        train = by_key.get((pass_idx, "train"))
        test = by_key.get((pass_idx, "test"))
        if not train or not test:
            continue
        row: dict[str, Any] = {"pass": pass_idx}
        for key in METRIC_KEYS:
            tr = float(train[f"{key}_mean"])
            te = float(test[f"{key}_mean"])
            row[f"train_{key}"] = tr
            row[f"test_{key}"] = te
            row[f"gap_{key}"] = te - tr
        rows.append(row)
    return rows


def aggregate_stage_gain(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        groups.setdefault((str(row["tensor"]), int(row["pass"]), int(row["level"])), []).append(float(row["residual_mse"]))
    baseline: dict[tuple[str, int], float] = {}
    for (tensor, pass_idx, level), vals in groups.items():
        if pass_idx == 0:
            baseline[(tensor, level)] = sum(vals) / len(vals)
    out: list[dict[str, Any]] = []
    for (tensor, pass_idx, level), vals in sorted(groups.items()):
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        base = baseline.get((tensor, level), mean)
        gain = (base - mean) / abs(base) * 100.0 if base else 0.0
        out.append(
            {
                "tensor": tensor,
                "pass": pass_idx,
                "level": level,
                "n": len(vals),
                "residual_mse_mean": mean,
                "residual_mse_std": math.sqrt(var),
                "baseline_pass0_residual_mse_mean": base,
                "gain_vs_pass0_pct": gain,
            }
        )
    return out


def make_plot(root: Path, summary: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot-skip] matplotlib unavailable: {exc}", flush=True)
        return
    test = sorted([r for r in summary if r["split"] == "test"], key=lambda r: int(r["pass"]))
    if not test:
        return
    passes = [int(r["pass"]) for r in test]
    metrics = [
        ("k_reconstruction_mse_mean", "K MSE"),
        ("v_reconstruction_mse_mean", "V MSE"),
        ("qkv_output_mse_mean", "QKV output MSE"),
        ("attention_kl_mean", "Attention KL"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, title) in zip(axes.reshape(-1), metrics):
        vals = [float(r[key]) for r in test]
        ax.plot(passes, vals, marker="o")
        ax.set_title(title)
        ax.set_xlabel("CD pass")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = root / "plots" / "cd_rvq_refine_convergence.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)


def pct_change(old: float, new: float, higher_is_better: bool = False) -> float:
    if old == 0:
        return 0.0
    if higher_is_better:
        return (new - old) / abs(old) * 100.0
    return (old - new) / abs(old) * 100.0


def write_report(root: Path, summary: list[dict[str, Any]]) -> None:
    test = {int(r["pass"]): r for r in summary if r["split"] == "test"}
    train = {int(r["pass"]): r for r in summary if r["split"] == "train"}
    base = test.get(0)
    best = None
    if base:
        best = min((r for p, r in test.items() if p > 0), key=lambda r: float(r["qkv_output_mse_mean"]), default=None)
    lines = [
        "# CD RVQ Refinement L8C64 INT4 Summary",
        "",
        "This is an offline cache-level quick validation. It uses fixed baseline RVQ assignments,",
        "then refines codebooks with coordinate descent. Every pass is evaluated after applying",
        "the same per-codeword INT4 dequantization used by runtime codebook storage.",
        "",
        f"- Root: `{root}`",
        "- Baseline: `sweep_k_l8c64_codebooks`, `sweep_v_l8c64_codebooks`",
        "- Evaluation: all train/test prompts, all 30 layers and 24 heads, sampled query/key tokens per head.",
        "- No video generation or VBench was run.",
        "",
        "## Test Means",
        "",
        "| pass | K MSE | K cos | V MSE | V cos | QK MSE | attn KL | top-k | QKV MSE | QKV cos |",
        "| ---: | ----: | ----: | ----: | ----: | -----: | ------: | ----: | ------: | ------: |",
    ]
    for pass_idx in sorted(test):
        r = test[pass_idx]
        lines.append(
            "| {pass_idx} | {k_mse:.6f} | {k_cos:.6f} | {v_mse:.6f} | {v_cos:.6f} | "
            "{qk:.6f} | {kl:.6f} | {topk:.6f} | {qkv:.6f} | {qkv_cos:.6f} |".format(
                pass_idx=pass_idx,
                k_mse=float(r["k_reconstruction_mse_mean"]),
                k_cos=float(r["k_reconstruction_cosine_mean"]),
                v_mse=float(r["v_reconstruction_mse_mean"]),
                v_cos=float(r["v_reconstruction_cosine_mean"]),
                qk=float(r["qk_logits_mse_mean"]),
                kl=float(r["attention_kl_mean"]),
                topk=float(r["topk_overlap_mean"]),
                qkv=float(r["qkv_output_mse_mean"]),
                qkv_cos=float(r["qkv_output_cosine_mean"]),
            )
        )
    lines.extend(["", "## Quick Decision", ""])
    if base and best:
        bp = int(best["pass"])
        qkv_gain = pct_change(float(base["qkv_output_mse_mean"]), float(best["qkv_output_mse_mean"]))
        k_gain = pct_change(float(base["k_reconstruction_mse_mean"]), float(best["k_reconstruction_mse_mean"]))
        v_gain = pct_change(float(base["v_reconstruction_mse_mean"]), float(best["v_reconstruction_mse_mean"]))
        kl_gain = pct_change(float(base["attention_kl_mean"]), float(best["attention_kl_mean"]))
        topk_gain = pct_change(float(base["topk_overlap_mean"]), float(best["topk_overlap_mean"]), higher_is_better=True)
        tr = train.get(bp)
        overfit_note = "train row unavailable"
        if tr:
            train_qkv_gain = pct_change(
                float(train[0]["qkv_output_mse_mean"]),
                float(tr["qkv_output_mse_mean"]),
            )
            overfit_note = f"train QKV gain {train_qkv_gain:.2f}% vs test QKV gain {qkv_gain:.2f}%"
        lines.extend(
            [
                f"- Best test pass by QKV output MSE: pass {bp}.",
                f"- Test K MSE gain: {k_gain:.2f}%.",
                f"- Test V MSE gain: {v_gain:.2f}%.",
                f"- Test QKV output MSE gain: {qkv_gain:.2f}%.",
                f"- Test attention KL gain: {kl_gain:.2f}%; top-k change: {topk_gain:.2f}%.",
                f"- Train/test gap note: {overfit_note}.",
            ]
        )
        max_any_gain = max(k_gain, v_gain, qkv_gain, kl_gain, topk_gain)
        worth = (max(k_gain, v_gain, qkv_gain) >= 3.0) or (kl_gain >= 3.0 and topk_gain >= 0.0)
        if max_any_gain < 1.0:
            lines.append("- Verdict: gain is below 1%; not worth pursuing as a paper main line.")
        elif worth:
            lines.append("- Verdict: positive enough for a follow-up video-level validation.")
        else:
            lines.append("- Verdict: mixed or small gain; inspect train/test gap before spending video runs.")
    else:
        lines.append("- Verdict unavailable because baseline or refined test rows are missing.")
    lines.append("")
    lines.append("Detailed rows are in `metrics/cd_rvq_refine_rows.csv`; stage residual rows are in `metrics/cd_rvq_refine_stage_gain.csv`.")
    out = root / "reports" / "cd_rvq_refine_l8c64_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {out}", flush=True)


def main() -> None:
    args = parse_args()
    mkdirs(args.output_dir)
    if args.worker == "refine":
        refine_worker(args)
        return
    if args.worker == "eval":
        eval_worker(args)
        return
    for tensor in ("K", "V"):
        if not codebook_complete(baseline_dir(args.output_dir, tensor)):
            raise FileNotFoundError(f"Missing baseline {tensor} codebooks: {baseline_dir(args.output_dir, tensor)}")
    gpus = parse_gpus(args.gpus)
    t0 = time.perf_counter()
    launch_refine(args, gpus)
    launch_eval(args, gpus)
    elapsed = time.perf_counter() - t0
    print(f"[done] cd_rvq_refine elapsed_s={elapsed:.1f}", flush=True)


if __name__ == "__main__":
    main()
