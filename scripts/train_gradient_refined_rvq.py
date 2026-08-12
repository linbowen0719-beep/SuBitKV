#!/usr/bin/env python3
"""KMeans-initialized gradient-refined RVQ training for KV cache codebooks.

This script refines existing per-layer/per-head RVQ codebooks without changing
the runtime RVQ format.  It trains codebooks in the same normalized space used
by the current runtime encoder:

    x_norm = x / rms(x)
    x_hat = rms(x) * sum_l codebook_l[index_l]

Example smoke test:

python scripts/train_gradient_refined_rvq.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --gpus 4 \
  --layers 0 \
  --heads 0 \
  --sample-size 2048 \
  --eval-sample-size 1024 \
  --stage1-epochs 1 \
  --stage2-epochs 1 \
  --batch-size 512 \
  --num-q 64 \
  --num-kv 128 \
  --eval-interval-steps 2 \
  --log-interval-steps 1 \
  --overwrite

Full run:

python scripts/train_gradient_refined_rvq.py \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test \
  --gpus auto
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")

LAYERS_DEFAULT = 30
HEADS_DEFAULT = 24
LEVELS_DEFAULT = 8
CODEBOOK_SIZE_DEFAULT = 64
LOGICAL_KV_DIM = 256

METRIC_KEYS = [
    "k_reconstruction_mse",
    "k_reconstruction_cosine",
    "v_reconstruction_mse",
    "v_reconstruction_cosine",
    "qk_logits_mse",
    "qk_normalized_mse",
    "attention_kl",
    "topk_overlap",
    "attn_output_mse",
    "av_normalized_mse",
    "qkv_output_mse",
    "qkv_output_cosine",
]


def parse_int_list(raw: str, default_count: int) -> list[int]:
    raw = str(raw).strip()
    if not raw or raw.lower() in {"all", "*"}:
        return list(range(default_count))
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            out.extend(range(int(lo_s), int(hi_s) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--init-k-dir", type=Path, default=None)
    p.add_argument("--init-v-dir", type=Path, default=None)
    p.add_argument("--worker-python", type=Path, default=DEFAULT_WORKER_PYTHON)
    p.add_argument("--gpus", default="auto")
    p.add_argument("--min-free-mb", type=int, default=30000)
    p.add_argument("--gpu-poll-s", type=int, default=60)
    p.add_argument("--layers", default="all")
    p.add_argument("--heads", default="all")
    p.add_argument("--levels", type=int, default=LEVELS_DEFAULT)
    p.add_argument("--codebook-size", type=int, default=CODEBOOK_SIZE_DEFAULT)
    p.add_argument("--sample-size", type=int, default=50000)
    p.add_argument("--eval-sample-size", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--stage1-epochs", type=int, default=5)
    p.add_argument("--stage2-epochs", type=int, default=5)
    p.add_argument("--lr-k-stage1", type=float, default=1e-3)
    p.add_argument("--lr-v-stage1", type=float, default=1e-3)
    p.add_argument("--lr-k-stage2", type=float, default=5e-4)
    p.add_argument("--lr-v-stage2", type=float, default=5e-4)
    p.add_argument("--beta-k", type=float, default=0.1)
    p.add_argument("--beta-v", type=float, default=0.1)
    p.add_argument("--lambda-drift", type=float, default=1e-4)
    p.add_argument("--num-q", type=int, default=512)
    p.add_argument("--num-kv", type=int, default=1024)
    p.add_argument("--eval-interval-steps", type=int, default=100)
    p.add_argument("--log-interval-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--offline-query-sample", type=int, default=128)
    p.add_argument("--offline-key-sample", type=int, default=2048)
    p.add_argument("--offline-head-batch", type=int, default=24)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-offline-eval", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--overwrite", action="store_true")

    # Worker-only args.
    p.add_argument("--worker", choices=["train", "offline-eval"], default="")
    p.add_argument("--tensor", choices=["K", "V"], default="K")
    p.add_argument("--task-values", default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--metrics-out", type=Path, default=None)
    return p.parse_args()


def default_init_k_dir(root: Path) -> Path:
    return root / "sweep_k_l8c64_codebooks"


def default_init_v_dir(root: Path) -> Path:
    return root / "sweep_v_l8c64_codebooks"


def init_dir(args: argparse.Namespace, tensor: str) -> Path:
    if tensor == "K":
        return args.init_k_dir or default_init_k_dir(args.output_dir)
    return args.init_v_dir or default_init_v_dir(args.output_dir)


def codebook_path(cb_dir: Path, layer: int, head: int) -> Path:
    return cb_dir / f"layer_{layer}_head_{head}.pt"


def stage_dir(root: Path, stage: int, tensor: str, kind: str) -> Path:
    return root / f"gradref_stage{stage}_{kind}_{tensor.lower()}_l8c64_codebooks"


def latest_dir(root: Path, stage: int, tensor: str) -> Path:
    return stage_dir(root, stage, tensor, "latest")


def best_dir(root: Path, stage: int, tensor: str) -> Path:
    return stage_dir(root, stage, tensor, "best")


def layer_file(root: Path, split: str, prompt_id: int, layer: int) -> Path:
    return root / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def split_items(root: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads((root / "prompt_split.json").read_text(encoding="utf-8"))
    return payload[split]


def mkdirs(root: Path) -> None:
    for rel in [
        "metrics",
        "plots",
        "reports",
        "logs/gradref",
        "tmp/gradref/train",
        "tmp/gradref/eval",
        "status/gradref",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)
    for stage in (1, 2):
        for tensor in ("K", "V"):
            latest_dir(root, stage, tensor).mkdir(parents=True, exist_ok=True)
            best_dir(root, stage, tensor).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def gpu_free_mib() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    free: dict[int, int] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            try:
                free[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return free


def parse_gpus(raw: str, min_free_mb: int) -> list[int]:
    raw = raw.strip().lower()
    if raw == "auto":
        free = gpu_free_mib()
        gpus = [idx for idx, mib in sorted(free.items()) if mib >= min_free_mb]
        if not gpus:
            raise RuntimeError(f"No GPU has >= {min_free_mb} MiB free; free={free}")
        return gpus
    gpus = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not gpus:
        raise ValueError("--gpus cannot be empty")
    return gpus


def all_tasks(args: argparse.Namespace) -> list[int]:
    layers = parse_int_list(args.layers, LAYERS_DEFAULT)
    heads = parse_int_list(args.heads, HEADS_DEFAULT)
    return [layer * 100 + head for layer in layers for head in heads]


def marker_path(root: Path, tensor: str, task: int) -> Path:
    return root / "status" / "gradref" / f"train_{tensor.lower()}_{task}.done"


def load_codebook_payload(cb_dir: Path, layer: int, head: int) -> dict[str, Any]:
    path = codebook_path(cb_dir, layer, head)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "codebooks" in payload:
        return payload
    return {"codebooks": payload, "config": {}}


def save_codebook_payload(
    out_dir: Path,
    layer: int,
    head: int,
    codebooks: torch.Tensor,
    config: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "codebooks": codebooks.detach().cpu().float().contiguous(),
            "config": config,
        },
        codebook_path(out_dir, layer, head),
    )


def sample_arrays(
    root: Path,
    split: str,
    layer: int,
    head: int,
    sample_size: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    items = split_items(root, split)
    per_prompt = max(1, math.ceil(sample_size / max(1, len(items))))
    q_chunks: list[torch.Tensor] = []
    k_chunks: list[torch.Tensor] = []
    v_chunks: list[torch.Tensor] = []
    remaining = sample_size
    for item in items:
        if remaining <= 0:
            break
        prompt_id = int(item["prompt_id"])
        payload = torch.load(layer_file(root, split, prompt_id, layer), map_location="cpu", weights_only=False)
        q = payload["Q"][:, head].reshape(-1, payload["Q"].shape[-1]).float()
        k = payload["K"][:, head].reshape(-1, payload["K"].shape[-1]).float()
        v = payload["V"][:, head].reshape(-1, payload["V"].shape[-1]).float()
        take = min(per_prompt, remaining, k.shape[0])
        gen_kv = torch.Generator(device="cpu").manual_seed(seed + prompt_id * 1000000 + layer * 10000 + head * 100 + 11)
        gen_q = torch.Generator(device="cpu").manual_seed(seed + prompt_id * 1000000 + layer * 10000 + head * 100 + 37)
        kv_idx = torch.randperm(k.shape[0], generator=gen_kv)[:take]
        q_idx = torch.randperm(q.shape[0], generator=gen_q)[:take]
        q_chunks.append(q[q_idx])
        k_chunks.append(k[kv_idx])
        v_chunks.append(v[kv_idx])
        remaining -= take
    if not k_chunks:
        raise RuntimeError(f"No samples for split={split} layer={layer} head={head}")
    return {
        "Q": torch.cat(q_chunks, dim=0).contiguous(),
        "K": torch.cat(k_chunks, dim=0).contiguous(),
        "V": torch.cat(v_chunks, dim=0).contiguous(),
    }


def nearest_indices(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    c_f = centroids.float()
    dist = (
        (x_f * x_f).sum(dim=-1, keepdim=True)
        + (c_f * c_f).sum(dim=-1)[None, :]
        - 2.0 * x_f @ c_f.T
    )
    return torch.argmin(dist, dim=-1)


@torch.no_grad()
def rvq_encode(x_norm: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    residual = x_norm.float()
    indices: list[torch.Tensor] = []
    for level in range(codebooks.shape[0]):
        idx = nearest_indices(residual, codebooks[level])
        residual = residual - codebooks[level][idx].float()
        indices.append(idx)
    return torch.stack(indices, dim=-1).long()


def rvq_decode(indices: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        (indices.shape[0], codebooks.shape[-1]),
        device=codebooks.device,
        dtype=codebooks.dtype,
    )
    for level in range(codebooks.shape[0]):
        out = out + F.embedding(indices[:, level], codebooks[level])
    return out


def rvq_reconstruct(x: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = torch.sqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True)).clamp_min(1e-6).to(x.device)
    x_norm = x.float() / scale.float()
    with torch.no_grad():
        indices = rvq_encode(x_norm, codebooks)
    recon_norm = rvq_decode(indices, codebooks)
    recon = recon_norm.float() * scale.float()
    return recon, indices, x_norm


def normalized_mse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.mse_loss(pred.float(), target.float()) / (target.float().pow(2).mean().detach() + eps)


def tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean()


def int4_dequant_codebooks(codebooks: torch.Tensor) -> torch.Tensor:
    max_abs = codebooks.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / 7.0
    q = torch.round(codebooks.float() / scale).clamp(-7, 7)
    return (q * scale).float().contiguous()


def sample_rows(tensor: torch.Tensor, take: int, gen: torch.Generator) -> torch.Tensor:
    if take <= 0 or take >= tensor.shape[0]:
        return tensor
    idx = torch.randperm(tensor.shape[0], generator=gen, device=tensor.device)[:take]
    return tensor.index_select(0, idx)


def batch_loss(
    tensor_name: str,
    stage: int,
    codebooks: torch.Tensor,
    init_codebooks: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    gen: torch.Generator,
) -> dict[str, torch.Tensor]:
    x = batch[tensor_name]
    x_hat, _indices, _x_norm = rvq_reconstruct(x, codebooks)
    loss_rec = F.mse_loss(x_hat.float(), x.float())
    out: dict[str, torch.Tensor] = {
        "loss_rec": loss_rec,
        "loss_qk": torch.zeros((), device=x.device),
        "loss_av": torch.zeros((), device=x.device),
        "loss_drift": torch.zeros((), device=x.device),
    }
    total = loss_rec
    if stage == 2 and tensor_name == "K":
        kv_take = min(args.num_kv, x.shape[0])
        q_take = min(args.num_q, batch["Q"].shape[0])
        kv_idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:kv_take]
        q_idx = torch.randperm(batch["Q"].shape[0], generator=gen, device=x.device)[:q_take]
        q_sub = batch["Q"].index_select(0, q_idx)
        k_sub = x.index_select(0, kv_idx)
        kh_sub = x_hat.index_select(0, kv_idx)
        logits = q_sub.float() @ k_sub.float().T
        logits_hat = q_sub.float() @ kh_sub.float().T
        out["loss_qk"] = normalized_mse(logits_hat, logits)
        total = total + float(args.beta_k) * out["loss_qk"]
    elif stage == 2 and tensor_name == "V":
        kv_take = min(args.num_kv, x.shape[0])
        q_take = min(args.num_q, batch["Q"].shape[0])
        kv_idx = torch.randperm(x.shape[0], generator=gen, device=x.device)[:kv_take]
        q_idx = torch.randperm(batch["Q"].shape[0], generator=gen, device=x.device)[:q_take]
        q_sub = batch["Q"].index_select(0, q_idx)
        k_sub = batch["K"].index_select(0, kv_idx)
        v_sub = x.index_select(0, kv_idx)
        vh_sub = x_hat.index_select(0, kv_idx)
        with torch.no_grad():
            logits = q_sub.float() @ k_sub.float().T / math.sqrt(k_sub.shape[-1])
            attn = F.softmax(logits, dim=-1)
            out_ref = attn @ v_sub.float()
        out_hat = attn @ vh_sub.float()
        out["loss_av"] = normalized_mse(out_hat, out_ref)
        total = total + float(args.beta_v) * out["loss_av"]
    if args.lambda_drift > 0:
        out["loss_drift"] = F.mse_loss(codebooks.float(), init_codebooks.float())
        total = total + float(args.lambda_drift) * out["loss_drift"]
    out["loss_total"] = total
    return out


@torch.no_grad()
def eval_task_loss(
    tensor_name: str,
    stage: int,
    codebooks: torch.Tensor,
    init_codebooks: torch.Tensor,
    data: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    n = data[tensor_name].shape[0]
    gen = torch.Generator(device=device).manual_seed(seed)
    for start in range(0, n, args.eval_batch_size):
        stop = min(n, start + args.eval_batch_size)
        batch = {k: v[start:stop].to(device=device, dtype=torch.float32) for k, v in data.items()}
        # batch_loss is gradient-capable, but this call is under no_grad.
        losses = batch_loss(tensor_name, stage, codebooks, init_codebooks, batch, args, gen)
        x = batch[tensor_name]
        x_hat, _indices, _x_norm = rvq_reconstruct(x, codebooks)
        rows.append(
            {
                "loss_total": float(losses["loss_total"].item()),
                "loss_rec": float(losses["loss_rec"].item()),
                "loss_qk": float(losses["loss_qk"].item()),
                "loss_av": float(losses["loss_av"].item()),
                "loss_drift": float(losses["loss_drift"].item()),
                "mse": float(F.mse_loss(x_hat.float(), x.float()).item()),
                "cosine": float(tensor_cosine(x_hat, x).item()),
                "n": float(stop - start),
            }
        )
    denom = max(1.0, sum(r["n"] for r in rows))
    return {
        key: sum(r[key] * r["n"] for r in rows) / denom
        for key in ["loss_total", "loss_rec", "loss_qk", "loss_av", "loss_drift", "mse", "cosine"]
    }


def make_batches(data: dict[str, torch.Tensor], batch_size: int, epoch: int, seed: int) -> list[torch.Tensor]:
    n = data["K"].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed + epoch * 100003)
    perm = torch.randperm(n, generator=gen)
    return [perm[start : start + batch_size] for start in range(0, n, batch_size)]


def metric_row(
    tensor: str,
    stage: int,
    split: str,
    layer: int,
    head: int,
    epoch: int,
    step: int,
    values: dict[str, float],
    lr: float,
) -> dict[str, Any]:
    return {
        "tensor": tensor,
        "stage": stage,
        "split": split,
        "layer": layer,
        "head": head,
        "epoch": epoch,
        "step": step,
        "loss_total": values.get("loss_total"),
        "loss_rec": values.get("loss_rec"),
        "loss_qk": values.get("loss_qk"),
        "loss_av": values.get("loss_av"),
        "loss_drift": values.get("loss_drift"),
        "mse": values.get("mse"),
        "cosine": values.get("cosine"),
        "lr": lr,
    }


def save_best_latest(
    root: Path,
    stage: int,
    tensor: str,
    layer: int,
    head: int,
    latest_codebooks: torch.Tensor,
    best_codebooks: torch.Tensor,
    base_config: dict[str, Any],
    train_config: dict[str, Any],
    best_metrics: dict[str, Any],
) -> None:
    common = dict(base_config)
    common.update(train_config)
    common.update(
        {
            "layer": layer,
            "head": head,
            "stage": stage,
            "tensor": tensor,
            "normalization": "per-vector RMS",
            "runtime_format": "standard_rvq_indices_plus_codebooks",
            "logical_kv_dim_for_memory": LOGICAL_KV_DIM,
        }
    )
    latest_cfg = dict(common)
    latest_cfg["checkpoint_kind"] = "latest"
    best_cfg = dict(common)
    best_cfg["checkpoint_kind"] = "best"
    best_cfg["best_metrics"] = best_metrics
    save_codebook_payload(latest_dir(root, stage, tensor), layer, head, latest_codebooks, latest_cfg)
    save_codebook_payload(best_dir(root, stage, tensor), layer, head, best_codebooks, best_cfg)


def train_one_task(args: argparse.Namespace, tensor: str, task: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    layer, head = task // 100, task % 100
    done = marker_path(args.output_dir, tensor, task)
    if done.exists() and not args.overwrite:
        print(f"[train-skip] tensor={tensor} layer={layer} head={head}", flush=True)
        return [], []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_payload = load_codebook_payload(init_dir(args, tensor), layer, head)
    init_cb_cpu = init_payload["codebooks"].float().contiguous()
    if init_cb_cpu.shape[:2] != (args.levels, args.codebook_size):
        raise ValueError(f"Expected L{args.levels}C{args.codebook_size}, got {tuple(init_cb_cpu.shape)}")

    train_data_cpu = sample_arrays(args.output_dir, "train", layer, head, args.sample_size, args.seed)
    test_data_cpu = sample_arrays(args.output_dir, "test", layer, head, args.eval_sample_size, args.seed + 911)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    base_config = init_payload.get("config", {}) if isinstance(init_payload, dict) else {}
    train_config = {
        "train_method": "kmeans_initialized_gradient_refined_rvq",
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "sample_size": args.sample_size,
        "eval_sample_size": args.eval_sample_size,
        "batch_size": args.batch_size,
        "beta_k": args.beta_k,
        "beta_v": args.beta_v,
        "lambda_drift": args.lambda_drift,
        "num_q": args.num_q,
        "num_kv": args.num_kv,
    }

    stage_start_codebooks = init_cb_cpu
    for stage in (1, 2):
        epochs = args.stage1_epochs if stage == 1 else args.stage2_epochs
        if epochs <= 0:
            continue
        lr = (
            args.lr_k_stage1
            if stage == 1 and tensor == "K"
            else args.lr_v_stage1
            if stage == 1
            else args.lr_k_stage2
            if tensor == "K"
            else args.lr_v_stage2
        )
        codebooks = stage_start_codebooks.clone().to(device=device, dtype=torch.float32)
        codebooks.requires_grad_(True)
        init_fixed = init_cb_cpu.to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW([codebooks], lr=lr, weight_decay=0.0)
        best_loss = float("inf")
        best_step = 0
        best_codebooks = codebooks.detach().cpu().clone()
        patience_counter = 0
        global_step = 0

        initial_train = eval_task_loss(tensor, stage, codebooks, init_fixed, train_data_cpu, args, device, args.seed + stage * 1000)
        initial_test = eval_task_loss(tensor, stage, codebooks, init_fixed, test_data_cpu, args, device, args.seed + stage * 1000 + 1)
        for split, values in (("train", initial_train), ("test", initial_test)):
            eval_rows.append(metric_row(tensor, stage, split, layer, head, 0, 0, values, lr))
        best_loss = initial_test["loss_total"]
        best_codebooks = codebooks.detach().cpu().clone()

        for epoch in range(1, epochs + 1):
            batches = make_batches(train_data_cpu, args.batch_size, epoch, args.seed + layer * 10000 + head * 100 + stage)
            for indices_cpu in batches:
                global_step += 1
                batch = {k: v.index_select(0, indices_cpu).to(device=device, dtype=torch.float32) for k, v in train_data_cpu.items()}
                gen = torch.Generator(device=device).manual_seed(
                    args.seed + stage * 10000000 + epoch * 100000 + global_step * 13 + layer * 100 + head
                )
                optimizer.zero_grad(set_to_none=True)
                losses = batch_loss(tensor, stage, codebooks, init_fixed, batch, args, gen)
                losses["loss_total"].backward()
                grad_norm = float(codebooks.grad.detach().float().norm().item()) if codebooks.grad is not None else 0.0
                optimizer.step()

                if global_step % max(1, args.log_interval_steps) == 0:
                    train_values = {
                        "loss_total": float(losses["loss_total"].detach().item()),
                        "loss_rec": float(losses["loss_rec"].detach().item()),
                        "loss_qk": float(losses["loss_qk"].detach().item()),
                        "loss_av": float(losses["loss_av"].detach().item()),
                        "loss_drift": float(losses["loss_drift"].detach().item()),
                        "mse": float(losses["loss_rec"].detach().item()),
                        "cosine": float("nan"),
                    }
                    row = metric_row(tensor, stage, "train_step", layer, head, epoch, global_step, train_values, lr)
                    row["grad_norm"] = grad_norm
                    train_rows.append(row)

                if global_step % max(1, args.eval_interval_steps) == 0:
                    train_eval = eval_task_loss(
                        tensor, stage, codebooks, init_fixed, train_data_cpu, args, device, args.seed + global_step
                    )
                    test_eval = eval_task_loss(
                        tensor, stage, codebooks, init_fixed, test_data_cpu, args, device, args.seed + global_step + 1
                    )
                    gap = test_eval["loss_total"] - train_eval["loss_total"]
                    rel_gap = gap / (abs(train_eval["loss_total"]) + 1e-8)
                    for split, values in (("train", train_eval), ("test", test_eval)):
                        row = metric_row(tensor, stage, split, layer, head, epoch, global_step, values, lr)
                        row["generalization_gap"] = gap
                        row["relative_gap"] = rel_gap
                        eval_rows.append(row)
                    if test_eval["loss_total"] < best_loss:
                        best_loss = test_eval["loss_total"]
                        best_step = global_step
                        best_codebooks = codebooks.detach().cpu().clone()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    if args.early_stop and patience_counter >= args.patience:
                        print(
                            f"[early-stop] tensor={tensor} stage={stage} layer={layer} head={head} "
                            f"step={global_step} best={best_loss:.8g}",
                            flush=True,
                        )
                        break
            if args.early_stop and patience_counter >= args.patience:
                break

        final_train = eval_task_loss(tensor, stage, codebooks, init_fixed, train_data_cpu, args, device, args.seed + 777 + stage)
        final_test = eval_task_loss(tensor, stage, codebooks, init_fixed, test_data_cpu, args, device, args.seed + 778 + stage)
        if final_test["loss_total"] < best_loss:
            best_loss = final_test["loss_total"]
            best_step = global_step
            best_codebooks = codebooks.detach().cpu().clone()
            patience_counter = 0
        gap = final_test["loss_total"] - final_train["loss_total"]
        rel_gap = gap / (abs(final_train["loss_total"]) + 1e-8)
        overfit = final_test["loss_total"] > best_loss * 1.02 and rel_gap > 0.05
        for split, values in (("train_final", final_train), ("test_final", final_test)):
            row = metric_row(tensor, stage, split, layer, head, epochs, global_step, values, lr)
            row["best_test_loss"] = best_loss
            row["best_step"] = best_step
            row["generalization_gap"] = gap
            row["relative_gap"] = rel_gap
            row["overfitting_flag"] = overfit
            eval_rows.append(row)

        best_metrics = {
            "best_test_loss": best_loss,
            "best_step": best_step,
            "final_train_loss": final_train["loss_total"],
            "final_test_loss": final_test["loss_total"],
            "generalization_gap": gap,
            "relative_gap": rel_gap,
            "overfitting_flag": overfit,
        }
        save_best_latest(
            args.output_dir,
            stage,
            tensor,
            layer,
            head,
            codebooks.detach().cpu(),
            best_codebooks,
            base_config,
            train_config,
            best_metrics,
        )
        stage_start_codebooks = best_codebooks.detach().cpu().clone()
        print(
            f"[train-done] tensor={tensor} stage={stage} layer={layer} head={head} "
            f"best_test={best_loss:.8g} final_test={final_test['loss_total']:.8g}",
            flush=True,
        )
    done.write_text("done\n", encoding="utf-8")
    return train_rows, eval_rows


def train_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    tasks = [int(x) for x in args.task_values.split(",") if x.strip()]
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for task in tasks:
        tr, ev = train_one_task(args, args.tensor, task)
        train_rows.extend(tr)
        eval_rows.extend(ev)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.metrics_out is not None:
        write_csv(args.metrics_out.with_suffix(".train.csv"), train_rows)
        write_csv(args.metrics_out.with_suffix(".eval.csv"), eval_rows)


def run_processes(procs: list[tuple[subprocess.Popen, Any, Path]]) -> None:
    failures = []
    for proc, log, log_path in procs:
        rc = proc.wait()
        log.close()
        if rc != 0:
            failures.append((rc, log_path))
    if failures:
        raise RuntimeError("Subprocess failures:\n" + "\n".join(f"rc={rc} log={path}" for rc, path in failures))


def launch_train(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_train:
        return
    tmp = args.output_dir / "tmp" / "gradref" / "train"
    if tmp.exists() and args.overwrite:
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for tensor in ("K", "V"):
        tasks = [task for task in all_tasks(args) if args.overwrite or not marker_path(args.output_dir, tensor, task).exists()]
        if not tasks:
            print(f"[train-skip-all] tensor={tensor}", flush=True)
            continue
        procs = []
        for i, gpu in enumerate(gpu_ids):
            assigned = tasks[i::len(gpu_ids)]
            if not assigned:
                continue
            metrics_out = tmp / f"{tensor.lower()}_gpu{gpu}.csv"
            log_path = args.output_dir / "logs" / "gradref" / f"train_{tensor.lower()}_gpu{gpu}.log"
            cmd = [
                str(args.worker_python),
                str(Path(__file__).resolve()),
                "--worker",
                "train",
                "--output-dir",
                str(args.output_dir),
                "--gpus",
                str(gpu),
                "--gpu",
                str(gpu),
                "--tensor",
                tensor,
                "--task-values",
                ",".join(str(x) for x in assigned),
                "--layers",
                args.layers,
                "--heads",
                args.heads,
                "--levels",
                str(args.levels),
                "--codebook-size",
                str(args.codebook_size),
                "--sample-size",
                str(args.sample_size),
                "--eval-sample-size",
                str(args.eval_sample_size),
                "--batch-size",
                str(args.batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--stage1-epochs",
                str(args.stage1_epochs),
                "--stage2-epochs",
                str(args.stage2_epochs),
                "--lr-k-stage1",
                str(args.lr_k_stage1),
                "--lr-v-stage1",
                str(args.lr_v_stage1),
                "--lr-k-stage2",
                str(args.lr_k_stage2),
                "--lr-v-stage2",
                str(args.lr_v_stage2),
                "--beta-k",
                str(args.beta_k),
                "--beta-v",
                str(args.beta_v),
                "--lambda-drift",
                str(args.lambda_drift),
                "--num-q",
                str(args.num_q),
                "--num-kv",
                str(args.num_kv),
                "--eval-interval-steps",
                str(args.eval_interval_steps),
                "--log-interval-steps",
                str(args.log_interval_steps),
                "--seed",
                str(args.seed),
                "--metrics-out",
                str(metrics_out),
            ]
            if args.init_k_dir is not None:
                cmd.extend(["--init-k-dir", str(args.init_k_dir)])
            if args.init_v_dir is not None:
                cmd.extend(["--init-v-dir", str(args.init_v_dir)])
            if args.early_stop:
                cmd.append("--early-stop")
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("w", encoding="utf-8")
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            print(f"[train-launch] tensor={tensor} gpu={gpu} tasks={len(assigned)} log={log_path}", flush=True)
            procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
        run_processes(procs)
    train_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for path in sorted(tmp.glob("*.train.csv")):
        train_rows.extend(read_csv_rows(path))
    for path in sorted(tmp.glob("*.eval.csv")):
        eval_rows.extend(read_csv_rows(path))
    write_csv(args.output_dir / "metrics" / "gradref_train_metrics.csv", train_rows)
    write_csv(args.output_dir / "metrics" / "gradref_eval_metrics.csv", eval_rows)
    write_overfitting_report(args.output_dir, eval_rows)
    if not args.skip_plots:
        plot_training_curves(args.output_dir, eval_rows)


def write_overfitting_report(root: Path, rows: list[dict[str, Any]]) -> None:
    report: dict[str, Any] = {}
    for stage in ("1", "2"):
        stage_rows = [r for r in rows if str(r.get("stage")) == stage]
        final_train = [float(r["loss_total"]) for r in stage_rows if r.get("split") == "train_final" and r.get("loss_total")]
        final_test = [float(r["loss_total"]) for r in stage_rows if r.get("split") == "test_final" and r.get("loss_total")]
        best = [float(r["best_test_loss"]) for r in stage_rows if r.get("best_test_loss")]
        gaps = [float(r["generalization_gap"]) for r in stage_rows if r.get("generalization_gap")]
        flags = [str(r.get("overfitting_flag")).lower() == "true" for r in stage_rows if r.get("overfitting_flag") != ""]
        report[f"stage{stage}"] = {
            "best_test_loss_mean": sum(best) / len(best) if best else None,
            "final_train_loss_mean": sum(final_train) / len(final_train) if final_train else None,
            "final_test_loss_mean": sum(final_test) / len(final_test) if final_test else None,
            "final_generalization_gap_mean": sum(gaps) / len(gaps) if gaps else None,
            "overfitting_task_fraction": sum(flags) / len(flags) if flags else None,
            "num_rows": len(stage_rows),
        }
    (root / "metrics" / "gradref_overfitting_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def plot_training_curves(root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (root / "plots" / "gradref_plot_error.txt").write_text(str(exc), encoding="utf-8")
        plot_training_curves_svg(root, rows)
        return
    for stage in (1, 2):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        stage_rows = [r for r in rows if str(r.get("stage")) == str(stage) and r.get("split") in {"train", "test"}]
        for ax, key, title in [
            (axes[0, 0], "loss_rec", "Reconstruction"),
            (axes[0, 1], "loss_qk", "QK"),
            (axes[1, 0], "loss_av", "AV"),
            (axes[1, 1], "loss_total", "Total"),
        ]:
            for split in ("train", "test"):
                vals_by_step: dict[int, list[float]] = {}
                for row in stage_rows:
                    if row.get("split") != split or not row.get(key):
                        continue
                    try:
                        step = int(float(row["step"]))
                        vals_by_step.setdefault(step, []).append(float(row[key]))
                    except Exception:
                        continue
                if vals_by_step:
                    xs = sorted(vals_by_step)
                    ys = [sum(vals_by_step[x]) / len(vals_by_step[x]) for x in xs]
                    ax.plot(xs, ys, label=split)
            ax.set_title(title)
            ax.set_xlabel("step")
            ax.legend()
        fig.tight_layout()
        fig.savefig(root / "plots" / f"gradref_stage{stage}_loss_curves.png", dpi=160)
        plt.close(fig)


def _svg_polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        + '" />'
    )


def plot_training_curves_svg(root: Path, rows: list[dict[str, Any]]) -> None:
    """Dependency-free fallback plots for environments without matplotlib."""
    width, height = 900, 560
    margin_l, margin_r, margin_t, margin_b = 70, 25, 45, 55
    colors = {"train": "#1f77b4", "test": "#d62728"}
    for stage in (1, 2):
        stage_rows = [r for r in rows if str(r.get("stage")) == str(stage) and r.get("split") in {"train", "test"}]
        panels = [
            ("loss_rec", "Reconstruction"),
            ("loss_qk", "QK"),
            ("loss_av", "AV"),
            ("loss_total", "Total"),
        ]
        panel_w = (width - margin_l - margin_r) / 2
        panel_h = (height - margin_t - margin_b) / 2
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">GradRef Stage {stage} Loss Curves</text>',
        ]
        for idx, (key, title) in enumerate(panels):
            col, row = idx % 2, idx // 2
            x0 = margin_l + col * panel_w
            y0 = margin_t + row * panel_h
            x1 = x0 + panel_w - 22
            y1 = y0 + panel_h - 42
            values_by_split: dict[str, dict[int, list[float]]] = {"train": {}, "test": {}}
            for item in stage_rows:
                split = str(item.get("split"))
                raw = item.get(key)
                if split not in values_by_split or raw in (None, ""):
                    continue
                try:
                    step = int(float(item["step"]))
                    val = float(raw)
                    if math.isfinite(val):
                        values_by_split[split].setdefault(step, []).append(val)
                except Exception:
                    continue
            series: dict[str, list[tuple[int, float]]] = {}
            all_x: list[int] = []
            all_y: list[float] = []
            for split, vals in values_by_split.items():
                series[split] = [(s, sum(v) / len(v)) for s, v in sorted(vals.items()) if v]
                all_x.extend(s for s, _ in series[split])
                all_y.extend(v for _, v in series[split])
            if not all_x or not all_y:
                continue
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            if max_x == min_x:
                max_x += 1
            if max_y == min_y:
                max_y += 1e-6
            svg.extend(
                [
                    f'<text x="{(x0+x1)/2}" y="{y0+8}" text-anchor="middle" font-family="sans-serif" font-size="13">{title}</text>',
                    f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#222"/>',
                    f'<line x1="{x0}" y1="{y0+20}" x2="{x0}" y2="{y1}" stroke="#222"/>',
                    f'<text x="{x0}" y="{y1+18}" font-family="sans-serif" font-size="10">{min_x}</text>',
                    f'<text x="{x1}" y="{y1+18}" text-anchor="end" font-family="sans-serif" font-size="10">{max_x}</text>',
                    f'<text x="{x0-5}" y="{y1}" text-anchor="end" font-family="sans-serif" font-size="10">{min_y:.3g}</text>',
                    f'<text x="{x0-5}" y="{y0+25}" text-anchor="end" font-family="sans-serif" font-size="10">{max_y:.3g}</text>',
                ]
            )
            for split, points in series.items():
                mapped = [
                    (
                        x0 + (s - min_x) / (max_x - min_x) * (x1 - x0),
                        y1 - (v - min_y) / (max_y - min_y) * (y1 - (y0 + 20)),
                    )
                    for s, v in points
                ]
                svg.append(_svg_polyline(mapped, colors[split]))
            svg.append(f'<text x="{x1-90}" y="{y0+24}" font-family="sans-serif" font-size="11" fill="{colors["train"]}">train</text>')
            svg.append(f'<text x="{x1-45}" y="{y0+24}" font-family="sans-serif" font-size="11" fill="{colors["test"]}">test</text>')
        svg.append("</svg>")
        (root / "plots" / f"gradref_stage{stage}_loss_curves.svg").write_text("\n".join(svg), encoding="utf-8")


def load_layer_codebooks(cb_dir: Path, layer: int, device: torch.device, heads: list[int]) -> torch.Tensor:
    values = []
    for head in heads:
        payload = load_codebook_payload(cb_dir, layer, head)
        values.append(payload["codebooks"].float())
    stacked = torch.stack(values, dim=1).contiguous()  # [L,H,C,D]
    return int4_dequant_codebooks(stacked).to(device=device, dtype=torch.float32)


@torch.no_grad()
def encode_reconstruct_heads(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    x_f = x.float()
    scale = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True)).clamp_min(1e-6)
    residual = x_f / scale
    recon_norm = torch.zeros_like(residual)
    for level in range(codebooks.shape[0]):
        centroids = codebooks[level].float()  # [H,C,D]
        dist = (
            (residual * residual).sum(dim=-1, keepdim=True)
            + (centroids * centroids).sum(dim=-1)[:, None, :]
            - 2.0 * torch.bmm(residual, centroids.transpose(1, 2))
        )
        ids = torch.argmin(dist, dim=-1)
        assigned = torch.gather(centroids, 1, ids[..., None].expand(-1, -1, centroids.shape[-1]))
        residual -= assigned
        recon_norm += assigned
    return recon_norm * scale


@torch.no_grad()
def eval_layer_prompt(
    args: argparse.Namespace,
    split: str,
    prompt_id: int,
    layer: int,
    heads: list[int],
    setting: str,
    k_dir: Path,
    v_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = torch.load(layer_file(args.output_dir, split, prompt_id, layer), map_location="cpu", weights_only=False)
    q_all = payload["Q"].permute(1, 0, 2, 3).reshape(HEADS_DEFAULT, -1, payload["Q"].shape[-1]).float()
    k_all = payload["K"].permute(1, 0, 2, 3).reshape(HEADS_DEFAULT, -1, payload["K"].shape[-1]).float()
    v_all = payload["V"].permute(1, 0, 2, 3).reshape(HEADS_DEFAULT, -1, payload["V"].shape[-1]).float()
    n_tokens = k_all.shape[1]
    gen = torch.Generator(device="cpu").manual_seed(args.seed + prompt_id * 1000000 + layer * 10000 + 97)
    q_take = min(args.offline_query_sample, n_tokens)
    k_take = min(args.offline_key_sample, n_tokens)
    q_idx = torch.randperm(n_tokens, generator=gen)[:q_take].to(device)
    kv_idx = torch.randperm(n_tokens, generator=gen)[:k_take].to(device)
    head_tensor = torch.tensor(heads, dtype=torch.long)
    q_s = q_all.index_select(0, head_tensor).to(device).index_select(1, q_idx)
    k_s = k_all.index_select(0, head_tensor).to(device).index_select(1, kv_idx)
    v_s = v_all.index_select(0, head_tensor).to(device).index_select(1, kv_idx)
    k_cb = load_layer_codebooks(k_dir, layer, device, heads)
    v_cb = load_layer_codebooks(v_dir, layer, device, heads)
    kh = encode_reconstruct_heads(k_s, k_cb)
    vh = encode_reconstruct_heads(v_s, v_cb)

    logits = torch.bmm(q_s.float(), k_s.float().transpose(1, 2)) / math.sqrt(k_s.shape[-1])
    logits_hat = torch.bmm(q_s.float(), kh.float().transpose(1, 2)) / math.sqrt(k_s.shape[-1])
    log_p = F.log_softmax(logits.float(), dim=-1)
    log_q = F.log_softmax(logits_hat.float(), dim=-1)
    p = log_p.exp()
    p_hat = log_q.exp()
    av = torch.bmm(p, v_s.float())
    av_vhat = torch.bmm(p, vh.float())
    qkv = torch.bmm(p_hat, vh.float())
    topk = min(args.topk, logits.shape[-1])
    top_a = torch.topk(logits, k=topk, dim=-1).indices
    top_b = torch.topk(logits_hat, k=topk, dim=-1).indices
    overlap = (top_a[..., :, None] == top_b[..., None, :]).any(dim=-1).float().mean(dim=(1, 2))

    rows = []
    for i, head in enumerate(heads):
        rows.append(
            {
                "setting": setting,
                "split": split,
                "prompt_id": prompt_id,
                "layer": layer,
                "head": head,
                "k_reconstruction_mse": float(((k_s[i] - kh[i]) ** 2).mean().item()),
                "k_reconstruction_cosine": float(F.cosine_similarity(k_s[i], kh[i], dim=-1).mean().item()),
                "v_reconstruction_mse": float(((v_s[i] - vh[i]) ** 2).mean().item()),
                "v_reconstruction_cosine": float(F.cosine_similarity(v_s[i], vh[i], dim=-1).mean().item()),
                "qk_logits_mse": float(((logits[i] - logits_hat[i]) ** 2).mean().item()),
                "qk_normalized_mse": float(normalized_mse(logits_hat[i], logits[i]).item()),
                "attention_kl": float((p[i] * (log_p[i] - log_q[i])).sum(dim=-1).mean().item()),
                "topk_overlap": float(overlap[i].item()),
                "attn_output_mse": float(((av[i] - av_vhat[i]) ** 2).mean().item()),
                "av_normalized_mse": float(normalized_mse(av_vhat[i], av[i]).item()),
                "qkv_output_mse": float(((av[i] - qkv[i]) ** 2).mean().item()),
                "qkv_output_cosine": float(F.cosine_similarity(av[i], qkv[i], dim=-1).mean().item()),
                "query_sample": q_take,
                "key_sample": k_take,
            }
        )
    return rows


def offline_eval_worker(args: argparse.Namespace) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    layers = [int(x) for x in args.task_values.split(",") if x.strip()]
    head_values = parse_int_list(args.heads, HEADS_DEFAULT)
    settings = [
        ("RVQ-L8C64-INT4 baseline", init_dir(args, "K"), init_dir(args, "V")),
        ("GradRef-Stage1-Latest-L8C64-INT4", latest_dir(args.output_dir, 1, "K"), latest_dir(args.output_dir, 1, "V")),
        ("GradRef-Stage2-Latest-L8C64-INT4", latest_dir(args.output_dir, 2, "K"), latest_dir(args.output_dir, 2, "V")),
    ]
    rows: list[dict[str, Any]] = []
    for split in ("train", "test"):
        prompt_ids = [int(item["prompt_id"]) for item in split_items(args.output_dir, split)]
        for layer in layers:
            for prompt_id in prompt_ids:
                for setting, k_dir, v_dir in settings:
                    rows.extend(eval_layer_prompt(args, split, prompt_id, layer, head_values, setting, k_dir, v_dir, device))
                print(f"[offline-eval] split={split} layer={layer} prompt={prompt_id}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    if args.metrics_out is None:
        raise ValueError("--metrics-out is required")
    write_csv(args.metrics_out, rows)


def aggregate_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["setting"], row["split"]), []).append(row)
    out = []
    for (setting, split), vals in sorted(groups.items()):
        summary: dict[str, Any] = {"setting": setting, "split": split, "n": len(vals)}
        for key in METRIC_KEYS:
            xs = []
            for row in vals:
                try:
                    xs.append(float(row[key]))
                except Exception:
                    pass
            if xs:
                mean = sum(xs) / len(xs)
                var = sum((x - mean) ** 2 for x in xs) / len(xs)
                summary[f"{key}_mean"] = mean
                summary[f"{key}_std"] = math.sqrt(var)
        out.append(summary)
    return out


def launch_offline_eval(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    if args.skip_offline_eval:
        return
    rows_path = args.output_dir / "metrics" / "gradref_offline_reconstruction_rows.csv"
    summary_path = args.output_dir / "metrics" / "gradref_offline_reconstruction.csv"
    if rows_path.exists() and summary_path.exists() and not args.overwrite:
        print(f"[offline-eval-skip] {summary_path}", flush=True)
        return
    tmp = args.output_dir / "tmp" / "gradref" / "eval"
    if tmp.exists() and args.overwrite:
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    layers = parse_int_list(args.layers, LAYERS_DEFAULT)
    procs = []
    for i, gpu in enumerate(gpu_ids):
        assigned = layers[i::len(gpu_ids)]
        if not assigned:
            continue
        out = tmp / f"offline_eval_gpu{gpu}.csv"
        log_path = args.output_dir / "logs" / "gradref" / f"offline_eval_gpu{gpu}.log"
        cmd = [
            str(args.worker_python),
            str(Path(__file__).resolve()),
            "--worker",
            "offline-eval",
            "--output-dir",
            str(args.output_dir),
            "--gpu",
            str(gpu),
            "--task-values",
            ",".join(str(x) for x in assigned),
            "--heads",
            args.heads,
            "--topk",
            str(args.topk),
            "--offline-query-sample",
            str(args.offline_query_sample),
            "--offline-key-sample",
            str(args.offline_key_sample),
            "--seed",
            str(args.seed),
            "--metrics-out",
            str(out),
        ]
        if args.init_k_dir is not None:
            cmd.extend(["--init-k-dir", str(args.init_k_dir)])
        if args.init_v_dir is not None:
            cmd.extend(["--init-v-dir", str(args.init_v_dir)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log = log_path.open("w", encoding="utf-8")
        log.write("[cmd] " + " ".join(cmd) + "\n")
        log.flush()
        print(f"[offline-eval-launch] gpu={gpu} layers={assigned}", flush=True)
        procs.append((subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT), log, log_path))
    run_processes(procs)
    rows: list[dict[str, str]] = []
    for path in sorted(tmp.glob("offline_eval_gpu*.csv")):
        rows.extend(read_csv_rows(path))
    write_csv(rows_path, rows)
    write_csv(summary_path, aggregate_summary(rows))
    write_report(args.output_dir)


def write_report(root: Path) -> None:
    summary_path = root / "metrics" / "gradref_offline_reconstruction.csv"
    overfit_path = root / "metrics" / "gradref_overfitting_report.json"
    lines = [
        "# Gradient-Refined RVQ L8C64",
        "",
        "This run refines existing KMeans RVQ codebooks and keeps the runtime RVQ format unchanged.",
        "",
    ]
    if summary_path.exists():
        rows = read_csv_rows(summary_path)
        lines.extend(["## Offline Cache Metrics", ""])
        keep = [
            "setting",
            "split",
            "k_reconstruction_mse_mean",
            "k_reconstruction_cosine_mean",
            "v_reconstruction_mse_mean",
            "v_reconstruction_cosine_mean",
            "qk_logits_mse_mean",
            "attention_kl_mean",
            "qkv_output_mse_mean",
        ]
        lines.append("| " + " | ".join(keep) + " |")
        lines.append("| " + " | ".join(["---"] * len(keep)) + " |")
        for row in rows:
            vals = []
            for key in keep:
                raw = row.get(key, "")
                try:
                    vals.append(f"{float(raw):.6g}")
                except Exception:
                    vals.append(str(raw))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    if overfit_path.exists():
        lines.extend(["## Overfitting", "", "```json", overfit_path.read_text(encoding="utf-8").strip(), "```", ""])
    (root / "reports" / "gradref_l8c64_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_config(args: argparse.Namespace, gpu_ids: list[int]) -> None:
    payload = {
        "output_dir": str(args.output_dir),
        "init_k_dir": str(init_dir(args, "K")),
        "init_v_dir": str(init_dir(args, "V")),
        "gpus": gpu_ids,
        "layers": args.layers,
        "heads": args.heads,
        "levels": args.levels,
        "codebook_size": args.codebook_size,
        "sample_size": args.sample_size,
        "eval_sample_size": args.eval_sample_size,
        "stage1_epochs": args.stage1_epochs,
        "stage2_epochs": args.stage2_epochs,
        "beta_k": args.beta_k,
        "beta_v": args.beta_v,
        "lambda_drift": args.lambda_drift,
        "num_q": args.num_q,
        "num_kv": args.num_kv,
        "logical_kv_dim_for_memory": LOGICAL_KV_DIM,
    }
    (args.output_dir / "gradref_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    mkdirs(args.output_dir)
    if args.worker == "train":
        train_worker(args)
        return
    if args.worker == "offline-eval":
        offline_eval_worker(args)
        return
    gpu_ids = parse_gpus(args.gpus, args.min_free_mb)
    write_config(args, gpu_ids)
    print(f"[gpus] {gpu_ids}", flush=True)
    launch_train(args, gpu_ids)
    launch_offline_eval(args, gpu_ids)
    print(f"[done] report={args.output_dir / 'reports' / 'gradref_l8c64_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
