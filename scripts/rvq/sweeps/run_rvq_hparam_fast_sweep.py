#!/usr/bin/env python3
"""Fast RVQ hyperparameter sweep over existing QuantVideo KV dumps.

This runner keeps the existing RVQ training path in `run_rvq_diag.py`, but
evaluates each variant by writing one aggregate summary instead of thousands of
per-head JSON shards.

Example:
python scripts/rvq/sweeps/run_rvq_hparam_fast_sweep.py \
  --output-dir /data/bowen/rvq_diag \
  --variants l1c64,l1c128,l2c64,l2c128,l2c256,l3c64,l3c128,l3c256,l4c128,l4c256,l5c256,l6c256,l4c512 \
  --tensors K,V \
  --gpu-groups '0,1;2,3;4,5;6,7'
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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RVQ_DIAG_DIR = REPO_ROOT / "scripts" / "rvq" / "diag"
if str(RVQ_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(RVQ_DIAG_DIR))
FULL_LAYERS = ",".join(str(i) for i in range(30))
LAYERS = 30
HEADS = 24
T = 44
SPATIAL = 880
DIM = 128


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int
    sample_size: int = 50000
    kmeans_max_iters: int = 20


DEFAULT_VARIANTS = [
    Variant("l1c64", 1, 64),
    Variant("l1c128", 1, 128),
    Variant("l1c256", 1, 256),
    Variant("l1c512", 1, 512),
    Variant("l1c1024", 1, 1024),
    Variant("l2c64", 2, 64),
    Variant("l2c128", 2, 128),
    Variant("l2c256", 2, 256),
    Variant("l2c512", 2, 512),
    Variant("l2c1024", 2, 1024),
    Variant("l3c64", 3, 64),
    Variant("l3c128", 3, 128),
    Variant("l3c256", 3, 256),
    Variant("l3c512", 3, 512),
    Variant("l3c1024", 3, 1024),
    Variant("l4c64", 4, 64),
    Variant("l4c128", 4, 128),
    Variant("l4c256", 4, 256),
    Variant("l4c512", 4, 512),
    Variant("l4c1024", 4, 1024),
    Variant("l5c128", 5, 128),
    Variant("l5c256", 5, 256),
    Variant("l5c512", 5, 512),
    Variant("l5c1024", 5, 1024),
    Variant("l6c64", 6, 64),
    Variant("l6c128", 6, 128),
    Variant("l6c256", 6, 256),
    Variant("l6c512", 6, 512),
    Variant("l6c1024", 6, 1024),
    Variant("l7c256", 7, 256),
    Variant("l7c512", 7, 512),
    Variant("l8c64", 8, 64),
    Variant("l8c32", 8, 32),
    Variant("l8c256", 8, 256),
    Variant("l8c512", 8, 512),
    Variant("l8c1024", 8, 1024),
    Variant("l10c256", 10, 256),
    Variant("l12c256", 12, 256),
    Variant("l64c64", 64, 64),
]


def variant_map() -> dict[str, Variant]:
    return {v.name: v for v in DEFAULT_VARIANTS}


def tensor_prefix(tensor: str, variant: Variant) -> str:
    return f"sweep_{tensor.lower()}_{variant.name}"


def codebook_subdir(tensor: str, variant: Variant) -> str:
    if variant.name == "l4c256":
        return "codebooks" if tensor == "K" else "v_codebooks"
    return f"{tensor_prefix(tensor, variant)}_codebooks"


def fast_dir(output_dir: Path) -> Path:
    return output_dir / "rvq_hparam_fast"


def summary_path(output_dir: Path, tensor: str, variant: Variant) -> Path:
    return fast_dir(output_dir) / f"{tensor_prefix(tensor, variant)}_summary.json"


def rows_path(output_dir: Path, tensor: str, variant: Variant) -> Path:
    return fast_dir(output_dir) / f"{tensor_prefix(tensor, variant)}_rows.csv"


def timing_path(output_dir: Path, tensor: str, variant: Variant) -> Path:
    return fast_dir(output_dir) / f"{tensor_prefix(tensor, variant)}_timing.json"


def partial_dir(output_dir: Path, tensor: str, variant: Variant) -> Path:
    return fast_dir(output_dir) / "partials" / tensor_prefix(tensor, variant)


def parse_gpu_groups(raw: str) -> list[list[int]]:
    groups = []
    for group in raw.split(";"):
        gpus = [int(x) for x in group.split(",") if x.strip()]
        if gpus:
            groups.append(gpus)
    if not groups:
        raise ValueError("--gpu-groups must contain at least one GPU")
    return groups


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def codebook_file(output_dir: Path, layer: int, head: int, subdir: str) -> Path:
    return output_dir / subdir / f"layer_{layer}_head_{head}.pt"


def all_codebooks_exist(output_dir: Path, tensor: str, variant: Variant) -> bool:
    subdir = codebook_subdir(tensor, variant)
    return all(codebook_file(output_dir, l, h, subdir).exists() for l in range(LAYERS) for h in range(HEADS))


def compression_estimates(levels: int, codebook_size: int) -> dict[str, float]:
    raw_bits = DIM * 16
    packed_index_bits = math.ceil(math.log2(codebook_size)) * levels
    runtime_index_bits = (8 if codebook_size <= 256 else 16) * levels
    scale_bits = 16
    vectors_per_tensor = LAYERS * HEADS * T * SPATIAL
    codebook_bits = LAYERS * HEADS * levels * codebook_size * DIM * 16
    return {
        "packed_bits_per_vector": float(packed_index_bits + scale_bits),
        "runtime_bits_per_vector": float(runtime_index_bits + scale_bits),
        "packed_ratio_no_codebook": raw_bits / (packed_index_bits + scale_bits),
        "runtime_ratio_no_codebook": raw_bits / (runtime_index_bits + scale_bits),
        "packed_ratio_with_codebook_per_video": raw_bits * vectors_per_tensor
        / ((packed_index_bits + scale_bits) * vectors_per_tensor + codebook_bits),
        "runtime_ratio_with_codebook_per_video": raw_bits * vectors_per_tensor
        / ((runtime_index_bits + scale_bits) * vectors_per_tensor + codebook_bits),
    }


def prefix_source_subdir(output_dir: Path, tensor: str, variant: Variant) -> tuple[str, int] | None:
    if variant.codebook_size == 64 and variant.levels <= 2:
        return f"sweep_{tensor.lower()}_l2c64_codebooks", 2
    if variant.codebook_size == 128 and variant.levels <= 2:
        return f"sweep_{tensor.lower()}_l2c128_codebooks", 2
    if variant.codebook_size == 256 and variant.levels <= 4:
        return ("codebooks" if tensor == "K" else "v_codebooks"), 4
    for source_levels in (12, 10, 8, 7, 6, 5, 4, 3, 2, 1):
        if source_levels <= variant.levels:
            continue
        source = output_dir / f"sweep_{tensor.lower()}_l{source_levels}c{variant.codebook_size}_codebooks"
        if source.exists():
            return source.name, source_levels
    source = output_dir / f"sweep_{tensor.lower()}_l{variant.levels}c{variant.codebook_size}_codebooks"
    if source.exists():
        return source.name, variant.levels
    return None


def materialize_prefix_codebooks(output_dir: Path, tensor: str, variant: Variant, overwrite: bool) -> bool:
    if variant.name == "l4c256":
        return all_codebooks_exist(output_dir, tensor, variant)
    if all_codebooks_exist(output_dir, tensor, variant) and not overwrite:
        return True
    source_info = prefix_source_subdir(output_dir, tensor, variant)
    if source_info is None:
        return False
    source_subdir, source_levels = source_info
    if source_levels < variant.levels:
        return False
    if not all(codebook_file(output_dir, l, h, source_subdir).exists() for l in range(LAYERS) for h in range(HEADS)):
        return False

    import torch

    target_subdir = codebook_subdir(tensor, variant)
    target_root = output_dir / target_subdir
    target_root.mkdir(parents=True, exist_ok=True)
    made = 0
    for layer in range(LAYERS):
        for head in range(HEADS):
            dst = codebook_file(output_dir, layer, head, target_subdir)
            if dst.exists() and not overwrite:
                continue
            payload = torch.load(codebook_file(output_dir, layer, head, source_subdir), map_location="cpu", weights_only=False)
            payload["codebooks"] = payload["codebooks"][: variant.levels].contiguous()
            cfg = dict(payload.get("config", {}))
            cfg.update(
                {
                    "levels": variant.levels,
                    "codebook_size": variant.codebook_size,
                    "derived_from_codebook_subdir": source_subdir,
                    "derived_by": "prefix copy",
                }
            )
            payload["config"] = cfg
            torch.save(payload, dst)
            made += 1
    print(f"[prefix] {tensor} {variant.name}: materialized {made} files from {source_subdir}", flush=True)
    return all_codebooks_exist(output_dir, tensor, variant)


def build_train_cmd(args: argparse.Namespace, tensor: str, variant: Variant, gpus: list[int]) -> list[str]:
    prefix = tensor_prefix(tensor, variant)
    return [
        str(args.worker_python),
        str(REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"),
        "--prompt_source",
        str(args.prompt_source),
        "--output_dir",
        str(args.output_dir),
        "--train_num",
        str(args.train_num),
        "--test_num",
        str(args.test_num),
        "--seed",
        str(args.seed),
        "--first_stage_layers",
        FULL_LAYERS,
        "--gpus",
        ",".join(str(g) for g in gpus),
        "--n_spatial",
        str(SPATIAL),
        "--levels",
        str(variant.levels),
        "--codebook_size",
        str(variant.codebook_size),
        "--sample_size",
        str(variant.sample_size),
        "--kmeans_max_iters",
        str(variant.kmeans_max_iters),
        "--codebook_tensor",
        tensor,
        "--codebook_subdir",
        codebook_subdir(tensor, variant),
        "--metrics_subdir",
        f"{prefix}_trainonly_metrics",
        "--indices_subdir",
        f"{prefix}_trainonly_indices",
        "--stage_prefix",
        f"{prefix}_trainonly_",
        "--dump_python",
        str(args.dump_python),
        "--worker_python",
        str(args.worker_python),
        "--skip_dump",
        "--skip_eval",
    ] + (["--overwrite"] if args.overwrite else [])


def train_codebooks(args: argparse.Namespace, tensor: str, variant: Variant, gpus: list[int], log_dir: Path) -> None:
    if materialize_prefix_codebooks(args.output_dir, tensor, variant, args.overwrite):
        print(f"[train-skip] {tensor} {variant.name}: codebooks ready", flush=True)
        return
    log_path = log_dir / f"train_{tensor_prefix(tensor, variant)}_{int(time.time())}.log"
    cmd = build_train_cmd(args, tensor, variant, gpus)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    with log_path.open("w", encoding="utf-8") as log_f:
        log_f.write("[launch] " + " ".join(cmd) + "\n")
        log_f.flush()
        print(f"[train] {tensor} {variant.name} gpus={gpus} log={log_path}", flush=True)
        rc = subprocess.call(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise RuntimeError(f"training failed for {tensor} {variant.name}; log={log_path}")
    if not all_codebooks_exist(args.output_dir, tensor, variant):
        raise RuntimeError(f"training finished but codebooks are incomplete for {tensor} {variant.name}")


def eval_rows_for_task(args: argparse.Namespace, tensor: str, variant: Variant, split_name: str, layer: int, head: int) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from run_rvq_diag import layer_file, rvq_encode_reconstruct, split_items, topk_overlap

    output_dir = args.output_dir
    cb_path = codebook_file(output_dir, layer, head, codebook_subdir(tensor, variant))
    codebooks = torch.load(cb_path, map_location="cpu", weights_only=False)["codebooks"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for item in split_items(output_dir, split_name):
        pid = int(item["prompt_id"])
        payload = torch.load(layer_file(output_dir, split_name, pid, layer), map_location="cpu", weights_only=False)
        target = payload[tensor][:, head].float().contiguous()
        flat_target = target.reshape(-1, target.shape[-1])
        total = flat_target.shape[0]
        gen = torch.Generator(device="cpu").manual_seed(args.seed + pid * 1000000 + layer * 10000 + head * 100 + 17)
        take = min(args.eval_vector_sample, total)
        recon_idx = torch.randperm(total, generator=gen)[:take]
        _idx, target_hat = rvq_encode_reconstruct(flat_target[recon_idx], codebooks, device, args.eval_batch_size)
        flat_hat = target_hat.reshape(-1, target_hat.shape[-1])
        flat_metric_target = flat_target[recon_idx]
        err = flat_metric_target - flat_hat
        mse = float((err * err).mean().item())
        rel = float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(flat_metric_target).clamp_min(1e-12))
        cosine = float(F.cosine_similarity(flat_metric_target, flat_hat, dim=-1).mean().item())

        logits_mse = None
        attn_kl = None
        overlap = None
        if tensor == "K":
            q = payload["Q"][:, head].float().contiguous().reshape(-1, target.shape[-1])
            k = payload["K"][:, head].float().contiguous().reshape(-1, target.shape[-1])
            gen = torch.Generator(device="cpu").manual_seed(args.seed + pid * 1000000 + layer * 10000 + head * 100)
            q_take = min(args.query_sample, total)
            k_take = min(args.key_sample, total)
            q_idx = torch.randperm(total, generator=gen)[:q_take]
            k_idx = torch.randperm(total, generator=gen)[:k_take]
            _ids, kh_cpu = rvq_encode_reconstruct(flat_target[k_idx], codebooks, device, args.eval_batch_size)
            q_sample = q[q_idx].to(device)
            k_sample = k[k_idx].to(device)
            kh_sample = kh_cpu.reshape(-1, kh_cpu.shape[-1]).float().to(device)
            logits = q_sample @ k_sample.T / math.sqrt(target.shape[-1])
            logits_hat = q_sample @ kh_sample.T / math.sqrt(target.shape[-1])
            logits_mse = float(((logits - logits_hat) ** 2).mean().item())
            log_p = F.log_softmax(logits, dim=-1)
            log_q = F.log_softmax(logits_hat, dim=-1)
            p = log_p.exp()
            attn_kl = float((p * (log_p - log_q)).sum(dim=-1).mean().item())
            overlap = topk_overlap(logits, logits_hat, args.topk)

        rows.append(
            {
                "split": split_name,
                "prompt_id": pid,
                "source_index": int(item["source_index"]),
                "tensor": tensor,
                "variant": variant.name,
                "levels": variant.levels,
                "codebook_size": variant.codebook_size,
                "layer": layer,
                "head": head,
                "reconstruction_cosine": cosine,
                "reconstruction_mse": mse,
                "relative_error": rel,
                "qk_logits_mse": logits_mse,
                "attention_kl": attn_kl,
                "topk_overlap": overlap,
            }
        )
    return rows


def eval_head_from_payload(
    args: argparse.Namespace,
    tensor: str,
    variant: Variant,
    split_name: str,
    item: dict[str, Any],
    layer: int,
    head: int,
    payload: dict[str, Any],
    codebooks: "Any",
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from run_rvq_diag import rvq_encode_reconstruct, topk_overlap

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pid = int(item["prompt_id"])
    target = payload[tensor][:, head].float().contiguous()
    flat_target = target.reshape(-1, target.shape[-1])
    total = flat_target.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(args.seed + pid * 1000000 + layer * 10000 + head * 100 + 17)
    take = min(args.eval_vector_sample, total)
    recon_idx = torch.randperm(total, generator=gen)[:take]
    _idx, target_hat = rvq_encode_reconstruct(flat_target[recon_idx], codebooks, device, args.eval_batch_size)
    flat_hat = target_hat.reshape(-1, target_hat.shape[-1])
    flat_metric_target = flat_target[recon_idx]
    err = flat_metric_target - flat_hat
    mse = float((err * err).mean().item())
    rel = float(torch.linalg.vector_norm(err) / torch.linalg.vector_norm(flat_metric_target).clamp_min(1e-12))
    cosine = float(F.cosine_similarity(flat_metric_target, flat_hat, dim=-1).mean().item())

    logits_mse = None
    attn_kl = None
    overlap = None
    if tensor == "K":
        q = payload["Q"][:, head].float().contiguous().reshape(-1, target.shape[-1])
        k = payload["K"][:, head].float().contiguous().reshape(-1, target.shape[-1])
        gen = torch.Generator(device="cpu").manual_seed(args.seed + pid * 1000000 + layer * 10000 + head * 100)
        q_take = min(args.query_sample, total)
        k_take = min(args.key_sample, total)
        q_idx = torch.randperm(total, generator=gen)[:q_take]
        k_idx = torch.randperm(total, generator=gen)[:k_take]
        _ids, kh_cpu = rvq_encode_reconstruct(flat_target[k_idx], codebooks, device, args.eval_batch_size)
        q_sample = q[q_idx].to(device)
        k_sample = k[k_idx].to(device)
        kh_sample = kh_cpu.reshape(-1, kh_cpu.shape[-1]).float().to(device)
        logits = q_sample @ k_sample.T / math.sqrt(target.shape[-1])
        logits_hat = q_sample @ kh_sample.T / math.sqrt(target.shape[-1])
        logits_mse = float(((logits - logits_hat) ** 2).mean().item())
        log_p = F.log_softmax(logits, dim=-1)
        log_q = F.log_softmax(logits_hat, dim=-1)
        p = log_p.exp()
        attn_kl = float((p * (log_p - log_q)).sum(dim=-1).mean().item())
        overlap = topk_overlap(logits, logits_hat, args.topk)

    return {
        "split": split_name,
        "prompt_id": pid,
        "source_index": int(item["source_index"]),
        "tensor": tensor,
        "variant": variant.name,
        "levels": variant.levels,
        "codebook_size": variant.codebook_size,
        "layer": layer,
        "head": head,
        "reconstruction_cosine": cosine,
        "reconstruction_mse": mse,
        "relative_error": rel,
        "qk_logits_mse": logits_mse,
        "attention_kl": attn_kl,
        "topk_overlap": overlap,
    }


def eval_rows_for_layer(args: argparse.Namespace, tensor: str, variant: Variant, split_name: str, layer: int) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from run_rvq_diag import layer_file, split_items

    output_dir = args.output_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    codebooks = torch.stack([
        torch.load(
            codebook_file(output_dir, layer, head, codebook_subdir(tensor, variant)),
            map_location="cpu",
            weights_only=False,
        )["codebooks"]
        for head in range(HEADS)
    ], dim=0).float().to(device)

    def gather_positions(x: "Any", indices: "Any") -> "Any":
        return x.gather(1, indices[:, :, None].expand(-1, -1, x.shape[-1]))

    def sample_matrix(total: int, take: int, seed_offset: int) -> "Any":
        take = min(take, total)
        idx = []
        for head in range(HEADS):
            gen = torch.Generator(device="cpu").manual_seed(seed_offset + head * 100)
            idx.append(torch.randperm(total, generator=gen)[:take])
        return torch.stack(idx, dim=0).to(device)

    def rvq_reconstruct_heads(samples: "Any") -> "Any":
        # samples: [heads, sample, dim], codebooks: [heads, levels, codebook, dim]
        scale = torch.sqrt(torch.mean(samples * samples, dim=-1, keepdim=True)).clamp_min(1e-6)
        residual = samples / scale
        recon_norm = torch.zeros_like(residual)
        for level in range(codebooks.shape[1]):
            centroids = codebooks[:, level]
            dist = (
                (residual * residual).sum(dim=-1, keepdim=True)
                + (centroids * centroids).sum(dim=-1)[:, None, :]
                - 2.0 * torch.bmm(residual, centroids.transpose(1, 2))
            )
            ids = torch.argmin(dist, dim=-1)
            assigned = centroids.gather(1, ids[:, :, None].expand(-1, -1, centroids.shape[-1]))
            recon_norm = recon_norm + assigned
            residual = residual - assigned
        return recon_norm * scale

    rows = []
    for item in split_items(output_dir, split_name):
        pid = int(item["prompt_id"])
        payload = torch.load(layer_file(output_dir, split_name, pid, layer), map_location="cpu", weights_only=False)
        target_all = payload[tensor].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float().to(device)
        total = target_all.shape[1]
        recon_idx = sample_matrix(
            total,
            args.eval_vector_sample,
            args.seed + pid * 1000000 + layer * 10000 + 17,
        )
        target_sample = gather_positions(target_all, recon_idx)
        target_hat = rvq_reconstruct_heads(target_sample)
        err = target_sample - target_hat
        mse = (err * err).mean(dim=(1, 2))
        rel = torch.linalg.vector_norm(err, dim=(1, 2)) / torch.linalg.vector_norm(target_sample, dim=(1, 2)).clamp_min(1e-12)
        cosine = F.cosine_similarity(target_sample, target_hat, dim=-1).mean(dim=1)

        logits_mse = torch.full((HEADS,), float("nan"), device=device)
        attn_kl = torch.full((HEADS,), float("nan"), device=device)
        overlap = torch.full((HEADS,), float("nan"), device=device)
        if tensor == "K":
            q_all = payload["Q"].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float().to(device)
            k_all = payload["K"].permute(1, 0, 2, 3).reshape(HEADS, -1, DIM).float().to(device)
            q_idx = sample_matrix(total, args.query_sample, args.seed + pid * 1000000 + layer * 10000)
            k_idx = sample_matrix(total, args.key_sample, args.seed + pid * 1000000 + layer * 10000 + 999)
            q_sample = gather_positions(q_all, q_idx)
            k_sample = gather_positions(k_all, k_idx)
            kh_sample = rvq_reconstruct_heads(k_sample)
            logits = torch.bmm(q_sample, k_sample.transpose(1, 2)) / math.sqrt(DIM)
            logits_hat = torch.bmm(q_sample, kh_sample.transpose(1, 2)) / math.sqrt(DIM)
            logits_mse = ((logits - logits_hat) ** 2).mean(dim=(1, 2))
            log_p = F.log_softmax(logits, dim=-1)
            log_q = F.log_softmax(logits_hat, dim=-1)
            p = log_p.exp()
            attn_kl = (p * (log_p - log_q)).sum(dim=-1).mean(dim=-1)
            k_top = min(args.topk, logits.shape[-1])
            top_a = torch.topk(logits, k=k_top, dim=-1).indices
            top_b = torch.topk(logits_hat, k=k_top, dim=-1).indices
            overlap = (top_a[:, :, :, None] == top_b[:, :, None, :]).any(dim=-1).float().mean(dim=(1, 2))

        mse_v = mse.detach().cpu().tolist()
        rel_v = rel.detach().cpu().tolist()
        cosine_v = cosine.detach().cpu().tolist()
        logits_mse_v = logits_mse.detach().cpu().tolist()
        attn_kl_v = attn_kl.detach().cpu().tolist()
        overlap_v = overlap.detach().cpu().tolist()
        for head in range(HEADS):
            rows.append(
                {
                    "split": split_name,
                    "prompt_id": pid,
                    "source_index": int(item["source_index"]),
                    "tensor": tensor,
                    "variant": variant.name,
                    "levels": variant.levels,
                    "codebook_size": variant.codebook_size,
                    "layer": layer,
                    "head": head,
                    "reconstruction_cosine": cosine_v[head],
                    "reconstruction_mse": mse_v[head],
                    "relative_error": rel_v[head],
                    "qk_logits_mse": None if math.isnan(logits_mse_v[head]) else logits_mse_v[head],
                    "attention_kl": None if math.isnan(attn_kl_v[head]) else attn_kl_v[head],
                    "topk_overlap": None if math.isnan(overlap_v[head]) else overlap_v[head],
                }
            )
        del payload, target_all, target_sample, target_hat, err
    return rows


def eval_worker(args: argparse.Namespace) -> None:
    from run_rvq_diag import setup_paths

    setup_paths()
    variant = variant_map()[args.variant]
    rows = []
    for layer in parse_int_list(args.tasks):
        print(f"[fast-eval] {args.tensor} {variant.name} layer={layer} start", flush=True)
        for split_name in [s.strip() for s in args.eval_splits.split(",") if s.strip()]:
            rows.extend(eval_rows_for_layer(args, args.tensor, variant, split_name, layer))
        print(f"[fast-eval] {args.tensor} {variant.name} layer={layer} done", flush=True)
    out = Path(args.partial_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(rows, f)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value in (None, ""):
            continue
        values.append(float(value))
    return values


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "reconstruction_cosine",
        "reconstruction_mse",
        "relative_error",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
    ]
    summary: dict[str, Any] = {"global": {}, "by_layer": {}, "by_head": {}}
    for split_name in ("train", "test"):
        split_rows = [r for r in rows if r["split"] == split_name]
        summary["global"][split_name] = {}
        for key in keys:
            m, s = mean_std(numeric_values(split_rows, key))
            summary["global"][split_name][f"{key}_mean"] = m
            summary["global"][split_name][f"{key}_std"] = s
    for group_key, target in (("layer", "by_layer"), ("head", "by_head")):
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((row["split"], int(row[group_key])), []).append(row)
        for (split_name, value), group_rows in sorted(groups.items()):
            target_key = f"{split_name}_{value}"
            summary[target][target_key] = {}
            for key in keys:
                m, s = mean_std(numeric_values(group_rows, key))
                summary[target][target_key][f"{key}_mean"] = m
                summary[target][target_key][f"{key}_std"] = s
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "prompt_id",
        "source_index",
        "tensor",
        "variant",
        "levels",
        "codebook_size",
        "layer",
        "head",
        "reconstruction_cosine",
        "reconstruction_mse",
        "relative_error",
        "qk_logits_mse",
        "attention_kl",
        "topk_overlap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_fast_eval(args: argparse.Namespace, tensor: str, variant: Variant, gpus: list[int], log_dir: Path) -> dict[str, Any]:
    out_summary = summary_path(args.output_dir, tensor, variant)
    if out_summary.exists() and not args.overwrite:
        with out_summary.open("r", encoding="utf-8") as f:
            return json.load(f)

    pdir = partial_dir(args.output_dir, tensor, variant)
    if pdir.exists():
        shutil.rmtree(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    tasks = list(range(LAYERS))
    procs = []
    for i, gpu in enumerate(gpus):
        assigned = tasks[i :: len(gpus)]
        if not assigned:
            continue
        partial = pdir / f"worker_gpu{gpu}.json"
        log_path = log_dir / f"eval_{tensor_prefix(tensor, variant)}_gpu{gpu}_{int(time.time())}.log"
        cmd = [
            str(args.worker_python),
            str(Path(__file__).resolve()),
            "eval-worker",
            "--output-dir",
            str(args.output_dir),
            "--seed",
            str(args.seed),
            "--tensor",
            tensor,
            "--variant",
            variant.name,
            "--eval-splits",
            args.eval_splits,
            "--eval-vector-sample",
            str(args.eval_vector_sample),
            "--query-sample",
            str(args.query_sample),
            "--key-sample",
            str(args.key_sample),
            "--topk",
            str(args.topk),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--tasks",
            ",".join(str(v) for v in assigned),
            "--partial-out",
            str(partial),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        with log_path.open("w", encoding="utf-8") as log_f:
            log_f.write("[launch] " + " ".join(cmd) + "\n")
            log_f.flush()
            procs.append((subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env), log_path))
    failures = []
    for proc, log_path in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append((rc, log_path))
    if failures:
        details = "\n".join(f"rc={rc} log={path}" for rc, path in failures)
        raise RuntimeError(f"fast eval failed for {tensor} {variant.name}:\n{details}")

    rows = []
    for path in sorted(pdir.glob("worker_gpu*.json")):
        with path.open("r", encoding="utf-8") as f:
            rows.extend(json.load(f))
    summary = summarize_rows(rows)
    payload = {
        "variant": variant.__dict__,
        "tensor": tensor,
        "codebook_subdir": codebook_subdir(tensor, variant),
        "eval": {
            "splits": args.eval_splits,
            "eval_vector_sample": args.eval_vector_sample,
            "query_sample": args.query_sample,
            "key_sample": args.key_sample,
            "topk": args.topk,
            "rows": len(rows),
        },
        "compression": compression_estimates(variant.levels, variant.codebook_size),
        "summary": summary,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    write_rows_csv(rows_path(args.output_dir, tensor, variant), rows)
    shutil.rmtree(pdir, ignore_errors=True)
    return payload


def write_overall_summary(output_dir: Path, variants: list[Variant], tensors: list[str]) -> Path:
    rows = []
    for variant in variants:
        for tensor in tensors:
            path = summary_path(output_dir, tensor, variant)
            row = {
                "variant": variant.name,
                "tensor": tensor,
                "levels": variant.levels,
                "codebook_size": variant.codebook_size,
                **compression_estimates(variant.levels, variant.codebook_size),
                "status": "missing",
            }
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                row["status"] = "complete"
                for split in ("train", "test"):
                    stats = payload["summary"]["global"].get(split, {})
                    for key in (
                        "reconstruction_cosine_mean",
                        "reconstruction_mse_mean",
                        "relative_error_mean",
                        "qk_logits_mse_mean",
                        "attention_kl_mean",
                        "topk_overlap_mean",
                    ):
                        row[f"{split}_{key}"] = stats.get(key)
            rows.append(row)
    out = output_dir / "rvq_hparam_fast_summary.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    return out


def run_one_job(args: argparse.Namespace, tensor: str, variant: Variant, gpus: list[int], log_dir: Path) -> None:
    print(f"[job] {tensor} {variant.name} gpus={gpus}", flush=True)
    t0 = time.perf_counter()
    train_codebooks(args, tensor, variant, gpus, log_dir)
    t1 = time.perf_counter()
    run_fast_eval(args, tensor, variant, gpus, log_dir)
    t2 = time.perf_counter()
    payload = {
        "tensor": tensor,
        "variant": variant.name,
        "levels": variant.levels,
        "codebook_size": variant.codebook_size,
        "gpus": gpus,
        "train_wall_time_s": t1 - t0,
        "eval_wall_time_s": t2 - t1,
        "total_wall_time_s": t2 - t0,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = timing_path(args.output_dir, tensor, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {tensor} {variant.name}", flush=True)


def parse_main_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/bowen/rvq_diag"))
    parser.add_argument(
        "--prompt-source",
        type=Path,
        default=Path("/home/bowen/QuantVid/external/Self-Forcing/prompts/vbench/all_dimension.txt"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-num", type=int, default=20)
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument("--variants", type=str, default=",".join(v.name for v in DEFAULT_VARIANTS))
    parser.add_argument("--tensors", type=str, default="K,V")
    parser.add_argument("--gpu-groups", type=str, default="0,1;2,3;4,5;6,7")
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--worker-python", type=str, default=os.environ.get("RVQ_DIAG_WORKER_PYTHON", sys.executable))
    parser.add_argument("--dump-python", type=str, default=os.environ.get("RVQ_DIAG_DUMP_PYTHON", sys.executable))
    parser.add_argument("--eval-splits", type=str, default="train,test")
    parser.add_argument("--eval-vector-sample", type=int, default=4096)
    parser.add_argument("--query-sample", type=int, default=128)
    parser.add_argument("--key-sample", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_worker_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=["eval-worker"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tensor", type=str, required=True)
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--eval-splits", type=str, required=True)
    parser.add_argument("--eval-vector-sample", type=int, required=True)
    parser.add_argument("--query-sample", type=int, required=True)
    parser.add_argument("--key-sample", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--eval-batch-size", type=int, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--partial-out", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "eval-worker":
        eval_worker(parse_worker_args(sys.argv[1:]))
        return

    args = parse_main_args()
    variants_by_name = variant_map()
    variants = [variants_by_name[name.strip()] for name in args.variants.split(",") if name.strip()]
    tensors = [t.strip().upper() for t in args.tensors.split(",") if t.strip()]
    gpu_groups = parse_gpu_groups(args.gpu_groups)
    max_parallel = min(args.max_parallel, len(gpu_groups))
    log_dir = fast_dir(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(tensor, variant) for variant in variants for tensor in tensors]
    active: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    next_job = 0
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        while next_job < len(jobs) or active:
            while next_job < len(jobs) and len(active) < max_parallel:
                active_groups = {tuple(job["gpus"]) for job in active}
                group = next((candidate for candidate in gpu_groups if tuple(candidate) not in active_groups), None)
                if group is None:
                    break
                tensor, variant = jobs[next_job]
                next_job += 1
                future = pool.submit(run_one_job, args, tensor, variant, group, log_dir)
                active.append({"future": future, "tensor": tensor, "variant": variant, "gpus": group})

            if not active:
                continue
            done_futures, _pending = wait(
                [job["future"] for job in active],
                timeout=30,
                return_when=FIRST_COMPLETED,
            )
            still_active = []
            for job in active:
                future = job["future"]
                if future not in done_futures:
                    still_active.append(job)
                    continue
                try:
                    future.result()
                except Exception as exc:
                    tensor = job["tensor"]
                    variant = job["variant"]
                    print(f"[failed] {tensor} {variant.name}: {exc}", flush=True)
                    failures.append({"tensor": tensor, "variant": variant.name, "error": str(exc)})
            active = still_active

    out = write_overall_summary(args.output_dir, variants, tensors)
    print(f"[summary] {out}", flush=True)
    if failures:
        fail_path = fast_dir(args.output_dir) / "failures.json"
        with fail_path.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        raise SystemExit(f"{len(failures)} jobs failed; see {fail_path}")


if __name__ == "__main__":
    main()
