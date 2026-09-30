#!/usr/bin/env python3
"""Evaluate RVQ runtime storage quantization and index lossless coding.

This is an offline diagnostic for the MovieGen T2V RVQ benchmark. It reuses
saved pre-RoPE K/Q/V dumps and trained RVQ codebooks, then measures the actual
reconstruction produced when runtime storage is INT8 or INT4:

  INT8 storage: int8 codebook vectors + uint8 per-token scale
  INT4 storage: packed int4 codebook vectors + packed uint4 per-token scale

For INT4 storage it also estimates lossless index compression with the same
three deployable modes used elsewhere:

  raw: bitpacked RVQ indices
  symbol: entropy code the original index stream
  temporal_delta: first frame raw + entropy code x[t] - x[t-1]

Offline lossless uses train-split histograms and applies cross-entropy to the
test split. Online lossless is an oracle upper bound using per-test-block
histograms.
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
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache


LAYERS = 30
HEADS = 24
TOKENS_PER_FRAME = 880
LOGICAL_DIM = 256
BF16_BYTES = 2
DEFAULT_OUTPUT_DIR = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


@dataclass(frozen=True)
class Variant:
    name: str
    levels: int
    codebook_size: int


VARIANTS = {
    "l4c256": Variant("l4c256", 4, 256),
    "l8c64": Variant("l8c64", 8, 64),
    "l28c64": Variant("l28c64", 28, 64),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--variants", default="l4c256,l8c64")
    p.add_argument("--storages", default="int8,int4")
    p.add_argument("--splits", default="train,test")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--worker-python", default=sys.executable)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--query-sample", type=int, default=128)
    p.add_argument("--key-sample", type=int, default=2048)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frames-for-memory", type=int, default=40)
    p.add_argument("--mode-bits", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-variant", default="")
    p.add_argument("--worker-storage", default="")
    p.add_argument("--worker-tensor", default="")
    p.add_argument("--worker-split", default="")
    p.add_argument("--worker-prompt-id", type=int, default=-1)
    return p.parse_args()


def csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def codebook_dir(output_dir: Path, tensor: str, variant: str) -> Path:
    if variant == "l4c256":
        return output_dir / ("codebooks" if tensor == "K" else "v_codebooks")
    return output_dir / f"sweep_{tensor.lower()}_{variant}_codebooks"


def dump_path(output_dir: Path, split: str, prompt_id: int, layer: int) -> Path:
    return output_dir / "kv_dump" / split / str(prompt_id) / f"layer_{layer}.pt"


def prompt_ids(output_dir: Path, split: str) -> list[int]:
    root = output_dir / "kv_dump" / split
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir())


def partial_path(
    args: argparse.Namespace,
    variant: str,
    storage: str,
    tensor: str,
    split: str,
    prompt_id: int | None = None,
) -> Path:
    name = f"{variant}_{storage}_{tensor}_{split}"
    if prompt_id is not None and prompt_id >= 0:
        name += f"_p{prompt_id}"
    return (
        args.output_dir
        / "metrics"
        / "rvq_storage_eval"
        / "partials"
        / f"{name}.pt"
    )


def scale_storage(storage: str) -> str:
    if storage == "int8":
        return "uint8"
    if storage == "int4":
        return "uint4"
    raise ValueError(storage)


def entropy_bits_from_counts(counts: torch.Tensor) -> float:
    total = float(counts.sum().item())
    if total <= 0:
        return 0.0
    nz = counts[counts > 0].double()
    probs = nz / total
    return float((-nz * torch.log2(probs)).sum().item())


def cross_entropy_bits_from_counts(values: torch.Tensor, table: torch.Tensor, alpha: float = 0.5) -> float:
    values = values.double()
    table = table.double()
    alphabet = int(table.numel())
    total = float(table.sum().item()) + alpha * alphabet
    if total <= 0:
        return 0.0
    probs = (table + alpha) / total
    return float((-values * torch.log2(probs)).sum().item())


def count_block(stage_idx: torch.Tensor, codebook_size: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    x = stage_idx.to(torch.long).cpu()
    symbol = torch.bincount(x.reshape(-1), minlength=codebook_size).to(torch.int64)
    delta = (x[1:] - x[:-1]) % codebook_size
    delta_counts = torch.bincount(delta.reshape(-1), minlength=codebook_size).to(torch.int64)
    first_symbols = int(x[:1].numel())
    total_symbols = int(x.numel())
    return symbol, delta_counts, first_symbols, total_symbols


def online_block_costs(
    symbol_counts: torch.Tensor,
    delta_counts: torch.Tensor,
    first_symbols: int,
    total_symbols: int,
    codebook_size: int,
    mode_bits: float,
) -> dict[str, float]:
    bits = math.ceil(math.log2(codebook_size))
    raw = total_symbols * bits + mode_bits
    symbol = entropy_bits_from_counts(symbol_counts) + mode_bits
    temporal = first_symbols * bits + entropy_bits_from_counts(delta_counts) + mode_bits
    return {
        "raw": raw,
        "symbol": min(symbol, raw),
        "temporal_delta": min(temporal, raw),
    }


def topk_overlap(logits: torch.Tensor, logits_hat: torch.Tensor, topk: int) -> float:
    k = min(int(topk), logits.shape[-1])
    top_a = torch.topk(logits, k=k, dim=-1).indices
    top_b = torch.topk(logits_hat, k=k, dim=-1).indices
    return float((top_a[:, :, None] == top_b[:, None, :]).any(dim=-1).float().mean().item())


@torch.no_grad()
def eval_one_layer(
    args: argparse.Namespace,
    variant: Variant,
    storage: str,
    tensor_name: str,
    split: str,
    prompt_id: int,
    layer: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = torch.load(dump_path(args.output_dir, split, prompt_id, layer), map_location="cpu", weights_only=False)
    target = payload[tensor_name].to(device=device, non_blocking=True).contiguous()

    packed = rvq_encode_tensor_cache(
        target,
        layer_idx=layer,
        codebook_dir=codebook_dir(args.output_dir, tensor_name, variant.name),
        batch_size=args.eval_batch_size,
        compute_dtype="bf16",
        codebook_storage_dtype=storage,
        scale_storage_dtype=scale_storage(storage),
    )
    recon = rvq_dequantize_tensor_cache(packed, target.dtype)

    target_f = target.float()
    recon_f = recon.float()
    err = target_f - recon_f
    mse = (err * err).mean(dim=(0, 2, 3)).detach().cpu()
    rel = (
        torch.linalg.vector_norm(err, dim=(0, 2, 3))
        / torch.linalg.vector_norm(target_f, dim=(0, 2, 3)).clamp_min(1e-12)
    ).detach().cpu()
    cosine = F.cosine_similarity(target_f, recon_f, dim=-1).mean(dim=(0, 2)).detach().cpu()

    rows: list[dict[str, Any]] = []
    for head in range(HEADS):
        row = {
            "split": split,
            "prompt_id": prompt_id,
            "tensor": tensor_name,
            "variant": variant.name,
            "storage": storage,
            "levels": variant.levels,
            "codebook_size": variant.codebook_size,
            "layer": layer,
            "head": head,
            "reconstruction_cosine": float(cosine[head].item()),
            "reconstruction_mse": float(mse[head].item()),
            "relative_error": float(rel[head].item()),
            "qk_logits_mse": None,
            "attention_kl": None,
            "topk_overlap": None,
        }
        rows.append(row)

    if tensor_name == "K":
        q_all = payload["Q"].to(device=device, non_blocking=True).float().contiguous()
        n_tokens = target.shape[0] * target.shape[2]
        k_full = target_f.permute(1, 0, 2, 3).reshape(HEADS, n_tokens, target.shape[-1])
        kh_full = recon_f.permute(1, 0, 2, 3).reshape(HEADS, n_tokens, target.shape[-1])
        q_full = q_all.permute(1, 0, 2, 3).reshape(HEADS, n_tokens, target.shape[-1])
        for head in range(HEADS):
            gen = torch.Generator(device="cpu").manual_seed(
                args.seed + prompt_id * 1000000 + layer * 10000 + head * 100
            )
            q_take = min(args.query_sample, n_tokens)
            k_take = min(args.key_sample, n_tokens)
            q_idx = torch.randperm(n_tokens, generator=gen)[:q_take].to(device)
            k_idx = torch.randperm(n_tokens, generator=gen)[:k_take].to(device)
            q_sample = q_full[head].index_select(0, q_idx)
            k_sample = k_full[head].index_select(0, k_idx)
            kh_sample = kh_full[head].index_select(0, k_idx)
            logits = q_sample @ k_sample.T / math.sqrt(target.shape[-1])
            logits_hat = q_sample @ kh_sample.T / math.sqrt(target.shape[-1])
            log_p = F.log_softmax(logits, dim=-1)
            log_q = F.log_softmax(logits_hat, dim=-1)
            p = log_p.exp()
            rows[head]["qk_logits_mse"] = float(((logits - logits_hat) ** 2).mean().item())
            rows[head]["attention_kl"] = float((p * (log_p - log_q)).sum(dim=-1).mean().item())
            rows[head]["topk_overlap"] = topk_overlap(logits, logits_hat, args.topk)

    lossless_blocks: list[dict[str, Any]] = []
    if storage == "int4":
        indices = packed["indices"].detach().cpu()
        for stage in range(variant.levels):
            for head in range(HEADS):
                symbol, delta, first_symbols, total_symbols = count_block(
                    indices[stage, :, head, :],
                    variant.codebook_size,
                )
                online = online_block_costs(
                    symbol,
                    delta,
                    first_symbols,
                    total_symbols,
                    variant.codebook_size,
                    args.mode_bits,
                )
                lossless_blocks.append(
                    {
                        "split": split,
                        "prompt_id": prompt_id,
                        "tensor": tensor_name,
                        "variant": variant.name,
                        "stage": stage + 1,
                        "layer": layer,
                        "head": head,
                        "symbol_counts": symbol,
                        "delta_counts": delta,
                        "first_symbols": first_symbols,
                        "total_symbols": total_symbols,
                        "online_costs": online,
                    }
                )

    del payload, target, packed, recon
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, lossless_blocks


def worker_main(args: argparse.Namespace) -> None:
    variant = VARIANTS[args.worker_variant]
    storage = args.worker_storage
    tensor_name = args.worker_tensor
    split = args.worker_split
    worker_prompt_id = int(args.worker_prompt_id)
    out = partial_path(args, variant.name, storage, tensor_name, split, worker_prompt_id)
    if out.exists() and not args.overwrite:
        print(f"[worker-skip] {out}", flush=True)
        return

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    lossless_blocks: list[dict[str, Any]] = []
    started = time.perf_counter()
    pids = [worker_prompt_id] if worker_prompt_id >= 0 else prompt_ids(args.output_dir, split)
    for prompt_id in pids:
        for layer in range(LAYERS):
            print(
                f"[storage-eval] {variant.name} {storage} {tensor_name} {split} "
                f"prompt={prompt_id} layer={layer}",
                flush=True,
            )
            layer_rows, layer_blocks = eval_one_layer(
                args,
                variant,
                storage,
                tensor_name,
                split,
                prompt_id,
                layer,
                device,
            )
            rows.extend(layer_rows)
            lossless_blocks.extend(layer_blocks)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rows": rows,
            "lossless_blocks": lossless_blocks,
            "wall_time_s": time.perf_counter() - started,
            "config": {
                "variant": variant.name,
                "storage": storage,
                "tensor": tensor_name,
                "split": split,
                "scale_storage": scale_storage(storage),
            },
        },
        out,
    )
    print(f"[worker-done] {out}", flush=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_reconstruction(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["variant"], row["storage"], row["tensor"], row["split"])].append(row)
    for (variant, storage, tensor_name, split), group in sorted(groups.items()):
        out.append(
            {
                "variant": variant,
                "storage": storage,
                "tensor": tensor_name,
                "split": split,
                "rows": len(group),
                "reconstruction_cosine": mean([float(r["reconstruction_cosine"]) for r in group]),
                "reconstruction_mse": mean([float(r["reconstruction_mse"]) for r in group]),
                "relative_error": mean([float(r["relative_error"]) for r in group]),
                "qk_logits_mse": mean([float(r["qk_logits_mse"]) for r in group if r["qk_logits_mse"] is not None]),
                "attention_kl": mean([float(r["attention_kl"]) for r in group if r["attention_kl"] is not None]),
                "topk_overlap": mean([float(r["topk_overlap"]) for r in group if r["topk_overlap"] is not None]),
            }
        )
    return out


def merge_counters(blocks: list[dict[str, Any]], variant: Variant) -> dict[tuple[str, int], dict[str, torch.Tensor]]:
    merged: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
    for block in blocks:
        key = (block["tensor"], int(block["stage"]))
        if key not in merged:
            merged[key] = {
                "symbol": torch.zeros(variant.codebook_size, dtype=torch.int64),
                "delta": torch.zeros(variant.codebook_size, dtype=torch.int64),
            }
        merged[key]["symbol"] += block["symbol_counts"].to(torch.int64)
        merged[key]["delta"] += block["delta_counts"].to(torch.int64)
    return merged


def summarize_lossless(
    args: argparse.Namespace,
    blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_variant[block["variant"]].append(block)

    for variant_name, variant_blocks in sorted(by_variant.items()):
        variant = VARIANTS[variant_name]
        train_blocks = [b for b in variant_blocks if b["split"] == "train"]
        test_blocks = [b for b in variant_blocks if b["split"] == "test"]
        train_tables = merge_counters(train_blocks, variant)

        accum: dict[tuple[str, str, int], dict[str, float]] = defaultdict(lambda: defaultdict(float))
        modes: dict[tuple[str, str, int], Counter[str]] = defaultdict(Counter)

        for block in test_blocks:
            bits = math.ceil(math.log2(variant.codebook_size))
            raw = block["total_symbols"] * bits + args.mode_bits
            online_costs = block["online_costs"]
            table = train_tables[(block["tensor"], int(block["stage"]))]
            symbol = cross_entropy_bits_from_counts(
                block["symbol_counts"],
                table["symbol"],
                alpha=0.5,
            ) + args.mode_bits
            temporal = (
                block["first_symbols"] * bits
                + cross_entropy_bits_from_counts(block["delta_counts"], table["delta"], alpha=0.5)
                + args.mode_bits
            )
            offline_costs = {
                "raw": raw,
                "symbol": min(symbol, raw),
                "temporal_delta": min(temporal, raw),
            }
            for family, costs in (("online_oracle", online_costs), ("offline_train_table", offline_costs)):
                best_mode = min(costs, key=lambda key: costs[key])
                key = (family, block["tensor"], int(block["stage"]))
                accum[key]["raw_bits"] += raw
                accum[key]["best_bits"] += costs[best_mode]
                accum[key]["symbol_bits"] += costs["symbol"]
                accum[key]["temporal_delta_bits"] += costs["temporal_delta"]
                modes[key][best_mode] += 1

        for (family, tensor_name, stage), vals in sorted(accum.items()):
            raw_bits = vals["raw_bits"]
            best_bits = vals["best_bits"]
            out.append(
                {
                    "variant": variant_name,
                    "storage": "int4",
                    "tensor": tensor_name,
                    "family": family,
                    "stage": stage,
                    "raw_mib": raw_bits / 8 / 1024**2,
                    "best_mib": best_bits / 8 / 1024**2,
                    "best_index_ratio": raw_bits / best_bits if best_bits else None,
                    "symbol_ratio_forced": raw_bits / vals["symbol_bits"] if vals["symbol_bits"] else None,
                    "temporal_delta_ratio_forced": (
                        raw_bits / vals["temporal_delta_bits"] if vals["temporal_delta_bits"] else None
                    ),
                    "mode_counts": json.dumps(dict(modes[(family, tensor_name, stage)]), sort_keys=True),
                }
            )

        for family in ("online_oracle", "offline_train_table"):
            for tensor_name in ("K", "V"):
                selected = [
                    r
                    for r in out
                    if r["variant"] == variant_name and r["family"] == family and r["tensor"] == tensor_name
                ]
                if selected:
                    raw_bits = sum(float(r["raw_mib"]) for r in selected)
                    best_bits = sum(float(r["best_mib"]) for r in selected)
                    out.append(
                        {
                            "variant": variant_name,
                            "storage": "int4",
                            "tensor": tensor_name,
                            "family": family,
                            "stage": "all",
                            "raw_mib": raw_bits,
                            "best_mib": best_bits,
                            "best_index_ratio": raw_bits / best_bits if best_bits else None,
                            "symbol_ratio_forced": None,
                            "temporal_delta_ratio_forced": None,
                            "mode_counts": "",
                        }
                    )
    return out


def theoretical_full_mib(frames: int) -> float:
    return 2 * LAYERS * HEADS * frames * TOKENS_PER_FRAME * LOGICAL_DIM * BF16_BYTES / 1024**2


def rvq_int4_components_mib(variant: Variant, frames: int, k_ratio: float, v_ratio: float) -> dict[str, float]:
    vectors = LAYERS * HEADS * frames * TOKENS_PER_FRAME
    index_bits = math.ceil(math.log2(variant.codebook_size))
    one_index = vectors * variant.levels * index_bits / 8.0
    one_scale = vectors / 2.0
    one_codebook = LAYERS * HEADS * variant.levels * variant.codebook_size * LOGICAL_DIM / 2.0
    codebook_meta = LAYERS * HEADS * variant.levels * variant.codebook_size * 2
    k_indices = 2 * one_index
    v_indices = 2 * one_index
    k_lossless = k_indices / k_ratio
    v_lossless = v_indices / v_ratio
    k_scale = 2 * one_scale
    v_scale = 2 * one_scale
    total_before = k_indices + v_indices + k_scale + v_scale + 2 * one_codebook + 2 * codebook_meta
    total_after = k_lossless + v_lossless + k_scale + v_scale + 2 * one_codebook + 2 * codebook_meta
    full = theoretical_full_mib(frames) * 1024**2
    return {
        "k_indices_mib": k_indices / 1024**2,
        "v_indices_mib": v_indices / 1024**2,
        "k_lossless_indices_mib": k_lossless / 1024**2,
        "v_lossless_indices_mib": v_lossless / 1024**2,
        "k_scale_mib": k_scale / 1024**2,
        "v_scale_mib": v_scale / 1024**2,
        "k_codebook_int4_mib": one_codebook / 1024**2,
        "v_codebook_int4_mib": one_codebook / 1024**2,
        "codebook_meta_mib": 2 * codebook_meta / 1024**2,
        "total_before_lossless_mib": total_before / 1024**2,
        "total_after_lossless_mib": total_after / 1024**2,
        "ratio_before_lossless": full / total_before,
        "ratio_after_lossless": full / total_after,
    }


def summarize_lossless_memory(args: argparse.Namespace, lossless_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant_name, variant in VARIANTS.items():
        for family in ("offline_train_table", "online_oracle"):
            ratios = {}
            for tensor_name in ("K", "V"):
                vals = [
                    float(r["best_index_ratio"])
                    for r in lossless_rows
                    if r["variant"] == variant_name
                    and r["family"] == family
                    and r["tensor"] == tensor_name
                    and r["stage"] == "all"
                    and r["best_index_ratio"] not in (None, "")
                ]
                ratios[tensor_name] = vals[0] if vals else 1.0
            row = {
                "variant": variant_name,
                "storage": "int4",
                "family": family,
                "k_index_ratio": ratios["K"],
                "v_index_ratio": ratios["V"],
                **rvq_int4_components_mib(variant, args.frames_for_memory, ratios["K"], ratios["V"]),
            }
            out.append(row)
    return out


def run_workers(args: argparse.Namespace) -> None:
    variants = csv_list(args.variants)
    storages = csv_list(args.storages)
    splits = csv_list(args.splits)
    gpus = csv_list(args.gpus)
    tasks = []
    for variant in variants:
        for storage in storages:
            for tensor_name in ("K", "V"):
                for split in splits:
                    for prompt_id in prompt_ids(args.output_dir, split):
                        tasks.append((variant, storage, tensor_name, split, prompt_id))

    pending = list(tasks)
    running: list[tuple[subprocess.Popen, Path, tuple[str, str, str, str, int], str]] = []
    log_dir = args.output_dir / "logs" / "rvq_storage_eval"
    log_dir.mkdir(parents=True, exist_ok=True)

    while pending or running:
        while pending and len(running) < len(gpus):
            used_gpus = {gpu for _proc, _log_path, _task, gpu in running}
            free_gpus = [gpu for gpu in gpus if gpu not in used_gpus]
            if not free_gpus:
                break
            task = pending.pop(0)
            variant, storage, tensor_name, split, prompt_id = task
            out = partial_path(args, variant, storage, tensor_name, split, prompt_id)
            if out.exists() and not args.overwrite:
                print(f"[main-skip] {out}", flush=True)
                continue
            gpu = free_gpus[0]
            log_path = log_dir / f"{variant}_{storage}_{tensor_name}_{split}_p{prompt_id}.log"
            cmd = [
                str(args.worker_python),
                str(Path(__file__).resolve()),
                "--worker",
                "--output-dir",
                str(args.output_dir),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--query-sample",
                str(args.query_sample),
                "--key-sample",
                str(args.key_sample),
                "--topk",
                str(args.topk),
                "--seed",
                str(args.seed),
                "--mode-bits",
                str(args.mode_bits),
                "--worker-variant",
                variant,
                "--worker-storage",
                storage,
                "--worker-tensor",
                tensor_name,
                "--worker-split",
                split,
                "--worker-prompt-id",
                str(prompt_id),
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            log_f = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            running.append((proc, log_path, task, gpu))
            print(f"[main-start] gpu={gpu} task={task} log={log_path}", flush=True)
        time.sleep(5)
        still_running = []
        for proc, log_path, task, gpu in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, log_path, task, gpu))
            elif rc != 0:
                raise RuntimeError(f"Task {task} failed with rc={rc}; see {log_path}")
            else:
                print(f"[main-done] task={task}", flush=True)
        running = still_running


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    timings = []
    for path in sorted((args.output_dir / "metrics" / "rvq_storage_eval" / "partials").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows.extend(payload.get("rows", []))
        blocks.extend(payload.get("lossless_blocks", []))
        timing = dict(payload.get("config", {}))
        timing["wall_time_s"] = payload.get("wall_time_s")
        timing["partial"] = str(path)
        timings.append(timing)

    write_csv(args.output_dir / "metrics" / "rvq_storage_reconstruction_rows.csv", rows)
    rec = summarize_reconstruction(rows)
    write_csv(args.output_dir / "metrics" / "rvq_storage_reconstruction.csv", rec)
    write_csv(args.output_dir / "metrics" / "rvq_storage_eval_timing.csv", timings)

    lossless = summarize_lossless(args, blocks)
    write_csv(args.output_dir / "metrics" / "rvq_storage_int4_lossless.csv", lossless)
    memory = summarize_lossless_memory(args, lossless)
    write_csv(args.output_dir / "metrics" / "rvq_storage_int4_lossless_memory.csv", memory)

    print(args.output_dir / "metrics" / "rvq_storage_reconstruction.csv")
    print(args.output_dir / "metrics" / "rvq_storage_int4_lossless.csv")
    print(args.output_dir / "metrics" / "rvq_storage_int4_lossless_memory.csv")


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_main(args)
        return
    run_workers(args)
    aggregate(args)


if __name__ == "__main__":
    main()
