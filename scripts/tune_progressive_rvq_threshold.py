#!/usr/bin/env python3
"""Tune the residual-RMS threshold for progressive runtime RVQ."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.rvq import _codebook_level, _compute_dtype, _input_norm_mode, load_layer_codebooks_from_dir


def parse_csv_ints(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("list must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dump-root", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk/kv_dump"))
    p.add_argument("--k-codebook-dir", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk/sweep_k_l64c64_codebooks"))
    p.add_argument("--v-codebook-dir", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk/sweep_v_l64c64_codebooks"))
    p.add_argument("--split", default="test")
    p.add_argument("--prompts", default="0,1")
    p.add_argument("--layers", default="0,5,10,15,20,25,29")
    p.add_argument("--token-sample", type=int, default=1024)
    p.add_argument("--min-depth", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=32)
    p.add_argument("--target-average-depth", type=float, default=16.0)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compute-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--codebook-storage-dtype", default="int4", choices=["bf16", "int8", "int4", "int2"])
    p.add_argument("--output-json", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk/metrics/progressive_rvq_threshold_l8_l32.json"))
    p.add_argument("--output-csv", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test_21chunk/metrics/progressive_rvq_threshold_l8_l32.csv"))
    return p.parse_args()


def sample_dump_vectors(
    dump_root: Path,
    split: str,
    prompt_id: int,
    layer: int,
    tensor_name: str,
    token_sample: int,
    seed: int,
) -> torch.Tensor:
    path = dump_root / split / str(prompt_id) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x = payload[tensor_name]
    if not torch.is_tensor(x) or x.ndim != 4:
        raise ValueError(f"Expected {path}:{tensor_name} to have shape [T,H,S,D], got {type(x)}")
    frames, heads, spatial, dim = x.shape
    flat = x.permute(1, 0, 2, 3).reshape(heads, frames * spatial, dim).contiguous()
    sample = min(int(token_sample), int(flat.shape[1]))
    generator = torch.Generator(device="cpu").manual_seed(int(seed) + prompt_id * 1009 + layer * 9176)
    indices = torch.randperm(int(flat.shape[1]), generator=generator)[:sample]
    return flat[:, indices, :].contiguous()


@torch.no_grad()
def residual_errors_by_level(
    x_hnd: torch.Tensor,
    *,
    codebook_dir: Path,
    layer: int,
    min_depth: int,
    max_depth: int,
    batch_size: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    codebook_storage_dtype: str,
) -> torch.Tensor:
    codebooks = load_layer_codebooks_from_dir(
        codebook_dir,
        int(layer),
        device,
        compute_dtype,
        codebook_storage_dtype=codebook_storage_dtype,
    )
    x_hnd = x_hnd.to(device=device)
    heads, tokens, dim = x_hnd.shape
    if _input_norm_mode(codebooks) == "none":
        residual = x_hnd.float().to(compute_dtype)
    else:
        scale = torch.sqrt(torch.mean(x_hnd.float() * x_hnd.float(), dim=-1, keepdim=True)).clamp_min(1e-6)
        residual = (x_hnd.float() / scale).to(compute_dtype)
    errors: list[torch.Tensor] = []
    for level in range(int(max_depth)):
        centroids = _codebook_level(codebooks, level, compute_dtype)
        centroid_sq = (centroids.float() * centroids.float()).sum(dim=-1)
        next_residual = torch.empty_like(residual)
        for start in range(0, tokens, int(batch_size)):
            end = min(start + int(batch_size), tokens)
            rb = residual[:, start:end, :]
            rb_sq = (rb.float() * rb.float()).sum(dim=-1, keepdim=True)
            dots = torch.bmm(rb, centroids.transpose(1, 2)).float()
            dist = rb_sq + centroid_sq[:, None, :] - 2.0 * dots
            ids = torch.argmin(dist, dim=-1)
            assigned = torch.gather(centroids, 1, ids[..., None].expand(-1, -1, dim))
            next_residual[:, start:end, :] = rb - assigned
        residual = next_residual
        if level + 1 >= int(min_depth):
            err = torch.sqrt(torch.mean(residual.float() * residual.float(), dim=-1))
            errors.append(err.detach().cpu())
    return torch.stack(errors, dim=0).contiguous()


def depths_for_threshold(errors: torch.Tensor, threshold: float, min_depth: int, max_depth: int) -> torch.Tensor:
    depths = torch.full(errors.shape[1:], int(max_depth), dtype=torch.int16)
    done = torch.zeros(errors.shape[1:], dtype=torch.bool)
    for offset in range(errors.shape[0]):
        level_depth = int(min_depth) + offset
        hit = (~done) & (errors[offset] <= float(threshold))
        depths[hit] = level_depth
        done |= hit
    return depths


def summarize(errors_by_tensor: dict[str, list[torch.Tensor]], threshold: float, min_depth: int, max_depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {"threshold": float(threshold)}
    all_depths = []
    for tensor_name, chunks in errors_by_tensor.items():
        depths = torch.cat(
            [depths_for_threshold(errors, threshold, min_depth, max_depth).reshape(-1) for errors in chunks],
            dim=0,
        )
        all_depths.append(depths)
        out[f"{tensor_name.lower()}_avg_depth"] = float(depths.float().mean().item())
        out[f"{tensor_name.lower()}_max_depth_hit_rate"] = float((depths >= int(max_depth)).float().mean().item())
        for q in (0.5, 0.9, 0.99):
            out[f"{tensor_name.lower()}_p{int(q * 100)}_depth"] = float(torch.quantile(depths.float(), q).item())
    combined = torch.cat(all_depths, dim=0)
    out["avg_depth"] = float(combined.float().mean().item())
    out["max_depth_hit_rate"] = float((combined >= int(max_depth)).float().mean().item())
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    prompts = parse_csv_ints(args.prompts)
    layers = parse_csv_ints(args.layers)
    if args.min_depth <= 0 or args.min_depth > args.max_depth:
        raise ValueError("--min-depth must be positive and <= --max-depth")
    device = torch.device(args.device)
    compute_dtype = _compute_dtype(args.compute_dtype, device)

    errors_by_tensor: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    for prompt_id in prompts:
        for layer in layers:
            for tensor_name, codebook_dir in (("K", args.k_codebook_dir), ("V", args.v_codebook_dir)):
                x = sample_dump_vectors(
                    args.dump_root,
                    args.split,
                    prompt_id,
                    layer,
                    tensor_name,
                    args.token_sample,
                    args.seed,
                )
                errors = residual_errors_by_level(
                    x,
                    codebook_dir=codebook_dir,
                    layer=layer,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    batch_size=args.batch_size,
                    device=device,
                    compute_dtype=compute_dtype,
                    codebook_storage_dtype=args.codebook_storage_dtype,
                )
                errors_by_tensor[tensor_name].append(errors)
                print(
                    f"[sample] tensor={tensor_name} prompt={prompt_id} layer={layer} "
                    f"vectors={x.shape[0] * x.shape[1]}",
                    flush=True,
                )

    max_error = max(float(torch.max(errors).item()) for chunks in errors_by_tensor.values() for errors in chunks)
    lo, hi = 0.0, max_error
    for _ in range(32):
        mid = (lo + hi) / 2.0
        summary = summarize(errors_by_tensor, mid, args.min_depth, args.max_depth)
        if summary["avg_depth"] <= float(args.target_average_depth):
            hi = mid
        else:
            lo = mid
    tuned = summarize(errors_by_tensor, hi, args.min_depth, args.max_depth)
    tuned.update(
        {
            "min_depth": int(args.min_depth),
            "max_depth": int(args.max_depth),
            "target_average_depth": float(args.target_average_depth),
            "split": args.split,
            "prompts": prompts,
            "layers": layers,
            "token_sample": int(args.token_sample),
            "k_codebook_dir": str(args.k_codebook_dir),
            "v_codebook_dir": str(args.v_codebook_dir),
        }
    )

    rows = []
    for threshold in sorted({0.0, hi * 0.9, hi, hi * 1.1, max_error}):
        row = summarize(errors_by_tensor, threshold, args.min_depth, args.max_depth)
        row["selected"] = abs(threshold - hi) <= max(1e-12, hi * 1e-9)
        rows.append(row)
    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(tuned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(tuned, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
