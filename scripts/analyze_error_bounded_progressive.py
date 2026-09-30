#!/usr/bin/env python3
"""Analyze fixed L8 RVQ against token-wise error-bounded progressive RVQ.

The analysis replays the exact non-normalized INT4 codebooks used by the Wan5B
11-chunk video benchmark.  It deliberately writes into a new result directory
and never modifies the source KV dumps or benchmark outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.rvq import (  # noqa: E402
    _codebook_level,
    _compute_dtype,
    load_layer_codebooks_from_dir,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


LAYERS = 30
HEADS = 24
DIM = 128
CODEBOOK_SIZE = 64
FIXED_DEPTH = 8
PROGRESSIVE_MIN_DEPTH = 4
PROGRESSIVE_MAX_DEPTH = 16
DEFAULT_THRESHOLD = 0.6850449743634424
MIB = 1024**2

DEFAULT_DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "results/long_video/no_scale_variants_11chunk_10prompt_20260707"


def parse_ints(raw: str, upper: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(upper))
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value < 0 or value >= upper for value in values):
        raise ValueError(f"Expected values in [0, {upper}), got {raw!r}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", default="all")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--token-sample", type=int, default=1024, help="Tokens sampled per prompt/layer/head")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--smoke", action="store_true", help="Run codec agreement checks and stop")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_hash(paths: list[Path], root: Path, content: bool) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(str(path.stat().st_size).encode())
        if content:
            digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sample_prompt_layer(
    path: Path,
    prompt_id: int,
    layer: int,
    token_sample: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out: dict[str, torch.Tensor] = {}
    for tensor_name in ("K", "V"):
        tensor = payload[tensor_name]
        if tuple(tensor.shape[1:]) != (HEADS, 880, DIM):
            raise ValueError(f"Unexpected {path}:{tensor_name} shape {tuple(tensor.shape)}")
        flat = tensor.permute(1, 0, 2, 3).reshape(HEADS, -1, DIM)
        take = min(int(token_sample), int(flat.shape[1]))
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + prompt_id * 1_000_003 + layer * 10_007
        )
        indices = torch.randperm(int(flat.shape[1]), generator=generator)[:take]
        out[tensor_name] = flat[:, indices, :].contiguous()
    del payload
    return out


def sample_layer(
    dump_root: Path,
    prompts: list[int],
    layer: int,
    token_sample: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {"K": [], "V": []}
    for prompt_id in prompts:
        path = dump_root / "test" / str(prompt_id) / f"layer_{layer}.pt"
        sampled = sample_prompt_layer(path, prompt_id, layer, token_sample, seed)
        for tensor_name in chunks:
            chunks[tensor_name].append(sampled[tensor_name])
    return {name: torch.cat(values, dim=1) for name, values in chunks.items()}


@torch.no_grad()
def assign_stage(
    residual: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, tokens, dim = residual.shape
    centroid_sq = (centroids.float() * centroids.float()).sum(dim=-1)
    next_residual = torch.empty_like(residual)
    all_assigned = torch.empty_like(residual)
    for start in range(0, tokens, batch_size):
        end = min(tokens, start + batch_size)
        block = residual[:, start:end]
        distance = (
            (block.float() * block.float()).sum(dim=-1, keepdim=True)
            + centroid_sq[:, None, :]
            - 2.0 * torch.bmm(block, centroids.transpose(1, 2)).float()
        )
        ids = torch.argmin(distance, dim=-1)
        assigned = torch.gather(centroids, 1, ids[..., None].expand(-1, -1, dim))
        next_residual[:, start:end] = block - assigned
        all_assigned[:, start:end] = assigned
    return next_residual, all_assigned


def vector_metrics(residual: torch.Tensor, original: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_mse = (residual.float() * residual.float()).mean(dim=-1)
    input_mse = (original.float() * original.float()).mean(dim=-1).clamp_min(1e-12)
    relative_mse = raw_mse / input_mse
    residual_rms = torch.sqrt(raw_mse)
    return raw_mse, relative_mse, residual_rms


def stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    quantiles = np.quantile(values, [0.10, 0.50, 0.90, 0.99])
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p10": float(quantiles[0]),
        f"{prefix}_p50": float(quantiles[1]),
        f"{prefix}_p90": float(quantiles[2]),
        f"{prefix}_p99": float(quantiles[3]),
    }


def append_head_rows(
    rows: list[dict[str, Any]],
    distributions: dict[tuple[str, str, str], list[np.ndarray]],
    *,
    method: str,
    tensor_name: str,
    layer: int,
    raw_mse: torch.Tensor,
    relative_mse: torch.Tensor,
    residual_rms: torch.Tensor,
    depths: torch.Tensor | None = None,
) -> None:
    raw_np = raw_mse.detach().cpu().numpy()
    rel_np = relative_mse.detach().cpu().numpy()
    rms_np = residual_rms.detach().cpu().numpy()
    distributions[(method, tensor_name, "raw_mse")].append(raw_np.reshape(-1))
    distributions[(method, tensor_name, "relative_mse")].append(rel_np.reshape(-1))
    distributions[(method, tensor_name, "residual_rms")].append(rms_np.reshape(-1))
    depth_np = depths.detach().cpu().numpy() if depths is not None else None
    for head in range(raw_np.shape[0]):
        row: dict[str, Any] = {
            "method": method,
            "tensor": tensor_name,
            "layer": layer,
            "head": head,
            "count": int(raw_np.shape[1]),
        }
        row.update(stats(raw_np[head], "raw_mse"))
        row.update(stats(rel_np[head], "relative_mse"))
        row.update(stats(rms_np[head], "residual_rms"))
        if depth_np is not None:
            row.update(stats(depth_np[head].astype(np.float64), "depth"))
            row["max_depth_hit_rate"] = float(np.mean(depth_np[head] >= PROGRESSIVE_MAX_DEPTH))
        rows.append(row)


@torch.no_grad()
def analyze_fixed(
    original: torch.Tensor,
    *,
    layer: int,
    codebook_dir: Path,
    device: torch.device,
    compute_dtype: torch.dtype,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    codebooks = load_layer_codebooks_from_dir(
        codebook_dir, layer, device, compute_dtype, codebook_storage_dtype="int4", num_heads=HEADS
    )
    residual = original.to(device=device, dtype=compute_dtype)
    reconstruction = torch.zeros_like(residual)
    for stage in range(FIXED_DEPTH):
        residual, assigned = assign_stage(
            residual, _codebook_level(codebooks, stage, compute_dtype), batch_size
        )
        reconstruction += assigned.to(compute_dtype)
    decoded_error = original.to(device=device).float() - reconstruction.to(original.dtype).float()
    return vector_metrics(decoded_error, original.to(device=device))


@torch.no_grad()
def analyze_progressive(
    original: torch.Tensor,
    *,
    layer: int,
    codebook_dir: Path,
    threshold: float,
    device: torch.device,
    compute_dtype: torch.dtype,
    batch_size: int,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
    list[tuple[float, int]],
]:
    codebooks = load_layer_codebooks_from_dir(
        codebook_dir, layer, device, compute_dtype, codebook_storage_dtype="int4", num_heads=HEADS
    )
    original_device = original.to(device=device)
    residual = original_device.to(dtype=compute_dtype)
    reconstruction = torch.zeros_like(residual)
    heads, tokens, _ = residual.shape
    depths = torch.full((heads, tokens), PROGRESSIVE_MAX_DEPTH, device=device, dtype=torch.uint8)
    selected_raw = torch.empty((heads, tokens), device=device, dtype=torch.float32)
    active = torch.ones((heads, tokens), device=device, dtype=torch.bool)
    stage_sums: list[tuple[float, int]] = [
        (float((original_device.float() * original_device.float()).sum().item()), int(residual.numel()))
    ]
    for stage in range(1, PROGRESSIVE_MAX_DEPTH + 1):
        residual, assigned = assign_stage(
            residual, _codebook_level(codebooks, stage - 1, compute_dtype), batch_size
        )
        reconstruction += assigned.to(compute_dtype)
        decoded_error = original_device.float() - reconstruction.to(original.dtype).float()
        decoded_raw_mse = (decoded_error * decoded_error).mean(dim=-1)
        threshold_raw_mse = (residual.float() * residual.float()).mean(dim=-1)
        stage_sums.append(
            (float(decoded_raw_mse.sum().item()) * DIM, int(decoded_raw_mse.numel()) * DIM)
        )
        if stage >= PROGRESSIVE_MIN_DEPTH:
            done = active & (torch.sqrt(threshold_raw_mse) <= float(threshold))
            if stage == PROGRESSIVE_MAX_DEPTH:
                done = active
            depths[done] = stage
            selected_raw[done] = decoded_raw_mse[done]
            active &= ~done
    input_mse = (original_device.float() * original_device.float()).mean(dim=-1).clamp_min(1e-12)
    selected_relative = selected_raw / input_mse
    selected_rms = torch.sqrt(selected_raw)
    return (selected_raw, selected_relative, selected_rms), depths, stage_sums


def codebook_storage(codebook_dir: Path, levels: int) -> dict[str, int]:
    files = sorted(codebook_dir.glob("layer_*_head_*.pt"))
    if len(files) != LAYERS * HEADS:
        raise ValueError(f"Expected {LAYERS * HEADS} files in {codebook_dir}, found {len(files)}")
    value_bytes = 0
    scale_bytes = 0
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        codebooks = payload["codebooks"]
        active_levels = min(levels, int(codebooks.shape[0]))
        value_bytes += math.ceil(active_levels * codebooks.shape[1] * codebooks.shape[2] / 2)
        scale_bytes += active_levels * codebooks.shape[1] * 2
    return {"int4_value_bytes": value_bytes, "fp16_scale_bytes": scale_bytes}


def read_formal_rows(benchmark_root: Path) -> dict[str, dict[str, str]]:
    with (benchmark_root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        return {row["setting_name"]: row for row in csv.DictReader(handle)}


def memory_breakdown(data_root: Path, benchmark_root: Path) -> dict[str, Any]:
    formal = read_formal_rows(benchmark_root)
    fixed = formal["nonorm_l8c64_int4"]
    progressive = formal["nonorm_progressive_token_l4_l16_avg7"]
    fixed_total = int(fixed["kv_stored_bytes"])
    progressive_total = int(progressive["kv_stored_bytes"])
    progressive_depth = float(progressive["rvq_actual_avg_depth"])

    fixed_cb_k = codebook_storage(data_root / "nonorm_k_l8c64_codebooks", FIXED_DEPTH)
    fixed_cb_v = codebook_storage(data_root / "nonorm_v_l8c64_codebooks", FIXED_DEPTH)
    prog_cb_k = codebook_storage(data_root / "nonorm_k_l16c64_codebooks", PROGRESSIVE_MAX_DEPTH)
    prog_cb_v = codebook_storage(data_root / "nonorm_v_l16c64_codebooks", PROGRESSIVE_MAX_DEPTH)

    def merge_codebooks(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
        return {key: a[key] + b[key] for key in a}

    fixed_cb = merge_codebooks(fixed_cb_k, fixed_cb_v)
    prog_cb = merge_codebooks(prog_cb_k, prog_cb_v)
    fixed_cb_total = sum(fixed_cb.values())
    prog_cb_total = sum(prog_cb.values())
    fixed_logical_index = fixed_total - fixed_cb_total
    vector_count = fixed_logical_index / (FIXED_DEPTH * 6 / 8)
    progressive_logical_index = vector_count * progressive_depth * 6 / 8
    progressive_metadata = progressive_total - prog_cb_total - progressive_logical_index
    active_depth = int(round(vector_count))
    offsets = progressive_metadata - active_depth

    schemes: dict[str, Any] = {}
    for name, depth, cb, logical_index, metadata, total in (
        ("fixed_l8c64_int4", FIXED_DEPTH, fixed_cb, fixed_logical_index, 0.0, fixed_total),
        (
            "error_bounded_l4_l16_int4",
            progressive_depth,
            prog_cb,
            progressive_logical_index,
            progressive_metadata,
            progressive_total,
        ),
    ):
        physical_index = vector_count * depth
        physical_total = physical_index + sum(cb.values()) + metadata
        schemes[name] = {
            "average_depth": depth,
            "vector_count": vector_count,
            "logical_6bit": {
                "index_bytes": logical_index,
                "codebook_int4_value_bytes": cb["int4_value_bytes"],
                "codebook_fp16_scale_bytes": cb["fp16_scale_bytes"],
                "active_depth_bytes": active_depth if metadata else 0,
                "extra_level_offset_bytes": offsets if metadata else 0,
                "total_bytes": total,
            },
            "runtime_uint8": {
                "index_bytes": physical_index,
                "codebook_int4_value_bytes": cb["int4_value_bytes"],
                "codebook_fp16_scale_bytes": cb["fp16_scale_bytes"],
                "active_depth_bytes": active_depth if metadata else 0,
                "extra_level_offset_bytes": offsets if metadata else 0,
                "total_bytes": physical_total,
            },
        }
    component_keys = (
        "index_bytes", "codebook_int4_value_bytes", "codebook_fp16_scale_bytes",
        "active_depth_bytes", "extra_level_offset_bytes",
    )
    for name, scheme in schemes.items():
        for accounting in ("logical_6bit", "runtime_uint8"):
            item = scheme[accounting]
            component_sum = sum(float(item[key]) for key in component_keys)
            if not math.isclose(component_sum, float(item["total_bytes"]), abs_tol=1e-4):
                raise AssertionError(
                    f"Memory components do not sum to total for {name}/{accounting}: "
                    f"{component_sum} != {item['total_bytes']}"
                )
    return {
        "accounting": {
            "logical_index_bits": 6,
            "runtime_index_bits": 8,
            "codebook_value_bits": 4,
            "codebook_scale_bits": 16,
            "formal_benchmark_uses_logical_6bit_indices": True,
        },
        "schemes": schemes,
    }


def memory_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scheme, scheme_payload in payload["schemes"].items():
        for accounting in ("logical_6bit", "runtime_uint8"):
            item = scheme_payload[accounting]
            row = {"scheme": scheme, "accounting": accounting, "average_depth": scheme_payload["average_depth"]}
            for key, value in item.items():
                row[key] = value
                row[key.replace("_bytes", "_mib")] = float(value) / MIB
            rows.append(row)
    return rows


def formal_comparison(benchmark_root: Path) -> dict[str, Any]:
    analysis = json.loads((REPO_ROOT / "results/analysis/progressive_rvq_effectiveness.json").read_text())
    target = next(item for item in analysis["comparisons"] if item["label"] == "no-norm token progressive vs L8 all chunks")
    return {"fixed": target["base"], "error_bounded": target["progressive"], "delta": target["delta"]}


def codec_smoke(
    data_root: Path,
    device: torch.device,
    compute_dtype: torch.dtype,
    threshold: float,
    batch_size: int,
) -> dict[str, Any]:
    sampled = sample_prompt_layer(data_root / "kv_dump/test/0/layer_0.pt", 0, 0, 64, 42)
    results: dict[str, Any] = {}
    for tensor_name, directory in (
        ("K", data_root / "nonorm_k_l8c64_codebooks"),
        ("V", data_root / "nonorm_v_l8c64_codebooks"),
    ):
        original = sampled[tensor_name].to(device)
        ours = analyze_fixed(
            sampled[tensor_name], layer=0, codebook_dir=directory, device=device,
            compute_dtype=compute_dtype, batch_size=batch_size
        )[0]
        packed = rvq_encode_tensor_cache(
            original[None].reshape(1, HEADS, -1, DIM), layer_idx=0, codebook_dir=directory,
            batch_size=batch_size, compute_dtype=str(compute_dtype).replace("torch.", ""),
            codebook_storage_dtype="int4", scale_storage_dtype="uint4"
        )
        decoded = rvq_dequantize_tensor_cache(packed, original.dtype).reshape(HEADS, -1, DIM)
        codec_mse = ((decoded.float() - original.float()) ** 2).mean(dim=-1)
        max_abs = float((codec_mse - ours).abs().max().item())
        if max_abs > 2e-4:
            raise AssertionError(f"{tensor_name} fixed replay/codec MSE mismatch: {max_abs}")
        progressive_directory = data_root / f"nonorm_{tensor_name.lower()}_l16c64_codebooks"
        progressive_packed = rvq_encode_tensor_cache(
            original[None].reshape(1, HEADS, -1, DIM), layer_idx=0,
            codebook_dir=progressive_directory, batch_size=batch_size,
            compute_dtype=str(compute_dtype).replace("torch.", ""),
            codebook_storage_dtype="int4", scale_storage_dtype="uint4",
            progressive=True, progressive_min_depth=PROGRESSIVE_MIN_DEPTH,
            progressive_max_depth=PROGRESSIVE_MAX_DEPTH,
            progressive_error_threshold=threshold, progressive_granularity="token",
        )
        progressive_metrics, depths, _ = analyze_progressive(
            sampled[tensor_name], layer=0,
            codebook_dir=progressive_directory,
            threshold=threshold, device=device, compute_dtype=compute_dtype, batch_size=batch_size,
        )
        progressive_decoded = rvq_dequantize_tensor_cache(
            progressive_packed, original.dtype
        ).reshape(HEADS, -1, DIM)
        progressive_codec_mse = (
            (progressive_decoded.float() - original.float()) ** 2
        ).mean(dim=-1)
        progressive_mse_max_abs = float(
            (progressive_codec_mse - progressive_metrics[0]).abs().max().item()
        )
        packed_depth = progressive_packed["active_depth"].reshape(HEADS, -1)
        depth_mismatch = int((packed_depth != depths).sum().item())
        if progressive_mse_max_abs > 2e-4 or depth_mismatch:
            raise AssertionError(
                f"{tensor_name} progressive replay/codec mismatch: "
                f"mse={progressive_mse_max_abs} depths={depth_mismatch}"
            )
        results[tensor_name] = {
            "fixed_codec_mse_max_abs_difference": max_abs,
            "progressive_codec_mse_max_abs_difference": progressive_mse_max_abs,
            "progressive_depth_mismatch_count": depth_mismatch,
            "progressive_average_depth": float(depths.float().mean().item()),
            "progressive_raw_mse_mean": float(progressive_metrics[0].mean().item()),
            "packed_state_audit": {
                "vectors": int(original.shape[0] * original.shape[1]),
                "fixed_uint8_index_bytes": int(packed["indices"].numel() * packed["indices"].element_size()),
                "progressive_prefix_index_bytes": int(
                    progressive_packed["indices"].numel() * progressive_packed["indices"].element_size()
                ),
                "progressive_extra_index_bytes": int(
                    progressive_packed["extra_indices"].numel()
                    * progressive_packed["extra_indices"].element_size()
                ),
                "progressive_active_depth_bytes": int(
                    progressive_packed["active_depth"].numel()
                    * progressive_packed["active_depth"].element_size()
                ),
                "progressive_extra_offset_bytes": int(
                    progressive_packed["extra_level_offsets"].numel()
                    * progressive_packed["extra_level_offsets"].element_size()
                ),
            },
        }
    return results


def build_stage_rows(stage_totals: dict[str, list[list[float]]], fixed_stage8: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_tensor_stage_index_mib = 580.078125 / FIXED_DEPTH / 2
    per_tensor_stage_codebook_mib = 46.40625 / FIXED_DEPTH / 2
    added_payload = per_tensor_stage_index_mib + per_tensor_stage_codebook_mib
    for stage in range(PROGRESSIVE_MAX_DEPTH + 1):
        row: dict[str, Any] = {
            "stage": stage,
            "logical_index_mib_per_tensor_full_stage": per_tensor_stage_index_mib if stage else 0.0,
            "codebook_mib_per_tensor_stage": per_tensor_stage_codebook_mib if stage else 0.0,
        }
        for tensor_name in ("K", "V"):
            sum_sq, count = stage_totals[tensor_name][stage]
            mse = sum_sq / count
            row[f"{tensor_name.lower()}_raw_mse_after"] = mse
            if stage:
                prev_sum, prev_count = stage_totals[tensor_name][stage - 1]
                previous = prev_sum / prev_count
                gain = previous - mse
                row[f"{tensor_name.lower()}_additional_mse_reduction"] = gain
                row[f"{tensor_name.lower()}_relative_reduction_pct"] = gain / previous * 100.0
                row[f"{tensor_name.lower()}_reduction_per_added_payload_mib"] = gain / added_payload
            else:
                row[f"{tensor_name.lower()}_additional_mse_reduction"] = None
                row[f"{tensor_name.lower()}_relative_reduction_pct"] = None
                row[f"{tensor_name.lower()}_reduction_per_added_payload_mib"] = None
            if stage == FIXED_DEPTH:
                row[f"{tensor_name.lower()}_independent_fixed_l8_raw_mse"] = (
                    fixed_stage8[tensor_name][0] / fixed_stage8[tensor_name][1]
                )
        rows.append(row)
    return rows


def plot_outputs(
    output_dir: Path,
    distributions: dict[tuple[str, str, str], list[np.ndarray]],
    error_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    memory: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"fixed_l8c64_int4": "#2563eb", "error_bounded_l4_l16_int4": "#dc2626"}

    def save(fig: Any, stem: str) -> None:
        fig.savefig(figures / f"{stem}.svg")
        fig.savefig(figures / f"{stem}.png", dpi=180)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, tensor_name in zip(axes, ("K", "V")):
        for method in colors:
            values = np.concatenate(distributions[(method, tensor_name, "relative_mse")])
            cap = np.quantile(values, 0.99)
            axis.hist(np.minimum(values, cap), bins=80, density=True, alpha=0.45, color=colors[method], label=method)
        axis.set_title(f"{tensor_name} relative MSE distribution")
        axis.set_xlabel("relative squared error (clipped at P99)")
        axis.set_ylabel("density")
        axis.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "error_histogram")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, tensor_name in zip(axes, ("K", "V")):
        for method in colors:
            values = np.sort(np.concatenate(distributions[(method, tensor_name, "relative_mse")]))
            sample = np.linspace(0, len(values) - 1, min(10000, len(values))).astype(int)
            axis.plot(values[sample], (sample + 1) / len(values), color=colors[method], label=method)
        axis.set_title(f"{tensor_name} relative MSE CDF")
        axis.set_xlabel("relative squared error")
        axis.set_ylabel("CDF")
        axis.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "error_cdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for row_idx, tensor_name in enumerate(("K", "V")):
        for col_idx, method in enumerate(colors):
            matrix = np.full((LAYERS, HEADS), np.nan)
            for row in error_rows:
                if row["tensor"] == tensor_name and row["method"] == method:
                    matrix[int(row["layer"]), int(row["head"])] = float(row["relative_mse_mean"])
            image = axes[row_idx, col_idx].imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
            axes[row_idx, col_idx].set_title(f"{tensor_name} {method}")
            axes[row_idx, col_idx].set_xlabel("head")
            axes[row_idx, col_idx].set_ylabel("transformer layer")
            fig.colorbar(image, ax=axes[row_idx, col_idx], fraction=0.046)
    fig.tight_layout()
    save(fig, "layer_head_error_heatmap")
    plt.close(fig)

    stages = np.array([row["stage"] for row in stage_rows])
    fig, axis = plt.subplots(figsize=(7, 4))
    for tensor_name, color in (("K", "#2563eb"), ("V", "#dc2626")):
        axis.plot(stages, [row[f"{tensor_name.lower()}_raw_mse_after"] for row in stage_rows], marker="o", color=color, label=tensor_name)
    axis.set_yscale("log")
    axis.set_xlabel("RVQ stage")
    axis.set_ylabel("raw reconstruction MSE")
    axis.set_title("Residual error by RVQ stage")
    axis.legend()
    fig.tight_layout()
    save(fig, "stage_error_curve")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for tensor_name, color in (("K", "#2563eb"), ("V", "#dc2626")):
        axis.plot(stages[1:], [row[f"{tensor_name.lower()}_additional_mse_reduction"] for row in stage_rows[1:]], marker="o", color=color, label=tensor_name)
    axis.set_yscale("log")
    axis.set_xlabel("RVQ stage")
    axis.set_ylabel("additional MSE reduction")
    axis.set_title("Diminishing marginal gain")
    axis.legend()
    fig.tight_layout()
    save(fig, "stage_marginal_gain")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    component_keys = (
        "index_bytes", "codebook_int4_value_bytes", "codebook_fp16_scale_bytes",
        "active_depth_bytes", "extra_level_offset_bytes",
    )
    labels = ("index", "INT4 values", "FP16 CB scales", "depth", "offsets")
    component_colors = ("#2563eb", "#16a34a", "#65a30d", "#dc2626", "#f59e0b")
    for axis, accounting in zip(axes, ("logical_6bit", "runtime_uint8")):
        bottoms = np.zeros(2)
        names = list(memory["schemes"])
        display_names = ["Fixed L8", "Error-Bounded"]
        for key, label, color in zip(component_keys, labels, component_colors):
            values = np.array([memory["schemes"][name][accounting][key] / MIB for name in names])
            axis.bar(display_names, values, bottom=bottoms, label=label, color=color)
            bottoms += values
        axis.set_title(accounting)
        axis.set_ylabel("MiB")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "memory_breakdown")
    plt.close(fig)


def render_report(
    output_dir: Path,
    formal: dict[str, Any],
    global_stats: list[dict[str, Any]],
    depth_summary: dict[str, Any],
    stage_rows: list[dict[str, Any]],
    memory: dict[str, Any],
    threshold: float,
) -> None:
    fixed = formal["fixed"]
    progressive = formal["error_bounded"]
    global_lookup = {(row["method"], row["tensor"]): row for row in global_stats}
    fixed_memory = memory["schemes"]["fixed_l8c64_int4"]
    prog_memory = memory["schemes"]["error_bounded_l4_l16_int4"]
    stage_lines = [
        "| Stage | K MSE | K extra reduction | K reduction % | V MSE | V extra reduction | V reduction % |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stage_rows[1:]:
        stage_lines.append(
            f"| {row['stage']} | {row['k_raw_mse_after']:.6g} | {row['k_additional_mse_reduction']:.6g} | "
            f"{row['k_relative_reduction_pct']:.2f}% | {row['v_raw_mse_after']:.6g} | "
            f"{row['v_additional_mse_reduction']:.6g} | {row['v_relative_reduction_pct']:.2f}% |"
        )

    memory_lines = [
        "| Accounting | Scheme | Index MiB | CB INT4 MiB | CB scale MiB | Depth MiB | Offset MiB | Total MiB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for accounting in ("logical_6bit", "runtime_uint8"):
        for label, payload in (("Fixed L8C64", fixed_memory), ("Error-Bounded L4-L16", prog_memory)):
            item = payload[accounting]
            memory_lines.append(
                f"| {accounting} | {label} | {item['index_bytes']/MIB:.2f} | "
                f"{item['codebook_int4_value_bytes']/MIB:.2f} | {item['codebook_fp16_scale_bytes']/MIB:.2f} | "
                f"{item['active_depth_bytes']/MIB:.2f} | {item['extra_level_offset_bytes']/MIB:.2f} | "
                f"{item['total_bytes']/MIB:.2f} |"
            )

    error_lines = [
        "| Method | Tensor | Raw MSE mean | Relative MSE mean | Relative P50 | Relative P90 | Relative P99 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("fixed_l8c64_int4", "error_bounded_l4_l16_int4"):
        for tensor_name in ("K", "V"):
            row = global_lookup[(method, tensor_name)]
            error_lines.append(
                f"| {method} | {tensor_name} | {row['raw_mse_mean']:.6g} | {row['relative_mse_mean']:.6g} | "
                f"{row['relative_mse_p50']:.6g} | {row['relative_mse_p90']:.6g} | {row['relative_mse_p99']:.6g} |"
            )

    text = f"""# Error-Bounded Progressive Encoding 为什么无效

## 实验范围

Wan 5B，11 generation chunks，10个MovieGen test prompts，seed 42。Fixed使用nonorm L8C64 codebook；Error-Bounded使用nonorm L16C64 codebook，逐token residual-RMS threshold={threshold:.6f}，深度范围L4-L16。所有codebook运行时按INT4 values与FP16 centroid scales统计。

## 真实推理结果

| Setting | Avg depth | PSNR | SSIM | LPIPS | VBench | Encode s | Decode s | Total s | KV MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed L8C64 INT4 | 8 | {fixed['psnr']:.3f} | {fixed['ssim']:.4f} | {fixed['lpips']:.4f} | {fixed['vbench_avg']:.4f} | {fixed['quant_s']:.2f} | {fixed['dequant_s']:.2f} | {fixed['total_s']:.2f} | {fixed['kv_mib']:.2f} |
| Error-Bounded L4-L16 | {progressive['avg_depth']:.3f} | {progressive['psnr']:.3f} | {progressive['ssim']:.4f} | {progressive['lpips']:.4f} | {progressive['vbench_avg']:.4f} | {progressive['quant_s']:.2f} | {progressive['dequant_s']:.2f} | {progressive['total_s']:.2f} | {progressive['kv_mib']:.2f} |

Error-Bounded只带来PSNR +{progressive['psnr']-fixed['psnr']:.3f}、VBench +{progressive['vbench_avg']-fixed['vbench_avg']:.4f}，但KV增加{progressive['kv_mib']-fixed['kv_mib']:.10g} MiB，encode增加{progressive['quant_s']-fixed['quant_s']:.2f}s，decode增加{progressive['dequant_s']-fixed['dequant_s']:.2f}s，端到端增加{progressive['total_s']-fixed['total_s']:.2f}s。

## Compression Reconstruction Error Distribution

{chr(10).join(error_lines)}

统一offline replay中，Error-Bounded的K平均深度为{depth_summary['K']['mean']:.3f}，V为{depth_summary['V']['mean']:.3f}；K/V max-depth hit rate分别为{depth_summary['K']['max_depth_hit_rate']:.2%}和{depth_summary['V']['max_depth_hit_rate']:.2%}。同一绝对RMS threshold给K和V分配了明显不同的深度预算；正式10-prompt推理的K/V合并平均深度为{progressive['avg_depth']:.3f}。

## 每个RVQ Stage的边际收益

{chr(10).join(stage_lines)}

Stage 8的L16 prefix与独立训练L8结果在CSV中并列报告。后续stage仍降低重建MSE，但每增加一个stage获得的绝对误差下降持续缩小；variable-depth表示同时引入depth map和不规则extra-index路径，因此这些尾部收益不足以补偿布局成本。

## KV Memory Breakdown

{chr(10).join(memory_lines)}

Benchmark主表采用C64理想6-bit index口径；当前runtime index tensor实际为uint8。逻辑口径下，Error-Bounded相对Fixed仅减少{(fixed_memory['logical_6bit']['index_bytes']-prog_memory['logical_6bit']['index_bytes'])/MIB:.2f} MiB index，却增加{(sum((prog_memory['logical_6bit'][k] for k in ('codebook_int4_value_bytes','codebook_fp16_scale_bytes')))-sum((fixed_memory['logical_6bit'][k] for k in ('codebook_int4_value_bytes','codebook_fp16_scale_bytes'))))/MIB:.2f} MiB codebook和{(prog_memory['logical_6bit']['active_depth_bytes']+prog_memory['logical_6bit']['extra_level_offset_bytes'])/MIB:.2f} MiB metadata，总KV反而增加{(prog_memory['logical_6bit']['total_bytes']-fixed_memory['logical_6bit']['total_bytes'])/MIB:.2f} MiB。

## 结论

当前Error-Bounded Progressive Encoding没有形成有效trade-off：它用更大的L16共享codebook、每token depth metadata和不规则decode，换取极小的视频质量变化。根本原因不是残差完全不下降，而是深层stage的边际收益递减，且节省的index小于新增codebook与metadata成本。

## Figures

- `figures/error_histogram.svg`
- `figures/error_cdf.svg`
- `figures/layer_head_error_heatmap.svg`
- `figures/stage_error_curve.svg`
- `figures/stage_marginal_gain.svg`
- `figures/memory_breakdown.svg`
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.token_sample <= 0 or args.batch_size <= 0:
        raise ValueError("--token-sample and --batch-size must be positive")
    prompts = parse_ints(args.prompts, 10)
    layers = parse_ints(args.layers, LAYERS)
    device = torch.device(args.device)
    compute_dtype = _compute_dtype(args.compute_dtype, device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    smoke = codec_smoke(args.data_root, device, compute_dtype, args.threshold, args.batch_size)
    atomic_json(args.output_dir / "codec_smoke.json", smoke)
    if args.smoke:
        print(json.dumps(smoke, indent=2))
        return

    error_rows: list[dict[str, Any]] = []
    distributions: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    depth_values: dict[str, list[np.ndarray]] = defaultdict(list)
    stage_totals: dict[str, list[list[float]]] = {
        name: [[0.0, 0] for _ in range(PROGRESSIVE_MAX_DEPTH + 1)] for name in ("K", "V")
    }
    fixed_stage8: dict[str, list[float]] = {name: [0.0, 0] for name in ("K", "V")}

    for layer in layers:
        print(f"[layer {layer:02d}] loading {len(prompts)} prompts", flush=True)
        sampled = sample_layer(args.data_root / "kv_dump", prompts, layer, args.token_sample, args.seed)
        for tensor_name in ("K", "V"):
            original = sampled[tensor_name]
            fixed_metrics = analyze_fixed(
                original, layer=layer,
                codebook_dir=args.data_root / f"nonorm_{tensor_name.lower()}_l8c64_codebooks",
                device=device, compute_dtype=compute_dtype, batch_size=args.batch_size,
            )
            append_head_rows(
                error_rows, distributions, method="fixed_l8c64_int4", tensor_name=tensor_name,
                layer=layer, raw_mse=fixed_metrics[0], relative_mse=fixed_metrics[1],
                residual_rms=fixed_metrics[2],
            )
            fixed_stage8[tensor_name][0] += float(fixed_metrics[0].sum().item()) * DIM
            fixed_stage8[tensor_name][1] += int(fixed_metrics[0].numel()) * DIM

            progressive_metrics, depths, stage_sums = analyze_progressive(
                original, layer=layer,
                codebook_dir=args.data_root / f"nonorm_{tensor_name.lower()}_l16c64_codebooks",
                threshold=args.threshold, device=device, compute_dtype=compute_dtype,
                batch_size=args.batch_size,
            )
            append_head_rows(
                error_rows, distributions, method="error_bounded_l4_l16_int4", tensor_name=tensor_name,
                layer=layer, raw_mse=progressive_metrics[0], relative_mse=progressive_metrics[1],
                residual_rms=progressive_metrics[2], depths=depths,
            )
            depth_values[tensor_name].append(depths.detach().cpu().numpy().reshape(-1))
            for stage, (sum_sq, count) in enumerate(stage_sums):
                stage_totals[tensor_name][stage][0] += sum_sq
                stage_totals[tensor_name][stage][1] += count
            del original, fixed_metrics, progressive_metrics, depths
            if device.type == "cuda":
                torch.cuda.empty_cache()
        del sampled

    write_csv(args.output_dir / "error_distribution.csv", error_rows)
    global_stats: list[dict[str, Any]] = []
    for method in ("fixed_l8c64_int4", "error_bounded_l4_l16_int4"):
        for tensor_name in ("K", "V"):
            relative = np.concatenate(distributions[(method, tensor_name, "relative_mse")])
            row = {"method": method, "tensor": tensor_name, "count": int(relative.size)}
            for metric in ("raw_mse", "relative_mse", "residual_rms"):
                values = np.concatenate(distributions[(method, tensor_name, metric)])
                row.update(stats(values, metric))
            global_stats.append(row)
    write_csv(args.output_dir / "error_distribution_global.csv", global_stats)

    depth_summary: dict[str, Any] = {}
    depth_rows: list[dict[str, Any]] = []
    for tensor_name, chunks in depth_values.items():
        values = np.concatenate(chunks)
        histogram = Counter(int(value) for value in values)
        depth_summary[tensor_name] = {
            "count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.quantile(values, 0.5)), "p90": float(np.quantile(values, 0.9)),
            "p99": float(np.quantile(values, 0.99)),
            "max_depth_hit_rate": float(np.mean(values >= PROGRESSIVE_MAX_DEPTH)),
            "histogram": dict(sorted(histogram.items())),
        }
        for depth, count in sorted(histogram.items()):
            depth_rows.append({"tensor": tensor_name, "depth": depth, "count": count, "fraction": count / values.size})
    atomic_json(args.output_dir / "depth_distribution.json", depth_summary)
    write_csv(args.output_dir / "depth_distribution.csv", depth_rows)

    stage_rows = build_stage_rows(stage_totals, fixed_stage8)
    write_csv(args.output_dir / "stage_marginal_gain.csv", stage_rows)
    memory = memory_breakdown(args.data_root, args.benchmark_root)
    atomic_json(args.output_dir / "memory_breakdown.json", memory)
    write_csv(args.output_dir / "memory_breakdown.csv", memory_rows(memory))
    formal = formal_comparison(args.benchmark_root)
    atomic_json(args.output_dir / "inference_comparison.json", formal)
    atomic_json(
        args.output_dir / "packed_state_audit.json",
        {tensor_name: payload["packed_state_audit"] for tensor_name, payload in smoke.items()},
    )
    plot_outputs(args.output_dir, distributions, error_rows, stage_rows, memory)
    render_report(args.output_dir, formal, global_stats, depth_summary, stage_rows, memory, args.threshold)

    config = {
        "data_root": str(args.data_root), "benchmark_root": str(args.benchmark_root),
        "prompts": prompts, "layers": layers, "token_sample_per_prompt_layer_head": args.token_sample,
        "seed": args.seed, "threshold": args.threshold, "progressive_min_depth": PROGRESSIVE_MIN_DEPTH,
        "progressive_max_depth": PROGRESSIVE_MAX_DEPTH, "fixed_depth": FIXED_DEPTH,
        "compute_dtype": args.compute_dtype, "codebook_storage_dtype": "int4",
    }
    atomic_json(args.output_dir / "analysis_config.json", config)
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    codebook_dirs = [
        args.data_root / f"nonorm_{tensor}_l{levels}c64_codebooks"
        for tensor in ("k", "v") for levels in (8, 16)
    ]
    codebook_hashes = {
        directory.name: inventory_hash(list(directory.glob("layer_*_head_*.pt")), directory, content=True)
        for directory in codebook_dirs
    }
    dump_paths = [args.data_root / "kv_dump/test" / str(prompt) / f"layer_{layer}.pt" for prompt in prompts for layer in layers]
    run_info = {
        "git_commit": git_commit(), "config_path": str(args.output_dir / "analysis_config.json"),
        "config_hash": config_hash, "codebook_path": [str(path) for path in codebook_dirs],
        "codebook_hash": codebook_hashes, "input_dump_inventory_hash": inventory_hash(dump_paths, args.data_root, content=False),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "hostname": socket.gethostname(), "platform": platform.platform(),
        "gpu_type": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": args.seed, "prompt_list": prompts, "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(args.output_dir / "run_info.json", run_info)
    print(f"[done] {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
