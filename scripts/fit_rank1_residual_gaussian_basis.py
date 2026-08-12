#!/usr/bin/env python3
"""Fit train-only rank-1 RVQ-residual channel predictors.

Each worker processes a disjoint set of Wan layers.  For every layer, tensor
(K/V), and head it accumulates the second moment of the runtime-faithful
L8C64-INT4 RVQ residual over 20 training prompts, then stores the leading
eigenvector in FP16.  No query, test prompt, K-means, VQ, or PQ is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from quant_videogen.rvq import (
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
PARAM_SCHEMA = "rank1_residual_gaussian.params.v1"
KNOWN_CODEBOOK_HASHES = {
    "K": "83501131ea25e074731b4932c6aa7b79ab96d42372934934f345050a63c172dc",
    "V": "6c111f59335dff4a9d9a4261d9195166dbf28635049e8abf124a7c5b156978a1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--samples-per-prompt", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-end", type=int, default=20)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        parser.error("invalid shard geometry")
    if args.samples_per_prompt <= 0:
        parser.error("samples-per-prompt must be positive")
    if not 0 <= args.prompt_start < args.prompt_end <= 20:
        parser.error("prompt range must satisfy 0 <= start < end <= 20")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def write_json_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv_create(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty diagnostic table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def layer_path(data_root: Path, prompt: int, layer: int) -> Path:
    return data_root / "kv_dump" / "train" / str(prompt) / f"layer_{layer}.pt"


def sampled_bhsd(
    payload: dict[str, Any],
    tensor_name: str,
    *,
    count: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    value = payload[tensor_name]
    if tuple(value.shape[1:])[-2:] != (880, 128):
        raise ValueError(f"unexpected {tensor_name} dump shape {tuple(value.shape)}")
    flat = value.permute(1, 0, 2, 3).reshape(24, -1, 128)
    generator = torch.Generator().manual_seed(int(seed))
    ids = torch.randperm(flat.shape[1], generator=generator)[:count]
    return flat[:, ids].unsqueeze(0).to(
        device=device, dtype=torch.bfloat16
    ).contiguous()


@torch.no_grad()
def fit_layer(
    args: argparse.Namespace,
    layer: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    moments = {
        name: torch.zeros((24, 128, 128), device=device, dtype=torch.float64)
        for name in ("K", "V")
    }
    energies = {
        name: torch.zeros(24, device=device, dtype=torch.float64)
        for name in ("K", "V")
    }
    total_vectors = 0
    prompt_list = list(range(args.prompt_start, args.prompt_end))
    for prompt in prompt_list:
        payload = torch.load(
            str(layer_path(args.data_root, prompt, layer)),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        for tensor_name in ("K", "V"):
            source = sampled_bhsd(
                payload,
                tensor_name,
                count=args.samples_per_prompt,
                seed=args.seed + prompt * 1_000_003 + layer * 1009,
                device=device,
            )
            packed = rvq_encode_tensor_cache(
                source,
                layer_idx=layer,
                codebook_dir=(
                    args.data_root
                    / f"nonorm_{tensor_name.lower()}_l8c64_codebooks"
                ),
                batch_size=4096,
                compute_dtype="bf16",
                codebook_storage_dtype="int4",
                scale_storage_dtype="uint4",
            )
            base = rvq_dequantize_tensor_cache(packed, torch.bfloat16)
            residual = (source - base).float()[0]
            moments[tensor_name].add_(
                torch.einsum("hsd,hse->hde", residual, residual).double()
            )
            energies[tensor_name].add_(
                residual.square().mean(dim=(1, 2)).double()
            )
        total_vectors += int(source.shape[2])
        print(f"layer={layer} prompt={prompt} complete", flush=True)

    result: dict[str, torch.Tensor] = {}
    diagnostics: list[dict[str, Any]] = []
    for tensor_name in ("K", "V"):
        covariance = moments[tensor_name] / max(total_vectors, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = eigenvalues.argsort(dim=-1, descending=True)
        sorted_values = eigenvalues.gather(1, order).clamp_min(0)
        top_index = order[:, 0].reshape(24, 1, 1).expand(24, 128, 1)
        basis = eigenvectors.gather(2, top_index).squeeze(-1)
        pivot = basis.abs().argmax(dim=-1, keepdim=True)
        pivot_sign = basis.gather(1, pivot).sign().clamp(min=-1, max=1)
        basis = basis * torch.where(
            pivot_sign == 0, torch.ones_like(pivot_sign), pivot_sign
        )
        stored = basis.to(torch.float16).cpu().contiguous()
        result[tensor_name] = stored
        total = sorted_values.sum(dim=-1).clamp_min(1e-30)
        probabilities = sorted_values / total.unsqueeze(-1)
        for head in range(24):
            diagnostics.append(
                {
                    "layer": layer,
                    "head": head,
                    "tensor": tensor_name,
                    "train_prompts": len(prompt_list),
                    "vectors_per_prompt": args.samples_per_prompt,
                    "residual_mse": float(
                        energies[tensor_name][head] / len(prompt_list)
                    ),
                    "top1_energy_fraction": float(
                        sorted_values[head, 0] / total[head]
                    ),
                    "top8_energy_fraction": float(
                        sorted_values[head, :8].sum() / total[head]
                    ),
                    "effective_rank_participation": float(
                        1.0
                        / probabilities[head]
                        .square()
                        .sum()
                        .clamp_min(1e-30)
                    ),
                }
            )
    return result, diagnostics


def worker(args: argparse.Namespace) -> None:
    worker_root = args.output_root / f"worker_{args.shard_id:02d}"
    worker_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    layers = list(range(30))[args.shard_id :: args.num_shards]
    tables: dict[int, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, Any]] = []
    for layer in layers:
        tables[layer], unit_rows = fit_layer(args, layer, device)
        rows.extend(unit_rows)
    params_path = worker_root / "basis.pt"
    torch.save(
        {
            "schema_version": f"{PARAM_SCHEMA}.worker",
            "layers": tables,
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        },
        params_path,
    )
    diagnostics_path = worker_root / "diagnostics.csv"
    write_csv_create(diagnostics_path, rows)
    config = {
        "schema_version": f"{PARAM_SCHEMA}.worker_config",
        "data_root": str(args.data_root),
        "split": "train",
        "prompt_list": list(range(args.prompt_start, args.prompt_end)),
        "layers": layers,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "uses_query": False,
        "uses_clustering": False,
        "test_prompts_read": False,
    }
    config_path = worker_root / "config.json"
    write_json_create(config_path, config)
    write_json_create(
        worker_root / "run_info.json",
        {
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                name: str(
                    args.data_root
                    / f"nonorm_{name.lower()}_l8c64_codebooks"
                )
                for name in ("K", "V")
            },
            "codebook_hash": KNOWN_CODEBOOK_HASHES,
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(range(args.prompt_start, args.prompt_end)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
        },
    )


def aggregate(args: argparse.Namespace) -> None:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    rows: list[dict[str, str]] = []
    sources: dict[str, str] = {}
    for shard in range(args.num_shards):
        root = args.output_root / f"worker_{shard:02d}"
        params_path = root / "basis.pt"
        payload = torch.load(params_path, map_location="cpu", weights_only=False)
        for layer, table in payload["layers"].items():
            if int(layer) in layers:
                raise ValueError(f"duplicate layer {layer}")
            layers[int(layer)] = table
        with (root / "diagnostics.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows.extend(csv.DictReader(handle))
        sources[str(params_path)] = sha256_file(params_path)
    if sorted(layers) != list(range(30)):
        raise ValueError("worker outputs do not cover all 30 layers")
    basis = {
        name: torch.stack([layers[layer][name] for layer in range(30)])
        for name in ("K", "V")
    }
    params_path = args.output_root / "params.pt"
    torch.save(
        {
            "schema_version": PARAM_SCHEMA,
            "basis": basis,
            "num_layers": 30,
            "num_heads": 24,
            "dim": 128,
            "basis_storage": "fp16",
            "uses_query": False,
            "uses_clustering": False,
            "test_prompts_read": False,
        },
        params_path,
    )
    diagnostics_path = args.output_root / "diagnostics.csv"
    write_csv_create(diagnostics_path, rows)
    config = {
        "schema_version": f"{PARAM_SCHEMA}.config",
        "data_root": str(args.data_root),
        "split": "train",
        "prompt_list": list(range(args.prompt_start, args.prompt_end)),
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "prompt_start": args.prompt_start,
        "prompt_end": args.prompt_end,
    }
    config_path = args.output_root / "config.json"
    write_json_create(config_path, config)
    write_json_create(
        args.output_root / "run_info.json",
        {
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                name: str(
                    args.data_root
                    / f"nonorm_{name.lower()}_l8c64_codebooks"
                )
                for name in ("K", "V")
            },
            "codebook_hash": KNOWN_CODEBOOK_HASHES,
            "hostname": socket.gethostname(),
            "gpu_type": "worker manifests",
            "seed": args.seed,
            "prompt_list": list(range(args.prompt_start, args.prompt_end)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
        },
    )
    write_json_create(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{PARAM_SCHEMA}.evidence",
            "params_hash": sha256_file(params_path),
            "diagnostics_hash": sha256_file(diagnostics_path),
            "worker_params": sources,
        },
    )


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
