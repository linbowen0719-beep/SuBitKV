#!/usr/bin/env python3
"""Validate the frozen uniform-grid Gaussian codec on one real K cache unit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_uniform_gaussian_k import load_k_residual, load_queries  # noqa: E402
from subitkv import uniform_grid_gaussian  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    load_int4_base_codebook,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
    "tree_params.pt"
)
DEFAULT_REFERENCE = (
    REPO_ROOT
    / "results/analysis/"
    "uniform_gaussian_grid22x10_f8p4_full30_test10_20260725/"
    "unit_metrics.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--reference-csv", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--layer", type=int, default=29)
    parser.add_argument("--head", type=int, default=17)
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in ("output_root", "data_root", "tree_params", "reference_csv"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.key_sample % uniform_grid_gaussian.SPATIAL_TOKENS:
        raise ValueError("key-sample must contain complete 22x40 latent frames")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as handle:
        handle.write(json_bytes(value))


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def reference_row(args: argparse.Namespace) -> dict[str, str]:
    with args.reference_csv.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row["prompt_id"]) == args.prompt
            and int(row["layer"]) == args.layer
            and int(row["head"]) == args.head
            and row["method"]
            == "rvq_uniform_gaussian_grid22x10_bilinear_f8p4_b4"
        ]
    if len(rows) != 1:
        raise ValueError(f"expected one matching analysis row, found {len(rows)}")
    return rows[0]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "uniform_grid_gaussian.physical_validation.config.v1",
        "data_root": str(args.data_root),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "reference_csv": str(args.reference_csv),
        "reference_csv_hash": sha256_file(args.reference_csv),
        "device": args.device,
        "prompt": args.prompt,
        "layer": args.layer,
        "head": args.head,
        "key_sample": args.key_sample,
        "query_sample": args.query_sample,
        "seed": args.seed,
        "uses_query_for_codec": False,
        "uses_clustering": False,
        "uses_routed_head_depth": False,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("physical validation requires CUDA")
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    load_args = SimpleNamespace(
        data_root=args.data_root,
        seed=args.seed,
        key_sample=args.key_sample,
        query_sample=args.query_sample,
    )
    codebook = load_int4_base_codebook(
        runtime_args, "K", args.layer, args.head, device
    )
    original, base, residual = load_k_residual(
        load_args,
        runtime_args,
        split="test",
        prompt=args.prompt,
        layer=args.layer,
        head=args.head,
        count=args.key_sample,
        codebook=codebook,
        device=device,
    )
    query = load_queries(
        load_args,
        prompt=args.prompt,
        layer=args.layer,
        head=args.head,
        device=device,
    )
    tree = torch.load(args.tree_params, map_location=device, weights_only=False)
    state = uniform_grid_gaussian.encode_uniform_grid_gaussian_residual(
        residual[None, None].to(torch.bfloat16),
        layer_idx=args.layer,
        tree_params=tree,
        seed=20260722,
    )
    decoded_residual = (
        uniform_grid_gaussian.decode_uniform_grid_gaussian_residual(
            state, output_dtype=torch.bfloat16
        )[0, 0]
    )
    reconstructed = (
        base.to(torch.bfloat16) + decoded_residual
    ).to(torch.bfloat16).float()
    error = original - reconstructed
    physical_k_mse = float(error.square().mean())
    physical_qk_mse = float(
        (query @ error.T / math.sqrt(uniform_grid_gaussian.DIM))
        .square()
        .mean()
    )
    reference = reference_row(args)
    analysis_k_mse = float(reference["K_MSE"])
    analysis_qk_mse = float(reference["QK_logits_MSE"])
    row = {
        "prompt_id": args.prompt,
        "layer": args.layer,
        "head": args.head,
        "physical_K_MSE": physical_k_mse,
        "analysis_K_MSE": analysis_k_mse,
        "physical_vs_analysis_K_delta_pct": (
            100.0 * (physical_k_mse / analysis_k_mse - 1.0)
        ),
        "physical_QK_logits_MSE": physical_qk_mse,
        "analysis_QK_logits_MSE": analysis_qk_mse,
        "physical_vs_analysis_QK_delta_pct": (
            100.0 * (physical_qk_mse / analysis_qk_mse - 1.0)
        ),
        "local_payload_bytes": int(state["local_payload_bytes"]),
        "predictor_payload_bytes": int(state["predictor_payload_bytes"]),
        "shared_table_bytes": int(state["shared_table_bytes"]),
        "local_physical_bits_per_value": float(
            state["diagnostics"]["local_physical_bits_per_value"]
        ),
        "physical_bits_per_value_including_shared_for_one_unit": float(
            uniform_grid_gaussian.payload_nbytes(state) * 8 / residual.numel()
        ),
    }
    metrics_path = args.output_root / "metrics.csv"
    with metrics_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    codebook_path = (
        args.data_root
        / "nonorm_k_l8c64_codebooks"
        / f"layer_{args.layer}_head_{args.head}.pt"
    )
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": str(codebook_path),
        "codebook_hash": sha256_file(codebook_path),
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": args.seed,
        "prompt_list": [args.prompt],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(metrics_path),
        "metrics_hash": sha256_file(metrics_path),
    }
    write_json(args.output_root / "run_info.json", run_info)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
