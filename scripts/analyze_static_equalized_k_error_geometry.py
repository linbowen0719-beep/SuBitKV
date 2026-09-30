#!/usr/bin/env python3
"""Measure why lower K MSE can fail to improve closed-loop quality.

This is a reporting-only diagnostic.  It compares the physical uniform and
static-equalized midrange reconstruction paths on frozen test captures and
reports error bias, covariance concentration, and frozen-query QK error.
"""

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

from subitkv.centered_absmax_scalar import (  # noqa: E402
    decode_centered_absmax_scalar_residual,
    encode_centered_absmax_scalar_residual,
)
from subitkv.static_equalized_midrange import (  # noqa: E402
    decode_static_equalized_midrange_residual,
    encode_static_equalized_midrange_residual,
    validate_params,
)
from subitkv.truncatable_residual import groupwise_hadamard  # noqa: E402
from run_fixed_budget_rvq_screen import layer_path, parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--layers", default="0,1,23,29")
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--key-sample", type=int, default=2048)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.params = args.params.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.layers = parse_ints(args.layers)
    args.prompts = parse_ints(args.prompts)
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty CSV")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def covariance_geometry(error: torch.Tensor) -> dict[str, float]:
    error = error.float()
    mse = error.square().mean()
    bias_energy = error.mean(0).square().mean()
    centered = error - error.mean(0)
    covariance = centered.T @ centered / max(1, int(centered.shape[0]))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-30)
    probability = eigenvalues / total
    return {
        "K_MSE": float(mse.item()),
        "error_bias_energy_fraction": float(
            (bias_energy / mse.clamp_min(1e-30)).item()
        ),
        "error_cov_top1_fraction": float((eigenvalues[:1].sum() / total).item()),
        "error_cov_top8_fraction": float((eigenvalues[:8].sum() / total).item()),
        "error_cov_effective_rank": float(
            (1.0 / probability.square().sum().clamp_min(1e-30)).item()
        ),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    params = torch.load(args.params, map_location=device, weights_only=False)
    validate_params(params)
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    config = {
        "schema_version": "static_equalized_k_error_geometry.config.v1",
        "params": str(args.params),
        "params_hash": sha256_file(args.params),
        "data_root": str(args.data_root),
        "layers": list(args.layers),
        "prompts": list(args.prompts),
        "split": args.split,
        "key_sample": args.key_sample,
        "query_sample": args.query_sample,
        "seed": args.seed,
        "selection": "reporting-only; no parameter or layer selection",
        "uses_clustering": False,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    rows: list[dict[str, Any]] = []
    for prompt in args.prompts:
        for layer in args.layers:
            saved = torch.load(
                str(layer_path(args.data_root, args.split, prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            originals = []
            bases = []
            residuals = []
            queries = []
            for head in range(24):
                flat_k = saved["K"][:, head].reshape(-1, 128)
                flat_q = saved["Q"][:, head].reshape(-1, 128)
                generator = torch.Generator().manual_seed(
                    args.seed + prompt * 1_000_003 + layer * 1009 + head
                )
                key_ids = deterministic_indices(
                    len(flat_k), args.key_sample, generator, "inference_span"
                )
                query_ids = torch.randperm(
                    len(flat_q), generator=generator
                )[: args.query_sample]
                original = flat_k[key_ids].to(device=device, dtype=torch.float32)
                base, residual = rvq_residual(
                    runtime_args,
                    original,
                    load_int4_base_codebook(
                        runtime_args, "K", layer, head, device
                    ),
                    device,
                )
                originals.append(original)
                bases.append(base)
                residuals.append(residual)
                queries.append(
                    flat_q[query_ids].to(device=device, dtype=torch.float32)
                )
            original = torch.stack(originals).unsqueeze(0)
            base = torch.stack(bases).unsqueeze(0)
            residual = torch.stack(residuals).unsqueeze(0)
            query = torch.stack(queries)
            uniform_state = encode_centered_absmax_scalar_residual(
                residual, bits=4, group_size=128, center_mode="midrange"
            )
            uniform_residual = decode_centered_absmax_scalar_residual(
                uniform_state, torch.bfloat16
            )
            static_state = encode_static_equalized_midrange_residual(
                residual,
                tensor_name="K",
                layer_idx=layer,
                params=params,
                bits=4,
            )
            static_residual = decode_static_equalized_midrange_residual(
                static_state, torch.bfloat16
            )
            channel_scale = params["k_channel_scale"][layer].float()[
                None, :, None, :
            ]
            equalized = residual.float() / channel_scale
            rotated = groupwise_hadamard(
                equalized, 128, seed=20260727, inverse=False
            )
            rotated_state = encode_centered_absmax_scalar_residual(
                rotated, bits=4, group_size=128, center_mode="midrange"
            )
            rotated_decoded = decode_centered_absmax_scalar_residual(
                rotated_state, torch.float32
            )
            equalized_hadamard_residual = (
                groupwise_hadamard(
                    rotated_decoded, 128, seed=20260727, inverse=True
                )
                * channel_scale
            ).to(torch.bfloat16)
            for method, decoded in (
                ("uniform_midrange", uniform_residual),
                ("static_equalized", static_residual),
                ("static_equalized_hadamard", equalized_hadamard_residual),
            ):
                reconstructed = (base + decoded.float()).to(torch.bfloat16).float()
                for head in range(24):
                    error = reconstructed[0, head] - original[0, head]
                    geometry = covariance_geometry(error)
                    q = query[head]
                    qk_error = q @ error.T / math.sqrt(128.0)
                    query_diagonal = q.square().mean(0)
                    error_diagonal = error.square().mean(0)
                    rows.append(
                        {
                            "prompt_id": prompt,
                            "layer": layer,
                            "head": head,
                            "method": method,
                            **geometry,
                            "QK_MSE": float(qk_error.square().mean().item()),
                            "QK_over_K_MSE": float(
                                qk_error.square().mean().item()
                                / max(geometry["K_MSE"], 1e-30)
                            ),
                            "query_diagonal_error_proxy": float(
                                (query_diagonal * error_diagonal).sum().item()
                                / 128.0
                            ),
                        }
                    )
            print(f"prompt={prompt} layer={layer}", flush=True)
    unit_path = args.output_root / "unit_metrics.csv"
    write_csv(unit_path, rows)
    summary_rows: list[dict[str, Any]] = []
    metrics = [
        "K_MSE",
        "QK_MSE",
        "QK_over_K_MSE",
        "query_diagonal_error_proxy",
        "error_bias_energy_fraction",
        "error_cov_top1_fraction",
        "error_cov_top8_fraction",
        "error_cov_effective_rank",
    ]
    scopes = [("all", rows)]
    scopes.extend(
        (f"layer_{layer}", [row for row in rows if row["layer"] == layer])
        for layer in args.layers
    )
    for scope, selected in scopes:
        for method in (
            "uniform_midrange",
            "static_equalized",
            "static_equalized_hadamard",
        ):
            subset = [row for row in selected if row["method"] == method]
            summary_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "units": len(subset),
                    **{
                        metric: sum(float(row[metric]) for row in subset)
                        / len(subset)
                        for metric in metrics
                    },
                }
            )
    summary_path = args.output_root / "summary.csv"
    write_csv(summary_path, summary_rows)
    codebook_paths = {
        name: args.data_root / f"nonorm_{name.lower()}_l8c64_codebooks"
        for name in ("K", "V")
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                name: str(path) for name, path in codebook_paths.items()
            },
            "codebook_hash": {
                name: stable_directory_hash(path)
                for name, path in codebook_paths.items()
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "output_hashes": {
                str(path): sha256_file(path) for path in (unit_path, summary_path)
            },
        },
    )


if __name__ == "__main__":
    main()
