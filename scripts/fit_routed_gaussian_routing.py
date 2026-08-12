#!/usr/bin/env python3
"""Fit a static cluster-free Gaussian bit-routing table on train prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from quant_videogen import routed_gaussian as codec  # noqa: E402
from run_fixed_budget_rvq_screen import layer_path  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_BASE_PARAMS = (
    DEFAULT_DATA
    / "truncatable_residual_strict_final_20260723/training/codec_params.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--base-tree-params", type=Path, default=DEFAULT_BASE_PARAMS)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument("--sample-tokens", type=int, default=1024)
    parser.add_argument("--prompt-repeats", type=int, default=3)
    parser.add_argument("--fractions", default="0.125,0.25,0.375")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.base_tree_params = args.base_tree_params.expanduser().resolve()
    args.fractions = tuple(float(item) for item in args.fractions.split(","))
    if any(not 0.0 < value < 0.5 for value in args.fractions):
        parser.error("fractions must be in (0,0.5)")
    if args.sample_tokens <= 0 or args.prompt_repeats <= 0:
        parser.error("sample-tokens and prompt-repeats must be positive")
    return args


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _depth_mse(
    residual: torch.Tensor,
    tree: dict[str, tuple[torch.Tensor, ...]],
    *,
    seed: int,
) -> tuple[float, float, float, float, float]:
    transformed = codec.groupwise_hadamard(
        residual.float(), 32, seed=seed
    ).reshape(-1, 128)
    rms = transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    raw = codec._to_positive_e4m3_raw(rms)
    scale = codec._from_positive_e4m3_raw(raw).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    def assign(current_scale: torch.Tensor) -> torch.Tensor:
        flat = (transformed / current_scale).reshape(-1)
        path = torch.zeros(flat.numel(), device=flat.device, dtype=torch.int64)
        for thresholds in tree["thresholds"]:
            branch = (flat >= thresholds[path].float()).to(torch.int64)
            path = path.bitwise_left_shift(1).bitwise_or(branch)
        return path

    result: list[float] = []
    for depth in range(1, 6):
        current_scale = scale
        path = assign(current_scale)
        prefix = path.bitwise_right_shift(5 - depth)
        normalized_reconstruction = (
            tree["reconstruction"][depth - 1][prefix]
            .float()
            .reshape_as(transformed)
        )
        refined = (
            (transformed * normalized_reconstruction).sum(-1, keepdim=True)
            / normalized_reconstruction.square().sum(-1, keepdim=True).clamp_min(1e-12)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        current_scale = codec._from_positive_e4m3_raw(
            codec._to_positive_e4m3_raw(refined)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        path = assign(current_scale)
        prefix = path.bitwise_right_shift(5 - depth)
        reconstructed = (
            tree["reconstruction"][depth - 1][prefix]
            .float()
            .reshape_as(transformed)
            * current_scale
        )
        result.append(
            float((transformed - reconstructed).square().mean().item())
        )
    return tuple(result)  # type: ignore[return-value]


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    base_params = torch.load(
        args.base_tree_params, map_location="cpu", weights_only=False
    )
    tree_by_name = {
        name: codec.extend_gaussian_tree_to_depth5(
            base_params,
            tensor_name=name,
            device=device,
        )
        for name in ("K", "V")
    }
    aggregates: dict[tuple[int, int, str], list[list[float]]] = {
        (layer, head, name): []
        for layer in range(30)
        for head in range(24)
        for name in ("K", "V")
    }
    for repeat in range(args.prompt_repeats):
        for layer in range(30):
            prompt = (layer * 7 + repeat * 11) % 20
            saved = torch.load(
                str(layer_path(args.data_root, "train", prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            total_tokens = int(saved["K"].shape[0] * saved["K"].shape[2])
            for head in range(24):
                generator = torch.Generator().manual_seed(
                    args.seed
                    + repeat * 10_000_019
                    + prompt * 1_000_003
                    + layer * 1009
                    + head * 97
                )
                sample_count = min(args.sample_tokens, total_tokens)
                ids = torch.randperm(total_tokens, generator=generator)[:sample_count]
                for name in ("K", "V"):
                    source = saved[name][:, head].reshape(-1, 128)[ids]
                    _base, residual = rvq_residual(
                        runtime_args,
                        source,
                        load_int4_base_codebook(
                            runtime_args, name, layer, head, device
                        ),
                        device,
                    )
                    mse = _depth_mse(
                        residual,
                        tree_by_name[name],
                        seed=args.codec_seed,
                    )
                    aggregates[(layer, head, name)].append(list(mse))
            print(
                f"repeat={repeat + 1}/{args.prompt_repeats} "
                f"layer={layer:02d} prompt={prompt:02d}",
                flush=True,
            )
    rows: list[dict[str, Any]] = []
    for (layer, head, name), samples in sorted(aggregates.items()):
        values = torch.tensor(samples, dtype=torch.float64).mean(0)
        rows.append(
            {
                "split": "train",
                "layer": layer,
                "head": head,
                "tensor": name,
                "prompt_repeats": args.prompt_repeats,
                "sample_tokens_per_repeat": args.sample_tokens,
                **{
                    f"depth_{depth}_mse": float(values[depth - 1])
                    for depth in range(1, 6)
                },
                "gain_4_to_5": float(values[3] - values[4]),
                "cost_4_to_3": float(values[2] - values[3]),
            }
        )
    return rows


def fit_routing(
    rows: list[dict[str, Any]], fractions: tuple[float, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    selected_offsets: dict[str, list[list[int]]] = {}
    for name in ("K", "V"):
        units = [row for row in rows if row["tensor"] == name]
        candidates: list[tuple[float, float, dict[tuple[int, int], int]]] = []
        for fraction in fractions:
            count = int(round(fraction * len(units)))
            high_order = sorted(
                units,
                key=lambda row: (
                    -float(row["gain_4_to_5"]),
                    int(row["layer"]),
                    int(row["head"]),
                ),
            )
            high = {
                (int(row["layer"]), int(row["head"]))
                for row in high_order[:count]
            }
            low_order = sorted(
                (
                    row
                    for row in units
                    if (int(row["layer"]), int(row["head"])) not in high
                ),
                key=lambda row: (
                    float(row["cost_4_to_3"]),
                    int(row["layer"]),
                    int(row["head"]),
                ),
            )
            low = {
                (int(row["layer"]), int(row["head"]))
                for row in low_order[:count]
            }
            offsets = {
                (int(row["layer"]), int(row["head"])): (
                    1
                    if (int(row["layer"]), int(row["head"])) in high
                    else -1
                    if (int(row["layer"]), int(row["head"])) in low
                    else 0
                )
                for row in units
            }
            uniform = sum(float(row["depth_4_mse"]) for row in units) / len(units)
            routed = sum(
                float(
                    row[
                        f"depth_{4 + offsets[(int(row['layer']), int(row['head']))]}_mse"
                    ]
                )
                for row in units
            ) / len(units)
            summaries.append(
                {
                    "tensor": name,
                    "fraction": fraction,
                    "low_units": count,
                    "normal_units": len(units) - 2 * count,
                    "high_units": count,
                    "uniform_b4_mse": uniform,
                    "routed_b4_mse": routed,
                    "relative_mse_improvement_pct": 100.0 * (1.0 - routed / uniform),
                }
            )
            candidates.append((routed, fraction, offsets))
        _score, fraction, winner = min(candidates, key=lambda item: (item[0], item[1]))
        selected_offsets[name] = [
            [winner[(layer, head)] for head in range(24)]
            for layer in range(30)
        ]
        for row in summaries:
            if row["tensor"] == name:
                row["selected"] = bool(row["fraction"] == fraction)
    routing = {
        "schema_version": codec.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "offsets": selected_offsets,
        "fit": {
            "split": "train",
            "objective": "minimum prompt-sampled B4 coefficient MSE",
            "high_score": "largest depth4-to-depth5 MSE gain",
            "low_score": "smallest depth4-to-depth3 MSE cost",
            "fractions": list(fractions),
        },
    }
    codec.validate_routing_table(routing)
    return routing, summaries


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "device": args.device,
        "base_tree_params": str(args.base_tree_params),
        "base_tree_params_hash": sha256_file(args.base_tree_params),
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "sample_tokens": args.sample_tokens,
        "prompt_repeats": args.prompt_repeats,
        "prompt_list": list(range(20)),
        "layers": list(range(30)),
        "heads": list(range(24)),
        "fractions": list(args.fractions),
        "rvq_assignment_mode": "runtime_bf16",
        "residual_codec": {
            "clustering": False,
            "tree": "fitted_residual_depth4_plus_fixed_conditional_gaussian_depth5",
            "group_size": 128,
            "hadamard_group_size": 32,
            "scale": "fp8_e4m3fn_rms",
            "hadamard": True,
        },
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    rows = collect(args)
    routing, summaries = fit_routing(rows, args.fractions)
    write_csv(args.output_root / "unit_marginals.csv", rows)
    write_csv(args.output_root / "fraction_summary.csv", summaries)
    routing_path = args.output_root / "routing.json"
    write_json(routing_path, routing)
    base_params = torch.load(
        args.base_tree_params, map_location="cpu", weights_only=False
    )
    torch.save(
        {
            "schema_version": "routed_gaussian.tree_params.v1",
            "K": codec.extend_gaussian_tree_to_depth5(
                base_params, tensor_name="K"
            ),
            "V": codec.extend_gaussian_tree_to_depth5(
                base_params, tensor_name="V"
            ),
        },
        args.output_root / "tree_params.pt",
    )
    codebook_paths = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": {key: str(value) for key, value in codebook_paths.items()},
        "codebook_hash": {
            key: stable_directory_hash(value) for key, value in codebook_paths.items()
        },
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.device(args.device))
            if torch.cuda.is_available()
            else None
        ),
        "seed": args.seed,
        "prompt_list": list(range(20)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "routing_path": str(routing_path),
        "routing_hash": sha256_file(routing_path),
        "base_tree_params_path": str(args.base_tree_params),
        "base_tree_params_hash": sha256_file(args.base_tree_params),
        "implementation_path": str(REPO_ROOT / "quant_videogen/routed_gaussian.py"),
        "implementation_hash": sha256_file(
            REPO_ROOT / "quant_videogen/routed_gaussian.py"
        ),
        "physical_bits_per_value_b4": 4.0625,
        "environment": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
    }
    write_json(args.output_root / "run_info.json", run_info)


if __name__ == "__main__":
    main()
