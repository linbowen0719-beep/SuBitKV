#!/usr/bin/env python3
"""Fit Gaussian low/normal/high routes from K-only/V-only attention output loss."""

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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from subitkv import routed_gaussian as codec  # noqa: E402
from subitkv import routed_midrange_scalar as midrange_codec  # noqa: E402
from fit_routed_gaussian_routing import (  # noqa: E402
    fit_routing,
    sha256_file,
    write_csv,
    write_json,
)
from run_fixed_budget_rvq_screen import layer_path  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tree-params", type=Path, required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument("--key-sample", type=int, default=2048)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--prompt-repeats", type=int, default=3)
    parser.add_argument("--fractions", default="0.125,0.25,0.375")
    parser.add_argument("--input-marginals", type=Path)
    parser.add_argument(
        "--allocation-mode",
        choices=("per_tensor", "cross_k_high_v_low"),
        default="per_tensor",
    )
    parser.add_argument(
        "--codec-family",
        choices=("gaussian", "midrange"),
        default="gaussian",
    )
    parser.add_argument("--layer-start", type=int, default=0)
    parser.add_argument("--layer-end", type=int, default=30)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument(
        "--input-marginal-shards",
        type=Path,
        nargs="*",
        default=(),
    )
    parser.add_argument(
        "--marginal-column-prefix",
        choices=("", "split_a_", "split_b_"),
        default="",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    if args.input_marginals is not None:
        args.input_marginals = args.input_marginals.expanduser().resolve()
    args.input_marginal_shards = tuple(
        item.expanduser().resolve() for item in args.input_marginal_shards
    )
    if not 0 <= args.layer_start < args.layer_end <= 30:
        parser.error("layer range must satisfy 0 <= start < end <= 30")
    if args.collect_only and (
        args.input_marginals is not None or args.input_marginal_shards
    ):
        parser.error("collect-only cannot read precomputed marginals")
    args.fractions = tuple(float(item) for item in args.fractions.split(","))
    return args


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def gpu_type_or_unavailable(device: str) -> str:
    """Return GPU metadata without making CPU-only aggregation depend on CUDA."""
    try:
        return torch.cuda.get_device_name(torch.device(device))
    except (AssertionError, RuntimeError):
        return "unavailable (CPU-only aggregation)"


def quantize_depths(
    residual: torch.Tensor,
    tree: dict[str, tuple[torch.Tensor, ...]],
    *,
    seed: int,
) -> dict[int, torch.Tensor]:
    shape = residual.shape
    transformed = codec.groupwise_hadamard(
        residual.float(), 128, seed=seed
    ).reshape(-1, 128)
    rms = transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    initial_scale = codec._from_positive_e4m3_raw(
        codec._to_positive_e4m3_raw(rms)
    ).clamp_min(torch.finfo(torch.float32).tiny)

    def assign(scale: torch.Tensor) -> torch.Tensor:
        flat = (transformed / scale).reshape(-1)
        path = torch.zeros(flat.numel(), device=flat.device, dtype=torch.int64)
        for thresholds in tree["thresholds"]:
            branch = (flat >= thresholds[path].float()).to(torch.int64)
            path = path.bitwise_left_shift(1).bitwise_or(branch)
        return path

    result: dict[int, torch.Tensor] = {}
    for depth in (3, 4, 5):
        scale = initial_scale
        path = assign(scale)
        prefix = path.bitwise_right_shift(5 - depth)
        normalized = (
            tree["reconstruction"][depth - 1][prefix]
            .float()
            .reshape_as(transformed)
        )
        refined = (
            (transformed * normalized).sum(-1, keepdim=True)
            / normalized.square().sum(-1, keepdim=True).clamp_min(1e-12)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        scale = codec._from_positive_e4m3_raw(
            codec._to_positive_e4m3_raw(refined)
        ).clamp_min(torch.finfo(torch.float32).tiny)
        path = assign(scale)
        prefix = path.bitwise_right_shift(5 - depth)
        reconstructed = (
            tree["reconstruction"][depth - 1][prefix]
            .float()
            .reshape_as(transformed)
            * scale
        ).reshape(shape)
        result[depth] = codec.groupwise_hadamard(
            reconstructed, 128, seed=seed, inverse=True
        ).to(torch.bfloat16)
    return result


def collect(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.codec_family == "gaussian":
        params = torch.load(
            args.tree_params, map_location=device, weights_only=False
        )
        trees = {
            name: codec._canonical_tree(
                params, tensor_name=name, device=device
            )
            for name in ("K", "V")
        }
    else:
        trees = {
            name: midrange_codec.build_qv_anchored_tree(device=device)
            for name in ("K", "V")
        }
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    values: dict[tuple[int, int, str], list[dict[int, float]]] = {
        (layer, head, name): []
        for layer in range(args.layer_start, args.layer_end)
        for head in range(24)
        for name in ("K", "V")
    }
    for repeat in range(args.prompt_repeats):
        for layer in range(args.layer_start, args.layer_end):
            prompt = (layer * 7 + repeat * 11) % 20
            saved = torch.load(
                str(layer_path(args.data_root, "train", prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            for head in range(24):
                flat = {
                    name: saved[name][:, head].reshape(-1, 128)
                    for name in ("Q", "K", "V")
                }
                generator = torch.Generator().manual_seed(
                    args.seed
                    + repeat * 10_000_019
                    + prompt * 1_000_003
                    + layer * 1009
                    + head * 97
                )
                key_ids = deterministic_indices(
                    flat["K"].shape[0],
                    args.key_sample,
                    generator,
                    "inference_span",
                )
                query_ids = torch.randperm(
                    flat["Q"].shape[0], generator=generator
                )[: args.query_sample]
                q = flat["Q"][query_ids].to(device=device, dtype=torch.float32)
                k = flat["K"][key_ids].to(device=device, dtype=torch.float32)
                v = flat["V"][key_ids].to(device=device, dtype=torch.float32)
                kb, kr = rvq_residual(
                    runtime_args,
                    k,
                    load_int4_base_codebook(
                        runtime_args, "K", layer, head, device
                    ),
                    device,
                )
                vb, vr = rvq_residual(
                    runtime_args,
                    v,
                    load_int4_base_codebook(
                        runtime_args, "V", layer, head, device
                    ),
                    device,
                )
                reference_logits = q @ k.T / math.sqrt(128)
                reference_probs = F.softmax(reference_logits, -1)
                reference_output = reference_probs @ v
                if args.codec_family == "midrange":
                    k_depths = midrange_codec.quantize_midrange_depths(
                        kr,
                        tensor_name="K",
                    )
                    v_depths = midrange_codec.quantize_midrange_depths(
                        vr,
                        tensor_name="V",
                    )
                else:
                    k_depths = quantize_depths(
                        kr, trees["K"], seed=args.codec_seed
                    )
                    v_depths = quantize_depths(
                        vr, trees["V"], seed=args.codec_seed
                    )
                k_losses: dict[int, float] = {}
                v_losses: dict[int, float] = {}
                for depth in (3, 4, 5):
                    kh = (kb + k_depths[depth]).to(torch.bfloat16).float()
                    vh = (vb + v_depths[depth]).to(torch.bfloat16).float()
                    k_output = F.softmax(
                        q @ kh.T / math.sqrt(128), -1
                    ) @ v
                    v_output = reference_probs @ vh
                    k_losses[depth] = float(
                        F.mse_loss(k_output, reference_output).item()
                    )
                    v_losses[depth] = float(
                        F.mse_loss(v_output, reference_output).item()
                    )
                values[(layer, head, "K")].append(k_losses)
                values[(layer, head, "V")].append(v_losses)
            print(
                f"repeat={repeat + 1}/{args.prompt_repeats} "
                f"layer={layer:02d} prompt={prompt:02d}",
                flush=True,
            )
    rows: list[dict[str, Any]] = []
    for (layer, head, name), samples in sorted(values.items()):
        means = {
            depth: sum(sample[depth] for sample in samples) / len(samples)
            for depth in (3, 4, 5)
        }
        midpoint = len(samples) // 2
        split_samples = {
            "split_a_": samples[:midpoint],
            "split_b_": samples[midpoint:],
        }
        split_values: dict[str, float] = {}
        if midpoint > 0 and split_samples["split_b_"]:
            for prefix, selected in split_samples.items():
                selected_means = {
                    depth: sum(sample[depth] for sample in selected)
                    / len(selected)
                    for depth in (3, 4, 5)
                }
                split_values.update(
                    {
                        f"{prefix}depth_{depth}_mse": selected_means[depth]
                        for depth in (3, 4, 5)
                    }
                )
                split_values[f"{prefix}gain_4_to_5"] = (
                    selected_means[4] - selected_means[5]
                )
                split_values[f"{prefix}cost_4_to_3"] = (
                    selected_means[3] - selected_means[4]
                )
        rows.append(
            {
                "split": "train",
                "layer": layer,
                "head": head,
                "tensor": name,
                "prompt_repeats": args.prompt_repeats,
                "key_sample": args.key_sample,
                "query_sample": args.query_sample,
                "depth_1_mse": float("nan"),
                "depth_2_mse": float("nan"),
                "depth_3_mse": means[3],
                "depth_4_mse": means[4],
                "depth_5_mse": means[5],
                "gain_4_to_5": means[4] - means[5],
                "cost_4_to_3": means[3] - means[4],
                **split_values,
            }
        )
    return rows


def fit_per_layer_balanced(
    rows: list[dict[str, Any]], fractions: tuple[float, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    selected_offsets: dict[str, list[list[int]]] = {}
    for name in ("K", "V"):
        units = [row for row in rows if row["tensor"] == name]
        candidates: list[tuple[float, float, dict[tuple[int, int], int]]] = []
        for fraction in fractions:
            per_layer = int(round(fraction * 24))
            offsets: dict[tuple[int, int], int] = {}
            for layer in range(30):
                layer_rows = [row for row in units if int(row["layer"]) == layer]
                high_order = sorted(
                    layer_rows,
                    key=lambda row: (
                        -float(row["gain_4_to_5"]),
                        int(row["head"]),
                    ),
                )
                high = {int(row["head"]) for row in high_order[:per_layer]}
                low_order = sorted(
                    (row for row in layer_rows if int(row["head"]) not in high),
                    key=lambda row: (
                        float(row["cost_4_to_3"]),
                        int(row["head"]),
                    ),
                )
                low = {int(row["head"]) for row in low_order[:per_layer]}
                for row in layer_rows:
                    head = int(row["head"])
                    offsets[(layer, head)] = (
                        1 if head in high else -1 if head in low else 0
                    )
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
                    "low_units": per_layer * 30,
                    "normal_units": (24 - 2 * per_layer) * 30,
                    "high_units": per_layer * 30,
                    "low_heads_per_layer": per_layer,
                    "high_heads_per_layer": per_layer,
                    "uniform_b4_mse": uniform,
                    "routed_b4_mse": routed,
                    "relative_mse_improvement_pct": 100.0 * (1.0 - routed / uniform),
                }
            )
            candidates.append((routed, fraction, offsets))
        _score, winner_fraction, winner = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        selected_offsets[name] = [
            [winner[(layer, head)] for head in range(24)]
            for layer in range(30)
        ]
        for row in summaries:
            if row["tensor"] == name:
                row["selected"] = bool(row["fraction"] == winner_fraction)
    routing = {
        "schema_version": codec.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "offsets": selected_offsets,
    }
    codec.validate_routing_table(routing)
    return routing, summaries


def fit_cross_k_high_v_low(
    rows: list[dict[str, Any]], fractions: tuple[float, ...]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Route one extra K bit against one fewer V bit inside every layer."""

    summaries: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, dict[str, list[list[int]]]]
    ] = []
    for fraction in fractions:
        per_layer = int(round(fraction * 24))
        offsets = {
            name: [[0 for _head in range(24)] for _layer in range(30)]
            for name in ("K", "V")
        }
        for layer in range(30):
            k_rows = [
                row
                for row in rows
                if row["tensor"] == "K" and int(row["layer"]) == layer
            ]
            v_rows = [
                row
                for row in rows
                if row["tensor"] == "V" and int(row["layer"]) == layer
            ]
            high_k = sorted(
                k_rows,
                key=lambda row: (
                    -float(row["gain_4_to_5"]),
                    int(row["head"]),
                ),
            )[:per_layer]
            low_v = sorted(
                v_rows,
                key=lambda row: (
                    float(row["cost_4_to_3"]),
                    int(row["head"]),
                ),
            )[:per_layer]
            for row in high_k:
                offsets["K"][layer][int(row["head"])] = 1
            for row in low_v:
                offsets["V"][layer][int(row["head"])] = -1

        uniform = sum(float(row["depth_4_mse"]) for row in rows) / len(rows)
        routed = sum(
            float(
                row[
                    f"depth_{4 + offsets[str(row['tensor'])][int(row['layer'])][int(row['head'])]}_mse"
                ]
            )
            for row in rows
        ) / len(rows)
        summaries.append(
            {
                "tensor": "K+V",
                "fraction": fraction,
                "low_units": per_layer * 30,
                "normal_units": (48 - 2 * per_layer) * 30,
                "high_units": per_layer * 30,
                "low_heads_per_layer": per_layer,
                "high_heads_per_layer": per_layer,
                "uniform_b4_mse": uniform,
                "routed_b4_mse": routed,
                "relative_mse_improvement_pct": 100.0 * (1.0 - routed / uniform),
            }
        )
        candidates.append((routed, fraction, offsets))
    _score, winner_fraction, winner = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    for row in summaries:
        row["selected"] = bool(row["fraction"] == winner_fraction)
    routing = {
        "schema_version": codec.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "budget_scope": "joint_kv",
        "offsets": winner,
    }
    codec.validate_routing_table(routing)
    return routing, summaries


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    config = {
        "data_root": str(args.data_root),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "device": args.device,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "key_sample": args.key_sample,
        "query_sample": args.query_sample,
        "prompt_repeats": args.prompt_repeats,
        "prompt_list": list(range(20)),
        "layers": list(range(args.layer_start, args.layer_end)),
        "heads": list(range(24)),
        "fractions": list(args.fractions),
        "input_marginals": (
            str(args.input_marginals) if args.input_marginals else None
        ),
        "allocation_mode": args.allocation_mode,
        "codec_family": args.codec_family,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "collect_only": args.collect_only,
        "input_marginal_shards": [
            str(item) for item in args.input_marginal_shards
        ],
        "marginal_column_prefix": args.marginal_column_prefix,
        "objective": {
            "K": "K-only attention output MSE",
            "V": "V-only attention output MSE",
        },
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    if args.input_marginal_shards:
        rows = []
        for path in args.input_marginal_shards:
            with path.open(encoding="utf-8", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        covered = sorted({int(row["layer"]) for row in rows})
        if covered != list(range(30)):
            raise ValueError(
                "input marginal shards must cover every layer exactly once"
            )
    elif args.input_marginals is None:
        rows = collect(args)
    else:
        with args.input_marginals.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if args.collect_only:
        marginals_path = args.output_root / "functional_unit_marginals.csv"
        write_csv(marginals_path, rows)
        codebook_paths = {
            "K": args.data_root / "nonorm_k_l8c64_codebooks",
            "V": args.data_root / "nonorm_v_l8c64_codebooks",
        }
        write_json(
            args.output_root / "run_info.json",
            {
                "git_commit": git_commit(),
                "config_path": str(config_path),
                "config_hash": sha256_file(config_path),
                "codebook_path": {
                    key: str(value) for key, value in codebook_paths.items()
                },
                "codebook_hash": {
                    key: stable_directory_hash(value)
                    for key, value in codebook_paths.items()
                },
                "hostname": socket.gethostname(),
                "gpu_type": gpu_type_or_unavailable(args.device),
                "seed": args.seed,
                "prompt_list": list(range(20)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "marginals_path": str(marginals_path),
                "marginals_hash": sha256_file(marginals_path),
                "implementation_path": str(
                    REPO_ROOT
                    / "subitkv/routed_midrange_scalar.py"
                ),
                "implementation_hash": sha256_file(
                    REPO_ROOT
                    / "subitkv/routed_midrange_scalar.py"
                ),
            },
        )
        return
    if args.marginal_column_prefix:
        prefix = args.marginal_column_prefix
        for row in rows:
            for metric in (
                "depth_3_mse",
                "depth_4_mse",
                "depth_5_mse",
                "gain_4_to_5",
                "cost_4_to_3",
            ):
                source = f"{prefix}{metric}"
                if source not in row:
                    raise ValueError(
                        f"marginal row is missing split column {source}"
                    )
                row[metric] = row[source]
    if args.allocation_mode == "cross_k_high_v_low":
        routing, summaries = fit_cross_k_high_v_low(rows, args.fractions)
    else:
        routing, summaries = fit_per_layer_balanced(rows, args.fractions)
    routing["fit"] = {
        "split": "train",
        "objective": {
            "K": "minimum K-only attention output MSE",
            "V": "minimum V-only attention output MSE",
        },
        "high_score": "largest depth4-to-depth5 functional gain",
        "low_score": "smallest depth4-to-depth3 functional cost",
        "fractions": list(args.fractions),
        "budget_constraint": "equal low/high head counts inside every layer",
        "allocation_mode": args.allocation_mode,
    }
    write_csv(args.output_root / "functional_unit_marginals.csv", rows)
    write_csv(args.output_root / "fraction_summary.csv", summaries)
    routing_path = args.output_root / "routing.json"
    write_json(routing_path, routing)
    codebook_paths = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": git_commit(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                key: str(value) for key, value in codebook_paths.items()
            },
            "codebook_hash": {
                key: stable_directory_hash(value)
                for key, value in codebook_paths.items()
            },
            "tree_params_path": str(args.tree_params),
            "tree_params_hash": sha256_file(args.tree_params),
            "routing_path": str(routing_path),
            "routing_hash": sha256_file(routing_path),
            "hostname": socket.gethostname(),
            "gpu_type": gpu_type_or_unavailable(args.device),
            "seed": args.seed,
            "prompt_list": list(range(20)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
