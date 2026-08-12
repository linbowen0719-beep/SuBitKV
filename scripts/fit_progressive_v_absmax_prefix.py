#!/usr/bin/env python3
"""Fit train-only scalar prefix tables for progressive V absmax INT4.

The B4 leaf assignment is the deployed sign-adaptive G128 INT4 codec.  This
script only learns (1) the transmission order of its four code bits and
(2) orientation-specific scalar reconstruction values for incomplete
prefixes.  Twenty train prompts are cross-validated; test prompts are never
read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from quant_videogen import progressive_sign_adaptive_absmax_scalar as codec  # noqa: E402
from quant_videogen.sign_adaptive_absmax_scalar import (  # noqa: E402
    encode_sign_adaptive_absmax_scalar_residual,
)
from run_fixed_budget_rvq_screen import layer_path, parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


SCHEMA = "progressive_v_absmax.prefix_fit.v1"
DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
# 0..4 are sum(1,u,u^2,u^3,u^4); 5..7 are
# sum(scale^2, scale^2*u, scale^2*u^2).
MOMENTS = 8
ORIENTATIONS = 2
LEAVES = 16


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts", default="0:20")
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--key-sample", type=int, default=880)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tail-alphas", default="4,16")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.prompts = tuple(parse_ints(args.prompts))
    args.layers = tuple(parse_ints(args.layers))
    args.tail_alphas = tuple(
        float(item) for item in args.tail_alphas.split(",") if item
    )
    if args.prompts != tuple(range(20)):
        parser.error("formal fit requires train prompts 0..19")
    if not args.layers or any(not 0 <= item < 30 for item in args.layers):
        parser.error("layers must lie in [0,29]")
    if args.key_sample < 1:
        parser.error("key-sample must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("shard-id must lie in [0,num-shards)")
    if args.folds < 2 or len(args.prompts) % args.folds:
        parser.error("folds must divide the prompt count")
    if any(item <= 0 for item in args.tail_alphas):
        parser.error("tail alphas must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def task_list(args: argparse.Namespace) -> list[tuple[int, int]]:
    return [
        (prompt, layer)
        for prompt in args.prompts
        for layer in args.layers
    ]


def unpack_codes(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return torch.stack((packed >> 4, packed & 15), dim=-1).reshape(
        tuple(int(item) for item in shape)
    )


@torch.no_grad()
def task_moments(
    args: argparse.Namespace,
    *,
    prompt: int,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    payload = torch.load(
        str(layer_path(args.data_root, "train", prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    residuals: list[torch.Tensor] = []
    for head in range(24):
        flat = payload["V"][:, head].reshape(-1, 128)
        generator = torch.Generator().manual_seed(
            args.seed + prompt * 1_000_003 + layer * 1009 + head * 97
        )
        indices = deterministic_indices(
            len(flat), args.key_sample, generator, "inference_span"
        )
        original = flat[indices].to(device=device, dtype=torch.float32)
        _base, residual = rvq_residual(
            runtime_args,
            original,
            load_int4_base_codebook(
                runtime_args, "V", layer, head, device
            ),
            device,
        )
        residuals.append(residual)
    residual = torch.stack(residuals).unsqueeze(0).to(torch.bfloat16)
    state = encode_sign_adaptive_absmax_scalar_residual(
        residual, group_size=128, use_fast=True
    )
    codes = unpack_codes(state["packed"], residual.shape).reshape(-1).long()
    signed_scale = state["scales"].float().reshape(
        1, 24, args.key_sample, 1
    )
    orientation = (signed_scale >= 0).expand_as(residual).reshape(-1).long()
    normalized = (
        residual.float()
        / signed_scale.abs().clamp_min(torch.finfo(torch.float32).tiny)
    ).reshape(-1)
    bucket = orientation * LEAVES + codes
    output = torch.empty(
        MOMENTS, ORIENTATIONS * LEAVES, device=device, dtype=torch.float64
    )
    output[0] = torch.bincount(
        bucket, minlength=ORIENTATIONS * LEAVES
    ).to(torch.float64)
    power = normalized.to(torch.float64)
    for moment in range(1, MOMENTS):
        output[moment] = torch.bincount(
            bucket,
            weights=power,
            minlength=ORIENTATIONS * LEAVES,
        )
        power = power * normalized
    scale2 = (
        signed_scale.float().abs().square().expand_as(residual).reshape(-1)
    ).to(torch.float64)
    output[5] = torch.bincount(
        bucket,
        weights=scale2,
        minlength=ORIENTATIONS * LEAVES,
    )
    output[6] = torch.bincount(
        bucket,
        weights=scale2 * normalized.to(torch.float64),
        minlength=ORIENTATIONS * LEAVES,
    )
    output[7] = torch.bincount(
        bucket,
        weights=scale2 * normalized.to(torch.float64).square(),
        minlength=ORIENTATIONS * LEAVES,
    )
    return output.reshape(MOMENTS, ORIENTATIONS, LEAVES).cpu()


def worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("runtime-faithful RVQ statistics require CUDA")
    torch.cuda.set_device(device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    stats_dir = args.output_root / "worker_stats"
    stats_dir.mkdir(exist_ok=True)
    output = stats_dir / (
        f"worker_{args.shard_id:03d}_of_{args.num_shards:03d}.pt"
    )
    if output.exists():
        raise FileExistsError(output)
    assigned = [
        task
        for index, task in enumerate(task_list(args))
        if index % args.num_shards == args.shard_id
    ]
    by_prompt = torch.zeros(
        len(args.prompts),
        MOMENTS,
        ORIENTATIONS,
        LEAVES,
        dtype=torch.float64,
    )
    for prompt, layer in assigned:
        by_prompt[prompt] += task_moments(
            args, prompt=prompt, layer=layer, device=device
        )
        print(f"[done] prompt={prompt} layer={layer}", flush=True)
    torch.save(
        {
            "schema_version": f"{SCHEMA}.worker",
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
            "tasks": assigned,
            "prompt_moments": by_prompt,
        },
        output,
    )
    codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    write_json(
        args.output_root
        / "run_info"
        / f"worker_{args.shard_id:03d}_of_{args.num_shards:03d}.json",
        {
            "schema_version": f"{SCHEMA}.worker_run_info",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": None,
            "config_hash": None,
            "codebook_path": {"V": str(codebook)},
            "codebook_hash": {"V": stable_directory_hash(codebook)},
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": sorted({prompt for prompt, _layer in assigned}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats_path": str(output),
            "stats_hash": sha256_file(output),
            "tasks": assigned,
        },
    )


def code_to_path(code: np.ndarray, order: Sequence[int]) -> np.ndarray:
    path = np.zeros_like(code, dtype=np.int64)
    for source_bit in order:
        path = (path << 1) | ((code >> int(source_bit)) & 1)
    return path


def fit_tables(
    moments: np.ndarray,
    *,
    order: Sequence[int],
    alpha: float,
    scale_weighted: bool = False,
) -> tuple[np.ndarray, ...]:
    total = moments.sum(axis=0)
    code = np.arange(LEAVES, dtype=np.int64)
    path = code_to_path(code, order)
    result: list[np.ndarray] = []
    for bits in (1, 2, 3):
        prefix = path >> (4 - bits)
        table = np.empty((ORIENTATIONS, 1 << bits), dtype=np.float64)
        for orientation, qmin in enumerate((-8.0, -7.0)):
            for index in range(1 << bits):
                selected = prefix == index
                if scale_weighted:
                    weighted_count = total[5, orientation, selected].sum()
                    weighted_sum = total[6, orientation, selected].sum()
                else:
                    count = total[0, orientation, selected].sum()
                    weighted_count = count + alpha * (
                        total[2, orientation, selected].sum() / 64.0
                    )
                    weighted_sum = (
                        total[1, orientation, selected].sum()
                        + alpha
                        * total[3, orientation, selected].sum()
                        / 64.0
                    )
                if weighted_count > 0:
                    table[orientation, index] = weighted_sum / weighted_count
                else:
                    table[orientation, index] = (
                        code[selected].mean() + qmin
                    )
        result.append(table)
    return tuple(result)


def evaluate_tables(
    moments: np.ndarray,
    *,
    order: Sequence[int],
    tables: Sequence[np.ndarray],
    alpha: float,
    scale_weighted: bool = False,
) -> tuple[float, tuple[float, float, float]]:
    total = moments.sum(axis=0)
    code = np.arange(LEAVES, dtype=np.int64)
    path = code_to_path(code, order)
    ratios: list[float] = []
    for bits, table in zip((1, 2, 3), tables):
        prefix = path >> (4 - bits)
        error = baseline = 0.0
        for orientation in range(ORIENTATIONS):
            reconstruction = table[orientation, prefix]
            count = total[0, orientation]
            sum1 = total[1, orientation]
            sum2 = total[2, orientation]
            if scale_weighted:
                weight_count = total[5, orientation]
                weight_sum1 = total[6, orientation]
                weight_sum2 = total[7, orientation]
                error += np.sum(
                    weight_sum2
                    - 2 * reconstruction * weight_sum1
                    + reconstruction**2 * weight_count
                )
                baseline += np.sum(weight_sum2)
            elif alpha == 0:
                error += np.sum(
                    sum2 - 2 * reconstruction * sum1 + reconstruction**2 * count
                )
                baseline += np.sum(sum2)
            else:
                sum3 = total[3, orientation]
                sum4 = total[4, orientation]
                weight_count = count + alpha * sum2 / 64.0
                weight_sum1 = sum1 + alpha * sum3 / 64.0
                weight_sum2 = sum2 + alpha * sum4 / 64.0
                error += np.sum(
                    weight_sum2
                    - 2 * reconstruction * weight_sum1
                    + reconstruction**2 * weight_count
                )
                baseline += np.sum(weight_sum2)
        ratios.append(float(error / max(baseline, 1e-30)))
    return float(np.mean(ratios)), tuple(ratios)


def aggregate(args: argparse.Namespace) -> None:
    stats_paths = sorted((args.output_root / "worker_stats").glob("worker_*.pt"))
    if len(stats_paths) != args.num_shards:
        raise ValueError(
            f"expected {args.num_shards} worker stats, found {len(stats_paths)}"
        )
    prompt_moments = torch.zeros(
        len(args.prompts),
        MOMENTS,
        ORIENTATIONS,
        LEAVES,
        dtype=torch.float64,
    )
    tasks: list[tuple[int, int]] = []
    for path in stats_paths:
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("schema_version") != f"{SCHEMA}.worker":
            raise ValueError(f"invalid worker stats: {path}")
        prompt_moments += value["prompt_moments"]
        tasks.extend(tuple(item) for item in value["tasks"])
    if sorted(tasks) != task_list(args) or len(tasks) != len(set(tasks)):
        raise ValueError("worker task coverage is incomplete or duplicated")
    raw = prompt_moments.numpy()
    fold_sets = [
        set(args.prompts[fold :: args.folds]) for fold in range(args.folds)
    ]
    orders = tuple(itertools.permutations((0, 1, 2, 3)))
    objectives = (("mse_cv", 0.0, False), ("scale2_mse_cv", 0.0, True)) + tuple(
        (f"tail{int(alpha)}_cv", alpha, False) for alpha in args.tail_alphas
    )
    cv_rows: list[dict[str, Any]] = []
    methods: dict[str, Any] = {
        "midpoint_msb": codec.build_midpoint_params()["methods"]["midpoint"]
    }
    for method, alpha, scale_weighted in objectives:
        order_scores: dict[tuple[int, ...], list[float]] = {
            order: [] for order in orders
        }
        for fold, heldout in enumerate(fold_sets):
            training = np.stack(
                [raw[prompt] for prompt in args.prompts if prompt not in heldout]
            )
            validation = np.stack([raw[prompt] for prompt in sorted(heldout)])
            for order in orders:
                tables = fit_tables(
                    training,
                    order=order,
                    alpha=alpha,
                    scale_weighted=scale_weighted,
                )
                score, depth_scores = evaluate_tables(
                    validation,
                    order=order,
                    tables=tables,
                    alpha=alpha,
                    scale_weighted=scale_weighted,
                )
                order_scores[order].append(score)
                cv_rows.append(
                    {
                        "method": method,
                        "alpha": alpha,
                        "scale_weighted": scale_weighted,
                        "bit_order": "".join(map(str, order)),
                        "fold": fold,
                        "heldout_prompts": ",".join(map(str, sorted(heldout))),
                        "objective_ratio_mean_B1_B3": score,
                        "B1_ratio": depth_scores[0],
                        "B2_ratio": depth_scores[1],
                        "B3_ratio": depth_scores[2],
                    }
                )
        means = {
            order: float(np.mean(values))
            for order, values in order_scores.items()
        }
        standard = (3, 2, 1, 0)
        best = min(means, key=lambda order: (means[order], order))
        best_se = float(
            np.std(order_scores[best], ddof=1) / math.sqrt(args.folds)
        )
        selected = (
            standard
            if means[standard] <= means[best] + best_se
            else best
        )
        tables = fit_tables(
            raw,
            order=selected,
            alpha=alpha,
            scale_weighted=scale_weighted,
        )
        methods[method] = {
            "bit_order": selected,
            "reconstruction": tuple(
                torch.tensor(table, dtype=torch.float16) for table in tables
            ),
            "objective": (
                "conditional source-space mean weighted by block scale^2"
                if scale_weighted
                else "conditional normalized-source mean"
                if alpha == 0
                else f"conditional weighted mean, weight=1+{alpha}*u^2/64"
            ),
            "cv_objective_ratio": means[selected],
            "best_order": best,
            "best_order_ratio": means[best],
            "best_order_se": best_se,
        }
        for row in cv_rows:
            if row["method"] == method:
                row["selected_order"] = row["bit_order"] == "".join(
                    map(str, selected)
                )
    params = {
        "schema_version": codec.PARAMS_SCHEMA,
        "methods": methods,
        "fit": {
            "schema_version": SCHEMA,
            "split": "train",
            "prompt_ids": list(args.prompts),
            "layers": list(args.layers),
            "key_sample": args.key_sample,
            "folds": args.folds,
            "test_prompts_read": False,
            "uses_clustering": False,
            "B4_leaf_codec": "sign_adaptive_absmax_scalar",
            "B4_reconstruction_changed": False,
            "worker_stats_hashes": {
                str(path): sha256_file(path) for path in stats_paths
            },
        },
    }
    params_path = args.output_root / "params.pt"
    if params_path.exists():
        raise FileExistsError(params_path)
    torch.save(params, params_path)
    cv_path = args.output_root / "bit_order_cv.csv"
    if cv_path.exists():
        raise FileExistsError(cv_path)
    with cv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_rows[0]))
        writer.writeheader()
        writer.writerows(cv_rows)
    config = {
        "schema_version": SCHEMA,
        "data_root": str(args.data_root),
        "prompts": args.prompts,
        "layers": args.layers,
        "key_sample": args.key_sample,
        "num_shards": args.num_shards,
        "folds": args.folds,
        "tail_alphas": args.tail_alphas,
        "seed": args.seed,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    write_json(
        args.output_root / "run_info.json",
        {
            "schema_version": f"{SCHEMA}.run_info",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {"V": str(codebook)},
            "codebook_hash": {"V": stable_directory_hash(codebook)},
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "params_path": str(params_path),
            "params_hash": sha256_file(params_path),
            "cv_path": str(cv_path),
            "cv_hash": sha256_file(cv_path),
        },
    )
    write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "files": {
                "params.pt": sha256_file(params_path),
                "bit_order_cv.csv": sha256_file(cv_path),
                "config.json": sha256_file(config_path),
            },
            "worker_stats": {
                str(path): sha256_file(path) for path in stats_paths
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "worker":
        worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
