#!/usr/bin/env python3
"""Train HY-specific K Gaussian tree/routing and V progressive prefix tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from quant_videogen import routed_gaussian  # noqa: E402
from quant_videogen.hyworldplay_residual_stats import RESIDUAL_STATS_SCHEMA  # noqa: E402
from quant_videogen.progressive_sign_adaptive_absmax_scalar import PARAMS_SCHEMA  # noqa: E402
from quant_videogen.sign_adaptive_absmax_scalar import (  # noqa: E402
    encode_sign_adaptive_absmax_scalar_residual,
)
from run_truncatable_residual_codec_bench import fit_gaussian_tree  # noqa: E402


LAYERS = 54
HEADS = 16
DIM = 128
DEFAULT_PROMPTS = 20
MOMENTS = 8
ORIENTATIONS = 2
LEAVES = 16


def split_label(prompts: int) -> str:
    return "moviegen_train20" if int(prompts) == 20 else f"moviegen_train{int(prompts)}_fast_validation"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_layers(value: str) -> list[int]:
    layers = [int(item) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < LAYERS for layer in layers):
        raise ValueError("layers must be unique values in 0..53")
    return layers


def stats_path(root: Path, prompt: int, layer: int) -> Path:
    return root / "residual_stats" / "train" / str(prompt) / f"layer_{layer}.pt"


def load_stats(root: Path, prompt: int, layer: int) -> dict[str, Any]:
    path = stats_path(root, prompt, layer)
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if saved.get("schema_version") != RESIDUAL_STATS_SCHEMA:
        raise ValueError(f"{path} is not a HY residual-stats v2 file")
    return saved


def normalized_k_scalars(
    value: torch.Tensor,
    *,
    seed: int,
    samples_per_head: int,
    sample_seed: int,
) -> torch.Tensor:
    transformed = routed_gaussian.groupwise_hadamard(value.float(), 128, seed=seed)
    normalized = transformed / transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    rows = []
    for head in range(int(normalized.shape[0])):
        flat = normalized[head].reshape(-1).cpu()
        generator = torch.Generator().manual_seed(sample_seed + head * 97)
        take = min(int(samples_per_head), int(flat.numel()))
        rows.append(flat[torch.randperm(int(flat.numel()), generator=generator)[:take]])
    return torch.cat(rows).to(torch.float16)


def tree_worker(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda")
    values = []
    layers = parse_layers(args.layers)
    started = time.perf_counter()
    for layer in layers:
        for prompt in range(int(args.train_prompts)):
            saved = load_stats(args.root, prompt, layer)
            centered = saved["K_centered"].to(device=device, dtype=torch.bfloat16)
            values.append(
                normalized_k_scalars(
                    centered,
                    seed=args.codec_seed,
                    samples_per_head=args.scalars_per_head,
                    sample_seed=args.seed + prompt * 1_000_003 + layer * 1009,
                )
            )
        print(f"TREE_LAYER {layer}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "hyworldplay_rvq_k_tree_samples.worker.v2",
            "layers": layers,
            "train_prompts": int(args.train_prompts),
            "normalized_scalars": torch.cat(values),
            "wall_seconds": time.perf_counter() - started,
        },
        args.output,
    )


def tree_aggregate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    workers = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(args.worker_dir.glob("tree_worker_gpu*.pt"))]
    if not workers or sorted(layer for worker in workers for layer in worker["layers"]) != list(range(LAYERS)):
        raise RuntimeError("tree aggregation requires one complete worker layer partition")
    values = torch.cat([worker["normalized_scalars"] for worker in workers]).float()
    generator = torch.Generator().manual_seed(args.seed)
    if int(values.numel()) > args.fit_scalars:
        indices = torch.randperm(int(values.numel()), generator=generator)[: args.fit_scalars]
        values = values.index_select(0, indices)
    tree = fit_gaussian_tree(values, args.tree_iters)
    packed = {
        "schema_version": "hyworldplay_rvq_k_gaussian_tree.v2",
        "K": tree,
        "V": tree,
        "fit": {
            "split": split_label(args.train_prompts),
            "prompts": int(args.train_prompts),
            "test_prompts_read": False,
            "normalized_scalars": int(values.numel()),
            "codec_seed": args.codec_seed,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "tree_base_params.pt"
    torch.save(packed, output)
    write_create(args.output_dir / "summary.json", {
        "schema_version": "hyworldplay_rvq_k_gaussian_tree.summary.v2",
        "normalized_scalars": int(values.numel()),
        "fit_audit": tree["fit_audit"],
    })


def gaussian_depth_sse(value: torch.Tensor, tree: dict[str, tuple[torch.Tensor, ...]], seed: int) -> dict[int, tuple[float, int]]:
    transformed = routed_gaussian.groupwise_hadamard(value.float(), 128, seed=seed).reshape(-1, 128)
    scale = transformed.square().mean(-1, keepdim=True).add(1e-12).sqrt()
    scale = routed_gaussian._from_positive_e4m3_raw(
        routed_gaussian._to_positive_e4m3_raw(scale)
    ).clamp_min(torch.finfo(torch.float32).tiny)
    flat = (transformed / scale).reshape(-1)
    path = torch.zeros(flat.numel(), device=flat.device, dtype=torch.int64)
    paths = []
    for threshold in tree["thresholds"][:5]:
        path = path.bitwise_left_shift(1).bitwise_or((flat >= threshold[path].float()).long())
        paths.append(path.clone())
    output = {}
    for depth in range(1, 6):
        restored = tree["reconstruction"][depth - 1][paths[depth - 1]].float().reshape_as(transformed) * scale
        error = (transformed - restored).square()
        output[depth] = (float(error.double().sum().item()), int(error.numel()))
    return output


def unpack_codes(packed: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return torch.stack((packed >> 4, packed & 15), dim=-1).reshape(tuple(int(item) for item in shape))


def v_moments(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 3:
        source = value.unsqueeze(0)
    elif value.ndim == 4:
        source = value
    else:
        raise ValueError("V residual samples must be [H,S,D] or [B,H,S,D]")
    state = encode_sign_adaptive_absmax_scalar_residual(
        source, group_size=128, use_fast=source.is_cuda
    )
    codes = unpack_codes(state["packed"], source.shape).reshape(-1).long()
    signed_scale = state["scales"].float().reshape(*source.shape[:-1], 1)
    orientation = (signed_scale >= 0).expand_as(source).reshape(-1).long()
    normalized = (
        source.float()
        / signed_scale.abs().clamp_min(torch.finfo(torch.float32).tiny)
    ).reshape(-1)
    bucket = orientation * LEAVES + codes
    output = torch.zeros(MOMENTS, ORIENTATIONS * LEAVES, device=value.device, dtype=torch.float64)
    output[0] = torch.bincount(bucket, minlength=ORIENTATIONS * LEAVES).double()
    power = normalized.double()
    for moment in range(1, 5):
        output[moment] = torch.bincount(bucket, weights=power, minlength=ORIENTATIONS * LEAVES)
        power = power * normalized
    scale2 = signed_scale.abs().square().expand_as(source).reshape(-1).double()
    output[5] = torch.bincount(bucket, weights=scale2, minlength=ORIENTATIONS * LEAVES)
    output[6] = torch.bincount(bucket, weights=scale2 * normalized.double(), minlength=ORIENTATIONS * LEAVES)
    output[7] = torch.bincount(bucket, weights=scale2 * normalized.double().square(), minlength=ORIENTATIONS * LEAVES)
    return output.reshape(MOMENTS, ORIENTATIONS, LEAVES).cpu()


def fit_worker(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device("cuda")
    base = torch.load(args.tree, map_location=device, weights_only=False)
    tree = routed_gaussian.extend_gaussian_tree_to_depth5(base, tensor_name="K", device=device)
    layers = parse_layers(args.layers)
    units: dict[str, dict[str, list[float | int]]] = {
        f"{layer}:{head}": {str(depth): [0.0, 0] for depth in range(1, 6)}
        for layer in layers for head in range(HEADS)
    }
    moments = torch.zeros(int(args.train_prompts), MOMENTS, ORIENTATIONS, LEAVES, dtype=torch.float64)
    base_sse = {name: [0.0, 0] for name in ("K", "V")}
    started = time.perf_counter()
    for layer in layers:
        for prompt in range(int(args.train_prompts)):
            saved = load_stats(args.root, prompt, layer)
            k_value = saved["K_centered"].to(device=device, dtype=torch.bfloat16)
            v_value = saved["V_residual"].to(device=device, dtype=torch.bfloat16)
            for head in range(HEADS):
                rows = gaussian_depth_sse(k_value[head : head + 1], tree, args.codec_seed)
                for depth, (total, count) in rows.items():
                    units[f"{layer}:{head}"][str(depth)][0] += total
                    units[f"{layer}:{head}"][str(depth)][1] += count
            moments[prompt] += v_moments(v_value)
            for name in ("K", "V"):
                base_sse[name][0] += float(saved["base_sse"][name])
                base_sse[name][1] += int(saved["base_count"][name])
        print(f"FIT_LAYER {layer}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": "hyworldplay_rvq_residual_params.worker.v2",
        "layers": layers,
        "train_prompts": int(args.train_prompts),
        "units": units,
        "v_prompt_moments": moments,
        "base_sse": base_sse,
        "wall_seconds": time.perf_counter() - started,
    }, args.output)


def code_to_path(code: np.ndarray, order: Sequence[int]) -> np.ndarray:
    path = np.zeros_like(code, dtype=np.int64)
    for source_bit in order:
        path = (path << 1) | ((code >> int(source_bit)) & 1)
    return path


def fit_v_tables(moments: np.ndarray, order: Sequence[int]) -> tuple[np.ndarray, ...]:
    total = moments.sum(axis=0)
    path = code_to_path(np.arange(LEAVES, dtype=np.int64), order)
    tables = []
    for bits in (1, 2, 3):
        prefix = path >> (4 - bits)
        table = np.empty((ORIENTATIONS, 1 << bits), dtype=np.float64)
        for orientation, qmin in enumerate((-8.0, -7.0)):
            for index in range(1 << bits):
                selected = prefix == index
                count = total[5, orientation, selected].sum()
                value = total[6, orientation, selected].sum()
                table[orientation, index] = value / count if count > 0 else np.arange(16)[selected].mean() + qmin
        tables.append(table)
    return tuple(tables)


def evaluate_v_tables(moments: np.ndarray, order: Sequence[int], tables: Sequence[np.ndarray]) -> float:
    total = moments.sum(axis=0)
    path = code_to_path(np.arange(LEAVES, dtype=np.int64), order)
    error = 0.0
    for bits, table in zip((1, 2, 3), tables):
        prefix = path >> (4 - bits)
        for orientation in range(ORIENTATIONS):
            reconstruction = table[orientation, prefix]
            error += float(np.sum(
                total[7, orientation]
                - 2 * reconstruction * total[6, orientation]
                + reconstruction**2 * total[5, orientation]
            ))
    return error


def select_v_params(prompt_moments: torch.Tensor) -> tuple[tuple[int, ...], tuple[np.ndarray, ...], dict[str, Any]]:
    values = prompt_moments.numpy()
    prompt_count = int(values.shape[0])
    folds = min(5, prompt_count)
    if folds < 2:
        raise ValueError("V prefix fitting requires at least two training prompts")
    candidates = []
    for order in itertools.permutations(range(4)):
        fold_errors = []
        for fold in range(folds):
            test = np.arange(prompt_count) % folds == fold
            tables = fit_v_tables(values[~test], order)
            fold_errors.append(evaluate_v_tables(values[test], order, tables))
        candidates.append((sum(fold_errors), order, fold_errors))
    objective, order, fold_errors = min(candidates, key=lambda row: (row[0], row[1]))
    tables = fit_v_tables(values, order)
    return order, tables, {
        "cross_validated_error": objective,
        "five_fold_error": objective,
        "fold_errors": fold_errors,
        "cross_validation_folds": folds,
        "candidate_orders": len(candidates),
    }


def fit_aggregate(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    workers = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(args.worker_dir.glob("fit_worker_gpu*.pt"))]
    if not workers or sorted(layer for worker in workers for layer in worker["layers"]) != list(range(LAYERS)):
        raise RuntimeError("fit aggregation requires one complete worker layer partition")
    units = {key: value for worker in workers for key, value in worker["units"].items()}
    unit_mse = {
        key: {depth: float(total) / max(1, int(count)) for depth, (total, count) in table.items()}
        for key, table in units.items()
    }
    routing_candidates = []
    for fraction in (0.125, 0.25, 0.375):
        count = int(round(fraction * len(unit_mse)))
        high = set(sorted(unit_mse, key=lambda key: (-(unit_mse[key]["4"] - unit_mse[key]["5"]), key))[:count])
        remaining = [key for key in unit_mse if key not in high]
        low = set(sorted(remaining, key=lambda key: (unit_mse[key]["3"] - unit_mse[key]["4"], key))[:count])
        offsets = {key: 1 if key in high else -1 if key in low else 0 for key in unit_mse}
        error = sum(unit_mse[key][str(4 + offsets[key])] for key in unit_mse) / len(unit_mse)
        routing_candidates.append((error, fraction, offsets))
    routing_error, fraction, offsets = min(routing_candidates, key=lambda row: (row[0], row[1]))
    routing = {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": LAYERS,
        "num_heads": HEADS,
        "offsets": {
            "K": [[int(offsets[f"{layer}:{head}"]) for head in range(HEADS)] for layer in range(LAYERS)],
            "V": [[0] * HEADS for _ in range(LAYERS)],
        },
        "budget_scope": "per_tensor",
        "fit": {"split": split_label(args.train_prompts), "prompts": int(args.train_prompts), "test_prompts_read": False, "fraction": fraction},
    }
    routed_gaussian.validate_routing_table(routing)
    prompt_moments = sum((worker["v_prompt_moments"] for worker in workers), torch.zeros_like(workers[0]["v_prompt_moments"]))
    order, tables, v_cv = select_v_params(prompt_moments)
    v_params = {
        "schema_version": PARAMS_SCHEMA,
        "methods": {
            "midpoint_msb": {
                "bit_order": tuple(int(value) for value in order),
                "reconstruction": tuple(torch.tensor(table, dtype=torch.float16) for table in tables),
                "objective": f"train{int(args.train_prompts)} {v_cv['cross_validation_folds']}-fold scale-weighted residual MSE",
            }
        },
        "fit": {"split": split_label(args.train_prompts), "prompts": int(args.train_prompts), "test_prompts_read": False, **v_cv},
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_create(args.output_dir / "routing.json", routing)
    tree_base = torch.load(args.tree, map_location="cpu", weights_only=False)
    torch.save({
        "schema_version": "routed_gaussian.tree_params.v1",
        "K": routed_gaussian.extend_gaussian_tree_to_depth5(tree_base, tensor_name="K"),
        "V": routed_gaussian.extend_gaussian_tree_to_depth5(tree_base, tensor_name="V"),
        "fit": {"split": split_label(args.train_prompts), "prompts": int(args.train_prompts), "test_prompts_read": False},
    }, args.output_dir / "tree_params.pt")
    torch.save(v_params, args.output_dir / "v_progressive_params.pt")
    write_create(args.output_dir / "summary.json", {
        "schema_version": "hyworldplay_rvq_residual_params.summary.v2",
        "K_routing_fraction": fraction,
        "K_routed_b4_mse": routing_error,
        "V_bit_order": list(order),
        "V_cross_validated_error": v_cv["cross_validated_error"],
        "V_five_fold_error": v_cv["five_fold_error"],
    })
    write_create(args.output_dir / "run_info.json", {
        "git_commit": subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "dirty_state": subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True).splitlines(),
        "config_path": str((args.output_dir / "summary.json").resolve()),
        "config_hash": sha256_file(args.output_dir / "summary.json"),
        "codebook_path": str(args.root.resolve()),
        "codebook_hash": sha256_file(args.root / "manifest.json"),
        "hostname": socket.gethostname(),
        "gpu_type": "NVIDIA RTX 6000 Ada Generation",
        "seed": args.seed,
        "prompt_list": [line.strip() for line in (REPO_ROOT / "assets/moviegen_train20.txt").read_text(encoding="utf-8").splitlines() if line.strip()][: int(args.train_prompts)],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common_worker = argparse.ArgumentParser(add_help=False)
    common_worker.add_argument("--root", type=Path, required=True)
    common_worker.add_argument("--layers", required=True)
    common_worker.add_argument("--output", type=Path, required=True)
    common_worker.add_argument("--seed", type=int, default=42)
    common_worker.add_argument("--codec-seed", type=int, default=20260722)
    common_worker.add_argument("--train-prompts", type=int, default=DEFAULT_PROMPTS)
    tree_w = sub.add_parser("tree-worker", parents=[common_worker])
    tree_w.add_argument("--scalars-per-head", type=int, default=256)
    tree_a = sub.add_parser("tree-aggregate")
    tree_a.add_argument("--worker-dir", type=Path, required=True)
    tree_a.add_argument("--output-dir", type=Path, required=True)
    tree_a.add_argument("--fit-scalars", type=int, default=1_000_000)
    tree_a.add_argument("--tree-iters", type=int, default=4)
    tree_a.add_argument("--seed", type=int, default=42)
    tree_a.add_argument("--codec-seed", type=int, default=20260722)
    tree_a.add_argument("--train-prompts", type=int, default=DEFAULT_PROMPTS)
    fit_w = sub.add_parser("fit-worker", parents=[common_worker])
    fit_w.add_argument("--tree", type=Path, required=True)
    fit_a = sub.add_parser("fit-aggregate")
    fit_a.add_argument("--root", type=Path, required=True)
    fit_a.add_argument("--worker-dir", type=Path, required=True)
    fit_a.add_argument("--tree", type=Path, required=True)
    fit_a.add_argument("--output-dir", type=Path, required=True)
    fit_a.add_argument("--seed", type=int, default=42)
    fit_a.add_argument("--train-prompts", type=int, default=DEFAULT_PROMPTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "tree-worker":
        tree_worker(args)
    elif args.command == "tree-aggregate":
        tree_aggregate(args)
    elif args.command == "fit-worker":
        fit_worker(args)
    else:
        fit_aggregate(args)


if __name__ == "__main__":
    main()
