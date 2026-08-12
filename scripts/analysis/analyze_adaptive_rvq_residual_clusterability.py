#!/usr/bin/env python3
"""Held-out clusterability audit for raw KV and Adaptive-RVQ residuals.

The experiment separates two notions that are often conflated:

* online C256: centroids are fitted on the very test chunk being encoded;
* train-transfer C256: centroids are fitted only on the 20 training prompts
  and frozen before the 10 test prompts are read.

Both are compared with the per-head test mean, and all predictor side bytes are
reported for the one-chunk (3520 token), D128 production boundary.
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
from typing import Any, Iterable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant_videogen.kmeans.euclid_assign import euclid_assign_triton
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
from quant_videogen.rvq import rvq_dequantize_tensor_cache
from scripts.benchmark_temporal_codec_optimizations import encode_base


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_ROUTING = REPO / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json"
DEFAULT_TREE = REPO / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt"
DEFAULT_V_PARAMS = REPO / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt"
DIM = 128
HEADS = 24
TOKENS = 3520
FRAMES = 4
CENTROIDS = 256


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paths(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        k_codebooks=args.data_root / "nonorm_k_l16c64_codebooks",
        v_codebooks=args.data_root / "nonorm_v_l16c64_codebooks",
        routing=args.routing,
        tree_params=args.tree,
        v_params=args.v_params,
        fast_approximate_rvq=False,
    )


def load_chunk(
    data_root: Path,
    split: str,
    prompt: int,
    layer: int,
    chunk: int,
    tensor_name: str,
    device: torch.device,
) -> torch.Tensor:
    source_path = data_root / "kv_dump" / split / str(prompt) / f"layer_{layer}.pt"
    raw = torch.load(str(source_path), map_location="cpu", weights_only=False, mmap=True)[tensor_name]
    start = int(chunk) * FRAMES
    value = (
        raw[start : start + FRAMES]
        .permute(1, 0, 2, 3)
        .reshape(1, raw.shape[1], TOKENS, DIM)
        .contiguous()
    )
    if value.dtype != torch.bfloat16 or tuple(value.shape) != (1, HEADS, TOKENS, DIM):
        raise ValueError(f"unexpected tensor {source_path} {tensor_name}: {value.shape} {value.dtype}")
    return value.to(device=device).contiguous()


@torch.no_grad()
def adaptive_residual(
    source: torch.Tensor,
    tensor_name: str,
    layer: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, int]:
    state = encode_base(source, tensor_name, layer, paths(args))
    restored = rvq_dequantize_tensor_cache(state, torch.bfloat16)
    return (source - restored).contiguous(), int(state["num_levels"])


def deterministic_indices(count: int, total: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:count]


def task_seed(args: argparse.Namespace, *items: Any) -> int:
    digest = hashlib.sha256(
        (str(args.seed) + ":" + ":".join(map(str, items))).encode()
    ).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def fit_tasks(args: argparse.Namespace) -> list[tuple[int, int, str]]:
    return [
        (layer, chunk, tensor_name)
        for layer in args.layers
        for chunk in args.chunks
        for tensor_name in ("K", "V")
    ]


def test_tasks(args: argparse.Namespace) -> list[tuple[int, int, int, str]]:
    return [
        (prompt, layer, chunk, tensor_name)
        for prompt in args.test_prompts
        for layer in args.layers
        for chunk in args.chunks
        for tensor_name in ("K", "V")
    ]


def initialize(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    config = {
        "schema_version": "adaptive_rvq_clusterability.config.v1",
        "data_root": str(args.data_root),
        "source_position": "pre_rope_pre_prope_D128",
        "train_prompts": list(args.train_prompts),
        "test_prompts": list(args.test_prompts),
        "layers": list(args.layers),
        "chunks": list(args.chunks),
        "tokens_per_chunk": TOKENS,
        "heads": HEADS,
        "dim": DIM,
        "centroids": CENTROIDS,
        "fit_samples_per_prompt": args.fit_samples_per_prompt,
        "fit_kmeans_iters": args.fit_kmeans_iters,
        "online_kmeans_iters": args.online_kmeans_iters,
        "intrinsic_samples": args.intrinsic_samples,
        "rvq": "Adaptive L4-16, two consecutive drops below 6.29%, INT4 codebook",
        "predictors": ["test_mean", "online_C256", "train_transfer_C256"],
        "seed": args.seed,
    }
    write_json(args.output_root / "config.json", config)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "script_path": str(Path(__file__).resolve()),
            "script_hash": sha256_file(Path(__file__).resolve()),
            "config_path": str(args.output_root / "config.json"),
            "config_hash": stable_hash(config),
            "codebook_path": {
                name: str(args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks")
                for name in ("K", "V")
            },
            "codebook_hash": {
                name: sha256_tree(args.data_root / f"nonorm_{name.lower()}_l16c64_codebooks")
                for name in ("K", "V")
            },
            "input_hashes": {
                "routing": sha256_file(args.routing),
                "tree_params": sha256_file(args.tree),
                "v_params": sha256_file(args.v_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "seed": args.seed,
            "prompt_list": {"train": list(args.train_prompts), "test": list(args.test_prompts)},
            "timestamp": utc_now(),
            "status": "initialized",
        },
    )


@torch.no_grad()
def fit_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    assigned = [
        task for index, task in enumerate(fit_tasks(args))
        if index % args.num_shards == args.shard_id
    ]
    worker_rows: list[dict[str, Any]] = []
    for task_index, (layer, chunk, tensor_name) in enumerate(assigned, start=1):
        raw_samples: list[torch.Tensor] = []
        residual_samples: list[torch.Tensor] = []
        depths: list[int] = []
        for prompt in args.train_prompts:
            source = load_chunk(
                args.data_root, "train", prompt, layer, chunk, tensor_name, device
            )
            residual, depth = adaptive_residual(source, tensor_name, layer, args)
            indices = deterministic_indices(
                args.fit_samples_per_prompt,
                TOKENS,
                task_seed(args, "fit_sample", prompt, layer, chunk, tensor_name),
            ).to(device)
            raw_samples.append(source[0, :, indices].contiguous())
            residual_samples.append(residual[0, :, indices].contiguous())
            depths.append(depth)
            del source, residual
        domains = {
            "raw": torch.cat(raw_samples, dim=1).contiguous(),
            "adaptive_rvq_residual": torch.cat(residual_samples, dim=1).contiguous(),
        }
        payload: dict[str, Any] = {
            "layer": layer,
            "chunk": chunk,
            "tensor": tensor_name,
            "train_prompts": tuple(args.train_prompts),
            "rvq_depth_mean": float(np.mean(depths)),
            "rvq_depth_min": int(min(depths)),
            "rvq_depth_max": int(max(depths)),
            "domains": {},
        }
        for domain, values in domains.items():
            torch.manual_seed(task_seed(args, "fit_kmeans", layer, chunk, tensor_name, domain))
            torch.cuda.manual_seed_all(task_seed(args, "fit_kmeans_cuda", layer, chunk, tensor_name, domain))
            ids, centroids, sizes, iterations = batch_kmeans_Euclid(
                values,
                n_clusters=CENTROIDS,
                max_iters=args.fit_kmeans_iters,
            )
            payload["domains"][domain] = {
                "centroids": centroids.to(torch.bfloat16).cpu().contiguous(),
                "mean": values.float().mean(dim=1).to(torch.bfloat16).cpu().contiguous(),
                "fit_iterations": int(iterations),
                "fit_dead_fraction": float((sizes == 0).float().mean()),
                "fit_mse": float(
                    (
                        values.float()
                        - torch.gather(
                            centroids.float(), 1,
                            ids[..., None].expand(-1, -1, DIM),
                        )
                    ).square().mean()
                ),
            }
        output = args.output_root / "centroids" / f"layer_{layer:02d}_chunk_{chunk:02d}_{tensor_name}.pt"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(output)
        torch.save(payload, output)
        worker_rows.append({
            "layer": layer,
            "chunk": chunk,
            "tensor": tensor_name,
            "rvq_depth_mean": payload["rvq_depth_mean"],
            "centroid_path": str(output),
            "centroid_hash": sha256_file(output),
        })
        print(
            f"fit [{task_index}/{len(assigned)}] layer={layer} chunk={chunk} tensor={tensor_name}",
            flush=True,
        )
    output = args.output_root / "fit_workers" / f"worker_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    write_csv(output, worker_rows)


def occupancy(ids: torch.Tensor) -> dict[str, float]:
    perplexities: list[float] = []
    dead: list[float] = []
    max_share: list[float] = []
    for head in range(ids.shape[0]):
        counts = torch.bincount(ids[head].long(), minlength=CENTROIDS).float()
        probabilities = counts / counts.sum().clamp_min(1.0)
        used = probabilities > 0
        entropy = -(probabilities[used] * probabilities[used].log()).sum()
        perplexities.append(float(entropy.exp() / CENTROIDS))
        dead.append(float((counts == 0).float().mean()))
        max_share.append(float(probabilities.max()))
    return {
        "normalized_perplexity": float(np.mean(perplexities)),
        "dead_code_fraction": float(np.mean(dead)),
        "largest_cluster_fraction": float(np.mean(max_share)),
    }


def intrinsic_geometry(values: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    sample = values[0, :, indices].float()
    centered = sample - sample.mean(dim=1, keepdim=True)
    covariance = torch.matmul(centered.transpose(1, 2), centered) / max(1, sample.shape[1] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(-1)
    total = eigenvalues.sum(-1).clamp_min(1e-30)
    probabilities = eigenvalues / total[:, None]
    participation = total.square() / eigenvalues.square().sum(-1).clamp_min(1e-30)
    effective_rank = torch.exp(
        -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
    )
    return {
        "participation_ratio": float(participation.mean()),
        "effective_rank": float(effective_rank.mean()),
        "cov_top1_fraction": float((eigenvalues[:, :1].sum(-1) / total).mean()),
        "cov_top8_fraction": float((eigenvalues[:, :8].sum(-1) / total).mean()),
        "cov_top16_fraction": float((eigenvalues[:, :16].sum(-1) / total).mean()),
        "cov_top32_fraction": float((eigenvalues[:, :32].sum(-1) / total).mean()),
    }


def predictor_metrics(
    values: torch.Tensor,
    ids: torch.Tensor,
    centroids: torch.Tensor,
    mean_mse: float,
) -> dict[str, float]:
    restored = torch.gather(
        centroids.float(), 1, ids[..., None].expand(-1, -1, DIM)
    )
    mse = float((values.float() - restored).square().mean())
    ratio = mse / max(mean_mse, 1e-30)
    return {
        "predictor_mse": mse,
        "distortion_ratio_vs_test_mean": ratio,
        "gain_vs_test_mean_percent": (1.0 - ratio) * 100.0,
        "equivalent_scalar_bits": -0.5 * math.log2(max(ratio, 1e-30)),
        **occupancy(ids),
    }


@torch.no_grad()
def test_worker(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    assigned = [
        task for index, task in enumerate(test_tasks(args))
        if index % args.num_shards == args.shard_id
    ]
    rows: list[dict[str, Any]] = []
    for task_index, (prompt, layer, chunk, tensor_name) in enumerate(assigned, start=1):
        source = load_chunk(
            args.data_root, "test", prompt, layer, chunk, tensor_name, device
        )
        residual, depth = adaptive_residual(source, tensor_name, layer, args)
        fitted = torch.load(
            args.output_root / "centroids" / f"layer_{layer:02d}_chunk_{chunk:02d}_{tensor_name}.pt",
            map_location="cpu", weights_only=False,
        )
        geometry_indices = deterministic_indices(
            min(args.intrinsic_samples, TOKENS), TOKENS,
            task_seed(args, "geometry", prompt, layer, chunk, tensor_name),
        ).to(device)
        for domain, values in (
            ("raw", source),
            ("adaptive_rvq_residual", residual),
        ):
            flattened = values.reshape(HEADS, TOKENS, DIM).contiguous()
            test_mean = flattened.float().mean(dim=1, keepdim=True)
            source_mse = float(flattened.float().square().mean())
            mean_mse = float((flattened.float() - test_mean).square().mean())
            geometry = intrinsic_geometry(values, geometry_indices)
            torch.manual_seed(task_seed(args, "online", prompt, layer, chunk, tensor_name, domain))
            torch.cuda.manual_seed_all(task_seed(args, "online_cuda", prompt, layer, chunk, tensor_name, domain))
            online_ids, online_centroids, _sizes, online_iterations = batch_kmeans_Euclid(
                flattened,
                n_clusters=CENTROIDS,
                max_iters=args.online_kmeans_iters,
            )
            transfer_centroids = fitted["domains"][domain]["centroids"].to(device).contiguous()
            transfer_ids = euclid_assign_triton(
                flattened,
                transfer_centroids,
                (flattened.float() * flattened.float()).sum(-1),
            )
            common = {
                "prompt": prompt,
                "layer": layer,
                "chunk": chunk,
                "tensor": tensor_name,
                "domain": domain,
                "rvq_depth": depth if domain == "adaptive_rvq_residual" else 0,
                "source_mean_square": source_mse,
                "test_mean_mse": mean_mse,
                "centroid_count": CENTROIDS,
                "predictor_payload_bits_per_value_bf16": (
                    (CENTROIDS * DIM * 16 + TOKENS * 8) / (TOKENS * DIM)
                ),
                **geometry,
            }
            rows.append({
                **common,
                "predictor": "online_C256",
                "kmeans_iterations": int(online_iterations),
                **predictor_metrics(
                    flattened, online_ids, online_centroids, mean_mse
                ),
            })
            rows.append({
                **common,
                "predictor": "train_transfer_C256",
                "kmeans_iterations": int(fitted["domains"][domain]["fit_iterations"]),
                **predictor_metrics(
                    flattened, transfer_ids, transfer_centroids, mean_mse
                ),
            })
        print(
            f"test [{task_index}/{len(assigned)}] prompt={prompt} layer={layer} chunk={chunk} tensor={tensor_name}",
            flush=True,
        )
    output = args.output_root / "test_workers" / f"worker_{args.shard_id:02d}_of_{args.num_shards:02d}.csv"
    write_csv(output, rows)


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def aggregate(args: argparse.Namespace) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    paths_found = sorted((args.output_root / "test_workers").glob("worker_*.csv"))
    if len(paths_found) != args.num_shards:
        raise ValueError(f"expected {args.num_shards} test workers, found {len(paths_found)}")
    raw = pd.DataFrame(read_rows(paths_found))
    numeric = [column for column in raw.columns if column not in {"tensor", "domain", "predictor"}]
    raw[numeric] = raw[numeric].apply(pd.to_numeric)
    raw.to_csv(args.output_root / "clusterability_per_unit.csv", index=False)
    metrics = [
        "source_mean_square", "test_mean_mse", "predictor_mse",
        "distortion_ratio_vs_test_mean", "gain_vs_test_mean_percent",
        "equivalent_scalar_bits", "normalized_perplexity", "dead_code_fraction",
        "largest_cluster_fraction", "participation_ratio", "effective_rank",
        "cov_top1_fraction", "cov_top8_fraction", "cov_top16_fraction",
        "cov_top32_fraction", "predictor_payload_bits_per_value_bf16", "rvq_depth",
    ]
    summary = raw.groupby(["tensor", "domain", "predictor"], as_index=False)[metrics].mean()
    summary["equivalent_bits_per_predictor_bit"] = (
        summary.equivalent_scalar_bits
        / summary.predictor_payload_bits_per_value_bf16
    )
    summary.to_csv(args.output_root / "clusterability_summary.csv", index=False)
    by_layer = raw.groupby(["layer", "tensor", "domain", "predictor"], as_index=False)[metrics].mean()
    by_layer.to_csv(args.output_root / "clusterability_by_layer.csv", index=False)
    by_chunk = raw.groupby(["chunk", "tensor", "domain", "predictor"], as_index=False)[metrics].mean()
    by_chunk.to_csv(args.output_root / "clusterability_by_chunk.csv", index=False)

    depth_units = raw[
        (raw.domain == "adaptive_rvq_residual")
        & (raw.predictor == "online_C256")
    ][["prompt", "layer", "chunk", "tensor", "rvq_depth"]].copy()
    depth_units.to_csv(args.output_root / "adaptive_rvq_depth_per_unit.csv", index=False)
    depth_stability = depth_units.groupby(
        ["tensor", "layer", "chunk"]
    )["rvq_depth"].agg(["mean", "std", "min", "max"]).reset_index()
    depth_stability["range"] = depth_stability["max"] - depth_stability["min"]
    depth_stability.to_csv(args.output_root / "adaptive_rvq_depth_stability.csv", index=False)
    depth_stability.groupby("tensor", as_index=False).agg(
        mean_depth=("mean", "mean"),
        mean_across_prompt_std=("std", "mean"),
        mean_across_prompt_range=("range", "mean"),
        fraction_layer_chunk_units_with_nonzero_range=("range", lambda x: float((x > 0).mean())),
        fraction_layer_chunk_units_with_range_gt2=("range", lambda x: float((x > 2).mean())),
    ).to_csv(args.output_root / "adaptive_rvq_depth_stability_summary.csv", index=False)

    paired_rows: list[dict[str, Any]] = []
    index = raw.set_index(["prompt", "layer", "chunk", "tensor", "domain", "predictor"])
    for predictor in ("online_C256", "train_transfer_C256"):
        for tensor in ("K", "V"):
            for prompt in args.test_prompts:
                raw_values = raw[
                    (raw.prompt == prompt) & (raw.tensor == tensor)
                    & (raw.domain == "raw") & (raw.predictor == predictor)
                ].gain_vs_test_mean_percent.mean()
                residual_values = raw[
                    (raw.prompt == prompt) & (raw.tensor == tensor)
                    & (raw.domain == "adaptive_rvq_residual") & (raw.predictor == predictor)
                ].gain_vs_test_mean_percent.mean()
                paired_rows.append({
                    "predictor": predictor,
                    "tensor": tensor,
                    "prompt": prompt,
                    "raw_gain_percent": raw_values,
                    "residual_gain_percent": residual_values,
                    "residual_minus_raw_gain_pp": residual_values - raw_values,
                })
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.output_root / "rvq_clusterability_effect_per_prompt.csv", index=False)
    rng = np.random.default_rng(args.seed)
    bootstrap_rows: list[dict[str, Any]] = []
    for (predictor, tensor), group in paired.groupby(["predictor", "tensor"]):
        values = group.residual_minus_raw_gain_pp.to_numpy(float)
        samples = np.array([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(10000)
        ])
        bootstrap_rows.append({
            "predictor": predictor,
            "tensor": tensor,
            "prompts": len(values),
            "mean_residual_minus_raw_gain_pp": values.mean(),
            "prompt_wins_residual_more_clusterable": int((values > 0).sum()),
            "bootstrap_ci95_low": np.quantile(samples, 0.025),
            "bootstrap_ci95_high": np.quantile(samples, 0.975),
        })
    pd.DataFrame(bootstrap_rows).to_csv(
        args.output_root / "rvq_clusterability_effect_bootstrap.csv", index=False
    )
    gain_prompt = raw.groupby(
        ["prompt", "tensor", "domain", "predictor"], as_index=False
    ).gain_vs_test_mean_percent.mean()
    gain_bootstrap_rows: list[dict[str, Any]] = []
    for (tensor, domain, predictor), group in gain_prompt.groupby(
        ["tensor", "domain", "predictor"]
    ):
        values = group.gain_vs_test_mean_percent.to_numpy(float)
        samples = np.array([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(10000)
        ])
        gain_bootstrap_rows.append({
            "tensor": tensor,
            "domain": domain,
            "predictor": predictor,
            "prompts": len(values),
            "mean_gain_percent": values.mean(),
            "prompt_positive_gain_fraction": float((values > 0).mean()),
            "bootstrap_ci95_low": np.quantile(samples, 0.025),
            "bootstrap_ci95_high": np.quantile(samples, 0.975),
        })
    pd.DataFrame(gain_bootstrap_rows).to_csv(
        args.output_root / "predictor_gain_bootstrap.csv", index=False
    )

    labels = {
        ("raw", "online_C256"): "Raw / online",
        ("adaptive_rvq_residual", "online_C256"): "RVQ residual / online",
        ("raw", "train_transfer_C256"): "Raw / train-transfer",
        ("adaptive_rvq_residual", "train_transfer_C256"): "RVQ residual / train-transfer",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
    for axis, tensor in zip(axes, ("K", "V")):
        selected = summary[summary.tensor == tensor]
        values = []
        names = []
        for key, label in labels.items():
            row = selected[(selected.domain == key[0]) & (selected.predictor == key[1])].iloc[0]
            values.append(row.gain_vs_test_mean_percent)
            names.append(label)
        axis.bar(np.arange(len(values)), values, color=["#4c78a8", "#72b7b2", "#f58518", "#eeca3b"])
        axis.set_xticks(np.arange(len(values)), names, rotation=24, ha="right")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("MSE reduction vs per-head mean (%)")
        axis.set_title(f"{tensor}: C256 cluster gain")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"clusterability_gain.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    styles = {
        ("raw", "online_C256"): ("Raw / online", "-"),
        ("adaptive_rvq_residual", "online_C256"): ("RVQ residual / online", "-"),
        ("raw", "train_transfer_C256"): ("Raw / train-transfer", "--"),
        ("adaptive_rvq_residual", "train_transfer_C256"): ("RVQ residual / train-transfer", "--"),
    }
    for axis, tensor in zip(axes, ("K", "V")):
        for (domain, predictor), (label, linestyle) in styles.items():
            selected = by_layer[
                (by_layer.tensor == tensor) & (by_layer.domain == domain)
                & (by_layer.predictor == predictor)
            ].sort_values("layer")
            axis.plot(
                selected.layer, selected.gain_vs_test_mean_percent,
                linestyle=linestyle, marker="o", markersize=2.5, label=label,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel("Transformer layer")
        axis.set_ylabel("MSE reduction vs per-head mean (%)")
        axis.set_title(tensor)
    axes[1].legend(fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"clusterability_by_layer.{suffix}", dpi=180)
    plt.close(fig)

    geometry = (
        raw.drop_duplicates(["prompt", "layer", "chunk", "tensor", "domain"])
        .groupby(["tensor", "domain"], as_index=False)[
            ["participation_ratio", "effective_rank", "cov_top16_fraction"]
        ].mean()
    )
    geometry.to_csv(args.output_root / "intrinsic_geometry_summary.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), constrained_layout=True)
    for axis, tensor in zip(axes, ("K", "V")):
        selected = geometry[geometry.tensor == tensor].set_index("domain")
        vals = [selected.loc["raw", "effective_rank"], selected.loc["adaptive_rvq_residual", "effective_rank"]]
        axis.bar(["Raw KV", "Adaptive-RVQ residual"], vals, color=["#4c78a8", "#e45756"])
        axis.set_ylabel("Covariance effective rank (maximum 128)")
        axis.set_title(tensor)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"intrinsic_rank.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.0), constrained_layout=True)
    for axis, tensor in zip(axes, ("K", "V")):
        table = depth_stability[depth_stability.tensor == tensor].pivot(
            index="layer", columns="chunk", values="mean"
        ).sort_index()
        image = axis.imshow(table.to_numpy(), aspect="auto", cmap="viridis", vmin=4, vmax=16)
        axis.set_xticks(range(len(table.columns)), [str(item) for item in table.columns])
        axis.set_yticks(range(len(table.index)), [str(item) for item in table.index])
        axis.set_xlabel("Chunk index")
        axis.set_ylabel("Transformer layer")
        axis.set_title(f"{tensor}: mean Adaptive-RVQ depth")
    fig.colorbar(image, ax=axes, label="RVQ stages")
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_root / f"adaptive_rvq_depth_heatmap.{suffix}", dpi=180)
    plt.close(fig)

    complete = json.loads((args.output_root / "run_info.json").read_text())
    output_names = [
        "clusterability_per_unit.csv", "clusterability_summary.csv",
        "clusterability_by_layer.csv", "clusterability_by_chunk.csv",
        "adaptive_rvq_depth_per_unit.csv", "adaptive_rvq_depth_stability.csv",
        "adaptive_rvq_depth_stability_summary.csv",
        "rvq_clusterability_effect_per_prompt.csv",
        "rvq_clusterability_effect_bootstrap.csv", "intrinsic_geometry_summary.csv",
        "predictor_gain_bootstrap.csv",
        "clusterability_gain.png", "clusterability_gain.pdf",
        "clusterability_by_layer.png", "clusterability_by_layer.pdf",
        "intrinsic_rank.png", "intrinsic_rank.pdf",
        "adaptive_rvq_depth_heatmap.png", "adaptive_rvq_depth_heatmap.pdf",
    ]
    complete.update({
        "status": "complete",
        "completed_timestamp": utc_now(),
        "output_hashes": {
            name: sha256_file(args.output_root / name) for name in output_names
        },
    })
    write_json(args.output_root / "run_info.complete.json", complete)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create", "fit", "worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--train-prompts", type=parse_ints, default=parse_ints(",".join(map(str, range(20)))))
    parser.add_argument("--test-prompts", type=parse_ints, default=parse_ints(",".join(map(str, range(10)))))
    parser.add_argument("--layers", type=parse_ints, default=parse_ints("0,10,20,29"))
    parser.add_argument("--chunks", type=parse_ints, default=parse_ints("0,4,9"))
    parser.add_argument("--fit-samples-per-prompt", type=int, default=384)
    parser.add_argument("--fit-kmeans-iters", type=int, default=10)
    parser.add_argument("--online-kmeans-iters", type=int, default=2)
    parser.add_argument("--intrinsic-samples", type=int, default=1024)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--v-params", type=Path, default=DEFAULT_V_PARAMS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()
    for name in ("output_root", "data_root", "routing", "tree", "v_params"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main() -> None:
    args = parse_args()
    if args.command == "create":
        initialize(args)
    elif args.command == "fit":
        fit_worker(args)
    elif args.command == "worker":
        test_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
