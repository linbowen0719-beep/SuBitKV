#!/usr/bin/env python3
"""Analyze KMeans outliers in saved full-precision HY-WorldPlay KV caches."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import torch

from subitkv.kmeans.kmeans_euclid import batch_kmeans_Euclid


QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.995, 0.999)


def parse_csv_ints(value: str, available: Iterable[int] | None = None) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        if available is None:
            raise ValueError("'all' requires available values")
        return list(available)
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def chunk_sort_key(path: Path) -> int:
    match = re.match(r"chunk_(\d+)_", path.name)
    if not match:
        return -1
    return int(match.group(1))


def select_chunks(cache_dir: Path, selector: str) -> list[Path]:
    chunks = sorted(
        [p for p in cache_dir.iterdir() if p.is_dir() and p.name.startswith("chunk_")],
        key=chunk_sort_key,
    )
    if not chunks:
        raise FileNotFoundError(f"No chunk_* directories under {cache_dir}")

    selector = selector.strip().lower()
    if selector == "all":
        return chunks

    selected: list[Path] = []
    for part in selector.split(","):
        part = part.strip()
        if part == "first":
            selected.append(chunks[0])
        elif part in {"middle", "mid"}:
            selected.append(chunks[len(chunks) // 2])
        elif part == "last":
            selected.append(chunks[-1])
        elif part:
            idx = int(part)
            match = [p for p in chunks if chunk_sort_key(p) == idx]
            if not match:
                raise FileNotFoundError(f"Requested chunk index {idx} not found")
            selected.append(match[0])

    deduped: list[Path] = []
    seen: set[Path] = set()
    for item in selected:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


def tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().flatten().cpu()
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            **{f"p{q * 100:g}": float("nan") for q in QUANTILES},
        }
    q = torch.quantile(values, torch.tensor(QUANTILES, dtype=torch.float32))
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        **{f"p{QUANTILES[i] * 100:g}": float(q[i].item()) for i in range(len(QUANTILES))},
    }


def robust_outlier_rate(values: torch.Tensor, z_threshold: float = 6.0) -> float:
    flat = values.detach().float().flatten()
    median = torch.quantile(flat, 0.5)
    mad = torch.quantile((flat - median).abs(), 0.5)
    if float(mad.item()) == 0.0:
        return 0.0
    robust_z = 0.6745 * (flat - median).abs() / mad
    return float((robust_z > z_threshold).float().mean().item())


def cluster_summary(cluster_sizes: torch.Tensor, seq_len: int) -> dict[str, float]:
    sizes = cluster_sizes.detach().cpu().to(torch.int64)
    nonempty = sizes[sizes > 0].float()
    out = {
        "empty_clusters": int((sizes == 0).sum().item()),
        "singleton_clusters": int((sizes == 1).sum().item()),
        "clusters_le2": int((sizes <= 2).sum().item()),
        "clusters_le5": int((sizes <= 5).sum().item()),
        "max_cluster_size": int(sizes.max().item()),
        "max_cluster_fraction": float(sizes.max().item() / seq_len),
    }
    if nonempty.numel():
        out.update(
            {
                "min_nonempty_cluster_size": int(nonempty.min().item()),
                "median_nonempty_cluster_size": float(nonempty.median().item()),
                "mean_nonempty_cluster_size": float(nonempty.mean().item()),
            }
        )
    else:
        out.update(
            {
                "min_nonempty_cluster_size": 0,
                "median_nonempty_cluster_size": 0.0,
                "mean_nonempty_cluster_size": 0.0,
            }
        )
    return out


def corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) == 0.0:
        return float("nan")
    return float((a * b).sum().div(denom).item())


def collect_top(
    metric: torch.Tensor,
    norm: torch.Tensor,
    dist: torch.Tensor,
    rel: torch.Tensor,
    cluster_ids: torch.Tensor,
    cluster_sizes: torch.Tensor,
    *,
    top_k: int,
    chunk_name: str,
    layer_idx: int,
    kv_name: str,
    token_start: int,
) -> list[dict[str, float | int | str]]:
    heads, seq_len = metric.shape
    k = min(top_k, metric.numel())
    vals, flat_idx = torch.topk(metric.flatten(), k=k)
    rows: list[dict[str, float | int | str]] = []
    for rank, idx_t in enumerate(flat_idx.detach().cpu().tolist(), start=1):
        head = idx_t // seq_len
        token = idx_t % seq_len
        cid = int(cluster_ids[head, token].item())
        rows.append(
            {
                "rank_within_group": rank,
                "chunk": chunk_name,
                "layer": layer_idx,
                "kv": kv_name,
                "head": head,
                "token_in_chunk": token,
                "global_token": int(token_start + token),
                "cluster_id": cid,
                "cluster_size": int(cluster_sizes[head, cid].item()),
                "norm": float(norm[head, token].item()),
                "dist": float(dist[head, token].item()),
                "rel": float(rel[head, token].item()),
                "metric": float(vals[rank - 1].item()),
            }
        )
    return rows


def analyze_group(
    x_bhsd: torch.Tensor,
    *,
    clusters: int,
    max_iters: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int]:
    if x_bhsd.ndim != 4 or x_bhsd.shape[0] != 1:
        raise ValueError(f"Expected [1,H,S,D], got {tuple(x_bhsd.shape)}")

    _, heads, seq_len, dim = x_bhsd.shape
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x = x_bhsd.squeeze(0).contiguous().to(device)
    cluster_ids, centroids, cluster_sizes, iters = batch_kmeans_Euclid(
        x.reshape(heads, seq_len, dim),
        n_clusters=clusters,
        max_iters=max_iters,
        tol=0.0,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    labels = cluster_ids.long()
    assigned = torch.gather(
        centroids.float(),
        dim=1,
        index=labels[..., None].expand(-1, -1, dim),
    )
    x_f = x.float()
    residual = x_f - assigned
    dist = torch.linalg.vector_norm(residual, dim=-1)
    norm = torch.linalg.vector_norm(x_f, dim=-1).clamp_min(1e-12)
    rel = dist / norm

    return {
        "cluster_ids": labels.detach().cpu(),
        "cluster_sizes": cluster_sizes.detach().cpu(),
        "norm": norm.detach().cpu(),
        "dist": dist.detach().cpu(),
        "rel": rel.detach().cpu(),
    }, int(iters)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        type=Path,
        required=True,
        help="Directory containing chunk_* folders, usually .../rank_000.",
    )
    parser.add_argument("--chunks", default="first,middle,last")
    parser.add_argument("--layers", default="0,15,29")
    parser.add_argument("--clusters", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=2)
    parser.add_argument("--top_per_group", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks = select_chunks(args.cache_dir, args.chunks)
    available_layers = sorted(
        int(p.stem.split("_")[-1])
        for p in chunks[0].glob("layer_*.pt")
        if p.stem.split("_")[-1].isdigit()
    )
    layers = parse_csv_ints(args.layers, available_layers)

    group_rows: list[dict[str, object]] = []
    head_rows: list[dict[str, object]] = []
    top_dist_rows: list[dict[str, object]] = []
    top_rel_rows: list[dict[str, object]] = []

    for chunk_path in chunks:
        for layer_idx in layers:
            layer_path = chunk_path / f"layer_{layer_idx:02d}.pt"
            payload = torch.load(layer_path, map_location="cpu", weights_only=False)
            token_start = int(payload.get("token_start", 0))
            token_end = int(payload.get("token_end", 0))
            layout = payload.get("layout", "BHSD")
            if layout != "BHSD":
                raise ValueError(f"{layer_path} layout is {layout}, expected BHSD")

            for kv_name in ("k", "v"):
                source = payload[kv_name]
                result, iters = analyze_group(
                    source,
                    clusters=args.clusters,
                    max_iters=args.max_iters,
                    device=device,
                    seed=args.seed + chunk_sort_key(chunk_path) * 1000 + layer_idx * 10 + (kv_name == "v"),
                )

                norm = result["norm"]
                dist = result["dist"]
                rel = result["rel"]
                cluster_ids = result["cluster_ids"]
                cluster_sizes = result["cluster_sizes"]
                heads, seq_len = norm.shape

                row: dict[str, object] = {
                    "chunk": chunk_path.name,
                    "layer": layer_idx,
                    "kv": kv_name,
                    "dtype": str(source.dtype).replace("torch.", ""),
                    "shape": "x".join(str(v) for v in source.shape),
                    "token_start": token_start,
                    "token_end": token_end,
                    "heads": heads,
                    "seq_len": seq_len,
                    "dim": source.shape[-1],
                    "clusters": args.clusters,
                    "kmeans_iters": iters,
                    "norm_dist_corr": corrcoef(norm, dist),
                    "norm_rel_corr": corrcoef(norm, rel),
                    "dist_robust_outlier_rate_z6": robust_outlier_rate(dist),
                    "rel_robust_outlier_rate_z6": robust_outlier_rate(rel),
                }
                for prefix, values in (("norm", norm), ("dist", dist), ("rel", rel)):
                    for key, value in tensor_stats(values).items():
                        row[f"{prefix}_{key}"] = value
                row.update(cluster_summary(cluster_sizes, seq_len))
                group_rows.append(row)

                for head in range(heads):
                    hrow: dict[str, object] = {
                        "chunk": chunk_path.name,
                        "layer": layer_idx,
                        "kv": kv_name,
                        "head": head,
                        "token_start": token_start,
                        "token_end": token_end,
                    }
                    for prefix, values in (
                        ("norm", norm[head]),
                        ("dist", dist[head]),
                        ("rel", rel[head]),
                    ):
                        for key, value in tensor_stats(values).items():
                            hrow[f"{prefix}_{key}"] = value
                    hrow.update(cluster_summary(cluster_sizes[head : head + 1], seq_len))
                    head_rows.append(hrow)

                top_dist_rows.extend(
                    collect_top(
                        dist,
                        norm,
                        dist,
                        rel,
                        cluster_ids,
                        cluster_sizes,
                        top_k=args.top_per_group,
                        chunk_name=chunk_path.name,
                        layer_idx=layer_idx,
                        kv_name=kv_name,
                        token_start=token_start,
                    )
                )
                top_rel_rows.extend(
                    collect_top(
                        rel,
                        norm,
                        dist,
                        rel,
                        cluster_ids,
                        cluster_sizes,
                        top_k=args.top_per_group,
                        chunk_name=chunk_path.name,
                        layer_idx=layer_idx,
                        kv_name=kv_name,
                        token_start=token_start,
                    )
                )
                print(
                    f"done {chunk_path.name} layer {layer_idx:02d} {kv_name}: "
                    f"dist_p99.5={row['dist_p99.5']:.4f} rel_p99.5={row['rel_p99.5']:.4f} "
                    f"empty={row['empty_clusters']}"
                )

    top_dist_rows = sorted(top_dist_rows, key=lambda r: float(r["dist"]), reverse=True)
    top_rel_rows = sorted(top_rel_rows, key=lambda r: float(r["rel"]), reverse=True)

    write_csv(args.out_dir / "groups.csv", group_rows)
    write_csv(args.out_dir / "heads.csv", head_rows)
    write_csv(args.out_dir / "top_dist_outliers.csv", top_dist_rows)
    write_csv(args.out_dir / "top_rel_outliers.csv", top_rel_rows)

    summary = {
        "cache_dir": str(args.cache_dir),
        "chunks": [p.name for p in chunks],
        "layers": layers,
        "clusters": args.clusters,
        "max_iters": args.max_iters,
        "device": str(device),
        "num_groups": len(group_rows),
        "outputs": {
            "groups": str(args.out_dir / "groups.csv"),
            "heads": str(args.out_dir / "heads.csv"),
            "top_dist_outliers": str(args.out_dir / "top_dist_outliers.csv"),
            "top_rel_outliers": str(args.out_dir / "top_rel_outliers.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
