#!/usr/bin/env python3
"""Train OSCAR-style K/V rotations from QuantVideo pre-RoPE Q/K/V dumps.

Example:

python scripts/train_subitkv_oscar_rotations.py \
  --dump-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/train \
  --output-dir /data/bowen/t2v_kv_bench_moviegen_20train_10test/oscar \
  --sample-tokens-per-prompt 4096 \
  --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.oscar import build_hadamard, compose_rotation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--layers", default="0-29")
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--sample-tokens-per-prompt", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--composition", default="r_h_pbr")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_layers(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(dict.fromkeys(out))


def prompt_dirs(dump_dir: Path) -> list[Path]:
    dirs = [p for p in dump_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(dirs, key=lambda p: int(p.name))


def flatten_thnd(x: torch.Tensor) -> torch.Tensor:
    # Dump layout is [T, H, N, D]; analysis wants token-major [T*N, H, D].
    return x.permute(0, 2, 1, 3).reshape(x.shape[0] * x.shape[2], x.shape[1], x.shape[3])


def sample_indices(total: int, sample: int, *, seed: int, layer: int, prompt_id: int, device: torch.device) -> torch.Tensor:
    if sample <= 0 or sample >= total:
        return torch.arange(total, device=device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + layer * 1000003 + prompt_id * 9176)
    idx = torch.randperm(total, generator=gen)[:sample]
    return idx.to(device=device)


def empty_result(objective: str, composition: str) -> dict[str, Any]:
    return {
        "format_version": 1,
        "objective": objective,
        "source_grouping": "layer",
        "composition": composition,
        "layers": {},
    }


def add_layer(result: dict[str, Any], layer_id: int, rotation: torch.Tensor, eigvals: torch.Tensor) -> None:
    result["layers"][int(layer_id)] = {
        "layer_id": int(layer_id),
        "rotation": rotation.float().cpu().contiguous(),
        "eigenvalues": eigvals.float().cpu().contiguous(),
    }


@torch.no_grad()
def train_layer(
    dump_dir: Path,
    prompt_paths: list[Path],
    layer: int,
    head_dim: int,
    sample_tokens_per_prompt: int,
    seed: int,
    device: torch.device,
    hadamard: torch.Tensor,
    composition: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    q_cov_heads: torch.Tensor | None = None
    q_counts: torch.Tensor | None = None
    metadata: dict[str, Any] = {"layer": layer, "prompts": len(prompt_paths), "sample_rows": 0}

    for pdir in prompt_paths:
        payload = torch.load(pdir / f"layer_{layer}.pt", map_location="cpu", weights_only=False)
        q = flatten_thnd(payload["Q"]).to(device=device, dtype=torch.float32)
        idx = sample_indices(q.shape[0], sample_tokens_per_prompt, seed=seed, layer=layer, prompt_id=int(pdir.name), device=device)
        q = q.index_select(0, idx)
        heads = q.shape[1]
        if q_cov_heads is None:
            q_cov_heads = torch.zeros(heads, head_dim, head_dim, device=device, dtype=torch.float64)
            q_counts = torch.zeros(heads, device=device, dtype=torch.float64)
        qd = q.double()
        for h in range(heads):
            qh = qd[:, h, :]
            q_cov_heads[h] += qh.T @ qh
            q_counts[h] += qh.shape[0]
        metadata["sample_rows"] += int(q.shape[0] * heads)
        del q, qd, payload

    assert q_cov_heads is not None and q_counts is not None
    q_cov_heads = q_cov_heads / q_counts[:, None, None].clamp(min=1)
    q_cov = q_cov_heads.mean(dim=0)
    q_cov = (q_cov + q_cov.T) / 2
    eigvals_k, eigvecs_k = torch.linalg.eigh(q_cov)
    k_rotation = compose_rotation(eigvecs_k.cpu(), eigvals_k.cpu(), hadamard, composition)

    v_cov_heads = torch.zeros_like(q_cov_heads)
    v_weight_sums = torch.zeros_like(q_counts)
    for pdir in prompt_paths:
        payload = torch.load(pdir / f"layer_{layer}.pt", map_location="cpu", weights_only=False)
        k = flatten_thnd(payload["K"]).to(device=device, dtype=torch.float32)
        v = flatten_thnd(payload["V"]).to(device=device, dtype=torch.float32)
        idx = sample_indices(k.shape[0], sample_tokens_per_prompt, seed=seed, layer=layer, prompt_id=int(pdir.name), device=device)
        k = k.index_select(0, idx).double()
        v = v.index_select(0, idx).double()
        for h in range(k.shape[1]):
            kh = k[:, h, :]
            vh = v[:, h, :]
            weights = (kh @ q_cov_heads[h] * kh).sum(dim=1).clamp(min=0)
            v_cov_heads[h] += vh.T @ (vh * weights[:, None])
            v_weight_sums[h] += weights.sum()
        del k, v, payload

    v_cov_heads = v_cov_heads / v_weight_sums[:, None, None].clamp(min=1e-12)
    v_cov = v_cov_heads.mean(dim=0)
    v_cov = (v_cov + v_cov.T) / 2
    eigvals_v, eigvecs_v = torch.linalg.eigh(v_cov)
    v_rotation = compose_rotation(eigvecs_v.cpu(), eigvals_v.cpu(), hadamard, composition)

    return k_rotation, eigvals_k.cpu(), v_rotation, eigvals_v.cpu(), metadata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    k_path = args.output_dir / f"k_rotation_qqt_{args.composition}.pt"
    v_path = args.output_dir / f"v_rotation_sst_{args.composition}.pt"
    meta_path = args.output_dir / "training_metadata.json"
    if k_path.exists() and v_path.exists() and not args.overwrite:
        print(f"Rotations already exist: {k_path} {v_path}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    layers = parse_layers(args.layers)
    prompts = prompt_dirs(args.dump_dir)
    if not prompts:
        raise FileNotFoundError(f"No prompt dirs in {args.dump_dir}")

    hadamard = build_hadamard(args.head_dim)
    k_result = empty_result("qqt", args.composition)
    v_result = empty_result("sst", args.composition)
    metadata: dict[str, Any] = {
        "dump_dir": str(args.dump_dir),
        "layers": layers,
        "sample_tokens_per_prompt": args.sample_tokens_per_prompt,
        "seed": args.seed,
        "device": str(device),
        "composition": args.composition,
        "layer_stats": [],
    }
    start = time.perf_counter()
    for layer in layers:
        layer_start = time.perf_counter()
        k_rot, k_ev, v_rot, v_ev, layer_meta = train_layer(
            args.dump_dir,
            prompts,
            layer,
            args.head_dim,
            args.sample_tokens_per_prompt,
            args.seed,
            device,
            hadamard,
            args.composition,
        )
        add_layer(k_result, layer, k_rot, k_ev)
        add_layer(v_result, layer, v_rot, v_ev)
        layer_meta["seconds"] = time.perf_counter() - layer_start
        metadata["layer_stats"].append(layer_meta)
        torch.save(k_result, args.output_dir / f"k_rotation_qqt_{args.composition}.partial.pt")
        torch.save(v_result, args.output_dir / f"v_rotation_sst_{args.composition}.partial.pt")
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(
            f"layer {layer:02d}: {layer_meta['seconds']:.2f}s "
            f"sample_rows={layer_meta['sample_rows']}",
            flush=True,
        )

    metadata["seconds"] = time.perf_counter() - start
    torch.save(k_result, k_path)
    torch.save(v_result, v_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {k_path}", flush=True)
    print(f"Saved {v_path}", flush=True)
    print(f"Total {metadata['seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
