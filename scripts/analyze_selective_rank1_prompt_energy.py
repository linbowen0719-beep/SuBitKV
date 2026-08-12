#!/usr/bin/env python3
"""Measure per-prompt energy explained by fixed sparse K rank-1 directions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from quant_videogen.rvq import (
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from quant_videogen.selective_rank1_gaussian import load_params


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples-per-prompt", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    args.params = args.params.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    params = load_params(args.params, device=device)
    prompt_list = list(range(20 if args.split == "train" else 10))
    rows = []
    layers = sorted(set(int(item) for item in params["layer_indices"].tolist()))
    for layer in layers:
        selected = params["layer_indices"] == layer
        heads = params["head_indices"][selected].long()
        basis = params["basis"][selected].float()
        for prompt in prompt_list:
            path = (
                DATA / "kv_dump" / args.split / str(prompt) / f"layer_{layer}.pt"
            )
            payload = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True
            )
            value = payload["K"]
            flat = value.permute(1, 0, 2, 3).reshape(24, -1, 128)
            generator = torch.Generator().manual_seed(
                args.seed + prompt * 1_000_003 + layer * 1009
            )
            ids = torch.randperm(flat.shape[1], generator=generator)[
                : args.samples_per_prompt
            ]
            source = (
                flat[:, ids]
                .unsqueeze(0)
                .to(device=device, dtype=torch.bfloat16)
                .contiguous()
            )
            packed = rvq_encode_tensor_cache(
                source,
                layer_idx=layer,
                codebook_dir=DATA / "nonorm_k_l8c64_codebooks",
                batch_size=4096,
                compute_dtype="bf16",
                codebook_storage_dtype="int4",
                scale_storage_dtype="uint4",
            )
            residual = (
                source - rvq_dequantize_tensor_cache(packed, torch.bfloat16)
            ).float()[0, heads]
            coefficient = torch.einsum("hsd,hd->hs", residual, basis)
            explained = coefficient.square().mean(-1)
            total = residual.square().mean(dim=(1, 2)).clamp_min(1e-30)
            ratio = explained / (128.0 * total)
            for index, head in enumerate(heads.tolist()):
                rows.append(
                    {
                        "split": args.split,
                        "prompt": prompt,
                        "layer": layer,
                        "head": head,
                        "explained_energy_fraction": float(ratio[index]),
                        "residual_mse": float(total[index]),
                    }
                )
            print(
                f"split={args.split} prompt={prompt} layer={layer} complete",
                flush=True,
            )
    csv_path = args.output_root / "prompt_energy.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "schema_version": "selective_rank1_prompt_energy.config.v1",
        "params": str(args.params),
        "params_hash": sha256_file(args.params),
        "split": args.split,
        "prompt_list": prompt_list,
        "layers": layers,
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "uses_query": False,
        "uses_clustering": False,
        "selection_use": args.split == "train",
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": str(DATA / "nonorm_k_l8c64_codebooks"),
            "codebook_hash": (
                "83501131ea25e074731b4932c6aa7b79ab96d42372934934"
                "f345050a63c172dc"
            ),
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": prompt_list,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "csv_path": str(csv_path),
            "csv_hash": sha256_file(csv_path),
        },
    )


if __name__ == "__main__":
    main()
