#!/usr/bin/env python3
"""Audit per-stage utilization of the real Wan2.2 L16C64 RVQ codebooks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT / "scripts"))

from run_fixed_budget_rvq_screen import (  # noqa: E402
    DEFAULT_DATA,
    layer_path,
    load_baseline,
    nearest,
    parse_ints,
    quantize_codebook_int4,
)

CARDINALITY = 64
DIM = 128
DEPTH = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("eval", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--output-root", type=Path,
        default=DEFAULT_DATA / "l16c64_codebook_usage_20260722",
    )
    parser.add_argument("--layers", default="0:30")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--tensors", default="K,V")
    parser.add_argument("--prompts", default="0:20")
    parser.add_argument("--sample-per-prompt", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sampled_vectors(args: argparse.Namespace, prompt: int, layer: int, head: int, tensor: str) -> torch.Tensor:
    payload = torch.load(
        layer_path(args.data_root, "train", prompt, layer),
        map_location="cpu", weights_only=False, mmap=True,
    )
    vectors = payload[tensor][:, head].reshape(-1, DIM)
    generator = torch.Generator().manual_seed(
        args.seed + prompt * 1_000_000 + layer * 10_000 + head * 100 + 17 + (tensor == "V")
    )
    count = min(args.sample_per_prompt, vectors.shape[0])
    indices = torch.randperm(vectors.shape[0], generator=generator)[:count]
    return vectors[indices].float().contiguous()


def evaluate_unit(args: argparse.Namespace, layer: int, head: int, tensor: str, device: torch.device) -> list[dict]:
    codebooks = quantize_codebook_int4(
        load_baseline(args.data_root, "l16", tensor, layer, head)[:DEPTH]
    ).to(device)
    counts = torch.zeros((DEPTH, CARDINALITY), dtype=torch.int64)
    energy_before = torch.zeros(DEPTH, dtype=torch.float64)
    energy_after = torch.zeros(DEPTH, dtype=torch.float64)
    vector_count = 0

    for prompt in parse_ints(args.prompts):
        residual = sampled_vectors(args, prompt, layer, head, tensor).to(device)
        vector_count += residual.shape[0]
        for stage in range(DEPTH):
            energy_before[stage] += residual.double().square().sum().cpu()
            ids = nearest(residual, codebooks[stage], device)
            counts[stage] += torch.bincount(ids.cpu(), minlength=CARDINALITY)
            residual -= codebooks[stage, ids]
            energy_after[stage] += residual.double().square().sum().cpu()

    rows = []
    for stage in range(DEPTH):
        stage_counts = counts[stage].double()
        total = stage_counts.sum()
        probabilities = stage_counts / total.clamp_min(1)
        nonzero = probabilities > 0
        entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log2()).sum())
        active = int(nonzero.sum())
        before = float(energy_before[stage]) / (vector_count * DIM)
        after = float(energy_after[stage]) / (vector_count * DIM)
        rows.append({
            "tensor": tensor,
            "layer": layer,
            "head": head,
            "stage": stage + 1,
            "vectors": vector_count,
            "active_codes": active,
            "dead_codes": CARDINALITY - active,
            "dead_ratio": (CARDINALITY - active) / CARDINALITY,
            "entropy_bits": entropy,
            "perplexity": 2.0 ** entropy,
            "normalized_perplexity": (2.0 ** entropy) / CARDINALITY,
            "max_code_share": float(probabilities.max()),
            "top4_code_share": float(torch.topk(probabilities, 4).values.sum()),
            "residual_mse_before": before,
            "residual_mse_after": after,
            "stage_mse_reduction": before - after,
            "stage_relative_mse_reduction": (before - after) / before if before else 0.0,
        })
    return rows


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def aggregate(args: argparse.Namespace) -> None:
    rows: list[dict] = []
    for path in sorted((args.output_root / "rows").glob("shard*.csv")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    expected = len(parse_ints(args.layers)) * len(parse_ints(args.heads)) * len(args.tensors.split(",")) * DEPTH
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} unit-stage rows, found {len(rows)}")

    stage_rows = []
    for tensor in args.tensors.split(","):
        for stage in range(1, DEPTH + 1):
            group = [row for row in rows if row["tensor"] == tensor and int(row["stage"]) == stage]
            stage_rows.append({
                "tensor": tensor,
                "stage": stage,
                "unit_count": len(group),
                "active_codes_mean": mean(group, "active_codes"),
                "active_codes_min": min(int(row["active_codes"]) for row in group),
                "dead_ratio_mean": mean(group, "dead_ratio"),
                "entropy_bits_mean": mean(group, "entropy_bits"),
                "perplexity_mean": mean(group, "perplexity"),
                "normalized_perplexity_mean": mean(group, "normalized_perplexity"),
                "max_code_share_mean": mean(group, "max_code_share"),
                "top4_code_share_mean": mean(group, "top4_code_share"),
                "residual_mse_before_mean": mean(group, "residual_mse_before"),
                "residual_mse_after_mean": mean(group, "residual_mse_after"),
                "stage_relative_mse_reduction_mean": mean(group, "stage_relative_mse_reduction"),
            })
    write_csv(args.output_root / "per_stage.csv", stage_rows)

    layer_rows = []
    for tensor in args.tensors.split(","):
        for layer in parse_ints(args.layers):
            for stage in range(1, DEPTH + 1):
                group = [row for row in rows if row["tensor"] == tensor and int(row["layer"]) == layer and int(row["stage"]) == stage]
                layer_rows.append({
                    "tensor": tensor, "layer": layer, "stage": stage,
                    "perplexity_mean": mean(group, "perplexity"),
                    "dead_ratio_mean": mean(group, "dead_ratio"),
                    "max_code_share_mean": mean(group, "max_code_share"),
                    "stage_relative_mse_reduction_mean": mean(group, "stage_relative_mse_reduction"),
                })
    write_csv(args.output_root / "per_layer_stage.csv", layer_rows)
    summary = {
        "status": "complete",
        "scope": "20 train prompts; real BF16 K/V; 30 layers x 24 heads; L16C64 INT4-restored codebooks",
        "sample_per_prompt_layer_head_tensor": args.sample_per_prompt,
        "unit_stage_rows": len(rows),
        "per_stage_csv": str(args.output_root / "per_stage.csv"),
        "per_layer_stage_csv": str(args.output_root / "per_layer_stage.csv"),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_run_info(args: argparse.Namespace) -> None:
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"command", "shard_index"}
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "config.json"
    config_bytes = json.dumps(config, indent=2, sort_keys=True).encode()
    config_path.write_bytes(config_bytes)
    split = json.loads((args.data_root / "prompt_split.json").read_text(encoding="utf-8"))
    info = {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "config_path": str(config_path),
        "config_hash": hashlib.sha256(config_bytes).hexdigest(),
        "codebook_path": [str(args.data_root / "nonorm_k_l16c64_codebooks"), str(args.data_root / "nonorm_v_l16c64_codebooks")],
        "codebook_hash": {
            tensor: directory_hash(args.data_root / f"nonorm_{tensor.lower()}_l16c64_codebooks")
            for tensor in ("K", "V")
        },
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": args.seed,
        "prompt_list": [item["prompt"] for item in split["train"]],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    target = args.output_root / "run_info" / f"{args.command}_shard{args.shard_index}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    write_run_info(args)
    if args.command == "aggregate":
        aggregate(args)
        return
    units = [(layer, head, tensor) for layer in parse_ints(args.layers) for head in parse_ints(args.heads) for tensor in args.tensors.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for layer, head, tensor in units[args.shard_index::args.num_shards]:
        rows.extend(evaluate_unit(args, layer, head, tensor, device))
        print(f"[usage] {tensor} layer={layer} head={head}", flush=True)
    write_csv(args.output_root / "rows" / f"shard{args.shard_index}.csv", rows)


if __name__ == "__main__":
    main()
