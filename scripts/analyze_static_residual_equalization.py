#!/usr/bin/env python3
"""Screen cluster-free static channel equalization for L8C64 RVQ residuals.

The first ten training prompts fit per-layer/head/channel statistics.  The
second ten training prompts select the statistic and companding exponent.
The disjoint ten-prompt test split is reported but never used for selection.
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

from run_fixed_budget_rvq_screen import layer_path  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
TENSORS = ("K", "V")


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return result


def parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one float")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--layers", type=parse_ints, default=(0, 10, 20, 29))
    parser.add_argument("--alphas", type=parse_floats, default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-sample", type=int, default=1024)
    parser.add_argument("--eval-sample", type=int, default=2048)
    parser.add_argument("--quantile", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    if any(layer < 0 or layer >= 30 for layer in args.layers):
        parser.error("layers must be in [0,30)")
    if any(alpha <= 0.0 or alpha > 1.0 for alpha in args.alphas):
        parser.error("alphas must be in (0,1]")
    if not 0.5 < args.quantile < 1.0:
        parser.error("quantile must be in (0.5,1)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sampled_rows(
    tensor: torch.Tensor,
    count: int,
    *,
    seed: int,
) -> torch.Tensor:
    flat = tensor.reshape(-1, tensor.shape[-1])
    generator = torch.Generator().manual_seed(seed)
    ids = deterministic_indices(
        flat.shape[0], count, generator, "inference_span"
    )
    return flat[ids]


def decoded_fp8_positive(value: torch.Tensor) -> torch.Tensor:
    tiny = torch.finfo(torch.float8_e4m3fn).tiny
    return (
        value.float()
        .clamp(min=tiny, max=448.0)
        .to(torch.float8_e4m3fn)
        .float()
        .clamp_min(tiny)
    )


def static_scale(statistic: torch.Tensor, alpha: float) -> torch.Tensor:
    powered = statistic.float().clamp_min(1e-12).pow(float(alpha))
    powered = powered / powered.log().mean().exp().clamp_min(1e-12)
    return decoded_fp8_positive(powered)


def quantize_midrange(
    residual: torch.Tensor,
    channel_scale: torch.Tensor | None,
) -> torch.Tensor:
    if channel_scale is None:
        normalized = residual.float()
        restore_scale = 1.0
    else:
        restore_scale = channel_scale.float()
        normalized = residual.float() / restore_scale
    low = normalized.amin(dim=-1, keepdim=True)
    high = normalized.amax(dim=-1, keepdim=True)
    center = (0.5 * (low + high)).to(torch.float8_e4m3fn).float()
    centered = normalized - center
    step = decoded_fp8_positive(centered.abs().amax(dim=-1, keepdim=True) / 7.0)
    code = (centered / step).round().clamp(-7, 7)
    return ((code * step + center) * restore_scale).to(torch.bfloat16)


def residual_for(
    runtime_args: SimpleNamespace,
    saved: dict[str, torch.Tensor],
    *,
    tensor_name: str,
    layer: int,
    head: int,
    sample: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = sampled_rows(
        saved[tensor_name][:, head],
        sample,
        seed=seed,
    ).to(device=device, dtype=torch.float32)
    base, residual = rvq_residual(
        runtime_args,
        source,
        load_int4_base_codebook(
            runtime_args, tensor_name, layer, head, device
        ),
        device,
    )
    return source, base, residual


def empty_accumulator(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "sum_square": torch.zeros(128, device=device, dtype=torch.float64),
        "count": torch.zeros((), device=device, dtype=torch.float64),
        "quantile_sum": torch.zeros(128, device=device, dtype=torch.float64),
        "prompts": torch.zeros((), device=device, dtype=torch.float64),
    }


def update_accumulator(
    accumulator: dict[str, torch.Tensor],
    residual: torch.Tensor,
    *,
    quantile: float,
) -> None:
    accumulator["sum_square"] += residual.double().square().sum(0)
    accumulator["count"] += residual.shape[0]
    accumulator["quantile_sum"] += torch.quantile(
        residual.float().abs(), quantile, dim=0
    ).double()
    accumulator["prompts"] += 1


def finalize_accumulator(
    accumulator: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "rms": (
            accumulator["sum_square"] / accumulator["count"].clamp_min(1)
        ).sqrt().float(),
        "q995": (
            accumulator["quantile_sum"]
            / accumulator["prompts"].clamp_min(1)
        ).float(),
    }


def method_name(statistic: str, alpha: float) -> str:
    return f"{statistic}_alpha_{alpha:g}"


def evaluate_residual(
    source: torch.Tensor,
    base: torch.Tensor,
    residual: torch.Tensor,
    statistics: dict[str, torch.Tensor],
    alphas: tuple[float, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    uniform = (base + quantize_midrange(residual, None)).to(torch.bfloat16)
    result["uniform_midrange"] = float(
        (source.float() - uniform.float()).square().mean().item()
    )
    for statistic_name, statistic in statistics.items():
        for alpha in alphas:
            scale = static_scale(statistic, alpha)
            restored = (
                base + quantize_midrange(residual, scale)
            ).to(torch.bfloat16)
            result[method_name(statistic_name, alpha)] = float(
                (source.float() - restored.float()).square().mean().item()
            )
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    config = {
        "schema_version": "static_residual_equalization_screen.config.v1",
        "data_root": str(args.data_root),
        "layers": list(args.layers),
        "alphas": list(args.alphas),
        "device": args.device,
        "train_fit_prompts": list(range(10)),
        "train_validation_prompts": list(range(10, 20)),
        "test_prompts": list(range(10)),
        "train_sample": args.train_sample,
        "eval_sample": args.eval_sample,
        "quantile": args.quantile,
        "seed": args.seed,
        "selection_contract": (
            "select statistic and alpha only from train prompts 10-19; "
            "test split is reporting-only"
        ),
        "uses_clustering": False,
        "uses_query": False,
        "local_physical_bits_per_value": 4.125,
        "shared_parameter_storage": "one FP8 channel scale per layer/head/KV",
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)

    unit_rows: list[dict[str, Any]] = []
    parameter_tables: dict[str, dict[int, torch.Tensor]] = {
        name: {} for name in TENSORS
    }
    for layer in args.layers:
        fit_accumulators = {
            (name, head): empty_accumulator(device)
            for name in TENSORS
            for head in range(24)
        }
        full_accumulators = {
            (name, head): empty_accumulator(device)
            for name in TENSORS
            for head in range(24)
        }
        for prompt in range(10):
            saved = torch.load(
                str(layer_path(args.data_root, "train", prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            for head in range(24):
                for name in TENSORS:
                    _source, _base, residual = residual_for(
                        runtime_args,
                        saved,
                        tensor_name=name,
                        layer=layer,
                        head=head,
                        sample=args.train_sample,
                        seed=args.seed + prompt * 1009 + layer * 97 + head * 13,
                        device=device,
                    )
                    update_accumulator(
                        fit_accumulators[(name, head)],
                        residual,
                        quantile=args.quantile,
                    )
                    update_accumulator(
                        full_accumulators[(name, head)],
                        residual,
                        quantile=args.quantile,
                    )
            print(f"fit layer={layer:02d} prompt={prompt:02d}", flush=True)
        fit_statistics = {
            key: finalize_accumulator(value)
            for key, value in fit_accumulators.items()
        }

        for prompt in range(10, 20):
            saved = torch.load(
                str(layer_path(args.data_root, "train", prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            for head in range(24):
                for name in TENSORS:
                    source, base, residual = residual_for(
                        runtime_args,
                        saved,
                        tensor_name=name,
                        layer=layer,
                        head=head,
                        sample=args.eval_sample,
                        seed=args.seed + prompt * 1009 + layer * 97 + head * 13,
                        device=device,
                    )
                    update_accumulator(
                        full_accumulators[(name, head)],
                        residual,
                        quantile=args.quantile,
                    )
                    metrics = evaluate_residual(
                        source,
                        base,
                        residual,
                        fit_statistics[(name, head)],
                        args.alphas,
                    )
                    uniform = metrics["uniform_midrange"]
                    for method, mse in metrics.items():
                        unit_rows.append(
                            {
                                "split": "train_validation",
                                "prompt_id": prompt,
                                "layer": layer,
                                "head": head,
                                "tensor": name,
                                "method": method,
                                "mse": mse,
                                "relative_improvement_vs_uniform_pct": (
                                    100.0 * (1.0 - mse / uniform)
                                ),
                            }
                        )
            print(f"validate layer={layer:02d} prompt={prompt:02d}", flush=True)

        full_statistics = {
            key: finalize_accumulator(value)
            for key, value in full_accumulators.items()
        }
        for name in TENSORS:
            parameter_tables[name][layer] = torch.stack(
                [
                    torch.stack(
                        [
                            full_statistics[(name, head)]["rms"],
                            full_statistics[(name, head)]["q995"],
                        ]
                    )
                    for head in range(24)
                ]
            ).cpu()

        for prompt in range(10):
            saved = torch.load(
                str(layer_path(args.data_root, "test", prompt, layer)),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            for head in range(24):
                for name in TENSORS:
                    source, base, residual = residual_for(
                        runtime_args,
                        saved,
                        tensor_name=name,
                        layer=layer,
                        head=head,
                        sample=args.eval_sample,
                        seed=(
                            args.seed
                            + 1_000_003
                            + prompt * 1009
                            + layer * 97
                            + head * 13
                        ),
                        device=device,
                    )
                    metrics = evaluate_residual(
                        source,
                        base,
                        residual,
                        full_statistics[(name, head)],
                        args.alphas,
                    )
                    uniform = metrics["uniform_midrange"]
                    for method, mse in metrics.items():
                        unit_rows.append(
                            {
                                "split": "test",
                                "prompt_id": prompt,
                                "layer": layer,
                                "head": head,
                                "tensor": name,
                                "method": method,
                                "mse": mse,
                                "relative_improvement_vs_uniform_pct": (
                                    100.0 * (1.0 - mse / uniform)
                                ),
                            }
                        )
            print(f"test layer={layer:02d} prompt={prompt:02d}", flush=True)

    metrics_path = args.output_root / "unit_metrics.csv"
    params_path = args.output_root / "statistics_full_train20.pt"
    write_csv(metrics_path, unit_rows)
    torch.save(
        {
            "schema_version": "static_residual_equalization.stats.v1",
            "layers": list(args.layers),
            "statistic_names": ["rms", "q995"],
            "tables": parameter_tables,
            "uses_clustering": False,
        },
        params_path,
    )

    summary_rows: list[dict[str, Any]] = []
    for split in ("train_validation", "test"):
        for name in TENSORS:
            selected = [
                row
                for row in unit_rows
                if row["split"] == split and row["tensor"] == name
            ]
            methods = sorted({str(row["method"]) for row in selected})
            uniform_mse = sum(
                float(row["mse"])
                for row in selected
                if row["method"] == "uniform_midrange"
            ) / sum(row["method"] == "uniform_midrange" for row in selected)
            for method in methods:
                rows = [row for row in selected if row["method"] == method]
                mse = sum(float(row["mse"]) for row in rows) / len(rows)
                summary_rows.append(
                    {
                        "split": split,
                        "tensor": name,
                        "method": method,
                        "units": len(rows),
                        "mse": mse,
                        "relative_improvement_vs_uniform_pct": (
                            100.0 * (1.0 - mse / uniform_mse)
                        ),
                    }
                )
    summary_path = args.output_root / "summary.csv"
    write_csv(summary_path, summary_rows)
    codebook_paths = {
        name: args.data_root / f"nonorm_{name.lower()}_l8c64_codebooks"
        for name in TENSORS
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
            "prompt_list": {
                "train_fit": list(range(10)),
                "train_validation": list(range(10, 20)),
                "test": list(range(10)),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "outputs": {
                str(metrics_path): sha256_file(metrics_path),
                str(summary_path): sha256_file(summary_path),
                str(params_path): sha256_file(params_path),
            },
        },
    )


if __name__ == "__main__":
    main()
