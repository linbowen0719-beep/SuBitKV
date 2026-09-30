#!/usr/bin/env python3
"""Exact K-only ablation of the frame-mean predictor and Hadamard rotation.

The experiment keeps the deployed L8C64 INT4 RVQ base, routed B4 plane
allocation, Gaussian tree, pre-RoPE/pre-PRoPE cache boundary, and runtime
two-chunk history partition fixed.  Only two residual-codec components vary:

* decoded FP8 frame/channel mean predictor;
* fixed sign-Hadamard G128 rotation.

V is kept at the captured BF16 reference so all attention errors are K-only.
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
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
for entry in (REPO_ROOT, SCRIPT_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_r2aq_exact_attention as exact  # noqa: E402
import evaluate_sfbr_train_marginals as replay  # noqa: E402
from subitkv import frame_mean_gaussian, routed_gaussian  # noqa: E402
from subitkv.truncatable_residual import groupwise_hadamard  # noqa: E402
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import stable_directory_hash  # noqa: E402


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_GEOMETRY = (
    REPO_ROOT
    / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
    / "prompt_08/paired_rope/rank_000"
)
DEFAULT_ROUTING = (
    REPO_ROOT
    / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728"
    / "routing.json"
)
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3"
    / "tree_params.pt"
)
METHODS = (
    ("frame_mean_hadamard", True, True),
    ("no_frame_mean_hadamard", False, True),
    ("frame_mean_no_hadamard", True, False),
    ("no_frame_mean_no_hadamard", False, False),
)
METRIC_FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "head",
    "method",
    "frame_mean",
    "hadamard",
    "attention_half",
    "historical_K_mse",
    "attention_logit_mse",
    "attention_logit_relative_l2",
    "attention_kl_reference_to_candidate",
    "attention_js",
    "attention_topk_overlap",
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
    "residual_payload_bytes",
    "residual_physical_bits_per_value",
    "residual_encode_seconds",
    "residual_decode_seconds",
)
DISTRIBUTION_FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "span_index",
    "method",
    "frame_mean",
    "hadamard",
    "rvq_residual_energy",
    "predictor_energy",
    "predictor_energy_fraction",
    "suffix_energy",
    "suffix_energy_ratio",
    "transformed_energy",
    "excess_kurtosis",
    "channel_energy_cv",
    "channel_energy_max_over_mean",
    "block_absmax_over_rms_mean",
    "block_absmax_over_rms_p99",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument(
        "--bits",
        type=int,
        choices=(2, 3, 4),
        default=4,
        help="Nominal Gaussian-tree truncation depth evaluated by the route.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    args = parser.parse_args()
    for name in (
        "output_root",
        "data_root",
        "geometry_root",
        "routing",
        "tree_params",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if not args.prompts or not args.layers:
        parser.error("prompts and layers must be non-empty")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_for(args: argparse.Namespace, prompt: int, layer: int) -> dict[str, Any]:
    dump = args.data_root / "kv_dump" / args.split / str(prompt) / f"layer_{layer}.pt"
    geometry = args.geometry_root / f"layer_{layer:02d}" / "call_00000.pt"
    if not dump.is_file() or not geometry.is_file():
        raise FileNotFoundError(dump if not dump.is_file() else geometry)
    metadata = torch.load(dump, map_location="cpu", weights_only=False, mmap=True)[
        "metadata"
    ]
    if (
        metadata.get("split") != args.split
        or int(metadata.get("prompt_id", -1)) != prompt
        or int(metadata.get("layer", -1)) != layer
    ):
        raise ValueError(f"invalid dump provenance: {dump}")
    return {
        "split": args.split,
        "prompt_id": prompt,
        "prompt": str(metadata["prompt"]),
        "layer": layer,
        "dump_path": str(dump),
        "dump_bytes": int(dump.stat().st_size),
        "geometry_path": str(geometry),
        "geometry_hash": replay.geometry_signature(geometry),
    }


def decode_frame_predictor(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = tuple(int(item) for item in residual.shape)
    batch, heads, tokens, dim = shape
    frames = tokens // frame_mean_gaussian.SPATIAL_TOKENS
    shaped = residual.float().reshape(
        batch,
        heads,
        frames,
        frame_mean_gaussian.SPATIAL_TOKENS,
        dim,
    )
    raw = frame_mean_gaussian._to_signed_e4m3_raw(shaped.mean(dim=3))
    predictor = frame_mean_gaussian._decode_predictor(raw, shape)
    return predictor, raw


def distribution_row(
    *,
    args: argparse.Namespace,
    task: dict[str, Any],
    span_index: int,
    method: str,
    uses_frame_mean: bool,
    uses_hadamard: bool,
    residual: torch.Tensor,
    predictor: torch.Tensor,
    suffix: torch.Tensor,
) -> dict[str, Any]:
    transformed = (
        groupwise_hadamard(suffix.float(), 128, seed=args.codec_seed)
        if uses_hadamard
        else suffix.float()
    )
    flat = transformed.reshape(-1)
    mean = flat.mean()
    centered = flat - mean
    variance = centered.square().mean().clamp_min(1e-20)
    kurtosis = centered.square().square().mean() / variance.square() - 3.0
    channel_energy = transformed.square().mean(dim=(0, 1, 2))
    channel_mean = channel_energy.mean().clamp_min(1e-20)
    blocks = transformed.reshape(-1, 128)
    block_rms = blocks.square().mean(-1).clamp_min(1e-20).sqrt()
    block_ratio = blocks.abs().amax(-1) / block_rms
    residual_energy = residual.float().square().mean().clamp_min(1e-20)
    predictor_energy = predictor.float().square().mean()
    suffix_energy = suffix.float().square().mean()
    return {
        "split": args.split,
        "prompt_id": int(task["prompt_id"]),
        "layer": int(task["layer"]),
        "span_index": span_index,
        "method": method,
        "frame_mean": uses_frame_mean,
        "hadamard": uses_hadamard,
        "rvq_residual_energy": float(residual_energy),
        "predictor_energy": float(predictor_energy),
        "predictor_energy_fraction": float(predictor_energy / residual_energy),
        "suffix_energy": float(suffix_energy),
        "suffix_energy_ratio": float(suffix_energy / residual_energy),
        "transformed_energy": float(transformed.square().mean()),
        "excess_kurtosis": float(kurtosis),
        "channel_energy_cv": float(channel_energy.std(unbiased=False) / channel_mean),
        "channel_energy_max_over_mean": float(channel_energy.max() / channel_mean),
        "block_absmax_over_rms_mean": float(block_ratio.mean()),
        "block_absmax_over_rms_p99": float(torch.quantile(block_ratio, 0.99)),
    }


@torch.no_grad()
def evaluate_task(
    args: argparse.Namespace,
    task: dict[str, Any],
    device: torch.device,
    routing: dict[str, Any],
    tree_params: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    (
        tensors,
        evidence,
        output_matrix,
        query_positions,
        key_positions,
        _positional_validation,
    ) = replay.positional_replay_from_dump(task, device)
    layer = int(task["layer"])
    history_tokens = 31680
    key_reference = tensors["key_pre"][:, :, :history_tokens]
    value_reference = tensors["value_pre"][:, :, :history_tokens]
    partition = exact.runtime_history_partition(history_tokens)
    spans = tuple(partition["quantized_spans"])
    pending = partition["pending_bf16_range"]
    codebook = args.data_root / "nonorm_k_l8c64_codebooks"
    base_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    for start, end in spans:
        source = key_reference[:, :, start:end].contiguous()
        packed = evidence.encode_fixed(
            source,
            layer=layer,
            directory=codebook,
            batch_size=args.assignment_batch_size,
        )
        base = evidence.rvq_dequantize_tensor_cache(packed, torch.bfloat16).contiguous()
        base_parts.append(base)
        residual_parts.append((source - base).contiguous())
        del packed

    halves = {
        "standard_rope": (
            tensors["query_post_rope"],
            tensors["key_post_rope"],
            tensors["value_post_rope"],
            None,
        ),
        "camera_prope": (
            tensors["query_post_prope"],
            tensors["key_post_prope"],
            tensors["value_post_prope"],
            output_matrix,
        ),
    }
    score_args = SimpleNamespace(
        query_batch_size=args.query_batch_size,
        topk=args.topk,
    )
    reference_context = {
        half: exact.replay.build_reference_context(
            query=query,
            key=key,
            value=value,
            history_tokens=history_tokens,
            query_positions=query_positions,
            key_positions=key_positions,
            query_batch_size=args.query_batch_size,
            topk=args.topk,
            output_matrix=matrix,
            evidence=evidence,
        )
        for half, (query, key, value, matrix) in halves.items()
    }

    rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    for method, uses_frame_mean, uses_hadamard in METHODS:
        reconstructed_parts: list[torch.Tensor] = []
        payload = 0
        encode_elapsed = 0.0
        decode_elapsed = 0.0
        seen_tables: set[str] = set()
        for span_index, residual in enumerate(residual_parts):
            if uses_frame_mean:
                (predictor, predictor_raw), predictor_elapsed = exact.replay.sync_timed(
                    device,
                    lambda residual=residual: decode_frame_predictor(residual),
                )
                predictor_bytes = int(predictor_raw.numel())
            else:
                predictor = torch.zeros_like(residual, dtype=torch.float32)
                predictor_raw = None
                predictor_bytes = 0
                predictor_elapsed = 0.0
            suffix = (residual.float() - predictor).contiguous()
            distribution_rows.append(
                distribution_row(
                    args=args,
                    task=task,
                    span_index=span_index,
                    method=method,
                    uses_frame_mean=uses_frame_mean,
                    uses_hadamard=uses_hadamard,
                    residual=residual,
                    predictor=predictor,
                    suffix=suffix,
                )
            )
            state, elapsed = exact.replay.sync_timed(
                device,
                lambda suffix=suffix, uses_hadamard=uses_hadamard: (
                    routed_gaussian.encode_routed_gaussian_residual(
                        suffix,
                        tensor_name="K",
                        layer_idx=layer,
                        routing_table=routing,
                        bits=args.bits,
                        group_size=128,
                        hadamard_group_size=128,
                        hadamard=uses_hadamard,
                        seed=args.codec_seed,
                        tree_params=tree_params,
                        scale_refine_iters=1,
                    )
                ),
            )
            if predictor_raw is not None:
                shape = tuple(int(item) for item in residual.shape)
                decoded, decode_time = exact.replay.sync_timed(
                    device,
                    lambda state=state, predictor_raw=predictor_raw, shape=shape: (
                        (
                            frame_mean_gaussian._decode_predictor(predictor_raw, shape)
                            + routed_gaussian.decode_routed_gaussian_residual(
                                state, torch.bfloat16
                            ).float()
                        )
                        .to(torch.bfloat16)
                        .contiguous()
                    ),
                )
            else:
                decoded, decode_time = exact.replay.sync_timed(
                    device,
                    lambda state=state: routed_gaussian.decode_routed_gaussian_residual(
                        state, torch.bfloat16
                    ),
                )
            payload += predictor_bytes + routed_gaussian.payload_nbytes(state)
            payload += routed_gaussian.shared_payload_nbytes(state, seen_tables)
            encode_elapsed += predictor_elapsed + elapsed
            decode_elapsed += decode_time
            reconstructed_parts.append(
                (base_parts[span_index] + decoded).to(torch.bfloat16).contiguous()
            )
        if pending is not None:
            reconstructed_parts.append(
                key_reference[:, :, pending[0] : pending[1]].contiguous()
            )
        reconstructed = torch.cat(reconstructed_parts, dim=2)
        squared = (reconstructed.float() - key_reference.float()).square()
        mse_by_head = squared.mean(dim=(0, 2, 3)).cpu()
        metrics, _current, _candidates = exact.score_exact_attention(
            args=score_args,
            tensors=tensors,
            evidence=evidence,
            query_positions=query_positions,
            key_positions=key_positions,
            query_output_matrix=output_matrix,
            history_tokens=history_tokens,
            reconstructed_k=reconstructed,
            reconstructed_v=value_reference,
            halves=halves,
            reference_context=reference_context,
        )
        quantized_values = sum(int(residual.numel()) for residual in residual_parts)
        for half in ("standard_rope", "camera_prope"):
            for head, item in enumerate(metrics[half]["K_only"]):
                rows.append(
                    {
                        "split": args.split,
                        "prompt_id": int(task["prompt_id"]),
                        "layer": layer,
                        "head": head,
                        "method": method,
                        "frame_mean": uses_frame_mean,
                        "hadamard": uses_hadamard,
                        "attention_half": half,
                        "historical_K_mse": float(mse_by_head[head]),
                        "attention_logit_mse": item["attention_logit_mse"],
                        "attention_logit_relative_l2": item[
                            "attention_logit_relative_l2"
                        ],
                        "attention_kl_reference_to_candidate": item[
                            "attention_kl_reference_to_candidate"
                        ],
                        "attention_js": item["attention_js"],
                        "attention_topk_overlap": item["attention_topk_overlap"],
                        "attention_output_squared_error_mean": item[
                            "attention_output_squared_error_mean"
                        ],
                        "attention_output_squared_error_P99": item[
                            "attention_output_squared_error_P99"
                        ],
                        "attention_output_squared_error_P999": item[
                            "attention_output_squared_error_P999"
                        ],
                        "attention_output_squared_error_CVaR99": item[
                            "attention_output_squared_error_CVaR99"
                        ],
                        "residual_payload_bytes": payload,
                        "residual_physical_bits_per_value": (
                            payload * 8 / quantized_values
                        ),
                        "residual_encode_seconds": encode_elapsed,
                        "residual_decode_seconds": decode_elapsed,
                    }
                )
    return rows, distribution_rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lower = (
        "historical_K_mse",
        "attention_logit_mse",
        "attention_logit_relative_l2",
        "attention_kl_reference_to_candidate",
        "attention_js",
        "attention_output_squared_error_mean",
        "attention_output_squared_error_P99",
        "attention_output_squared_error_P999",
        "attention_output_squared_error_CVaR99",
        "residual_physical_bits_per_value",
        "residual_encode_seconds",
        "residual_decode_seconds",
    )
    fields = (
        "method",
        "frame_mean",
        "hadamard",
        "attention_half",
        "rows",
        *lower,
        "attention_topk_overlap",
    )
    result: list[dict[str, Any]] = []
    for method, uses_frame_mean, uses_hadamard in METHODS:
        for half in ("standard_rope", "camera_prope"):
            selected = [
                row
                for row in rows
                if row["method"] == method and row["attention_half"] == half
            ]
            result.append(
                {
                    "method": method,
                    "frame_mean": uses_frame_mean,
                    "hadamard": uses_hadamard,
                    "attention_half": half,
                    "rows": len(selected),
                    **{
                        field: sum(float(row[field]) for row in selected)
                        / len(selected)
                        for field in lower
                    },
                    "attention_topk_overlap": sum(
                        float(row["attention_topk_overlap"]) for row in selected
                    )
                    / len(selected),
                }
            )
    if any(tuple(row) != fields for row in result):
        raise AssertionError("aggregate field ordering changed")
    return result


def effect_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "historical_K_mse",
        "attention_logit_mse",
        "attention_kl_reference_to_candidate",
        "attention_output_squared_error_mean",
        "attention_output_squared_error_CVaR99",
        "attention_topk_overlap",
        "residual_physical_bits_per_value",
        "residual_encode_seconds",
        "residual_decode_seconds",
    )
    fields = (
        "attention_half",
        "ablation",
        "baseline_method",
        "candidate_method",
        "metric",
        "baseline",
        "candidate",
        "candidate_minus_baseline",
        "relative_change_percent",
    )
    by_key = {(row["method"], row["attention_half"]): row for row in summary}
    result: list[dict[str, Any]] = []
    comparisons = (
        ("remove_frame_mean", "no_frame_mean_hadamard"),
        ("remove_hadamard", "frame_mean_no_hadamard"),
        ("remove_both", "no_frame_mean_no_hadamard"),
    )
    for half in ("standard_rope", "camera_prope"):
        baseline = by_key[("frame_mean_hadamard", half)]
        for label, candidate_method in comparisons:
            candidate = by_key[(candidate_method, half)]
            for metric in metrics:
                base_value = float(baseline[metric])
                candidate_value = float(candidate[metric])
                result.append(
                    {
                        "attention_half": half,
                        "ablation": label,
                        "baseline_method": "frame_mean_hadamard",
                        "candidate_method": candidate_method,
                        "metric": metric,
                        "baseline": base_value,
                        "candidate": candidate_value,
                        "candidate_minus_baseline": candidate_value - base_value,
                        "relative_change_percent": (
                            (candidate_value / base_value - 1.0) * 100.0
                            if base_value != 0
                            else math.nan
                        ),
                    }
                )
    if any(tuple(row) != fields for row in result):
        raise AssertionError("effect field ordering changed")
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    routing = json.loads(args.routing.read_text(encoding="utf-8"))
    routed_gaussian.validate_routing_table(routing)
    tree_params = torch.load(args.tree_params, map_location=device, weights_only=False)
    config = {
        "schema_version": "frame_mean_hadamard_exact_k_ablation.config.v1",
        "split": args.split,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "routing": str(args.routing),
        "routing_hash": sha256_file(args.routing),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "methods": [
            {
                "name": name,
                "frame_mean": frame_mean,
                "hadamard": hadamard,
            }
            for name, frame_mean, hadamard in METHODS
        ],
        "history_tokens": 31680,
        "runtime_history_partition": "four_7040_token_compressed_spans_plus_3520_BF16",
        "rvq": "L8C64_INT4_codebook_UINT4_scale",
        "residual_nominal_bits_per_value": args.bits,
        "residual_group_size": 128,
        "intervention": "K_only",
        "v_cache": "captured_BF16_reference",
        "position_boundary": "pre_RoPE_pre_PRoPE",
        "attention_halves": ["standard_rope", "camera_prope"],
        "query_batch_size": args.query_batch_size,
        "topk": args.topk,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    rows: list[dict[str, Any]] = []
    distribution: list[dict[str, Any]] = []
    source_files: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for prompt in args.prompts:
        for layer in args.layers:
            task = task_for(args, prompt, layer)
            task_rows, task_distribution = evaluate_task(
                args, task, device, routing, tree_params
            )
            rows.extend(task_rows)
            distribution.extend(task_distribution)
            source_files[task["dump_path"]] = {
                "bytes": task["dump_bytes"],
                "metadata_validated": True,
            }
            source_files[task["geometry_path"]] = {
                "geometry_signature": task["geometry_hash"],
            }
            print(f"[done] prompt={prompt} layer={layer}", flush=True)
    metrics_path = args.output_root / "exact_k_metrics.csv"
    distribution_path = args.output_root / "distribution.csv"
    summary_path = args.output_root / "summary.csv"
    effects_path = args.output_root / "effects_vs_full.csv"
    write_csv(metrics_path, rows, METRIC_FIELDS)
    write_csv(distribution_path, distribution, DISTRIBUTION_FIELDS)
    summary = aggregate_rows(rows)
    write_csv(summary_path, summary, tuple(summary[0]))
    effects = effect_rows(summary)
    write_csv(effects_path, effects, tuple(effects[0]))
    codebook = args.data_root / "nonorm_k_l8c64_codebooks"
    output_paths = (
        metrics_path,
        distribution_path,
        summary_path,
        effects_path,
    )
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                "K": str(codebook),
                "tree": str(args.tree_params),
            },
            "codebook_hash": {
                "K": stable_directory_hash(codebook),
                "tree": sha256_file(args.tree_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_files": source_files,
            "output_hashes": {str(path): sha256_file(path) for path in output_paths},
        },
    )


if __name__ == "__main__":
    main()
