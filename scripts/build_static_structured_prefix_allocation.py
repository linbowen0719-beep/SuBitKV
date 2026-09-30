#!/usr/bin/env python3
"""Build a leakage-free static structured-prefix allocation from train data.

The default evidence is the existing 20-training-prompt L8C64 functional
replay.  K priority uses query-projected logit/KL error; V priority uses the
V-only attention-output error.  Prompt means and prompt P90 values are mixed
to retain tail risk.  Global nested-codec prefix MSE supplies the diminishing
return of each additional plane.

This is a *training functional proxy*: it measures L8C64 RVQ sensitivity, not
the yet-unmeasured per-head marginal gain of the structured codec.  The output
manifest records that limitation.  Test rows present in the source CSV are
discarded before every statistic and do not participate in allocation.

The command is CPU-only and create-only:

  python scripts/build_static_structured_prefix_allocation.py \
    --output-root results/analysis/structured_static_prefix_allocation_20260723
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import platform
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.static_prefix_allocation import (  # noqa: E402
    ALLOCATION_SCHEMA,
    validate_allocation,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_EVIDENCE = (
    DEFAULT_DATA / "metrics" / "gradref_offline_reconstruction_rows.csv"
)
DEFAULT_PARAMS = (
    DEFAULT_DATA
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
DEFAULT_RUN_CONFIG = DEFAULT_DATA / "run_config.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_static_prefix_allocation_20260723"
)
BASELINE_SETTING = "RVQ-L8C64-INT4 baseline"
CONFIG_SCHEMA = "structured_static_prefix_builder.config.v1"
RUN_SCHEMA = "structured_static_prefix_builder.run_info.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--evidence-csv", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--codec-params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--setting", default=BASELINE_SETTING)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--num-heads", type=int, default=24)
    parser.add_argument("--prompt-count", type=int, default=20)
    parser.add_argument("--tail-quantile", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--canonical-history-tokens",
        type=int,
        default=7040,
        help="Only used for the illustrative physical-rate audit.",
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--predictor-centroids", type=int, default=256)
    args = parser.parse_args()
    for field in ("evidence_csv", "codec_params", "run_config", "output_root"):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    if args.num_layers < 1 or args.num_heads < 1 or args.prompt_count < 1:
        parser.error("layer/head/prompt counts must be positive")
    if not 0.5 <= args.tail_quantile < 1.0:
        parser.error("--tail-quantile must be in [0.5,1)")
    if args.head_dim < 1 or args.group_size < 1:
        parser.error("head dimension and group size must be positive")
    if args.head_dim % args.group_size:
        parser.error("--group-size must divide --head-dim")
    return args


def finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {result}")
    return result


def load_training_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    required = {
        "split",
        "setting",
        "prompt_id",
        "layer",
        "head",
        "qk_logits_mse",
        "attention_kl",
        "attn_output_mse",
        "av_normalized_mse",
    }
    rows: list[dict[str, Any]] = []
    with args.evidence_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Evidence CSV is missing columns {sorted(missing)}")
        for raw in reader:
            if raw["split"] != "train" or raw["setting"] != args.setting:
                continue
            layer = int(raw["layer"])
            head = int(raw["head"])
            prompt_id = int(raw["prompt_id"])
            if not 0 <= layer < args.num_layers:
                raise ValueError(f"Training evidence has out-of-range layer {layer}")
            if not 0 <= head < args.num_heads:
                raise ValueError(f"Training evidence has out-of-range head {head}")
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "layer": layer,
                    "head": head,
                    "qk_logits_mse": finite_float(
                        raw["qk_logits_mse"], "qk_logits_mse"
                    ),
                    "attention_kl": finite_float(
                        raw["attention_kl"], "attention_kl"
                    ),
                    "attn_output_mse": finite_float(
                        raw["attn_output_mse"], "attn_output_mse"
                    ),
                    "av_normalized_mse": finite_float(
                        raw["av_normalized_mse"], "av_normalized_mse"
                    ),
                }
            )
    expected = args.num_layers * args.num_heads * args.prompt_count
    if len(rows) != expected:
        raise ValueError(
            f"Selected {len(rows)} train-baseline rows, expected exactly {expected}"
        )
    prompts = sorted({row["prompt_id"] for row in rows})
    if len(prompts) != args.prompt_count:
        raise ValueError(
            f"Selected {len(prompts)} distinct train prompts, expected "
            f"{args.prompt_count}"
        )
    counts = Counter((row["layer"], row["head"]) for row in rows)
    bad = {key: count for key, count in counts.items() if count != args.prompt_count}
    if len(counts) != args.num_layers * args.num_heads or bad:
        raise ValueError(
            "Every layer/head must have exactly one row per training prompt; "
            f"bad units={list(bad.items())[:5]}"
        )
    return rows


def summarize_units(
    rows: Sequence[Mapping[str, Any]],
    tail_quantile: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer"]), int(row["head"]))].append(row)
    result: list[dict[str, Any]] = []
    for (layer, head), unit_rows in sorted(grouped.items()):
        def stats(name: str) -> tuple[float, float]:
            values = np.asarray(
                [float(row[name]) for row in unit_rows], dtype=np.float64
            )
            return float(values.mean()), float(
                np.quantile(values, tail_quantile, method="linear")
            )

        qk_mean, qk_tail = stats("qk_logits_mse")
        kl_mean, kl_tail = stats("attention_kl")
        vout_mean, vout_tail = stats("attn_output_mse")
        avnorm_mean, avnorm_tail = stats("av_normalized_mse")
        common = {
            "layer": layer,
            "head": head,
            "training_prompt_count": len(unit_rows),
        }
        result.extend(
            (
                {
                    **common,
                    "tensor": "K",
                    "qk_logits_mse_mean": qk_mean,
                    "qk_logits_mse_tail": qk_tail,
                    "attention_kl_mean": kl_mean,
                    "attention_kl_tail": kl_tail,
                    "attn_output_mse_mean": None,
                    "attn_output_mse_tail": None,
                    "av_normalized_mse_mean": None,
                    "av_normalized_mse_tail": None,
                },
                {
                    **common,
                    "tensor": "V",
                    "qk_logits_mse_mean": None,
                    "qk_logits_mse_tail": None,
                    "attention_kl_mean": None,
                    "attention_kl_tail": None,
                    "attn_output_mse_mean": vout_mean,
                    "attn_output_mse_tail": vout_tail,
                    "av_normalized_mse_mean": avnorm_mean,
                    "av_normalized_mse_tail": avnorm_tail,
                },
            )
        )
    return result


def clipped_median_ratio(values: Sequence[float]) -> list[float]:
    center = median(values)
    if center <= 0:
        positive = [value for value in values if value > 0]
        center = median(positive) if positive else 1.0
    return [min(20.0, max(0.05, value / center)) for value in values]


def attach_sensitivity_scores(rows: list[dict[str, Any]]) -> None:
    definitions = {
        "K": (
            ("qk_logits_mse_mean", 0.35),
            ("qk_logits_mse_tail", 0.25),
            ("attention_kl_mean", 0.20),
            ("attention_kl_tail", 0.20),
        ),
        "V": (
            ("attn_output_mse_mean", 0.45),
            ("attn_output_mse_tail", 0.35),
            ("av_normalized_mse_mean", 0.10),
            ("av_normalized_mse_tail", 0.10),
        ),
    }
    for tensor, terms in definitions.items():
        subset = [row for row in rows if row["tensor"] == tensor]
        scores = np.zeros(len(subset), dtype=np.float64)
        for metric, weight in terms:
            ratios = clipped_median_ratio(
                [float(row[metric]) for row in subset]
            )
            for index, ratio in enumerate(ratios):
                scores[index] += float(weight) * ratio
        # Equal K/V mean score makes the cross-tensor exchange explicit rather
        # than inheriting unrelated metric units.
        scores /= scores.mean()
        for row, score in zip(subset, scores.tolist()):
            row["sensitivity_score"] = float(score)


def prefix_distortions(params_path: Path) -> dict[str, list[float]]:
    value = torch.load(
        str(params_path), map_location="cpu", weights_only=False
    )
    try:
        summaries = value["fit_objective_summary"]["gaussian_tree"]["hadamard"]
        result = {
            tensor: [
                float(item)
                for item in summaries[tensor]["packed_fit_prefix_MSE"]
            ]
            for tensor in ("K", "V")
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Codec params are missing Gaussian/Hadamard packed prefix MSE"
        ) from error
    for tensor, values in result.items():
        if len(values) != 4 or any(
            not math.isfinite(item) or item <= 0 for item in values
        ):
            raise ValueError(f"Invalid {tensor} prefix distortions {values}")
        if any(left <= right for left, right in zip(values, values[1:])):
            raise ValueError(f"{tensor} prefix distortion is not decreasing: {values}")
    return result


def allocate(
    sensitivity: Sequence[Mapping[str, Any]],
    distortions: Mapping[str, Sequence[float]],
    *,
    num_layers: int,
    num_heads: int,
    target_bits: int,
) -> tuple[dict[str, list[list[int]]], list[dict[str, Any]]]:
    units = {
        (str(row["tensor"]), int(row["layer"]), int(row["head"])): row
        for row in sensitivity
    }
    expected_units = 2 * num_layers * num_heads
    if len(units) != expected_units:
        raise ValueError(f"Found {len(units)} sensitivity units, expected {expected_units}")
    depths = {key: 1 for key in units}
    additions = expected_units * (target_bits - 1)
    heap: list[tuple[float, str, int, int, int]] = []

    def push(key: tuple[str, int, int]) -> None:
        tensor, layer, head = key
        depth = depths[key]
        if depth >= 4:
            return
        gain = float(distortions[tensor][depth - 1]) - float(
            distortions[tensor][depth]
        )
        utility = float(units[key]["sensitivity_score"]) * gain
        heapq.heappush(heap, (-utility, tensor, layer, head, depth + 1))

    for key in sorted(units):
        push(key)
    increments: list[dict[str, Any]] = []
    for rank in range(1, additions + 1):
        if not heap:
            raise RuntimeError("Prefix allocator exhausted candidates before its budget")
        negative, tensor, layer, head, new_depth = heapq.heappop(heap)
        key = (tensor, layer, head)
        if depths[key] + 1 != new_depth:
            raise RuntimeError("Stale prefix candidate escaped heap invariants")
        depths[key] = new_depth
        increments.append(
            {
                "rank": rank,
                "tensor": tensor,
                "layer": layer,
                "head": head,
                "new_depth": new_depth,
                "marginal_utility": -negative,
                "sensitivity_score": float(units[key]["sensitivity_score"]),
                "normalized_prefix_mse_before": float(
                    distortions[tensor][new_depth - 2]
                ),
                "normalized_prefix_mse_after": float(
                    distortions[tensor][new_depth - 1]
                ),
            }
        )
        push(key)
    table = {
        tensor: [
            [depths[(tensor, layer, head)] for head in range(num_heads)]
            for layer in range(num_layers)
        ]
        for tensor in ("K", "V")
    }
    return table, increments


def prompt_list(run_config: Path) -> list[dict[str, Any]]:
    value = json.loads(run_config.read_text(encoding="utf-8"))
    prompts = value.get("prompt_split", {}).get("train")
    if not isinstance(prompts, list):
        raise ValueError(f"Training prompt list is missing from {run_config}")
    return prompts


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def allocation_document(
    *,
    table: Mapping[str, Any],
    args: argparse.Namespace,
    distortions: Mapping[str, Sequence[float]],
    evidence_hash: str,
    target_bits: int,
) -> dict[str, Any]:
    value = {
        "schema_version": ALLOCATION_SCHEMA,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "min_bits": 1,
        "max_bits": 4,
        "target_uniform_bits": target_bits,
        "allocation": table,
        "derivation": {
            "protocol": "train_functional_proxy_v1",
            "split_used": "train only",
            "prompt_count": args.prompt_count,
            "setting": args.setting,
            "evidence_selected_rows_sha256": evidence_hash,
            "K_objective": (
                "0.35 mean QK-logit MSE + 0.25 prompt-P90 QK-logit MSE + "
                "0.20 mean attention KL + 0.20 prompt-P90 attention KL; "
                "each metric median-normalized and clipped"
            ),
            "V_objective": (
                "0.45 mean V-only attention-output MSE + 0.35 prompt-P90 "
                "V-only attention-output MSE + 0.10 mean normalized output MSE "
                "+ 0.10 prompt-P90 normalized output MSE; each metric "
                "median-normalized and clipped"
            ),
            "cross_tensor_calibration": "K and V sensitivity means normalized to 1",
            "plane_gain_source": (
                "20-train-prompt fitted Gaussian/Hadamard global packed-prefix MSE"
            ),
            "prefix_distortions": {
                key: list(values) for key, values in distortions.items()
            },
            "limitation": (
                "The layer/head sensitivity is measured for the L8C64 RVQ "
                "baseline, not as structured-codec B1/B2/B3/B4 marginal "
                "attention error. It is a leakage-free launch candidate, not "
                "a claimed quality improvement."
            ),
        },
    }
    validate_allocation(value)
    return value


def main() -> None:
    args = parse_args()
    for path in (args.evidence_csv, args.codec_params, args.run_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output_root.exists():
        raise FileExistsError(
            f"Create-only output already exists: {args.output_root}"
        )

    selected = load_training_rows(args)
    selected_hash = canonical_hash(selected)
    sensitivity = summarize_units(selected, args.tail_quantile)
    attach_sensitivity_scores(sensitivity)
    distortions = prefix_distortions(args.codec_params)

    b2_table, b2_increments = allocate(
        sensitivity,
        distortions,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        target_bits=2,
    )
    b4_table, b4_increments = allocate(
        sensitivity,
        distortions,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        target_bits=4,
    )
    b2 = allocation_document(
        table=b2_table,
        args=args,
        distortions=distortions,
        evidence_hash=selected_hash,
        target_bits=2,
    )
    b4 = allocation_document(
        table=b4_table,
        args=args,
        distortions=distortions,
        evidence_hash=selected_hash,
        target_bits=4,
    )
    if any(
        bits != 4
        for tensor in ("K", "V")
        for row in b4_table[tensor]
        for bits in row
    ):
        raise RuntimeError("B4 budget should force the unique all-B4 allocation")

    config = {
        "schema_version": CONFIG_SCHEMA,
        "command": " ".join(sys.argv),
        "evidence_csv": str(args.evidence_csv),
        "evidence_csv_sha256": sha256_file(args.evidence_csv),
        "selected_evidence_rows_sha256": selected_hash,
        "codec_params": str(args.codec_params),
        "codec_params_sha256": sha256_file(args.codec_params),
        "run_config": str(args.run_config),
        "run_config_sha256": sha256_file(args.run_config),
        "setting": args.setting,
        "split_filter": "train",
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "prompt_count": args.prompt_count,
        "tail_quantile": args.tail_quantile,
        "seed": args.seed,
        "canonical_shape": {
            "history_tokens": args.canonical_history_tokens,
            "head_dim": args.head_dim,
            "group_size": args.group_size,
            "predictor_centroids": args.predictor_centroids,
        },
    }
    config_hash = canonical_hash(config)
    train_prompts = prompt_list(args.run_config)
    if len(train_prompts) != args.prompt_count:
        raise ValueError(
            f"Run config has {len(train_prompts)} train prompts, expected "
            f"{args.prompt_count}"
        )

    args.output_root.mkdir(parents=True, exist_ok=False)
    config_path = args.output_root / "config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    b2_path = args.output_root / "allocation_b2.json"
    b2_path.write_text(
        json.dumps(b2, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    b4_path = args.output_root / "allocation_b4.json"
    b4_path.write_text(
        json.dumps(b4, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_root / "sensitivity.csv", sensitivity)
    write_csv(args.output_root / "selected_increments_b2.csv", b2_increments)
    write_csv(args.output_root / "selected_increments_b4.csv", b4_increments)

    audits = []
    for name, allocation in (("B2", b2), ("B4", b4)):
        audit = validate_allocation(allocation)
        for tensor in ("K", "V"):
            audits.append(
                {
                    "target": name,
                    "tensor": tensor,
                    **{
                        f"heads_at_B{bits}": audit["counts"][tensor][str(bits)]
                        for bits in range(1, 5)
                    },
                    "allocated_plane_units_total_KV": audit[
                        "allocated_plane_units"
                    ],
                    "uniform_plane_units_total_KV": audit[
                        "uniform_plane_units"
                    ],
                    "nontrivial_allocation": name == "B2",
                }
            )
    write_csv(args.output_root / "allocation_summary.csv", audits)

    tokens = args.canonical_history_tokens
    dim = args.head_dim
    fixed_bpp = (
        args.predictor_centroids * dim * 16.0 / (tokens * dim)
        + 8.0 / dim
        + 16.0 / args.group_size
    )
    rate_rows = []
    total_values = (
        2 * args.num_layers * args.num_heads * tokens * dim
    )
    for bits in (2, 4):
        total_bpp = fixed_bpp + bits
        payload_bytes = total_values * total_bpp / 8.0
        rate_rows.append(
            {
                "target": f"B{bits}",
                "history_tokens": tokens,
                "head_dim": dim,
                "predictor_centroid_bpp": (
                    args.predictor_centroids * 16.0 / tokens
                ),
                "token_id_bpp": 8.0 / dim,
                "scale_bpp": 16.0 / args.group_size,
                "average_suffix_plane_bpp": bits,
                "total_residual_bpp": total_bpp,
                "residual_compression_ratio_vs_BF16": 16.0 / total_bpp,
                "full_30_layer_KV_residual_payload_MiB": (
                    payload_bytes / (1024.0**2)
                ),
                "allocated_equals_uniform_bytes": True,
            }
        )
    write_csv(args.output_root / "payload_accounting.csv", rate_rows)

    run_info = {
        "schema_version": RUN_SCHEMA,
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": None,
        "codebook_hash": None,
        "codec_params_path": str(args.codec_params),
        "codec_params_hash": config["codec_params_sha256"],
        "hostname": socket.gethostname(),
        "gpu_type": "not used; CPU-only allocation derivation",
        "seed": args.seed,
        "prompt_list": train_prompts,
        "timestamp": utc_now(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "outputs": {
            "allocation_b2": str(b2_path),
            "allocation_b2_sha256": sha256_file(b2_path),
            "allocation_b4": str(b4_path),
            "allocation_b4_sha256": sha256_file(b4_path),
        },
        "test_prompt_rows_used": 0,
        "status": "complete",
    }
    (args.output_root / "run_info.json").write_text(
        json.dumps(run_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "b2_audit": validate_allocation(b2),
                "b4_nontrivial": False,
                "selected_train_rows": len(selected),
                "test_prompt_rows_used": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
