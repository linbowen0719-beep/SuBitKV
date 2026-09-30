#!/usr/bin/env python3
"""Rate-matched screen of QuantVideo and a structure-preserving nested suffix.

The statistical unit is one prompt/layer/head.  Both methods encode the BF16
residual after the same runtime L8C64-INT4 RVQ base.  Structured B2/B4 uses a
C256 online vector predictor, uint8 labels, and a G128 FP16 suffix scale, so
its physical payload is exactly equal to QuantVideo INT2/INT4 for S=7040.

This screen deliberately reports tail-sensitive QK diagnostics in addition to
mean reconstruction error.  It is not an end-to-end video benchmark.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (REPO_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_subitkv_cascade import qv_residual_pair  # noqa: E402
from run_fixed_budget_rvq_screen import (  # noqa: E402
    baseline_path,
    layer_path,
    parse_ints,
    rvq_reconstruct,
    sha256_file,
)
from subitkv.structured_truncatable import (  # noqa: E402
    decode_structured_residual,
    encode_structured_residual,
    payload_nbytes as structured_payload_nbytes,
    truncate_structured_state,
)


DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
PARAMS = (
    DATA
    / "truncatable_residual_strict_final_20260723"
    / "training"
    / "codec_params.pt"
)
OUTPUT = DATA / "structured_truncatable_screen_20260723"
DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("command", choices=("eval", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--params-path", type=Path, default=PARAMS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--prompts", default="0:10")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--variants", default="h0r0,h1r0,h1r1")
    parser.add_argument("--bits", default="2,4")
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--kmeans-iters", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.params_path = args.params_path.resolve()
    args.output_root = args.output_root.resolve()
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    args.heads = parse_ints(args.heads)
    args.bits = parse_ints(args.bits)
    args.variants = tuple(
        value.strip() for value in args.variants.split(",") if value.strip()
    )
    if not args.variants or any(
        value not in {"h0r0", "h1r0", "h1r1"} for value in args.variants
    ):
        parser.error("--variants must be a subset of h0r0,h1r0,h1r1")
    if any(value not in (2, 4) for value in args.bits):
        parser.error("--bits must be a subset of 2,4")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0,num-shards)")
    return args


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_create(
        path,
        (
            json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False
            )
            + "\n"
        ).encode(),
    )


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    if not rows:
        raise ValueError("Refusing to serialize an empty table")
    fields = sorted({key for row in rows for key in row})
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_create(path, csv_bytes(rows))


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            result.extend(csv.DictReader(handle))
    return result


def config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "structured_truncatable_screen.config.v1",
        "data_root": str(args.data_root),
        "params_path": str(args.params_path),
        "output_root": str(args.output_root),
        "prompts": args.prompts,
        "layers": args.layers,
        "heads": args.heads,
        "key_sample": args.key_sample,
        "key_sampling": "aligned inference_span",
        "query_sample": args.query_sample,
        "variants": list(args.variants),
        "bits": args.bits,
        "centroids": args.centroids,
        "kmeans_iters": args.kmeans_iters,
        "group_size": args.group_size,
        "seed": args.seed,
        "rate_baseline": "BF16 residual, 16 physical bits/scalar",
        "rvq": "runtime L8C64 with INT4 codebooks and UINT4 scales",
    }


def write_run_info(args: argparse.Namespace) -> None:
    cfg = config(args)
    cfg_bytes = (
        json.dumps(cfg, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()
    cfg_path = args.output_root / "config.json"
    if cfg_path.exists():
        if cfg_path.read_bytes() != cfg_bytes:
            raise RuntimeError(f"Existing config differs: {cfg_path}")
    else:
        try:
            atomic_create(cfg_path, cfg_bytes)
        except FileExistsError:
            if not cfg_path.exists() or cfg_path.read_bytes() != cfg_bytes:
                raise RuntimeError(f"Concurrent config differs: {cfg_path}")
    split = json.loads(
        (args.data_root / "prompt_split.json").read_text(encoding="utf-8")
    )
    markers = {
        str(
            args.data_root / "nonorm_k_l8c64_codebooks"
        ): sha256_file(
            args.data_root
            / "nonorm_k_l8c64_codebooks"
            / "layer_0_head_0.pt"
        ),
        str(
            args.data_root / "nonorm_v_l8c64_codebooks"
        ): sha256_file(
            args.data_root
            / "nonorm_v_l8c64_codebooks"
            / "layer_0_head_0.pt"
        ),
        str(args.params_path): sha256_file(args.params_path),
    }
    run_info = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(cfg_path),
        "config_hash": cfg_hash,
        "codebook_path": list(markers),
        "codebook_hash": markers,
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available()
            else None
        ),
        "seed": args.seed,
        "prompt_list": [
            split["test"][index]["prompt"] for index in args.prompts
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric_accounting": {
            "K_and_V": "reported separately before prompt macro averaging",
            "QK_tail": "absolute sampled pre-position logit error",
            "runtime": "synchronized encode/decode of the residual codec only",
            "payload": (
                "physical tensor bytes; shared static Gaussian table excluded "
                "and RVQ base reported separately"
            ),
        },
    }
    path = (
        args.output_root
        / "run_info"
        / f"{args.command}_shard{args.shard_index}.json"
    )
    if not path.exists():
        write_json(path, run_info)


@functools.lru_cache(maxsize=None)
def codebook(
    data_root: str, tensor_name: str, layer: int, head: int
) -> torch.Tensor:
    payload = torch.load(
        baseline_path(
            Path(data_root), "l8", tensor_name, layer, head
        ),
        map_location="cpu",
        weights_only=False,
    )
    return payload["codebooks"].float()


def sample_indices(
    total: int,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    count = min(int(count), int(total))
    starts = list(range(0, total - count + 1, count))
    if not starts:
        return torch.arange(total)
    start = starts[
        int(torch.randint(len(starts), (1,), generator=generator).item())
    ]
    return torch.arange(start, start + count)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, function):
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return result, time.perf_counter() - started


def error_coherence(error: torch.Tensor) -> float:
    denominator = error.square().mean().clamp_min(1e-20)
    return float(error.mean(dim=0).square().mean() / denominator)


def quantile(value: torch.Tensor, probability: float) -> float:
    return float(torch.quantile(value.reshape(-1), probability))


def metrics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
) -> dict[str, float]:
    query, key, value, key_hat, value_hat = (
        item.float() for item in (query, key, value, key_hat, value_hat)
    )
    scale = DIM**-0.5
    logits = query @ key.T * scale
    logits_hat = query @ key_hat.T * scale
    probability = torch.softmax(logits, dim=-1)
    probability_hat = torch.softmax(logits_hat, dim=-1)
    output = probability @ value
    output_hat = probability_hat @ value_hat
    output_k_only = probability_hat @ value
    output_v_only = probability @ value_hat
    qk_abs = (logits_hat - logits).abs()
    k_error = key_hat - key
    v_error = value_hat - value
    return {
        "K_MSE": float(k_error.square().mean()),
        "V_MSE": float(v_error.square().mean()),
        "K_relative_MSE": float(
            k_error.square().mean() / key.square().mean().clamp_min(1e-20)
        ),
        "V_relative_MSE": float(
            v_error.square().mean()
            / value.square().mean().clamp_min(1e-20)
        ),
        "K_error_coherence": error_coherence(k_error),
        "V_error_coherence": error_coherence(v_error),
        "QK_MSE": float((logits_hat - logits).square().mean()),
        "QK_abs_p95": quantile(qk_abs, 0.95),
        "QK_abs_p99": quantile(qk_abs, 0.99),
        "QK_abs_p999": quantile(qk_abs, 0.999),
        "QK_abs_max": float(qk_abs.max()),
        "attention_KL": float(
            (
                probability.clamp_min(1e-20)
                * (
                    probability.clamp_min(1e-20).log()
                    - probability_hat.clamp_min(1e-20).log()
                )
            )
            .sum(dim=-1)
            .mean()
        ),
        "attention_probability_MSE": float(
            (probability_hat - probability).square().mean()
        ),
        "K_only_output_MSE": float(
            (output_k_only - output).square().mean()
        ),
        "V_only_output_MSE": float(
            (output_v_only - output).square().mean()
        ),
        "QKV_output_MSE": float((output_hat - output).square().mean()),
        "QKV_output_cosine": float(
            F.cosine_similarity(output_hat, output, dim=-1).mean()
        ),
    }


def unit_seed(args: argparse.Namespace, prompt: int, layer: int, head: int) -> int:
    return args.seed + prompt * 1_000_003 + layer * 1009 + head * 97


def evaluate_unit(
    args: argparse.Namespace,
    prompt: int,
    layer: int,
    head: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    payload = torch.load(
        layer_path(args.data_root, "test", prompt, layer),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    flat = {
        name: payload[name][:, head].reshape(-1, DIM)
        for name in ("Q", "K", "V")
    }
    generator = torch.Generator().manual_seed(
        unit_seed(args, prompt, layer, head)
    )
    key_ids = sample_indices(
        len(flat["K"]), args.key_sample, generator
    )
    query_ids = torch.randperm(
        len(flat["Q"]), generator=generator
    )[: args.query_sample]
    query = flat["Q"][query_ids].float()
    key = flat["K"][key_ids].float()
    value = flat["V"][key_ids].float()
    base_key = rvq_reconstruct(
        key,
        codebook(str(args.data_root), "K", layer, head),
        device,
    ).to(torch.bfloat16)
    base_value = rvq_reconstruct(
        value,
        codebook(str(args.data_root), "V", layer, head),
        device,
    ).to(torch.bfloat16)
    source_bytes = int((key.numel() + value.numel()) * 2)
    common = {
        "schema_version": "structured_truncatable_screen.metrics.v1",
        "prompt_id": prompt,
        "layer": layer,
        "head": head,
        "key_tokens": len(key_ids),
        "query_tokens": len(query_ids),
        "source_BF16_residual_bytes": source_bytes,
    }
    rows = [
        {
            **common,
            "method": "l8c64_int4_base",
            "bits": 0,
            "residual_payload_bytes": 0,
            "residual_bits_per_scalar": 0.0,
            "residual_compression_ratio_vs_BF16": None,
            "encode_seconds": 0.0,
            "decode_seconds": 0.0,
            **metrics(
                query,
                key,
                value,
                base_key.cpu(),
                base_value.cpu(),
            ),
        }
    ]

    qv_args = argparse.Namespace(
        centroids=args.centroids,
        kmeans_iters=args.kmeans_iters,
        block_size=64,
        prq_stages=1,
    )
    for bits in args.bits:
        seed = unit_seed(args, prompt, layer, head)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        (qv_key, qv_value, qv_bytes), qv_encode = timed(
            device,
            lambda bits=bits: qv_residual_pair(
                key,
                value,
                base_key,
                base_value,
                bits,
                qv_args,
                device,
            ),
        )
        # qv_residual_pair includes the QuantVideo decode in this interval.
        rows.append(
            {
                **common,
                "method": f"rvq_plus_quantvideo_int{bits}",
                "bits": bits,
                "residual_payload_bytes": qv_bytes,
                "residual_bits_per_scalar": (
                    8.0 * qv_bytes / (key.numel() + value.numel())
                ),
                "residual_compression_ratio_vs_BF16": (
                    source_bytes / qv_bytes
                ),
                "encode_seconds": qv_encode,
                "decode_seconds": None,
                "runtime_note": (
                    "combined qv encode+decode round trip; standalone runtime "
                    "is taken from the existing full benchmark"
                ),
                **metrics(
                    query,
                    key,
                    value,
                    qv_key.reshape(-1, DIM).cpu(),
                    qv_value.reshape(-1, DIM).cpu(),
                ),
            }
        )

    for variant in args.variants:
        hadamard = variant[1] == "1"
        refine = int(variant[-1])
        packed = []
        encode_seconds = 0.0
        for tensor_name, reference, base in (
            ("K", key, base_key),
            ("V", value, base_value),
        ):
            residual = (
                reference.to(device=device, dtype=torch.bfloat16)
                - base
            ).reshape(1, 1, -1, DIM).contiguous()
            state, elapsed = timed(
                device,
                lambda residual=residual, tensor_name=tensor_name: (
                    encode_structured_residual(
                        residual,
                        bits=4,
                        tensor_name=tensor_name,
                        params_path=args.params_path,
                        scheme="gaussian_tree",
                        group_size=args.group_size,
                        n_clusters=args.centroids,
                        kmeans_max_iters=args.kmeans_iters,
                        kmeans_seed=unit_seed(
                            args, prompt, layer, head
                        ),
                        hadamard=hadamard,
                        scale_refine_iters=refine,
                    )
                ),
            )
            encode_seconds += elapsed
            packed.append((base, state))
        for bits in args.bits:
            states = [
                state
                if bits == 4
                else truncate_structured_state(state, bits)
                for _, state in packed
            ]
            decoded = []
            decode_seconds = 0.0
            for (base, _), state in zip(packed, states):
                suffix, elapsed = timed(
                    device,
                    lambda state=state: decode_structured_residual(
                        state, torch.bfloat16
                    ),
                )
                decode_seconds += elapsed
                decoded.append(
                    (base.reshape_as(suffix) + suffix)
                    .reshape(-1, DIM)
                    .cpu()
                )
            encoded_bytes = sum(
                structured_payload_nbytes(state) for state in states
            )
            rows.append(
                {
                    **common,
                    "method": f"structured_gaussian_{variant}_b{bits}",
                    "bits": bits,
                    "residual_payload_bytes": encoded_bytes,
                    "residual_bits_per_scalar": (
                        8.0
                        * encoded_bytes
                        / (key.numel() + value.numel())
                    ),
                    "residual_compression_ratio_vs_BF16": (
                        source_bytes / encoded_bytes
                    ),
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                    "runtime_note": (
                        "B4 encode shared by all physical prefixes; eager "
                        "structured suffix decode"
                    ),
                    **metrics(
                        query, key, value, decoded[0], decoded[1]
                    ),
                }
            )
    return rows


def finite_mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = []
    for row in rows:
        raw = row.get(name)
        if raw in (None, "", "None"):
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return sum(values) / len(values) if values else None


METRICS = (
    "K_MSE",
    "V_MSE",
    "K_relative_MSE",
    "V_relative_MSE",
    "K_error_coherence",
    "V_error_coherence",
    "QK_MSE",
    "QK_abs_p95",
    "QK_abs_p99",
    "QK_abs_p999",
    "QK_abs_max",
    "attention_KL",
    "attention_probability_MSE",
    "K_only_output_MSE",
    "V_only_output_MSE",
    "QKV_output_MSE",
    "QKV_output_cosine",
    "residual_bits_per_scalar",
    "residual_compression_ratio_vs_BF16",
    "encode_seconds",
    "decode_seconds",
)


def grouped_summary(
    rows: list[dict[str, Any]], axes: tuple[str, ...]
) -> list[dict[str, Any]]:
    keys = sorted(
        {
            tuple(row[name] for name in ("method", *axes))
            for row in rows
        }
    )
    result = []
    for key in keys:
        method, *axis_values = key
        group = [
            row
            for row in rows
            if row["method"] == method
            and all(row[name] == value for name, value in zip(axes, axis_values))
        ]
        result.append(
            {
                "method": method,
                **dict(zip(axes, axis_values)),
                "units": len(group),
                **{name: finite_mean(group, name) for name in METRICS},
            }
        )
    return result


def bootstrap_improvement(
    candidate: dict[int, float],
    baseline: dict[int, float],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float, float, int]:
    prompts = sorted(set(candidate) & set(baseline))

    def statistic(indices: list[int]) -> float:
        cand = sum(candidate[prompts[index]] for index in indices) / len(indices)
        base = sum(baseline[prompts[index]] for index in indices) / len(indices)
        return 100.0 * (base - cand) / base

    point = statistic(list(range(len(prompts))))
    generator = random.Random(seed)
    draws = sorted(
        statistic(
            [generator.randrange(len(prompts)) for _ in prompts]
        )
        for _ in range(samples)
    )
    return (
        point,
        draws[round(0.025 * (samples - 1))],
        draws[round(0.975 * (samples - 1))],
        sum(candidate[prompt] < baseline[prompt] for prompt in prompts),
    )


def aggregate(args: argparse.Namespace) -> None:
    paths = sorted((args.output_root / "rows").glob("shard*.csv"))
    if len(paths) != args.num_shards:
        raise RuntimeError(
            f"Expected {args.num_shards} shards, found {len(paths)}"
        )
    raw = read_rows(paths)
    expected_units = (
        len(args.prompts) * len(args.layers) * len(args.heads)
    )
    methods = sorted({row["method"] for row in raw})
    for method in methods:
        count = sum(row["method"] == method for row in raw)
        if count != expected_units:
            raise RuntimeError(
                f"{method} has {count} units, expected {expected_units}"
            )

    prompt_rows = grouped_summary(raw, ("prompt_id",))
    summary = grouped_summary(prompt_rows, ())
    layer_summary = grouped_summary(raw, ("layer",))
    head_summary = grouped_summary(raw, ("head",))
    layer_head_summary = grouped_summary(raw, ("layer", "head"))

    paired = []
    prompt_lookup = {
        (row["method"], int(row["prompt_id"])): row
        for row in prompt_rows
    }
    for method in methods:
        if not method.startswith("structured_"):
            continue
        bits = int(method.rsplit("b", 1)[1])
        baseline = f"rvq_plus_quantvideo_int{bits}"
        for metric in (
            "K_MSE",
            "V_MSE",
            "QK_MSE",
            "QK_abs_p999",
            "attention_KL",
            "QKV_output_MSE",
        ):
            candidate = {
                prompt: float(prompt_lookup[(method, prompt)][metric])
                for prompt in args.prompts
            }
            control = {
                prompt: float(prompt_lookup[(baseline, prompt)][metric])
                for prompt in args.prompts
            }
            point, low, high, wins = bootstrap_improvement(
                candidate,
                control,
                seed=args.seed
                + int(
                    hashlib.sha256(
                        f"{method}.{metric}".encode()
                    ).hexdigest()[:8],
                    16,
                ),
                samples=args.bootstrap_samples,
            )
            paired.append(
                {
                    "candidate": method,
                    "baseline": baseline,
                    "metric": metric,
                    "paired_prompts": len(args.prompts),
                    "prompt_wins": wins,
                    "improvement_pct": point,
                    "bootstrap_ci95_low": low,
                    "bootstrap_ci95_high": high,
                }
            )

    tail_rows = []
    for method in methods:
        group = [row for row in raw if row["method"] == method]
        for metric in (
            "K_MSE",
            "V_MSE",
            "QK_MSE",
            "QK_abs_p999",
            "attention_KL",
            "QKV_output_MSE",
        ):
            values = torch.tensor(
                [float(row[metric]) for row in group], dtype=torch.float64
            )
            ordered = values.sort(descending=True).values
            top_count = max(1, math.ceil(0.01 * len(ordered)))
            tail_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "units": len(group),
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "p95": float(torch.quantile(values, 0.95)),
                    "p99": float(torch.quantile(values, 0.99)),
                    "p999": float(torch.quantile(values, 0.999)),
                    "maximum": float(values.max()),
                    "top_1pct_error_mass_fraction": float(
                        ordered[:top_count].sum()
                        / ordered.sum().clamp_min(1e-30)
                    ),
                }
            )

    unit_lookup = {
        (
            row["method"],
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["head"]),
        ): row
        for row in raw
    }
    unit_paired = []
    for method in methods:
        if not method.startswith("structured_"):
            continue
        bits = int(method.rsplit("b", 1)[1])
        baseline = f"rvq_plus_quantvideo_int{bits}"
        for layer_scope in ("all", *args.layers):
            units = [
                (prompt, layer, head)
                for prompt in args.prompts
                for layer in args.layers
                for head in args.heads
                if layer_scope == "all" or layer == layer_scope
            ]
            for metric in (
                "K_MSE",
                "V_MSE",
                "QK_MSE",
                "QK_abs_p999",
                "attention_KL",
                "QKV_output_MSE",
            ):
                candidate_values = torch.tensor(
                    [
                        float(
                            unit_lookup[(method, prompt, layer, head)][
                                metric
                            ]
                        )
                        for prompt, layer, head in units
                    ],
                    dtype=torch.float64,
                )
                baseline_values = torch.tensor(
                    [
                        float(
                            unit_lookup[(baseline, prompt, layer, head)][
                                metric
                            ]
                        )
                        for prompt, layer, head in units
                    ],
                    dtype=torch.float64,
                )
                ratios = candidate_values / baseline_values.clamp_min(1e-30)
                unit_paired.append(
                    {
                        "candidate": method,
                        "baseline": baseline,
                        "layer_scope": layer_scope,
                        "metric": metric,
                        "units": len(units),
                        "aggregate_improvement_pct": float(
                            100.0
                            * (
                                baseline_values.mean()
                                - candidate_values.mean()
                            )
                            / baseline_values.mean().clamp_min(1e-30)
                        ),
                        "unit_win_rate": float(
                            (candidate_values < baseline_values).double().mean()
                        ),
                        "median_candidate_over_baseline": float(
                            ratios.median()
                        ),
                        "p95_candidate_over_baseline": float(
                            torch.quantile(ratios, 0.95)
                        ),
                        "p99_candidate_over_baseline": float(
                            torch.quantile(ratios, 0.99)
                        ),
                        "maximum_candidate_over_baseline": float(
                            ratios.max()
                        ),
                    }
                )

    write_csv(args.output_root / "unit_rows.csv", raw)
    write_csv(args.output_root / "prompt_macro.csv", prompt_rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_csv(args.output_root / "layer_summary.csv", layer_summary)
    write_csv(args.output_root / "head_summary.csv", head_summary)
    write_csv(
        args.output_root / "layer_head_summary.csv", layer_head_summary
    )
    write_csv(args.output_root / "paired_vs_quantvideo.csv", paired)
    write_csv(args.output_root / "tail_risk.csv", tail_rows)
    write_csv(
        args.output_root / "unit_paired_vs_quantvideo.csv", unit_paired
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    write_run_info(args)
    if args.command == "aggregate":
        aggregate(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("This formal screen requires CUDA")
    device = torch.device("cuda")
    units = [
        (prompt, layer, head)
        for prompt in args.prompts
        for layer in args.layers
        for head in args.heads
    ]
    rows = []
    for index, (prompt, layer, head) in enumerate(
        units[args.shard_index :: args.num_shards], start=1
    ):
        rows.extend(
            evaluate_unit(args, prompt, layer, head, device)
        )
        print(
            f"[{index}/{math.ceil(len(units)/args.num_shards)}] "
            f"p={prompt} l={layer} h={head}",
            flush=True,
        )
    write_csv(
        args.output_root / "rows" / f"shard{args.shard_index}.csv",
        rows,
    )


if __name__ == "__main__":
    main()
