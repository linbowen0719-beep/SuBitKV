#!/usr/bin/env python3
"""Evaluate the frozen cluster-free 2-D predictor on V RVQ residuals.

The predictor geometry and storage precision are copied unchanged from the
K-only experiment.  No V result is used to select the 22x10 grid.  Attention
output is evaluated V-only: Q and K remain BF16 references and only V is
reconstructed by the tested residual codec.
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
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_subitkv_cascade import qv_pair  # noqa: E402
from subitkv import routed_gaussian, uniform_grid_gaussian  # noqa: E402
from run_fixed_budget_rvq_screen import layer_path, parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/"
    "tree_params.pt"
)
DIM = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--heads", default="0:24")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    args = parser.parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    args.heads = parse_ints(args.heads)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    if args.key_sample % uniform_grid_gaussian.SPATIAL_TOKENS:
        raise ValueError("key-sample must contain complete 22x40 latent frames")
    if tuple(args.heads) != tuple(range(24)):
        raise ValueError("QV protocol reproduction currently requires all 24 heads")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def uniform_routing() -> dict[str, Any]:
    matrix = [[0 for _head in range(24)] for _layer in range(30)]
    return {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "offsets": {"K": matrix, "V": matrix},
    }


def robust_output_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, float]:
    per_query = (reference.float() - candidate.float()).square().mean(dim=-1)
    sorted_error = per_query.sort().values
    tail_count = max(1, math.ceil(0.01 * int(per_query.numel())))
    return {
        "V_only_output_MSE": float(per_query.mean()),
        "V_only_output_P99": float(torch.quantile(per_query, 0.99)),
        "V_only_output_P99_9": float(torch.quantile(per_query, 0.999)),
        "V_only_output_CVaR99": float(sorted_error[-tail_count:].mean()),
        "V_only_output_cosine": float(
            F.cosine_similarity(
                reference.float(), candidate.float(), dim=-1
            ).mean()
        ),
    }


@torch.no_grad()
def evaluate_layer(
    args: argparse.Namespace,
    *,
    prompt: int,
    layer: int,
    device: torch.device,
    tree_params: dict[str, Any],
) -> list[dict[str, Any]]:
    saved = torch.load(
        str(layer_path(args.data_root, "test", prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    samples = {
        name: [] for name in ("q", "k", "v", "kb", "vb", "kr", "vr")
    }
    for head in args.heads:
        flat = {
            name: saved[name][:, head].reshape(-1, DIM)
            for name in ("Q", "K", "V")
        }
        generator = torch.Generator().manual_seed(
            args.seed + prompt * 1_000_003 + layer * 1009 + head
        )
        key_ids = deterministic_indices(
            len(flat["K"]), args.key_sample, generator, "inference_span"
        )
        query_ids = torch.randperm(
            len(flat["Q"]), generator=generator
        )[: args.query_sample]
        q = flat["Q"][query_ids].to(device=device, dtype=torch.float32)
        k = flat["K"][key_ids].to(device=device, dtype=torch.float32)
        v = flat["V"][key_ids].to(device=device, dtype=torch.float32)
        kb, kr = rvq_residual(
            runtime_args,
            k,
            load_int4_base_codebook(
                runtime_args, "K", layer, head, device
            ),
            device,
        )
        vb, vr = rvq_residual(
            runtime_args,
            v,
            load_int4_base_codebook(
                runtime_args, "V", layer, head, device
            ),
            device,
        )
        for name, value in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("kb", kb),
            ("vb", vb),
            ("kr", kr),
            ("vr", vr),
        ):
            samples[name].append(value)
    stacked = {
        name: torch.stack(values, dim=0) for name, values in samples.items()
    }
    kr4 = stacked["kr"].unsqueeze(0)
    vr4 = stacked["vr"].unsqueeze(0)

    qv_args = SimpleNamespace(
        centroids=256,
        kmeans_iters=2,
        block_size=64,
        prq_stages=1,
    )
    qv_seed = args.seed + prompt * 1_000_003 + layer * 1009
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(qv_seed)
        torch.cuda.manual_seed(qv_seed)
        _qv_kr, qv_vr, _qv_pair_payload = qv_pair(
            kr4, vr4, 4, qv_args
        )
    qv_v = (
        stacked["vb"].unsqueeze(0) + qv_vr
    ).to(torch.bfloat16)[0]

    uniform_state = routed_gaussian.encode_routed_gaussian_residual(
        vr4,
        tensor_name="V",
        layer_idx=layer,
        routing_table=uniform_routing(),
        bits=4,
        group_size=128,
        hadamard_group_size=128,
        hadamard=True,
        seed=args.codec_seed,
        tree_params=tree_params,
        scale_refine_iters=1,
    )
    uniform_vr = routed_gaussian.decode_routed_gaussian_residual(
        uniform_state, output_dtype=torch.bfloat16
    )
    uniform_v = (
        stacked["vb"].unsqueeze(0) + uniform_vr
    ).to(torch.bfloat16)[0]

    grid_state = (
        uniform_grid_gaussian.encode_uniform_grid_gaussian_residual(
            vr4,
            layer_idx=layer,
            tensor_name="V",
            tree_params=tree_params,
            seed=args.codec_seed,
        )
    )
    grid_vr = uniform_grid_gaussian.decode_uniform_grid_gaussian_residual(
        grid_state, output_dtype=torch.bfloat16
    )
    grid_v = (
        stacked["vb"].unsqueeze(0) + grid_vr
    ).to(torch.bfloat16)[0]

    candidates = {
        "rvq_quantvideo_int4": (qv_v, 4.769318181818182),
        "uniform_gaussian_g128_fp8_b4": (uniform_v, 4.0625),
        "rvq_uniform_grid22x10_bilinear_f8p4_v_b4": (
            grid_v,
            float(
                grid_state["diagnostics"]["local_physical_bits_per_value"]
            ),
        ),
    }
    rows: list[dict[str, Any]] = []
    for head in args.heads:
        q = stacked["q"][head]
        k = stacked["k"][head]
        v = stacked["v"][head]
        logits = q @ k.T / math.sqrt(DIM)
        probabilities = F.softmax(logits, dim=-1)
        reference_output = probabilities @ v
        for method, (decoded_v, rate) in candidates.items():
            reconstructed = decoded_v[head].float()
            error = v - reconstructed
            rows.append(
                {
                    "split": "test",
                    "prompt_id": prompt,
                    "layer": layer,
                    "head": head,
                    "method": method,
                    "V_residual_physical_bits_per_value": rate,
                    "V_MSE": float(error.square().mean()),
                    "V_relative_MSE": float(
                        error.square().mean()
                        / v.square().mean().clamp_min(1e-30)
                    ),
                    **robust_output_metrics(
                        reference_output,
                        probabilities @ reconstructed,
                    ),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "V_MSE",
        "V_relative_MSE",
        "V_only_output_MSE",
        "V_only_output_P99",
        "V_only_output_P99_9",
        "V_only_output_CVaR99",
        "V_only_output_cosine",
    )
    result: list[dict[str, Any]] = []
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {
            "method": method,
            "units": len(selected),
            "prompts": len({row["prompt_id"] for row in selected}),
            "layers": len({row["layer"] for row in selected}),
            "V_residual_physical_bits_per_value": sum(
                float(row["V_residual_physical_bits_per_value"])
                for row in selected
            )
            / len(selected),
        }
        for metric in metrics:
            item[metric] = sum(float(row[metric]) for row in selected) / len(
                selected
            )
        result.append(item)
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("V evaluation requires CUDA")
    torch.cuda.set_device(device)
    tree_params = torch.load(
        args.tree_params, map_location=device, weights_only=False
    )
    config = {
        "schema_version": "uniform_grid_gaussian_v_eval.config.v1",
        "data_root": str(args.data_root),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "device": args.device,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "heads": list(args.heads),
        "key_sample": args.key_sample,
        "query_sample": args.query_sample,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "candidate_frozen_from": "K-only prompt-0/4/9 grid selection",
        "predictor_selection_uses_V": False,
        "scalar_tree_uses_existing_V_training": True,
        "selection_uses_Q": False,
        "attention_scope": "V-only; BF16 Q/K probabilities",
        "position_scope": "saved post-position Q/K; no positional replay",
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    rows: list[dict[str, Any]] = []
    for prompt in args.prompts:
        for layer in args.layers:
            rows.extend(
                evaluate_layer(
                    args,
                    prompt=prompt,
                    layer=layer,
                    device=device,
                    tree_params=tree_params,
                )
            )
            print(f"prompt={prompt} layer={layer} complete", flush=True)
    write_csv(args.output_root / "unit_metrics.csv", rows)
    summary = aggregate(rows)
    write_csv(args.output_root / "summary.csv", summary)
    write_json(
        args.output_root / "summary.json",
        {row["method"]: row for row in summary},
    )
    codebook_paths = {
        "K": args.data_root / "nonorm_k_l8c64_codebooks",
        "V": args.data_root / "nonorm_v_l8c64_codebooks",
    }
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": git_commit(),
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
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
