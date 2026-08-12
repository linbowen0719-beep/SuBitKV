#!/usr/bin/env python3
"""Protocol-matched screen for routed Gaussian residual compression."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import subprocess
import sys
from collections.abc import Mapping
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

from quant_videogen import routed_gaussian  # noqa: E402
from quant_videogen import truncatable_residual  # noqa: E402
from eval_rvq_quantvideo_cascade import qv_pair  # noqa: E402
from run_fixed_budget_rvq_screen import layer_path  # noqa: E402
from run_truncatable_residual_codec_bench import (  # noqa: E402
    attention_metrics,
    deterministic_indices,
    load_int4_base_codebook,
    rvq_residual,
    stable_directory_hash,
)


DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_PARAMS = (
    DEFAULT_DATA
    / "truncatable_residual_strict_final_20260723/training/codec_params.pt"
)


def parse_ints(value: str) -> tuple[int, ...]:
    result: list[int] = []
    for part in value.split(","):
        if ":" in part:
            start, stop = (int(item) for item in part.split(":"))
            result.extend(range(start, stop))
        else:
            result.append(int(part))
    return tuple(dict.fromkeys(result))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--tree-params", type=Path, required=True)
    parser.add_argument("--old-params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--prompts", default="0,4,9")
    parser.add_argument("--layers", default="0,10,20,29")
    parser.add_argument("--key-sample", type=int, default=7040)
    parser.add_argument("--query-sample", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    parser.add_argument("--hadamard-group-size", type=int, default=128)
    parser.add_argument("--uniform-k-route", action="store_true")
    parser.add_argument("--uniform-v-route", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.routing = args.routing.expanduser().resolve()
    args.tree_params = args.tree_params.expanduser().resolve()
    args.old_params = args.old_params.expanduser().resolve()
    args.prompts = parse_ints(args.prompts)
    args.layers = parse_ints(args.layers)
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def uniform_routing() -> dict[str, Any]:
    matrix = [[0 for _head in range(24)] for _layer in range(30)]
    return {
        "schema_version": routed_gaussian.ROUTING_SCHEMA,
        "num_layers": 30,
        "num_heads": 24,
        "offsets": {"K": matrix, "V": matrix},
    }


def metric_rows_for_layer(
    args: argparse.Namespace,
    *,
    prompt: int,
    layer: int,
    device: torch.device,
    old_params: Mapping[str, Any],
) -> list[dict[str, Any]]:
    saved = torch.load(
        str(layer_path(args.data_root, "test", prompt, layer)),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    samples: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("q", "k", "v", "kb", "vb", "kr", "vr")
    }
    runtime_args = SimpleNamespace(
        data_root=args.data_root,
        rvq_assignment_mode="runtime_bf16",
    )
    for head in range(24):
        flat = {
            name: saved[name][:, head].reshape(-1, 128)
            for name in ("Q", "K", "V")
        }
        generator = torch.Generator().manual_seed(
            args.seed + prompt * 1_000_003 + layer * 1009 + head
        )
        key_ids = deterministic_indices(
            flat["K"].shape[0],
            args.key_sample,
            generator,
            "inference_span",
        )
        query_ids = torch.randperm(
            flat["Q"].shape[0], generator=generator
        )[: args.query_sample]
        q = flat["Q"][query_ids].to(device=device, dtype=torch.float32)
        k = flat["K"][key_ids].to(device=device, dtype=torch.float32)
        v = flat["V"][key_ids].to(device=device, dtype=torch.float32)
        kb, kr = rvq_residual(
            runtime_args,
            k,
            load_int4_base_codebook(runtime_args, "K", layer, head, device),
            device,
        )
        vb, vr = rvq_residual(
            runtime_args,
            v,
            load_int4_base_codebook(runtime_args, "V", layer, head, device),
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
        name: torch.stack(values, dim=0)
        for name, values in samples.items()
    }
    kr4 = stacked["kr"].unsqueeze(0)
    vr4 = stacked["vr"].unsqueeze(0)
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = {}

    qv_args = SimpleNamespace(
        centroids=256,
        kmeans_iters=2,
        block_size=64,
        prq_stages=1,
    )
    with torch.random.fork_rng(devices=[device.index]):
        qv_seed = args.seed + prompt * 1_000_003 + layer * 1009
        torch.manual_seed(qv_seed)
        torch.cuda.manual_seed(qv_seed)
        qv_kr, qv_vr, qv_payload = qv_pair(kr4, vr4, 4, qv_args)
    candidates["rvq_quantvideo_int4"] = (
        (stacked["kb"].unsqueeze(0) + qv_kr).to(torch.bfloat16)[0],
        (stacked["vb"].unsqueeze(0) + qv_vr).to(torch.bfloat16)[0],
        int(qv_payload),
        0,
    )
    with torch.random.fork_rng(devices=[device.index]):
        torch.manual_seed(qv_seed)
        torch.cuda.manual_seed(qv_seed)
        direct_k, direct_v, direct_payload = qv_pair(
            stacked["k"].unsqueeze(0).to(torch.bfloat16),
            stacked["v"].unsqueeze(0).to(torch.bfloat16),
            4,
            qv_args,
        )
    candidates["direct_quantvideo_int4"] = (
        direct_k[0],
        direct_v[0],
        int(direct_payload),
        0,
    )

    old_k = truncatable_residual.encode_residual(
        kr4,
        "gaussian_tree",
        4,
        "K",
        params_path=old_params,
        hadamard=True,
        seed=args.codec_seed,
    )
    old_v = truncatable_residual.encode_residual(
        vr4,
        "gaussian_tree",
        4,
        "V",
        params_path=old_params,
        hadamard=True,
        seed=args.codec_seed,
    )
    old_kr = truncatable_residual.decode_residual(old_k, torch.bfloat16)
    old_vr = truncatable_residual.decode_residual(old_v, torch.bfloat16)
    old_shared = sum(
        tensor.numel() * tensor.element_size()
        for name in ("K", "V")
        for kind in ("thresholds", "centroids")
        for tensor in old_params["schemes"]["gaussian_tree"][name][kind]
    )
    candidates["old_gaussian_g64_fp16_b4"] = (
        (stacked["kb"].unsqueeze(0) + old_kr).to(torch.bfloat16)[0],
        (stacked["vb"].unsqueeze(0) + old_vr).to(torch.bfloat16)[0],
        truncatable_residual.payload_nbytes(old_k)
        + truncatable_residual.payload_nbytes(old_v),
        int(old_shared),
    )

    for method, routing in (
        ("uniform_gaussian_g128_fp8_b4", uniform_routing()),
        (
            "routed_gaussian_g128_fp8_b4",
            json.loads(args.routing.read_text(encoding="utf-8")),
        ),
    ):
        tree_params = torch.load(
            args.tree_params, map_location=device, weights_only=False
        )
        if method == "routed_gaussian_g128_fp8_b4" and (
            args.uniform_k_route or args.uniform_v_route
        ):
            routing = dict(routing)
            routing["offsets"] = {
                name: (
                    [[0 for _head in range(24)] for _layer in range(30)]
                    if (
                        (name == "K" and args.uniform_k_route)
                        or (name == "V" and args.uniform_v_route)
                    )
                    else routing["offsets"][name]
                )
                for name in ("K", "V")
            }
        state_k = routed_gaussian.encode_routed_gaussian_residual(
            kr4,
            tensor_name="K",
            layer_idx=layer,
            routing_table=routing,
            bits=4,
            seed=args.codec_seed,
            tree_params=tree_params,
            hadamard_group_size=args.hadamard_group_size,
        )
        state_v = routed_gaussian.encode_routed_gaussian_residual(
            vr4,
            tensor_name="V",
            layer_idx=layer,
            routing_table=routing,
            bits=4,
            seed=args.codec_seed,
            tree_params=tree_params,
            hadamard_group_size=args.hadamard_group_size,
        )
        decoded_k = routed_gaussian.decode_routed_gaussian_residual(
            state_k, torch.bfloat16
        )
        decoded_v = routed_gaussian.decode_routed_gaussian_residual(
            state_v, torch.bfloat16
        )
        seen: set[str] = set()
        shared = routed_gaussian.shared_payload_nbytes(state_k, seen)
        shared += routed_gaussian.shared_payload_nbytes(state_v, seen)
        candidates[method] = (
            (stacked["kb"].unsqueeze(0) + decoded_k).to(torch.bfloat16)[0],
            (stacked["vb"].unsqueeze(0) + decoded_v).to(torch.bfloat16)[0],
            routed_gaussian.payload_nbytes(state_k)
            + routed_gaussian.payload_nbytes(state_v),
            int(shared),
        )

    rows: list[dict[str, Any]] = []
    source_bytes = (kr4.numel() + vr4.numel()) * 2
    for method, (kh, vh, payload, shared) in candidates.items():
        for head in range(24):
            q = stacked["q"][head]
            k = stacked["k"][head]
            v = stacked["v"][head]
            logits = q @ k.T / math.sqrt(128)
            probs = F.softmax(logits, -1)
            reference = (logits, probs, probs @ v)
            rows.append(
                {
                    "split": "test",
                    "prompt_id": prompt,
                    "layer": layer,
                    "head": head,
                    "method": method,
                    "bits": 4,
                    "key_tokens": args.key_sample,
                    "query_tokens": args.query_sample,
                    "residual_payload_bytes_layer": payload,
                    "shared_parameter_bytes": shared,
                    "residual_compression_ratio_vs_bf16": source_bytes / payload,
                    **attention_metrics(
                        q,
                        k,
                        v,
                        kh[head].float(),
                        vh[head].float(),
                        reference,
                    ),
                }
            )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "K_MSE",
        "V_MSE",
        "QK_logits_MSE",
        "attention_KL",
        "QKV_output_MSE",
        "QKV_output_cosine",
    )
    result: list[dict[str, Any]] = []
    methods = sorted({str(row["method"]) for row in rows})
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        item: dict[str, Any] = {
            "method": method,
            "units": len(selected),
            "prompts": len({row["prompt_id"] for row in selected}),
            "layers": len({row["layer"] for row in selected}),
            "residual_compression_ratio_vs_bf16": sum(
                float(row["residual_compression_ratio_vs_bf16"])
                for row in selected
            )
            / len(selected),
        }
        for metric in metrics:
            item[metric] = sum(float(row[metric]) for row in selected) / len(selected)
        result.append(item)
    return result


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    old_params = torch.load(
        args.old_params, map_location=device, weights_only=False
    )
    config = {
        "data_root": str(args.data_root),
        "routing": str(args.routing),
        "tree_params": str(args.tree_params),
        "tree_params_hash": sha256_file(args.tree_params),
        "old_params": str(args.old_params),
        "device": args.device,
        "prompts": list(args.prompts),
        "layers": list(args.layers),
        "heads": list(range(24)),
        "key_sample": args.key_sample,
        "query_sample": args.query_sample,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "hadamard_group_size": args.hadamard_group_size,
        "uniform_k_route": args.uniform_k_route,
        "uniform_v_route": args.uniform_v_route,
        "protocol": "runtime_bf16_l8c64_int4_sampled_attention_inference_span",
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    rows: list[dict[str, Any]] = []
    for prompt in args.prompts:
        for layer in args.layers:
            rows.extend(
                metric_rows_for_layer(
                    args,
                    prompt=prompt,
                    layer=layer,
                    device=device,
                    old_params=old_params,
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
                key: str(value) for key, value in codebook_paths.items()
            },
            "codebook_hash": {
                key: stable_directory_hash(value)
                for key, value in codebook_paths.items()
            },
            "routing_path": str(args.routing),
            "routing_hash": sha256_file(args.routing),
            "tree_params_path": str(args.tree_params),
            "tree_params_hash": sha256_file(args.tree_params),
            "old_params_path": str(args.old_params),
            "old_params_hash": sha256_file(args.old_params),
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": args.seed,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
