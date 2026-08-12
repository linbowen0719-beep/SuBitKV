#!/usr/bin/env python3
"""Create-only one-layer smoke for KIVI-style and TurboQuant KV adapters."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant_videogen.kivi import decode_kivi_tensor, encode_kivi_tensor
from quant_videogen.turboquant import (
    decode_turboquant,
    encode_turboquant_mse,
    encode_turboquant_prod,
    shared_parameter_manifest,
)


DEFAULT_INPUT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump/test/0/layer_0.pt"
)
METHODS = (
    "kivi_all_int2",
    "kivi_all_int4",
    "turboquant_mse_k3_v2",
    "turboquant_mse_k4_v3",
    "turboquant_prod_k3_v2",
    "turboquant_prod_k4_v3",
    "turboquant_mse_k2_v2",
    "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2",
    "turboquant_prod_k4_v4",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def encode(method: str, k: torch.Tensor, v: torch.Tensor):
    if method.startswith("kivi_all_int"):
        bits = int(method.rsplit("int", 1)[1])
        return (
            encode_kivi_tensor(k, tensor_name="k", bits=bits, group_size=32),
            encode_kivi_tensor(v, tensor_name="v", bits=bits, group_size=32),
        )
    _, setting, k_text, v_text = method.split("_")
    k_bits = int(k_text[1:])
    v_bits = int(v_text[1:])
    k_state = (
        encode_turboquant_prod(k, bits=k_bits)
        if setting == "prod"
        else encode_turboquant_mse(k, bits=k_bits)
    )
    return k_state, encode_turboquant_mse(v, bits=v_bits)


def decode(state) -> torch.Tensor:
    if str(state["quant_method"]).startswith("kivi"):
        return decode_kivi_tensor(state)
    return decode_turboquant(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--query-sample", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"create-only output already exists: {args.output}")
    args.output.mkdir(parents=True)

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    frame_count = min(int(args.frames), int(payload["K"].shape[0]))
    device = torch.device(args.device)
    tensors = {}
    for name in ("K", "Q", "V"):
        value = payload[name][:frame_count]
        tensors[name] = value.permute(1, 0, 2, 3).reshape(
            1, value.shape[1], -1, value.shape[-1]
        ).to(device=device, non_blocking=True).contiguous()
    k, q, v = tensors["K"], tensors["Q"], tensors["V"]
    q = q[:, :, -min(args.query_sample, q.shape[2]) :]
    torch.cuda.synchronize(device)

    rows = []
    for method in METHODS:
        # Populate immutable tables/device matrices outside the measured pass.
        warm_k, warm_v = encode(method, k[:, :, :32], v[:, :, :32])
        decode(warm_k)
        decode(warm_v)
        torch.cuda.synchronize(device)
        del warm_k, warm_v

        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        k_state, v_state = encode(method, k, v)
        end.record()
        end.synchronize()
        encode_s = start.elapsed_time(end) / 1000.0

        start.record()
        k_hat, v_hat = decode(k_state), decode(v_state)
        end.record()
        end.synchronize()
        decode_s = start.elapsed_time(end) / 1000.0
        cpu_unchanged = torch.equal(cpu_before, torch.random.get_rng_state())
        cuda_unchanged = torch.equal(cuda_before, torch.cuda.get_rng_state(device))

        scale = 1.0 / math.sqrt(k.shape[-1])
        ref_logits = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
        test_logits = torch.matmul(q.float(), k_hat.float().transpose(-1, -2)) * scale
        ref_attention = ref_logits.softmax(-1)
        test_attention = test_logits.softmax(-1)
        ref_output = torch.matmul(ref_attention, v.float())
        test_output = torch.matmul(test_attention, v_hat.float())
        kl = (
            ref_attention
            * (
                ref_attention.clamp_min(1e-12).log()
                - test_attention.clamp_min(1e-12).log()
            )
        ).sum(-1).mean()
        rows.append({
            "method": method,
            "K_MSE": float((k.float() - k_hat.float()).square().mean()),
            "V_MSE": float((v.float() - v_hat.float()).square().mean()),
            "logit_MSE": float((ref_logits - test_logits).square().mean()),
            "attention_KL": float(kl),
            "attention_output_MSE": float(
                (ref_output - test_output).square().mean()
            ),
            "payload_MiB": (tensor_bytes(k_state) + tensor_bytes(v_state)) / 2**20,
            "BF16_MiB": (tensor_bytes(k) + tensor_bytes(v)) / 2**20,
            "codec_only_ratio": (tensor_bytes(k) + tensor_bytes(v))
            / (tensor_bytes(k_state) + tensor_bytes(v_state)),
            "encode_s": encode_s,
            "decode_s": decode_s,
            "cpu_rng_unchanged": cpu_unchanged,
            "cuda_rng_unchanged": cuda_unchanged,
        })
        del k_state, v_state, k_hat, v_hat

    with (args.output / "single_layer_smoke.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if not all(row["cpu_rng_unchanged"] and row["cuda_rng_unchanged"] for row in rows):
        raise RuntimeError("codec changed global CPU/CUDA RNG state")
    by_name = {row["method"]: row for row in rows}
    for prefix in ("kivi_all", "turboquant_mse", "turboquant_prod"):
        low = by_name[f"{prefix}_int2"] if prefix == "kivi_all" else by_name[f"{prefix}_k2_v2"]
        high = by_name[f"{prefix}_int4"] if prefix == "kivi_all" else by_name[f"{prefix}_k4_v4"]
        if high["K_MSE"] > low["K_MSE"] or high["V_MSE"] > low["V_MSE"]:
            raise RuntimeError(f"non-monotone K/V reconstruction for {prefix}")

    config = {
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "frames": frame_count,
        "query_sample": int(q.shape[2]),
        "shape": list(k.shape),
        "methods": list(METHODS),
        "device": str(device),
    }
    write_json(args.output / "config.json", config)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    gpu_name = torch.cuda.get_device_name(device)
    write_json(args.output / "run_info.json", {
        "git_commit": commit,
        "config_path": str((args.output / "config.json").resolve()),
        "config_hash": sha256(args.output / "config.json"),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_name,
        "seed": 20260803,
        "prompt_list": [payload.get("metadata", {}).get("prompt", "prompt_0")],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "turboquant_shared_parameters": shared_parameter_manifest(),
    })


if __name__ == "__main__":
    main()
