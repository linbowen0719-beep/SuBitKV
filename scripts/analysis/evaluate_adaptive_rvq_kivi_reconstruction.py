#!/usr/bin/env python3
"""Create-only 10-prompt reconstruction audit for Adaptive-RVQ residual codecs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quant_videogen.frame_mean_gaussian import (
    decode_frame_mean_gaussian_residual,
    encode_frame_mean_gaussian_residual,
    payload_nbytes as k_payload_nbytes,
)
from quant_videogen.kivi import decode_kivi_tensor, encode_kivi_tensor, payload_nbytes as kivi_payload_nbytes
from quant_videogen.compress import compress_tensor_cache, get_quantize_fn
from quant_videogen.functions import triton_prq_dequantize_tensor
from quant_videogen.rvq import rvq_dequantize_tensor_cache
from quant_videogen.progressive_sign_adaptive_absmax_scalar import (
    decode_progressive_sign_adaptive_absmax_scalar_residual,
    encode_progressive_sign_adaptive_absmax_scalar_residual,
    payload_nbytes as v_payload_nbytes,
)
from quant_videogen.sim.quant.quantize_config import QuantizeConfig
from quant_videogen.temporal_adaptive_residual import (
    create_b16_state,
    decode_state,
    payload_nbytes as current_payload_nbytes,
    transition_state,
)
from scripts.benchmark_temporal_codec_optimizations import (
    DEFAULT_DATA,
    ROOT,
    codec_config,
    encode_base,
    parse_ints,
)


METHODS = (
    "adaptive_rvq_only",
    "adaptive_rvq_kivi_b1_g32",
    "adaptive_rvq_kivi_b2_g32",
    "adaptive_rvq_kivi_b3_g32",
    "adaptive_rvq_kivi_b4_g32",
    "adaptive_rvq_quantvideo_int2",
    "adaptive_rvq_quantvideo_int4",
    "direct_quantvideo_int2",
    "direct_quantvideo_int4",
    "adaptive_rvq_current_b1",
    "adaptive_rvq_current_b2",
    "adaptive_rvq_current_b3",
    "adaptive_rvq_current_b4",
    "direct_kivi_int2",
    "direct_kivi_int4",
    "direct_current_b2",
    "adaptive_rvq_uniform_b2",
)
CHUNKS = 10
CHUNK_FRAMES = 4
CHUNK_TOKENS = 3520
DIM = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty rows for {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(source: torch.Tensor, restored: torch.Tensor) -> dict[str, float]:
    x = source.float()
    y = restored.float()
    error = x - y
    vector_se = error.square().mean(dim=-1).reshape(-1)
    mse = float(vector_se.mean())
    source_mse = float(x.square().mean())
    cosine_error = float(
        (1.0 - F.cosine_similarity(x, y, dim=-1, eps=1e-12)).mean()
    )
    p99 = float(torch.quantile(vector_se, 0.99))
    p999 = float(torch.quantile(vector_se, 0.999))
    tail = vector_se[vector_se >= p99]
    return {
        "mse": mse,
        "normalized_mse": mse / max(source_mse, 1e-30),
        "cosine_error": cosine_error,
        "vector_se_mean": mse,
        "vector_se_p99": p99,
        "vector_se_p999": p999,
        "vector_se_cvar99": float(tail.mean()),
        "source_mean_square": source_mse,
    }


def source_chunks(path: Path, name: str, device: torch.device) -> list[torch.Tensor]:
    raw = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)[name]
    if raw.dtype != torch.bfloat16 or raw.shape[-1] != DIM:
        raise ValueError(f"unexpected source {path} {name}: {raw.shape} {raw.dtype}")
    return [
        raw[index * CHUNK_FRAMES : (index + 1) * CHUNK_FRAMES]
        .permute(1, 0, 2, 3)
        .reshape(1, raw.shape[1], CHUNK_TOKENS, DIM)
        .to(device=device)
        .contiguous()
        for index in range(CHUNKS)
    ]


def method_bits(method: str) -> int:
    if "_b" in method:
        return int(method.rsplit("_b", 1)[1].split("_", 1)[0])
    if method.endswith("int2"):
        return 2
    if method.endswith("int4"):
        return 4
    return 0


@torch.no_grad()
def run_worker(args: argparse.Namespace) -> None:
    out = args.output_root / "units" / f"prompt_{args.prompt:02d}.csv"
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    paths = SimpleNamespace(
        k_codebooks=args.data_root / "nonorm_k_l16c64_codebooks",
        v_codebooks=args.data_root / "nonorm_v_l16c64_codebooks",
        routing=args.routing,
        tree_params=args.tree_params,
        v_params=args.v_params,
        fast_approximate_rvq=False,
    )
    current_config = codec_config(paths)
    uniform_values = vars(paths).copy()
    uniform_values["routing"] = args.uniform_routing
    uniform_paths = SimpleNamespace(**uniform_values)
    uniform_config = codec_config(uniform_paths)
    direct_tree_params = torch.load(
        str(args.tree_params), map_location=device, weights_only=False
    )
    qv = {}
    for bits in (2, 4):
        quant_type = f"triton-nstages-kmeans-int{bits}"
        config = QuantizeConfig(
            quant_type=quant_type,
            cache_num_k_centroids=256,
            cache_num_v_centroids=256,
            kmeans_max_iters=2,
            quant_block_size=64,
            num_prq_stages=1,
        )
        qv[bits] = (quant_type, config, get_quantize_fn(quant_type, config))

    def encode_qv_preserve_rng(value: torch.Tensor, bits: int):
        cpu_before = torch.random.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(device).clone()
        quant_type, config, quantize = qv[bits]
        state = compress_tensor_cache(value, quant_type, config, quantize, 256)
        # Production QV samples online centroids, then restores model RNG so
        # codec choice cannot alter the subsequent closed-loop trajectory.
        torch.random.set_rng_state(cpu_before)
        torch.cuda.set_rng_state(cuda_before, device)
        if not torch.equal(cpu_before, torch.random.get_rng_state()):
            raise RuntimeError("QuantVideo changed CPU RNG")
        if not torch.equal(cuda_before, torch.cuda.get_rng_state(device)):
            raise RuntimeError("QuantVideo changed CUDA RNG")
        return state
    rows: list[dict[str, Any]] = []
    for layer in args.layers:
        path = args.data_root / "kv_dump" / "test" / str(args.prompt) / f"layer_{layer}.pt"
        for name in ("K", "V"):
            chunks = source_chunks(path, name, device)
            for chunk_index, source in enumerate(chunks):
                pristine = source.clone()
                base_state = encode_base(source, name, layer, paths)
                base = rvq_dequantize_tensor_cache(base_state, torch.bfloat16)
                if not torch.equal(source, pristine):
                    raise RuntimeError("RVQ mutated the BF16 input")
                base_hash = hashlib.sha256(
                    json.dumps({
                        "depth": int(base_state["num_levels"]),
                        "shape": list(base_state["indices"].shape),
                        "sample": base_state["indices"].reshape(-1)[:64].cpu().tolist(),
                    }, sort_keys=True).encode()
                ).hexdigest()
                rvq_bytes = tensor_bytes(base_state)
                residual = (source - base).contiguous()
                for method in args.methods:
                    bits = method_bits(method)
                    if method == "adaptive_rvq_only":
                        restored = base
                        residual_bytes = 0
                        local_rvq_bytes = rvq_bytes
                        observed_hash = base_hash
                    elif method.startswith("adaptive_rvq_kivi"):
                        state = encode_kivi_tensor(
                            residual, tensor_name=name.lower(), bits=bits, group_size=32
                        )
                        suffix = decode_kivi_tensor(state)
                        restored = (base + suffix).to(torch.bfloat16).contiguous()
                        residual_bytes = kivi_payload_nbytes(state)
                        local_rvq_bytes = rvq_bytes
                        observed_hash = base_hash
                    elif method.startswith(("adaptive_rvq_current", "adaptive_rvq_uniform")):
                        selected_config = (
                            uniform_config
                            if method.startswith("adaptive_rvq_uniform")
                            else current_config
                        )
                        state = transition_state(
                            create_b16_state(
                                source,
                                base_state,
                                tensor_name=name,
                                layer_idx=layer,
                                portable_identity=False,
                            ),
                            bits,
                            quant_config=selected_config,
                            verify_identity=False,
                        )
                        restored = decode_state(state)
                        total = current_payload_nbytes(state, include_rvq=True)
                        local_rvq_bytes = current_payload_nbytes(state, include_rvq=True) - current_payload_nbytes(state, include_rvq=False)
                        residual_bytes = total - local_rvq_bytes
                        observed_hash = base_hash
                    elif method.startswith("adaptive_rvq_quantvideo"):
                        state = encode_qv_preserve_rng(residual, bits)
                        suffix = triton_prq_dequantize_tensor(
                            state, 64, bits, output_dtype=torch.bfloat16
                        )
                        restored = (base + suffix).to(torch.bfloat16).contiguous()
                        residual_bytes = tensor_bytes(state)
                        local_rvq_bytes = rvq_bytes
                        observed_hash = base_hash
                    elif method.startswith("direct_quantvideo"):
                        state = encode_qv_preserve_rng(source, bits)
                        restored = triton_prq_dequantize_tensor(
                            state, 64, bits, output_dtype=torch.bfloat16
                        )
                        residual_bytes = tensor_bytes(state)
                        local_rvq_bytes = 0
                        observed_hash = "none"
                    elif method.startswith("direct_current"):
                        if name == "K":
                            state = encode_frame_mean_gaussian_residual(
                                source,
                                layer_idx=layer,
                                tensor_name="K",
                                bits=bits,
                                routing_table=args.uniform_routing,
                                tree_params=direct_tree_params,
                                seed=20260722,
                                mean_shrinkage="none",
                                spatial_tokens=880,
                            )
                            restored = decode_frame_mean_gaussian_residual(
                                state, torch.bfloat16
                            )
                            # Shared decoder tables are model-level payload and must
                            # be charged once by the final accounting, not once per
                            # physical history chunk.
                            residual_bytes = k_payload_nbytes(
                                state, include_shared=False
                            )
                        else:
                            state = encode_progressive_sign_adaptive_absmax_scalar_residual(
                                source,
                                method="midpoint_msb",
                                params=args.v_params,
                                group_size=128,
                                max_bits=bits,
                            )
                            restored = decode_progressive_sign_adaptive_absmax_scalar_residual(
                                state, torch.bfloat16
                            )
                            residual_bytes = v_payload_nbytes(state)
                        local_rvq_bytes = 0
                        observed_hash = "none"
                    else:
                        state = encode_kivi_tensor(
                            source, tensor_name=name.lower(), bits=bits, group_size=32
                        )
                        restored = decode_kivi_tensor(state)
                        residual_bytes = kivi_payload_nbytes(state)
                        local_rvq_bytes = 0
                        observed_hash = "none"
                    stats = metrics(source, restored)
                    rows.append({
                        "prompt": args.prompt,
                        "layer": layer,
                        "chunk": chunk_index,
                        "tensor": name,
                        "method": method,
                        "nominal_bits": bits,
                        "values": int(source.numel()),
                        "source_bytes": int(source.numel() * source.element_size()),
                        "rvq_payload_bytes": int(local_rvq_bytes),
                        "residual_payload_bytes": int(residual_bytes),
                        "local_payload_bytes": int(local_rvq_bytes + residual_bytes),
                        "rvq_depth": int(base_state["num_levels"]) if local_rvq_bytes else 0,
                        "rvq_state_hash": observed_hash,
                        **stats,
                    })
                    del restored
                del source, pristine, base_state, base, residual
            del chunks
        print(f"prompt={args.prompt} layer={layer} complete", flush=True)
    write_csv(out, rows)


def aggregate(args: argparse.Namespace) -> None:
    raw: list[dict[str, str]] = []
    for prompt in args.prompts:
        path = args.output_root / "units" / f"prompt_{prompt:02d}.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            raw.extend(csv.DictReader(handle))
    expected = len(args.prompts) * len(args.layers) * CHUNKS * 2 * len(args.methods)
    if len(raw) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(raw)}")
    write_csv(args.output_root / "kv_reconstruction_per_chunk.csv", raw)
    outputs: dict[str, list[dict[str, Any]]] = {}
    for axis, keys in (
        ("prompt", ("method", "tensor", "prompt")),
        ("layer", ("method", "tensor", "layer")),
        ("main", ("method", "tensor")),
    ):
        grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in raw:
            grouped[tuple(row[key] for key in keys)].append(row)
        result = []
        for key, items in sorted(grouped.items()):
            values = sum(int(row["values"]) for row in items)
            source_bytes = sum(int(row["source_bytes"]) for row in items)
            payload = sum(int(row["local_payload_bytes"]) for row in items)
            record = {name: value for name, value in zip(keys, key)}
            record.update({
                "rows": len(items),
                "values": values,
                "mse": sum(float(row["mse"]) * int(row["values"]) for row in items) / values,
                "normalized_mse": sum(float(row["normalized_mse"]) * int(row["values"]) for row in items) / values,
                "cosine_error": sum(float(row["cosine_error"]) * int(row["values"]) for row in items) / values,
                "vector_se_p99_mean": sum(float(row["vector_se_p99"]) for row in items) / len(items),
                "vector_se_p999_mean": sum(float(row["vector_se_p999"]) for row in items) / len(items),
                "vector_se_cvar99_mean": sum(float(row["vector_se_cvar99"]) for row in items) / len(items),
                "rvq_depth_mean": sum(float(row["rvq_depth"]) for row in items) / len(items),
                "source_MiB": source_bytes / 2**20,
                "local_payload_MiB": payload / 2**20,
                "local_compression_ratio": source_bytes / payload,
            })
            result.append(record)
        outputs[axis] = result
    write_csv(args.output_root / "kv_reconstruction_per_prompt.csv", outputs["prompt"])
    write_csv(args.output_root / "kv_reconstruction_per_layer.csv", outputs["layer"])
    write_csv(args.output_root / "kv_reconstruction_main.csv", outputs["main"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "worker", "aggregate"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prompts", type=parse_ints, default=parse_ints("0,1,2,3,4,5,6,7,8,9"))
    parser.add_argument("--layers", type=parse_ints, default=parse_ints(",".join(map(str, range(30)))))
    parser.add_argument("--prompt", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--methods")
    parser.add_argument("--routing", type=Path, default=ROOT / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json")
    parser.add_argument("--uniform-routing", type=Path, default=ROOT / "results/analysis/frame_mean_gaussian_preposition_uniformK_routeV_control_20260728/routing.json")
    parser.add_argument("--tree-params", type=Path, default=ROOT / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt")
    parser.add_argument("--v-params", type=Path, default=ROOT / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt")
    args = parser.parse_args()
    if args.methods is None:
        config_path = args.output_root / "config.json"
        configured = (
            json.loads(config_path.read_text(encoding="utf-8"))["methods"]
            if config_path.is_file()
            else METHODS
        )
        args.methods = tuple(configured)
    else:
        args.methods = tuple(
            item.strip() for item in args.methods.split(",") if item.strip()
        )
    unknown = set(args.methods).difference(METHODS)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}")
    if args.command == "create":
        if args.output_root.exists():
            raise FileExistsError(args.output_root)
        args.output_root.mkdir(parents=True)
        config = {
            "schema": "adaptive_rvq_kivi_reconstruction.v1",
            "methods": args.methods,
            "prompts": list(args.prompts),
            "layers": list(args.layers),
            "chunks": CHUNKS,
            "source_position": "pre_rope_pre_prope",
            "rvq_backend": "strict_staged_bit_exact",
            "rvq_rule": "L4-L16; stop after two consecutive drops below 6.29%",
        }
        (args.output_root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        run_info = {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "config_path": str((args.output_root / "config.json").resolve()),
            "config_hash": sha256_json(config),
            "codebook_path": {"K": str((args.data_root / "nonorm_k_l16c64_codebooks").resolve()), "V": str((args.data_root / "nonorm_v_l16c64_codebooks").resolve())},
            "codebook_hash": {"K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e", "V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae"},
            "hostname": socket.gethostname(),
            "gpu_type": "NVIDIA RTX 6000 Ada Generation",
            "seed": 42,
            "prompt_list": list(args.prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        (args.output_root / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    elif args.command == "worker":
        if args.prompt is None:
            raise ValueError("worker requires --prompt")
        run_worker(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
