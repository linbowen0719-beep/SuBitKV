#!/usr/bin/env python3
"""Single-layer CUDA smoke for a complete trained HY avg2 head group."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.hyworldplay8b_cache_codec import (  # noqa: E402
    AVG2_RESIDUAL_BITS,
    encode_avg2_prefix_candidates,
    make_codec_settings,
    tensor_tree_nbytes,
)
from subitkv.uncompress import uncompress_kv_cache  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--v-params", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": "hyworldplay_rvq_avg2_cuda_smoke.config.v2",
        "layer": 0,
        "heads": 16,
        "streams": 2,
        "spatial_tokens": 1537,
        "latent_frames": 4,
        "head_dim": 128,
        "bits": list(AVG2_RESIDUAL_BITS),
        "seed": 42,
    }
    write_create(args.output_dir / "config.json", config)
    settings = make_codec_settings(
        "rvq_ours_avg2",
        k_rvq_codebook_dir=str(args.root / "nonorm_k_l16c64_codebooks"),
        v_rvq_codebook_dir=str(args.root / "nonorm_v_l16c64_codebooks"),
        routed_gaussian_routing_path=str(args.routing),
        routed_gaussian_tree_params_path=str(args.tree),
        progressive_v_absmax_params_path=str(args.v_params),
        temporal_profile_path="",
        frame_spatial_tokens=1537,
        allow_missing_temporal_profile_for_calibration=True,
    )
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(42)
    # HY routing is trained jointly for all 16 heads.  Exercise one full layer
    # so the smoke uses the same routing contract as formal inference.
    shape = (2, 16, 4 * 1537, 128)
    k = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    v = torch.randn(shape, generator=generator, device=device, dtype=torch.bfloat16)
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    torch.cuda.reset_peak_memory_stats(device)
    encode_started = time.perf_counter()
    candidates = encode_avg2_prefix_candidates(k, v, layer_idx=0, settings=settings)
    torch.cuda.synchronize(device)
    encode_seconds = time.perf_counter() - encode_started
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(
        cuda_rng, torch.cuda.get_rng_state(device)
    ):
        raise RuntimeError("avg2 codec changed the model RNG streams")
    rows = []
    for bits in AVG2_RESIDUAL_BITS:
        decode_started = time.perf_counter()
        k_hat, v_hat = uncompress_kv_cache(*candidates[bits])
        torch.cuda.synchronize(device)
        rows.append(
            {
                "bits": bits,
                "K_mse": float((k_hat.float() - k.float()).square().mean().item()),
                "V_mse": float((v_hat.float() - v.float()).square().mean().item()),
                "decode_seconds": time.perf_counter() - decode_started,
                "physical_payload_bytes": tensor_tree_nbytes(candidates[bits]),
            }
        )
    for name in ("K_mse", "V_mse"):
        values = [float(row[name]) for row in rows]
        if any(left > right + 1e-12 for left, right in zip(values, values[1:])):
            raise RuntimeError(f"{name} is not precision monotone: {values}")
    payload = {
        "schema_version": "hyworldplay_rvq_avg2_cuda_smoke.v2",
        "status": "success",
        "encode_seconds": encode_seconds,
        "rows": rows,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "RNG_isolated": True,
    }
    write_create(args.output_dir / "result.json", payload)
    write_create(
        args.output_dir / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((args.output_dir / "config.json").resolve()),
            "config_hash": sha256_file(args.output_dir / "config.json"),
            "codebook_path": str(args.root.resolve()),
            "codebook_hash": sha256_file(args.root / "manifest.json"),
            "routing_hash": sha256_file(args.routing),
            "tree_hash": sha256_file(args.tree),
            "v_params_hash": sha256_file(args.v_params),
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(device),
            "seed": 42,
            "prompt_list": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
