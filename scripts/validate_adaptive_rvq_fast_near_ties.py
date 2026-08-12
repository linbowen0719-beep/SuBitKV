#!/usr/bin/env python3
"""Short real-cache audit of exact-staged versus monolithic adaptive RVQ."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from quant_videogen.temporal_adaptive_residual import (
    create_b16_state,
    decode_state,
    transition_state,
)
from scripts.benchmark_temporal_codec_optimizations import (
    DEFAULT_DATA,
    ROOT,
    codec_config,
    encode_base,
    load_chunk,
    parse_ints,
)


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument(
        "--layers", type=parse_ints, default=parse_ints("2,4,14,25")
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--chunks", type=parse_ints, default=parse_ints("0,1,2,3,4,5,6,7,8,9")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("CUDA is required")
    if args.bits not in (1, 2, 3, 4, 8):
        raise ValueError("bits must be one of 1/2/3/4/8")

    shared = {
        "k_codebooks": args.data_root / "nonorm_k_l16c64_codebooks",
        "v_codebooks": args.data_root / "nonorm_v_l16c64_codebooks",
        "routing": ROOT
        / "results/analysis/frame_mean_gaussian_preposition_routeK_uniformV_control_20260728/routing.json",
        "tree_params": ROOT
        / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt",
        "v_params": ROOT
        / "results/analysis/progressive_v_absmax_prefix_fit_train20x30_v2_20260730/params.pt",
    }
    exact_paths = SimpleNamespace(**shared, fast_approximate_rvq=False)
    fast_paths = SimpleNamespace(**shared, fast_approximate_rvq=True)
    residual_config = codec_config(exact_paths)
    config = {
        "schema_version": "adaptive_rvq_fast_near_tie.config.v1",
        "source_position": "pre_rope_pre_prope",
        "shape": [1, 24, 3520, 128],
        "prompt": args.prompt,
        "layers": args.layers,
        "bits": args.bits,
        "chunks": args.chunks,
        "exact_backend": "triton_staged_exact_relative_stop",
        "fast_backend": "triton_monolithic_near_tie_relative_stop",
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    input_paths: list[str] = []
    for layer in args.layers:
        input_path = (
            args.data_root / "kv_dump" / "test" / str(args.prompt)
            / f"layer_{layer}.pt"
        )
        payload = torch.load(input_path, map_location="cpu", weights_only=False)
        input_paths.append(str(input_path.resolve()))
        sources = []
        for name in ("K", "V"):
            frames = payload[name]
            for chunk in args.chunks:
                source = (
                    frames[chunk * 4 : (chunk + 1) * 4]
                    .permute(1, 0, 2, 3)
                    .reshape(1, frames.shape[1], 3520, frames.shape[-1])
                    .to(device=device)
                    .contiguous()
                )
                sources.append((name, chunk, source))
        for name, chunk, source in sources:
            pristine = source.clone()
            exact_base = encode_base(source, name, layer, exact_paths)
            fast_base = encode_base(source, name, layer, fast_paths)
            if not torch.equal(source, pristine):
                raise RuntimeError("RVQ encoder mutated its BF16 input")
            depth_equal = (
                exact_base["relative_stop_actual_depth"]
                == fast_base["relative_stop_actual_depth"]
            )
            same_shape = exact_base["indices"].shape == fast_base["indices"].shape
            index_count = int(exact_base["indices"].numel())
            mismatch_count = (
                int((exact_base["indices"] != fast_base["indices"]).sum())
                if same_shape
                else index_count
            )

            def finish(base: dict[str, Any]) -> torch.Tensor:
                state = create_b16_state(
                    source,
                    base,
                    tensor_name=name,
                    layer_idx=layer,
                    portable_identity=False,
                )
                state = transition_state(
                    state,
                    args.bits,
                    quant_config=residual_config,
                    verify_identity=False,
                )
                return decode_state(state)

            exact = finish(exact_base)
            fast = finish(fast_base)
            delta = fast.float() - exact.float()
            exact_error = exact.float() - source.float()
            fast_error = fast.float() - source.float()
            rows.append(
                {
                    "prompt": args.prompt,
                    "layer": layer,
                    "chunk": chunk,
                    "tensor_name": name,
                    "bits": args.bits,
                    "exact_depth": exact_base["relative_stop_actual_depth"],
                    "fast_depth": fast_base["relative_stop_actual_depth"],
                    "depth_equal": depth_equal,
                    "index_count": index_count,
                    "index_mismatch_count": mismatch_count,
                    "index_mismatch_fraction": mismatch_count
                    / max(index_count, 1),
                    "final_value_mismatch_fraction": float(
                        (exact != fast).float().mean()
                    ),
                    "final_exact_vs_fast_mse": float(delta.square().mean()),
                    "final_exact_vs_fast_max_abs": float(delta.abs().max()),
                    "exact_source_mse": float(exact_error.square().mean()),
                    "fast_source_mse": float(fast_error.square().mean()),
                }
            )
    write_csv(args.output_dir / "near_tie_audit.csv", rows)
    summary = {
        "rows": len(rows),
        "depth_equal_rows": sum(bool(row["depth_equal"]) for row in rows),
        "index_mismatch_count": sum(
            int(row["index_mismatch_count"]) for row in rows
        ),
        "index_count": sum(int(row["index_count"]) for row in rows),
        "final_exact_vs_fast_mse": sum(
            float(row["final_exact_vs_fast_mse"]) for row in rows
        )
        / len(rows),
        "exact_source_mse": sum(float(row["exact_source_mse"]) for row in rows)
        / len(rows),
        "fast_source_mse": sum(float(row["fast_source_mse"]) for row in rows)
        / len(rows),
    }
    summary["source_mse_relative_change"] = (
        summary["fast_source_mse"] / summary["exact_source_mse"] - 1.0
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    run_info = {
        "schema_version": "adaptive_rvq_fast_near_tie.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_json(config),
        "codebook_path": {
            "K": str(shared["k_codebooks"].resolve()),
            "V": str(shared["v_codebooks"].resolve()),
        },
        "codebook_hash": {
            "K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e",
            "V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
        },
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": 20260731,
        "prompt_list": [args.prompt],
        "layer_list": args.layers,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_paths": sorted(set(input_paths)),
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
