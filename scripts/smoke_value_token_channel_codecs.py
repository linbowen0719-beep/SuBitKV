#!/usr/bin/env python3
"""Run a non-generative codec smoke on one real calibration prompt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for item in sorted(path.glob("layer_*_head_*.pt")):
            digest.update(item.name.encode())
            digest.update(sha256(item).encode())
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_type() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    return ", ".join(dict.fromkeys(line.strip() for line in output.splitlines() if line.strip())) or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("wan5b", "self_forcing_1p3b"), required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--token-v-codebooks", type=Path, required=True)
    parser.add_argument("--channel-v-codebooks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=2)
    args = parser.parse_args()
    completed = args.output_dir / "completed.json"
    if completed.exists():
        print(f"already complete: {args.output_dir}")
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite partial smoke output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_path = args.calibration_root / "prompt_split.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    prompt_item = split["train"][0]
    prompt_id = int(prompt_item["prompt_id"])
    dump_path = args.calibration_root / "kv_dump" / "train" / str(prompt_id) / "layer_0.pt"
    payload = torch.load(dump_path, map_location="cpu", weights_only=False)
    value = payload["V"][:, :, : args.tokens, :].float().contiguous()
    expected_heads = 24 if args.model == "wan5b" else 12
    if value.shape[1:] != (expected_heads, args.tokens, 128):
        raise ValueError(f"Unexpected calibration Value shape: {tuple(value.shape)}")
    os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = "32"

    results = {}
    for name, directory in (
        ("per_token", args.token_v_codebooks),
        ("per_channel", args.channel_v_codebooks),
    ):
        packed = rvq_encode_tensor_cache(
            value,
            layer_idx=0,
            codebook_dir=directory,
            batch_size=64,
            compute_dtype="fp32",
            codebook_storage_dtype="int4",
            scale_storage_dtype="uint4",
        )
        vectors = value.shape[0] * value.shape[1] * value.shape[2]
        indices_per_vector = packed["indices"].numel() // vectors
        if packed["indices"].dtype != torch.uint8 or indices_per_vector != 32:
            raise ValueError(f"{name} did not produce 32 uint8 indices/vector")
        if any("scale" in key for key in packed if key != "codebook_storage_dtype"):
            raise ValueError(f"{name} unexpectedly stored a per-vector scale")
        reconstructed = rvq_dequantize_tensor_cache(packed, value.dtype)
        error = value - reconstructed
        results[name] = {
            "indices_per_vector": indices_per_vector,
            "index_dtype": str(packed["indices"].dtype),
            "raw_mse": float(error.square().mean().item()),
            "relative_l2": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(value)),
        }

    config = {
        "model": args.model,
        "calibration_dump": str(dump_path),
        "prompt_id": prompt_id,
        "prompt": prompt_item["prompt"],
        "tokens": args.tokens,
        "seed": 42,
        "token_v_codebooks": str(args.token_v_codebooks),
        "channel_v_codebooks": str(args.channel_v_codebooks),
    }
    config_path = args.output_dir / "smoke_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    run_info = {
        "git_commit": commit,
        "config_path": str(config_path),
        "config_hash": sha256(config_path),
        "codebook_path": {
            "token": str(args.token_v_codebooks),
            "channel": str(args.channel_v_codebooks),
        },
        "codebook_hash": {
            "token": sha256(args.token_v_codebooks),
            "channel": sha256(args.channel_v_codebooks),
        },
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "seed": 42,
        "prompt_list": [prompt_item["prompt"]],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "result_source": "new",
        "results": results,
    }
    (args.output_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completed.write_text(json.dumps({"status": "complete", "results": results}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "output": str(args.output_dir)}))


if __name__ == "__main__":
    main()
