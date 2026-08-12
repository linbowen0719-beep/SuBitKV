#!/usr/bin/env python3
"""Build coarse offline K/V post-RVQ calibration scale tables on a train split."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.compress import _k_query_block_scale, _v_attention_block_scale
from quant_videogen.rvq import _compute_dtype, rvq_dequantize_tensor_cache, rvq_encode_tensor_cache


def parse_csv_ints(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(30))
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("integer list cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dump-root", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test/kv_dump"))
    p.add_argument("--split-json", type=Path, default=Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompt_split.json"))
    p.add_argument("--split", default="train")
    p.add_argument("--prompt-limit", type=int, default=20)
    p.add_argument("--layers", default="all")
    p.add_argument("--frame-limit", type=int, default=4)
    p.add_argument("--block-size", type=int, default=880)
    p.add_argument("--query-sample", type=int, default=96)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-codebook-dir", type=Path, required=True)
    p.add_argument("--v-codebook-dir", type=Path, required=True)
    p.add_argument("--codebook-storage-dtype", default="int4", choices=["bf16", "int8", "int4", "int2"])
    p.add_argument("--scale-storage-dtype", default="uint4", choices=["bf16", "fp16", "fp32", "uint8", "uint4"])
    p.add_argument("--compute-dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--clamp-min", type=float, default=0.75)
    p.add_argument("--clamp-max", type=float, default=1.33)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_prompt_ids(path: Path, split: str, prompt_limit: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get(split)
    if not isinstance(items, list):
        raise ValueError(f"{path} missing split {split!r}")
    ids = [int(item["prompt_id"]) for item in items]
    if prompt_limit > 0:
        ids = ids[: int(prompt_limit)]
    if not ids:
        raise ValueError(f"No prompt ids selected for split={split}")
    return ids


def flatten_bhsd(x: torch.Tensor, frame_limit: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected tensor [T,H,S,D], got {tuple(x.shape)}")
    frames = min(int(frame_limit), int(x.shape[0])) if int(frame_limit) > 0 else int(x.shape[0])
    x = x[:frames].contiguous()
    return x.permute(1, 0, 2, 3).reshape(1, x.shape[1], frames * x.shape[2], x.shape[3]).to(device=device, dtype=dtype)


def sample_query(q: torch.Tensor, sample: int, seed: int) -> torch.Tensor:
    total = int(q.shape[2])
    count = min(int(sample), total)
    if count >= total:
        return q
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(total, generator=gen)[:count].to(device=q.device)
    return q.index_select(2, idx).contiguous()


@torch.no_grad()
def layer_prompt_scales(
    args: argparse.Namespace,
    *,
    prompt_id: int,
    layer: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = args.dump_root / args.split / str(prompt_id) / f"layer_{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    k = flatten_bhsd(payload["K"], args.frame_limit, device, compute_dtype)
    v = flatten_bhsd(payload["V"], args.frame_limit, device, compute_dtype)
    q = flatten_bhsd(payload["Q"], args.frame_limit, device, compute_dtype)
    q = sample_query(q, args.query_sample, args.seed + prompt_id * 1009 + layer * 9176)

    k_packed = rvq_encode_tensor_cache(
        k,
        layer_idx=layer,
        codebook_dir=args.k_codebook_dir,
        batch_size=args.batch_size,
        compute_dtype=args.compute_dtype,
        codebook_storage_dtype=args.codebook_storage_dtype,
        scale_storage_dtype=args.scale_storage_dtype,
    )
    k_recon = rvq_dequantize_tensor_cache(k_packed, k.dtype)
    k_scale = _k_query_block_scale(
        q,
        k,
        k_recon,
        block_size=args.block_size,
        clamp=(args.clamp_min, args.clamp_max),
    )

    v_packed = rvq_encode_tensor_cache(
        v,
        layer_idx=layer,
        codebook_dir=args.v_codebook_dir,
        batch_size=args.batch_size,
        compute_dtype=args.compute_dtype,
        codebook_storage_dtype=args.codebook_storage_dtype,
        scale_storage_dtype=args.scale_storage_dtype,
    )
    v_recon = rvq_dequantize_tensor_cache(v_packed, v.dtype)
    v_scale = _v_attention_block_scale(
        q,
        k,
        v,
        v_recon,
        block_size=args.block_size,
        clamp=(args.clamp_min, args.clamp_max),
    )
    return k_scale.detach().cpu(), v_scale.detach().cpu()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.clamp_min <= 0 or args.clamp_max < args.clamp_min:
        raise ValueError("--clamp-min/--clamp-max are invalid")
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; choose a new path to avoid overwriting")
    device = torch.device(args.device)
    compute_dtype = _compute_dtype(args.compute_dtype, device)
    layers = parse_csv_ints(args.layers)
    prompt_ids = split_prompt_ids(args.split_json, args.split, args.prompt_limit)

    k_rows: dict[int, torch.Tensor] = {}
    v_rows: dict[int, torch.Tensor] = {}
    for layer in layers:
        k_sum = None
        v_sum = None
        count = 0
        for prompt_id in prompt_ids:
            k_scale, v_scale = layer_prompt_scales(
                args,
                prompt_id=prompt_id,
                layer=int(layer),
                device=device,
                compute_dtype=compute_dtype,
            )
            k_sum = k_scale if k_sum is None else k_sum + k_scale
            v_sum = v_scale if v_sum is None else v_sum + v_scale
            count += 1
            print(
                f"[calib] layer={layer:02d} prompt={prompt_id} "
                f"k_blocks={k_scale.shape[2]} v_blocks={v_scale.shape[2]}",
                flush=True,
            )
        if k_sum is None or v_sum is None or count == 0:
            raise RuntimeError(f"No calibration samples for layer {layer}")
        k_rows[int(layer)] = (k_sum / count).squeeze(0).contiguous()
        v_rows[int(layer)] = (v_sum / count).squeeze(0).contiguous()

    max_layer = max(layers)
    if set(layers) != set(range(max_layer + 1)):
        raise ValueError("Calibration table currently expects contiguous layers starting at 0")
    k_table = torch.stack([k_rows[layer] for layer in range(max_layer + 1)], dim=0).contiguous()
    v_table = torch.stack([v_rows[layer] for layer in range(max_layer + 1)], dim=0).contiguous()
    metadata: dict[str, Any] = {
        "git_commit": git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "dump_root": str(args.dump_root),
        "split_json": str(args.split_json),
        "split": args.split,
        "prompt_ids": prompt_ids,
        "layers": layers,
        "frame_limit": int(args.frame_limit),
        "block_size": int(args.block_size),
        "query_sample": int(args.query_sample),
        "seed": int(args.seed),
        "k_codebook_dir": str(args.k_codebook_dir),
        "v_codebook_dir": str(args.v_codebook_dir),
        "codebook_storage_dtype": args.codebook_storage_dtype,
        "compute_dtype": args.compute_dtype,
        "clamp": [float(args.clamp_min), float(args.clamp_max)],
    }
    payload = {
        "k": k_table,
        "v": v_table,
        "block_size": int(args.block_size),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata["output"] = str(args.output)
    metadata["output_sha256"] = sha256_file(args.output)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[done] wrote {args.output}", flush=True)
    print(f"[done] metadata {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
