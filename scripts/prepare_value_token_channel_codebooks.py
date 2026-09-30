#!/usr/bin/env python3
"""Prepare matched-rate no-norm Value RVQ codebooks for Wan/Self-Forcing.

The two arms use the same sampled raw Value vectors per layer/head:

* token: standard L32C64 over 128 dimensions;
* channel: four 32-D groups, each L8C64.

Both therefore store 32 uint8 indices (192 logical bits) per Value vector.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subitkv.rvq import rvq_codebook_directory_metadata  # noqa: E402


TOKEN_DIRNAME = "value_token_nonorm_v_l32c64_codebooks"
CHANNEL_DIRNAME = "grouped_direct_nonorm_v_l8c64_codebooks"
TRAINER = REPO_ROOT / "scripts" / "rvq" / "diag" / "run_rvq_diag.py"
GROUPED_TRAINER = REPO_ROOT / "scripts" / "run_grouped_dim_rvq_bench.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", choices=["wan5b", "self_forcing_1p3b"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prompt-source", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--stage", choices=["token", "channel", "verify", "all"], default="all")
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--kmeans-max-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token-dirname", default=TOKEN_DIRNAME)
    parser.add_argument("--channel-dirname", default=CHANNEL_DIRNAME)
    return parser.parse_args()


def model_shape(model: str) -> tuple[int, int]:
    return (30, 24) if model == "wan5b" else (30, 12)


def prompt_split(root: Path) -> dict[str, Any]:
    path = root / "prompt_split.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing existing calibration split: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("train"):
        raise ValueError(f"Calibration split has no training prompts: {path}")
    return payload


def run(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def train_token(args: argparse.Namespace) -> None:
    layers, heads = model_shape(args.model)
    split = prompt_split(args.root)
    run([
        str(args.python),
        str(TRAINER),
        "--prompt_source", str(args.prompt_source),
        "--output_dir", str(args.root),
        "--train_num", str(len(split["train"])),
        "--test_num", str(len(split.get("test", []))),
        "--seed", str(args.seed),
        "--first_stage_layers", ",".join(str(index) for index in range(layers)),
        "--gpus", args.gpus,
        "--num_layers", str(layers),
        "--num_heads", str(heads),
        "--levels", "32",
        "--codebook_size", "64",
        "--codebook_tensor", "V",
        "--codebook_subdir", args.token_dirname,
        "--metrics_subdir", "value_token_nonorm_v_l32c64_metrics",
        "--indices_subdir", "value_token_nonorm_v_l32c64_indices",
        "--stage_prefix", "value_token_nonorm_v_l32c64_",
        "--run_config_name", "value_token_nonorm_v_l32c64_run_config.json",
        "--sample_size", str(args.sample_size),
        "--sample_seed_mode", "grouped_dim_v1",
        "--kmeans_max_iters", str(args.kmeans_max_iters),
        "--input_norm_mode", "none",
        "--train_objective", "recon",
        "--skip_dump",
        "--worker_python", str(args.python),
    ])


def train_channel(args: argparse.Namespace) -> None:
    layers, heads = model_shape(args.model)
    if args.channel_dirname != CHANNEL_DIRNAME:
        raise ValueError(
            f"The grouped trainer uses the deployable directory name {CHANNEL_DIRNAME}; "
            "custom --channel-dirname is supported only for verification"
        )
    channel_dir = args.root / args.channel_dirname
    existing = len(list(channel_dir.glob("layer_*_head_*.pt")))
    if existing == layers * heads:
        print(f"[reuse] complete channel codebook set: {channel_dir}", flush=True)
        return
    if existing:
        raise RuntimeError(
            f"Refusing to mix a partial existing channel codebook set in {channel_dir}: "
            f"found {existing}, expected {layers * heads}"
        )
    run([
        str(args.python),
        str(GROUPED_TRAINER),
        "--output-dir", str(args.root),
        "--configs", "l8c64",
        "--modes", "direct_nonorm",
        "--tensors", "V",
        "--gpus", args.gpus,
        "--num-layers", str(layers),
        "--num-heads", str(heads),
        "--sample-size", str(args.sample_size),
        "--kmeans-max-iters", str(args.kmeans_max_iters),
        "--skip-video",
        "--skip-quality",
        "--skip-vbench",
    ])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_once(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FileExistsError(f"Refusing to overwrite different metadata: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.glob("layer_*_head_*.pt")):
        digest.update(item.name.encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def codebook_config(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing codebook config: {path}")
    return config


def calibration_sample_hash(
    root: Path,
    *,
    layer: int,
    head: int,
    sample_size: int,
    seed: int,
) -> str:
    """Recompute the grouped_dim_v1 V sample hash for legacy codebooks."""

    items = prompt_split(root)["train"]
    per_prompt = max(1, math.ceil(sample_size / len(items)))
    worker_seed = seed + layer * 100000 + head * 1000 + 97
    samples = []
    for item in items:
        prompt_id = int(item["prompt_id"])
        path = root / "kv_dump" / "train" / str(prompt_id) / f"layer_{layer}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        value = payload["V"][:, head].reshape(-1, payload["V"].shape[-1])
        generator = torch.Generator(device="cpu").manual_seed(
            worker_seed + prompt_id * 1000003 + layer * 1009 + head
        )
        take = min(per_prompt, value.shape[0])
        indices = torch.randperm(value.shape[0], generator=generator)[:take]
        samples.append(value[indices].float())
    sample = torch.cat(samples, dim=0)
    if sample.shape[0] > sample_size:
        generator = torch.Generator(device="cpu").manual_seed(
            worker_seed + layer * 100000 + head
        )
        sample = sample[torch.randperm(sample.shape[0], generator=generator)[:sample_size]]
    return hashlib.sha256(sample.contiguous().numpy().tobytes()).hexdigest()


def verify(args: argparse.Namespace) -> dict[str, Any]:
    layers, heads = model_shape(args.model)
    expected_files = layers * heads
    token_dir = args.root / args.token_dirname
    channel_dir = args.root / args.channel_dirname
    for directory in (token_dir, channel_dir):
        count = len(list(directory.glob("layer_*_head_*.pt")))
        if count != expected_files:
            raise RuntimeError(f"Expected {expected_files} codebooks in {directory}, found {count}")

    token = rvq_codebook_directory_metadata(token_dir, storage_dtype="int4")
    channel = rvq_codebook_directory_metadata(channel_dir, storage_dtype="int4")
    expected = {
        "codebook_size": 64,
        "physical_index_bytes_per_vector": 32,
        "logical_index_bits_per_vector": 192,
        "input_norm_mode": "none",
        "per_vector_scale_count": 0,
    }
    for name, metadata, rvq_format in (
        ("token", token, "standard"),
        ("channel", channel, "grouped_dim"),
    ):
        if metadata["rvq_format"] != rvq_format:
            raise ValueError(f"{name} format mismatch: {metadata['rvq_format']} != {rvq_format}")
        for key, value in expected.items():
            if metadata[key] != value:
                raise ValueError(f"{name} {key} mismatch: {metadata[key]} != {value}")
        if name == "channel" and metadata["group_transform"] != "none":
            raise ValueError(f"channel group transform must be none, got {metadata['group_transform']}")

    sample_hashes: dict[str, str] = {}
    missing_channel_hashes = 0
    permutation_hashes: set[str] = set()
    for layer in range(layers):
        for head in range(heads):
            key = f"layer_{layer}_head_{head}"
            token_config = codebook_config(token_dir / f"{key}.pt")
            channel_config = codebook_config(channel_dir / f"{key}.pt")
            channel_payload = torch.load(
                channel_dir / f"{key}.pt", map_location="cpu", weights_only=False
            )
            permutation = channel_payload.get("dim_permutation")
            if not torch.is_tensor(permutation) or permutation.numel() != 128:
                raise ValueError(f"Channel codebook has invalid dimension permutation: {key}")
            permutation_hashes.add(
                hashlib.sha256(permutation.long().contiguous().numpy().tobytes()).hexdigest()
            )
            token_hash = token_config.get("sample_manifest_sha256")
            channel_hash = channel_config.get("sample_manifest_sha256")
            if int(token_config.get("sample_size", -1)) != args.sample_size:
                raise ValueError(f"Token codebook sample-size mismatch: {key}")
            if int(channel_config.get("sample_size", -1)) != args.sample_size:
                raise ValueError(f"Channel codebook sample-size mismatch: {key}")
            if not token_hash:
                raise ValueError(f"Token codebook is missing sample hash: {key}")
            if channel_hash and channel_hash != token_hash:
                raise ValueError(f"Calibration sample mismatch for {key}: token={token_hash} channel={channel_hash}")
            if not channel_hash:
                missing_channel_hashes += 1
                channel_hash = calibration_sample_hash(
                    args.root,
                    layer=layer,
                    head=head,
                    sample_size=args.sample_size,
                    seed=args.seed,
                )
                if channel_hash != token_hash:
                    raise ValueError(
                        f"Recomputed legacy channel sample mismatch for {key}: "
                        f"token={token_hash} channel={channel_hash}"
                    )
            sample_hashes[key] = str(token_hash)

    sample_digest = hashlib.sha256(
        json.dumps(sample_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if len(permutation_hashes) != 1:
        raise ValueError(f"Channel permutation differs across codebooks: {len(permutation_hashes)} hashes")
    result = {
        "model": args.model,
        "seed": args.seed,
        "sample_size": args.sample_size,
        "sample_seed_mode": "grouped_dim_v1",
        "sample_manifest_hash": sample_digest,
        "sample_hashes": sample_hashes,
        "legacy_channel_codebooks_without_embedded_sample_hash": missing_channel_hashes,
        "channel_permutation_hash": next(iter(permutation_hashes)),
        "token": {**token, "sha256": sha256_directory(token_dir)},
        "channel": {**channel, "sha256": sha256_directory(channel_dir)},
        "budget_match": {
            "physical_index_bytes_per_vector": 32,
            "logical_index_bits_per_vector": 192,
            "scale_bytes_per_vector": 0,
        },
    }
    manifest_path = args.root / f"value_token_channel_{args.model}_manifest.json"
    write_json_once(manifest_path, result)
    return result


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


def write_run_info(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    split = prompt_split(args.root)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    manifest_path = args.root / f"value_token_channel_{args.model}_manifest.json"
    payload = {
        "git_commit": commit,
        "config_path": str(manifest_path),
        "config_hash": sha256_file(manifest_path),
        "codebook_path": {"token": manifest["token"]["codebook_dir"], "channel": manifest["channel"]["codebook_dir"]},
        "codebook_hash": {"token": manifest["token"]["sha256"], "channel": manifest["channel"]["sha256"]},
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "seed": args.seed,
        "prompt_list": [item["prompt"] for item in split["train"]],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sample_manifest_hash": manifest["sample_manifest_hash"],
        "command": sys.argv,
    }
    path = args.root / f"value_token_channel_{args.model}_run_info.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        stable_keys = ("config_hash", "codebook_hash", "seed", "sample_manifest_hash")
        if any(existing.get(key) != payload.get(key) for key in stable_keys):
            raise FileExistsError(f"Refusing to overwrite different run metadata: {path}")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.root = args.root.resolve()
    if args.stage in {"token", "all"}:
        train_token(args)
    if args.stage in {"channel", "all"}:
        train_channel(args)
    if args.stage in {"verify", "all"}:
        manifest = verify(args)
        write_run_info(args, manifest)
        print(json.dumps({"status": "verified", "manifest": manifest["sample_manifest_hash"]}), flush=True)


if __name__ == "__main__":
    main()
