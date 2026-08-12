#!/usr/bin/env python3
"""Collect prompt-aligned exact Wan2.2 5B attention captures on empty GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from typing import Any

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_config(path: Path) -> tuple[dict[str, Any], str]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return cfg, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() or "unknown"


def prompts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gpu_name() -> str | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def gpu_memory_used() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True, capture_output=True, check=True,
    )
    values = {}
    for line in result.stdout.splitlines():
        index, memory = line.split(",")
        values[int(index.strip())] = int(memory.strip())
    return values


def prompt_root(cfg: dict[str, Any], prompt_id: int) -> Path:
    return resolve(cfg["output_dir"]) / f"prompt_{int(prompt_id):02d}"


def generation_command(cfg: dict[str, Any], prompt_id: int) -> list[str]:
    destination = prompt_root(cfg, prompt_id) / "driver" / "reference_bf16"
    command = [
        str(cfg["python"]), str(RUNNER),
        "--prompt-file", str(cfg["prompt_file"]),
        "--prompt-start", str(int(prompt_id)),
        "--num-prompts", "1",
        "--output-dir", str(destination),
        "--seed", str(int(cfg["seed"])),
        "--num_frames", str(int(cfg["num_frames"])),
        "--num_chunk", str(int(cfg["num_chunks"])),
        "--setting", "bf16",
        "--k_rvq_codebook_dir", "/data/bowen/rvq_diag/codebooks",
        "--v_rvq_codebook_dir", "/data/bowen/rvq_diag/v_codebooks",
    ]
    if bool(cfg.get("skip_decode", False)):
        command.append("--skip_decode")
    return command


def generation_environment(
    cfg: dict[str, Any], prompt_id: int, gpu: int
) -> dict[str, str]:
    env = os.environ.copy()
    capture = cfg["capture"]
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(int(gpu)),
            "MASTER_PORT": str(29700 + int(prompt_id)),
            "HYWORLDPLAY_KV_CACHE_MODE": str(cfg["kv_cache_mode"]),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP": "1",
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP_DIR": str(
                prompt_root(cfg, prompt_id) / "paired_rope"
            ),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_LAYERS": ",".join(
                str(int(value)) for value in capture["layers"]
            ),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_MIN_CACHE_TOKENS": str(
                int(capture["min_historical_cache_tokens"])
            ),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_QUERY_SAMPLE": str(int(capture["query_sample"])),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_KEY_SAMPLE": str(int(capture["key_sample_limit"])),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_MAX_CALLS_PER_LAYER": str(
                int(capture["max_calls_per_layer"])
            ),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_SEED": str(int(capture["seed"])),
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_LEAN": "1" if bool(capture.get("lean", False)) else "0",
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_POST_VALIDATION_SAMPLE": str(
                int(capture.get("post_validation_sample", 64))
            ),
            "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
            "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
            "HYWORLDPLAY_ENABLE_VAE_TILING": "0",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def base_run_info(
    cfg: dict[str, Any], cfg_path: Path, cfg_hash: str, prompt_list: list[str]
) -> dict[str, Any]:
    registry = resolve(cfg["hypothesis_registry"])
    return {
        "git_commit": git_commit(),
        "config_path": str(cfg_path.resolve()),
        "config_hash": cfg_hash,
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_name(),
        "seed": int(cfg["seed"]),
        "prompt_list": prompt_list,
        "timestamp": utc_now(),
        "model_id": cfg["model_id"],
        "source_capture_is_bf16": True,
        "hypothesis_registry": str(registry.resolve()),
        "hypothesis_registry_hash": sha256_file(registry),
        "capture_scope": cfg["capture"],
        "kv_cache_mode": cfg["kv_cache_mode"],
    }


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg_path = resolve(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = resolve(cfg["output_dir"])
    info_path = root / "run_info.json"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite capture root: {root}")
    selected = [int(value) for value in cfg["prompt_ids"]]
    gpus = [int(value) for value in cfg["gpus"]]
    if len(selected) != len(gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("Capture requires one distinct GPU per prompt")
    all_prompts = prompts(Path(cfg["prompt_file"]))
    if min(selected) < 0 or max(selected) >= len(all_prompts):
        raise ValueError("Prompt IDs are out of range")
    base = base_run_info(
        cfg, cfg_path, cfg_hash, [all_prompts[prompt_id] for prompt_id in selected]
    )
    atomic_json(info_path, {**base, "prompt_ids": selected, "gpus": gpus})
    for prompt_id, gpu in zip(selected, gpus):
        destination = prompt_root(cfg, prompt_id)
        atomic_json(
            destination / "run_info.json",
            {
                **base,
                "prompt_id": prompt_id,
                "prompt_start": prompt_id,
                "prompt_list": [all_prompts[prompt_id]],
                "gpu_physical_index": gpu,
                "command": generation_command(cfg, prompt_id),
                "status": "prepared",
            },
        )
    print(f"[prepare] root={root} prompts={selected} gpus={gpus} config_hash={cfg_hash}")


def validate_capture_file(path: Path, expected_layer: int) -> dict[str, Any]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(raw.get("metadata", {}))
    required = {
        "query_pre", "key_pre", "value_pre", "query_post_rope", "query_post_prope",
        "query_position_indices", "key_position_indices", "key_indices",
    }
    if bool(metadata.get("lean_reference_reconstructed", False)):
        required.update(
            {
                "post_validation_offsets", "key_post_rope_validation",
                "value_post_rope_validation", "key_post_prope_validation",
                "value_post_prope_validation",
            }
        )
    else:
        required.update(
            {"key_post_rope", "value_post_rope", "key_post_prope", "value_post_prope"}
        )
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"Capture {path} missing fields: {missing}")
    if int(metadata.get("layer", -1)) != int(expected_layer):
        raise ValueError(f"Capture layer mismatch: {path} {metadata}")
    if metadata.get("alignment") != "exact_same_call_same_indices":
        raise ValueError(f"Capture alignment mismatch: {path} {metadata}")
    key_len = int(metadata["key_len"])
    if int(metadata["key_sample"]) != key_len or int(raw["key_pre"].shape[2]) != key_len:
        raise ValueError(f"Capture key axis is incomplete: {path}")
    if not torch.equal(raw["key_indices"], torch.arange(key_len)):
        raise ValueError(f"Capture key order is not canonical: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "layer": int(expected_layer),
        "history_tokens": int(metadata["cache_tokens"]),
        "current_tokens": key_len - int(metadata["cache_tokens"]),
        "alignment": metadata["alignment"],
        "lean_reference_reconstructed": bool(metadata.get("lean_reference_reconstructed", False)),
        "post_validation_sample": int(metadata.get("post_validation_sample", key_len)),
    }


def cmd_launch(args: argparse.Namespace) -> None:
    cfg_path = resolve(args.config)
    cfg, cfg_hash = parse_config(cfg_path)
    root = resolve(cfg["output_dir"])
    root_info_path = root / "run_info.json"
    if not root_info_path.exists():
        raise FileNotFoundError("Run prepare first")
    root_info = json.loads(root_info_path.read_text(encoding="utf-8"))
    if root_info.get("config_hash") != cfg_hash:
        raise ValueError("Prepared config hash mismatch")
    selected = [int(value) for value in cfg["prompt_ids"]]
    gpus = [int(value) for value in cfg["gpus"]]
    used = gpu_memory_used()
    max_preexisting_mib = int(cfg.get("max_preexisting_gpu_mib", 1024))
    unavailable = {
        gpu: used.get(gpu)
        for gpu in gpus
        if used.get(gpu, 10**9) >= max_preexisting_mib
    }
    if unavailable:
        raise RuntimeError(f"Capture GPUs are no longer empty: {unavailable}")
    processes = []
    for prompt_id, gpu in zip(selected, gpus):
        destination = prompt_root(cfg, prompt_id)
        log_path = destination / "generation.log"
        if log_path.exists():
            raise FileExistsError(log_path)
        handle = log_path.open("x", encoding="utf-8")
        process = subprocess.Popen(
            generation_command(cfg, prompt_id), cwd=REPO_ROOT,
            env=generation_environment(cfg, prompt_id, gpu),
            stdout=handle, stderr=subprocess.STDOUT,
        )
        processes.append((prompt_id, gpu, process, handle, log_path))
        print(f"[launch] prompt={prompt_id} gpu={gpu} pid={process.pid} log={log_path}", flush=True)
    last_report = 0.0
    try:
        while any(process.poll() is None for _prompt, _gpu, process, _handle, _log in processes):
            if time.time() - last_report >= 10:
                print(
                    "[monitor] " + ", ".join(
                        f"p{prompt}:gpu{gpu}:pid{process.pid}:"
                        f"{'running' if process.poll() is None else process.returncode}"
                        for prompt, gpu, process, _handle, _log in processes
                    ), flush=True,
                )
                last_report = time.time()
            time.sleep(2)
    finally:
        for _prompt, _gpu, _process, handle, _log in processes:
            handle.close()
    failed = [item for item in processes if item[2].returncode != 0]
    if failed:
        for prompt_id, gpu, process, _handle, log_path in failed:
            print(
                f"[failed] prompt={prompt_id} gpu={gpu} rc={process.returncode} log={log_path}\n"
                f"{log_path.read_text(encoding='utf-8')[-16000:]}", flush=True,
            )
        raise RuntimeError(f"{len(failed)} capture process(es) failed; outputs preserved")
    manifest = []
    layers = [int(value) for value in cfg["capture"]["layers"]]
    for prompt_id, gpu, _process, _handle, _log_path in processes:
        prompt_manifest = []
        for layer in layers:
            path = (
                prompt_root(cfg, prompt_id) / "paired_rope" / "rank_000"
                / f"layer_{layer:02d}" / "call_00000.pt"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            prompt_manifest.append(validate_capture_file(path, layer))
        manifest.extend({"prompt_id": prompt_id, **row} for row in prompt_manifest)
        info_path = prompt_root(cfg, prompt_id) / "run_info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info.update(
            {
                "status": "complete", "completed_timestamp": utc_now(),
                "capture_files": prompt_manifest,
                "paired_dump_bytes": sum(int(row["bytes"]) for row in prompt_manifest),
            }
        )
        atomic_json(info_path, info)
    atomic_json(
        root / "capture_manifest.json",
        {
            "schema_version": "wan5b_p1_multi_prompt_capture_manifest.v1",
            "config_hash": cfg_hash, "completed": utc_now(),
            "files": manifest,
            "total_bytes": sum(int(row["bytes"]) for row in manifest),
        },
    )
    root_info.update(
        {
            "status": "complete", "completed_timestamp": utc_now(),
            "capture_manifest": "capture_manifest.json",
            "capture_manifest_hash": sha256_file(root / "capture_manifest.json"),
        }
    )
    atomic_json(root_info_path, root_info)
    print(
        f"[complete] prompts={len(selected)} captures={len(manifest)} "
        f"bytes={sum(int(row['bytes']) for row in manifest)}", flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "launch"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "launch":
        cmd_launch(args)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
