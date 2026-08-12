#!/usr/bin/env python3
"""Long-video KV-cache benchmark orchestrator.

The default actions are intentionally non-destructive. Use --dry-run to inspect
retention/compression decisions, --print-commands to generate the full command
plan, and --execute only when you want to launch jobs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "long_video" / "l8c64_int4_21chunk.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "long_video"
DEFAULT_WORKER_PYTHON = Path("/home/bowen/anaconda3/envs/hw8/bin/python")
DEFAULT_VBENCH_BIN = Path("/home/bowen/anaconda3/bin/vbench")
DEFAULT_VBENCH_DIMS = [
    "subject_consistency",
    "background_consistency",
    "imaging_quality",
    "aesthetic_quality",
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.rvq import rvq_codebook_directory_metadata  # noqa: E402


@dataclass(frozen=True)
class SettingSpec:
    name: str
    label: str
    retention_policy: str
    compression_method: str
    extra_args: tuple[str, ...]
    env: dict[str, str]
    is_reference: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--settings", default="all", help="Comma-separated setting names or 'all'.")
    parser.add_argument("--prompt-start-override", type=int)
    parser.add_argument("--num-prompts-override", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Print the 21-chunk retention/compression plan.")
    parser.add_argument("--print-commands", action="store_true", help="Print generation, quality, and optional VBench commands.")
    parser.add_argument("--write-commands", action="store_true", help="Write the command plan under the run directory.")
    parser.add_argument("--execute", action="store_true", help="Run the planned commands.")
    parser.add_argument("--allow-full-run", action="store_true", help="Required with --execute unless --smoke is set.")
    parser.add_argument("--smoke", action="store_true", help="Use the config's minimal smoke overrides.")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate existing outputs into summary.csv.")
    parser.add_argument("--no-aggregate", action="store_true", help="Do not refresh summary.csv after --execute.")
    parser.add_argument("--resume", action="store_true", help="Reuse an existing run directory and skip completed settings.")
    parser.add_argument("--skip-decode", action="store_true", help="Run latent-only generation without VAE video decode/export.")
    parser.add_argument("--skip-quality", action="store_true")
    vbench = parser.add_mutually_exclusive_group()
    vbench.add_argument("--run_vbench", action="store_true", help="Run VBench if the binary is available.")
    vbench.add_argument("--skip_vbench", action="store_true", help="Do not run VBench.")
    parser.add_argument("--gpus", default="", help="CUDA_VISIBLE_DEVICES value for execution. Empty uses current env.")
    parser.add_argument("--vbench-ngpus", type=int, default=1)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError("PyYAML is required to load YAML long-video configs.") from exc
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def repo_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_id_from_args(args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    mode = "smoke" if args.smoke else "full"
    return f"{utc_timestamp()}_{mode}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(item).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


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


def gpu_type() -> str | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    names = [line.strip() for line in output.splitlines() if line.strip()]
    return ", ".join(dict.fromkeys(names)) if names else None


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(out.get(key), dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolved_benchmark_config(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    resolved = dict(config)
    if smoke:
        smoke_cfg = config.get("smoke", {})
        if not isinstance(smoke_cfg, dict):
            raise ValueError("smoke config must be a mapping")
        resolved = deep_merge(config, smoke_cfg)

    benchmark = resolved.setdefault("benchmark", {})
    if not isinstance(benchmark, dict):
        raise ValueError("benchmark config must be a mapping")
    chunk_size = int(benchmark.get("chunk_size", 4))
    pred_latent_size = int(benchmark.get("pred_latent_size", 4))
    num_chunks = int(benchmark.get("num_chunks", 21))
    benchmark.setdefault("num_frames", 4 * num_chunks * pred_latent_size - 3)
    benchmark.setdefault("kv_reference_frames", num_chunks * chunk_size)
    benchmark.setdefault("kv_logical_head_dim", 256)
    benchmark.setdefault("prompt_start", 0)
    benchmark.setdefault("num_prompts", 1)
    benchmark.setdefault("seed", 42)
    benchmark.setdefault("image_path", "none")
    benchmark.setdefault("pose", "w-96")
    benchmark.setdefault("memory_frames", 80)
    benchmark.setdefault("temporal_context_size", 76)
    benchmark.setdefault("num_inference_steps", 50)
    benchmark.setdefault("offload_text_encoder", True)
    return resolved


def setting_specs(config: dict[str, Any]) -> list[SettingSpec]:
    benchmark = config["benchmark"]
    chunk_size = int(benchmark.get("chunk_size", 4))
    rvq = config.get("rvq", {})
    rvq_l64 = config.get("rvq_l64", {})
    rvq_l16 = config.get("rvq_l16", rvq_l64)
    rvq_progressive = config.get("rvq_progressive", {})
    rvq_progressive_v2 = config.get("rvq_progressive_v2", rvq_progressive)
    rvq_age_ramp16 = config.get("rvq_age_ramp16")
    rvq_calibration = config.get("rvq_calibration", {})
    rvq_diagnostics = config.get("rvq_diagnostics", {})
    rvq_k_token_v_channel = config.get("rvq_k_token_v_channel", {})
    rvq_k_token_v_token_l32 = config.get("rvq_k_token_v_token_l32", {})
    native_int4 = config.get("native_int4", {})
    native_int2 = config.get("native_int2", native_int4)
    fp8 = config.get("fp8", {})
    k_codebook_dir = str(repo_path(rvq.get("k_codebook_dir", "/data/bowen/t2v_kv_bench_moviegen_20train_10test/sweep_k_l8c64_codebooks")))
    v_codebook_dir = str(repo_path(rvq.get("v_codebook_dir", "/data/bowen/t2v_kv_bench_moviegen_20train_10test/sweep_v_l8c64_codebooks")))
    k_l64_codebook_dir = str(repo_path(rvq_l64.get("k_codebook_dir", "/data/bowen/t2v_kv_bench_moviegen_20train_10test/sweep_k_l64c64_codebooks")))
    v_l64_codebook_dir = str(repo_path(rvq_l64.get("v_codebook_dir", "/data/bowen/t2v_kv_bench_moviegen_20train_10test/sweep_v_l64c64_codebooks")))
    k_l16_codebook_dir = str(repo_path(rvq_l16.get("k_codebook_dir", k_l64_codebook_dir)))
    v_l16_codebook_dir = str(repo_path(rvq_l16.get("v_codebook_dir", v_l64_codebook_dir)))
    mixed_k_codebook_dir = str(repo_path(rvq_k_token_v_channel.get("k_codebook_dir", k_codebook_dir)))
    mixed_v_codebook_dir = str(repo_path(rvq_k_token_v_channel.get("v_codebook_dir", v_codebook_dir)))
    matched_token_k_codebook_dir = str(
        repo_path(rvq_k_token_v_token_l32.get("k_codebook_dir", mixed_k_codebook_dir))
    )
    matched_token_v_codebook_dir = str(
        repo_path(rvq_k_token_v_token_l32.get("v_codebook_dir", v_codebook_dir))
    )

    cpu_offload_env = {
        "HYWORLDPLAY_OFFLOAD_FOR_DECODE": "1",
        "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE": "1",
        "HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD": "1",
    }
    matched_value_env = {**cpu_offload_env, "HYWORLDPLAY_RVQ_NUM_LEVELS": "32"}
    common_reference = (
        "--kv_logical_head_dim",
        str(int(benchmark["kv_logical_head_dim"])),
        "--kv_reference_frames",
        str(int(benchmark["kv_reference_frames"])),
    )
    perchunk = ("--kv_quant_factor", "1")
    first_recent_quant = (
        "--kv_quant_skip_chunk_policy",
        "first",
        "--kv_quant_recent_policy",
        "nearest1",
    )
    first_recent_sparse = (
        "--kv_sparse_policy",
        "first_recent_chunks",
        "--kv_sparse_first_frames",
        str(chunk_size),
        "--kv_sparse_recent_frames",
        str(chunk_size),
    )
    def native_args(setting: str, native_cfg: dict[str, Any], keep_first_recent: bool) -> tuple[str, ...]:
        args = (
            "--setting",
            setting,
            "--quant_block_size",
            str(int(native_cfg.get("quant_block_size", 64))),
            "--cache_num_k_centroids",
            str(int(native_cfg.get("cache_num_k_centroids", 256))),
            "--cache_num_v_centroids",
            str(int(native_cfg.get("cache_num_v_centroids", 256))),
            "--kmeans_max_iters",
            str(int(native_cfg.get("kmeans_max_iters", 2))),
            "--num_prq_stages",
            str(int(native_cfg.get("num_prq_stages", 1))),
            *common_reference,
            *perchunk,
        )
        if keep_first_recent:
            args = (*args, *first_recent_quant)
        return args

    native_int4_args = native_args("int4", native_int4, keep_first_recent=True)
    all_int4_args = native_args("int4", native_int4, keep_first_recent=False)
    all_int2_args = native_args("int2", native_int2, keep_first_recent=False)
    fp8_args = (
        "--setting",
        str(fp8.get("setting", "fp8")),
        "--quant_type",
        str(fp8.get("quant_type", "fp8-e4m3")),
        "--quant_block_size",
        str(int(fp8.get("quant_block_size", 64))),
        *common_reference,
        *perchunk,
        *first_recent_quant,
    )
    first_recent_fp8_sparse_args = (
        "--setting",
        str(fp8.get("setting", "fp8")),
        "--quant_type",
        str(fp8.get("quant_type", "fp8-e4m3")),
        "--quant_block_size",
        str(int(fp8.get("quant_block_size", 64))),
        *common_reference,
        *perchunk,
        *first_recent_sparse,
        "--kv_sparse_quantize_retained",
    )
    rvq_args_base = (
        "--setting",
        "kvrvq-i4",
        "--k_rvq_codebook_dir",
        k_codebook_dir,
        "--v_rvq_codebook_dir",
        v_codebook_dir,
        "--rvq_codebook_storage_dtype",
        str(rvq.get("codebook_storage_dtype", "int4")),
        "--rvq_scale_storage_dtype",
        str(rvq.get("scale_storage_dtype", "uint4")),
        "--kv_logical_head_dim",
        str(int(benchmark["kv_logical_head_dim"])),
        "--kv_reference_frames",
        str(int(benchmark["kv_reference_frames"])),
        *perchunk,
    )
    rvq_args = (*rvq_args_base, *first_recent_quant)
    rvq_diagnostic_args: tuple[str, ...] = ()
    if as_bool(rvq_diagnostics.get("enabled"), False):
        rvq_diagnostic_args = (
            "--rvq_diagnostics",
            "--rvq_feedback_query_sample",
            str(int(rvq_diagnostics.get("query_sample", 96))),
        )

    def rvq_args_for_dirs(
        *,
        k_dir: str,
        v_dir: str,
        codebook_dtype: str,
        scale_dtype: str,
        keep_first_recent: bool,
        num_levels: int | None = None,
    ) -> tuple[str, ...]:
        args = (
            "--setting",
            "kvrvq-i4",
            "--k_rvq_codebook_dir",
            k_dir,
            "--v_rvq_codebook_dir",
            v_dir,
            "--rvq_codebook_storage_dtype",
            str(codebook_dtype),
            "--rvq_scale_storage_dtype",
            str(scale_dtype),
            "--kv_logical_head_dim",
            str(int(benchmark["kv_logical_head_dim"])),
            "--kv_reference_frames",
            str(int(benchmark["kv_reference_frames"])),
            *perchunk,
            *rvq_diagnostic_args,
        )
        if num_levels is not None:
            args = (*args, "--rvq_num_levels", str(int(num_levels)))
        if keep_first_recent:
            args = (*args, *first_recent_quant)
        return args

    def calibration_args(mode: str, *, keep_first_recent: bool) -> tuple[str, ...]:
        args = (
            *rvq_args_for_dirs(
                k_dir=k_codebook_dir,
                v_dir=v_codebook_dir,
                codebook_dtype=str(rvq.get("codebook_storage_dtype", "int4")),
                scale_dtype=str(rvq.get("scale_storage_dtype", "uint4")),
                keep_first_recent=keep_first_recent,
            ),
            "--rvq_k_calibration",
            mode,
            "--rvq_v_calibration",
            mode,
            "--rvq_calibration_block_size",
            str(int(rvq_calibration.get("block_size", 880))),
            "--rvq_calibration_min",
            str(float(rvq_calibration.get("min", 0.75))),
            "--rvq_calibration_max",
            str(float(rvq_calibration.get("max", 1.33))),
            "--rvq_calibration_storage_dtype",
            str(rvq_calibration.get("storage_dtype", "uint4")),
            "--rvq_feedback_query_sample",
            str(int(rvq_calibration.get("query_sample", 96))),
        )
        path = rvq_calibration.get("path")
        if path:
            args = (*args, "--rvq_calibration_path", str(repo_path(path)))
        return args

    def mixed_k_token_v_channel_args(*, keep_first_recent: bool) -> tuple[str, ...]:
        return rvq_args_for_dirs(
            k_dir=mixed_k_codebook_dir,
            v_dir=mixed_v_codebook_dir,
            codebook_dtype=str(rvq_k_token_v_channel.get("codebook_storage_dtype", rvq.get("codebook_storage_dtype", "int4"))),
            scale_dtype=str(rvq_k_token_v_channel.get("scale_storage_dtype", rvq.get("scale_storage_dtype", "uint4"))),
            keep_first_recent=keep_first_recent,
        )

    def mixed_k_token_v_token_l32_args(*, keep_first_recent: bool) -> tuple[str, ...]:
        return rvq_args_for_dirs(
            k_dir=matched_token_k_codebook_dir,
            v_dir=matched_token_v_codebook_dir,
            codebook_dtype=str(
                rvq_k_token_v_token_l32.get(
                    "codebook_storage_dtype", rvq.get("codebook_storage_dtype", "int4")
                )
            ),
            scale_dtype=str(
                rvq_k_token_v_token_l32.get(
                    "scale_storage_dtype", rvq.get("scale_storage_dtype", "uint4")
                )
            ),
            keep_first_recent=keep_first_recent,
        )

    def progressive_v2_args(granularity: str, *, keep_first_recent: bool) -> tuple[str, ...]:
        threshold = rvq_progressive_v2.get(f"{granularity}_error_threshold", rvq_progressive_v2.get("error_threshold", 0.0))
        args = (
            *rvq_args_for_dirs(
                k_dir=k_l16_codebook_dir,
                v_dir=v_l16_codebook_dir,
                codebook_dtype=str(rvq_l16.get("codebook_storage_dtype", rvq_l64.get("codebook_storage_dtype", "int4"))),
                scale_dtype=str(rvq_l16.get("scale_storage_dtype", rvq_l64.get("scale_storage_dtype", "uint4"))),
                keep_first_recent=keep_first_recent,
            ),
            "--rvq_progressive",
            "--rvq_progressive_min_depth",
            str(int(rvq_progressive_v2.get("min_depth", 4))),
            "--rvq_progressive_max_depth",
            str(int(rvq_progressive_v2.get("max_depth", 16))),
            "--rvq_progressive_error_threshold",
            str(float(threshold)),
            "--rvq_progressive_granularity",
            granularity,
        )
        if granularity == "block":
            args = (
                *args,
                "--rvq_progressive_block_size",
                str(int(rvq_progressive_v2.get("block_size", 880))),
            )
        return args

    def rvq_l64_args(num_levels: int, *, keep_first_recent: bool = True) -> tuple[str, ...]:
        args = (
            "--setting",
            "kvrvq-i4",
            "--k_rvq_codebook_dir",
            k_l64_codebook_dir,
            "--v_rvq_codebook_dir",
            v_l64_codebook_dir,
            "--rvq_num_levels",
            str(int(num_levels)),
            "--rvq_codebook_storage_dtype",
            str(rvq_l64.get("codebook_storage_dtype", "int4")),
            "--rvq_scale_storage_dtype",
            str(rvq_l64.get("scale_storage_dtype", "uint4")),
            "--kv_logical_head_dim",
            str(int(benchmark["kv_logical_head_dim"])),
            "--kv_reference_frames",
            str(int(benchmark["kv_reference_frames"])),
            *perchunk,
        )
        if keep_first_recent:
            args = (*args, *first_recent_quant)
        return args

    def progressive_args(*, keep_first_recent: bool = True) -> tuple[str, ...]:
        args = (
            "--setting",
            "kvrvq-i4",
            "--k_rvq_codebook_dir",
            k_l64_codebook_dir,
            "--v_rvq_codebook_dir",
            v_l64_codebook_dir,
            "--rvq_progressive",
            "--rvq_progressive_min_depth",
            str(int(rvq_progressive.get("min_depth", 8))),
            "--rvq_progressive_max_depth",
            str(int(rvq_progressive.get("max_depth", 32))),
            "--rvq_progressive_error_threshold",
            str(float(rvq_progressive.get("error_threshold", 0.0))),
            "--rvq_codebook_storage_dtype",
            str(rvq_l64.get("codebook_storage_dtype", "int4")),
            "--rvq_scale_storage_dtype",
            str(rvq_l64.get("scale_storage_dtype", "uint4")),
            "--kv_logical_head_dim",
            str(int(benchmark["kv_logical_head_dim"])),
            "--kv_reference_frames",
            str(int(benchmark["kv_reference_frames"])),
            *perchunk,
        )
        if keep_first_recent:
            args = (*args, *first_recent_quant)
        return args

    def age_ramp16_args() -> tuple[str, ...]:
        if not isinstance(rvq_age_ramp16, dict):
            raise ValueError("rvq_age_ramp16 config is required for the age-ramp setting")
        k_dir = str(repo_path(rvq_age_ramp16["k_codebook_dir"]))
        v_dir = str(repo_path(rvq_age_ramp16["v_codebook_dir"]))
        return (
            *rvq_args_for_dirs(
                k_dir=k_dir,
                v_dir=v_dir,
                codebook_dtype=str(rvq_age_ramp16.get("codebook_storage_dtype", "int4")),
                scale_dtype=str(rvq_age_ramp16.get("scale_storage_dtype", "uint4")),
                keep_first_recent=False,
                num_levels=16,
            ),
            "--kv_quant_tiered_depth_policy",
            "--kv_quant_tiered_depth_policy_name",
            "age_ramp16_middle4_recent6_8_12_16",
        )

    def age_ramp16_uniform_l8_args() -> tuple[str, ...]:
        if not isinstance(rvq_age_ramp16, dict):
            raise ValueError("rvq_age_ramp16 config is required for the L16-prefix control")
        return rvq_args_for_dirs(
            k_dir=str(repo_path(rvq_age_ramp16["k_codebook_dir"])),
            v_dir=str(repo_path(rvq_age_ramp16["v_codebook_dir"])),
            codebook_dtype=str(rvq_age_ramp16.get("codebook_storage_dtype", "int4")),
            scale_dtype=str(rvq_age_ramp16.get("scale_storage_dtype", "uint4")),
            keep_first_recent=False,
            num_levels=8,
        )

    specs = [
        SettingSpec(
            name="full_precision_cpu_offload",
            label="Full precision BF16/FP16 CPU offload reference",
            retention_policy="all_chunks_full_precision_cpu_offload",
            compression_method="none",
            extra_args=("--setting", "bf16", *common_reference),
            env=cpu_offload_env,
            is_reference=True,
        ),
        SettingSpec(
            name="first_recent_bf16_only",
            label="First and recent chunk BF16 only; middle chunks evicted",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_evicted",
            compression_method="evict_middle_chunks",
            extra_args=("--setting", "bf16", *common_reference, *first_recent_sparse),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_middle_int4",
            label="First and recent chunk BF16; middle chunks native INT4 per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="native_int4_perchunk",
            extra_args=native_int4_args,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="all_int4",
            label="All chunks native INT4 per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="native_int4_all_chunks_perchunk",
            extra_args=all_int4_args,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="all_int2",
            label="All chunks native INT2 per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="native_int2_all_chunks_perchunk",
            extra_args=all_int2_args,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="rvq_l8c64_int4",
            label="All chunks RVQ L8C64 INT4 per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="rvq_l8c64_int4_perchunk",
            extra_args=rvq_args_base,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="rvq_l16c64_int4",
            label="All chunks RVQ L16C64 INT4 per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="rvq_l16c64_int4_perchunk",
            extra_args=rvq_l64_args(16, keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="rvq_l32c64_int4",
            label="All chunks RVQ L32C64 INT4 per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="rvq_l32c64_int4_perchunk",
            extra_args=rvq_l64_args(32, keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="rvq_prog_l8_l32c64_int4",
            label="All chunks progressive RVQ L8-L32C64 INT4 per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="rvq_progressive_l8_l32c64_int4_perchunk",
            extra_args=progressive_args(keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_middle_fp8",
            label="First and recent chunk BF16; middle chunks FP8 E4M3 per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="fp8_e4m3_perchunk",
            extra_args=fp8_args,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_fp8_only",
            label="First and recent chunk FP8 E4M3; middle chunks evicted",
            retention_policy="first_chunk_and_recent_chunk_compressed_middle_evicted",
            compression_method="fp8_e4m3_retained_first_recent",
            extra_args=first_recent_fp8_sparse_args,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_middle_rvq",
            label="First and recent chunk BF16; middle chunks RVQ L8C64 INT4 per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="rvq_l8c64_int4_perchunk",
            extra_args=rvq_args,
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_middle_rvq_l16c64_int4",
            label="First and recent chunk BF16; middle chunks RVQ L16C64 INT4 per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="rvq_l16c64_int4_perchunk",
            extra_args=rvq_l64_args(16),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_middle_rvq_l32c64_int4",
            label="First and recent chunk BF16; middle chunks RVQ L32C64 INT4 per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="rvq_l32c64_int4_perchunk",
            extra_args=rvq_l64_args(32),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_middle_rvq_prog_l8_l32c64_int4",
            label="First and recent chunk BF16; middle chunks progressive RVQ L8-L32C64 INT4 per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="rvq_progressive_l8_l32c64_int4_perchunk",
            extra_args=progressive_args(keep_first_recent=True),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="nonorm_l8c64_int4",
            label="All chunks RVQ-L8C64 INT4 no-norm/no-scale per-chunk",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_l8c64_int4",
            extra_args=(*rvq_args_base, *rvq_diagnostic_args),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_l8c64_int4",
            label="First and recent chunk BF16; middle chunks RVQ-L8C64 INT4 no-norm/no-scale per-chunk",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_l8c64_int4",
            extra_args=(*rvq_args, *rvq_diagnostic_args),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="nonorm_l8c64_online_kv_block_scale",
            label="All chunks RVQ-L8C64 INT4 no-norm with online KV block calibration scale",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_l8c64_online_kv_block_scale",
            extra_args=calibration_args("online", keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_l8c64_online_kv_block_scale",
            label="First and recent chunk BF16; middle chunks RVQ-L8C64 INT4 no-norm with online KV block calibration scale",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_l8c64_online_kv_block_scale",
            extra_args=calibration_args("online", keep_first_recent=True),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="nonorm_l8c64_offline_kv_block_scale",
            label="All chunks RVQ-L8C64 INT4 no-norm with offline KV block calibration scale",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_l8c64_offline_kv_block_scale",
            extra_args=calibration_args("offline", keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_l8c64_offline_kv_block_scale",
            label="First and recent chunk BF16; middle chunks RVQ-L8C64 INT4 no-norm with offline KV block calibration scale",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_l8c64_offline_kv_block_scale",
            extra_args=calibration_args("offline", keep_first_recent=True),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="nonorm_k_token_v_channel_l8c64",
            label="All chunks K token-level + V grouped/channel RVQ-L8C64 INT4 no-norm",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_k_token_v_channel_l8c64",
            extra_args=mixed_k_token_v_channel_args(keep_first_recent=False),
            env=matched_value_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_k_token_v_channel_l8c64",
            label="First and recent chunk BF16; middle chunks K token-level + V grouped/channel RVQ-L8C64 INT4 no-norm",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_k_token_v_channel_l8c64",
            extra_args=mixed_k_token_v_channel_args(keep_first_recent=True),
            env=matched_value_env,
        ),
        SettingSpec(
            name="nonorm_k_token_l8_v_token_l32c64",
            label="All chunks K token L8 + V token L32C64 INT4 no-norm (matched Value rate)",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_k_token_l8_v_token_l32c64",
            extra_args=mixed_k_token_v_token_l32_args(keep_first_recent=False),
            env=matched_value_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_k_token_l8_v_token_l32c64",
            label="First and recent chunk BF16; middle chunks K token L8 + V token L32C64 INT4 no-norm",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_k_token_l8_v_token_l32c64",
            extra_args=mixed_k_token_v_token_l32_args(keep_first_recent=True),
            env=matched_value_env,
        ),
        SettingSpec(
            name="nonorm_progressive_token_l4_l16_avg7",
            label="All chunks progressive token RVQ no-norm L4-L16C64 INT4 target avg<=7",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_progressive_token_l4_l16_avg7",
            extra_args=progressive_v2_args("token", keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_progressive_token_l4_l16_avg7",
            label="First and recent chunk BF16; middle chunks progressive token RVQ no-norm L4-L16C64 INT4 target avg<=7",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_progressive_token_l4_l16_avg7",
            extra_args=progressive_v2_args("token", keep_first_recent=True),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="nonorm_progressive_layer_l4_l16_avg7",
            label="All chunks progressive layer RVQ no-norm L4-L16C64 INT4 target avg<=7",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_progressive_layer_l4_l16_avg7",
            extra_args=progressive_v2_args("layer", keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_progressive_layer_l4_l16_avg7",
            label="First and recent chunk BF16; middle chunks progressive layer RVQ no-norm L4-L16C64 INT4 target avg<=7",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_progressive_layer_l4_l16_avg7",
            extra_args=progressive_v2_args("layer", keep_first_recent=True),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="nonorm_progressive_block_l4_l16_avg7",
            label="All chunks progressive block RVQ no-norm L4-L16C64 INT4 target avg<=7",
            retention_policy="all_chunks_compressed",
            compression_method="nonorm_progressive_block_l4_l16_avg7",
            extra_args=progressive_v2_args("block", keep_first_recent=False),
            env=cpu_offload_env,
        ),
        SettingSpec(
            name="first_recent_nonorm_progressive_block_l4_l16_avg7",
            label="First and recent chunk BF16; middle chunks progressive block RVQ no-norm L4-L16C64 INT4 target avg<=7",
            retention_policy="first_chunk_and_recent_chunk_full_precision_middle_compressed",
            compression_method="nonorm_progressive_block_l4_l16_avg7",
            extra_args=progressive_v2_args("block", keep_first_recent=True),
            env=cpu_offload_env,
        ),
    ]
    if isinstance(rvq_age_ramp16, dict):
        specs.extend(
            [
            SettingSpec(
                name="nonorm_l16prefix_uniform_l8_int4",
                label="All chunks uniform L8 using the no-norm L16C64 INT4 codebook prefix",
                retention_policy="all_chunks_compressed_uniform_depth8",
                compression_method="nonorm_l16prefix_uniform_l8_int4",
                extra_args=age_ramp16_uniform_l8_args(),
                env=cpu_offload_env,
            ),
            SettingSpec(
                name="nonorm_age_ramp16_middle4_recent6_8_12_16",
                label=(
                    "All chunks no-norm L16C64 INT4 with first L16, middle L4, "
                    "and recent L6/L8/L12/L16"
                ),
                retention_policy="all_chunks_compressed_cache_age_ramp",
                compression_method="nonorm_age_ramp16_middle4_recent6_8_12_16",
                extra_args=age_ramp16_args(),
                env=cpu_offload_env,
            ),
            ]
        )
    return specs


def select_settings(all_settings: list[SettingSpec], raw: str) -> list[SettingSpec]:
    if raw.strip().lower() == "all":
        return all_settings
    wanted = [item.strip() for item in raw.split(",") if item.strip()]
    by_name = {spec.name: spec for spec in all_settings}
    missing = sorted(set(wanted) - set(by_name))
    if missing:
        raise ValueError(f"Unknown setting(s): {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def chunk_decision(spec: SettingSpec, chunk_idx: int, final_chunk_idx: int) -> str:
    no_scale_all = {
        "nonorm_l8c64_int4",
        "nonorm_l8c64_online_kv_block_scale",
        "nonorm_l8c64_offline_kv_block_scale",
        "nonorm_k_token_v_channel_l8c64",
        "nonorm_k_token_l8_v_token_l32c64",
        "nonorm_progressive_token_l4_l16_avg7",
        "nonorm_progressive_layer_l4_l16_avg7",
        "nonorm_progressive_block_l4_l16_avg7",
    }
    no_scale_first_recent = {
        "first_recent_nonorm_l8c64_int4",
        "first_recent_nonorm_l8c64_online_kv_block_scale",
        "first_recent_nonorm_l8c64_offline_kv_block_scale",
        "first_recent_nonorm_k_token_v_channel_l8c64",
        "first_recent_nonorm_k_token_l8_v_token_l32c64",
        "first_recent_nonorm_progressive_token_l4_l16_avg7",
        "first_recent_nonorm_progressive_layer_l4_l16_avg7",
        "first_recent_nonorm_progressive_block_l4_l16_avg7",
    }
    if spec.name == "full_precision_cpu_offload":
        return "full_precision_cpu_offload"
    if spec.name in no_scale_all:
        return f"{spec.compression_method}_all_chunks"
    if spec.name == "all_int4":
        return "native_int4_perchunk_all_chunks"
    if spec.name == "all_int2":
        return "native_int2_perchunk_all_chunks"
    if spec.name == "rvq_l8c64_int4":
        return "rvq_l8c64_int4_perchunk_all_chunks"
    if spec.name == "rvq_l16c64_int4":
        return "rvq_l16c64_int4_perchunk_all_chunks"
    if spec.name == "rvq_l32c64_int4":
        return "rvq_l32c64_int4_perchunk_all_chunks"
    if spec.name == "rvq_prog_l8_l32c64_int4":
        return "rvq_progressive_l8_l32c64_int4_perchunk_all_chunks"
    if spec.name == "first_recent_fp8_only":
        if chunk_idx == 0:
            return "fp8_e4m3_retained_first_chunk"
        if chunk_idx == final_chunk_idx:
            return "fp8_e4m3_retained_recent_current_chunk"
        return "evicted_middle_chunk"
    if chunk_idx == 0:
        return "full_precision_first_chunk"
    if chunk_idx == final_chunk_idx:
        return "full_precision_recent_current_chunk"
    if spec.name == "first_recent_bf16_only":
        return "evicted_middle_chunk"
    if spec.name in no_scale_first_recent:
        return f"{spec.compression_method}_middle_chunk"
    if spec.name == "first_recent_middle_int4":
        return "native_int4_perchunk_middle_chunk"
    if spec.name == "first_recent_middle_fp8":
        return "fp8_e4m3_perchunk_middle_chunk"
    if spec.name == "first_recent_middle_rvq":
        return "rvq_l8c64_int4_perchunk_middle_chunk"
    if spec.name == "first_recent_middle_rvq_l16c64_int4":
        return "rvq_l16c64_int4_perchunk_middle_chunk"
    if spec.name == "first_recent_middle_rvq_l32c64_int4":
        return "rvq_l32c64_int4_perchunk_middle_chunk"
    if spec.name == "first_recent_middle_rvq_prog_l8_l32c64_int4":
        return "rvq_progressive_l8_l32c64_int4_perchunk_middle_chunk"
    raise ValueError(spec.name)


def print_dry_run(settings: list[SettingSpec], config: dict[str, Any]) -> None:
    num_chunks = int(config["benchmark"]["num_chunks"])
    print(f"[dry-run] num_chunks={num_chunks}")
    for spec in settings:
        print(f"\n[{spec.name}] {spec.label}")
        print(f"retention_policy={spec.retention_policy}")
        print(f"compression_method={spec.compression_method}")
        for idx in range(num_chunks):
            print(f"  chunk {idx:02d}: {chunk_decision(spec, idx, num_chunks - 1)}")
        if spec.compression_method.endswith("_perchunk"):
            print("  quantization_schedule: per-chunk (--kv_quant_factor 1)")


def benchmark_base_cmd(
    config: dict[str, Any],
    spec: SettingSpec,
    run_dir: Path,
    worker_python: Path,
) -> list[str]:
    benchmark = config["benchmark"]
    setting_dir = run_dir / spec.name
    prompt_file = repo_path(benchmark["prompt_file"])
    if prompt_file is None:
        raise ValueError("benchmark.prompt_file is required")

    cmd = [
        str(worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "run_small_eval.py"),
        "--prompt-file",
        str(prompt_file),
        "--prompt-start",
        str(int(benchmark["prompt_start"])),
        "--num-prompts",
        str(int(benchmark["num_prompts"])),
        "--output-dir",
        str(setting_dir),
        "--seed",
        str(int(benchmark["seed"])),
        "--num_frames",
        str(int(benchmark["num_frames"])),
        "--num_chunk",
        str(int(benchmark["num_chunks"])),
        "--num_inference_steps",
        str(int(benchmark["num_inference_steps"])),
        "--image_path",
        str(benchmark.get("image_path", "none")),
        "--pose",
        str(benchmark.get("pose", "w-96")),
        "--memory_frames",
        str(int(benchmark["memory_frames"])),
        "--temporal_context_size",
        str(int(benchmark["temporal_context_size"])),
        "--pred_latent_size",
        str(int(benchmark["pred_latent_size"])),
        "--metrics-name",
        "metrics.csv",
        "--config-name",
        "config.json",
    ]
    if as_bool(benchmark.get("offload_text_encoder"), True):
        cmd.append("--offload_text_encoder")
    if as_bool(benchmark.get("skip_decode"), False):
        cmd.append("--skip_decode")
    cmd.extend(spec.extra_args)
    return cmd


def quality_cmd(config: dict[str, Any], spec: SettingSpec, run_dir: Path, worker_python: Path) -> list[str]:
    configured_reference = config.get("quality_reference_dir")
    reference_dir = (
        repo_path(configured_reference)
        if configured_reference is not None
        else run_dir / "full_precision_cpu_offload"
    )
    return [
        str(worker_python),
        str(REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"),
        "--reference-dir",
        str(reference_dir),
        "--test-dir",
        str(run_dir / spec.name),
        "--metrics-csv",
        str(run_dir / spec.name / "metrics.csv"),
        "--setting",
        spec.name,
        "--output-json",
        str(run_dir / spec.name / "quality.json"),
        "--output-csv",
        str(run_dir / spec.name / "quality.csv"),
        "--device",
        "cuda",
    ]


def vbench_prompt_json(setting_dir: Path) -> Path:
    metrics_csv = setting_dir / "metrics.csv"
    out = setting_dir / "vbench_prompts.json"
    prompt_map: dict[str, str] = {}
    if metrics_csv.exists():
        with metrics_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                video_path = Path(row["video_path"])
                if not video_path.is_absolute():
                    video_path = (REPO_ROOT / video_path).resolve()
                prompt_map[str(video_path)] = row.get("prompt", "")
    out.write_text(json.dumps(prompt_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def vbench_cmds(config: dict[str, Any], spec: SettingSpec, run_dir: Path, vbench_bin: Path, ngpus: int) -> list[list[str]]:
    dims = config.get("vbench", {}).get("dimensions", DEFAULT_VBENCH_DIMS)
    setting_dir = run_dir / spec.name
    prompt_file = setting_dir / "vbench_prompts.json"
    return [
        [
            str(vbench_bin),
            "evaluate",
            "--ngpus",
            str(int(ngpus)),
            "--output_path",
            str(setting_dir / "vbench" / dim),
            "--videos_path",
            str(setting_dir),
            "--dimension",
            str(dim),
            "--mode",
            "custom_input",
            "--prompt_file",
            str(prompt_file),
        ]
        for dim in dims
    ]


def command_plan(
    config: dict[str, Any],
    settings: list[SettingSpec],
    run_dir: Path,
    *,
    run_vbench: bool,
    skip_quality: bool,
    vbench_ngpus: int,
) -> list[tuple[str, SettingSpec | None, list[str]]]:
    runner = config.get("runner", {})
    worker_python = repo_path(runner.get("python", DEFAULT_WORKER_PYTHON)) or DEFAULT_WORKER_PYTHON
    vbench_bin = repo_path(config.get("vbench", {}).get("bin", runner.get("vbench_bin", DEFAULT_VBENCH_BIN))) or DEFAULT_VBENCH_BIN
    plan: list[tuple[str, SettingSpec | None, list[str]]] = []
    for spec in settings:
        plan.append(("generate", spec, benchmark_base_cmd(config, spec, run_dir, worker_python)))
    configured_reference = config.get("quality_reference_dir")
    reference_available = (
        any(spec.is_reference for spec in settings)
        or (run_dir / "full_precision_cpu_offload" / "metrics.csv").exists()
        or (
            configured_reference is not None
            and (repo_path(configured_reference) / "metrics.csv").exists()
        )
    )
    if not skip_quality and reference_available:
        for spec in settings:
            if not spec.is_reference:
                plan.append(("quality", spec, quality_cmd(config, spec, run_dir, worker_python)))
    if run_vbench:
        for spec in settings:
            for cmd in vbench_cmds(config, spec, run_dir, vbench_bin, vbench_ngpus):
                plan.append(("vbench", spec, cmd))
    return plan


def print_commands(plan: list[tuple[str, SettingSpec | None, list[str]]]) -> None:
    for phase, spec, cmd in plan:
        prefix = f"[{phase}]"
        if spec is not None:
            prefix += f"[{spec.name}]"
        print(prefix, shlex.join(cmd))


def write_commands(run_dir: Path, plan: list[tuple[str, SettingSpec | None, list[str]]], resume: bool) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "commands.sh"
    if out.exists() and not resume:
        raise FileExistsError(f"{out} exists; use --resume or a new --run-id")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for phase, spec, cmd in plan:
        label = f"{phase}:{spec.name}" if spec else phase
        lines.append(f"# {label}")
        lines.append(shlex.join(cmd))
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    out.chmod(0o755)
    return out


def command_env(spec: SettingSpec | None, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if spec is not None:
        env.update(spec.env)
    if args.gpus:
        env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[cmd] " + shlex.join(cmd) + "\n")
        log.flush()
        rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}; log={log_path}")
    return elapsed


def read_metric_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any) -> float | None:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def parse_log_timing(log_path: Path) -> tuple[list[dict[str, float | int | None]], float | None]:
    if not log_path.exists():
        return [], None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    by_chunk: dict[int, dict[str, float | int | None]] = {}
    for chunk, seconds in re.findall(r"Generate time for chunk\s+(\d+)\s+is\s+([0-9.eE+-]+)", text):
        idx = int(chunk)
        by_chunk.setdefault(idx, {"chunk": idx})
        by_chunk[idx]["generation_time_sec"] = float(seconds)
    for chunk, seconds, calls in re.findall(
        r"KV Cache Dequantization Time for chunk\s+(\d+):\s+([0-9.eE+-]+)\s+s over\s+(\d+) calls",
        text,
    ):
        idx = int(chunk)
        by_chunk.setdefault(idx, {"chunk": idx})
        by_chunk[idx]["decompression_time_sec"] = float(seconds)
        by_chunk[idx]["decompression_calls"] = int(calls)
    rows = [by_chunk[idx] for idx in sorted(by_chunk)]
    decompression_values = [
        float(row["decompression_time_sec"])
        for row in rows
        if row.get("decompression_time_sec") is not None
    ]
    return rows, sum(decompression_values) if decompression_values else None


def parse_vbench_scores(setting_dir: Path, dimensions: list[str]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {dim: None for dim in dimensions}
    root = setting_dir / "vbench"
    if not root.exists():
        return scores
    for dim in dimensions:
        for path in sorted((root / dim).rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            stack: list[Any] = [payload]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    value = item.get(dim)
                    if isinstance(value, (int, float)):
                        scores[dim] = float(value)
                    elif isinstance(value, list) and value and isinstance(value[0], (int, float)):
                        scores[dim] = float(value[0])
                    elif isinstance(value, dict):
                        for key in ("score", "mean", "value"):
                            if isinstance(value.get(key), (int, float)):
                                scores[dim] = float(value[key])
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
    return scores


def prompt_values(config: dict[str, Any]) -> list[str]:
    benchmark = config["benchmark"]
    prompt_file = repo_path(benchmark["prompt_file"])
    if prompt_file is None or not prompt_file.exists():
        return []
    lines = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = int(benchmark["prompt_start"])
    end = start + int(benchmark["num_prompts"])
    return lines[start:end]


def codebook_metadata(spec: SettingSpec) -> tuple[list[str], dict[str, str | None]]:
    paths: list[str] = []
    args = list(spec.extra_args)
    for flag in ("--k_rvq_codebook_dir", "--v_rvq_codebook_dir", "--rvq_calibration_path"):
        if flag in args:
            paths.append(args[args.index(flag) + 1])
    hashes = {path: sha256_path(Path(path)) for path in paths}
    return paths, hashes


def matched_value_metadata(spec: SettingSpec) -> dict[str, Any] | None:
    matched = {
        "nonorm_k_token_v_channel_l8c64": "channel_4x32d_l8c64",
        "first_recent_nonorm_k_token_v_channel_l8c64": "channel_4x32d_l8c64",
        "nonorm_k_token_l8_v_token_l32c64": "token_128d_l32c64",
        "first_recent_nonorm_k_token_l8_v_token_l32c64": "token_128d_l32c64",
    }
    granularity = matched.get(spec.name)
    if granularity is None:
        return None
    args = list(spec.extra_args)
    dtype = args[args.index("--rvq_codebook_storage_dtype") + 1]
    k_dir = args[args.index("--k_rvq_codebook_dir") + 1]
    v_dir = args[args.index("--v_rvq_codebook_dir") + 1]
    k_metadata = rvq_codebook_directory_metadata(k_dir, storage_dtype=dtype)
    v_metadata = rvq_codebook_directory_metadata(v_dir, storage_dtype=dtype)
    expected_format = "grouped_dim" if granularity.startswith("channel") else "standard"
    checks = {
        "rvq_format": expected_format,
        "codebook_size": 64,
        "physical_index_bytes_per_vector": 32,
        "logical_index_bits_per_vector": 192,
        "input_norm_mode": "none",
        "per_vector_scale_count": 0,
    }
    for key, expected in checks.items():
        if v_metadata.get(key) != expected:
            raise ValueError(
                f"Matched Value codebook invariant failed for {spec.name}: "
                f"{key}={v_metadata.get(key)!r}, expected {expected!r}"
            )
    if expected_format == "grouped_dim" and v_metadata.get("group_transform") != "none":
        raise ValueError(f"Matched Value channel arm must not use Hadamard: {spec.name}")
    codebook_bytes = int(k_metadata["codebook_storage_bytes"]) + int(
        v_metadata["codebook_storage_bytes"]
    )
    return {
        "value_granularity": granularity,
        "value_index_bytes_per_vector": 32,
        "value_logical_bits_per_vector": 192,
        "value_scale_bytes_per_vector": 0,
        "codebook_storage_dtype": dtype,
        "shared_codebook_bytes": codebook_bytes,
        "shared_codebook_mib": codebook_bytes / 1024**2,
        "codebooks": {"K": k_metadata, "V": v_metadata},
    }


def write_run_info(
    config_path: Path,
    config_hash: str,
    config: dict[str, Any],
    spec: SettingSpec,
    setting_dir: Path,
    command: list[str],
    smoke: bool,
) -> None:
    benchmark = config["benchmark"]
    codebook_paths, codebook_hashes = codebook_metadata(spec)
    existing_path = setting_dir / "run_info.json"
    if existing_path.exists():
        try:
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing.get("codebook_paths") == codebook_paths and isinstance(existing.get("codebook_hashes"), dict):
                codebook_hashes = existing["codebook_hashes"]
        except Exception:
            pass
    prompts = prompt_values(config)
    payload = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": codebook_paths[0] if len(codebook_paths) == 1 else codebook_paths or None,
        "codebook_hash": codebook_hashes.get(codebook_paths[0]) if len(codebook_paths) == 1 else codebook_hashes or None,
        "codebook_paths": codebook_paths,
        "codebook_hashes": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "seed": int(benchmark["seed"]),
        "prompt": prompts[0] if len(prompts) == 1 else None,
        "prompt_list": prompts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_chunks": int(benchmark["num_chunks"]),
        "retention_policy": spec.retention_policy,
        "compression_method": spec.compression_method,
        "setting_name": spec.name,
        "setting_label": spec.label,
        "smoke": smoke,
        "skip_decode": as_bool(benchmark.get("skip_decode"), False),
        "command": shlex.join(command),
        "output_dir": str(setting_dir),
        "result_source": str(config.get("result_source", "new")),
    }
    value_metadata = matched_value_metadata(spec)
    if value_metadata is not None:
        payload["matched_value_budget"] = value_metadata
    age_ramp = config.get("rvq_age_ramp16", {})
    age_ramp_settings = {
        "nonorm_l16prefix_uniform_l8_int4",
        "nonorm_age_ramp16_middle4_recent6_8_12_16",
    }
    if isinstance(age_ramp, dict) and spec.name in age_ramp_settings:
        audit_paths = sorted(setting_dir.glob("*-tiered_depth_audit.json"))
        terminal_storage: dict[str, Any] | None = None
        if audit_paths:
            observed: list[dict[str, Any]] = []
            for audit_path in audit_paths:
                audit = json.loads(audit_path.read_text(encoding="utf-8"))
                events = audit.get("pipeline_audit", {}).get("events", [])
                if events:
                    observed.append(events[-1].get("storage", {}))
            if observed:
                terminal_storage = observed[0]
                if any(item != terminal_storage for item in observed[1:]):
                    raise ValueError("age-ramp audit terminal storage differs across prompts")
        is_tiered = spec.name == "nonorm_age_ramp16_middle4_recent6_8_12_16"
        payload.update(
            {
                "input_norm_mode": str(age_ramp.get("input_norm_mode", "none")),
                "runtime_codebook_storage": str(
                    age_ramp.get("codebook_storage_dtype", "int4")
                ),
                "per_token_rms_scale_stored": bool(
                    age_ramp.get("per_token_rms_scale_stored", False)
                ),
                "tiered_depth_policy": str(age_ramp.get("policy", "")) if is_tiered else None,
                "terminal_depth_schedule": (
                    age_ramp.get("terminal_schedule") if is_tiered else [8] * 10
                ),
                "terminal_average_depth": (
                    age_ramp.get("terminal_average_depth") if is_tiered else 8.0
                ),
                "actual_storage_accounting": terminal_storage,
                "audit_paths": [str(path) for path in audit_paths],
            }
        )
    (setting_dir / "run_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_setting_outputs(
    config: dict[str, Any],
    spec: SettingSpec,
    setting_dir: Path,
    *,
    config_path: Path,
    config_hash: str,
    command: list[str],
    smoke: bool,
) -> dict[str, Any]:
    rows = read_metric_rows(setting_dir / "metrics.csv")
    dimensions = list(config.get("vbench", {}).get("dimensions", DEFAULT_VBENCH_DIMS))

    def vals(key: str) -> list[float]:
        return [v for v in (to_float(row.get(key)) for row in rows) if v is not None]

    kv_stored_mib = mean(vals("kv_cache_mem_mb"))
    kv_ratio = mean(vals("compression_ratio"))
    k_payload_mib = mean(vals("k_payload_mib"))
    v_payload_mib = mean(vals("v_payload_mib"))
    kv_payload_mib = mean(vals("kv_payload_mib"))
    shared_codebook_mib = mean(vals("shared_codebook_mib"))
    payload_ratio = mean(vals("compression_ratio_excluding_shared_codebook"))
    deployment_ratio = mean(vals("compression_ratio_including_shared_codebook"))
    kv_stored_bytes = int(round(kv_stored_mib * 1024 * 1024)) if kv_stored_mib is not None else None
    kv_original_bytes = (
        int(round(kv_stored_bytes * kv_ratio))
        if kv_stored_bytes is not None and kv_ratio is not None
        else None
    )
    generation_time_sec = mean(vals("denoise_time_s"))
    compression_time_sec = mean(vals("quant_time_s"))
    total_time_sec = mean(vals("e2e_time_s"))
    rvq_actual_avg_depth = mean(vals("rvq_actual_avg_depth"))
    rvq_k_attention_kl = mean(vals("rvq_k_attention_kl"))
    rvq_v_output_mse = mean(vals("rvq_v_output_mse"))
    rvq_v_raw_mse = mean(vals("rvq_v_raw_mse"))
    rvq_v_relative_l2 = mean(vals("rvq_v_relative_l2"))
    rvq_v_mean_vector_relative_squared_error = mean(
        vals("rvq_v_mean_vector_relative_squared_error")
    )
    peak_gpu_memory_mib = max_or_none(vals("peak_gpu_mem_mb"))
    video_paths = [row.get("video_path") for row in rows if row.get("video_path")]
    quality_path = setting_dir / "quality.json"
    quality: dict[str, Any] = {}
    if quality_path.exists():
        quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
        quality = {
            "psnr": quality_payload.get("psnr_mean"),
            "ssim": quality_payload.get("ssim_mean"),
            "lpips": quality_payload.get("lpips_mean"),
            "quality_reference": quality_payload.get("reference"),
        }
    else:
        quality = {"psnr": None, "ssim": None, "lpips": None, "quality_reference": None}
    vbench_scores = parse_vbench_scores(setting_dir, dimensions)

    per_chunk_timing, decompression_time_sec = parse_log_timing(setting_dir / "run.log")
    timing = {
        "generation_time_sec": generation_time_sec,
        "compression_time_sec": compression_time_sec,
        "decompression_time_sec": decompression_time_sec,
        "total_time_sec": total_time_sec,
        "per_chunk": per_chunk_timing,
        "notes": {
            "generation_time_sec": "Mean denoise_time_s from run_small_eval metrics.csv.",
            "compression_time_sec": "Mean quant_time_s from run_small_eval metrics.csv.",
            "decompression_time_sec": "Sum of per-chunk KV Cache Dequantization Time lines from run.log.",
            "total_time_sec": "Mean e2e_time_s from run_small_eval metrics.csv.",
        },
    }
    memory = {
        "kv_original_bytes": kv_original_bytes,
        "kv_stored_bytes": kv_stored_bytes,
        "kv_compression_ratio": kv_ratio,
        "k_payload_mib": k_payload_mib,
        "v_payload_mib": v_payload_mib,
        "kv_payload_mib": kv_payload_mib,
        "shared_codebook_mib": shared_codebook_mib,
        "compression_ratio_excluding_shared_codebook": payload_ratio,
        "compression_ratio_including_shared_codebook": deployment_ratio,
        "peak_gpu_memory_mib": peak_gpu_memory_mib,
        "baseline": f"Full BF16 KV cache over {config['benchmark']['kv_reference_frames']} latent frames.",
    }
    metrics = {
        "setting_name": spec.name,
        "setting_label": spec.label,
        "generated_video_path": video_paths[0] if len(video_paths) == 1 else None,
        "generated_video_paths": video_paths,
        **quality,
        "rvq_actual_avg_depth": rvq_actual_avg_depth,
        "rvq_k_attention_kl": rvq_k_attention_kl,
        "rvq_v_output_mse": rvq_v_output_mse,
        "rvq_v_raw_mse": rvq_v_raw_mse,
        "rvq_v_relative_l2": rvq_v_relative_l2,
        "rvq_v_mean_vector_relative_squared_error": rvq_v_mean_vector_relative_squared_error,
        "vbench": vbench_scores,
    }

    (setting_dir / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")
    (setting_dir / "memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")
    (setting_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_run_info(config_path, config_hash, config, spec, setting_dir, command, smoke)

    row = {
        "setting_name": spec.name,
        "video_path": metrics["generated_video_path"],
        **{k: timing[k] for k in ("generation_time_sec", "compression_time_sec", "decompression_time_sec", "total_time_sec")},
        **memory,
        "psnr": quality["psnr"],
        "ssim": quality["ssim"],
        "lpips": quality["lpips"],
        "rvq_actual_avg_depth": rvq_actual_avg_depth,
        "rvq_k_attention_kl": rvq_k_attention_kl,
        "rvq_v_output_mse": rvq_v_output_mse,
        "rvq_v_raw_mse": rvq_v_raw_mse,
        "rvq_v_relative_l2": rvq_v_relative_l2,
        "rvq_v_mean_vector_relative_squared_error": rvq_v_mean_vector_relative_squared_error,
    }
    for dim, value in vbench_scores.items():
        row[f"vbench_{dim}"] = value
    return row


def write_vbench_placeholder(config: dict[str, Any], spec: SettingSpec, setting_dir: Path, vbench_bin: Path, ngpus: int) -> None:
    dimensions = config.get("vbench", {}).get("dimensions", DEFAULT_VBENCH_DIMS)
    placeholder = {
        "status": "vbench_unavailable",
        "reason": f"VBench binary not found: {vbench_bin}",
        "prepare_prompt_file": str(setting_dir / "vbench_prompts.json"),
        "commands": [
            shlex.join(cmd)
            for cmd in vbench_cmds(config, spec, setting_dir.parent, vbench_bin, ngpus)
        ],
        "dimensions": dimensions,
    }
    out = setting_dir / "vbench_placeholder.json"
    out.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
    print(f"[vbench-placeholder] {spec.name}: {out}", flush=True)


def aggregate_run(config: dict[str, Any], settings: list[SettingSpec], run_dir: Path) -> Path:
    rows = []
    dimensions = list(config.get("vbench", {}).get("dimensions", DEFAULT_VBENCH_DIMS))
    for spec in settings:
        setting_dir = run_dir / spec.name
        timing_path = setting_dir / "timing.json"
        memory_path = setting_dir / "memory.json"
        metrics_path = setting_dir / "metrics.json"
        if not (timing_path.exists() and memory_path.exists() and metrics_path.exists()):
            continue
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "setting_name": spec.name,
            "video_path": metrics.get("generated_video_path"),
            "generation_time_sec": timing.get("generation_time_sec"),
            "compression_time_sec": timing.get("compression_time_sec"),
            "decompression_time_sec": timing.get("decompression_time_sec"),
            "total_time_sec": timing.get("total_time_sec"),
            "kv_original_bytes": memory.get("kv_original_bytes"),
            "kv_stored_bytes": memory.get("kv_stored_bytes"),
            "kv_compression_ratio": memory.get("kv_compression_ratio"),
            "peak_gpu_memory_mib": memory.get("peak_gpu_memory_mib"),
            "psnr": metrics.get("psnr"),
            "ssim": metrics.get("ssim"),
            "lpips": metrics.get("lpips"),
            "rvq_actual_avg_depth": metrics.get("rvq_actual_avg_depth"),
            "rvq_k_attention_kl": metrics.get("rvq_k_attention_kl"),
            "rvq_v_output_mse": metrics.get("rvq_v_output_mse"),
        }
        vbench = metrics.get("vbench", {}) or {}
        for dim in dimensions:
            row[f"vbench_{dim}"] = vbench.get(dim)
        rows.append(row)

    out_csv = run_dir / "summary.csv"
    fields = [
        "setting_name",
        "video_path",
        "generation_time_sec",
        "compression_time_sec",
        "decompression_time_sec",
        "total_time_sec",
        "kv_original_bytes",
        "kv_stored_bytes",
        "kv_compression_ratio",
        "peak_gpu_memory_mib",
        "psnr",
        "ssim",
        "lpips",
        "rvq_actual_avg_depth",
        "rvq_k_attention_kl",
        "rvq_v_output_mse",
        "rvq_v_raw_mse",
        "rvq_v_relative_l2",
        "rvq_v_mean_vector_relative_squared_error",
        *[f"vbench_{dim}" for dim in dimensions],
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    (run_dir / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_csv


def execute_plan(
    args: argparse.Namespace,
    config: dict[str, Any],
    settings: list[SettingSpec],
    run_dir: Path,
    config_path: Path,
    config_hash: str,
) -> None:
    if not args.smoke and not args.allow_full_run:
        raise RuntimeError("--execute for the full 21-chunk benchmark requires --allow-full-run")
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory exists: {run_dir}; use a new --run-id or --resume")
    run_dir.mkdir(parents=True, exist_ok=True)

    runner = config.get("runner", {})
    worker_python = repo_path(runner.get("python", DEFAULT_WORKER_PYTHON)) or DEFAULT_WORKER_PYTHON
    vbench_bin = repo_path(config.get("vbench", {}).get("bin", runner.get("vbench_bin", DEFAULT_VBENCH_BIN))) or DEFAULT_VBENCH_BIN

    for spec in settings:
        setting_dir = run_dir / spec.name
        if (setting_dir / "metrics.json").exists() and args.resume:
            print(f"[skip-complete] {spec.name}", flush=True)
            continue
        if setting_dir.exists() and any(setting_dir.iterdir()) and not args.resume:
            raise FileExistsError(f"Setting directory exists: {setting_dir}")
        setting_dir.mkdir(parents=True, exist_ok=True)
        cmd = benchmark_base_cmd(config, spec, run_dir, worker_python)
        print(f"[run] generate {spec.name}", flush=True)
        run_command(cmd, setting_dir / "run.log", command_env(spec, args))
        normalize_setting_outputs(
            config,
            spec,
            setting_dir,
            config_path=config_path,
            config_hash=config_hash,
            command=cmd,
            smoke=args.smoke,
        )

    configured_reference = config.get("quality_reference_dir")
    reference_available = (
        any(spec.is_reference for spec in settings)
        or (run_dir / "full_precision_cpu_offload" / "metrics.csv").exists()
        or (
            configured_reference is not None
            and (repo_path(configured_reference) / "metrics.csv").exists()
        )
    )
    if not args.skip_quality and reference_available:
        for spec in settings:
            if spec.is_reference:
                continue
            setting_dir = run_dir / spec.name
            if (setting_dir / "quality.json").exists() and args.resume:
                print(f"[skip-quality] {spec.name}", flush=True)
            else:
                cmd = quality_cmd(config, spec, run_dir, worker_python)
                print(f"[run] quality {spec.name}", flush=True)
                run_command(cmd, setting_dir / "quality.log", command_env(spec, args))
            normalize_setting_outputs(
                config,
                spec,
                setting_dir,
                config_path=config_path,
                config_hash=config_hash,
                command=benchmark_base_cmd(config, spec, run_dir, worker_python),
                smoke=args.smoke,
            )

    if args.run_vbench:
        for spec in settings:
            setting_dir = run_dir / spec.name
            vbench_prompt_json(setting_dir)
            if not vbench_bin.exists():
                write_vbench_placeholder(config, spec, setting_dir, vbench_bin, args.vbench_ngpus)
                continue
            for cmd in vbench_cmds(config, spec, run_dir, vbench_bin, args.vbench_ngpus):
                dim = cmd[cmd.index("--dimension") + 1]
                sentinel = setting_dir / "vbench" / dim / "vbench.done"
                if sentinel.exists() and args.resume:
                    print(f"[skip-vbench] {spec.name} {dim}", flush=True)
                    continue
                print(f"[run] vbench {spec.name} {dim}", flush=True)
                run_command(cmd, setting_dir / "vbench" / f"{dim}.log", command_env(spec, args))
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text("ok\n", encoding="utf-8")
            normalize_setting_outputs(
                config,
                spec,
                setting_dir,
                config_path=config_path,
                config_hash=config_hash,
                command=benchmark_base_cmd(config, spec, run_dir, worker_python),
                smoke=args.smoke,
            )

    if args.no_aggregate:
        print("[summary] skipped by --no-aggregate", flush=True)
    else:
        out_csv = aggregate_run(config, settings, run_dir)
        print(f"[summary] {out_csv}", flush=True)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    raw_config = load_config(config_path)
    config = resolved_benchmark_config(raw_config, args.smoke)
    if args.prompt_start_override is not None:
        config["benchmark"]["prompt_start"] = int(args.prompt_start_override)
    if args.num_prompts_override is not None:
        if args.num_prompts_override <= 0:
            raise ValueError("--num-prompts-override must be positive")
        config["benchmark"]["num_prompts"] = int(args.num_prompts_override)
    if args.skip_decode:
        config.setdefault("benchmark", {})["skip_decode"] = True
    if args.prompt_start_override is None and args.num_prompts_override is None:
        config_hash = sha256_file(config_path)
    else:
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    settings = select_settings(setting_specs(config), args.settings)
    run_dir = args.output_root / run_id_from_args(args)

    if not any([args.dry_run, args.print_commands, args.write_commands, args.execute, args.aggregate]):
        args.dry_run = True

    if args.dry_run:
        print_dry_run(settings, config)

    if args.print_commands or args.write_commands:
        run_vbench = args.run_vbench and not args.skip_vbench
        plan = command_plan(
            config,
            settings,
            run_dir,
            run_vbench=run_vbench,
            skip_quality=args.skip_quality,
            vbench_ngpus=args.vbench_ngpus,
        )
        if args.print_commands:
            print_commands(plan)
        if args.write_commands:
            path = write_commands(run_dir, plan, args.resume)
            print(f"[commands] {path}", flush=True)

    if args.execute:
        execute_plan(args, config, settings, run_dir, config_path, config_hash)

    if args.aggregate and not args.execute:
        out_csv = aggregate_run(config, settings, run_dir)
        print(f"[summary] {out_csv}", flush=True)


if __name__ == "__main__":
    main()
