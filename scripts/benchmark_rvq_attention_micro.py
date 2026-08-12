#!/usr/bin/env python3
"""Reproducible RVQ-aware attention microbenchmark on saved Q/K/V tensors.

This runner intentionally does not import or execute a video pipeline.  File
loading, CPU setup, CUDA context initialization, codebook loading, and Triton
compilation/warmup happen outside the reported CUDA-event intervals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.rvq import (  # noqa: E402
    _unpack_scale,
    load_layer_codebooks_from_dir,
    materialize_rvq_codebooks,
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)
from quant_videogen.rvq_attention import (  # noqa: E402
    build_rvq_attention_table,
    rvq_compressed_attention_reference,
    rvq_histogram_attention_b1,
    rvq_histogram_attention_packed_scale_b1,
    rvq_histogram_contract,
    rvq_linear_decode_reference,
    rvq_online_histogram_b1,
    rvq_table_tile_v_attention_b1,
    rvq_table_tile_v_attention_packed_scale_b1,
    rvq_table_tile_v_attention_from_table_b1,
)

MIB = 1024.0 * 1024.0
DEFAULT_CONFIG = REPO_ROOT / "experiments/microbench/rvq_attention_l8c64.yaml"


@dataclass
class LoadedTrack:
    name: str
    source_path: Path
    layer: int
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    q_token_counts: list[int]
    kv_token_counts: list[int]
    metadata: dict[str, Any]
    representation_assumption: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", default="default", help="Grid under benchmark.profiles.")
    parser.add_argument("--track", action="append", default=[], help="Run only this track (repeatable).")
    parser.add_argument("--q-frames", default=None, help="Override THND Q frame counts, comma-separated.")
    parser.add_argument("--kv-tokens", default=None, help="Override K/V token counts, comma-separated.")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and source paths without loading tensors or creating results.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("PyYAML is required to read the microbenchmark config") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def parse_int_list(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return [int(item) for item in raw]


def dtype_from_name(name: str) -> torch.dtype:
    names = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return names[str(name).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_layer_codebooks(directory: Path, layer: int, heads: int) -> str:
    """Hash the ordered set of codebooks actually used, not unrelated layers."""
    digest = hashlib.sha256()
    for head in range(heads):
        path = directory / f"layer_{layer}_head_{head}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "shape": [int(x) for x in value.shape],
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("Cannot compute percentile of no samples")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_stats(samples_ms: list[float]) -> dict[str, float | int]:
    return {
        "median_ms": float(statistics.median(samples_ms)),
        "p10_ms": percentile(samples_ms, 0.10),
        "p90_ms": percentile(samples_ms, 0.90),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": float(statistics.fmean(samples_ms)),
        "samples": len(samples_ms),
    }


def call_supported(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Pass tuning kwargs supported by the installed kernel revision."""
    signature = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{key: value for key, value in kwargs.items() if key in signature.parameters})


def sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def cuda_samples(
    fn: Callable[[], Any], *, warmup: int, repeat: int, device: torch.device
) -> tuple[dict[str, float | int], list[float]]:
    for _ in range(warmup):
        out = fn()
        del out
    sync(device)
    samples: list[float] = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        del out
    return timing_stats(samples), samples


def cuda_peak(fn: Callable[[], Any], device: torch.device) -> dict[str, float]:
    sync(device)
    base_alloc = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    out = fn()
    sync(device)
    result = {
        "baseline_allocated_mib": base_alloc / MIB,
        "baseline_reserved_mib": base_reserved / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "incremental_peak_allocated_mib": max(0, torch.cuda.max_memory_allocated(device) - base_alloc) / MIB,
        "incremental_peak_reserved_mib": max(0, torch.cuda.max_memory_reserved(device) - base_reserved) / MIB,
    }
    del out
    sync(device)
    return result


def error_stats(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.float()
    reference = reference.float()
    candidate_nan = bool(torch.isnan(candidate).any().item())
    candidate_inf = bool(torch.isinf(candidate).any().item())
    reference_nan = bool(torch.isnan(reference).any().item())
    reference_inf = bool(torch.isinf(reference).any().item())
    if candidate.shape != reference.shape:
        raise ValueError(f"Correctness shape mismatch: {candidate.shape} vs {reference.shape}")
    diff = candidate - reference
    abs_diff = diff.abs()
    ref_norm = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(diff) / ref_norm
    mean_relative = (abs_diff / reference.abs().clamp_min(1e-6)).mean()
    cosine = F.cosine_similarity(candidate.reshape(1, -1), reference.reshape(1, -1), dim=-1)
    return {
        "max_abs_error": float(abs_diff.max().item()),
        "mean_abs_error": float(abs_diff.mean().item()),
        "mse": float((diff * diff).mean().item()),
        "relative_l2_error": float(relative_l2.item()),
        "mean_relative_error": float(mean_relative.item()),
        "cosine_similarity": float(cosine.item()),
        "candidate_has_nan": candidate_nan,
        "candidate_has_inf": candidate_inf,
        "reference_has_nan": reference_nan,
        "reference_has_inf": reference_inf,
        "all_finite": not (candidate_nan or candidate_inf or reference_nan or reference_inf),
    }


def normalize_attention_output(out: torch.Tensor) -> torch.Tensor:
    return out.unsqueeze(0) if out.ndim == 3 else out


def unique_run_dir(root: Path, tag: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S.%f%z")
    base = root / f"{stamp}_{tag}_pid{os.getpid()}"
    for suffix in range(1000):
        candidate = base if suffix == 0 else Path(f"{base}_{suffix:03d}")
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique results directory under {root}")


def validate_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Expected schema_version: 1")
    profiles = config.get("benchmark", {}).get("profiles", {})
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile {args.profile!r}; choices={sorted(profiles)}")
    profile = dict(profiles[args.profile])
    if args.q_frames is not None:
        profile["q_frame_counts"] = parse_int_list(args.q_frames)
    if args.kv_tokens is not None:
        profile["kv_tokens"] = parse_int_list(args.kv_tokens)
    selected = set(args.track or profile.get("tracks", []))
    tracks = [item for item in config.get("data", {}).get("tracks", []) if item.get("enabled") and item["name"] in selected]
    if not tracks:
        raise ValueError(f"No enabled data tracks selected: {sorted(selected)}")
    for track in tracks:
        if not Path(track["path"]).expanduser().exists():
            raise FileNotFoundError(track["path"])
    codebooks = config["codebooks"]
    for key in ("k_dir", "v_dir"):
        if not Path(codebooks[key]).expanduser().is_dir():
            raise FileNotFoundError(codebooks[key])
    warmup = args.warmup if args.warmup is not None else int(config["benchmark"]["warmup"])
    repeat = args.repeat if args.repeat is not None else int(config["benchmark"]["repeat"])
    if warmup < 0 or repeat <= 0:
        raise ValueError(f"warmup must be >=0 and repeat >0, got {warmup}, {repeat}")
    return {"profile": profile, "tracks": tracks, "warmup": warmup, "repeat": repeat}


def _thnd_to_bhsd(x: torch.Tensor, token_start: int, token_count: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected THND tensor, got {tuple(x.shape)}")
    # THND stores H between T and N.  Flattening T/N directly would interleave
    # head and token values, so move N next to T before forming the token axis.
    flat = x.permute(0, 2, 1, 3).reshape(
        x.shape[0] * x.shape[2], x.shape[1], x.shape[3]
    )
    if token_start < 0 or token_start + token_count > flat.shape[0]:
        raise ValueError(f"Requested [{token_start},{token_start + token_count}) from THND shape {tuple(x.shape)}")
    return flat[token_start : token_start + token_count].permute(1, 0, 2).unsqueeze(0).contiguous()


def load_track(spec: dict[str, Any], profile: dict[str, Any]) -> LoadedTrack:
    path = Path(spec["path"]).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = spec.get("tensor_keys", {"q": "Q", "k": "K", "v": "V"})
    q_source, k_source, v_source = (payload[keys[name]] for name in ("q", "k", "v"))
    metadata = dict(payload.get("metadata", {})) if isinstance(payload, dict) else {}
    fmt = str(spec["format"])
    profile_kv = parse_int_list(spec.get("kv_tokens", profile["kv_tokens"]))
    if fmt == "thnd":
        tokens_per_frame = int(spec.get("tokens_per_frame", q_source.shape[2]))
        q_frames = parse_int_list(profile.get("q_frame_counts", [1]))
        q_counts = [frames * tokens_per_frame for frames in q_frames]
        q_start_frame = int(spec.get("q_start_frame", 0))
        if q_start_frame < 0:
            # Negative starts address the end of the selected window: -1 means
            # "the last q_frame_count frames", for both 1- and 4-frame grids.
            q_start_frame = int(q_source.shape[0]) + q_start_frame - max(q_frames) + 1
        q_start = q_start_frame * tokens_per_frame
        kv_start = int(spec.get("kv_start_token", 0))
        q = _thnd_to_bhsd(q_source, q_start, max(q_counts))
        k = _thnd_to_bhsd(k_source, kv_start, max(profile_kv))
        v = _thnd_to_bhsd(v_source, kv_start, max(profile_kv))
    elif fmt == "bhsd_sample":
        sample_half = int(spec.get("sample_half", 0))
        if q_source.ndim != 4 or k_source.ndim != 4 or v_source.ndim != 4:
            raise ValueError(f"Expected BHSD-like sample tensors in {path}")
        q = q_source[sample_half : sample_half + 1].contiguous()
        k = k_source[sample_half : sample_half + 1].contiguous()
        v = v_source[sample_half : sample_half + 1].contiguous()
        q_counts = parse_int_list(spec.get("q_token_counts", [q.shape[2]]))
        profile_kv = [value for value in profile_kv if value <= k.shape[2]]
    else:
        raise ValueError(f"Unknown track format: {fmt}")
    del payload, q_source, k_source, v_source
    if not profile_kv:
        raise ValueError(f"No requested K lengths fit track {spec['name']}")
    if q.shape[1] != k.shape[1] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError(f"Incompatible Q/K/V shapes: {q.shape}, {k.shape}, {v.shape}")
    metadata.update(spec.get("metadata", {}))
    return LoadedTrack(
        name=str(spec["name"]),
        source_path=path,
        layer=int(spec["layer"]),
        q=q,
        k=k,
        v=v,
        q_token_counts=q_counts,
        kv_token_counts=profile_kv,
        metadata=metadata,
        representation_assumption=str(spec.get("representation_assumption", "unspecified")),
    )


def codebook_source_metadata(directory: Path, layer: int) -> dict[str, Any]:
    path = directory / f"layer_{layer}_head_0.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload.get("codebooks")
    if not torch.is_tensor(tensor):
        raise ValueError(f"Missing codebooks tensor in {path}")
    return {
        "serialized_dtype": str(tensor.dtype).replace("torch.", ""),
        "shape_per_head": [int(x) for x in tensor.shape],
        "input_norm_mode": payload.get("input_norm_mode", payload.get("config", {}).get("input_norm_mode")),
    }


def without_runtime_cache(payload: torch.Tensor | dict[str, Any]) -> torch.Tensor | dict[str, Any]:
    if torch.is_tensor(payload):
        return payload
    return {key: value for key, value in payload.items() if key not in {"runtime_codebooks", "runtime_codebook_sq"}}


def tensor_tree_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    return 0


def codebook_residency_bytes(value: torch.Tensor | dict[str, Any]) -> dict[str, int]:
    if torch.is_tensor(value):
        size = tensor_tree_nbytes(value)
        return {
            "packed_storage_bytes": 0,
            "runtime_codebooks_bytes": size,
            "runtime_squared_norm_bytes": 0,
            "other_metadata_tensor_bytes": 0,
            "total_tensor_tree_bytes": size,
        }
    runtime = tensor_tree_nbytes(value.get("runtime_codebooks"))
    squared = tensor_tree_nbytes(value.get("runtime_codebook_sq"))
    packed_keys = {"q_packed", "q_shape", "q", "scale"}
    packed = sum(tensor_tree_nbytes(value.get(key)) for key in packed_keys)
    total = tensor_tree_nbytes(value)
    return {
        "packed_storage_bytes": packed,
        "runtime_codebooks_bytes": runtime,
        "runtime_squared_norm_bytes": squared,
        "other_metadata_tensor_bytes": max(0, total - packed - runtime - squared),
        "total_tensor_tree_bytes": total,
    }


def peak_fields(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in (
            "baseline_allocated_mib",
            "baseline_reserved_mib",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "incremental_peak_allocated_mib",
            "incremental_peak_reserved_mib",
        )
    }


def materialized_gather_logits(table: torch.Tensor, indices: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Correctness-only gather that explicitly materializes [H,Q,K] logits."""
    levels, batch, heads, tokens = indices.shape
    if batch != 1:
        raise ValueError("RVQ attention microbenchmark requires B=1")
    if table.ndim != 3 or table.shape[0] != heads or table.shape[1] % levels:
        raise ValueError(f"Unexpected table/indices shapes: {table.shape}, {indices.shape}")
    centroids = table.shape[1] // levels
    q_tokens = int(table.shape[2])
    table_hlcq = table.reshape(heads, levels, centroids, q_tokens)
    ids = indices[:, 0].permute(1, 0, 2).long()  # [H,L,K]
    logits = torch.zeros((heads, q_tokens, tokens), device=table.device, dtype=table.dtype)
    for level in range(levels):
        gather_ids = ids[:, level, :, None].expand(heads, tokens, q_tokens)
        selected_kq = torch.gather(table_hlcq[:, level, :, :], 1, gather_ids)
        logits.add_(selected_kq.transpose(1, 2))
    return logits * scale[0, :, :, 0].to(logits.dtype)[:, None, :]


def materialized_probability(logits: torch.Tensor, softmax_scale: float) -> torch.Tensor:
    return torch.softmax(logits.float() * float(softmax_scale), dim=-1)


def materialized_v_histogram(
    probability: torch.Tensor,
    v_indices: torch.Tensor,
    v_scale: torch.Tensor,
    centroids: int,
) -> torch.Tensor:
    levels, batch, heads, tokens = v_indices.shape
    if batch != 1 or probability.shape != (heads, probability.shape[1], tokens):
        raise ValueError("Materialized V histogram shape mismatch")
    ids = v_indices[:, 0].permute(1, 0, 2).long()
    weighted = probability * v_scale[0, :, :, 0].float()[:, None, :]
    hist = torch.zeros(
        (heads, probability.shape[1], levels, centroids),
        device=probability.device,
        dtype=torch.float32,
    )
    for level in range(levels):
        hist[:, :, level, :].scatter_add_(
            2,
            ids[:, level, None, :].expand(heads, probability.shape[1], tokens),
            weighted,
        )
    return hist.reshape(heads, probability.shape[1], levels * centroids).contiguous()


class Recorder:
    def __init__(self, *, warmup: int, repeat: int, device: torch.device) -> None:
        self.warmup = warmup
        self.repeat = repeat
        self.device = device
        self.rows: list[dict[str, Any]] = []
        self.raw: dict[str, Any] = {}

    def stage(
        self,
        fn: Callable[[], Any],
        *,
        track: str,
        q_tokens: int | None,
        kv_tokens: int,
        dim: int,
        path: str,
        stage: str,
        timing_scope: str,
        warmup: int | None = None,
        repeat: int | None = None,
        memory: bool = True,
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage_warmup = self.warmup if warmup is None else int(warmup)
        stage_repeat = self.repeat if repeat is None else int(repeat)
        stats, samples = cuda_samples(
            fn,
            warmup=stage_warmup,
            repeat=stage_repeat,
            device=self.device,
        )
        peak = cuda_peak(fn, self.device) if memory else {}
        row = {
            "track": track,
            "q_tokens": q_tokens,
            "kv_tokens": kv_tokens,
            "head_dim": dim,
            "path": path,
            "stage": stage,
            "timing_scope": timing_scope,
            "warmup": stage_warmup,
            "repeat": stage_repeat,
            **stats,
            **peak,
            **(labels or {}),
        }
        self.rows.append(row)
        key = f"{track}/q{q_tokens if q_tokens is not None else 'na'}/k{kv_tokens}/{path}/{stage}/{timing_scope}"
        self.raw[key] = {"samples_ms": samples, "summary": row}
        return row


def median_ms(row: dict[str, Any] | None) -> float | None:
    return None if row is None else float(row["median_ms"])


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator / denominator)


def encode_kwargs(config: dict[str, Any], layer: int, tensor_name: str) -> dict[str, Any]:
    codebooks = config["codebooks"]
    return {
        "layer_idx": layer,
        "codebook_dir": Path(codebooks["k_dir"] if tensor_name == "K" else codebooks["v_dir"]),
        "batch_size": int(codebooks["encode_batch_size"]),
        "compute_dtype": str(codebooks["compute_dtype"]),
        "codebook_storage_dtype": str(codebooks["storage_dtype"]),
        "scale_storage_dtype": str(codebooks["scale_storage_dtype"]),
    }


def kernel_kwargs(config: dict[str, Any], path: str) -> dict[str, int]:
    raw = config["benchmark"][path]
    return {key: int(raw[key]) for key in ("block_q", "block_k", "num_warps", "num_stages")}


def make_histogram_call(
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    v_codebooks: torch.Tensor,
    *,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    softmax_scale: float,
    output_dtype: torch.dtype,
    tuning: dict[str, int],
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        actual_k_scale = _unpack_scale(packed_k, query.device).contiguous() if k_scale is None else k_scale
        actual_v_scale = _unpack_scale(packed_v, query.device).contiguous() if v_scale is None else v_scale
        return call_supported(
            rvq_histogram_attention_b1,
            query,
            packed_k["indices"],
            k_codebooks,
            actual_k_scale,
            packed_v["indices"],
            v_codebooks,
            actual_v_scale,
            softmax_scale=softmax_scale,
            output_dtype=output_dtype,
            **tuning,
        )

    return run


def make_tile_call(
    query: torch.Tensor,
    packed_k: dict[str, Any],
    k_codebooks: torch.Tensor,
    packed_v: dict[str, Any],
    v_codebooks: torch.Tensor,
    *,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    softmax_scale: float,
    output_dtype: torch.dtype,
    tuning: dict[str, int],
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        actual_k_scale = _unpack_scale(packed_k, query.device).contiguous() if k_scale is None else k_scale
        actual_v_scale = _unpack_scale(packed_v, query.device).contiguous() if v_scale is None else v_scale
        return call_supported(
            rvq_table_tile_v_attention_b1,
            query,
            packed_k["indices"],
            k_codebooks,
            actual_k_scale,
            packed_v["indices"],
            v_codebooks,
            actual_v_scale,
            softmax_scale=softmax_scale,
            output_dtype=output_dtype,
            **tuning,
        )

    return run


def benchmark_track(
    track: LoadedTrack,
    config: dict[str, Any],
    recorder: Recorder,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    codebook_cfg = config["codebooks"]
    heads, dim = int(track.k.shape[1]), int(track.k.shape[-1])
    k_dir = Path(codebook_cfg["k_dir"]).expanduser().resolve()
    v_dir = Path(codebook_cfg["v_dir"]).expanduser().resolve()
    compute_dtype = dtype_from_name(codebook_cfg["compute_dtype"])
    storage_dtype = str(codebook_cfg["storage_dtype"])
    source_k_meta = codebook_source_metadata(k_dir, track.layer)
    source_v_meta = codebook_source_metadata(v_dir, track.layer)
    labels = {
        "codebook_source_serialized_dtype": f"K:{source_k_meta['serialized_dtype']};V:{source_v_meta['serialized_dtype']}",
        "codebook_resident_storage_dtype": storage_dtype,
        "codebook_kernel_compute_dtype": str(compute_dtype).replace("torch.", ""),
        "codebook_runtime_strategy": str(codebook_cfg["runtime_strategy"]),
        "scale_storage_dtype": str(codebook_cfg["scale_storage_dtype"]),
    }

    loaded_k = load_layer_codebooks_from_dir(
        k_dir,
        track.layer,
        device,
        compute_dtype,
        codebook_storage_dtype=storage_dtype,
        num_heads=heads,
    )
    loaded_v = load_layer_codebooks_from_dir(
        v_dir,
        track.layer,
        device,
        compute_dtype,
        codebook_storage_dtype=storage_dtype,
        num_heads=heads,
    )
    codebook_bytes = {
        "k": codebook_residency_bytes(loaded_k),
        "v": codebook_residency_bytes(loaded_v),
    }
    codebook_bytes["combined"] = {
        key: int(codebook_bytes["k"][key] + codebook_bytes["v"][key])
        for key in codebook_bytes["k"]
    }
    # Time the actual resident INT4 -> compute tensor policy separately.  Main
    # attention timings below use the persistent materialization by design.
    cold_k = without_runtime_cache(loaded_k)
    cold_v = without_runtime_cache(loaded_v)
    cold_materialize_k = recorder.stage(
        lambda: materialize_rvq_codebooks(cold_k, compute_dtype),
        track=track.name,
        q_tokens=None,
        kv_tokens=0,
        dim=dim,
        path="codebook_residency",
        stage="int4_cold_materialize_k",
        timing_scope="cold_single_sample_diagnostic",
        warmup=0,
        repeat=1,
        labels=labels,
    )
    cold_materialize_v = recorder.stage(
        lambda: materialize_rvq_codebooks(cold_v, compute_dtype),
        track=track.name,
        q_tokens=None,
        kv_tokens=0,
        dim=dim,
        path="codebook_residency",
        stage="int4_cold_materialize_v",
        timing_scope="cold_single_sample_diagnostic",
        warmup=0,
        repeat=1,
        labels=labels,
    )
    k_codebooks = materialize_rvq_codebooks(loaded_k, compute_dtype).contiguous()
    v_codebooks = materialize_rvq_codebooks(loaded_v, compute_dtype).contiguous()
    if k_codebooks.ndim != 4 or v_codebooks.ndim != 4:
        raise ValueError("Only standard [L,H,C,D] RVQ codebooks are supported")
    if k_codebooks.shape != v_codebooks.shape:
        raise ValueError(f"K/V codebook shape mismatch: {k_codebooks.shape} vs {v_codebooks.shape}")
    levels, cb_heads, centroids, cb_dim = (int(x) for x in k_codebooks.shape)
    expected_levels = int(codebook_cfg["expected_levels"])
    expected_centroids = int(codebook_cfg["expected_centroids"])
    if (levels, cb_heads, centroids, cb_dim) != (expected_levels, heads, expected_centroids, dim):
        raise ValueError(
            "Codebook/data mismatch: "
            f"actual={(levels, cb_heads, centroids, cb_dim)} "
            f"expected={(expected_levels, heads, expected_centroids, dim)}"
        )

    shape_results: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    encode_by_k: dict[int, dict[str, Any]] = {}
    hist_tuning = kernel_kwargs(config, "histogram")
    tile_tuning = kernel_kwargs(config, "tile_decode")
    materialized_limit = int(config["benchmark"]["materialized_max_qk_elements"])

    for kv_tokens in track.kv_token_counts:
        key = track.k[:, :, :kv_tokens].to(device=device, dtype=dtype, non_blocking=True).contiguous()
        value = track.v[:, :, :kv_tokens].to(device=device, dtype=dtype, non_blocking=True).contiguous()
        k_kwargs = encode_kwargs(config, track.layer, "K")
        v_kwargs = encode_kwargs(config, track.layer, "V")

        def encode_k_fn() -> dict[str, Any]:
            return rvq_encode_tensor_cache(key, **k_kwargs)

        def encode_v_fn() -> dict[str, Any]:
            return rvq_encode_tensor_cache(value, **v_kwargs)

        def encode_kv_fn() -> tuple[dict[str, Any], dict[str, Any]]:
            return encode_k_fn(), encode_v_fn()

        # Ensure loaders, allocator pools, and encoder kernels are warm before
        # any event is recorded. Recorder performs the configured warmup again.
        packed_k = encode_k_fn()
        packed_v = encode_v_fn()
        sync(device)
        encode_k = recorder.stage(
            encode_k_fn,
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="rvq_encode",
            stage="encode_k",
            timing_scope="kernel_latency",
            labels=labels,
        )
        encode_v = recorder.stage(
            encode_v_fn,
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="rvq_encode",
            stage="encode_v",
            timing_scope="kernel_latency",
            labels=labels,
        )
        encode_kv = recorder.stage(
            encode_kv_fn,
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="rvq_encode",
            stage="encode_kv_joint",
            timing_scope="kernel_latency",
            labels=labels,
        )
        encode_by_k[kv_tokens] = {"k": encode_k, "v": encode_v, "kv": encode_kv}
        # Stable packed states used by every attention reuse. Encoding is not
        # repeated inside any attention-only closure.
        packed_k = encode_k_fn()
        packed_v = encode_v_fn()
        if tuple(packed_k["indices"].shape) != (levels, 1, heads, kv_tokens):
            raise ValueError(f"Unexpected K index layout: {tuple(packed_k['indices'].shape)}")
        if tuple(packed_v["indices"].shape) != (levels, 1, heads, kv_tokens):
            raise ValueError(f"Unexpected V index layout: {tuple(packed_v['indices'].shape)}")
        k_scale = _unpack_scale(packed_k, device).contiguous()
        v_scale = _unpack_scale(packed_v, device).contiguous()

        unpack_k = recorder.stage(
            lambda: _unpack_scale(packed_k, device).contiguous(),
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="scale_residency",
            stage="unpack_k_scale",
            timing_scope="packed_uint4_to_compute",
            labels=labels,
        )
        unpack_v = recorder.stage(
            lambda: _unpack_scale(packed_v, device).contiguous(),
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="scale_residency",
            stage="unpack_v_scale",
            timing_scope="packed_uint4_to_compute",
            labels=labels,
        )

        def decode_k_fn() -> torch.Tensor:
            return rvq_dequantize_tensor_cache(packed_k, dtype)

        def decode_v_fn() -> torch.Tensor:
            return rvq_dequantize_tensor_cache(packed_v, dtype)

        def decode_kv_fn() -> tuple[torch.Tensor, torch.Tensor]:
            return decode_k_fn(), decode_v_fn()

        decode_k = recorder.stage(
            decode_k_fn,
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="explicit_decode",
            stage="decode_k",
            timing_scope="attention_only_component",
            labels=labels,
        )
        decode_v = recorder.stage(
            decode_v_fn,
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="explicit_decode",
            stage="decode_v",
            timing_scope="attention_only_component",
            labels=labels,
        )
        decode_kv = recorder.stage(
            decode_kv_fn,
            track=track.name,
            q_tokens=None,
            kv_tokens=kv_tokens,
            dim=dim,
            path="explicit_decode",
            stage="decode_kv_joint",
            timing_scope="attention_only_component",
            labels=labels,
        )

        for q_tokens in track.q_token_counts:
            query = track.q[:, :, :q_tokens].to(device=device, dtype=dtype, non_blocking=True).contiguous()
            softmax_scale = 1.0 / math.sqrt(dim)
            fused_scale_max_k = int(config["benchmark"].get("fused_uint4_scale_max_k", -1))
            use_fused_scale = kv_tokens <= fused_scale_max_k
            scale_strategy = "fused_uint4_in_attention" if use_fused_scale else "external_uint4_unpack_once"
            bf16_ref = F.scaled_dot_product_attention(query, key, value)
            decoded_key = decode_k_fn()
            decoded_value = decode_v_fn()
            decoded_ref = F.scaled_dot_product_attention(query, decoded_key, decoded_value)
            linear_key = rvq_linear_decode_reference(
                packed_k["indices"], k_codebooks, k_scale, output_dtype=dtype
            )
            linear_value = rvq_linear_decode_reference(
                packed_v["indices"], v_codebooks, v_scale, output_dtype=dtype
            )
            linear_decode_ref = F.scaled_dot_product_attention(query, linear_key, linear_value)
            sync(device)

            bf16_sdpa = recorder.stage(
                lambda: F.scaled_dot_product_attention(query, key, value),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="bf16_reference",
                stage="sdpa",
                timing_scope="attention_only",
                labels=labels,
            )
            decoded_sdpa = recorder.stage(
                lambda: F.scaled_dot_product_attention(query, decoded_key, decoded_value),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="explicit_decode",
                stage="sdpa_on_decoded_kv",
                timing_scope="attention_only_component",
                labels=labels,
            )

            def decode_attention_fn() -> torch.Tensor:
                decoded_k_now = decode_k_fn()
                decoded_v_now = decode_v_fn()
                return F.scaled_dot_product_attention(query, decoded_k_now, decoded_v_now)

            decode_attention = recorder.stage(
                decode_attention_fn,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="explicit_decode",
                stage="decode_kv_plus_sdpa",
                timing_scope="attention_only_indices_cached",
                labels=labels,
            )
            correctness_rows.extend(
                [
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "linear_rvq_decode_attention_vs_bf16_original",
                        "error_class": "rvq_reconstruction_approximation",
                        **error_stats(linear_decode_ref, bf16_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "production_rvq_decode_attention_vs_bf16_original",
                        "error_class": "rvq_reconstruction_plus_decode_rounding",
                        **error_stats(decoded_ref, bf16_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "production_rvq_decode_vs_linear_rvq_decode_attention",
                        "error_class": "existing_per_level_bf16_decode_rounding_semantics",
                        **error_stats(decoded_ref, linear_decode_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "linear_rvq_k_reconstruction_vs_original_k",
                        "error_class": "rvq_k_reconstruction_approximation",
                        **error_stats(linear_key, key),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "linear_rvq_v_reconstruction_vs_original_v",
                        "error_class": "rvq_v_reconstruction_approximation",
                        **error_stats(linear_value, value),
                    },
                ]
            )

            table = build_rvq_attention_table(query, k_codebooks)
            lut = recorder.stage(
                lambda: build_rvq_attention_table(query, k_codebooks),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="shared_k_lut",
                stage="q_codebook_lut",
                timing_scope="preunpacked_kernel_component",
                labels=labels,
            )

            histogram_preunpacked = make_histogram_call(
                query,
                packed_k,
                k_codebooks,
                packed_v,
                v_codebooks,
                k_scale=k_scale,
                v_scale=v_scale,
                softmax_scale=softmax_scale,
                output_dtype=dtype,
                tuning=hist_tuning,
            )
            histogram_main = make_histogram_call(
                query,
                packed_k,
                k_codebooks,
                packed_v,
                v_codebooks,
                k_scale=None,
                v_scale=None,
                softmax_scale=softmax_scale,
                output_dtype=dtype,
                tuning=hist_tuning,
            )
            def histogram_fused_scale() -> torch.Tensor:
                return call_supported(
                    rvq_histogram_attention_packed_scale_b1,
                    query,
                    packed_k,
                    k_codebooks,
                    packed_v,
                    v_codebooks,
                    softmax_scale=softmax_scale,
                    output_dtype=dtype,
                    **hist_tuning,
                )
            numerator, denominator = call_supported(
                rvq_online_histogram_b1,
                table,
                packed_k["indices"],
                k_scale,
                packed_v["indices"],
                v_scale,
                k_centroids=centroids,
                v_centroids=centroids,
                softmax_scale=softmax_scale,
                **hist_tuning,
            )
            histogram_out = normalize_attention_output(
                call_supported(
                    rvq_histogram_contract,
                    numerator,
                    denominator,
                    v_codebooks,
                    output_dtype=dtype,
                )
            )
            histogram_out_fp32_contract = normalize_attention_output(
                call_supported(
                    rvq_histogram_contract,
                    numerator,
                    denominator,
                    v_codebooks,
                    output_dtype=dtype,
                    contract_dtype=torch.float32,
                )
            )
            histogram_fused_out = normalize_attention_output(histogram_fused_scale())
            correctness_rows.extend(
                [
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "online_histogram_bf16_contract_vs_linear_explicit_rvq_decode_attention",
                        "error_class": "performance_kernel_numerical",
                        **error_stats(histogram_out, linear_decode_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "online_histogram_fp32_contract_vs_linear_explicit_rvq_decode_attention",
                        "error_class": "online_core_numerical_with_fp32_contract",
                        **error_stats(histogram_out_fp32_contract, linear_decode_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "online_histogram_bf16_contract_vs_production_explicit_rvq_decode_attention",
                        "error_class": "kernel_numerical_plus_existing_decode_rounding_delta",
                        **error_stats(histogram_out, decoded_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "online_histogram_fused_uint4_scale_vs_external_unpack",
                        "error_class": "fused_scale_unpack_numerical",
                        **error_stats(histogram_fused_out, histogram_out),
                    },
                ]
            )
            online_core = recorder.stage(
                lambda: call_supported(
                    rvq_online_histogram_b1,
                    table,
                    packed_k["indices"],
                    k_scale,
                    packed_v["indices"],
                    v_scale,
                    k_centroids=centroids,
                    v_centroids=centroids,
                    softmax_scale=softmax_scale,
                    **hist_tuning,
                ),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage="online_histogram_core",
                timing_scope="precomputed_lut_preunpacked_scale",
                labels=labels,
            )
            online_contract = recorder.stage(
                lambda: call_supported(
                    rvq_histogram_contract,
                    numerator,
                    denominator,
                    v_codebooks,
                    output_dtype=dtype,
                ),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage="online_histogram_contract",
                timing_scope="preunpacked_kernel_component",
                labels=labels,
            )
            online_contract_fp32 = recorder.stage(
                lambda: call_supported(
                    rvq_histogram_contract,
                    numerator,
                    denominator,
                    v_codebooks,
                    output_dtype=dtype,
                    contract_dtype=torch.float32,
                ),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage="online_histogram_contract_fp32",
                timing_scope="correctness_diagnostic_not_performance_result",
                labels=labels,
            )
            histogram_kernel = recorder.stage(
                histogram_preunpacked,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage="online_attention_total_preunpacked_kernel",
                timing_scope="attention_only_lut_core_contract",
                labels=labels,
            )
            histogram_external_unpack = recorder.stage(
                histogram_main,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage="online_attention_total_external_scale_unpack",
                timing_scope="comparison_external_uint4_scale_unpack",
                labels=labels,
            )
            histogram_total = recorder.stage(
                histogram_fused_scale if use_fused_scale else histogram_main,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage=f"online_attention_total_adaptive_{scale_strategy}",
                timing_scope="attention_only_indices_cached",
                labels=labels,
            )

            tile_preunpacked = make_tile_call(
                query,
                packed_k,
                k_codebooks,
                packed_v,
                v_codebooks,
                k_scale=k_scale,
                v_scale=v_scale,
                softmax_scale=softmax_scale,
                output_dtype=dtype,
                tuning=tile_tuning,
            )
            tile_main = make_tile_call(
                query,
                packed_k,
                k_codebooks,
                packed_v,
                v_codebooks,
                k_scale=None,
                v_scale=None,
                softmax_scale=softmax_scale,
                output_dtype=dtype,
                tuning=tile_tuning,
            )
            def tile_fused_scale() -> torch.Tensor:
                return call_supported(
                    rvq_table_tile_v_attention_packed_scale_b1,
                    query,
                    packed_k,
                    k_codebooks,
                    packed_v,
                    v_codebooks,
                    softmax_scale=softmax_scale,
                    output_dtype=dtype,
                    **tile_tuning,
                )
            tile_out = normalize_attention_output(tile_preunpacked())
            tile_fused_out = normalize_attention_output(tile_fused_scale())
            correctness_rows.extend(
                [
                    {
                    "track": track.name,
                    "q_tokens": q_tokens,
                    "kv_tokens": kv_tokens,
                    "comparison": "tile_v_decode_vs_linear_explicit_rvq_decode_attention",
                    "error_class": "kernel_numerical",
                    **error_stats(tile_out, linear_decode_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "tile_v_decode_vs_production_explicit_rvq_decode_attention",
                        "error_class": "kernel_numerical_plus_existing_decode_rounding_delta",
                        **error_stats(tile_out, decoded_ref),
                    },
                    {
                        "track": track.name,
                        "q_tokens": q_tokens,
                        "kv_tokens": kv_tokens,
                        "comparison": "tile_v_fused_uint4_scale_vs_external_unpack",
                        "error_class": "fused_scale_unpack_numerical",
                        **error_stats(tile_fused_out, tile_out),
                    },
                ]
            )
            tile_core = recorder.stage(
                lambda: call_supported(
                    rvq_table_tile_v_attention_from_table_b1,
                    table,
                    packed_k["indices"],
                    k_scale,
                    packed_v["indices"],
                    v_codebooks,
                    v_scale,
                    k_centroids=centroids,
                    softmax_scale=softmax_scale,
                    output_dtype=dtype,
                    **tile_tuning,
                ),
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="tile_v_decode",
                stage="tile_v_decode_core_from_lut",
                timing_scope="precomputed_lut_preunpacked_scale",
                labels=labels,
            )
            tile_kernel = recorder.stage(
                tile_preunpacked,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="tile_v_decode",
                stage="tile_attention_total_preunpacked_kernel",
                timing_scope="attention_only_lut_tile_core",
                labels=labels,
            )
            tile_external_unpack = recorder.stage(
                tile_main,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="tile_v_decode",
                stage="tile_attention_total_external_scale_unpack",
                timing_scope="comparison_external_uint4_scale_unpack",
                labels=labels,
            )
            tile_total = recorder.stage(
                tile_fused_scale if use_fused_scale else tile_main,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="tile_v_decode",
                stage=f"tile_attention_total_adaptive_{scale_strategy}",
                timing_scope="attention_only_indices_cached",
                labels=labels,
            )

            def histogram_one_shot_fn() -> torch.Tensor:
                one_k = encode_k_fn()
                one_v = encode_v_fn()
                if not use_fused_scale:
                    return call_supported(
                        rvq_histogram_attention_b1,
                        query,
                        one_k["indices"],
                        k_codebooks,
                        _unpack_scale(one_k, device).contiguous(),
                        one_v["indices"],
                        v_codebooks,
                        _unpack_scale(one_v, device).contiguous(),
                        softmax_scale=softmax_scale,
                        output_dtype=dtype,
                        **hist_tuning,
                    )
                return call_supported(
                    rvq_histogram_attention_packed_scale_b1,
                    query,
                    one_k,
                    k_codebooks,
                    one_v,
                    v_codebooks,
                    softmax_scale=softmax_scale,
                    output_dtype=dtype,
                    **hist_tuning,
                )

            def tile_one_shot_fn() -> torch.Tensor:
                one_k = encode_k_fn()
                one_v = encode_v_fn()
                if not use_fused_scale:
                    return call_supported(
                        rvq_table_tile_v_attention_b1,
                        query,
                        one_k["indices"],
                        k_codebooks,
                        _unpack_scale(one_k, device).contiguous(),
                        one_v["indices"],
                        v_codebooks,
                        _unpack_scale(one_v, device).contiguous(),
                        softmax_scale=softmax_scale,
                        output_dtype=dtype,
                        **tile_tuning,
                    )
                return call_supported(
                    rvq_table_tile_v_attention_packed_scale_b1,
                    query,
                    one_k,
                    k_codebooks,
                    one_v,
                    v_codebooks,
                    softmax_scale=softmax_scale,
                    output_dtype=dtype,
                    **tile_tuning,
                )

            def explicit_one_shot_fn() -> torch.Tensor:
                one_k = encode_k_fn()
                one_v = encode_v_fn()
                decoded_k_now = rvq_dequantize_tensor_cache(one_k, dtype)
                decoded_v_now = rvq_dequantize_tensor_cache(one_v, dtype)
                return F.scaled_dot_product_attention(query, decoded_k_now, decoded_v_now)

            histogram_one_shot = recorder.stage(
                histogram_one_shot_fn,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="online_histogram",
                stage="encode_kv_plus_online_attention",
                timing_scope="measured_one_shot",
                labels=labels,
            )
            tile_one_shot = recorder.stage(
                tile_one_shot_fn,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="tile_v_decode",
                stage="encode_kv_plus_tile_attention",
                timing_scope="measured_one_shot",
                labels=labels,
            )
            explicit_one_shot = recorder.stage(
                explicit_one_shot_fn,
                track=track.name,
                q_tokens=q_tokens,
                kv_tokens=kv_tokens,
                dim=dim,
                path="explicit_decode",
                stage="encode_kv_plus_decode_kv_plus_sdpa",
                timing_scope="measured_one_shot",
                labels=labels,
            )

            materialized: dict[str, Any] = {"status": "skipped", "reason": None}
            qk_elements = heads * q_tokens * kv_tokens
            if qk_elements <= materialized_limit:
                logits = materialized_gather_logits(table, packed_k["indices"], k_scale)
                probability = materialized_probability(logits, softmax_scale)
                hist = materialized_v_histogram(probability, packed_v["indices"], v_scale, centroids)
                ones = torch.ones((heads, q_tokens), device=device, dtype=torch.float32)
                materialized_out = normalize_attention_output(
                    call_supported(rvq_histogram_contract, hist, ones, v_codebooks, output_dtype=dtype)
                )
                prototype_out = normalize_attention_output(
                    call_supported(
                        rvq_compressed_attention_reference,
                        query,
                        packed_k["indices"],
                        k_codebooks,
                        k_scale,
                        packed_v["indices"],
                        v_codebooks,
                        v_scale,
                        softmax_scale=softmax_scale,
                        output_dtype=dtype,
                    )
                )
                correctness_rows.extend(
                    [
                        {
                            "track": track.name,
                            "q_tokens": q_tokens,
                            "kv_tokens": kv_tokens,
                            "comparison": "materialized_histogram_vs_linear_explicit_rvq_decode_attention",
                            "error_class": "kernel_numerical",
                            **error_stats(materialized_out, linear_decode_ref),
                        },
                        {
                            "track": track.name,
                            "q_tokens": q_tokens,
                            "kv_tokens": kv_tokens,
                            "comparison": "compressed_reference_vs_linear_explicit_rvq_decode_attention",
                            "error_class": "kernel_numerical",
                            **error_stats(prototype_out, linear_decode_ref),
                        },
                        {
                            "track": track.name,
                            "q_tokens": q_tokens,
                            "kv_tokens": kv_tokens,
                            "comparison": "online_histogram_bf16_contract_vs_compressed_materialized_fp32_accum_reference",
                            "error_class": "kernel_numerical_isolated_from_rvq_approximation",
                            **error_stats(histogram_out, prototype_out),
                        },
                        {
                            "track": track.name,
                            "q_tokens": q_tokens,
                            "kv_tokens": kv_tokens,
                            "comparison": "online_histogram_fp32_contract_vs_compressed_materialized_fp32_accum_reference",
                            "error_class": "online_core_numerical_isolated_from_rvq_approximation",
                            **error_stats(histogram_out_fp32_contract, prototype_out),
                        },
                        {
                            "track": track.name,
                            "q_tokens": q_tokens,
                            "kv_tokens": kv_tokens,
                            "comparison": "tile_v_decode_vs_compressed_materialized_fp32_accum_reference",
                            "error_class": "kernel_numerical_isolated_from_rvq_approximation",
                            **error_stats(tile_out, prototype_out),
                        },
                    ]
                )
                gather_row = recorder.stage(
                    lambda: materialized_gather_logits(table, packed_k["indices"], k_scale),
                    track=track.name,
                    q_tokens=q_tokens,
                    kv_tokens=kv_tokens,
                    dim=dim,
                    path="materialized_correctness_prototype",
                    stage="k_index_gather_logits",
                    timing_scope="diagnostic_materializes_qk",
                    labels=labels,
                )
                softmax_row = recorder.stage(
                    lambda: materialized_probability(logits, softmax_scale),
                    track=track.name,
                    q_tokens=q_tokens,
                    kv_tokens=kv_tokens,
                    dim=dim,
                    path="materialized_correctness_prototype",
                    stage="row_softmax",
                    timing_scope="diagnostic_materializes_qk",
                    labels=labels,
                )
                histogram_row = recorder.stage(
                    lambda: materialized_v_histogram(probability, packed_v["indices"], v_scale, centroids),
                    track=track.name,
                    q_tokens=q_tokens,
                    kv_tokens=kv_tokens,
                    dim=dim,
                    path="materialized_correctness_prototype",
                    stage="v_histogram",
                    timing_scope="diagnostic_materializes_qk",
                    labels=labels,
                )
                contract_row = recorder.stage(
                    lambda: call_supported(rvq_histogram_contract, hist, ones, v_codebooks, output_dtype=dtype),
                    track=track.name,
                    q_tokens=q_tokens,
                    kv_tokens=kv_tokens,
                    dim=dim,
                    path="materialized_correctness_prototype",
                    stage="histogram_codebook_contract",
                    timing_scope="diagnostic_materializes_qk",
                    labels=labels,
                )
                materialized_total = recorder.stage(
                    lambda: call_supported(
                        rvq_compressed_attention_reference,
                        query,
                        packed_k["indices"],
                        k_codebooks,
                        k_scale,
                        packed_v["indices"],
                        v_codebooks,
                        v_scale,
                        softmax_scale=softmax_scale,
                        output_dtype=dtype,
                    ),
                    track=track.name,
                    q_tokens=q_tokens,
                    kv_tokens=kv_tokens,
                    dim=dim,
                    path="materialized_correctness_prototype",
                    stage="materialized_attention_total",
                    timing_scope="diagnostic_not_performance_result",
                    labels=labels,
                )
                materialized = {
                    "status": "ran_correctness_prototype",
                    "gather_ms": median_ms(gather_row),
                    "softmax_ms": median_ms(softmax_row),
                    "histogram_ms": median_ms(histogram_row),
                    "contract_ms": median_ms(contract_row),
                    "total_ms": median_ms(materialized_total),
                }
                del logits, probability, hist, ones, materialized_out, prototype_out
            else:
                materialized["reason"] = (
                    f"H*Q*K={qk_elements} exceeds materialized_max_qk_elements={materialized_limit}"
                )

            encode_joint_ms = median_ms(encode_kv)
            encode_stage_sum_ms = float(median_ms(encode_k) + median_ms(encode_v))
            fixed_reuse = int(config["reuse"]["fixed_attention_reuse"])
            hist_ms = median_ms(histogram_total)
            tile_ms = median_ms(tile_total)
            decode_attention_ms = median_ms(decode_attention)
            latency_models = {
                "online_histogram_one_shot": median_ms(histogram_one_shot),
                "tile_v_one_shot": median_ms(tile_one_shot),
                "explicit_decode_one_shot": median_ms(explicit_one_shot),
                "online_histogram_one_shot_component_sum": encode_joint_ms + hist_ms,
                "tile_v_one_shot_component_sum": encode_joint_ms + tile_ms,
                "explicit_decode_one_shot_component_sum": encode_joint_ms + decode_attention_ms,
                f"online_histogram_amortized_n{fixed_reuse}": encode_joint_ms / fixed_reuse + hist_ms,
                f"tile_v_amortized_n{fixed_reuse}": encode_joint_ms / fixed_reuse + tile_ms,
                f"explicit_decode_amortized_n{fixed_reuse}": encode_joint_ms / fixed_reuse + decode_attention_ms,
            }
            speedup = {
                "online_histogram_vs_bf16_attention_only": safe_ratio(median_ms(bf16_sdpa), hist_ms),
                "tile_v_vs_bf16_attention_only": safe_ratio(median_ms(bf16_sdpa), tile_ms),
                "online_histogram_vs_explicit_decode_attention": safe_ratio(decode_attention_ms, hist_ms),
                "tile_v_vs_explicit_decode_attention": safe_ratio(decode_attention_ms, tile_ms),
                "online_histogram_one_shot_vs_explicit_decode_one_shot": safe_ratio(
                    median_ms(explicit_one_shot), median_ms(histogram_one_shot)
                ),
                "tile_v_one_shot_vs_explicit_decode_one_shot": safe_ratio(
                    median_ms(explicit_one_shot), median_ms(tile_one_shot)
                ),
                f"online_histogram_amortized_n{fixed_reuse}_vs_bf16": safe_ratio(
                    median_ms(bf16_sdpa), encode_joint_ms / fixed_reuse + hist_ms
                ),
                f"tile_v_amortized_n{fixed_reuse}_vs_bf16": safe_ratio(
                    median_ms(bf16_sdpa), encode_joint_ms / fixed_reuse + tile_ms
                ),
                f"online_histogram_amortized_n{fixed_reuse}_vs_explicit_decode_amortized": safe_ratio(
                    encode_joint_ms / fixed_reuse + decode_attention_ms,
                    encode_joint_ms / fixed_reuse + hist_ms,
                ),
                f"tile_v_amortized_n{fixed_reuse}_vs_explicit_decode_amortized": safe_ratio(
                    encode_joint_ms / fixed_reuse + decode_attention_ms,
                    encode_joint_ms / fixed_reuse + tile_ms,
                ),
            }
            result = {
                "track": track.name,
                "layer": track.layer,
                "scale_unpack_strategy": scale_strategy,
                "shape": {"batch": 1, "heads": heads, "q_tokens": q_tokens, "kv_tokens": kv_tokens, "head_dim": dim},
                "timing_ms": {
                    "encode_k": median_ms(encode_k),
                    "encode_v": median_ms(encode_v),
                    "encode_k_plus_v_stage_sum": encode_stage_sum_ms,
                    "encode_kv_joint": encode_joint_ms,
                    "decode_k": median_ms(decode_k),
                    "decode_v": median_ms(decode_v),
                    "decode_kv_joint": median_ms(decode_kv),
                    "bf16_sdpa": median_ms(bf16_sdpa),
                    "decoded_sdpa_only": median_ms(decoded_sdpa),
                    "explicit_decode_attention_total": decode_attention_ms,
                    "q_codebook_lut": median_ms(lut),
                    "online_histogram_core": median_ms(online_core),
                    "online_histogram_contract": median_ms(online_contract),
                    "online_histogram_contract_fp32_correctness": median_ms(online_contract_fp32),
                    "online_histogram_preunpacked_kernel": median_ms(histogram_kernel),
                    "online_histogram_external_scale_unpack": median_ms(histogram_external_unpack),
                    "online_histogram_attention_only": hist_ms,
                    "tile_v_core_from_lut": median_ms(tile_core),
                    "tile_v_preunpacked_kernel": median_ms(tile_kernel),
                    "tile_v_external_scale_unpack": median_ms(tile_external_unpack),
                    "tile_v_attention_only": tile_ms,
                    "unpack_k_scale": median_ms(unpack_k),
                    "unpack_v_scale": median_ms(unpack_v),
                },
                "latency_models_ms": latency_models,
                "speedup": speedup,
                "peak_gpu_memory_mib": {
                    "encode_kv": peak_fields(encode_kv),
                    "bf16_sdpa": peak_fields(bf16_sdpa),
                    "explicit_decode_attention": peak_fields(decode_attention),
                    "online_histogram_attention": peak_fields(histogram_total),
                    "tile_v_attention": peak_fields(tile_total),
                    "online_histogram_one_shot": peak_fields(histogram_one_shot),
                    "tile_v_one_shot": peak_fields(tile_one_shot),
                    "explicit_decode_one_shot": peak_fields(explicit_one_shot),
                },
                "packed_cache_residency_bytes": {
                    "k": tensor_tree_nbytes(packed_k),
                    "v": tensor_tree_nbytes(packed_v),
                    "kv_total": tensor_tree_nbytes(packed_k) + tensor_tree_nbytes(packed_v),
                },
                "materialized_correctness_prototype": materialized,
            }
            shape_results.append(result)
            sync(device)
        sync(device)

    residency = {
        "source": {
            "k": source_k_meta,
            "v": source_v_meta,
        },
        "configured_resident_storage_dtype": storage_dtype,
        "kernel_compute_dtype": str(compute_dtype).replace("torch.", ""),
        "runtime_strategy": codebook_cfg["runtime_strategy"],
        "tensor_tree_residency_bytes": codebook_bytes,
        "tensor_tree_residency_mib": {
            tensor_name: {key: value / MIB for key, value in breakdown.items()}
            for tensor_name, breakdown in codebook_bytes.items()
        },
        "cold_materialize_k_ms": median_ms(cold_materialize_k),
        "cold_materialize_v_ms": median_ms(cold_materialize_v),
        "scale_storage_dtype": str(codebook_cfg["scale_storage_dtype"]),
        "index_layout": "LBHS ([levels,batch,heads,sequence])",
        "input_layout": "BHSD",
    }
    return shape_results, correctness_rows, residency


def compute_break_even(shape_results: list[dict[str, Any]], fixed_reuse: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in shape_results:
        groups.setdefault((row["track"], int(row["shape"]["q_tokens"])), []).append(row)
    for (track, q_tokens), rows in groups.items():
        rows.sort(key=lambda row: int(row["shape"]["kv_tokens"]))
        group_key = f"{track}/q{q_tokens}"
        result[group_key] = {}
        comparisons = {
            "online_histogram_attention_vs_explicit_decode_attention": lambda row: (
                row["timing_ms"]["online_histogram_attention_only"],
                row["timing_ms"]["explicit_decode_attention_total"],
            ),
            "tile_v_attention_vs_explicit_decode_attention": lambda row: (
                row["timing_ms"]["tile_v_attention_only"],
                row["timing_ms"]["explicit_decode_attention_total"],
            ),
            f"online_histogram_amortized_n{fixed_reuse}_vs_bf16": lambda row: (
                row["latency_models_ms"][f"online_histogram_amortized_n{fixed_reuse}"],
                row["timing_ms"]["bf16_sdpa"],
            ),
            f"tile_v_amortized_n{fixed_reuse}_vs_bf16": lambda row: (
                row["latency_models_ms"][f"tile_v_amortized_n{fixed_reuse}"],
                row["timing_ms"]["bf16_sdpa"],
            ),
            f"online_histogram_amortized_n{fixed_reuse}_vs_explicit_decode_amortized": lambda row: (
                row["latency_models_ms"][f"online_histogram_amortized_n{fixed_reuse}"],
                row["latency_models_ms"][f"explicit_decode_amortized_n{fixed_reuse}"],
            ),
            f"tile_v_amortized_n{fixed_reuse}_vs_explicit_decode_amortized": lambda row: (
                row["latency_models_ms"][f"tile_v_amortized_n{fixed_reuse}"],
                row["latency_models_ms"][f"explicit_decode_amortized_n{fixed_reuse}"],
            ),
        }
        for name, select in comparisons.items():
            wins = [select(row)[0] < select(row)[1] for row in rows]
            winning_k = [
                int(row["shape"]["kv_tokens"])
                for row, wins_at_k in zip(rows, wins, strict=True)
                if wins_at_k
            ]
            sustained = [
                int(rows[index]["shape"]["kv_tokens"])
                for index in range(len(rows))
                if all(wins[index:])
            ]
            result[group_key][name] = {
                "first_measured_win_k": winning_k[0] if winning_k else None,
                "first_sustained_win_k": sustained[0] if sustained else None,
                "winning_measured_k": winning_k,
            }
    return result


def compute_run_integrated(
    shape_results: list[dict[str, Any]], reuse: dict[str, Any]
) -> dict[str, Any]:
    schedule = reuse["run_integrated_schedule"]
    q_tokens = int(schedule["q_tokens"])
    span_tokens = int(schedule["span_tokens"])
    spans = int(schedule["spans"])
    calls = int(schedule["attention_calls_per_cache_state"])
    expected_k = [span_tokens * index for index in range(1, spans + 1)]
    candidates = {
        int(row["shape"]["kv_tokens"]): row
        for row in shape_results
        if row["track"] == "real_thnd" and int(row["shape"]["q_tokens"]) == q_tokens
    }
    missing = [tokens for tokens in expected_k if tokens not in candidates]
    if missing or span_tokens not in candidates:
        return {
            "status": "not_computed",
            "reason": "Exact measured schedule points are required; interpolation is disabled.",
            "missing_shapes": [{"q_tokens": q_tokens, "kv_tokens": tokens} for tokens in missing],
            "formula": "(sum(E_delta) + 4*sum_c(A(Q=3520,K=3520*c))) / (4*20)",
        }
    delta_encode = float(candidates[span_tokens]["timing_ms"]["encode_kv_joint"])
    encode_total = spans * delta_encode
    denominator = calls * spans

    def aggregate(attention_key: str) -> float:
        return (
            encode_total
            + calls * sum(float(candidates[t]["timing_ms"][attention_key]) for t in expected_k)
        ) / denominator

    hist = aggregate("online_histogram_attention_only")
    tile = aggregate("tile_v_attention_only")
    explicit = aggregate("explicit_decode_attention_total")
    bf16 = (
        calls * sum(float(candidates[t]["timing_ms"]["bf16_sdpa"]) for t in expected_k) / denominator
    )
    return {
        "status": "computed_from_exact_points",
        "q_tokens": q_tokens,
        "kv_tokens": expected_k,
        "attention_calls_per_cache_state": calls,
        "span_encode_events": spans,
        "delta_encode_kv_ms": delta_encode,
        "total_encode_ms": encode_total,
        "average_per_attention_call_ms": {
            "bf16_sdpa_no_encoding": bf16,
            "explicit_decode_encoding_inclusive": explicit,
            "online_histogram_encoding_inclusive": hist,
            "tile_v_encoding_inclusive": tile,
        },
        "speedup": {
            "online_histogram_vs_bf16": safe_ratio(bf16, hist),
            "tile_v_vs_bf16": safe_ratio(bf16, tile),
            "online_histogram_vs_explicit_decode": safe_ratio(explicit, hist),
            "tile_v_vs_explicit_decode": safe_ratio(explicit, tile),
        },
        "formula": "(sum(E_delta) + 4*sum_c(A(Q=3520,K=3520*c))) / (4*20)",
    }


def summarize_bottlenecks(shape_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in shape_results:
        timing = item["timing_ms"]
        scale_unpack = float(timing["unpack_k_scale"] + timing["unpack_v_scale"])
        histogram_components = {
            "q_codebook_lut": float(timing["q_codebook_lut"]),
            "online_histogram_core": float(timing["online_histogram_core"]),
            "histogram_contract": float(timing["online_histogram_contract"]),
        }
        tile_components = {
            "q_codebook_lut": float(timing["q_codebook_lut"]),
            "tile_v_core_from_lut": float(timing["tile_v_core_from_lut"]),
        }
        histogram_largest = max(histogram_components, key=histogram_components.get)
        tile_largest = max(tile_components, key=tile_components.get)
        encode = float(timing["encode_kv_joint"])
        hist_attention = float(timing["online_histogram_attention_only"])
        tile_attention = float(timing["tile_v_attention_only"])
        rows.append(
            {
                "track": item["track"],
                **item["shape"],
                "histogram_largest_measured_component": histogram_largest,
                "histogram_largest_component_ms": histogram_components[histogram_largest],
                "tile_largest_measured_component": tile_largest,
                "tile_largest_component_ms": tile_components[tile_largest],
                "encode_vs_histogram_attention_ratio": safe_ratio(encode, hist_attention),
                "encode_vs_tile_attention_ratio": safe_ratio(encode, tile_attention),
                "external_scale_unpack_k_plus_v_comparison_ms": scale_unpack,
                "fixed_cost_warning": (
                    "Q-codebook LUT dominates this measured decomposition"
                    if histogram_largest == "q_codebook_lut" or tile_largest == "q_codebook_lut"
                    else None
                ),
            }
        )
    return rows


def next_step_suggestions(bottlenecks: list[dict[str, Any]]) -> list[str]:
    observed = {
        row[key]
        for row in bottlenecks
        for key in ("histogram_largest_measured_component", "tile_largest_measured_component")
    }
    suggestions: list[str] = []
    if "online_histogram_core" in observed:
        suggestions.append(
            "Profile online_histogram_core with Nsight Compute for shared-memory bank conflicts, register spill, occupancy, and atomics/reduction pressure."
        )
    if "tile_v_core_from_lut" in observed:
        suggestions.append(
            "Profile tile_v_core_from_lut for Tensor-Core utilization and tune BLOCK_Q/BLOCK_K before preferring histogram aggregation."
        )
    if "q_codebook_lut" in observed:
        suggestions.append(
            "Amortize or fuse the Q-codebook LUT at short K, and verify the table is consumed from SRAM/cache rather than random HBM lookup."
        )
    if any((row.get("encode_vs_histogram_attention_ratio") or 0.0) > 1.0 for row in bottlenecks):
        suggestions.append("Encoding is larger than histogram attention for at least one shape; prioritize encoder batching/fusion and cache indices across reuse.")
    suggestions.append(
        "Use Nsight Systems/Compute for the production grid to quantify HBM traffic for U/G, register pressure, occupancy, spills, and Tensor-Core utilization; CUDA events alone do not establish these hardware counters."
    )
    return suggestions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_float(value: Any) -> str:
    return "null" if value is None else f"{float(value):.4f}"


def render_summary(summary: dict[str, Any]) -> str:
    fixed_reuse = int(summary["reuse"]["fixed_attention_reuse"])
    lines = [
        "# RVQ-aware attention microbenchmark",
        "",
        f"Run: `{summary['run_dir']}`",
        "",
        "All latency values below are CUDA-event medians in ms. File I/O, CUDA initialization, and warmup are excluded.",
        "The materialized path is a correctness prototype only; performance conclusions use the online-histogram and tile-decode paths.",
        "",
        "| track | Q | K | BF16 SDPA | decode+SDPA | hist attn | tile attn | encode K+V | hist one-shot | tile one-shot | "
        f"hist amort. N={fixed_reuse} | tile amort. N={fixed_reuse} | decode/hist | decode/tile |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["shapes"]:
        shape, timing, latency, speedup = row["shape"], row["timing_ms"], row["latency_models_ms"], row["speedup"]
        lines.append(
            "| {track} | {q} | {k} | {bf16} | {decode} | {hist} | {tile} | {encode} | {hist_one} | {tile_one} | "
            "{hist_amort} | {tile_amort} | {decode_hist} | {decode_tile} |".format(
                track=row["track"],
                q=shape["q_tokens"],
                k=shape["kv_tokens"],
                bf16=markdown_float(timing["bf16_sdpa"]),
                decode=markdown_float(timing["explicit_decode_attention_total"]),
                hist=markdown_float(timing["online_histogram_attention_only"]),
                tile=markdown_float(timing["tile_v_attention_only"]),
                encode=markdown_float(timing["encode_kv_joint"]),
                hist_one=markdown_float(latency["online_histogram_one_shot"]),
                tile_one=markdown_float(latency["tile_v_one_shot"]),
                hist_amort=markdown_float(latency[f"online_histogram_amortized_n{fixed_reuse}"]),
                tile_amort=markdown_float(latency[f"tile_v_amortized_n{fixed_reuse}"]),
                decode_hist=markdown_float(speedup["online_histogram_vs_explicit_decode_attention"]),
                decode_tile=markdown_float(speedup["tile_v_vs_explicit_decode_attention"]),
            )
        )
    lines.extend(
        [
            "",
            "## Correctness",
            "",
            "| track | Q | K | comparison | max abs | mean abs | MSE | relative L2 | cosine | finite |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["correctness"]:
        lines.append(
            "| {track} | {q} | {k} | {comparison} | {max_abs} | {mean_abs} | {mse} | {relative} | {cosine} | {finite} |".format(
                track=row["track"],
                q=row["q_tokens"],
                k=row["kv_tokens"],
                comparison=row["comparison"],
                max_abs=markdown_float(row["max_abs_error"]),
                mean_abs=markdown_float(row["mean_abs_error"]),
                mse=markdown_float(row["mse"]),
                relative=markdown_float(row["relative_l2_error"]),
                cosine=markdown_float(row["cosine_similarity"]),
                finite="yes" if row["all_finite"] else "NO",
            )
        )
    lines.extend(["", "## Break-even", "", "Both the first isolated measured win and the first win sustained through every larger measured K are reported; no interpolation is used.", "", "```json", json.dumps(summary["break_even_k_tokens"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Run-integrated 21-chunk schedule", "", "```json", json.dumps(summary["run_integrated"], indent=2, sort_keys=True), "```"])
    lines.extend(["", "## Measured bottlenecks and next steps", "", "```json", json.dumps(summary["bottlenecks"], indent=2, sort_keys=True), "```", ""])
    lines.extend(f"- {item}" for item in summary["next_step_suggestions"])
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `attention_only` fuses packed uint4 per-token scale unpack into the core; external-unpack and preunpacked paths are reported separately.",
            "- Source, resident-storage, and kernel-compute codebook dtypes are separate labels; cold materialization is a separate diagnostic.",
            "- Peak memory reports baseline, absolute peak, and incremental peak. Reserved-memory values retain allocator history in benchmark order.",
            "- BF16-vs-decoded error is RVQ approximation error. Compressed-vs-decoded error is kernel numerical error.",
            "- `speedup > 1` means the numerator/reference path is slower than the RVQ-aware path.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    validated = validate_config(config, args)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "config": str(config_path),
                    "profile": args.profile,
                    "tracks": [item["name"] for item in validated["tracks"]],
                },
                indent=2,
            )
        )
        return

    device = torch.device(args.device or config.get("device", "cuda"))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    dtype = dtype_from_name(config.get("dtype", "bf16"))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Initialize the context and SDPA backend before any measured interval.
    init = torch.zeros((1, 1, 1, 16), device=device, dtype=dtype)
    F.scaled_dot_product_attention(init, init, init)
    sync(device)
    del init

    tracks = [load_track(spec, validated["profile"]) for spec in validated["tracks"]]
    output_root = (args.output_root or Path(config["output_root"])).expanduser()
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    tag = args.tag or f"{config['name']}_{args.profile}"
    run_dir = unique_run_dir(output_root, tag)
    config_hash = sha256_file(config_path)
    heads_by_track = {track.name: int(track.k.shape[1]) for track in tracks}
    codebook_paths = {
        "k": str(Path(config["codebooks"]["k_dir"]).expanduser().resolve()),
        "v": str(Path(config["codebooks"]["v_dir"]).expanduser().resolve()),
    }
    # All selected tracks use one layer in the shipped profiles. Hash each
    # layer/head set explicitly if a custom multi-layer config is supplied.
    codebook_hashes: dict[str, dict[str, str]] = {"k": {}, "v": {}}
    for track in tracks:
        for tensor_name, directory_key in (("k", "k_dir"), ("v", "v_dir")):
            codebook_hashes[tensor_name][str(track.layer)] = sha256_layer_codebooks(
                Path(config["codebooks"][directory_key]).expanduser().resolve(),
                track.layer,
                heads_by_track[track.name],
            )
    tensor_hashes = {track.name: sha256_file(track.source_path) for track in tracks}
    now = datetime.now().astimezone().isoformat()
    gpu_props = torch.cuda.get_device_properties(device)
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": codebook_paths,
        "codebook_hash": codebook_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": torch.cuda.get_device_name(device),
        "seed": seed,
        "prompt_list": [
            str(
                track.metadata.get("prompt")
                or track.metadata.get("prompt_text")
                or f"{spec.get('split', 'sample')}:{spec.get('prompt_id', spec['name'])}"
            )
            for track, spec in zip(tracks, validated["tracks"], strict=True)
        ],
        "timestamp": now,
        "tensor_path": {track.name: str(track.source_path) for track in tracks},
        "tensor_hash": tensor_hashes,
        "tensor_residency": {
            "source": "CPU torch.save payload",
            "timed_inputs": f"CUDA BHSD {str(dtype).replace('torch.', '')}",
            "packed_indices": "CUDA LBHS",
        },
        "representation_provenance": {
            track.name: {
                "assumption": track.representation_assumption,
                "source_metadata": track.metadata,
            }
            for track in tracks
        },
        "reuse_provenance": config["reuse"],
        "kernel_implementation": {
            "k_lut_layout": "H_(level_centroid)_Q_bf16",
            "k_indexing": "coalesced_direct_indexed_load_from_l2_global_table",
            "online_softmax": "single_pass_tiled_no_qk_materialization",
            "histogram_v": "dense_one_hot_tensor_core_correctness_performance_candidate",
            "tile_v": "direct_indexed_codeword_gather_tile_local_decode_then_tensor_core_pv",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu_total_memory_mib": gpu_props.total_memory / MIB,
            "gpu_compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
            "warmup": validated["warmup"],
            "repeat": validated["repeat"],
            "profile": args.profile,
            "command": [sys.executable, *sys.argv],
            "rvq_runtime_codebook_cache": os.getenv("HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE", "1"),
        },
    }
    (run_dir / "run_info.json").write_text(json.dumps(jsonable(run_info), indent=2, sort_keys=True), encoding="utf-8")

    recorder = Recorder(warmup=validated["warmup"], repeat=validated["repeat"], device=device)
    shapes: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    residency: dict[str, Any] = {}
    try:
        with torch.inference_mode():
            for track in tracks:
                track_shapes, track_correctness, track_residency = benchmark_track(
                    track, config, recorder, device, dtype
                )
                shapes.extend(track_shapes)
                correctness.extend(track_correctness)
                residency[track.name] = track_residency
        fixed_reuse = int(config["reuse"]["fixed_attention_reuse"])
        summary = {
            "schema_version": 1,
            "run_dir": str(run_dir),
            "profile": args.profile,
            "config_hash": config_hash,
            "shapes": shapes,
            "correctness": correctness,
            "codebook_and_scale_residency": residency,
            "reuse": config["reuse"],
            "break_even_k_tokens": compute_break_even(shapes, fixed_reuse),
            "run_integrated": compute_run_integrated(shapes, config["reuse"]),
            "bottlenecks": summarize_bottlenecks(shapes),
            "performance_conclusion_paths": ["online_histogram", "tile_v_decode"],
            "kernel_implementation": run_info["kernel_implementation"],
            "materialized_path_is_correctness_only": True,
        }
        summary["next_step_suggestions"] = next_step_suggestions(summary["bottlenecks"])
        write_csv(run_dir / "timings.csv", recorder.rows)
        write_csv(run_dir / "correctness.csv", correctness)
        (run_dir / "raw_timings.json").write_text(
            json.dumps(jsonable({"warmup": validated["warmup"], "repeat": validated["repeat"], "stages": recorder.raw}), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "summary.json").write_text(json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    except Exception as exc:
        (run_dir / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc), "timestamp": datetime.now().astimezone().isoformat()}, indent=2),
            encoding="utf-8",
        )
        raise
    print(json.dumps({"status": "complete", "run_dir": str(run_dir), "shapes": len(shapes)}, indent=2))


if __name__ == "__main__":
    main()
