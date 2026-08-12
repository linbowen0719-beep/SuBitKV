#!/usr/bin/env python3
"""Benchmark bit-exact lossless compression for saved FP16/BF16 KV cache files.

The script is intentionally offline-only: it loads existing cache artifacts,
never regenerates KV cache, and writes results only under ``--out_dir``.

The NVIDIA nvCOMP Python package is used when available. The GPU path keeps
tensor bytes resident on CUDA during the measured compression/decompression
section and uses CUDA events for backend timing. CPU zstd/lz4 fallbacks are
kept as honest sanity/benchmark backends.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


RESULT_COLUMNS = [
    "file",
    "tensor_name",
    "kv_type",
    "layer",
    "head",
    "block_index",
    "method",
    "backend",
    "dtype",
    "shape",
    "raw_bytes",
    "compressed_bytes",
    "compression_ratio",
    "compress_ms",
    "decompress_ms",
    "compress_gbs",
    "decompress_gbs",
    "lossless_pass",
    "notes",
]

SUPPORTED_EXTENSIONS = {
    ".pt",
    ".pth",
    ".npy",
    ".npz",
    ".bin",
    ".safetensors",
    ".ckpt",
}

SUPPORTED_TORCH_DTYPES = {torch.float16, torch.bfloat16}
TOKENS_PER_FRAME = 880


@dataclass(frozen=True)
class TensorBlockRef:
    file_path: Path
    rel_file: str
    tensor_name: str
    kv_type: str
    layer: str
    head: str
    block_index: int
    dtype: str
    shape: tuple[int, ...]
    token_start: int | None
    token_end: int | None
    chunk_i: int | None
    source_format: str

    def stream_key(self) -> tuple[str, str, str, str, tuple[int, ...]]:
        shape_key = self.shape
        if self.shape:
            # Token length can vary on edge chunks; temporal XOR only needs
            # equal byte lengths, but stream identity should not split on S.
            shape_key = tuple(-1 if i == 2 else dim for i, dim in enumerate(self.shape))
        return (self.kv_type, self.layer, self.head, self.tensor_name, shape_key)


class Codec:
    name = "codec"

    def compress(self, payload: bytes) -> bytes:
        raise NotImplementedError

    def decompress(self, payload: bytes) -> bytes:
        raise NotImplementedError


@dataclass(frozen=True)
class NvcompBackend:
    name: str
    algorithm: str
    extra_kwargs: tuple[tuple[str, Any], ...] = ()

    def codec_kwargs(self) -> dict[str, Any]:
        kwargs = {"algorithm": self.algorithm, "data_type": "|u1"}
        kwargs.update(dict(self.extra_kwargs))
        return kwargs


class ZstdCodec(Codec):
    name = "zstd_cpu"

    def __init__(self, level: int = 1) -> None:
        import zstandard as zstd

        self._compressor = zstd.ZstdCompressor(level=level)
        self._decompressor = zstd.ZstdDecompressor()

    def compress(self, payload: bytes) -> bytes:
        return self._compressor.compress(payload)

    def decompress(self, payload: bytes) -> bytes:
        return self._decompressor.decompress(payload)


class Lz4Codec(Codec):
    name = "lz4_cpu"

    def __init__(self, compression_level: int = 0) -> None:
        import lz4.frame

        self._lz4 = lz4.frame
        self._compression_level = compression_level

    def compress(self, payload: bytes) -> bytes:
        return self._lz4.compress(payload, compression_level=self._compression_level)

    def decompress(self, payload: bytes) -> bytes:
        return self._lz4.decompress(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bit-exact lossless compression benchmark for saved FP16/BF16 KV cache tensors."
    )
    parser.add_argument("--kv_path", type=Path, help="Saved KV cache file or directory.")
    parser.add_argument("--out_dir", type=Path, help="Directory for CSV/Markdown outputs.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--max_blocks", type=int, default=None)
    parser.add_argument(
        "--block_mode",
        choices=("chunk", "tensor", "head", "token_tile"),
        default="chunk",
        help=(
            "'chunk'/'tensor' benchmarks one saved tensor block; 'head' splits BHSD tensors by head; "
            "'token_tile' splits BHSD tensors into one head x token tile blocks."
        ),
    )
    parser.add_argument(
        "--tile_tokens",
        type=int,
        default=64,
        help="Token tile length used when --block_mode token_tile.",
    )
    parser.add_argument(
        "--backends",
        default="auto",
        help=(
            "Comma-separated backends. 'auto' selects fast nvCOMP backends when CUDA is available, "
            "otherwise CPU fallbacks. Examples: nvcomp_gdeflate,nvcomp_ans,nvcomp_lz4,zstd_cpu."
        ),
    )
    parser.add_argument("--zstd_level", type=int, default=1)
    parser.add_argument("--lz4_level", type=int, default=0)
    parser.add_argument("--label", default=None, help="Human-readable label for this benchmark run.")
    parser.add_argument(
        "--combine_result",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Combine an existing result CSV into a Markdown comparison. May be repeated.",
    )
    parser.add_argument("--combined_out", type=Path, help="Output Markdown path for --combine_result mode.")
    return parser.parse_args()


def natural_key(path: Path) -> list[Any]:
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part for part in parts]


def scan_candidate_files(kv_path: Path) -> list[Path]:
    if kv_path.is_file():
        return [kv_path]
    if not kv_path.exists():
        raise FileNotFoundError(kv_path)
    candidates = [
        path
        for path in kv_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(candidates, key=natural_key)


def print_candidates(candidates: list[Path]) -> None:
    print(f"Discovered {len(candidates)} candidate tensor/cache files:")
    for path in candidates:
        print(f"  {path}")


def detect_nvcomp() -> dict[str, Any]:
    modules = {}
    nvcomp_version = None
    for module_name in ("nvidia.nvcomp", "nvcomp", "pynvcomp", "cupy"):
        try:
            if module_name == "nvidia.nvcomp":
                from nvidia import nvcomp as nvidia_nvcomp

                nvcomp_version = getattr(nvidia_nvcomp, "__version__", None)
                modules[module_name] = True
            else:
                __import__(module_name)
                modules[module_name] = True
        except Exception as exc:  # noqa: BLE001 - import failures are reported.
            modules[module_name] = f"unavailable: {exc.__class__.__name__}: {exc}"
    available = modules.get("nvidia.nvcomp") is True
    cuda_available = torch.cuda.is_available()
    if available and cuda_available:
        note = "nvidia.nvcomp is available; GPU backends use CUDA-event timing on device-resident uint8 tensors."
        algorithm_status = "available through nvidia.nvcomp.Codec"
    elif available:
        note = "nvidia.nvcomp is importable, but CUDA is not available to PyTorch."
        algorithm_status = "nvCOMP importable, CUDA unavailable"
    else:
        note = "No implemented Python nvCOMP backend was found. CPU fallback backends are used only as fallback measurements."
        algorithm_status = "unavailable through nvidia.nvcomp"
    return {
        "available": available and cuda_available,
        "importable": available,
        "version": nvcomp_version,
        "modules": modules,
        "note": note,
        "algorithms": {
            "lz4": algorithm_status,
            "gdeflate": algorithm_status,
            "zstd": algorithm_status,
            "deflate": algorithm_status,
            "ans": algorithm_status,
            "snappy": algorithm_status,
            "bitcomp": algorithm_status,
            "cascaded": algorithm_status,
        },
    }


def nvcomp_backend_catalog() -> dict[str, NvcompBackend]:
    return {
        "nvcomp_lz4": NvcompBackend("nvcomp_lz4", "LZ4"),
        "nvcomp_gdeflate": NvcompBackend(
            "nvcomp_gdeflate",
            "GDeflate",
            extra_kwargs=(("algorithm_type", 0),),
        ),
        "nvcomp_ans": NvcompBackend("nvcomp_ans", "ANS"),
        "nvcomp_snappy": NvcompBackend("nvcomp_snappy", "Snappy"),
        "nvcomp_zstd": NvcompBackend("nvcomp_zstd", "Zstd"),
        "nvcomp_deflate": NvcompBackend("nvcomp_deflate", "Deflate"),
        "nvcomp_bitcomp": NvcompBackend("nvcomp_bitcomp", "Bitcomp"),
        "nvcomp_cascaded": NvcompBackend("nvcomp_cascaded", "Cascaded"),
    }


def available_codecs(
    args: argparse.Namespace,
    nvcomp_info: dict[str, Any],
) -> tuple[list[Codec], list[NvcompBackend], list[str]]:
    requested = [b.strip().lower() for b in args.backends.split(",") if b.strip()]
    catalog = nvcomp_backend_catalog()
    if requested == ["auto"]:
        requested = ["nvcomp_gdeflate", "nvcomp_ans", "nvcomp_lz4"] if nvcomp_info["available"] else ["zstd_cpu", "lz4_cpu"]

    codecs: list[Codec] = []
    nvcomp_backends: list[NvcompBackend] = []
    unavailable: list[str] = []

    for backend in requested:
        if backend in {"zstd", "zstd_cpu"}:
            try:
                codecs.append(ZstdCodec(level=args.zstd_level))
            except Exception as exc:  # noqa: BLE001
                unavailable.append(f"zstd_cpu unavailable: {exc}")
        elif backend in {"lz4", "lz4_cpu"}:
            try:
                codecs.append(Lz4Codec(compression_level=args.lz4_level))
            except Exception as exc:  # noqa: BLE001
                unavailable.append(f"lz4_cpu unavailable: {exc}")
        elif backend in catalog:
            if nvcomp_info["available"]:
                nvcomp_backends.append(catalog[backend])
            else:
                unavailable.append(f"{backend} unavailable: {nvcomp_info['note']}")
        elif backend.startswith("nvcomp"):
            unavailable.append(f"{backend} unavailable: unknown nvCOMP backend")
        else:
            unavailable.append(f"{backend} unavailable: unknown backend")

    return codecs, nvcomp_backends, unavailable


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_object(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt"}:
        return torch_load(path)
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        return np.load(path, allow_pickle=False)
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"safetensors is not importable: {exc}") from exc
        return load_file(str(path), device="cpu")
    if suffix == ".bin":
        raise RuntimeError("raw .bin files do not carry tensor dtype/shape metadata; listed but skipped")
    raise RuntimeError(f"unsupported extension {suffix}")


def iter_leaf_tensors(obj: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(obj, torch.Tensor):
        yield prefix or "tensor", obj
        return
    if isinstance(obj, np.ndarray):
        yield prefix or "array", obj
        return
    if isinstance(obj, np.lib.npyio.NpzFile):
        for key in sorted(obj.files, key=natural_text_key):
            yield key, obj[key]
        return
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=lambda item: natural_text_key(str(item))):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_leaf_tensors(obj[key], child_prefix)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from iter_leaf_tensors(value, child_prefix)


def natural_text_key(text: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def infer_kv_type(tensor_name: str) -> str:
    normalized = tensor_name.lower().replace("-", "_")
    parts = [part for part in re.split(r"[^a-z0-9_]+", normalized) if part]
    last = parts[-1] if parts else normalized
    if "key_pre_rope" in normalized or "pre_rope_key" in normalized:
        return "pre_rope_k"
    if last in {"k", "key"} or last.startswith("k_") or last.endswith("_key"):
        return "k"
    if last in {"v", "value"} or last.startswith("v_") or last.endswith("_value"):
        return "v"
    if "key" in parts:
        return "k"
    if "value" in parts:
        return "v"
    return "unknown"


def metadata_int(obj: Any, key: str) -> int | None:
    if isinstance(obj, dict):
        value = obj.get(key)
        if isinstance(value, (int, np.integer)):
            return int(value)
    return None


def parse_layer_from_path(path: Path) -> str:
    match = re.search(r"layer_(\d+)", path.name)
    if match:
        return str(int(match.group(1)))
    return ""


def parse_chunk_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        match = re.search(r"chunk_(\d+)(?:_(\d+)_(\d+))?", part)
        if match:
            return int(match.group(1))
    return None


def parse_token_span_from_path(path: Path) -> tuple[int | None, int | None]:
    for part in reversed(path.parts):
        match = re.search(r"chunk_\d+_(\d+)_(\d+)", part)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def as_torch_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == np.float16:
            return torch.from_numpy(np.ascontiguousarray(value))
    return None


def make_refs_for_file(
    path: Path,
    root: Path,
    obj: Any,
    block_mode: str,
    tile_tokens: int = 64,
) -> list[TensorBlockRef]:
    rel_file = str(path.relative_to(root)) if root.is_dir() else path.name
    token_start = metadata_int(obj, "token_start")
    token_end = metadata_int(obj, "token_end")
    if token_start is None or token_end is None:
        parsed_start, parsed_end = parse_token_span_from_path(path)
        token_start = token_start if token_start is not None else parsed_start
        token_end = token_end if token_end is not None else parsed_end
    chunk_i = metadata_int(obj, "chunk_i")
    if chunk_i is None:
        chunk_i = parse_chunk_from_path(path)
    layer = metadata_int(obj, "layer_idx")
    layer_str = str(layer) if layer is not None else parse_layer_from_path(path)
    block_index = int(chunk_i if chunk_i is not None else 0)

    refs: list[TensorBlockRef] = []
    for tensor_name, value in iter_leaf_tensors(obj):
        tensor = as_torch_tensor(value)
        if tensor is None:
            continue
        if tensor.dtype not in SUPPORTED_TORCH_DTYPES:
            continue
        kv_type = infer_kv_type(tensor_name)
        if kv_type == "unknown" and tensor_name not in {"tensor", "array"}:
            continue
        shape = tuple(int(dim) for dim in tensor.shape)
        if block_mode == "token_tile":
            if len(shape) < 4 or shape[1] <= 0 or shape[2] <= 0:
                continue
            if tile_tokens <= 0:
                raise ValueError("--tile_tokens must be positive for --block_mode token_tile")
            for head_idx in range(shape[1]):
                for local_start in range(0, shape[2], tile_tokens):
                    local_end = min(local_start + tile_tokens, shape[2])
                    head_name = f"{tensor_name}#head_{head_idx:02d}#tok_{local_start:06d}_{local_end:06d}"
                    head_shape = (shape[0], 1, local_end - local_start, *shape[3:])
                    abs_start = token_start + local_start if token_start is not None else local_start
                    abs_end = token_start + local_end if token_start is not None else local_end
                    refs.append(
                        TensorBlockRef(
                            file_path=path,
                            rel_file=rel_file,
                            tensor_name=head_name,
                            kv_type=kv_type,
                            layer=layer_str,
                            head=str(head_idx),
                            block_index=block_index,
                            dtype=str(tensor.dtype).replace("torch.", ""),
                            shape=head_shape,
                            token_start=abs_start,
                            token_end=abs_end,
                            chunk_i=chunk_i,
                            source_format=path.suffix.lower(),
                        )
                    )
        elif block_mode == "head" and len(shape) >= 4 and shape[1] > 1:
            for head_idx in range(shape[1]):
                head_name = f"{tensor_name}#head_{head_idx:02d}"
                head_shape = (shape[0], 1, *shape[2:])
                refs.append(
                    TensorBlockRef(
                        file_path=path,
                        rel_file=rel_file,
                        tensor_name=head_name,
                        kv_type=kv_type,
                        layer=layer_str,
                        head=str(head_idx),
                        block_index=block_index,
                        dtype=str(tensor.dtype).replace("torch.", ""),
                        shape=head_shape,
                        token_start=token_start,
                        token_end=token_end,
                        chunk_i=chunk_i,
                        source_format=path.suffix.lower(),
                    )
                )
        else:
            refs.append(
                TensorBlockRef(
                    file_path=path,
                    rel_file=rel_file,
                    tensor_name=tensor_name,
                    kv_type=kv_type,
                    layer=layer_str,
                    head="",
                    block_index=block_index,
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    shape=shape,
                    token_start=token_start,
                    token_end=token_end,
                    chunk_i=chunk_i,
                    source_format=path.suffix.lower(),
                )
            )
    return refs


def load_tensor_for_ref(ref: TensorBlockRef) -> torch.Tensor:
    obj = load_object(ref.file_path)
    tensor_name = ref.tensor_name
    head_match = re.search(r"#head_(\d+)", tensor_name)
    token_match = re.search(r"#tok_(\d+)_(\d+)", tensor_name)
    first_marker = len(tensor_name)
    for marker in ("#head_", "#tok_"):
        marker_pos = tensor_name.find(marker)
        if marker_pos >= 0:
            first_marker = min(first_marker, marker_pos)
    base_name = tensor_name[:first_marker]
    for name, value in iter_leaf_tensors(obj):
        if name != base_name:
            continue
        tensor = as_torch_tensor(value)
        if tensor is None:
            break
        if head_match:
            head_idx = int(head_match.group(1))
            tensor = tensor[:, head_idx : head_idx + 1, ...]
        if token_match:
            token_start = int(token_match.group(1))
            token_end = int(token_match.group(2))
            tensor = tensor[:, :, token_start:token_end, ...]
        return tensor
    raise KeyError(f"tensor {ref.tensor_name} not found in {ref.file_path}")


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype not in SUPPORTED_TORCH_DTYPES:
        raise TypeError(f"expected FP16/BF16 tensor, got {tensor.dtype}")
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def tensor_to_cuda_bytes(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in SUPPORTED_TORCH_DTYPES:
        raise TypeError(f"expected FP16/BF16 tensor, got {tensor.dtype}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return tensor.detach().contiguous().view(torch.uint8).reshape(-1).cuda(non_blocking=True)


def count_mismatched_u16(expected: bytes, actual: bytes) -> int:
    if len(expected) != len(actual):
        return max(len(expected), len(actual))
    even_len = len(expected) - (len(expected) % 2)
    if even_len == 0:
        return 0 if expected == actual else 1
    expected_u16 = np.frombuffer(expected[:even_len], dtype=np.uint16)
    actual_u16 = np.frombuffer(actual[:even_len], dtype=np.uint16)
    mismatches = int(np.count_nonzero(expected_u16 != actual_u16))
    if len(expected) % 2 and expected[-1:] != actual[-1:]:
        mismatches += 1
    return mismatches


def save_debug_record(
    out_dir: Path,
    ref: TensorBlockRef,
    method: str,
    backend: str,
    mismatch_count: int,
    note: str,
) -> None:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{ref.rel_file}_{ref.tensor_name}_{method}_{backend}")
    payload = {
        "file": ref.rel_file,
        "tensor_name": ref.tensor_name,
        "method": method,
        "backend": backend,
        "mismatch_count": mismatch_count,
        "note": note,
    }
    (out_dir / f"debug_mismatch_{safe_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def timed_codec_roundtrip(codec: Codec, payload: bytes, warmup: int, repeat: int) -> tuple[bytes, float, float]:
    compressed = b""
    recovered = b""
    for _ in range(max(0, warmup)):
        compressed = codec.compress(payload)
        recovered = codec.decompress(compressed)
    compress_times: list[float] = []
    decompress_times: list[float] = []
    iterations = max(1, repeat)
    for _ in range(iterations):
        start = time.perf_counter()
        compressed = codec.compress(payload)
        after_compress = time.perf_counter()
        recovered = codec.decompress(compressed)
        after_decompress = time.perf_counter()
        compress_times.append((after_compress - start) * 1000.0)
        decompress_times.append((after_decompress - after_compress) * 1000.0)
    return recovered, float(np.mean(compress_times)), float(np.mean(decompress_times))


def make_result_row(
    ref: TensorBlockRef,
    method: str,
    backend: str,
    raw_bytes: int,
    compressed_bytes: int,
    compress_ms: float,
    decompress_ms: float,
    lossless_pass: bool,
    notes: str,
) -> dict[str, Any]:
    ratio = raw_bytes / compressed_bytes if lossless_pass and compressed_bytes > 0 else ""
    compress_gbs = raw_bytes / (compress_ms / 1000.0) / 1e9 if lossless_pass and compress_ms > 0 else ""
    decompress_gbs = raw_bytes / (decompress_ms / 1000.0) / 1e9 if lossless_pass and decompress_ms > 0 else ""
    return {
        "file": ref.rel_file,
        "tensor_name": ref.tensor_name,
        "kv_type": ref.kv_type,
        "layer": ref.layer,
        "head": ref.head,
        "block_index": ref.block_index,
        "method": method,
        "backend": backend,
        "dtype": ref.dtype,
        "shape": "x".join(str(dim) for dim in ref.shape),
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes if compressed_bytes >= 0 else "",
        "compression_ratio": f"{ratio:.6f}" if isinstance(ratio, float) else ratio,
        "compress_ms": f"{compress_ms:.6f}" if math.isfinite(compress_ms) else "",
        "decompress_ms": f"{decompress_ms:.6f}" if math.isfinite(decompress_ms) else "",
        "compress_gbs": f"{compress_gbs:.6f}" if isinstance(compress_gbs, float) else compress_gbs,
        "decompress_gbs": f"{decompress_gbs:.6f}" if isinstance(decompress_gbs, float) else decompress_gbs,
        "lossless_pass": str(bool(lossless_pass)),
        "notes": notes,
    }


def benchmark_raw(
    ref: TensorBlockRef,
    raw: bytes,
    codec: Codec,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    recovered, compress_ms, decompress_ms = timed_codec_roundtrip(codec, raw, args.warmup, args.repeat)
    compressed = codec.compress(raw)
    lossless = recovered == raw
    notes = "bit-exact byte equality"
    if not lossless:
        mismatches = count_mismatched_u16(raw, recovered)
        notes = f"FAILED byte equality; mismatched_uint16={mismatches}"
        save_debug_record(out_dir, ref, "raw", codec.name, mismatches, notes)
    return make_result_row(
        ref,
        method="raw_bytes",
        backend=codec.name,
        raw_bytes=len(raw),
        compressed_bytes=len(compressed),
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        lossless_pass=lossless,
        notes=notes,
    )


def benchmark_temporal_xor(
    ref: TensorBlockRef,
    raw: bytes,
    prev_raw: bytes,
    codec: Codec,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any] | None:
    if len(raw) != len(prev_raw):
        return None
    if len(raw) % 2 != 0:
        return None
    current_u16 = np.frombuffer(raw, dtype=np.uint16)
    previous_u16 = np.frombuffer(prev_raw, dtype=np.uint16)
    residual = np.bitwise_xor(current_u16, previous_u16).astype(np.uint16, copy=False)
    residual_bytes = residual.tobytes()
    recovered_residual_bytes, compress_ms, decompress_ms = timed_codec_roundtrip(
        codec,
        residual_bytes,
        args.warmup,
        args.repeat,
    )
    compressed = codec.compress(residual_bytes)

    residual_recovered = np.frombuffer(recovered_residual_bytes, dtype=np.uint16)
    reconstructed = np.bitwise_xor(residual_recovered, previous_u16).astype(np.uint16, copy=False)
    reconstructed_bytes = reconstructed.tobytes()
    lossless = reconstructed_bytes == raw
    notes = "residual_recovered XOR previous equals current bytes"
    if not lossless:
        mismatches = count_mismatched_u16(raw, reconstructed_bytes)
        notes = f"FAILED temporal XOR reconstruction; mismatched_uint16={mismatches}"
        save_debug_record(out_dir, ref, "temporal_xor", codec.name, mismatches, notes)
    return make_result_row(
        ref,
        method="temporal_xor",
        backend=codec.name,
        raw_bytes=len(raw),
        compressed_bytes=len(compressed),
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        lossless_pass=lossless,
        notes=notes,
    )


def make_nvcomp_codec(backend: NvcompBackend) -> Any:
    from nvidia import nvcomp

    kwargs = backend.codec_kwargs()
    kwargs["cuda_stream"] = torch.cuda.current_stream().cuda_stream
    return nvcomp.Codec(**kwargs)


def timed_nvcomp_roundtrip(
    backend: NvcompBackend,
    payload_cuda: torch.Tensor,
    warmup: int,
    repeat: int,
) -> tuple[Any, int, float, float]:
    from nvidia import nvcomp

    if payload_cuda.dtype != torch.uint8 or payload_cuda.device.type != "cuda":
        raise TypeError("payload_cuda must be a CUDA uint8 tensor")
    payload_cuda = payload_cuda.contiguous().reshape(-1)
    codec = make_nvcomp_codec(backend)
    array = nvcomp.as_array(payload_cuda)
    decoded = None
    encoded = None

    for _ in range(max(0, warmup)):
        encoded = codec.encode(array)
        decoded = codec.decode(encoded, data_type="|u1")
    torch.cuda.synchronize()

    compress_times: list[float] = []
    decompress_times: list[float] = []
    stream = torch.cuda.current_stream()
    iterations = max(1, repeat)
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        encoded = codec.encode(array)
        end.record(stream)
        end.synchronize()
        compress_times.append(start.elapsed_time(end))

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        decoded = codec.decode(encoded, data_type="|u1")
        end.record(stream)
        end.synchronize()
        decompress_times.append(start.elapsed_time(end))

    if encoded is None or decoded is None:
        raise RuntimeError(f"{backend.name} did not produce an encoded/decoded payload")
    return decoded, int(encoded.buffer_size), float(np.mean(compress_times)), float(np.mean(decompress_times))


def cuda_bytes_to_host_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def benchmark_raw_nvcomp(
    ref: TensorBlockRef,
    raw_cuda: torch.Tensor,
    backend: NvcompBackend,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    decoded_array, compressed_bytes, compress_ms, decompress_ms = timed_nvcomp_roundtrip(
        backend,
        raw_cuda,
        args.warmup,
        args.repeat,
    )
    recovered = torch.from_dlpack(decoded_array).reshape(-1)
    lossless = torch.equal(raw_cuda.reshape(-1), recovered)
    notes = "CUDA event timing; tensor resident on GPU; bit-exact byte equality"
    if not lossless:
        expected = cuda_bytes_to_host_bytes(raw_cuda.reshape(-1))
        actual = cuda_bytes_to_host_bytes(recovered)
        mismatches = count_mismatched_u16(expected, actual)
        notes = f"FAILED CUDA byte equality; mismatched_uint16={mismatches}"
        save_debug_record(out_dir, ref, "raw", backend.name, mismatches, notes)
    return make_result_row(
        ref,
        method="raw_bytes",
        backend=backend.name,
        raw_bytes=int(raw_cuda.numel()),
        compressed_bytes=compressed_bytes,
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        lossless_pass=lossless,
        notes=notes,
    )


def benchmark_temporal_xor_nvcomp(
    ref: TensorBlockRef,
    raw_cuda: torch.Tensor,
    prev_raw_cuda: torch.Tensor,
    backend: NvcompBackend,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any] | None:
    if raw_cuda.numel() != prev_raw_cuda.numel():
        return None
    residual = torch.bitwise_xor(raw_cuda.reshape(-1), prev_raw_cuda.reshape(-1))
    decoded_array, compressed_bytes, compress_ms, decompress_ms = timed_nvcomp_roundtrip(
        backend,
        residual,
        args.warmup,
        args.repeat,
    )
    recovered_residual = torch.from_dlpack(decoded_array).reshape(-1)
    reconstructed = torch.bitwise_xor(recovered_residual, prev_raw_cuda.reshape(-1))
    lossless = torch.equal(raw_cuda.reshape(-1), reconstructed)
    notes = "CUDA event timing; bytewise XOR residual resident on GPU; decoded residual XOR previous equals current bytes"
    if not lossless:
        expected = cuda_bytes_to_host_bytes(raw_cuda.reshape(-1))
        actual = cuda_bytes_to_host_bytes(reconstructed)
        mismatches = count_mismatched_u16(expected, actual)
        notes = f"FAILED CUDA temporal XOR reconstruction; mismatched_uint16={mismatches}"
        save_debug_record(out_dir, ref, "temporal_xor", backend.name, mismatches, notes)
    return make_result_row(
        ref,
        method="temporal_xor",
        backend=backend.name,
        raw_bytes=int(raw_cuda.numel()),
        compressed_bytes=compressed_bytes,
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        lossless_pass=lossless,
        notes=notes,
    )


def environment_info(nvcomp_info: dict[str, Any], unavailable_backends: list[str]) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_names = []
    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(idx))
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "gpu_names": gpu_names,
        "nvcomp": nvcomp_info,
        "unavailable_backends": unavailable_backends,
    }


def aggregate_rows(rows: list[dict[str, str]], kv_filter: set[str] | None = None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        if row.get("lossless_pass") != "True":
            continue
        kv_type = row.get("kv_type", "unknown")
        if kv_filter is not None and kv_type not in kv_filter:
            continue
        try:
            raw = float(row["raw_bytes"])
            compressed = float(row["compressed_bytes"])
            compress_ms = float(row["compress_ms"])
            decompress_ms = float(row["decompress_ms"])
        except (TypeError, ValueError, KeyError):
            continue
        key = (kv_type, row.get("method", ""), row.get("backend", ""))
        group = groups.setdefault(
            key,
            {
                "raw_bytes": 0.0,
                "compressed_bytes": 0.0,
                "compress_ms": 0.0,
                "decompress_ms": 0.0,
                "blocks": 0.0,
            },
        )
        group["raw_bytes"] += raw
        group["compressed_bytes"] += compressed
        group["compress_ms"] += compress_ms
        group["decompress_ms"] += decompress_ms
        group["blocks"] += 1.0

    result = []
    for (kv_type, method, backend), values in groups.items():
        raw = values["raw_bytes"]
        compressed = values["compressed_bytes"]
        ratio = raw / compressed if compressed > 0 else float("nan")
        compress_gbs = raw / (values["compress_ms"] / 1000.0) / 1e9 if values["compress_ms"] > 0 else float("nan")
        decompress_gbs = raw / (values["decompress_ms"] / 1000.0) / 1e9 if values["decompress_ms"] > 0 else float("nan")
        result.append(
            {
                "kv_type": kv_type,
                "method": method,
                "backend": backend,
                "blocks": int(values["blocks"]),
                "raw_bytes": int(raw),
                "compressed_bytes": int(compressed),
                "compression_ratio": ratio,
                "compress_gbs": compress_gbs,
                "decompress_gbs": decompress_gbs,
            }
        )
    return sorted(result, key=lambda item: (item["kv_type"], -item["compression_ratio"], item["method"], item["backend"]))


def aggregate_combined_kv_rows(
    rows: list[dict[str, str]],
    kv_filter: set[str],
    kv_type_name: str,
    label: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, float]] = {}
    present_kv_types: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row.get("lossless_pass") != "True":
            continue
        kv_type = row.get("kv_type", "unknown")
        if kv_type not in kv_filter:
            continue
        try:
            raw = float(row["raw_bytes"])
            compressed = float(row["compressed_bytes"])
            compress_ms = float(row["compress_ms"])
            decompress_ms = float(row["decompress_ms"])
        except (TypeError, ValueError, KeyError):
            continue
        key = (row.get("method", ""), row.get("backend", ""))
        group = groups.setdefault(
            key,
            {
                "raw_bytes": 0.0,
                "compressed_bytes": 0.0,
                "compress_ms": 0.0,
                "decompress_ms": 0.0,
                "blocks": 0.0,
            },
        )
        group["raw_bytes"] += raw
        group["compressed_bytes"] += compressed
        group["compress_ms"] += compress_ms
        group["decompress_ms"] += decompress_ms
        group["blocks"] += 1.0
        present_kv_types.setdefault(key, set()).add(kv_type)

    result = []
    for (method, backend), values in groups.items():
        if not kv_filter.issubset(present_kv_types.get((method, backend), set())):
            continue
        raw = values["raw_bytes"]
        compressed = values["compressed_bytes"]
        ratio = raw / compressed if compressed > 0 else float("nan")
        compress_gbs = raw / (values["compress_ms"] / 1000.0) / 1e9 if values["compress_ms"] > 0 else float("nan")
        decompress_gbs = raw / (values["decompress_ms"] / 1000.0) / 1e9 if values["decompress_ms"] > 0 else float("nan")
        item = {
            "kv_type": kv_type_name,
            "method": method,
            "backend": backend,
            "blocks": int(values["blocks"]),
            "raw_bytes": int(raw),
            "compressed_bytes": int(compressed),
            "compression_ratio": ratio,
            "compress_gbs": compress_gbs,
            "decompress_gbs": decompress_gbs,
        }
        if label is not None:
            item["label"] = label
        result.append(item)
    return sorted(result, key=lambda item: (-item["compression_ratio"], item["method"], item["backend"]))


def best_by_ratio(aggregates: list[dict[str, Any]], kv_type: str) -> dict[str, Any] | None:
    matching = [item for item in aggregates if item["kv_type"] == kv_type]
    if not matching:
        return None
    return max(matching, key=lambda item: item["compression_ratio"])


def best_by_speed(aggregates: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    valid = [item for item in aggregates if math.isfinite(float(item[field]))]
    if not valid:
        return None
    return max(valid, key=lambda item: item[field])


def format_aggregate_row(item: dict[str, Any]) -> str:
    return (
        f"| {item['kv_type']} | {item['method']} | {item['backend']} | {item['blocks']} | "
        f"{item['compression_ratio']:.4f} | {item['compress_gbs']:.3f} | {item['decompress_gbs']:.3f} |"
    )


def write_summary(
    out_dir: Path,
    args: argparse.Namespace,
    env: dict[str, Any],
    rows: list[dict[str, str]],
    candidates: list[Path],
    skipped_files: list[str],
    unavailable_backends: list[str],
) -> None:
    aggregates = aggregate_rows(rows)
    k_best = best_by_ratio(aggregates, "k")
    v_best = best_by_ratio(aggregates, "v")
    pre_best = best_by_ratio(aggregates, "pre_rope_k")
    kv_combined = aggregate_combined_kv_rows(rows, {"k", "v"}, "k+v")
    kv_combined_best = max(kv_combined, key=lambda item: item["compression_ratio"], default=None)
    fastest_compress = best_by_speed(aggregates, "compress_gbs")
    fastest_decompress = best_by_speed(aggregates, "decompress_gbs")
    failures = [row for row in rows if row.get("lossless_pass") != "True"]

    parts = [
        "# Lossless KV nvCOMP Benchmark Summary",
        "",
        f"- Label: {args.label or args.kv_path.name}",
        f"- KV path: `{args.kv_path}`",
        f"- Candidate files discovered: {len(candidates)}",
        f"- Result rows: {len(rows)}",
        f"- Block mode: `{args.block_mode}`",
        f"- Warmup/repeat: {args.warmup}/{args.repeat}",
        f"- Max blocks: {args.max_blocks if args.max_blocks is not None else 'all'}",
        "",
        "## Environment",
        "",
        f"- Python: `{env['python']}`",
        f"- Platform: `{env['platform']}`",
        f"- PyTorch: `{env['torch']}`",
        f"- CUDA available: `{env['cuda_available']}`",
        f"- Torch CUDA: `{env['torch_cuda']}`",
        f"- GPUs: `{', '.join(env['gpu_names']) if env['gpu_names'] else 'none detected'}`",
        f"- nvCOMP available: `{env['nvcomp']['available']}`",
        f"- nvCOMP note: {env['nvcomp']['note']}",
        "",
        "## Backend Availability",
        "",
    ]
    if unavailable_backends:
        parts.extend(f"- {item}" for item in unavailable_backends)
    else:
        parts.append("- No requested backend was unavailable.")
    parts.extend(
        [
            "",
            "nvCOMP algorithm status:",
        ]
    )
    parts.extend(f"- {name}: {status}" for name, status in env["nvcomp"]["algorithms"].items())

    parts.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| kv_type | method | backend | blocks | ratio | comp GB/s | decomp GB/s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    if aggregates:
        parts.extend(format_aggregate_row(item) for item in aggregates)
    else:
        parts.append("| none | none | none | 0 | n/a | n/a | n/a |")

    def best_line(label: str, item: dict[str, Any] | None) -> str:
        if item is None:
            return f"- {label}: not present"
        return (
            f"- {label}: {item['compression_ratio']:.4f}x via "
            f"{item['method']} / {item['backend']} over {item['blocks']} blocks"
        )

    parts.extend(
        [
            "",
            "## Key Points",
            "",
            best_line("Best K compression ratio", k_best),
            best_line("Best V compression ratio", v_best),
            best_line("Best pre-RoPE K compression ratio", pre_best),
            best_line("Best overall K+V compression ratio", kv_combined_best),
            (
                f"- Fastest compression backend: {fastest_compress['backend']} "
                f"({fastest_compress['compress_gbs']:.3f} GB/s, {fastest_compress['method']}, {fastest_compress['kv_type']})"
                if fastest_compress is not None
                else "- Fastest compression backend: not present"
            ),
            (
                f"- Fastest decompression backend: {fastest_decompress['backend']} "
                f"({fastest_decompress['decompress_gbs']:.3f} GB/s, {fastest_decompress['method']}, {fastest_decompress['kv_type']})"
                if fastest_decompress is not None
                else "- Fastest decompression backend: not present"
            ),
            f"- Every tested row was bit-exact lossless: `{not failures}`",
        ]
    )
    if skipped_files:
        parts.extend(["", "## Skipped Files", ""])
        parts.extend(f"- {item}" for item in skipped_files[:200])
        if len(skipped_files) > 200:
            parts.append(f"- ... {len(skipped_files) - 200} more")

    (out_dir / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_csv(out_dir: Path, rows: list[dict[str, str]]) -> Path:
    csv_path = out_dir / "lossless_kv_nvcomp_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def benchmark(args: argparse.Namespace) -> None:
    if args.kv_path is None or args.out_dir is None:
        raise SystemExit("--kv_path and --out_dir are required in benchmark mode")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = scan_candidate_files(args.kv_path)
    print_candidates(candidates)
    (args.out_dir / "candidate_files.txt").write_text(
        "\n".join(str(path) for path in candidates) + ("\n" if candidates else ""),
        encoding="utf-8",
    )

    nvcomp_info = detect_nvcomp()
    codecs, nvcomp_backends, unavailable_backends = available_codecs(args, nvcomp_info)
    if not nvcomp_info["available"]:
        print(f"WARNING: nvCOMP is unavailable. {nvcomp_info['note']}", flush=True)
    if not codecs and not nvcomp_backends:
        raise SystemExit("No compression backends are available.")
    backend_names = [codec.name for codec in codecs] + [backend.name for backend in nvcomp_backends]
    print("Testing backends: " + ", ".join(backend_names), flush=True)

    rows: list[dict[str, str]] = []
    skipped_files: list[str] = []
    previous_by_stream: dict[tuple[str, str, str, str, tuple[int, ...]], TensorBlockRef] = {}
    tested_blocks = 0

    for path in candidates:
        if args.max_blocks is not None and tested_blocks >= args.max_blocks:
            break
        try:
            obj = load_object(path)
            refs = make_refs_for_file(path, args.kv_path, obj, args.block_mode, args.tile_tokens)
        except Exception as exc:  # noqa: BLE001
            skipped_files.append(f"{path}: {exc}")
            print(f"SKIP {path}: {exc}", flush=True)
            continue
        if not refs:
            skipped_files.append(f"{path}: no supported FP16/BF16 KV tensors found")
            print(f"SKIP {path}: no supported FP16/BF16 KV tensors found", flush=True)
            continue

        for ref in refs:
            if args.max_blocks is not None and tested_blocks >= args.max_blocks:
                break
            tensor = load_tensor_for_ref(ref)
            raw_nbytes = tensor_nbytes(tensor)
            raw = tensor_to_bytes(tensor) if codecs else None
            raw_cuda = tensor_to_cuda_bytes(tensor) if nvcomp_backends else None
            print(
                "Tensor "
                f"{ref.rel_file}::{ref.tensor_name} "
                f"kv_type={ref.kv_type} layer={ref.layer or 'n/a'} "
                f"head={ref.head or 'all'} dtype={ref.dtype} "
                f"shape={list(ref.shape)} raw_bytes={raw_nbytes}",
                flush=True,
            )
            prev_ref = previous_by_stream.get(ref.stream_key())
            prev_raw = None
            prev_raw_cuda = None
            if prev_ref is not None:
                try:
                    prev_tensor = load_tensor_for_ref(prev_ref)
                    prev_raw = tensor_to_bytes(prev_tensor) if codecs else None
                    prev_raw_cuda = tensor_to_cuda_bytes(prev_tensor) if nvcomp_backends else None
                except Exception as exc:  # noqa: BLE001
                    print(f"Temporal XOR skip for {ref.rel_file}::{ref.tensor_name}: {exc}", flush=True)

            for codec in codecs:
                if raw is None:
                    continue
                rows.append(benchmark_raw(ref, raw, codec, args, args.out_dir))
                if prev_raw is not None:
                    temporal_row = benchmark_temporal_xor(ref, raw, prev_raw, codec, args, args.out_dir)
                    if temporal_row is not None:
                        rows.append(temporal_row)

            for backend in nvcomp_backends:
                if raw_cuda is None:
                    continue
                rows.append(benchmark_raw_nvcomp(ref, raw_cuda, backend, args, args.out_dir))
                if prev_raw_cuda is not None:
                    temporal_row = benchmark_temporal_xor_nvcomp(ref, raw_cuda, prev_raw_cuda, backend, args, args.out_dir)
                    if temporal_row is not None:
                        rows.append(temporal_row)

            previous_by_stream[ref.stream_key()] = ref
            tested_blocks += 1

    csv_path = write_csv(args.out_dir, rows)
    env = environment_info(nvcomp_info, unavailable_backends)
    (args.out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    write_summary(args.out_dir, args, env, rows, candidates, skipped_files, unavailable_backends)
    print(f"Wrote {csv_path}", flush=True)
    print(f"Wrote {args.out_dir / 'summary.md'}", flush=True)


def read_result_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_combine_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"expected LABEL=CSV, got {value}")
    label, csv_path = value.split("=", 1)
    if not label:
        raise ValueError(f"empty label in {value}")
    return label, Path(csv_path)


def aggregate_for_label(label: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    aggregates = aggregate_rows(rows)
    for item in aggregates:
        item["label"] = label
    return aggregates


def write_combined_summary(args: argparse.Namespace) -> None:
    if not args.combine_result:
        raise SystemExit("--combine_result is required in combine mode")
    if args.combined_out is None:
        raise SystemExit("--combined_out is required in combine mode")
    args.combined_out.parent.mkdir(parents=True, exist_ok=True)

    all_aggregates: list[dict[str, Any]] = []
    all_rows_by_label: dict[str, list[dict[str, str]]] = {}
    for item in args.combine_result:
        label, csv_path = parse_combine_arg(item)
        rows = read_result_csv(csv_path)
        all_rows_by_label[label] = rows
        all_aggregates.extend(aggregate_for_label(label, rows))
        all_aggregates.extend(aggregate_combined_kv_rows(rows, {"k", "v"}, "k+v", label=label))

    post_candidates = [
        item
        for item in all_aggregates
        if item["kv_type"] == "k" and "pre" not in item["label"].lower()
    ]
    pre_candidates = [item for item in all_aggregates if item["kv_type"] == "pre_rope_k"]
    regular_v_candidates = [item for item in all_aggregates if item["kv_type"] == "v"]
    regular_kv_candidates = [
        item
        for item in all_aggregates
        if item["kv_type"] == "k+v" and "pre" not in item["label"].lower()
    ]
    post_best = max(post_candidates, key=lambda item: item["compression_ratio"], default=None)
    pre_best = max(pre_candidates, key=lambda item: item["compression_ratio"], default=None)
    v_best = max(regular_v_candidates, key=lambda item: item["compression_ratio"], default=None)
    kv_best = max(regular_kv_candidates, key=lambda item: item["compression_ratio"], default=None)

    all_tested_rows_lossless = True
    failed_rows = 0
    total_rows = 0
    for rows in all_rows_by_label.values():
        for row in rows:
            total_rows += 1
            if row.get("lossless_pass") != "True":
                all_tested_rows_lossless = False
                failed_rows += 1

    parts = [
        "# Combined Lossless KV Compression Comparison",
        "",
        f"- Result files compared: {len(args.combine_result)}",
        f"- Total result rows: {total_rows}",
        f"- Failed/non-lossless rows: {failed_rows}",
        f"- Every tested row passed bit-exact lossless checks: `{all_tested_rows_lossless}`",
        "- Every row marked successful was admitted only after byte-level equality after decompression/reconstruction.",
        "",
        "## Aggregate Table",
        "",
        "| label | kv_type | method | backend | blocks | ratio | comp GB/s | decomp GB/s |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(
        all_aggregates,
        key=lambda value: (value["label"], value["kv_type"], -value["compression_ratio"], value["method"], value["backend"]),
    ):
        parts.append(
            f"| {item['label']} | {item['kv_type']} | {item['method']} | {item['backend']} | "
            f"{item['blocks']} | {item['compression_ratio']:.4f} | "
            f"{item['compress_gbs']:.3f} | {item['decompress_gbs']:.3f} |"
        )

    def describe_best(title: str, item: dict[str, Any] | None) -> str:
        if item is None:
            return f"- {title}: not present"
        return (
            f"- {title}: {item['compression_ratio']:.4f}x, "
            f"{item['compress_gbs']:.3f} GB/s compress, {item['decompress_gbs']:.3f} GB/s decompress "
            f"({item['label']}, {item['method']}, {item['backend']}, {item['blocks']} blocks)"
        )

    if post_best is not None and pre_best is not None:
        more_compressible = pre_best["compression_ratio"] > post_best["compression_ratio"]
        compressibility_line = (
            f"- Pre-RoPE K is more compressible than post-RoPE K: `{more_compressible}` "
            f"(pre {pre_best['compression_ratio']:.4f}x vs post {post_best['compression_ratio']:.4f}x)."
        )
    else:
        compressibility_line = "- Pre-RoPE K vs post-RoPE K compressibility: not enough data."

    parts.extend(
        [
            "",
            "## Comparison",
            "",
            describe_best("Regular BF16 post-RoPE K+V best ratio/speed", kv_best),
            describe_best("Regular BF16 post-RoPE K best ratio/speed", post_best),
            describe_best("Regular BF16 post-RoPE V best ratio/speed", v_best),
            describe_best("Pre-RoPE BF16 K-only best ratio/speed", pre_best),
            compressibility_line,
            (
                "- All rows marked successful used byte-level equality after decompression/reconstruction; "
                "failed rows, if any, are excluded from ratios."
            ),
        ]
    )
    args.combined_out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {args.combined_out}", flush=True)


def main() -> None:
    args = parse_args()
    if args.combine_result:
        write_combined_summary(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
