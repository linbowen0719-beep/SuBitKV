#!/usr/bin/env python3
"""Hypothesis-driven offline evidence runner for Wan 5B no-norm RVQ.

The runner is deliberately separate from generation/model integration.  Its
accuracy path accepts an index-aligned paired pre/post positional-transform
dump, encodes only ``key_pre``/``value_pre`` with the raw-vector L8C64
codebooks, replays the saved standard RoPE, and evaluates the exact tensors
that entered SDPA.  A legacy post-RoPE-only dump may be used for explicitly
labelled, data-independent timing, but never for RVQ accuracy claims.

Each output is self describing: hypotheses are snapshotted before benchmarks,
the physical and logical payload accounting are both recorded, and
``run_info.json`` contains source/config/codebook hashes and machine metadata.
No generation job is launched by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
HY_WORLDPLAY_ROOT = REPO_ROOT / "experiments/HY-WorldPlay"
if str(HY_WORLDPLAY_ROOT) not in sys.path:
    sys.path.insert(0, str(HY_WORLDPLAY_ROOT))

from hyvideo.prope.camera_rope import prope_qkv  # noqa: E402
from subitkv.rvq import (  # noqa: E402
    rvq_dequantize_tensor_cache,
    rvq_encode_tensor_cache,
)


BASELINE = "nonorm_l8c64_int4"
MIB = 1024.0**2
EXACT_STREAMING_RELATIVE_L2_TOLERANCE = 5e-3
DEFAULT_K_CODEBOOK_DIR = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l8c64_codebooks"
)
DEFAULT_V_CODEBOOK_DIR = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l8c64_codebooks"
)
SCALE_KEYS = {
    "scale",
    "scale_q",
    "scale_q_packed",
    "scale_q_shape",
    "scale_min",
    "scale_step",
    "scale_storage_dtype",
    "grouped_scale_shape",
}

HYPOTHESES: dict[str, Any] = {
    "schema_version": 1,
    "registered_before_measurement": True,
    "baseline": BASELINE,
    "constraints": {
        "model": "Wan2.2-TI2V-5B / HY-WorldPlay",
        "input_norm_mode": "none",
        "levels": 8,
        "centroids": 64,
        "resident_codebook_storage": "int4",
        "per_token_scale": False,
        "accuracy_requires": "aligned pre/post dump plus replayable positional transform",
    },
    "H1": {
        "claim": "Exact restore/attention is limited by eight codebook gathers and positional restore.",
        "predictions": [
            "decode+RoPE+SDPA is slower than dense post-RoPE SDPA at sampled K",
            "standalone compressed QK can cross over only at larger K",
            "decode/position restore explains most added latency",
        ],
        "reference_equivalence_gate": {
            "relative_l2_max": EXACT_STREAMING_RELATIVE_L2_TOLERANCE,
            "reference": "explicit L8 decode plus saved token RoPE plus PyTorch SDPA",
        },
        "implementation_scope": (
            "blockwise PyTorch diagnostic reference; not a fused Triton production kernel"
        ),
    },
    "H2": {
        "claim": "Token progressive depth is limited independently by metadata, sensitivity mismatch, and irregular decode.",
        "predictions": [
            "logical 6-bit and physical uint8 accounting produce different break-even depths",
            "fixed L6 decode is faster than variable token depth near average depth seven",
            "raw residual RMS is imperfectly correlated with QK/PV sensitivity",
        ],
    },
    "H3": {
        "claim": "A scalar K correction recovers only the radial part of Q-weighted error.",
        "predictions": [
            "optimal scalar is concentrated near one",
            "scalar recoverable ratio is below diagonal/full-linear upper bounds",
            "lower QK error need not lower final PV error",
        ],
    },
    "H4": {
        "claim": "Codebook reconstruction is more useful for page routing than for replacing all tokens by group means.",
        "predictions": [
            "group/page means have high output error",
            "routing recall and captured mass improve with selection fraction",
            "speedup appears only when saved dense work exceeds route/gather/decode overhead",
        ],
        "feasible_gate": {"output_relative_l2_max": 0.15, "must_beat_dense": True},
    },
    "H5": {
        "claim": "Recent KV supplies local continuity while the first chunk supplies a global anchor.",
        "predictions": [
            "recent beats uniform at matched token count",
            "first+recent beats recent-only when an anchor matters",
            "content/oracle routing bounds the fixed temporal policies",
        ],
    },
    "H6": {
        "claim": "Pre-RoPE routing is useful only if a position-aware approximation preserves post-RoPE page ranking.",
        "predictions": [
            "naive pre-RoPE routing is weaker than a post-RoPE upper bound",
            "page-center RoPE closes part of the recall gap",
            "selected-only RoPE latency falls with selection fraction",
        ],
        "feasible_gate": {"selection_fraction": 0.20, "output_relative_l2_max": 0.15, "must_beat_dense": True},
    },
    "H7": {
        "claim": "At matched average depth, first/recent L8 plus old L4 should beat uniform L6 if temporal sensitivity is real.",
        "predictions": [
            "tiered depth has lower PV error than uniform L6 at equal index rate",
            "raw reconstruction MSE need not improve",
            "coarse tiered decode overhead is at most ten percent",
        ],
        "offline_gate": {"pv_not_worse_than_l6": True, "decode_overhead_max_fraction": 0.10},
    },
}


@dataclass
class Track:
    path: Path
    layer: int
    metadata: dict[str, Any]
    query_pre: torch.Tensor | None
    key_pre: torch.Tensor | None
    value_pre: torch.Tensor | None
    query_post_rope: torch.Tensor
    key_post_rope: torch.Tensor
    value_post_rope: torch.Tensor
    key_rope_cos: torch.Tensor | None
    key_rope_sin: torch.Tensor | None
    query_post_prope: torch.Tensor | None
    key_post_prope: torch.Tensor | None
    value_post_prope: torch.Tensor | None
    query_viewmats: torch.Tensor | None
    query_Ks: torch.Tensor | None
    key_viewmats: torch.Tensor | None
    key_Ks: torch.Tensor | None
    query_indices: torch.Tensor | None
    key_indices: torch.Tensor | None
    paired: bool
    accuracy_valid: bool
    accuracy_reason: str
    camera_prope_present: bool
    camera_prope_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--sample", action="append", type=Path, default=[])
    parser.add_argument("--k-codebook-dir", type=Path, default=DEFAULT_K_CODEBOOK_DIR)
    parser.add_argument("--v-codebook-dir", type=Path, default=DEFAULT_V_CODEBOOK_DIR)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "results/report_evidence_wan5b")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--encode-batch-size", type=int, default=2048)
    parser.add_argument("--query-tokens", type=int, default=64)
    parser.add_argument("--max-key-tokens", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=32)
    parser.add_argument("--chunk-tokens", type=int, default=880)
    parser.add_argument("--fractions", default="0.05,0.10,0.20")
    parser.add_argument("--sections", default="H1,H2,H3,H4,H5,H6,H7")
    parser.add_argument("--progressive-target-depth", type=float, default=7.0)
    parser.add_argument("--exact-reference-max-elements", type=int, default=100_000_000)
    parser.add_argument(
        "--exact-key-block-tokens",
        type=int,
        default=1024,
        help="Contiguous key block size for H1 compressed PyTorch diagnostic references.",
    )
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate source/codebook compatibility without creating results.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_codebook_layer(directory: Path, layer: int, heads: int) -> str:
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
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tensor(payload: dict[str, Any], key: str, *, required: bool = False) -> torch.Tensor | None:
    value = payload.get(key)
    if torch.is_tensor(value):
        return value
    if required:
        raise ValueError(f"Sample is missing tensor {key!r}")
    return None


def load_track(path: Path, device: torch.device, *, q_limit: int, k_limit: int) -> Track:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Sample must be a dict: {path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    layer = int(metadata.get("layer", -1))
    if layer < 0:
        raise ValueError(f"Sample metadata lacks a valid layer: {path}")

    paired_keys = {
        "query_pre",
        "key_pre",
        "value_pre",
        "query_post_rope",
        "key_post_rope",
        "value_post_rope",
    }
    paired = paired_keys.issubset(payload)
    if paired:
        query_pre = _tensor(payload, "query_pre", required=True)
        key_pre = _tensor(payload, "key_pre", required=True)
        value_pre = _tensor(payload, "value_pre", required=True)
        query_post = _tensor(payload, "query_post_rope", required=True)
        key_post = _tensor(payload, "key_post_rope", required=True)
        value_post = _tensor(payload, "value_post_rope", required=True)
        key_cos = _tensor(payload, "key_rope_cos")
        key_sin = _tensor(payload, "key_rope_sin")
        aligned = metadata.get("alignment") == "exact_same_call_same_indices"
        transform_ok = key_cos is not None and key_sin is not None
        accuracy_valid = bool(aligned and transform_ok)
        reason = "valid_standard_rope_replay" if accuracy_valid else (
            "paired dump is not exact-index aligned" if not aligned else "saved standard RoPE cos/sin missing"
        )
        camera_tensor_keys = {
            "query_post_prope",
            "key_post_prope",
            "value_post_prope",
            "query_viewmats",
            "key_viewmats",
        }
        camera_metadata_keys = {"query_Ks", "key_Ks"}
        camera_present = camera_tensor_keys.issubset(payload) and camera_metadata_keys.issubset(payload)
        if camera_present:
            query_post_prope = _tensor(payload, "query_post_prope", required=True)
            key_post_prope = _tensor(payload, "key_post_prope", required=True)
            value_post_prope = _tensor(payload, "value_post_prope", required=True)
            query_viewmats = _tensor(payload, "query_viewmats", required=True)
            query_Ks = _tensor(payload, "query_Ks")
            key_viewmats = _tensor(payload, "key_viewmats", required=True)
            key_Ks = _tensor(payload, "key_Ks")
            camera_reason = (
                "camera_prope_metadata_and_post_tensors_present"
                if aligned
                else "camera tensors are present but paired alignment metadata is invalid"
            )
        else:
            query_post_prope = key_post_prope = value_post_prope = None
            query_viewmats = query_Ks = key_viewmats = key_Ks = None
            missing_camera = sorted((camera_tensor_keys | camera_metadata_keys) - set(payload))
            camera_reason = f"camera replay fields missing: {missing_camera}"
    else:
        query_pre = key_pre = value_pre = None
        query_post = _tensor(payload, "query", required=True)
        key_post = _tensor(payload, "key", required=True)
        value_post = _tensor(payload, "value", required=True)
        key_cos = key_sin = None
        accuracy_valid = False
        query_post_prope = key_post_prope = value_post_prope = None
        query_viewmats = query_Ks = key_viewmats = key_Ks = None
        camera_present = False
        camera_reason = "legacy post-RoPE-only sample has no aligned camera-PRoPE replay"
        reason = (
            "legacy post-RoPE-only sample: raw-vector codebooks cannot validly encode post-RoPE K; "
            "accuracy and data-dependent routing are disabled"
        )

    tensors = [query_post, key_post, value_post]
    if paired:
        tensors.extend([query_pre, key_pre, value_pre])
    assert all(torch.is_tensor(item) and item.ndim == 4 for item in tensors)
    assert query_post is not None and key_post is not None and value_post is not None
    q_take = min(int(query_post.shape[2]), int(q_limit)) if q_limit > 0 else int(query_post.shape[2])
    k_take = min(int(key_post.shape[2]), int(k_limit)) if k_limit > 0 else int(key_post.shape[2])

    def move_slice(value: torch.Tensor | None, tokens: int) -> torch.Tensor | None:
        return None if value is None else value[:, :, :tokens].to(device=device).contiguous()

    def move_token_metadata(value: torch.Tensor | None, tokens: int) -> torch.Tensor | None:
        return None if value is None else value[:, :tokens].to(device=device).contiguous()

    query_indices = _tensor(payload, "query_indices")
    key_indices = _tensor(payload, "key_indices")
    return Track(
        path=path,
        layer=layer,
        metadata=metadata,
        query_pre=move_slice(query_pre, q_take),
        key_pre=move_slice(key_pre, k_take),
        value_pre=move_slice(value_pre, k_take),
        query_post_rope=move_slice(query_post, q_take),  # type: ignore[arg-type]
        key_post_rope=move_slice(key_post, k_take),  # type: ignore[arg-type]
        value_post_rope=move_slice(value_post, k_take),  # type: ignore[arg-type]
        key_rope_cos=move_slice(key_cos, k_take),
        key_rope_sin=move_slice(key_sin, k_take),
        query_post_prope=move_slice(query_post_prope, q_take),
        key_post_prope=move_slice(key_post_prope, k_take),
        value_post_prope=move_slice(value_post_prope, k_take),
        query_viewmats=move_token_metadata(query_viewmats, q_take),
        query_Ks=move_token_metadata(query_Ks, q_take),
        key_viewmats=move_token_metadata(key_viewmats, k_take),
        key_Ks=move_token_metadata(key_Ks, k_take),
        query_indices=None if query_indices is None else query_indices[:q_take].long(),
        key_indices=None if key_indices is None else key_indices[:k_take].long(),
        paired=paired,
        accuracy_valid=accuracy_valid,
        accuracy_reason=reason,
        camera_prope_present=bool(camera_present and aligned),
        camera_prope_reason=camera_reason,
    )


def validate_codebook_dir(
    directory: Path,
    *,
    layer: int,
    heads: int,
    expected_dim: int,
) -> dict[str, Any]:
    modes: set[str] = set()
    shapes: set[tuple[int, ...]] = set()
    for head in range(heads):
        path = directory / f"layer_{layer}_head_{head}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not torch.is_tensor(payload.get("codebooks")):
            raise ValueError(f"Invalid codebook payload: {path}")
        config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
        mode = str(payload.get("input_norm_mode", config.get("input_norm_mode", "rms"))).lower()
        modes.add("none" if mode in {"none", "raw"} else mode)
        shapes.add(tuple(int(x) for x in payload["codebooks"].shape))
    if modes != {"none"}:
        raise ValueError(f"{directory} must contain only input_norm_mode=none, got {sorted(modes)}")
    expected = (8, 64, int(expected_dim))
    if shapes != {expected}:
        raise ValueError(f"{directory} must be L8C64D{expected_dim}, got {sorted(shapes)}")
    return {
        "directory": str(directory),
        "input_norm_mode": "none",
        "shape": list(expected),
        "heads": heads,
        "layer_hash": sha256_codebook_layer(directory, layer, heads),
    }


def assert_no_token_scale(packed: dict[str, Any]) -> None:
    mode = str(packed.get("input_norm_mode", "")).lower()
    if mode not in {"none", "raw"}:
        raise AssertionError(f"Expected packed input_norm_mode=none, got {mode!r}")
    present = sorted(key for key in SCALE_KEYS if key in packed)
    if present:
        raise AssertionError(f"No-norm packed state unexpectedly stores scale fields: {present}")


def prefix_packed(packed: dict[str, Any], levels: int) -> dict[str, Any]:
    levels = int(levels)
    total = int(packed["indices"].shape[0])
    if levels <= 0 or levels > total:
        raise ValueError(f"levels must be in [1,{total}], got {levels}")
    out = {key: value for key, value in packed.items() if key != "indices"}
    out["indices"] = packed["indices"][:levels].contiguous()
    out["num_levels"] = levels
    out.pop("progressive", None)
    return out


def tensor_storage_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_storage_bytes(item) for item in value)
    return 0


def payload_storage(packed: dict[str, Any], *, centroids: int = 64) -> dict[str, Any]:
    assert_no_token_scale(packed)
    indices = packed["indices"]
    prefix_count = int(indices.numel())
    extra = packed.get("extra_indices")
    extra_count = int(extra.numel()) if torch.is_tensor(extra) else 0
    vectors = int(indices.shape[1] * indices.shape[2] * indices.shape[3])
    logical_bits_per_index = int(math.ceil(math.log2(centroids)))
    logical_index_bytes = math.ceil((prefix_count + extra_count) * logical_bits_per_index / 8)
    depth = packed.get("active_depth")
    depth_bytes = tensor_storage_bytes(depth)
    offsets = packed.get("extra_level_offsets")
    offsets_bytes = tensor_storage_bytes(offsets)
    physical_indices = tensor_storage_bytes(indices) + tensor_storage_bytes(extra)
    total_physical = tensor_storage_bytes(packed)
    return {
        "vectors": vectors,
        "logical_index_bits": logical_bits_per_index,
        "logical_index_bytes": logical_index_bytes,
        "depth_metadata_bytes": depth_bytes,
        "offset_metadata_bytes": offsets_bytes,
        "logical_payload_bytes_including_tensor_metadata": logical_index_bytes + depth_bytes + offsets_bytes,
        "physical_index_tensor_bytes": physical_indices,
        "physical_tensor_payload_bytes": total_physical,
        "physical_index_dtype": str(indices.dtype).replace("torch.", ""),
        "per_token_scale_bytes": 0,
    }


def apply_saved_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if x.shape != cos.shape or x.shape != sin.shape:
        try:
            cos = cos.expand_as(x)
            sin = sin.expand_as(x)
        except RuntimeError as exc:
            raise ValueError(
                f"RoPE tensors are not broadcast-compatible: x={tuple(x.shape)}, "
                f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            ) from exc
    pair = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = pair[..., 0], pair[..., 1]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
    out[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]
    return out


def _lift_camera_intrinsics(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(
        (*Ks.shape[:-2], 4, 4), device=Ks.device, dtype=Ks.dtype
    )
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1
    return out


def _invert_camera_intrinsics(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1
    return out


def _invert_camera_se3(viewmats: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(viewmats)
    rotation_inv = viewmats[..., :3, :3].transpose(-1, -2)
    out[..., :3, :3] = rotation_inv
    out[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation_inv, viewmats[..., :3, 3]
    )
    out[..., 3, 3] = 1
    return out


def camera_prope_matrices(
    viewmats: torch.Tensor,
    Ks: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact Q, KV and output matrices used by Wan camera-PRoPE.

    Unlike ``prope_qkv``, this helper also accepts a head axis in the camera
    metadata.  That is required after each attention head routes different
    pages/tokens.  The matrix construction deliberately mirrors
    ``hyvideo.prope.camera_rope`` so the unsampled/full replay remains exact.
    """
    if viewmats.shape[-2:] != (4, 4):
        raise ValueError(f"viewmats must end in 4x4, got {tuple(viewmats.shape)}")
    if Ks is not None:
        if Ks.shape[:-2] != viewmats.shape[:-2] or Ks.shape[-2:] != (3, 3):
            raise ValueError(
                f"Ks/viewmats token axes differ: Ks={tuple(Ks.shape)}, "
                f"viewmats={tuple(viewmats.shape)}"
            )
        normalized = torch.zeros_like(Ks)
        normalized[..., 0, 0] = Ks[..., 0, 0]
        normalized[..., 1, 1] = Ks[..., 1, 1]
        normalized[..., 2, 2] = 1
        projection = torch.einsum(
            "...ij,...jk->...ik", _lift_camera_intrinsics(normalized), viewmats
        )
        projection_inv = torch.einsum(
            "...ij,...jk->...ik",
            _invert_camera_se3(viewmats),
            _lift_camera_intrinsics(_invert_camera_intrinsics(normalized)),
        )
    else:
        projection = viewmats
        projection_inv = _invert_camera_se3(viewmats)
    return projection.transpose(-1, -2), projection_inv, projection


def apply_camera_prope_matrix(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a token-wise 4x4 PRoPE matrix, optionally selected per head."""
    if feats.ndim != 4 or feats.shape[-1] % 4:
        raise ValueError(f"PRoPE features must be BHSD with D divisible by 4: {tuple(feats.shape)}")
    bsz, heads, tokens, dim = feats.shape
    tiled = feats.reshape(bsz, heads, tokens, dim // 4, 4)
    if matrix.ndim == 4:
        if matrix.shape[:2] != (bsz, tokens):
            raise ValueError(
                f"shared PRoPE matrix must be [B,S,4,4], got {tuple(matrix.shape)}"
            )
        out = torch.einsum("bsij,bhsrj->bhsri", matrix, tiled)
    elif matrix.ndim == 5:
        if matrix.shape[:3] != (bsz, heads, tokens):
            raise ValueError(
                f"head-specific PRoPE matrix must be [B,H,S,4,4], got {tuple(matrix.shape)}"
            )
        out = torch.einsum("bhsij,bhsrj->bhsri", matrix, tiled)
    else:
        raise ValueError(f"PRoPE matrix must have rank 4 or 5, got {matrix.ndim}")
    return out.reshape_as(feats)


def gather_camera_metadata(x: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
    """Gather repeated-per-token camera metadata for per-head token indices."""
    if x is None:
        return None
    if indices.ndim != 3 or x.ndim != 4 or indices.shape[0] != x.shape[0]:
        raise ValueError(
            f"camera metadata/indices must be [B,S,M,M]/[B,H,T], got "
            f"{tuple(x.shape)}/{tuple(indices.shape)}"
        )
    expanded = x[:, None].expand(-1, indices.shape[1], -1, -1, -1)
    gather_index = indices[..., None, None].expand(
        *indices.shape, x.shape[-2], x.shape[-1]
    )
    return torch.gather(expanded, 2, gather_index).contiguous()


def camera_prope_replay(track: Track) -> dict[str, Any]:
    """Numerically replay both sampled camera-PRoPE transforms from raw Q/K/V."""
    if not track.camera_prope_present:
        return {
            "status": "unavailable",
            "exact_replay": False,
            "reason": track.camera_prope_reason,
        }
    assert track.query_pre is not None and track.key_pre is not None and track.value_pre is not None
    assert track.query_post_prope is not None
    assert track.key_post_prope is not None and track.value_post_prope is not None
    assert track.query_viewmats is not None and track.key_viewmats is not None
    replay_q, _, _, _ = prope_qkv(
        track.query_pre,
        track.query_pre,
        track.query_pre,
        viewmats=track.query_viewmats,
        Ks=track.query_Ks,
        patches_x=40,
        patches_y=22,
    )
    _, replay_k, replay_v, _ = prope_qkv(
        track.key_pre,
        track.key_pre,
        track.value_pre,
        viewmats=track.key_viewmats,
        Ks=track.key_Ks,
        patches_x=40,
        patches_y=22,
    )

    def max_abs(candidate: torch.Tensor, reference: torch.Tensor) -> float:
        return float(torch.max(torch.abs(candidate.float() - reference.float())).item())

    errors = {
        "query_replay_max_abs": max_abs(replay_q, track.query_post_prope),
        "key_replay_max_abs": max_abs(replay_k, track.key_post_prope),
        "value_replay_max_abs": max_abs(replay_v, track.value_post_prope),
    }
    exact = all(value == 0.0 for value in errors.values())
    return {
        "status": "available" if exact else "replay_mismatch",
        "exact_replay": exact,
        "replay_tolerance_max_abs": 0.0,
        **errors,
        "reason": "exact hyvideo.prope.camera_rope.prope_qkv replay" if exact else "saved camera-PRoPE tensors do not replay exactly",
    }


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm((candidate.float() - reference.float()).reshape(-1))
    denom = torch.linalg.vector_norm(reference.float().reshape(-1)).clamp_min(1e-12)
    return float((diff / denom).item())


def cosine_similarity(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a = candidate.float().reshape(-1)
    b = reference.float().reshape(-1)
    return float(F.cosine_similarity(a, b, dim=0).item())


def mse(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.mean((candidate.float() - reference.float()) ** 2).item())


def attention_probs(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    return torch.softmax(
        torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(query.shape[-1]),
        dim=-1,
    )


def attention_kl(query: torch.Tensor, key: torch.Tensor, reference_key: torch.Tensor) -> float:
    p = attention_probs(query, reference_key).clamp_min(1e-12)
    q = attention_probs(query, key).clamp_min(1e-12)
    return float(torch.mean(torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)).item())


def output_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    return {
        "relative_l2": relative_l2(candidate, reference),
        "cosine": cosine_similarity(candidate, reference),
        "mse": mse(candidate, reference),
    }


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(x) for x in values)
    if not ordered:
        raise ValueError("Cannot summarize no timing samples")
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def timing_stats(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "median_ms": float(statistics.median(values)),
        "p10_ms": percentile(values, 0.10),
        "p90_ms": percentile(values, 0.90),
        "mean_ms": float(statistics.fmean(values)),
        "min_ms": min(values),
        "max_ms": max(values),
        "samples": len(values),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    fn: Callable[[], Any], *, warmup: int, repeat: int, device: torch.device
) -> dict[str, float | int]:
    for _ in range(max(0, warmup)):
        out = fn()
        del out
    synchronize(device)
    values: list[float] = []
    for _ in range(max(1, repeat)):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        else:
            start_cpu = time.perf_counter()
            out = fn()
            values.append((time.perf_counter() - start_cpu) * 1000.0)
        del out
    return timing_stats(values)


@contextmanager
def temporary_num_levels(levels: int) -> Iterable[None]:
    old = os.environ.get("HYWORLDPLAY_RVQ_NUM_LEVELS")
    os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = str(int(levels))
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("HYWORLDPLAY_RVQ_NUM_LEVELS", None)
        else:
            os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = old


def encode_fixed(
    tensor: torch.Tensor,
    *,
    layer: int,
    directory: Path,
    batch_size: int,
) -> dict[str, Any]:
    with temporary_num_levels(8):
        packed = rvq_encode_tensor_cache(
            tensor.contiguous(),
            layer_idx=layer,
            codebook_dir=directory,
            batch_size=batch_size,
            compute_dtype="bf16" if tensor.device.type == "cuda" else "fp32",
            codebook_storage_dtype="int4",
            scale_storage_dtype="uint4",
        )
    assert_no_token_scale(packed)
    if int(packed["indices"].shape[0]) != 8:
        raise AssertionError("Expected exactly eight RVQ levels")
    return packed


def decode_prefix(
    packed: dict[str, Any], levels: int, *, output_dtype: torch.dtype
) -> torch.Tensor:
    return rvq_dequantize_tensor_cache(prefix_packed(packed, levels), output_dtype)


def gather_packed_tokens(packed: dict[str, Any], token_indices: torch.Tensor) -> dict[str, Any]:
    """Gather per-(batch,head) token ids without introducing scale state."""
    assert_no_token_scale(packed)
    indices = packed["indices"]
    if token_indices.ndim != 3 or tuple(token_indices.shape[:2]) != tuple(indices.shape[1:3]):
        raise ValueError(
            f"token_indices must be [B,H,T] matching packed indices, got {tuple(token_indices.shape)}"
        )
    gather_index = token_indices[None].expand(indices.shape[0], -1, -1, -1)
    out = {key: value for key, value in packed.items() if key != "indices"}
    out["indices"] = torch.gather(indices, 3, gather_index).contiguous()
    return out


def decode_selected_tokens(
    packed: dict[str, Any], token_indices: torch.Tensor, *, output_dtype: torch.dtype
) -> torch.Tensor:
    return rvq_dequantize_tensor_cache(gather_packed_tokens(packed, token_indices), output_dtype)


def slice_packed_token_block(
    packed: dict[str, Any], start: int, end: int
) -> dict[str, Any]:
    """Slice a contiguous token block from fixed-depth no-norm packed indices.

    H1 intentionally operates on the pre-RoPE packed state.  Only after a
    block is decoded do we apply that block's saved, token-dependent RoPE.
    This avoids the invalid shortcut of encoding post-RoPE keys with a
    pre-RoPE codebook.
    """
    assert_no_token_scale(packed)
    indices = packed["indices"]
    tokens = int(indices.shape[-1])
    start, end = int(start), int(end)
    if start < 0 or end <= start or end > tokens:
        raise ValueError(f"Invalid packed token block [{start},{end}) for {tokens} tokens")
    if bool(packed.get("progressive", False)):
        raise ValueError("H1 exact block references require fixed-depth packed indices")
    out = {key: value for key, value in packed.items() if key != "indices"}
    out["indices"] = indices[..., start:end].contiguous()
    return out


def compressed_qk_blockwise_reference(
    query_post_rope: torch.Tensor,
    packed_k_pre_rope: dict[str, Any],
    key_rope_cos: torch.Tensor,
    key_rope_sin: torch.Tensor,
    *,
    block_tokens: int,
) -> torch.Tensor:
    """Materialize QK while decoding and applying saved RoPE by key block.

    This is a PyTorch diagnostic reference, not a fused production kernel.
    It never materializes the complete decoded K tensor, but by design it
    materializes the complete Q-by-K logits requested by the standalone-QK
    experiment.
    """
    block_tokens = int(block_tokens)
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    tokens = int(packed_k_pre_rope["indices"].shape[-1])
    if int(key_rope_cos.shape[2]) < tokens or int(key_rope_sin.shape[2]) < tokens:
        raise ValueError("Saved key RoPE metadata is shorter than packed K")
    if tokens <= 0:
        raise ValueError("Packed K must contain at least one token")
    logits = torch.empty(
        (*query_post_rope.shape[:-1], tokens),
        device=query_post_rope.device,
        dtype=query_post_rope.dtype,
    )
    for start in range(0, tokens, block_tokens):
        end = min(tokens, start + block_tokens)
        key_pre = rvq_dequantize_tensor_cache(
            slice_packed_token_block(packed_k_pre_rope, start, end),
            query_post_rope.dtype,
        )
        key_post = apply_saved_rope(
            key_pre,
            key_rope_cos[:, :, start:end],
            key_rope_sin[:, :, start:end],
        )
        logits[..., start:end] = torch.matmul(
            query_post_rope, key_post.transpose(-1, -2)
        )
    return logits


def exact_compressed_streaming_attention_reference(
    query_post_rope: torch.Tensor,
    packed_k_pre_rope: dict[str, Any],
    packed_v_pre_prope: dict[str, Any],
    key_rope_cos: torch.Tensor,
    key_rope_sin: torch.Tensor,
    *,
    block_tokens: int,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Exact blockwise compressed-attention reference with online softmax.

    K/V indices are decoded in contiguous key blocks, standard RoPE is applied
    only to decoded pre-RoPE K, and FP32 running max/normalizer/PV state makes
    the softmax stable across blocks.  Full decoded K/V and full logits are
    never materialized.  This path is deliberately plain PyTorch so it is a
    correctness and latency diagnostic, not a claim about a fused Triton
    implementation.
    """
    block_tokens = int(block_tokens)
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    k_tokens = int(packed_k_pre_rope["indices"].shape[-1])
    v_tokens = int(packed_v_pre_prope["indices"].shape[-1])
    if k_tokens <= 0 or k_tokens != v_tokens:
        raise ValueError(f"Packed K/V token mismatch: K={k_tokens}, V={v_tokens}")
    if int(key_rope_cos.shape[2]) < k_tokens or int(key_rope_sin.shape[2]) < k_tokens:
        raise ValueError("Saved key RoPE metadata is shorter than packed K")

    q = query_post_rope.float()
    scale = float(query_post_rope.shape[-1]) ** -0.5
    running_max: torch.Tensor | None = None
    running_sum: torch.Tensor | None = None
    running_pv: torch.Tensor | None = None
    for start in range(0, k_tokens, block_tokens):
        end = min(k_tokens, start + block_tokens)
        key_pre = rvq_dequantize_tensor_cache(
            slice_packed_token_block(packed_k_pre_rope, start, end),
            query_post_rope.dtype,
        )
        value = rvq_dequantize_tensor_cache(
            slice_packed_token_block(packed_v_pre_prope, start, end),
            query_post_rope.dtype,
        ).float()
        key_post = apply_saved_rope(
            key_pre,
            key_rope_cos[:, :, start:end],
            key_rope_sin[:, :, start:end],
        )
        logits = torch.matmul(q, key_post.float().transpose(-1, -2)) * scale
        block_max = logits.amax(dim=-1, keepdim=True)
        if running_max is None:
            running_max = block_max
            weights = torch.exp(logits - running_max)
            running_sum = weights.sum(dim=-1, keepdim=True)
            running_pv = torch.matmul(weights, value)
            continue

        assert running_sum is not None and running_pv is not None
        new_max = torch.maximum(running_max, block_max)
        old_factor = torch.exp(running_max - new_max)
        weights = torch.exp(logits - new_max)
        running_sum = running_sum * old_factor + weights.sum(dim=-1, keepdim=True)
        running_pv = running_pv * old_factor + torch.matmul(weights, value)
        running_max = new_max

    if running_sum is None or running_pv is None:
        raise ValueError("Packed K/V must contain at least one token")
    dtype = query_post_rope.dtype if output_dtype is None else output_dtype
    return (running_pv / running_sum.clamp_min(1e-20)).to(dtype)


def tier_depths(num_chunks: int) -> list[int]:
    """Return first/recent L8, two oldest non-anchor L4, remaining L6."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if num_chunks == 1:
        return [8]
    result = [6] * num_chunks
    result[0] = result[-1] = 8
    old_candidates = list(range(1, max(1, num_chunks - 1)))[:2]
    for index in old_candidates:
        result[index] = 4
    return result


def chunk_ids_for_track(track: Track, *, chunk_tokens: int, token_count: int) -> torch.Tensor:
    if track.key_indices is not None and int(track.key_indices.numel()) >= token_count:
        positions = track.key_indices[:token_count]
    else:
        positions = torch.arange(token_count, dtype=torch.long)
    minimum = int(positions.min().item()) if positions.numel() else 0
    return torch.div(positions - minimum, int(chunk_tokens), rounding_mode="floor")


def decode_tiered(
    packed: dict[str, Any],
    *,
    chunk_ids: torch.Tensor,
    depths: Sequence[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    indices = packed["indices"]
    bsz, heads, tokens = map(int, indices.shape[1:])
    out: torch.Tensor | None = None
    chunk_ids_device = chunk_ids.to(device=indices.device)
    for depth in sorted(set(int(x) for x in depths)):
        token_mask = torch.zeros(tokens, device=indices.device, dtype=torch.bool)
        for chunk, chunk_depth in enumerate(depths):
            if int(chunk_depth) == depth:
                token_mask |= chunk_ids_device == chunk
        positions = torch.nonzero(token_mask, as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        selected = positions[None, None, :].expand(bsz, heads, -1).contiguous()
        decoded = decode_selected_tokens(
            prefix_packed(packed, depth), selected, output_dtype=output_dtype
        )
        if out is None:
            out = torch.empty(
                (bsz, heads, tokens, decoded.shape[-1]),
                device=indices.device,
                dtype=output_dtype,
            )
        out[:, :, positions, :] = decoded
    if out is None:
        raise ValueError("tiered schedule selected no tokens")
    return out


def coarse_residual_depths(
    original: torch.Tensor,
    packed: dict[str, Any],
    *,
    chunk_ids: torch.Tensor,
    target_average_depth: float,
) -> tuple[list[int], list[float]]:
    """Assign one L4/L6/L8 depth per chunk from L6 residual RMS.

    This is the regular-runtime counterpart to token-progressive coding.  It
    deliberately uses the same reconstruction-residual family of objectives,
    so the experiment can separate target mismatch from per-token metadata and
    gather/scatter overhead.  For at least three chunks, all three depths are
    represented whenever the closest-rate integer allocation permits it.
    """
    assert_no_token_scale(packed)
    if chunk_ids.ndim != 1 or int(chunk_ids.numel()) != int(original.shape[2]):
        raise ValueError("chunk_ids must contain one id per token")
    num_chunks = int(chunk_ids.max().item()) + 1 if chunk_ids.numel() else 0
    if num_chunks <= 0:
        raise ValueError("coarse depth assignment needs at least one chunk")
    l6 = decode_prefix(packed, 6, output_dtype=original.dtype)
    token_error = torch.mean((original.float() - l6.float()) ** 2, dim=(0, 1, 3))
    scores: list[float] = []
    chunk_ids_device = chunk_ids.to(token_error.device)
    for chunk in range(num_chunks):
        selected = token_error[chunk_ids_device == chunk]
        scores.append(float(torch.sqrt(selected.mean()).item()) if selected.numel() else 0.0)

    candidates: list[tuple[float, int, int, int]] = []
    for n4 in range(num_chunks + 1):
        for n8 in range(num_chunks - n4 + 1):
            n6 = num_chunks - n4 - n8
            if num_chunks >= 3 and min(n4, n6, n8) == 0:
                continue
            average = (4 * n4 + 6 * n6 + 8 * n8) / num_chunks
            # Prefer the closest rate, then the fewest deliberately degraded
            # L4 chunks, then the fewest non-L6 chunks.
            candidates.append((abs(average - target_average_depth), n4, n4 + n8, n8))
    if not candidates:
        for n4 in range(num_chunks + 1):
            for n8 in range(num_chunks - n4 + 1):
                n6 = num_chunks - n4 - n8
                average = (4 * n4 + 6 * n6 + 8 * n8) / num_chunks
                candidates.append((abs(average - target_average_depth), n4, n4 + n8, n8))
    _, n4, _, n8 = min(candidates)
    order = sorted(range(num_chunks), key=lambda index: (scores[index], index))
    depths = [6] * num_chunks
    for chunk in order[:n4]:
        depths[chunk] = 4
    for chunk in order[num_chunks - n8 :]:
        depths[chunk] = 8
    return depths, scores


def rankdata(values: torch.Tensor) -> torch.Tensor:
    flat = values.float().reshape(-1)
    order = torch.argsort(flat, stable=True)
    ranks = torch.empty_like(flat)
    ranks[order] = torch.arange(flat.numel(), device=flat.device, dtype=torch.float32)
    # Ties are rare for these continuous errors; average exact ties to make the
    # helper mathematically complete without scipy.
    sorted_values = flat[order]
    if flat.numel() > 1:
        boundaries = torch.cat(
            [
                torch.tensor([True], device=flat.device),
                sorted_values[1:] != sorted_values[:-1],
                torch.tensor([True], device=flat.device),
            ]
        ).nonzero(as_tuple=False).flatten()
        for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist()):
            if end - start > 1:
                ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks.reshape(values.shape)


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel() or a.numel() < 2:
        return float("nan")
    ra = rankdata(a).reshape(-1)
    rb = rankdata(b).reshape(-1)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.linalg.vector_norm(ra) * torch.linalg.vector_norm(rb)
    return float((torch.dot(ra, rb) / denom.clamp_min(1e-12)).item())


def error_sensitivity_correlations(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
) -> dict[str, float]:
    k_error = key.float() - key_hat.float()
    v_error = value.float() - value_hat.float()
    raw_k_rms = torch.sqrt(torch.mean(k_error * k_error, dim=-1))
    raw_v_rms = torch.sqrt(torch.mean(v_error * v_error, dim=-1))
    qk_delta = torch.matmul(query.float(), k_error.transpose(-1, -2))
    k_sensitivity = torch.mean(qk_delta * qk_delta, dim=2)
    probs = attention_probs(query, key)
    v_energy = torch.sum(v_error * v_error, dim=-1)
    v_sensitivity = torch.mean(probs * probs, dim=2) * v_energy
    return {
        "k_residual_rms_vs_qk_sensitivity_spearman": spearman(raw_k_rms, k_sensitivity),
        "v_residual_rms_vs_pv_sensitivity_spearman": spearman(raw_v_rms, v_sensitivity),
    }


def calibration_candidates(
    query: torch.Tensor,
    key: torch.Tensor,
    key_hat: torch.Tensor,
    *,
    block_tokens: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build scalar, diagonal, and full-linear QK calibration candidates."""
    bsz, heads, tokens, dim = key.shape
    scalar_key = torch.empty_like(key_hat)
    scalars: list[float] = []
    for b in range(bsz):
        for h in range(heads):
            q = query[b, h].float()
            for start in range(0, tokens, block_tokens):
                end = min(tokens, start + block_tokens)
                target = torch.matmul(q, key[b, h, start:end].float().T)
                base = torch.matmul(q, key_hat[b, h, start:end].float().T)
                alpha = torch.sum(base * target) / torch.sum(base * base).clamp_min(1e-12)
                scalar_key[b, h, start:end] = (key_hat[b, h, start:end].float() * alpha).to(key_hat.dtype)
                scalars.append(float(alpha.item()))

    diagonal_key = torch.empty_like(key_hat)
    full_key = torch.empty_like(key_hat)
    diag_values: list[torch.Tensor] = []
    for b in range(bsz):
        for h in range(heads):
            q = query[b, h].float()
            kh = key_hat[b, h].float()
            kt = key[b, h].float()
            sq = q.T @ q
            sh = kh.T @ kh
            shk = kh.T @ kt
            normal = sq * sh
            rhs = torch.sum(sq * shk, dim=1)
            ridge = torch.eye(dim, device=key.device) * (normal.diag().mean().clamp_min(1e-8) * 1e-5)
            diag = torch.linalg.solve(normal + ridge, rhs)
            diagonal_key[b, h] = (kh * diag[None, :]).to(key_hat.dtype)
            diag_values.append(diag.detach())

            full_ridge = torch.eye(dim, device=key.device) * ((kh.T @ kh).diag().mean().clamp_min(1e-8) * 1e-5)
            transform = torch.linalg.solve(kh.T @ kh + full_ridge, kh.T @ kt)
            full_key[b, h] = (kh @ transform).to(key_hat.dtype)

    scalar_tensor = torch.tensor(scalars, dtype=torch.float32)
    return (
        {"uncalibrated": key_hat, "block_scalar": scalar_key, "per_channel_diagonal": diagonal_key, "q_cov_full_linear_upper_bound": full_key},
        {
            "block_scalar": {
                "count": int(scalar_tensor.numel()),
                "mean": float(scalar_tensor.mean().item()),
                "std": float(scalar_tensor.std(unbiased=False).item()),
                "min": float(scalar_tensor.min().item()),
                "max": float(scalar_tensor.max().item()),
                "fraction_abs_minus_one_lt_0p02": float(((scalar_tensor - 1).abs() < 0.02).float().mean().item()),
            },
            "diagonal_mean_abs_minus_one": float(
                torch.mean(torch.abs(torch.stack(diag_values) - 1.0)).item()
            ),
        },
    )


def evaluate_calibrations(
    query: torch.Tensor,
    key: torch.Tensor,
    value_hat: torch.Tensor,
    candidates: dict[str, torch.Tensor],
) -> dict[str, Any]:
    dense_ref = F.scaled_dot_product_attention(query, key, value_hat)
    true_logits = torch.matmul(query.float(), key.float().transpose(-1, -2))
    rows: dict[str, Any] = {}
    for name, candidate in candidates.items():
        logits = torch.matmul(query.float(), candidate.float().transpose(-1, -2))
        output = F.scaled_dot_product_attention(query, candidate, value_hat)
        rows[name] = {
            "qk_mse": mse(logits, true_logits),
            "attention_kl": attention_kl(query, candidate, key),
            "pv": output_metrics(output, dense_ref),
        }
    base = rows["uncalibrated"]["qk_mse"]
    for row in rows.values():
        row["qk_error_recoverable_fraction_vs_uncalibrated"] = (
            (base - row["qk_mse"]) / base if base > 0 else 0.0
        )
    return rows


def contiguous_page_eligible(track: Track, tokens: int, page_size: int) -> tuple[bool, str]:
    if tokens % page_size:
        return False, "sample length is not divisible by page size"
    if track.key_indices is None:
        return True, "implicit contiguous token order"
    indices = track.key_indices[:tokens]
    if indices.numel() <= 1 or bool(torch.all(indices[1:] - indices[:-1] == 1).item()):
        return True, "saved key indices are contiguous"
    return False, "saved key sample is non-contiguous; page means would not represent real pages"


def gather_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    # x [B,H,K,D], indices [B,H,T]
    return torch.gather(x, 2, indices[..., None].expand(*indices.shape, x.shape[-1])).contiguous()


def gather_rope_metadata(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expanded = x.expand(indices.shape[0], indices.shape[1], x.shape[2], x.shape[3])
    return gather_tokens(expanded, indices)


def page_means(x: torch.Tensor, page_size: int) -> torch.Tensor:
    bsz, heads, tokens, dim = x.shape
    pages = tokens // page_size
    return x[:, :, : pages * page_size].reshape(bsz, heads, pages, page_size, dim).mean(dim=3)


def page_mass(probs: torch.Tensor, page_size: int) -> torch.Tensor:
    bsz, heads, queries, tokens = probs.shape
    pages = tokens // page_size
    return probs[..., : pages * page_size].reshape(bsz, heads, queries, pages, page_size).sum(dim=-1)


def temporal_attention_profile(
    probs: torch.Tensor,
    *,
    chunk_tokens: int,
) -> dict[str, Any]:
    """Measure dense attention mass by contiguous generation-cache age."""
    if probs.ndim != 4 or int(chunk_tokens) <= 0:
        raise ValueError("probs must be [B,H,Q,K] and chunk_tokens positive")
    tokens = int(probs.shape[-1])
    chunks = math.ceil(tokens / int(chunk_tokens))
    masses: list[float] = []
    for chunk in range(chunks):
        start = chunk * int(chunk_tokens)
        end = min(tokens, start + int(chunk_tokens))
        masses.append(float(probs[..., start:end].sum(dim=-1).mean().item()))
    ages = list(reversed(range(chunks)))
    age_tensor = torch.tensor(ages, dtype=torch.float32, device=probs.device)
    mass_tensor = torch.tensor(masses, dtype=torch.float32, device=probs.device)
    rho = spearman(age_tensor, mass_tensor) if chunks >= 2 else float("nan")
    newest_to_oldest = list(reversed(masses))
    adjacent_decay = (
        sum(
            newest_to_oldest[index] >= newest_to_oldest[index + 1]
            for index in range(len(newest_to_oldest) - 1)
        )
        / max(1, len(newest_to_oldest) - 1)
    )
    return {
        "chunk_tokens": int(chunk_tokens),
        "num_chunks": chunks,
        "age_definition": "0=newest contiguous saved key chunk",
        "spearman_age_vs_attention_mass": float(rho),
        "adjacent_decay_fraction_newest_to_oldest": float(adjacent_decay),
        "rows": [
            {
                "chunk_index_oldest_to_newest": chunk,
                "age_chunks": ages[chunk],
                "attention_mass": masses[chunk],
            }
            for chunk in range(chunks)
        ],
    }


def top_page_indices(scores: torch.Tensor, count: int) -> torch.Tensor:
    # score [B,H,Q,P] or [B,H,P]
    if scores.ndim == 4:
        scores = scores.amax(dim=2)
    return torch.topk(scores, k=min(int(count), int(scores.shape[-1])), dim=-1).indices


def pages_to_tokens(pages: torch.Tensor, page_size: int) -> torch.Tensor:
    offsets = torch.arange(page_size, device=pages.device)
    return (pages[..., None] * page_size + offsets).reshape(*pages.shape[:-1], -1)


def selection_metrics(
    selected_pages: torch.Tensor,
    oracle_pages: torch.Tensor,
    mass_by_page: torch.Tensor,
) -> dict[str, float]:
    selected = selected_pages[..., :, None]
    oracle = oracle_pages[..., None, :]
    overlap = (selected == oracle).any(dim=-1).float().mean()
    selected_mass = torch.gather(
        mass_by_page.mean(dim=2), -1, selected_pages
    ).sum(dim=-1).mean()
    return {"oracle_page_recall": float(overlap.item()), "captured_attention_mass": float(selected_mass.item())}


def selected_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    token_indices = pages_to_tokens(selected_pages, page_size)
    return F.scaled_dot_product_attention(query, gather_tokens(key, token_indices), gather_tokens(value, token_indices))


def hybrid_page_tail_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_pages: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    bsz, heads, tokens, dim = key.shape
    pages = tokens // page_size
    exact_indices = pages_to_tokens(selected_pages, page_size)
    exact_k = gather_tokens(key, exact_indices)
    exact_v = gather_tokens(value, exact_indices)
    k_mean = page_means(key, page_size)
    v_mean = page_means(value, page_size)
    all_pages = torch.arange(pages, device=key.device)[None, None, :].expand(bsz, heads, -1)
    keep_tail = ~(all_pages[..., None] == selected_pages[..., None, :]).any(dim=-1)
    tail_count = int(keep_tail.sum(dim=-1).max().item())
    tail_indices = torch.empty((bsz, heads, tail_count), device=key.device, dtype=torch.long)
    for b in range(bsz):
        for h in range(heads):
            ids = all_pages[b, h][keep_tail[b, h]]
            tail_indices[b, h, : ids.numel()] = ids
    tail_k = gather_tokens(k_mean, tail_indices)
    tail_v = gather_tokens(v_mean, tail_indices)
    combined_k = torch.cat([exact_k, tail_k], dim=2)
    combined_v = torch.cat([exact_v, tail_v], dim=2)
    bias = torch.zeros((bsz, heads, 1, combined_k.shape[2]), device=key.device, dtype=query.dtype)
    bias[..., exact_k.shape[2] :] = math.log(page_size)
    return F.scaled_dot_product_attention(query, combined_k, combined_v, attn_mask=bias)


def temporal_page_policies(
    pages: int,
    count: int,
    *,
    device: torch.device,
    bsz: int,
    heads: int,
) -> dict[str, torch.Tensor]:
    count = min(max(1, count), pages)
    recent = torch.arange(pages - count, pages, device=device)
    first_count = count // 2
    recent_count = count - first_count
    first_recent = torch.cat(
        [torch.arange(first_count, device=device), torch.arange(pages - recent_count, pages, device=device)]
    ).unique(sorted=True)
    if first_recent.numel() < count:
        missing = torch.tensor(
            [idx for idx in range(pages) if idx not in set(first_recent.tolist())], device=device
        )[: count - first_recent.numel()]
        first_recent = torch.cat([first_recent, missing])
    uniform = torch.linspace(0, pages - 1, steps=count, device=device).round().long().unique(sorted=True)
    if uniform.numel() < count:
        missing = torch.tensor([idx for idx in range(pages) if idx not in set(uniform.tolist())], device=device)[
            : count - uniform.numel()
        ]
        uniform = torch.cat([uniform, missing])
    return {
        "recent_only": recent.expand(bsz, heads, -1).contiguous(),
        "first_plus_recent": first_recent[:count].expand(bsz, heads, -1).contiguous(),
        "uniform_temporal": uniform[:count].expand(bsz, heads, -1).contiguous(),
    }


def logical_tier_storage(
    *, heads: int, tokens: int, chunk_ids: torch.Tensor, depths: Sequence[int], centroids: int = 64
) -> dict[str, Any]:
    bits = int(math.ceil(math.log2(centroids)))
    token_depth = torch.tensor([depths[int(idx)] for idx in chunk_ids.tolist()], dtype=torch.int64)
    indices = int(token_depth.sum().item()) * int(heads)
    logical = math.ceil(indices * bits / 8)
    # Physical today is one uint8 per index.  The schedule is coarse config,
    # not a per-token depth tensor; one byte per chunk is listed separately.
    physical = indices
    return {
        "average_depth": float(token_depth.float().mean().item()),
        "logical_index_bytes": logical,
        "physical_uint8_index_bytes": physical,
        "coarse_schedule_bytes": len(depths),
        "per_token_scale_bytes": 0,
    }


def evaluate_h1(
    track: Track,
    packed_k: dict[str, Any] | None,
    packed_v: dict[str, Any] | None,
    *,
    warmup: int,
    repeat: int,
    device: torch.device,
    exact_reference_max_elements: int,
    exact_key_block_tokens: int,
) -> dict[str, Any]:
    q, k, v = track.query_post_rope, track.key_post_rope, track.value_post_rope
    result: dict[str, Any] = {
        "theory_test": (
            "separate dense SDPA, explicit restore, blockwise compressed QK, and "
            "online-softmax compressed full-attention timing"
        ),
        "accuracy_valid": track.accuracy_valid,
        "accuracy_reason": track.accuracy_reason,
        "implementation_scope": (
            "PyTorch diagnostic block reference; not a fused Triton production kernel"
        ),
        "timing_interval": {
            "standalone_compressed_qk": (
                "packed-index block slice through K decode, saved RoPE, QK, and full-logit write"
            ),
            "exact_compressed_streaming_full_attention": (
                "packed-index block slice through K/V decode, saved K RoPE, FP32 online softmax, and PV output"
            ),
            "dense_post_rope_sdpa": "PyTorch SDPA call only on resident BF16 Q/K/V",
        },
        "key_block_tokens": int(exact_key_block_tokens),
        "timing": {
            "dense_post_rope_qk": benchmark(lambda: torch.matmul(q, k.transpose(-1, -2)), warmup=warmup, repeat=repeat, device=device),
            "dense_post_rope_sdpa": benchmark(lambda: F.scaled_dot_product_attention(q, k, v), warmup=warmup, repeat=repeat, device=device),
        },
    }
    if packed_k is None or packed_v is None:
        result["compressed_paths"] = {"status": "unavailable", "reason": track.accuracy_reason}
        return result

    output_dtype = q.dtype
    assert track.key_rope_cos is not None and track.key_rope_sin is not None

    def decode_k() -> torch.Tensor:
        return rvq_dequantize_tensor_cache(packed_k, output_dtype)

    def decode_v() -> torch.Tensor:
        return rvq_dequantize_tensor_cache(packed_v, output_dtype)

    def restore_attention() -> torch.Tensor:
        restored_k = apply_saved_rope(decode_k(), track.key_rope_cos, track.key_rope_sin)
        restored_v = decode_v()
        return F.scaled_dot_product_attention(q, restored_k, restored_v)

    def restore_qk() -> torch.Tensor:
        restored_k = apply_saved_rope(decode_k(), track.key_rope_cos, track.key_rope_sin)
        return torch.matmul(q, restored_k.transpose(-1, -2))

    def compressed_qk() -> torch.Tensor:
        return compressed_qk_blockwise_reference(
            q,
            packed_k,
            track.key_rope_cos,
            track.key_rope_sin,
            block_tokens=exact_key_block_tokens,
        )

    def compressed_streaming_attention() -> torch.Tensor:
        return exact_compressed_streaming_attention_reference(
            q,
            packed_k,
            packed_v,
            track.key_rope_cos,
            track.key_rope_sin,
            block_tokens=exact_key_block_tokens,
            output_dtype=q.dtype,
        )

    result["timing"].update(
        {
            "decode_pre_k": benchmark(decode_k, warmup=warmup, repeat=repeat, device=device),
            "decode_pre_v": benchmark(decode_v, warmup=warmup, repeat=repeat, device=device),
            "explicit_decode_rope_qk": benchmark(
                restore_qk, warmup=warmup, repeat=repeat, device=device
            ),
            "decode_plus_real_rope_plus_sdpa": benchmark(restore_attention, warmup=warmup, repeat=repeat, device=device),
            "exact_compressed_streaming_full_attention": benchmark(
                compressed_streaming_attention,
                warmup=warmup,
                repeat=repeat,
                device=device,
            ),
        }
    )
    restored_k = apply_saved_rope(decode_k(), track.key_rope_cos, track.key_rope_sin)
    restored_v = decode_v()
    dense_out = F.scaled_dot_product_attention(q, k, v)
    restored_out = F.scaled_dot_product_attention(q, restored_k, restored_v)
    streaming_out = compressed_streaming_attention()
    streaming_equivalence = output_metrics(streaming_out, restored_out)
    result["accuracy"] = {
        "key_post_rope": output_metrics(restored_k, k),
        "value_post_rope": output_metrics(restored_v, v),
        "attention_output": output_metrics(restored_out, dense_out),
        "streaming_vs_explicit_decode_rope_sdpa": streaming_equivalence,
    }
    result["nonmaterializing_exact_compressed"] = {
        "status": "measured",
        "path": "blockwise_decode_saved_rope_fp32_online_softmax_pv",
        "materializes_full_decoded_kv": False,
        "materializes_full_logits": False,
        "reference_relative_l2_tolerance": EXACT_STREAMING_RELATIVE_L2_TOLERANCE,
        "reference_equivalence_pass": bool(
            streaming_equivalence["relative_l2"]
            <= EXACT_STREAMING_RELATIVE_L2_TOLERANCE
        ),
        "note": "PyTorch diagnostic reference, not a fused Triton production kernel",
    }
    qk_elements = int(q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2])
    if qk_elements <= int(exact_reference_max_elements):
        result["timing"]["standalone_compressed_qk"] = benchmark(
            compressed_qk, warmup=warmup, repeat=repeat, device=device
        )
        explicit_qk = torch.matmul(q, restored_k.transpose(-1, -2))
        compressed_logits = compressed_qk()
        result["standalone_compressed_qk"] = {
            "status": "measured",
            "materializes_full_logits": True,
            "materializes_full_decoded_k": False,
            "accuracy_vs_explicit_decode_rope_qk": output_metrics(
                compressed_logits, explicit_qk
            ),
            "note": "PyTorch diagnostic reference, not a fused Triton production kernel",
        }
    else:
        result["standalone_compressed_qk"] = {
            "status": "skipped_element_guard",
            "reason": (
                f"QK has {qk_elements} elements, above configured guard "
                f"{int(exact_reference_max_elements)}"
            ),
            "materializes_full_logits": True,
        }
    result["exact_reference_guard"] = {
        "qk_elements": qk_elements,
        "configured_max": int(exact_reference_max_elements),
        "standalone_qk_called": qk_elements <= int(exact_reference_max_elements),
        "streaming_full_attention_called": True,
    }
    result["compressed_paths"] = {
        "status": "measured",
        "encoded_space": "pre_rope_raw_k_and_pre_prope_raw_v",
        "position_restore": "saved per-token standard RoPE applied after block decode",
    }
    return result


def calibrate_progressive_threshold(
    original: torch.Tensor,
    packed_l8: dict[str, Any],
    *,
    target_depth: float,
) -> float:
    if not 6.0 <= target_depth <= 8.0:
        raise ValueError("This L8 runner supports progressive target depth in [6,8]")
    errors = {}
    for depth in (6, 7):
        recon = decode_prefix(packed_l8, depth, output_dtype=original.dtype)
        errors[depth] = torch.sqrt(torch.mean((original.float() - recon.float()) ** 2, dim=-1)).reshape(-1)
    candidates = torch.unique(torch.cat([errors[6], errors[7]])).sort().values
    if candidates.numel() > 2048:
        positions = torch.linspace(0, candidates.numel() - 1, 2048, device=candidates.device).long()
        candidates = candidates[positions]
    best_threshold = float(candidates[0].item()) if candidates.numel() else 0.0
    best_delta = float("inf")
    for threshold_tensor in candidates:
        threshold = float(threshold_tensor.item())
        depth = torch.full_like(errors[6], 8.0)
        stop6 = errors[6] <= threshold
        depth[stop6] = 6.0
        stop7 = (~stop6) & (errors[7] <= threshold)
        depth[stop7] = 7.0
        delta = abs(float(depth.mean().item()) - target_depth)
        if delta < best_delta:
            best_delta = delta
            best_threshold = threshold
    return best_threshold


def encode_progressive(
    tensor: torch.Tensor,
    *,
    layer: int,
    directory: Path,
    batch_size: int,
    threshold: float,
) -> dict[str, Any]:
    packed = rvq_encode_tensor_cache(
        tensor.contiguous(),
        layer_idx=layer,
        codebook_dir=directory,
        batch_size=batch_size,
        compute_dtype="bf16" if tensor.device.type == "cuda" else "fp32",
        codebook_storage_dtype="int4",
        scale_storage_dtype="uint4",
        progressive=True,
        progressive_min_depth=6,
        progressive_max_depth=8,
        progressive_error_threshold=float(threshold),
        progressive_granularity="token",
    )
    assert_no_token_scale(packed)
    return packed


def evaluate_h2(
    track: Track,
    packed_k: dict[str, Any],
    packed_v: dict[str, Any],
    key_hat_post: torch.Tensor,
    value_hat: torch.Tensor,
    *,
    k_dir: Path,
    v_dir: Path,
    batch_size: int,
    target_depth: float,
    chunk_tokens: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> dict[str, Any]:
    assert track.key_pre is not None and track.value_pre is not None
    threshold_k = calibrate_progressive_threshold(track.key_pre, packed_k, target_depth=target_depth)
    threshold_v = calibrate_progressive_threshold(track.value_pre, packed_v, target_depth=target_depth)

    encode_k_timing = benchmark(
        lambda: encode_progressive(track.key_pre, layer=track.layer, directory=k_dir, batch_size=batch_size, threshold=threshold_k),
        warmup=0,
        repeat=1,
        device=device,
    )
    encode_v_timing = benchmark(
        lambda: encode_progressive(track.value_pre, layer=track.layer, directory=v_dir, batch_size=batch_size, threshold=threshold_v),
        warmup=0,
        repeat=1,
        device=device,
    )
    progressive_k = encode_progressive(
        track.key_pre, layer=track.layer, directory=k_dir, batch_size=batch_size, threshold=threshold_k
    )
    progressive_v = encode_progressive(
        track.value_pre, layer=track.layer, directory=v_dir, batch_size=batch_size, threshold=threshold_v
    )
    k6, v6 = prefix_packed(packed_k, 6), prefix_packed(packed_v, 6)
    progressive_k_hat = rvq_dequantize_tensor_cache(progressive_k, track.key_pre.dtype)
    progressive_v_hat = rvq_dequantize_tensor_cache(progressive_v, track.value_pre.dtype)
    fixed8_k_hat_pre = rvq_dequantize_tensor_cache(packed_k, track.key_pre.dtype)
    fixed6_k_hat = rvq_dequantize_tensor_cache(k6, track.key_pre.dtype)
    fixed6_v_hat = rvq_dequantize_tensor_cache(v6, track.value_pre.dtype)
    chunk_ids = chunk_ids_for_track(
        track, chunk_tokens=chunk_tokens, token_count=int(track.key_pre.shape[2])
    )
    coarse_k_depths, coarse_k_scores = coarse_residual_depths(
        track.key_pre,
        packed_k,
        chunk_ids=chunk_ids,
        target_average_depth=target_depth,
    )
    coarse_v_depths, coarse_v_scores = coarse_residual_depths(
        track.value_pre,
        packed_v,
        chunk_ids=chunk_ids,
        target_average_depth=target_depth,
    )
    coarse_k_hat = decode_tiered(
        packed_k,
        chunk_ids=chunk_ids,
        depths=coarse_k_depths,
        output_dtype=track.key_pre.dtype,
    )
    coarse_v_hat = decode_tiered(
        packed_v,
        chunk_ids=chunk_ids,
        depths=coarse_v_depths,
        output_dtype=track.value_pre.dtype,
    )
    assert track.key_rope_cos is not None and track.key_rope_sin is not None
    fixed8_k_post = key_hat_post
    fixed6_k_post = apply_saved_rope(fixed6_k_hat, track.key_rope_cos, track.key_rope_sin)
    progressive_k_post = apply_saved_rope(
        progressive_k_hat, track.key_rope_cos, track.key_rope_sin
    )
    coarse_k_post = apply_saved_rope(coarse_k_hat, track.key_rope_cos, track.key_rope_sin)
    dense_output = F.scaled_dot_product_attention(
        track.query_post_rope, track.key_post_rope, track.value_post_rope
    )
    fixed8_output = F.scaled_dot_product_attention(
        track.query_post_rope, fixed8_k_post, value_hat
    )
    fixed6_output = F.scaled_dot_product_attention(
        track.query_post_rope, fixed6_k_post, fixed6_v_hat
    )
    progressive_output = F.scaled_dot_product_attention(
        track.query_post_rope, progressive_k_post, progressive_v_hat
    )
    coarse_output = F.scaled_dot_product_attention(
        track.query_post_rope, coarse_k_post, coarse_v_hat
    )
    result: dict[str, Any] = {
        "thresholds": {"k": threshold_k, "v": threshold_v, "target_average_depth": target_depth},
        "observed_average_depth": {
            "k": float(progressive_k.get("avg_depth", float("nan"))),
            "v": float(progressive_v.get("avg_depth", float("nan"))),
        },
        "storage": {
            "fixed_l8_k": payload_storage(packed_k),
            "fixed_l8_v": payload_storage(packed_v),
            "fixed_l6_k": payload_storage(k6),
            "fixed_l6_v": payload_storage(v6),
            "token_progressive_k": payload_storage(progressive_k),
            "token_progressive_v": payload_storage(progressive_v),
            "coarse_chunk_k": logical_tier_storage(
                heads=int(track.key_pre.shape[1]),
                tokens=int(track.key_pre.shape[2]),
                chunk_ids=chunk_ids,
                depths=coarse_k_depths,
            ),
            "coarse_chunk_v": logical_tier_storage(
                heads=int(track.value_pre.shape[1]),
                tokens=int(track.value_pre.shape[2]),
                chunk_ids=chunk_ids,
                depths=coarse_v_depths,
            ),
            "note": (
                "logical index bytes assume 6-bit packing; physical bytes are the current uint8 tensors. "
                "Break-even must be evaluated separately for both representations."
            ),
        },
        "timing": {
            "progressive_encode_k_single_sample": encode_k_timing,
            "progressive_encode_v_single_sample": encode_v_timing,
            "fixed_l8_decode_k": benchmark(
                lambda: rvq_dequantize_tensor_cache(packed_k, track.key_pre.dtype), warmup=warmup, repeat=repeat, device=device
            ),
            "fixed_l8_decode_v": benchmark(
                lambda: rvq_dequantize_tensor_cache(packed_v, track.value_pre.dtype), warmup=warmup, repeat=repeat, device=device
            ),
            "fixed_l6_decode_k": benchmark(
                lambda: rvq_dequantize_tensor_cache(k6, track.key_pre.dtype), warmup=warmup, repeat=repeat, device=device
            ),
            "fixed_l6_decode_v": benchmark(
                lambda: rvq_dequantize_tensor_cache(v6, track.value_pre.dtype), warmup=warmup, repeat=repeat, device=device
            ),
            "token_progressive_decode_k": benchmark(
                lambda: rvq_dequantize_tensor_cache(progressive_k, track.key_pre.dtype), warmup=warmup, repeat=repeat, device=device
            ),
            "token_progressive_decode_v": benchmark(
                lambda: rvq_dequantize_tensor_cache(progressive_v, track.value_pre.dtype), warmup=warmup, repeat=repeat, device=device
            ),
            "coarse_chunk_decode_k": benchmark(
                lambda: decode_tiered(
                    packed_k,
                    chunk_ids=chunk_ids,
                    depths=coarse_k_depths,
                    output_dtype=track.key_pre.dtype,
                ),
                warmup=warmup,
                repeat=repeat,
                device=device,
            ),
            "coarse_chunk_decode_v": benchmark(
                lambda: decode_tiered(
                    packed_v,
                    chunk_ids=chunk_ids,
                    depths=coarse_v_depths,
                    output_dtype=track.value_pre.dtype,
                ),
                warmup=warmup,
                repeat=repeat,
                device=device,
            ),
        },
        "coarse_chunk_policy": {
            "chunk_tokens": int(chunk_tokens),
            "objective": "per-chunk L6 residual RMS",
            "allowed_depths": [4, 6, 8],
            "k_depths": coarse_k_depths,
            "v_depths": coarse_v_depths,
            "k_residual_rms": coarse_k_scores,
            "v_residual_rms": coarse_v_scores,
        },
        "pre_space_reconstruction": {
            "fixed_l8_k_relative_l2": relative_l2(fixed8_k_hat_pre, track.key_pre),
            "fixed_l8_v_relative_l2": relative_l2(value_hat, track.value_pre),
            "fixed_l6_k_relative_l2": relative_l2(fixed6_k_hat, track.key_pre),
            "fixed_l6_v_relative_l2": relative_l2(fixed6_v_hat, track.value_pre),
            "progressive_k_relative_l2": relative_l2(progressive_k_hat, track.key_pre),
            "progressive_v_relative_l2": relative_l2(progressive_v_hat, track.value_pre),
            "coarse_chunk_k_relative_l2": relative_l2(coarse_k_hat, track.key_pre),
            "coarse_chunk_v_relative_l2": relative_l2(coarse_v_hat, track.value_pre),
        },
        "post_rope_attention": {
            "fixed_l8": {
                "qk_kl": attention_kl(track.query_post_rope, fixed8_k_post, track.key_post_rope),
                "pv": output_metrics(fixed8_output, dense_output),
            },
            "fixed_l6": {
                "qk_kl": attention_kl(track.query_post_rope, fixed6_k_post, track.key_post_rope),
                "pv": output_metrics(fixed6_output, dense_output),
            },
            "token_progressive": {
                "qk_kl": attention_kl(
                    track.query_post_rope, progressive_k_post, track.key_post_rope
                ),
                "pv": output_metrics(progressive_output, dense_output),
            },
            "coarse_chunk": {
                "qk_kl": attention_kl(track.query_post_rope, coarse_k_post, track.key_post_rope),
                "pv": output_metrics(coarse_output, dense_output),
            },
        },
        "sensitivity_correlation_l8": error_sensitivity_correlations(
            track.query_post_rope,
            track.key_post_rope,
            track.value_post_rope,
            key_hat_post,
            value_hat,
        ),
    }
    # Physical break-even: fixed L8 uses 8 uint8 indices/vector.  Progressive
    # uses avg_depth uint8 indices plus an actual uint8 active_depth/vector and
    # offset tensors, so it needs avg_depth < 7 even before offsets.  Logical
    # 6-bit packing gives 0.75*d + 1 < 6 -> d < 6.6667.
    result["break_even"] = {
        "logical_6bit_with_one_depth_byte": 20.0 / 3.0,
        "physical_uint8_with_one_depth_byte_before_offsets": 7.0,
        "observed_physical_includes_offsets": True,
    }
    return result


def evaluate_h3(
    track: Track,
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
    *,
    block_tokens: int,
) -> dict[str, Any]:
    candidates, parameter_stats = calibration_candidates(
        track.query_post_rope,
        track.key_post_rope,
        key_hat,
        block_tokens=block_tokens,
    )
    rows = evaluate_calibrations(
        track.query_post_rope,
        track.key_post_rope,
        value_hat,
        candidates,
    )
    return {"parameter_stats": parameter_stats, "methods": rows}


def evaluate_h4_h5(
    track: Track,
    packed_k: dict[str, Any],
    packed_v: dict[str, Any],
    key_hat: torch.Tensor,
    value_hat: torch.Tensor,
    *,
    fractions: Sequence[float],
    page_size: int,
    chunk_tokens: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    q, k, v = track.query_post_rope, track.key_post_rope, track.value_post_rope
    assert track.key_rope_cos is not None and track.key_rope_sin is not None
    eligible, reason = contiguous_page_eligible(track, int(k.shape[2]), page_size)
    if not eligible:
        unavailable = {"status": "unavailable", "reason": reason}
        return unavailable, unavailable.copy()
    tokens = int(k.shape[2] // page_size * page_size)
    q, k, v = q, k[:, :, :tokens], v[:, :, :tokens]
    key_hat, value_hat = key_hat[:, :, :tokens], value_hat[:, :, :tokens]
    probs = attention_probs(q, k)
    mass = page_mass(probs, page_size)
    page_k_hat = page_means(key_hat, page_size)
    route_score = torch.matmul(q.float(), page_k_hat.float().transpose(-1, -2))
    dense_ref = F.scaled_dot_product_attention(q, k, v)
    grouped = F.scaled_dot_product_attention(
        q,
        page_means(key_hat, page_size),
        page_means(value_hat, page_size),
        attn_mask=torch.full(
            (q.shape[0], q.shape[1], q.shape[2], tokens // page_size),
            math.log(page_size),
            device=device,
            dtype=q.dtype,
        ),
    )
    h4: dict[str, Any] = {
        "page_size": page_size,
        "group_mean_output": output_metrics(grouped, dense_ref),
        "fractions": {},
    }
    h5: dict[str, Any] = {
        "page_size": page_size,
        "effective_key_count": float(torch.exp(-(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)).mean().item()),
        "temporal_attention_profile": temporal_attention_profile(
            probs,
            chunk_tokens=chunk_tokens,
        ),
        "fractions": {},
        "scope": "full contiguous saved key sample",
    }
    pages = tokens // page_size
    for fraction in fractions:
        count = max(1, int(math.ceil(pages * float(fraction))))
        oracle = top_page_indices(mass, count)
        routed = top_page_indices(route_score, count)
        routed_output = selected_attention(q, key_hat, value_hat, routed, page_size)
        hybrid = hybrid_page_tail_attention(q, key_hat, value_hat, routed, page_size)

        def route_decode_rope_attention() -> torch.Tensor:
            scores = torch.matmul(q.float(), page_k_hat.float().transpose(-1, -2))
            selected = top_page_indices(scores, count)
            token_ids = pages_to_tokens(selected, page_size)
            restored_k_pre = decode_selected_tokens(
                packed_k, token_ids, output_dtype=q.dtype
            )
            restored_v = decode_selected_tokens(packed_v, token_ids, output_dtype=q.dtype)
            restored_k = apply_saved_rope(
                restored_k_pre,
                gather_rope_metadata(track.key_rope_cos[:, :, :tokens], token_ids),
                gather_rope_metadata(track.key_rope_sin[:, :, :tokens], token_ids),
            )
            return F.scaled_dot_product_attention(q, restored_k, restored_v)

        metrics = selection_metrics(routed, oracle, mass)
        metrics.update(
            {
                "selection_fraction": float(count / pages),
                "selected_output": output_metrics(routed_output, dense_ref),
                "grouped_tail_output": output_metrics(hybrid, dense_ref),
                "route_selected_index_decode_rope_attention_timing": benchmark(
                    route_decode_rope_attention, warmup=warmup, repeat=repeat, device=device
                ),
            }
        )
        dense_ms = benchmark(
            lambda: F.scaled_dot_product_attention(q, k, v), warmup=warmup, repeat=repeat, device=device
        )
        metrics["dense_sdpa_timing"] = dense_ms
        metrics["feasible_gate"] = bool(
            metrics["selected_output"]["relative_l2"] <= 0.15
            and metrics["route_selected_index_decode_rope_attention_timing"]["median_ms"]
            < dense_ms["median_ms"]
        )
        h4["fractions"][str(fraction)] = metrics

        policies = temporal_page_policies(
            pages,
            count,
            device=device,
            bsz=int(q.shape[0]),
            heads=int(q.shape[1]),
        )
        policies["content_routing"] = routed
        policies["oracle_top_attention"] = oracle
        policy_rows: dict[str, Any] = {}
        for name, selected in policies.items():
            output = selected_attention(q, key_hat, value_hat, selected, page_size)
            policy_rows[name] = {
                **selection_metrics(selected, oracle, mass),
                "output": output_metrics(output, dense_ref),
            }
        h5["fractions"][str(fraction)] = policy_rows
    return h4, h5


def evaluate_h6(
    track: Track,
    packed_k: dict[str, Any],
    packed_v: dict[str, Any],
    key_hat_post: torch.Tensor,
    value_hat: torch.Tensor,
    *,
    fractions: Sequence[float],
    page_size: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate standard-RoPE and camera-PRoPE as separate routing halves."""
    assert track.query_pre is not None and track.key_pre is not None
    assert track.key_rope_cos is not None and track.key_rope_sin is not None
    q_pre = track.query_pre
    eligible, reason = contiguous_page_eligible(
        track, int(track.key_post_rope.shape[2]), page_size
    )
    replay = camera_prope_replay(track)
    result: dict[str, Any] = {}
    if not eligible:
        result["standard_rope"] = {"status": "unavailable", "reason": reason}
        result["camera_prope"] = {
            "status": "unavailable",
            "reason": reason,
            "replay": replay,
        }
        return result

    tokens = int(track.key_post_rope.shape[2] // page_size * page_size)
    pages = tokens // page_size
    key_hat_pre = rvq_dequantize_tensor_cache(packed_k, track.key_pre.dtype)[:, :, :tokens]
    value_hat_pre = rvq_dequantize_tensor_cache(packed_v, track.value_pre.dtype)[:, :, :tokens]
    pre_summary = page_means(key_hat_pre, page_size)
    center = torch.arange(pages, device=device) * page_size + page_size // 2

    def evaluate_half(
        *,
        half_name: str,
        query_post: torch.Tensor,
        key_post_true: torch.Tensor,
        value_post_true: torch.Tensor,
        key_post_reconstructed: torch.Tensor,
        value_post_reconstructed: torch.Tensor,
        page_center_transform: Callable[[torch.Tensor], torch.Tensor],
        selected_transform: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        full_transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
        output_transform: Callable[[torch.Tensor], torch.Tensor],
        replay_payload: dict[str, Any],
    ) -> dict[str, Any]:
        key_post_true = key_post_true[:, :, :tokens]
        value_post_true = value_post_true[:, :, :tokens]
        key_post_reconstructed = key_post_reconstructed[:, :, :tokens]
        value_post_reconstructed = value_post_reconstructed[:, :, :tokens]
        probs = attention_probs(query_post, key_post_true)
        mass = page_mass(probs, page_size)
        dense_ref = output_transform(
            F.scaled_dot_product_attention(query_post, key_post_true, value_post_true)
        )
        position_summary = page_center_transform(pre_summary)
        post_summary = page_means(key_post_reconstructed, page_size)

        def route_score(variant: str) -> torch.Tensor:
            if variant == "naive_pre_rope":
                return torch.matmul(q_pre.float(), pre_summary.float().transpose(-1, -2))
            if variant == "position_aware_page_center":
                return torch.matmul(
                    query_post.float(), position_summary.float().transpose(-1, -2)
                )
            if variant == "post_rope_upper_bound":
                return torch.matmul(
                    query_post.float(), post_summary.float().transpose(-1, -2)
                )
            raise ValueError(f"unknown routing variant: {variant}")

        dense_timing = benchmark(
            lambda: output_transform(
                F.scaled_dot_product_attention(query_post, key_post_true, value_post_true)
            ),
            warmup=warmup,
            repeat=repeat,
            device=device,
        )
        full_position_timing = benchmark(
            lambda: full_transform(key_hat_pre, value_hat_pre),
            warmup=warmup,
            repeat=repeat,
            device=device,
        )

        def full_decode_position_attention() -> torch.Tensor:
            decoded_k = rvq_dequantize_tensor_cache(packed_k, query_post.dtype)[:, :, :tokens]
            decoded_v = rvq_dequantize_tensor_cache(packed_v, query_post.dtype)[:, :, :tokens]
            restored_k, restored_v = full_transform(decoded_k, decoded_v)
            return output_transform(
                F.scaled_dot_product_attention(query_post, restored_k, restored_v)
            )

        half: dict[str, Any] = {
            "status": "available",
            "reason": reason,
            "replay": replay_payload,
            "dense_sdpa_timing": dense_timing,
            "dense_bf16_position_attention_timing": dense_timing,
            "full_position_transform_timing": full_position_timing,
            "full_decode_position_attention_timing": benchmark(
                full_decode_position_attention,
                warmup=warmup,
                repeat=repeat,
                device=device,
            ),
            "fractions": {},
        }
        if half_name == "standard_rope":
            half["full_saved_rope_timing"] = full_position_timing

        for fraction in fractions:
            count = max(1, int(math.ceil(pages * float(fraction))))
            oracle = top_page_indices(mass, count)
            rows: dict[str, Any] = {}
            for variant in (
                "naive_pre_rope",
                "position_aware_page_center",
                "post_rope_upper_bound",
            ):
                selected = top_page_indices(route_score(variant), count)
                token_ids = pages_to_tokens(selected, page_size)
                candidate = output_transform(
                    selected_attention(
                        query_post,
                        key_post_reconstructed,
                        value_post_reconstructed,
                        selected,
                        page_size,
                    )
                )

                def selected_only_position(ids: torch.Tensor = token_ids) -> tuple[torch.Tensor, torch.Tensor]:
                    return selected_transform(
                        gather_tokens(key_hat_pre, ids),
                        gather_tokens(value_hat_pre, ids),
                        ids,
                    )

                def total_selected_path(active_variant: str = variant) -> torch.Tensor:
                    chosen = top_page_indices(route_score(active_variant), count)
                    ids = pages_to_tokens(chosen, page_size)
                    selected_k_pre = decode_selected_tokens(
                        packed_k, ids, output_dtype=query_post.dtype
                    )
                    selected_v_pre = decode_selected_tokens(
                        packed_v, ids, output_dtype=query_post.dtype
                    )
                    selected_k, selected_v = selected_transform(
                        selected_k_pre, selected_v_pre, ids
                    )
                    return output_transform(
                        F.scaled_dot_product_attention(
                            query_post, selected_k, selected_v
                        )
                    )

                position_timing = benchmark(
                    selected_only_position,
                    warmup=warmup,
                    repeat=repeat,
                    device=device,
                )
                total_timing = benchmark(
                    total_selected_path,
                    warmup=warmup,
                    repeat=repeat,
                    device=device,
                )
                row = {
                    **selection_metrics(selected, oracle, mass),
                    "output": output_metrics(candidate, dense_ref),
                    "selected_only_position_transform_timing": position_timing,
                    "route_selected_decode_position_attention_timing": total_timing,
                }
                # Preserve the pre-camera schema for old report consumers.
                if half_name == "standard_rope":
                    row["selected_only_saved_rope_timing"] = position_timing
                    row["route_selected_decode_rope_attention_timing"] = total_timing
                row["feasible_gate"] = bool(
                    row["output"]["relative_l2"] <= 0.15
                    and total_timing["median_ms"] < dense_timing["median_ms"]
                )
                rows[variant] = row
            half["fractions"][str(fraction)] = rows
        return half

    center_ids = center[None, None, :].expand(
        q_pre.shape[0], q_pre.shape[1], -1
    )

    def standard_page_center(summary: torch.Tensor) -> torch.Tensor:
        return apply_saved_rope(
            summary,
            gather_rope_metadata(track.key_rope_cos[:, :, :tokens], center_ids),
            gather_rope_metadata(track.key_rope_sin[:, :, :tokens], center_ids),
        )

    def standard_selected(
        selected_k: torch.Tensor, selected_v: torch.Tensor, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_saved_rope(
                selected_k,
                gather_rope_metadata(track.key_rope_cos[:, :, :tokens], ids),
                gather_rope_metadata(track.key_rope_sin[:, :, :tokens], ids),
            ),
            selected_v,
        )

    def standard_full(
        restored_k: torch.Tensor, restored_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_saved_rope(
                restored_k,
                track.key_rope_cos[:, :, :tokens],
                track.key_rope_sin[:, :, :tokens],
            ),
            restored_v,
        )

    result["standard_rope"] = evaluate_half(
        half_name="standard_rope",
        query_post=track.query_post_rope,
        key_post_true=track.key_post_rope,
        value_post_true=track.value_post_rope,
        key_post_reconstructed=key_hat_post,
        value_post_reconstructed=value_hat,
        page_center_transform=standard_page_center,
        selected_transform=standard_selected,
        full_transform=standard_full,
        output_transform=lambda output: output,
        replay_payload={"status": "available", "exact_replay": True, "reason": track.accuracy_reason},
    )

    if replay.get("status") != "available" or replay.get("exact_replay") is not True:
        result["camera_prope"] = {
            "status": "unavailable",
            "reason": replay.get("reason", track.camera_prope_reason),
            "replay": replay,
        }
        return result

    assert track.query_post_prope is not None
    assert track.key_post_prope is not None and track.value_post_prope is not None
    assert track.query_viewmats is not None and track.key_viewmats is not None
    _, full_kv_matrix, _ = camera_prope_matrices(
        track.key_viewmats[:, :tokens],
        None if track.key_Ks is None else track.key_Ks[:, :tokens],
    )
    _, _, query_output_matrix = camera_prope_matrices(
        track.query_viewmats,
        track.query_Ks,
    )
    camera_key_hat = apply_camera_prope_matrix(key_hat_pre, full_kv_matrix)
    camera_value_hat = apply_camera_prope_matrix(value_hat_pre, full_kv_matrix)

    center_viewmats = track.key_viewmats[:, center]
    center_Ks = None if track.key_Ks is None else track.key_Ks[:, center]
    _, center_kv_matrix, _ = camera_prope_matrices(center_viewmats, center_Ks)

    def camera_page_center(summary: torch.Tensor) -> torch.Tensor:
        return apply_camera_prope_matrix(summary, center_kv_matrix)

    def camera_selected(
        selected_k: torch.Tensor, selected_v: torch.Tensor, ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected_viewmats = gather_camera_metadata(track.key_viewmats[:, :tokens], ids)
        selected_Ks = gather_camera_metadata(
            None if track.key_Ks is None else track.key_Ks[:, :tokens], ids
        )
        assert selected_viewmats is not None
        _, matrix, _ = camera_prope_matrices(selected_viewmats, selected_Ks)
        return (
            apply_camera_prope_matrix(selected_k, matrix),
            apply_camera_prope_matrix(selected_v, matrix),
        )

    def camera_full(
        restored_k: torch.Tensor, restored_v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            apply_camera_prope_matrix(restored_k, full_kv_matrix),
            apply_camera_prope_matrix(restored_v, full_kv_matrix),
        )

    result["camera_prope"] = evaluate_half(
        half_name="camera_prope",
        query_post=track.query_post_prope,
        key_post_true=track.key_post_prope,
        value_post_true=track.value_post_prope,
        key_post_reconstructed=camera_key_hat,
        value_post_reconstructed=camera_value_hat,
        page_center_transform=camera_page_center,
        selected_transform=camera_selected,
        full_transform=camera_full,
        output_transform=lambda output: apply_camera_prope_matrix(output, query_output_matrix),
        replay_payload=replay,
    )
    return result


def evaluate_h7(
    track: Track,
    packed_k: dict[str, Any],
    packed_v: dict[str, Any],
    *,
    chunk_tokens: int,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> dict[str, Any]:
    assert track.key_pre is not None and track.value_pre is not None
    assert track.key_rope_cos is not None and track.key_rope_sin is not None
    tokens = int(track.key_pre.shape[2])
    chunk_ids = chunk_ids_for_track(track, chunk_tokens=chunk_tokens, token_count=tokens)
    num_chunks = int(chunk_ids.max().item()) + 1 if chunk_ids.numel() else 0
    depths = tier_depths(num_chunks)
    k6 = decode_prefix(packed_k, 6, output_dtype=track.key_pre.dtype)
    v6 = decode_prefix(packed_v, 6, output_dtype=track.value_pre.dtype)
    kt = decode_tiered(packed_k, chunk_ids=chunk_ids, depths=depths, output_dtype=track.key_pre.dtype)
    vt = decode_tiered(packed_v, chunk_ids=chunk_ids, depths=depths, output_dtype=track.value_pre.dtype)
    k6_post = apply_saved_rope(k6, track.key_rope_cos, track.key_rope_sin)
    kt_post = apply_saved_rope(kt, track.key_rope_cos, track.key_rope_sin)
    q = track.query_post_rope
    dense_ref = F.scaled_dot_product_attention(q, track.key_post_rope, track.value_post_rope)
    out6 = F.scaled_dot_product_attention(q, k6_post, v6)
    outt = F.scaled_dot_product_attention(q, kt_post, vt)
    fixed6_timing = benchmark(
        lambda: (
            decode_prefix(packed_k, 6, output_dtype=track.key_pre.dtype),
            decode_prefix(packed_v, 6, output_dtype=track.value_pre.dtype),
        ),
        warmup=warmup,
        repeat=repeat,
        device=device,
    )
    tiered_timing = benchmark(
        lambda: (
            decode_tiered(packed_k, chunk_ids=chunk_ids, depths=depths, output_dtype=track.key_pre.dtype),
            decode_tiered(packed_v, chunk_ids=chunk_ids, depths=depths, output_dtype=track.value_pre.dtype),
        ),
        warmup=warmup,
        repeat=repeat,
        device=device,
    )
    overhead = tiered_timing["median_ms"] / fixed6_timing["median_ms"] - 1.0
    fixed_storage = {
        "average_depth": 6.0,
        "logical_index_bytes": math.ceil(tokens * track.key_pre.shape[1] * 6 * 6 / 8),
        "physical_uint8_index_bytes": tokens * track.key_pre.shape[1] * 6,
        "coarse_schedule_bytes": 0,
        "per_token_scale_bytes": 0,
    }
    tier_storage = logical_tier_storage(
        heads=int(track.key_pre.shape[1]),
        tokens=tokens,
        chunk_ids=chunk_ids,
        depths=depths,
    )
    pv6 = output_metrics(out6, dense_ref)
    pvt = output_metrics(outt, dense_ref)
    gate = bool(
        pvt["relative_l2"] <= pv6["relative_l2"]
        and overhead <= HYPOTHESES["H7"]["offline_gate"]["decode_overhead_max_fraction"]
    )
    return {
        "chunk_tokens": chunk_tokens,
        "num_observed_chunks": num_chunks,
        "depth_schedule": depths,
        "storage_per_tensor": {"uniform_l6": fixed_storage, "tiered": tier_storage},
        "pre_space_reconstruction": {
            "uniform_l6_k_mse": mse(k6, track.key_pre),
            "uniform_l6_v_mse": mse(v6, track.value_pre),
            "tiered_k_mse": mse(kt, track.key_pre),
            "tiered_v_mse": mse(vt, track.value_pre),
        },
        "post_rope_attention_output": {"uniform_l6": pv6, "tiered": pvt},
        "decode_timing": {
            "uniform_l6": fixed6_timing,
            "tiered": tiered_timing,
            "tiered_overhead_fraction": float(overhead),
        },
        "offline_gate_pass": gate,
        "video_smoke_authorized": gate,
    }


def run_track(args: argparse.Namespace, track: Track) -> dict[str, Any]:
    heads = int(track.key_post_rope.shape[1])
    dim = int(track.key_post_rope.shape[-1])
    codebooks = {
        "k": validate_codebook_dir(args.k_codebook_dir, layer=track.layer, heads=heads, expected_dim=dim),
        "v": validate_codebook_dir(args.v_codebook_dir, layer=track.layer, heads=heads, expected_dim=dim),
    }
    sections = {item.strip().upper() for item in str(args.sections).split(",") if item.strip()}
    result: dict[str, Any] = {
        "source": str(track.path),
        "source_sha256": sha256_file(track.path),
        "layer": track.layer,
        "metadata": jsonable(track.metadata),
        "shape": {
            "query": list(track.query_post_rope.shape),
            "key": list(track.key_post_rope.shape),
            "value": list(track.value_post_rope.shape),
        },
        "paired": track.paired,
        "standard_rope_accuracy_valid": track.accuracy_valid,
        "accuracy_reason": track.accuracy_reason,
        "camera_prope": {
            "present_in_dump": track.camera_prope_present,
            "accuracy_status": "pending_h6_replay" if "H6" in sections else "not_requested",
            "reason": track.camera_prope_reason,
        },
        "codebooks": codebooks,
    }
    packed_k = packed_v = None
    key_hat = value_hat = None
    if track.accuracy_valid:
        assert track.key_pre is not None and track.value_pre is not None
        packed_k = encode_fixed(
            track.key_pre,
            layer=track.layer,
            directory=args.k_codebook_dir,
            batch_size=args.encode_batch_size,
        )
        packed_v = encode_fixed(
            track.value_pre,
            layer=track.layer,
            directory=args.v_codebook_dir,
            batch_size=args.encode_batch_size,
        )
        assert track.key_rope_cos is not None and track.key_rope_sin is not None
        key_hat = apply_saved_rope(
            rvq_dequantize_tensor_cache(packed_k, track.key_pre.dtype),
            track.key_rope_cos,
            track.key_rope_sin,
        )
        value_hat = rvq_dequantize_tensor_cache(packed_v, track.value_pre.dtype)
        result["packed_invariants"] = {
            "k_no_scale": True,
            "v_no_scale": True,
            "input_norm_mode": "none",
            "resident_codebook_storage": "int4",
            "runtime_strategy": "persistent materialization/restoration measured separately",
        }

    if "H1" in sections:
        result["H1"] = evaluate_h1(
            track,
            packed_k,
            packed_v,
            warmup=args.warmup,
            repeat=args.repeat,
            device=track.query_post_rope.device,
            exact_reference_max_elements=args.exact_reference_max_elements,
            exact_key_block_tokens=args.exact_key_block_tokens,
        )
    if track.accuracy_valid:
        assert packed_k is not None and packed_v is not None and key_hat is not None and value_hat is not None
        if "H2" in sections:
            result["H2"] = evaluate_h2(
                track,
                packed_k,
                packed_v,
                key_hat,
                value_hat,
                k_dir=args.k_codebook_dir,
                v_dir=args.v_codebook_dir,
                batch_size=args.encode_batch_size,
                target_depth=args.progressive_target_depth,
                chunk_tokens=args.chunk_tokens,
                warmup=args.warmup,
                repeat=args.repeat,
                device=track.query_post_rope.device,
            )
        if "H3" in sections:
            result["H3"] = evaluate_h3(track, key_hat, value_hat, block_tokens=args.chunk_tokens)
        if "H6" in sections:
            result["H6"] = evaluate_h6(
                track,
                packed_k,
                packed_v,
                key_hat,
                value_hat,
                fractions=[float(item) for item in str(args.fractions).split(",") if item.strip()],
                page_size=args.page_size,
                warmup=args.warmup,
                repeat=args.repeat,
                device=track.query_post_rope.device,
            )
            camera_half = result["H6"].get("camera_prope", {})
            camera_replay = camera_half.get("replay", {})
            result["camera_prope"] = {
                "present_in_dump": track.camera_prope_present,
                "accuracy_status": camera_replay.get("status", camera_half.get("status", "unavailable")),
                **camera_replay,
            }
        if "H4" in sections or "H5" in sections:
            h4, h5 = evaluate_h4_h5(
                track,
                packed_k,
                packed_v,
                key_hat,
                value_hat,
                fractions=[float(item) for item in str(args.fractions).split(",") if item.strip()],
                page_size=args.page_size,
                chunk_tokens=args.chunk_tokens,
                warmup=args.warmup,
                repeat=args.repeat,
                device=track.query_post_rope.device,
            )
            if "H4" in sections:
                result["H4"] = h4
            if "H5" in sections:
                result["H5"] = h5
        if "H7" in sections:
            result["H7"] = evaluate_h7(
                track,
                packed_k,
                packed_v,
                chunk_tokens=args.chunk_tokens,
                warmup=args.warmup,
                repeat=args.repeat,
                device=track.query_post_rope.device,
            )
    else:
        for section in sections - {"H1"}:
            result[section] = {"status": "unavailable", "reason": track.accuracy_reason}
    return result


def default_samples() -> list[Path]:
    paired_root = REPO_ROOT / "results/paired_rope_diag"
    paired = sorted(paired_root.glob("rank_*/layer_*/call_*.pt"))
    if paired:
        return paired
    return sorted((REPO_ROOT / "results/attention_diag_qkv_smoke").glob("rank_*/layer_*/call_*.pt"))


def validate_args(args: argparse.Namespace) -> tuple[torch.device, list[Path]]:
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("warmup must be non-negative and repeat must be positive")
    if args.exact_key_block_tokens <= 0 or args.exact_reference_max_elements <= 0:
        raise ValueError("H1 exact block size and element guard must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")
    samples = list(args.sample) or default_samples()
    if not samples:
        raise FileNotFoundError("No paired or legacy attention diagnostic samples found")
    for sample in samples:
        if not sample.exists():
            raise FileNotFoundError(sample)
    if not args.k_codebook_dir.is_dir() or not args.v_codebook_dir.is_dir():
        raise FileNotFoundError("K/V codebook directories must exist")
    return device, samples


def make_run_dir(root: Path, tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = f"_{tag}" if tag else ""
    path = root / f"{stamp}_{BASELINE}{suffix}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_canonical_evidence(
    run_dir: Path,
    *,
    results: Sequence[dict[str, Any]],
    samples: Sequence[Path],
    sections: set[str],
) -> list[Path]:
    """Write one report-aggregator record per preregistered hypothesis."""
    output_dir = run_dir / "canonical"
    output_dir.mkdir(exist_ok=False)
    paths: list[Path] = []
    layers = sorted({int(item["layer"]) for item in results})
    experiment_ids = {
        "H1": "h1_real_post_rope_attention_timing",
        "H2": "h2_fixed_progressive_control",
        "H3": "h3_radial_angular_decomposition",
        "H4": "h4_real_post_rope_codebook_routing",
        "H5": "h5_real_post_rope_temporal_masks",
        "H6": "h6_paired_pre_post_rope_routing",
        "H7": "h7_matched_rate_offline_gate",
    }
    for hypothesis_id in ("H1", "H2", "H3", "H4", "H5", "H6", "H7"):
        rows = [
            {
                "layer": int(item["layer"]),
                "source": item["source"],
                "standard_rope_accuracy_valid": bool(item["standard_rope_accuracy_valid"]),
                "data": item.get(hypothesis_id),
            }
            for item in results
            if hypothesis_id in item
        ]
        unavailable = [
            row
            for row in rows
            if isinstance(row.get("data"), dict)
            and row["data"].get("status") == "unavailable"
        ]
        h1_incomplete = hypothesis_id == "H1" and any(
            not isinstance(row.get("data"), dict)
            or row["data"].get("compressed_paths", {}).get("status") != "measured"
            or row["data"].get("nonmaterializing_exact_compressed", {}).get("status")
            != "measured"
            or row["data"].get("standalone_compressed_qk", {}).get("status")
            != "measured"
            for row in rows
        )
        h6_incomplete = hypothesis_id == "H6" and any(
            not isinstance(row.get("data"), dict)
            or row["data"].get("standard_rope", {}).get("status") != "available"
            or row["data"].get("camera_prope", {}).get("status") != "available"
            or row["data"].get("standard_rope", {}).get("replay", {}).get("exact_replay") is not True
            or row["data"].get("camera_prope", {}).get("replay", {}).get("exact_replay") is not True
            for row in rows
        )
        requested = hypothesis_id in sections
        if not requested or not rows:
            status = "missing"
            verdict_state = "not_measured"
            reason = "hypothesis was not requested or produced no table"
        elif unavailable or h1_incomplete or h6_incomplete:
            # H1 is partial when an exact path is guarded/unavailable; H6 is
            # partial unless both positional halves have exact standalone replay.
            status = "partial"
            verdict_state = "measured_with_unsupported_variant"
            reason = "at least one preregistered transform/variant is explicitly unavailable"
        else:
            status = "complete"
            verdict_state = "measured"
            reason = "all requested offline variants in this runner produced measurements"
        record = {
            "schema_version": "wan5b_report_evidence.v1",
            "hypothesis_id": hypothesis_id,
            "experiment_id": experiment_ids[hypothesis_id],
            "baseline_id": BASELINE,
            "status": status,
            "scope": {
                "model_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "implementation": "HY-WorldPlay",
                "input_norm_mode": "none",
                "rvq_levels": 8,
                "rvq_centroids": 64,
                "codebook_storage_dtype": "int4",
                "stores_token_scale": False,
            },
            "run_info_path": "../run_info.json",
            "tables": [
                {
                    "title": f"{hypothesis_id} raw offline measurements",
                    "columns": ["layer", "source", "standard_rope_accuracy_valid", "data"],
                    "rows": [
                        [
                            row["layer"],
                            row["source"],
                            row["standard_rope_accuracy_valid"],
                            row["data"],
                        ]
                        for row in rows
                    ],
                }
            ],
            "observations": [
                {
                    "prediction_id": f"{hypothesis_id}-P1",
                    "status": "not_tested" if status == "missing" else "inconclusive",
                    "explanation": (
                        f"layer={row['layer']} accuracy_valid={row['standard_rope_accuracy_valid']}; "
                        + (
                            str(row["data"].get("reason", "measurement recorded"))
                            if isinstance(row.get("data"), dict)
                            else "measurement missing"
                        )
                    ),
                }
                for row in rows
            ],
            "verdict": {
                "status": "missing" if status == "missing" else "inconclusive",
                "explanation": (
                    f"{reason}; numerical support/refutation must follow the preregistered thresholds."
                ),
            },
            "artifacts": [str(path) for path in samples],
            "measurement_scope": {
                "layers": layers,
                "position_transform": "saved exact standard RoPE replay plus exact camera-PRoPE replay",
                "camera_prope": "separate naive/position-aware/post-upper-bound routing track",
                "codebook_training_space": "pre_rope_raw_k_and_pre_prope_v",
            },
        }
        path = output_dir / f"{hypothesis_id.lower()}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    device, samples = validate_args(args)
    tracks = [
        load_track(
            path,
            device,
            q_limit=args.query_tokens,
            k_limit=args.max_key_tokens,
        )
        for path in samples
    ]
    # Validation intentionally loads each source and every used codebook header,
    # but creates no result directory.
    if args.validate_only:
        validation = []
        for track in tracks:
            heads, dim = int(track.key_post_rope.shape[1]), int(track.key_post_rope.shape[-1])
            validation.append(
                {
                    "sample": str(track.path),
                    "paired": track.paired,
                    "accuracy_valid": track.accuracy_valid,
                    "accuracy_reason": track.accuracy_reason,
                    "k": validate_codebook_dir(args.k_codebook_dir, layer=track.layer, heads=heads, expected_dim=dim),
                    "v": validate_codebook_dir(args.v_codebook_dir, layer=track.layer, heads=heads, expected_dim=dim),
                }
            )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return

    run_dir = make_run_dir(args.output_root, args.tag)
    hypotheses_path = run_dir / "hypotheses.json"
    hypotheses_path.write_text(json.dumps(HYPOTHESES, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config = {
        "baseline": BASELINE,
        "samples": [str(path) for path in samples],
        "k_codebook_dir": str(args.k_codebook_dir),
        "v_codebook_dir": str(args.v_codebook_dir),
        "device": str(device),
        "seed": args.seed,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "query_tokens": args.query_tokens,
        "max_key_tokens": args.max_key_tokens,
        "page_size": args.page_size,
        "chunk_tokens": args.chunk_tokens,
        "fractions": args.fractions,
        "sections": args.sections,
        "progressive_target_depth": args.progressive_target_depth,
        "exact_reference_max_elements": args.exact_reference_max_elements,
        "exact_key_block_tokens": args.exact_key_block_tokens,
        "exact_streaming_relative_l2_tolerance": EXACT_STREAMING_RELATIVE_L2_TOLERANCE,
        "codebook_storage_dtype": "int4",
        "input_norm_mode": "none",
        "per_token_scale": False,
    }
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    config_hash = sha256_file(config_path)
    start = time.time()
    results: list[dict[str, Any]] = []
    for track in tracks:
        results.append(run_track(args, track))
    summary = {
        "schema_version": 1,
        "baseline": BASELINE,
        "elapsed_seconds": time.time() - start,
        "tracks": results,
        "coverage": {
            "requested_layers": [0, 10, 20, 29],
            "observed_layers": sorted({track.layer for track in tracks}),
            "missing_requested_layers": sorted({0, 10, 20, 29} - {track.layer for track in tracks}),
            "note": "Missing layers are reported, never synthesized from another model/layer.",
        },
    }
    (run_dir / "evidence.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gpu_type = None
    if device.type == "cuda":
        gpu_type = torch.cuda.get_device_name(device)
    layer_hashes = {
        str(item["layer"]): {
            "k": item["codebooks"]["k"]["layer_hash"],
            "v": item["codebooks"]["v"]["layer_hash"],
        }
        for item in results
    }
    prompt_list = [
        str(track.metadata.get("prompt", track.metadata.get("prompt_id", track.path))) for track in tracks
    ]
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "hypotheses_path": str(hypotheses_path),
        "hypotheses_hash": sha256_file(hypotheses_path),
        "codebook_path": {"k": str(args.k_codebook_dir), "v": str(args.v_codebook_dir)},
        "codebook_hash": layer_hashes,
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type,
        "device": str(device),
        "platform": platform.platform(),
        "seed": args.seed,
        "prompt_list": prompt_list,
        "source_paths": [str(path) for path in samples],
        "source_hashes": {str(path): sha256_file(path) for path in samples},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_norm_mode": "none",
        "codebook_storage_dtype": "int4",
        "per_token_scale": False,
        "physical_storage_note": "indices are current uint8 tensors; logical storage separately reports 6-bit packing",
    }
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    canonical_paths = write_canonical_evidence(
        run_dir,
        results=results,
        samples=samples,
        sections={item.strip().upper() for item in str(args.sections).split(",") if item.strip()},
    )
    (run_dir / "canonical_manifest.json").write_text(
        json.dumps({"schema_version": "wan5b_report_evidence.v1", "records": [str(path) for path in canonical_paths]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(run_dir)


if __name__ == "__main__":
    main()
