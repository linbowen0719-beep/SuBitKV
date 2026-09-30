"""Structure-preserving, prefix-truncatable quantization of an RVQ residual.

QuantVideo's practical advantage is not only its final INT2/INT4 scalar
quantizer.  Before scalar quantization it learns an online vector predictor
for every cache head.  This module keeps that predictor, then replaces the
independently encoded scalar suffix with the nested bitplanes implemented in
``truncatable_residual``::

    r = x - RVQ(x)
    r = centroid[token] + epsilon
    r_hat_b = centroid[token] + Q_prefix_b(epsilon)

With 256 BF16 centroids, uint8 token labels, and a G128 FP16 suffix scale, the
local metadata rate matches QuantVideo's G64 FP8 scale rate.  Thus prefix
depths two and four can be compared with QuantVideo INT2 and INT4 at the same
physical payload, while every suffix prefix remains directly decodable.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from . import truncatable_residual as _residual_codec
from .truncatable_residual import (
    decode_residual,
    encode_residual,
    payload_nbytes as residual_payload_nbytes,
    truncate_state as truncate_residual_state,
)


_QUANT_METHOD = "structured_truncatable"
_FAST_GROUP_SIZE = 128


@functools.lru_cache(maxsize=None)
def _fast_signs_cpu(group_size: int, seed: int) -> torch.Tensor:
    """Cache the exact deterministic sign vector used by the eager codec."""

    return torch.tensor(
        _residual_codec._fixed_sign_values(int(group_size), int(seed)),
        dtype=torch.float32,
    )


_FAST_SIGNS_DEVICE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _validate_source(tensor: torch.Tensor, n_clusters: int) -> tuple[int, int, int, int]:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        raise TypeError("tensor must be a floating-point torch.Tensor")
    if tensor.ndim != 4:
        raise ValueError(
            "structured truncatable residual expects [B,H,S,D], got "
            f"{tuple(tensor.shape)}"
        )
    batch, heads, tokens, dim = (int(value) for value in tensor.shape)
    if n_clusters < 2 or n_clusters > 256:
        raise ValueError(f"n_clusters must be in [2,256], got {n_clusters}")
    return batch, heads, tokens, dim


def _gather_centroids(
    centroids: torch.Tensor,
    cluster_ids: torch.Tensor,
) -> torch.Tensor:
    dim = int(centroids.shape[-1])
    index = cluster_ids.long().unsqueeze(-1).expand(*cluster_ids.shape, dim)
    return torch.gather(centroids, dim=2, index=index)


def _subtract_centroids(
    tensor: torch.Tensor,
    cluster_ids: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    # The Triton kernel is materially faster for real video caches.  Keeping a
    # torch fallback makes the state format and unit tests CUDA-independent.
    if tensor.is_cuda:
        from .real.smooth import minus_centroid

        return minus_centroid(tensor, cluster_ids, centroids)
    return (tensor - _gather_centroids(centroids, cluster_ids)).contiguous()


def _online_vector_predictor(
    flattened: torch.Tensor,
    *,
    n_clusters: int,
    max_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    from .kmeans.kmeans_euclid import batch_kmeans_Euclid

    return batch_kmeans_Euclid(
        flattened,
        n_clusters=int(n_clusters),
        max_iters=int(max_iters),
    )


def encode_structured_residual(
    tensor: torch.Tensor,
    *,
    bits: int,
    tensor_name: str,
    params_path: str | Path | Mapping[str, Any],
    scheme: str = "gaussian_tree",
    group_size: int = 128,
    n_clusters: int = 256,
    kmeans_max_iters: int = 2,
    kmeans_seed: int = 42,
    hadamard: bool = False,
    transform_seed: int = 20260722,
    scale_refine_iters: int = 1,
) -> dict[str, Any]:
    """Encode an online vector predictor plus a nested scalar suffix."""

    batch, heads, tokens, dim = _validate_source(tensor, int(n_clusters))
    canonical_tensor_name = str(tensor_name).strip().upper()
    if canonical_tensor_name not in {"K", "V"}:
        raise ValueError(f"tensor_name must be K or V, got {tensor_name!r}")
    effective_kmeans_seed = int(kmeans_seed) + (
        1 if canonical_tensor_name == "V" else 0
    )
    if int(kmeans_max_iters) < 1:
        raise ValueError(
            f"kmeans_max_iters must be positive, got {kmeans_max_iters}"
        )
    if dim % int(group_size):
        raise ValueError(
            f"channel dimension {dim} must be divisible by group_size={group_size}"
        )

    flattened = tensor.reshape(batch * heads, tokens, dim).contiguous()
    devices = [tensor.device.index] if tensor.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        if tensor.is_cuda:
            torch.cuda.manual_seed(effective_kmeans_seed)
        else:
            torch.manual_seed(effective_kmeans_seed)
        cluster_ids, centroids, cluster_sizes, iterations = _online_vector_predictor(
            flattened,
            n_clusters=int(n_clusters),
            max_iters=int(kmeans_max_iters),
        )

    cluster_ids = cluster_ids.reshape(batch, heads, tokens).contiguous()
    centroids = (
        centroids.reshape(batch, heads, int(n_clusters), dim)
        .to(dtype=tensor.dtype)
        .contiguous()
    )
    # Recompute the suffix from the exact centroids that are physically stored.
    suffix = _subtract_centroids(tensor, cluster_ids, centroids)
    suffix_state = encode_residual(
        suffix,
        scheme=scheme,
        bits=int(bits),
        tensor_name=tensor_name,
        params_path=params_path,
        hadamard=bool(hadamard),
        seed=int(transform_seed),
        group_size=int(group_size),
        scale_refine_iters=int(scale_refine_iters),
    )
    return {
        "quant_method": _QUANT_METHOD,
        "schema_version": "structured_truncatable.v1",
        "shape": tuple(int(value) for value in tensor.shape),
        "source_dtype": str(tensor.dtype).replace("torch.", ""),
        "centroids": centroids,
        "cluster_ids": cluster_ids.to(torch.uint8).contiguous(),
        "suffix": suffix_state,
        "n_clusters": int(n_clusters),
        "nonempty_clusters": int((cluster_sizes > 0).sum().item()),
        "minimum_cluster_size": int(cluster_sizes.min().item()),
        "maximum_cluster_size": int(cluster_sizes.max().item()),
        "kmeans_max_iters": int(kmeans_max_iters),
        "kmeans_iterations": int(iterations),
        "kmeans_seed": int(kmeans_seed),
        "effective_kmeans_seed": effective_kmeans_seed,
        "group_size": int(group_size),
        "scale_refine_iters": int(scale_refine_iters),
    }


def _validate_state(state: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(state, Mapping) or state.get("quant_method") != _QUANT_METHOD:
        raise ValueError("Expected a structured_truncatable packed state")
    centroids = state.get("centroids")
    cluster_ids = state.get("cluster_ids")
    if not torch.is_tensor(centroids) or centroids.ndim != 4:
        raise ValueError("structured state requires [B,H,K,D] centroids")
    if (
        not torch.is_tensor(cluster_ids)
        or cluster_ids.dtype != torch.uint8
        or cluster_ids.ndim != 3
    ):
        raise ValueError("structured state requires uint8 [B,H,S] cluster_ids")
    shape = tuple(int(value) for value in state.get("shape", ()))
    expected = (
        int(centroids.shape[0]),
        int(centroids.shape[1]),
        int(cluster_ids.shape[2]),
        int(centroids.shape[3]),
    )
    if shape != expected or tuple(cluster_ids.shape[:2]) != tuple(centroids.shape[:2]):
        raise ValueError(
            f"inconsistent structured state shape={shape}, expected={expected}"
        )
    # Encoder-produced uint8 labels are range-safe by construction.  Retain
    # the stronger check for CPU/deserialized diagnostics without forcing a
    # device synchronization on every online CUDA decode.
    if (
        cluster_ids.device.type == "cpu"
        and int(cluster_ids.max().item()) >= int(centroids.shape[2])
    ):
        raise ValueError("cluster_ids contain an out-of-range centroid index")
    return centroids, cluster_ids


def _fast_decode_unsupported_reason(
    state: Mapping[str, Any],
    centroids: torch.Tensor,
    cluster_ids: torch.Tensor,
    output_dtype: torch.dtype,
) -> str | None:
    """Return why the fused production path cannot decode this state."""

    if not centroids.is_cuda:
        return "predictor centroids are not on CUDA"
    expected_shape = tuple(int(value) for value in state.get("shape", ()))
    expected_predictor_shape = (
        expected_shape[0],
        expected_shape[1],
        256,
        expected_shape[3],
    )
    if (
        centroids.dtype != torch.bfloat16
        or tuple(centroids.shape) != expected_predictor_shape
    ):
        return (
            "predictor centroids are not production BF16 "
            f"{expected_predictor_shape}"
        )
    if tuple(cluster_ids.shape) != expected_shape[:3]:
        return "cluster_ids shape does not match [B,H,S]"
    if output_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return f"output dtype {output_dtype} is unsupported"
    if not centroids.is_contiguous() or not cluster_ids.is_contiguous():
        return "predictor tensors are not contiguous"
    suffix = state.get("suffix")
    if not isinstance(suffix, Mapping):
        return "suffix is not a packed residual mapping"
    scheme, _bits, planes = _residual_codec._validate_state(suffix)
    if scheme != "gaussian_tree":
        return f"suffix scheme {scheme!r} is unsupported"
    if not bool(suffix.get("hadamard", False)):
        return "suffix does not use the sign-Hadamard transform"
    if int(suffix.get("group_size", 0)) != _FAST_GROUP_SIZE:
        return f"suffix group size is not {_FAST_GROUP_SIZE}"
    shape = tuple(int(value) for value in suffix.get("shape", ()))
    if shape != tuple(int(value) for value in state.get("shape", ())):
        return "suffix and structured-state shapes differ"
    if len(shape) != 4 or shape[-1] % _FAST_GROUP_SIZE:
        return "suffix shape is not [B,H,S,D] with D divisible by 128"
    if int(suffix.get("plane_count", -1)) != math.prod(shape):
        return "suffix plane_count does not equal the stored shape product"
    scale = suffix.get("scale")
    if not torch.is_tensor(scale) or not scale.is_cuda or not scale.is_contiguous():
        return "suffix scale is not a contiguous CUDA tensor"
    if scale.dtype != torch.float16:
        return f"suffix scale dtype {scale.dtype} is not float16"
    expected_groups = math.prod(shape) // _FAST_GROUP_SIZE
    if int(scale.numel()) != expected_groups:
        return "suffix scale count is inconsistent with its shape"
    device = centroids.device
    if cluster_ids.device != device or scale.device != device:
        return "structured tensors are on different CUDA devices"
    if any(
        not plane.is_cuda
        or plane.device != device
        or not plane.is_contiguous()
        for plane in planes
    ):
        return "bitplanes are not contiguous tensors on the predictor CUDA device"
    return None


def _decode_structured_residual_fast(
    state: Mapping[str, Any],
    centroids: torch.Tensor,
    cluster_ids: torch.Tensor,
    output_dtype: torch.dtype,
    *,
    algorithm: str = "fwht",
    output_tile: int = 8,
) -> torch.Tensor:
    suffix = state["suffix"]
    _scheme, bits, planes = _residual_codec._validate_state(suffix)
    parameters = _residual_codec._state_params(suffix, planes[0].device)
    suffix_centroids = parameters["centroids"][bits - 1]
    if (
        not torch.is_tensor(suffix_centroids)
        or suffix_centroids.dtype != torch.float32
        or tuple(suffix_centroids.shape) != (1 << bits,)
        or suffix_centroids.device != centroids.device
        or not suffix_centroids.is_contiguous()
    ):
        raise ValueError(
            "fused structured decode requires contiguous float32 Gaussian "
            f"centroids [{1 << bits}] on the predictor device"
        )
    seed = int(suffix.get("seed", 20260722))
    cache_key = (_FAST_GROUP_SIZE, seed, str(centroids.device))
    signs = _FAST_SIGNS_DEVICE_CACHE.get(cache_key)
    if signs is None:
        signs = _fast_signs_cpu(_FAST_GROUP_SIZE, seed).to(
            device=centroids.device, non_blocking=True
        )
        _FAST_SIGNS_DEVICE_CACHE[cache_key] = signs

    from .kernels.triton.structured_truncatable import (
        decode_gaussian_g128_hadamard,
    )

    return decode_gaussian_g128_hadamard(
        planes=planes,
        scale=suffix["scale"],
        suffix_centroids=suffix_centroids,
        predictor_centroids=centroids,
        cluster_ids=cluster_ids,
        signs=signs,
        shape=tuple(int(value) for value in state["shape"]),
        output_dtype=output_dtype,
        algorithm=algorithm,
        output_tile=output_tile,
    )


def decode_structured_residual(
    state: Mapping[str, Any],
    output_dtype: torch.dtype,
    *,
    use_fast_path: bool | None = None,
) -> torch.Tensor:
    """Decode the vector predictor and the physically present suffix prefix.

    The default ``None`` and explicit ``False`` retain the eager reference
    path.  Passing ``True`` opts into the fused CUDA implementation and rejects
    unsupported states instead of silently falling back.  Keeping opt-in
    explicit prevents one multi-process quality run from mixing decoder
    implementations when workers import the module at different times.
    """

    centroids, cluster_ids = _validate_state(state)
    if not isinstance(output_dtype, torch.dtype):
        raise TypeError(f"output_dtype must be a torch.dtype, got {output_dtype!r}")
    if use_fast_path is not None and not isinstance(use_fast_path, bool):
        raise TypeError(
            "use_fast_path must be None, True, or False, got "
            f"{use_fast_path!r}"
        )
    reason = _fast_decode_unsupported_reason(
        state, centroids, cluster_ids, output_dtype
    )
    if use_fast_path is True and reason is not None:
        raise ValueError(f"fused structured decode is unavailable: {reason}")
    if use_fast_path is True:
        # The register-level FWHT preserves the eager transform's operation
        # order and is bit-exact for FP16/BF16 at the production G128 shape.
        # It is also substantially faster than the tiled direct-Hadamard
        # kernel, which remains available as an explicit diagnostic option.
        return _decode_structured_residual_fast(
            state, centroids, cluster_ids, output_dtype
        )

    suffix = decode_residual(state["suffix"], output_dtype)
    predictor = _gather_centroids(centroids, cluster_ids).to(output_dtype)
    return (predictor + suffix).to(output_dtype).contiguous()


def truncate_structured_state(
    state: Mapping[str, Any],
    bits: int,
) -> dict[str, Any]:
    """Drop suffix bitplanes while retaining the mandatory vector predictor."""

    _validate_state(state)
    truncated = dict(state)
    truncated["suffix"] = truncate_residual_state(state["suffix"], int(bits))
    return truncated


def payload_nbytes(state: Mapping[str, Any]) -> int:
    """Physical bytes excluding globally shared nested-codec parameters."""

    centroids, cluster_ids = _validate_state(state)
    total = (
        int(centroids.numel()) * int(centroids.element_size())
        + int(cluster_ids.numel()) * int(cluster_ids.element_size())
        + residual_payload_nbytes(state["suffix"])
    )
    return int(total)


def physical_bits_per_value(state: Mapping[str, Any]) -> float:
    """Return the realized payload rate for the represented residual tensor."""

    shape = tuple(int(value) for value in state["shape"])
    return 8.0 * payload_nbytes(state) / float(math.prod(shape))


__all__ = [
    "decode_structured_residual",
    "encode_structured_residual",
    "payload_nbytes",
    "physical_bits_per_value",
    "truncate_structured_state",
]
