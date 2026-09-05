from functools import lru_cache
from collections.abc import Mapping
import os
import torch
from enum import Enum
import re
import math
from pathlib import Path
from .sim.quant.lowbit_quantize import (
    nvfp4_e2m1_quantize_triton,
    blockwise_int4_quantize_triton,
    blockwise_int3_quantize_triton,
    blockwise_int2_quantize_triton,
)
from .sim.quant.quantize_config import (
    QuantizeConfig,
    parse_layer_spec,
    parse_residual_codec_span_schedule,
)

from .functions import (
    kmeans_quantize_tensor,
    prq_quantize_tensor,
    triton_prq_quantize_tensor,
)
from .rvq import pack_rvq_calibration_scale, rvq_dequantize_tensor_cache, rvq_encode_tensor_cache
from .oscar import oscar_int2_quantize_tensor
from .higgs import higgs_quantize_tensor
from .truncatable_residual import encode_residual
from .routed_gaussian import encode_routed_gaussian_residual
from .uniform_grid_gaussian import encode_uniform_grid_gaussian_residual
from .frame_mean_gaussian import encode_frame_mean_gaussian_residual
from .cache_mean_gaussian import encode_cache_mean_gaussian_residual
from .absmax_scalar import encode_absmax_scalar_residual
from .sign_adaptive_absmax_scalar import (
    encode_sign_adaptive_absmax_scalar_residual,
)
from .progressive_sign_adaptive_absmax_scalar import (
    encode_progressive_sign_adaptive_absmax_scalar_residual,
    truncate_progressive_sign_adaptive_absmax_state,
)
from .temporal_adaptive_residual import create_b16_state, transition_state
from .centered_absmax_scalar import (
    encode_centered_absmax_scalar_residual,
)
from .routed_midrange_scalar import (
    encode_routed_midrange_scalar_residual,
)
from .static_equalized_midrange import (
    encode_static_equalized_midrange_residual,
)
from .selective_rank1_gaussian import (
    encode_selective_rank1_gaussian_residual,
    load_params as load_selective_rank1_params,
)
from .structured_truncatable import encode_structured_residual
from .r2aq import encode_r2aq_residual, owner_shared_budget_nbytes
from .static_prefix_allocation import (
    encode_static_allocated_structured_residual,
)
from .kivi import encode_kivi_tensor
from .turboquant import encode_turboquant_mse, encode_turboquant_prod


_ENCODE_BREAKDOWN_EVENTS: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
_ENCODE_BREAKDOWN_CALLS: dict[str, int] = {}


def reset_encode_breakdown_stats() -> None:
    _ENCODE_BREAKDOWN_EVENTS.clear()
    _ENCODE_BREAKDOWN_CALLS.clear()


def get_encode_breakdown_stats(*, reset: bool = False) -> dict[str, object]:
    seconds: dict[str, float] = {}
    for phase, pairs in _ENCODE_BREAKDOWN_EVENTS.items():
        total_ms = 0.0
        for start, end in pairs:
            end.synchronize()
            total_ms += start.elapsed_time(end)
        seconds[phase] = total_ms / 1000.0
    result: dict[str, object] = {
        "seconds": seconds,
        "calls": dict(_ENCODE_BREAKDOWN_CALLS),
        "instrumentation": "per-call CUDA events",
    }
    if reset:
        reset_encode_breakdown_stats()
    return result


def _start_encode_phase(phase: str, device: torch.device):
    if (
        os.getenv("QVG_CODEC_BREAKDOWN", "0").lower()
        in {"", "0", "false", "no"}
        or device.type != "cuda"
    ):
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    return phase, start, end


def _stop_encode_phase(timer) -> None:
    if timer is None:
        return
    phase, start, end = timer
    end.record()
    _ENCODE_BREAKDOWN_EVENTS.setdefault(phase, []).append((start, end))
    _ENCODE_BREAKDOWN_CALLS[phase] = _ENCODE_BREAKDOWN_CALLS.get(phase, 0) + 1


@lru_cache(maxsize=16)
def _load_routed_gaussian_tree_params(
    path: str, device: str
) -> dict:
    return torch.load(
        Path(path).expanduser().resolve(),
        map_location=torch.device(device),
        weights_only=False,
    )


@lru_cache(maxsize=16)
def _load_selective_rank1_gaussian_params(
    path: str, device: str
) -> dict:
    return load_selective_rank1_params(path, device=torch.device(device))


@lru_cache(maxsize=16)
def _load_progressive_v_absmax_params(path: str) -> dict:
    return torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )


########################################################
# Entrypoints
########################################################

# Make it a enum
class QuantizeFunctions(Enum):
    NAIVE = "naive"
    KMEANS = "kmeans"
    NSTAGE_KMEANS = "prq"
    NSTAGE_KMEANS_CLIP = "prq_clip"
    TRITON_PRQ = "triton_prq"
    TRITON_PRQ_CLIP = "triton_prq_clip"
    SVD_TRITON_PRQ = "svd_triton_prq"
    SVD_TRITON_PRQ_CLIP = "svd_triton_prq_clip"
    OSCAR_INT2 = "oscar_int2"


_SVD_QUANT_RE = re.compile(
    r"^svd(?P<rank>\d+)-(?P<base>triton-nstages-kmeans-int(?:2|4)(?:-clip)?(?:-flc)?)$"
)


def uses_lossless_flc(quant_type: str) -> bool:
    return quant_type.endswith("-flc")


def strip_lossless_flc(quant_type: str) -> str:
    if uses_lossless_flc(quant_type):
        return quant_type[: -len("-flc")]
    return quant_type


def parse_svd_quant_type(quant_type: str) -> tuple[int, str] | None:
    match = _SVD_QUANT_RE.match(quant_type)
    if match is None:
        return None
    rank = int(match.group("rank"))
    if rank < 1:
        raise ValueError(f"SVD rank must be >= 1, got {rank}")
    return rank, match.group("base")


def fp8_e4m3_quantize_tensor_cache(tensor: torch.Tensor) -> dict:
    """Store a KV tensor as FP8 E4M3 with no learned codebook."""
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    values = torch.nan_to_num(
        tensor.float(),
        nan=0.0,
        posinf=fp8_max,
        neginf=-fp8_max,
    ).clamp(-fp8_max, fp8_max)
    return {
        "quant_method": "fp8_e4m3",
        "data": values.to(fp8_dtype).contiguous(),
        "shape": tuple(tensor.shape),
    }


def lowrank_svd_residual(x: torch.Tensor, rank: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return per-head SVD factors and residual for x [B,H,S,D]."""
    bsz, heads, _, dim = x.shape
    mats = x.float()
    us, ss, vhs, approx = [], [], [], []
    for batch in range(bsz):
        batch_us, batch_ss, batch_vhs, batch_approx = [], [], [], []
        for head in range(heads):
            u, s, vh = torch.linalg.svd(mats[batch, head], full_matrices=False)
            actual_rank = min(rank, s.numel())
            u_r = u[:, :actual_rank].contiguous()
            s_r = s[:actual_rank].contiguous()
            vh_r = vh[:actual_rank, :].contiguous()
            batch_us.append(u_r.to(x.dtype))
            batch_ss.append(s_r.to(x.dtype))
            batch_vhs.append(vh_r.to(x.dtype))
            batch_approx.append(((u_r * s_r.unsqueeze(0)) @ vh_r).to(x.dtype))
        us.append(torch.stack(batch_us, dim=0))
        ss.append(torch.stack(batch_ss, dim=0))
        vhs.append(torch.stack(batch_vhs, dim=0))
        approx.append(torch.stack(batch_approx, dim=0))

    factors = {
        "u": torch.stack(us, dim=0).contiguous(),
        "s": torch.stack(ss, dim=0).contiguous(),
        "vh": torch.stack(vhs, dim=0).contiguous(),
    }
    residual = (x - torch.stack(approx, dim=0).contiguous()).contiguous()
    return factors, residual


def svd_triton_prq_quantize_tensor(
    tensor: torch.Tensor,
    rank: int,
    num_stages: int,
    num_clusters: int,
    block_size: int,
    max_iters: int,
    quantize_fn: callable,
    use_percentile_clipping: bool = False,
    use_lossless_flc: bool = False,
    lossless_flc_block_size: int = 16,
) -> dict:
    factors, residual = lowrank_svd_residual(tensor, rank)
    residual_state = triton_prq_quantize_tensor(
        residual,
        num_stages=num_stages,
        num_clusters=num_clusters,
        block_size=block_size,
        max_iters=max_iters,
        quantize_fn=quantize_fn,
        use_percentile_clipping=use_percentile_clipping,
        use_lossless_flc=use_lossless_flc,
        lossless_flc_block_size=lossless_flc_block_size,
    )
    return {
        "svd_rank": rank,
        "svd_factors": factors,
        "residual_state": residual_state,
    }


def get_quantize_fn(quant_type: str, quant_config: QuantizeConfig):
    if quant_type.startswith("kivi-all-int") or quant_type.startswith("turboquant-"):
        # Direct KV adapters are dispatched in compress_kv_cache.  Returning a
        # sentinel keeps the public configuration path compatible without
        # accidentally applying a second generic quantizer.
        def quantize_fn(_x):
            raise RuntimeError("direct KV codec must be dispatched by tensor role")

    elif quant_type in {"fp8", "fp8-e4m3", "fp8-e4m3fn", "naive-fp8", "naive-fp8-e4m3"}:
        def quantize_fn(x):
            return fp8_e4m3_quantize_tensor_cache(x.contiguous())

    elif quant_type in ["naive-fp4", "kmeans-fp4", "nstages-kmeans-fp4", "nstages-kmeans-fp4-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-fp4-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return nvfp4_e2m1_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int4", "kmeans-int4", "nstages-kmeans-int4", "nstages-kmeans-int4-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int4-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int4_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int3", "kmeans-int3", "nstages-kmeans-int3", "nstages-kmeans-int3-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int3-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int3_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )

    elif quant_type in ["naive-int2", "kmeans-int2", "nstages-kmeans-int2", "nstages-kmeans-int2-clip"]:
        def quantize_fn(x):
            """Quantization function - replace this to use different methods."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
            )

    elif quant_type == "kmeans-int2-clip":
        def quantize_fn(x):
            """Quantization function with percentile clipping."""
            return blockwise_int2_quantize_triton(
                x.contiguous(),
                block_size=quant_config.quant_block_size,
                use_percentile_clipping=True,
                percentile=99.0,
            )
    elif quant_type == "oscar-int2":
        def quantize_fn(x):
            return 2
    elif strip_lossless_flc(quant_type) in ["triton-nstages-kmeans-int2", "triton-nstages-kmeans-int2-clip", "triton-nstages-kmeans-int4", "triton-nstages-kmeans-int4-clip"] or parse_svd_quant_type(quant_type) is not None:
        """Do not quantize here"""
        def quantize_fn(x):
            m = re.search(r'int(\d+)', quant_config.quant_type)
            if m is None:
                raise ValueError(f"Cannot identify num_bits from {quant_config.quant_type}")
            num_bits = int(m.group(1))
            return num_bits
        
    else:
        raise ValueError(
            f"Unsupported quant type: {quant_type}"
        )
        
    return quantize_fn


def get_quantize_type(quant_type: str):
    base_quant_type = strip_lossless_flc(quant_type)
    # ==========================================================
    # Determine preprocessing mode
    # ==========================================================
    if base_quant_type in [
        "kmeans-fp4",
        "kmeans-fp4-clip",
        "kmeans-int4",
        "kmeans-int4-clip",
        "kmeans-int3",
        "kmeans-int3-clip",
        "kmeans-int2",
        "kmeans-int2-clip",
    ]:
        quantize_type = QuantizeFunctions.KMEANS

    elif base_quant_type in [
        "nstages-kmeans-fp4",
        "nstages-kmeans-int4",
        "nstages-kmeans-int3",
        "nstages-kmeans-int2",
    ]:
        quantize_type = QuantizeFunctions.NSTAGE_KMEANS

    elif base_quant_type in [
        "nstages-kmeans-fp4-clip",
        "nstages-kmeans-int4-clip",
        "nstages-kmeans-int3-clip",
        "nstages-kmeans-int2-clip",
    ]:
        quantize_type = QuantizeFunctions.NSTAGE_KMEANS_CLIP
    elif base_quant_type in [
        "triton-nstages-kmeans-int2",
        "triton-nstages-kmeans-int4",
    ]:
        quantize_type = QuantizeFunctions.TRITON_PRQ
    elif base_quant_type in [
        "triton-nstages-kmeans-int2-clip",
        "triton-nstages-kmeans-int4-clip",
    ]:
        quantize_type = QuantizeFunctions.TRITON_PRQ_CLIP
    elif (svd_quant := parse_svd_quant_type(quant_type)) is not None:
        _, base_quant_type = svd_quant
        if strip_lossless_flc(base_quant_type).endswith("-clip"):
            quantize_type = QuantizeFunctions.SVD_TRITON_PRQ_CLIP
        else:
            quantize_type = QuantizeFunctions.SVD_TRITON_PRQ
    elif base_quant_type == "oscar-int2":
        quantize_type = QuantizeFunctions.OSCAR_INT2
    else:
        quantize_type = QuantizeFunctions.NAIVE

    return quantize_type


def _rvq_feedback_query_like(query: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor | None:
    if not torch.is_tensor(query) or query.ndim != 4:
        return None
    if (
        int(query.shape[0]) != int(tensor.shape[0])
        or int(query.shape[1]) != int(tensor.shape[1])
        or int(query.shape[-1]) != int(tensor.shape[-1])
    ):
        return None
    return query.to(device=tensor.device, dtype=torch.float32)


def _rvq_calibration_mode(quant_config: QuantizeConfig, name: str) -> str:
    specific = getattr(quant_config, f"rvq_{name.lower()}_calibration", None)
    if specific is None:
        specific = getattr(quant_config, "rvq_calibration", "none")
    mode = str(specific).lower().replace("-", "_")
    if mode in {"", "0", "false", "off", "none"}:
        return "none"
    if mode not in {"online", "offline"}:
        raise ValueError(f"Unsupported RVQ {name} calibration mode: {specific}")
    return mode


def _rvq_calibration_query_like(query: torch.Tensor | None, tensor: torch.Tensor) -> torch.Tensor | None:
    return _rvq_feedback_query_like(query, tensor)


def _rvq_block_count(seq_len: int, block_size: int) -> int:
    return max(1, (int(seq_len) + int(block_size) - 1) // int(block_size))


def _rvq_clamp_pair(quant_config: QuantizeConfig) -> tuple[float, float]:
    raw = getattr(quant_config, "rvq_calibration_clamp", None)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo, hi = float(raw[0]), float(raw[1])
    else:
        lo = float(getattr(quant_config, "rvq_calibration_min", 0.75))
        hi = float(getattr(quant_config, "rvq_calibration_max", 1.33))
    if lo <= 0.0 or hi < lo:
        raise ValueError(f"Invalid RVQ calibration clamp: {lo}, {hi}")
    return lo, hi


def _blockwise_lstsq_scale(
    ref: torch.Tensor,
    recon: torch.Tensor,
    *,
    block_size: int,
    clamp: tuple[float, float],
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz, heads, seq_len, _dim = ref.shape
    blocks = _rvq_block_count(seq_len, block_size)
    out = torch.ones((bsz, heads, blocks, 1), device=ref.device, dtype=torch.float32)
    for block in range(blocks):
        start = block * int(block_size)
        end = min(seq_len, start + int(block_size))
        ref_b = ref[:, :, start:end, :].float()
        rec_b = recon[:, :, start:end, :].float()
        dot = (ref_b * rec_b).sum(dim=-1)
        denom = (rec_b * rec_b).sum(dim=-1).clamp_min(1e-8)
        if weights is not None:
            w_b = weights[:, :, start:end].float()
            num = (w_b * dot).sum(dim=-1)
            den = (w_b * denom).sum(dim=-1).clamp_min(1e-8)
        else:
            num = dot.sum(dim=-1)
            den = denom.sum(dim=-1).clamp_min(1e-8)
        out[:, :, block, 0] = (num / den).clamp(float(clamp[0]), float(clamp[1]))
    return out


def _k_query_block_scale(
    query: torch.Tensor | None,
    ref_k: torch.Tensor,
    recon_k: torch.Tensor,
    *,
    block_size: int,
    clamp: tuple[float, float],
) -> torch.Tensor:
    if query is None:
        return _blockwise_lstsq_scale(ref_k, recon_k, block_size=block_size, clamp=clamp)
    bsz, heads, seq_len, dim = ref_k.shape
    blocks = _rvq_block_count(seq_len, block_size)
    q = query.float().to(device=ref_k.device)
    out = torch.ones((bsz, heads, blocks, 1), device=ref_k.device, dtype=torch.float32)
    for block in range(blocks):
        start = block * int(block_size)
        end = min(seq_len, start + int(block_size))
        ref_b = ref_k[:, :, start:end, :].float()
        rec_b = recon_k[:, :, start:end, :].float()
        logits_ref = torch.einsum("bhqd,bhtd->bhqt", q, ref_b) / math.sqrt(dim)
        logits_rec = torch.einsum("bhqd,bhtd->bhqt", q, rec_b) / math.sqrt(dim)
        num = (logits_ref * logits_rec).sum(dim=(2, 3))
        den = (logits_rec * logits_rec).sum(dim=(2, 3)).clamp_min(1e-8)
        out[:, :, block, 0] = (num / den).clamp(float(clamp[0]), float(clamp[1]))
    return out


def _v_attention_block_scale(
    query: torch.Tensor | None,
    ref_k: torch.Tensor,
    ref_v: torch.Tensor,
    recon_v: torch.Tensor,
    *,
    block_size: int,
    clamp: tuple[float, float],
) -> torch.Tensor:
    if query is None:
        return _blockwise_lstsq_scale(ref_v, recon_v, block_size=block_size, clamp=clamp)
    dim = ref_k.shape[-1]
    q = query.float().to(device=ref_k.device)
    logits = torch.einsum("bhqd,bhsd->bhqs", q, ref_k.float()) / math.sqrt(dim)
    attn = torch.softmax(logits.float(), dim=-1)
    weights = (attn * attn).sum(dim=2)
    return _blockwise_lstsq_scale(ref_v, recon_v, block_size=block_size, clamp=clamp, weights=weights)


def _k_attention_kl(
    query: torch.Tensor | None,
    ref_k: torch.Tensor,
    recon_k: torch.Tensor,
) -> float | None:
    if query is None:
        return None
    dim = ref_k.shape[-1]
    q = query.float().to(device=ref_k.device)
    logits_ref = torch.einsum("bhqd,bhsd->bhqs", q, ref_k.float()) / math.sqrt(dim)
    logits_rec = torch.einsum("bhqd,bhsd->bhqs", q, recon_k.float()) / math.sqrt(dim)
    log_p = torch.log_softmax(logits_ref.float(), dim=-1)
    log_q = torch.log_softmax(logits_rec.float(), dim=-1)
    p = torch.softmax(logits_ref.float(), dim=-1)
    return float((p * (log_p - log_q)).sum(dim=-1).mean().item())


def _v_attention_output_mse(
    query: torch.Tensor | None,
    ref_k: torch.Tensor,
    ref_v: torch.Tensor,
    recon_v: torch.Tensor,
) -> float | None:
    if query is None:
        return None
    dim = ref_k.shape[-1]
    q = query.float().to(device=ref_k.device)
    logits = torch.einsum("bhqd,bhsd->bhqs", q, ref_k.float()) / math.sqrt(dim)
    attn = torch.softmax(logits.float(), dim=-1)
    out_ref = torch.einsum("bhqs,bhsd->bhqd", attn, ref_v.float())
    out_rec = torch.einsum("bhqs,bhsd->bhqd", attn, recon_v.float())
    diff = out_ref - out_rec
    return float((diff * diff).mean().item())


def _record_rvq_diagnostic(packed: dict, key: str, value: float | None) -> None:
    if value is None:
        return
    diagnostics = packed.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        diagnostics[key] = float(value)


def _reconstruction_diagnostics(ref: torch.Tensor, recon: torch.Tensor) -> dict[str, float]:
    error_sq = (ref.float() - recon.float()).square()
    source_sq = ref.float().square()
    relative_squared = error_sq.sum() / source_sq.sum().clamp_min(1e-20)
    vector_relative = error_sq.sum(dim=-1) / source_sq.sum(dim=-1).clamp_min(1e-20)
    return {
        "raw_mse": float(error_sq.mean().item()),
        "relative_l2": float(torch.sqrt(relative_squared).item()),
        "mean_vector_relative_squared_error": float(vector_relative.mean().item()),
    }


@lru_cache(maxsize=4)
def _load_rvq_calibration_table(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"RVQ calibration table must be a dict: {path}")
    return payload


def _offline_block_scale(
    quant_config: QuantizeConfig,
    name: str,
    layer_idx: int,
    ref: torch.Tensor,
    *,
    block_size: int,
    clamp: tuple[float, float],
) -> torch.Tensor:
    path = str(getattr(quant_config, "rvq_calibration_path", "") or "")
    if not path:
        raise ValueError("Offline RVQ calibration requires rvq_calibration_path")
    table = _load_rvq_calibration_table(str(Path(path)))
    tensor_key = name.lower()
    value = table.get(tensor_key)
    if value is None:
        value = table.get(tensor_key.upper())
    if not torch.is_tensor(value):
        raise ValueError(f"Calibration table {path} missing tensor key {tensor_key!r}")
    layer_scale = value[int(layer_idx)].float()
    if layer_scale.ndim == 2:
        layer_scale = layer_scale[:, :, None]
    if layer_scale.ndim == 3:
        layer_scale = layer_scale[None, :, :, :]
    if layer_scale.ndim != 4:
        raise ValueError(f"Invalid calibration scale shape for {tensor_key}: {tuple(layer_scale.shape)}")
    scale = layer_scale.to(device=ref.device, dtype=torch.float32)
    bsz, heads, seq_len, _dim = ref.shape
    blocks = _rvq_block_count(seq_len, block_size)
    if int(scale.shape[0]) == 1 and bsz != 1:
        scale = scale.expand(bsz, -1, -1, -1)
    if int(scale.shape[1]) != heads:
        raise ValueError(f"Calibration head mismatch for {tensor_key}: {tuple(scale.shape)} vs heads={heads}")
    if int(scale.shape[2]) < blocks:
        repeats = math.ceil(blocks / int(scale.shape[2]))
        scale = scale.repeat(1, 1, repeats, 1)
    scale = scale[:, :, :blocks, :1].contiguous()
    return scale.clamp(float(clamp[0]), float(clamp[1]))


def _attach_rvq_calibration(
    packed: dict,
    *,
    name: str,
    mode: str,
    layer_idx: int,
    ref: torch.Tensor,
    recon: torch.Tensor,
    quant_config: QuantizeConfig,
    query: torch.Tensor | None,
    key_for_v: torch.Tensor | None = None,
) -> dict:
    if mode == "none":
        return packed
    block_size = int(getattr(quant_config, "rvq_calibration_block_size", 880))
    clamp = _rvq_clamp_pair(quant_config)
    if mode == "offline":
        scale = _offline_block_scale(
            quant_config,
            name,
            int(layer_idx),
            ref,
            block_size=block_size,
            clamp=clamp,
        )
    elif name.lower() == "k":
        scale = _k_query_block_scale(query, ref, recon, block_size=block_size, clamp=clamp)
    else:
        k_ref = ref if key_for_v is None else key_for_v
        scale = _v_attention_block_scale(query, k_ref, ref, recon, block_size=block_size, clamp=clamp)
    storage = str(getattr(quant_config, "rvq_calibration_storage_dtype", "uint4"))
    return pack_rvq_calibration_scale(
        packed,
        scale,
        block_size=block_size,
        storage_dtype=storage,
        source=mode,
        target=name.lower(),
    )


def _rvq_state_tensor(
    state: dict | None,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(state, dict):
        value = state.get(name)
        if torch.is_tensor(value) and tuple(value.shape) == shape:
            return value.to(device=device, dtype=torch.float32)
    return torch.zeros(shape, device=device, dtype=torch.float32)


def _rvq_current_error(state: dict | None, tensor: torch.Tensor) -> torch.Tensor | None:
    if not isinstance(state, dict):
        return None
    value = state.get("prev_error")
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(tensor.shape):
        return None
    return value.to(device=tensor.device, dtype=torch.float32)


def _rvq_store_cpu_state(state: dict | None, name: str, value: torch.Tensor) -> None:
    if isinstance(state, dict):
        state[name] = value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _rvq_attn_k_guidance(
    query: torch.Tensor | None,
    state: dict | None,
    seq_len: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if query is None or not isinstance(state, dict):
        return None, None
    bsz, heads, q_len, dim = query.shape
    logit_error = _rvq_state_tensor(
        state,
        "attn_logit_error",
        (bsz, heads, q_len),
        query.device,
    )
    guidance = torch.einsum("bhq,bhqd->bhd", logit_error, query) / math.sqrt(dim)
    guidance = guidance[:, :, None, :].expand(-1, -1, int(seq_len), -1).contiguous()
    curvature = (query * query).sum(dim=(2, 3))[:, :, None, None] / float(dim)
    curvature = curvature.expand(-1, -1, int(seq_len), 1).contiguous()
    return guidance, curvature


def _rvq_attn_v_guidance(
    query: torch.Tensor | None,
    key: torch.Tensor,
    state: dict | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if query is None or not isinstance(state, dict):
        return None, None, None
    bsz, heads, q_len, dim = query.shape
    output_error = _rvq_state_tensor(
        state,
        "attn_output_error",
        (bsz, heads, q_len, key.shape[-1]),
        query.device,
    )
    logits = torch.einsum("bhqd,bhsd->bhqs", query, key.float()) / math.sqrt(dim)
    attn = torch.softmax(logits.float(), dim=-1)
    guidance = torch.einsum("bhqs,bhqd->bhsd", attn, output_error)
    curvature = (attn * attn).sum(dim=2)[..., None]
    return guidance.contiguous(), curvature.contiguous(), attn


def _rvq_update_attn_k_error(
    query: torch.Tensor | None,
    key: torch.Tensor,
    state: dict | None,
) -> None:
    err = _rvq_current_error(state, key)
    if query is None or err is None:
        return
    bsz, heads, q_len, dim = query.shape
    prev = _rvq_state_tensor(
        state,
        "attn_logit_error",
        (bsz, heads, q_len),
        query.device,
    )
    logits_err = torch.einsum("bhqd,bhsd->bhqs", query, err) / math.sqrt(dim)
    _rvq_store_cpu_state(state, "attn_logit_error", prev + logits_err.mean(dim=-1))


def _rvq_update_attn_v_error(
    attn: torch.Tensor | None,
    value: torch.Tensor,
    state: dict | None,
) -> None:
    err = _rvq_current_error(state, value)
    if attn is None or err is None:
        return
    bsz, heads, q_len, _ = attn.shape
    prev = _rvq_state_tensor(
        state,
        "attn_output_error",
        (bsz, heads, q_len, value.shape[-1]),
        value.device,
    )
    output_err = torch.einsum("bhqs,bhsd->bhqd", attn, err)
    _rvq_store_cpu_state(state, "attn_output_error", prev + output_err)


def compress_tensor_cache(
    tensor: torch.Tensor,
    quant_type: str,
    quant_config: QuantizeConfig,
    quantize_fn: callable,
    num_centroids: int,
):
    quantize_type = get_quantize_type(quant_type)
    use_lossless_flc = uses_lossless_flc(quant_type)
    lossless_flc_block_size = getattr(quant_config, "lossless_flc_block_size", 16)

    if quantize_type == QuantizeFunctions.NSTAGE_KMEANS:
        return prq_quantize_tensor(
            tensor,
            num_stages=quant_config.num_prq_stages,
            codebook_size=num_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.NSTAGE_KMEANS_CLIP:
        return prq_quantize_tensor(
            tensor,
            num_stages=quant_config.num_prq_stages,
            codebook_size=num_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
        )
    elif quantize_type == QuantizeFunctions.KMEANS:
        return kmeans_quantize_tensor(
            tensor,
            num_centroids=num_centroids,
            kmeans_max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
        )
    elif quantize_type == QuantizeFunctions.TRITON_PRQ:
        return triton_prq_quantize_tensor(
            tensor,
            num_stages=quant_config.num_prq_stages,
            num_clusters=num_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_lossless_flc=use_lossless_flc,
            lossless_flc_block_size=lossless_flc_block_size,
        )
    elif quantize_type == QuantizeFunctions.TRITON_PRQ_CLIP:
        return triton_prq_quantize_tensor(
            tensor,
            num_stages=quant_config.num_prq_stages,
            num_clusters=num_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=True,
            use_lossless_flc=use_lossless_flc,
            lossless_flc_block_size=lossless_flc_block_size,
        )
    elif quantize_type in (QuantizeFunctions.SVD_TRITON_PRQ, QuantizeFunctions.SVD_TRITON_PRQ_CLIP):
        svd_quant = parse_svd_quant_type(quant_type)
        assert svd_quant is not None
        svd_rank, _ = svd_quant
        use_clip = quantize_type == QuantizeFunctions.SVD_TRITON_PRQ_CLIP
        return svd_triton_prq_quantize_tensor(
            tensor,
            rank=svd_rank,
            num_stages=quant_config.num_prq_stages,
            num_clusters=num_centroids,
            block_size=quant_config.quant_block_size,
            max_iters=quant_config.kmeans_max_iters,
            quantize_fn=quantize_fn,
            use_percentile_clipping=use_clip,
            use_lossless_flc=use_lossless_flc,
            lossless_flc_block_size=lossless_flc_block_size,
        )
    elif quantize_type == QuantizeFunctions.NAIVE:
        return quantize_fn(tensor)

    raise ValueError(f"Unsupported quant type: {quant_type}")


def _quantize_rvq_residual(
    tensor: torch.Tensor,
    rvq_base: dict,
    *,
    tensor_name: str,
    quant_type: str,
    quant_config: QuantizeConfig,
    quantize_fn: callable,
    num_centroids: int,
    layer_idx: int | None = None,
    span_idx: int | None = None,
) -> dict:
    """Store an RVQ reconstruction plus a compressed BF16 residual."""

    original_bf16 = tensor.to(dtype=torch.bfloat16)
    if bool(getattr(quant_config, "temporal_adaptive_residual", False)):
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for temporal adaptive residual admission"
            )
        state = create_b16_state(
            original_bf16,
            rvq_base,
            tensor_name=tensor_name,
            layer_idx=int(layer_idx),
            portable_identity=not bool(
                getattr(
                    quant_config,
                    "temporal_adaptive_fast_identity",
                    True,
                )
            ),
        )
        # The temporal scheduler can determine the new chunk's target depth
        # from the already-resident precision vector before admission.  Encode
        # directly to that target so a B16 bypass is never installed in the
        # cache when it will be discarded immediately.  Keeping the default
        # at B16 preserves the standalone codec and legacy pipeline behavior.
        tensor_admission_bits = getattr(
            quant_config,
            f"temporal_adaptive_{str(tensor_name).lower()}_admission_target_bits",
            None,
        )
        layer_tensor_targets = getattr(
            quant_config,
            "temporal_adaptive_admission_target_bits_by_layer_tensor",
            None,
        )
        mapped_admission_bits = None
        if layer_tensor_targets is not None:
            if not isinstance(layer_tensor_targets, Mapping):
                raise TypeError(
                    "temporal_adaptive_admission_target_bits_by_layer_tensor "
                    "must be a mapping"
                )
            key = f"{str(tensor_name).strip().upper()}:{int(layer_idx)}"
            if key not in layer_tensor_targets:
                raise ValueError(f"missing layer/tensor admission target {key}")
            mapped_admission_bits = int(layer_tensor_targets[key])
        admission_bits = int(
            mapped_admission_bits
            if mapped_admission_bits is not None
            else (
                tensor_admission_bits
                if tensor_admission_bits is not None
                else getattr(
                    quant_config,
                    "temporal_adaptive_admission_target_bits",
                    16,
                )
            )
        )
        # Frozen sub-bit schedules can deliberately omit a newly admitted
        # residual (B0).  transition_state handles the zero-depth terminal
        # representation, so admission must accept it as well.
        if admission_bits not in {16, 8, 4, 3, 2, 1, 0}:
            raise ValueError(
                "temporal_adaptive_admission_target_bits must be one of "
                "16/8/4/3/2/1/0"
            )
        if admission_bits < 16:
            state = transition_state(
                state,
                admission_bits,
                quant_config=quant_config,
                verify_identity=False,
            )
        return state
    base_bf16 = rvq_dequantize_tensor_cache(rvq_base, torch.bfloat16)
    residual_bf16 = (original_bf16 - base_bf16).contiguous()
    canonical_tensor_name = str(tensor_name).strip().lower()
    if canonical_tensor_name not in {"k", "v"}:
        raise ValueError(
            f"tensor_name must be K or V for RVQ residual dispatch, got {tensor_name!r}"
        )
    tensor_override = str(
        getattr(
            quant_config,
            f"rvq_{canonical_tensor_name}_residual_codec",
            "",
        )
        or ""
    ).strip()
    k_structured_layer_spec = str(
        getattr(quant_config, "rvq_k_structured_layers", "") or ""
    ).strip()
    span_schedule_spec = str(
        getattr(quant_config, "rvq_residual_codec_span_schedule", "") or ""
    ).strip()
    span_schedule = parse_residual_codec_span_schedule(span_schedule_spec)
    if span_schedule:
        if tensor_override or k_structured_layer_spec:
            raise ValueError(
                "rvq_residual_codec_span_schedule cannot be combined with "
                "per-tensor residual overrides or rvq_k_structured_layers"
            )
        if span_idx is None:
            raise ValueError(
                "span_idx is required when "
                "rvq_residual_codec_span_schedule is set"
            )
        if int(span_idx) < 0 or int(span_idx) >= len(span_schedule):
            raise ValueError(
                f"span_idx={span_idx} is outside the configured residual-codec "
                f"schedule of length {len(span_schedule)}"
            )
        residual_codec = span_schedule[int(span_idx)]
    elif canonical_tensor_name == "k" and k_structured_layer_spec:
        if tensor_override:
            raise ValueError(
                "rvq_k_structured_layers cannot be combined with "
                "rvq_k_residual_codec; the layer-selective policy fully defines "
                "K residual dispatch"
            )
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required when rvq_k_structured_layers is set"
            )
        structured_layers = parse_layer_spec(k_structured_layer_spec)
        residual_codec = (
            "structured_truncatable"
            if int(layer_idx) in structured_layers
            else "quantvideo"
        )
    else:
        residual_codec = (
            tensor_override
            or str(getattr(quant_config, "rvq_residual_codec", "quantvideo"))
        ).strip().lower()
    if not residual_codec:
        raise ValueError("rvq_residual_codec cannot be empty after tensor override resolution")
    if residual_codec == "higgs":
        match = re.search(r"int(2|4)", quant_type)
        if match is None:
            raise ValueError(
                "HIGGS-lite RVQ residual quantization requires an INT2/INT4 "
                f"quant_type, got {quant_type!r}"
            )
        residual_quant = higgs_quantize_tensor(
            residual_bf16,
            bits=int(match.group(1)),
            group_size=int(getattr(quant_config, "higgs_group_size", 32)),
            refine_iters=int(getattr(quant_config, "higgs_refine_iters", 1)),
            seed=int(getattr(quant_config, "higgs_seed", 20260722)),
        )
        quant_method = "rvq_higgs_residual"
    elif residual_codec == "truncatable":
        residual_quant = encode_residual(
            residual_bf16,
            scheme=str(
                getattr(
                    quant_config,
                    "truncatable_residual_scheme",
                    "nested_affine",
                )
            ),
            bits=int(getattr(quant_config, "truncatable_residual_bits", 4)),
            tensor_name=str(tensor_name).lower(),
            params_path=(
                str(
                    getattr(
                        quant_config,
                        "truncatable_residual_params_path",
                        "",
                    )
                )
                or None
            ),
            hadamard=bool(
                getattr(
                    quant_config,
                    "truncatable_residual_hadamard",
                    False,
                )
            ),
            seed=int(
                getattr(
                    quant_config,
                    "truncatable_residual_seed",
                    20260722,
                )
            ),
        )
        quant_method = "rvq_truncatable_residual"
    elif residual_codec == "routed_gaussian":
        if layer_idx is None:
            raise ValueError("layer_idx is required for routed_gaussian")
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError("routed_gaussian_routing_path is required")
        tree_params_path = str(
            getattr(quant_config, "routed_gaussian_tree_params_path", "") or ""
        )
        tree_params = (
            _load_routed_gaussian_tree_params(
                tree_params_path,
                str(residual_bf16.device),
            )
            if tree_params_path
            else None
        )
        residual_quant = encode_routed_gaussian_residual(
            residual_bf16,
            tensor_name=canonical_tensor_name,
            layer_idx=int(layer_idx),
            routing_table=routing_path,
            bits=int(getattr(quant_config, "routed_gaussian_bits", 4)),
            group_size=int(
                getattr(quant_config, "routed_gaussian_group_size", 128)
            ),
            hadamard_group_size=int(
                getattr(
                    quant_config,
                    "routed_gaussian_hadamard_group_size",
                    128,
                )
            ),
            seed=int(
                getattr(quant_config, "routed_gaussian_seed", 20260722)
            ),
            tree_params=tree_params,
            scale_refine_iters=int(
                getattr(
                    quant_config,
                    "routed_gaussian_scale_refine_iters",
                    1,
                )
            ),
        )
        quant_method = "rvq_routed_gaussian_residual"
    elif residual_codec == "uniform_grid_gaussian":
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for uniform_grid_gaussian"
            )
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError("routed_gaussian_routing_path is required")
        tree_params_path = str(
            getattr(
                quant_config, "routed_gaussian_tree_params_path", ""
            )
            or ""
        )
        tree_params = (
            _load_routed_gaussian_tree_params(
                tree_params_path,
                str(residual_bf16.device),
            )
            if tree_params_path
            else None
        )
        residual_quant = encode_uniform_grid_gaussian_residual(
            residual_bf16,
            layer_idx=int(layer_idx),
            tensor_name=canonical_tensor_name,
            bits=int(getattr(quant_config, "routed_gaussian_bits", 4)),
            routing_table=routing_path,
            tree_params=tree_params,
            seed=int(
                getattr(quant_config, "routed_gaussian_seed", 20260722)
            ),
        )
        quant_method = "rvq_uniform_grid_gaussian_residual"
    elif residual_codec in {
        "frame_mean_gaussian",
        "shrunk_frame_mean_gaussian",
    }:
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for frame_mean_gaussian"
            )
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError("routed_gaussian_routing_path is required")
        tree_params_path = str(
            getattr(
                quant_config, "routed_gaussian_tree_params_path", ""
            )
            or ""
        )
        tree_params = (
            _load_routed_gaussian_tree_params(
                tree_params_path,
                str(residual_bf16.device),
            )
            if tree_params_path
            else None
        )
        residual_quant = encode_frame_mean_gaussian_residual(
            residual_bf16,
            layer_idx=int(layer_idx),
            tensor_name=canonical_tensor_name,
            bits=int(getattr(quant_config, "routed_gaussian_bits", 4)),
            routing_table=routing_path,
            tree_params=tree_params,
            seed=int(
                getattr(quant_config, "routed_gaussian_seed", 20260722)
            ),
            spatial_tokens=int(
                getattr(quant_config, "frame_mean_spatial_tokens", 880)
            ),
            mean_shrinkage=(
                "universal"
                if residual_codec == "shrunk_frame_mean_gaussian"
                else "none"
            ),
        )
        quant_method = (
            "rvq_shrunk_frame_mean_gaussian_residual"
            if residual_codec == "shrunk_frame_mean_gaussian"
            else "rvq_frame_mean_gaussian_residual"
        )
    elif residual_codec == "selective_rank1_gaussian":
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for selective_rank1_gaussian"
            )
        if canonical_tensor_name != "k":
            raise ValueError(
                "selective_rank1_gaussian is intentionally K-only"
            )
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError("routed_gaussian_routing_path is required")
        params_path = str(
            getattr(
                quant_config,
                "selective_rank1_gaussian_params_path",
                "",
            )
            or ""
        )
        if not params_path:
            raise ValueError(
                "selective_rank1_gaussian_params_path is required"
            )
        tree_params_path = str(
            getattr(
                quant_config, "routed_gaussian_tree_params_path", ""
            )
            or ""
        )
        tree_params = (
            _load_routed_gaussian_tree_params(
                tree_params_path,
                str(residual_bf16.device),
            )
            if tree_params_path
            else None
        )
        residual_quant = encode_selective_rank1_gaussian_residual(
            residual_bf16,
            layer_idx=int(layer_idx),
            tensor_name=canonical_tensor_name,
            params=_load_selective_rank1_gaussian_params(
                params_path, str(residual_bf16.device)
            ),
            routing_table=routing_path,
            bits=int(getattr(quant_config, "routed_gaussian_bits", 4)),
            tree_params=tree_params,
            seed=int(
                getattr(quant_config, "routed_gaussian_seed", 20260722)
            ),
        )
        quant_method = "rvq_selective_rank1_gaussian_residual"
    elif residual_codec == "cache_mean_gaussian":
        if layer_idx is None:
            raise ValueError("layer_idx is required for cache_mean_gaussian")
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError("routed_gaussian_routing_path is required")
        tree_params_path = str(
            getattr(
                quant_config, "routed_gaussian_tree_params_path", ""
            )
            or ""
        )
        tree_params = (
            _load_routed_gaussian_tree_params(
                tree_params_path,
                str(residual_bf16.device),
            )
            if tree_params_path
            else None
        )
        residual_quant = encode_cache_mean_gaussian_residual(
            residual_bf16,
            layer_idx=int(layer_idx),
            tensor_name=canonical_tensor_name,
            routing_table=routing_path,
            bits=int(getattr(quant_config, "routed_gaussian_bits", 4)),
            tree_params=tree_params,
            seed=int(
                getattr(quant_config, "routed_gaussian_seed", 20260722)
            ),
        )
        quant_method = "rvq_cache_mean_gaussian_residual"
    elif residual_codec == "structured_truncatable":
        default_structured_bits = int(
            getattr(quant_config, "truncatable_residual_bits", 4)
        )
        tensor_structured_bits = int(
            getattr(
                quant_config,
                f"structured_residual_{str(tensor_name).lower()}_bits",
                0,
            )
        )
        if tensor_structured_bits <= 0:
            tensor_structured_bits = default_structured_bits
        residual_quant = encode_structured_residual(
            residual_bf16,
            scheme=str(
                getattr(
                    quant_config,
                    "truncatable_residual_scheme",
                    "gaussian_tree",
                )
            ),
            bits=tensor_structured_bits,
            tensor_name=str(tensor_name).lower(),
            params_path=str(
                getattr(
                    quant_config,
                    "truncatable_residual_params_path",
                    "",
                )
            ),
            hadamard=bool(
                getattr(
                    quant_config,
                    "truncatable_residual_hadamard",
                    False,
                )
            ),
            transform_seed=int(
                getattr(
                    quant_config,
                    "truncatable_residual_seed",
                    20260722,
                )
            ),
            group_size=int(
                getattr(
                    quant_config,
                    "structured_residual_group_size",
                    128,
                )
            ),
            n_clusters=int(
                getattr(
                    quant_config,
                    "structured_residual_clusters",
                    256,
                )
            ),
            kmeans_max_iters=int(
                getattr(
                    quant_config,
                    "structured_residual_kmeans_iters",
                    2,
                )
            ),
            kmeans_seed=int(
                getattr(
                    quant_config,
                    "structured_residual_kmeans_seed",
                    42,
                )
            ),
            scale_refine_iters=int(
                getattr(
                    quant_config,
                    "structured_residual_scale_refine_iters",
                    1,
                )
            ),
        )
        quant_method = "rvq_structured_truncatable_residual"
    elif residual_codec == "structured_static_prefix":
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for structured_static_prefix residuals"
            )
        allocation_path = str(
            getattr(quant_config, "structured_prefix_allocation_path", "")
        )
        if not allocation_path:
            raise ValueError(
                "structured_prefix_allocation_path is required for "
                "structured_static_prefix residuals"
            )
        residual_quant = encode_static_allocated_structured_residual(
            residual_bf16,
            layer_idx=int(layer_idx),
            allocation_path=allocation_path,
            tensor_name=str(tensor_name),
            params_path=str(
                getattr(
                    quant_config,
                    "truncatable_residual_params_path",
                    "",
                )
            ),
            scheme=str(
                getattr(
                    quant_config,
                    "truncatable_residual_scheme",
                    "gaussian_tree",
                )
            ),
            hadamard=bool(
                getattr(
                    quant_config,
                    "truncatable_residual_hadamard",
                    True,
                )
            ),
            transform_seed=int(
                getattr(
                    quant_config,
                    "truncatable_residual_seed",
                    20260722,
                )
            ),
            group_size=int(
                getattr(
                    quant_config,
                    "structured_residual_group_size",
                    128,
                )
            ),
            n_clusters=int(
                getattr(
                    quant_config,
                    "structured_residual_clusters",
                    256,
                )
            ),
            kmeans_max_iters=int(
                getattr(
                    quant_config,
                    "structured_residual_kmeans_iters",
                    2,
                )
            ),
            kmeans_seed=int(
                getattr(
                    quant_config,
                    "structured_residual_kmeans_seed",
                    42,
                )
            ),
            scale_refine_iters=int(
                getattr(
                    quant_config,
                    "structured_residual_scale_refine_iters",
                    1,
                )
            ),
        )
        quant_method = "rvq_structured_static_prefix_residual"
    elif residual_codec == "r2aq":
        default_bits = int(getattr(quant_config, "r2aq_residual_bits", 4))
        tensor_bits = int(
            getattr(
                quant_config,
                f"r2aq_{canonical_tensor_name}_bits",
                0,
            )
        )
        if tensor_bits <= 0:
            tensor_bits = default_bits
        predictor_group_size = int(
            getattr(quant_config, "r2aq_predictor_group_size", 128)
        )
        suffix_group_size = int(
            getattr(quant_config, "r2aq_suffix_group_size", 64)
        )
        explicit_ablation = str(
            getattr(quant_config, "r2aq_ablation", "") or ""
        ).strip().lower()
        coherent_tail_group_size = int(
            getattr(
                quant_config,
                "r2aq_coherent_tail_group_size",
                128,
            )
        )
        coherent_tail_ablations = {
            "coherent_tail",
            "coherent_tail_g128",
        }
        if explicit_ablation in coherent_tail_ablations:
            if not bool(
                getattr(
                    quant_config,
                    "r2aq_coherent_tail_eager_decode",
                    False,
                )
            ):
                raise ValueError(
                    f"{explicit_ablation} requires the explicit "
                    "r2aq_coherent_tail_eager_decode opt-in; the fused "
                    "decoder cannot restore coherent-tail states"
                )
            if tensor_bits != 4:
                raise ValueError(
                    f"{explicit_ablation} is a B4 encode path; got B{tensor_bits}"
                )
        if explicit_ablation == "coherent_tail_g128":
            if coherent_tail_group_size != 128:
                raise ValueError(
                    "coherent_tail_g128 requires "
                    "r2aq_coherent_tail_group_size=128"
                )
            if predictor_group_size != coherent_tail_group_size:
                raise ValueError(
                    "coherent_tail_g128 derives corrections from predictor "
                    "groups, so r2aq_predictor_group_size must equal "
                    "r2aq_coherent_tail_group_size=128"
                )
        encode_kwargs = {
            "bits": tensor_bits,
            "max_nominal_bits": tensor_bits,
            "tensor_name": canonical_tensor_name,
            "predictor_group_size": predictor_group_size,
            "suffix_group_size": suffix_group_size,
            "n_clusters": int(getattr(quant_config, "r2aq_clusters", 256)),
            "kmeans_max_iters": int(
                getattr(quant_config, "r2aq_kmeans_iters", 2)
            ),
            "kmeans_seed": int(
                getattr(quant_config, "r2aq_kmeans_seed", 42)
            ),
            "utility_weights_path": (
                str(
                    getattr(
                        quant_config,
                        "r2aq_utility_weights_path",
                        "",
                    )
                )
                or None
            ),
            "tree_params_path": (
                str(
                    getattr(
                        quant_config,
                        "r2aq_tree_params_path",
                        "",
                    )
                )
                or None
            ),
            "layer_idx": layer_idx,
            "age_bin": span_idx,
            "coherent_tail_group_fraction": float(
                getattr(
                    quant_config,
                    "r2aq_coherent_tail_group_fraction",
                    0.55,
                )
            ),
        }
        selection_fraction = getattr(
            quant_config,
            "r2aq_selection_fraction",
            None,
        )
        if selection_fraction is not None:
            encode_kwargs["selection_fraction"] = float(selection_fraction)
        if explicit_ablation:
            encode_kwargs["ablation"] = explicit_ablation
        else:
            # ``selection_mode`` is the compatibility spelling used by the
            # first R²AQ video runner.  It intentionally remains absent when
            # an explicit ablation is supplied because the reference encoder
            # treats it as an alias that would otherwise override ablation.
            encode_kwargs["selection_mode"] = str(
                getattr(quant_config, "r2aq_selection_mode", "scale")
            )
        owner_layer = int(
            getattr(
                quant_config,
                "r2aq_shared_budget_owner_layer",
                -1,
            )
        )
        owner_span = int(
            getattr(
                quant_config,
                "r2aq_shared_budget_owner_span",
                -1,
            )
        )
        if (owner_layer == -1) != (owner_span == -1):
            raise ValueError(
                "r2aq_shared_budget_owner_layer and "
                "r2aq_shared_budget_owner_span must be enabled together"
            )
        if owner_layer < -1 or owner_span < -1:
            raise ValueError(
                "R²AQ shared-budget owner indices must be >=0 or -1"
            )
        if owner_layer >= 0:
            if layer_idx is None or span_idx is None:
                raise ValueError(
                    "R²AQ shared-budget ownership requires layer_idx and "
                    "span_idx on every encoded state"
                )
            if int(layer_idx) == owner_layer and int(span_idx) == owner_span:
                # The codec helper derives the actual tree/threshold/utility
                # bytes from the selected tables.  This keeps the bridge free
                # of fragile byte constants while the explicit layer/span
                # binding guarantees a unique owner per K/V stream.
                owner_shared_bytes = owner_shared_budget_nbytes(
                    canonical_tensor_name,
                    utility_weights_path=(
                        encode_kwargs["utility_weights_path"]
                    ),
                )
                logical_multiplier = int(
                    getattr(
                        quant_config,
                        "r2aq_payload_logical_multiplier",
                        1,
                    )
                )
                if logical_multiplier <= 0:
                    raise ValueError(
                        "r2aq_payload_logical_multiplier must be positive"
                    )
                if owner_shared_bytes % logical_multiplier:
                    raise ValueError(
                        "R²AQ shared owner bytes are not divisible by the "
                        "runtime logical payload multiplier"
                    )
                encode_kwargs["shared_budget_reserve_bytes"] = (
                    owner_shared_bytes // logical_multiplier
                )
        residual_quant = encode_r2aq_residual(
            residual_bf16,
            **encode_kwargs,
        )
        quant_method = "rvq_r2aq_residual"
    elif residual_codec == "quantvideo":
        quantize_type = get_quantize_type(quant_type)
        if quantize_type not in {
            QuantizeFunctions.TRITON_PRQ,
            QuantizeFunctions.TRITON_PRQ_CLIP,
        }:
            raise ValueError(
                "QuantVideo RVQ residual quantization requires "
                "triton-nstages-kmeans-int2/int4, got "
                f"{quant_type!r}"
            )
        residual_quant = compress_tensor_cache(
            residual_bf16,
            quant_type,
            quant_config,
            quantize_fn,
            num_centroids,
        )
        quant_method = "rvq_quantvideo_residual"
    elif residual_codec == "kivi":
        residual_quant = encode_kivi_tensor(
            residual_bf16,
            tensor_name=canonical_tensor_name,
            bits=int(getattr(quant_config, "kivi_residual_bits", 4)),
            group_size=int(
                getattr(quant_config, "kivi_residual_group_size", 32)
            ),
        )
        quant_method = "rvq_kivi_residual"
    elif residual_codec == "absmax_scalar":
        residual_quant = encode_absmax_scalar_residual(
            residual_bf16,
            bits=int(getattr(quant_config, "absmax_scalar_bits", 4)),
            group_size=int(
                getattr(quant_config, "absmax_scalar_group_size", 128)
            ),
        )
        quant_method = "rvq_absmax_scalar_residual"
    elif residual_codec == "sign_adaptive_absmax_scalar":
        residual_quant = encode_sign_adaptive_absmax_scalar_residual(
            residual_bf16,
            group_size=int(
                getattr(
                    quant_config,
                    "sign_adaptive_absmax_group_size",
                    128,
                )
            ),
        )
        quant_method = "rvq_sign_adaptive_absmax_scalar_residual"
    elif residual_codec == "progressive_sign_adaptive_absmax_scalar":
        if canonical_tensor_name.lower() != "v":
            raise ValueError(
                "progressive_sign_adaptive_absmax_scalar is a V-only codec"
            )
        params_path = str(
            getattr(
                quant_config,
                "progressive_v_absmax_params_path",
                "",
            )
            or ""
        )
        if not params_path:
            raise ValueError("progressive_v_absmax_params_path is required")
        requested_bits = int(
            getattr(quant_config, "progressive_v_absmax_bits", 4)
        )
        residual_quant = (
            encode_progressive_sign_adaptive_absmax_scalar_residual(
                residual_bf16,
                method=str(
                    getattr(
                        quant_config,
                        "progressive_v_absmax_method",
                        "midpoint_msb",
                    )
                ),
                params=_load_progressive_v_absmax_params(params_path),
                group_size=int(
                    getattr(
                        quant_config,
                        "progressive_v_absmax_group_size",
                        128,
                    )
                ),
                max_bits=8 if requested_bits > 4 else 4,
            )
        )
        residual_quant = truncate_progressive_sign_adaptive_absmax_state(
            residual_quant,
            bits=requested_bits,
        )
        quant_method = (
            "rvq_progressive_sign_adaptive_absmax_scalar_residual"
        )
    elif residual_codec == "centered_absmax_scalar":
        residual_quant = encode_centered_absmax_scalar_residual(
            residual_bf16,
            bits=int(getattr(quant_config, "centered_absmax_scalar_bits", 4)),
            group_size=int(
                getattr(
                    quant_config,
                    "centered_absmax_scalar_group_size",
                    128,
                )
            ),
            center_mode=str(
                getattr(
                    quant_config,
                    "centered_absmax_scalar_center_mode",
                    "mean",
                )
            ),
        )
        quant_method = "rvq_centered_absmax_scalar_residual"
    elif residual_codec == "routed_midrange_scalar":
        if layer_idx is None:
            raise ValueError("layer_idx is required for routed_midrange_scalar")
        routing_path = str(
            getattr(quant_config, "routed_gaussian_routing_path", "") or ""
        )
        if not routing_path:
            raise ValueError(
                "routed_gaussian_routing_path is required for "
                "routed_midrange_scalar"
            )
        residual_quant = encode_routed_midrange_scalar_residual(
            residual_bf16,
            tensor_name=canonical_tensor_name.upper(),
            layer_idx=int(layer_idx),
            routing_table=routing_path,
            bits=int(
                getattr(quant_config, "routed_midrange_scalar_bits", 4)
            ),
        )
        quant_method = "rvq_routed_midrange_scalar_residual"
    elif residual_codec == "static_equalized_midrange":
        if layer_idx is None:
            raise ValueError(
                "layer_idx is required for static_equalized_midrange"
            )
        params_path = str(
            getattr(
                quant_config,
                "static_equalized_midrange_params_path",
                "",
            )
            or ""
        )
        if not params_path:
            raise ValueError(
                "static_equalized_midrange_params_path is required"
            )
        residual_quant = encode_static_equalized_midrange_residual(
            residual_bf16,
            tensor_name=canonical_tensor_name.upper(),
            layer_idx=int(layer_idx),
            params=params_path,
            bits=4,
        )
        quant_method = "rvq_static_equalized_midrange_residual"
    else:
        raise ValueError(
            f"Unsupported RVQ residual codec {residual_codec!r}; "
            "expected quantvideo, kivi, higgs, truncatable, structured_truncatable, "
            "structured_static_prefix, routed_gaussian, "
            "uniform_grid_gaussian, frame_mean_gaussian, "
            "shrunk_frame_mean_gaussian, cache_mean_gaussian, "
            "selective_rank1_gaussian, absmax_scalar, "
            "sign_adaptive_absmax_scalar, "
            "progressive_sign_adaptive_absmax_scalar, "
            "centered_absmax_scalar, "
            "routed_midrange_scalar, "
            "static_equalized_midrange, or r2aq"
        )
    return {
        "quant_method": quant_method,
        "base": rvq_base,
        "residual": residual_quant,
        "residual_codec": residual_codec,
        "residual_quant_type": quant_type,
        "residual_source_dtype": "bfloat16",
    }


def compress_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    quant_type: str,
    quant_config: QuantizeConfig,
    quantize_fn: callable,
    layer_idx: int | None = None,
    span_idx: int | None = None,
    rvq_feedback_state: dict | None = None,
    rvq_feedback_query: torch.Tensor | None = None,
):
    quantize_k = getattr(quant_config, "quantize_k", True)
    quantize_v = getattr(quant_config, "quantize_v", True)
    k_rvq = getattr(quant_config, "k_rvq", False)
    v_rvq = getattr(quant_config, "v_rvq", False)
    use_oscar = quant_type == "oscar-int2"
    kivi_match = re.fullmatch(r"kivi-all-int(2|4)", quant_type)
    turbo_match = re.fullmatch(
        r"turboquant-(mse|prod)-k(2|3|4)-v(2|3|4)", quant_type
    )
    if turbo_match is not None:
        turbo_setting, turbo_k_bits_text, turbo_v_bits_text = turbo_match.groups()
        turbo_k_bits = int(turbo_k_bits_text)
        turbo_v_bits = int(turbo_v_bits_text)
        if (turbo_k_bits, turbo_v_bits) not in {
            (2, 2), (3, 2), (4, 3), (4, 4)
        }:
            raise ValueError(f"unsupported TurboQuant K/V bit pair in {quant_type!r}")
    else:
        turbo_setting = None
        turbo_k_bits = turbo_v_bits = None
    rvq_feedback_method = getattr(quant_config, "rvq_feedback_method", "none")
    rvq_feedback_tensors = str(getattr(quant_config, "rvq_feedback_tensors", "kv")).lower()
    rvq_feedback_lambda = float(getattr(quant_config, "rvq_feedback_lambda", 0.0))
    rvq_feedback_beta = float(getattr(quant_config, "rvq_feedback_beta", 0.0))
    rvq_progressive = getattr(quant_config, "rvq_progressive", False)
    rvq_progressive_min_depth = int(getattr(quant_config, "rvq_progressive_min_depth", 0))
    rvq_progressive_max_depth = int(getattr(quant_config, "rvq_progressive_max_depth", 0))
    rvq_progressive_error_threshold = float(getattr(quant_config, "rvq_progressive_error_threshold", 0.0))
    rvq_progressive_granularity = str(getattr(quant_config, "rvq_progressive_granularity", "token"))
    rvq_progressive_block_size = int(getattr(quant_config, "rvq_progressive_block_size", 880))
    rvq_relative_stop = bool(getattr(quant_config, "rvq_relative_stop", False))
    rvq_relative_stop_threshold = float(
        getattr(quant_config, "rvq_relative_stop_threshold", 0.0629)
    )
    rvq_relative_stop_min_depth = int(
        getattr(quant_config, "rvq_relative_stop_min_depth", 4)
    )
    rvq_relative_stop_max_depth = int(
        getattr(quant_config, "rvq_relative_stop_max_depth", 16)
    )
    rvq_relative_stop_fast_approximate = bool(
        getattr(
            quant_config,
            "rvq_relative_stop_fast_approximate",
            False,
        )
    )
    rvq_relative_stop_monolithic_exact = bool(
        getattr(quant_config, "rvq_relative_stop_monolithic_exact", False)
    )
    rvq_relative_stop_exact_staged_k_layers = tuple(
        int(value)
        for value in getattr(
            quant_config, "rvq_relative_stop_exact_staged_k_layers", ()
        )
    )
    rvq_relative_stop_exact_staged_v_layers = tuple(
        int(value)
        for value in getattr(
            quant_config, "rvq_relative_stop_exact_staged_v_layers", ()
        )
    )
    rvq_k_calibration = _rvq_calibration_mode(quant_config, "k")
    rvq_v_calibration = _rvq_calibration_mode(quant_config, "v")
    rvq_diagnostics = bool(getattr(quant_config, "rvq_diagnostics", False))
    rvq_residual_quantize = bool(
        getattr(quant_config, "rvq_residual_quantize", False)
    )

    def feedback_enabled_for(name: str) -> bool:
        if str(rvq_feedback_method).lower() in {"", "none", "off", "false", "0"}:
            return False
        if rvq_feedback_tensors == "kv":
            return True
        return name.lower() == rvq_feedback_tensors

    def tensor_feedback_state(name: str) -> dict | None:
        if not isinstance(rvq_feedback_state, dict) or layer_idx is None:
            return None
        return rvq_feedback_state.setdefault((int(layer_idx), name.lower()), {})

    use_attn_feedback = str(rvq_feedback_method).lower().replace("-", "_") in {
        "attn_output",
        "attention_output",
        "attn",
        "attention",
    }
    feedback_query = _rvq_feedback_query_like(rvq_feedback_query, k) if use_attn_feedback else None
    calibration_query = (
        _rvq_calibration_query_like(rvq_feedback_query, k)
        if rvq_diagnostics or "online" in {rvq_k_calibration, rvq_v_calibration}
        else None
    )
    k_feedback_state = tensor_feedback_state("k") if feedback_enabled_for("k") else None
    v_feedback_state = tensor_feedback_state("v") if feedback_enabled_for("v") else None

    if quantize_k and kivi_match is not None:
        k_quant = encode_kivi_tensor(
            k,
            tensor_name="k",
            bits=int(kivi_match.group(1)),
            group_size=int(getattr(quant_config, "kivi_group_size", 32)),
        )
    elif quantize_k and turbo_match is not None:
        turbo_seed = int(getattr(quant_config, "turboquant_seed", 20260803))
        if turbo_setting == "prod":
            k_quant = encode_turboquant_prod(
                k, bits=int(turbo_k_bits), seed=turbo_seed
            )
        else:
            k_quant = encode_turboquant_mse(
                k, bits=int(turbo_k_bits), seed=turbo_seed
            )
    elif quantize_k and use_oscar:
        if layer_idx is None:
            raise ValueError("layer_idx is required for OSCAR quantization")
        k_quant = oscar_int2_quantize_tensor(
            k,
            layer_idx=int(layer_idx),
            tensor_name="k",
            rotation_path=getattr(quant_config, "oscar_k_rotation_path"),
            clip_percentile=getattr(quant_config, "oscar_k_clip_percentile", 1.0),
        )
    elif quantize_k and k_rvq:
        if layer_idx is None:
            raise ValueError("layer_idx is required when quant_config.k_rvq is enabled")
        k_guidance = k_curvature = None
        if use_attn_feedback and feedback_enabled_for("k"):
            k_guidance, k_curvature = _rvq_attn_k_guidance(
                feedback_query,
                k_feedback_state,
                k.shape[2],
            )
        encode_timer = _start_encode_phase("rvq_base_k", k.device)
        k_quant = rvq_encode_tensor_cache(
            k,
            tensor_name="K",
            layer_idx=int(layer_idx),
            codebook_dir=getattr(quant_config, "k_rvq_codebook_dir"),
            batch_size=getattr(quant_config, "k_rvq_batch_size", 2048),
            compute_dtype=getattr(quant_config, "k_rvq_compute_dtype", "bf16"),
            codebook_storage_dtype=getattr(quant_config, "k_rvq_codebook_storage_dtype", "bf16"),
            scale_storage_dtype=getattr(quant_config, "k_rvq_scale_storage_dtype", "bf16"),
            feedback_method=rvq_feedback_method if feedback_enabled_for("k") else "none",
            feedback_lambda=rvq_feedback_lambda,
            feedback_beta=rvq_feedback_beta,
            feedback_state=k_feedback_state,
            feedback_guidance=k_guidance,
            feedback_curvature=k_curvature,
            progressive=rvq_progressive,
            progressive_min_depth=rvq_progressive_min_depth,
            progressive_max_depth=rvq_progressive_max_depth,
            progressive_error_threshold=rvq_progressive_error_threshold,
            progressive_granularity=rvq_progressive_granularity,
            progressive_block_size=rvq_progressive_block_size,
            relative_stop=rvq_relative_stop,
            relative_stop_threshold=rvq_relative_stop_threshold,
            relative_stop_min_depth=rvq_relative_stop_min_depth,
            relative_stop_max_depth=rvq_relative_stop_max_depth,
            relative_stop_fast_approximate=(
                rvq_relative_stop_fast_approximate
            ),
            relative_stop_monolithic_exact=(
                rvq_relative_stop_monolithic_exact
            ),
            relative_stop_exact_staged_layers=(
                rvq_relative_stop_exact_staged_k_layers
            ),
        )
        _stop_encode_phase(encode_timer)
        if rvq_k_calibration != "none":
            k_recon = rvq_dequantize_tensor_cache(k_quant, k.dtype)
            k_quant = _attach_rvq_calibration(
                k_quant,
                name="k",
                mode=rvq_k_calibration,
                layer_idx=int(layer_idx),
                ref=k,
                recon=k_recon,
                quant_config=quant_config,
                query=calibration_query,
            )
            del k_recon
        if rvq_diagnostics:
            k_diag_recon = rvq_dequantize_tensor_cache(k_quant, k.dtype)
            _record_rvq_diagnostic(
                k_quant,
                "k_attention_kl",
                _k_attention_kl(calibration_query, k, k_diag_recon),
            )
            del k_diag_recon
        if use_attn_feedback and feedback_enabled_for("k"):
            _rvq_update_attn_k_error(feedback_query, k, k_feedback_state)
    elif quantize_k:
        encode_timer = _start_encode_phase("direct_quantvideo_k", k.device)
        k_quant = compress_tensor_cache(
            k,
            quant_type,
            quant_config,
            quantize_fn,
            quant_config.cache_num_k_centroids,
        )
        _stop_encode_phase(encode_timer)
    else:
        k_quant = k

    if quantize_v and kivi_match is not None:
        v_quant = encode_kivi_tensor(
            v,
            tensor_name="v",
            bits=int(kivi_match.group(1)),
            group_size=int(getattr(quant_config, "kivi_group_size", 32)),
        )
    elif quantize_v and turbo_match is not None:
        v_quant = encode_turboquant_mse(
            v,
            bits=int(turbo_v_bits),
            seed=int(getattr(quant_config, "turboquant_seed", 20260803)),
        )
    elif quantize_v and use_oscar:
        if layer_idx is None:
            raise ValueError("layer_idx is required for OSCAR quantization")
        v_quant = oscar_int2_quantize_tensor(
            v,
            layer_idx=int(layer_idx),
            tensor_name="v",
            rotation_path=getattr(quant_config, "oscar_v_rotation_path"),
            clip_percentile=getattr(quant_config, "oscar_v_clip_percentile", 1.0),
        )
    elif quantize_v and v_rvq:
        if layer_idx is None:
            raise ValueError("layer_idx is required when quant_config.v_rvq is enabled")
        v_guidance = v_curvature = v_attn = None
        if use_attn_feedback and feedback_enabled_for("v"):
            v_guidance, v_curvature, v_attn = _rvq_attn_v_guidance(
                feedback_query,
                k,
                v_feedback_state,
            )
        encode_timer = _start_encode_phase("rvq_base_v", v.device)
        v_quant = rvq_encode_tensor_cache(
            v,
            tensor_name="V",
            layer_idx=int(layer_idx),
            codebook_dir=getattr(quant_config, "v_rvq_codebook_dir"),
            batch_size=getattr(quant_config, "v_rvq_batch_size", 2048),
            compute_dtype=getattr(quant_config, "v_rvq_compute_dtype", "bf16"),
            codebook_storage_dtype=getattr(quant_config, "v_rvq_codebook_storage_dtype", "bf16"),
            scale_storage_dtype=getattr(quant_config, "v_rvq_scale_storage_dtype", "bf16"),
            feedback_method=rvq_feedback_method if feedback_enabled_for("v") else "none",
            feedback_lambda=rvq_feedback_lambda,
            feedback_beta=rvq_feedback_beta,
            feedback_state=v_feedback_state,
            feedback_guidance=v_guidance,
            feedback_curvature=v_curvature,
            progressive=rvq_progressive,
            progressive_min_depth=rvq_progressive_min_depth,
            progressive_max_depth=rvq_progressive_max_depth,
            progressive_error_threshold=rvq_progressive_error_threshold,
            progressive_granularity=rvq_progressive_granularity,
            progressive_block_size=rvq_progressive_block_size,
            relative_stop=rvq_relative_stop,
            relative_stop_threshold=rvq_relative_stop_threshold,
            relative_stop_min_depth=rvq_relative_stop_min_depth,
            relative_stop_max_depth=rvq_relative_stop_max_depth,
            relative_stop_fast_approximate=(
                rvq_relative_stop_fast_approximate
            ),
            relative_stop_monolithic_exact=(
                rvq_relative_stop_monolithic_exact
            ),
            relative_stop_exact_staged_layers=(
                rvq_relative_stop_exact_staged_v_layers
            ),
        )
        _stop_encode_phase(encode_timer)
        if rvq_v_calibration != "none":
            v_recon = rvq_dequantize_tensor_cache(v_quant, v.dtype)
            v_quant = _attach_rvq_calibration(
                v_quant,
                name="v",
                mode=rvq_v_calibration,
                layer_idx=int(layer_idx),
                ref=v,
                recon=v_recon,
                quant_config=quant_config,
                query=calibration_query,
                key_for_v=k,
            )
            del v_recon
        if rvq_diagnostics:
            v_diag_recon = rvq_dequantize_tensor_cache(v_quant, v.dtype)
            for diagnostic_name, diagnostic_value in _reconstruction_diagnostics(
                v, v_diag_recon
            ).items():
                _record_rvq_diagnostic(v_quant, f"v_{diagnostic_name}", diagnostic_value)
            _record_rvq_diagnostic(
                v_quant,
                "v_output_mse",
                _v_attention_output_mse(calibration_query, k, v, v_diag_recon),
            )
            del v_diag_recon
        if use_attn_feedback and feedback_enabled_for("v"):
            _rvq_update_attn_v_error(v_attn, v, v_feedback_state)
    elif quantize_v:
        encode_timer = _start_encode_phase("direct_quantvideo_v", v.device)
        v_quant = compress_tensor_cache(
            v,
            quant_type,
            quant_config,
            quantize_fn,
            quant_config.cache_num_v_centroids,
        )
        _stop_encode_phase(encode_timer)
    else:
        v_quant = v

    if rvq_residual_quantize:
        if quantize_k and k_rvq:
            encode_timer = _start_encode_phase("residual_materialize_k", k.device)
            k_quant = _quantize_rvq_residual(
                k,
                k_quant,
                tensor_name="k",
                quant_type=quant_type,
                quant_config=quant_config,
                quantize_fn=quantize_fn,
                num_centroids=quant_config.cache_num_k_centroids,
                layer_idx=layer_idx,
                span_idx=span_idx,
            )
            _stop_encode_phase(encode_timer)
        if quantize_v and v_rvq:
            encode_timer = _start_encode_phase("residual_materialize_v", v.device)
            v_quant = _quantize_rvq_residual(
                v,
                v_quant,
                tensor_name="v",
                quant_type=quant_type,
                quant_config=quant_config,
                quantize_fn=quantize_fn,
                num_centroids=quant_config.cache_num_v_centroids,
                layer_idx=layer_idx,
                span_idx=span_idx,
            )
            _stop_encode_phase(encode_timer)

    return k_quant, v_quant
