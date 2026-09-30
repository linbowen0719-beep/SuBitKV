# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from platform import processor
from typing import Any, Dict, Optional, Tuple, Union, List

import os
import torch
import json
import datetime
import time
import torch.nn as nn
import torch.nn.functional as F

from hyvideo.prope.camera_rope import prope_qkv
from distributed.communication_op import (
    all_to_all_sp,
    sequence_model_parallel_all_gather,
)
from distributed.parallel_state import get_sp_parallel_rank, get_sp_world_size

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm

try:
    from sageattention import sageattn
except ImportError:
    import warnings

    warnings.warn("SageAttention is unavailable; Wan uses PyTorch SDPA.", stacklevel=2)

    def sageattn(query, key, value, *, tensor_layout="HND", is_causal=False):
        if tensor_layout != "HND":
            raise ValueError("Wan SDPA compatibility path requires HND tensors")
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=is_causal
        )
from quant_videogen.rvq import q_rvq_enabled, rvq_reconstruct_query



logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

TOKENS_PER_FRAME = 880
DEFAULT_CHUNK_FRAMES = 4


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no", ""}


def _gather_rotary_emb_by_frame_indices(
    full_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    frame_indices: Optional[List[int]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not frame_indices:
        return None
    device = full_rotary_emb[0].device
    frames = torch.as_tensor(frame_indices, dtype=torch.long, device=device)
    frame_tokens = torch.arange(TOKENS_PER_FRAME, dtype=torch.long, device=device)
    token_indices = (
        frames[:, None] * TOKENS_PER_FRAME + frame_tokens[None, :]
    ).reshape(-1)
    return (
        full_rotary_emb[0].index_select(2, token_indices),
        full_rotary_emb[1].index_select(2, token_indices),
    )


def _kv_cache_mode() -> str:
    return os.getenv("HYWORLDPLAY_KV_CACHE_MODE", "final").lower()


def _use_pre_kv_reconstruct(kv_cache: Optional[dict]) -> bool:
    if kv_cache is not None and kv_cache.get("cache_mode") == "pre_kv_reconstruct":
        return True
    return _kv_cache_mode() == "pre_kv_reconstruct"


def _save_pre_rope_key(
    key: torch.Tensor,
    *,
    layer_idx: Optional[int],
    token_start: int,
    token_end: int,
) -> None:
    """Save K after projection/norm and before RoPE/PRoPE."""
    if not _env_enabled("HYWORLDPLAY_SAVE_PRE_ROPE_KEY"):
        return

    save_root = os.getenv("HYWORLDPLAY_PRE_ROPE_KEY_SAVE_DIR")
    if not save_root:
        return
    if token_start >= token_end:
        return

    rank = int(os.getenv("RANK", "0"))
    frame_start = token_start // TOKENS_PER_FRAME
    frame_end = token_end // TOKENS_PER_FRAME
    chunk_i = (
        frame_end // DEFAULT_CHUNK_FRAMES
        if frame_end % DEFAULT_CHUNK_FRAMES == 0
        else frame_end
    )
    layer = int(layer_idx if layer_idx is not None else -1)
    save_dir = os.path.join(
        save_root,
        f"rank_{rank:03d}",
        f"chunk_{chunk_i:04d}_{token_start}_{token_end}",
    )
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"layer_{layer:02d}.pt")
    key_to_save = key[:, :, token_start:token_end, :].detach().cpu()
    torch.save(
        {
            "key_pre_rope": key_to_save,
            "token_start": token_start,
            "token_end": token_end,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "chunk_i": chunk_i,
            "layer_idx": layer,
            "rank": rank,
            "layout": "BHSD",
            "note": "K after to_k/norm_k and before RoPE/PRoPE.",
        },
        save_path,
    )
    del key_to_save

    metadata_path = os.path.join(save_dir, "metadata.json")
    if layer == 0 or not os.path.exists(metadata_path):
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "chunk_i": chunk_i,
                    "rank": rank,
                    "token_start": token_start,
                    "token_end": token_end,
                    "tokens_per_frame": TOKENS_PER_FRAME,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "layout": "BHSD",
                    "dtype": str(key.dtype),
                    "num_layers": 30,
                    "note": "Pre-RoPE K: after to_k/norm_k, before RoPE and PRoPE.",
                },
                f,
                indent=2,
            )


def _rvq_diag_enabled() -> bool:
    return _env_enabled("HYWORLDPLAY_RVQ_DIAG_DUMP")


def _rvq_diag_layers() -> Optional[set[int]]:
    raw = os.getenv("HYWORLDPLAY_RVQ_DIAG_LAYERS", "").strip()
    if not raw:
        return None
    return {int(item) for item in raw.split(",") if item.strip()}


def _attention_diag_layers() -> Optional[set[int]]:
    raw = os.getenv("HYWORLDPLAY_ATTENTION_DIAG_LAYERS", "").strip()
    if not raw:
        return None
    return {int(item) for item in raw.split(",") if item.strip()}


def _paired_rope_diag_enabled() -> bool:
    return _env_enabled("HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP")


def _paired_rope_diag_layers() -> Optional[set[int]]:
    raw = os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_LAYERS", "").strip()
    if not raw:
        return None
    return {int(item) for item in raw.split(",") if item.strip()}


def _sample_sequence_tensor(
    tensor: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return tensor.index_select(2, indices.to(tensor.device)).detach().cpu()


def _sample_frame_metadata(
    tensor: Optional[torch.Tensor],
    token_indices: torch.Tensor,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if token_indices.numel() == 0:
        return tensor[:, :0].detach().cpu()
    frame_indices = torch.div(token_indices, TOKENS_PER_FRAME, rounding_mode="floor")
    max_frame = int(frame_indices.max().item())
    if max_frame >= tensor.shape[1]:
        raise ValueError(
            "Paired RoPE diagnostic frame metadata is shorter than its token axis: "
            f"max sampled frame={max_frame}, metadata frames={tensor.shape[1]}."
        )
    return tensor.index_select(1, frame_indices.to(tensor.device)).detach().cpu()


def _reserve_paired_rope_diag_call(
    kv_cache: dict,
    *,
    layer_idx: Optional[int],
    cache_tokens: int,
) -> Optional[int]:
    """Reserve one bounded diagnostic call before materializing full pre tensors."""
    if not _paired_rope_diag_enabled():
        return None

    layer = int(layer_idx if layer_idx is not None else -1)
    layers = _paired_rope_diag_layers()
    if layers is not None and layer not in layers:
        return None
    if int(os.getenv("RANK", "0")) != 0:
        return None
    if not os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP_DIR"):
        raise ValueError(
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP_DIR is required when "
            "HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP is enabled."
        )

    min_cache_tokens = int(
        os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_MIN_CACHE_TOKENS", "1")
    )
    if cache_tokens < min_cache_tokens:
        return None
    max_calls = int(
        os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_MAX_CALLS_PER_LAYER", "1")
    )
    if max_calls <= 0:
        return None

    call_idx = int(kv_cache.get("_paired_rope_diag_call_count", 0))
    kv_cache["_paired_rope_diag_call_count"] = call_idx + 1
    return call_idx if call_idx < max_calls else None


def _validate_paired_rope_cache_mode(
    *,
    cache_tokens: int,
    pre_kv_reconstruct: bool,
) -> None:
    if cache_tokens > 0 and not pre_kv_reconstruct:
        raise ValueError(
            "Paired pre/post-RoPE diagnostics require "
            "HYWORLDPLAY_KV_CACHE_MODE=pre_kv_reconstruct when historical "
            "KV is present; the final cache does not retain pre-RoPE K/V."
        )


def _save_paired_rope_diag_sample(
    *,
    query_pre: torch.Tensor,
    key_pre: torch.Tensor,
    value_pre: torch.Tensor,
    query_post_rope: torch.Tensor,
    key_post_rope: torch.Tensor,
    value_post_rope: torch.Tensor,
    query_post_prope: torch.Tensor,
    key_post_prope: torch.Tensor,
    value_post_prope: torch.Tensor,
    query_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    key_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    query_position_indices: torch.Tensor,
    key_position_indices: torch.Tensor,
    query_viewmats: Optional[torch.Tensor],
    query_Ks: Optional[torch.Tensor],
    key_viewmats: Optional[torch.Tensor],
    key_Ks: Optional[torch.Tensor],
    kv_cache: dict,
    layer_idx: Optional[int],
    cache_tokens: int,
    current_token_start: int,
    call_idx: Optional[int] = None,
) -> None:
    """Save bounded, index-aligned pre/post RoPE and PRoPE attention samples."""
    layer = int(layer_idx if layer_idx is not None else -1)
    if call_idx is None:
        call_idx = _reserve_paired_rope_diag_call(
            kv_cache,
            layer_idx=layer_idx,
            cache_tokens=cache_tokens,
        )
    if call_idx is None:
        return

    rank = int(os.getenv("RANK", "0"))
    save_root = os.environ["HYWORLDPLAY_PAIRED_ROPE_DIAG_DUMP_DIR"]
    q_len = int(query_pre.shape[2])
    k_len = int(key_pre.shape[2])
    q_tensors = (query_post_rope, query_post_prope)
    k_tensors = (key_post_rope, key_post_prope)
    v_tensors = (value_post_rope, value_post_prope)
    if any(tensor.shape != query_pre.shape for tensor in q_tensors):
        raise ValueError("Paired RoPE diagnostic query/key/value sequence lengths are not aligned.")
    if any(tensor.shape != key_pre.shape for tensor in k_tensors):
        raise ValueError("Paired RoPE diagnostic pre/post key sequence lengths are not aligned.")
    if any(tensor.shape != value_pre.shape for tensor in v_tensors):
        raise ValueError("Paired RoPE diagnostic pre/post value sequence lengths are not aligned.")
    if query_position_indices.numel() != q_len or key_position_indices.numel() != k_len:
        raise ValueError("Paired RoPE diagnostic position indices are not aligned with Q/K.")
    if any(emb.shape[2] != q_len for emb in query_rotary_emb):
        raise ValueError("Paired RoPE diagnostic query cos/sin are not aligned with Q.")
    if any(emb.shape[2] != k_len for emb in key_rotary_emb):
        raise ValueError("Paired RoPE diagnostic key cos/sin are not aligned with K.")

    query_sample = min(
        max(int(os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_QUERY_SAMPLE", "64")), 0),
        q_len,
    )
    key_sample = min(
        max(int(os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_KEY_SAMPLE", "256")), 0),
        k_len,
    )
    if query_sample == 0 or key_sample == 0:
        return

    seed = int(os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_SEED", "42"))
    generator = torch.Generator(device="cpu").manual_seed(
        seed + layer * 100000 + call_idx
    )
    q_idx = torch.randperm(q_len, generator=generator)[:query_sample].sort().values
    k_idx = torch.randperm(k_len, generator=generator)[:key_sample].sort().values

    save_dir = os.path.join(save_root, f"rank_{rank:03d}", f"layer_{layer:02d}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"call_{call_idx:05d}.pt")
    if os.path.exists(save_path):
        raise FileExistsError(
            f"Refusing to overwrite paired RoPE diagnostic sample: {save_path}"
        )
    lean = _env_enabled("HYWORLDPLAY_PAIRED_ROPE_DIAG_LEAN")
    payload = {
            "query_pre": _sample_sequence_tensor(query_pre, q_idx),
            "key_pre": _sample_sequence_tensor(key_pre, k_idx),
            "value_pre": _sample_sequence_tensor(value_pre, k_idx),
            "query_post_rope": _sample_sequence_tensor(query_post_rope, q_idx),
            "query_post_prope": _sample_sequence_tensor(query_post_prope, q_idx),
            "query_indices": q_idx.cpu(),
            "key_indices": k_idx.cpu(),
            "query_position_indices": query_position_indices.index_select(0, q_idx).cpu(),
            "key_position_indices": key_position_indices.index_select(0, k_idx).cpu(),
            "query_rope_cos": _sample_sequence_tensor(query_rotary_emb[0], q_idx),
            "query_rope_sin": _sample_sequence_tensor(query_rotary_emb[1], q_idx),
            "key_rope_cos": _sample_sequence_tensor(key_rotary_emb[0], k_idx),
            "key_rope_sin": _sample_sequence_tensor(key_rotary_emb[1], k_idx),
            "query_viewmats": _sample_frame_metadata(query_viewmats, q_idx),
            "query_Ks": _sample_frame_metadata(query_Ks, q_idx),
            "key_viewmats": _sample_frame_metadata(key_viewmats, k_idx),
            "key_Ks": _sample_frame_metadata(key_Ks, k_idx),
            "metadata": {
                "layer": layer,
                "call_idx": call_idx,
                "cache_mode": str(kv_cache.get("cache_mode", "final")),
                "cache_tokens": int(cache_tokens),
                "current_token_start": int(current_token_start),
                "query_len": q_len,
                "key_len": k_len,
                "query_sample": query_sample,
                "key_sample": key_sample,
                "seed": seed,
                "tokens_per_frame": TOKENS_PER_FRAME,
                "layout": "BHSD",
                "alignment": "exact_same_call_same_indices",
                "lean_reference_reconstructed": lean,
                "rope_halves": ["standard_rope", "camera_prope"],
                "note": (
                    "Pre tensors are after Q/K projection and norm, before positional "
                    "transforms. Full post tensors are stored in the default format; lean "
                    "format stores deterministic post-validation points and reconstructs "
                    "the full reference from pre tensors. Camera matrices are repeated "
                    "per sampled token."
                ),
            },
        }
    if lean:
        validation_count = min(
            int(os.getenv("HYWORLDPLAY_PAIRED_ROPE_DIAG_POST_VALIDATION_SAMPLE", "64")),
            key_sample,
        )
        validation_offsets = torch.linspace(
            0, key_sample - 1, steps=validation_count, dtype=torch.float64
        ).round().long().unique(sorted=True)
        validation_indices = k_idx.index_select(0, validation_offsets)
        payload.update(
            {
                "post_validation_offsets": validation_offsets.cpu(),
                "key_post_rope_validation": _sample_sequence_tensor(
                    key_post_rope, validation_indices
                ),
                "value_post_rope_validation": _sample_sequence_tensor(
                    value_post_rope, validation_indices
                ),
                "key_post_prope_validation": _sample_sequence_tensor(
                    key_post_prope, validation_indices
                ),
                "value_post_prope_validation": _sample_sequence_tensor(
                    value_post_prope, validation_indices
                ),
            }
        )
        payload["metadata"]["post_validation_sample"] = int(validation_offsets.numel())
    else:
        payload.update(
            {
                "key_post_rope": _sample_sequence_tensor(key_post_rope, k_idx),
                "value_post_rope": _sample_sequence_tensor(value_post_rope, k_idx),
                "key_post_prope": _sample_sequence_tensor(key_post_prope, k_idx),
                "value_post_prope": _sample_sequence_tensor(value_post_prope, k_idx),
            }
        )
    torch.save(payload, save_path)


def _save_attention_diag_sample(
    query_all: torch.Tensor,
    key_all: torch.Tensor,
    value_all: torch.Tensor,
    *,
    kv_cache: Optional[dict],
    layer_idx: Optional[int],
    cache_tokens: int,
) -> None:
    """Save bounded true denoise-attention Q/K/V samples for cache diagnostics."""
    if not _env_enabled("HYWORLDPLAY_ATTENTION_DIAG_DUMP"):
        return
    if kv_cache is None or cache_tokens <= 0:
        return
    min_cache_tokens = int(os.getenv("HYWORLDPLAY_ATTENTION_DIAG_MIN_CACHE_TOKENS", "1"))
    if cache_tokens < min_cache_tokens:
        return

    layer = int(layer_idx if layer_idx is not None else -1)
    layers = _attention_diag_layers()
    if layers is not None and layer not in layers:
        return

    rank = int(os.getenv("RANK", "0"))
    if rank != 0:
        return

    save_root = os.getenv("HYWORLDPLAY_ATTENTION_DIAG_DUMP_DIR")
    if not save_root:
        return

    max_calls = int(os.getenv("HYWORLDPLAY_ATTENTION_DIAG_MAX_CALLS_PER_LAYER", "1"))
    if max_calls <= 0:
        return
    call_idx = int(kv_cache.get("_attention_diag_call_count", 0))
    kv_cache["_attention_diag_call_count"] = call_idx + 1
    if call_idx >= max_calls:
        return

    q_len = query_all.shape[2]
    k_len = min(cache_tokens, key_all.shape[2])
    if q_len <= 0 or k_len <= 0:
        return

    query_sample = min(int(os.getenv("HYWORLDPLAY_ATTENTION_DIAG_QUERY_SAMPLE", "64")), q_len)
    key_sample = min(int(os.getenv("HYWORLDPLAY_ATTENTION_DIAG_KEY_SAMPLE", "256")), k_len)
    seed = int(os.getenv("HYWORLDPLAY_ATTENTION_DIAG_SEED", "42"))
    generator = torch.Generator(device="cpu").manual_seed(
        seed + layer * 100000 + call_idx
    )
    q_idx = torch.randperm(q_len, generator=generator)[:query_sample].sort().values
    k_idx = torch.randperm(k_len, generator=generator)[:key_sample].sort().values

    save_dir = os.path.join(save_root, f"rank_{rank:03d}", f"layer_{layer:02d}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"call_{call_idx:05d}.pt")
    torch.save(
        {
            "query": query_all[:, :, q_idx.to(query_all.device), :].detach().cpu(),
            "key": key_all[:, :, k_idx.to(key_all.device), :].detach().cpu(),
            "value": value_all[:, :, k_idx.to(value_all.device), :].detach().cpu(),
            "query_indices": q_idx.cpu(),
            "key_indices": k_idx.cpu(),
            "metadata": {
                "layer": layer,
                "call_idx": call_idx,
                "cache_tokens": int(cache_tokens),
                "query_len": int(q_len),
                "key_len": int(key_all.shape[2]),
                "layout": "BHSD with first dim containing rope/prope halves",
                "note": (
                    "Samples are taken from the tensors passed to "
                    "scaled_dot_product_attention during is_cache=False."
                ),
            },
        },
        save_path,
    )


def _store_rvq_feedback_query_sample(
    query: torch.Tensor,
    *,
    kv_cache: Optional[dict],
    layer_idx: Optional[int],
) -> None:
    """Keep a bounded pre-RoPE Q sample for transient RVQ feedback."""
    if kv_cache is None:
        return
    sample_size = int(kv_cache.get("_rvq_feedback_query_sample_size", 0) or 0)
    if sample_size <= 0 or query.ndim != 4:
        return

    q_len = int(query.shape[2])
    if q_len <= 0:
        return
    take = min(sample_size, q_len)
    layer = int(layer_idx if layer_idx is not None else -1)
    call_idx = int(kv_cache.get("_rvq_feedback_query_call_count", 0))
    kv_cache["_rvq_feedback_query_call_count"] = call_idx + 1
    seed = int(kv_cache.get("_rvq_feedback_query_seed", 42))
    generator = torch.Generator(device="cpu").manual_seed(
        seed + layer * 100000 + call_idx
    )
    q_idx = torch.randperm(q_len, generator=generator)[:take].sort().values
    kv_cache["_rvq_feedback_query_sample"] = (
        query[:, :, q_idx.to(query.device), :].detach().to("cpu", dtype=torch.float16)
    )
    kv_cache["_rvq_feedback_query_indices"] = q_idx.cpu()


def _save_rvq_diag_qkv_span(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layer_idx: Optional[int],
    token_start: int,
    token_end: int,
    local_start: Optional[int] = None,
    local_end: Optional[int] = None,
) -> None:
    """Save pre-RoPE Q/K/V spans for the offline RVQ diagnostic pipeline."""
    if not _rvq_diag_enabled():
        return
    if token_start >= token_end:
        return

    layer = int(layer_idx if layer_idx is not None else -1)
    layers = _rvq_diag_layers()
    if layers is not None and layer not in layers:
        return

    rank = int(os.getenv("RANK", "0"))
    if rank != 0:
        return

    save_root = os.getenv("HYWORLDPLAY_RVQ_DIAG_DUMP_DIR")
    split = os.getenv("HYWORLDPLAY_RVQ_DIAG_SPLIT")
    prompt_id = os.getenv("HYWORLDPLAY_RVQ_DIAG_PROMPT_ID")
    if not save_root or not split or prompt_id is None:
        return

    span_dir = os.path.join(
        save_root,
        "kv_dump",
        split,
        str(prompt_id),
        "_spans",
        f"layer_{layer}",
    )
    os.makedirs(span_dir, exist_ok=True)
    save_path = os.path.join(span_dir, f"span_{token_start:06d}_{token_end:06d}.pt")

    overwrite = _env_enabled("HYWORLDPLAY_RVQ_DIAG_OVERWRITE")
    if os.path.exists(save_path) and not overwrite:
        return

    local_start = token_start if local_start is None else local_start
    local_end = token_end if local_end is None else local_end
    q_span = query[:, :, local_start:local_end, :].detach().cpu()
    k_span = key[:, :, local_start:local_end, :].detach().cpu()
    v_span = value[:, :, local_start:local_end, :].detach().cpu()
    torch.save(
        {
            "Q": q_span,
            "K": k_span,
            "V": v_span,
            "metadata": {
                "prompt": os.getenv("HYWORLDPLAY_RVQ_DIAG_PROMPT", ""),
                "split": split,
                "prompt_id": str(prompt_id),
                "layer": layer,
                "token_start": int(token_start),
                "token_end": int(token_end),
                "frame_start": int(token_start // TOKENS_PER_FRAME),
                "frame_end": int(token_end // TOKENS_PER_FRAME),
                "tokens_per_frame": TOKENS_PER_FRAME,
                "layout": "BHSD",
                "key_type": "pre_rope",
                "query_type": "pre_rope",
                "value_type": "pre_prope",
                "note": "Pre-RoPE Q/K/V after projection/norm, before RoPE/PRoPE.",
            },
        },
        save_path,
    )
    del q_span, k_span


class CausalCameraPRopeWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        is_cache: Optional[bool] = False,
        idx: Optional[int] = None,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        cache_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_viewmats: Optional[torch.Tensor] = None,
        cache_Ks: Optional[torch.Tensor] = None,
        context_frames_list: Optional[List[int]] = None,
        current_token_start: Optional[int] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if q_rvq_enabled():
            query = rvq_reconstruct_query(query, idx)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                x1, x2 = x[..., 0], x[..., 1]
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query_rope = apply_rotary_emb(query, *rotary_emb)
            key_rope = apply_rotary_emb(key, *rotary_emb)

        else:
            query_rope = query
            key_rope = key

        pad_size = 0
        if os.getenv("WORLD_SIZE", "0") != "0":
            world_size = get_sp_world_size()
            head_num = query.shape[1]
            if head_num % world_size != 0:
                # pad head_num
                pad_size = world_size - head_num % world_size
                query = torch.nn.functional.pad(query, (0, 0, 0, 0, 0, pad_size))
                key = torch.nn.functional.pad(key, (0, 0, 0, 0, 0, pad_size))
                value = torch.nn.functional.pad(value, (0, 0, 0, 0, 0, pad_size))
                query_rope = torch.nn.functional.pad(
                    query_rope, (0, 0, 0, 0, 0, pad_size)
                )
                key_rope = torch.nn.functional.pad(key_rope, (0, 0, 0, 0, 0, pad_size))

            query = all_to_all_sp(query, scatter_dim=1, gather_dim=2)
            key = all_to_all_sp(key, scatter_dim=1, gather_dim=2)
            value = all_to_all_sp(value, scatter_dim=1, gather_dim=2)
            query_rope = all_to_all_sp(query_rope, scatter_dim=1, gather_dim=2)
            key_rope = all_to_all_sp(key_rope, scatter_dim=1, gather_dim=2)

        value_rope = value

        query_prope, key_prope, value_prope, apply_fn_o = prope_qkv(
            query,
            key,
            value,
            viewmats=viewmats,
            Ks=Ks,
            patches_x=40,  # hard code
            patches_y=22,  # hard code
        )  # [batch, num_heads, seqlen, head_dim]


        kv_cache_return = {}

        end_index = kv_cache.get("end_index", 0)
        cache_key = kv_cache["k"].read(0, end_index) if end_index > 0 else None
        cache_value = kv_cache["v"].read(0, end_index) if end_index > 0 else None

        pre_kv_reconstruct = _use_pre_kv_reconstruct(kv_cache)
        if cache_value is not None and not is_cache:
            if pre_kv_reconstruct:
                if cache_rotary_emb is None:
                    raise ValueError("cache_rotary_emb is required for pre_kv_reconstruct cache mode")
                if cache_viewmats is None:
                    raise ValueError("cache_viewmats is required for pre_kv_reconstruct cache mode")

                cache_key_rope = apply_rotary_emb(cache_key, *cache_rotary_emb)
                cache_value_rope = cache_value
                _, cache_key_prope, cache_value_prope, _ = prope_qkv(
                    cache_key,
                    cache_key,
                    cache_value,
                    viewmats=cache_viewmats,
                    Ks=cache_Ks,
                    patches_x=40,
                    patches_y=22,
                )
            else:
                cache_key_rope, cache_key_prope = cache_key.chunk(2, dim=-1)
                cache_value_rope, cache_value_prope = cache_value.chunk(2, dim=-1)

            key_rope = torch.cat([cache_key_rope, key_rope], dim=-2)
            value_rope = torch.cat([cache_value_rope, value_rope], dim=-2)

            key_prope = torch.cat([cache_key_prope, key_prope], dim=-2)
            value_prope = torch.cat([cache_value_prope, value_prope], dim=-2)

        if is_cache:
            old_end = kv_cache.get("end_index", 0)
            token_offset = int(kv_cache.get("_rvq_diag_token_offset", 0))
            total_tokens = token_offset + key.shape[2]
            dump_only = bool(kv_cache.get("_rvq_diag_dump_only", False))
            if old_end < token_offset or old_end > total_tokens:
                raise ValueError(
                    "Invalid KV cache write range: "
                    f"token_offset={token_offset}, old_end={old_end}, "
                    f"total_tokens={total_tokens}, local_tokens={key.shape[2]}"
                )
            local_start = old_end - token_offset
            local_end = total_tokens - token_offset
            _save_rvq_diag_qkv_span(
                query,
                key,
                value,
                layer_idx=idx,
                token_start=old_end,
                token_end=total_tokens,
                local_start=local_start,
                local_end=local_end,
            )
            if pre_kv_reconstruct:
                new_k = key
                new_v = value
            else:
                new_k = torch.cat([key_rope, key_prope], dim=-1)
                new_v = torch.cat([value_rope, value_prope], dim=-1)
            if not dump_only:
                _save_pre_rope_key(
                    key,
                    layer_idx=idx,
                    token_start=local_start,
                    token_end=local_end,
                )
                kv_cache["k"].write(old_end, total_tokens, new_k[:, :, local_start:local_end, :])
                kv_cache["v"].write(old_end, total_tokens, new_v[:, :, local_start:local_end, :])
                kv_cache["end_index"] = total_tokens
                if _env_enabled("HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD"):
                    kv_cache["k"].offload()
                    kv_cache["v"].offload()

        if kv_cache is not None:
            _store_rvq_feedback_query_sample(
                query,
                kv_cache=kv_cache,
                layer_idx=idx,
            )

        paired_call_idx = None
        if not is_cache and kv_cache is not None:
            paired_call_idx = _reserve_paired_rope_diag_call(
                kv_cache,
                layer_idx=idx,
                cache_tokens=end_index,
            )
        if paired_call_idx is not None:
            _validate_paired_rope_cache_mode(
                cache_tokens=end_index,
                pre_kv_reconstruct=pre_kv_reconstruct,
            )
            if rotary_emb is None:
                raise ValueError("Paired RoPE diagnostics require rotary_emb.")

            if cache_key is not None:
                key_pre_all = torch.cat([cache_key, key], dim=-2)
                value_pre_all = torch.cat([cache_value, value], dim=-2)
                key_rotary_all = (
                    torch.cat([cache_rotary_emb[0], rotary_emb[0]], dim=2),
                    torch.cat([cache_rotary_emb[1], rotary_emb[1]], dim=2),
                )
                key_viewmats_all = torch.cat([cache_viewmats, viewmats], dim=1)
                if cache_Ks is None and Ks is None:
                    key_Ks_all = None
                elif cache_Ks is not None and Ks is not None:
                    key_Ks_all = torch.cat([cache_Ks, Ks], dim=1)
                else:
                    raise ValueError(
                        "Paired RoPE diagnostics require consistent cache/current Ks."
                    )
            else:
                key_pre_all = key
                value_pre_all = value
                key_rotary_all = rotary_emb
                key_viewmats_all = viewmats
                key_Ks_all = Ks

            current_start = (
                int(current_token_start)
                if current_token_start is not None
                else int(end_index)
            )
            query_positions = torch.arange(
                current_start,
                current_start + query.shape[2],
                dtype=torch.long,
            )
            if end_index > 0 and context_frames_list:
                cache_frames = torch.as_tensor(context_frames_list, dtype=torch.long)
                frame_tokens = torch.arange(TOKENS_PER_FRAME, dtype=torch.long)
                cache_positions = (
                    cache_frames[:, None] * TOKENS_PER_FRAME + frame_tokens[None, :]
                ).reshape(-1)
                if cache_positions.numel() != end_index:
                    raise ValueError(
                        "Paired RoPE diagnostic context frame positions do not match "
                        f"the cache length: positions={cache_positions.numel()}, "
                        f"cache_tokens={end_index}."
                    )
            else:
                cache_positions = torch.arange(end_index, dtype=torch.long)
            key_positions = torch.cat([cache_positions, query_positions], dim=0)

            _save_paired_rope_diag_sample(
                query_pre=query,
                key_pre=key_pre_all,
                value_pre=value_pre_all,
                query_post_rope=query_rope,
                key_post_rope=key_rope,
                value_post_rope=value_rope,
                query_post_prope=query_prope,
                key_post_prope=key_prope,
                value_post_prope=value_prope,
                query_rotary_emb=rotary_emb,
                key_rotary_emb=key_rotary_all,
                query_position_indices=query_positions,
                key_position_indices=key_positions,
                query_viewmats=viewmats,
                query_Ks=Ks,
                key_viewmats=key_viewmats_all,
                key_Ks=key_Ks_all,
                kv_cache=kv_cache,
                layer_idx=idx,
                cache_tokens=end_index,
                current_token_start=current_start,
                call_idx=paired_call_idx,
            )
            del key_pre_all, value_pre_all

        if os.getenv("SUBBITKV_CALIBRATION_SETTINGS") and not is_cache and cache_key is not None:
            if not pre_kv_reconstruct:
                raise ValueError("SubBitKV teacher calibration requires pre-RoPE cache")
            from quant_videogen.subbitkv.live_calibration import observe
            def positioned(history_k, history_v):
                hkr = apply_rotary_emb(history_k, *cache_rotary_emb)
                _, hkp, hvp, _ = prope_qkv(history_k, history_k, history_v,
                    viewmats=cache_viewmats, Ks=cache_Ks, patches_x=40, patches_y=22)
                kr = torch.cat((hkr, key_rope[:, :, end_index:]), dim=2)
                vr = torch.cat((history_v, value_rope[:, :, end_index:]), dim=2)
                kp = torch.cat((hkp, key_prope[:, :, end_index:]), dim=2)
                vp = torch.cat((hvp, value_prope[:, :, end_index:]), dim=2)
                return torch.cat((kr, kp), dim=0), torch.cat((vr, vp), dim=0)
            observe(torch.cat((query_rope, query_prope), dim=0), cache_key, cache_value,
                    layer=int(idx), transform=positioned)

        query_all = torch.cat([query_rope, query_prope], dim=0)
        key_all = torch.cat([key_rope, key_prope], dim=0)
        value_all = torch.cat([value_rope, value_prope], dim=0)

        if not is_cache and kv_cache is not None:
            _save_attention_diag_sample(
                query_all,
                key_all,
                value_all,
                kv_cache=kv_cache,
                layer_idx=idx,
                cache_tokens=end_index,
            )

        # Avoid big memory usage
        del query_rope, key_rope, value_rope, query_prope, key_prope, value_prope

        # hidden_states_all = sageattn(
        #     query_all, key_all, value_all, tensor_layout="HND", is_causal=False
        # )

        hidden_states_all = F.scaled_dot_product_attention(
           query_all, key_all, value_all, dropout_p=0.0
        )

        hidden_states_all = hidden_states_all.transpose(
            1, 2
        )  # [batch * 2, seqlen, per_sp_num_heads, head_dim]

        hidden_states_rope, hidden_states_prope = hidden_states_all.chunk(2, dim=0)
        hidden_states_prope = apply_fn_o(hidden_states_prope.transpose(1, 2)).transpose(
            1, 2
        )

        if os.getenv("WORLD_SIZE", "0") != "0":
            hidden_states_rope = all_to_all_sp(
                hidden_states_rope, scatter_dim=1, gather_dim=2
            )
            hidden_states_prope = all_to_all_sp(
                hidden_states_prope, scatter_dim=1, gather_dim=2
            )

        if pad_size != 0:
            hidden_states_rope, _ = hidden_states_rope.split(
                [hidden_states_rope.shape[2] - pad_size, pad_size], dim=2
            )
            hidden_states_prope, _ = hidden_states_prope.split(
                [hidden_states_prope.shape[2] - pad_size, pad_size], dim=2
            )

        hidden_states_prope = hidden_states_prope.flatten(2, 3)
        hidden_states_prope = hidden_states_prope.type_as(query)
        hidden_states_prope = attn.to_out_prope[0](hidden_states_prope)

        hidden_states_rope = hidden_states_rope.flatten(2, 3)
        hidden_states_rope = hidden_states_rope.type_as(query)
        hidden_states_rope = attn.to_out[0](hidden_states_rope)
        hidden_states_rope = attn.to_out[1](hidden_states_rope)

        hidden_states = hidden_states_rope + hidden_states_prope

        return hidden_states, kv_cache_return


class WanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
                x1, x2 = x[..., 0], x[..., 1]
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            assert False
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        # SP 16 Pad

        assert attention_mask is None

        hidden_states = sageattn(
            query, key, value, tensor_layout="HND", is_causal=False
        )
        # hidden_states = F.scaled_dot_product_attention(
        #    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_seq_len, in_features)
            )
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


# add the discrete action
class WanActionTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        action_embed_dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.action_embed_dim = action_embed_dim
        self.dim = dim
        self.time_freq_dim = time_freq_dim

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

    def forward(
        self,
        action: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        action = self.timesteps_proj(action.squeeze(0))
        action_embedder_dtype = next(iter(self.action_embedder.parameters())).dtype
        if (
            action.dtype != action_embedder_dtype
            and action_embedder_dtype != torch.int8
        ):
            action = action.to(action_embedder_dtype)
        action = self.action_embedder(action).type_as(encoder_hidden_states)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        temb = temb + action

        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: Optional[int] = None,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len
            )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
    ):
        timestep = self.timesteps_proj(timestep)

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(
                encoder_hidden_states_image
            )

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, 1, ppf * pph * ppw, -1
        )

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=CausalCameraPRopeWanAttnProcessor2_0(),
        )

        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=WanAttnProcessor2_0(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _expand_adaln(self, shift, scale, gate, hidden_states, temb):
        """Expand AdaLN parameters to full sequence length for diffusion forcing."""
        if hidden_states.shape[0] == temb.shape[0]:
            return shift, scale, gate
        latent_length = temb.shape[0] // hidden_states.shape[0]
        token_length = hidden_states.shape[1] // latent_length
        if os.getenv("WORLD_SIZE", "0") != "0":
            token_length = token_length * get_sp_world_size()
        batch, _, dim = hidden_states.shape
        shift = shift.reshape(batch, -1, dim).repeat_interleave(token_length, dim=1)
        scale = scale.reshape(batch, -1, dim).repeat_interleave(token_length, dim=1)
        gate = gate.reshape(batch, -1, dim).repeat_interleave(token_length, dim=1)
        if os.getenv("WORLD_SIZE", "0") != "0":
            world_size = get_sp_world_size()
            rank = get_sp_parallel_rank()
            shift = shift.chunk(world_size, dim=1)[rank]
            scale = scale.chunk(world_size, dim=1)[rank]
            gate = gate.chunk(world_size, dim=1)[rank]
        return shift, scale, gate

    # @torch.compile
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        kv_cache,
        is_cache,
        idx,
        viewmats,
        Ks,
        cache_rotary_emb=None,
        cache_viewmats=None,
        cache_Ks=None,
        context_frames_list=None,
        current_token_start=None,
        attention_output_perturbation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        # 1. Self-attention
        shift_msa, scale_msa, gate_msa = self._expand_adaln(
            shift_msa, scale_msa, gate_msa, hidden_states, temb
        )
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        del shift_msa, scale_msa

        attn_output, kv_cache_return = self.attn1(
            hidden_states=norm_hidden_states,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            is_cache=is_cache,
            idx=idx,
            viewmats=viewmats,
            Ks=Ks,
            cache_rotary_emb=cache_rotary_emb,
            cache_viewmats=cache_viewmats,
            cache_Ks=cache_Ks,
            context_frames_list=context_frames_list,
            current_token_start=current_token_start,
        )
        if attention_output_perturbation is not None:
            if (
                attention_output_perturbation.shape != attn_output.shape
                or attention_output_perturbation.device != attn_output.device
                or not attention_output_perturbation.is_floating_point()
            ):
                raise ValueError(
                    "attention-output propagation perturbation must match "
                    "the self-attention output"
                )
            attn_output = (
                attn_output.float() + attention_output_perturbation.float()
            ).type_as(attn_output)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )
        del gate_msa

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        c_shift_msa, c_scale_msa, c_gate_msa = self._expand_adaln(
            c_shift_msa, c_scale_msa, c_gate_msa, hidden_states, temb
        )
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        del c_shift_msa, c_scale_msa
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)
        del c_gate_msa

        return hidden_states, kv_cache_return


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanActionTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
            action_embed_dim=8,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.gradient_checkpointing = False
        self.inner_dim = inner_dim

    @staticmethod
    def _propagation_error_stats(error: torch.Tensor) -> dict[str, float]:
        per_token = error.float().square().mean(dim=-1).reshape(-1)
        tail_count = max(1, math.ceil(0.01 * int(per_token.numel())))
        ordered = per_token.sort().values
        return {
            "mean": float(per_token.mean().item()),
            "p99": float(torch.quantile(per_token, 0.99).item()),
            "cvar99": float(ordered[-tail_count:].mean().item()),
            "tokens": int(per_token.numel()),
        }

    def _project_propagation_tokens(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the real output AdaLN/projection without unpatchifying."""

        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(
            2, dim=1
        )
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)
        if shift.shape[0] != hidden_states.shape[0]:
            latent_length = temb.shape[0] // hidden_states.shape[0]
            token_length = hidden_states.shape[1] // latent_length
            shift = shift.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            scale = scale.reshape(
                hidden_states.shape[0], -1, hidden_states.shape[2]
            )
            normalized = (
                self.norm_out(hidden_states.float())
                * (1 + scale.repeat_interleave(token_length, dim=1))
                + shift.repeat_interleave(token_length, dim=1)
            ).type_as(hidden_states)
        else:
            normalized = (
                self.norm_out(hidden_states.float()) * (1 + scale) + shift
            ).type_as(hidden_states)
        return self.proj_out(normalized)

    @staticmethod
    def _reserve_layer_propagation_probe(
        kv_cache: Optional[list[dict]],
        *,
        is_cache: bool,
    ) -> tuple[dict[str, Any], int] | None:
        if (
            not _env_enabled("HYWORLDPLAY_LAYER_PROPAGATION_DIAG")
            or is_cache
            or int(os.getenv("RANK", "0")) != 0
        ):
            return None
        if os.getenv("WORLD_SIZE", "0") not in {"", "0", "1"}:
            raise RuntimeError(
                "layer propagation diagnostic currently requires one GPU"
            )
        if not kv_cache or not isinstance(kv_cache[0], dict):
            return None
        first = kv_cache[0]
        cache_tokens = int(first.get("end_index", 0) or 0)
        minimum = int(
            os.getenv(
                "HYWORLDPLAY_LAYER_PROPAGATION_MIN_CACHE_TOKENS", "31680"
            )
        )
        if cache_tokens < minimum:
            return None
        call_index = int(first.get("_layer_propagation_diag_calls", 0))
        first["_layer_propagation_diag_calls"] = call_index + 1
        maximum = int(
            os.getenv("HYWORLDPLAY_LAYER_PROPAGATION_MAX_CALLS", "1")
        )
        if call_index >= maximum:
            return None
        return first, call_index

    @torch.no_grad()
    def _run_layer_propagation_probes(
        self,
        *,
        block_inputs: list[torch.Tensor],
        baseline_hidden: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        kv_cache: list[dict],
        viewmats: Optional[torch.Tensor],
        Ks: Optional[torch.Tensor],
        cache_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]],
        cache_viewmats: Optional[torch.Tensor],
        cache_Ks: Optional[torch.Tensor],
        context_frames_list: Optional[List[int]],
        current_start: Optional[int],
        cache_tokens: int,
        call_index: int,
    ) -> None:
        """Measure finite-difference error propagation from every layer.

        A deterministic Rademacher perturbation is inserted into the real
        self-attention output, before AdaLN gating, cross-attention and FFN.
        The perturbed branch is then replayed through every downstream block
        and the real output projection.  This measures layer order/propagation,
        not codec quality; codec-specific K/V/head marginals are multiplied by
        these coefficients offline.
        """

        if len(block_inputs) != len(self.blocks):
            raise ValueError("layer propagation block input capture is incomplete")
        raw_layers = os.getenv("HYWORLDPLAY_LAYER_PROPAGATION_LAYERS", "")
        layers = (
            tuple(int(item) for item in raw_layers.split(",") if item)
            if raw_layers
            else tuple(range(len(self.blocks)))
        )
        if (
            not layers
            or len(set(layers)) != len(layers)
            or any(not 0 <= item < len(self.blocks) for item in layers)
        ):
            raise ValueError("invalid layer propagation layer list")
        epsilon = float(
            os.getenv("HYWORLDPLAY_LAYER_PROPAGATION_EPSILON", "0.001")
        )
        seed = int(os.getenv("HYWORLDPLAY_LAYER_PROPAGATION_SEED", "20260725"))
        if not 0 < epsilon <= 0.1:
            raise ValueError("layer propagation epsilon must lie in (0,0.1]")
        root = os.getenv("HYWORLDPLAY_LAYER_PROPAGATION_DUMP_DIR")
        if not root:
            raise ValueError("layer propagation dump directory is required")
        started = time.perf_counter()
        baseline_projected = self._project_propagation_tokens(
            baseline_hidden, temb
        )
        operating_points_path = os.getenv(
            "HYWORLDPLAY_LAYER_PROPAGATION_OPERATING_POINTS", ""
        )
        operating_points_by_layer: dict[int, list[dict[str, Any]]] = {}
        if operating_points_path:
            with open(operating_points_path, encoding="utf-8") as handle:
                operating_payload = json.load(handle)
            if operating_payload.get("schema_version") != (
                "uniform_grid_gaussian.propagation_operating_points.v1"
            ):
                raise ValueError(
                    "unexpected layer propagation operating-point schema"
                )
            for point in operating_payload.get("points", []):
                operating_points_by_layer.setdefault(
                    int(point["layer"]), []
                ).append(dict(point))
            expected_points = {
                (layer, tensor, depth)
                for layer in range(len(self.blocks))
                for tensor in ("K", "V")
                for depth in (3, 4, 5)
            }
            observed_points = {
                (
                    layer,
                    str(point["tensor"]),
                    int(point["depth"]),
                )
                for layer, points in operating_points_by_layer.items()
                for point in points
            }
            if observed_points != expected_points:
                raise ValueError(
                    "layer propagation operating points are incomplete"
                )
        records: list[dict[str, Any]] = []
        for layer in layers:
            source = block_inputs[layer]

            def replay_suffix(
                perturbation: Optional[torch.Tensor],
            ) -> torch.Tensor:
                shadow, _unused = self.blocks[layer](
                    source,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    kv_cache[layer],
                    False,
                    layer,
                    viewmats,
                    Ks,
                    cache_rotary_emb,
                    cache_viewmats,
                    cache_Ks,
                    context_frames_list=context_frames_list,
                    current_token_start=current_start,
                    attention_output_perturbation=perturbation,
                )
                for downstream in range(layer + 1, len(self.blocks)):
                    shadow, _unused = self.blocks[downstream](
                        shadow,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
                        kv_cache[downstream],
                        False,
                        downstream,
                        viewmats,
                        Ks,
                        cache_rotary_emb,
                        cache_viewmats,
                        cache_Ks,
                        context_frames_list=context_frames_list,
                        current_token_start=current_start,
                    )
                return shadow

            control_shadow = replay_suffix(None)
            control_projected = self._project_propagation_tokens(
                control_shadow, temb
            )
            control_terminal = self._propagation_error_stats(
                control_projected - baseline_projected
            )
            del control_shadow, control_projected
            if operating_points_by_layer:
                point_records: list[dict[str, Any]] = []
                for tensor_name in ("K", "V"):
                    generator = torch.Generator(
                        device=source.device
                    ).manual_seed(
                        seed
                        + call_index * 1_000_003
                        + layer * 1009
                        + (0 if tensor_name == "K" else 53)
                    )
                    signs = torch.randint(
                        0,
                        2,
                        source.shape,
                        generator=generator,
                        device=source.device,
                        dtype=torch.int8,
                    )
                    tensor_points = sorted(
                        (
                            point
                            for point in operating_points_by_layer[layer]
                            if point["tensor"] == tensor_name
                        ),
                        key=lambda point: int(point["depth"]),
                    )
                    for point in tensor_points:
                        target = float(point["target_local_mean"])
                        if not math.isfinite(target) or target <= 0:
                            raise ValueError(
                                "propagation target local MSE must be positive"
                            )
                        perturbation = (
                            signs.to(source.dtype).mul(2).sub(1)
                            * torch.tensor(
                                math.sqrt(target),
                                device=source.device,
                                dtype=source.dtype,
                            )
                        )
                        local_stats = self._propagation_error_stats(
                            perturbation
                        )
                        shadow = replay_suffix(perturbation)
                        terminal_hidden = self._propagation_error_stats(
                            shadow - baseline_hidden
                        )
                        shadow_projected = (
                            self._project_propagation_tokens(shadow, temb)
                        )
                        terminal_projected = self._propagation_error_stats(
                            shadow_projected - baseline_projected
                        )
                        signal_mean = max(
                            terminal_projected["mean"]
                            - control_terminal["mean"],
                            0.0,
                        )
                        signal_cvar99 = max(
                            terminal_projected["cvar99"]
                            - control_terminal["cvar99"],
                            0.0,
                        )
                        point_records.append(
                            {
                                "tensor": tensor_name,
                                "depth": int(point["depth"]),
                                "target_local_mean": target,
                                "target_local_cvar99": float(
                                    point["target_local_cvar99"]
                                ),
                                "realized_local_attention_output": local_stats,
                                "terminal_hidden": terminal_hidden,
                                "terminal_projected": terminal_projected,
                                "terminal_projected_noise_floor_subtracted": {
                                    "mean": signal_mean,
                                    "cvar99": signal_cvar99,
                                },
                            }
                        )
                        del (
                            shadow,
                            shadow_projected,
                            perturbation,
                        )
                    del signs
                records.append(
                    {
                        "layer": int(layer),
                        "zero_perturbation_terminal_projected": (
                            control_terminal
                        ),
                        "operating_points": point_records,
                    }
                )
                continue
            generator = torch.Generator(device=source.device).manual_seed(
                seed + call_index * 1_000_003 + layer * 1009
            )
            signs = torch.randint(
                0,
                2,
                source.shape,
                generator=generator,
                device=source.device,
                dtype=torch.int8,
            )
            source_rms = source.float().square().mean().sqrt()
            perturbation = (
                signs.to(source.dtype).mul_(2).sub_(1)
                * (source_rms * epsilon).to(source.dtype)
            )
            local_stats = self._propagation_error_stats(perturbation)
            shadow = replay_suffix(perturbation)
            terminal_hidden = self._propagation_error_stats(
                shadow - baseline_hidden
            )
            shadow_projected = self._project_propagation_tokens(shadow, temb)
            terminal_projected = self._propagation_error_stats(
                shadow_projected - baseline_projected
            )
            signal_mean = max(
                terminal_projected["mean"] - control_terminal["mean"], 0.0
            )
            signal_cvar99 = max(
                terminal_projected["cvar99"] - control_terminal["cvar99"],
                0.0,
            )
            records.append(
                {
                    "layer": int(layer),
                    "local_attention_output": local_stats,
                    "zero_perturbation_terminal_projected": control_terminal,
                    "terminal_hidden": terminal_hidden,
                    "terminal_projected": terminal_projected,
                    "terminal_projected_noise_floor_subtracted": {
                        "mean": signal_mean,
                        "cvar99": signal_cvar99,
                    },
                    "amplification_mean": (
                        signal_mean
                        / max(local_stats["mean"], 1e-30)
                    ),
                    "amplification_cvar99": (
                        signal_cvar99
                        / max(local_stats["cvar99"], 1e-30)
                    ),
                    "control_fraction_mean": (
                        control_terminal["mean"]
                        / max(terminal_projected["mean"], 1e-30)
                    ),
                    "control_fraction_cvar99": (
                        control_terminal["cvar99"]
                        / max(terminal_projected["cvar99"], 1e-30)
                    ),
                }
            )
            del shadow, shadow_projected, perturbation, signs
        rank = int(os.getenv("RANK", "0"))
        destination = os.path.join(
            root, f"rank_{rank:03d}", f"call_{call_index:05d}.json"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.exists(destination):
            raise FileExistsError(
                f"refusing to overwrite layer propagation probe: {destination}"
            )
        payload = {
            "schema_version": "wan5b.layer_propagation_probe.v1",
            "prompt_id": os.getenv("HYWORLDPLAY_LAYER_PROPAGATION_PROMPT_ID"),
            "call_index": int(call_index),
            "cache_tokens": int(cache_tokens),
            "current_token_start": (
                int(current_start) if current_start is not None else None
            ),
            "epsilon_relative_to_hidden_rms": epsilon,
            "operating_points_path": (
                operating_points_path or None
            ),
            "probe_mode": (
                "codec_D3_D4_D5_operating_points"
                if operating_points_path
                else "relative_hidden_rms_epsilon"
            ),
            "seed": seed,
            "probe": (
                "deterministic_rademacher_at_self_attention_output_"
                "before_gate_cross_attention_ffn"
            ),
            "terminal": "real_output_adaln_and_projection_before_unpatchify",
            "records": records,
            "elapsed_seconds": time.perf_counter() - started,
            "timestamp": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }
        with open(destination, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        current_start=None,
        current_end=None,
        kv_cache: dict = None,
        is_cache: bool = False,
        viewmats: Optional[torch.Tensor] = None,
        Ks: Optional[torch.Tensor] = None,
        cache_viewmats: Optional[torch.Tensor] = None,
        cache_Ks: Optional[torch.Tensor] = None,
        action: Optional[torch.Tensor] = None,
        window_frames: int = None,
        context_frames_list: List[int] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_states = hidden_states.to(encoder_hidden_states.dtype)
        viewmats = (
            viewmats.to(encoder_hidden_states.dtype) if viewmats is not None else None
        )
        Ks = Ks.to(encoder_hidden_states.dtype) if Ks is not None else None
        cache_viewmats = (
            cache_viewmats.to(encoder_hidden_states.dtype)
            if cache_viewmats is not None
            else None
        )
        cache_Ks = cache_Ks.to(encoder_hidden_states.dtype) if cache_Ks is not None else None
        action = action.to(encoder_hidden_states.dtype) if action is not None else None
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w


        ## if hidden_states is smaller than the window size, we need to pad it
        if post_patch_num_frames < window_frames:
            padding = window_frames - post_patch_num_frames
            # pad hidden_state to the window size in the dim=2: [B, C, T, H, W] -> [B, C, T + padding, H, W]
            padding_hidden_states = F.pad(
                hidden_states, (0, 0, 0, 0, 0, padding), value=0.0
            )  # [B, C, T + padding, H, W]
        elif post_patch_num_frames == window_frames:
            padding_hidden_states = hidden_states.clone()
        else:
            # if hidden_states is larger than the window size, we need to slice it
            padding_hidden_states = hidden_states[:, :, :window_frames, :, :]

        # 得到所有的rope参数
        rotary_emb = self.rope(padding_hidden_states)  # 2 * [1, ,1 , T' * H' * W']([1, 1, 21 * 30 * 52])
        full_rotary_emb = rotary_emb
        context_rotary_emb = _gather_rotary_emb_by_frame_indices(
            full_rotary_emb,
            context_frames_list,
        )

        hidden_states = self.patch_embedding(hidden_states)  # [B, C, T, H, W] -> [B, C', T', H', W']
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                action, timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))  # [B, 9216] -> [B, 6, 1536]

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        cache_rotary_emb = None
        if kv_cache is not None and not is_cache and kv_cache:
            cache_mode = kv_cache[0].get("cache_mode", "final")
            cache_end = kv_cache[0].get("end_index", 0)
            if cache_mode == "pre_kv_reconstruct" and cache_end > 0:
                if context_rotary_emb is not None:
                    cache_rotary_emb = context_rotary_emb
                else:
                    cache_rotary_emb0 = full_rotary_emb[0][:, :, :cache_end, :]
                    cache_rotary_emb1 = full_rotary_emb[1][:, :, :cache_end, :]
                    cache_rotary_emb = (cache_rotary_emb0, cache_rotary_emb1)

        if is_cache and context_rotary_emb is not None:
            rotary_emb0, rotary_emb1 = context_rotary_emb
        else:
            rotary_emb0 = full_rotary_emb[0][:, :, current_start:current_end, :]
            rotary_emb1 = full_rotary_emb[1][:, :, current_start:current_end, :]

        # SP
        if os.getenv("WORLD_SIZE", "0") != "0":
            world_size = get_sp_world_size()
            rank = get_sp_parallel_rank()
            hidden_states = hidden_states.chunk(chunks=world_size, dim=1)[rank]
            rotary_emb0 = rotary_emb0.chunk(chunks=world_size, dim=2)[rank]
            rotary_emb1 = rotary_emb1.chunk(chunks=world_size, dim=2)[rank]
            if cache_rotary_emb is not None:
                cache_rotary_emb = (
                    cache_rotary_emb[0].chunk(chunks=world_size, dim=2)[rank],
                    cache_rotary_emb[1].chunk(chunks=world_size, dim=2)[rank],
                )
        rotary_emb = (rotary_emb0, rotary_emb1)

        if is_cache:
            _kv_cache_now = [None] * 30

        # 4. Transformer blocks
        propagation_reservation = self._reserve_layer_propagation_probe(
            kv_cache, is_cache=is_cache
        )
        propagation_inputs: list[torch.Tensor] = []
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            if propagation_reservation is not None:
                raise RuntimeError(
                    "layer propagation diagnostic is incompatible with "
                    "gradient checkpointing"
                )
            for idx, block in enumerate(self.blocks):
                hidden_states, t_kv = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    kv_cache[idx] if kv_cache is not None else None,
                    is_cache,
                    idx,
                    viewmats,
                    Ks,
                    cache_rotary_emb,
                    cache_viewmats,
                    cache_Ks,
                    context_frames_list,
                    current_start,
                )
                if is_cache:
                    _kv_cache_now[idx] = t_kv
        else:
            for idx, block in enumerate(self.blocks):
                if propagation_reservation is not None:
                    propagation_inputs.append(hidden_states.detach())
                hidden_states, t_kv = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    kv_cache[idx] if kv_cache is not None else None,
                    is_cache,
                    idx,
                    viewmats,
                    Ks,
                    cache_rotary_emb,
                    cache_viewmats,
                    cache_Ks,
                    context_frames_list=context_frames_list,
                    current_token_start=current_start,
                )
                if is_cache:
                    _kv_cache_now[idx] = t_kv

        if is_cache:
            return _kv_cache_now

        if propagation_reservation is not None:
            first_cache, propagation_call_index = propagation_reservation
            self._run_layer_propagation_probes(
                block_inputs=propagation_inputs,
                baseline_hidden=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep_proj=timestep_proj,
                temb=temb,
                rotary_emb=rotary_emb,
                kv_cache=kv_cache,
                viewmats=viewmats,
                Ks=Ks,
                cache_rotary_emb=cache_rotary_emb,
                cache_viewmats=cache_viewmats,
                cache_Ks=cache_Ks,
                context_frames_list=context_frames_list,
                current_start=current_start,
                cache_tokens=int(first_cache.get("end_index", 0) or 0),
                call_index=propagation_call_index,
            )

        if os.getenv("WORLD_SIZE", "0") != "0":
            hidden_states = sequence_model_parallel_all_gather(
                hidden_states.contiguous(), dim=1
            )

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if shift.shape[0] != hidden_states.shape[0]:
            # Diffusion forcing training, adjust the AdaLN operation
            latent_length = temb.shape[0] // hidden_states.shape[0]
            token_length = hidden_states.shape[1] // latent_length
            shift = shift.reshape(hidden_states.shape[0], -1, hidden_states.shape[2])
            scale = scale.reshape(hidden_states.shape[0], -1, hidden_states.shape[2])
            # operate on the hidden_states
            hidden_states = (
                self.norm_out(hidden_states.float())
                * (1 + scale.repeat_interleave(token_length, dim=1))
                + shift.repeat_interleave(token_length, dim=1)
            ).type_as(hidden_states)
        else:
            hidden_states = (
                self.norm_out(hidden_states.float()) * (1 + scale) + shift
            ).type_as(hidden_states)

        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def add_discrete_action_parameters(self):
        self.condition_embedder.action_embedder = TimestepEmbedding(
            in_channels=self.condition_embedder.time_freq_dim,
            time_embed_dim=self.condition_embedder.dim,
        )
        nn.init.zeros_(self.condition_embedder.action_embedder.linear_2.weight)

        if self.condition_embedder.action_embedder.linear_2.bias is not None:
            nn.init.zeros_(self.condition_embedder.action_embedder.linear_2.bias)

        # prope
        for block in self.blocks:
            block.attn1.to_out_prope = torch.nn.ModuleList(
                [
                    torch.nn.Linear(block.attn1.inner_dim, self.inner_dim, bias=True),
                ]
            )
            nn.init.zeros_(block.attn1.to_out_prope[0].weight)

            if block.attn1.to_out_prope[0].bias is not None:
                nn.init.zeros_(block.attn1.to_out_prope[0].bias)
