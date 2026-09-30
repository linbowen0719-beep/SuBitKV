import html

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, Union

import os
import time
import regex as re
from termcolor import colored, cprint

import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from types import SimpleNamespace

from models.utils import (
    shard_latents_dim_across_sp,
    select_mem_frames_wan,
)
from subitkv.compress import get_quantize_fn, compress_kv_cache
from subitkv.sim.quant.quantize_config import (
    parse_residual_codec_span_schedule,
)
from subitkv.uncompress import (
    get_dequant_stats,
    reset_dequant_stats,
    uncompress_kv_cache,
    uncompress_single_kv_cache,
)
from subitkv.kv_cache import ChunkedKVCache, offload_kv_cache_layer, onload_kv_cache_layer
from subitkv.temporal_tiered_rvq import (
    AGE_RAMP16_TIERED_POLICY,
    H7_TIERED_POLICY,
    apply_tiered_cache_updates,
    normalize_tiered_depth_policy,
    plan_tiered_cache_updates,
    temporal_tiered_depth_schedule,
    tiered_depth_policy_spec,
    tiered_index_storage_audit,
)
from subitkv.temporal_structured_residual import (
    FORMAL_PENDING_BF16_BYTES,
    FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES,
    FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES,
    STRUCTURED_TEMPORAL_KV_POLICY_31114,
    STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
    STRUCTURED_TEMPORAL_KV_POLICY_41113,
    STRUCTURED_TEMPORAL_MAX_SPANS,
    apply_structured_temporal_cache_updates,
    normalize_structured_temporal_policy,
    packed_tensor_leaf_nbytes,
    plan_structured_temporal_kv_updates,
    structured_temporal_depth_schedule,
    structured_temporal_formal_payload_timeline,
    structured_temporal_storage_audit,
)
from subitkv.temporal_adaptive_residual import (
    QUANT_METHOD as TEMPORAL_ADAPTIVE_RESIDUAL_METHOD,
    load_profile as load_temporal_adaptive_profile,
    payload_nbytes as temporal_payload_nbytes,
    rvq_identity_hash,
    rvq_identity_is_valid,
    schedule_precisions as schedule_temporal_precisions,
    transition_state as transition_temporal_state,
    transition_steps as temporal_transition_steps,
    validate_state as validate_temporal_state,
)
from hyvideo.utils.retrieval_context import generate_points_in_sphere
from distributed.parallel_state import (
    get_sp_parallel_rank,
    get_sp_parallel_local_rank,
    get_sp_world_size,
)
from distributed.communication_op import sequence_model_parallel_all_gather
from models.par_vae.tools import distribute_data_to_gpus, gather_results
from inference.helper import CHUNK_SIZE

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS = 11
_STRUCTURED_TEMPORAL_PARAMS_SHA256 = (
    "a8d85f8848e182516da51aa49e7883d576a248f1b3a91facbe566e22861eca96"
)

if is_ftfy_available():
    import ftfy


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: Optional[WanTransformer3DModel] = None,
        transformer_2: Optional[WanTransformer3DModel] = None,
        boundary_ratio: Optional[float] = None,
        expand_timesteps: bool = False,  # Wan2.2 ti2v
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = (
            self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        )
        self.vae_scale_factor_spatial = (
            self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        )
        self.video_processor = VideoProcessor(
            vae_scale_factor=self.vae_scale_factor_spatial
        )
        if os.getenv("WORLD_SIZE", "0") != "0":
            self.parallel_run = True
        else:
            self.parallel_run = False
        self.par_vae_encode = False
        self.par_vae_decode = self.parallel_run and True

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_videos_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        offload_text_encoder: bool = False,
    ):
        device = device or self._execution_device

        if offload_text_encoder and self.text_encoder is not None:
            self.text_encoder.to(device)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if offload_text_encoder and self.text_encoder is not None:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 16 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str)
            and not isinstance(negative_prompt, list)
        ):
            raise ValueError(
                f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}"
            )

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError(
                "`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None."
            )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def init_kv_cache(self, batch_size, max_frames, dtype, device):
        TOKENS_PER_FRAME = 880
        num_heads = self.transformer.blocks[0].attn1.heads
        base_head_dim = self.transformer.blocks[0].attn1.inner_dim // num_heads
        self._kv_cache_mode = os.getenv("HYWORLDPLAY_KV_CACHE_MODE", "final").lower()
        if self._kv_cache_mode == "pre_kv_reconstruct":
            head_dim = base_head_dim
        else:
            self._kv_cache_mode = "final"
            head_dim = base_head_dim * 2  # rope + prope
        cprint(
            f"Initializing HY-WorldPlay KV cache mode={self._kv_cache_mode}, "
            f"head_dim={head_dim}",
            "light_cyan",
        )

        self._kv_cache = []
        self._kv_cache_neg = []
        transformer_num_layers = len(self.transformer.blocks)
        for _ in range(transformer_num_layers):
            self._kv_cache.append({
                "k": ChunkedKVCache(batch_size, TOKENS_PER_FRAME, num_heads, head_dim, max_frames, dtype, device, layout="BHSD"),
                "v": ChunkedKVCache(batch_size, TOKENS_PER_FRAME, num_heads, head_dim, max_frames, dtype, device, layout="BHSD"),
                "end_index": 0,
                "cache_mode": self._kv_cache_mode,
            })
            self._kv_cache_neg.append({"k": None, "v": None})

        self.use_kv_cache = True

    @staticmethod
    def _env_flag(name: str, default: str = "1") -> bool:
        return os.getenv(name, default).lower() not in {"0", "false", "no", "off", ""}

    @staticmethod
    def _parse_prewarm_layers(raw: str, num_layers: int, rvq_enabled: bool) -> list[int]:
        raw = str(raw or "").strip().lower()
        if not raw:
            return list(range(num_layers)) if rvq_enabled else [0]
        if raw in {"all", "*"}:
            return list(range(num_layers))
        if raw in {"none", "off"}:
            return []
        layers = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            idx = int(part)
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"HYWORLDPLAY_KV_QUANT_PREWARM_LAYERS index {idx} out of range 0:{num_layers}")
            layers.append(idx)
        return sorted(set(layers))

    def _make_kv_prewarm_tensor(
        self,
        *,
        batch_size: int,
        heads: int,
        tokens: int,
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
        offset: int,
    ) -> torch.Tensor:
        values = torch.arange(tokens * dim, device=device, dtype=torch.float32).reshape(1, 1, tokens, dim)
        values = ((values + float(offset)) % 257.0 - 128.0) / 64.0
        return values.expand(batch_size, heads, tokens, dim).to(dtype=dtype).contiguous()

    def _prewarm_kv_quantization(self) -> None:
        quant_config = getattr(self, "_quant_config", None)
        if quant_config is None or getattr(quant_config, "quant_type", "none") == "none":
            return
        if not self._env_flag("HYWORLDPLAY_KV_QUANT_PREWARM", "1"):
            return

        rvq_feedback_method = str(getattr(quant_config, "rvq_feedback_method", "none")).lower()
        if rvq_feedback_method not in {"", "none", "off", "false", "0"}:
            cprint(
                "Skipping KV quant prewarm because RVQ feedback stateful assignment is enabled",
                "yellow",
            )
            return

        if not getattr(self, "_kv_cache", None):
            return
        first_cache: ChunkedKVCache = self._kv_cache[0]["k"]
        tokens = int(os.getenv("HYWORLDPLAY_KV_QUANT_PREWARM_TOKENS", str(CHUNK_SIZE * 880)))
        tokens = max(first_cache.frame_seq_length, tokens)
        if tokens % first_cache.frame_seq_length != 0:
            tokens = ((tokens + first_cache.frame_seq_length - 1) // first_cache.frame_seq_length) * first_cache.frame_seq_length

        k_rvq = bool(getattr(quant_config, "k_rvq", False))
        v_rvq = bool(getattr(quant_config, "v_rvq", False))
        layers = self._parse_prewarm_layers(
            os.getenv("HYWORLDPLAY_KV_QUANT_PREWARM_LAYERS", ""),
            len(self._kv_cache),
            k_rvq or v_rvq,
        )
        if not layers:
            return

        cprint(
            f"Prewarming KV quantization for layers {layers[:4]}"
            f"{'...' if len(layers) > 4 else ''}, tokens={tokens}",
            "light_cyan",
        )
        temporal_prewarm_bits: tuple[int, ...] = ()
        if bool(getattr(quant_config, "temporal_adaptive_residual", False)):
            profile = self._temporal_adaptive_profile()
            capacity = int(
                getattr(
                    quant_config,
                    "temporal_adaptive_capacity_bit_chunks",
                    40,
                )
            )
            allocation_mode = str(
                getattr(
                    quant_config,
                    "temporal_adaptive_allocation_mode",
                    "optimal",
                )
                or "optimal"
            )
            current: list[int] = []
            active_bits: set[int] = set()
            for _ in range(10):
                current = list(
                    schedule_temporal_precisions(
                        [*current, 16],
                        profile,
                        capacity_bit_chunks=capacity,
                        allocation_mode=allocation_mode,
                    )
                )
                active_bits.update(int(bits) for bits in current if int(bits) < 16)
            # Descending order lets each later state be obtained by a cheap
            # prefix truncation from the already materialized higher-depth
            # state.  This compiles one restore variant per actually used bit
            # depth, outside the reported encode/decode intervals.
            temporal_prewarm_bits = tuple(sorted(active_bits, reverse=True))
            cprint(
                f"Temporal restore prewarm bits: {temporal_prewarm_bits}",
                "light_cyan",
            )
        start_time = time.perf_counter()
        quantize_fn = get_quantize_fn(quant_config.quant_type, quant_config)
        span_schedule = parse_residual_codec_span_schedule(
            getattr(
                quant_config,
                "rvq_residual_codec_span_schedule",
                "",
            )
        )
        k = self._make_kv_prewarm_tensor(
            batch_size=first_cache.batch_size,
            heads=first_cache.num_heads,
            tokens=tokens,
            dim=first_cache.head_dim,
            dtype=first_cache.dtype,
            device=first_cache.device,
            offset=0,
        )
        v = self._make_kv_prewarm_tensor(
            batch_size=first_cache.batch_size,
            heads=first_cache.num_heads,
            tokens=tokens,
            dim=first_cache.head_dim,
            dtype=first_cache.dtype,
            device=first_cache.device,
            offset=17,
        )
        try:
            with torch.no_grad():
                for layer_idx in layers:
                    k_quant, v_quant = compress_kv_cache(
                        k,
                        v,
                        quant_config.quant_type,
                        quant_config,
                        quantize_fn,
                        layer_idx=int(layer_idx),
                        span_idx=0 if span_schedule else None,
                        rvq_feedback_state=None,
                        rvq_feedback_query=None,
                    )
                    if isinstance(k_quant, dict):
                        k_quant = self._pack_info_into_single_kv_cache(k_quant, k.dtype)
                        if temporal_prewarm_bits:
                            current_k = k_quant
                            for bits in temporal_prewarm_bits:
                                current_k = transition_temporal_state(
                                    current_k,
                                    bits,
                                    quant_config=quant_config,
                                )
                                _ = uncompress_single_kv_cache(current_k)
                        else:
                            _ = uncompress_single_kv_cache(k_quant)
                    if isinstance(v_quant, dict):
                        v_quant = self._pack_info_into_single_kv_cache(v_quant, v.dtype)
                        if temporal_prewarm_bits:
                            current_v = v_quant
                            for bits in temporal_prewarm_bits:
                                current_v = transition_temporal_state(
                                    current_v,
                                    bits,
                                    quant_config=quant_config,
                                )
                                _ = uncompress_single_kv_cache(current_v)
                        else:
                            _ = uncompress_single_kv_cache(v_quant)
                    del k_quant, v_quant
            if first_cache.device.type == "cuda":
                torch.cuda.synchronize(first_cache.device)
        finally:
            del k, v
            reset_dequant_stats()
            if first_cache.device.type == "cuda":
                torch.cuda.empty_cache()
        cprint(f"KV quantization prewarm done in {time.perf_counter() - start_time:.2f} s", "light_cyan")

    # ------------------------------------------------------------------
    # KV cache quantization (mirrors CausalInferencePipeline)
    # ------------------------------------------------------------------

    def _configure_rvq_feedback_query_capture(self):
        quant_config = getattr(self, "_quant_config", None)
        method = str(getattr(quant_config, "rvq_feedback_method", "none")).lower()
        k_calibration = str(getattr(quant_config, "rvq_k_calibration", "none")).lower()
        v_calibration = str(getattr(quant_config, "rvq_v_calibration", "none")).lower()
        diagnostics = bool(getattr(quant_config, "rvq_diagnostics", False))
        sample_size = 0
        if diagnostics or method == "attn_output" or "online" in {k_calibration, v_calibration}:
            sample_size = int(getattr(quant_config, "rvq_feedback_query_sample", 96))

        for layer in getattr(self, "_kv_cache", []) or []:
            if sample_size > 0:
                layer["_rvq_feedback_query_sample_size"] = sample_size
                layer.setdefault("_rvq_feedback_query_seed", 42)
            else:
                layer.pop("_rvq_feedback_query_sample_size", None)
                layer.pop("_rvq_feedback_query_sample", None)
                layer.pop("_rvq_feedback_query_indices", None)

    def quantize_kv_cache(self, tokens_to_quantize_start: int, tokens_to_quantize_end: int):
        """Quantize a token range [start, end) of the KV cache via ChunkedKVCache.

        Indices are in token space (frame-aligned). Only the specified range
        is compressed; the rest of the cache stays at full precision.
        """
        quant_config = self._quant_config
        if quant_config is None or quant_config.quant_type == "none":
            self._print_memory_usage()
            return

        cprint(
            f"Quantizing {tokens_to_quantize_end - tokens_to_quantize_start} tokens, "
            f"from {tokens_to_quantize_start} to {tokens_to_quantize_end}",
            "light_cyan",
        )
        cprint(f"Quantizing KV cache with config: {quant_config}", "light_cyan")

        span_schedule = parse_residual_codec_span_schedule(
            getattr(
                quant_config,
                "rvq_residual_codec_span_schedule",
                "",
            )
        )
        span_idx = None
        if span_schedule:
            span_tokens = (
                int(tokens_to_quantize_end) - int(tokens_to_quantize_start)
            )
            if span_tokens <= 0:
                raise ValueError(
                    "residual-codec span routing requires a positive token range"
                )
            if int(tokens_to_quantize_start) % span_tokens:
                raise ValueError(
                    "residual-codec span routing requires equal, origin-aligned "
                    f"physical spans; start={tokens_to_quantize_start}, "
                    f"span_tokens={span_tokens}"
                )
            span_idx = int(tokens_to_quantize_start) // span_tokens
            if span_idx >= len(span_schedule):
                raise ValueError(
                    f"physical span {span_idx} exceeds configured residual-codec "
                    f"schedule length {len(span_schedule)}"
                )
            cprint(
                "Residual-codec physical span "
                f"{span_idx}: {span_schedule[span_idx]}",
                "light_cyan",
            )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        with torch.no_grad():
            quantize_fn = get_quantize_fn(quant_config.quant_type, quant_config)
            rvq_feedback_state = getattr(self, "_rvq_feedback_state", None)

            for layer_idx, layer in enumerate(self._kv_cache):
                k_cache: ChunkedKVCache = layer["k"]
                v_cache: ChunkedKVCache = layer["v"]

                k = k_cache.read(tokens_to_quantize_start, tokens_to_quantize_end)  # BHSD
                v = v_cache.read(tokens_to_quantize_start, tokens_to_quantize_end)

                if layer_idx == 0:
                    cprint(f"K shape: {k.shape} | V shape: {v.shape}", "magenta")

                k_quant, v_quant = compress_kv_cache(
                    k,
                    v,
                    quant_config.quant_type,
                    quant_config,
                    quantize_fn,
                    layer_idx=layer_idx,
                    span_idx=span_idx,
                    rvq_feedback_state=rvq_feedback_state,
                    rvq_feedback_query=layer.get("_rvq_feedback_query_sample"),
                )

                self._print_kv_cache_mse_error(k, k_quant, v, v_quant, layer_idx)

                if isinstance(k_quant, dict):
                    k_quant = self._pack_info_into_single_kv_cache(k_quant, k.dtype)
                if isinstance(v_quant, dict):
                    v_quant = self._pack_info_into_single_kv_cache(v_quant, v.dtype)
                if isinstance(k_quant, dict) or isinstance(v_quant, dict):
                    self._save_quantized_kv_cache_layer(
                        layer_idx,
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                        k_quant,
                        v_quant,
                    )

                # store_quantized handles both tensor (fake) and dict (real)
                k_cache.store_quantized(tokens_to_quantize_start, tokens_to_quantize_end, k_quant)
                v_cache.store_quantized(tokens_to_quantize_start, tokens_to_quantize_end, v_quant)

        end_event.record()
        torch.cuda.synchronize()
        duration = start_event.elapsed_time(end_event)
        cprint(f"KV Cache Quantization Time: {(duration / 1000):.2f} s", "light_cyan")

        if not bool(
            getattr(quant_config, "suppress_quantize_memory_report", False)
        ):
            self._print_memory_usage()

    @staticmethod
    def _policy_has(policy: str, name: str) -> bool:
        policy = str(policy or "none").lower()
        return policy == name or policy == "first_last"

    @staticmethod
    def _last_base_quant_trigger_chunk(num_chunks: int, quant_factor: int) -> Optional[int]:
        max_chunk_i = max(0, int(num_chunks) - 1)
        trigger = max_chunk_i - (max_chunk_i % int(quant_factor))
        if trigger < int(quant_factor):
            return None
        return trigger

    def _static_skip_frame_ranges(
        self,
        n_latent: int,
        chunk_size: int,
        quant_factor: int,
        num_chunks: int,
    ) -> list[tuple[int, int]]:
        chunk_policy = str(self.ctx.get("kv_quant_skip_chunk_policy", "none")).lower()
        frame_policy = str(self.ctx.get("kv_quant_skip_frame_policy", "none")).lower()
        skip_first_frames = max(0, int(self.ctx.get("kv_quant_skip_first_frames", 0)))
        skip_last_frames = max(0, int(self.ctx.get("kv_quant_skip_last_frames", 0)))
        anchor_stride = max(0, int(self.ctx.get("kv_quant_anchor_frame_stride", 0)))
        anchor_offset = max(0, int(self.ctx.get("kv_quant_anchor_frame_offset", 0)))
        ranges: list[tuple[int, int]] = []

        last_trigger = self._last_base_quant_trigger_chunk(num_chunks, quant_factor)
        last_quantizable_chunk = None if last_trigger is None else max(0, last_trigger - 1)
        last_quantizable_frame = None if last_trigger is None else max(0, last_trigger * chunk_size - 1)

        if self._policy_has(chunk_policy, "first"):
            ranges.append((0, min(chunk_size, n_latent)))
        if self._policy_has(chunk_policy, "last") and last_quantizable_chunk is not None:
            start = last_quantizable_chunk * chunk_size
            ranges.append((start, min(start + chunk_size, n_latent)))
        if chunk_policy == "last_two" and last_quantizable_chunk is not None:
            start_chunk = max(0, last_quantizable_chunk - 1)
            start = start_chunk * chunk_size
            ranges.append((start, min(start + 2 * chunk_size, n_latent)))

        if self._policy_has(frame_policy, "first"):
            ranges.append((0, min(1, n_latent)))
        if self._policy_has(frame_policy, "last") and last_quantizable_frame is not None:
            ranges.append((last_quantizable_frame, min(last_quantizable_frame + 1, n_latent)))
        if skip_first_frames > 0:
            ranges.append((0, min(skip_first_frames, n_latent)))
        if skip_last_frames > 0 and last_quantizable_frame is not None:
            start = max(0, last_quantizable_frame - skip_last_frames + 1)
            ranges.append((start, min(last_quantizable_frame + 1, n_latent)))
        if anchor_stride > 0:
            offset = anchor_offset % anchor_stride
            for frame in range(offset, n_latent, anchor_stride):
                ranges.append((frame, min(frame + 1, n_latent)))

        return [(max(0, s), min(n_latent, e)) for s, e in ranges if e > s]

    @staticmethod
    def _subtract_frame_ranges(
        frame_start: int,
        frame_end: int,
        skip_ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        cursor = int(frame_start)
        for raw_start, raw_end in sorted(skip_ranges):
            start = max(int(frame_start), int(raw_start))
            end = min(int(frame_end), int(raw_end))
            if end <= cursor or start >= frame_end:
                continue
            if cursor < start:
                segments.append((cursor, start))
            cursor = max(cursor, end)
        if cursor < frame_end:
            segments.append((cursor, int(frame_end)))
        return segments

    def quantize_kv_cache_frame_range_with_policy(
        self,
        frame_start: int,
        frame_end: int,
        *,
        n_latent: int,
        chunk_size: int,
        quant_factor: int,
        num_chunks: int,
        tokens_per_frame: int,
    ) -> None:
        skip_ranges = self._static_skip_frame_ranges(
            n_latent,
            chunk_size,
            quant_factor,
            num_chunks,
        )
        segments = self._subtract_frame_ranges(frame_start, frame_end, skip_ranges)
        if not segments:
            cprint(
                f"Skipping KV quantization for frames {frame_start}:{frame_end}; "
                "entire range selected for BF16 retention",
                "light_cyan",
            )
            self._print_memory_usage()
            return
        for start, end in segments:
            if start < end:
                self.quantize_kv_cache(start * tokens_per_frame, end * tokens_per_frame)

    def _tiered_depth_policy(self) -> str:
        return normalize_tiered_depth_policy(
            getattr(self, "ctx", {}).get("kv_quant_tiered_depth_policy", "none")
        )

    def _structured_temporal_policy(self) -> str:
        return normalize_structured_temporal_policy(
            getattr(self, "ctx", {}).get(
                "kv_quant_structured_temporal_policy", "none"
            )
        )

    def _validate_structured_temporal_configuration(
        self, *, chunk_size: int
    ) -> None:
        """Fail closed before mutating a K-B2/V-B4 structured cache."""

        policy = self._structured_temporal_policy()
        if policy == "none":
            return
        if policy not in {
            STRUCTURED_TEMPORAL_KV_POLICY_41113,
            STRUCTURED_TEMPORAL_KV_POLICY_31114,
            STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE,
        }:
            raise ValueError(f"unsupported structured temporal policy {policy!r}")
        if self._tiered_depth_policy() != "none":
            raise ValueError(
                "structured temporal residual policy cannot be combined with "
                "tiered RVQ depth"
            )
        configured_num_chunks = self.ctx.get("num_chunks")
        if (
            configured_num_chunks is None
            or int(configured_num_chunks)
            != _STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS
        ):
            raise ValueError(
                "structured temporal policy freezes the formal video axis at "
                f"num_chunks={_STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS}, got "
                f"{configured_num_chunks!r}"
            )
        quant_config = getattr(self, "_quant_config", None)
        if quant_config is None or str(
            getattr(quant_config, "quant_type", "none")
        ).lower() == "none":
            raise ValueError(
                "structured temporal policy requires real KV quantization"
            )
        if not bool(getattr(quant_config, "quantize_k", True)) or not bool(
            getattr(quant_config, "quantize_v", True)
        ):
            raise ValueError(
                "structured temporal policy requires both K and V quantization"
            )
        if not bool(getattr(quant_config, "k_rvq", False)) or not bool(
            getattr(quant_config, "v_rvq", False)
        ):
            raise ValueError(
                "structured temporal policy requires packed K/V RVQ bases"
            )
        if not bool(getattr(quant_config, "rvq_residual_quantize", False)):
            raise ValueError(
                "structured temporal policy requires RVQ residual quantization"
            )
        if str(
            getattr(quant_config, "rvq_residual_codec", "")
        ).lower() != "structured_truncatable":
            raise ValueError(
                "structured temporal policy requires "
                "rvq_residual_codec=structured_truncatable"
            )
        if int(getattr(quant_config, "structured_residual_k_bits", 0)) != 2:
            raise ValueError(
                "structured temporal policy requires direct K-B2 encoding"
            )
        if int(getattr(quant_config, "structured_residual_v_bits", 0)) != 4:
            raise ValueError(
                "structured temporal policy requires V-B4 encoding before truncation"
            )
        if str(
            getattr(quant_config, "truncatable_residual_scheme", "")
        ).lower() != "gaussian_tree":
            raise ValueError(
                "structured temporal policy requires gaussian_tree suffixes"
            )
        if not bool(
            getattr(quant_config, "truncatable_residual_hadamard", False)
        ):
            raise ValueError(
                "structured temporal policy requires fixed Hadamard preprocessing"
            )
        if not bool(
            getattr(quant_config, "structured_residual_fast_decode", False)
        ):
            raise ValueError(
                "structured temporal production bridge requires the "
                "fail-closed fused FWHT decoder"
            )
        exact_fields = {
            "truncatable_residual_bits": 4,
            "truncatable_residual_seed": 20260722,
            "structured_residual_group_size": 128,
            "structured_residual_clusters": 256,
            "structured_residual_kmeans_iters": 2,
            "structured_residual_kmeans_seed": 42,
            "structured_residual_scale_refine_iters": 1,
        }
        for name, expected in exact_fields.items():
            observed = int(getattr(quant_config, name, -1))
            if observed != expected:
                raise ValueError(
                    "structured temporal policy freezes the formal codec "
                    f"parameter {name}={expected}, got {observed}"
                )
        params_path = str(
            getattr(quant_config, "truncatable_residual_params_path", "")
        )
        if not params_path or not os.path.isfile(params_path):
            raise ValueError(
                "structured temporal policy requires an existing frozen "
                "structured codec parameter file"
            )
        params_hasher = hashlib.sha256()
        with open(params_path, "rb") as params_handle:
            for block in iter(lambda: params_handle.read(1024 * 1024), b""):
                params_hasher.update(block)
        params_hash = params_hasher.hexdigest()
        if params_hash != _STRUCTURED_TEMPORAL_PARAMS_SHA256:
            raise ValueError(
                "structured temporal codec parameter hash mismatch: "
                f"{params_hash} != {_STRUCTURED_TEMPORAL_PARAMS_SHA256}"
            )
        if str(getattr(quant_config, "quant_type", "")).lower() != (
            "triton-nstages-kmeans-int2"
        ):
            raise ValueError(
                "structured temporal policy freezes the formal outer "
                "quant_type=triton-nstages-kmeans-int2"
            )
        if bool(getattr(quant_config, "rvq_progressive", False)):
            raise ValueError(
                "structured temporal policy requires fixed L8 RVQ bases"
            )
        if str(
            getattr(quant_config, "rvq_feedback_method", "none")
        ).lower() not in {"", "none", "off", "false", "0"}:
            raise ValueError(
                "structured temporal policy does not support RVQ feedback"
            )
        if bool(getattr(quant_config, "rvq_diagnostics", False)):
            raise ValueError(
                "structured temporal policy rejects RVQ diagnostics metadata"
            )
        for tensor_name in ("k", "v"):
            calibration = str(
                getattr(
                    quant_config,
                    f"rvq_{tensor_name}_calibration",
                    "none",
                )
            ).lower()
            if calibration not in {"", "none", "off", "false", "0"}:
                raise ValueError(
                    "structured temporal policy rejects RVQ calibration; "
                    f"{tensor_name.upper()} uses {calibration!r}"
                )
        for tensor_name in ("k", "v"):
            storage = str(
                getattr(
                    quant_config,
                    f"{tensor_name}_rvq_codebook_storage_dtype",
                    "bf16",
                )
            ).lower()
            if storage != "int4":
                raise ValueError(
                    "structured temporal policy requires INT4 runtime "
                    f"codebooks; {tensor_name.upper()} uses {storage!r}"
                )
        if int(self.ctx.get("kv_quant_factor", 2)) != 2:
            raise ValueError(
                "structured temporal policy requires kv_quant_factor=2 so "
                "each physical span matches the formal 8-frame/7040-token axis"
            )
        if int(chunk_size) != int(CHUNK_SIZE):
            raise ValueError(
                "structured temporal policy requires generation "
                f"chunk_size={CHUNK_SIZE}, got {chunk_size}"
            )
        if int(self.ctx.get("first_chunk_size", chunk_size)) != int(chunk_size):
            raise ValueError(
                "structured temporal policy requires first_chunk_size to equal "
                "the generation chunk size"
            )

        incompatible = {
            "kv_quant_skip_chunk_policy": {"", "none", "off", "false", "0"},
            "kv_quant_skip_frame_policy": {"", "none", "off", "false", "0"},
            "kv_quant_recent_policy": {"", "none", "off", "false", "0"},
            "kv_sparse_policy": {"", "none", "off", "false", "0"},
        }
        for name, off_values in incompatible.items():
            value = str(self.ctx.get(name, "none")).lower()
            if value not in off_values:
                raise ValueError(
                    "structured temporal policy is incompatible with "
                    f"{name}={value!r}"
                )
        for name in (
            "kv_quant_skip_first_frames",
            "kv_quant_skip_last_frames",
            "kv_quant_recent_frames",
            "kv_quant_anchor_frame_stride",
        ):
            if int(self.ctx.get(name, 0)) != 0:
                raise ValueError(
                    "structured temporal policy is incompatible with "
                    f"nonzero {name}"
                )

        env_depth = str(os.getenv("HYWORLDPLAY_RVQ_NUM_LEVELS", "")).strip()
        if env_depth and env_depth != "8":
            raise ValueError(
                "structured temporal policy requires L8 RVQ bases; "
                f"HYWORLDPLAY_RVQ_NUM_LEVELS={env_depth!r}"
            )

    def _structured_temporal_audit_container(
        self, *, chunk_size: int
    ) -> dict[str, Any]:
        """Return the audit with explicit suffix and persistent-cache scopes."""

        policy = self._structured_temporal_policy()
        quant_factor = int(self.ctx.get("kv_quant_factor", 2))
        span_frames = int(chunk_size) * quant_factor
        span_tokens = span_frames * 880
        schedule_history = [
            {
                "physical_spans": count,
                "k_schedule": structured_temporal_depth_schedule(
                    count, tensor_name="K", policy=policy
                ),
                "v_schedule": structured_temporal_depth_schedule(
                    count, tensor_name="V", policy=policy
                ),
            }
            for count in range(1, STRUCTURED_TEMPORAL_MAX_SPANS + 1)
        ]
        header = {
            "schema_version": "wan5b_structured_temporal_runtime_audit.v3",
            "policy": policy,
            "policy_scope": (
                "fixed-horizon absolute-position policy for exactly five "
                "physical spans; not an arbitrary-length online age policy"
            ),
            "generation_frames_per_physical_span": span_frames,
            "generation_chunks_per_physical_span": quant_factor,
            "tokens_per_physical_span": span_tokens,
            "tokens_per_frame": 880,
            "formal_num_generation_chunks": (
                _STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS
            ),
            "k_encode_protocol": "direct_b2",
            "v_encode_protocol": "b4_then_immediate_physical_prefix_truncate",
            "terminal_v_schedule": structured_temporal_depth_schedule(
                STRUCTURED_TEMPORAL_MAX_SPANS,
                tensor_name="V",
                policy=policy,
            ),
            "schedule_history": schedule_history,
            "terminal_uniform_control": "direct K/V B2",
            "outer_quant_type_control": "triton-nstages-kmeans-int2",
            "structured_params_sha256": _STRUCTURED_TEMPORAL_PARAMS_SHA256,
            "screen_bitstream_relation": {
                "classification": "closed_loop_production_bridge",
                "formal_exact_screen_predictor_seed": (
                    "derived from task/layer/span/tensor context"
                ),
                "production_predictor_seed": (
                    "passed K/V seed=42; codec effective K=42, V=43"
                ),
                "bit_for_bit_screen_replay_claimed": False,
                "matched_scope": (
                    "video/cache axis, codec family, frozen parameters, "
                    "prefix rates, and terminal payload budget"
                ),
            },
            "decoder": {
                "mode": "fail_closed_fused",
                "backend": "triton_fwht_g128",
                "fallback": "forbidden",
                "generic_structured_default": "eager",
            },
            "quantization_events": [],
            "visibility_events": [],
            "dequantization_events": [],
            "memory_accounting": {
                "scope": (
                    "post-policy attention-visible K+V physical uint8 L8 RVQ "
                    "indices + structured predictor/labels/scale/suffix planes "
                    "+ odd-chunk BF16 pending span; shared compressed-codebook "
                    "payload, runtime materialized codebook/squared-norm cache, "
                    "pre-quantization even-chunk resident BF16, encoder "
                    "workspace, and allocator/container overhead excluded"
                ),
                "pending_bf16_bytes": (
                    FORMAL_PENDING_BF16_BYTES
                ),
                "uniform_compressed_span_bytes": (
                    FORMAL_UNIFORM_COMPRESSED_SPAN_BYTES
                ),
                "one_v_suffix_plane_across_30_layers_bytes": (
                    FORMAL_V_PLANE_ACROSS_30_LAYERS_BYTES
                ),
                "shared_compressed_codebook_payload": (
                    "EXCLUDED_AND_REPORTED_SEPARATELY"
                ),
                "runtime_materialized_codebook_and_squared_norm_cache": (
                    "NOT_AUDITED"
                ),
                "allocator_peak_memory": "NOT_AUDITED",
                "pre_quantization_even_chunk_resident_payload": "NOT_AUDITED",
                "encoder_workspace": "NOT_AUDITED",
                "fairness": (
                    "terminal compressed payload is exact-equal; odd pending "
                    "BF16 is identical to uniform, while nonterminal suffix "
                    "plane deltas are reported per event"
                ),
            },
        }
        audit = self.ctx.setdefault(
            "kv_quant_structured_temporal_audit", header
        )
        if audit.get("policy") != policy:
            raise RuntimeError("structured temporal audit policy changed mid-run")
        return audit

    def _structured_temporal_visibility_event(
        self,
        *,
        chunk_i: int,
        end_index: int,
        action: str,
        token_start: int | None,
        token_end: int | None,
        storage: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Describe the exact cache representation visible to noise prediction."""

        policy = self._structured_temporal_policy()
        analytic = structured_temporal_formal_payload_timeline(
            policy=policy
        )[int(chunk_i) - 1]
        if int(analytic["chunk_i"]) != int(chunk_i):
            raise AssertionError("structured temporal analytic timeline mismatch")
        observed = self._observe_structured_temporal_cache_visibility(
            chunk_i=chunk_i,
            end_index=end_index,
            analytic=analytic,
        )
        return {
            "chunk_i": int(chunk_i),
            "hook_phase": (
                "after_kv_cache_write_before_current_chunk_noise_prediction"
            ),
            "action": str(action),
            "cache_end_token": int(end_index),
            "quantized_token_start": (
                int(token_start) if token_start is not None else None
            ),
            "quantized_token_end": (
                int(token_end) if token_end is not None else None
            ),
            "compressed_physical_spans": analytic[
                "compressed_physical_spans"
            ],
            "k_schedule_visible_to_attention": analytic["k_schedule"],
            "v_schedule_visible_to_attention": analytic["v_schedule"],
            "pending_bf16_frames": analytic["pending_bf16_frames"],
            "pending_bf16_tokens": analytic["pending_bf16_tokens"],
            "pending_bf16_physical_bytes": analytic[
                "pending_bf16_bytes"
            ],
            "pending_bf16_identical_to_uniform_control": True,
            "v_plane_delta_vs_current_uniform_b2": analytic[
                "v_plane_delta_vs_current_uniform_b2"
            ],
            "post_policy_attention_visible_physical_payload_excluding_shared_codebook_bytes": (
                analytic["candidate_bytes"]
            ),
            "post_policy_attention_visible_uniform_same_chunk_payload_excluding_shared_codebook_bytes": (
                analytic["uniform_same_chunk_bytes"]
            ),
            "post_policy_attention_visible_delta_vs_uniform_same_chunk_bytes": (
                analytic["delta_bytes"]
            ),
            "post_policy_attention_visible_physical_payload_excluding_shared_codebook_MiB": (
                analytic["candidate_MiB"]
            ),
            "post_policy_attention_visible_uniform_same_chunk_payload_excluding_shared_codebook_MiB": (
                analytic["uniform_same_chunk_MiB"]
            ),
            "post_update_structured_residual_storage": storage,
            "observed_persistent_cache": observed,
            "allocator_peak_memory": "NOT_AUDITED",
            "pre_quantization_resident_payload": "NOT_AUDITED",
            "encoder_workspace": "NOT_AUDITED",
        }

    def _observe_structured_temporal_cache_visibility(
        self,
        *,
        chunk_i: int,
        end_index: int,
        analytic: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate all 30xK/V cache objects against the analytic expectation."""

        if len(self._kv_cache) != 30:
            raise RuntimeError(
                "formal structured temporal audit requires 30 layers, got "
                f"{len(self._kv_cache)}"
            )
        if int(end_index) != int(chunk_i) * int(CHUNK_SIZE) * 880:
            raise RuntimeError(
                "cache end does not match generation chronology: "
                f"{end_index} != {chunk_i}*{CHUNK_SIZE}*880"
            )
        frame_end = int(end_index) // 880
        compressed_spans = int(analytic["compressed_physical_spans"])
        expected_boundaries = [
            (index * 8, (index + 1) * 8)
            for index in range(compressed_spans)
        ]
        expected_quantized_chunks = list(range(compressed_spans * 8))
        expected_bf16_chunks = (
            list(range(compressed_spans * 8, compressed_spans * 8 + 4))
            if int(chunk_i) % 2
            else []
        )
        expected_schedules = {
            "K": list(analytic["k_schedule"]),
            "V": list(analytic["v_schedule"]),
        }
        packed_bytes = 0
        bf16_bytes = 0
        records: list[dict[str, Any]] = []
        for layer_idx, layer in enumerate(self._kv_cache):
            for tensor_name in ("K", "V"):
                cache = layer.get(tensor_name.lower())
                if not isinstance(cache, ChunkedKVCache):
                    raise RuntimeError("observed cache is not ChunkedKVCache")
                cache_spec = (
                    int(cache.batch_size),
                    int(cache.num_heads),
                    int(cache.head_dim),
                    int(cache.frame_seq_length),
                )
                if cache_spec != (1, 24, 128, 880):
                    raise RuntimeError(
                        "formal cache geometry mismatch "
                        f"B/H/D/Sframe={cache_spec}"
                    )
                states = [
                    getattr(state, "name", str(state))
                    for state in cache.chunk_state[:frame_end]
                ]
                observed_quantized = [
                    index
                    for index, state in enumerate(states)
                    if state == "QUANTIZED"
                ]
                observed_bf16 = [
                    index
                    for index, state in enumerate(states)
                    if state == "BF16"
                ]
                if observed_quantized != expected_quantized_chunks:
                    raise RuntimeError(
                        f"{tensor_name} layer {layer_idx} quantized chunks "
                        f"{observed_quantized} != {expected_quantized_chunks}"
                    )
                if observed_bf16 != expected_bf16_chunks:
                    raise RuntimeError(
                        f"{tensor_name} layer {layer_idx} BF16 chunks "
                        f"{observed_bf16} != {expected_bf16_chunks}"
                    )
                if any(state == "EMPTY" for state in states):
                    raise RuntimeError(
                        f"{tensor_name} layer {layer_idx} has an EMPTY history chunk"
                    )

                spans = sorted(
                    cache.quantized_spans,
                    key=lambda span: int(span["start_chunk"]),
                )
                boundaries = [
                    (int(span["start_chunk"]), int(span["end_chunk"]))
                    for span in spans
                ]
                if boundaries != expected_boundaries:
                    raise RuntimeError(
                        f"{tensor_name} layer {layer_idx} span boundaries "
                        f"{boundaries} != {expected_boundaries}"
                    )
                observed_schedule: list[int] = []
                cache_packed_bytes = 0
                for span in spans:
                    packed = span["quant_data"]
                    residual = packed["residual"]
                    shape = tuple(int(value) for value in residual["shape"])
                    if shape != (1, 24, 7040, 128):
                        raise RuntimeError(
                            "formal structured residual shape mismatch: "
                            f"{shape}"
                        )
                    if int(residual.get("n_clusters", -1)) != 256:
                        raise RuntimeError("formal structured predictor is not C256")
                    if int(residual.get("group_size", -1)) != 128:
                        raise RuntimeError("formal structured suffix is not G128")
                    centroids = residual.get("centroids")
                    cluster_ids = residual.get("cluster_ids")
                    suffix = residual.get("suffix")
                    suffix_scale = (
                        suffix.get("scale")
                        if isinstance(suffix, dict)
                        else None
                    )
                    suffix_planes = (
                        tuple(suffix.get("planes", ()))
                        if isinstance(suffix, dict)
                        else ()
                    )
                    if (
                        not torch.is_tensor(centroids)
                        or centroids.dtype != torch.bfloat16
                        or tuple(centroids.shape)
                        != (1, 24, 256, 128)
                        or not centroids.is_contiguous()
                    ):
                        raise RuntimeError(
                            "formal structured predictor centroids are not "
                            "contiguous BF16 [1,24,256,128]"
                        )
                    if (
                        not torch.is_tensor(cluster_ids)
                        or cluster_ids.dtype != torch.uint8
                        or tuple(cluster_ids.shape) != (1, 24, 7040)
                        or not cluster_ids.is_contiguous()
                    ):
                        raise RuntimeError(
                            "formal structured labels are not contiguous "
                            "uint8 [1,24,7040]"
                        )
                    if (
                        not torch.is_tensor(suffix_scale)
                        or suffix_scale.dtype != torch.float16
                        or int(suffix_scale.numel())
                        != 1 * 24 * 7040
                        or not suffix_scale.is_contiguous()
                    ):
                        raise RuntimeError(
                            "formal structured suffix scale is not contiguous "
                            "FP16 with one value per G128 block"
                        )
                    if (
                        not suffix_planes
                        or any(
                            not torch.is_tensor(plane)
                            or plane.dtype != torch.uint8
                            or not plane.is_contiguous()
                            for plane in suffix_planes
                        )
                    ):
                        raise RuntimeError(
                            "formal structured suffix bitplanes are not "
                            "contiguous uint8 tensors"
                        )
                    if int(suffix.get("plane_count", -1)) != (
                        1 * 24 * 7040 * 128
                    ):
                        raise RuntimeError(
                            "formal structured suffix plane_count does not "
                            "match [1,24,7040,128]"
                        )
                    base_indices = packed["base"].get("indices")
                    if (
                        not torch.is_tensor(base_indices)
                        or tuple(base_indices.shape)
                        != (8, 1, 24, 7040)
                        or base_indices.dtype != torch.uint8
                    ):
                        raise RuntimeError(
                            "formal RVQ base is not physical uint8 "
                            "L8xB1xH24xS7040"
                        )
                    observed_schedule.append(
                        int(residual["suffix"]["stored_bits"])
                    )
                    cache_packed_bytes += packed_tensor_leaf_nbytes(packed)
                if observed_schedule != expected_schedules[tensor_name]:
                    raise RuntimeError(
                        f"{tensor_name} layer {layer_idx} observed schedule "
                        f"{observed_schedule} != "
                        f"{expected_schedules[tensor_name]}"
                    )
                packed_bytes += cache_packed_bytes

                cache_bf16_bytes = 0
                for frame_chunk in observed_bf16:
                    tensor = cache.chunks[frame_chunk]
                    if (
                        not torch.is_tensor(tensor)
                        or tensor.dtype != torch.bfloat16
                        or int(tensor.numel()) != 1 * 24 * 880 * 128
                    ):
                        raise RuntimeError(
                            f"{tensor_name} layer {layer_idx} pending BF16 "
                            "chunk geometry/dtype mismatch"
                        )
                    cache_bf16_bytes += int(
                        tensor.numel() * tensor.element_size()
                    )
                bf16_bytes += cache_bf16_bytes
                records.append(
                    {
                        "layer_idx": layer_idx,
                        "tensor_name": tensor_name,
                        "cache_geometry_B_H_D_Sframe": list(cache_spec),
                        "span_boundaries": [
                            list(boundary) for boundary in boundaries
                        ],
                        "schedule": observed_schedule,
                        "quantized_frame_chunks": observed_quantized,
                        "bf16_frame_chunks": observed_bf16,
                        "packed_tensor_leaf_bytes": cache_packed_bytes,
                        "pending_bf16_tensor_bytes": cache_bf16_bytes,
                    }
                )
        total = packed_bytes + bf16_bytes
        expected_total = int(analytic["candidate_bytes"])
        if bf16_bytes != int(analytic["pending_bf16_bytes"]):
            raise RuntimeError(
                f"observed pending BF16 bytes {bf16_bytes} != "
                f"{analytic['pending_bf16_bytes']}"
            )
        if total != expected_total:
            raise RuntimeError(
                "observed persistent cache payload does not match analytic "
                f"expectation: {total} != {expected_total}"
            )
        return {
            "layer_count": 30,
            "cache_objects": len(records),
            "geometry": {
                "batch": 1,
                "heads": 24,
                "head_dim": 128,
                "tokens_per_frame": 880,
                "tokens_per_compressed_span": 7040,
                "rvq": (
                    "L8C64 physical uint8 indices; external INT4 compressed "
                    "codebook payload is excluded, while runtime materialized "
                    "codebook caches are not audited here"
                ),
                "structured": "C256/G128",
                "structured_tensor_dtypes": {
                    "predictor_centroids": "bfloat16",
                    "cluster_ids": "uint8",
                    "suffix_scale": "float16",
                    "suffix_planes": "uint8",
                },
            },
            "expected_span_boundaries": [
                list(boundary) for boundary in expected_boundaries
            ],
            "observed_packed_tensor_leaf_bytes": packed_bytes,
            "observed_pending_bf16_tensor_bytes": bf16_bytes,
            "observed_post_policy_attention_visible_persistent_cache_bytes": (
                total
            ),
            "analytic_expected_bytes": expected_total,
            "observed_equals_analytic": True,
            "cache_records": records,
        }

    def _finalize_structured_temporal_runtime_audit(
        self, audit: dict[str, Any]
    ) -> None:
        """Fail closed unless chronology, ranges, rate, and peak gates hold."""

        quantization_events = audit["quantization_events"]
        visibility_events = audit["visibility_events"]
        expected_chunks = [2, 4, 6, 8, 10]
        expected_ranges = [
            [index * 7040, (index + 1) * 7040] for index in range(5)
        ]
        observed_chunks = [
            int(event["chunk_i"]) for event in quantization_events
        ]
        observed_ranges = [
            [
                int(event["quantized_token_start"]),
                int(event["quantized_token_end"]),
            ]
            for event in quantization_events
        ]
        visibility_chunks = [
            int(event["chunk_i"]) for event in visibility_events
        ]
        visibility_cache_ends = [
            int(event["cache_end_token"]) for event in visibility_events
        ]
        pending_bf16_frames = [
            int(event["pending_bf16_frames"])
            for event in visibility_events
        ]
        observed_payload_matches = [
            bool(
                event["observed_persistent_cache"][
                    "observed_equals_analytic"
                ]
            )
            for event in visibility_events
        ]
        expected_deltas = (
            [1, 0, -1, -2, 0]
            if audit["policy"]
            == STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE
            else [2, 4, 3, 2, 0]
        )
        observed_deltas = [
            int(event["storage"]["v_plane_delta_vs_current_uniform_b2"])
            for event in quantization_events
        ]
        suffix_capacity_ok = all(
            bool(
                event["storage"][
                    "actual_structured_residual_payload_within_fixed_terminal_uniform_b2_capacity"
                ]
            )
            for event in quantization_events
        )
        persistent_payloads = [
            int(
                event[
                    "post_policy_attention_visible_physical_payload_excluding_shared_codebook_bytes"
                ]
            )
            for event in visibility_events
        ]
        uniform_global_peak_bytes = 3_418_767_360
        candidate_peak_bytes = max(persistent_payloads)
        primary_peak_gate = (
            candidate_peak_bytes <= uniform_global_peak_bytes
            if audit["policy"]
            == STRUCTURED_TEMPORAL_KV_POLICY_31114_PEAK_SAFE
            else None
        )
        final_storage = quantization_events[-1]["storage"]
        checks = {
            "trigger_chunks_exact": observed_chunks == expected_chunks,
            "token_ranges_exact": observed_ranges == expected_ranges,
            "visibility_chunks_exact": visibility_chunks == list(range(1, 11)),
            "visibility_cache_ends_exact": visibility_cache_ends
            == [3520 * index for index in range(1, 11)],
            "pending_bf16_pattern_exact": pending_bf16_frames
            == [4, 0, 4, 0, 4, 0, 4, 0, 4, 0],
            "all_observed_persistent_cache_payloads_match_analytic": all(
                observed_payload_matches
            ),
            "v_plane_deltas_exact": observed_deltas == expected_deltas,
            "all_post_update_structured_suffix_payloads_within_terminal_capacity": (
                suffix_capacity_ok
            ),
            "terminal_payload_exactly_matches_uniform_kv_b2": bool(
                final_storage[
                    "terminal_payload_exactly_matches_uniform_kv_b2"
                ]
            ),
            "primary_post_policy_attention_visible_payload_peak_within_uniform": (
                primary_peak_gate
            ),
        }
        required = [
            value for value in checks.values() if value is not None
        ]
        if not all(required):
            raise RuntimeError(
                "structured temporal final runtime audit failed: "
                f"{checks}"
            )
        audit["final_verification"] = {
            "checks": checks,
            "expected_trigger_chunks": expected_chunks,
            "expected_token_ranges": expected_ranges,
            "expected_v_plane_deltas": expected_deltas,
            "candidate_post_policy_attention_visible_peak_bytes": (
                candidate_peak_bytes
            ),
            "candidate_post_policy_attention_visible_peak_MiB": (
                candidate_peak_bytes / 2**20
            ),
            "uniform_post_policy_attention_visible_peak_bytes": (
                uniform_global_peak_bytes
            ),
            "uniform_post_policy_attention_visible_peak_MiB": (
                uniform_global_peak_bytes / 2**20
            ),
            "candidate_over_uniform_post_policy_attention_visible_peak": (
                candidate_peak_bytes / uniform_global_peak_bytes
            ),
            "allocator_peak_memory": "NOT_AUDITED",
            "pre_quantization_resident_payload": "NOT_AUDITED",
            "encoder_workspace": "NOT_AUDITED",
        }

    def _validate_tiered_depth_configuration(self, *, chunk_size: int) -> None:
        """Fail closed before an experiment-only tiered run mutates the cache."""

        policy = self._tiered_depth_policy()
        if policy == "none":
            return
        spec = tiered_depth_policy_spec(policy)

        quant_config = getattr(self, "_quant_config", None)
        if quant_config is None or str(getattr(quant_config, "quant_type", "none")).lower() == "none":
            raise ValueError("Tiered depth policy requires real RVQ KV quantization")
        if not bool(getattr(quant_config, "quantize_k", True)) or not bool(
            getattr(quant_config, "quantize_v", True)
        ):
            raise ValueError("Tiered depth policy requires both K and V quantization")
        if not bool(getattr(quant_config, "k_rvq", False)) or not bool(
            getattr(quant_config, "v_rvq", False)
        ):
            raise ValueError("Tiered depth policy requires packed RVQ for both K and V")
        if bool(getattr(quant_config, "rvq_progressive", False)):
            raise ValueError("Tiered depth policy is incompatible with progressive RVQ")
        if str(getattr(quant_config, "rvq_feedback_method", "none")).lower() not in {
            "",
            "none",
            "off",
            "false",
            "0",
        }:
            raise ValueError("Tiered depth policy does not support RVQ feedback re-assignment")
        for tensor_name in ("k", "v"):
            storage = str(
                getattr(quant_config, f"{tensor_name}_rvq_codebook_storage_dtype", "bf16")
            ).lower()
            if storage != "int4":
                raise ValueError(
                    "Tiered depth policy requires INT4 runtime codebooks; "
                    f"{tensor_name.upper()} uses {storage!r}"
                )

        if int(self.ctx.get("kv_quant_factor", 2)) != 1:
            raise ValueError("Tiered depth policy requires kv_quant_factor=1")
        if int(chunk_size) != int(CHUNK_SIZE):
            raise ValueError(
                f"Tiered depth policy requires generation chunk_size={CHUNK_SIZE}, got {chunk_size}"
            )
        if int(self.ctx.get("first_chunk_size", chunk_size)) != int(chunk_size):
            raise ValueError(
                "Tiered depth policy requires first_chunk_size to equal the generation chunk size"
            )

        incompatible = {
            "kv_quant_skip_chunk_policy": {"", "none", "off", "false", "0"},
            "kv_quant_skip_frame_policy": {"", "none", "off", "false", "0"},
            "kv_quant_recent_policy": {"", "none", "off", "false", "0"},
            "kv_sparse_policy": {"", "none", "off", "false", "0"},
        }
        for name, off_values in incompatible.items():
            value = str(self.ctx.get(name, "none")).lower()
            if value not in off_values:
                raise ValueError(f"Tiered depth policy is incompatible with {name}={value!r}")
        for name in (
            "kv_quant_skip_first_frames",
            "kv_quant_skip_last_frames",
            "kv_quant_recent_frames",
            "kv_quant_anchor_frame_stride",
        ):
            if int(self.ctx.get(name, 0)) != 0:
                raise ValueError(f"Tiered depth policy is incompatible with nonzero {name}")

        env_depth = str(os.getenv("HYWORLDPLAY_RVQ_NUM_LEVELS", "")).strip()
        if env_depth and env_depth != str(spec.encode_depth):
            raise ValueError(
                "Tiered depth policy must encode each new chunk at "
                f"L{spec.encode_depth}; "
                f"HYWORLDPLAY_RVQ_NUM_LEVELS={env_depth!r}"
            )

    @staticmethod
    def _assert_new_generation_span_is_bf16(
        cache: ChunkedKVCache,
        *,
        token_start: int,
        token_end: int,
    ) -> None:
        if token_start % cache.frame_seq_length or token_end % cache.frame_seq_length:
            raise ValueError("Tiered RVQ token span is not latent-frame aligned")
        start_chunk = token_start // cache.frame_seq_length
        end_chunk = token_end // cache.frame_seq_length
        for frame_chunk in range(start_chunk, end_chunk):
            state = cache.chunk_state[frame_chunk]
            if getattr(state, "name", str(state)) != "BF16":
                raise ValueError(
                    "Tiered RVQ encodes each generation span exactly once; "
                    f"frame chunk {frame_chunk} is {getattr(state, 'name', state)}"
                )

    def _apply_tiered_depth_policy(
        self,
        *,
        chunk_i: int,
        chunk_size: int,
        end_index: int,
    ) -> None:
        """Encode a new span at policy max depth, then truncate aged prefixes."""

        self._validate_tiered_depth_configuration(chunk_size=chunk_size)
        policy = self._tiered_depth_policy()
        spec = tiered_depth_policy_spec(policy)
        tokens_per_frame = 880
        generation_tokens = int(chunk_size) * tokens_per_frame
        if int(end_index) <= 0 or int(end_index) % generation_tokens:
            raise ValueError(
                "Tiered RVQ requires a generation-aligned contiguous cache end; "
                f"end_index={end_index}, generation_tokens={generation_tokens}"
            )
        num_cache_chunks = int(end_index) // generation_tokens
        if int(chunk_i) != num_cache_chunks:
            raise ValueError(
                "Tiered RVQ expects one newly cached generation span per generation call; "
                f"chunk_i={chunk_i}, cached_generation_chunks={num_cache_chunks}"
            )
        token_start = int(end_index) - generation_tokens

        caches: list[ChunkedKVCache] = []
        for layer in self._kv_cache:
            for tensor_name in ("k", "v"):
                cache = layer.get(tensor_name)
                if not isinstance(cache, ChunkedKVCache):
                    raise ValueError("Tiered RVQ requires ChunkedKVCache K/V storage")
                caches.append(cache)

        # Validate every existing cache before replacing the new BF16 span.
        if num_cache_chunks > 1:
            for cache in caches:
                plan_tiered_cache_updates(
                    cache,
                    generation_frames=int(chunk_size),
                    num_cache_chunks=num_cache_chunks - 1,
                    policy=policy,
                )
        for cache in caches:
            self._assert_new_generation_span_is_bf16(
                cache,
                token_start=token_start,
                token_end=int(end_index),
            )

        self.quantize_kv_cache(token_start, int(end_index))

        # Two-phase update: validate all K/V layers before truncating any one
        # of them, so an unsupported/misaligned payload cannot leave mixed depth.
        all_updates = [
            plan_tiered_cache_updates(
                cache,
                generation_frames=int(chunk_size),
                num_cache_chunks=num_cache_chunks,
                policy=policy,
            )
            for cache in caches
        ]
        transition_counts: dict[str, int] = {}
        for updates in all_updates:
            for item in apply_tiered_cache_updates(updates):
                transition = f"{item['previous_depth']}->{item['current_depth']}"
                transition_counts[transition] = transition_counts.get(transition, 0) + 1

        storage = tiered_index_storage_audit(caches)
        expected_schedule = temporal_tiered_depth_schedule(
            num_cache_chunks,
            policy=policy,
        )
        if storage["schedule"] != expected_schedule:
            raise RuntimeError(
                "Tiered RVQ schedule audit mismatch: "
                f"{storage['schedule']} != {expected_schedule}"
            )
        event = {
            "cache_generation_chunks": num_cache_chunks,
            "schedule": expected_schedule,
            "average_depth": storage["average_depth"],
            "transition_counts_across_kv_layers": transition_counts,
            "storage": storage,
        }
        audit_header = {
            "policy": policy,
            "generation_frames": int(chunk_size),
            "tokens_per_frame": tokens_per_frame,
            "new_span_encode_depth": spec.encode_depth,
            "allowed_depths": list(spec.allowed_depths),
            "aging_operation": "indices_prefix_clone_only_no_reencode",
            "events": [],
        }
        if policy == H7_TIERED_POLICY:
            audit_header.update(
                {
                    "boundary_depth": 8,
                    "middle_depth": 6,
                    "old_depth": 4,
                    "oldest_middle_chunks": 2,
                    "recent_chunks": 1,
                }
            )
        elif policy == AGE_RAMP16_TIERED_POLICY:
            audit_header.update(
                {
                    "anchor_depth": 16,
                    "middle_depth": 4,
                    "recent_depths_old_to_new": [6, 8, 12, 16],
                }
            )
        audit = self.ctx.setdefault(
            "kv_quant_tiered_depth_audit",
            audit_header,
        )
        audit["events"].append(event)
        audit["final"] = event
        cprint(
            f"{policy} tiered RVQ schedule "
            f"{expected_schedule}, avg_depth={storage['average_depth']:.3f}, "
            f"actual_uint8_indices={storage['actual_uint8_index_payload_bytes'] / 1024**2:.3f} MiB",
            "light_green",
        )
        self._print_memory_usage()

    def _apply_structured_temporal_policy(
        self,
        *,
        chunk_i: int,
        chunk_size: int,
        end_index: int,
    ) -> dict[str, Any]:
        """Encode K-B2/V-B4 once, then physically age only V prefixes."""

        self._validate_structured_temporal_configuration(
            chunk_size=chunk_size
        )
        tokens_per_frame = 880
        quant_factor = int(self.ctx.get("kv_quant_factor", 2))
        span_frames = int(chunk_size) * quant_factor
        span_tokens = span_frames * tokens_per_frame
        if int(end_index) <= 0 or int(end_index) % span_tokens:
            raise ValueError(
                "structured temporal policy requires a generation-aligned "
                "two-chunk cache end; "
                f"end_index={end_index}, span_tokens={span_tokens}"
            )
        num_cache_spans = int(end_index) // span_tokens
        if num_cache_spans > STRUCTURED_TEMPORAL_MAX_SPANS:
            raise ValueError(
                "structured temporal policy is defined for at most "
                f"{STRUCTURED_TEMPORAL_MAX_SPANS} cached spans, got "
                f"{num_cache_spans}"
            )
        if int(chunk_i) != num_cache_spans * quant_factor:
            raise ValueError(
                "structured temporal policy expects one new physical span per "
                f"two generation chunks; chunk_i={chunk_i}, "
                f"spans={num_cache_spans}, quant_factor={quant_factor}"
            )
        token_start = int(end_index) - span_tokens

        caches: list[tuple[str, ChunkedKVCache]] = []
        for layer in self._kv_cache:
            for tensor_name in ("K", "V"):
                cache = layer.get(tensor_name.lower())
                if not isinstance(cache, ChunkedKVCache):
                    raise ValueError(
                        "structured temporal policy requires ChunkedKVCache K/V"
                    )
                caches.append((tensor_name, cache))

        # Validate every old K/V layer before the new span replaces BF16.
        if num_cache_spans > 1:
            plan_structured_temporal_kv_updates(
                caches,
                generation_frames=span_frames,
                num_cache_spans=num_cache_spans - 1,
                policy=self._structured_temporal_policy(),
            )
        for _tensor_name, cache in caches:
            self._assert_new_generation_span_is_bf16(
                cache,
                token_start=token_start,
                token_end=int(end_index),
            )

        # compress_kv_cache reads the explicit per-tensor suffix bits from the
        # validated quant_config: K is direct B2, V is direct B4.
        self.quantize_kv_cache(token_start, int(end_index))

        # Post-encode prefix updates are atomically preflighted across every
        # layer/tensor before the first physical bitplane mutation.  This does
        # not claim transactional rollback for quantize_kv_cache itself.
        all_updates = plan_structured_temporal_kv_updates(
            caches,
            generation_frames=span_frames,
            num_cache_spans=num_cache_spans,
            policy=self._structured_temporal_policy(),
        )
        transition_counts: dict[str, int] = {}
        for updates in all_updates:
            for item in apply_structured_temporal_cache_updates(updates):
                transition = (
                    f"{item['tensor_name']}:"
                    f"{item['previous_bits']}->{item['current_bits']}"
                )
                transition_counts[transition] = (
                    transition_counts.get(transition, 0) + 1
                )

        policy = self._structured_temporal_policy()
        storage = structured_temporal_storage_audit(caches, policy=policy)
        expected_k = structured_temporal_depth_schedule(
            num_cache_spans, tensor_name="K", policy=policy
        )
        expected_v = structured_temporal_depth_schedule(
            num_cache_spans, tensor_name="V", policy=policy
        )
        if storage["k_schedule"] != expected_k:
            raise RuntimeError(
                "structured temporal K schedule audit mismatch: "
                f"{storage['k_schedule']} != {expected_k}"
            )
        if storage["v_schedule"] != expected_v:
            raise RuntimeError(
                "structured temporal V schedule audit mismatch: "
                f"{storage['v_schedule']} != {expected_v}"
            )
        if (
            num_cache_spans == STRUCTURED_TEMPORAL_MAX_SPANS
            and not storage["terminal_payload_exactly_matches_uniform_kv_b2"]
        ):
            raise RuntimeError(
                "terminal structured temporal payload is not exactly matched "
                "to direct uniform K/V B2"
            )
        event = {
            "chunk_i": int(chunk_i),
            "cache_generation_spans": num_cache_spans,
            "quantized_token_start": int(token_start),
            "quantized_token_end": int(end_index),
            "k_schedule": expected_k,
            "v_schedule": expected_v,
            "transition_counts_across_kv_layers": transition_counts,
            "storage": storage,
        }
        audit = self._structured_temporal_audit_container(
            chunk_size=chunk_size
        )
        audit["quantization_events"].append(event)
        audit["final"] = event
        cprint(
            "structured temporal residual schedules "
            f"K={expected_k}, V={expected_v}, "
            "delta_vs_uniform_B2="
            f"{storage['payload_delta_vs_uniform_kv_b2_bytes'] / 1024**2:.3f} MiB",
            "light_green",
        )
        self._print_memory_usage()
        return event

    def _temporal_adaptive_residual_enabled(self) -> bool:
        return bool(
            getattr(
                getattr(self, "_quant_config", None),
                "temporal_adaptive_residual",
                False,
            )
        )

    def _validate_temporal_adaptive_residual_configuration(
        self,
        *,
        chunk_size: int,
    ) -> None:
        if not self._temporal_adaptive_residual_enabled():
            return
        config = self._quant_config
        errors: list[str] = []
        if int(self.ctx.get("kv_quant_factor", 2)) != 1:
            errors.append("kv_quant_factor must be 1")
        if not bool(getattr(config, "rvq_residual_quantize", False)):
            errors.append("rvq_residual_quantize must be enabled")
        if not (
            bool(getattr(config, "k_rvq", False))
            and bool(getattr(config, "v_rvq", False))
        ):
            errors.append("both K and V must use RVQ")
        if not bool(getattr(config, "rvq_relative_stop", False)):
            errors.append("Adaptive RVQ relative stopping must be enabled")
        if int(getattr(config, "rvq_relative_stop_min_depth", -1)) != 4:
            errors.append("Adaptive RVQ minimum depth must be L4")
        if int(getattr(config, "rvq_relative_stop_max_depth", -1)) != 16:
            errors.append("Adaptive RVQ maximum depth must be L16")
        if abs(
            float(getattr(config, "rvq_relative_stop_threshold", -1.0))
            - 0.0629
        ) > 1e-9:
            errors.append("Adaptive RVQ relative threshold must be 0.0629")
        if not str(
            getattr(config, "temporal_adaptive_profile_path", "") or ""
        ):
            errors.append("temporal_adaptive_profile_path is required")
        if not str(getattr(config, "routed_gaussian_routing_path", "") or ""):
            errors.append("K routed Gaussian routing table is required")
        if not str(
            getattr(config, "progressive_v_absmax_params_path", "") or ""
        ):
            errors.append("V progressive absmax params are required")
        allocation_mode = str(
            getattr(config, "temporal_adaptive_allocation_mode", "optimal")
            or "optimal"
        ).lower()
        if allocation_mode not in {
            "optimal",
            "force_recent_b16",
            "frozen_joint_schedule",
        }:
            errors.append(
                "temporal_adaptive_allocation_mode must be optimal, "
                "force_recent_b16, or frozen_joint_schedule"
            )
        if int(chunk_size) * 880 <= 0:
            errors.append("generation chunk must contain complete latent frames")
        incompatible = {
            "tiered RVQ": self._tiered_depth_policy() != "none",
            "structured temporal": self._structured_temporal_policy() != "none",
            "recent BF16": str(
                self.ctx.get("kv_quant_recent_policy", "none")
            ).lower()
            not in {"", "none"},
            "recent frames": int(self.ctx.get("kv_quant_recent_frames", 0)) > 0,
            "sparse cache": self._use_sparse_kv_policy(),
        }
        errors.extend(
            f"incompatible policy enabled: {name}"
            for name, enabled in incompatible.items()
            if enabled
        )
        if errors:
            raise ValueError(
                "invalid adaptive temporal residual configuration: "
                + "; ".join(errors)
            )

    def _temporal_adaptive_profile(self):
        profile = self.ctx.get("kv_quant_temporal_adaptive_profile")
        if profile is None:
            profile = load_temporal_adaptive_profile(
                self._quant_config.temporal_adaptive_profile_path
            )
            self.ctx["kv_quant_temporal_adaptive_profile"] = profile
        return profile

    @staticmethod
    def _combined_rvq_hash(states: list[dict]) -> str:
        digest = hashlib.sha256()
        for state in states:
            digest.update(str(state["rvq_identity_hash"]).encode("ascii"))
        return digest.hexdigest()

    def _apply_temporal_adaptive_residual_policy(
        self,
        *,
        chunk_i: int,
        chunk_size: int,
        token_start: int,
        end_index: int,
    ) -> dict[str, Any]:
        """Admit one B16 chunk, then solve/apply the monotone precision plan."""

        # This wall-clock interval deliberately surrounds both adaptive-RVQ
        # admission and the later B16->B8 materialization/truncation policy.
        # The legacy ``KV Cache Quantization Time`` printed inside
        # ``quantize_kv_cache`` only measures the former and is retained for
        # backward-compatible metric accounting.
        torch.cuda.synchronize()
        full_policy_started = time.perf_counter()

        self._validate_temporal_adaptive_residual_configuration(
            chunk_size=chunk_size
        )
        first_cache = self._kv_cache[0]["k"]
        if not isinstance(first_cache, ChunkedKVCache):
            raise ValueError("temporal adaptive policy requires ChunkedKVCache")
        span_tokens = int(chunk_size) * 880
        if (
            int(end_index) - int(token_start) != span_tokens
            or int(token_start) % span_tokens
        ):
            raise ValueError(
                "temporal adaptive admission must be one origin-aligned "
                f"{span_tokens}-token chunk, got [{token_start},{end_index})"
            )
        new_chunk = int(token_start) // span_tokens
        if new_chunk >= 10:
            raise ValueError("temporal adaptive schedule supports 10 cache chunks")

        def span_bounds(cache_chunk: int) -> tuple[int, int]:
            start_frame = int(cache_chunk) * int(chunk_size)
            return start_frame, start_frame + int(chunk_size)

        def packed_state(cache: ChunkedKVCache, cache_chunk: int) -> dict:
            start_frame, end_frame = span_bounds(cache_chunk)
            return cache.quantized_span_data(start_frame, end_frame)

        profile = self._temporal_adaptive_profile()
        capacity = int(
            getattr(
                self._quant_config,
                "temporal_adaptive_capacity_bit_chunks",
                40,
            )
        )
        # The rate-distortion schedule depends only on the resident precision
        # vector and the frozen profile, so the new chunk's target is known
        # before its B16 admission.  Pass that target into the encoder to avoid
        # installing a BF16 bypass that the policy immediately discards.
        resident_current: list[int] = []
        for cache_chunk in range(new_chunk):
            reference = packed_state(first_cache, cache_chunk)
            validate_temporal_state(reference)
            resident_current.append(int(reference["bits"]))
        preview_target = list(
            schedule_temporal_precisions(
                [*resident_current, 16],
                profile,
                capacity_bit_chunks=capacity,
                allocation_mode=str(
                    getattr(
                        self._quant_config,
                        "temporal_adaptive_allocation_mode",
                        "optimal",
                    )
                    or "optimal"
                ),
            )
        )
        admission_target_bits = int(preview_target[-1])

        # compress_kv_cache performs adaptive RVQ exactly once and, under the
        # temporal flag, now materializes the new chunk directly at the frozen
        # schedule target.  Restore the transient config attribute even when
        # admission raises so later pipeline calls cannot inherit stale state.
        admission_attr = "temporal_adaptive_admission_target_bits"
        memory_attr = "suppress_quantize_memory_report"
        had_admission_attr = hasattr(self._quant_config, admission_attr)
        had_memory_attr = hasattr(self._quant_config, memory_attr)
        previous_admission_target = getattr(
            self._quant_config, admission_attr, None
        )
        previous_memory_suppression = getattr(
            self._quant_config, memory_attr, None
        )
        setattr(self._quant_config, admission_attr, admission_target_bits)
        # ``quantize_kv_cache`` historically performs a full 30-layer
        # offload/onload solely to print memory.  The temporal policy already
        # reports memory once after all transitions, so suppress the duplicate
        # mid-policy audit.  It is diagnostic I/O, not codec encode work.
        setattr(self._quant_config, memory_attr, True)
        try:
            self.quantize_kv_cache(int(token_start), int(end_index))
        finally:
            if had_admission_attr:
                setattr(
                    self._quant_config,
                    admission_attr,
                    previous_admission_target,
                )
            else:
                delattr(self._quant_config, admission_attr)
            if had_memory_attr:
                setattr(
                    self._quant_config,
                    memory_attr,
                    previous_memory_suppression,
                )
            else:
                delattr(self._quant_config, memory_attr)

        caches: list[tuple[str, int, ChunkedKVCache]] = []
        for layer_index, layer in enumerate(self._kv_cache):
            for tensor_name in ("K", "V"):
                cache = layer[tensor_name.lower()]
                if not isinstance(cache, ChunkedKVCache):
                    raise ValueError("temporal adaptive policy requires chunk caches")
                caches.append((tensor_name, layer_index, cache))

        exhaustive_validation = self._env_flag(
            "HYWORLDPLAY_TEMPORAL_EXHAUSTIVE_VALIDATION", "0"
        )
        current: list[int] = []
        for cache_chunk in range(new_chunk + 1):
            reference = packed_state(first_cache, cache_chunk)
            validate_temporal_state(reference)
            current.append(int(reference["bits"]))
        if exhaustive_validation:
            for _name, _layer, cache in caches:
                observed = []
                for cache_chunk in range(new_chunk + 1):
                    state = packed_state(cache, cache_chunk)
                    validate_temporal_state(state)
                    observed.append(int(state["bits"]))
                if observed != current:
                    raise RuntimeError(
                        f"K/V layer precision schedules diverged: {observed} != {current}"
                    )

        target = list(
            schedule_temporal_precisions(
                current,
                profile,
                capacity_bit_chunks=capacity,
                allocation_mode=str(
                    getattr(
                        self._quant_config,
                        "temporal_adaptive_allocation_mode",
                        "optimal",
                    )
                    or "optimal"
                ),
            )
        )
        if target != preview_target:
            raise RuntimeError(
                "pre-admission and post-admission temporal schedules diverged: "
                f"{preview_target} != {target}"
            )

        # The first cache is the scheduler source of truth.  Walking the full
        # nested RVQ tree for every layer, K/V state and resident chunk costs
        # more than the codec itself.  Keep that audit opt-in for formal debug
        # runs; production still validates every state touched by a transition,
        # enforces monotonicity and compares its stored RVQ identity before and
        # after the update.
        if exhaustive_validation:
            for _name, _layer, cache in caches:
                for cache_chunk, target_bits in enumerate(target):
                    state = packed_state(cache, cache_chunk)
                    if int(target_bits) > int(state["bits"]):
                        raise RuntimeError("scheduler requested a precision increase")
                    if not rvq_identity_is_valid(state):
                        raise RuntimeError("adaptive RVQ identity audit failed")

        transition_rows: list[dict[str, Any]] = []
        step = new_chunk + 1
        for cache_chunk, (old_bits, new_bits) in enumerate(zip(current, target)):
            if old_bits == new_bits:
                continue
            affected_before = [
                packed_state(cache, cache_chunk)
                for _name, _layer, cache in caches
            ]
            before_hash = self._combined_rvq_hash(affected_before)
            for _name, _layer, cache in caches:
                old_state = packed_state(cache, cache_chunk)
                new_state = transition_temporal_state(
                    old_state,
                    new_bits,
                    quant_config=self._quant_config,
                    verify_identity=exhaustive_validation,
                )
                start_frame, end_frame = span_bounds(cache_chunk)
                cache.replace_quantized_span_data(
                    start_frame, end_frame, new_state
                )
            affected_after = [
                packed_state(cache, cache_chunk)
                for _name, _layer, cache in caches
            ]
            after_hash = self._combined_rvq_hash(affected_after)
            if before_hash != after_hash:
                raise RuntimeError("RVQ hash changed while truncating residual")
            age = new_chunk - cache_chunk
            weight = profile.weight(
                chunk_index=cache_chunk,
                current_chunks=new_chunk + 1,
            )
            for previous, present in temporal_transition_steps(
                old_bits, new_bits
            ):
                transition_rows.append(
                    {
                        "step": step,
                        "chunk": cache_chunk,
                        "old_bits": previous,
                        "new_bits": present,
                        "age": age,
                        "weight": float(weight),
                        "distortion_delta": float(
                            weight
                            * (
                                profile.distortion(present)
                                - profile.distortion(previous)
                            )
                        ),
                        "released_nominal_bits": previous - present,
                        "rvq_hash_before": before_hash,
                        "rvq_hash_after": after_hash,
                    }
                )

        # Direct admission bypasses the physical B16 state but keeps the same
        # logical 16->target transition rows for schedule auditing.
        if admission_target_bits < 16:
            admitted_states = [
                packed_state(cache, new_chunk)
                for _name, _layer, cache in caches
            ]
            admitted_hash = self._combined_rvq_hash(admitted_states)
            weight = profile.weight(
                chunk_index=new_chunk,
                current_chunks=new_chunk + 1,
            )
            for previous, present in temporal_transition_steps(
                16, admission_target_bits
            ):
                transition_rows.append(
                    {
                        "step": step,
                        "chunk": new_chunk,
                        "old_bits": previous,
                        "new_bits": present,
                        "age": 0,
                        "weight": float(weight),
                        "distortion_delta": float(
                            weight
                            * (
                                profile.distortion(present)
                                - profile.distortion(previous)
                            )
                        ),
                        "released_nominal_bits": previous - present,
                        "rvq_hash_before": admitted_hash,
                        "rvq_hash_after": admitted_hash,
                    }
                )

        audit = self.ctx.setdefault(
            "kv_quant_temporal_adaptive_audit",
            {
                "schema_version": "wan5b_temporal_adaptive_residual.audit.v1",
                "budget_definition": (
                    "rvq_base_reservation_plus_nominal_residual"
                    if profile.residual_capacity_by_step is not None
                    else "residual_nominal_bits_only"
                ),
                "capacity_bit_chunks": capacity,
                "allocation_mode": str(
                    getattr(
                        self._quant_config,
                        "temporal_adaptive_allocation_mode",
                        "optimal",
                    )
                    or "optimal"
                ),
                "allowed_bits": [16, 8, 4, 3, 2, 1, 0],
                "events": [],
                "transitions": [],
            },
        )
        cells = target + [None] * (10 - len(target))
        event = {
            "step": step,
            "chunk_i": int(chunk_i),
            "current_cache_chunks": len(target),
            "schedule": cells,
            "nominal_bits_sum": int(sum(target)),
            "current_average": float(sum(target) / len(target)),
            "token_start": int(token_start),
            "token_end": int(end_index),
        }
        if profile.residual_capacity_by_step is not None:
            event["residual_capacity_after_base"] = profile.residual_capacity_by_step[step - 1]
            event["cache_accounting"] = []
            for name, layer, cache in caches:
                states = [packed_state(cache, i) for i in range(step)]
                if [int(state["bits"]) for state in states] != target:
                    raise RuntimeError("actual K/V cache precision differs from the schedule")
                shape = states[0]["shape"]
                event["cache_accounting"].append({
                    "tensor": name,
                    "layer": layer,
                    "elements_per_chunk": int(shape[0] * shape[1] * shape[2] * shape[3]),
                    "rvq_base_bytes": sum(
                        temporal_payload_nbytes(state) - temporal_payload_nbytes(state, include_rvq=False)
                        for state in states
                    ),
                    "physical_payload_bytes": sum(temporal_payload_nbytes(state) for state in states),
                    "selected_bits": list(target),
                })
        audit["events"].append(event)
        audit["transitions"].extend(transition_rows)
        audit["final"] = event
        terminal_capacity = (
            profile.joint_schedule_terminal_bit_chunks
            if profile.joint_schedule_terminal_bit_chunks is not None
            and (
                str(getattr(self._quant_config, "temporal_adaptive_allocation_mode", "optimal"))
                == "frozen_joint_schedule"
                or profile.residual_capacity_by_step is not None
            )
            else capacity
        )
        if step == 10 and int(sum(target)) != terminal_capacity:
            raise RuntimeError(
                f"terminal temporal budget is {sum(target)}/{terminal_capacity}, expected exact"
            )
        cprint(
            f"temporal adaptive residual step {step}: {target}, "
            f"nominal={sum(target)}/{event.get('residual_capacity_after_base', capacity)}, "
            f"peak_capacity={capacity}, terminal_target={terminal_capacity}",
            "light_green",
        )
        torch.cuda.synchronize()
        full_policy_seconds = time.perf_counter() - full_policy_started
        cprint(
            f"KV Cache Temporal Policy Time: {full_policy_seconds:.6f} s",
            "light_cyan",
        )
        self._print_memory_usage()
        return event

    def maybe_quantize_kv_cache_with_policy(
        self,
        *,
        chunk_i: int,
        chunk_size: int,
        n_latent: int,
        num_chunks: int,
        token_start: int,
        end_index: int,
    ) -> None:
        quant_factor = int(self.ctx.get("kv_quant_factor", 2))
        tokens_per_frame = 880
        recent_policy = str(self.ctx.get("kv_quant_recent_policy", "none")).lower()
        recent_frames = max(0, int(self.ctx.get("kv_quant_recent_frames", 0)))

        if self._temporal_adaptive_residual_enabled():
            self._apply_temporal_adaptive_residual_policy(
                chunk_i=chunk_i,
                chunk_size=chunk_size,
                token_start=token_start,
                end_index=end_index,
            )
            return

        if self._tiered_depth_policy() != "none":
            self._apply_tiered_depth_policy(
                chunk_i=chunk_i,
                chunk_size=chunk_size,
                end_index=end_index,
            )
            return

        if self._structured_temporal_policy() != "none":
            if int(num_chunks) != _STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS:
                raise ValueError(
                    "structured temporal runtime hook requires "
                    f"num_chunks={_STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS}, "
                    f"got {num_chunks}"
                )
            audit = self._structured_temporal_audit_container(
                chunk_size=chunk_size
            )
            if chunk_i >= quant_factor and chunk_i % quant_factor == 0:
                quantization_event = self._apply_structured_temporal_policy(
                    chunk_i=chunk_i,
                    chunk_size=chunk_size,
                    end_index=end_index,
                )
                visibility_event = self._structured_temporal_visibility_event(
                    chunk_i=chunk_i,
                    end_index=end_index,
                    action="quantize_two_chunks_then_apply_prefix",
                    token_start=quantization_event[
                        "quantized_token_start"
                    ],
                    token_end=quantization_event["quantized_token_end"],
                    storage=quantization_event["storage"],
                )
            else:
                visibility_event = self._structured_temporal_visibility_event(
                    chunk_i=chunk_i,
                    end_index=end_index,
                    action="wait_with_one_pending_bf16_generation_chunk",
                    token_start=None,
                    token_end=None,
                    storage=None,
                )
                cprint(
                    "structured temporal residual waiting for the second "
                    f"generation chunk of an 8-frame span (chunk_i={chunk_i})",
                    "light_cyan",
                )
                self._print_memory_usage()
            audit["visibility_events"].append(visibility_event)
            if int(chunk_i) == 10:
                self._finalize_structured_temporal_runtime_audit(audit)
            return

        if recent_frames > 0:
            current_cached_frames = int(end_index) // tokens_per_frame
            last_quantized_frame = int(self.ctx.get("kv_recent_quantized_frame_end", 0))
            frame_end = max(0, current_cached_frames - recent_frames)
            if frame_end > last_quantized_frame:
                cprint(
                    f"Recent-frame BF16 policy quantizing frames {last_quantized_frame}:{frame_end}; "
                    f"keeping recent frames {frame_end}:{current_cached_frames} BF16",
                    "light_cyan",
                )
                self.quantize_kv_cache_frame_range_with_policy(
                    last_quantized_frame,
                    frame_end,
                    n_latent=n_latent,
                    chunk_size=chunk_size,
                    quant_factor=quant_factor,
                    num_chunks=num_chunks,
                    tokens_per_frame=tokens_per_frame,
                )
                self.ctx["kv_recent_quantized_frame_end"] = frame_end
            else:
                cprint(
                    f"Recent-frame BF16 policy keeping frames {last_quantized_frame}:{current_cached_frames} "
                    "unquantized",
                    "light_cyan",
                )
                self._print_memory_usage()
            return

        if recent_policy in {"nearest1", "nearest2"}:
            keep_chunks = 1 if recent_policy == "nearest1" else 2
            current_cached_frames = int(end_index) // tokens_per_frame
            last_quantized_frame = int(self.ctx.get("kv_recent_quantized_frame_end", 0))
            frame_end = max(0, current_cached_frames - keep_chunks * chunk_size)
            if frame_end > last_quantized_frame:
                cprint(
                    f"Nearest-BF16 policy quantizing frames {last_quantized_frame}:{frame_end}; "
                    f"keeping recent frames {frame_end}:{current_cached_frames} BF16",
                    "light_cyan",
                )
                self.quantize_kv_cache_frame_range_with_policy(
                    last_quantized_frame,
                    frame_end,
                    n_latent=n_latent,
                    chunk_size=chunk_size,
                    quant_factor=quant_factor,
                    num_chunks=num_chunks,
                    tokens_per_frame=tokens_per_frame,
                )
                self.ctx["kv_recent_quantized_frame_end"] = frame_end
            else:
                cprint(
                    f"Nearest-BF16 policy keeping frames {last_quantized_frame}:{current_cached_frames} "
                    "unquantized",
                    "light_cyan",
                )
                self._print_memory_usage()
            return

        if recent_policy == "recent1or2":
            current_cached_frames = int(end_index) // tokens_per_frame
            last_quantized_frame = int(self.ctx.get("kv_recent_quantized_frame_end", 0))
            if current_cached_frames - last_quantized_frame >= 3 * chunk_size:
                frame_start = last_quantized_frame
                frame_end = min(frame_start + 2 * chunk_size, current_cached_frames)
                cprint(
                    f"Recent-BF16 policy quantizing frames {frame_start}:{frame_end}; "
                    f"keeping recent frames {frame_end}:{current_cached_frames} BF16",
                    "light_cyan",
                )
                self.quantize_kv_cache_frame_range_with_policy(
                    frame_start,
                    frame_end,
                    n_latent=n_latent,
                    chunk_size=chunk_size,
                    quant_factor=quant_factor,
                    num_chunks=num_chunks,
                    tokens_per_frame=tokens_per_frame,
                )
                self.ctx["kv_recent_quantized_frame_end"] = frame_end
            else:
                cprint(
                    f"Recent-BF16 policy keeping frames {last_quantized_frame}:{current_cached_frames} "
                    "unquantized",
                    "light_cyan",
                )
                self._print_memory_usage()
            return

        if chunk_i >= quant_factor and chunk_i % quant_factor == 0:
            frame_start = (chunk_i - quant_factor) * chunk_size
            frame_end = chunk_i * chunk_size
            self.quantize_kv_cache_frame_range_with_policy(
                frame_start,
                frame_end,
                n_latent=n_latent,
                chunk_size=chunk_size,
                quant_factor=quant_factor,
                num_chunks=num_chunks,
                tokens_per_frame=tokens_per_frame,
            )

    def _clear_kv_cache_for_sparse_policy(self) -> None:
        for layer in self._kv_cache:
            k_cache = layer.get("k")
            v_cache = layer.get("v")
            if hasattr(k_cache, "clear"):
                k_cache.clear()
            if hasattr(v_cache, "clear"):
                v_cache.clear()
            layer["end_index"] = 0

    def _sparse_selected_frame_indices(
        self,
        current_frame_idx: int,
        chunk_size: int,
    ) -> Optional[list[int]]:
        policy = str(self.ctx.get("kv_sparse_policy", "none")).lower()
        if policy in {"", "none", "off", "false", "0"}:
            return None
        if policy != "first_recent_chunks":
            raise ValueError(f"Unsupported kv_sparse_policy={policy!r}")

        first_frames = int(self.ctx.get("kv_sparse_first_frames", 0))
        recent_frames = int(self.ctx.get("kv_sparse_recent_frames", 0))
        if first_frames <= 0:
            first_frames = int(chunk_size)
        if recent_frames <= 0:
            recent_frames = int(chunk_size)

        selected = set(range(0, min(first_frames, current_frame_idx)))
        recent_start = max(0, current_frame_idx - recent_frames)
        selected.update(range(recent_start, current_frame_idx))
        return sorted(selected)

    def _use_sparse_kv_policy(self) -> bool:
        policy = str(self.ctx.get("kv_sparse_policy", "none")).lower()
        return policy not in {"", "none", "off", "false", "0"}

    def _save_quantized_kv_cache_layer(
        self,
        layer_idx: int,
        token_start: int,
        token_end: int,
        k_quant: dict,
        v_quant: dict,
    ):
        enabled = os.getenv("HYWORLDPLAY_SAVE_QUANTIZED_KV_CACHE", "0").lower() not in {
            "0",
            "false",
            "no",
            "",
        }
        if not enabled:
            return

        save_dir = os.getenv("HYWORLDPLAY_QUANTIZED_KV_CACHE_SAVE_DIR")
        if not save_dir:
            save_dir = os.path.join("results", "hyworldplay", "quantized_kv_cache")

        span_dir = os.path.join(save_dir, f"span_{token_start:06d}_{token_end:06d}")
        os.makedirs(span_dir, exist_ok=True)
        save_path = os.path.join(span_dir, f"layer_{layer_idx:02d}.pt")
        payload = {
            "layer_idx": layer_idx,
            "token_start": token_start,
            "token_end": token_end,
            "cache_mode": self._kv_cache_mode,
            "k": self._detach_to_cpu(k_quant),
            "v": self._detach_to_cpu(v_quant),
        }
        torch.save(payload, save_path)

        if layer_idx == 0:
            manifest_path = os.path.join(save_dir, "manifest.jsonl")
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "token_start": token_start,
                            "token_end": token_end,
                            "span_dir": span_dir,
                            "cache_mode": self._kv_cache_mode,
                        }
                    )
                    + "\n"
                )

    def _detach_to_cpu(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {key: self._detach_to_cpu(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [self._detach_to_cpu(value) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._detach_to_cpu(value) for value in obj)
        return obj

    def _print_kv_cache_mse_error(self, k, k_quant, v, v_quant, layer_idx):
        """Print the Rel L2 error between the original and quantized KV cache."""
        if isinstance(k_quant, dict) or isinstance(v_quant, dict):
            return
        k_rel_l2 = torch.norm(k - k_quant, p=2) / torch.norm(k, p=2)
        v_rel_l2 = torch.norm(v - v_quant, p=2) / torch.norm(v, p=2)
        cprint(
            f"Layer {layer_idx:02d} | K Rel L2: {k_rel_l2:.4f} | V Rel L2: {v_rel_l2:.4f}",
            "light_blue",
        )

    def _print_memory_usage(self):
        """Offload all KV cache, then onload layer-by-layer to measure memory usage."""
        torch.cuda.empty_cache()
        if self._env_flag("HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD", "0"):
            for layer in self._kv_cache:
                offload_kv_cache_layer(layer)
            memory_usage_before_onload = torch.cuda.memory_allocated() / 1024**2
            per_layer_memory_usage = [
                self._stored_kv_cache_layer_bytes(layer) / 1024**2
                for layer in self._kv_cache
            ]
            total_kv_cache_memory_usage = sum(per_layer_memory_usage)
            median_per_layer_memory_usage = (
                float(np.median(per_layer_memory_usage))
                if per_layer_memory_usage
                else 0.0
            )
            peak_memory_usage = memory_usage_before_onload + total_kv_cache_memory_usage

            cprint(
                f"Peak Memory Usage: {peak_memory_usage:.2f} MB | Other Memory Usage: {memory_usage_before_onload:.2f} MB | "
                f"Per Layer Memory Usage: {median_per_layer_memory_usage:.2f} MB | Total KV Cache Memory Usage: {total_kv_cache_memory_usage:.2f} MB",
                "light_blue",
            )
            torch.cuda.empty_cache()
            return

        cuda_device = torch.device("cuda")

        for layer in self._kv_cache:
            offload_kv_cache_layer(layer)

        memory_usage_before_onload = torch.cuda.memory_allocated() / 1024**2
        memory_usage = [memory_usage_before_onload]

        for layer in self._kv_cache:
            onload_kv_cache_layer(layer, cuda_device)
            memory_usage.append(torch.cuda.memory_allocated() / 1024**2)

        peak_memory_usage = max(memory_usage)
        diff_per_layer = [memory_usage[i + 1] - memory_usage[i] for i in range(len(memory_usage) - 1)]
        per_layer_memory_usage = np.median(diff_per_layer)
        total_kv_cache_memory_usage = per_layer_memory_usage * len(self._kv_cache)

        cprint(
            f"Peak Memory Usage: {peak_memory_usage:.2f} MB | Other Memory Usage: {memory_usage[0]:.2f} MB | "
            f"Per Layer Memory Usage: {per_layer_memory_usage:.2f} MB | Total KV Cache Memory Usage: {total_kv_cache_memory_usage:.2f} MB",
            "light_blue",
        )
        if self._env_flag("HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD", "0"):
            for layer in self._kv_cache:
                offload_kv_cache_layer(layer)
            torch.cuda.empty_cache()

    def _stored_kv_cache_layer_bytes(self, layer):
        return sum(self._stored_kv_cache_bytes(value) for value in layer.values())

    def _stored_kv_cache_bytes(self, value):
        if isinstance(value, ChunkedKVCache):
            total = 0
            for chunk in value.chunks:
                total += self._stored_kv_cache_bytes(chunk)
            for span in value.quantized_spans:
                total += self._stored_kv_cache_bytes(span.get("quant_data"))
            return total
        if torch.is_tensor(value):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(self._stored_kv_cache_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._stored_kv_cache_bytes(item) for item in value)
        return 0

    def _pack_info_into_kv_cache(self, k_cache, v_cache, output_dtype):
        """Pack decompression metadata into real-quantized KV cache dicts."""
        k_cache = self._pack_info_into_single_kv_cache(k_cache, output_dtype)
        v_cache = self._pack_info_into_single_kv_cache(v_cache, output_dtype)
        return k_cache, v_cache

    def _pack_info_into_single_kv_cache(self, cache, output_dtype):
        """Pack decompression metadata into one real-quantized cache dict."""
        cache["info"] = {
            "output_dtype": output_dtype,
            "quant_config": self._quant_config,
        }
        return cache

    def _decompress_kv_cache_layer(self, layer: dict) -> dict:
        """Decompress a single KV cache layer if it contains quantized dicts.

        Returns a new dict with tensor k/v, leaving the original unchanged.
        """
        k, v = layer["k"], layer["v"]
        if isinstance(k, dict) or isinstance(v, dict):
            k = uncompress_single_kv_cache(k)
            v = uncompress_single_kv_cache(v)
        return {"k": k, "v": v}

    def save_full_precision_kv_cache(
        self,
        chunk_i: int,
        token_start: int,
        token_end: int,
        selected_frame_indices: Optional[List[int]],
    ):
        """Save BF16 KV cache tensors for debugging or offline analysis."""
        if not getattr(self, "_save_full_precision_kv_cache", False):
            return
        if not getattr(self, "_kv_cache_save_dir", None):
            return

        end_index = self._kv_cache[0].get("end_index", 0)
        if self._kv_cache_save_mode == "all":
            save_start, save_end = 0, end_index
        else:
            save_start, save_end = token_start, token_end

        if save_start >= save_end:
            return

        rank = int(os.environ.get("RANK", 0))
        save_dir = os.path.join(
            self._kv_cache_save_dir,
            f"rank_{rank:03d}",
            f"chunk_{chunk_i:04d}_{save_start}_{save_end}",
        )
        os.makedirs(save_dir, exist_ok=True)

        cprint(
            f"Saving full-precision KV cache to {save_dir} "
            f"(tokens {save_start}:{save_end})",
            "light_green",
        )

        layer_files = []
        for layer_idx, layer in enumerate(self._kv_cache):
            k_cache: ChunkedKVCache = layer["k"]
            v_cache: ChunkedKVCache = layer["v"]
            k = k_cache.read(save_start, save_end).detach().cpu()
            v = v_cache.read(save_start, save_end).detach().cpu()

            layer_file = f"layer_{layer_idx:02d}.pt"
            torch.save(
                {
                    "k": k,
                    "v": v,
                    "token_start": save_start,
                    "token_end": save_end,
                    "chunk_i": chunk_i,
                    "rank": rank,
                    "layout": "BHSD",
                },
                os.path.join(save_dir, layer_file),
            )
            layer_files.append(
                {
                    "file": layer_file,
                    "layer_idx": layer_idx,
                    "k_shape": list(k.shape),
                    "v_shape": list(v.shape),
                    "dtype": str(k.dtype),
                }
            )
            del k, v

        metadata = {
            "chunk_i": chunk_i,
            "rank": rank,
            "save_mode": self._kv_cache_save_mode,
            "token_start": save_start,
            "token_end": save_end,
            "tokens_per_frame": 880,
            "frame_start": save_start // 880,
            "frame_end": save_end // 880,
            "selected_frame_indices": selected_frame_indices,
            "quant_config": (
                vars(self._quant_config) if self._quant_config is not None else None
            ),
            "note": (
                "mode=new stores freshly computed BF16 spans before quantization; "
                "mode=all stores the current readable cache snapshot."
            ),
            "layers": layer_files,
        }
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def resize_center_crop_to_tensor(self, img_path, target_size):
        """
        Read image -> Resize -> Center crop -> Convert to torch.tensor -> Normalize to [-1, 1]
        :param img_path: Path to the image
        :param target_size: (width, height)
        :return: torch.Tensor, shape [C, H, W], range [-1, 1]
        """
        img = Image.open(img_path).convert("RGB")
        target_w, target_h = target_size
        orig_w, orig_h = img.size

        # --- Step 1: proportional resize ---
        scale = max(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        # --- Step 2: center crop ---
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        img = img.crop((left, top, right, bottom))

        # --- Step 3: to numpy，and then tensor ---
        img_np = np.array(img).astype(np.float32)  # [H, W, C], 0~255
        img_np = img_np / 127.5 - 1.0  # normalize to [-1, 1]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]

        return img_tensor

    def init_decode_state(self, all_latents, chunk_i):
        """
        Initialize the decoding state and set all latents required for decoding.
        """
        self._decode_state["total_latents"] = all_latents.shape[2]  # all latent numbers
        self._decode_state["all_latents"] = all_latents
        self._decode_state["current_latent_idx"] = 0
        self._decode_state["chunk_i"] = chunk_i
        assert self._decode_state["par_vae_decode"]

        if self._decode_state["latents_mean"] is None:
            latents_mean = (
                torch.tensor(self.dist_vae.pipe.config.latents_mean)
                .view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1)
                .to(all_latents.device, all_latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(
                self.dist_vae.pipe.config.latents_std
            ).view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1).to(
                all_latents.device, all_latents.dtype
            )

            self._decode_state["latents_mean"] = latents_mean
            self._decode_state["latents_std"] = latents_std

    def decode_next_latent(self, output_type="np"):
        """
        Decode the next latents and return the decoded frames.
        If all latents have been decoded, return None.
        """

        current_idx = self._decode_state["current_latent_idx"]
        total_latents = self._decode_state["total_latents"]

        if current_idx >= total_latents:
            return None  # all latents have been decoded

        # Get the current latent to be decoded.
        all_latents = self._decode_state["all_latents"]
        chunk_i = self._decode_state["chunk_i"]
        current_latent = all_latents[
            :, :, current_idx : current_idx + 1, :, :
        ]  # [B, C, 1, H, W]

        sp_world_size = self._decode_state["sp_world_size"]
        rank_in_sp_group = self._decode_state["rank_in_sp_group"]
        latents_mean = self._decode_state["latents_mean"]
        latents_std = self._decode_state["latents_std"]

        # Multi-GPU parallel VAE decoding
        current_latent = current_latent.to(self.dist_vae.pipe.dtype)
        # Ensure latents_mean and latents_std have the same type as current_latent
        latents_mean = latents_mean.to(current_latent.dtype)
        latents_std = latents_std.to(current_latent.dtype)
        current_latent = current_latent / latents_std + latents_mean

        local_rank_in_sp_group_vae = self.ctx["local_rank_in_sp_group"]

        decode_input = distribute_data_to_gpus(
            current_latent,
            dim=4,
            rank=rank_in_sp_group,
            local_rank=local_rank_in_sp_group_vae,
            num_gpus=sp_world_size,
            dtype=current_latent.dtype,
        )
        video = self.dist_vae.pipe.decode(
            decode_input,
            return_dict=False,
            is_first_chunk=(current_idx == 0 and chunk_i == 0),
        )
        video = gather_results(
            video[0],
            dim=4,
            world_size=sp_world_size,
        )

        video = self.video_processor.postprocess_video(video, output_type=output_type)

        torch.cuda.synchronize()

        # Update the index
        self._decode_state["current_latent_idx"] += 1

        return video

    def _print_model_memory_report(self):
        """Print parameter count, dtype, and memory usage for each model component."""
        print("=" * 80)
        print("Model Memory Report")
        print("=" * 80)
        for name, model in [
            ("text_encoder", self.text_encoder),
            ("transformer", self.transformer),
            ("transformer_2", self.transformer_2),
            ("vae", self.vae),
        ]:
            if model is None:
                print(f"  {name:20s} | not loaded")
                continue
            num_params = sum(p.numel() for p in model.parameters())
            dtypes = {p.dtype for p in model.parameters()}
            devices = {str(p.device) for p in model.parameters()}
            mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            mem_mb = mem_bytes / 1024**2
            print(
                f"  {name:20s} | params: {num_params / 1e6:>8.2f}M | "
                f"dtype: {dtypes} | device: {devices} | memory: {mem_mb:>10.2f} MB"
            )
        total_mb = sum(
            sum(p.numel() * p.element_size() for p in m.parameters()) / 1024**2
            for _, m in [
                ("text_encoder", self.text_encoder),
                ("transformer", self.transformer),
                ("vae", self.vae),
            ]
            if m is not None
        )
        print(f"  {'TOTAL':20s} | {total_mb:>36.2f} MB")
        print(f"  {'CUDA allocated':20s} | {torch.cuda.memory_allocated() / 1024**2:>36.2f} MB")
        print("=" * 80)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: Optional[float] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        few_step=False,
        first_chunk_size=1,
        image_path=None,
        viewmats=None,
        Ks=None,
        action=None,
        use_memory=True,
        context_window_length=None,
        memory_frames: int = 16,
        temporal_context_size: int = 12,
        pred_latent_size: int = 4,
        offload_text_encoder: bool = False,
        quant_config: Optional[dict] = None,
        save_full_precision_kv_cache: bool = False,
        kv_cache_save_dir: Optional[str] = None,
        kv_cache_save_mode: str = "new",
        kv_quant_skip_chunk_policy: str = "none",
        kv_quant_skip_frame_policy: str = "none",
        kv_quant_skip_first_frames: int = 0,
        kv_quant_skip_last_frames: int = 0,
        kv_quant_recent_policy: str = "none",
        kv_quant_recent_frames: int = 0,
        kv_quant_anchor_frame_stride: int = 0,
        kv_quant_anchor_frame_offset: int = 0,
        kv_sparse_policy: str = "none",
        kv_sparse_first_frames: int = 0,
        kv_sparse_recent_frames: int = 0,
        kv_sparse_quantize_retained: bool = False,
        kv_quant_factor: int = 2,
        kv_quant_tiered_depth_policy: str = "none",
        kv_quant_structured_temporal_policy: str = "none",
        num_chunks: Optional[int] = None,
        chunk_i: int = 0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor`
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """


        if chunk_i == 0:
            self._print_model_memory_report()

            # print("init run")
            # 1. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                negative_prompt,
                height,
                width,
                prompt_embeds,
                negative_prompt_embeds,
                callback_on_step_end_tensor_inputs,
                guidance_scale_2,
            )

            if num_frames % self.vae_scale_factor_temporal != 1:
                logger.warning(
                    "`num_frames - 1` has to be divisible by %d. "
                    "Rounding to the nearest number.",
                    self.vae_scale_factor_temporal,
                )
                num_frames = (
                    num_frames
                    // self.vae_scale_factor_temporal
                    * self.vae_scale_factor_temporal
                    + 1
                )
            num_frames = max(num_frames, 1)

            if self.config.boundary_ratio is not None and guidance_scale_2 is None:
                guidance_scale_2 = guidance_scale

            self._guidance_scale = guidance_scale
            self._guidance_scale_2 = guidance_scale_2
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if self.transformer is not None:
                device = next(self.transformer.parameters()).device
            elif self.transformer_2 is not None:
                device = next(self.transformer_2.parameters()).device
            else:
                device = self._execution_device

            # 2. Define call parameters
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            self.points_local = generate_points_in_sphere(50000, 8.0).to(device)

            # 3. Encode input prompt
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
                device=device,
                offload_text_encoder=offload_text_encoder,
            )

            transformer_dtype = (
                self.transformer.dtype
                if self.transformer is not None
                else self.transformer_2.dtype
            )
            prompt_embeds = prompt_embeds.to(transformer_dtype)
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

            # 4. Prepare timesteps
            if not few_step:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                zeros_t = torch.zeros(1).to(timesteps.device).to(timesteps.dtype)
                timesteps = torch.cat([timesteps, zeros_t])
            else:
                sigmas = self.scheduler.sigmas
                timesteps = torch.tensor(
                    [1000.0000, 960.0000, 888.8889, 727.2728, 0.0]
                ).to(sigmas.device)
                sigmas = timesteps / 1000.0

            # 5. Prepare latent variables
            num_channels_latents = (
                self.transformer.config.in_channels
                if self.transformer is not None
                else self.transformer_2.config.in_channels
            )
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                height,
                width,
                num_frames,
                torch.float32,
                device,
                generator,
                latents,
            )

            self._num_timesteps = len(timesteps)
            chunk_size = CHUNK_SIZE
            n_latent = latents.shape[2]

            # set the noise level for the inference stage
            stabilization_level = 15

            sp_world_size = 1
            rank_in_sp_group = 0
            local_rank_in_sp_group = 0
            if self.parallel_run:
                sp_world_size = get_sp_world_size()
                rank_in_sp_group = get_sp_parallel_rank()
                local_rank_in_sp_group = get_sp_parallel_local_rank()

            first_image_condition = None
            if image_path:
                first_image = self.resize_center_crop_to_tensor(image_path, (width, height)) # (B C T H W)
                first_image = first_image.to(device).to(self.vae.dtype)
                first_image = first_image.unsqueeze(0).unsqueeze(2) # [B C T H W]
                first_image_condition = self.vae.encode(first_image, return_dict=False)[0].sample()
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                first_image_condition = (first_image_condition - latents_mean) * latents_std


            self.use_kv_cache = False
            self.init_kv_cache(batch_size, n_latent, transformer_dtype, device)

            self._decode_state = {
                "current_latent_idx": 0,
                "total_latents": None,
                "device": device,
                "transformer_dtype": transformer_dtype,
                "vae": self.vae,
                "video_processor": self.video_processor,
                "parallel_run": self.parallel_run,
                "par_vae_decode": self.par_vae_decode,
                "sp_world_size": sp_world_size,
                "rank_in_sp_group": rank_in_sp_group,
                "latents_mean": None,
                "latents_std": None,
                "is_initialized": False,
            }


            self.ctx = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "guidance_scale": guidance_scale,
                "guidance_scale_2": guidance_scale_2,
                "attention_kwargs": attention_kwargs,
                "transformer_dtype": (
                    self.transformer.dtype
                    if self.transformer is not None
                    else self.transformer_2.dtype
                ),
                # Denoising scheduler and state
                "timesteps": timesteps,
                "num_inference_steps": num_inference_steps,
                "latents": latents,
                "first_image_condition": first_image_condition,
                "chunk_size": chunk_size,
                "first_chunk_size": first_chunk_size,
                "n_latent": n_latent,
                "viewmats": torch.zeros(
                    1, n_latent, 4, 4, device=latents.device, dtype=latents.dtype
                ),
                "Ks": torch.zeros(
                    1, n_latent, 3, 3, device=latents.device, dtype=latents.dtype
                ),
                "action": torch.zeros(
                    1, n_latent, device=latents.device, dtype=latents.dtype
                ),
                "use_memory": use_memory,
                "context_window_length": context_window_length,
                "memory_frames": memory_frames,
                "temporal_context_size": temporal_context_size,
                "pred_latent_size": pred_latent_size,
                "kv_cache": self._kv_cache,
                "kv_cache_neg": self._kv_cache_neg,
                "use_kv_cache": self.use_kv_cache,
                # Distributed related
                "sp_world_size": sp_world_size,
                "rank_in_sp_group": rank_in_sp_group,
                "local_rank_in_sp_group": local_rank_in_sp_group,
                # Other auxiliary parameters
                "device": device,
                "batch_size": batch_size,
                "few_step": few_step,
                "sigmas": sigmas if few_step else None,
                "stabilization_level": stabilization_level,
                "quant_config": quant_config,
                "save_full_precision_kv_cache": save_full_precision_kv_cache,
                "kv_cache_save_dir": kv_cache_save_dir,
                "kv_cache_save_mode": kv_cache_save_mode,
                "kv_quant_skip_chunk_policy": kv_quant_skip_chunk_policy,
                "kv_quant_skip_frame_policy": kv_quant_skip_frame_policy,
                "kv_quant_skip_first_frames": int(kv_quant_skip_first_frames),
                "kv_quant_skip_last_frames": int(kv_quant_skip_last_frames),
                "kv_quant_recent_policy": kv_quant_recent_policy,
                "kv_quant_recent_frames": int(kv_quant_recent_frames),
                "kv_quant_anchor_frame_stride": int(kv_quant_anchor_frame_stride),
                "kv_quant_anchor_frame_offset": int(kv_quant_anchor_frame_offset),
                "kv_sparse_policy": kv_sparse_policy,
                "kv_sparse_first_frames": int(kv_sparse_first_frames),
                "kv_sparse_recent_frames": int(kv_sparse_recent_frames),
                "kv_sparse_quantize_retained": bool(kv_sparse_quantize_retained),
                "kv_quant_factor": int(kv_quant_factor),
                "kv_quant_tiered_depth_policy": normalize_tiered_depth_policy(
                    kv_quant_tiered_depth_policy
                ),
                "kv_quant_structured_temporal_policy": (
                    normalize_structured_temporal_policy(
                        kv_quant_structured_temporal_policy
                    )
                ),
                "num_chunks": (
                    int(num_chunks) if num_chunks is not None else None
                ),
                "kv_recent_quantized_frame_end": 0,
                "rvq_feedback_state": {},
            }
            self._quant_config = SimpleNamespace(**quant_config) if quant_config is not None else None
            self._rvq_feedback_state = self.ctx["rvq_feedback_state"]
            self._validate_tiered_depth_configuration(chunk_size=chunk_size)
            self._validate_structured_temporal_configuration(
                chunk_size=chunk_size
            )
            self._validate_temporal_adaptive_residual_configuration(
                chunk_size=chunk_size
            )
            self._prewarm_kv_quantization()

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # ==========================================================
        if self.ctx is None and chunk_i > 0:
            raise ValueError(
                "When chunk_i > 0, the context self.ctx cannot be empty. Please ensure the call starts from chunk_i = 0."
            )
        prompt_embeds = self.ctx["prompt_embeds"]
        negative_prompt_embeds = self.ctx["negative_prompt_embeds"]
        guidance_scale = self.ctx["guidance_scale"]
        guidance_scale_2 = self.ctx["guidance_scale_2"]
        attention_kwargs = self.ctx["attention_kwargs"]
        transformer_dtype = self.ctx["transformer_dtype"]

        timesteps = self.ctx["timesteps"]
        num_inference_steps = self.ctx["num_inference_steps"]
        latents = self.ctx["latents"]
        first_image_condition = self.ctx["first_image_condition"]

        chunk_size = self.ctx["chunk_size"]
        first_chunk_size = self.ctx["first_chunk_size"]
        n_latent = self.ctx["n_latent"]
        stabilization_level = self.ctx["stabilization_level"]

        # Fill the poses of the current chunk into the corresponding positions.
        start_idx = chunk_i * chunk_size
        end_idx = start_idx + chunk_size

        self.ctx["viewmats"][:, start_idx:end_idx, ...] = viewmats
        self.ctx["Ks"][:, start_idx:end_idx, ...] = Ks
        self.ctx["action"][:, start_idx:end_idx, ...] = action

        viewmats = self.ctx["viewmats"]
        Ks = self.ctx["Ks"]
        action = self.ctx["action"]

        use_memory = self.ctx["use_memory"]
        context_window_length = self.ctx["context_window_length"]
        memory_frames = self.ctx["memory_frames"]
        temporal_context_size = self.ctx["temporal_context_size"]
        pred_latent_size = self.ctx["pred_latent_size"]
        self._kv_cache = self.ctx["kv_cache"]
        self._kv_cache_neg = self.ctx["kv_cache_neg"]
        self.use_kv_cache = self.ctx["use_kv_cache"]

        sp_world_size_vae = self.ctx["sp_world_size"]
        rank_in_sp_group_vae = self.ctx["rank_in_sp_group"]
        local_rank_in_sp_group_vae = self.ctx["local_rank_in_sp_group"]

        device = self.ctx["device"]
        batch_size = self.ctx["batch_size"]
        few_step = self.ctx["few_step"]
        sigmas = self.ctx["sigmas"]

        quant_config_dict = self.ctx.get("quant_config", None)
        if quant_config_dict is not None:
            self._quant_config = SimpleNamespace(**quant_config_dict)
        else:
            self._quant_config = None
        self._rvq_feedback_state = self.ctx.setdefault("rvq_feedback_state", {})
        self._configure_rvq_feedback_query_capture()
        self._save_full_precision_kv_cache = self.ctx.get(
            "save_full_precision_kv_cache", False
        )
        self._kv_cache_save_dir = self.ctx.get("kv_cache_save_dir", None)
        self._kv_cache_save_mode = self.ctx.get("kv_cache_save_mode", "new")
        reset_dequant_stats()

        # Restore the attributes of the pipeline
        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._num_timesteps = len(timesteps)

        selected_frame_indices = None
        if chunk_i == 0:
            already_generate_num = first_chunk_size
            generate_latent_num = first_chunk_size
            if first_image_condition is not None:
                latents[:, :, :1] = first_image_condition
            latents_curr = latents[:, :, :already_generate_num].to(device=device, dtype=prompt_embeds.dtype)
        else:
            if not few_step:
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                timesteps = self.scheduler.timesteps
                zeros_t = torch.zeros(1).to(timesteps.device).to(timesteps.dtype)
                timesteps = torch.cat([timesteps, zeros_t])

            already_generate_num = chunk_i * chunk_size + first_chunk_size
            latents_curr = latents[:, :, :already_generate_num].to(device=device, dtype=prompt_embeds.dtype)
            generate_latent_num = chunk_size

            # add memory selection here
            current_frame_idx = chunk_i * chunk_size  # the current frame index to generate
            if (
                use_memory
                and (current_frame_idx >= context_window_length)
                and current_frame_idx < n_latent
            ):
                selected_frame_indices = select_mem_frames_wan(
                    viewmats[0].cpu().detach().numpy(),
                    current_frame_idx,
                    memory_frames=memory_frames,
                    temporal_context_size=temporal_context_size,
                    pred_latent_size=pred_latent_size,
                    points_local=self.points_local,
                    device=device,
                )

            elif use_memory:
                selected_frame_indices = list(range(0, current_frame_idx))

            sparse_selected_frame_indices = self._sparse_selected_frame_indices(
                current_frame_idx,
                chunk_size,
            )
            if sparse_selected_frame_indices is not None:
                selected_frame_indices = sparse_selected_frame_indices
            else:
                # All frame indices must be contiguous so that old quantized
                # tokens remain valid when we only quantize the new part.
                assert selected_frame_indices == list(range(len(selected_frame_indices))), (
                    f"selected_frame_indices must be contiguous starting from 0, "
                    f"got {selected_frame_indices}"
                )

        cprint(f"selected_frame_indices: {selected_frame_indices}", "magenta")

        # print memory usage in MB
        cprint(f"memory usage: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB", "cyan")

        with self.progress_bar(total=4) as progress_bar:
            for i, t in enumerate(timesteps[:-1]):
                if self.interrupt:
                    continue

                self._current_timestep = t

                current_model = self.transformer
                current_guidance_scale = guidance_scale

                if chunk_i > 0:  # context latent frame
                    t_now = torch.full((batch_size, chunk_size), t, device=device, dtype=timesteps.dtype)
                    t_ctx = torch.full((batch_size, first_chunk_size + (chunk_i - 1) * chunk_size),
                        stabilization_level - 1,
                        device=device,
                        dtype=timesteps.dtype,
                    )  # [B, F]
                    timestep = torch.cat([t_ctx, t_now], dim=1)  # [B, F+1]: predict the next frame in timestep t
                else:
                    if first_image_condition is not None:
                        t_now = torch.full((batch_size, chunk_size - 1), t, device=device, dtype=timesteps.dtype)
                        t_ctx = torch.full((batch_size, 1), stabilization_level - 1,
                            device=device,
                            dtype=timesteps.dtype,
                        )  # [B, F]
                        timestep = torch.cat([t_ctx, t_now], dim=1)
                    else:
                        timestep = torch.full((batch_size, first_chunk_size), t,
                            device=device,
                            dtype=timesteps.dtype,
                        )  # [B, 1]

                all_window_latent_num = latents_curr.shape[2]

                ## kv cache
                if chunk_i > 0 and i == 0 and self.use_kv_cache:
                    sparse_kv_policy = self._use_sparse_kv_policy()
                    if sparse_kv_policy:
                        cprint(
                            "Sparse KV policy rebuilding cache from selected frames; "
                            "dropped middle frames are not stored",
                            "light_cyan",
                        )
                        self._clear_kv_cache_for_sparse_policy()

                    # Print Using kv cache
                    latents_cache = latents_curr.clone()
                    latents_cache = latents_cache[:, :, selected_frame_indices]
                    t_cache = timestep[:, selected_frame_indices]

                    latent_model_cache = latents_cache.to(transformer_dtype)
                    # flatten the timestep for the transformer
                    timestep_cache = t_cache.flatten()
                    if action is not None:
                        action_cache = action[:, selected_frame_indices]
                        viewmats_cache = viewmats[:, selected_frame_indices]
                        Ks_cache = Ks[:, selected_frame_indices]

                    kv_start_rope = 0
                    kv_end_rope = len(selected_frame_indices)
                    kv_cache_token_start = self._kv_cache[0].get("end_index", 0)

                    cprint("Calculating kv cache", "green")

                    torch.cuda.nvtx.range_push("kv_cache")
                    current_model(
                        hidden_states=latent_model_cache,
                        timestep=timestep_cache,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        current_start=kv_start_rope * 880,
                        current_end=kv_end_rope * 880,
                        kv_cache=self._kv_cache,
                        is_cache=True,
                        window_frames=n_latent,
                        viewmats=(viewmats_cache if viewmats is not None else None),
                        Ks=Ks_cache if Ks is not None else None,
                        action=action_cache if action is not None else None,
                        context_frames_list=selected_frame_indices if sparse_kv_policy else None,
                    )
                    torch.cuda.nvtx.range_pop()

                    end_index = self._kv_cache[0]["end_index"]
                    cprint(f"KV cache tokens: {end_index} ({end_index // 880} frames)", "magenta")
                    self.save_full_precision_kv_cache(
                        chunk_i=chunk_i,
                        token_start=kv_cache_token_start,
                        token_end=end_index,
                        selected_frame_indices=selected_frame_indices,
                    )

                    if not sparse_kv_policy:
                        self.maybe_quantize_kv_cache_with_policy(
                            chunk_i=chunk_i,
                            chunk_size=chunk_size,
                            n_latent=n_latent,
                            num_chunks=int(num_chunks) if num_chunks is not None else chunk_i + 1,
                            token_start=kv_cache_token_start,
                            end_index=end_index,
                        )
                    elif bool(self.ctx.get("kv_sparse_quantize_retained", False)):
                        cprint(
                            "Sparse KV policy quantizing retained first/recent cache entries",
                            "light_cyan",
                        )
                        self.quantize_kv_cache(0, end_index)
                    else:
                        self._print_memory_usage()

                # Print the memory usage after the kv cache
                cprint(f"memory usage after kv cache: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB", "cyan")

                if not self.use_kv_cache:
                    generate_latent_num = all_window_latent_num

                if selected_frame_indices is not None:
                    now_window_latent_num = len(selected_frame_indices) + chunk_size
                else:
                    now_window_latent_num = chunk_size

                latent_model_input = latents_curr.clone()
                latent_model_input = latent_model_input[:, :, -generate_latent_num:].to(transformer_dtype)
                timestep = timestep[:, -generate_latent_num:]
                timestep = timestep.flatten()

                if self._use_sparse_kv_policy():
                    generate_rope_start = (all_window_latent_num - generate_latent_num) * 880
                    generate_rope_end = all_window_latent_num * 880
                else:
                    generate_rope_start = (now_window_latent_num - generate_latent_num) * 880
                    generate_rope_end = now_window_latent_num * 880

                cprint("Calculating noise prediction", "green")

                torch.cuda.nvtx.range_push("noise_pred")
                cache_viewmats_for_reconstruct = (
                    viewmats[:, selected_frame_indices]
                    if selected_frame_indices is not None
                    else None
                )
                cache_Ks_for_reconstruct = (
                    Ks[:, selected_frame_indices]
                    if selected_frame_indices is not None
                    else None
                )
                noise_pred = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    current_start=generate_rope_start,
                    current_end=generate_rope_end,
                    kv_cache=self._kv_cache if self.use_kv_cache else None,
                    is_cache=False,
                    window_frames=n_latent,
                    viewmats=viewmats[:, all_window_latent_num - generate_latent_num: all_window_latent_num],
                    Ks=Ks[:, all_window_latent_num - generate_latent_num: all_window_latent_num],
                    cache_viewmats=cache_viewmats_for_reconstruct,
                    cache_Ks=cache_Ks_for_reconstruct,
                    action=action[:, all_window_latent_num - generate_latent_num: all_window_latent_num],
                    context_frames_list=selected_frame_indices if self._use_sparse_kv_policy() else None,
                )[0]

                torch.cuda.nvtx.range_pop()

                # compute the previous noisy sample x_t -> x_t-1
                if chunk_i == 0:
                    if not few_step:
                        latent_model_input = self.scheduler.step( noise_pred, t, latent_model_input, return_dict=False)[0]
                        if first_image_condition is not None:
                            latents_curr[:, :, -first_chunk_size + 1:] = latent_model_input[:, :, 1:]
                        else:
                            latents_curr[:, :, -first_chunk_size:] = latent_model_input
                    else:
                        sigma = sigmas[i]
                        sigma_next = sigmas[i + 1]
                        dt = sigma_next - sigma
                        prev_sample = latent_model_input + dt * noise_pred
                        if first_image_condition is not None:
                            latents_curr[:, :, -first_chunk_size + 1:] = prev_sample[:, :, 1:]
                        else:
                            latents_curr[:, :, -first_chunk_size:] = prev_sample
                else:
                    noise_pred_chunk = noise_pred[:, :, -chunk_size:]  # [B, C, chunk_size, H, W
                    latents_curr_chunk = latent_model_input[:, :, -chunk_size:]  # [B, C, chunk_size, H, W]
                    if not few_step:
                        latent_curr_pred_chunk = self.scheduler.step(
                            noise_pred_chunk,
                            t,
                            latents_curr_chunk,
                            return_dict=False,
                        )[0]  # [B, C, chunk_size, H, W]
                    else:
                        sigma = sigmas[i]
                        sigma_next = sigmas[i + 1]
                        dt = sigma_next - sigma
                        latent_curr_pred_chunk = latents_curr_chunk + dt * noise_pred_chunk

                    latents_curr[:, :, -chunk_size:] = latent_curr_pred_chunk  # update the latents

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        dequant_stats = get_dequant_stats(reset=True)
        if self._structured_temporal_policy() != "none":
            temporal_audit = self.ctx.get(
                "kv_quant_structured_temporal_audit"
            )
            # Chunk zero has no history and therefore precedes creation of the
            # temporal audit.  Every later chunk must prove that all packed
            # structured reconstructions used the explicitly selected fused
            # backend, with no eager fallback.
            if isinstance(temporal_audit, dict):
                fast_calls = int(
                    dequant_stats.get("structured_fast_calls", 0)
                )
                eager_calls = int(
                    dequant_stats.get("structured_eager_calls", 0)
                )
                total_calls = int(dequant_stats["calls"])
                decoder_event = {
                    "chunk_i": int(chunk_i),
                    "total_dequant_calls": total_calls,
                    "structured_fast_calls": fast_calls,
                    "structured_eager_calls": eager_calls,
                    "structured_fast_backend": dequant_stats.get(
                        "structured_fast_backend"
                    ),
                    "seconds": float(dequant_stats["seconds"]),
                    "all_calls_used_fail_closed_fused_fwht": bool(
                        fast_calls == total_calls and eager_calls == 0
                    ),
                }
                temporal_audit.setdefault(
                    "dequantization_events", []
                ).append(decoder_event)
                if not decoder_event[
                    "all_calls_used_fail_closed_fused_fwht"
                ]:
                    raise RuntimeError(
                        "structured temporal decoder mixed or bypassed the "
                        f"fused FWHT path: {decoder_event}"
                    )
                if int(chunk_i) == (
                    _STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS - 1
                ):
                    events = temporal_audit["dequantization_events"]
                    expected_chunks = list(
                        range(1, _STRUCTURED_TEMPORAL_FORMAL_NUM_CHUNKS)
                    )
                    observed_chunks = [
                        int(event["chunk_i"]) for event in events
                    ]
                    decoder_verification = {
                        "expected_chunks": expected_chunks,
                        "observed_chunks": observed_chunks,
                        "chunks_exact": observed_chunks == expected_chunks,
                        "all_events_fused_only": all(
                            bool(
                                event[
                                    "all_calls_used_fail_closed_fused_fwht"
                                ]
                            )
                            for event in events
                        ),
                        "nonzero_fast_calls_after_first_history_chunk": all(
                            int(event["structured_fast_calls"]) > 0
                            for event in events[1:]
                        ),
                        "backend": "triton_fwht_g128",
                    }
                    if not all(
                        (
                            decoder_verification["chunks_exact"],
                            decoder_verification["all_events_fused_only"],
                            decoder_verification[
                                "nonzero_fast_calls_after_first_history_chunk"
                            ],
                        )
                    ):
                        raise RuntimeError(
                            "structured temporal fused decoder verification "
                            f"failed: {decoder_verification}"
                        )
                    temporal_audit[
                        "decoder_verification"
                    ] = decoder_verification
        if (
            os.getenv("HYWORLDPLAY_RVQ_DIAG_POST_CHUNK_CACHE", "0").lower()
            not in {"0", "false", "no", ""}
            and self.use_kv_cache
        ):
            TOKENS_PER_FRAME = 880
            cache_frame_start = chunk_i * chunk_size
            cache_frame_end = min(cache_frame_start + chunk_size, n_latent)
            cache_token_start = cache_frame_start * TOKENS_PER_FRAME
            cache_token_end = cache_frame_end * TOKENS_PER_FRAME
            current_cache_end = self._kv_cache[0].get("end_index", 0)
            is_final_cache_span = cache_frame_end >= n_latent
            if (
                is_final_cache_span
                and cache_token_start < cache_token_end
                and current_cache_end < cache_token_end
            ):
                if current_cache_end != cache_token_start:
                    raise ValueError(
                        "RVQ diagnostic cache dump expects contiguous cache writes; "
                        f"got current_cache_end={current_cache_end}, "
                        f"cache_token_start={cache_token_start}."
                    )

                cprint(
                    f"RVQ diagnostic cache-only dump for frames "
                    f"{cache_frame_start}:{cache_frame_end}",
                    "light_green",
                )
                latent_model_cache = latents_curr[
                    :, :, cache_frame_start:cache_frame_end
                ].to(device=device, dtype=transformer_dtype)
                timestep_cache = torch.full(
                    (batch_size, cache_frame_end - cache_frame_start),
                    stabilization_level - 1,
                    device=device,
                    dtype=timesteps.dtype,
                ).flatten()

                for layer in self._kv_cache:
                    layer["_rvq_diag_token_offset"] = cache_token_start
                    layer["_rvq_diag_dump_only"] = True

                try:
                    current_model(
                        hidden_states=latent_model_cache,
                        timestep=timestep_cache,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        current_start=cache_token_start,
                        current_end=cache_token_end,
                        kv_cache=self._kv_cache,
                        is_cache=True,
                        window_frames=n_latent,
                        viewmats=(
                            viewmats[:, cache_frame_start:cache_frame_end]
                            if viewmats is not None
                            else None
                        ),
                        Ks=(
                            Ks[:, cache_frame_start:cache_frame_end]
                            if Ks is not None
                            else None
                        ),
                        action=(
                            action[:, cache_frame_start:cache_frame_end]
                            if action is not None
                            else None
                        ),
                    )
                finally:
                    for layer in self._kv_cache:
                        layer.pop("_rvq_diag_token_offset", None)
                        layer.pop("_rvq_diag_dump_only", None)
                    del latent_model_cache, timestep_cache

        cprint(
            f"KV Cache Dequantization Time for chunk {chunk_i}: "
            f"{dequant_stats['seconds']:.4f} s over {dequant_stats['calls']} calls",
            "light_cyan",
        )

        latents[:, :, :already_generate_num, :, :] = latents_curr

        # ==========================================================
        # Save the state updated during this call back to self.ctx
        self.ctx["latents"] = latents
        self.ctx["kv_cache"] = self._kv_cache
        self.ctx["kv_cache_neg"] = self._kv_cache_neg
        # ==========================================================

        re_latents = latents
        self._current_timestep = None
        latents = latents[:, :, start_idx:end_idx, :, :]

        # Automatically initialize the decoding state (for subsequent frame-by-frame decoding)
        # Note: Must be initialized here because the latents will be modified later
        if output_type == "latent":
            self.init_decode_state(latents, chunk_i)

        torch.cuda.synchronize()
        vae_decode_begin_time = time.time()
        if not output_type == "latent":
            if not self.par_vae_decode:
                latents = latents.to(self.vae.dtype)
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(latents.device, latents.dtype)
                latents = latents / latents_std + latents_mean
                video = self.vae.decode(
                    latents, return_dict=False, is_first_chunk=chunk_i == 0
                )[0]
            else:
                latents = latents.to(self.dist_vae.pipe.dtype)
                latents_mean = (
                    torch.tensor(self.dist_vae.pipe.config.latents_mean)
                    .view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1)
                    .to(latents.device, latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(
                    self.dist_vae.pipe.config.latents_std
                ).view(1, self.dist_vae.pipe.config.z_dim, 1, 1, 1).to(
                    latents.device, latents.dtype
                )
                latents = latents / latents_std + latents_mean

                # Accelerated VAE
                decode_input = distribute_data_to_gpus(
                    latents,
                    dim=4,
                    rank=rank_in_sp_group_vae,
                    local_rank=local_rank_in_sp_group_vae,
                    num_gpus=sp_world_size_vae,
                    dtype=latents.dtype,
                )
                video = self.dist_vae.pipe.decode(
                    decode_input, return_dict=False, is_first_chunk=chunk_i == 0
                )
                video = gather_results(
                    video[0],
                    dim=4,
                    world_size=sp_world_size_vae,
                )
            video = self.video_processor.postprocess_video(
                video, output_type=output_type
            )
        else:
            return None, None

        torch.cuda.synchronize()
        print("Vae decode time", time.time() - vae_decode_begin_time)
        torch.cuda.nvtx.range_pop()

        if not return_dict:
            return (
                video,
                re_latents,
            )

        return WanPipelineOutput(frames=video)
