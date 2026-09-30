from typing import List, Optional
from contextlib import nullcontext
import json
import torch
import os
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import (
    gpu,
    get_cuda_free_memory_gb,
    move_model_to_device_with_memory_preservation,
)

from types import SimpleNamespace
from termcolor import cprint

from subitkv.compress import get_quantize_fn, compress_kv_cache
from subitkv.chunk_policy import completed_chunk_compression_range
from subitkv.kv_cache import ChunkedKVCache, offload_kv_cache_layer, onload_kv_cache_layer
from subitkv.rvq import rvq_codebook_directory_metadata
from subitkv.rng import (
    compression_uses_online_rng,
    preserve_torch_rng_state,
)
from subitkv.temporal_adaptive_residual import (
    load_profile as load_temporal_profile,
    schedule_precisions as schedule_temporal_precisions,
    transition_state as transition_temporal_state,
    validate_state as validate_temporal_state,
)
from subitkv.uncompress import (
    get_dequant_stats,
    reset_dequant_stats,
    uncompress_single_kv_cache,
)

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        # Environment variable for dumping KV cache
        self.dump_kv_level = int(os.getenv("DUMP_KV_LEVEL", "0"))  # 0: off, 1: final, 2: every block
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            
        # Quant Config using SimpleNamespace
        self.quant_config = SimpleNamespace(**args.quant_config)

        self.generator.model.kv_cache_cpu_offload = getattr(self.quant_config, "kv_cache_cpu_offload", False)
        self.runtime_metrics = {}
        self._quantized_frame_ranges: list[tuple[int, int]] = []
        self._encode_seconds = 0.0
        self._encode_calls = 0
        self._decode_seconds = 0.0
        self._decode_calls = 0
        self._temporal_profile_cache = None
        self._temporal_schedules: list = []
        self._peak_kv_payload_bytes = 0
        self._recon_audit_enabled = os.getenv("SELF_FORCING_RECON_AUDIT", "0") == "1"
        self._recon_audit_rows: list[dict] = []


    def quantize_kv_cache(self, tokens_to_quantize_start: int, tokens_to_quantize_end: int, max_tokens_to_quantize: int):
        """
        Quantize a range of the KV cache.  Indices are in token space and
        must be frame-aligned.
        """
        # Check the inputs
        assert tokens_to_quantize_start <= max_tokens_to_quantize and tokens_to_quantize_end <= max_tokens_to_quantize
        cprint(
            f"Quantizing {tokens_to_quantize_end - tokens_to_quantize_start} tokens, "
            f"from {tokens_to_quantize_start} to {tokens_to_quantize_end} tokens. "
            f"The max tokens to quantize is {max_tokens_to_quantize}",
            "light_cyan",
        )

        # Record the time using torch.cuda.Event
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # Do nothing if quantization type is none
        if self.quant_config.quant_type == "none":
            self._print_memory_usage(self.kv_cache1)
            return

        cprint(f"Quantizing kv cache with config:\n{self.quant_config}", "light_blue")

        do_offload = getattr(self.quant_config, "kv_cache_cpu_offload", False)
        cuda_device = torch.device("cuda")

        # Online K-means samples its initial centroids from the global CUDA
        # generator. Self-Forcing also uses that generator for three denoising
        # noise draws per chunk. Isolate randomized compression so changing the
        # cache codec cannot silently change the model's future noise trajectory.
        rng_context = (
            preserve_torch_rng_state(cuda_device)
            if compression_uses_online_rng(self.quant_config)
            else nullcontext()
        )
        with torch.no_grad(), rng_context:
            quantize_fn = get_quantize_fn(self.quant_config.quant_type, self.quant_config)

            for layer_idx, layer in enumerate(self.kv_cache1):
                if do_offload:
                    onload_kv_cache_layer(layer, cuda_device)

                # Read the span to quantize — always full precision [B, S, H, D]
                k = layer["k"].read(tokens_to_quantize_start, tokens_to_quantize_end)
                v = layer["v"].read(tokens_to_quantize_start, tokens_to_quantize_end)

                # compress_kv_cache expects [B, H, S, D]
                k = k.permute(0, 2, 1, 3).contiguous()
                v = v.permute(0, 2, 1, 3).contiguous()

                k_quant, v_quant = compress_kv_cache(
                    k,
                    v,
                    self.quant_config.quant_type,
                    self.quant_config,
                    quantize_fn,
                    layer_idx=layer_idx,
                )

                self._print_kv_cache_mse_error(k, k_quant, v, v_quant, layer_idx)

                # Pack decompression metadata for real-quantized dicts
                if isinstance(k_quant, dict) and isinstance(v_quant, dict):
                    k_quant, v_quant = self._pack_info_into_kv_cache(
                        k_quant, v_quant, k.dtype
                    )
                if self._recon_audit_enabled:
                    self._audit_kv_reconstruction(
                        k,
                        k_quant,
                        v,
                        v_quant,
                        layer_idx,
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                    )
                    self._audit_kv_attention(
                        layer,
                        k,
                        k_quant,
                        v,
                        v_quant,
                        layer_idx,
                        tokens_to_quantize_start,
                        tokens_to_quantize_end,
                    )
                    layer.pop("recon_audit_q", None)
                    layer.pop("recon_audit_start", None)
                    layer.pop("recon_audit_position", None)
                    layer.pop("recon_audit_grid_sizes", None)
                    layer.pop("recon_audit_freqs", None)

                # store_quantized handles both tensor (fake) and dict (real)
                layer["k"].store_quantized(
                    tokens_to_quantize_start, tokens_to_quantize_end, k_quant
                )
                layer["v"].store_quantized(
                    tokens_to_quantize_start, tokens_to_quantize_end, v_quant
                )

                if do_offload:
                    offload_kv_cache_layer(layer)

        end_time.record()
        torch.cuda.synchronize()
        duration = start_time.elapsed_time(end_time)
        self._encode_seconds += duration / 1000.0
        self._encode_calls += 1
        cprint(f"Quantization KV Cache Time: {(duration / 1000):.2f} s", "light_cyan")
        self._quantized_frame_ranges.append(
            (
                tokens_to_quantize_start // self.frame_seq_length,
                tokens_to_quantize_end // self.frame_seq_length,
            )
        )

    def _temporal_residual_enabled(self) -> bool:
        return bool(
            getattr(self.quant_config, "temporal_adaptive_residual", False)
        )

    def _temporal_profile(self):
        if self._temporal_profile_cache is None:
            path = str(
                getattr(
                    self.quant_config,
                    "temporal_adaptive_profile_path",
                    "",
                )
                or ""
            )
            if not path:
                raise ValueError(
                    "temporal_adaptive_profile_path is required for "
                    "time-adaptive residual compression"
                )
            self._temporal_profile_cache = load_temporal_profile(path)
        return self._temporal_profile_cache

    def _record_persistent_kv_payload(self) -> None:
        payload = self._kv_cache_payload_bytes()
        self._peak_kv_payload_bytes = max(
            self._peak_kv_payload_bytes,
            int(payload["K"] + payload["V"]),
        )

    def _quantize_temporal_residual_chunk(
        self,
        start_frame: int,
        end_frame: int,
        max_tokens: int,
    ) -> None:
        """Encode one 3-frame chunk and apply a frozen monotone bit schedule."""

        frames_per_chunk = int(self.num_frame_per_block)
        if (
            end_frame - start_frame != frames_per_chunk
            or start_frame % frames_per_chunk
        ):
            raise ValueError(
                "progressive Self-Forcing compression requires one aligned "
                f"{frames_per_chunk}-frame chunk"
            )
        chunk_index = start_frame // frames_per_chunk
        if not 0 <= chunk_index < 10:
            raise ValueError("the formal progressive benchmark uses 10 chunks")
        first_cache = self.kv_cache1[0]["k"]
        current: list[int] = []
        for index in range(chunk_index):
            span_start = index * frames_per_chunk
            span_end = span_start + frames_per_chunk
            state = first_cache.quantized_span_data(span_start, span_end)
            validate_temporal_state(state)
            current.append(int(state["bits"]))

        uniform_bits = int(
            getattr(self.quant_config, "temporal_uniform_bits", 0) or 0
        )
        uniform_k_bits = int(
            getattr(self.quant_config, "temporal_uniform_k_bits", 0) or 0
        )
        uniform_v_bits = int(
            getattr(self.quant_config, "temporal_uniform_v_bits", 0) or 0
        )
        split_uniform = bool(uniform_k_bits or uniform_v_bits)
        if split_uniform:
            if uniform_bits or not (uniform_k_bits and uniform_v_bits):
                raise ValueError(
                    "tensor-specific uniform precision requires both K/V bits "
                    "and is incompatible with temporal_uniform_bits"
                )
            if uniform_k_bits not in {1, 2, 3, 4, 8} or uniform_v_bits not in {
                1, 2, 3, 4, 8
            }:
                raise ValueError("tensor-specific uniform bits must be 1/2/3/4/8")
            targets = {
                "k": [uniform_k_bits] * (chunk_index + 1),
                "v": [uniform_v_bits] * (chunk_index + 1),
            }
            target = targets["k"]
        elif uniform_bits:
            if uniform_bits not in {1, 2, 3, 4, 8}:
                raise ValueError("temporal_uniform_bits must be 1/2/3/4/8")
            target = [uniform_bits] * (chunk_index + 1)
            targets = {"k": target, "v": target}
        else:
            capacity = int(
                getattr(
                    self.quant_config,
                    "temporal_adaptive_capacity_bit_chunks",
                    40,
                )
            )
            target = list(
                schedule_temporal_precisions(
                    [*current, 16],
                    self._temporal_profile(),
                    capacity_bit_chunks=capacity,
                    allocation_mode=str(
                        getattr(
                            self.quant_config,
                            "temporal_adaptive_allocation_mode",
                            "optimal",
                        )
                        or "optimal"
                    ),
                )
            )
            targets = {"k": target, "v": target}

        torch.cuda.synchronize()
        started = time.perf_counter()
        encode_before = self._encode_seconds
        admission_attributes = (
            (
                "temporal_adaptive_k_admission_target_bits",
                int(targets["k"][-1]),
            ),
            (
                "temporal_adaptive_v_admission_target_bits",
                int(targets["v"][-1]),
            ),
        )
        previous_admission = {
            name: (
                hasattr(self.quant_config, name),
                getattr(self.quant_config, name, None),
            )
            for name, _value in admission_attributes
        }
        for name, value in admission_attributes:
            setattr(self.quant_config, name, value)
        try:
            self.quantize_kv_cache(
                start_frame * self.frame_seq_length,
                end_frame * self.frame_seq_length,
                max_tokens,
            )
        finally:
            for name, _value in admission_attributes:
                existed, previous = previous_admission[name]
                if existed:
                    setattr(self.quant_config, name, previous)
                else:
                    delattr(self.quant_config, name)

        for layer in self.kv_cache1:
            for name in ("k", "v"):
                cache = layer[name]
                for index, bits in enumerate(targets[name]):
                    span_start = index * frames_per_chunk
                    span_end = span_start + frames_per_chunk
                    state = cache.quantized_span_data(span_start, span_end)
                    if int(state["bits"]) == int(bits):
                        continue
                    transitioned = transition_temporal_state(
                        state,
                        int(bits),
                        quant_config=self.quant_config,
                        verify_identity=False,
                    )
                    cache.replace_quantized_span_data(
                        span_start,
                        span_end,
                        transitioned,
                    )
        torch.cuda.synchronize()
        # Replace the admission-only CUDA-event interval with the complete
        # scheduling, admission and physical-prefix transition interval.
        self._encode_seconds = encode_before + (time.perf_counter() - started)
        if split_uniform:
            self._temporal_schedules.append(
                {
                    "K": [int(item) for item in targets["k"]],
                    "V": [int(item) for item in targets["v"]],
                }
            )
        else:
            self._temporal_schedules.append([int(item) for item in target])
        self._record_persistent_kv_payload()

    
    def _print_kv_cache_mse_error(self, k, k_quant, v, v_quant, layer_idx):
        """Print the Rel L2 error between the original and quantized KV cache."""

        if isinstance(k_quant, dict) and isinstance(v_quant, dict):
            return

        k_rel_l2 = torch.norm(k - k_quant, p=2) / torch.norm(k, p=2)
        v_rel_l2 = torch.norm(v - v_quant, p=2) / torch.norm(v, p=2)
        cprint(f"Layer {layer_idx:02d} | K Rel L2: {k_rel_l2:.4f} | V Rel L2: {v_rel_l2:.4f}", "light_blue")

    def _audit_kv_reconstruction(
        self,
        k,
        k_quant,
        v,
        v_quant,
        layer_idx,
        token_start,
        token_end,
    ):
        """Record exact encode/decode error for an analysis-only run."""

        for tensor_name, reference, packed in (
            ("K", k, k_quant),
            ("V", v, v_quant),
        ):
            reconstructed = uncompress_single_kv_cache(packed)
            reference_float = reference.float()
            error = reference_float - reconstructed.float()
            error_sq = error.square()
            source_sq = reference_float.square()
            vector_error_sq = error_sq.sum(dim=-1)
            vector_source_sq = source_sq.sum(dim=-1).clamp_min(1e-20)
            self._recon_audit_rows.append({
                "tensor": tensor_name,
                "layer": int(layer_idx),
                "token_start": int(token_start),
                "token_end": int(token_end),
                "element_count": int(reference.numel()),
                "vector_count": int(reference.numel() // reference.shape[-1]),
                "squared_error_sum": float(error_sq.sum().item()),
                "source_squared_sum": float(source_sq.sum().item()),
                "vector_relative_squared_error_sum": float(
                    (vector_error_sq / vector_source_sq).sum().item()
                ),
            })
            del reconstructed, reference_float, error, error_sq, source_sq
            del vector_error_sq, vector_source_sq

    def _audit_kv_attention(
        self,
        layer,
        k,
        k_quant,
        v,
        v_quant,
        layer_idx,
        token_start,
        token_end,
    ):
        """Measure chunk-local QK/KL and role-isolated attention output errors."""

        q = layer.get("recon_audit_q")
        q_start = int(layer.get("recon_audit_start", -1))
        if not torch.is_tensor(q) or q_start != int(token_start):
            return
        q = q.permute(0, 2, 1, 3).contiguous()
        query_count = min(128, int(q.shape[2]))
        q = q[:, :, -query_count:].float()
        k_hat = uncompress_single_kv_cache(k_quant).float()
        v_hat = uncompress_single_kv_cache(v_quant).float()
        k_reference = k.float()
        if str(layer.get("recon_audit_position", "pre_rope")) == "pre_rope":
            from wan.modules.causal_model import causal_rope_apply
            grid_sizes = layer.get("recon_audit_grid_sizes")
            freqs = layer.get("recon_audit_freqs")
            if not torch.is_tensor(grid_sizes) or not torch.is_tensor(freqs):
                return
            start_frame = int(token_start) // int(self.frame_seq_length)
            k_reference = causal_rope_apply(
                k_reference.permute(0, 2, 1, 3),
                grid_sizes,
                freqs,
                start_frame=start_frame,
            ).permute(0, 2, 1, 3).float()
            k_hat = causal_rope_apply(
                k_hat.permute(0, 2, 1, 3),
                grid_sizes,
                freqs,
                start_frame=start_frame,
            ).permute(0, 2, 1, 3).float()
        scale = float(k.shape[-1]) ** -0.5
        ref_logits = torch.matmul(q, k_reference.transpose(-1, -2)) * scale
        test_logits = torch.matmul(q, k_hat.transpose(-1, -2)) * scale
        ref_attention = ref_logits.softmax(-1)
        test_attention = test_logits.softmax(-1)
        ref_output = torch.matmul(ref_attention, v.float())
        k_only_output = torch.matmul(test_attention, v.float())
        v_only_output = torch.matmul(ref_attention, v_hat)
        joint_output = torch.matmul(test_attention, v_hat)
        kl = (
            ref_attention
            * (
                ref_attention.clamp_min(1e-12).log()
                - test_attention.clamp_min(1e-12).log()
            )
        ).sum(-1).mean()
        topk = min(32, int(k.shape[2]))
        ref_topk = ref_logits.topk(topk, dim=-1).indices
        test_topk = test_logits.topk(topk, dim=-1).indices
        topk_overlap = (
            ref_topk.unsqueeze(-1) == test_topk.unsqueeze(-2)
        ).any(-1).float().mean()
        self._recon_audit_rows.append({
            "tensor": "ATTENTION",
            "layer": int(layer_idx),
            "token_start": int(token_start),
            "token_end": int(token_end),
            "query_count": query_count,
            "logit_mse": float((ref_logits - test_logits).square().mean()),
            "attention_kl": float(kl),
            "top32_overlap": float(topk_overlap),
            "k_only_output_mse": float(
                (ref_output - k_only_output).square().mean()
            ),
            "v_only_output_mse": float(
                (ref_output - v_only_output).square().mean()
            ),
            "joint_output_mse": float(
                (ref_output - joint_output).square().mean()
            ),
        })

    def _reconstruction_audit_metrics(self):
        if not self._recon_audit_enabled:
            return None
        summary = {}
        for tensor_name in ("K", "V"):
            rows = [row for row in self._recon_audit_rows if row["tensor"] == tensor_name]
            if not rows:
                summary[tensor_name] = {
                    "raw_mse": 0.0,
                    "relative_squared_error": 0.0,
                    "relative_l2": 0.0,
                    "mean_vector_relative_squared_error": 0.0,
                    "element_count": 0,
                    "vector_count": 0,
                }
                continue
            element_count = sum(row["element_count"] for row in rows)
            vector_count = sum(row["vector_count"] for row in rows)
            squared_error_sum = sum(row["squared_error_sum"] for row in rows)
            source_squared_sum = sum(row["source_squared_sum"] for row in rows)
            vector_relative_sum = sum(
                row["vector_relative_squared_error_sum"] for row in rows
            )
            relative_squared_error = squared_error_sum / source_squared_sum
            summary[tensor_name] = {
                "raw_mse": squared_error_sum / element_count,
                "relative_squared_error": relative_squared_error,
                "relative_l2": relative_squared_error ** 0.5,
                "mean_vector_relative_squared_error": vector_relative_sum / vector_count,
                "element_count": element_count,
                "vector_count": vector_count,
            }
        return {
            "definition": {
                "raw_mse": "sum((x-x_hat)^2) / number_of_elements",
                "relative_squared_error": "sum((x-x_hat)^2) / sum(x^2)",
                "mean_vector_relative_squared_error": (
                    "mean_over_vectors(sum((x-x_hat)^2) / sum(x^2)); "
                    "matches per-vector RMS-normalized reconstruction MSE"
                ),
            },
            "summary": summary,
            "rows": self._recon_audit_rows,
            "attention_rows": [
                row for row in self._recon_audit_rows
                if row.get("tensor") == "ATTENTION"
            ],
        }

    def _print_memory_usage(self, kv_cache_list: list[dict]):
        """Offload all KV cache, then onload layer-by-layer to measure memory usage."""

        torch.cuda.empty_cache()
        cuda_device = torch.device("cuda")

        for layer in kv_cache_list:
            offload_kv_cache_layer(layer)

        memory_usage_before_onload = torch.cuda.memory_allocated() / 1024**2
        memory_usage = [memory_usage_before_onload]

        for layer in kv_cache_list:
            onload_kv_cache_layer(layer, cuda_device)
            memory_usage.append(torch.cuda.memory_allocated() / 1024**2)

        peak_memory_usage = max(memory_usage)
        diff_per_layer = [memory_usage[i + 1] - memory_usage[i] for i in range(len(memory_usage) - 1)]
        per_layer_memory_usage = np.median(diff_per_layer)
        total_kv_cache_memory_usage = per_layer_memory_usage * len(kv_cache_list)

        cprint(
            f"Peak Memory Usage: {peak_memory_usage:.2f} MB | Other Memory Usage: {memory_usage[0]:.2f} MB | "
            f"Per Layer Memory Usage: {per_layer_memory_usage:.2f} MB | Total KV Cache Memory Usage: {total_kv_cache_memory_usage:.2f} MB",
            "light_blue",
        )

    def _pack_info_into_kv_cache(self, k_cache, v_cache, output_dtype):
        """Pack metadata into KV cache. Only when the cached value is a real quantized dict."""
        if isinstance(k_cache, dict) and isinstance(v_cache, dict):
            k_cache["info"] = {
                "output_dtype": output_dtype,
                "quant_config": self.quant_config,
            }
            v_cache["info"] = {
                "output_dtype": output_dtype,
                "quant_config": self.quant_config,
            }
        return k_cache, v_cache
    

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        wall_start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(noise.device)
        reset_dequant_stats()
        self._encode_seconds = 0.0
        self._encode_calls = 0
        self._decode_seconds = 0.0
        self._decode_calls = 0
        self._quantized_frame_ranges = []
        self._temporal_schedules = []
        self._peak_kv_payload_bytes = 0
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros (size depends on target length when using global attention)
        if self.kv_cache1 is None:
            # When local_attn_size == -1 we need a cache large enough for the entire sequence
            if self.local_attn_size == -1:
                # reserve all output frames (input + to-generate) plus 1 extra block for safety
                target_frames = num_output_frames + 1
            else:
                target_frames = self.local_attn_size

            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                target_frames=target_frames
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for layer in self.kv_cache1:
                layer["k"].clear()
                layer["v"].clear()
                layer["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                layer["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for chunk_index, current_num_frames in enumerate(tqdm(all_num_frames)):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]
            
            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            # Update compressed storage once per completed generation chunk.
            # The first+recent policy delays one chunk so the newest context stays BF16.
            if self.quant_config.quant_type != "none":
                policy = getattr(self.quant_config, "compression_policy", "all")
                frame_range = completed_chunk_compression_range(chunk_index, all_num_frames, policy)
                if frame_range is not None:
                    start_frame, end_frame = frame_range
                    max_tokens = self.kv_cache1[0]["local_end_index"].item()
                    if self._temporal_residual_enabled():
                        self._quantize_temporal_residual_chunk(
                            start_frame,
                            end_frame,
                            max_tokens,
                        )
                    else:
                        self.quantize_kv_cache(
                            start_frame * self.frame_seq_length,
                            end_frame * self.frame_seq_length,
                            max_tokens,
                        )
                        self._record_persistent_kv_payload()

            # Bound CUDA-event bookkeeping while retaining exact dequant timing.
            chunk_dequant = get_dequant_stats(reset=True)
            self._decode_seconds += float(chunk_dequant["seconds"])
            self._decode_calls += int(chunk_dequant["calls"])

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        self._dump_calibration_samples(text_prompts, num_output_frames)

        # Step 4: Decode the output
        denoise_wall_seconds = time.perf_counter() - wall_start
        vae_wall_start = time.perf_counter()
        video = self.vae.decode_to_pixel(output, use_cache=False)
        
        # video = (video * 0.5 + 0.5).clamp(0, 1)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)
        torch.cuda.synchronize()
        vae_wall_seconds = time.perf_counter() - vae_wall_start
        dequant_stats = get_dequant_stats()
        self._decode_seconds += float(dequant_stats["seconds"])
        self._decode_calls += int(dequant_stats["calls"])
        cache_payload = self._kv_cache_payload_bytes()
        cache_bytes = cache_payload["K"] + cache_payload["V"]
        self._peak_kv_payload_bytes = max(
            self._peak_kv_payload_bytes, int(cache_bytes)
        )
        bf16_bytes = self._bf16_cache_bytes(num_output_frames, batch_size)
        codebook_metadata = self._shared_codebook_metadata()
        codebook_bytes = sum(int(item["codebook_storage_bytes"]) for item in codebook_metadata.values())
        shared_fixed_codec = {}
        if str(getattr(self.quant_config, "quant_type", "")).startswith("turboquant-"):
            from subitkv.turboquant import shared_parameter_manifest
            shared_fixed_codec = shared_parameter_manifest(
                dim=128,
                seed=int(getattr(self.quant_config, "turboquant_seed", 20260803)),
            )
            shared_fixed_codec["resident_gpu_bytes"] = int(
                shared_fixed_codec["rotation_bytes"]
                + (
                    shared_fixed_codec["qjl_bytes"]
                    if "-prod-" in str(self.quant_config.quant_type)
                    else 0
                )
            )
        self.runtime_metrics = {
            "num_latent_frames": int(num_output_frames),
            "num_generation_chunks": int(len(all_num_frames)),
            "latent_frames_per_chunk": int(self.num_frame_per_block),
            "wall_seconds": time.perf_counter() - wall_start,
            "e2e_seconds": time.perf_counter() - wall_start,
            "denoise_seconds": denoise_wall_seconds,
            "vae_decode_seconds": vae_wall_seconds,
            "quant_encode_seconds": self._encode_seconds,
            "quant_encode_calls": self._encode_calls,
            "quant_decode_seconds": self._decode_seconds,
            "quant_decode_calls": self._decode_calls,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(noise.device) / 1024**2,
            "bf16_kv_bytes": bf16_bytes,
            "stored_kv_bytes": cache_bytes,
            "k_payload_bytes": cache_payload["K"],
            "v_payload_bytes": cache_payload["V"],
            "k_payload_mib": cache_payload["K"] / 1024**2,
            "v_payload_mib": cache_payload["V"] / 1024**2,
            "kv_payload_mib": cache_bytes / 1024**2,
            "kv_peak_payload_mib": self._peak_kv_payload_bytes / 1024**2,
            "shared_codebook_bytes": codebook_bytes,
            "shared_codebook_mib": codebook_bytes / 1024**2,
            "shared_codebook_storage_basis": (
                "canonical_packed_values_plus_per_centroid_fp16_scale" if codebook_bytes else "none"
            ),
            "shared_codebook_metadata": codebook_metadata,
            "shared_fixed_codec": shared_fixed_codec,
            "shared_fixed_codec_mib": (
                float(shared_fixed_codec.get("resident_gpu_bytes", 0)) / 1024**2
            ),
            "compression_ratio_vs_bf16": bf16_bytes / cache_bytes if cache_bytes else None,
            "compression_ratio_vs_bf16_including_shared_codebook": (
                bf16_bytes / (cache_bytes + codebook_bytes)
                if cache_bytes + codebook_bytes else None
            ),
            "compressed_frame_ranges": self._quantized_frame_ranges,
            "temporal_precision_schedules": self._temporal_schedules,
        }
        reconstruction_audit = self._reconstruction_audit_metrics()
        if reconstruction_audit is not None:
            self.runtime_metrics["kv_reconstruction_audit"] = reconstruction_audit
        metrics_path = os.getenv("SELF_FORCING_METRICS_PATH")
        if metrics_path:
            path = Path(metrics_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(self.runtime_metrics, indent=2), encoding="utf-8")
            temporary.replace(path)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        # ----------------------------------------------------------
        # Optional: dump KV / CrossAttn cache for analysis
        # ----------------------------------------------------------
        if self.dump_kv_level >= 1:
            # Only rank-0 process dumps to avoid duplicates in DDP
            if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
                dump_dir = os.path.join(os.getenv("KV_DUMP_DIR", "kv_dumps"))
                os.makedirs(dump_dir, exist_ok=True)
                # Compose filename with prompt hash & frames
                filename = f"kv_cache_frames{num_output_frames}.pt"
                torch.save({
                    "kv_cache": self.kv_cache1,
                    "crossattn_cache": getattr(self, "crossattn_cache", None),
                }, os.path.join(dump_dir, filename))
                print(f"KV cache dumped to {os.path.join(dump_dir, filename)}")

        if return_latents:
            return video, output
        else:
            return video

    @staticmethod
    def _tensor_bytes(value) -> int:
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(CausalInferencePipeline._tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(CausalInferencePipeline._tensor_bytes(item) for item in value)
        return 0

    def _kv_cache_payload_bytes(self) -> dict[str, int]:
        totals = {"K": 0, "V": 0}
        for layer in self.kv_cache1:
            for name in ("k", "v"):
                cache = layer[name]
                total = sum(self._tensor_bytes(chunk) for chunk in cache.chunks if chunk is not None)
                total += sum(self._tensor_bytes(span["quant_data"]) for span in cache.quantized_spans)
                totals[name.upper()] += total
        return totals

    def _kv_cache_stored_bytes(self) -> int:
        """Compatibility total; shared codebooks are deliberately excluded."""

        payload = self._kv_cache_payload_bytes()
        return payload["K"] + payload["V"]

    def _bf16_cache_bytes(self, frames: int, batch_size: int) -> int:
        return (
            self.num_transformer_blocks * 2 * batch_size * frames * self.frame_seq_length
            * 12 * 128 * torch.tensor([], dtype=torch.bfloat16).element_size()
        )

    def _shared_codebook_metadata(self) -> dict:
        if not (getattr(self.quant_config, "k_rvq", False) and getattr(self.quant_config, "v_rvq", False)):
            return {}
        storage_dtype = str(getattr(self.quant_config, "k_rvq_codebook_storage_dtype", "int4"))
        k_metadata = rvq_codebook_directory_metadata(
            getattr(self.quant_config, "k_rvq_codebook_dir"),
            storage_dtype=storage_dtype,
        )
        v_metadata = rvq_codebook_directory_metadata(
            getattr(self.quant_config, "v_rvq_codebook_dir"),
            storage_dtype=str(getattr(self.quant_config, "v_rvq_codebook_storage_dtype", storage_dtype)),
        )
        return {"K": k_metadata, "V": v_metadata}

    def _shared_codebook_bytes(self) -> int:
        """Compatibility wrapper retained for callers/tests outside this pipeline."""

        return sum(
            int(item["codebook_storage_bytes"])
            for item in self._shared_codebook_metadata().values()
        )

    def _dump_calibration_samples(self, prompts: List[str], num_output_frames: int) -> None:
        root = getattr(self.quant_config, "calibration_dump_dir", "")
        if not root:
            return
        prompt_index = int(getattr(self.quant_config, "calibration_prompt_index", -1))
        sample_size = int(getattr(self.quant_config, "calibration_sample_tokens", 5000))
        if prompt_index < 0:
            raise ValueError("calibration_prompt_index is required when calibration dumping is enabled")
        prompt_dir = Path(root) / "kv_dump" / "train" / str(prompt_index)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        total_tokens = num_output_frames * self.frame_seq_length
        sampling_mode = str(
            getattr(self.quant_config, "calibration_sampling_mode", "random")
        )
        if sampling_mode == "contiguous_generation_chunk":
            chunk_frames = int(self.num_frame_per_block)
            if chunk_frames <= 0 or num_output_frames % chunk_frames:
                raise ValueError(
                    "contiguous calibration requires frame-aligned generation chunks"
                )
            chunk_count = num_output_frames // chunk_frames
            selected_chunk = prompt_index % chunk_count
            start = selected_chunk * chunk_frames * self.frame_seq_length
            available = chunk_frames * self.frame_seq_length
            take = min(sample_size, available)
            indices = torch.arange(start, start + take, dtype=torch.long)
        elif sampling_mode == "random":
            generator = torch.Generator(device="cpu").manual_seed(
                int(getattr(self.quant_config, "calibration_seed", 42)) + prompt_index
            )
            indices = torch.randperm(total_tokens, generator=generator)[: min(sample_size, total_tokens)]
            indices = indices.sort().values
            selected_chunk = None
        else:
            raise ValueError(f"unsupported calibration_sampling_mode={sampling_mode!r}")
        for layer_idx, layer in enumerate(self.kv_cache1):
            destination = prompt_dir / f"layer_{layer_idx}.pt"
            if destination.exists():
                continue
            tensors = {}
            for name in ("K", "V"):
                cache = layer[name.lower()]
                pieces = []
                for chunk_idx in indices.div(self.frame_seq_length, rounding_mode="floor").unique().tolist():
                    mask = indices.div(self.frame_seq_length, rounding_mode="floor") == chunk_idx
                    local = indices[mask] - chunk_idx * self.frame_seq_length
                    chunk = cache.chunks[chunk_idx]
                    if chunk is None:
                        raise RuntimeError("Calibration dumping requires an uncompressed BF16 cache")
                    pieces.append(chunk[:, local.to(chunk.device), :, :].cpu())
                tensors[name] = torch.cat(pieces, dim=1).permute(0, 2, 1, 3).contiguous()
            payload = {
                **tensors,
                "prompt": prompts[0],
                "prompt_id": prompt_index,
                "sample_indices": indices,
                "source_sequence_length": total_tokens,
                "sampling_mode": sampling_mode,
                "selected_generation_chunk": selected_chunk,
                "frame_seq_length": int(self.frame_seq_length),
                "generation_chunk_frames": int(self.num_frame_per_block),
            }
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)

    def _initialize_kv_cache(self, batch_size, dtype, device, target_frames: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # allocate enough tokens for entire sequence when global attention
            total_frames = target_frames if target_frames is not None else 21
            kv_cache_size = total_frames * self.frame_seq_length

        max_num_chunks = kv_cache_size // self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "subitkv_layer": _,
                "k": ChunkedKVCache(batch_size, self.frame_seq_length, 12, 128, max_num_chunks, dtype, device, layout="BSHD"),
                "v": ChunkedKVCache(batch_size, self.frame_seq_length, 12, 128, max_num_chunks, dtype, device, layout="BSHD"),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "cache_position": str(
                    getattr(self.quant_config, "cache_position", "pre_rope")
                ),
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
