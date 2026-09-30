# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

from typing import Optional
from dataclasses import dataclass, field

import torch


@dataclass
class InferState:
    enable_sageattn: bool = False  # whether to use SageAttention
    sage_blocks_range: Optional[range] = None  # block range to use SageAttention
    enable_torch_compile: bool = False  # whether to use torch compile

    # fp8 gemm related
    use_fp8_gemm: bool = False  # whether to use fp8 gemm
    quant_type: str = "fp8-per-block"  # fp8 quantization type
    include_patterns: list = field(
        default_factory=lambda: ["double_blocks"]
    )  # include patterns for fp8 gemm

    # vae related
    use_vae_parallel: bool = False  # whether to use vae parallel

    # AR KV-cache CPU offload and accounting. When enabled, only the cache for
    # the block currently being evaluated is resident on the CUDA device.
    kv_cache_offload: bool = False
    # Keep compact encoded vision payloads on CUDA while still discarding each
    # materialized BF16 layer after attention.
    kv_cache_gpu_resident_compressed: bool = False
    kv_cache_cpu_bytes_peak: int = 0
    kv_cache_gpu_layer_bytes_peak: int = 0
    kv_cache_gpu_compressed_bytes_current: int = 0
    kv_cache_gpu_compressed_bytes_peak: int = 0
    kv_cache_gpu_resident_hits: int = 0
    kv_cache_gpu_resident_promotions: int = 0
    peak_allocated_excluding_kv_estimate_bytes: int = 0
    kv_cache_onload_count: int = 0
    kv_cache_offload_count: int = 0
    full_history_attention: bool = False
    # Experimental causalized history encoder: freeze old vision K/V and only
    # encode the newly completed AR span.  The official rebuild path remains
    # the default because its history encoder is bidirectional.
    append_only_vision_cache: bool = False
    kv_cache_codec: str = "none"
    kv_codec_k_codebook_dir: str = ""
    kv_codec_v_codebook_dir: str = ""
    kv_codec_routing_path: str = ""
    kv_codec_tree_path: str = ""
    kv_codec_v_params_path: str = ""
    kv_codec_temporal_profile_path: str = ""
    kv_codec_frame_spatial_tokens: int = 1537
    kv_codec_encode_seconds: float = 0.0
    kv_codec_decode_seconds: float = 0.0
    kv_codec_encode_count: int = 0
    kv_codec_decode_count: int = 0
    kv_cache_bf16_vision_bytes_peak: int = 0
    kv_cache_compressed_vision_bytes_peak: int = 0
    kv_codec_encode_events: list = field(default_factory=list)
    kv_codec_decode_events: list = field(default_factory=list)
    # GPU-resident compressed KV can enqueue decode/encode and immediately
    # launch the consuming attention/cache work on the same CUDA stream.  Keep
    # timing events alive until the existing pipeline boundary instead of
    # synchronizing the whole device once per transformer layer.
    kv_codec_pending_cuda_timings: list = field(default_factory=list)
    kv_codec_timing_mode: str = "synchronized_layer_wall"
    kv_codec_int2_budget_bytes_peak: int = 0
    kv_codec_post_truncation_bytes_peak: int = 0
    kv_codec_nominal_k_bit_sum_peak: int = 0
    kv_codec_nominal_v_bit_sum_peak: int = 0
    kv_codec_avg2_physical_bytes_peak: int = 0
    kv_capture_dir: str = ""
    kv_capture_prompt_index: int = -1
    kv_capture_sample_tokens: int = 1250
    kv_capture_seed: int = 42
    kv_capture_history_calls: int = 9
    kv_residual_stats_dir: str = ""
    kv_residual_stats_prompt_index: int = -1
    kv_residual_stats_samples_per_head: int = 128
    kv_residual_stats_seed: int = 42
    kv_residual_stats_events: list = field(default_factory=list)
    kv_avg2_calibration_dir: str = ""
    kv_avg2_calibration_prompt_index: int = -1
    kv_avg2_calibration_queries_per_stream: int = 4
    kv_avg2_calibration_seed: int = 42
    kv_avg2_calibration_events: list = field(default_factory=list)
    kv_avg2_calibration_seen: set = field(default_factory=set)
    record_latent_hashes: bool = False
    terminal_cache_audit: bool = False
    terminal_cache_audit_result: dict = field(default_factory=dict)
    # Auditable wall-clock component timings. Values include required CUDA
    # synchronization at phase boundaries and are reported separately from E2E.
    part_seconds: dict = field(default_factory=dict)
    chunk_timing_events: list = field(default_factory=list)

    def add_part_seconds(self, name: str, seconds: float) -> None:
        self.part_seconds[str(name)] = self.part_seconds.get(str(name), 0.0) + float(seconds)

    def defer_codec_cuda_timing(
        self,
        *,
        kind: str,
        event: dict,
        start_event: torch.cuda.Event,
        end_event: torch.cuda.Event,
        host_submit_seconds: float,
        transfer_seconds: float,
    ) -> None:
        if kind not in {"encode", "decode"}:
            raise ValueError(f"unsupported codec timing kind {kind!r}")
        event["timing_mode"] = "deferred_cuda_event"
        event["host_submit_seconds"] = float(host_submit_seconds)
        self.kv_codec_pending_cuda_timings.append(
            {
                "kind": kind,
                "event": event,
                "start_event": start_event,
                "end_event": end_event,
                "host_submit_seconds": float(host_submit_seconds),
                "transfer_seconds": float(transfer_seconds),
            }
        )
        self.kv_codec_timing_mode = "deferred_cuda_events_no_layer_sync"

    def finalize_codec_cuda_timings(self) -> None:
        """Resolve deferred per-layer events at an existing CUDA boundary."""

        pending = self.kv_codec_pending_cuda_timings
        if not pending:
            return
        for item in pending:
            end_event = item["end_event"]
            end_event.synchronize()
            cuda_seconds = float(
                item["start_event"].elapsed_time(end_event) / 1000.0
            )
            transfer_seconds = float(item["transfer_seconds"])
            host_submit_seconds = float(item["host_submit_seconds"])
            kind = str(item["kind"])
            event = item["event"]
            event[f"{kind}_compute_seconds"] = cuda_seconds
            event["cuda_seconds"] = cuda_seconds
            event["seconds"] = cuda_seconds + transfer_seconds
            self.add_part_seconds(f"kv_{kind}_compute", cuda_seconds)
            self.add_part_seconds(
                f"kv_{kind}_host_submit", host_submit_seconds
            )
            if kind == "encode":
                self.kv_codec_encode_seconds += cuda_seconds + transfer_seconds
            else:
                self.kv_codec_decode_seconds += cuda_seconds + transfer_seconds
        pending.clear()


__infer_state = None


def parse_range(value):
    if "-" in value:
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    else:
        return [int(x) for x in value.split(",")]


def initialize_infer_state(args):
    global __infer_state
    sage_blocks_range = parse_range(args.sage_blocks_range)
    # Map CLI argument use_sageattn to internal enable_sageattn field
    use_sageattn = getattr(args, "use_sageattn", False)

    # Parse include_patterns from args
    include_patterns = getattr(args, "include_patterns", "double_blocks")
    if isinstance(include_patterns, str):
        # Split by comma and strip whitespace
        include_patterns = [p.strip() for p in include_patterns.split(",") if p.strip()]

    __infer_state = InferState(
        enable_sageattn=use_sageattn,
        sage_blocks_range=sage_blocks_range,
        enable_torch_compile=args.enable_torch_compile,
        # fp8 gemm related
        use_fp8_gemm=args.use_fp8_gemm,
        quant_type=args.quant_type,
        include_patterns=include_patterns,
        # vae related
        use_vae_parallel=args.use_vae_parallel,
        kv_cache_offload=getattr(args, "kv_cache_offload", False),
        kv_cache_gpu_resident_compressed=getattr(
            args, "kv_cache_gpu_resident_compressed", False
        ),
        full_history_attention=getattr(args, "full_history_attention", False),
        append_only_vision_cache=getattr(args, "append_only_vision_cache", False),
        kv_cache_codec=getattr(args, "kv_cache_codec", "none"),
        kv_codec_k_codebook_dir=getattr(args, "kv_codec_k_codebook_dir", ""),
        kv_codec_v_codebook_dir=getattr(args, "kv_codec_v_codebook_dir", ""),
        kv_codec_routing_path=getattr(args, "kv_codec_routing_path", ""),
        kv_codec_tree_path=getattr(args, "kv_codec_tree_path", ""),
        kv_codec_v_params_path=getattr(args, "kv_codec_v_params_path", ""),
        kv_codec_temporal_profile_path=getattr(args, "kv_codec_temporal_profile_path", ""),
        kv_codec_frame_spatial_tokens=getattr(args, "kv_codec_frame_spatial_tokens", 1537),
        kv_capture_dir=getattr(args, "kv_capture_dir", ""),
        kv_capture_prompt_index=getattr(args, "kv_capture_prompt_index", -1),
        kv_capture_sample_tokens=getattr(args, "kv_capture_sample_tokens", 1250),
        kv_capture_seed=getattr(args, "kv_capture_seed", 42),
        kv_capture_history_calls=getattr(args, "kv_capture_history_calls", 9),
        kv_residual_stats_dir=getattr(args, "kv_residual_stats_dir", ""),
        kv_residual_stats_prompt_index=getattr(args, "kv_residual_stats_prompt_index", -1),
        kv_residual_stats_samples_per_head=getattr(args, "kv_residual_stats_samples_per_head", 128),
        kv_residual_stats_seed=getattr(args, "kv_residual_stats_seed", 42),
        kv_avg2_calibration_dir=getattr(args, "kv_avg2_calibration_dir", ""),
        kv_avg2_calibration_prompt_index=getattr(args, "kv_avg2_calibration_prompt_index", -1),
        kv_avg2_calibration_queries_per_stream=getattr(args, "kv_avg2_calibration_queries_per_stream", 4),
        kv_avg2_calibration_seed=getattr(args, "kv_avg2_calibration_seed", 42),
        record_latent_hashes=getattr(args, "record_latent_hashes", False),
        terminal_cache_audit=getattr(args, "terminal_cache_audit", False),
    )
    return __infer_state


def get_infer_state():
    return __infer_state
