#!/usr/bin/env python
"""Small SuBitKV evaluation runner for one HY-WorldPlay setting.

Example:
python scripts/eval/run_small_eval.py \
  --prompt-file prompts_small.txt \
  --num-prompts 20 \
  --output-dir outputs/int2_eval \
  --seed 42 \
  --setting int2 \
  --quant_block_size 64 \
  --cache_num_k_centroids 256 \
  --cache_num_v_centroids 256 \
  --kmeans_max_iters 2 \
  --num_prq_stages 1

python scripts/eval/run_small_eval.py \
  --prompt-file prompts_small.txt \
  --num-prompts 20 \
  --output-dir outputs/int4_flc_eval \
  --seed 42 \
  --setting int4-flc \
  --lossless_flc_block_size 16

SVD residual INT4:
python scripts/eval/run_small_eval.py \
  --prompt-file prompts_small.txt \
  --num-prompts 1 \
  --output-dir outputs/svd_int4_eval \
  --seed 42 \
  --setting svd-int4

For multi-GPU runs, launch this script with torchrun in the same way as the
existing HY-WorldPlay scripts. This runner intentionally resolves exactly one
setting per process invocation.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
HYWORLDPLAY_ROOT = REPO_ROOT / "experiments" / "HY-WorldPlay"
HYWORLDPLAY_WAN_ROOT = HYWORLDPLAY_ROOT / "wan"
HYWORLDPLAY_GENERATE = HYWORLDPLAY_ROOT / "wan" / "generate.py"

DEFAULT_CKPT_ROOT = os.environ.get(
    "HYWORLDPLAY_CKPT_ROOT",
    str(Path(os.environ.get("SUBITKV_CKPT_ROOT", "ckpts")) / "HY-WorldPlay"),
)

DEFAULT_NEGATIVE_PROMPT = (
    "\u8272\u8c03\u8273\u4e3d,\u8fc7\u66dd,\u9759\u6001,"
    "\u7ec6\u8282\u6a21\u7cca\u4e0d\u6e05,\u5b57\u5e55,\u98ce\u683c,"
    "\u4f5c\u54c1,\u753b\u4f5c,\u753b\u9762,\u9759\u6b62,"
    "\u6574\u4f53\u53d1\u7070,\u6700\u5dee\u8d28\u91cf,\u4f4e\u8d28\u91cf,"
    "JPEG\u538b\u7f29\u6b8b\u7559,\u4e11\u964b\u7684,\u6b8b\u7f3a\u7684,"
    "\u591a\u4f59\u7684\u624b\u6307,\u753b\u5f97\u4e0d\u597d\u7684\u624b\u90e8,"
    "\u753b\u5f97\u4e0d\u597d\u7684\u8138\u90e8,\u7578\u5f62\u7684,"
    "\u6bc1\u5bb9\u7684,\u5f62\u6001\u7578\u5f62\u7684\u80a2\u4f53,"
    "\u624b\u6307\u878d\u5408,\u9759\u6b62\u4e0d\u52a8\u7684\u753b\u9762,"
    "\u6742\u4e71\u7684\u80cc\u666f,\u4e09\u6761\u817f,"
    "\u80cc\u666f\u4eba\u5f88\u591a,\u5012\u7740\u8d70"
)

CSV_COLUMNS = [
    "prompt_id",
    "prompt",
    "seed",
    "setting",
    "video_path",
    "e2e_time_s",
    "denoise_time_s",
    "quant_time_s",
    "temporal_policy_time_s",
    "full_encode_time_s",
    "dequant_time_s",
    "dequant_calls",
    "peak_gpu_mem_mb",
    "kv_cache_mem_mb",
    "compression_ratio",
    "k_payload_mib",
    "v_payload_mib",
    "kv_payload_mib",
    "k_payload_bytes",
    "v_payload_bytes",
    "kv_payload_bytes",
    "residual_k_payload_mib",
    "residual_v_payload_mib",
    "residual_kv_payload_mib",
    "residual_k_payload_bytes",
    "residual_v_payload_bytes",
    "residual_kv_payload_bytes",
    "residual_compression_ratio_vs_bf16",
    "shared_codebook_mib",
    "r2aq_shared_table_mib",
    "global_shared_mib",
    "r2aq_shared_table_bytes",
    "global_shared_bytes",
    "physical_kv_payload_bytes",
    "r2aq_shared_budget_owner_states",
    "r2aq_shared_budget_reserved_local_bytes",
    "r2aq_shared_budget_reserved_logical_bytes",
    "r2aq_shared_budget_exactly_repaid",
    "compression_ratio_excluding_shared_codebook",
    "compression_ratio_including_shared_codebook",
    "rvq_actual_avg_depth",
    "rvq_k_attention_kl",
    "rvq_v_output_mse",
    "rvq_v_raw_mse",
    "rvq_v_relative_l2",
    "rvq_v_mean_vector_relative_squared_error",
]


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def validate_tensor_residual_codec_overrides(args: argparse.Namespace) -> None:
    overrides = (
        str(getattr(args, "rvq_k_residual_codec", "") or ""),
        str(getattr(args, "rvq_v_residual_codec", "") or ""),
    )
    k_structured_layers = str(
        getattr(args, "rvq_k_structured_layers", "") or ""
    ).strip()
    span_schedule = str(
        getattr(args, "rvq_residual_codec_span_schedule", "") or ""
    ).strip()
    if (any(overrides) or k_structured_layers or span_schedule) and not bool(
        args.rvq_residual_quantize
    ):
        raise ValueError(
            "per-tensor/layer/span RVQ residual codec selection requires "
            "--rvq_residual_quantize"
        )
    if (any(overrides) or k_structured_layers or span_schedule) and bool(
        args.kv_quant_structured_temporal_policy
    ):
        raise ValueError(
            "per-tensor/layer/span RVQ residual codec selection is incompatible with "
            "--kv_quant_structured_temporal_policy"
        )
    if k_structured_layers:
        from subitkv.sim.quant.quantize_config import parse_layer_spec

        parse_layer_spec(k_structured_layers)
        if overrides[0]:
            raise ValueError(
                "--rvq_k_structured_layers cannot be combined with "
                "--rvq_k_residual_codec"
            )
    if span_schedule:
        from subitkv.sim.quant.quantize_config import (
            parse_residual_codec_span_schedule,
        )

        parsed = parse_residual_codec_span_schedule(span_schedule)
        if any(overrides) or k_structured_layers:
            raise ValueError(
                "--rvq_residual_codec_span_schedule cannot be combined with "
                "per-tensor overrides or --rvq_k_structured_layers"
            )
        if len(parsed) != 5:
            raise ValueError(
                "the Wan 11-chunk protocol requires exactly five physical "
                "residual-codec span entries"
            )


def validate_r2aq_codec_options(args: argparse.Namespace) -> None:
    """Fail closed on ambiguous coherent-tail runtime configurations."""

    selection_fraction = getattr(args, "r2aq_selection_fraction", None)
    if selection_fraction is not None and (
        not math.isfinite(float(selection_fraction))
        or not 0.0 <= float(selection_fraction) <= 1.0
    ):
        raise ValueError("--r2aq_selection_fraction must be in [0,1]")
    tail_fraction = float(
        getattr(args, "r2aq_coherent_tail_group_fraction", 0.55)
    )
    if not math.isfinite(tail_fraction) or not 0.0 <= tail_fraction <= 1.0:
        raise ValueError(
            "--r2aq_coherent_tail_group_fraction must be in [0,1]"
        )

    ablation = str(getattr(args, "r2aq_ablation", "") or "").strip().lower()
    coherent_tail_ablations = {"coherent_tail", "coherent_tail_g128"}
    owner_layer = int(
        getattr(args, "r2aq_shared_budget_owner_layer", -1)
    )
    owner_span = int(
        getattr(args, "r2aq_shared_budget_owner_span", -1)
    )
    logical_multiplier = int(
        getattr(args, "r2aq_payload_logical_multiplier", 1)
    )
    if logical_multiplier <= 0:
        raise ValueError("--r2aq_payload_logical_multiplier must be positive")
    if (owner_layer == -1) != (owner_span == -1):
        raise ValueError(
            "--r2aq_shared_budget_owner_layer and "
            "--r2aq_shared_budget_owner_span must be enabled together"
        )
    if owner_layer < -1 or owner_span < -1:
        raise ValueError("R²AQ shared-budget owner indices must be >=0 or -1")
    eager_opt_in = bool(
        getattr(args, "r2aq_coherent_tail_eager_decode", False)
    )
    if eager_opt_in and ablation not in coherent_tail_ablations:
        raise ValueError(
            "--r2aq_coherent_tail_eager_decode requires an explicit "
            "coherent-tail --r2aq_ablation"
        )
    if ablation not in coherent_tail_ablations:
        return
    if not eager_opt_in:
        raise ValueError(
            f"--r2aq_ablation {ablation} requires the explicit "
            "--r2aq_coherent_tail_eager_decode opt-in"
        )
    if not str(getattr(args, "r2aq_utility_weights_path", "") or "").strip():
        raise ValueError(
            f"--r2aq_ablation {ablation} requires "
            "--r2aq_utility_weights_path"
        )
    if int(getattr(args, "r2aq_residual_bits", 0)) != 4:
        raise ValueError(f"--r2aq_ablation {ablation} requires B4")
    if int(getattr(args, "r2aq_k_bits", 0)) not in {0, 4}:
        raise ValueError(f"--r2aq_ablation {ablation} requires K B4")
    if int(getattr(args, "r2aq_v_bits", 0)) not in {0, 4}:
        raise ValueError(f"--r2aq_ablation {ablation} requires V B4")
    if ablation == "coherent_tail_g128":
        if int(getattr(args, "r2aq_coherent_tail_group_size", 0)) != 128:
            raise ValueError(
                "--r2aq_ablation coherent_tail_g128 requires "
                "--r2aq_coherent_tail_group_size 128"
            )
        if int(getattr(args, "r2aq_predictor_group_size", 0)) != 128:
            raise ValueError(
                "--r2aq_ablation coherent_tail_g128 requires "
                "--r2aq_predictor_group_size 128"
            )
        if not math.isclose(tail_fraction, 1.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "the preregistered coherent_tail_g128 video gate requires "
                "--r2aq_coherent_tail_group_fraction 1.0"
            )
        if owner_layer < 0:
            raise ValueError(
                "the strict-rate coherent_tail_g128 video path requires an "
                "explicit shared-budget owner layer/span"
            )
        logical_head_dim = getattr(args, "kv_logical_head_dim", None)
        if logical_head_dim is not None:
            logical_head_dim = int(logical_head_dim)
            if (
                logical_head_dim <= 0
                or logical_head_dim % 128
                or logical_multiplier != logical_head_dim // 128
            ):
                raise ValueError(
                    "coherent_tail_g128 requires "
                    "r2aq_payload_logical_multiplier == "
                    "kv_logical_head_dim/128"
                )


def rvq_residual_quant_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the complete residual-codec fragment passed into QuantizeConfig."""

    return {
        "rvq_residual_quantize": True,
        "temporal_adaptive_residual": bool(
            getattr(args, "kv_quant_temporal_adaptive_residual", False)
        ),
        "temporal_adaptive_profile_path": str(
            getattr(args, "temporal_adaptive_profile_path", "") or ""
        ),
        "temporal_adaptive_capacity_bit_chunks": int(
            getattr(args, "temporal_adaptive_capacity_bit_chunks", 40)
        ),
        "temporal_adaptive_allocation_mode": str(
            getattr(args, "temporal_adaptive_allocation_mode", "optimal")
            or "optimal"
        ),
        "rvq_residual_codec": args.rvq_residual_codec,
        "rvq_k_residual_codec": args.rvq_k_residual_codec,
        "rvq_v_residual_codec": args.rvq_v_residual_codec,
        "kivi_residual_bits": int(args.kivi_residual_bits),
        "kivi_residual_group_size": int(args.kivi_residual_group_size),
        "rvq_k_structured_layers": args.rvq_k_structured_layers,
        "rvq_residual_codec_span_schedule": (
            args.rvq_residual_codec_span_schedule
        ),
        "higgs_group_size": args.higgs_group_size,
        "higgs_refine_iters": args.higgs_refine_iters,
        "higgs_seed": args.higgs_seed,
        "truncatable_residual_scheme": args.truncatable_residual_scheme,
        "truncatable_residual_bits": args.truncatable_residual_bits,
        "truncatable_residual_params_path": (
            args.truncatable_residual_params_path
        ),
        "truncatable_residual_hadamard": args.truncatable_residual_hadamard,
        "truncatable_residual_seed": args.truncatable_residual_seed,
        "routed_gaussian_bits": getattr(args, "routed_gaussian_bits", 4),
        "routed_gaussian_group_size": getattr(
            args, "routed_gaussian_group_size", 128
        ),
        "routed_gaussian_hadamard_group_size": (
            getattr(args, "routed_gaussian_hadamard_group_size", 128)
        ),
        "routed_gaussian_scale_refine_iters": (
            getattr(args, "routed_gaussian_scale_refine_iters", 1)
        ),
        "routed_gaussian_seed": getattr(
            args, "routed_gaussian_seed", 20260722
        ),
        "routed_gaussian_routing_path": (
            getattr(args, "routed_gaussian_routing_path", "")
        ),
        "routed_gaussian_tree_params_path": (
            getattr(args, "routed_gaussian_tree_params_path", "")
        ),
        "selective_rank1_gaussian_params_path": (
            getattr(args, "selective_rank1_gaussian_params_path", "")
        ),
        "absmax_scalar_bits": getattr(args, "absmax_scalar_bits", 4),
        "absmax_scalar_group_size": getattr(
            args, "absmax_scalar_group_size", 128
        ),
        "sign_adaptive_absmax_group_size": getattr(
            args, "sign_adaptive_absmax_group_size", 128
        ),
        "progressive_v_absmax_bits": getattr(
            args, "progressive_v_absmax_bits", 4
        ),
        "progressive_v_absmax_group_size": getattr(
            args, "progressive_v_absmax_group_size", 128
        ),
        "progressive_v_absmax_method": getattr(
            args, "progressive_v_absmax_method", "midpoint_msb"
        ),
        "progressive_v_absmax_params_path": getattr(
            args, "progressive_v_absmax_params_path", ""
        ),
        "centered_absmax_scalar_bits": getattr(
            args, "centered_absmax_scalar_bits", 4
        ),
        "centered_absmax_scalar_group_size": (
            getattr(args, "centered_absmax_scalar_group_size", 128)
        ),
        "centered_absmax_scalar_center_mode": (
            getattr(args, "centered_absmax_scalar_center_mode", "mean")
        ),
        "routed_midrange_scalar_bits": getattr(
            args, "routed_midrange_scalar_bits", 4
        ),
        "static_equalized_midrange_params_path": getattr(
            args, "static_equalized_midrange_params_path", ""
        ),
        "structured_residual_group_size": args.structured_residual_group_size,
        "structured_residual_clusters": args.structured_residual_clusters,
        "structured_residual_kmeans_iters": (
            args.structured_residual_kmeans_iters
        ),
        "structured_residual_kmeans_seed": (
            args.structured_residual_kmeans_seed
        ),
        "structured_residual_scale_refine_iters": (
            args.structured_residual_scale_refine_iters
        ),
        "structured_residual_k_bits": args.structured_residual_k_bits,
        "structured_residual_v_bits": args.structured_residual_v_bits,
        "structured_residual_fast_decode": (
            args.structured_residual_fast_decode
        ),
        "structured_prefix_allocation_path": (
            args.structured_prefix_allocation_path
        ),
        "r2aq_residual_bits": args.r2aq_residual_bits,
        "r2aq_k_bits": args.r2aq_k_bits,
        "r2aq_v_bits": args.r2aq_v_bits,
        "r2aq_predictor_group_size": args.r2aq_predictor_group_size,
        "r2aq_suffix_group_size": args.r2aq_suffix_group_size,
        "r2aq_clusters": args.r2aq_clusters,
        "r2aq_kmeans_iters": args.r2aq_kmeans_iters,
        "r2aq_kmeans_seed": args.r2aq_kmeans_seed,
        "r2aq_selection_mode": args.r2aq_selection_mode,
        "r2aq_ablation": getattr(args, "r2aq_ablation", ""),
        "r2aq_selection_fraction": getattr(
            args, "r2aq_selection_fraction", None
        ),
        "r2aq_coherent_tail_group_fraction": (
            getattr(args, "r2aq_coherent_tail_group_fraction", 0.55)
        ),
        "r2aq_coherent_tail_group_size": (
            getattr(args, "r2aq_coherent_tail_group_size", 128)
        ),
        "r2aq_utility_weights_path": args.r2aq_utility_weights_path,
        "r2aq_tree_params_path": args.r2aq_tree_params_path,
        "r2aq_fast_decode": args.r2aq_fast_decode,
        "r2aq_coherent_tail_eager_decode": (
            getattr(args, "r2aq_coherent_tail_eager_decode", False)
        ),
        "r2aq_shared_budget_owner_layer": getattr(
            args, "r2aq_shared_budget_owner_layer", -1
        ),
        "r2aq_shared_budget_owner_span": getattr(
            args, "r2aq_shared_budget_owner_span", -1
        ),
        "r2aq_payload_logical_multiplier": getattr(
            args, "r2aq_payload_logical_multiplier", 1
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small HY-WorldPlay SuBitKV eval for one setting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num-prompts", type=positive_int, default=1)
    parser.add_argument("--prompt-start", type=int, default=0)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt", type=str, help="Text for a single prompt.")
    prompt_source.add_argument(
        "--prompt-file",
        type=str,
        default=str(REPO_ROOT / "assets/default_prompt.txt"),
        help="Prompt file; defaults to the frog detective sample.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metrics-name", type=str, default="metrics.csv")
    parser.add_argument("--config-name", type=str, default="config.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--setting",
        type=str,
        default="bf16",
        help=(
            "Single setting label for metrics. Known labels bf16/none, int2, "
            "int4, int2-flc, int4-flc, qrvq-vonly-int2, krvq-vonly-int2, "
            "kvrvq/krvq-vrvq, kvrvq-i8/krvq-vrvq-i8, and SVD variants resolve quant_type automatically. "
            "oscar-int2 resolves to OSCAR rotation INT2 KV quantization. "
            "Other labels require --quant_type."
        ),
    )

    # Existing HY-WorldPlay generation arguments.
    parser.add_argument("--num_frames", type=int, default=961)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--image_path", type=str, default="assets/hyworld.png")
    parser.add_argument(
        "--pose",
        type=str,
        default="w-8,s-8,a-8,d-8,up-8,down-8",
        help="Pose string or pose JSON path, as accepted by HY-WorldPlay.",
    )
    parser.add_argument(
        "--ar_model_path",
        type=str,
        default=os.path.join(DEFAULT_CKPT_ROOT, "wan_transformer"),
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=os.path.join(DEFAULT_CKPT_ROOT, "wan_distilled_model", "model.pt"),
    )
    parser.add_argument("--num_chunk", type=int, default=12)
    parser.add_argument("--offload_text_encoder", action="store_true")
    parser.add_argument(
        "--skip_decode",
        action="store_true",
        help="Generate latents and KV-cache metrics without VAE video decode/export.",
    )
    parser.add_argument("--memory_frames", type=int, default=48)
    parser.add_argument("--temporal_context_size", type=int, default=44)
    parser.add_argument("--pred_latent_size", type=int, default=4)
    parser.add_argument(
        "--model_id",
        type=str,
        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    )

    # Existing SuBitKV KV-cache quantization arguments.
    parser.add_argument(
        "--quant_type",
        type=str,
        default=None,
        help=(
            "KV cache quantization type from the existing generator. If omitted, "
            "--setting resolves bf16/none to none, int2 to "
            "triton-nstages-kmeans-int2, int4 to triton-nstages-kmeans-int4, "
            "int4-flc to triton-nstages-kmeans-int4-flc, and svd-int4/svd8-int4 "
            "to svd8-triton-nstages-kmeans-int4."
        ),
    )
    parser.add_argument("--quant_block_size", type=int, default=64)
    parser.add_argument("--kivi_group_size", type=int, default=32)
    parser.add_argument("--kivi_residual_bits", type=int, default=4)
    parser.add_argument("--kivi_residual_group_size", type=int, default=32)
    parser.add_argument("--turboquant_seed", type=int, default=20260803)
    parser.add_argument("--cache_num_k_centroids", type=int, default=256)
    parser.add_argument("--cache_num_v_centroids", type=int, default=256)
    parser.add_argument("--kmeans_max_iters", type=int, default=2)
    parser.add_argument("--num_prq_stages", type=int, default=1)
    parser.add_argument(
        "--rvq_residual_quantize",
        action="store_true",
        help=(
            "Store RVQ(x) plus QuantVideo(x-RVQ(x)); requires an RVQ K and/or V "
            "setting and triton-nstages-kmeans INT2/INT4."
        ),
    )
    parser.add_argument(
        "--rvq_residual_codec",
        choices=(
            "quantvideo",
            "kivi",
            "higgs",
            "truncatable",
            "structured_truncatable",
            "structured_static_prefix",
            "routed_gaussian",
            "uniform_grid_gaussian",
            "frame_mean_gaussian",
            "shrunk_frame_mean_gaussian",
            "cache_mean_gaussian",
            "selective_rank1_gaussian",
            "absmax_scalar",
            "sign_adaptive_absmax_scalar",
            "progressive_sign_adaptive_absmax_scalar",
            "centered_absmax_scalar",
            "routed_midrange_scalar",
            "static_equalized_midrange",
            "r2aq",
        ),
        default="quantvideo",
        help="Codec applied to x-RVQ(x) when --rvq_residual_quantize is enabled.",
    )
    parser.add_argument(
        "--rvq_k_residual_codec",
        choices=(
            "quantvideo",
            "higgs",
            "truncatable",
            "structured_truncatable",
            "structured_static_prefix",
            "routed_gaussian",
            "uniform_grid_gaussian",
            "frame_mean_gaussian",
            "shrunk_frame_mean_gaussian",
            "cache_mean_gaussian",
            "selective_rank1_gaussian",
            "absmax_scalar",
            "sign_adaptive_absmax_scalar",
            "progressive_sign_adaptive_absmax_scalar",
            "centered_absmax_scalar",
            "routed_midrange_scalar",
            "static_equalized_midrange",
            "r2aq",
        ),
        default="",
        help=(
            "Optional K-only residual codec override. Empty inherits "
            "--rvq_residual_codec."
        ),
    )
    parser.add_argument(
        "--rvq_v_residual_codec",
        choices=(
            "quantvideo",
            "higgs",
            "truncatable",
            "structured_truncatable",
            "structured_static_prefix",
            "routed_gaussian",
            "uniform_grid_gaussian",
            "frame_mean_gaussian",
            "shrunk_frame_mean_gaussian",
            "cache_mean_gaussian",
            "selective_rank1_gaussian",
            "absmax_scalar",
            "sign_adaptive_absmax_scalar",
            "progressive_sign_adaptive_absmax_scalar",
            "centered_absmax_scalar",
            "routed_midrange_scalar",
            "static_equalized_midrange",
            "r2aq",
        ),
        default="",
        help=(
            "Optional V-only residual codec override. Empty inherits "
            "--rvq_residual_codec."
        ),
    )
    parser.add_argument(
        "--rvq_k_structured_layers",
        type=str,
        default="",
        help=(
            "Comma/range K-layer set (for example 0,4-7,10) that uses "
            "structured_truncatable residuals; all other K layers use "
            "quantvideo. Incompatible with --rvq_k_residual_codec and the "
            "structured temporal policy."
        ),
    )
    parser.add_argument(
        "--rvq_residual_codec_span_schedule",
        type=str,
        default="",
        help=(
            "Fixed five-span residual-codec schedule for the Wan 11-chunk "
            "protocol, e.g. q,q,q,s,s. q=QuantVideo and "
            "s=structured_truncatable; incompatible with tensor/layer "
            "overrides and the structured temporal prefix policy."
        ),
    )
    parser.add_argument("--higgs_group_size", type=int, choices=(32, 64), default=32)
    parser.add_argument("--higgs_refine_iters", type=int, choices=(0, 1), default=1)
    parser.add_argument("--higgs_seed", type=int, default=20260722)
    parser.add_argument(
        "--truncatable_residual_scheme",
        choices=(
            "nested_affine",
            "gaussian_tree",
            "additive_binary",
            "hierarchical_vq",
        ),
        default="nested_affine",
    )
    parser.add_argument(
        "--truncatable_residual_bits",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
    )
    parser.add_argument("--truncatable_residual_params_path", type=str, default="")
    parser.add_argument("--truncatable_residual_hadamard", action="store_true")
    parser.add_argument("--truncatable_residual_seed", type=int, default=20260722)
    parser.add_argument(
        "--routed_gaussian_bits",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8),
        default=4,
    )
    parser.add_argument(
        "--routed_gaussian_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--routed_gaussian_hadamard_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--routed_gaussian_scale_refine_iters",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )
    parser.add_argument("--routed_gaussian_seed", type=int, default=20260722)
    parser.add_argument(
        "--routed_gaussian_routing_path", type=str, default=""
    )
    parser.add_argument(
        "--routed_gaussian_tree_params_path", type=str, default=""
    )
    parser.add_argument(
        "--selective_rank1_gaussian_params_path", type=str, default=""
    )
    parser.add_argument(
        "--absmax_scalar_bits",
        type=int,
        choices=(2, 4),
        default=4,
    )
    parser.add_argument(
        "--absmax_scalar_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--sign_adaptive_absmax_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--progressive_v_absmax_bits",
        type=int,
        choices=(1, 2, 3, 4, 5, 6, 7, 8),
        default=4,
    )
    parser.add_argument(
        "--progressive_v_absmax_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--progressive_v_absmax_method",
        type=str,
        default="midpoint_msb",
    )
    parser.add_argument(
        "--progressive_v_absmax_params_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--centered_absmax_scalar_bits",
        type=int,
        choices=(2, 4),
        default=4,
    )
    parser.add_argument(
        "--centered_absmax_scalar_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--centered_absmax_scalar_center_mode",
        choices=("mean", "midrange"),
        default="mean",
    )
    parser.add_argument(
        "--routed_midrange_scalar_bits",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
    )
    parser.add_argument(
        "--static_equalized_midrange_params_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--structured_residual_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--structured_residual_clusters",
        type=int,
        choices=(64, 128, 256),
        default=256,
    )
    parser.add_argument(
        "--structured_residual_kmeans_iters",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--structured_residual_kmeans_seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--structured_residual_scale_refine_iters",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )
    parser.add_argument(
        "--structured_residual_k_bits",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help=(
            "K structured-residual prefix depth; zero inherits "
            "--truncatable_residual_bits."
        ),
    )
    parser.add_argument(
        "--structured_residual_v_bits",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help=(
            "V structured-residual prefix depth; zero inherits "
            "--truncatable_residual_bits."
        ),
    )
    parser.add_argument(
        "--structured_residual_fast_decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt into the fail-closed fused G128 CUDA decoder for structured "
            "residual payloads."
        ),
    )
    parser.add_argument(
        "--structured_prefix_allocation_path",
        type=str,
        default="",
        help=(
            "Shared layer/head K/V prefix table used by "
            "rvq_residual_codec=structured_static_prefix."
        ),
    )
    parser.add_argument(
        "--r2aq_residual_bits",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help=(
            "Nominal R²AQ depth. The physically equal-rate payload also stores "
            "one sparse next-plane refinement selected from recycled predictor bits."
        ),
    )
    parser.add_argument(
        "--r2aq_k_bits",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help="K nominal R²AQ depth; zero inherits --r2aq_residual_bits.",
    )
    parser.add_argument(
        "--r2aq_v_bits",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help="V nominal R²AQ depth; zero inherits --r2aq_residual_bits.",
    )
    parser.add_argument(
        "--r2aq_predictor_group_size",
        type=int,
        choices=(32, 64, 128),
        default=128,
    )
    parser.add_argument(
        "--r2aq_suffix_group_size",
        type=int,
        choices=(32, 64, 128),
        default=64,
    )
    parser.add_argument(
        "--r2aq_clusters",
        type=int,
        choices=(64, 128, 256),
        default=256,
    )
    parser.add_argument("--r2aq_kmeans_iters", type=positive_int, default=2)
    parser.add_argument("--r2aq_kmeans_seed", type=int, default=42)
    parser.add_argument(
        "--r2aq_selection_mode",
        choices=("scale", "scale_utility"),
        default="scale",
    )
    parser.add_argument(
        "--r2aq_ablation",
        choices=(
            "",
            "int4_predictor_qv",
            "uniform_bonus",
            "scale_waterfill",
            "utility_waterfill",
            "full_tree",
            "coherent_tail",
            "coherent_tail_g128",
        ),
        default="",
        help=(
            "Explicit R²AQ ablation. Empty preserves compatibility with "
            "--r2aq_selection_mode."
        ),
    )
    parser.add_argument(
        "--r2aq_selection_fraction",
        type=float,
        default=None,
        help=(
            "Optional fraction in [0,1] of reclaimed predictor bytes spent on "
            "the refinement packet; omission uses exact auto-budgeting."
        ),
    )
    parser.add_argument(
        "--r2aq_coherent_tail_group_fraction",
        type=float,
        default=0.55,
        help="Requested fraction in [0,1] of coherent-tail groups.",
    )
    parser.add_argument(
        "--r2aq_coherent_tail_group_size",
        type=positive_int,
        default=128,
        help=(
            "Explicit coherent-tail channel group size. The staged "
            "coherent_tail_g128 path requires 128."
        ),
    )
    parser.add_argument("--r2aq_utility_weights_path", type=str, default="")
    parser.add_argument("--r2aq_tree_params_path", type=str, default="")
    parser.add_argument(
        "--r2aq_fast_decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Opt into the fail-closed fused CUDA R²AQ decoder. Unsupported "
            "geometry raises instead of silently using eager decode."
        ),
    )
    parser.add_argument(
        "--r2aq_coherent_tail_eager_decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly permit eager reconstruction for coherent-tail V/B4 "
            "states. Non-tail K states can still use the fused decoder."
        ),
    )
    parser.add_argument(
        "--r2aq_shared_budget_owner_layer",
        type=int,
        default=-1,
        help=(
            "Layer index of the unique state that repays shared R²AQ table "
            "bytes; -1 disables shared-budget repayment."
        ),
    )
    parser.add_argument(
        "--r2aq_shared_budget_owner_span",
        type=int,
        default=-1,
        help=(
            "Physical span of the unique shared-budget owner; -1 disables "
            "shared-budget repayment."
        ),
    )
    parser.add_argument(
        "--r2aq_payload_logical_multiplier",
        type=positive_int,
        default=1,
        help=(
            "Logical memory-accounting multiplier applied to each core R²AQ "
            "state. Wan pre-RoPE D128 stored for logical D256 uses 2."
        ),
    )
    parser.add_argument(
        "--lossless_flc_block_size",
        type=positive_int,
        default=16,
        help="Sequence block size for experimental block fixed-length lossless residual coding.",
    )
    parser.add_argument("--save_full_precision_kv_cache", action="store_true")
    parser.add_argument("--kv_cache_save_dir", type=str, default=None)
    parser.add_argument(
        "--kv_cache_save_mode",
        type=str,
        default="new",
        choices=["new", "all"],
    )
    parser.add_argument(
        "--q_rvq_codebook_dir",
        type=str,
        default=str(Path(os.environ.get("SUBITKV_CODEBOOK_ROOT", "codebooks")) / "q_rvq"),
        help="Existing per-layer/head RVQ codebooks used to reconstruct Q for qrvq-vonly-int2.",
    )
    parser.add_argument("--q_rvq_batch_size", type=int, default=4096)
    parser.add_argument(
        "--q_rvq_compute_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument("--q_rvq_log_time", action="store_true")
    parser.add_argument(
        "--k_rvq_codebook_dir",
        type=str,
        default=str(Path(os.environ.get("SUBITKV_CODEBOOK_ROOT", "codebooks")) / "k_rvq"),
        help="Existing per-layer/head pre-RoPE K RVQ codebooks for krvq-vonly-int2.",
    )
    parser.add_argument("--k_rvq_batch_size", type=int, default=4096)
    parser.add_argument(
        "--k_rvq_compute_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--v_rvq_codebook_dir",
        type=str,
        default=str(Path(os.environ.get("SUBITKV_CODEBOOK_ROOT", "codebooks")) / "v_rvq"),
        help="Existing per-layer/head pre-PRoPE V RVQ codebooks for kvrvq/krvq-vrvq.",
    )
    parser.add_argument("--v_rvq_batch_size", type=int, default=4096)
    parser.add_argument(
        "--v_rvq_compute_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
    )
    parser.add_argument(
        "--rvq_8bit_storage",
        action="store_true",
        help="Store runtime RVQ codebooks as int8 and packed KV scales as uint8.",
    )
    parser.add_argument(
        "--rvq_codebook_storage_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "int8", "int4", "int2"],
        help="Resident runtime storage for RVQ codebooks.",
    )
    parser.add_argument(
        "--rvq_scale_storage_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "uint8", "uint4"],
        help="Per-span RVQ RMS scale storage.",
    )
    parser.add_argument(
        "--rvq_num_levels",
        type=int,
        default=0,
        help="If >0, use only the first N levels from the RVQ codebooks at runtime.",
    )
    parser.add_argument(
        "--rvq_progressive",
        action="store_true",
        help="Use adaptive RVQ prefix depth with compact extra-level index storage.",
    )
    parser.add_argument(
        "--rvq_progressive_min_depth",
        type=int,
        default=0,
        help="Minimum RVQ depth for every vector when --rvq_progressive is enabled.",
    )
    parser.add_argument(
        "--rvq_progressive_max_depth",
        type=int,
        default=0,
        help="Maximum RVQ depth for every vector when --rvq_progressive is enabled.",
    )
    parser.add_argument(
        "--rvq_progressive_error_threshold",
        type=float,
        default=0.0,
        help="Normalized residual RMS threshold for progressive RVQ early stop.",
    )
    parser.add_argument(
        "--rvq_progressive_granularity",
        default="token",
        choices=["token", "layer", "block"],
        help="Depth metadata granularity for progressive RVQ.",
    )
    parser.add_argument(
        "--rvq_progressive_block_size",
        type=positive_int,
        default=880,
        help="Token block size for block-level progressive RVQ.",
    )
    parser.add_argument(
        "--rvq_relative_stop",
        action="store_true",
        help=(
            "Stop dense span-level RVQ when two consecutive residual-MSE "
            "reductions fall below the configured threshold."
        ),
    )
    parser.add_argument(
        "--rvq_relative_stop_threshold",
        type=float,
        default=0.0629,
    )
    parser.add_argument(
        "--rvq_relative_stop_min_depth",
        type=positive_int,
        default=4,
    )
    parser.add_argument(
        "--rvq_relative_stop_max_depth",
        type=positive_int,
        default=16,
    )
    parser.add_argument(
        "--rvq_relative_stop_fast_approximate",
        action="store_true",
        help=(
            "Use the one-kernel adaptive RVQ encoder. The stopping depth is "
            "preserved; exceptionally close FP32 centroid ties may choose a "
            "different index."
        ),
    )
    parser.add_argument(
        "--rvq_feedback_method",
        type=str,
        default="none",
        choices=["none", "inner", "input_comp", "attn_output"],
        help="Optional encoder-only RVQ feedback assignment mode.",
    )
    parser.add_argument(
        "--rvq_feedback_tensors",
        type=str,
        default="kv",
        choices=["k", "v", "kv"],
        help="Apply RVQ feedback to K, V, or both.",
    )
    parser.add_argument(
        "--rvq_feedback_lambda",
        type=float,
        default=0.0,
        help="Penalty weight for inner/attn_output RVQ feedback modes.",
    )
    parser.add_argument(
        "--rvq_feedback_beta",
        type=float,
        default=0.0,
        help="Input compensation weight for --rvq_feedback_method=input_comp.",
    )
    parser.add_argument(
        "--rvq_feedback_query_sample",
        type=int,
        default=96,
        help="Reserved sampled-query budget for attention-output RVQ feedback diagnostics.",
    )
    parser.add_argument(
        "--rvq_diagnostics",
        action="store_true",
        help="Record sampled K attention KL and V attention-output MSE for RVQ spans.",
    )
    parser.add_argument(
        "--rvq_k_calibration",
        type=str,
        default="none",
        choices=["none", "online", "offline"],
        help="Post-RVQ K block-scale calibration mode.",
    )
    parser.add_argument(
        "--rvq_v_calibration",
        type=str,
        default="none",
        choices=["none", "online", "offline"],
        help="Post-RVQ V block-scale calibration mode.",
    )
    parser.add_argument(
        "--rvq_calibration_path",
        type=str,
        default="",
        help="Offline RVQ calibration table used by --rvq_*_calibration=offline.",
    )
    parser.add_argument(
        "--rvq_calibration_block_size",
        type=int,
        default=880,
        help="Sequence-token block size for coarse post-RVQ calibration scales.",
    )
    parser.add_argument(
        "--rvq_calibration_min",
        type=float,
        default=0.75,
        help="Lower clamp for post-RVQ calibration scales.",
    )
    parser.add_argument(
        "--rvq_calibration_max",
        type=float,
        default=1.33,
        help="Upper clamp for post-RVQ calibration scales.",
    )
    parser.add_argument(
        "--rvq_calibration_storage_dtype",
        type=str,
        default="uint4",
        choices=["uint4", "uint8", "bf16", "fp16", "fp32"],
        help="Runtime storage dtype for post-RVQ calibration scales.",
    )
    parser.add_argument(
        "--kv_logical_head_dim",
        type=int,
        default=0,
        help=(
            "If >0, use this head dimension as the BF16 denominator when "
            "reporting KV-cache compression ratio. This keeps pre-RoPE/RVQ "
            "methods comparable to the logical 256-dim full KV cache."
        ),
    )
    parser.add_argument(
        "--kv_quant_skip_chunk_policy",
        type=str,
        default="none",
        choices=["none", "first", "last", "first_last", "last_two"],
        help=(
            "Leave selected generation chunks in BF16 instead of quantizing them. "
            "'last' means the last chunk that would be quantized by the base policy; "
            "'last_two' leaves the final two quantizable chunks in BF16."
        ),
    )
    parser.add_argument(
        "--kv_quant_factor",
        type=positive_int,
        default=2,
        help=(
            "Quantize KV cache every N generation chunks. The original QuantVideo "
            "schedule uses 2; use 1 for per-chunk quantization."
        ),
    )
    parser.add_argument(
        "--kv_quant_temporal_adaptive_residual",
        action="store_true",
        help=(
            "Use immutable Adaptive L4-L16 RVQ per 4-frame cache chunk and "
            "online residual precision states 16/4/3/2/1 under a nominal "
            "40 bit-chunk budget."
        ),
    )
    parser.add_argument(
        "--temporal_adaptive_profile_path",
        type=str,
        default="",
        help="Frozen train-only time/rate-distortion profile JSON.",
    )
    parser.add_argument(
        "--temporal_adaptive_capacity_bit_chunks",
        type=positive_int,
        default=40,
        help="Peak residual bit-count capacity; base-aware profiles enforce lower per-step limits.",
    )
    parser.add_argument(
        "--temporal_adaptive_allocation_mode",
        choices=["optimal", "force_recent_b16", "frozen_joint_schedule"],
        default="optimal",
        help=(
            "Online residual scheduler. force_recent_b16 is a controlled "
            "counterfactual that preserves the newest chunk at B16 and forces "
            "older B2/B3 states under the same 40 bit-chunk budget; "
            "frozen_joint_schedule follows a train-only finite-horizon plan "
            "embedded in the profile."
        ),
    )
    parser.add_argument(
        "--kv_quant_tiered_depth_policy",
        action="store_true",
        help=(
            "Enable the experiment-only H7 temporal RVQ policy. Each 4-frame "
            "generation span is encoded once at L8, then aged by index-prefix "
            "truncation using boundary=8, middle=6, old=4, two oldest middle "
            "chunks, and one recent chunk. Requires --kv_quant_factor 1 and "
            "INT4 no-norm K/V RVQ codebooks."
        ),
    )
    parser.add_argument(
        "--kv_quant_tiered_depth_policy_name",
        type=str,
        default="h7_boundary8_middle6_old4",
        choices=[
            "h7_boundary8_middle6_old4",
            "age_ramp16_middle4_recent6_8_12_16",
        ],
        help=(
            "Select the experiment-only schedule enabled by "
            "--kv_quant_tiered_depth_policy. The age-ramp policy encodes new "
            "chunks at L16, keeps the first at L16, assigns the newest four "
            "non-anchor chunks L6/L8/L12/L16, and truncates older middle "
            "chunks to L4."
        ),
    )
    parser.add_argument(
        "--kv_quant_structured_temporal_policy",
        action="store_true",
        help=(
            "Enable the production K/V-decoupled structured-residual policy. "
            "Each K span is encoded directly at B2; each V span is encoded "
            "once at B4 and physically prefix-truncated to the selected frozen "
            "schedule. Requires exactly eleven generation chunks (five cached "
            "8-frame/7040-token spans), --kv_quant_factor 2, L8C64 INT4 "
            "no-norm K/V RVQ, and "
            "rvq_residual_codec=structured_truncatable."
        ),
    )
    parser.add_argument(
        "--kv_quant_structured_temporal_policy_name",
        type=str,
        default="structured_k_b2_v_31114_peak_safe",
        choices=(
            "structured_k_b2_v_31114_peak_safe",
            "structured_k_b2_v_31114",
            "structured_k_b2_v_41113",
        ),
        help=(
            "Frozen terminal V schedule selected by "
            "--kv_quant_structured_temporal_policy. 31114_peak_safe is the "
            "primary runtime history; 31114 retains fresh B4 as a "
            "capacity-greedy ablation. All terminal schedules retain ten V "
            "planes."
        ),
    )
    parser.add_argument(
        "--kv_quant_skip_frame_policy",
        type=str,
        default="none",
        choices=["none", "first", "last", "first_last"],
        help=(
            "Leave selected latent cache frames in BF16 instead of quantizing them. "
            "'last' means the last frame that would be quantized by the base policy."
        ),
    )
    parser.add_argument(
        "--kv_quant_skip_first_frames",
        type=int,
        default=0,
        help=(
            "Leave this many first latent cache frames in BF16. This generalizes "
            "--kv_quant_skip_frame_policy first for frame-count ablations."
        ),
    )
    parser.add_argument(
        "--kv_quant_skip_last_frames",
        type=int,
        default=0,
        help=(
            "Leave this many final quantizable latent cache frames in BF16. This "
            "generalizes --kv_quant_skip_frame_policy last for frame-count ablations."
        ),
    )
    parser.add_argument(
        "--kv_quant_recent_policy",
        type=str,
        default="none",
        choices=["none", "recent1or2", "nearest1", "nearest2"],
        help=(
            "Alternative online quantization schedule. recent1or2 keeps the newest "
            "1 or 2 generation chunks BF16 and quantizes the oldest two chunks when "
            "three unquantized chunks accumulate. nearest1/nearest2 keep the most "
            "recent 1/2 cached generation chunks BF16 at every quantization point."
        ),
    )
    parser.add_argument(
        "--kv_quant_recent_frames",
        type=int,
        default=0,
        help=(
            "Keep this many newest latent cache frames in BF16 at each online "
            "quantization step. This is dynamic recent-context retention, unlike "
            "--kv_quant_skip_last_frames which is a static final-frame range."
        ),
    )
    parser.add_argument(
        "--kv_quant_anchor_frame_stride",
        type=int,
        default=0,
        help=(
            "If >0, keep every Nth latent cache frame in BF16. For example, "
            "--kv_quant_anchor_frame_stride 4 --kv_quant_anchor_frame_offset 0 "
            "keeps the first frame of each 4-frame generation chunk."
        ),
    )
    parser.add_argument(
        "--kv_quant_anchor_frame_offset",
        type=int,
        default=0,
        help="Frame offset used with --kv_quant_anchor_frame_stride.",
    )
    parser.add_argument(
        "--kv_sparse_policy",
        type=str,
        default="none",
        choices=["none", "first_recent_chunks"],
        help=(
            "Drop middle KV cache entries instead of quantizing them. "
            "first_recent_chunks rebuilds cache from the first N frames and "
            "the newest M previous frames at each generation chunk."
        ),
    )
    parser.add_argument(
        "--kv_sparse_first_frames",
        type=int,
        default=0,
        help="Number of initial latent frames retained by --kv_sparse_policy.",
    )
    parser.add_argument(
        "--kv_sparse_recent_frames",
        type=int,
        default=0,
        help="Number of newest previous latent frames retained by --kv_sparse_policy.",
    )
    parser.add_argument(
        "--kv_sparse_quantize_retained",
        action="store_true",
        help=(
            "When --kv_sparse_policy rebuilds a reduced first/recent KV cache, "
            "quantize the retained cache entries with the selected quantization method."
        ),
    )
    parser.add_argument(
        "--kv_reference_frames",
        type=int,
        default=0,
        help=(
            "If >0, report compression_ratio against this many full BF16 "
            "KV-cache latent frames instead of only the stored chunks."
        ),
    )
    parser.add_argument(
        "--oscar_k_rotation_path",
        type=str,
        default="",
        help="OSCAR K rotation checkpoint path for --setting oscar-int2.",
    )
    parser.add_argument(
        "--oscar_v_rotation_path",
        type=str,
        default="",
        help="OSCAR V rotation checkpoint path for --setting oscar-int2.",
    )
    parser.add_argument(
        "--oscar_k_clip_percentile",
        type=float,
        default=1.0,
        help="Per-row symmetric clip percentile before OSCAR INT2 K quantization. 1.0 disables clipping.",
    )
    parser.add_argument(
        "--oscar_v_clip_percentile",
        type=float,
        default=1.0,
        help="Per-row symmetric clip percentile before OSCAR INT2 V quantization. 1.0 disables clipping.",
    )

    return parser.parse_args()


def setup_import_paths_and_env() -> None:
    for path in (str(HYWORLDPLAY_WAN_ROOT), str(HYWORLDPLAY_ROOT), str(REPO_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)

    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath_parts = [str(REPO_ROOT), str(HYWORLDPLAY_ROOT), str(HYWORLDPLAY_WAN_ROOT)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HYWORLDPLAY_OFFLOAD_FOR_DECODE", "1")
    os.environ.setdefault("HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE", "1")
    os.environ.setdefault("HYWORLDPLAY_ENABLE_VAE_TILING", "0")

    cuda_stubs = "/usr/local/cuda/targets/x86_64-linux/lib/stubs"
    if os.path.isdir(cuda_stubs):
        library_path = os.environ.get("LIBRARY_PATH")
        if library_path:
            os.environ["LIBRARY_PATH"] = cuda_stubs + os.pathsep + library_path
        else:
            os.environ["LIBRARY_PATH"] = cuda_stubs


def resolve_quant_type(setting: str, quant_type: str | None) -> str:
    if quant_type:
        return quant_type

    known_settings = {
        "bf16": "none",
        "bf16-prerope": "none",
        "none": "none",
        "int2": "triton-nstages-kmeans-int2",
        "int2-prerope": "triton-nstages-kmeans-int2",
        "quantvideo_prerope_int2": "triton-nstages-kmeans-int2",
        "quantvideo_postrope_int2": "triton-nstages-kmeans-int2",
        "qrvq-vint2": "triton-nstages-kmeans-int2",
        "qrvq-vonly-int2": "triton-nstages-kmeans-int2",
        "krvq-vint2": "triton-nstages-kmeans-int2",
        "krvq-vonly-int2": "triton-nstages-kmeans-int2",
        "kvrvq": "triton-nstages-kmeans-int2",
        "krvq-vrvq": "triton-nstages-kmeans-int2",
        "krvq-vrvq-int2": "triton-nstages-kmeans-int2",
        "kvrvq-i8": "triton-nstages-kmeans-int2",
        "krvq-vrvq-i8": "triton-nstages-kmeans-int2",
        "kvrvq-i4": "triton-nstages-kmeans-int2",
        "krvq-vrvq-i4": "triton-nstages-kmeans-int2",
        "int4": "triton-nstages-kmeans-int4",
        "int4-prerope": "triton-nstages-kmeans-int4",
        "quantvideo_prerope_int4": "triton-nstages-kmeans-int4",
        "quantvideo_postrope_int4": "triton-nstages-kmeans-int4",
        "kivi_all_int2": "kivi-all-int2",
        "kivi_all_int4": "kivi-all-int4",
        "turboquant_mse_k3_v2": "turboquant-mse-k3-v2",
        "turboquant_mse_k4_v3": "turboquant-mse-k4-v3",
        "turboquant_prod_k3_v2": "turboquant-prod-k3-v2",
        "turboquant_prod_k4_v3": "turboquant-prod-k4-v3",
        "turboquant_mse_k2_v2": "turboquant-mse-k2-v2",
        "turboquant_mse_k4_v4": "turboquant-mse-k4-v4",
        "turboquant_prod_k2_v2": "turboquant-prod-k2-v2",
        "turboquant_prod_k4_v4": "turboquant-prod-k4-v4",
        "int2-flc": "triton-nstages-kmeans-int2-flc",
        "int4-flc": "triton-nstages-kmeans-int4-flc",
        "svd-int4": "svd8-triton-nstages-kmeans-int4",
        "svd8-int4": "svd8-triton-nstages-kmeans-int4",
        "svd4-int4": "svd4-triton-nstages-kmeans-int4",
        "svd8-int2": "svd8-triton-nstages-kmeans-int2",
        "svd4-int2": "svd4-triton-nstages-kmeans-int2",
        "svd-int4-flc": "svd8-triton-nstages-kmeans-int4-flc",
        "svd8-int4-flc": "svd8-triton-nstages-kmeans-int4-flc",
        "svd4-int4-flc": "svd4-triton-nstages-kmeans-int4-flc",
        "svd4-int2-flc": "svd4-triton-nstages-kmeans-int2-flc",
        "svd8-int2-flc": "svd8-triton-nstages-kmeans-int2-flc",
        "oscar-int2": "oscar-int2",
    }
    setting_key = setting.lower()
    if setting_key not in known_settings:
        raise ValueError(f"--setting {setting!r} requires an explicit --quant_type")
    return known_settings[setting_key]


def configure_rvq_num_levels_env(
    *,
    rvq_num_levels: int,
    tiered_depth: bool,
    tiered_encode_depth: int = 8,
) -> int:
    """Configure fixed RVQ depth without changing the flag-off L6 contract."""

    requested = int(rvq_num_levels)
    if tiered_depth:
        encode_depth = int(tiered_encode_depth)
        if requested not in {0, encode_depth}:
            raise ValueError(
                f"Tiered RVQ always encodes new spans at L{encode_depth}"
            )
        os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = str(encode_depth)
        return encode_depth
    if requested > 0:
        os.environ["HYWORLDPLAY_RVQ_NUM_LEVELS"] = str(requested)
        return requested
    os.environ.pop("HYWORLDPLAY_RVQ_NUM_LEVELS", None)
    return 0


def parse_prompt_text(line: str) -> dict[str, str | None]:
    if "@" in line:
        prompt, image_path = line.split("@", 1)
        return {"prompt": prompt, "image_path": image_path or None}
    return {"prompt": line, "image_path": None}


def normalize_image_path(value: str | None) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw.lower() in {"", "none", "null", "nil", "t2v"}:
        return None
    return raw


def prompt_from_object(value: Any) -> dict[str, str | None]:
    if isinstance(value, str):
        return parse_prompt_text(value.strip())
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported prompt item: {value!r}")

    prompt = None
    for key in ("prompt", "text", "caption", "input"):
        if key in value:
            prompt = value[key]
            break
    if prompt is None:
        raise ValueError(f"Prompt object has no prompt/text/caption/input key: {value!r}")

    image_path = None
    for key in ("image_path", "image", "init_image"):
        if key in value:
            image_path = value[key]
            break

    return {
        "prompt": str(prompt),
        "image_path": str(image_path) if image_path else None,
    }


def load_prompt_file(path: str) -> list[dict[str, str | None]]:
    prompt_path = Path(path)
    suffix = prompt_path.suffix.lower()

    if suffix == ".jsonl":
        prompts = []
        with prompt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(prompt_from_object(json.loads(line)))
        return prompts

    if suffix == ".json":
        with prompt_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("prompts", "data", "items"):
                if key in data:
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError("JSON prompt file must be a list or contain prompts/data/items")
        return [prompt_from_object(item) for item in data]

    prompts = []
    with prompt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(parse_prompt_text(line))
    return prompts


def save_config(args: argparse.Namespace, quant_type: str, output_dir: Path) -> None:
    relevant_env_keys = [
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "HYWORLDPLAY_CKPT_ROOT",
        "HYWORLDPLAY_KV_CACHE_MODE",
        "HYWORLDPLAY_OFFLOAD_FOR_DECODE",
        "HYWORLDPLAY_OFFLOAD_KV_CACHE_FOR_DECODE",
        "HYWORLDPLAY_ENABLE_VAE_TILING",
        "HYWORLDPLAY_SAVE_QUANTIZED_KV_CACHE",
        "HYWORLDPLAY_QUANTIZED_KV_CACHE_SAVE_DIR",
        "HYWORLDPLAY_Q_RVQ_ENABLE",
        "HYWORLDPLAY_Q_RVQ_CODEBOOK_DIR",
        "HYWORLDPLAY_Q_RVQ_BATCH_SIZE",
        "HYWORLDPLAY_Q_RVQ_COMPUTE_DTYPE",
        "HYWORLDPLAY_Q_RVQ_LOG_TIME",
        "PYTORCH_ALLOC_CONF",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TIME_BENCH",
        "CLEAR_TIME",
    ]
    payload = {
        "command": shlex.join([sys.executable] + sys.argv),
        "argv": [sys.executable] + sys.argv,
        "entrypoint": str(HYWORLDPLAY_GENERATE),
        "args": vars(args),
        "resolved_quant_type": quant_type,
        "environment": {
            key: os.environ[key] for key in relevant_env_keys if key in os.environ
        },
    }
    if quant_type.startswith("turboquant-"):
        from subitkv.turboquant import shared_parameter_manifest
        shared = shared_parameter_manifest(
            dim=128, seed=int(getattr(args, "turboquant_seed", 20260803))
        )
        shared["resident_gpu_bytes"] = int(
            shared["rotation_bytes"]
            + (shared["qjl_bytes"] if "-prod-" in quant_type else 0)
        )
        payload["turboquant_shared_parameters"] = shared
    with (output_dir / args.config_name).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_logged_metrics(log_text: str) -> dict[str, float | None]:
    generation_times = [
        float(match)
        for match in re.findall(
            r"Generate time for chunk\s+\d+\s+is\s+([0-9.eE+-]+)",
            log_text,
        )
    ]
    quant_times = [
        float(match)
        for match in re.findall(r"KV Cache Quantization Time:\s*([0-9.eE+-]+)\s*s", log_text)
    ]
    temporal_policy_times = [
        float(match)
        for match in re.findall(
            r"KV Cache Temporal Policy Time:\s*([0-9.eE+-]+)\s*s",
            log_text,
        )
    ]
    dequant_events = [
        (float(seconds), int(calls))
        for seconds, calls in re.findall(
            r"KV Cache Dequantization Time for chunk\s+\d+:\s*"
            r"([0-9.eE+-]+)\s*s over\s+(\d+)\s+calls",
            log_text,
        )
    ]
    kv_cache_mem_values = [
        float(match)
        for match in re.findall(r"Total KV Cache Memory Usage:\s*([0-9.eE+-]+)\s*MB", log_text)
    ]

    return {
        # TODO: Prefer a structured timer emitted by WanPipeline instead of
        # parsing the existing human-readable chunk log.
        "denoise_time_s": sum(generation_times) if generation_times else None,
        # TODO: Prefer returning this from quantize_kv_cache once the pipeline
        # exposes a structured measurement object.
        "quant_time_s": sum(quant_times) if quant_times else None,
        # ``quant_time_s`` remains the historical adaptive-RVQ admission
        # interval.  The policy field additionally includes residual
        # materialization, physical prefix truncation, and scheduler/audit
        # work, so metric meaning is explicit rather than silently changed.
        "temporal_policy_time_s": (
            sum(temporal_policy_times) if temporal_policy_times else None
        ),
        "full_encode_time_s": (
            sum(temporal_policy_times)
            if temporal_policy_times
            else (sum(quant_times) if quant_times else None)
        ),
        "dequant_time_s": (
            sum(seconds for seconds, _calls in dequant_events)
            if dequant_events
            else None
        ),
        "dequant_calls": (
            sum(calls for _seconds, calls in dequant_events)
            if dequant_events
            else None
        ),
        "logged_kv_cache_mem_mb": max(kv_cache_mem_values) if kv_cache_mem_values else None,
    }


def tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    return 0


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def nested_tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return tensor_nbytes(value)
    if isinstance(value, dict):
        return sum(nested_tensor_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_nbytes(v) for v in value)
    return 0


def dtype_element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def torch_dtype_from_name(name: str) -> torch.dtype:
    raw = str(name).lower()
    if raw in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if raw in {"fp16", "float16"}:
        return torch.float16
    if raw in {"fp32", "float32"}:
        return torch.float32
    return getattr(torch, str(name))


def rvq_codebook_nbytes(
    codebook_dir: str | Path,
    layer_idx: int,
    compute_dtype: str,
    codebook_storage_dtype: str = "bf16",
    logical_head_dim: int = 0,
    num_levels: int | None = None,
) -> int:
    dtype = torch_dtype_from_name(compute_dtype)
    itemsize = dtype_element_size(dtype)
    total = 0
    paths = sorted(Path(codebook_dir).glob(f"layer_{layer_idx}_head_*.pt"))
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        codebooks = payload.get("codebooks")
        if isinstance(codebooks, (list, tuple)):
            selected = list(codebooks)
            if num_levels is not None and int(num_levels) > 0:
                selected = selected[: int(num_levels)]
            if not selected or not all(
                torch.is_tensor(value) and value.ndim == 2 for value in selected
            ):
                raise ValueError(f"Invalid variable-cardinality RVQ payload: {path}")
            actual_dim = int(selected[0].shape[-1])
            if any(int(value.shape[-1]) != actual_dim for value in selected):
                raise ValueError(f"Variable RVQ dimensions disagree: {path}")
            vector_dim = (
                int(logical_head_dim) if int(logical_head_dim) > 0 else actual_dim
            )
            centroid_vectors = sum(int(value.shape[0]) for value in selected)
            logical_numel = centroid_vectors * vector_dim
            storage_dtype = str(codebook_storage_dtype).lower()
            if storage_dtype == "int8":
                total += logical_numel + centroid_vectors * 2
            elif storage_dtype == "int4":
                total += (logical_numel + 1) // 2 + centroid_vectors * 2
            elif storage_dtype == "int2":
                total += (logical_numel + 3) // 4 + centroid_vectors * 2
            else:
                total += logical_numel * itemsize
        elif torch.is_tensor(codebooks):
            levels = int(codebooks.shape[0])
            if num_levels is not None and int(num_levels) > 0:
                levels = min(levels, int(num_levels))
            config = payload.get("config", {}) if isinstance(payload, dict) else {}
            rvq_format = str(payload.get("rvq_format", config.get("rvq_format", "standard"))).lower()
            if rvq_format in {"grouped_dim", "grouped"} or codebooks.ndim == 4:
                groups = int(payload.get("num_groups", config.get("num_groups", codebooks.shape[1])))
                codebook_size = int(payload.get("codebook_size", config.get("codebook_size", codebooks.shape[-2])))
                group_dim = int(payload.get("group_dim", config.get("group_dim", codebooks.shape[-1])))
                actual_dim = groups * group_dim
                logical_factor = (
                    max(1.0, float(logical_head_dim) / float(actual_dim))
                    if int(logical_head_dim) > 0 and actual_dim > 0
                    else 1.0
                )
                centroid_vectors = levels * groups * codebook_size
                logical_numel = int(math.ceil(centroid_vectors * group_dim * logical_factor))
            else:
                vector_dim = int(logical_head_dim) if int(logical_head_dim) > 0 else int(codebooks.shape[-1])
                centroid_vectors = levels * int(codebooks.shape[1])
                logical_numel = centroid_vectors * vector_dim
            storage_dtype = str(codebook_storage_dtype).lower()
            if storage_dtype == "int8":
                # Runtime int8 storage keeps one signed byte per centroid
                # element and one fp16 scale per centroid vector.
                total += logical_numel
                total += centroid_vectors * 2
            elif storage_dtype == "int4":
                # Runtime int4 storage packs two signed 4-bit centroid values
                # per byte and keeps one fp16 scale per centroid vector.
                total += (logical_numel + 1) // 2
                total += centroid_vectors * 2
            elif storage_dtype == "int2":
                # Runtime int2 storage packs four codebook values per byte and
                # keeps one fp16 scale per centroid vector.
                total += (logical_numel + 3) // 4
                total += centroid_vectors * 2
            else:
                total += logical_numel * itemsize
            for stat_name in ("residual_mean", "residual_std"):
                stat = payload.get(stat_name)
                if torch.is_tensor(stat):
                    stat_levels = int(stat.shape[0])
                    if num_levels is not None and int(num_levels) > 0:
                        stat_levels = min(stat_levels, int(num_levels))
                    # Residual norm stats are stored per level/head vector.
                    # Use logical_head_dim when reporting pre-RoPE/RVQ cache
                    # memory so the accounting stays comparable to BF16 KV.
                    stat_dim = int(logical_head_dim) if int(logical_head_dim) > 0 else int(stat.shape[-1])
                    total += stat_levels * stat_dim * stat.element_size()
    return total


def rvq_codebook_refs_nbytes(
    value: Any,
    seen: set[tuple[str, int, str, str, int, int]],
    logical_head_dim: int = 0,
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "rvq":
            codebook_dir = value.get("codebook_dir")
            layer_idx = value.get("layer_idx")
            compute_dtype = value.get("compute_dtype", "bf16")
            codebook_storage_dtype = value.get("codebook_storage_dtype", "bf16")
            num_levels = value.get("num_levels")
            # Span-local relative stopping still needs the complete candidate
            # prefix resident at encode time.  Charge that Lmax table once per
            # layer instead of charging a separate truncated table for every
            # distinct span depth.
            resident_levels = (
                value.get("relative_stop_max_depth", num_levels)
                if value.get("relative_stop")
                else num_levels
            )
            if codebook_dir is not None and layer_idx is not None:
                key = (
                    str(codebook_dir),
                    int(layer_idx),
                    str(compute_dtype),
                    str(codebook_storage_dtype),
                    int(logical_head_dim),
                    int(resident_levels)
                    if resident_levels is not None
                    else -1,
                )
                if key not in seen:
                    seen.add(key)
                    # RVQ packed spans keep indices/scales in-cache and load
                    # codebooks separately; include each resident layer codebook
                    # once in the reported KV-cache footprint.
                    return rvq_codebook_nbytes(
                        codebook_dir,
                        int(layer_idx),
                        str(compute_dtype),
                        str(codebook_storage_dtype),
                        logical_head_dim=logical_head_dim,
                        num_levels=(
                            int(resident_levels)
                            if resident_levels is not None
                            else None
                        ),
                    )
        return sum(rvq_codebook_refs_nbytes(v, seen, logical_head_dim) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(rvq_codebook_refs_nbytes(v, seen, logical_head_dim) for v in value)
    return 0


def rvq_codebook_size(value: dict) -> int:
    codebook_dir = value.get("codebook_dir")
    layer_idx = value.get("layer_idx")
    if codebook_dir is None or layer_idx is None:
        return 256
    paths = sorted(Path(codebook_dir).glob(f"layer_{int(layer_idx)}_head_*.pt"))
    if not paths:
        return 256
    payload = torch.load(paths[0], map_location="cpu", weights_only=False)
    codebooks = payload.get("codebooks")
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if "codebook_size" in payload:
        return int(payload["codebook_size"])
    if "codebook_size" in config:
        return int(config["codebook_size"])
    if torch.is_tensor(codebooks) and codebooks.ndim >= 4:
        return int(codebooks.shape[-2])
    if torch.is_tensor(codebooks) and codebooks.ndim >= 2:
        return int(codebooks.shape[1])
    if isinstance(codebooks, (list, tuple)) and codebooks:
        return max(int(value.shape[0]) for value in codebooks)
    return 256


def rvq_scale_theoretical_nbytes(value: dict) -> int:
    storage = str(value.get("scale_storage_dtype", "bf16")).lower()
    if storage == "uint8":
        return nested_tensor_nbytes(
            {
                "scale_q": value.get("scale_q"),
                "scale_min": value.get("scale_min"),
                "scale_step": value.get("scale_step"),
            }
        )
    if storage == "uint4":
        scale_q_shape = value.get("scale_q_shape")
        if scale_q_shape is not None:
            total = 1
            for dim in scale_q_shape:
                total *= int(dim)
            packed = (total + 1) // 2
        else:
            packed = tensor_nbytes(value.get("scale_q_packed"))
        return packed + nested_tensor_nbytes(
            {
                "scale_min": value.get("scale_min"),
                "scale_step": value.get("scale_step"),
            }
        )
    scale = value.get("scale")
    if torch.is_tensor(scale):
        # Paper-accounting default: BF16/FP16 scale metadata.
        return scale.numel() * 2
    return 0


def rvq_calibration_scale_nbytes(value: dict) -> int:
    storage = str(value.get("calibration_scale_storage_dtype", "")).lower()
    if not storage:
        return 0
    if storage == "uint8":
        return nested_tensor_nbytes(
            {
                "calibration_scale_q": value.get("calibration_scale_q"),
                "calibration_scale_min": value.get("calibration_scale_min"),
                "calibration_scale_step": value.get("calibration_scale_step"),
            }
        )
    if storage == "uint4":
        shape = value.get("calibration_scale_q_shape")
        if shape is not None:
            total = 1
            for dim in shape:
                total *= int(dim)
            packed = (total + 1) // 2
        else:
            packed = tensor_nbytes(value.get("calibration_scale_q_packed"))
        return packed + nested_tensor_nbytes(
            {
                "calibration_scale_min": value.get("calibration_scale_min"),
                "calibration_scale_step": value.get("calibration_scale_step"),
            }
        )
    scale = value.get("calibration_scale")
    if torch.is_tensor(scale):
        return scale.numel() * (4 if scale.dtype == torch.float32 else 2)
    return 0


def rvq_residual_norm_stats_nbytes(value: dict) -> int:
    return nested_tensor_nbytes(
        {
            "residual_mean": value.get("residual_mean"),
            "residual_std": value.get("residual_std"),
        }
    )


RVQ_RESIDUAL_METHODS = {
    "rvq_quantvideo_residual",
    "rvq_higgs_residual",
    "rvq_truncatable_residual",
    "rvq_structured_truncatable_residual",
    "rvq_structured_static_prefix_residual",
    "rvq_r2aq_residual",
    "rvq_selective_rank1_gaussian_residual",
    "rvq_cache_mean_gaussian_residual",
    "rvq_absmax_scalar_residual",
    "rvq_sign_adaptive_absmax_scalar_residual",
    "rvq_progressive_sign_adaptive_absmax_scalar_residual",
    "rvq_centered_absmax_scalar_residual",
    "rvq_routed_midrange_scalar_residual",
    "rvq_static_equalized_midrange_residual",
    "rvq_routed_gaussian_residual",
    "rvq_uniform_grid_gaussian_residual",
    "rvq_frame_mean_gaussian_residual",
    "rvq_shrunk_frame_mean_gaussian_residual",
    "rvq_temporal_adaptive_residual",
}


def rvq_residual_state_nbytes(method: str, residual: Any) -> float:
    if method == "rvq_r2aq_residual":
        from subitkv.r2aq import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    if method == "rvq_selective_rank1_gaussian_residual":
        from subitkv.selective_rank1_gaussian import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    if method == "rvq_cache_mean_gaussian_residual":
        from subitkv.cache_mean_gaussian import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    if method == "rvq_absmax_scalar_residual":
        from subitkv.absmax_scalar import payload_nbytes

        return float(payload_nbytes(residual))
    if method == "rvq_sign_adaptive_absmax_scalar_residual":
        from subitkv.sign_adaptive_absmax_scalar import payload_nbytes

        return float(payload_nbytes(residual))
    if method == "rvq_progressive_sign_adaptive_absmax_scalar_residual":
        from subitkv.progressive_sign_adaptive_absmax_scalar import (
            payload_nbytes,
        )

        return float(payload_nbytes(residual))
    if method in {
        "rvq_frame_mean_gaussian_residual",
        "rvq_shrunk_frame_mean_gaussian_residual",
    }:
        from subitkv.frame_mean_gaussian import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    if method == "rvq_routed_gaussian_residual":
        from subitkv.routed_gaussian import payload_nbytes

        return float(payload_nbytes(residual))
    if method == "rvq_uniform_grid_gaussian_residual":
        from subitkv.uniform_grid_gaussian import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    if method == "rvq_centered_absmax_scalar_residual":
        from subitkv.centered_absmax_scalar import payload_nbytes

        return float(payload_nbytes(residual))
    if method == "rvq_routed_midrange_scalar_residual":
        from subitkv.routed_midrange_scalar import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    if method == "rvq_static_equalized_midrange_residual":
        from subitkv.static_equalized_midrange import payload_nbytes

        return float(payload_nbytes(residual, include_shared=False))
    return float(nested_tensor_nbytes(residual))


def rvq_packed_tensor_nbytes(value: Any, bitpack: bool = False) -> float:
    if isinstance(value, dict):
        method = value.get("quant_method")
        if method == "rvq_temporal_adaptive_residual":
            return rvq_packed_tensor_nbytes(
                value.get("base"), bitpack
            ) + float(value["local_payload_bytes"])
        if method in RVQ_RESIDUAL_METHODS:
            return rvq_packed_tensor_nbytes(
                value.get("base"), bitpack
            ) + rvq_residual_state_nbytes(method, value.get("residual"))
        if method == "rvq":
            if bitpack:
                indices = value.get("indices")
                if torch.is_tensor(indices):
                    levels = int(indices.shape[0])
                    vectors = int(indices.numel() // max(1, levels))
                    bits = math.ceil(math.log2(max(2, rvq_codebook_size(value))))
                    if value.get("progressive"):
                        extra_indices = value.get("extra_indices")
                        active_depth = value.get("active_depth")
                        active_depth_block = value.get("active_depth_block")
                        extra_offsets = value.get("extra_level_offsets")
                        index_count = int(indices.numel())
                        if torch.is_tensor(extra_indices):
                            index_count += int(extra_indices.numel())
                        depth_bytes = tensor_nbytes(active_depth) + tensor_nbytes(active_depth_block)
                        offset_bytes = tensor_nbytes(extra_offsets)
                        return (
                            index_count * bits / 8.0
                            + depth_bytes
                            + offset_bytes
                            + rvq_scale_theoretical_nbytes(value)
                            + rvq_calibration_scale_nbytes(value)
                            + rvq_residual_norm_stats_nbytes(value)
                        )
                    return (
                        vectors * levels * bits / 8.0
                        + tensor_nbytes(value.get("relative_stop_depth"))
                        + rvq_scale_theoretical_nbytes(value)
                        + rvq_calibration_scale_nbytes(value)
                        + rvq_residual_norm_stats_nbytes(value)
                    )
                head_indices = value.get("head_indices")
                if isinstance(head_indices, (list, tuple)) and head_indices and all(
                    torch.is_tensor(item) and item.ndim == 3 for item in head_indices
                ):
                    bits = math.ceil(math.log2(max(2, rvq_codebook_size(value))))
                    return (
                        sum(int(item.numel()) for item in head_indices) * bits / 8.0
                        + rvq_scale_theoretical_nbytes(value)
                        + rvq_calibration_scale_nbytes(value)
                        + rvq_residual_norm_stats_nbytes(value)
                    )
            return nested_tensor_nbytes(value)
        return sum(rvq_packed_tensor_nbytes(v, bitpack) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(rvq_packed_tensor_nbytes(v, bitpack) for v in value)
    if torch.is_tensor(value):
        # A composite RVQ payload may contain a non-RVQ sibling such as the
        # QuantVideo-compressed correction residual.  Preserve RVQ's
        # theoretical bitpacking while counting those real tensor leaves.
        return float(tensor_nbytes(value))
    return 0


def rvq_quantvideo_residual_nbytes(value: Any) -> float:
    """Return only the stored RVQ residual-codec tensor bytes."""
    if isinstance(value, dict):
        method = value.get("quant_method")
        if method == "rvq_temporal_adaptive_residual":
            return float(value["local_payload_bytes"])
        if method in RVQ_RESIDUAL_METHODS:
            return rvq_residual_state_nbytes(method, value.get("residual"))
        return sum(rvq_quantvideo_residual_nbytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(rvq_quantvideo_residual_nbytes(v) for v in value)
    return 0.0


def r2aq_shared_refs_nbytes(
    value: Any,
    seen_table_ids: set[str],
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "r2aq":
            from subitkv.r2aq import shared_payload_nbytes

            return int(shared_payload_nbytes(value, seen_table_ids))
        return sum(
            r2aq_shared_refs_nbytes(item, seen_table_ids)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            r2aq_shared_refs_nbytes(item, seen_table_ids) for item in value
        )
    return 0


def selective_rank1_shared_refs_nbytes(
    value: Any,
    seen_table_ids: set[str],
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "selective_rank1_gaussian":
            from subitkv.selective_rank1_gaussian import (
                shared_payload_nbytes,
            )

            return int(shared_payload_nbytes(value, seen_table_ids))
        return sum(
            selective_rank1_shared_refs_nbytes(item, seen_table_ids)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            selective_rank1_shared_refs_nbytes(item, seen_table_ids)
            for item in value
        )
    return 0


def cache_mean_shared_refs_nbytes(
    value: Any,
    seen_table_ids: set[str],
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "cache_mean_gaussian":
            from subitkv.routed_gaussian import shared_payload_nbytes

            return int(
                shared_payload_nbytes(value["suffix_state"], seen_table_ids)
            )
        return sum(
            cache_mean_shared_refs_nbytes(item, seen_table_ids)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            cache_mean_shared_refs_nbytes(item, seen_table_ids)
            for item in value
        )
    return 0


def routed_gaussian_shared_refs_nbytes(
    value: Any,
    seen_table_ids: set[str],
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "routed_gaussian":
            from subitkv.routed_gaussian import shared_payload_nbytes

            return int(shared_payload_nbytes(value, seen_table_ids))
        return sum(
            routed_gaussian_shared_refs_nbytes(item, seen_table_ids)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            routed_gaussian_shared_refs_nbytes(item, seen_table_ids)
            for item in value
        )
    return 0


def routed_midrange_shared_refs_nbytes(
    value: Any,
    seen_table_ids: set[str],
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "routed_midrange_scalar":
            from subitkv.routed_midrange_scalar import (
                shared_payload_nbytes,
            )

            return int(shared_payload_nbytes(value, seen_table_ids))
        return sum(
            routed_midrange_shared_refs_nbytes(item, seen_table_ids)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            routed_midrange_shared_refs_nbytes(item, seen_table_ids)
            for item in value
        )
    return 0


def static_equalized_shared_refs_nbytes(
    value: Any,
    seen_table_ids: set[str],
) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "static_equalized_midrange":
            from subitkv.static_equalized_midrange import (
                shared_payload_nbytes,
            )

            return int(shared_payload_nbytes(value, seen_table_ids))
        return sum(
            static_equalized_shared_refs_nbytes(item, seen_table_ids)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            static_equalized_shared_refs_nbytes(item, seen_table_ids)
            for item in value
        )
    return 0


def r2aq_shared_budget_entries(value: Any) -> list[dict[str, Any]]:
    """Collect explicit shared-budget owner records from packed R²AQ states."""

    if isinstance(value, dict):
        if value.get("quant_method") == "r2aq":
            reserve = int(value.get("shared_budget_reserve_bytes", 0))
            if reserve <= 0:
                return []
            return [
                {
                    "tensor_name": str(value.get("tensor_name", "")),
                    "layer_idx": int(value.get("layer_idx", -1)),
                    "age_bin": int(value.get("age_bin", -1)),
                    "reserve_local_bytes": reserve,
                    "local_payload_target_bytes": int(
                        value.get("local_payload_target_bytes", -1)
                    ),
                }
            ]
        result: list[dict[str, Any]] = []
        for item in value.values():
            result.extend(r2aq_shared_budget_entries(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(r2aq_shared_budget_entries(item))
        return result
    return []


def contains_rvq(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("quant_method") == "rvq":
            return True
        return any(contains_rvq(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_rvq(v) for v in value)
    return False


def rvq_avg_depth_values(value: Any) -> list[tuple[float, int]]:
    if isinstance(value, dict):
        if value.get("quant_method") == "rvq":
            avg_depth = value.get("avg_depth")
            if isinstance(avg_depth, (int, float)):
                indices = value.get("indices")
                weight = 1
                if torch.is_tensor(indices) and indices.ndim >= 4:
                    weight = int(indices.shape[1]) * int(indices.shape[2]) * int(indices.shape[-1])
                head_indices = value.get("head_indices")
                if isinstance(head_indices, (list, tuple)) and head_indices and torch.is_tensor(head_indices[0]):
                    weight = len(head_indices) * int(head_indices[0].shape[1]) * int(head_indices[0].shape[2])
                return [(float(avg_depth), max(1, weight))]
            return []
        out: list[tuple[float, int]] = []
        for item in value.values():
            out.extend(rvq_avg_depth_values(item))
        return out
    if isinstance(value, (list, tuple)):
        out: list[tuple[float, int]] = []
        for item in value:
            out.extend(rvq_avg_depth_values(item))
        return out
    return []


def rvq_diagnostic_values(value: Any, key: str) -> list[float]:
    if isinstance(value, dict):
        if value.get("quant_method") == "rvq":
            diagnostics = value.get("diagnostics")
            if isinstance(diagnostics, dict):
                item = diagnostics.get(key)
                if isinstance(item, (int, float)):
                    return [float(item)]
            return []
        out: list[float] = []
        for item in value.values():
            out.extend(rvq_diagnostic_values(item, key))
        return out
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(rvq_diagnostic_values(item, key))
        return out
    return []


def contains_oscar(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("quant_method") == "oscar_int2":
            return True
        return any(contains_oscar(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_oscar(v) for v in value)
    return False


def oscar_rotation_refs_nbytes(value: Any, seen: set[tuple[str, int]]) -> int:
    if isinstance(value, dict):
        if value.get("quant_method") == "oscar_int2":
            rotation_path = value.get("rotation_path")
            layer_idx = value.get("layer_idx")
            if rotation_path is not None and layer_idx is not None:
                key = (str(rotation_path), int(layer_idx))
                if key not in seen:
                    seen.add(key)
                    from subitkv.oscar import oscar_rotation_ref_nbytes

                    return oscar_rotation_ref_nbytes(str(rotation_path), int(layer_idx))
            return 0
        return sum(oscar_rotation_refs_nbytes(v, seen) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(oscar_rotation_refs_nbytes(v, seen) for v in value)
    return 0


def chunked_cache_bytes(
    cache: Any,
    rvq_seen: set[tuple[str, int, str, str, int, int]],
    oscar_seen: set[tuple[str, int]],
    r2aq_seen: set[str],
    cache_key: str,
    logical_head_dim: int = 0,
    physical_pre_position: bool = False,
) -> tuple[float, int, float, float, float]:
    current_bytes: float = 0.0
    full_precision_bytes = 0
    shared_reference_bytes = 0.0
    r2aq_shared_reference_bytes = 0.0
    residual_payload_bytes = 0.0

    for chunk, state in zip(cache.chunks, cache.chunk_state):
        state_name = getattr(state, "name", str(state))
        if state_name == "BF16" and torch.is_tensor(chunk):
            bytes_used = float(tensor_nbytes(chunk))
            if (
                not physical_pre_position
                and int(logical_head_dim) > 0
                and getattr(cache, "head_dim", 0) > 0
            ):
                bytes_used *= max(1.0, float(logical_head_dim) / float(cache.head_dim))
            current_bytes += bytes_used
            full_precision_bytes += bytes_used

    full_precision_itemsize = dtype_element_size(cache.dtype)
    denominator_head_dim = int(logical_head_dim) if int(logical_head_dim) > 0 else cache.head_dim
    for span in cache.quantized_spans:
        quant_data = span.get("quant_data")
        if contains_rvq(quant_data):
            # Theoretical RVQ accounting:
            # - codebook vectors use the logical final KV dimension
            # - indices are bitpacked
            # - both K and V payloads count two rope/prope streams
            logical_multiplier = 1.0 if physical_pre_position else 2.0
            current_bytes += logical_multiplier * rvq_packed_tensor_nbytes(
                quant_data, bitpack=True
            )
            residual_payload_bytes += (
                logical_multiplier * rvq_quantvideo_residual_nbytes(quant_data)
            )
        else:
            span_bytes = float(nested_tensor_nbytes(quant_data))
            if (
                not physical_pre_position
                and int(logical_head_dim) > 0
                and getattr(cache, "head_dim", 0) > 0
            ):
                span_bytes *= max(1.0, float(logical_head_dim) / float(cache.head_dim))
            current_bytes += span_bytes
        reference_head_dim = 0 if physical_pre_position else logical_head_dim
        reference_bytes = float(
            rvq_codebook_refs_nbytes(quant_data, rvq_seen, reference_head_dim)
        )
        reference_bytes += float(oscar_rotation_refs_nbytes(quant_data, oscar_seen))
        r2aq_reference_bytes = float(
            r2aq_shared_refs_nbytes(quant_data, r2aq_seen)
        )
        reference_bytes += float(
            selective_rank1_shared_refs_nbytes(quant_data, r2aq_seen)
        )
        reference_bytes += float(
            routed_gaussian_shared_refs_nbytes(quant_data, r2aq_seen)
        )
        reference_bytes += float(
            routed_midrange_shared_refs_nbytes(quant_data, r2aq_seen)
        )
        reference_bytes += float(
            static_equalized_shared_refs_nbytes(quant_data, r2aq_seen)
        )
        reference_bytes += r2aq_reference_bytes
        r2aq_shared_reference_bytes += r2aq_reference_bytes
        current_bytes += reference_bytes
        shared_reference_bytes += reference_bytes
        num_chunks = span["end_chunk"] - span["start_chunk"]
        full_precision_bytes += (
            cache.batch_size
            * cache.num_heads
            * num_chunks
            * cache.frame_seq_length
            * denominator_head_dim
            * full_precision_itemsize
        )

    return (
        current_bytes,
        full_precision_bytes,
        shared_reference_bytes,
        residual_payload_bytes,
        r2aq_shared_reference_bytes,
    )


def cache_metrics_from_runner(
    runner: Any,
    logical_head_dim: int = 0,
    reference_frames: int = 0,
) -> dict[str, float | None]:
    pipe = getattr(runner, "pipe", None)
    layers = getattr(pipe, "_kv_cache", None)
    if not layers and getattr(pipe, "ctx", None) is not None:
        layers = pipe.ctx.get("kv_cache")
    if not layers:
        # TODO: Instrument integrations that do not expose a readable KV-cache
        # object on the pipeline and return the final/peak KV-cache bytes here.
        return {
            "kv_cache_mem_mb": None,
            "compression_ratio": None,
            "k_payload_mib": None,
            "v_payload_mib": None,
            "kv_payload_mib": None,
            "k_payload_bytes": None,
            "v_payload_bytes": None,
            "kv_payload_bytes": None,
            "residual_k_payload_mib": None,
            "residual_v_payload_mib": None,
            "residual_kv_payload_mib": None,
            "residual_k_payload_bytes": None,
            "residual_v_payload_bytes": None,
            "residual_kv_payload_bytes": None,
            "residual_compression_ratio_vs_bf16": None,
            "shared_codebook_mib": None,
            "r2aq_shared_table_mib": None,
            "global_shared_mib": None,
            "r2aq_shared_table_bytes": None,
            "global_shared_bytes": None,
            "physical_kv_payload_bytes": None,
            "r2aq_shared_budget_owner_states": None,
            "r2aq_shared_budget_reserved_local_bytes": None,
            "r2aq_shared_budget_reserved_logical_bytes": None,
            "r2aq_shared_budget_exactly_repaid": None,
            "compression_ratio_excluding_shared_codebook": None,
            "compression_ratio_including_shared_codebook": None,
        }

    current_bytes: float = 0.0
    full_precision_bytes = 0
    rvq_seen: set[tuple[str, int, str, str, int, int]] = set()
    oscar_seen: set[tuple[str, int]] = set()
    r2aq_seen: set[str] = set()
    rvq_depth_values: list[tuple[float, int]] = []
    rvq_k_attention_kl_values: list[float] = []
    rvq_v_output_mse_values: list[float] = []
    rvq_v_raw_mse_values: list[float] = []
    rvq_v_relative_l2_values: list[float] = []
    rvq_v_mean_vector_relative_squared_error_values: list[float] = []
    payload_bytes = {"k": 0.0, "v": 0.0}
    residual_payload_bytes = {"k": 0.0, "v": 0.0}
    shared_reference_bytes = 0.0
    r2aq_shared_reference_bytes = 0.0
    r2aq_shared_budget_owners: list[dict[str, Any]] = []
    for layer in layers:
        physical_pre_position = (
            isinstance(layer, dict)
            and layer.get("cache_mode") == "pre_kv_reconstruct"
        )
        for key in ("k", "v"):
            cache = layer.get(key) if isinstance(layer, dict) else None
            if hasattr(cache, "chunks") and hasattr(cache, "quantized_spans"):
                (
                    current,
                    full_precision,
                    references,
                    residual,
                    r2aq_references,
                ) = chunked_cache_bytes(
                    cache,
                    rvq_seen,
                    oscar_seen,
                    r2aq_seen,
                    key,
                    logical_head_dim=logical_head_dim,
                    physical_pre_position=physical_pre_position,
                )
                current_bytes += current
                payload_bytes[key] += current - references
                residual_payload_bytes[key] += residual
                shared_reference_bytes += references
                r2aq_shared_reference_bytes += r2aq_references
                full_precision_bytes += full_precision
                for span in cache.quantized_spans:
                    quant_data = span.get("quant_data")
                    for owner in r2aq_shared_budget_entries(quant_data):
                        r2aq_shared_budget_owners.append(
                            {
                                **owner,
                                "logical_multiplier": (
                                    1 if physical_pre_position else 2
                                ),
                            }
                        )
                    rvq_depth_values.extend(rvq_avg_depth_values(quant_data))
                    rvq_k_attention_kl_values.extend(rvq_diagnostic_values(quant_data, "k_attention_kl"))
                    rvq_v_output_mse_values.extend(rvq_diagnostic_values(quant_data, "v_output_mse"))
                    rvq_v_raw_mse_values.extend(rvq_diagnostic_values(quant_data, "v_raw_mse"))
                    rvq_v_relative_l2_values.extend(rvq_diagnostic_values(quant_data, "v_relative_l2"))
                    rvq_v_mean_vector_relative_squared_error_values.extend(
                        rvq_diagnostic_values(quant_data, "v_mean_vector_relative_squared_error")
                    )
            elif torch.is_tensor(cache):
                bytes_used = tensor_nbytes(cache)
                current_bytes += bytes_used
                payload_bytes[key] += bytes_used
                full_precision_bytes += bytes_used
            elif isinstance(cache, dict):
                if contains_rvq(cache):
                    logical_multiplier = 1.0 if physical_pre_position else 2.0
                    packed_bytes = logical_multiplier * rvq_packed_tensor_nbytes(
                        cache, bitpack=True
                    )
                    residual_payload_bytes[key] += (
                        logical_multiplier * rvq_quantvideo_residual_nbytes(cache)
                    )
                else:
                    packed_bytes = float(nested_tensor_nbytes(cache))
                reference_head_dim = 0 if physical_pre_position else logical_head_dim
                reference_bytes = float(
                    rvq_codebook_refs_nbytes(cache, rvq_seen, reference_head_dim)
                )
                reference_bytes += float(oscar_rotation_refs_nbytes(cache, oscar_seen))
                r2aq_reference_bytes = float(
                    r2aq_shared_refs_nbytes(cache, r2aq_seen)
                )
                reference_bytes += float(
                    selective_rank1_shared_refs_nbytes(cache, r2aq_seen)
                )
                reference_bytes += float(
                    cache_mean_shared_refs_nbytes(cache, r2aq_seen)
                )
                reference_bytes += float(
                    routed_midrange_shared_refs_nbytes(cache, r2aq_seen)
                )
                reference_bytes += float(
                    static_equalized_shared_refs_nbytes(cache, r2aq_seen)
                )
                reference_bytes += r2aq_reference_bytes
                r2aq_shared_reference_bytes += r2aq_reference_bytes
                for owner in r2aq_shared_budget_entries(cache):
                    r2aq_shared_budget_owners.append(
                        {
                            **owner,
                            "logical_multiplier": (
                                (1 if physical_pre_position else 2)
                                if contains_rvq(cache) else 1
                            ),
                        }
                    )
                payload_bytes[key] += packed_bytes
                shared_reference_bytes += reference_bytes
                current_bytes += packed_bytes + reference_bytes
                rvq_depth_values.extend(rvq_avg_depth_values(cache))
                rvq_k_attention_kl_values.extend(rvq_diagnostic_values(cache, "k_attention_kl"))
                rvq_v_output_mse_values.extend(rvq_diagnostic_values(cache, "v_output_mse"))
                rvq_v_raw_mse_values.extend(rvq_diagnostic_values(cache, "v_raw_mse"))
                rvq_v_relative_l2_values.extend(rvq_diagnostic_values(cache, "v_relative_l2"))
                rvq_v_mean_vector_relative_squared_error_values.extend(
                    rvq_diagnostic_values(cache, "v_mean_vector_relative_squared_error")
                )

    if int(reference_frames) > 0 and layers:
        first_cache = None
        first_layer = layers[0]
        if isinstance(first_layer, dict):
            first_cache = first_layer.get("k")
        if first_cache is not None and hasattr(first_cache, "batch_size"):
            denominator_head_dim = (
                int(logical_head_dim)
                if int(logical_head_dim) > 0
                else int(getattr(first_cache, "head_dim", 0))
            )
            full_precision_bytes = (
                2
                * len(layers)
                * int(getattr(first_cache, "batch_size", 1))
                * int(getattr(first_cache, "num_heads", 1))
                * int(reference_frames)
                * int(getattr(first_cache, "frame_seq_length", 880))
                * denominator_head_dim
                * dtype_element_size(getattr(first_cache, "dtype", torch.bfloat16))
            )

    kv_cache_mem_mb = current_bytes / 1024**2 if current_bytes else None
    kv_payload_bytes = payload_bytes["k"] + payload_bytes["v"]
    residual_kv_payload_bytes = (
        residual_payload_bytes["k"] + residual_payload_bytes["v"]
    )
    r2aq_reserved_local_bytes = sum(
        int(owner["reserve_local_bytes"])
        for owner in r2aq_shared_budget_owners
    )
    r2aq_reserved_logical_bytes = sum(
        int(owner["reserve_local_bytes"])
        * int(owner["logical_multiplier"])
        for owner in r2aq_shared_budget_owners
    )
    compression_ratio = (
        full_precision_bytes / current_bytes
        if current_bytes and full_precision_bytes
        else None
    )
    rvq_actual_avg_depth = (
        sum(depth * weight for depth, weight in rvq_depth_values) / sum(weight for _depth, weight in rvq_depth_values)
        if rvq_depth_values
        else None
    )
    return {
        "kv_cache_mem_mb": kv_cache_mem_mb,
        "compression_ratio": compression_ratio,
        "k_payload_mib": payload_bytes["k"] / 1024**2 if payload_bytes["k"] else None,
        "v_payload_mib": payload_bytes["v"] / 1024**2 if payload_bytes["v"] else None,
        "kv_payload_mib": kv_payload_bytes / 1024**2 if kv_payload_bytes else None,
        "k_payload_bytes": int(payload_bytes["k"]) if payload_bytes["k"] else None,
        "v_payload_bytes": int(payload_bytes["v"]) if payload_bytes["v"] else None,
        "kv_payload_bytes": int(kv_payload_bytes) if kv_payload_bytes else None,
        "residual_k_payload_mib": (
            residual_payload_bytes["k"] / 1024**2
            if residual_payload_bytes["k"] else None
        ),
        "residual_v_payload_mib": (
            residual_payload_bytes["v"] / 1024**2
            if residual_payload_bytes["v"] else None
        ),
        "residual_kv_payload_mib": (
            residual_kv_payload_bytes / 1024**2
            if residual_kv_payload_bytes else None
        ),
        "residual_k_payload_bytes": (
            int(residual_payload_bytes["k"])
            if residual_payload_bytes["k"] else None
        ),
        "residual_v_payload_bytes": (
            int(residual_payload_bytes["v"])
            if residual_payload_bytes["v"] else None
        ),
        "residual_kv_payload_bytes": (
            int(residual_kv_payload_bytes)
            if residual_kv_payload_bytes else None
        ),
        "residual_compression_ratio_vs_bf16": (
            full_precision_bytes / residual_kv_payload_bytes
            if full_precision_bytes and residual_kv_payload_bytes else None
        ),
        "shared_codebook_mib": shared_reference_bytes / 1024**2,
        "r2aq_shared_table_mib": (
            r2aq_shared_reference_bytes / 1024**2
        ),
        "global_shared_mib": shared_reference_bytes / 1024**2,
        "r2aq_shared_table_bytes": int(r2aq_shared_reference_bytes),
        "global_shared_bytes": int(shared_reference_bytes),
        "physical_kv_payload_bytes": int(
            kv_payload_bytes + shared_reference_bytes
        ),
        "r2aq_shared_budget_owner_states": len(
            r2aq_shared_budget_owners
        ),
        "r2aq_shared_budget_reserved_local_bytes": (
            r2aq_reserved_local_bytes
        ),
        "r2aq_shared_budget_reserved_logical_bytes": (
            r2aq_reserved_logical_bytes
        ),
        "r2aq_shared_budget_exactly_repaid": (
            r2aq_reserved_logical_bytes
            == int(r2aq_shared_reference_bytes)
        ),
        "compression_ratio_excluding_shared_codebook": (
            full_precision_bytes / kv_payload_bytes
            if full_precision_bytes and kv_payload_bytes else None
        ),
        "compression_ratio_including_shared_codebook": (
            full_precision_bytes / (kv_payload_bytes + shared_reference_bytes)
            if full_precision_bytes and kv_payload_bytes + shared_reference_bytes else None
        ),
        "rvq_actual_avg_depth": rvq_actual_avg_depth,
        "rvq_k_attention_kl": mean(rvq_k_attention_kl_values),
        "rvq_v_output_mse": mean(rvq_v_output_mse_values),
        "rvq_v_raw_mse": mean(rvq_v_raw_mse_values),
        "rvq_v_relative_l2": mean(rvq_v_relative_l2_values),
        "rvq_v_mean_vector_relative_squared_error": mean(
            rvq_v_mean_vector_relative_squared_error_values
        ),
    }


def clear_runner_kv_cache(runner: Any) -> None:
    pipe = getattr(runner, "pipe", None)
    if pipe is None:
        return

    for attr_name in ("_kv_cache", "_kv_cache_neg"):
        layers = getattr(pipe, attr_name, None)
        if not layers:
            continue
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            for value in layer.values():
                if hasattr(value, "clear"):
                    value.clear()
            if "end_index" in layer:
                layer["end_index"] = 0
        setattr(pipe, attr_name, [])

    if getattr(pipe, "ctx", None) is not None:
        pipe.ctx["kv_cache"] = []
        pipe.ctx["kv_cache_neg"] = []

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def csv_value(value: Any) -> Any:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def configure_qrvq_env(args: argparse.Namespace) -> bool:
    qrvq_setting = args.setting.lower() in {"qrvq-vint2", "qrvq-vonly-int2"}
    if qrvq_setting:
        q_codebook_dir = Path(args.q_rvq_codebook_dir)
        if not q_codebook_dir.exists():
            raise FileNotFoundError(f"Q-RVQ codebook dir does not exist: {q_codebook_dir}")
        os.environ["HYWORLDPLAY_Q_RVQ_ENABLE"] = "1"
        os.environ["HYWORLDPLAY_Q_RVQ_CODEBOOK_DIR"] = str(q_codebook_dir)
        os.environ["HYWORLDPLAY_Q_RVQ_BATCH_SIZE"] = str(args.q_rvq_batch_size)
        os.environ["HYWORLDPLAY_Q_RVQ_COMPUTE_DTYPE"] = args.q_rvq_compute_dtype
        os.environ["HYWORLDPLAY_Q_RVQ_LOG_TIME"] = "1" if args.q_rvq_log_time else "0"
    else:
        os.environ.pop("HYWORLDPLAY_Q_RVQ_ENABLE", None)
    return qrvq_setting


def configure_krvq_env(args: argparse.Namespace) -> bool:
    prerope_native_setting = args.setting.lower() in {
        "int2-prerope",
        "int4-prerope",
        "bf16-prerope",
        "quantvideo_prerope_int2",
        "quantvideo_prerope_int4",
        "kivi_all_int2",
        "kivi_all_int4",
        "turboquant_mse_k3_v2",
        "turboquant_mse_k4_v3",
        "turboquant_prod_k3_v2",
        "turboquant_prod_k4_v3",
        "turboquant_mse_k2_v2",
        "turboquant_mse_k4_v4",
        "turboquant_prod_k2_v2",
        "turboquant_prod_k4_v4",
    }
    oscar_setting = args.setting.lower() in {"oscar-int2"}
    krvq_setting = args.setting.lower() in {
        "krvq-vint2",
        "krvq-vonly-int2",
        "kvrvq",
        "krvq-vrvq",
        "krvq-vrvq-int2",
        "kvrvq-i8",
        "krvq-vrvq-i8",
        "kvrvq-i4",
        "krvq-vrvq-i4",
    }
    if prerope_native_setting or oscar_setting:
        os.environ["HYWORLDPLAY_KV_CACHE_MODE"] = "pre_kv_reconstruct"
    if oscar_setting:
        k_rotation = Path(args.oscar_k_rotation_path)
        v_rotation = Path(args.oscar_v_rotation_path)
        if not k_rotation.exists():
            raise FileNotFoundError(f"OSCAR K rotation checkpoint does not exist: {k_rotation}")
        if not v_rotation.exists():
            raise FileNotFoundError(f"OSCAR V rotation checkpoint does not exist: {v_rotation}")
    if krvq_setting:
        k_codebook_dir = Path(args.k_rvq_codebook_dir)
        if not k_codebook_dir.exists():
            raise FileNotFoundError(f"K-RVQ codebook dir does not exist: {k_codebook_dir}")
        os.environ["HYWORLDPLAY_KV_CACHE_MODE"] = "pre_kv_reconstruct"
    return krvq_setting


def configure_vrvq(args: argparse.Namespace) -> bool:
    vrvq_setting = args.setting.lower() in {
        "kvrvq",
        "krvq-vrvq",
        "krvq-vrvq-int2",
        "kvrvq-i8",
        "krvq-vrvq-i8",
        "kvrvq-i4",
        "krvq-vrvq-i4",
    }
    if vrvq_setting:
        v_codebook_dir = Path(args.v_rvq_codebook_dir)
        if not v_codebook_dir.exists():
            raise FileNotFoundError(f"V-RVQ codebook dir does not exist: {v_codebook_dir}")
    return vrvq_setting


def main() -> None:
    args = parse_args()
    setup_import_paths_and_env()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    output_dir = Path(args.output_dir)
    quant_type = resolve_quant_type(args.setting, args.quant_type)
    qrvq_setting = configure_qrvq_env(args)
    krvq_setting = configure_krvq_env(args)
    vrvq_setting = configure_vrvq(args)
    tiered_policy = (
        args.kv_quant_tiered_depth_policy_name
        if args.kv_quant_tiered_depth_policy
        else "none"
    )
    tiered_encode_depth = (
        16
        if tiered_policy == "age_ramp16_middle4_recent6_8_12_16"
        else 8
    )
    structured_temporal_policy = (
        args.kv_quant_structured_temporal_policy_name
        if args.kv_quant_structured_temporal_policy
        else "none"
    )
    validate_tensor_residual_codec_overrides(args)
    validate_r2aq_codec_options(args)
    if (
        args.kv_quant_tiered_depth_policy
        and args.kv_quant_structured_temporal_policy
    ):
        raise ValueError(
            "tiered RVQ and structured temporal residual policies are mutually "
            "exclusive"
        )
    if args.kv_quant_temporal_adaptive_residual:
        if args.kv_quant_tiered_depth_policy or args.kv_quant_structured_temporal_policy:
            raise ValueError(
                "adaptive residual time allocation is mutually exclusive with "
                "tiered RVQ and structured temporal policies"
            )
        if not (krvq_setting and vrvq_setting):
            raise ValueError(
                "--kv_quant_temporal_adaptive_residual requires packed K/V RVQ"
            )
        if not args.rvq_residual_quantize:
            raise ValueError(
                "--kv_quant_temporal_adaptive_residual requires "
                "--rvq_residual_quantize"
            )
        if args.kv_quant_factor != 1:
            raise ValueError(
                "--kv_quant_temporal_adaptive_residual requires "
                "--kv_quant_factor 1"
            )
        if args.num_chunk != 11:
            raise ValueError(
                "--kv_quant_temporal_adaptive_residual requires 11 generation "
                "chunks, yielding 10 cache admissions"
            )
        if not args.rvq_relative_stop:
            raise ValueError(
                "--kv_quant_temporal_adaptive_residual requires "
                "--rvq_relative_stop"
            )
        if (
            not math.isclose(
                args.rvq_relative_stop_threshold,
                0.0629,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or args.rvq_relative_stop_min_depth != 4
            or args.rvq_relative_stop_max_depth != 16
        ):
            raise ValueError(
                "temporal adaptive residual requires the 6.29% two-stage "
                "Adaptive RVQ rule with min L4 and max L16"
            )
        if args.rvq_num_levels not in {0, 16}:
            raise ValueError(
                "temporal adaptive residual requires --rvq_num_levels 16 or omission"
            )
        profile_path = Path(args.temporal_adaptive_profile_path)
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"temporal adaptive profile does not exist: {profile_path}"
            )
        if not Path(args.routed_gaussian_routing_path).is_file():
            raise FileNotFoundError("K routed Gaussian routing table is required")
        if not Path(args.progressive_v_absmax_params_path).is_file():
            raise FileNotFoundError("V progressive absmax params are required")
        from subitkv.temporal_adaptive_residual import load_profile

        temporal_profile = load_profile(profile_path)
        if temporal_profile.residual_capacity_by_step is not None:
            if (
                args.temporal_adaptive_capacity_bit_chunks
                != temporal_profile.joint_schedule_capacity_bit_chunks
            ):
                raise ValueError("runtime capacity disagrees with RVQ Base budget profile")
        elif args.temporal_adaptive_capacity_bit_chunks not in {
            10,
            20,
            30,
            40,
            80,
        }:
            raise ValueError(
                "formal 10-chunk temporal protocols require capacity 10, 20, "
                "30, 40, or 80 for average 1, 2, 3, 4, or 8 residual bits"
            )
        if any(
            (
                args.kv_quant_skip_chunk_policy != "none",
                args.kv_quant_skip_frame_policy != "none",
                args.kv_quant_skip_first_frames != 0,
                args.kv_quant_skip_last_frames != 0,
                args.kv_quant_recent_policy != "none",
                args.kv_quant_recent_frames != 0,
                args.kv_quant_anchor_frame_stride != 0,
                args.kv_sparse_policy != "none",
            )
        ):
            raise ValueError(
                "temporal adaptive residual cannot be combined with skip, "
                "recent-BF16, anchor, or sparse cache policies"
            )
    if args.kv_quant_tiered_depth_policy:
        if not (krvq_setting and vrvq_setting):
            raise ValueError(
                "--kv_quant_tiered_depth_policy requires packed RVQ for both K and V"
            )
        if args.kv_quant_factor != 1:
            raise ValueError(
                "--kv_quant_tiered_depth_policy requires --kv_quant_factor 1"
            )
        if args.rvq_progressive:
            raise ValueError(
                "--kv_quant_tiered_depth_policy is incompatible with --rvq_progressive"
            )
        if args.rvq_num_levels not in {0, tiered_encode_depth}:
            raise ValueError(
                "--kv_quant_tiered_depth_policy always encodes new spans at "
                f"L{tiered_encode_depth} for {tiered_policy}; use "
                f"--rvq_num_levels {tiered_encode_depth} or omit --rvq_num_levels"
            )
        if any(
            (
                args.kv_quant_skip_chunk_policy != "none",
                args.kv_quant_skip_frame_policy != "none",
                args.kv_quant_skip_first_frames != 0,
                args.kv_quant_skip_last_frames != 0,
                args.kv_quant_recent_policy != "none",
                args.kv_quant_recent_frames != 0,
                args.kv_quant_anchor_frame_stride != 0,
                args.kv_sparse_policy != "none",
            )
        ):
            raise ValueError(
                "--kv_quant_tiered_depth_policy cannot be combined with skip, "
                "recent-BF16, anchor, or sparse cache policies"
            )
    if args.kv_quant_structured_temporal_policy:
        if not (krvq_setting and vrvq_setting):
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires packed RVQ "
                "for both K and V"
            )
        if not args.rvq_residual_quantize:
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires "
                "--rvq_residual_quantize"
            )
        if args.rvq_residual_codec != "structured_truncatable":
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires "
                "--rvq_residual_codec structured_truncatable"
            )
        if args.truncatable_residual_scheme != "gaussian_tree":
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires "
                "--truncatable_residual_scheme gaussian_tree"
            )
        if not args.truncatable_residual_hadamard:
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires "
                "--truncatable_residual_hadamard"
            )
        if args.truncatable_residual_bits != 4:
            raise ValueError(
                "--kv_quant_structured_temporal_policy uses B4 as the V encode "
                "source; set --truncatable_residual_bits 4"
            )
        if args.structured_residual_k_bits not in {0, 2}:
            raise ValueError(
                "--kv_quant_structured_temporal_policy fixes K at B2; use "
                "--structured_residual_k_bits 2 or omit it"
            )
        if args.structured_residual_v_bits not in {0, 4}:
            raise ValueError(
                "--kv_quant_structured_temporal_policy encodes V at B4; use "
                "--structured_residual_v_bits 4 or omit it"
            )
        if args.kv_quant_factor != 2:
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires "
                "--kv_quant_factor 2 so each physical span is the formal "
                "8-frame/7040-token unit"
            )
        if args.num_chunk != 11:
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires exactly "
                "--num_chunk 11, yielding five terminal 7040-token cache spans"
            )
        if args.rvq_num_levels not in {0, 8}:
            raise ValueError(
                "--kv_quant_structured_temporal_policy requires "
                "--rvq_num_levels 8 or omission"
            )
        if args.rvq_progressive:
            raise ValueError(
                "--kv_quant_structured_temporal_policy is incompatible with "
                "--rvq_progressive"
            )
        if any(
            (
                args.kv_quant_skip_chunk_policy != "none",
                args.kv_quant_skip_frame_policy != "none",
                args.kv_quant_skip_first_frames != 0,
                args.kv_quant_skip_last_frames != 0,
                args.kv_quant_recent_policy != "none",
                args.kv_quant_recent_frames != 0,
                args.kv_quant_anchor_frame_stride != 0,
                args.kv_sparse_policy != "none",
            )
        ):
            raise ValueError(
                "--kv_quant_structured_temporal_policy cannot be combined with "
                "skip, recent-BF16, anchor, or sparse cache policies"
            )
    if args.rvq_progressive:
        if not (krvq_setting or vrvq_setting):
            raise ValueError("--rvq_progressive requires an RVQ setting")
        if args.rvq_progressive_min_depth <= 0:
            raise ValueError("--rvq_progressive_min_depth must be positive")
        if args.rvq_progressive_max_depth <= 0:
            raise ValueError("--rvq_progressive_max_depth must be positive")
        if args.rvq_progressive_min_depth > args.rvq_progressive_max_depth:
            raise ValueError("--rvq_progressive_min_depth must be <= --rvq_progressive_max_depth")
        if args.rvq_progressive_error_threshold < 0:
            raise ValueError("--rvq_progressive_error_threshold must be non-negative")
    if (
        args.rvq_relative_stop_fast_approximate
        and not args.rvq_relative_stop
    ):
        raise ValueError(
            "--rvq_relative_stop_fast_approximate requires "
            "--rvq_relative_stop"
        )
    if args.rvq_relative_stop:
        if not (krvq_setting or vrvq_setting):
            raise ValueError("--rvq_relative_stop requires an RVQ setting")
        if args.rvq_progressive:
            raise ValueError(
                "--rvq_relative_stop is incompatible with --rvq_progressive"
            )
        if args.kv_quant_tiered_depth_policy:
            raise ValueError(
                "--rvq_relative_stop is incompatible with tiered RVQ"
            )
        if args.kv_quant_structured_temporal_policy:
            raise ValueError(
                "--rvq_relative_stop is incompatible with structured temporal policy"
            )
        if not 0.0 <= args.rvq_relative_stop_threshold < 1.0:
            raise ValueError(
                "--rvq_relative_stop_threshold must be in [0,1)"
            )
        if args.rvq_relative_stop_min_depth < 2:
            raise ValueError(
                "--rvq_relative_stop_min_depth must be at least 2"
            )
        if (
            args.rvq_relative_stop_min_depth
            > args.rvq_relative_stop_max_depth
        ):
            raise ValueError(
                "--rvq_relative_stop_min_depth must be <= max depth"
            )
        if args.rvq_num_levels not in {
            0,
            args.rvq_relative_stop_max_depth,
        }:
            raise ValueError(
                "--rvq_relative_stop requires --rvq_num_levels to be omitted "
                "or equal to --rvq_relative_stop_max_depth"
            )
    if args.rvq_residual_quantize and not (krvq_setting or vrvq_setting):
        raise ValueError("--rvq_residual_quantize requires an RVQ K and/or V setting")
    if args.rvq_calibration_block_size <= 0:
        raise ValueError("--rvq_calibration_block_size must be positive")
    if args.rvq_calibration_min <= 0 or args.rvq_calibration_max < args.rvq_calibration_min:
        raise ValueError("--rvq_calibration_min/max must define a positive ordered range")
    if "offline" in {args.rvq_k_calibration, args.rvq_v_calibration} and not args.rvq_calibration_path:
        raise ValueError("Offline RVQ calibration requires --rvq_calibration_path")
    # Keep this separate from the uniform comparator: flag-off plus
    # --rvq_num_levels 6 continues to encode/decode fixed L6 normally.
    configure_rvq_num_levels_env(
        rvq_num_levels=args.rvq_num_levels,
        tiered_depth=args.kv_quant_tiered_depth_policy,
        tiered_encode_depth=tiered_encode_depth,
    )
    rvq_codebook_storage_dtype = (
        "int4"
        if args.setting.lower() in {"kvrvq-i4", "krvq-vrvq-i4"}
        else "int8"
        if args.rvq_8bit_storage or args.setting.lower() in {"kvrvq-i8", "krvq-vrvq-i8"}
        else args.rvq_codebook_storage_dtype
    )
    rvq_scale_storage_dtype = (
        "uint4"
        if args.setting.lower() in {"kvrvq-i4", "krvq-vrvq-i4"}
        else "uint8"
        if args.rvq_8bit_storage or args.setting.lower() in {"kvrvq-i8", "krvq-vrvq-i8"}
        else args.rvq_scale_storage_dtype
    )
    if (
        args.kv_quant_tiered_depth_policy
        or args.kv_quant_structured_temporal_policy
    ) and rvq_codebook_storage_dtype != "int4":
        raise ValueError(
            "temporal KV policies require "
            "--rvq_codebook_storage_dtype int4 (or an i4 RVQ setting)"
        )

    if args.prompt is not None:
        args.prompt_file = None
        if not args.prompt.strip():
            raise ValueError("--prompt must not be empty")
        all_prompts = [parse_prompt_text(args.prompt.strip())]
    else:
        all_prompts = load_prompt_file(args.prompt_file)
    if args.prompt_start < 0:
        raise ValueError("--prompt-start must be >= 0")
    prompts = all_prompts[args.prompt_start : args.prompt_start + args.num_prompts]
    if not prompts:
        raise ValueError(
            f"No prompts found in {args.prompt_file or '--prompt'} "
            f"at --prompt-start={args.prompt_start}"
        )

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_config(args, quant_type, output_dir)

    if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    from diffusers.utils import export_to_video
    from subitkv.logger import logger
    from subitkv.temporal_tiered_rvq import (
        temporal_tiered_depth_schedule,
    )
    from subitkv.temporal_structured_residual import (
        structured_temporal_depth_schedule,
    )
    from wan.generate import WanRunner

    logger.info(f"Running setting={args.setting} quant_type={quant_type}")
    runner = WanRunner(
        model_id=args.model_id,
        ckpt_path=args.ckpt_path,
        ar_model_path=args.ar_model_path,
        offload_text_encoder=args.offload_text_encoder,
    )

    metrics_file = None
    writer = None
    if rank == 0:
        metrics_file = (output_dir / args.metrics_name).open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(metrics_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        metrics_file.flush()

    try:
        for prompt_id, item in enumerate(prompts, start=args.prompt_start):
            prompt = item["prompt"]
            image_path = normalize_image_path(item.get("image_path") or args.image_path)
            output_path = output_dir / f"{prompt_id}-{args.seed}.mp4"

            if image_path and not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path does not exist: {image_path}")

            quant_config = None
            if quant_type != "none":
                quant_config = {
                    "quant_type": quant_type,
                    "quant_block_size": args.quant_block_size,
                    "cache_num_k_centroids": args.cache_num_k_centroids,
                    "cache_num_v_centroids": args.cache_num_v_centroids,
                    "kmeans_max_iters": args.kmeans_max_iters,
                    "num_prq_stages": args.num_prq_stages,
                    "lossless_flc_block_size": args.lossless_flc_block_size,
                    "kivi_group_size": args.kivi_group_size,
                    "turboquant_seed": args.turboquant_seed,
                }
                if qrvq_setting:
                    quant_config["quantize_k"] = False
                    quant_config["quantize_v"] = True
                if krvq_setting:
                    quant_config["quantize_k"] = True
                    quant_config["quantize_v"] = True
                    quant_config["k_rvq"] = True
                    quant_config["k_rvq_codebook_dir"] = str(Path(args.k_rvq_codebook_dir))
                    quant_config["k_rvq_batch_size"] = args.k_rvq_batch_size
                    quant_config["k_rvq_compute_dtype"] = args.k_rvq_compute_dtype
                    quant_config["k_rvq_codebook_storage_dtype"] = rvq_codebook_storage_dtype
                    quant_config["k_rvq_scale_storage_dtype"] = rvq_scale_storage_dtype
                if vrvq_setting:
                    quant_config["v_rvq"] = True
                    quant_config["v_rvq_codebook_dir"] = str(Path(args.v_rvq_codebook_dir))
                    quant_config["v_rvq_batch_size"] = args.v_rvq_batch_size
                    quant_config["v_rvq_compute_dtype"] = args.v_rvq_compute_dtype
                    quant_config["v_rvq_codebook_storage_dtype"] = rvq_codebook_storage_dtype
                    quant_config["v_rvq_scale_storage_dtype"] = rvq_scale_storage_dtype
                if args.rvq_residual_quantize:
                    quant_config.update(rvq_residual_quant_config(args))
                    if args.kv_quant_structured_temporal_policy:
                        # K is encoded directly at the retained B2 rate.  V is
                        # encoded once at B4 and only physically truncated by
                        # the temporal policy after storage.  The production
                        # bridge opts into a fail-closed fused FWHT decoder;
                        # generic structured-codec runs remain eager by
                        # default.
                        quant_config["structured_residual_k_bits"] = 2
                        quant_config["structured_residual_v_bits"] = 4
                        quant_config["structured_residual_fast_decode"] = True
                if args.rvq_progressive:
                    quant_config["rvq_progressive"] = True
                    quant_config["rvq_progressive_min_depth"] = args.rvq_progressive_min_depth
                    quant_config["rvq_progressive_max_depth"] = args.rvq_progressive_max_depth
                    quant_config["rvq_progressive_error_threshold"] = args.rvq_progressive_error_threshold
                    quant_config["rvq_progressive_granularity"] = args.rvq_progressive_granularity
                    quant_config["rvq_progressive_block_size"] = args.rvq_progressive_block_size
                if args.rvq_relative_stop:
                    quant_config["rvq_relative_stop"] = True
                    quant_config["rvq_relative_stop_threshold"] = (
                        args.rvq_relative_stop_threshold
                    )
                    quant_config["rvq_relative_stop_min_depth"] = (
                        args.rvq_relative_stop_min_depth
                    )
                    quant_config["rvq_relative_stop_max_depth"] = (
                        args.rvq_relative_stop_max_depth
                    )
                    quant_config[
                        "rvq_relative_stop_fast_approximate"
                    ] = bool(args.rvq_relative_stop_fast_approximate)
                if args.rvq_feedback_method != "none":
                    quant_config["rvq_feedback_method"] = args.rvq_feedback_method
                    quant_config["rvq_feedback_tensors"] = args.rvq_feedback_tensors
                    quant_config["rvq_feedback_lambda"] = args.rvq_feedback_lambda
                    quant_config["rvq_feedback_beta"] = args.rvq_feedback_beta
                    quant_config["rvq_feedback_query_sample"] = args.rvq_feedback_query_sample
                if args.rvq_diagnostics:
                    quant_config["rvq_diagnostics"] = True
                    quant_config["rvq_feedback_query_sample"] = args.rvq_feedback_query_sample
                if args.rvq_k_calibration != "none" or args.rvq_v_calibration != "none":
                    quant_config["rvq_k_calibration"] = args.rvq_k_calibration
                    quant_config["rvq_v_calibration"] = args.rvq_v_calibration
                    quant_config["rvq_calibration_path"] = args.rvq_calibration_path
                    quant_config["rvq_calibration_block_size"] = args.rvq_calibration_block_size
                    quant_config["rvq_calibration_min"] = args.rvq_calibration_min
                    quant_config["rvq_calibration_max"] = args.rvq_calibration_max
                    quant_config["rvq_calibration_storage_dtype"] = args.rvq_calibration_storage_dtype
                    quant_config["rvq_feedback_query_sample"] = args.rvq_feedback_query_sample
                if args.setting.lower() == "oscar-int2":
                    quant_config["oscar_k_rotation_path"] = str(Path(args.oscar_k_rotation_path))
                    quant_config["oscar_v_rotation_path"] = str(Path(args.oscar_v_rotation_path))
                    quant_config["oscar_k_clip_percentile"] = args.oscar_k_clip_percentile
                    quant_config["oscar_v_clip_percentile"] = args.oscar_v_clip_percentile

            kv_cache_save_dir = args.kv_cache_save_dir
            if args.save_full_precision_kv_cache and kv_cache_save_dir is None:
                kv_cache_save_dir = str(output_dir / "full_precision_kv_cache")

            input_dict = {
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "num_frames": args.num_frames,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": 1,
                "height": 704,
                "width": 1280,
                "image_path": image_path,
                "use_memory": True,
                "context_window_length": args.memory_frames,
                "memory_frames": args.memory_frames,
                "temporal_context_size": args.temporal_context_size,
                "pred_latent_size": args.pred_latent_size,
                "offload_text_encoder": args.offload_text_encoder,
                "quant_config": quant_config,
                "save_full_precision_kv_cache": args.save_full_precision_kv_cache,
                "kv_cache_save_dir": kv_cache_save_dir,
                "kv_cache_save_mode": args.kv_cache_save_mode,
                "kv_quant_skip_chunk_policy": args.kv_quant_skip_chunk_policy,
                "kv_quant_skip_frame_policy": args.kv_quant_skip_frame_policy,
                "kv_quant_skip_first_frames": args.kv_quant_skip_first_frames,
                "kv_quant_skip_last_frames": args.kv_quant_skip_last_frames,
                "kv_quant_recent_policy": args.kv_quant_recent_policy,
                "kv_quant_recent_frames": args.kv_quant_recent_frames,
                "kv_quant_anchor_frame_stride": args.kv_quant_anchor_frame_stride,
                "kv_quant_anchor_frame_offset": args.kv_quant_anchor_frame_offset,
                "kv_sparse_policy": args.kv_sparse_policy,
                "kv_sparse_first_frames": args.kv_sparse_first_frames,
                "kv_sparse_recent_frames": args.kv_sparse_recent_frames,
                "kv_sparse_quantize_retained": args.kv_sparse_quantize_retained,
                "kv_quant_factor": args.kv_quant_factor,
                "kv_quant_tiered_depth_policy": (
                    tiered_policy
                    if args.kv_quant_tiered_depth_policy
                    else "none"
                ),
                "kv_quant_structured_temporal_policy": (
                    structured_temporal_policy
                ),
                "seed": args.seed,
                "pose": args.pose,
                "num_chunk": args.num_chunk,
                "skip_decode": args.skip_decode,
            }

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            log_buffer = io.StringIO()
            stdout_tee = Tee(sys.stdout, log_buffer)
            stderr_tee = Tee(sys.stderr, log_buffer)

            start_time = time.perf_counter()
            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                result = runner.predict(input_dict)

            video_path_value = str(output_path)
            if rank == 0:
                video = result["video"]
                if video is not None:
                    export_to_video(video[0], str(output_path), fps=16)
                else:
                    video_path_value = ""

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            else:
                peak_gpu_mem_mb = None

            e2e_time_s = time.perf_counter() - start_time
            parsed_metrics = parse_logged_metrics(log_buffer.getvalue())
            cache_metrics = cache_metrics_from_runner(
                runner,
                args.kv_logical_head_dim,
                args.kv_reference_frames,
            )
            if args.kv_quant_tiered_depth_policy and rank == 0:
                ctx = getattr(getattr(runner, "pipe", None), "ctx", None)
                tiered_audit = (
                    ctx.get("kv_quant_tiered_depth_audit")
                    if isinstance(ctx, dict)
                    else None
                )
                if not isinstance(tiered_audit, dict) or not isinstance(
                    tiered_audit.get("final"), dict
                ):
                    raise RuntimeError(
                        "Tiered RVQ run completed without a structured depth/storage audit"
                    )
                expected_cache_chunks = max(0, int(args.num_chunk) - 1)
                expected_schedule = temporal_tiered_depth_schedule(
                    expected_cache_chunks,
                    policy=tiered_policy,
                )
                final_schedule = tiered_audit["final"].get("schedule")
                if final_schedule != expected_schedule:
                    raise RuntimeError(
                        "Tiered RVQ final schedule mismatch: "
                        f"{final_schedule} != {expected_schedule}"
                    )
                audit_payload = {
                    "schema_version": 1,
                    "policy": tiered_policy,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": args.seed,
                    "num_generation_chunks": int(args.num_chunk),
                    "num_cache_chunks": expected_cache_chunks,
                    "runtime_codebook_storage": "int4",
                    "input_norm_mode_required": "none",
                    "pipeline_audit": tiered_audit,
                }
                audit_path = output_dir / f"{prompt_id}-{args.seed}-tiered_depth_audit.json"
                with audit_path.open("w", encoding="utf-8") as f:
                    json.dump(audit_payload, f, indent=2)
            if args.kv_quant_structured_temporal_policy and rank == 0:
                ctx = getattr(getattr(runner, "pipe", None), "ctx", None)
                temporal_audit = (
                    ctx.get("kv_quant_structured_temporal_audit")
                    if isinstance(ctx, dict)
                    else None
                )
                if not isinstance(temporal_audit, dict) or not isinstance(
                    temporal_audit.get("final"), dict
                ):
                    raise RuntimeError(
                        "structured temporal run completed without a physical "
                        "prefix/payload audit"
                    )
                expected_cache_spans = max(
                    0, (int(args.num_chunk) - 1) // int(args.kv_quant_factor)
                )
                expected_k = structured_temporal_depth_schedule(
                    expected_cache_spans,
                    tensor_name="K",
                    policy=structured_temporal_policy,
                )
                expected_v = structured_temporal_depth_schedule(
                    expected_cache_spans,
                    tensor_name="V",
                    policy=structured_temporal_policy,
                )
                final_event = temporal_audit["final"]
                if final_event.get("k_schedule") != expected_k:
                    raise RuntimeError(
                        "structured temporal final K schedule mismatch: "
                        f"{final_event.get('k_schedule')} != {expected_k}"
                    )
                if final_event.get("v_schedule") != expected_v:
                    raise RuntimeError(
                        "structured temporal final V schedule mismatch: "
                        f"{final_event.get('v_schedule')} != {expected_v}"
                    )
                storage = final_event.get("storage", {})
                if not storage.get(
                    "terminal_payload_exactly_matches_uniform_kv_b2", False
                ):
                    raise RuntimeError(
                        "structured temporal terminal payload did not exactly "
                        "match direct uniform K/V B2"
                    )
                final_verification = temporal_audit.get(
                    "final_verification"
                )
                if not isinstance(final_verification, dict):
                    raise RuntimeError(
                        "structured temporal run lacks final chronology/payload "
                        "verification"
                    )
                failed_checks = {
                    name: value
                    for name, value in final_verification.get(
                        "checks", {}
                    ).items()
                    if value is False
                }
                if failed_checks:
                    raise RuntimeError(
                        "structured temporal final verification failed: "
                        f"{failed_checks}"
                    )
                decoder_verification = temporal_audit.get(
                    "decoder_verification"
                )
                if not isinstance(decoder_verification, dict) or not all(
                    (
                        decoder_verification.get("chunks_exact", False),
                        decoder_verification.get(
                            "all_events_fused_only", False
                        ),
                        decoder_verification.get(
                            "nonzero_fast_calls_after_first_history_chunk",
                            False,
                        ),
                        decoder_verification.get("backend")
                        == "triton_fwht_g128",
                    )
                ):
                    raise RuntimeError(
                        "structured temporal run lacks a successful "
                        "fail-closed fused FWHT decoder audit: "
                        f"{decoder_verification}"
                    )
                audit_payload = {
                    "schema_version": (
                        "wan5b_structured_temporal_residual_audit.v2"
                    ),
                    "policy": structured_temporal_policy,
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": args.seed,
                    "num_generation_chunks": int(args.num_chunk),
                    "num_cache_spans": expected_cache_spans,
                    "generation_chunks_per_physical_span": int(
                        args.kv_quant_factor
                    ),
                    "latent_frames_per_physical_span": (
                        int(args.kv_quant_factor) * 4
                    ),
                    "tokens_per_physical_span": (
                        int(args.kv_quant_factor) * 4 * 880
                    ),
                    "runtime_codebook_storage": "int4",
                    "input_norm_mode_required": "none",
                    "formal_uniform_protocol": (
                        "K direct B2; V direct B2"
                    ),
                    "temporal_protocol": (
                        "K direct B2; V direct B4 then physical prefix truncation"
                    ),
                    "decoder_protocol": {
                        "mode": "fail_closed_fused",
                        "backend": "triton_fwht_g128",
                        "generic_structured_default": "eager",
                    },
                    "screen_bitstream_relation": (
                        "closed-loop production bridge; the formal exact-screen "
                        "uses task/layer/span-derived predictor seeds, whereas "
                        "this frozen runtime passes kmeans seed 42 (effective "
                        "K=42, V=43), so no bit-for-bit screen replay is claimed"
                    ),
                    "pipeline_audit": temporal_audit,
                }
                audit_path = (
                    output_dir
                    / f"{prompt_id}-{args.seed}-structured_temporal_audit.json"
                )
                with audit_path.open("x", encoding="utf-8") as f:
                    json.dump(audit_payload, f, indent=2)
            if args.kv_quant_temporal_adaptive_residual and rank == 0:
                ctx = getattr(getattr(runner, "pipe", None), "ctx", None)
                adaptive_audit = (
                    ctx.get("kv_quant_temporal_adaptive_audit")
                    if isinstance(ctx, dict)
                    else None
                )
                if not isinstance(adaptive_audit, dict):
                    raise RuntimeError(
                        "temporal adaptive run completed without a schedule audit"
                    )
                events = adaptive_audit.get("events", [])
                if len(events) != 10:
                    raise RuntimeError(
                        f"expected 10 cache admission events, observed {len(events)}"
                    )
                final_event = adaptive_audit.get("final", {})
                capacity = int(args.temporal_adaptive_capacity_bit_chunks)
                terminal = (
                    temporal_profile.joint_schedule_terminal_bit_chunks
                    if args.temporal_adaptive_allocation_mode == "frozen_joint_schedule"
                    or temporal_profile.residual_capacity_by_step is not None
                    else capacity
                )
                final_sum = int(final_event.get("nominal_bits_sum", -1))
                if final_sum != terminal:
                    raise RuntimeError(
                        "terminal nominal residual budget is not "
                        f"{terminal}/{terminal}"
                    )
                prompt_dir = output_dir / f"prompt_{int(prompt_id):02d}"
                prompt_dir.mkdir(parents=True, exist_ok=False)
                schedule_path = prompt_dir / "chunk_precision_schedule.csv"
                schedule_columns = [
                    "step",
                    "current cache chunks",
                    *[f"chunk{index}" for index in range(10)],
                    "nominal bits sum",
                    "current average",
                ]
                with schedule_path.open("x", newline="", encoding="utf-8") as handle:
                    schedule_writer = csv.DictWriter(
                        handle, fieldnames=schedule_columns
                    )
                    schedule_writer.writeheader()
                    for event in events:
                        row = {
                            "step": event["step"],
                            "current cache chunks": event[
                                "current_cache_chunks"
                            ],
                            "nominal bits sum": event["nominal_bits_sum"],
                            "current average": event["current_average"],
                        }
                        row.update(
                            {
                                f"chunk{index}": (
                                    "—" if value is None else int(value)
                                )
                                for index, value in enumerate(event["schedule"])
                            }
                        )
                        schedule_writer.writerow(row)
                transition_path = (
                    prompt_dir / "chunk_precision_transitions.csv"
                )
                transition_columns = [
                    "step",
                    "chunk",
                    "old_bits",
                    "new_bits",
                    "age",
                    "weight",
                    "distortion_delta",
                    "released_nominal_bits",
                    "rvq_hash_before",
                    "rvq_hash_after",
                ]
                with transition_path.open(
                    "x", newline="", encoding="utf-8"
                ) as handle:
                    transition_writer = csv.DictWriter(
                        handle, fieldnames=transition_columns
                    )
                    transition_writer.writeheader()
                    transition_writer.writerows(
                        adaptive_audit.get("transitions", [])
                    )
                audit_path = (
                    output_dir
                    / f"{prompt_id}-{args.seed}-temporal_adaptive_audit.json"
                )
                with audit_path.open("x", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "schema_version": (
                                "wan5b_temporal_adaptive_residual.audit.v1"
                            ),
                            "prompt_id": prompt_id,
                            "prompt": prompt,
                            "seed": args.seed,
                            "terminal_budget_utilization": (
                                f"{final_sum}/{terminal}"
                            ),
                            "pipeline_audit": adaptive_audit,
                        },
                        handle,
                        indent=2,
                    )
            kv_cache_mem_mb = cache_metrics["kv_cache_mem_mb"]
            if parsed_metrics["logged_kv_cache_mem_mb"] is not None:
                if kv_cache_mem_mb is None:
                    kv_cache_mem_mb = parsed_metrics["logged_kv_cache_mem_mb"]
                else:
                    kv_cache_mem_mb = max(kv_cache_mem_mb, parsed_metrics["logged_kv_cache_mem_mb"])

            if rank == 0 and writer is not None and metrics_file is not None:
                row = {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "seed": args.seed,
                    "setting": args.setting,
                    "video_path": video_path_value,
                    "e2e_time_s": e2e_time_s,
                    "denoise_time_s": parsed_metrics["denoise_time_s"],
                    "quant_time_s": parsed_metrics["quant_time_s"],
                    "temporal_policy_time_s": parsed_metrics[
                        "temporal_policy_time_s"
                    ],
                    "full_encode_time_s": parsed_metrics["full_encode_time_s"],
                    "dequant_time_s": parsed_metrics["dequant_time_s"],
                    "dequant_calls": parsed_metrics["dequant_calls"],
                    "peak_gpu_mem_mb": peak_gpu_mem_mb,
                    "kv_cache_mem_mb": kv_cache_mem_mb,
                    "compression_ratio": cache_metrics["compression_ratio"],
                    "k_payload_mib": cache_metrics.get("k_payload_mib"),
                    "v_payload_mib": cache_metrics.get("v_payload_mib"),
                    "kv_payload_mib": cache_metrics.get("kv_payload_mib"),
                    "k_payload_bytes": cache_metrics.get("k_payload_bytes"),
                    "v_payload_bytes": cache_metrics.get("v_payload_bytes"),
                    "kv_payload_bytes": cache_metrics.get("kv_payload_bytes"),
                    "residual_k_payload_mib": cache_metrics.get("residual_k_payload_mib"),
                    "residual_v_payload_mib": cache_metrics.get("residual_v_payload_mib"),
                    "residual_kv_payload_mib": cache_metrics.get("residual_kv_payload_mib"),
                    "residual_k_payload_bytes": cache_metrics.get(
                        "residual_k_payload_bytes"
                    ),
                    "residual_v_payload_bytes": cache_metrics.get(
                        "residual_v_payload_bytes"
                    ),
                    "residual_kv_payload_bytes": cache_metrics.get(
                        "residual_kv_payload_bytes"
                    ),
                    "residual_compression_ratio_vs_bf16": cache_metrics.get(
                        "residual_compression_ratio_vs_bf16"
                    ),
                    "shared_codebook_mib": cache_metrics.get("shared_codebook_mib"),
                    "r2aq_shared_table_mib": cache_metrics.get(
                        "r2aq_shared_table_mib"
                    ),
                    "global_shared_mib": cache_metrics.get(
                        "global_shared_mib"
                    ),
                    "r2aq_shared_table_bytes": cache_metrics.get(
                        "r2aq_shared_table_bytes"
                    ),
                    "global_shared_bytes": cache_metrics.get(
                        "global_shared_bytes"
                    ),
                    "physical_kv_payload_bytes": cache_metrics.get(
                        "physical_kv_payload_bytes"
                    ),
                    "r2aq_shared_budget_owner_states": cache_metrics.get(
                        "r2aq_shared_budget_owner_states"
                    ),
                    "r2aq_shared_budget_reserved_local_bytes": (
                        cache_metrics.get(
                            "r2aq_shared_budget_reserved_local_bytes"
                        )
                    ),
                    "r2aq_shared_budget_reserved_logical_bytes": (
                        cache_metrics.get(
                            "r2aq_shared_budget_reserved_logical_bytes"
                        )
                    ),
                    "r2aq_shared_budget_exactly_repaid": cache_metrics.get(
                        "r2aq_shared_budget_exactly_repaid"
                    ),
                    "compression_ratio_excluding_shared_codebook": cache_metrics.get(
                        "compression_ratio_excluding_shared_codebook"
                    ),
                    "compression_ratio_including_shared_codebook": cache_metrics.get(
                        "compression_ratio_including_shared_codebook"
                    ),
                    "rvq_actual_avg_depth": cache_metrics.get("rvq_actual_avg_depth"),
                    "rvq_k_attention_kl": cache_metrics.get("rvq_k_attention_kl"),
                    "rvq_v_output_mse": cache_metrics.get("rvq_v_output_mse"),
                    "rvq_v_raw_mse": cache_metrics.get("rvq_v_raw_mse"),
                    "rvq_v_relative_l2": cache_metrics.get("rvq_v_relative_l2"),
                    "rvq_v_mean_vector_relative_squared_error": cache_metrics.get(
                        "rvq_v_mean_vector_relative_squared_error"
                    ),
                }
                writer.writerow({key: csv_value(row[key]) for key in CSV_COLUMNS})
                metrics_file.flush()

            clear_runner_kv_cache(runner)

            if world_size > 1:
                torch.distributed.barrier()
    finally:
        if metrics_file is not None:
            metrics_file.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
