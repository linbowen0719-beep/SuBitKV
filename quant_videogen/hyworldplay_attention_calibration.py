"""Train-only attention calibration for HY-WorldPlay avg2 RVQ schedules."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from .hyworldplay8b_cache_codec import (
    AVG2_RESIDUAL_BITS,
    encode_avg2_prefix_candidates,
    make_codec_settings,
)
from .rng import preserve_torch_rng_state
from .uncompress import uncompress_kv_cache


CALIBRATION_SCHEMA = "hyworldplay_rvq_attention_calibration.event.v2"


def _query_indices(length: int, *, layer: int, spans: int, count: int, seed: int) -> torch.Tensor:
    take = min(int(count), int(length))
    generator = torch.Generator(device="cpu").manual_seed(
        int(seed) + int(layer) * 1009 + int(spans) * 9176
    )
    return torch.randperm(int(length), generator=generator)[:take]


def _normalized_mse(actual: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (actual.float() - reference.float()).square().double().sum()
    denominator = reference.float().square().double().sum().clamp_min(1e-20)
    return float((numerator / denominator).item())


@torch.no_grad()
def calibrate_attention_once(
    *,
    query: torch.Tensor,
    history_k: torch.Tensor,
    history_v: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    text_k: torch.Tensor,
    text_v: torch.Tensor,
    layer_idx: int,
    state: Any,
) -> dict[str, Any] | None:
    """Measure exact sampled-query attention damage for one rebuilt history."""

    if not getattr(state, "kv_avg2_calibration_dir", ""):
        return None
    spatial = int(state.kv_codec_frame_spatial_tokens)
    span_tokens = spatial * 4
    if int(history_k.shape[2]) % span_tokens:
        raise ValueError("HY calibration history is not a whole number of spans")
    spans = int(history_k.shape[2]) // span_tokens
    identity = (int(layer_idx), spans)
    if identity in state.kv_avg2_calibration_seen:
        return None
    if not 1 <= spans <= 9:
        raise ValueError(f"HY calibration expected 1..9 history spans, got {spans}")
    if any(
        tensor.ndim != 4 or int(tensor.shape[-1]) != 128
        for tensor in (query, history_k, history_v, current_k, current_v, text_k, text_v)
    ):
        raise ValueError("HY attention calibration requires BHSD D128 tensors")
    if int(query.shape[0]) != 2 or int(history_k.shape[0]) != 2:
        raise ValueError("HY attention calibration requires RoPE and ProPE streams")
    started = time.perf_counter()
    settings = make_codec_settings(
        "rvq_ours_avg2",
        k_rvq_codebook_dir=state.kv_codec_k_codebook_dir,
        v_rvq_codebook_dir=state.kv_codec_v_codebook_dir,
        routed_gaussian_routing_path=state.kv_codec_routing_path,
        routed_gaussian_tree_params_path=state.kv_codec_tree_path,
        progressive_v_absmax_params_path=state.kv_codec_v_params_path,
        temporal_profile_path="",
        frame_spatial_tokens=spatial,
        allow_missing_temporal_profile_for_calibration=True,
    )
    indices = _query_indices(
        int(query.shape[2]),
        layer=int(layer_idx),
        spans=spans,
        count=int(state.kv_avg2_calibration_queries_per_stream),
        seed=int(state.kv_avg2_calibration_seed),
    ).to(query.device)
    sampled_q = query.index_select(2, indices).float()
    all_k = torch.cat((text_k, history_k, current_k), dim=2)
    all_v = torch.cat((text_v, history_v, current_v), dim=2)
    scale = 1.0 / math.sqrt(int(query.shape[-1]))
    reference_logits = torch.matmul(sampled_q, all_k.float().transpose(-1, -2)) * scale
    reference_log_prob = reference_logits.log_softmax(dim=-1)
    reference_prob = reference_log_prob.exp()
    reference_output = torch.matmul(reference_prob, all_v.float())
    text_tokens = int(text_k.shape[2])
    age_mass = [0.0] * spans
    for span_id in range(spans):
        start = text_tokens + span_id * span_tokens
        end = start + span_tokens
        age = spans - 1 - span_id
        age_mass[age] = float(reference_prob[..., start:end].sum(-1).mean().item())
    candidates = encode_avg2_prefix_candidates(
        history_k,
        history_v,
        layer_idx=int(layer_idx),
        settings=settings,
    )
    metrics: dict[str, dict[str, float]] = {"K": {}, "V": {}}
    raw: dict[str, dict[str, dict[str, float]]] = {"K": {}, "V": {}}
    with preserve_torch_rng_state(query.device):
        for bits in AVG2_RESIDUAL_BITS:
            candidate_k, candidate_v = uncompress_kv_cache(*candidates[bits])
            k_all = torch.cat((text_k, candidate_k, current_k), dim=2)
            candidate_logits = torch.matmul(
                sampled_q, k_all.float().transpose(-1, -2)
            ) * scale
            candidate_log_prob = candidate_logits.log_softmax(dim=-1)
            kl = float(
                (
                    reference_prob
                    * (reference_log_prob - candidate_log_prob)
                )
                .sum(-1)
                .mean()
                .item()
            )
            k_output = torch.matmul(candidate_log_prob.exp(), all_v.float())
            k_output_nmse = _normalized_mse(k_output, reference_output)
            v_all = torch.cat((text_v, candidate_v, current_v), dim=2)
            v_output = torch.matmul(reference_prob, v_all.float())
            v_output_nmse = _normalized_mse(v_output, reference_output)
            raw["K"][str(bits)] = {
                "attention_kl": kl,
                "normalized_output_mse": k_output_nmse,
            }
            raw["V"][str(bits)] = {"normalized_output_mse": v_output_nmse}
            metrics["K"][str(bits)] = kl + k_output_nmse
            metrics["V"][str(bits)] = v_output_nmse
            del candidate_k, candidate_v, k_all, candidate_logits, candidate_log_prob
            del k_output, v_all, v_output
    event = {
        "schema_version": CALIBRATION_SCHEMA,
        "layer": int(layer_idx),
        "history_spans": spans,
        "query_indices": indices.cpu().tolist(),
        "queries_per_stream": int(indices.numel()),
        "attention_streams": ["rope", "prope"],
        "includes_real_text_kv": True,
        "age_attention_mass": age_mass,
        "distortion": metrics,
        "raw_metrics": raw,
        "seconds": time.perf_counter() - started,
    }
    state.kv_avg2_calibration_seen.add(identity)
    state.kv_avg2_calibration_events.append(event)
    state.add_part_seconds("rvq_attention_calibration", float(event["seconds"]))
    return event


__all__ = ["CALIBRATION_SCHEMA", "calibrate_attention_once"]
