"""Opt-in BF16 teacher calibration at the actual positioned attention boundary.

Hooks never replace attention outputs. Reconstruction is performed in cache
coordinates, then the model's own RoPE/ProPE transform is replayed. K and V
are perturbed separately, including unchanged current/text tokens in softmax.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

_SETTINGS = None
_SEEN = set()


def enabled():
    return bool(os.environ.get("SUBITKV_CALIBRATION_SETTINGS"))


def settings():
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = json.loads(
            Path(os.environ["SUBITKV_CALIBRATION_SETTINGS"]).read_text()
        )
    return _SETTINGS


def base(source, layer, tensor, cfg, bundle):
    import torch

    from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache

    state = rvq_encode_tensor_cache(
        source,
        layer_idx=layer,
        codebook_dir=bundle / f"nonorm_{tensor.lower()}_l16c64_codebooks",
        batch_size=cfg["assignment_batch_size"],
        compute_dtype="bf16",
        codebook_storage_dtype=cfg["codebook_storage_dtype"],
        scale_storage_dtype=cfg["scale_storage_dtype"],
        relative_stop=True,
        relative_stop_threshold=cfg["early_stop"]["threshold"],
        relative_stop_min_depth=cfg["early_stop"]["min_depth"],
        relative_stop_max_depth=cfg["early_stop"]["max_depth"],
    )
    return rvq_dequantize_tensor_cache(state, torch.bfloat16)


def candidates(source, layer, tensor, cfg, bundle, spatial, span_tokens):
    import torch

    from subitkv.frame_mean_gaussian import (
        decode_frame_mean_gaussian_residual,
        encode_frame_mean_gaussian_residual,
        truncate_frame_mean_gaussian_state,
    )
    from subitkv.progressive_sign_adaptive_absmax_scalar import (
        decode_progressive_sign_adaptive_absmax_scalar_residual,
        encode_progressive_sign_adaptive_absmax_scalar_residual,
        truncate_progressive_sign_adaptive_absmax_state,
    )

    bits = (8, 4, 3, 2, 1, 0)
    outputs = {b: [] for b in bits}
    tree = torch.load(
        bundle / "tree_params.pt", map_location=source.device, weights_only=False
    )
    for chunk in source.split(span_tokens, dim=2):
        reconstructed = base(chunk, layer, tensor, cfg, bundle)
        residual = (chunk.float() - reconstructed.float()).to(chunk.dtype)
        if tensor == "K":
            packed = encode_frame_mean_gaussian_residual(
                residual,
                layer_idx=layer,
                bits=8,
                routing_table=bundle / "routing.json",
                tree_params=tree,
                seed=cfg["residual"]["codec_seed"],
                spatial_tokens=spatial,
                use_fast=False,
            )

            def decode(b):
                return decode_frame_mean_gaussian_residual(
                    packed if b == 8 else truncate_frame_mean_gaussian_state(packed, b),
                    source.dtype,
                    use_fast=False,
                )
        else:
            packed = encode_progressive_sign_adaptive_absmax_scalar_residual(
                residual,
                params=bundle / "v_params.pt",
                method="midpoint_msb",
                max_bits=8,
                group_size=cfg["residual"]["group_size"],
                use_fast=False,
            )

            def decode(b):
                return decode_progressive_sign_adaptive_absmax_scalar_residual(
                    packed
                    if b == 8
                    else truncate_progressive_sign_adaptive_absmax_state(packed, b),
                    source.dtype,
                    use_fast=False,
                )

        for b in bits:
            # Runtime's zero-bit mode retains the RVQ base and discards residuals.
            outputs[b].append(
                reconstructed
                if b == 0
                else (reconstructed + decode(b)).to(source.dtype)
            )
    return {b: torch.cat(parts, dim=2) for b, parts in outputs.items()}


def observe(query, history_k, history_v, *, layer, transform, prefix_tokens=0):
    """All inputs use BHSD. transform(history K,V) returns complete positioned K,V."""
    if not enabled() or history_k is None or not history_k.shape[2]:
        return
    import torch

    from subitkv.rng import preserve_torch_rng_state

    raw = settings()
    config, phase = raw["config"], raw["phase"]
    m, c = config["model_config"], config["calibration"]
    spatial, span_tokens = (
        m["spatial_tokens"],
        m["spatial_tokens"] * m["latent_chunk_frames"],
    )
    tokens = history_k.shape[2]
    if tokens % span_tokens:
        raise ValueError(
            f"Calibration history has {tokens} tokens, not whole {span_tokens}-token chunks"
        )
    spans = tokens // span_tokens
    if not 1 <= spans <= m["calibration_history_steps"]:
        return
    key = (phase, int(layer), int(spans))
    if key in _SEEN or (
        phase == "residual" and spans != m["calibration_history_steps"]
    ):
        return
    _SEEN.add(key)
    root, bundle = Path(raw["output"]), Path(raw["bundle"])
    root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), preserve_torch_rng_state(query.device):
        if phase == "residual":
            residuals = {}
            base_sse, base_count = {}, {}
            for tensor, source in (("K", history_k), ("V", history_v)):
                rebuilt = torch.cat(
                    [
                        base(part, layer, tensor, c, bundle)
                        for part in source.split(span_tokens, dim=2)
                    ],
                    dim=2,
                )
                residual = (source.float() - rebuilt.float()).to(torch.bfloat16)
                base_sse[tensor], base_count[tensor] = (
                    float(residual.double().square().sum()),
                    residual.numel(),
                )
                if tensor == "K":
                    shape = (
                        *residual.shape[:2],
                        tokens // spatial,
                        spatial,
                        residual.shape[-1],
                    )
                    frames = residual.float().reshape(shape)
                    mean = frames.mean(3).to(torch.float8_e4m3fn).float()
                    residual = (
                        (frames - mean.unsqueeze(3))
                        .reshape_as(residual)
                        .to(torch.bfloat16)
                    )
                # Equal deterministic samples per head, across streams and complete frames.
                heads = residual.permute(1, 0, 2, 3).flatten(1, 2)
                generator = torch.Generator(device="cpu").manual_seed(
                    c["seed"] + layer * 1009 + raw["prompt_id"] * 1000003
                )
                indices = torch.randperm(heads.shape[1], generator=generator)[
                    : c["residual"]["samples_per_head"]
                ].to(heads.device)
                residuals[tensor] = heads[:, indices].cpu()
            RESIDUAL_STATS_SCHEMA = "hyworldplay_rvq_residual_stats.v2"

            torch.save(
                dict(
                    schema_version=RESIDUAL_STATS_SCHEMA,
                    K_centered=residuals["K"],
                    V_residual=residuals["V"],
                    base_sse=base_sse,
                    base_count=base_count,
                ),
                root / f"layer_{layer}.pt",
            )
            return
        generator = torch.Generator(device="cpu").manual_seed(
            c["weights"]["seed"] + layer * 1009 + spans * 9176
        )
        indices = torch.randperm(query.shape[2], generator=generator)[
            : c["weights"]["queries_per_stream"]
        ].to(query.device)
        q = query.index_select(2, indices).float()
        k, v = transform(history_k, history_v)
        scale = 1 / math.sqrt(q.shape[-1])
        logp = (q @ k.float().transpose(-1, -2) * scale).log_softmax(-1)
        probability = logp.exp()
        reference = probability @ v.float()
        denominator = reference.double().square().sum().clamp_min(1e-20)
        age_weights = [
            float(
                probability[
                    ...,
                    prefix_tokens + i * span_tokens : prefix_tokens
                    + (i + 1) * span_tokens,
                ]
                .sum(-1)
                .mean()
            )
            for i in reversed(range(spans))
        ]
        distortions = {}
        for tensor, source in (("K", history_k), ("V", history_v)):
            distortion = {}
            for bits, rebuilt in candidates(
                source, layer, tensor, c, bundle, spatial, span_tokens
            ).items():
                ck, cv = transform(
                    rebuilt if tensor == "K" else history_k,
                    rebuilt if tensor == "V" else history_v,
                )
                altered_logp = (
                    (q @ ck.float().transpose(-1, -2) * scale).log_softmax(-1)
                    if tensor == "K"
                    else logp
                )
                output = altered_logp.exp() @ cv.float()
                nmse = float(
                    (output.double() - reference.double()).square().sum() / denominator
                )
                kl = (
                    max(
                        0.0, float((probability * (logp - altered_logp)).sum(-1).mean())
                    )
                    if tensor == "K"
                    else 0.0
                )
                distortion[str(bits)] = dict(output_nmse=nmse, kl=kl, loss=nmse + kl)
            distortions[tensor] = distortion
        from .common import write_json

        write_json(
            root / f"layer_{layer}_span_{spans}.json",
            dict(
                layer=layer,
                spans=spans,
                age_weights=age_weights,
                distortions=distortions,
                split="train",
                test_prompts_read=False,
            ),
        )
