#!/usr/bin/env python3
"""Create-only same-input BF16 pre/post-RoPE equivalence evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


REPO = Path(__file__).resolve().parents[2]
SELF_FORCING = REPO / "experiments/Self-Forcing"
if str(SELF_FORCING) not in sys.path:
    sys.path.insert(0, str(SELF_FORCING))

from wan.modules.causal_model import (  # noqa: E402
    causal_rope_apply,
    causal_rope_apply_long_input,
)
from wan.modules.model import rope_params  # noqa: E402


OUTPUT = REPO / (
    "results/analysis/"
    "wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/"
    "self_forcing/bf16_position_equivalence_evidence.json"
)


def tensor_hash(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def source_hash(function) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cpu_before = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(20260803)
    batch, chunk_tokens, heads, dim = 1, 8, 4, 128
    grid = torch.tensor([[2, 2, 2]], dtype=torch.long)
    freqs = torch.cat([
        rope_params(32, dim - 4 * (dim // 6)),
        rope_params(32, 2 * (dim // 6)),
        rope_params(32, 2 * (dim // 6)),
    ], dim=1)
    key_pre = torch.randn(
        batch, 2 * chunk_tokens, heads, dim,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    value = torch.randn(
        batch, 2 * chunk_tokens, heads, dim,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)
    query = torch.randn(
        batch, chunk_tokens, heads, dim,
        generator=generator, dtype=torch.float32,
    ).to(torch.bfloat16)

    # Production post-RoPE storage transforms each admitted chunk once.
    key_post = torch.cat([
        causal_rope_apply(key_pre[:, :chunk_tokens], grid, freqs, start_frame=0),
        causal_rope_apply(key_pre[:, chunk_tokens:], grid, freqs, start_frame=2),
    ], dim=1)
    # Production pre-RoPE storage transforms the complete cache at restore.
    key_restored_from_pre = causal_rope_apply_long_input(key_pre, grid, freqs)
    query_post = causal_rope_apply(query, grid, freqs, start_frame=2)

    post_output = F.scaled_dot_product_attention(
        query_post.transpose(1, 2).float(),
        key_post.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
    )
    pre_output = F.scaled_dot_product_attention(
        query_post.transpose(1, 2).float(),
        key_restored_from_pre.transpose(1, 2).float(),
        value.transpose(1, 2).float(),
    )
    key_max_abs = float((key_post.float() - key_restored_from_pre.float()).abs().max())
    output_max_abs = float((post_output - pre_output).abs().max())
    cpu_unchanged = torch.equal(cpu_before, torch.random.get_rng_state())
    payload = {
        "schema_version": "self_forcing.bf16_position_equivalence.v1",
        "scope": "same deterministic two-chunk BF16 input; production causal RoPE functions",
        "seed": 20260803,
        "shape": {
            "batch": batch, "chunks": 2, "tokens_per_chunk": chunk_tokens,
            "heads": heads, "head_dim": dim,
        },
        "key_max_abs": key_max_abs,
        "attention_max_abs": output_max_abs,
        "cpu_rng_unchanged": cpu_unchanged,
        "pass": key_max_abs == 0.0 and output_max_abs == 0.0 and cpu_unchanged,
        "input_hashes": {
            "key_pre": tensor_hash(key_pre), "value": tensor_hash(value),
            "query": tensor_hash(query),
        },
        "source_hashes": {
            "causal_rope_apply": source_hash(causal_rope_apply),
            "causal_rope_apply_long_input": source_hash(causal_rope_apply_long_input),
        },
    }
    if not payload["pass"]:
        raise RuntimeError(payload)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
