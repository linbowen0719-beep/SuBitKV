#!/usr/bin/env python3
"""Aggregate Wan direct-codec exact shards into paper-level rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def summarize(frame: pd.DataFrame, prompt: int | str, scope: str) -> list[dict[str, object]]:
    rows = []
    for method, data in frame.groupby("method"):
        base = data.groupby(["prompt_id", "layer"], as_index=False).first()
        joint = data[data.intervention == "K_and_V"]
        k_only = data[data.intervention == "K_only"]
        v_only = data[data.intervention == "V_only"]
        rows.append({
            "model": "Wan2.2 5B", "method": method, "prompt": prompt,
            "exact_scope": scope,
            "prompts": base.prompt_id.nunique(), "layers": base.layer.nunique(),
            "K_mse": base.historical_K_mse.mean(),
            "V_mse": base.historical_V_mse.mean(),
            "K_relative_mse": base.historical_K_relative_mse.mean(),
            "V_relative_mse": base.historical_V_relative_mse.mean(),
            "qk_mse": joint.attention_logit_mse.mean(),
            "attention_kl": joint.attention_kl_reference_to_candidate.mean(),
            "topk_overlap": joint.attention_topk_overlap.mean(),
            "K_output_mse": k_only.attention_output_mse.mean(),
            "V_output_mse": v_only.attention_output_mse.mean(),
            "attention_output_mse": joint.attention_output_mse.mean(),
            "attention_output_p99": joint.attention_output_squared_error_P99.mean(),
            "attention_output_p999": joint.attention_output_squared_error_P999.mean(),
            "attention_output_cvar99": joint.attention_output_squared_error_CVaR99.mean(),
            "payload_K_mib": base.total_payload_K_MiB.mean(),
            "payload_V_mib": base.total_payload_V_MiB.mean(),
            "codec_ratio_K": base.joint_compression_ratio_K_vs_BF16.mean(),
            "codec_ratio_V": base.joint_compression_ratio_V_vs_BF16.mean(),
            "encode_K_s": base.total_encode_K_seconds.mean(),
            "encode_V_s": base.total_encode_V_seconds.mean(),
            "decode_K_s": base.total_decode_K_seconds.mean(),
            "decode_V_s": base.total_decode_V_seconds.mean(),
            "position_validation_max_abs": base.position_validation_max_abs.max(),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-layers", default=",".join(map(str, range(30))))
    args = parser.parse_args()
    complete = args.output.with_suffix(".complete.json")
    if args.output.exists() or complete.exists():
        raise FileExistsError("create-only Wan exact aggregate already exists")
    inputs = sorted(path for root in args.roots for path in root.glob("exact_shard_*.csv"))
    if not inputs:
        raise FileNotFoundError("no exact shards")
    frame = pd.concat([pd.read_csv(path) for path in inputs], ignore_index=True)
    expected_layers = tuple(int(value) for value in args.expected_layers.split(","))
    expected = {(prompt, layer) for prompt in range(10) for layer in expected_layers}
    observed = set(zip(frame.prompt_id.astype(int), frame.layer.astype(int)))
    if observed != expected:
        raise RuntimeError(f"exact coverage mismatch: missing={sorted(expected-observed)[:20]}")
    rows = []
    for prompt, data in frame.groupby("prompt_id"):
        rows.extend(summarize(
            data, int(prompt),
            f"late-history paired capture; layers={list(expected_layers)}; standard RoPE + camera PRoPE",
        ))
    rows.extend(summarize(
        frame, "aggregate",
        f"10 prompts x {len(expected_layers)} layers {list(expected_layers)}; late-history paired capture; standard RoPE + camera PRoPE",
    ))
    output = pd.DataFrame(rows)
    output.to_csv(args.output, index=False)
    complete.write_text(json.dumps({
        "schema_version": "wan2_2_5b.direct_exact.aggregate.v1",
        "inputs": {str(path): digest(path) for path in inputs},
        "raw_rows": len(frame), "aggregate_rows": len(output),
        "prompts": list(range(10)), "layers": list(expected_layers),
        "output_sha256": digest(args.output),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
