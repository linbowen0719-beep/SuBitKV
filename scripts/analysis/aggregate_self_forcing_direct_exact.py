#!/usr/bin/env python3
"""Aggregate the create-only 10-prompt Self-Forcing direct-codec audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803/self_forcing/exact_audit10")
METHODS = (
    "quantvideo_prerope_int2", "quantvideo_prerope_int4",
    "quantvideo_postrope_int2", "quantvideo_postrope_int4",
    "kivi_all_int2", "kivi_all_int4", "turboquant_mse_k3_v2",
    "turboquant_mse_k4_v3", "turboquant_prod_k3_v2",
    "turboquant_prod_k4_v3",
    "turboquant_mse_k2_v2", "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2", "turboquant_prod_k4_v4",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    methods = tuple(value for value in args.methods.split(",") if value)
    unknown = set(methods).difference(METHODS)
    if not methods or unknown:
        raise ValueError(f"invalid methods: {sorted(unknown)}")
    output = args.root / "exact_kv_attention.csv"
    complete = args.root / "complete.json"
    if output.exists() or complete.exists():
        raise FileExistsError("create-only exact aggregate already exists")
    rows = []
    inputs = {}
    for method in methods:
        prompt_rows = []
        for prompt in range(10):
            path = args.root / method / f"prompt_{prompt:04d}_seed_0/metrics.json"
            payload = json.loads(path.read_text())
            audit = payload["kv_reconstruction_audit"]
            summary = audit["summary"]
            attention = pd.DataFrame(audit["attention_rows"])
            weights = attention["query_count"].to_numpy(dtype=np.float64)
            weighted = {
                field: float(np.average(attention[field], weights=weights))
                for field in ("logit_mse", "attention_kl", "top32_overlap",
                              "k_only_output_mse", "v_only_output_mse",
                              "joint_output_mse")
            }
            prompt_rows.append({
                "model": "Self-Forcing", "method": method, "prompt": prompt,
                "exact_scope": "full 10-admission lifecycle reconstruction audit",
                "K_mse": summary["K"]["raw_mse"],
                "V_mse": summary["V"]["raw_mse"],
                "K_relative_mse": summary["K"]["relative_squared_error"],
                "V_relative_mse": summary["V"]["relative_squared_error"],
                "qk_mse": weighted["logit_mse"],
                "attention_kl": weighted["attention_kl"],
                "topk_overlap": weighted["top32_overlap"],
                "K_output_mse": weighted["k_only_output_mse"],
                "V_output_mse": weighted["v_only_output_mse"],
                "attention_output_mse": weighted["joint_output_mse"],
                "attention_rows": len(attention),
            })
            inputs[str(path)] = digest(path)
        frame = pd.DataFrame(prompt_rows)
        aggregate = frame.mean(numeric_only=True).to_dict()
        aggregate.update({"model": "Self-Forcing", "method": method,
                          "prompt": "aggregate", "prompts": 10,
                          "layers": 30})
        rows.extend(prompt_rows)
        rows.append(aggregate)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    complete.write_text(json.dumps({
        "schema_version": "self_forcing.direct_exact.aggregate.v1",
        "rows": len(frame), "methods": list(methods), "prompts": list(range(10)),
        "layers": list(range(30)), "inputs": inputs,
        "output_sha256": digest(output),
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
