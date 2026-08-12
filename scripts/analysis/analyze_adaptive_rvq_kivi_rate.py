#!/usr/bin/env python3
"""Build nominal-bit, physical-budget, and rate-matched RVQ+KIVI tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined(main: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # 30 layers x 24 heads x L16 x C64.  Each D128 centroid uses 64 INT4
    # bytes plus one FP16 scale.  K and V are separate shared tables.
    codebook_one_tensor_bytes = 30 * 16 * 24 * 64 * (128 / 2 + 2)
    prompts = int(raw.prompt.nunique())
    for method, data in main.groupby("method"):
        source = float(data.source_MiB.sum())
        local = float(data.local_payload_MiB.sum())
        codebook = (
            2 * prompts * codebook_one_tensor_bytes / 2**20
            if method.startswith("adaptive_rvq_")
            else 0.0
        )
        payload = local + codebook
        raw_method = raw[raw.method == method]
        values = int(raw_method["values"].sum())
        residual_bpp = (
            float(raw_method.residual_payload_bytes.sum()) * 8 / values
            if values else 0.0
        )
        row = {
            "method": method,
            "nominal_bits": int(raw_method.nominal_bits.iloc[0]),
            "K_mse": float(data[data.tensor == "K"].mse.iloc[0]),
            "V_mse": float(data[data.tensor == "V"].mse.iloc[0]),
            "K_normalized_mse": float(data[data.tensor == "K"].normalized_mse.iloc[0]),
            "V_normalized_mse": float(data[data.tensor == "V"].normalized_mse.iloc[0]),
            "residual_physical_bit_per_value": residual_bpp,
            "local_payload_MiB_10prompts": local,
            "codebook_MiB_10prompts": codebook,
            "joint_payload_MiB_10prompts": payload,
            "codec_only_compression_ratio_vs_D128_BF16": source / payload,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method")


def relative(new: float, reference: float) -> float:
    return (new / reference - 1.0) * 100.0


def pair_rows(table: pd.DataFrame, pairs: list[tuple[str, str]], kind: str) -> pd.DataFrame:
    by = table.set_index("method")
    rows = []
    for candidate, reference in pairs:
        c, r = by.loc[candidate], by.loc[reference]
        rows.append({
            "comparison": kind,
            "candidate": candidate,
            "reference": reference,
            "candidate_nominal_bits": c.nominal_bits,
            "reference_nominal_bits": r.nominal_bits,
            "candidate_residual_bpp": c.residual_physical_bit_per_value,
            "reference_residual_bpp": r.residual_physical_bit_per_value,
            "candidate_joint_payload_MiB": c.joint_payload_MiB_10prompts,
            "reference_joint_payload_MiB": r.joint_payload_MiB_10prompts,
            "payload_delta_percent": relative(c.joint_payload_MiB_10prompts, r.joint_payload_MiB_10prompts),
            "candidate_K_mse": c.K_mse,
            "reference_K_mse": r.K_mse,
            "K_mse_delta_percent": relative(c.K_mse, r.K_mse),
            "candidate_V_mse": c.V_mse,
            "reference_V_mse": r.V_mse,
            "V_mse_delta_percent": relative(c.V_mse, r.V_mse),
            "strict_within_2pct_payload": abs(relative(c.joint_payload_MiB_10prompts, r.joint_payload_MiB_10prompts)) <= 2,
        })
    return pd.DataFrame(rows)


def interpolation(table: pd.DataFrame) -> pd.DataFrame:
    by = table.set_index("method")
    kivi = table[table.method.str.startswith("adaptive_rvq_kivi")].sort_values("joint_payload_MiB_10prompts")
    rows = []
    for bits in (2, 3, 4):
        reference_name = f"adaptive_rvq_current_b{bits}"
        reference = by.loc[reference_name]
        target = float(reference.joint_payload_MiB_10prompts)
        lower = kivi[kivi.joint_payload_MiB_10prompts <= target].tail(1)
        upper = kivi[kivi.joint_payload_MiB_10prompts >= target].head(1)
        if lower.empty or upper.empty:
            continue
        lo, hi = lower.iloc[0], upper.iloc[0]
        if lo.method == hi.method:
            alpha = 0.0
        else:
            alpha = (target - lo.joint_payload_MiB_10prompts) / (hi.joint_payload_MiB_10prompts - lo.joint_payload_MiB_10prompts)
        row = {
            "reference": reference_name,
            "target_joint_payload_MiB_10prompts": target,
            "lower_kivi_method": lo.method,
            "upper_kivi_method": hi.method,
            "interpolation_alpha": alpha,
        }
        for metric in ("K_mse", "V_mse", "K_normalized_mse", "V_normalized_mse"):
            estimated = float(lo[metric] + alpha * (hi[metric] - lo[metric]))
            row[f"estimated_kivi_{metric}"] = estimated
            row[f"reference_{metric}"] = float(reference[metric])
            row[f"{metric}_delta_percent"] = relative(estimated, reference[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(per_prompt: pd.DataFrame, output: Path) -> None:
    rng = np.random.default_rng(20260803)
    by = per_prompt.set_index(["method", "tensor", "prompt"])
    rows = []
    for bits in (1, 2, 3, 4):
        candidate = f"adaptive_rvq_kivi_b{bits}_g32"
        reference = f"adaptive_rvq_current_b{bits}"
        for tensor in ("K", "V"):
            c = np.array([by.loc[(candidate, tensor, str(p)), "mse"] for p in range(10)], dtype=float)
            r = np.array([by.loc[(reference, tensor, str(p)), "mse"] for p in range(10)], dtype=float)
            improvement = 1.0 - c / r
            samples = np.array([improvement[rng.integers(0, 10, 10)].mean() for _ in range(10000)])
            rows.append({
                "candidate": candidate,
                "reference": reference,
                "tensor": tensor,
                "mean_relative_improvement": improvement.mean(),
                "prompt_wins": int((c < r).sum()),
                "prompts": 10,
                "bootstrap_ci_low": np.quantile(samples, 0.025),
                "bootstrap_ci_high": np.quantile(samples, 0.975),
            })
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    main_path = args.root / "kv_reconstruction_main.csv"
    raw_path = args.root / "kv_reconstruction_per_chunk.csv"
    prompt_path = args.root / "kv_reconstruction_per_prompt.csv"
    main_frame = pd.read_csv(main_path)
    raw = pd.read_csv(raw_path)
    table = combined(main_frame, raw)
    table.to_csv(args.root / "payload_accounting.csv", index=False)
    same_bits = [
        (f"adaptive_rvq_kivi_b{bits}_g32", f"adaptive_rvq_current_b{bits}")
        for bits in (1, 2, 3, 4)
    ]
    pair_rows(table, same_bits, "same_nominal_bits").to_csv(args.root / "same_nominal_bits.csv", index=False)
    physical = [
        ("adaptive_rvq_kivi_b1_g32", "adaptive_rvq_current_b2"),
        ("adaptive_rvq_kivi_b2_g32", "adaptive_rvq_current_b3"),
        ("adaptive_rvq_kivi_b3_g32", "adaptive_rvq_current_b4"),
    ]
    pair_rows(table, physical, "nearest_physical_budget").to_csv(args.root / "same_residual_budget.csv", index=False)
    interpolation(table).to_csv(args.root / "same_joint_rate.csv", index=False)
    bootstrap(pd.read_csv(prompt_path, dtype={"prompt": str}), args.root / "paired_bootstrap.csv")
    manifest = {
        "schema": "adaptive_rvq_kivi_rate_analysis.v1",
        "inputs": {str(path): digest(path) for path in (main_path, raw_path, prompt_path)},
        "outputs": {
            name: digest(args.root / name)
            for name in (
                "payload_accounting.csv", "same_nominal_bits.csv",
                "same_residual_budget.csv", "same_joint_rate.csv", "paired_bootstrap.csv",
            )
        },
        "interpolation": "piecewise linear in measured joint payload; no extrapolation",
    }
    (args.root / "rate_analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
