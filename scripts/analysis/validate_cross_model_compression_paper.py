#!/usr/bin/env python3
"""Validate final cross-model paper artifacts and preregistered hard gates."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803"
REPORT = REPO / "docs/wan2_2_5b_adaptive_rvq_progressive_kv_compression_paper_zh.md"
REQUIRED = (
    "closed_loop_main.csv", "per_prompt.csv", "exact_kv_attention.csv",
    "turboquant_setting_ablation.csv", "qv_position_compare.csv", "runtime.csv",
    "memory_accounting.csv", "pareto_peak_points.csv", "pareto_peak_frontier.csv",
    "pareto_terminal_points.csv", "pareto_terminal_frontier.csv",
    "reused_ablation_evidence.csv", "first_chunk_gate.csv",
    "bf16_position_equivalence.csv", "evidence_manifest.json", "run_info.json",
)


def aggregate(exact: pd.DataFrame) -> pd.DataFrame:
    if "prompt" in exact:
        return exact[exact.prompt.astype(str) == "aggregate"]
    return exact


def main() -> None:
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    if missing or not REPORT.is_file():
        raise FileNotFoundError({"missing": missing, "report": REPORT.is_file()})
    closed = pd.read_csv(ROOT / "closed_loop_main.csv")
    expected = 20  # BF16 + 2 QV positions + KIVI + Turbo + 9 RVQ points.
    counts = closed.groupby("model").method.nunique().to_dict()
    if any(value != expected for value in counts.values()):
        raise RuntimeError(f"closed-loop method coverage mismatch: {counts}")
    gate = pd.read_csv(ROOT / "first_chunk_gate.csv")
    if not gate["pass"].astype(bool).all():
        raise RuntimeError("first-uncompressed-chunk gate failed")
    equivalence = pd.read_csv(ROOT / "bf16_position_equivalence.csv")
    if not equivalence["pass"].astype(bool).all():
        raise RuntimeError("BF16 pre/post attention equivalence failed")
    exact = aggregate(pd.read_csv(ROOT / "exact_kv_attention.csv"))
    for model in exact.model.unique():
        data = exact[exact.model == model].set_index("method")
        for low, high in (
            ("kivi_all_int2", "kivi_all_int4"),
            ("turboquant_mse_k3_v2", "turboquant_mse_k4_v3"),
            ("turboquant_prod_k3_v2", "turboquant_prod_k4_v3"),
        ):
            if not (data.loc[high, ["K_mse", "V_mse"]]
                    <= data.loc[low, ["K_mse", "V_mse"]]).all():
                raise RuntimeError(f"monotonic exact gate failed: {model} {low}->{high}")
    text = REPORT.read_text(encoding="utf-8")
    if "\\[" in text or "\\]" in text:
        raise RuntimeError("report contains forbidden bracket display math")
    for target in re.findall(r"!\[[^]]*\]\(([^)]+)\)", text):
        if not (REPORT.parent / target).resolve().is_file():
            raise FileNotFoundError(f"broken report image: {target}")
    print({"status": "pass", "method_counts": counts,
           "first_chunk_rows": len(gate), "exact_rows": len(exact)})


if __name__ == "__main__":
    main()
