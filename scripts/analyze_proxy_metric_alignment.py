#!/usr/bin/env python3
"""Audit how KV/attention proxies align with closed-loop video metrics.

The analysis deliberately keeps unlike error protocols separate.  The
cross-codec correlation study only uses the 16 ``strict-v4`` rows and removes
the fixed effect of prefix bit depth before computing correlations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = (
    REPO
    / "results/analysis/structured_metric_master_table_20260723"
    / "master_comparison.csv"
)
DEFAULT_FORMAL = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "structured_residual_video_benchmark_v1_20260723"
)
DEFAULT_CRITICAL = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "structured_truncatable_critical_screen_v1_20260723"
)
DEFAULT_CLOSED_LOOP = (
    REPO
    / "results/analysis/structured_error_coherence_video10_b2_20260723"
)
DEFAULT_CODEBOOK = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "truncatable_residual_strict_final_20260723/training/codec_params.pt"
)
DEFAULT_OUTPUT = (
    REPO / "results/analysis/proxy_metric_alignment_final_20260723"
)

PROXIES = ["K_MSE", "V_MSE", "QK_logits_MSE", "attention_output_MSE"]
OUTCOMES = [
    "pixel_MSE",
    "PSNR",
    "SSIM",
    "LPIPS_normalized_v2",
    "vbench_mean_10d",
]
HIGHER_IS_BETTER = {
    "PSNR",
    "SSIM",
    "vbench_mean_10d",
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "human_action",
    "temporal_style",
    "overall_consistency",
    "end_to_end_compression_vs_bf16",
    "residual_compression_vs_bf16",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def gpu_type() -> str | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
        )
        names = sorted({line.strip() for line in output.splitlines() if line.strip()})
        return ", ".join(names) or None
    except Exception:
        return None


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")


def strict_protocol_analysis(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict = master.loc[master["error_protocol_id"].eq("strict-v4")].copy()
    numeric(strict, ["prefix_bits", *PROXIES, *OUTCOMES])
    if len(strict) != 16:
        raise RuntimeError(f"Expected 16 strict-v4 rows, found {len(strict)}")

    correlation_rows: list[dict[str, object]] = []
    for proxy in PROXIES:
        log_proxy = np.log(strict[proxy])
        proxy_within = log_proxy - log_proxy.groupby(strict["prefix_bits"]).transform(
            "mean"
        )
        for outcome in OUTCOMES:
            outcome_within = strict[outcome] - strict[outcome].groupby(
                strict["prefix_bits"]
            ).transform("mean")
            rho, p_value = spearmanr(proxy_within, outcome_within)
            aligned_rho = -rho if outcome in HIGHER_IS_BETTER else rho
            correlation_rows.append(
                {
                    "error_protocol_id": "strict-v4",
                    "rows": len(strict),
                    "bit_depth_fixed_effect": True,
                    "proxy": proxy,
                    "outcome": outcome,
                    "raw_spearman_rho": rho,
                    "loss_aligned_spearman_rho": aligned_rho,
                    "two_sided_p": p_value,
                    "interpretation": (
                        "positive loss-aligned rho means lower proxy error "
                        "associates with better outcome after removing bit depth"
                    ),
                }
            )

    concordance_rows: list[dict[str, object]] = []
    for bits, group in strict.groupby("prefix_bits", sort=True):
        for proxy in PROXIES:
            for outcome in OUTCOMES:
                aligned_outcome = (
                    -group[outcome]
                    if outcome in HIGHER_IS_BETTER
                    else group[outcome]
                )
                tau, p_value = kendalltau(group[proxy], aligned_outcome)
                concordance_rows.append(
                    {
                        "error_protocol_id": "strict-v4",
                        "prefix_bits": int(bits),
                        "methods": len(group),
                        "proxy": proxy,
                        "outcome": outcome,
                        "loss_aligned_kendall_tau": tau,
                        "two_sided_p": p_value,
                    }
                )
    return pd.DataFrame(correlation_rows), pd.DataFrame(concordance_rows)


def improvement_pct(candidate: float, baseline: float, metric: str) -> float:
    if not (math.isfinite(candidate) and math.isfinite(baseline)) or baseline == 0:
        return math.nan
    if metric in HIGHER_IS_BETTER:
        return 100.0 * (candidate / baseline - 1.0)
    return 100.0 * (1.0 - candidate / baseline)


def formal_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    by_name = metrics.set_index("setting")
    pairs = [
        (
            "structured_b2_vs_rvq_qv2",
            "l8c64_rvq_residual_structured_gaussian_h1r1_b2",
            "l8c64_rvq_residual_qv_int2",
        ),
        (
            "structured_b4_vs_rvq_qv4",
            "l8c64_rvq_residual_structured_gaussian_h1r1_b4",
            "l8c64_rvq_residual_qv_int4",
        ),
        (
            "rvq_qv2_vs_direct_qv2",
            "l8c64_rvq_residual_qv_int2",
            "direct_quantvideo_int2",
        ),
        (
            "rvq_qv4_vs_direct_qv4",
            "l8c64_rvq_residual_qv_int4",
            "direct_quantvideo_int4",
        ),
        (
            "structured_b2_vs_direct_qv2",
            "l8c64_rvq_residual_structured_gaussian_h1r1_b2",
            "direct_quantvideo_int2",
        ),
        (
            "structured_b4_vs_direct_qv4",
            "l8c64_rvq_residual_structured_gaussian_h1r1_b4",
            "direct_quantvideo_int4",
        ),
    ]
    vbench = [
        "subject_consistency",
        "background_consistency",
        "temporal_flickering",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
        "human_action",
        "temporal_style",
        "overall_consistency",
    ]
    compare_metrics = [
        "mse",
        "psnr",
        "ssim",
        "lpips",
        *vbench,
        "quant_mean_s",
        "kv_cache_mem_mib",
        "end_to_end_compression_vs_bf16",
    ]
    rows: list[dict[str, object]] = []
    for contrast, candidate, baseline in pairs:
        c_row, b_row = by_name.loc[candidate], by_name.loc[baseline]
        c_vbench = float(np.mean([float(c_row[x]) for x in vbench]))
        b_vbench = float(np.mean([float(b_row[x]) for x in vbench]))
        rows.append(
            {
                "contrast": contrast,
                "candidate": candidate,
                "baseline": baseline,
                "metric": "vbench_mean_10d",
                "candidate_value": c_vbench,
                "baseline_value": b_vbench,
                "candidate_improvement_pct": improvement_pct(
                    c_vbench, b_vbench, "vbench_mean_10d"
                ),
            }
        )
        for metric in compare_metrics:
            candidate_value = float(c_row[metric])
            baseline_value = float(b_row[metric])
            canonical = {
                "mse": "pixel_MSE",
                "psnr": "PSNR",
                "ssim": "SSIM",
                "lpips": "LPIPS_normalized_v2",
            }.get(metric, metric)
            rows.append(
                {
                    "contrast": contrast,
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": canonical,
                    "candidate_value": candidate_value,
                    "baseline_value": baseline_value,
                    "candidate_improvement_pct": improvement_pct(
                        candidate_value, baseline_value, canonical
                    ),
                }
            )
    return pd.DataFrame(rows)


def proxy_video_gap(
    critical_pairs: pd.DataFrame, formal: pd.DataFrame
) -> pd.DataFrame:
    mapping = {
        2: (
            "structured_gaussian_h1r1_b2",
            "structured_b2_vs_rvq_qv2",
        ),
        4: (
            "structured_gaussian_h1r1_b4",
            "structured_b4_vs_rvq_qv4",
        ),
    }
    rows: list[dict[str, object]] = []
    local_metrics = {
        "K_MSE": "local_K_MSE",
        "V_MSE": "local_V_MSE",
        "QK_MSE": "local_QK_MSE",
        "QK_abs_p999": "local_QK_abs_p999",
        "attention_KL": "local_attention_KL",
        "QKV_output_MSE": "local_attention_output_MSE",
    }
    video_metrics = [
        "pixel_MSE",
        "PSNR",
        "SSIM",
        "LPIPS_normalized_v2",
        "vbench_mean_10d",
    ]
    for bits, (candidate, contrast) in mapping.items():
        selected = critical_pairs.loc[critical_pairs["candidate"].eq(candidate)]
        for source_metric, label in local_metrics.items():
            row = selected.loc[selected["metric"].eq(source_metric)]
            if len(row) != 1:
                raise RuntimeError(
                    f"Missing critical metric {source_metric} for {candidate}"
                )
            rows.append(
                {
                    "prefix_bits": bits,
                    "scope": "teacher_forced_local_critical_screen",
                    "metric": label,
                    "improvement_pct": float(row.iloc[0]["improvement_pct"]),
                }
            )
        selected_formal = formal.loc[formal["contrast"].eq(contrast)]
        for metric in video_metrics:
            row = selected_formal.loc[selected_formal["metric"].eq(metric)]
            if len(row) != 1:
                raise RuntimeError(f"Missing formal metric {metric} for {contrast}")
            rows.append(
                {
                    "prefix_bits": bits,
                    "scope": "closed_loop_video_10_prompt",
                    "metric": metric,
                    "improvement_pct": float(
                        row.iloc[0]["candidate_improvement_pct"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def render_readme(
    strict_corr: pd.DataFrame,
    formal: pd.DataFrame,
    gap: pd.DataFrame,
    closed_loop: pd.DataFrame,
) -> str:
    def formal_value(contrast: str, metric: str) -> float:
        row = formal.loc[
            formal["contrast"].eq(contrast) & formal["metric"].eq(metric)
        ]
        return float(row.iloc[0]["candidate_improvement_pct"])

    proxy_lines = []
    for proxy in PROXIES:
        for outcome in ["pixel_MSE", "LPIPS_normalized_v2", "vbench_mean_10d"]:
            row = strict_corr.loc[
                strict_corr["proxy"].eq(proxy)
                & strict_corr["outcome"].eq(outcome)
            ].iloc[0]
            proxy_lines.append(
                f"| {proxy} | {outcome} | "
                f"{row['loss_aligned_spearman_rho']:.4f} | "
                f"{row['two_sided_p']:.4g} |"
            )

    contrast_lines = []
    for label, contrast in [
        ("Structured B2 / RVQ+QV2", "structured_b2_vs_rvq_qv2"),
        ("Structured B4 / RVQ+QV4", "structured_b4_vs_rvq_qv4"),
        ("RVQ+QV2 / Direct QV2", "rvq_qv2_vs_direct_qv2"),
        ("RVQ+QV4 / Direct QV4", "rvq_qv4_vs_direct_qv4"),
    ]:
        contrast_lines.append(
            "| "
            + label
            + " | "
            + " | ".join(
                f"{formal_value(contrast, metric):+.3f}%"
                for metric in [
                    "pixel_MSE",
                    "PSNR",
                    "SSIM",
                    "LPIPS_normalized_v2",
                    "vbench_mean_10d",
                ]
            )
            + " |"
        )

    drift = closed_loop.iloc[0]
    b2_local = gap.loc[
        gap["prefix_bits"].eq(2)
        & gap["metric"].isin(
            ["local_K_MSE", "local_V_MSE", "local_attention_output_MSE"]
        )
    ]
    local_text = ", ".join(
        f"{row.metric} {row.improvement_pct:.2f}%"
        for row in b2_local.itertuples()
    )
    return f"""# KV/attention 代理指标与闭环视频指标对齐审计

结论先行：K/V 重建 MSE 本身是弱代理；同协议旧 codec 中，QK 与
attention-output MSE 对 PSNR/SSIM/LPIPS 的排序明显更可靠，但对
VBench-10 几乎没有解释力。Structured B2 则展示了更强的
teacher-forced/闭环错位：局部 {local_text}，正式 10-prompt pixel MSE
反而劣化 {abs(formal_value('structured_b2_vs_rvq_qv2', 'pixel_MSE')):.2f}%。

## 同协议、去除 bit 深度后的相关性

以下只使用 `strict-v4` 的 16 行；先在每个 bit 内中心化，再算 Spearman。
`rho>0` 表示代理越低，最终 loss 越好。样本仍小，p 值仅作诊断。

| Proxy | Outcome | loss-aligned rho | p |
|---|---|---:|---:|
{chr(10).join(proxy_lines)}

QK/attention-output 对 Similarity 的相关性强，但 VBench 相关性接近零。
因此仅把 K/V MSE 压低，或仅在单步 attention 上优化，不足以宣称最终
视频质量会提升。

## 正式 10-prompt 闭环对照

正数表示候选更好；Structured 与对应 RVQ+QuantVideo 是精确同 payload。

| 对照 | Pixel MSE | PSNR | SSIM | LPIPS | VBench-10 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(contrast_lines)}

B2 时间漂移审计还发现 Structured/QV2 pixel-error ratio 的 chunk slope
为 `{float(drift['mean']):+.5f}`，95% CI
`[{float(drift['bootstrap_ci95_low']):+.5f}, {float(drift['bootstrap_ci95_high']):+.5f}]`。
即问题不是初始压缩精度，而是误差随 rollout 后期累积。

## 可执行的 metric-aligned gate

1. 先用 K/V MSE 排除明显失败，但不用于最终晋级。
2. 同时要求 QK/KL、attention-output、p99/p999、age-center shift 与
   late-history/CVaR 不退化；mean 好而 late/CVaR 差直接淘汰。
3. 代理必须在量化闭环状态上重放，而不是只在 BF16 teacher-forced
   capture 上测一次；训练集选策略，10 个测试 prompt 只做一次确认。
4. Similarity 与 VBench 分开优化。VBench 当前两维恒定且
   `temporal_style=overall_consistency`，10D 均值会稀释变化；必须同时
   报每一维和 7 个独立变化信号。
5. 只有同 payload、held-out 10/10 prompt、bootstrap CI 与 tail gate
   同时通过，才进入完整视频；避免按测试 prompt 挑 schedule 的指标过拟合。

## 文件

- `strict_within_bit_correlations.csv`
- `strict_per_bit_concordance.csv`
- `formal_comparisons.csv`
- `proxy_video_gap.csv`
- `sources_manifest.json`
- `run_info.json`
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--critical-root", type=Path, default=DEFAULT_CRITICAL)
    parser.add_argument("--closed-loop-root", type=Path, default=DEFAULT_CLOSED_LOOP)
    parser.add_argument("--codebook", type=Path, default=DEFAULT_CODEBOOK)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    sources = [
        args.master.resolve(),
        (args.formal_root / "tables/metrics.csv").resolve(),
        (args.critical_root / "paired_vs_quantvideo.csv").resolve(),
        (args.closed_loop_root / "drift_paired_bootstrap.csv").resolve(),
        args.codebook.resolve(),
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    master = pd.read_csv(args.master, keep_default_na=False)
    metrics = pd.read_csv(args.formal_root / "tables/metrics.csv")
    critical_pairs = pd.read_csv(args.critical_root / "paired_vs_quantvideo.csv")
    drift = pd.read_csv(args.closed_loop_root / "drift_paired_bootstrap.csv")

    strict_corr, strict_concordance = strict_protocol_analysis(master)
    formal = formal_comparisons(metrics)
    gap = proxy_video_gap(critical_pairs, formal)

    output.mkdir(parents=True)
    config = {
        "schema_version": "proxy_metric_alignment.v1",
        "strict_protocol": "strict-v4 only",
        "strict_rows_expected": 16,
        "fixed_effect": "prefix_bits",
        "formal_prompts": list(range(10)),
        "bootstrap_source": "precomputed closed-loop prompt bootstrap",
        "metric_accounting_changed": False,
    }
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    strict_corr.to_csv(output / "strict_within_bit_correlations.csv", index=False)
    strict_concordance.to_csv(
        output / "strict_per_bit_concordance.csv", index=False
    )
    formal.to_csv(output / "formal_comparisons.csv", index=False)
    gap.to_csv(output / "proxy_video_gap.csv", index=False)
    (output / "README_zh.md").write_text(
        render_readme(strict_corr, formal, gap, drift)
    )

    manifest = {
        str(path): {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in sources
    }
    (output / "sources_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    config_path = output / "config.json"
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(config_path),
        "config_hash": sha256(config_path),
        "codebook_path": str(args.codebook.resolve()),
        "codebook_hash": sha256(args.codebook.resolve()),
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type(),
        "seed": 20260723,
        "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_device": "CPU; no codec/video/VBench executed",
    }
    (output / "run_info.json").write_text(
        json.dumps(run_info, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
