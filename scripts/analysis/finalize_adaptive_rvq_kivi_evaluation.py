#!/usr/bin/env python3
"""Consolidate the fair Adaptive-RVQ/KIVI/QV evaluation and report section."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
FORMAL = (
    REPO
    / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803"
    / "wan2_2_5b/formal10_v2"
)
CURRENT = REPO / "results/analysis/adaptive_rvq_progressive_int8_video10_final_20260731"
CURRENT_MAP = {
    "adaptive_uniform_progressive_b1": "adaptive_rvq_current_b1",
    "adaptive_uniform_progressive_b2": "adaptive_rvq_current_b2",
    "adaptive_uniform_progressive_b3": "adaptive_rvq_current_b3",
    "adaptive_uniform_progressive": "adaptive_rvq_current_b4",
}
QV_RECON_MAP = {
    "direct_quantvideo_int2": "quantvideo_prerope_int2",
    "direct_quantvideo_int4": "quantvideo_prerope_int4",
    "direct_kivi_int2": "kivi_all_int2",
    "direct_kivi_int4": "kivi_all_int4",
}
LABELS = {
    "bf16": "BF16",
    "kivi_all_int2": "Direct KIVI INT2",
    "kivi_all_int4": "Direct KIVI INT4",
    "quantvideo_prerope_int2": "Direct QuantVideo INT2",
    "quantvideo_prerope_int4": "Direct QuantVideo INT4",
    "adaptive_rvq_only": "Adaptive RVQ only",
    "adaptive_rvq_kivi_b1_g32": "Adaptive RVQ + KIVI B1",
    "adaptive_rvq_kivi_b2_g32": "Adaptive RVQ + KIVI B2",
    "adaptive_rvq_kivi_b3_g32": "Adaptive RVQ + KIVI B3",
    "adaptive_rvq_kivi_b4_g32": "Adaptive RVQ + KIVI B4",
    "adaptive_rvq_current_b1": "Adaptive RVQ + current B1",
    "adaptive_rvq_current_b2": "Adaptive RVQ + current B2",
    "adaptive_rvq_current_b3": "Adaptive RVQ + current B3",
    "adaptive_rvq_current_b4": "Adaptive RVQ + current B4",
    "adaptive_rvq_quantvideo_int2": "Adaptive RVQ + QuantVideo INT2",
    "adaptive_rvq_quantvideo_int4": "Adaptive RVQ + QuantVideo INT4",
}
QUALITY_COLUMNS = ["psnr", "ssim", "lpips", "vbench10_mean"]
VBENCH_COLUMNS = [
    "vbench_subject_consistency", "vbench_background_consistency",
    "vbench_temporal_flickering", "vbench_motion_smoothness",
    "vbench_dynamic_degree", "vbench_aesthetic_quality",
    "vbench_imaging_quality", "vbench_human_action",
    "vbench_temporal_style", "vbench_overall_consistency",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        expected = frame.to_csv(index=False)
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"refusing to overwrite non-identical {path}")
        return
    frame.to_csv(path, index=False)


def quality_table(root: Path) -> pd.DataFrame:
    sources = []
    formal = pd.read_csv(FORMAL / "summary.csv")
    sources.append(
        formal[formal.method.isin([
            "bf16", "kivi_all_int2", "kivi_all_int4",
            "quantvideo_prerope_int2", "quantvideo_prerope_int4",
        ])]
    )
    current = pd.read_csv(CURRENT / "summary.csv")
    current = current[current.method.isin(CURRENT_MAP)].copy()
    current["method"] = current.method.map(CURRENT_MAP)
    sources.append(current)
    sources.append(pd.read_csv(root / "closed_loop/summary.csv"))
    sources.append(pd.read_csv(root / "closed_loop_qv_controls/summary.csv"))
    table = pd.concat(sources, ignore_index=True)
    table = table.drop_duplicates("method", keep="last")
    keep = ["method", *QUALITY_COLUMNS, *VBENCH_COLUMNS]
    table = table[keep].copy()
    table.insert(1, "method_label", table.method.map(LABELS))
    if table.method_label.isna().any():
        raise RuntimeError(f"unlabelled quality methods: {table[table.method_label.isna()].method.tolist()}")
    return table.sort_values("method")


def similarity_sources(root: Path) -> pd.DataFrame:
    frames = []
    roots_and_methods = [
        (FORMAL, [
            "kivi_all_int2", "kivi_all_int4", "quantvideo_prerope_int2",
            "quantvideo_prerope_int4",
        ]),
        (CURRENT, list(CURRENT_MAP)),
        (root / "closed_loop", [f"adaptive_rvq_kivi_b{b}_g32" for b in range(1, 5)]),
        (root / "closed_loop_qv_controls", [
            "adaptive_rvq_quantvideo_int2", "adaptive_rvq_quantvideo_int4",
        ]),
    ]
    for source_root, methods in roots_and_methods:
        for method in methods:
            frame = pd.read_csv(source_root / "similarity" / f"{method}.csv")
            frame["method"] = CURRENT_MAP.get(method, method)
            frames.append(frame[["method", "prompt_id", "psnr", "ssim", "lpips"]])
    return pd.concat(frames, ignore_index=True).sort_values(["method", "prompt_id"])


def payload_from_reconstruction(main_path: Path, raw_path: Path) -> pd.DataFrame:
    main = pd.read_csv(main_path)
    raw = pd.read_csv(raw_path)
    prompts = int(raw.prompt.nunique())
    codebook_one_tensor_bytes = 30 * 16 * 24 * 64 * (128 / 2 + 2)
    rows = []
    for method, data in main.groupby("method"):
        raw_method = raw[raw.method == method]
        source = float(data.source_MiB.sum())
        local = float(data.local_payload_MiB.sum())
        codebook = (
            2 * prompts * codebook_one_tensor_bytes / 2**20
            if method.startswith("adaptive_rvq_")
            else 0.0
        )
        payload = local + codebook
        values = int(raw_method["values"].sum())
        residual_bpp = float(raw_method.residual_payload_bytes.sum()) * 8 / values
        by_tensor = data.set_index("tensor")
        rows.append({
            "method": QV_RECON_MAP.get(method, method),
            "nominal_bits": int(raw_method.nominal_bits.iloc[0]),
            "residual_physical_bit_per_value": residual_bpp,
            "K_mse": float(by_tensor.loc["K", "mse"]),
            "V_mse": float(by_tensor.loc["V", "mse"]),
            "K_normalized_mse": float(by_tensor.loc["K", "normalized_mse"]),
            "V_normalized_mse": float(by_tensor.loc["V", "normalized_mse"]),
            "terminal_payload_MiB_per_prompt": payload / prompts,
            "codec_ratio_vs_D128_BF16": source / payload,
            "codebook_MiB_per_prompt": codebook / prompts,
        })
    return pd.DataFrame(rows)


def payload_table(root: Path) -> pd.DataFrame:
    primary = payload_from_reconstruction(
        root / "kv_reconstruction_main.csv",
        root / "kv_reconstruction_per_chunk.csv",
    )
    qv = payload_from_reconstruction(
        root / "qv_reconstruction_all/kv_reconstruction_main.csv",
        root / "qv_reconstruction_all/kv_reconstruction_per_chunk.csv",
    )
    table = pd.concat([primary, qv], ignore_index=True)
    table = table.drop_duplicates("method", keep="last")
    # The source tensor size is shared by all methods. BF16 has no codec state.
    source_frame = pd.read_csv(root / "kv_reconstruction_main.csv")
    source_per_prompt = (
        source_frame[source_frame.method == "adaptive_rvq_only"].source_MiB.sum()
        / 10
    )
    bf16 = pd.DataFrame([{
        "method": "bf16", "nominal_bits": 16,
        "residual_physical_bit_per_value": 16.0,
        "K_mse": 0.0, "V_mse": 0.0,
        "K_normalized_mse": 0.0, "V_normalized_mse": 0.0,
        "terminal_payload_MiB_per_prompt": source_per_prompt,
        "codec_ratio_vs_D128_BF16": 1.0, "codebook_MiB_per_prompt": 0.0,
    }])
    table = pd.concat([bf16, table], ignore_index=True)
    table.insert(1, "method_label", table.method.map(LABELS))
    return table.sort_values("terminal_payload_MiB_per_prompt")


def pareto(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in QUALITY_COLUMNS:
        lower_quality_is_better = metric == "lpips"
        candidates = points.dropna(subset=[metric, "terminal_payload_MiB_per_prompt"])
        for row in candidates.itertuples(index=False):
            dominated = False
            for other in candidates.itertuples(index=False):
                memory_ok = other.terminal_payload_MiB_per_prompt <= row.terminal_payload_MiB_per_prompt
                quality_ok = (
                    getattr(other, metric) <= getattr(row, metric)
                    if lower_quality_is_better
                    else getattr(other, metric) >= getattr(row, metric)
                )
                strict = (
                    other.terminal_payload_MiB_per_prompt < row.terminal_payload_MiB_per_prompt
                    or getattr(other, metric) != getattr(row, metric)
                )
                if memory_ok and quality_ok and strict:
                    dominated = True
                    break
            if not dominated:
                rows.append({"metric": metric, **row._asdict()})
    return pd.DataFrame(rows)


def pair_table(
    payload: pd.DataFrame,
    quality: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    kind: str,
) -> pd.DataFrame:
    payload_by = payload.set_index("method")
    records = []
    for candidate, reference in comparisons:
        c, r = payload_by.loc[candidate], payload_by.loc[reference]
        records.append({
            "comparison": kind, "candidate": candidate, "reference": reference,
            "candidate_nominal_bits": c.nominal_bits,
            "reference_nominal_bits": r.nominal_bits,
            "candidate_residual_bpp": c.residual_physical_bit_per_value,
            "reference_residual_bpp": r.residual_physical_bit_per_value,
            "candidate_terminal_MiB": c.terminal_payload_MiB_per_prompt,
            "reference_terminal_MiB": r.terminal_payload_MiB_per_prompt,
            "payload_delta_percent": (
                c.terminal_payload_MiB_per_prompt
                / r.terminal_payload_MiB_per_prompt - 1
            ) * 100,
            "candidate_K_mse": c.K_mse, "reference_K_mse": r.K_mse,
            "K_mse_delta_percent": (c.K_mse / r.K_mse - 1) * 100,
            "candidate_V_mse": c.V_mse, "reference_V_mse": r.V_mse,
            "V_mse_delta_percent": (c.V_mse / r.V_mse - 1) * 100,
            "strict_within_2pct_payload": abs(
                (c.terminal_payload_MiB_per_prompt
                 / r.terminal_payload_MiB_per_prompt - 1) * 100
            ) <= 2,
        })
    pairs = pd.DataFrame(records)
    by = quality.set_index("method")
    for metric in QUALITY_COLUMNS:
        pairs[f"candidate_{metric}"] = pairs.candidate.map(by[metric])
        pairs[f"reference_{metric}"] = pairs.reference.map(by[metric])
        if metric == "lpips":
            pairs[f"{metric}_improvement_percent"] = (
                1 - pairs[f"candidate_{metric}"] / pairs[f"reference_{metric}"]
            ) * 100
        else:
            pairs[f"{metric}_delta"] = (
                pairs[f"candidate_{metric}"] - pairs[f"reference_{metric}"]
            )
    return pairs


def joint_rate_quality(root: Path, payload: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    kivi = payload[payload.method.str.startswith("adaptive_rvq_kivi")].sort_values(
        "terminal_payload_MiB_per_prompt"
    )
    payload_by = payload.set_index("method")
    quality_by = quality.set_index("method")
    rows = []
    for bits in (2, 3, 4):
        reference = f"adaptive_rvq_current_b{bits}"
        target = payload_by.loc[reference, "terminal_payload_MiB_per_prompt"]
        lo = kivi[kivi.terminal_payload_MiB_per_prompt <= target].tail(1)
        hi = kivi[kivi.terminal_payload_MiB_per_prompt >= target].head(1)
        if lo.empty or hi.empty:
            continue
        lo, hi = lo.iloc[0], hi.iloc[0]
        alpha = (
            0.0 if lo.method == hi.method else
            (target - lo.terminal_payload_MiB_per_prompt)
            / (hi.terminal_payload_MiB_per_prompt - lo.terminal_payload_MiB_per_prompt)
        )
        row = {
            "reference": reference, "target_terminal_MiB": target,
            "lower_kivi_method": lo.method, "upper_kivi_method": hi.method,
            "interpolation_alpha": alpha,
        }
        for metric in ["K_mse", "V_mse"]:
            estimate = lo[metric] + alpha * (hi[metric] - lo[metric])
            row[f"estimated_kivi_{metric}"] = estimate
            row[f"reference_{metric}"] = payload_by.loc[reference, metric]
        for metric in QUALITY_COLUMNS:
            estimate = (
                quality_by.loc[lo.method, metric]
                + alpha * (quality_by.loc[hi.method, metric] - quality_by.loc[lo.method, metric])
            )
            row[f"estimated_kivi_{metric}"] = estimate
            row[f"reference_{metric}"] = quality_by.loc[reference, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def quality_bootstrap(per_prompt: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260803)
    by = per_prompt.set_index(["method", "prompt_id"])
    rows = []
    for bits in range(1, 5):
        candidate = f"adaptive_rvq_kivi_b{bits}_g32"
        reference = f"adaptive_rvq_current_b{bits}"
        for metric in ("psnr", "ssim", "lpips"):
            c = np.array([by.loc[(candidate, p), metric] for p in range(10)])
            r = np.array([by.loc[(reference, p), metric] for p in range(10)])
            improvement = r - c if metric == "lpips" else c - r
            samples = np.array([
                improvement[rng.integers(0, 10, 10)].mean() for _ in range(10000)
            ])
            rows.append({
                "candidate": candidate, "reference": reference, "metric": metric,
                "mean_directional_improvement": improvement.mean(),
                "prompt_wins": int((improvement > 0).sum()), "prompts": 10,
                "bootstrap_ci_low": np.quantile(samples, 0.025),
                "bootstrap_ci_high": np.quantile(samples, 0.975),
            })
    return pd.DataFrame(rows)


def build_report_section(
    root: Path,
    quality: pd.DataFrame,
    payload: pd.DataFrame,
    exact: pd.DataFrame,
    same_nominal: pd.DataFrame,
    same_physical: pd.DataFrame,
    same_joint: pd.DataFrame,
) -> str:
    runtime = pd.read_csv(
        REPO
        / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803"
        / "runtime_30layer_estimate.csv"
    )[["method", "encode_30layer_estimate_s", "decode_30layer_estimate_s"]]
    runtime_labels = {
        "direct_qv_int2": "Direct QuantVideo INT2",
        "direct_qv_int4": "Direct QuantVideo INT4",
        **{f"adaptive_uniform_b{b}": f"Adaptive RVQ + Uniform B{b}" for b in (1, 2, 3, 4, 8)},
        **{f"adaptive_temporal_avg{b}": f"Adaptive RVQ + Time-adaptive avg{b}" for b in (1, 2, 3, 4)},
    }
    runtime["method"] = runtime.method.map(runtime_labels)
    workspace = pd.read_csv(
        REPO
        / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803"
        / "runtime_workspace_audit.csv"
    )[["method", "peak_allocated_delta_mib_mean"]].copy()
    workspace["method"] = workspace.method.map(runtime_labels)
    exact_aggregate = exact[exact.prompt.astype(str) == "aggregate"].copy()
    exact_aggregate["method_label"] = exact_aggregate.method.map(LABELS)
    exact_view = exact_aggregate[[
        "method_label", "K_mse", "V_mse", "qk_mse", "attention_kl",
        "topk_overlap", "K_output_mse", "V_output_mse",
        "attention_output_mse", "attention_output_cvar99",
    ]]
    payload_view = payload[[
        "method_label", "nominal_bits", "residual_physical_bit_per_value",
        "K_mse", "V_mse", "terminal_payload_MiB_per_prompt",
        "codec_ratio_vs_D128_BF16",
    ]]
    quality_view = quality[["method_label", *QUALITY_COLUMNS]]
    b4 = same_nominal[same_nominal.candidate == "adaptive_rvq_kivi_b4_g32"].iloc[0]
    physical = same_physical[
        same_physical.candidate == "adaptive_rvq_kivi_b3_g32"
    ].iloc[0]
    text = f"""
## 18. 最高效Runtime复测与RVQ+KIVI完整评估（2026-08-03）

本节使用pre-RoPE/pre-PRoPE D128 BF16 KV。重建与exact-attention实验均覆盖10个test prompts、全部30层、每层10个独立历史chunk；exact-attention分别重放standard RoPE与camera PRoPE。闭环为seed 42、173帧、一chunk一次压缩。所有RVQ residual方法共享Adaptive L4–16停止规则；首个未压缩chunk必须在latent、raw VAE decode和pre-MP4 RGB上与BF16逐值相同。

### 18.1 当前最高效实现的公平runtime

下表为两次独立`3 prompts × 4 layers × 10 chunks`实测均值，再乘`30/4`得到30层估计；JIT已预热，decode按正式生成调用乘数重放，不是一次真实30层测量。

{runtime.to_markdown(index=False, floatfmt='.4f')}

峰值workspace采用“相对已驻留源tensor的CUDA allocated增量”口径，包含编码state和restore输出；该审计独立于计时运行，避免显存采样扰动runtime：

{workspace.to_markdown(index=False, floatfmt='.3f')}

strict staged与monolithic near-tie审计在prompts `0/4/9`、层`0/10/20/29`、全部10 chunks上均保持100%停止深度一致；source MSE相对变化绝对值最大不超过`2.82e-7`，因此主runtime采用near-tie admission和融合Triton restore。

### 18.2 D128真实payload与30层KV重建

`residual physical bit/value`包含真实indices、FP16 scale/offset及padding；联合payload另含RVQ indices、depth map和完整30层×24头的INT4 codebook。K/V codebook合计`92.8125 MiB`，不做跨prompt摊销。

{payload_view.to_markdown(index=False, floatfmt='.6g')}

同名义B4时，RVQ+KIVI相对当前B4的K/V MSE分别变化`{b4.K_mse_delta_percent:+.2f}%/{b4.V_mse_delta_percent:+.2f}%`，但payload增加`{b4.payload_delta_percent:+.2f}%`。真正接近物理预算的RVQ+KIVI B3对当前B4，payload变化`{physical.payload_delta_percent:+.2f}%`，K/V MSE却变化`{physical.K_mse_delta_percent:+.2f}%/{physical.V_mse_delta_percent:+.2f}%`。因此不能用“同名义bit”替代“同真实字节”。

### 18.3 Exact attention

K-only和V-only独立执行；`K_output_mse`、`V_output_mse`分别禁止K/V误差抵消，`attention_output_mse/CVaR99`为联合恢复。

{exact_view.to_markdown(index=False, floatfmt='.6g')}

### 18.4 10-prompt闭环质量与VBench-10

不报告单独RGB MSE；PSNR、SSIM、LPIPS和VBench-10均保留全部10 prompts，不筛除非单调轨迹。

{quality_view.to_markdown(index=False, floatfmt='.6g')}

### 18.5 三种预算口径

同名义bit：

{same_nominal.to_markdown(index=False, floatfmt='.6g')}

最近物理residual预算；超过2%者只标记为最近实测点：

{same_physical.to_markdown(index=False, floatfmt='.6g')}

同联合payload仅在KIVI相邻实测点包围目标时分段线性插值，不外推：

{same_joint.to_markdown(index=False, floatfmt='.6g')}

完整逐prompt、逐层、逐chunk数据、bootstrap CI、Pareto点和原始VBench维度位于`results/analysis/adaptive_rvq_kivi_rate_matched_20260803/`。旧的4个代表层exact结果保留为`exact_attention_all_methods_representative4.csv`，正文采用新增全30层审计结果。
""".strip() + "\n"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    quality = quality_table(root)
    write_csv(root / "closed_loop_quality.csv", quality[["method", "method_label", *QUALITY_COLUMNS]])
    write_csv(root / "vbench.csv", quality[["method", "method_label", "vbench10_mean", *VBENCH_COLUMNS]])
    per_prompt = similarity_sources(root)
    write_csv(root / "closed_loop_quality_per_prompt.csv", per_prompt)
    write_csv(root / "closed_loop_quality_bootstrap.csv", quality_bootstrap(per_prompt))

    payload = payload_table(root)
    write_csv(root / "payload_accounting_all_methods.csv", payload)
    write_csv(root / "payload_accounting.csv", payload)
    points = payload.merge(quality, on=["method", "method_label"], how="left")
    write_csv(root / "pareto_points.csv", points)
    write_csv(root / "pareto_frontier.csv", pareto(points))
    same_nominal = [
        (f"adaptive_rvq_kivi_b{bits}_g32", f"adaptive_rvq_current_b{bits}")
        for bits in range(1, 5)
    ]
    same_nominal_frame = pair_table(payload, quality, same_nominal, "same_nominal_bits")
    write_csv(
        root / "same_nominal_bits_all_metrics.csv",
        same_nominal_frame,
    )
    write_csv(root / "same_nominal_bits.csv", same_nominal_frame)
    same_physical = [
        ("adaptive_rvq_kivi_b1_g32", "adaptive_rvq_current_b2"),
        ("adaptive_rvq_kivi_b2_g32", "adaptive_rvq_current_b3"),
        ("adaptive_rvq_kivi_b3_g32", "adaptive_rvq_current_b4"),
    ]
    same_physical_frame = pair_table(payload, quality, same_physical, "nearest_physical_budget")
    write_csv(
        root / "same_residual_budget_all_metrics.csv",
        same_physical_frame,
    )
    write_csv(root / "same_residual_budget.csv", same_physical_frame)
    same_joint_frame = joint_rate_quality(root, payload, quality)
    write_csv(root / "same_joint_rate_all_metrics.csv", same_joint_frame)
    write_csv(root / "same_joint_rate.csv", same_joint_frame)

    exact = pd.read_csv(root / "exact_attention_full30.csv")
    write_csv(root / "exact_attention_all_methods.csv", exact)

    report_section = build_report_section(
        root, quality, payload, exact, same_nominal_frame,
        same_physical_frame, same_joint_frame,
    )
    (root / "report_section.md").write_text(report_section, encoding="utf-8")

    gates = []
    for subroot in (root / "closed_loop", root / "closed_loop_qv_controls"):
        gate = subroot / "full_run_first_chunk_gate.csv"
        if gate.is_file():
            frame = pd.read_csv(gate)
            frame["source_root"] = str(subroot)
            gates.append(frame)
    write_csv(root / "first_chunk_gate.csv", pd.concat(gates, ignore_index=True))

    outputs = [
        "closed_loop_quality.csv", "vbench.csv", "closed_loop_quality_per_prompt.csv",
        "closed_loop_quality_bootstrap.csv", "payload_accounting_all_methods.csv",
        "payload_accounting.csv",
        "pareto_points.csv", "pareto_frontier.csv",
        "same_nominal_bits_all_metrics.csv", "same_residual_budget_all_metrics.csv",
        "same_joint_rate_all_metrics.csv", "exact_attention_all_methods.csv",
        "same_nominal_bits.csv", "same_residual_budget.csv", "same_joint_rate.csv",
        "first_chunk_gate.csv", "report_section.md",
    ]
    input_paths = [
        root / "kv_reconstruction_main.csv",
        root / "kv_reconstruction_per_chunk.csv",
        root / "qv_reconstruction_all/kv_reconstruction_main.csv",
        root / "qv_reconstruction_all/kv_reconstruction_per_chunk.csv",
        root / "exact_attention_full30.csv",
        root / "closed_loop/summary.csv",
        root / "closed_loop_qv_controls/summary.csv",
        FORMAL / "summary.csv",
        CURRENT / "summary.csv",
        REPO / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803/runtime_30layer_estimate.csv",
        REPO / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803/runtime_backend_audit.csv",
        REPO / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803/runtime_workspace_audit.csv",
        REPO / "quant_videogen/kivi.py",
        REPO / "quant_videogen/compress.py",
        REPO / "quant_videogen/uncompress.py",
    ]
    input_paths.extend(
        root / "exact_attention_full30_units" / f"run_info_shard_{shard:03d}.json"
        for shard in range(8)
    )
    input_paths.extend(
        root / "exact_capture_full30_view" / f"prompt_{prompt:02d}" / "run_info.json"
        for prompt in range(10)
    )
    input_paths.append(root / "exact_attention.complete.json")
    manifest = {
        "schema": "adaptive_rvq_kivi_complete_evaluation.v1",
        "inputs": {str(path): sha256(path) for path in input_paths},
        "outputs": {name: sha256(root / name) for name in outputs},
        "quality_protocol": "Wan2.2 5B; 10 prompts; seed 42; 173 frames; one chunk per compression",
        "memory_protocol": "actual D128 pre-position tensor payload; 10 history chunks; codebook included per prompt",
        "attention_scope": "10 prompts x all 30 layers; standard RoPE + camera PRoPE",
        "interpolation": "piecewise-linear only between bracketing measured KIVI points; no extrapolation",
    }
    (root / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
