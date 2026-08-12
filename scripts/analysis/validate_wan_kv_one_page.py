#!/usr/bin/env python3
"""Validate the one-page Wan2.2 KV summary against audited result CSVs."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs/wan2_2_5b_adaptive_rvq_progressive_kv_one_page_zh.md"
CROSS_ROOT = (
    REPO
    / "results/analysis/wan2_2_5b_adaptive_rvq_progressive_cross_model_paper_20260803"
)
RUNTIME_ROOT = REPO / "results/analysis/adaptive_rvq_optimized_runtime_fair_20260803"

COMPRESSOR_ROWS = {
    "我们的Uniform B1": "adaptive_uniform_progressive_b1",
    "我们的Uniform B2": "adaptive_uniform_progressive_b2",
    "我们的Uniform B3": "adaptive_uniform_progressive_b3",
    "我们的Uniform B4": "adaptive_uniform_progressive",
    "TurboQuant KV-MSE 2.5-bit": "turboquant_mse_k3_v2",
    "TurboQuant K-prod/V-MSE 2.5-bit": "turboquant_prod_k3_v2",
    "KIVI-style INT4": "kivi_all_int4",
    "QuantVideo pre-RoPE INT4": "quantvideo_prerope_int4",
}
HIGHEST_RATE_ROWS = {
    "我们的Time-adaptive avg1": "adaptive_temporal_int8_avg1",
    "TurboQuant KV-MSE 2.5-bit": "turboquant_mse_k3_v2",
    "KIVI-style INT2": "kivi_all_int2",
    "QuantVideo pre-RoPE INT2": "quantvideo_prerope_int2",
}
UNIFORM_BY_BIT = {
    1: "adaptive_uniform_progressive_b1",
    2: "adaptive_uniform_progressive_b2",
    3: "adaptive_uniform_progressive_b3",
    4: "adaptive_uniform_progressive",
}
TEMPORAL_BY_BIT = {
    bit: f"adaptive_temporal_int8_avg{bit}" for bit in range(1, 5)
}
RUNTIME_ROWS = {
    "QuantVideo INT4": "direct_qv_int4",
    "我们的Uniform B4": "adaptive_uniform_b4",
    "我们的Time-adaptive avg4": "adaptive_temporal_avg4",
}
CROSS_RUNTIME_ROWS = {
    "KIVI-style INT2": "kivi_all_int2",
    "KIVI-style INT4": "kivi_all_int4",
    "TurboQuant KV-MSE 2.5-bit": "turboquant_mse_k3_v2",
    "TurboQuant KV-MSE 3.5-bit": "turboquant_mse_k4_v3",
    "TurboQuant K-prod/V-MSE 2.5-bit": "turboquant_prod_k3_v2",
    "TurboQuant K-prod/V-MSE 3.5-bit": "turboquant_prod_k4_v3",
}


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: {actual} != {expected} within {tolerance}")


def numeric(value: str) -> float:
    value = value.replace("×", "").replace("**", "").replace("`", "").strip()
    return float(value)


def parse_slash_metrics(value: str) -> tuple[float, float, float, float]:
    parts = [numeric(item) for item in value.split("/")]
    if len(parts) != 4:
        raise AssertionError(f"expected ratio/PSNR/SSIM/LPIPS cell: {value}")
    return tuple(parts)  # type: ignore[return-value]


def markdown_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-", ":"}:
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


def unique_row(rows: list[list[str]], label: str, width: int) -> list[str]:
    matches = [row for row in rows if row and row[0].replace("**", "") == label and len(row) == width]
    if len(matches) != 1:
        raise AssertionError(f"expected one {width}-column row for {label}, found {len(matches)}")
    return matches[0]


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    rows = markdown_rows(text)

    if text.count("$$") != 2:
        raise AssertionError("one-page summary must contain exactly one display formula")
    if any(ord(char) < 9 or 13 < ord(char) < 32 for char in text):
        raise AssertionError("Markdown contains control characters")
    headings = [
        "## 方法简介",
        "## 我们实际跑了什么",
        "## 结果一：不加Time-adaptive，压缩器本身更好",
        "## 结果二：加上Time-adaptive，同一个压缩器更有用",
        "## Pareto与最高压缩率",
        "## Encode/decode速度：encode已能领先，decode仍是限制",
    ]
    positions = [text.index(heading) for heading in headings]
    if positions != sorted(positions):
        raise AssertionError("one-page narrative sections are out of order")
    if (
        "系统压缩率" in text
        or "系统率" in text
        or "D256系统缓存" in text
        or "20.115×" in text
    ):
        raise AssertionError("displayed compression ratios must use only the D128 codec-only basis")
    if text.count("codec-only压缩率") != 4:
        raise AssertionError("all three displayed ratio tables must name the codec-only basis")

    quality = pd.read_csv(CROSS_ROOT / "closed_loop_main.csv")
    quality = quality[quality.model == "Wan2.2 5B"].set_index("method")
    memory = pd.read_csv(CROSS_ROOT / "memory_accounting.csv")
    memory = memory[memory.model == "Wan2.2 5B"].set_index("method")
    runtime = pd.read_csv(RUNTIME_ROOT / "runtime_30layer_estimate.csv").set_index("method")
    workspace = pd.read_csv(RUNTIME_ROOT / "runtime_workspace_audit.csv").set_index("method")
    cross_runtime = pd.read_csv(CROSS_ROOT / "runtime.csv")
    cross_runtime = cross_runtime[cross_runtime.model == "Wan2.2 5B"].set_index("method")

    bf16_mib = float(memory.loc["bf16", "terminal_kv_mib"])
    if bf16_mib != 24750.0:
        raise AssertionError(f"unexpected Wan system BF16 baseline: {bf16_mib}")

    # Every formal Wan pre-position point must use the D256 system baseline and
    # the D128 codec-only baseline consistently.
    audited_methods = set(COMPRESSOR_ROWS.values()) | set(HIGHEST_RATE_ROWS.values())
    audited_methods |= set(UNIFORM_BY_BIT.values()) | set(TEMPORAL_BY_BIT.values())
    for method in audited_methods:
        mem = memory.loc[method]
        close(
            float(mem.system_ratio),
            bf16_mib / float(mem.terminal_kv_mib),
            5e-6,
            f"{method} system ratio denominator",
        )
        close(
            float(mem.codec_only_ratio),
            float(mem.system_ratio) / 2.0,
            5e-6,
            f"{method} D128 codec-only denominator",
        )

    # Static compressor evidence table.
    for label, method in COMPRESSOR_ROWS.items():
        row = unique_row(rows, label, 6)
        displayed_ratio = numeric(row[1])
        mem = memory.loc[method]
        q = quality.loc[method]
        close(displayed_ratio, float(mem.codec_only_ratio), 5.1e-4, f"{label} D128 ratio")
        close(numeric(row[2]), float(q.psnr), 5.1e-5, f"{label} PSNR")
        close(numeric(row[3]), float(q.ssim), 5.1e-5, f"{label} SSIM")
        close(numeric(row[4]), float(q.lpips), 5.1e-5, f"{label} LPIPS")
        close(numeric(row[5]), float(q.vbench10), 5.1e-7, f"{label} VBench")

    # Uniform versus temporal uses identical terminal budgets at each nominal bit.
    for bit in range(1, 5):
        row = unique_row(rows, f"{bit} bit", 3)
        uniform_values = parse_slash_metrics(row[1])
        temporal_values = parse_slash_metrics(row[2])
        for values, method, name in (
            (uniform_values, UNIFORM_BY_BIT[bit], "uniform"),
            (temporal_values, TEMPORAL_BY_BIT[bit], "temporal"),
        ):
            mem = memory.loc[method]
            q = quality.loc[method]
            expected = (float(mem.codec_only_ratio), float(q.psnr), float(q.ssim), float(q.lpips))
            for actual, target, tolerance, metric in zip(
                values, expected, (5.1e-4, 5.1e-5, 5.1e-5, 5.1e-5),
                ("D128 ratio", "PSNR", "SSIM", "LPIPS"),
            ):
                close(actual, target, tolerance, f"B{bit} {name} {metric}")
        uniform_payload = float(memory.loc[UNIFORM_BY_BIT[bit], "terminal_kv_mib"])
        temporal_payload = float(memory.loc[TEMPORAL_BY_BIT[bit], "terminal_kv_mib"])
        if abs(uniform_payload - temporal_payload) / uniform_payload > 2e-4:
            raise AssertionError(f"B{bit} uniform/temporal terminal payloads are not matched")

    # Maximum-rate comparison table and the time-adaptive peak-rate statement.
    for label, method in HIGHEST_RATE_ROWS.items():
        row = unique_row(rows, label, 5)
        displayed_ratio = numeric(row[1])
        mem = memory.loc[method]
        q = quality.loc[method]
        close(displayed_ratio, float(mem.codec_only_ratio), 5.1e-4, f"{label} max D128 ratio")
        close(numeric(row[2]), float(q.psnr), 5.1e-5, f"{label} max PSNR")
        close(numeric(row[3]), float(q.ssim), 5.1e-5, f"{label} max SSIM")
        close(numeric(row[4]), float(q.lpips), 5.1e-5, f"{label} max LPIPS")
    avg1 = memory.loc["adaptive_temporal_int8_avg1"]
    expected_peak_ratio = (bf16_mib / 2.0) / float(avg1.peak_kv_mib)
    peak_match = re.search(r"峰值口径为`([0-9.]+)×`", text)
    if not peak_match:
        raise AssertionError("time-adaptive avg1 peak ratio statement missing")
    close(float(peak_match.group(1)), expected_peak_ratio, 5.1e-4, "avg1 peak ratio")

    # The existing Pareto CSV must put every temporal point on each primary frontier.
    frontier = pd.read_csv(CROSS_ROOT / "pareto_terminal_frontier.csv")
    frontier = frontier[(frontier.model == "Wan2.2 5B") & frontier.metric.isin(["psnr", "ssim", "lpips"])]
    for method in TEMPORAL_BY_BIT.values():
        observed = set(frontier[frontier.method == method].metric)
        if observed != {"psnr", "ssim", "lpips"}:
            raise AssertionError(f"{method} is not on all three primary terminal frontiers: {observed}")

    # Latest fair runtime and workspace table only; no historical runtime mixing.
    for label, method in RUNTIME_ROWS.items():
        row = unique_row(rows, label, 4)
        measured = runtime.loc[method]
        space = workspace.loc[method]
        close(numeric(row[1]), float(measured.encode_30layer_estimate_s), 5.1e-4, f"{label} encode")
        close(numeric(row[2]), float(measured.decode_30layer_estimate_s), 5.1e-4, f"{label} decode")
        close(numeric(row[3]), float(space.peak_allocated_delta_mib_mean), 5.1e-3, f"{label} workspace")

    for label, method in CROSS_RUNTIME_ROWS.items():
        row = unique_row(rows, label, 3)
        measured = cross_runtime.loc[method]
        close(numeric(row[1]), float(measured.encode_s), 5.1e-4, f"{label} formal encode")
        close(numeric(row[2]), float(measured.decode_s), 5.1e-4, f"{label} formal decode")
    if "不能与上一表的绝对时间直接横向比较" not in text:
        raise AssertionError("cross-protocol KIVI/TurboQuant runtime caveat missing")

    qv4 = runtime.loc["direct_qv_int4"]
    current_b4 = runtime.loc["adaptive_uniform_b4"]
    encode_gain = 100.0 * (1.0 - current_b4.encode_30layer_estimate_s / qv4.encode_30layer_estimate_s)
    decode_cost = 100.0 * (current_b4.decode_30layer_estimate_s / qv4.decode_30layer_estimate_s - 1.0)
    workspace_gain = 100.0 * (
        1.0
        - workspace.loc["adaptive_uniform_b4", "peak_allocated_delta_mib_mean"]
        / workspace.loc["direct_qv_int4", "peak_allocated_delta_mib_mean"]
    )
    for pattern, expected, label in (
        (r"encode比QV4快约`([0-9.]+)%`", encode_gain, "encode gain"),
        (r"workspace少约`([0-9.]+)%`", workspace_gain, "workspace gain"),
        (r"standalone decode仍慢约`([0-9.]+)%`", decode_cost, "decode overhead"),
    ):
        match = re.search(pattern, text)
        if not match:
            raise AssertionError(f"missing {label} statement")
        close(float(match.group(1)), expected, 0.051, label)

    # Cross-model scope and local links are part of the one-page contract.
    if not any(row and row[0] == "Wan2.2 5B" for row in rows):
        raise AssertionError("Wan2.2 experiment-scope row missing")
    if not any(row and row[0] == "Self-Forcing" for row in rows):
        raise AssertionError("Self-Forcing experiment-scope row missing")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if "://" in target:
            continue
        path = (DOC.parent / target).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

    print(
        "one-page summary validation passed: formal cross-model quality/memory, "
        "four matched uniform/temporal pairs, Pareto, and latest fair runtime"
    )


if __name__ == "__main__":
    main()
