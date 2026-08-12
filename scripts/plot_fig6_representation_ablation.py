#!/usr/bin/env python3
"""Plot corrected Fig. 6 from matched Wan2.2 5B exact-attention replay."""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "Wan2.2 5B"
BITS = (1, 2, 3, 4)
DATA_PATH = (
    REPO_ROOT
    / "results/analysis/fig6_representation_ablation_corrected_formal10x30_20260812/attention_aggregate.csv"
)
RECON_PATH = (
    REPO_ROOT
    / "results/analysis/fig6_int2_specialized_reconstruction_formal10x30_v2_20260812/kv_reconstruction_per_chunk.csv"
)
OUTPUT_PATH = REPO_ROOT / "figures/fig6_representation_ablation_corrected_v12.pdf"
OUTPUT_PNG = OUTPUT_PATH.with_suffix(".png")
OUTPUT_DATA = OUTPUT_PATH.with_name("fig6_representation_ablation_corrected_v12_data.csv")
OUTPUT_RUN_INFO = OUTPUT_PATH.with_name(
    "fig6_representation_ablation_corrected_v12_run_info.json"
)

METHODS = {
    "Direct": {bit: f"direct_current_b{bit}" for bit in BITS},
    "+ RVQ": {bit: f"adaptive_rvq_uniform_b{bit}" for bit in BITS},
    "Adaptive routing": {bit: f"adaptive_rvq_current_b{bit}" for bit in BITS},
}
COLORS = {
    "Direct": "#C7CBD1",
    "+ RVQ": "#4C78A8",
    "Adaptive routing": "#E59A53",
}
RVQ_INDEX_BITS = 6
RVQ_VECTOR_DIM = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
]:
    frame = pd.read_csv(DATA_PATH)
    aggregate = frame[frame.prompt.astype(str) == "aggregate"]
    assert set(aggregate.model.astype(str)) == {MODEL}
    assert set(aggregate.prompts.astype(int)) == {10}
    assert set(aggregate.layers.astype(int)) == {30}
    attention_raw: dict[str, dict[int, float]] = {label: {} for label in METHODS}
    for label, by_bit in METHODS.items():
        for bit, method in by_bit.items():
            selected = aggregate[aggregate.method.astype(str) == method]
            assert len(selected) == 1, f"Expected one aggregate row for {method}"
            attention_raw[label][bit] = float(selected.iloc[0].attention_kl)
    attention_normalized = {
        label: {
            bit: value / attention_raw["Direct"][bit]
            for bit, value in values.items()
        }
        for label, values in attention_raw.items()
    }
    values = [value for group in attention_raw.values() for value in group.values()]
    assert all(np.isfinite(values)) and all(value > 0 for value in values)
    assert all(
        np.isclose(attention_normalized["Direct"][bit], 1.0) for bit in BITS
    )

    raw_frame = pd.read_csv(RECON_PATH)
    assert len(raw_frame) == 12_000
    assert raw_frame.duplicated(
        ["prompt", "layer", "chunk", "tensor", "method"]
    ).sum() == 0
    assert raw_frame.prompt.nunique() == 10
    assert raw_frame.layer.nunique() == 30
    assert raw_frame.chunk.nunique() == 10
    panel_a: dict[str, dict[str, float]] = {}
    for label, method in (
        ("Direct", "direct_current_b2"),
        ("+ RVQ", "adaptive_rvq_uniform_b2"),
    ):
        selected = raw_frame[raw_frame.method.astype(str) == method].copy()
        assert len(selected) == 6_000
        by_tensor = selected.groupby("tensor")
        mse = {
            tensor: float((group.mse * group["values"]).sum() / group["values"].sum())
            for tensor, group in by_tensor
        }
        source_bytes = float(selected.source_bytes.sum())
        if label == "Direct":
            # The v2 audit included the 3,426-byte K decoder table in every
            # chunk. The per-prompt cache rate excludes shared model tables;
            # the matching +RVQ K residual row records the correct local bytes.
            payload_bytes = float(
                selected[selected.tensor == "V"].residual_payload_bytes.sum()
                + raw_frame[
                    (raw_frame.method == "adaptive_rvq_uniform_b2")
                    & (raw_frame.tensor == "K")
                ].residual_payload_bytes.sum()
            )
        else:
            residual_bytes = float(selected.residual_payload_bytes.sum())
            # C64 RVQ indices use six logical bits. The raw reconstruction
            # states retain uint8 tensors for decoding; report the actual
            # bit-packed cache payload instead of their in-memory tensor size.
            packed_rvq_bytes = float(
                (
                    selected["values"].astype(float)
                    / RVQ_VECTOR_DIM
                    * selected.rvq_depth.astype(float)
                    * RVQ_INDEX_BITS
                    / 8
                ).sum()
            )
            payload_bytes = residual_bytes + packed_rvq_bytes
        panel_a[label] = {
            "K MSE ↓": mse["K"],
            "V MSE ↓": mse["V"],
            "Compression\nrate ↑": float(source_bytes / payload_bytes),
        }
    panel_a_normalized = {
        label: {
            metric: value / panel_a["Direct"][metric]
            for metric, value in values.items()
        }
        for label, values in panel_a.items()
    }
    return panel_a, panel_a_normalized, attention_raw, attention_normalized


def write_audit_data(
    panel_a: dict[str, dict[str, float]],
    panel_a_normalized: dict[str, dict[str, float]],
    attention_raw: dict[str, dict[int, float]],
    attention_normalized: dict[str, dict[int, float]],
) -> None:
    rows = []
    for label in ("Direct", "+ RVQ"):
        for metric, unit in (
            ("K MSE ↓", "MSE"),
            ("V MSE ↓", "MSE"),
            ("Compression\nrate ↑", "x vs BF16"),
        ):
            rows.append(
                {
                    "panel": "a",
                    "model": MODEL,
                    "bit": 2,
                    "variant": label,
                    "method": METHODS[label][2],
                    "quantity": metric.replace("\n", " "),
                    "raw_value": panel_a[label][metric],
                    "normalized_value": panel_a_normalized[label][metric],
                    "raw_unit": unit,
                    "normalization_baseline": METHODS["Direct"][2],
                }
            )
    for label in tuple(METHODS):
        for bit in BITS:
            rows.append(
                {
                    "panel": "b",
                    "model": MODEL,
                    "bit": bit,
                    "variant": label,
                    "method": METHODS[label][bit],
                    "quantity": "attention KL",
                    "raw_value": attention_raw[label][bit],
                    "normalized_value": attention_normalized[label][bit],
                    "raw_unit": "KL divergence",
                    "normalization_baseline": METHODS["Direct"][bit],
                }
            )
    with OUTPUT_DATA.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E7E9EC", linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.spines["left"].set_linewidth(0.6)
    axis.spines["bottom"].set_linewidth(0.6)
    axis.tick_params(width=0.6, length=2.5, labelsize=8)


def grouped_bars(
    axis: plt.Axes,
    normalized: dict[str, dict[int, float]],
    labels: tuple[str, ...],
) -> None:
    x = np.arange(len(BITS))
    width = 0.34 if len(labels) == 2 else 0.245
    offsets = (
        np.array((-0.5, 0.5)) * width
        if len(labels) == 2
        else np.array((-1.0, 0.0, 1.0)) * width
    )
    bars_by_label: dict[str, object] = {}
    for offset, label in zip(offsets, labels):
        values = [normalized[label][bit] for bit in BITS]
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.35,
        )
        bars_by_label[label] = bars

    def reduction_from_direct(target_label: str) -> None:
        target_bars = bars_by_label[target_label]
        for bit, target_bar in zip(BITS, target_bars):
            source = normalized["Direct"][bit]
            target = normalized[target_label][bit]
            ratio = source / target
            label = f"{ratio:.2f}×"
            target_x = target_bar.get_x() + target_bar.get_width() / 2
            marker_top = source
            cap = target_bar.get_width() * 0.20
            axis.plot(
                [target_x - cap, target_x + cap],
                [marker_top, marker_top],
                color="black",
                linewidth=0.7,
                clip_on=False,
                zorder=5,
            )
            axis.annotate(
                "",
                xy=(target_x, target + 0.012),
                xytext=(target_x, marker_top),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "linewidth": 0.7,
                    "shrinkA": 0,
                    "shrinkB": 1.5,
                },
                zorder=5,
            )
            axis.text(
                target_x + (-0.018 if target_label == "+ RVQ" else 0.018),
                0.5 * (marker_top + target),
                label,
                ha="right" if target_label == "+ RVQ" else "left",
                va="center",
                fontsize=5.4,
                color="black",
                zorder=6,
            )

    reduction_from_direct("+ RVQ")
    if "Adaptive routing" in labels:
        reduction_from_direct("Adaptive routing")
    axis.set_xticks(x, [str(bit) for bit in BITS])
    axis.set_xlabel("Encoder precision (bit)", fontsize=8.5)
    axis.set_ylim(0, 1.16)
    axis.legend(
        loc="upper right",
        frameon=False,
        fontsize=7.4,
        ncols=len(labels),
        handlelength=1.2,
        columnspacing=0.8,
    )
    style_axis(axis)


def panel_a_bars(
    axis: plt.Axes,
    normalized: dict[str, dict[str, float]],
) -> None:
    metrics = ("K MSE ↓", "V MSE ↓", "Compression\nrate ↑")
    x = np.arange(len(metrics))
    width = 0.33
    direct = axis.bar(
        x - width / 2,
        [normalized["Direct"][metric] for metric in metrics],
        width,
        label="Direct",
        color=COLORS["Direct"],
        edgecolor="white",
        linewidth=0.35,
    )
    rvq = axis.bar(
        x + width / 2,
        [normalized["+ RVQ"][metric] for metric in metrics],
        width,
        label="+ RVQ",
        color=COLORS["+ RVQ"],
        edgecolor="white",
        linewidth=0.35,
    )
    for index, (source_bar, target_bar) in enumerate(zip(direct, rvq)):
        source = source_bar.get_height()
        target = target_bar.get_height()
        ratio = source / target
        source_x = source_bar.get_x() + source_bar.get_width() / 2
        target_x = target_bar.get_x() + target_bar.get_width() / 2
        marker_top = source
        cap = target_bar.get_width() * 0.20
        axis.plot(
            [target_x - cap, target_x + cap],
            [marker_top, marker_top],
            color="black",
            linewidth=0.7,
            clip_on=False,
            zorder=5,
        )
        axis.annotate(
            "",
            xy=(target_x, target + 0.015),
            xytext=(target_x, marker_top),
            arrowprops={
                "arrowstyle": "->",
                "color": "black",
                "linewidth": 0.7,
                "shrinkA": 0,
                "shrinkB": 1.5,
            },
            zorder=5,
        )
        axis.text(
            target_x - 0.025,
            0.5 * (marker_top + target),
            f"{ratio:.2f}×",
            ha="right",
            va="center",
            fontsize=6.1,
            color="black",
            zorder=6,
        )
    axis.set_xticks(x, [metric.replace(" ↓", "").replace(" ↑", "") for metric in metrics])
    axis.set_ylim(0, 1.18)
    axis.legend(loc="upper right", frameon=False, fontsize=7.4, ncols=2)
    style_axis(axis)


def plot(
    panel_a_normalized: dict[str, dict[str, float]],
    attention_normalized: dict[str, dict[int, float]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(8.6, 2.7), facecolor="white")
    grid = GridSpec(1, 2, figure=figure, width_ratios=[1.05, 1.20], wspace=0.28)
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    panel_a_bars(ax_a, panel_a_normalized)
    grouped_bars(
        ax_b,
        attention_normalized,
        ("Direct", "+ RVQ", "Adaptive routing"),
    )
    ax_a.set_ylabel("Relative to Direct INT2", fontsize=8.5)
    ax_a.set_title("(a) INT2 reconstruction and rate", fontsize=9, fontweight="semibold", pad=5)
    ax_b.set_ylabel("Normalized attention KL ↓", fontsize=8.5)
    ax_b.set_xlabel("Encoder precision (bit)", fontsize=8.2)
    ax_b.set_title("(b) Attention preservation", fontsize=9, fontweight="semibold", pad=5)
    figure.savefig(OUTPUT_PATH, bbox_inches="tight", pad_inches=0.02)
    figure.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def write_run_info() -> None:
    source = json.loads(
        (
            REPO_ROOT
            / "results/analysis/adaptive_rvq_kivi_rate_matched_20260803/run_info.json"
        ).read_text(encoding="utf-8")
    )
    info = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(Path(__file__).resolve()),
        "config_hash": sha256(Path(__file__).resolve()),
        "codebook_path": source["codebook_path"],
        "codebook_hash": source["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": "not applicable (plotting only)",
        "seed": 20260803,
        "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "attention_scope": (
            "10 prompts x 30 layers; standard RoPE + camera PRoPE; "
            "joint K_and_V intervention"
        ),
        "reconstruction_scope": (
            "10 prompts x 30 layers x 10 independent 3520-token chunks; "
            "pre-RoPE/pre-PRoPE D128 BF16; element-weighted K/V MSE"
        ),
        "compression_accounting": (
            "joint BF16 K+V bytes / per-prompt cache payload; C64 RVQ indices "
            "are bit-packed at 6 bits/index; RVQ codebooks and specialized "
            "decoder tables are shared model parameters and excluded from the "
            "per-prompt KV-cache payload"
        ),
        "method_semantics": {
            "Direct": "uniform RVQ-specialized K/V encoder applied directly to raw KV",
            "+ RVQ": "adaptive RVQ plus the same uniform specialized encoder on the residual",
            "Adaptive routing": (
                "adaptive RVQ plus specialized encoder and frozen layer-head "
                "adaptive routing"
            ),
        },
        "data_path": str(DATA_PATH),
        "data_hash": sha256(DATA_PATH),
        "reconstruction_data_path": str(RECON_PATH),
        "reconstruction_data_hash": sha256(RECON_PATH),
        "outputs": [str(OUTPUT_PATH), str(OUTPUT_PNG), str(OUTPUT_DATA)],
    }
    with OUTPUT_RUN_INFO.open("x", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    outputs = (OUTPUT_PATH, OUTPUT_PNG, OUTPUT_DATA, OUTPUT_RUN_INFO)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite corrected Fig. 6 outputs: {existing}")
    panel_a, panel_a_normalized, attention_raw, attention_normalized = load_data()
    print(json.dumps({
        "panel_a_raw": panel_a,
        "panel_a_normalized": panel_a_normalized,
        "attention_raw": attention_raw,
        "attention_normalized": attention_normalized,
    }, indent=2))
    write_audit_data(
        panel_a, panel_a_normalized, attention_raw, attention_normalized
    )
    plot(panel_a_normalized, attention_normalized)
    write_run_info()


if __name__ == "__main__":
    main()
