#!/usr/bin/env python3
"""Plot ICLR-width progressive-RVQ and residual-heterogeneity evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
STAGE_SOURCE = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/l16c64_codebook_usage_20260722/per_stage.csv"
)
SENSITIVITY_SOURCE = (
    REPO
    / "results/analysis/structured_static_prefix_allocation_proxy_v1_20260723/sensitivity.csv"
)
DEFAULT_OUTPUT = REPO / "results/analysis/wan2_2_5b_iclr2027_section3_20260811"
COLORS = {"K": "#2563EB", "V": "#DC2626"}
DIM = 128
CODEBOOK_SIZE = 64
BITS_PER_STAGE = math.log2(CODEBOOK_SIZE) / DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.linewidth": 0.45,
            "grid.alpha": 0.22,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def prepare_progressive(stage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for tensor in ("K", "V"):
        group = stage[stage["tensor"] == tensor].sort_values("stage")
        if len(group) != 16 or group["stage"].tolist() != list(range(1, 17)):
            raise ValueError(f"Incomplete L16 stage rows for {tensor}")
        d0 = float(group.iloc[0]["residual_mse_before_mean"])
        rows.append(
            {
                "tensor": tensor,
                "stage": 0,
                "rate_bits_per_dim": 0.0,
                "residual_mse": d0,
                "normalized_mse": 1.0,
                "marginal_gain_percent": np.nan,
                "normalization_d0": d0,
            }
        )
        for row in group.itertuples():
            rows.append(
                {
                    "tensor": tensor,
                    "stage": int(row.stage),
                    "rate_bits_per_dim": float(row.stage) * BITS_PER_STAGE,
                    "residual_mse": float(row.residual_mse_after_mean),
                    "normalized_mse": float(row.residual_mse_after_mean) / d0,
                    "marginal_gain_percent": 100 * float(row.stage_relative_mse_reduction_mean),
                    "normalization_d0": d0,
                }
            )
    progressive = pd.DataFrame(rows)
    rates = np.linspace(0.0, 16 * BITS_PER_STAGE, 201)
    ideal = pd.DataFrame(
        {
            "rate_bits_per_dim": rates,
            "normalized_mse": np.power(2.0, -2.0 * rates),
            "formula": "D_ref(B) = 1 * 2^(-2B)",
        }
    )
    return progressive, ideal


def prepare_sensitivity(source: pd.DataFrame) -> pd.DataFrame:
    definitions = {
        "K": ("qk_logits_mse_mean", "normalized QK-logit MSE proxy"),
        "V": ("attn_output_mse_mean", "normalized attention-output MSE proxy"),
    }
    rows: list[pd.DataFrame] = []
    for tensor, (field, metric) in definitions.items():
        group = source[source["tensor"] == tensor][["layer", "head", field]].copy()
        if len(group) != 30 * 24 or group[field].isna().any():
            raise ValueError(f"Incomplete 30x24 sensitivity grid for {tensor}:{field}")
        group = group.rename(columns={field: "raw_sensitivity"})
        group["normalized_sensitivity"] = (
            group["raw_sensitivity"] / group["raw_sensitivity"].mean()
        )
        group["tensor"] = tensor
        group["metric"] = metric
        group["normalization"] = "raw sensitivity / tensor-wide layer-head mean"
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def sensitivity_matrix(data: pd.DataFrame, tensor: str) -> np.ndarray:
    return (
        data[data["tensor"] == tensor]
        .pivot(index="layer", columns="head", values="normalized_sensitivity")
        .reindex(index=range(30), columns=range(24))
        .to_numpy()
    )


def heterogeneity_statistics(data: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for tensor in ("K", "V"):
        values = data[data["tensor"] == tensor]["normalized_sensitivity"].to_numpy()
        p10, p90 = np.percentile(values, [10, 90])
        result[tensor] = {
            "p10": float(p10),
            "p90": float(p90),
            "p90_over_p10": float(p90 / p10),
        }
    return result


def plot(
    progressive: pd.DataFrame,
    ideal: pd.DataFrame,
    sensitivity: pd.DataFrame,
    statistics: dict[str, dict[str, float]],
    figs_dir: Path,
) -> dict[str, dict[str, float]]:
    configure_style()
    fig = plt.figure(figsize=(7.0, 2.3))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.0, 1.0], wspace=0.38)
    ax_rate = fig.add_subplot(grid[0, 0])
    heat_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])]

    for tensor in ("K", "V"):
        group = progressive[progressive["tensor"] == tensor]
        ax_rate.plot(
            group["rate_bits_per_dim"],
            group["normalized_mse"],
            color=COLORS[tensor],
            linewidth=1.4,
            marker="o",
            markersize=2.3,
            markevery=[0, 1, 4, 8, 12, 16],
            label=f"{tensor} RVQ",
        )
    ax_rate.plot(
        ideal["rate_bits_per_dim"],
        ideal["normalized_mse"],
        color="#475569",
        linestyle="--",
        linewidth=1.1,
        label="Isotropic $2^{-2B}$ reference",
    )
    ax_rate.set_yscale("log")
    ax_rate.set_xlabel("Rate (bits/dim)")
    ax_rate.set_ylabel("Normalized MSE")
    ax_rate.set_xticks([0, 0.25, 0.5, 0.75])
    ax_rate.grid(True)
    ax_rate.legend(loc="lower left", handlelength=1.55, fontsize=7.0)
    ax_rate.text(-0.19, 1.04, "(a)", transform=ax_rate.transAxes, fontsize=9, fontweight="bold")
    ax_rate.text(0.0, 1.04, "Progressive refinement", transform=ax_rate.transAxes, fontsize=8.5)

    scales: dict[str, dict[str, float]] = {}
    for index, (ax, tensor) in enumerate(zip(heat_axes, ("K", "V"))):
        matrix = sensitivity_matrix(sensitivity, tensor)
        positive = matrix[matrix > 0]
        vmin, vmax = float(positive.min()), float(positive.max())
        scales[tensor] = {"vmin": vmin, "vmax": vmax}
        image = ax.pcolormesh(
            np.arange(matrix.shape[1] + 1) - 0.5,
            np.arange(matrix.shape[0] + 1) - 0.5,
            matrix,
            shading="flat",
            cmap="magma",
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(-0.5, matrix.shape[0] - 0.5)
        ax.set_aspect("auto")
        ax.set_title(tensor, pad=3)
        if index == 0:
            ax.set_ylabel("Layer")
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xticks([0, 7, 15, 23])
        ax.set_yticks([0, 10, 20, 29])
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.18, fraction=0.045)
        cbar.solids.set_rasterized(False)
        cbar.ax.tick_params(labelsize=6.5, length=2, width=0.5)
        cbar.set_label("Relative to tensor mean", fontsize=6.5, labelpad=1)
        ax.text(
            0.04,
            0.95,
            f"P90/P10 = {statistics[tensor]['p90_over_p10']:.1f}$\\times$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 1.3},
        )
    heat_axes[0].text(-0.25, 1.04, "(b)", transform=heat_axes[0].transAxes, fontsize=9, fontweight="bold")
    fig.text(0.73, 0.965, "Residual sensitivity", ha="center", va="top", fontsize=8.5)

    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.23, top=0.86)
    fig.savefig(figs_dir / "fig03_progressive_heterogeneity.pdf", dpi=300, facecolor="white")
    fig.savefig(figs_dir / "fig03_progressive_heterogeneity.png", dpi=300, facecolor="white")
    plt.close(fig)
    return scales


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    figs_dir = output / "figs"
    data_dir = output / "figure_data"
    targets = [
        figs_dir / "fig03_progressive_heterogeneity.pdf",
        figs_dir / "fig03_progressive_heterogeneity.png",
        data_dir / "fig03_progressive_rvq.csv",
        data_dir / "fig03_isotropic_reference.csv",
        data_dir / "fig03_layer_head_sensitivity.csv",
        data_dir / "fig03_config.json",
        output / "fig03_run_info.json",
    ]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite:\n" + "\n".join(existing))
    for path in (STAGE_SOURCE, SENSITIVITY_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    stage = pd.read_csv(STAGE_SOURCE)
    sensitivity_source = pd.read_csv(SENSITIVITY_SOURCE)
    stage_fields = {
        "tensor", "stage", "residual_mse_before_mean", "residual_mse_after_mean",
        "stage_relative_mse_reduction_mean",
    }
    sensitivity_fields = {
        "tensor", "layer", "head", "qk_logits_mse_mean", "attn_output_mse_mean",
    }
    if stage_fields - set(stage.columns):
        raise ValueError(f"Missing stage fields: {sorted(stage_fields - set(stage.columns))}")
    if sensitivity_fields - set(sensitivity_source.columns):
        raise ValueError(
            f"Missing sensitivity fields: {sorted(sensitivity_fields - set(sensitivity_source.columns))}"
        )

    progressive, ideal = prepare_progressive(stage)
    sensitivity = prepare_sensitivity(sensitivity_source)
    statistics = heterogeneity_statistics(sensitivity)
    progressive.to_csv(data_dir / "fig03_progressive_rvq.csv", index=False)
    ideal.to_csv(data_dir / "fig03_isotropic_reference.csv", index=False)
    sensitivity.to_csv(data_dir / "fig03_layer_head_sensitivity.csv", index=False)
    config = {
        "figure_size_inches": [7.0, 2.3],
        "rvq": {
            "codebook_size": CODEBOOK_SIZE,
            "dimension": DIM,
            "stages": 16,
            "bits_per_dim_formula": "B(L) = L * log2(C) / d",
            "bits_per_stage": BITS_PER_STAGE,
            "normalization": "D_tensor(L) / D_tensor(0)",
            "reference": "D_ref(B) = 1 * 2^(-2B)",
            "reference_label": "Isotropic 2^(-2B) reference",
        },
        "sensitivity": {
            "K_field": "qk_logits_mse_mean",
            "V_field": "attn_output_mse_mean",
            "normalization": "value / tensor-wide 30x24 layer-head mean",
            "color_scale": "separate full-range LogNorm per tensor",
            "heterogeneity_statistic": "P90/P10 over normalized 30x24 layer-head sensitivity",
            "statistics": statistics,
        },
    }
    write_json(data_dir / "fig03_config.json", config)
    scales = plot(progressive, ideal, sensitivity, statistics, figs_dir)

    run_info = {
        "git_commit": git_commit(),
        "config_path": str((data_dir / "fig03_config.json").resolve()),
        "config_hash": sha256(data_dir / "fig03_config.json"),
        "codebook_path": {
            "K": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_k_l16c64_codebooks",
            "V": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/nonorm_v_l16c64_codebooks",
        },
        "codebook_hash": {
            "K": "2aa5748ceb4594291c3ebbeb878bd6d98eb446c8994c97e46940383bb1069b1e",
            "V": "4667236477996d2bfa873517f696dddb1d000ceaae94218da575f9cb69dd25ae",
        },
        "hostname": socket.gethostname(),
        "gpu_type": "not used; CPU-only aggregation and plotting",
        "seed": 42,
        "prompt_list": {"stagewise": "20 train prompts", "sensitivity": "20 train prompts"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            str(STAGE_SOURCE): sha256(STAGE_SOURCE),
            str(SENSITIVITY_SOURCE): sha256(SENSITIVITY_SOURCE),
        },
        "output_hashes": {
            str(path.relative_to(output)): sha256(path)
            for path in targets[:-1]
            if path.is_file()
        },
    }
    write_json(output / "fig03_run_info.json", run_info)

    print(f"SOURCE Fig3(a): {STAGE_SOURCE}")
    print("FIELDS Fig3(a): tensor,stage,residual_mse_before_mean,residual_mse_after_mean,stage_relative_mse_reduction_mean")
    print(f"SOURCE Fig3(b): {SENSITIVITY_SOURCE}")
    print("FIELDS Fig3(b)-K: tensor=K,layer,head,qk_logits_mse_mean")
    print("FIELDS Fig3(b)-V: tensor=V,layer,head,attn_output_mse_mean")
    print(f"OUTPUT: {figs_dir / 'fig03_progressive_heterogeneity.pdf'}")
    print(f"OUTPUT: {figs_dir / 'fig03_progressive_heterogeneity.png'}")
    print("FIGURE DIMENSIONS: 7.0 x 2.3 inches; PNG 300 dpi")
    print(
        "PLOTTED RANGE Fig3(a): rate=[{:.6g}, {:.6g}] normalized_MSE=[{:.6g}, {:.6g}]".format(
            progressive["rate_bits_per_dim"].min(), progressive["rate_bits_per_dim"].max(),
            progressive["normalized_mse"].min(), progressive["normalized_mse"].max(),
        )
    )
    for tensor in ("K", "V"):
        group = sensitivity[sensitivity["tensor"] == tensor]
        print(
            "PLOTTED RANGE Fig3(b)-{}: normalized_sensitivity=[{:.6g}, {:.6g}], LogNorm=[{:.6g}, {:.6g}]".format(
                tensor, group["normalized_sensitivity"].min(), group["normalized_sensitivity"].max(),
                scales[tensor]["vmin"], scales[tensor]["vmax"],
            )
        )
        print(
            "HETEROGENEITY {}: P10={:.9g}, P90={:.9g}, P90/P10={:.6f}x".format(
                tensor,
                statistics[tensor]["p10"],
                statistics[tensor]["p90"],
                statistics[tensor]["p90_over_p10"],
            )
        )
    print(f"RATE FORMULA: B(L) = L * log2({CODEBOOK_SIZE}) / {DIM} = L * {BITS_PER_STAGE:.6f} bits/dim")
    print("NORMALIZATION Fig3(a): each tensor residual MSE divided by its measured pre-stage D0")
    print("ISOTROPIC REFERENCE: D_ref(B) = 1 * 2^(-2B)")
    print("NORMALIZATION Fig3(b): raw sensitivity divided by its tensor-wide 30x24 layer-head mean")


if __name__ == "__main__":
    main()
