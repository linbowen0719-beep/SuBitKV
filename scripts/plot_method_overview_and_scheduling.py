#!/usr/bin/env python3
"""Draw the conceptual method overview and scheduling behavior figures.

These are deliberately schematic paper figures.  The scheduling states and
memory curves are illustrative normalized coordinates, not experimental data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


SEED = 20270811
INK = "#172033"
MUTED = "#64748B"
GRID = "#E2E8F0"
BASE = "#2563A6"
BASE_LIGHT = "#DBEAFE"
RESIDUAL = ["#F59E0B", "#FBBF24", "#FCD34D", "#FDE68A"]
RESIDUAL_EDGE = "#B45309"
BF16 = "#27364A"
UNIFORM = "#8B98A9"
PROPOSED = "#0F8A72"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis/wan2_2_5b_iclr2027_method_figures_20260811"),
    )
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    face: str,
    edge: str,
    radius: float = 0.025,
    linewidth: float = 0.8,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = MUTED,
    linewidth: float = 1.0,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=linewidth,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def assert_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs:\n" + "\n".join(existing))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def draw_figure_a(figs_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    k_color = "#4F46E5"
    v_color = "#C2410C"
    ax.text(0.50, 0.965, "Fixed structural base + schedulable residual detail", ha="center", va="top", fontsize=8.2, fontweight="bold", color=INK)

    rounded_box(ax, (0.015, 0.735), 0.105, 0.14, "#F8FAFC", MUTED)
    ax.text(0.0675, 0.805, "Raw K/V", ha="center", va="center", fontsize=8.0, fontweight="bold", color=INK)
    arrow(ax, (0.12, 0.805), (0.16, 0.805))

    rounded_box(ax, (0.16, 0.71), 0.20, 0.19, BASE_LIGHT, BASE)
    ax.text(0.26, 0.83, "Offline-learned", ha="center", va="center", fontsize=6.7, color=MUTED)
    ax.text(0.26, 0.765, "Shared RVQ", ha="center", va="center", fontsize=8.2, fontweight="bold", color=BASE)
    ax.text(0.26, 0.72, "fixed-depth", ha="center", va="center", fontsize=6.2, color=MUTED)
    arrow(ax, (0.36, 0.805), (0.40, 0.805))

    rounded_box(ax, (0.40, 0.71), 0.21, 0.19, "#F8FAFC", MUTED)
    ax.text(0.505, 0.825, "Structural base", ha="center", va="center", fontsize=7.4, fontweight="bold", color=BASE)
    ax.text(0.505, 0.762, "+ residual", ha="center", va="center", fontsize=7.4, fontweight="bold", color=RESIDUAL_EDGE)
    ax.text(0.505, 0.72, "decomposition", ha="center", va="center", fontsize=6.0, color=MUTED)

    arrow(ax, (0.61, 0.805), (0.67, 0.805), color=BASE)
    rounded_box(ax, (0.67, 0.735), 0.305, 0.14, BASE_LIGHT, BASE)
    ax.text(0.8225, 0.825, "Structural base", ha="center", va="center", fontsize=7.6, fontweight="bold", color=BASE)
    ax.text(0.8225, 0.775, "fixed • shared • always retained", ha="center", va="center", fontsize=6.3, color=MUTED)

    ax.plot([0.505, 0.505], [0.71, 0.655], color=RESIDUAL_EDGE, linewidth=0.9)
    ax.plot([0.355, 0.655], [0.655, 0.655], color=RESIDUAL_EDGE, linewidth=0.9)
    arrow(ax, (0.355, 0.655), (0.355, 0.61), color=k_color)
    arrow(ax, (0.655, 0.655), (0.655, 0.61), color=v_color)

    rounded_box(ax, (0.275, 0.525), 0.16, 0.085, "#EEF2FF", k_color, radius=0.018)
    ax.text(0.355, 0.5675, "K residual", ha="center", va="center", fontsize=7.2, fontweight="bold", color=k_color)
    rounded_box(ax, (0.575, 0.525), 0.16, 0.085, "#FFF7ED", v_color, radius=0.018)
    ax.text(0.655, 0.5675, "V residual", ha="center", va="center", fontsize=7.2, fontweight="bold", color=v_color)

    arrow(ax, (0.355, 0.525), (0.355, 0.485), color=k_color)
    rounded_box(ax, (0.275, 0.40), 0.16, 0.085, "white", k_color, radius=0.018)
    ax.text(0.355, 0.4425, "Frame predictor", ha="center", va="center", fontsize=6.8, color=INK)
    arrow(ax, (0.355, 0.40), (0.355, 0.36), color=k_color)
    rounded_box(ax, (0.275, 0.275), 0.16, 0.085, "white", k_color, radius=0.018)
    ax.text(0.355, 0.3175, "Hadamard", ha="center", va="center", fontsize=6.8, color=INK)
    arrow(ax, (0.355, 0.275), (0.355, 0.235), color=k_color)

    rounded_box(ax, (0.275, 0.15), 0.16, 0.085, "#EEF2FF", k_color, radius=0.018)
    ax.text(0.355, 0.1925, "Block quantization", ha="center", va="center", fontsize=6.5, color=INK)
    arrow(ax, (0.655, 0.525), (0.655, 0.235), color=v_color)
    rounded_box(ax, (0.575, 0.15), 0.16, 0.085, "#FFF7ED", v_color, radius=0.018)
    ax.text(0.655, 0.1925, "Block quantization", ha="center", va="center", fontsize=6.5, color=INK)

    ax.plot([0.355, 0.355], [0.15, 0.11], color=k_color, linewidth=0.9)
    ax.plot([0.655, 0.655], [0.15, 0.11], color=v_color, linewidth=0.9)
    ax.plot([0.355, 0.75], [0.11, 0.11], color=RESIDUAL_EDGE, linewidth=0.9)
    arrow(ax, (0.75, 0.11), (0.78, 0.1775), color=RESIDUAL_EDGE, connectionstyle="arc3,rad=-0.22")
    rounded_box(ax, (0.78, 0.12), 0.20, 0.115, "#FFF7ED", RESIDUAL_EDGE, radius=0.018)
    ax.text(0.88, 0.192, "Truncatable residual codes", ha="center", va="center", fontsize=6.35, fontweight="bold", color=RESIDUAL_EDGE)
    ax.text(0.88, 0.15, "B1  B2  B3  B4", ha="center", va="center", fontsize=5.8, color=MUTED)

    arrow(ax, (0.88, 0.12), (0.88, 0.095), color=PROPOSED)
    rounded_box(ax, (0.78, 0.015), 0.20, 0.08, "#F0FDFA", PROPOSED, radius=0.018)
    ax.text(0.88, 0.066, "Budget-aware scheduler", ha="center", va="center", fontsize=6.3, fontweight="bold", color=PROPOSED)
    ax.text(0.88, 0.036, "residual codes only", ha="center", va="center", fontsize=5.4, color=MUTED)

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.99)
    fig.savefig(figs_dir / "figure_a_method_overview.pdf")
    fig.savefig(figs_dir / "figure_a_method_overview.png", dpi=300)
    plt.close(fig)


def schedule_states() -> tuple[np.ndarray, np.ndarray]:
    steps = 7
    fixed = np.full((steps, steps), np.nan)
    adaptive = np.full((steps, steps), np.nan)
    for time in range(steps):
        fixed[: time + 1, time] = 3
        for chunk in range(time + 1):
            age = time - chunk
            if chunk == 0:
                precision = 4
            else:
                precision = max(1, 4 - age)
            adaptive[chunk, time] = precision
    return fixed, adaptive


def draw_fixed_cell(ax: plt.Axes, x: float, y: float, precision: int) -> None:
    width, height = 0.72, 0.72
    ax.add_patch(
        Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            facecolor="#FBBF24",
            edgecolor=RESIDUAL_EDGE,
            linewidth=0.45,
        )
    )
    ax.text(x, y, f"B{precision}", ha="center", va="center", fontsize=5.6, color=INK)


def draw_adaptive_cell(ax: plt.Axes, x: float, y: float, precision: int) -> None:
    width, height = 0.72, 0.72
    left, bottom = x - width / 2, y - height / 2
    base_h = 0.12
    ax.add_patch(Rectangle((left, bottom), width, base_h, facecolor=BASE, edgecolor=BASE, linewidth=0.35))
    available = height - base_h
    layer_h = available / 4
    for bit in range(precision):
        ax.add_patch(
            Rectangle(
                (left, bottom + base_h + bit * layer_h),
                width,
                layer_h,
                facecolor=RESIDUAL[bit],
                edgecolor="white",
                linewidth=0.25,
            )
        )
    ax.add_patch(Rectangle((left, bottom), width, height, facecolor="none", edgecolor=MUTED, linewidth=0.4))


def prepare_memory_curves() -> pd.DataFrame:
    time = np.linspace(0, 7, 141)
    bf16 = 0.45 + 0.92 * time
    uniform = 0.35 + 0.47 * time
    budget = 3.05
    proposed = np.minimum(bf16, budget)
    return pd.DataFrame(
        {
            "generation_time": time,
            "uncompressed_bf16": bf16,
            "uniform_low_bit": uniform,
            "budget_aware": proposed,
            "peak_memory_budget": budget,
        }
    )


def style_schedule_axis(ax: plt.Axes, panel: str, title: str) -> None:
    ax.set_xlim(0.45, 7.55)
    ax.set_ylim(0.45, 7.55)
    ax.set_xticks([1, 3, 5, 7])
    ax.set_yticks([1, 3, 5, 7])
    ax.set_xticklabels(["1", "3", "5", "7"])
    ax.set_yticklabels(["C1", "C3", "C5", "C7"])
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Cache chunk")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.19, 1.05, panel, transform=ax.transAxes, fontsize=8.8, fontweight="bold", color=INK)
    ax.text(0.04, 1.05, title, transform=ax.transAxes, fontsize=8.2, color=INK)


def draw_figure_b(figs_dir: Path, fixed: np.ndarray, adaptive: np.ndarray, memory: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(7.0, 2.75))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 2.15, 1.45], wspace=0.42)
    ax_fixed = fig.add_subplot(grid[0, 0])
    ax_adapt = fig.add_subplot(grid[0, 1])
    ax_mem = fig.add_subplot(grid[0, 2])

    for time in range(fixed.shape[1]):
        for chunk in range(fixed.shape[0]):
            value = fixed[chunk, time]
            if np.isfinite(value):
                draw_fixed_cell(ax_fixed, time + 1, chunk + 1, int(value))
    style_schedule_axis(ax_fixed, "(a)", "Fixed residual precision")
    ax_fixed.text(
        0.02,
        0.97,
        "fixed RVQ base\nomitted for clarity",
        transform=ax_fixed.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=MUTED,
    )

    for time in range(adaptive.shape[1]):
        for chunk in range(adaptive.shape[0]):
            value = adaptive[chunk, time]
            if np.isfinite(value):
                draw_adaptive_cell(ax_adapt, time + 1, chunk + 1, int(value))
    style_schedule_axis(ax_adapt, "(b)", "Budget-aware residual scheduling")
    ax_adapt.axvline(4.5, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax_adapt.annotate(
        "Borrow unused\nfuture capacity",
        xy=(2.0, 2.0),
        xytext=(1.05, 7.34),
        ha="left",
        va="top",
        fontsize=5.5,
        color=PROPOSED,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 0.7},
        arrowprops={"arrowstyle": "->", "linewidth": 0.55, "color": PROPOSED},
    )
    ax_adapt.annotate(
        "Reclaim memory\nby truncating\nlow-utility history",
        xy=(6.0, 3.0),
        xytext=(4.45, 7.34),
        ha="left",
        va="top",
        fontsize=4.95,
        color=RESIDUAL_EDGE,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none", "pad": 0.7},
        arrowprops={"arrowstyle": "->", "linewidth": 0.55, "color": RESIDUAL_EDGE},
    )
    ax_adapt.text(
        0.50,
        -0.25,
        "Monotonic truncation: B4 → B3 → B2 → B1\nOnly residual refinement bits are discarded.",
        transform=ax_adapt.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
        color=RESIDUAL_EDGE,
    )
    ax_adapt.text(0.50, -0.38, "Preserve structure, discard detail.", transform=ax_adapt.transAxes, ha="center", va="top", fontsize=6.4, fontweight="bold", color=INK)

    base_handle = Rectangle((0, 0), 1, 1, facecolor=BASE, edgecolor=BASE)
    residual_handle = Rectangle((0, 0), 1, 1, facecolor=RESIDUAL[1], edgecolor=RESIDUAL_EDGE, linewidth=0.4)
    ax_adapt.legend(
        [base_handle, residual_handle],
        ["fixed RVQ base", "residual refinements"],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.84),
        frameon=False,
        fontsize=5.5,
        handlelength=1.1,
        borderpad=0,
    )

    time = memory["generation_time"].to_numpy()
    ax_mem.plot(time, memory["uncompressed_bf16"], color=BF16, linewidth=2.0)
    ax_mem.plot(time, memory["uniform_low_bit"], color=UNIFORM, linewidth=1.35)
    ax_mem.plot(time, memory["budget_aware"], color=PROPOSED, linewidth=1.25)
    budget = float(memory["peak_memory_budget"].iloc[0])
    ax_mem.axhline(budget, color=RESIDUAL_EDGE, linewidth=0.8, linestyle=(0, (3, 2)))
    ax_mem.axvline(4.5, color=MUTED, linewidth=0.7, linestyle=(0, (2, 2)), zorder=0)
    ax_mem.set_xlim(0, 7.1)
    ax_mem.set_ylim(0, 7.2)
    ax_mem.set_xticks([1, 3, 5, 7])
    ax_mem.set_yticks([])
    ax_mem.set_xlabel("Chunk")
    ax_mem.set_ylabel("KV-cache memory")
    ax_mem.spines[["top", "right"]].set_visible(False)
    ax_mem.grid(axis="y", color=GRID, linewidth=0.45)
    ax_mem.text(-0.19, 1.05, "(c)", transform=ax_mem.transAxes, fontsize=8.8, fontweight="bold", color=INK)
    ax_mem.text(0.04, 1.05, "KV-cache memory", transform=ax_mem.transAxes, fontsize=8.2, color=INK)
    ax_mem.text(6.86, memory["uncompressed_bf16"].iloc[-1] - 0.12, "BF16", ha="right", va="top", fontsize=5.9, color=BF16)
    ax_mem.text(6.86, memory["uniform_low_bit"].iloc[-1] + 0.12, "Uniform low-bit", ha="right", va="bottom", fontsize=5.8, color=UNIFORM)
    ax_mem.text(6.86, memory["budget_aware"].iloc[-1] - 0.22, "Budget-aware", ha="right", va="top", fontsize=5.9, color=PROPOSED)
    ax_mem.text(0.18, budget + 0.10, r"Peak memory budget $B_{\mathrm{peak}}$", ha="left", va="bottom", fontsize=5.7, color=RESIDUAL_EDGE)
    ax_mem.text(4.62, 0.72, "new chunk\n→ truncate\nhistory", ha="left", va="bottom", fontsize=5.2, color=MUTED)
    ax_mem.text(0.98, 0.02, "conceptual, not to scale", transform=ax_mem.transAxes, ha="right", va="bottom", fontsize=5.2, color=MUTED)

    fig.subplots_adjust(left=0.055, right=0.992, bottom=0.35, top=0.86)
    fig.savefig(figs_dir / "figure_b_scheduling_behavior.pdf")
    fig.savefig(figs_dir / "figure_b_scheduling_behavior.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    figs_dir = output_dir / "figs"
    data_dir = output_dir / "figure_data"
    targets = [
        figs_dir / "figure_a_method_overview.pdf",
        figs_dir / "figure_a_method_overview.png",
        figs_dir / "figure_b_scheduling_behavior.pdf",
        figs_dir / "figure_b_scheduling_behavior.png",
        data_dir / "figure_a_spec.json",
        data_dir / "figure_b_schedule.csv",
        data_dir / "figure_b_memory_curves.csv",
        data_dir / "config.json",
        output_dir / "run_info.json",
    ]
    assert_absent(targets)
    figs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    fixed, adaptive = schedule_states()
    memory = prepare_memory_curves()
    schedule_rows = []
    for policy, values in (("fixed", fixed), ("budget_aware", adaptive)):
        for chunk in range(values.shape[0]):
            for time in range(values.shape[1]):
                value = values[chunk, time]
                if np.isfinite(value):
                    schedule_rows.append(
                        {
                            "policy": policy,
                            "generation_step": time + 1,
                            "chunk_id": chunk + 1,
                            "retained_residual_blocks": int(value),
                            "fixed_base_retained": policy == "budget_aware",
                            "data_kind": "conceptual schematic",
                        }
                    )
    schedule = pd.DataFrame(schedule_rows)

    spec = {
        "data_kind": "conceptual schematic; no experimental measurements",
        "pipeline": [
            "raw K/V",
            "offline-learned shared fixed-depth RVQ",
            "fixed structural base plus K/V residuals",
            "K residual: frame predictor -> Hadamard -> block quantization",
            "V residual: block quantization",
            "merged truncatable residual codes",
            "budget-aware residual-code scheduler",
        ],
        "adaptive_scope": "residual refinement blocks only",
        "persistent_scope": "fixed RVQ structural base",
    }
    config = {
        "seed": SEED,
        "figure_a_inches": [7.0, 2.6],
        "figure_b_inches": [7.0, 2.75],
        "schedule_steps": 7,
        "residual_blocks": ["B1", "B2", "B3", "B4"],
        "memory_units": "illustrative normalized coordinates",
        "vector_rendering": True,
    }
    (data_dir / "figure_a_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    schedule.to_csv(data_dir / "figure_b_schedule.csv", index=False)
    memory.assign(data_kind="conceptual schematic").to_csv(data_dir / "figure_b_memory_curves.csv", index=False)
    (data_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    draw_figure_a(figs_dir)
    draw_figure_b(figs_dir, fixed, adaptive, memory)

    output_hashes = {}
    for path in targets[:-1]:
        output_hashes[str(path.relative_to(output_dir))] = sha256(path)
    run_info = {
        "git_commit": git_commit(),
        "config_path": str(data_dir / "config.json"),
        "config_hash": sha256(data_dir / "config.json"),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": "not used; CPU-only conceptual plotting",
        "seed": SEED,
        "prompt_list": "not applicable; conceptual schematic",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "source_reference": "existing fig01/fig07 layout inspected; no experimental data reused",
        "output_hashes": output_hashes,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n")

    print("FIGURE TYPE: conceptual schematic; no experimental measurements")
    print("FIGURE A: fixed structural base s=RVQ(x); adaptation applies only to Q_b(r)")
    print("FIGURE B: seven illustrative generation steps; historical residual precision is monotone non-increasing")
    print("MEMORY CURVES: illustrative normalized coordinates; not to scale")
    print(f"OUTPUT: {figs_dir / 'figure_a_method_overview.pdf'}")
    print(f"OUTPUT: {figs_dir / 'figure_b_scheduling_behavior.pdf'}")


if __name__ == "__main__":
    main()
