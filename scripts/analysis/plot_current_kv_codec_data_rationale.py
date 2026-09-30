#!/usr/bin/env python3
"""Plot data-distribution evidence for the deployed K/V residual codecs.

This is a read-only analysis over existing exact-attention CSVs and cached
pre-RoPE/pre-PRoPE BF16 KV tensors.  It does not generate videos or run
Similarity/VBench.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from scipy.stats import norm  # noqa: E402


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"
ANALYSIS_DIR = SCRIPT.parent
for entry in (REPO_ROOT, SCRIPT_DIR, ANALYSIS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import ablate_frame_mean_hadamard_exact_k as k_ablation  # noqa: E402
import evaluate_sfbr_train_marginals as replay  # noqa: E402
from subitkv.truncatable_residual import groupwise_hadamard  # noqa: E402
from run_fixed_budget_rvq_screen import parse_ints  # noqa: E402
from run_truncatable_residual_codec_bench import stable_directory_hash  # noqa: E402


DEFAULT_K_ROOT = (
    REPO_ROOT
    / "results/analysis/frame_mean_hadamard_exact_k_test10_all30_v2_20260728"
)
DEFAULT_V_ROOT = (
    REPO_ROOT
    / "results/analysis/frame_mean_hadamard_exact_v_test10_all30_20260728"
)
DEFAULT_ROUTING_ROOT = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_functional_routing_per_layer_f0125_20260725"
)
DEFAULT_TREE = (
    REPO_ROOT
    / "results/analysis/routed_gaussian_routing_train20_20260725_v3/tree_params.pt"
)
DEFAULT_DATA = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
DEFAULT_GEOMETRY = (
    REPO_ROOT
    / "results/analysis/wan5b_rvq_mechanisms_p1_all_layer_map_captures"
    / "prompt_08/paired_rope/rank_000"
)
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
ZH_FONT: FontProperties | None = None

TENSOR_COLORS = {"K": "#2563eb", "V": "#dc2626"}
METHOD_COLORS = {
    "K，无Hadamard": "#93c5fd",
    "K，Hadamard": "#1d4ed8",
    "V，无Hadamard": "#fca5a5",
    "V，Hadamard": "#b91c1c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--k-root", type=Path, default=DEFAULT_K_ROOT)
    parser.add_argument("--v-root", type=Path, default=DEFAULT_V_ROOT)
    parser.add_argument("--routing-root", type=Path, default=DEFAULT_ROUTING_ROOT)
    parser.add_argument("--tree-params", type=Path, default=DEFAULT_TREE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--sample-prompts", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--sample-layers", default="0,10,20,29")
    parser.add_argument("--samples-per-head-span", type=int, default=128)
    parser.add_argument("--block-ratios-per-head-span", type=int, default=32)
    parser.add_argument("--assignment-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codec-seed", type=int, default=20260722)
    args = parser.parse_args()
    for name in (
        "output_root",
        "k_root",
        "v_root",
        "routing_root",
        "tree_params",
        "data_root",
        "geometry_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.sample_prompts = parse_ints(args.sample_prompts)
    args.sample_layers = parse_ints(args.sample_layers)
    if args.output_root.exists():
        parser.error(f"output root exists: {args.output_root}")
    if not args.sample_prompts or not args.sample_layers:
        parser.error("sample prompts/layers must be non-empty")
    if args.samples_per_head_span <= 0 or args.block_ratios_per_head_span <= 0:
        parser.error("sample counts must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    global ZH_FONT
    if FONT_PATH.is_file():
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(FONT_PATH))
        ZH_FONT = FontProperties(
            family=["DejaVu Sans", "Droid Sans Fallback"]
        )
    else:
        ZH_FONT = FontProperties(family="DejaVu Sans")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Droid Sans Fallback"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Matplotlib probes the literal string "lp" while computing text extents.
    # The Chinese fallback font intentionally has no Latin glyphs.
    warnings.filterwarnings(
        "ignore",
        message=r"Glyph (108|112).*Droid Sans Fallback",
        category=UserWarning,
    )


def save_figure(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.savefig(figures / f"{stem}.png", bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def validate_distribution(df: pd.DataFrame, name: str) -> None:
    required = {
        "split",
        "prompt_id",
        "layer",
        "span_index",
        "method",
        "predictor_energy_fraction",
        "excess_kurtosis",
        "channel_energy_cv",
        "channel_energy_max_over_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} distribution missing columns: {missing}")
    if set(df["split"]) != {"test"}:
        raise ValueError(f"{name} distribution is not test-only")
    if df["prompt_id"].nunique() != 10 or df["layer"].nunique() != 30:
        raise ValueError(f"{name} distribution is not 10-prompt x 30-layer")
    expected_methods = {
        "frame_mean_hadamard",
        "no_frame_mean_hadamard",
        "frame_mean_no_hadamard",
        "no_frame_mean_no_hadamard",
    }
    if set(df["method"]) != expected_methods:
        raise ValueError(f"{name} distribution method set changed")


def load_sources(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "k_distribution": args.k_root / "distribution.csv",
        "v_distribution": args.v_root / "distribution.csv",
        "v_exact": args.v_root / "exact_v_metrics.csv",
        "routing_marginals": args.routing_root / "functional_unit_marginals.csv",
        "routing": args.routing_root / "routing.json",
        "tree": args.tree_params,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing inputs: {missing}")
    k_distribution = pd.read_csv(paths["k_distribution"])
    v_distribution = pd.read_csv(paths["v_distribution"])
    v_exact = pd.read_csv(paths["v_exact"])
    marginals = pd.read_csv(paths["routing_marginals"])
    routing = json.loads(paths["routing"].read_text(encoding="utf-8"))
    validate_distribution(k_distribution, "K")
    validate_distribution(v_distribution, "V")
    if (
        v_exact["prompt_id"].nunique() != 10
        or v_exact["layer"].nunique() != 30
        or v_exact["head"].nunique() != 24
    ):
        raise ValueError("V exact metrics are not 10-prompt x 30-layer x 24-head")
    expected_v_methods = {
        "frame_mean_hadamard",
        "no_frame_mean_hadamard",
        "frame_mean_no_hadamard",
        "no_frame_mean_no_hadamard",
        "absmax_g128",
    }
    if set(v_exact["method"]) != expected_v_methods:
        raise ValueError("V exact method set changed")
    if set(marginals["tensor"]) != {"K", "V"}:
        raise ValueError("routing marginals must contain K and V")
    if len(marginals) != 30 * 24 * 2:
        raise ValueError("routing marginals row count changed")
    offsets = routing.get("offsets", {})
    for tensor in ("K", "V"):
        array = np.asarray(offsets.get(tensor), dtype=np.int8)
        if array.shape != (30, 24):
            raise ValueError(f"invalid {tensor} routing offsets")
        unique, counts = np.unique(array, return_counts=True)
        count_map = dict(zip(unique.tolist(), counts.tolist()))
        if count_map != {-1: 90, 0: 540, 1: 90}:
            raise ValueError(f"{tensor} route is not 12.5/75/12.5")
    return {
        "paths": paths,
        "k_distribution": k_distribution,
        "v_distribution": v_distribution,
        "v_exact": v_exact,
        "marginals": marginals,
        "routing": routing,
    }


def figure_frame_mean(
    k_df: pd.DataFrame,
    v_df: pd.DataFrame,
    output_root: Path,
    figures: Path,
) -> dict[str, float]:
    rows: list[pd.DataFrame] = []
    for tensor, df in (("K", k_df), ("V", v_df)):
        selected = df[df["method"] == "frame_mean_no_hadamard"].copy()
        selected["tensor"] = tensor
        rows.append(selected)
    data = pd.concat(rows, ignore_index=True)
    prompt_layer = (
        data.groupby(["tensor", "prompt_id", "layer"], as_index=False)[
            "predictor_energy_fraction"
        ]
        .mean()
        .rename(columns={"predictor_energy_fraction": "energy_fraction"})
    )
    layer_rows: list[dict[str, Any]] = []
    for (tensor, layer), group in prompt_layer.groupby(["tensor", "layer"]):
        values = group["energy_fraction"].to_numpy()
        layer_rows.append(
            {
                "tensor": tensor,
                "layer": int(layer),
                "mean": float(values.mean()),
                "p10": float(np.quantile(values, 0.10)),
                "p90": float(np.quantile(values, 0.90)),
            }
        )
    write_rows(output_root / "frame_mean_energy_by_layer.csv", layer_rows)
    prompt = (
        prompt_layer.groupby(["tensor", "prompt_id"], as_index=False)[
            "energy_fraction"
        ]
        .mean()
        .rename(columns={"energy_fraction": "prompt_mean_energy_fraction"})
    )
    write_rows(
        output_root / "frame_mean_energy_by_prompt.csv",
        prompt.to_dict("records"),
    )
    headline = {
        tensor: float(
            data[data["tensor"] == tensor]["predictor_energy_fraction"].mean()
        )
        for tensor in ("K", "V")
    }

    layers = pd.DataFrame(layer_rows)
    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 4.4), gridspec_kw={"width_ratios": [2.1, 1]}
    )
    ax = axes[0]
    for tensor in ("K", "V"):
        view = layers[layers["tensor"] == tensor].sort_values("layer")
        x = view["layer"].to_numpy()
        ax.plot(
            x,
            100 * view["mean"].to_numpy(),
            marker="o",
            markersize=3,
            linewidth=2,
            label=tensor,
            color=TENSOR_COLORS[tensor],
        )
        ax.fill_between(
            x,
            100 * view["p10"].to_numpy(),
            100 * view["p90"].to_numpy(),
            alpha=0.14,
            color=TENSOR_COLORS[tensor],
        )
    ax.set_title("各层帧均值可解释的残差能量", fontproperties=ZH_FONT)
    ax.set_xlabel("Transformer层", fontproperties=ZH_FONT)
    ax.set_ylabel("预测器能量 / RVQ残差能量（%）", fontproperties=ZH_FONT)
    ax.legend(title="缓存张量", prop=ZH_FONT, title_fontproperties=ZH_FONT)
    ax.grid(alpha=0.22)
    ax.text(
        0.01,
        0.97,
        "阴影：10个测试提示词的P10–P90",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        fontproperties=ZH_FONT,
    )

    ax = axes[1]
    arrays = [
        100
        * prompt[prompt["tensor"] == tensor][
            "prompt_mean_energy_fraction"
        ].to_numpy()
        for tensor in ("K", "V")
    ]
    boxes = ax.boxplot(arrays, tick_labels=["K", "V"], patch_artist=True)
    for patch, tensor in zip(boxes["boxes"], ("K", "V")):
        patch.set_facecolor(TENSOR_COLORS[tensor])
        patch.set_alpha(0.45)
    for index, tensor in enumerate(("K", "V"), start=1):
        ax.scatter(
            np.full(len(arrays[index - 1]), index),
            arrays[index - 1],
            s=20,
            color=TENSOR_COLORS[tensor],
            alpha=0.72,
            zorder=3,
        )
        ax.text(
            index,
            max(arrays[index - 1]) * 1.06,
            f"均值 {100 * headline[tensor]:.2f}%",
            ha="center",
            fontsize=9,
            fontproperties=ZH_FONT,
        )
    ax.set_title("跨提示词分布", fontproperties=ZH_FONT)
    ax.set_ylabel("提示词的跨层均值（%）", fontproperties=ZH_FONT)
    ax.grid(axis="y", alpha=0.22)
    fig.suptitle(
        "图一　键缓存含有更强的帧级可预测结构",
        fontproperties=ZH_FONT,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures, "01_frame_mean_predictable_energy")
    return {
        "k_frame_mean_energy_fraction": headline["K"],
        "v_frame_mean_energy_fraction": headline["V"],
    }


def figure_hadamard(
    k_df: pd.DataFrame,
    v_df: pd.DataFrame,
    output_root: Path,
    figures: Path,
) -> dict[str, float]:
    metric_specs = (
        ("channel_energy_cv", "通道能量变异系数"),
        ("channel_energy_max_over_mean", "通道最大/平均能量"),
        ("excess_kurtosis", "超额峰度"),
    )
    pieces: list[pd.DataFrame] = []
    for tensor, frame in (("K", k_df), ("V", v_df)):
        selected = frame[
            frame["method"].isin(
                ("frame_mean_no_hadamard", "frame_mean_hadamard")
            )
        ].copy()
        selected["tensor"] = tensor
        selected["hadamard_label"] = np.where(
            selected["method"] == "frame_mean_hadamard",
            f"{tensor}，Hadamard",
            f"{tensor}，无Hadamard",
        )
        pieces.append(selected)
    data = pd.concat(pieces, ignore_index=True)
    aggregate_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for tensor in ("K", "V"):
        for method in ("frame_mean_no_hadamard", "frame_mean_hadamard"):
            view = data[
                (data["tensor"] == tensor) & (data["method"] == method)
            ]
            for metric, _label in metric_specs:
                aggregate_rows.append(
                    {
                        "tensor": tensor,
                        "method": method,
                        "metric": metric,
                        "mean": float(view[metric].mean()),
                        "median": float(view[metric].median()),
                    }
                )
                for layer, group in view.groupby("layer"):
                    layer_rows.append(
                        {
                            "tensor": tensor,
                            "method": method,
                            "metric": metric,
                            "layer": int(layer),
                            "mean": float(group[metric].mean()),
                        }
                    )
    write_rows(output_root / "hadamard_isotropy_summary.csv", aggregate_rows)
    write_rows(output_root / "hadamard_isotropy_by_layer.csv", layer_rows)
    aggregates = pd.DataFrame(aggregate_rows)

    labels = ("K,no-H", "K,H", "V,no-H", "V,H")
    color_by_label = {
        "K,no-H": "#93c5fd",
        "K,H": "#1d4ed8",
        "V,no-H": "#fca5a5",
        "V,H": "#b91c1c",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.1))
    for column, (metric, metric_label) in enumerate(metric_specs):
        ax = axes[0, column]
        values = [
            data[
                data["hadamard_label"]
                == {
                    "K,no-H": "K，无Hadamard",
                    "K,H": "K，Hadamard",
                    "V,no-H": "V，无Hadamard",
                    "V,H": "V，Hadamard",
                }[label]
            ][metric].to_numpy()
            for label in labels
        ]
        boxes = ax.boxplot(
            values,
            tick_labels=["K\nno-H", "K\nH", "V\nno-H", "V\nH"],
            patch_artist=True,
            showfliers=False,
        )
        for patch, label in zip(boxes["boxes"], labels):
            patch.set_facecolor(color_by_label[label])
            patch.set_alpha(0.62)
        if metric != "excess_kurtosis":
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(metric_label, fontproperties=ZH_FONT)
        ax.grid(axis="y", alpha=0.2)

        ax = axes[1, column]
        for label in labels:
            tensor = label[0]
            method = (
                "frame_mean_hadamard"
                if label.endswith(",H")
                else "frame_mean_no_hadamard"
            )
            view = pd.DataFrame(layer_rows)
            view = view[
                (view["tensor"] == tensor)
                & (view["method"] == method)
                & (view["metric"] == metric)
            ].sort_values("layer")
            ax.plot(
                view["layer"],
                view["mean"],
                color=color_by_label[label],
                linewidth=1.9,
                label=label,
            )
        if metric != "excess_kurtosis":
            ax.set_yscale("log")
        else:
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlabel("Transformer层", fontproperties=ZH_FONT)
        ax.set_ylabel("层内均值", fontproperties=ZH_FONT)
        ax.grid(alpha=0.2)
        if column == 0:
            ax.legend(
                ["K，无H", "K，有H", "V，无H", "V，有H"],
                fontsize=8,
                prop=ZH_FONT,
            )
    fig.suptitle(
        "图二　旋转主要修复键缓存的通道各向异性；值缓存本身已经较均匀",
        fontproperties=ZH_FONT,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures, "02_hadamard_isotropy")

    def mean_value(tensor: str, method: str, metric: str) -> float:
        row = aggregates[
            (aggregates["tensor"] == tensor)
            & (aggregates["method"] == method)
            & (aggregates["metric"] == metric)
        ]
        return float(row.iloc[0]["mean"])

    return {
        f"{tensor.lower()}_{short}_{suffix}": mean_value(
            tensor, method, metric
        )
        for tensor in ("K", "V")
        for method, suffix in (
            ("frame_mean_no_hadamard", "no_hadamard"),
            ("frame_mean_hadamard", "hadamard"),
        )
        for metric, short in (
            ("channel_energy_cv", "channel_cv"),
            ("channel_energy_max_over_mean", "channel_max_over_mean"),
            ("excess_kurtosis", "excess_kurtosis"),
        )
    }


def deterministic_indices(
    length: int,
    count: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    count = min(int(count), int(length))
    step = 7919
    offset = int(seed * 104729 + 17) % length
    return (torch.arange(count, device=device) * step + offset) % length


def sampling_task_for(
    args: SimpleNamespace, prompt: int, layer: int
) -> dict[str, Any]:
    dump = (
        args.data_root
        / "kv_dump"
        / args.split
        / str(prompt)
        / f"layer_{layer}.pt"
    )
    geometry = (
        args.geometry_root / f"layer_{layer:02d}" / "call_00000.pt"
    )
    if not dump.is_file() or not geometry.is_file():
        raise FileNotFoundError(dump if not dump.is_file() else geometry)
    metadata = torch.load(
        str(dump), map_location="cpu", weights_only=False, mmap=True
    )["metadata"]
    if (
        metadata.get("split") != args.split
        or int(metadata.get("prompt_id", -1)) != prompt
        or int(metadata.get("layer", -1)) != layer
    ):
        raise ValueError(f"invalid dump provenance: {dump}")
    return {
        "split": args.split,
        "prompt_id": prompt,
        "prompt": str(metadata["prompt"]),
        "layer": layer,
        "dump_path": str(dump),
        "dump_bytes": int(dump.stat().st_size),
        "geometry_path": str(geometry),
        "geometry_hash": replay.geometry_signature(geometry),
    }


@torch.no_grad()
def sample_value_geometry(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, float], list[str]]:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("cached residual sampling requires a CUDA device")
    torch.cuda.set_device(device)
    common_args = SimpleNamespace(
        data_root=args.data_root,
        geometry_root=args.geometry_root,
        split="test",
        codec_seed=args.codec_seed,
        assignment_batch_size=args.assignment_batch_size,
    )
    samples: dict[str, list[np.ndarray]] = {"K": [], "V": []}
    ratios: dict[str, list[np.ndarray]] = {"K": [], "V": []}
    source_files: list[str] = []
    history_tokens = 31680
    spans = ((0, 7040), (7040, 14080), (14080, 21120), (21120, 28160))
    for prompt in args.sample_prompts:
        for layer in args.sample_layers:
            task = sampling_task_for(common_args, prompt, layer)
            source_files.extend((task["dump_path"], task["geometry_path"]))
            tensors, evidence, *_rest = replay.positional_replay_from_dump(
                task, device
            )
            references = {
                "K": tensors["key_pre"][:, :, :history_tokens],
                "V": tensors["value_pre"][:, :, :history_tokens],
            }
            for tensor_name, reference in references.items():
                codebook = (
                    args.data_root
                    / f"nonorm_{tensor_name.lower()}_l8c64_codebooks"
                )
                for span_index, (start, end) in enumerate(spans):
                    source = reference[:, :, start:end].contiguous()
                    packed = evidence.encode_fixed(
                        source,
                        layer=layer,
                        directory=codebook,
                        batch_size=args.assignment_batch_size,
                    )
                    base = evidence.rvq_dequantize_tensor_cache(
                        packed, torch.bfloat16
                    ).contiguous()
                    residual = (source - base).contiguous()
                    if tensor_name == "K":
                        predictor, _raw = k_ablation.decode_frame_predictor(
                            residual
                        )
                        suffix = (residual.float() - predictor).contiguous()
                        transformed = groupwise_hadamard(
                            suffix, 128, seed=args.codec_seed
                        )
                    else:
                        transformed = residual.float()
                    blocks = transformed.reshape(
                        1, 24, -1, 128
                    ).contiguous()
                    rms = blocks.square().mean(-1).clamp_min(1e-20).sqrt()
                    normalized = blocks / rms.unsqueeze(-1)
                    ratio = blocks.abs().amax(-1) / rms
                    for head in range(24):
                        flat = normalized[0, head].reshape(-1)
                        value_idx = deterministic_indices(
                            flat.numel(),
                            args.samples_per_head_span,
                            seed=(
                                args.seed
                                + prompt * 100003
                                + layer * 1009
                                + span_index * 97
                                + head
                                + (0 if tensor_name == "K" else 1_000_003)
                            ),
                            device=device,
                        )
                        samples[tensor_name].append(
                            flat[value_idx].float().cpu().numpy()
                        )
                        flat_ratio = ratio[0, head].reshape(-1)
                        ratio_idx = deterministic_indices(
                            flat_ratio.numel(),
                            args.block_ratios_per_head_span,
                            seed=(
                                args.seed
                                + prompt * 100019
                                + layer * 1013
                                + span_index * 101
                                + head
                                + (0 if tensor_name == "K" else 2_000_003)
                            ),
                            device=device,
                        )
                        ratios[tensor_name].append(
                            flat_ratio[ratio_idx].float().cpu().numpy()
                        )
                    del packed, base, residual, transformed, blocks, normalized
            del tensors, evidence, references
            torch.cuda.empty_cache()
            print(
                f"[sampled] prompt={prompt} layer={layer}",
                flush=True,
            )

    bins = np.linspace(-6.0, 6.0, 161)
    probabilities = np.linspace(0.001, 0.999, 199)
    rows: list[dict[str, Any]] = []
    stats: dict[str, float] = {}
    for tensor_name in ("K", "V"):
        values = np.concatenate(samples[tensor_name]).astype(np.float64)
        block_ratios = np.concatenate(ratios[tensor_name]).astype(np.float64)
        count, edges = np.histogram(values, bins=bins)
        density, _ = np.histogram(values, bins=bins, density=True)
        quantiles = np.quantile(values, probabilities)
        for index in range(len(count)):
            rows.append(
                {
                    "record_type": "histogram",
                    "tensor": tensor_name,
                    "index": index,
                    "x": float((edges[index] + edges[index + 1]) / 2),
                    "empirical": float(density[index]),
                    "reference": float(
                        norm.pdf((edges[index] + edges[index + 1]) / 2)
                    ),
                    "count": int(count[index]),
                    "probability": math.nan,
                }
            )
        for index, probability in enumerate(probabilities):
            rows.append(
                {
                    "record_type": "quantile",
                    "tensor": tensor_name,
                    "index": index,
                    "x": float(norm.ppf(probability)),
                    "empirical": float(quantiles[index]),
                    "reference": float(norm.ppf(probability)),
                    "count": int(len(values)),
                    "probability": float(probability),
                }
            )
        stats[f"{tensor_name.lower()}_sample_count"] = int(len(values))
        stats[f"{tensor_name.lower()}_sample_mean"] = float(values.mean())
        stats[f"{tensor_name.lower()}_sample_std"] = float(values.std())
        stats[f"{tensor_name.lower()}_sample_excess_kurtosis"] = float(
            np.mean((values - values.mean()) ** 4) / values.var() ** 2 - 3.0
        )
        for quantile in (0.10, 0.50, 0.90):
            stats[
                f"{tensor_name.lower()}_absmax_over_rms_q{int(100 * quantile)}"
            ] = float(np.quantile(block_ratios, quantile))
    write_rows(output_root / "standardized_suffix_plot_data.csv", rows)
    write_rows(
        output_root / "standardized_suffix_statistics.csv",
        [{"metric": key, "value": value} for key, value in stats.items()],
    )
    return pd.DataFrame(rows), stats, sorted(set(source_files))


def figure_value_geometry(
    plot_data: pd.DataFrame,
    stats: dict[str, float],
    tree_path: Path,
    figures: Path,
) -> None:
    tree = torch.load(tree_path, map_location="cpu", weights_only=False)
    k_levels = tree["K"]["reconstruction"][3].float().numpy()
    v_ratio = stats["v_absmax_over_rms_q50"]
    absmax_levels = np.arange(-7, 8, dtype=np.float64) / 7.0 * v_ratio
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))
    for column, tensor_name in enumerate(("K", "V")):
        hist = plot_data[
            (plot_data["record_type"] == "histogram")
            & (plot_data["tensor"] == tensor_name)
        ].sort_values("index")
        ax = axes[0, column]
        ax.plot(
            hist["x"],
            hist["empirical"],
            color=TENSOR_COLORS[tensor_name],
            linewidth=2.1,
            label=f"{tensor_name}经验分布",
        )
        ax.plot(
            hist["x"],
            hist["reference"],
            color="#111827",
            linestyle="--",
            linewidth=1.4,
            label="标准高斯 N(0,1)",
        )
        levels = k_levels if tensor_name == "K" else absmax_levels
        label = (
            "K Gaussian-tree B4重建点"
            if tensor_name == "K"
            else "V absmax B4网格（中位block）"
        )
        ymax = float(hist["empirical"].max())
        for level in levels:
            ax.vlines(
                level,
                0,
                ymax * 0.12,
                color="#7c3aed" if tensor_name == "K" else "#ea580c",
                alpha=0.72,
                linewidth=0.9,
            )
        ax.plot([], [], color="#7c3aed" if tensor_name == "K" else "#ea580c", label=label)
        ax.set_xlim(-5, 5)
        ax.set_title(
            "键缓存：标准化后缀直方图"
            if tensor_name == "K"
            else "值缓存：标准化残差直方图",
            fontproperties=ZH_FONT,
        )
        ax.set_xlabel("数值 / block RMS", fontproperties=ZH_FONT)
        ax.set_ylabel("概率密度", fontproperties=ZH_FONT)
        ax.legend(fontsize=8, prop=ZH_FONT)
        ax.grid(alpha=0.18)

        qq = plot_data[
            (plot_data["record_type"] == "quantile")
            & (plot_data["tensor"] == tensor_name)
        ].sort_values("index")
        ax = axes[1, column]
        ax.plot(
            qq["x"],
            qq["empirical"],
            color=TENSOR_COLORS[tensor_name],
            linewidth=2,
        )
        bound = max(
            abs(float(qq["x"].min())),
            abs(float(qq["x"].max())),
            abs(float(qq["empirical"].min())),
            abs(float(qq["empirical"].max())),
        )
        ax.plot([-bound, bound], [-bound, bound], "--", color="#111827", linewidth=1.2)
        ax.set_title(
            "键缓存：高斯分位数图"
            if tensor_name == "K"
            else "值缓存：高斯分位数图",
            fontproperties=ZH_FONT,
        )
        ax.text(
            0.03,
            0.94,
            f"采样超额峰度 = "
            f"{stats[f'{tensor_name.lower()}_sample_excess_kurtosis']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            fontproperties=ZH_FONT,
        )
        ax.set_xlabel("N(0,1)理论分位数", fontproperties=ZH_FONT)
        ax.set_ylabel("经验分位数", fontproperties=ZH_FONT)
        ax.grid(alpha=0.18)
    fig.suptitle(
        "图三　键/值缓存的标准化边缘分布几乎相同，不能据此决定量化分流",
        fontproperties=ZH_FONT,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "两者的差别在码流约束与下游attention误差，而不在一维直方图；"
        "V的mean–tail证据见图四。",
        ha="center",
        fontsize=9,
        fontproperties=ZH_FONT,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    save_figure(fig, figures, "03_suffix_distribution_quantizer_geometry")


def aggregate_v_exact(v_exact: pd.DataFrame) -> pd.DataFrame:
    methods = ("no_frame_mean_no_hadamard", "absmax_g128")
    metrics = (
        "historical_V_mse",
        "attention_output_squared_error_mean",
        "attention_output_squared_error_P99",
        "attention_output_squared_error_P999",
        "attention_output_squared_error_CVaR99",
        "residual_physical_bits_per_value",
    )
    return (
        v_exact[v_exact["method"].isin(methods)]
        .groupby(["method", "attention_half"], as_index=False)[list(metrics)]
        .mean()
    )


def figure_v_tail(
    v_exact: pd.DataFrame,
    output_root: Path,
    figures: Path,
) -> dict[str, float]:
    aggregate = aggregate_v_exact(v_exact)
    write_rows(
        output_root / "v_gaussian_vs_absmax_aggregate.csv",
        aggregate.to_dict("records"),
    )
    gaussian = aggregate[
        (aggregate["method"] == "no_frame_mean_no_hadamard")
        & (aggregate["attention_half"] == "standard_rope")
    ].iloc[0]
    absmax = aggregate[
        (aggregate["method"] == "absmax_g128")
        & (aggregate["attention_half"] == "standard_rope")
    ].iloc[0]
    rate_delta = abs(
        float(gaussian["residual_physical_bits_per_value"])
        - float(absmax["residual_physical_bits_per_value"])
    )
    if rate_delta >= 0.001:
        raise AssertionError(f"Gaussian/absmax rate delta is {rate_delta}")

    metric_labels = (
        ("historical_V_mse", "V MSE"),
        ("attention_output_squared_error_mean", "Output mean"),
        ("attention_output_squared_error_P99", "P99"),
        ("attention_output_squared_error_P999", "P99.9"),
        ("attention_output_squared_error_CVaR99", "CVaR99"),
    )
    layer_rows: list[dict[str, Any]] = []
    selected = v_exact[
        v_exact["method"].isin(
            ("no_frame_mean_no_hadamard", "absmax_g128")
        )
    ]
    for half in ("standard_rope", "camera_prope"):
        half_data = selected[selected["attention_half"] == half]
        for layer in sorted(half_data["layer"].unique()):
            layer_data = half_data[half_data["layer"] == layer]
            for metric, _label in metric_labels[1:]:
                values = (
                    layer_data.groupby("method")[metric].mean().to_dict()
                )
                layer_rows.append(
                    {
                        "attention_half": half,
                        "layer": int(layer),
                        "metric": metric,
                        "gaussian": float(
                            values["no_frame_mean_no_hadamard"]
                        ),
                        "absmax": float(values["absmax_g128"]),
                        "gaussian_over_absmax": float(
                            values["no_frame_mean_no_hadamard"]
                            / values["absmax_g128"]
                        ),
                    }
                )
    write_rows(output_root / "v_tail_tradeoff_by_layer.csv", layer_rows)
    layer_frame = pd.DataFrame(layer_rows)

    fig, axes = plt.subplots(2, 2, figsize=(13.4, 8.4))
    for column, half in enumerate(("standard_rope", "camera_prope")):
        ax = axes[0, column]
        g = aggregate[
            (aggregate["method"] == "no_frame_mean_no_hadamard")
            & (aggregate["attention_half"] == half)
        ].iloc[0]
        a = aggregate[
            (aggregate["method"] == "absmax_g128")
            & (aggregate["attention_half"] == half)
        ].iloc[0]
        ratios = [float(g[metric] / a[metric]) for metric, _ in metric_labels]
        colors = ["#2563eb" if value < 1 else "#dc2626" for value in ratios]
        ax.bar(
            [label for _metric, label in metric_labels],
            ratios,
            color=colors,
            alpha=0.82,
        )
        ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.2)
        for index, ratio in enumerate(ratios):
            ax.text(
                index,
                ratio + (0.03 if ratio >= 1 else -0.06),
                f"{ratio:.2f}×",
                ha="center",
                va="bottom" if ratio >= 1 else "top",
                fontsize=9,
            )
        ax.set_title(
            "标准RoPE" if half == "standard_rope" else "相机PRoPE",
            fontproperties=ZH_FONT,
        )
        ax.set_ylabel(
            "Gaussian / absmax（越低越好）", fontproperties=ZH_FONT
        )
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.2)

    for column, metric in enumerate(
        (
            "attention_output_squared_error_P999",
            "attention_output_squared_error_CVaR99",
        )
    ):
        ax = axes[1, column]
        for half, style, label in (
            ("standard_rope", "-", "标准RoPE"),
            ("camera_prope", "--", "相机PRoPE"),
        ):
            view = layer_frame[
                (layer_frame["attention_half"] == half)
                & (layer_frame["metric"] == metric)
            ].sort_values("layer")
            ax.plot(
                view["layer"],
                view["gaussian_over_absmax"],
                linestyle=style,
                linewidth=2,
                label=label,
            )
        ax.axhline(1.0, color="#111827", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Transformer层", fontproperties=ZH_FONT)
        ax.set_ylabel("Gaussian / absmax", fontproperties=ZH_FONT)
        ax.set_title(
            "逐层P99.9比值"
            if column == 0
            else "逐层CVaR99比值",
            fontproperties=ZH_FONT,
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, prop=ZH_FONT)
    fig.suptitle(
        "图四　值缓存选择极值量化是为了保护尾部，而不是最小化平均误差",
        fontproperties=ZH_FONT,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, figures, "04_v_mean_tail_tradeoff")

    improvements = {
        "v_mse_gaussian_improvement_vs_absmax_pct": float(
            (1 - gaussian["historical_V_mse"] / absmax["historical_V_mse"])
            * 100
        ),
        "v_output_mean_gaussian_improvement_vs_absmax_pct": float(
            (
                1
                - gaussian["attention_output_squared_error_mean"]
                / absmax["attention_output_squared_error_mean"]
            )
            * 100
        ),
        "v_p999_gaussian_change_vs_absmax_pct": float(
            (
                gaussian["attention_output_squared_error_P999"]
                / absmax["attention_output_squared_error_P999"]
                - 1
            )
            * 100
        ),
        "v_cvar99_gaussian_change_vs_absmax_pct": float(
            (
                gaussian["attention_output_squared_error_CVaR99"]
                / absmax["attention_output_squared_error_CVaR99"]
                - 1
            )
            * 100
        ),
        "v_gaussian_bpv": float(
            gaussian["residual_physical_bits_per_value"]
        ),
        "v_absmax_bpv": float(absmax["residual_physical_bits_per_value"]),
    }
    expected = {
        "v_mse_gaussian_improvement_vs_absmax_pct": 30.6213,
        "v_output_mean_gaussian_improvement_vs_absmax_pct": 15.7960,
        "v_p999_gaussian_change_vs_absmax_pct": 86.3068,
        "v_cvar99_gaussian_change_vs_absmax_pct": 66.3352,
    }
    for key, target in expected.items():
        if abs(improvements[key] - target) > 0.02:
            raise AssertionError(
                f"{key}={improvements[key]:.6f}, expected {target:.6f}"
            )
    return improvements


def figure_routing(
    marginals: pd.DataFrame,
    routing: dict[str, Any],
    output_root: Path,
    figures: Path,
) -> dict[str, float]:
    k = marginals[marginals["tensor"] == "K"].copy()
    gain = k.pivot(index="layer", columns="head", values="gain_4_to_5")
    cost = k.pivot(index="layer", columns="head", values="cost_4_to_3")
    offset = np.asarray(routing["offsets"]["K"], dtype=np.int8)
    if gain.shape != (30, 24) or cost.shape != (30, 24):
        raise ValueError("invalid K marginal grid")
    heatmap_rows: list[dict[str, Any]] = []
    for layer in range(30):
        for head in range(24):
            heatmap_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "gain_4_to_5": float(gain.loc[layer, head]),
                    "cost_4_to_3": float(cost.loc[layer, head]),
                    "route_offset": int(offset[layer, head]),
                }
            )
    write_rows(output_root / "k_routing_heatmap_data.csv", heatmap_rows)

    fig, axes = plt.subplots(
        1, 3, figsize=(15.6, 6.3), gridspec_kw={"width_ratios": [1, 1, 0.85]}
    )
    for ax, matrix, title in (
        (
            axes[0],
            np.log10(np.clip(gain.to_numpy(), 1e-20, None)),
            "B4→B5功能收益（log10）",
        ),
        (
            axes[1],
            np.log10(np.clip(cost.to_numpy(), 1e-20, None)),
            "B4→B3功能代价（log10）",
        ),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xlabel("注意力头", fontproperties=ZH_FONT)
        ax.set_ylabel("层", fontproperties=ZH_FONT)
        ax.set_title(title, fontproperties=ZH_FONT)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    route_cmap = mcolors.ListedColormap(["#60a5fa", "#e5e7eb", "#f97316"])
    image = axes[2].imshow(
        offset, aspect="auto", cmap=route_cmap, vmin=-1.5, vmax=1.5
    )
    axes[2].set_xlabel("注意力头", fontproperties=ZH_FONT)
    axes[2].set_ylabel("层", fontproperties=ZH_FONT)
    axes[2].set_title("历史12.5%功能位路由", fontproperties=ZH_FONT)
    legend = [
        Line2D([0], [0], color="#60a5fa", lw=7, label="低价值：B3"),
        Line2D([0], [0], color="#e5e7eb", lw=7, label="普通：B4"),
        Line2D([0], [0], color="#f97316", lw=7, label="高价值：B5"),
    ]
    axes[2].legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        prop=ZH_FONT,
    )
    fig.suptitle(
        "图五　键缓存的功能敏感度高度不均匀，支持静态位路由",
        fontproperties=ZH_FONT,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "注意：该12.5%路由在Frame-mean预测器之前拟合；"
        "它只支持“应当非均匀分配”，不证明当前路由已经最优。",
        ha="center",
        fontsize=9,
        fontproperties=ZH_FONT,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    save_figure(fig, figures, "05_k_layer_head_functional_routing")

    k_gain_cv = float(k["gain_4_to_5"].std() / k["gain_4_to_5"].mean())
    k_cost_cv = float(k["cost_4_to_3"].std() / k["cost_4_to_3"].mean())
    return {
        "historical_route_low_fraction": float((offset == -1).mean()),
        "historical_route_normal_fraction": float((offset == 0).mean()),
        "historical_route_high_fraction": float((offset == 1).mean()),
        "k_gain_4_to_5_cv": k_gain_cv,
        "k_cost_4_to_3_cv": k_cost_cv,
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    configure_plotting()
    source = load_sources(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    figures = args.output_root / "figures"
    figures.mkdir(exist_ok=False)

    config = {
        "schema_version": "current_kv_codec_data_rationale.config.v1",
        "analysis_only": True,
        "runs_video_generation": False,
        "runs_similarity": False,
        "runs_vbench": False,
        "position_boundary": "pre_RoPE_pre_PRoPE",
        "k_codec": "frame_mean_predictor+G128_hadamard+Gaussian_tree+static_layer_head_routing",
        "v_codec": "G128_absmax_symmetric_INT4",
        "k_root": str(args.k_root),
        "v_root": str(args.v_root),
        "routing_root": str(args.routing_root),
        "tree_params": str(args.tree_params),
        "data_root": str(args.data_root),
        "geometry_root": str(args.geometry_root),
        "sample_prompts": list(args.sample_prompts),
        "sample_layers": list(args.sample_layers),
        "samples_per_head_span": args.samples_per_head_span,
        "block_ratios_per_head_span": args.block_ratios_per_head_span,
        "assignment_batch_size": args.assignment_batch_size,
        "device": args.device,
        "seed": args.seed,
        "codec_seed": args.codec_seed,
        "historical_routing_caveat": (
            "the plotted 12.5-percent functional routing was fitted before "
            "the frame-mean predictor and is mechanism evidence, not proof "
            "that the current routing table is optimal"
        ),
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)

    headline: dict[str, float] = {}
    headline.update(
        figure_frame_mean(
            source["k_distribution"],
            source["v_distribution"],
            args.output_root,
            figures,
        )
    )
    headline.update(
        figure_hadamard(
            source["k_distribution"],
            source["v_distribution"],
            args.output_root,
            figures,
        )
    )
    plot_data, sample_stats, sampled_sources = sample_value_geometry(
        args, args.output_root
    )
    headline.update(sample_stats)
    figure_value_geometry(
        plot_data, sample_stats, args.tree_params, figures
    )
    headline.update(
        figure_v_tail(source["v_exact"], args.output_root, figures)
    )
    headline.update(
        figure_routing(
            source["marginals"],
            source["routing"],
            args.output_root,
            figures,
        )
    )
    headline_rows = [
        {"metric": key, "value": value} for key, value in headline.items()
    ]
    write_rows(args.output_root / "headline_statistics.csv", headline_rows)

    source_hashes = {
        str(path): sha256_file(path)
        for path in source["paths"].values()
        if path.is_file()
    }
    output_paths = sorted(
        path
        for path in args.output_root.rglob("*")
        if path.is_file() and path.name != "run_info.json"
    )
    k_codebook = args.data_root / "nonorm_k_l8c64_codebooks"
    v_codebook = args.data_root / "nonorm_v_l8c64_codebooks"
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    write_json(
        args.output_root / "run_info.json",
        {
            "git_commit": git_commit,
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": {
                "K": str(k_codebook),
                "V": str(v_codebook),
                "tree": str(args.tree_params),
            },
            "codebook_hash": {
                "K": stable_directory_hash(k_codebook),
                "V": stable_directory_hash(v_codebook),
                "tree": sha256_file(args.tree_params),
            },
            "hostname": socket.gethostname(),
            "gpu_type": torch.cuda.get_device_name(torch.device(args.device)),
            "seed": args.seed,
            "prompt_list": list(args.sample_prompts),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "source_hashes": source_hashes,
            "sampled_capture_files": sampled_sources,
            "output_hashes": {
                str(path): sha256_file(path) for path in output_paths
            },
            "environment": {
                "python": sys.version,
                "torch": torch.__version__,
                "matplotlib": matplotlib.__version__,
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", ""
                ),
            },
        },
    )
    print(f"[complete] {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
