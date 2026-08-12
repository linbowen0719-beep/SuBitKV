#!/usr/bin/env python3
"""Plot Section 3 shared-structure evidence from completed Wan2.2 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

try:
    import umap  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency gate
    raise RuntimeError("Install umap-learn before running this script") from exc


REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/data/bowen/t2v_kv_bench_moviegen_20train_10test")
TRANSFER_SOURCE = DATA_ROOT / "metrics/rvq_residual_by_level_rows.csv"
USAGE_DIR = DATA_ROOT / "l16c64_codebook_usage_20260722/rows"
CODEBOOK_DIR = DATA_ROOT / "nonorm_k_l16c64_codebooks"
DEFAULT_OUTPUT = REPO / "results/analysis/wan2_2_5b_iclr2027_section3_20260811"
COLORS = {"K": "#2563EB", "V": "#DC2626"}
SEED = 20270811
TEST_PROMPTS = tuple(range(10))
SAMPLES_PER_PROMPT = 1000
PCA_DIMS = 32
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.08


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
            "font.size": 7.5,
            "axes.labelsize": 8.25,
            "axes.titlesize": 8.75,
            "xtick.labelsize": 7.25,
            "ytick.labelsize": 7.25,
            "legend.fontsize": 7.0,
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


def select_representative_unit() -> tuple[pd.DataFrame, int, int, float]:
    paths = sorted(USAGE_DIR.glob("shard*.csv"))
    if not paths:
        raise FileNotFoundError(f"No usage shards under {USAGE_DIR}")
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    stage1 = rows[(rows["tensor"] == "K") & (rows["stage"] == 1)].copy()
    if len(stage1) != 30 * 24:
        raise ValueError(f"Expected 720 K stage-1 units, got {len(stage1)}")
    stage1["stage1_distortion_ratio"] = (
        stage1["residual_mse_after"] / stage1["residual_mse_before"]
    )
    median = float(stage1["stage1_distortion_ratio"].median())
    stage1["distance_to_median"] = (
        stage1["stage1_distortion_ratio"] - median
    ).abs()
    stage1 = stage1.sort_values(["distance_to_median", "layer", "head"]).reset_index(drop=True)
    stage1["selected"] = False
    stage1.loc[0, "selected"] = True
    return stage1, int(stage1.loc[0, "layer"]), int(stage1.loc[0, "head"]), median


def sample_k_vectors(layer: int, head: int) -> tuple[np.ndarray, pd.DataFrame, list[Path]]:
    rng = np.random.default_rng(SEED)
    arrays: list[np.ndarray] = []
    metadata: list[dict[str, int]] = []
    source_paths: list[Path] = []
    for prompt in TEST_PROMPTS:
        path = DATA_ROOT / "kv_dump/test" / str(prompt) / f"layer_{layer}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
        value = payload["K"][:4, head].reshape(-1, 128).float().numpy()
        if len(value) != 3520:
            raise ValueError(f"Unexpected K sample shape at {path}: {value.shape}")
        indices = np.sort(rng.choice(len(value), size=SAMPLES_PER_PROMPT, replace=False))
        arrays.append(value[indices])
        metadata.extend(
            {"prompt_id": prompt, "chunk_id": 0, "flat_token_index": int(index)}
            for index in indices
        )
        source_paths.append(path)
    return np.concatenate(arrays, axis=0), pd.DataFrame(metadata), source_paths


def embed_k_and_prototypes(
    vectors: np.ndarray,
    layer: int,
    head: int,
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    codebook_path = CODEBOOK_DIR / f"layer_{layer}_head_{head}.pt"
    payload = torch.load(codebook_path, map_location="cpu", weights_only=False)
    prototypes = payload["codebooks"][0].float().numpy()
    if prototypes.shape != (64, 128) or payload.get("input_norm_mode") != "none":
        raise ValueError(
            f"Expected 64x128 first-stage prototypes in raw feature space: {codebook_path}"
        )

    pca = PCA(n_components=PCA_DIMS, svd_solver="randomized", random_state=SEED)
    vectors_pca = pca.fit_transform(vectors)
    prototypes_pca = pca.transform(prototypes)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=SEED,
        transform_seed=SEED,
        n_jobs=1,
    )
    vectors_2d = reducer.fit_transform(vectors_pca)
    prototypes_2d = reducer.transform(prototypes_pca)

    np.savez_compressed(
        data_dir / "fig02_pca_parameters.npz",
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        singular_values=pca.singular_values_,
        seed=np.array([SEED]),
    )
    joblib.dump(reducer, data_dir / "fig02_umap_model.joblib", compress=3)
    points = pd.DataFrame({"umap_x": vectors_2d[:, 0], "umap_y": vectors_2d[:, 1]})
    proto = pd.DataFrame(
        {
            "prototype_id": np.arange(len(prototypes_2d)),
            "umap_x": prototypes_2d[:, 0],
            "umap_y": prototypes_2d[:, 1],
            "rvq_stage": 1,
        }
    )
    return points, proto, codebook_path


def prepare_transfer(source: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fixed = source[(source["variant"] == "l4c256") & (source["level"] == 1)]
    transfer = (
        fixed.pivot(
            index=["tensor", "layer", "head"],
            columns="split",
            values="fraction_of_initial_norm_mse",
        )
        .reset_index()
        .rename(columns={"train": "train_distortion", "test": "heldout_distortion"})
    )
    transfer["absolute_relative_gap"] = (
        (transfer["heldout_distortion"] - transfer["train_distortion"]).abs()
        / transfer["train_distortion"]
    )
    transfer["variant"] = "l4c256"
    transfer["rvq_stage"] = 1
    summary: dict[str, dict[str, float | int]] = {}
    for tensor in ("K", "V"):
        group = transfer[transfer["tensor"] == tensor]
        pearson = float(group["train_distortion"].corr(group["heldout_distortion"]))
        summary[tensor] = {
            "components": int(len(group)),
            "pearson_r": pearson,
            "mean_absolute_relative_gap_percent": float(
                100 * group["absolute_relative_gap"].mean()
            ),
        }
    summary["formula"] = "mean(abs(D_heldout - D_train) / D_train)"
    return transfer, summary


def plot(
    points: pd.DataFrame,
    prototypes: pd.DataFrame,
    transfer: pd.DataFrame,
    summary: dict,
    figs_dir: Path,
) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"wspace": 0.28})

    ax = axes[0]
    ax.scatter(
        points["umap_x"],
        points["umap_y"],
        s=1.0,
        alpha=0.22,
        linewidths=0,
        color="#64748B",
        label="K samples",
    )
    ax.scatter(
        prototypes["umap_x"],
        prototypes["umap_y"],
        s=24,
        marker="X",
        linewidths=0.45,
        edgecolors="white",
        color="#EA580C",
        zorder=3,
        label="Stage-1 prototypes",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(
        loc="upper right",
        fontsize=6.2,
        handletextpad=0.25,
        borderpad=0.25,
        framealpha=0.88,
    )
    ax.text(-0.15, 1.04, "(a)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.text(0.0, 1.04, "Recurring KV structure", transform=ax.transAxes, fontsize=8.75)

    ax = axes[1]
    all_values = transfer[["train_distortion", "heldout_distortion"]].to_numpy()
    lower = float(all_values[all_values > 0].min()) * 0.8
    upper = float(all_values.max()) * 1.18
    ax.plot([lower, upper], [lower, upper], color="#64748B", linestyle="--", linewidth=0.9, label="$y=x$")
    markers = {"K": "o", "V": "^"}
    for tensor in ("K", "V"):
        group = transfer[transfer["tensor"] == tensor]
        ax.scatter(
            group["train_distortion"],
            group["heldout_distortion"],
            s=9,
            marker=markers[tensor],
            alpha=0.25,
            linewidths=0,
            color=COLORS[tensor],
            label=tensor,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Train distortion")
    ax.set_ylabel("Held-out distortion")
    ax.grid(True)
    ax.legend(loc="lower right", ncol=3, handlelength=1.1, columnspacing=0.7)
    annotation = (
        "Pearson $r$ / mean gap\n"
        f"K: {summary['K']['pearson_r']:.3f} / {summary['K']['mean_absolute_relative_gap_percent']:.1f}%\n"
        f"V: {summary['V']['pearson_r']:.3f} / {summary['V']['mean_absolute_relative_gap_percent']:.1f}%"
    )
    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        bbox={"facecolor": "white", "edgecolor": "#CBD5E1", "linewidth": 0.5, "pad": 1.6},
    )
    ax.text(-0.18, 1.04, "(b)", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax.text(0.0, 1.04, "Cross-video generalization", transform=ax.transAxes, fontsize=8.75)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.88, wspace=0.3)
    fig.savefig(figs_dir / "fig02_shared_structure.pdf", dpi=300, facecolor="white")
    fig.savefig(figs_dir / "fig02_shared_structure.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    figs_dir = output / "figs"
    data_dir = output / "figure_data"
    fixed_targets = [
        figs_dir / "fig02_shared_structure.pdf",
        figs_dir / "fig02_shared_structure.png",
        data_dir / "fig02_umap_points.csv",
        data_dir / "fig02_umap_prototypes.csv",
        data_dir / "fig02_representative_selection.csv",
        data_dir / "fig02_cross_video_generalization.csv",
        data_dir / "fig02_generalization_summary.json",
        data_dir / "fig02_pca_parameters.npz",
        data_dir / "fig02_umap_model.joblib",
        data_dir / "fig02_config.json",
        output / "fig02_run_info.json",
    ]
    existing = [str(path) for path in fixed_targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite:\n" + "\n".join(existing))
    if not TRANSFER_SOURCE.is_file():
        raise FileNotFoundError(TRANSFER_SOURCE)
    figs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    selection, layer, head, median = select_representative_unit()
    vectors, sample_meta, vector_paths = sample_k_vectors(layer, head)
    points, prototypes, codebook_path = embed_k_and_prototypes(
        vectors, layer, head, data_dir
    )
    points = pd.concat([points, sample_meta], axis=1)
    transfer, summary = prepare_transfer(pd.read_csv(TRANSFER_SOURCE))

    points.to_csv(data_dir / "fig02_umap_points.csv", index=False)
    prototypes.to_csv(data_dir / "fig02_umap_prototypes.csv", index=False)
    selection.to_csv(data_dir / "fig02_representative_selection.csv", index=False)
    transfer.to_csv(data_dir / "fig02_cross_video_generalization.csv", index=False)
    write_json(data_dir / "fig02_generalization_summary.json", summary)
    config = {
        "figure_size_inches": [7.0, 2.3],
        "representative_selection": {
            "criterion": "minimum absolute distance to median K stage-1 distortion ratio across 30x24 units; ties by layer then head",
            "selected_layer": layer,
            "selected_head": head,
            "selected_ratio": float(selection.loc[0, "stage1_distortion_ratio"]),
            "population_median": median,
        },
        "sampling": {
            "tensor": "K",
            "test_prompts": list(TEST_PROMPTS),
            "chunk_id": 0,
            "vectors_per_prompt": SAMPLES_PER_PROMPT,
            "total_vectors": len(vectors),
            "seed": SEED,
        },
        "pca": {"n_components": PCA_DIMS, "svd_solver": "randomized", "seed": SEED},
        "umap": {
            "version": umap.__version__,
            "n_components": 2,
            "n_neighbors": UMAP_NEIGHBORS,
            "min_dist": UMAP_MIN_DIST,
            "metric": "euclidean",
            "random_state": SEED,
            "transform_seed": SEED,
            "fit": "sampled KV points only; stage-1 prototypes transformed through the fitted PCA and UMAP",
        },
        "generalization": {
            "variant": "l4c256",
            "rvq_stage": 1,
            "component": "layer-head",
            "gap_formula": summary["formula"],
        },
    }
    write_json(data_dir / "fig02_config.json", config)
    plot(points, prototypes, transfer, summary, figs_dir)

    source_hashes = {str(TRANSFER_SOURCE): sha256(TRANSFER_SOURCE)}
    for path in sorted(USAGE_DIR.glob("shard*.csv")) + vector_paths + [codebook_path]:
        source_hashes[str(path)] = sha256(path)
    run_info = {
        "git_commit": git_commit(),
        "config_path": str((data_dir / "fig02_config.json").resolve()),
        "config_hash": sha256(data_dir / "fig02_config.json"),
        "codebook_path": str(codebook_path),
        "codebook_hash": sha256(codebook_path),
        "hostname": socket.gethostname(),
        "gpu_type": "not used; CPU-only PCA/UMAP and plotting",
        "seed": SEED,
        "prompt_list": {"test": list(TEST_PROMPTS)},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_files": source_hashes,
        "output_hashes": {
            str(path.relative_to(output)): sha256(path)
            for path in fixed_targets[:-1]
            if path.is_file()
        },
    }
    write_json(output / "fig02_run_info.json", run_info)

    print(f"SOURCE Fig2(a) selection: {USAGE_DIR}/shard*.csv fields=tensor,stage,layer,head,residual_mse_before,residual_mse_after")
    print(f"SELECTED UNIT: K layer={layer}, head={head}; criterion=closest to median stage-1 distortion ratio ({median:.9f})")
    print(f"SOURCE Fig2(a) vectors: kv_dump/test/<0..9>/layer_{layer}.pt field=K[:4,{head}]")
    print(f"SAMPLED KV VECTORS: {len(vectors)} ({SAMPLES_PER_PROMPT} per test prompt), seed={SEED}")
    print(f"PROTOTYPE SOURCE: {codebook_path} field=codebooks[0], shape=64x128")
    print(f"PCA: dimensions={PCA_DIMS}, solver=randomized, seed={SEED}")
    print(f"UMAP: version={umap.__version__}, neighbors={UMAP_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, metric=euclidean, seed={SEED}")
    print("SHARED PIPELINE: PCA fit on KV samples; UMAP fit on PCA(KV); prototypes transformed by the same fitted PCA and UMAP")
    print(f"SOURCE Fig2(b): {TRANSFER_SOURCE} fields=variant,level,tensor,layer,head,split,fraction_of_initial_norm_mse")
    print(f"TRAIN/TEST GAP FORMULA: {summary['formula']}")
    print(f"PEARSON: K={summary['K']['pearson_r']:.9f}, V={summary['V']['pearson_r']:.9f}")
    print(f"MEAN GAP: K={summary['K']['mean_absolute_relative_gap_percent']:.6f}%, V={summary['V']['mean_absolute_relative_gap_percent']:.6f}%")
    print(f"OUTPUT: {figs_dir / 'fig02_shared_structure.pdf'}")
    print(f"OUTPUT: {figs_dir / 'fig02_shared_structure.png'}")
    print("FIGURE DIMENSIONS: 7.0 x 2.3 inches; PNG 300 dpi")


if __name__ == "__main__":
    main()
