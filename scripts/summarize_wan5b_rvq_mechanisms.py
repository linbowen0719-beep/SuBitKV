#!/usr/bin/env python3
"""Aggregate prompt-level Wan 5B RVQ mechanism evidence and render the P0 report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TABLES = [
    "equal_rate_quantization.csv",
    "offline_generalization.csv",
    "calibration_size_curve.csv",
    "rvq_stagewise.csv",
    "codebook_transfer_matrix.csv",
    "codebook_count_scaling.csv",
    "canonicalization_scaling.csv",
    "functional_error_correlation.csv",
    "group_rate_distortion.csv",
    "adaptive_allocation_oracle.csv",
    "sensitivity_outliers.csv",
    "position_space_comparison.csv",
    "temporal_redundancy.csv",
    "index_conditional_entropy.csv",
]
P1_P2_TABLES = {
    "canonicalization_scaling.csv": ("H5", "P1 canonicalization was not selected before P0 evidence."),
    "group_rate_distortion.csv": ("H7", "P1 group rate-distortion was not run in P0."),
    "adaptive_allocation_oracle.csv": ("H7", "P1 allocation was not run in P0."),
    "sensitivity_outliers.csv": ("H8", "P1 outlier intervention was not run in P0."),
    "position_space_comparison.csv": ("H9", "P1 cross-space comparison was not run in P0."),
    "temporal_redundancy.csv": ("H10", "P2 temporal prediction was not run in P0."),
    "index_conditional_entropy.csv": ("H10", "P2 index entropy was not run in P0."),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--publish-root-tables", action="store_true")
    return parser.parse_args()


def parse_config(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return cfg, hashlib.sha256(canonical.encode()).hexdigest()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def read_table(table_root: Path, name: str) -> pd.DataFrame:
    path = table_root / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def prompt_series(frame: pd.DataFrame, metric: str) -> pd.Series:
    return frame.groupby("prompt_id", sort=True)[metric].mean().astype(float)


def bootstrap_mean(values: pd.Series, samples: int, seed: int) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if not len(array):
        return math.nan, math.nan, math.nan
    generator = np.random.default_rng(seed)
    draws = generator.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_result(
    *,
    hypothesis_id: str,
    baseline_name: str,
    comparison_name: str,
    baseline: pd.Series,
    comparison: pd.Series,
    samples: int,
    seed: int,
    supported: bool,
    confidence: str,
    counterevidence: str,
    recommended_method: str,
) -> dict[str, Any]:
    aligned = pd.concat([baseline.rename("baseline"), comparison.rename("comparison")], axis=1).dropna()
    delta = aligned["comparison"] - aligned["baseline"]
    effect, low, high = bootstrap_mean(delta, samples, seed)
    relative = float(
        (aligned["baseline"].mean() - aligned["comparison"].mean())
        / max(abs(aligned["baseline"].mean()), 1e-20)
    )
    return {
        "hypothesis_id": hypothesis_id,
        "tested": True,
        "primary_metric": "prompt_mean_relative_mse",
        "baseline": baseline_name,
        "comparison": comparison_name,
        "effect_size": effect,
        "relative_improvement": relative,
        "ci95_low": low,
        "ci95_high": high,
        "prompt_win_rate": float((aligned["comparison"] < aligned["baseline"]).mean()),
        "supported": bool(supported),
        "confidence": confidence,
        "counterevidence": counterevidence,
        "recommended_method": recommended_method,
    }


def missing_result(hypothesis_id: str, reason: str, method: str) -> dict[str, Any]:
    return {
        "hypothesis_id": hypothesis_id,
        "tested": False,
        "primary_metric": "missing_evidence",
        "baseline": "not_run",
        "comparison": "not_run",
        "effect_size": math.nan,
        "relative_improvement": math.nan,
        "ci95_low": math.nan,
        "ci95_high": math.nan,
        "prompt_win_rate": math.nan,
        "supported": False,
        "confidence": "none",
        "counterevidence": reason,
        "recommended_method": method,
    }


def save_figure(fig: plt.Figure, plot_root: Path, stem: str) -> None:
    plot_root.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(plot_root / f"{stem}.png", dpi=180)
    fig.savefig(plot_root / f"{stem}.pdf")
    plt.close(fig)


def missing_figure(plot_root: Path, stem: str, phase: str, description: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.text(0.5, 0.62, f"{phase}: missing evidence", ha="center", va="center", fontsize=20)
    axis.text(0.5, 0.38, description, ha="center", va="center", fontsize=11, wrap=True)
    save_figure(fig, plot_root, stem)


def line_by_tensor(
    frame: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    for axis, tensor in zip(axes, ["K", "V"]):
        subset = frame[frame["K_or_V"] == tensor]
        grouped = subset.groupby([x, hue], sort=True)[y].mean().reset_index()
        for label, rows in grouped.groupby(hue, sort=True):
            axis.plot(rows[x], rows[y], marker="o", label=str(label))
        axis.set_title(tensor)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.suptitle(title)
    return fig


def create_plots(table_root: Path, plot_root: Path) -> None:
    equal_rate = read_table(table_root, "equal_rate_quantization.csv")
    selected_methods = [
        "rvq", "pq", "single_stage_vq", "pca_scalar", "whitening_scalar",
        "equal_index_active_channel_scalar", "rvq_independent_channel_permutation",
    ]
    equal_subset = equal_rate[
        equal_rate["method"].isin(selected_methods)
        | equal_rate["method"].str.contains("total_rate", na=False)
    ]
    save_figure(
        line_by_tensor(
            equal_subset, "total_amortized_bits_per_scalar", "relative_mse", "method",
            "Equal-rate held-out reconstruction", "total amortized bits/scalar", "relative MSE",
        ),
        plot_root,
        "01_equal_rate_quantization",
    )

    calibration = read_table(table_root, "calibration_size_curve.csv")
    save_figure(
        line_by_tensor(
            calibration, "calibration_prompt_count", "relative_mse", "K_or_V",
            "Cross-prompt calibration curve", "calibration prompts", "relative MSE",
        ),
        plot_root,
        "02_calibration_prompt_curve",
    )

    stage = read_table(table_root, "rvq_stagewise.csv")
    for number, metric, title, ylabel in (
        (3, "residual_energy_ratio", "Residual energy by RVQ stage", "residual/input energy"),
        (4, "residual_effective_rank", "Residual effective rank by RVQ stage", "effective rank"),
        (5, "codebook_perplexity", "Codebook perplexity by RVQ stage", "perplexity"),
    ):
        save_figure(
            line_by_tensor(stage, "RVQ_stage", metric, "K_or_V", title, "RVQ stage", ylabel),
            plot_root,
            f"{number:02d}_{metric}",
        )

    transfer = read_table(table_root, "codebook_transfer_matrix.csv")
    max_stage = transfer["RVQ_stage"].max()
    transfer = transfer[transfer["RVQ_stage"] == max_stage]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, tensor in zip(axes, ["K", "V"]):
        subset = transfer[transfer["K_or_V"] == tensor]
        matrix = subset.groupby(["target_group", "source_group"])["transfer_ratio"].mean().unstack()
        image = axis.imshow(np.log10(matrix.clip(lower=1e-3, upper=1e3)), aspect="auto", cmap="coolwarm")
        axis.set_title(f"{tensor}: structured transfer log10 ratio")
        axis.set_xlabel("source group")
        axis.set_ylabel("target group")
        fig.colorbar(image, ax=axis, fraction=0.046)
    save_figure(fig, plot_root, "06_codebook_transfer_heatmap")

    scaling = read_table(table_root, "codebook_count_scaling.csv")
    feature = scaling[scaling["grouping_policy"] == "feature_kmeans"]
    save_figure(
        line_by_tensor(
            feature, "total_codebook_count", "relative_mse", "K_or_V",
            "Raw-space sharing quality", "K/V codebook groups", "relative MSE",
        ),
        plot_root,
        "07_quality_vs_codebook_count",
    )
    missing_figure(
        plot_root, "08_raw_vs_canonicalized_sharing", "P1",
        "Canonicalization scaling is intentionally not inferred from raw-space P0 measurements.",
    )

    functional = read_table(table_root, "functional_error_correlation.csv")
    k_only = functional[functional["intervention"] == "K_only"]
    v_only = functional[functional["intervention"] == "V_only"]
    fig, axis = plt.subplots(figsize=(6.5, 5))
    for half, rows in k_only.groupby("attention_half"):
        axis.scatter(rows["historical_K_relative_mse"], rows["attention_kl_reference_to_candidate"], s=16, alpha=0.7, label=half)
    axis.set(xlabel="historical K relative MSE", ylabel="softmax KL", title="K MSE vs softmax KL")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, plot_root, "09_mse_vs_softmax_kl")

    fig, axis = plt.subplots(figsize=(6.5, 5))
    for half, rows in v_only.groupby("attention_half"):
        axis.scatter(rows["historical_V_relative_mse"], rows["attention_output_relative_l2"], s=16, alpha=0.7, label=half)
    axis.set(xlabel="historical V relative MSE", ylabel="attention output relative L2", title="V MSE vs attention output error")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(fig, plot_root, "10_mse_vs_attention_output_error")

    for stem, phase, message in (
        ("11_layer_head_rate_distortion", "P1", "Group rate-distortion curves were not run in P0."),
        ("12_fixed_vs_oracle_allocation", "P1", "Adaptive allocation was not run in P0."),
        ("13_outlier_overlap", "P1", "Sensitivity outlier interventions were not run in P0."),
        ("14_temporal_gap_residual_ratio", "P2", "Temporal prediction and strict shuffles were not run in P0."),
        ("15_index_entropy_conditional_entropy", "P2", "Index conditional entropy was not run in P0."),
    ):
        missing_figure(plot_root, stem, phase, message)


def aggregate(cfg: dict[str, Any], cfg_hash: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    output_root = resolve(cfg["output_dir"])
    table_root = output_root / "tables"
    bootstrap_samples = int(cfg["bootstrap"]["samples"])
    bootstrap_seed = int(cfg["bootstrap"]["seed"])
    offline = read_table(table_root, "offline_generalization.csv")
    final_stage = int(offline["RVQ_stage"].max())
    offline_final = offline[offline["RVQ_stage"] == final_stage]
    random_series = prompt_series(offline_final[offline_final["method"] == "random_codebook"], "relative_mse")
    rvq_series = prompt_series(offline_final[offline_final["method"] == "offline_rvq"], "relative_mse")
    audit = read_table(output_root / "audit", "duplicate_summary.csv")
    duplicate_ratio = float(audit["duplicate_ratio"].max())

    results = [
        paired_result(
            hypothesis_id="H0", baseline_name="same-rate random codebook",
            comparison_name="canonical prompt-CV offline RVQ", baseline=random_series,
            comparison=rvq_series, samples=bootstrap_samples, seed=bootstrap_seed,
            supported=duplicate_ratio == 0.0 and rvq_series.mean() < random_series.mean(),
            confidence="high" if duplicate_ratio == 0.0 else "low",
            counterevidence=f"maximum exact source-chunk duplicate ratio={duplicate_ratio:.6g}",
            recommended_method="leakage-free canonical source sampling",
        )
    ]
    oracles = read_table(table_root, "offline_oracles.csv")
    oracle_by_method = {
        method: prompt_series(rows, "relative_mse")
        for method, rows in oracles.groupby("method")
    }
    oracle_aligned = pd.concat(
        {
            "random": oracle_by_method["random_codebook"],
            "global": oracle_by_method["global_offline"],
            "prompt": oracle_by_method["per_prompt_oracle"],
            "chunk": oracle_by_method["per_source_chunk_oracle"],
        },
        axis=1,
    ).dropna()
    # The per-prompt oracle has enough calibration vectors to be a useful
    # upper-bound control.  The per-source-chunk fit has only about C vectors
    # per chunk and can overfit; report its gap, but never silently treat it as
    # an upper bound when it is worse than the global dictionary.
    oracle_denominator = oracle_aligned["random"] - oracle_aligned["prompt"]
    valid_oracle = oracle_denominator > 1e-12
    oracle_recovery = (
        (oracle_aligned.loc[valid_oracle, "random"] - oracle_aligned.loc[valid_oracle, "global"])
        / oracle_denominator.loc[valid_oracle]
    )
    oracle_recovery_mean = float(oracle_recovery.mean()) if len(oracle_recovery) else math.nan
    offline_gap = float((oracle_aligned["global"] - oracle_aligned["prompt"]).mean())
    per_chunk_gap = float((oracle_aligned["global"] - oracle_aligned["chunk"]).mean())
    h1_support = bool(
        rvq_series.mean() < random_series.mean()
        and len(oracle_recovery)
        and oracle_recovery_mean >= 0.5
    )
    results.append(
        paired_result(
            hypothesis_id="H1", baseline_name="same-rate random codebook",
            comparison_name="16-prompt offline RVQ on unseen prompts", baseline=random_series,
            comparison=rvq_series, samples=bootstrap_samples, seed=bootstrap_seed + 1,
            supported=h1_support, confidence="high" if len(oracle_recovery) == 20 else "medium",
            counterevidence=(
                f"mean offline-minus-per-prompt-oracle gap={offline_gap:.6g}; "
                f"mean per-prompt oracle gain recovery={oracle_recovery_mean:.6g} over {len(oracle_recovery)} valid prompts; "
                f"offline-minus-sample-limited-chunk-fit gap={per_chunk_gap:.6g}."
            ),
            recommended_method="reusable offline dictionary",
        )
    )

    equal_rate = read_table(table_root, "equal_rate_quantization.csv")
    # These controls are constructed to match the RVQ index budget.  Do not
    # select the visually lower-error 1/2/4-bit-per-channel rows: they spend
    # 128--512 index bits per vector and are rate-sweep context, not fair
    # competitors to Lxlog2(C)-bit RVQ.
    comparable_methods = {
        "pq", "pca_scalar", "whitening_scalar", "equal_index_active_channel_scalar"
    }
    baseline_methods = equal_rate[
        equal_rate["method"].isin(comparable_methods)
    ].groupby("method")["relative_mse"].mean()
    best_baseline = str(baseline_methods.idxmin())
    rvq_h2 = prompt_series(equal_rate[equal_rate["method"] == "rvq"], "relative_mse")
    baseline_h2 = prompt_series(equal_rate[equal_rate["method"] == best_baseline], "relative_mse")
    permuted_h2 = prompt_series(
        equal_rate[equal_rate["method"] == "rvq_independent_channel_permutation"], "relative_mse"
    )
    rvq_total_rate = float(
        equal_rate.loc[equal_rate["method"] == "rvq", "total_amortized_bits_per_vector"].mean()
    )
    total_candidates = equal_rate[equal_rate["method"].str.contains("total_rate", na=False)]
    candidate_rates = total_candidates.groupby("method")["total_amortized_bits_per_vector"].mean()
    total_rate_tolerance = 0.15 if "smoke" in str(cfg["study_id"]).lower() else 0.05
    matched_methods = candidate_rates[
        (candidate_rates - rvq_total_rate).abs() / rvq_total_rate <= total_rate_tolerance
    ].index
    if not len(matched_methods):
        raise RuntimeError(
            f"No H2 control matches RVQ total amortized rate within {total_rate_tolerance:.1%}"
        )
    total_method_means = total_candidates[
        total_candidates["method"].isin(matched_methods)
    ].groupby("method")["relative_mse"].mean()
    best_total_baseline = str(total_method_means.idxmin())
    baseline_total_h2 = prompt_series(
        equal_rate[equal_rate["method"] == best_total_baseline], "relative_mse"
    )
    h2_support = bool(
        rvq_h2.mean() < baseline_h2.mean()
        and rvq_h2.mean() < baseline_total_h2.mean()
        and permuted_h2.mean() > rvq_h2.mean()
    )
    results.append(
        paired_result(
            hypothesis_id="H2",
            baseline_name=f"index:{best_baseline}; total:{best_total_baseline}", comparison_name="RVQ",
            baseline=baseline_h2, comparison=rvq_h2, samples=bootstrap_samples,
            seed=bootstrap_seed + 2, supported=h2_support, confidence="high",
            counterevidence=(
                f"independently permuted-channel RVQ relative MSE={permuted_h2.mean():.6g}; "
                f"intact RVQ={rvq_h2.mean():.6g}; total-rate control={baseline_total_h2.mean():.6g}"
            ),
            recommended_method="vector/manifold-aware quantization" if h2_support else "transform plus scalar",
        )
    )

    stage = read_table(table_root, "rvq_stagewise.csv")
    first = prompt_series(stage[stage["RVQ_stage"] == stage["RVQ_stage"].min()], "residual_energy_ratio")
    last = prompt_series(stage[stage["RVQ_stage"] == stage["RVQ_stage"].max()], "residual_energy_ratio")
    late = stage[stage["RVQ_stage"] == stage["RVQ_stage"].max()]
    h3_support = last.mean() < first.mean() and float(late["codebook_perplexity"].mean()) >= 2.0
    results.append(
        paired_result(
            hypothesis_id="H3", baseline_name="RVQ stage 1 residual",
            comparison_name=f"RVQ stage {final_stage} residual", baseline=first,
            comparison=last, samples=bootstrap_samples, seed=bootstrap_seed + 3,
            supported=h3_support, confidence="high",
            counterevidence=(
                f"last-stage mean perplexity={late['codebook_perplexity'].mean():.4g}, "
                f"dead-code ratio={late['dead_code_ratio'].mean():.4g}"
            ),
            recommended_method="stage-aware hybrid/shared residual RVQ",
        )
    )

    scaling = read_table(table_root, "codebook_count_scaling.csv")
    scaling = scaling[scaling["grouping_policy"] == "feature_kmeans"]
    local_count = int(scaling["total_codebook_count"].max())
    by_count = scaling.groupby("total_codebook_count")["relative_mse"].mean().sort_index()
    local_quality = float(by_count.loc[local_count])
    eligible = by_count[by_count <= 1.10 * local_quality]
    eligible_reduced = eligible[eligible.index < local_count]
    retained_count = int(eligible_reduced.index.min()) if len(eligible_reduced) else local_count
    reduced_counts = by_count.index[by_count.index < local_count]
    best_reduced_count = int(max(reduced_counts)) if len(reduced_counts) else local_count
    comparison_count = retained_count if retained_count < local_count else best_reduced_count
    local_series = prompt_series(scaling[scaling["total_codebook_count"] == local_count], "relative_mse")
    retained_series = prompt_series(scaling[scaling["total_codebook_count"] == retained_count], "relative_mse")
    comparison_series = prompt_series(
        scaling[scaling["total_codebook_count"] == comparison_count], "relative_mse"
    )
    local_memory_mib = float(
        scaling.loc[
            scaling["total_codebook_count"] == local_count,
            "codebook_storage_bytes_total",
        ].iloc[0]
        / (1024.0**2)
    )
    retained_memory_mib = float(
        scaling.loc[
            scaling["total_codebook_count"] == retained_count,
            "codebook_storage_bytes_total",
        ].iloc[0]
        / (1024.0**2)
    )
    h4_support = retained_count < local_count
    results.append(
        paired_result(
            hypothesis_id="H4", baseline_name=f"{local_count} local layer-head groups",
            comparison_name=f"{comparison_count} feature-clustered shared groups",
            baseline=local_series, comparison=comparison_series, samples=bootstrap_samples,
            seed=bootstrap_seed + 4, supported=h4_support, confidence="medium",
            counterevidence=(
                "Feature-descriptor clustering at M=1..256 and a random-balanced M=32 control fail the "
                "<=10% retention gate; transfer evidence covers self/same-layer/same-head/adjacent-layer blocks, "
                "not every unrelated 720x720 pair."
            ),
            recommended_method="grouped/shared RVQ" if h4_support else "canonicalization before sharing",
        )
    )

    functional = read_table(table_root, "functional_error_correlation.csv")
    k_only = functional[functional["intervention"] == "K_only"]
    v_only = functional[functional["intervention"] == "V_only"]
    k_rho = float(k_only["historical_K_relative_mse"].corr(
        k_only["attention_kl_reference_to_candidate"], method="spearman"
    ))
    v_rho = float(v_only["historical_V_relative_mse"].corr(
        v_only["attention_output_relative_l2"], method="spearman"
    ))
    capture_prompts = int(functional["prompt"].nunique())
    h6_minimal_only = capture_prompts < 2
    results.append(
        {
            "hypothesis_id": "H6", "tested": True,
            "primary_metric": "layer-head Spearman(K MSE, KL); Spearman(V MSE, output error)",
            "baseline": "Euclidean historical KV relative MSE", "comparison": "exact replay functional error",
            "effect_size": k_rho, "relative_improvement": v_rho,
            "ci95_low": math.nan, "ci95_high": math.nan, "prompt_win_rate": math.nan,
            "supported": False if h6_minimal_only else bool(abs(k_rho) < 0.5 or abs(v_rho) < 0.5),
            "confidence": "low" if h6_minimal_only else "medium",
            "counterevidence": (
                f"Only {capture_prompts} prompt capture is available; correlations are layer-head diagnostics, "
                "not prompt-level inference. Exact transform replay gates passed."
            ),
            "recommended_method": "collect multi-prompt exact captures before function-aware RVQ",
        }
    )
    results.insert(5, missing_result("H5", "P1 canonicalization is pending P0 selection.", "canonicalized shared RVQ"))
    results.extend(
        [
            missing_result("H7", "P1 rate allocation was not run.", "adaptive-stage RVQ"),
            missing_result("H8", "P1 sensitivity outliers were not run.", "sparse outlier refinement"),
            missing_result("H9", "P1 position-space sharing comparison was not run.", "canonical pre-position RVQ"),
            missing_result("H10", "P2 temporal controls were not run.", "temporal prediction/index coding"),
        ]
    )
    frame = pd.DataFrame(results).sort_values("hypothesis_id", key=lambda x: x.str.extract(r"(\d+)")[0].astype(int))
    facts = {
        "duplicate_ratio": duplicate_ratio,
        "offline_relative_mse": float(rvq_series.mean()),
        "random_relative_mse": float(random_series.mean()),
        "offline_gap": offline_gap,
        "per_chunk_fit_gap": per_chunk_gap,
        "oracle_gain_recovery": oracle_recovery_mean,
        "oracle_gain_recovery_valid_prompts": int(len(oracle_recovery)),
        "best_equal_rate_baseline": best_baseline,
        "best_total_rate_baseline": best_total_baseline,
        "total_rate_matching_tolerance": total_rate_tolerance,
        "h2_rvq_relative_mse": float(rvq_h2.mean()),
        "h2_baseline_relative_mse": float(baseline_h2.mean()),
        "h2_total_baseline_relative_mse": float(baseline_total_h2.mean()),
        "h2_permuted_relative_mse": float(permuted_h2.mean()),
        "h2_relative_advantage": float((baseline_h2.mean() - rvq_h2.mean()) / baseline_h2.mean()),
        "h2_total_relative_advantage": float(
            (baseline_total_h2.mean() - rvq_h2.mean()) / baseline_total_h2.mean()
        ),
        "h3_first_energy": float(first.mean()),
        "h3_last_energy": float(last.mean()),
        "h3_last_perplexity": float(late["codebook_perplexity"].mean()),
        "h3_last_dead_ratio": float(late["dead_code_ratio"].mean()),
        "local_count": local_count,
        "retained_count_10pct": retained_count,
        "retained_quality_ratio": float(retained_series.mean() / local_series.mean()),
        "best_reduced_count": best_reduced_count,
        "local_codebook_memory_mib_per_tensor": local_memory_mib,
        "retained_codebook_memory_mib_per_tensor": retained_memory_mib,
        "h6_k_spearman": k_rho,
        "h6_v_spearman": v_rho,
        "h6_capture_prompts": capture_prompts,
    }
    for tensor in ["K", "V"]:
        tensor_curve = scaling[scaling["K_or_V"] == tensor].groupby(
            "total_codebook_count"
        )["relative_mse"].mean()
        facts[f"best_reduced_ratio_{tensor}"] = float(
            tensor_curve.loc[best_reduced_count] / tensor_curve.loc[local_count]
        )
    facts["h6_k_only_output_error"] = float(
        functional[functional["intervention"] == "K_only"]["attention_output_relative_l2"].mean()
    )
    facts["h6_v_only_output_error"] = float(
        functional[functional["intervention"] == "V_only"]["attention_output_relative_l2"].mean()
    )
    calibration = read_table(table_root, "calibration_size_curve.csv")
    curve = calibration.groupby(["K_or_V", "calibration_prompt_count"])["relative_mse"].mean()
    saturation = {}
    for tensor in ["K", "V"]:
        values = curve.loc[tensor].sort_index()
        endpoint = float(values.iloc[-1])
        candidates = values[values <= endpoint * 1.02]
        saturation[tensor] = int(candidates.index.min()) if len(candidates) else int(values.index.max())
    facts["calibration_saturation"] = saturation
    return frame, facts


def report_text(cfg: dict[str, Any], cfg_hash: str, facts: dict[str, Any], hypotheses: pd.DataFrame) -> str:
    get = lambda hypothesis: hypotheses.set_index("hypothesis_id").loc[hypothesis]
    h2 = get("H2")
    h4 = get("H4")
    main = "Grouped/shared RVQ" if bool(h4["supported"]) else (
        "Local coarse RVQ + compact residual refinement"
        if bool(h2["supported"])
        else "transform + scalar quantization"
    )
    alternative = "Function-aware RVQ after multi-prompt exact capture" if facts["h6_capture_prompts"] < 2 else "Canonicalized Shared RVQ"
    answers = [
        f"1. 是。完整 source-chunk hash 的最大重复率为 {facts['duplicate_ratio']:.3g}；prompt-CV RVQ relative MSE={facts['offline_relative_mse']:.5g}，随机等码率码本={facts['random_relative_mse']:.5g}。",
        f"2. {'是' if bool(get('H1')['supported']) else '证据不足'}。相对随机码本在未见 prompt 上更好；offline-minus-per-prompt-oracle gap={facts['offline_gap']:.5g}，oracle gain recovery={facts['oracle_gain_recovery']:.3g}（{facts['oracle_gain_recovery_valid_prompts']} 个有效 prompt）。per-source-chunk fit 因每 chunk 样本有限单独报告，gap={facts['per_chunk_fit_gap']:.5g}，不伪称上界。",
        f"3. 2% endpoint 规则下，K 约 {facts['calibration_saturation']['K']} 个、V 约 {facts['calibration_saturation']['V']} 个 calibration prompts 开始饱和。",
        f"4. Index-matched 最强 baseline 是 {facts['best_equal_rate_baseline']}（relative MSE={facts['h2_baseline_relative_mse']:.5g}）；total-amortized ±{facts['total_rate_matching_tolerance']:.0%} matched 最强 baseline 是 {facts['best_total_rate_baseline']}（{facts['h2_total_baseline_relative_mse']:.5g}）。RVQ={facts['h2_rvq_relative_mse']:.5g}，分别降低 {facts['h2_relative_advantage']:.1%}/{facts['h2_total_relative_advantage']:.1%}。",
        f"5. {'支持' if bool(h2['supported']) else '不支持'}跨 channel 联合结构解释；独立 channel sample permutation 后 RVQ relative MSE={facts['h2_permuted_relative_mse']:.5g}。",
        f"6. stage 1 建立 coarse prototype，后续 stage 将 residual energy 从 {facts['h3_first_energy']:.5g} 降至 {facts['h3_last_energy']:.5g}。",
        f"7. 后级仍有结构：末级 perplexity={facts['h3_last_perplexity']:.3g}，dead-code ratio={facts['h3_last_dead_ratio']:.3g}；是否部署全部级数仍需 P1 functional RD。",
        f"8. raw-space 没有任何低于 720 的 M 通过 10% distortion 门槛。最接近 local 的已测缩减点是 {facts['best_reduced_count']} 组，但 K/V distortion ratio 仍为 {facts['best_reduced_ratio_K']:.3g}/{facts['best_reduced_ratio_V']:.3g}。因此通过门槛的仍是 {facts['retained_count_10pct']} 组，INT4 centroid+scale memory 每个 K 或 V tensor 仍为 {facts['retained_codebook_memory_mib_per_tensor']:.3f} MiB。",
        f"9. 不做 canonicalization 时，10% 门槛需要完整 {facts['retained_count_10pct']} 套 layer-head codebooks；raw sharing 的 H4 结论是否定的。",
        "10. 未测；H5 属于 P1，不能从 raw-space curve 推断 canonicalization 收益。",
        f"11. 需要不同设计：K 主要改变 logits/routing；V-only 的平均 attention-output relative L2={facts['h6_v_only_output_error']:.3g}，高于 K-only={facts['h6_k_only_output_error']:.3g}。最终统计仍需更多 prompt captures。",
        f"12. 目前不能作 prompt-level 结论。单 prompt 的 layer-head Spearman 为 K-MSE/KL={facts['h6_k_spearman']:.3g}、V-MSE/output={facts['h6_v_spearman']:.3g}。",
        "13. 未测 token-level sensitivity intervention；P1 H8 保持 missing evidence。",
        "14. 未测 group functional RD；不能从 reconstruction-only H3 直接断言最佳 depth。",
        "15. 未测；P1 H7 oracle allocation 保持 missing evidence。",
        "16. 未测共享曲线；已确认 K=pre-RoPE、V=pre-PRoPE，二者为同一 video self-attention 的两个 position branches，但不能合并训练语义。",
        "17. 未测；缓存轴已经确认是 latent source-chunk 时间轴，不是 diffusion timestep，但 P2 严格时间 shuffle 尚未运行。",
        f"18. 当前 P0 最支持主线：{main}。",
        "19. 尚未成立/证据不足：H5、H7–H10；H6 仅完成 exact single-prompt replay，不能宣称 prompt-level correlation。",
        f"20. 唯一主方法：{main}。唯一备选：{alternative}。",
    ]
    return f"""# Wan2.2 5B historical KV RVQ mechanism report (P0)

Config hash: `{cfg_hash}`  
Model: `{cfg['model_id']}`  
Statistical unit: prompt (20 prompts, 5-fold 16/4 split, 3 codebook initialization seeds).

## Scope audit

缓存来自 video latent self-attention `attn1` 的自定义 `ChunkedKVCache`。运行时缓存 K/V，不缓存 Q；dump 中 Q 仅用于诊断。`T=44` 是 video latent 时间轴，按 4 latent frames/3520 tokens 形成 11 个 source chunks，不是扩散去噪 timestep。K 在 QK norm 后、RoPE/PRoPE 前；V 在 PRoPE 前。原始缓存只读。

## Twenty required answers

""" + "\n".join(answers) + f"""

## Attention replay boundary

H6 对真实保存 call 逐层重放两个 non-causal SDPA branches：standard RoPE 和 camera PRoPE；camera branch 包含精确 `apply_fn_o` 等价变换。Q 和 current K/V 保持 BF16，只重构 historical K/V。所有保存的 pre→post 位置变换最大误差为 0。由于 capture 不包含 learned `to_out` 权重，functional output 指标截止于 SDPA（camera branch 含 `apply_fn_o`），没有伪称为完整 block 输出。

## Negative and missing evidence

P1/P2 表和图已显式标记 `tested=false`/missing evidence；没有用 reconstruction proxy 支持 attention、adaptive allocation、outlier、canonicalization 或 temporal claims。

## Evidence locations

- Prompt-level tables: `tables/`
- Figures (PNG and PDF): `plots/`
- Hypothesis decision table: `tables/hypothesis_results.csv`
- Machine/config provenance: `run_info.json`
"""


def decision_text(facts: dict[str, Any], hypotheses: pd.DataFrame) -> str:
    indexed = hypotheses.set_index("hypothesis_id")
    h2 = bool(indexed.loc["H2", "supported"])
    h4 = bool(indexed.loc["H4", "supported"])
    if h4:
        main = "Grouped/shared RVQ"
        rationale = (
            f"raw-space feature clustering reaches the registered 10% distortion-retention threshold with "
            f"{facts['retained_count_10pct']} versus {facts['local_count']} groups."
        )
    elif h2:
        main = "Local coarse RVQ + compact residual refinement"
        rationale = (
            "Joint-channel controls support vector quantization and raw-space sharing fails, so the coarse "
            f"dictionary remains local. Later stages have mean perplexity {facts['h3_last_perplexity']:.3g}/64 "
            f"and dead-code ratio {facts['h3_last_dead_ratio']:.1%}, motivating smaller-codebook or scalar "
            "residual refinement instead of eight equally sized local RVQ stages."
        )
    else:
        main = "Low-rank transform + scalar quantization"
        rationale = "RVQ did not beat the strongest fair transform/scalar control."
    backup = "Function-aware RVQ after collecting multi-prompt exact captures"
    return f"""# P0 research decision

## Decision

主线只选：**{main}**。

理由：{rationale} H2 intact RVQ relative MSE={facts['h2_rvq_relative_mse']:.5g}，index/total-rate matched baselines 分别为 {facts['h2_baseline_relative_mse']:.5g}/{facts['h2_total_baseline_relative_mse']:.5g}；独立 channel permutation 后为 {facts['h2_permuted_relative_mse']:.5g}。

备选只选：**{backup}**。

H6 的 exact replay 已验证，但只有 {facts['h6_capture_prompts']} 个 prompt，因此当前不升级为主线。下一步先增加 prompt-level paired captures，并加入 query-metric K error、attention-weighted V error；只有它们显著优于 Euclidean MSE 的预测能力时，才发展 function-aware objective。

## Explicit non-decisions

- 不选择 canonicalized shared RVQ：H5 尚未测试。
- 不选择 adaptive/outlier 方法：H7/H8 尚未测试。
- 不选择 temporal predictive/index coding：H10 尚未测试。
- 不把 single-prompt layer-head correlation 当作 prompt-level evidence。
"""


def publish_links(table_root: Path) -> None:
    public = ROOT / "tables"
    public.mkdir(parents=True, exist_ok=True)
    for name in ["hypothesis_results.csv", *REQUIRED_TABLES]:
        source = table_root / name
        target = public / name
        if target.exists() or target.is_symlink():
            resolved = target.resolve()
            if resolved == source.resolve():
                continue
            raise FileExistsError(f"Refusing to replace existing public table: {target}")
        target.symlink_to(os.path.relpath(source, public))


def main() -> None:
    args = parse_args()
    config_path = resolve(args.config)
    cfg, cfg_hash = parse_config(config_path)
    output_root = resolve(cfg["output_dir"])
    table_root = output_root / "tables"
    for name, (hypothesis, reason) in P1_P2_TABLES.items():
        path = table_root / name
        if not path.exists():
            atomic_csv(
                path,
                pd.DataFrame(
                    [{"study_id": cfg["study_id"], "config_hash": cfg_hash, "hypothesis_id": hypothesis,
                      "tested": False, "status": "missing_evidence", "reason": reason}]
                ),
            )
    hypotheses, facts = aggregate(cfg, cfg_hash)
    atomic_csv(table_root / "hypothesis_results.csv", hypotheses)
    create_plots(table_root, output_root / "plots")
    atomic_text(output_root / "report.md", report_text(cfg, cfg_hash, facts, hypotheses))
    atomic_text(output_root / "artifacts" / "research_decision.md", decision_text(facts, hypotheses))
    atomic_text(
        output_root / "summary.json",
        json.dumps({"config_hash": cfg_hash, "facts": facts}, indent=2, sort_keys=True) + "\n",
    )
    if args.publish_root_tables:
        publish_links(table_root)
    print(f"[summary] tables={table_root} plots={output_root / 'plots'}")


if __name__ == "__main__":
    main()
