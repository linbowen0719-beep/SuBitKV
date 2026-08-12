#!/usr/bin/env python3
"""Build the immutable P1 research synthesis, plots, and deployment decision."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_wan5b_rvq_mechanisms as p0  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def comparison(
    rows: list[dict[str, Any]], candidate: str, baseline: str, intervention: str
) -> dict[str, Any]:
    matches = [
        row for row in rows
        if row["candidate"] == candidate
        and row["baseline"] == baseline
        and row["intervention"] == intervention
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one comparison {(candidate, baseline, intervention)}, got {len(matches)}")
    return matches[0]


def save_figure(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(root / "plots" / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(root / "plots" / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def errorbar_bars(
    labels: list[str], means: list[float], lows: list[float], highs: list[float],
    colors: list[str], title: str, ylabel: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    y = 100 * np.asarray(means)
    yerr = np.vstack([100 * (np.asarray(means) - np.asarray(lows)), 100 * (np.asarray(highs) - np.asarray(means))])
    ax.bar(x, y, color=colors, edgecolor="#202020", linewidth=0.6)
    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="#202020", capsize=4, linewidth=1.2)
    ax.axhline(0, color="#202020", linewidth=1)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return fig


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    result = ["| " + " | ".join(title for _key, title in columns) + " |"]
    result.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        result.append("| " + " | ".join(str(row[key]) for key, _title in columns) + " |")
    return result


def write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg_path = p0.resolve_path(args.config)
    cfg, cfg_hash = p0.parse_config(cfg_path)
    root = p0.resolve_path(cfg["output_dir"])
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite final synthesis: {root}")
    (root / "tables").mkdir(parents=True)
    (root / "plots").mkdir()
    inputs = {name: p0.resolve_path(path) for name, path in cfg["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing synthesis inputs: {missing}")
    source_records = {
        name: {"path": str(path.resolve()), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
        for name, path in inputs.items()
    }
    compact = load_json(inputs["compact_summary"])
    canonical = load_json(inputs["canonical_summary"])
    exact4 = load_json(inputs["exact_four_prompt_summary"])
    schedule_search = load_json(inputs["schedule_search_summary"])
    schedule_exact = load_json(inputs["schedule_exact_summary"])
    proxy = load_json(inputs["proxy_gap_summary"])
    codec = load_json(inputs["codec_benchmark"])
    hybrid = load_json(inputs["hybrid_holdout_summary"])

    compact_joint = compact["P1-R1"]["compact30"]["joint_improvement"]
    exact4_joint = exact4["primary_joint_K_and_V"]
    compact_holdout = comparison(
        hybrid["comparisons"], "compact30", "uniform_l5_c64", "K_and_V"
    )
    hybrid_primary = hybrid["primary"]
    uniform8_exact = comparison(
        schedule_exact["comparisons"], "s30_l10_uniform8", "compact30", "K_and_V"
    )
    uniform8_offline = schedule_search["global"]

    hypotheses = [
        {
            "id": "P1-R1", "type": "confirmatory", "status": "supported",
            "result": f"compact30 offline joint MSE +{100*compact_joint['mean_relative_mse_improvement']:.3f}% vs L5C64; N=20, 20/20 wins",
            "decision": "retain compact residual schedule",
        },
        {
            "id": "P1-C1", "type": "confirmatory", "status": "not_supported",
            "result": "no <=256 shared-codebook transform met the <=1.10 K and V distortion/local gate",
            "decision": "do not deploy canonical sharing",
        },
        {
            "id": "P1-F2", "type": "confirmatory", "status": "supported",
            "result": f"compact30 exact joint +{100*exact4_joint['mean_prompt_relative_improvement']:.3f}% vs L5C64; N=4, 4/4 wins",
            "decision": "reconstruction gain transfers weakly but positively to exact attention",
        },
        {
            "id": "P1-S2", "type": "confirmatory_reconstruction", "status": "supported",
            "result": f"10xC8 offline +{100*uniform8_offline['confirmation_mean_relative_improvement']:.3f}% vs compact30; N=10",
            "decision": "advance to functional replay only",
        },
        {
            "id": "P1-S3", "type": "confirmatory_functional", "status": "not_supported",
            "result": f"10xC8 exact joint {100*uniform8_exact['mean_prompt_relative_improvement']:.3f}% vs compact30; N=4, 0/4 wins",
            "decision": "reject 10xC8 despite lower reconstruction MSE",
        },
        {
            "id": "P1-D2", "type": "posthoc_explanatory", "status": "mechanism_supported",
            "result": f"V MSE {100*proxy['V_mechanism']['candidate_uniform_error_relative_change']:.3f}%, coherence +{100*proxy['V_mechanism']['candidate_cross_token_coherence_relative_change']:.3f}%, output energy +{100*proxy['V_mechanism']['candidate_attention_output_energy_relative_change']:.3f}%",
            "decision": "use attention-aware diagnostics, not MSE-only selection",
        },
        {
            "id": "P1-H1", "type": "confirmatory_holdout", "status": "supported",
            "result": f"frozen hybrid exact joint +{100*hybrid_primary['mean_prompt_relative_improvement']:.3f}% vs L5C64; N=4 new prompts, 4/4 wins",
            "decision": "use hybrid on captured representative layers",
        },
        {
            "id": "P1-CODEC", "type": "engineering_prototype", "status": "passed",
            "result": f"30-bit payload exact round trip; pack {codec['pack_ms']:.3f} ms, unpack {codec['unpack_ms']:.3f} ms",
            "decision": "int32 packing is viable; runtime integration remains",
        },
    ]
    p0.write_csv(root / "tables" / "hypothesis_outcomes.csv", hypotheses)

    key_results = [
        {"result": "compact30_offline_vs_L5", "mean_pct": 100*compact_joint["mean_relative_mse_improvement"], "ci_low_pct": 100*compact_joint["bootstrap_95_ci_low"], "ci_high_pct": 100*compact_joint["bootstrap_95_ci_high"], "prompts": 20},
        {"result": "compact30_exact_dev_vs_L5", "mean_pct": 100*exact4_joint["mean_prompt_relative_improvement"], "ci_low_pct": 100*exact4_joint["bootstrap_95_ci_low"], "ci_high_pct": 100*exact4_joint["bootstrap_95_ci_high"], "prompts": 4},
        {"result": "compact30_exact_holdout_vs_L5", "mean_pct": 100*compact_holdout["mean_prompt_relative_improvement"], "ci_low_pct": 100*compact_holdout["bootstrap_95_ci_low"], "ci_high_pct": 100*compact_holdout["bootstrap_95_ci_high"], "prompts": 4},
        {"result": "hybrid_exact_holdout_vs_L5", "mean_pct": 100*hybrid_primary["mean_prompt_relative_improvement"], "ci_low_pct": 100*hybrid_primary["bootstrap_95_ci_low"], "ci_high_pct": 100*hybrid_primary["bootstrap_95_ci_high"], "prompts": 4},
        {"result": "10xC8_offline_vs_compact30", "mean_pct": 100*uniform8_offline["confirmation_mean_relative_improvement"], "ci_low_pct": 100*uniform8_offline["confirmation_bootstrap_95_ci_low"], "ci_high_pct": 100*uniform8_offline["confirmation_bootstrap_95_ci_high"], "prompts": 10},
        {"result": "10xC8_exact_vs_compact30", "mean_pct": 100*uniform8_exact["mean_prompt_relative_improvement"], "ci_low_pct": 100*uniform8_exact["bootstrap_95_ci_low"], "ci_high_pct": 100*uniform8_exact["bootstrap_95_ci_high"], "prompts": 4},
    ]
    p0.write_csv(root / "tables" / "key_results.csv", key_results)

    failures = [
        {"event": "canonical_sharing_gate", "status": "scientific_negative", "cause": "shared representations remained far above local distortion", "resolution": "reject sharing at <=256 codebooks"},
        {"event": "10xC8_proxy_reversal", "status": "scientific_negative", "cause": "unweighted MSE missed q-projected K and coherent V error", "resolution": "reject schedule; add exact attention gate"},
        {"event": "schedule_search_prepare", "status": "software_fixed", "cause": "undefined resolve_path in optional registry path", "resolution": "use p0.resolve_path; regression test; failed root preserved"},
        {"event": "proxy_gap_v1", "status": "analysis_corrected", "cause": "absolute logit MSE was aggregated across layers with incompatible energy scales", "resolution": "v2 normalizes within head/capture; v1 preserved"},
        {"event": "hybrid_summary_v1", "status": "analysis_corrected", "cause": "comparison lookup key omitted baseline", "resolution": "v2 key includes candidate/baseline/intervention; KNOWN_ISSUE added"},
    ]
    p0.write_csv(root / "tables" / "failure_log.csv", failures)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    labels = [row["result"].replace("_", "\n") for row in key_results]
    save_figure(
        errorbar_bars(
            labels,
            [row["mean_pct"]/100 for row in key_results],
            [row["ci_low_pct"]/100 for row in key_results],
            [row["ci_high_pct"]/100 for row in key_results],
            ["#4C78A8", "#4C78A8", "#4C78A8", "#59A14F", "#F28E2B", "#E15759"],
            "Offline reconstruction and exact-attention effects are not interchangeable",
            "Relative improvement in error (%)",
        ),
        root, "01_offline_vs_exact",
    )

    layer = pd.read_csv(inputs["hybrid_layer_table"])
    layer = layer[layer["intervention"] == "K_and_V"].sort_values("layer")
    save_figure(
        errorbar_bars(
            [f"layer {int(value)}\n{method}" for value, method in zip(layer["layer"], layer["policy_method"])],
            layer["mean_prompt_relative_improvement"].tolist(),
            layer["bootstrap_95_ci_low"].tolist(), layer["bootstrap_95_ci_high"].tolist(),
            ["#BAB0AC" if method == "uniform_l5_c64" else "#59A14F" for method in layer["policy_method"]],
            "Holdout layer heterogeneity replicates the frozen policy",
            "Compact30 improvement vs L5C64 (%)",
        ),
        root, "02_holdout_layer_heterogeneity",
    )

    proxy_table = pd.read_csv(inputs["proxy_gap_table"]).set_index("metric")
    proxy_metrics = [
        ("uniform_K_error_per_dim", "K uniform MSE"),
        ("K_logit_error_relative_energy", "K q-projected error"),
        ("K_output_error_relative_energy", "K-only output error"),
        ("uniform_V_error_per_dim", "V uniform MSE"),
        ("V_cross_token_coherence_multiplier", "V error coherence"),
        ("V_output_error_relative_energy", "V-only output error"),
    ]
    rows = [proxy_table.loc[name] for name, _label in proxy_metrics]
    save_figure(
        errorbar_bars(
            [label for _name, label in proxy_metrics],
            [float(row["mean_candidate_relative_change"]) for row in rows],
            [float(row["bootstrap_95_ci_low"]) for row in rows],
            [float(row["bootstrap_95_ci_high"]) for row in rows],
            ["#4C78A8"]*3 + ["#F28E2B"]*3,
            "Why 10xC8 lowers MSE but worsens attention",
            "10xC8 relative change vs compact30 (%)",
        ),
        root, "03_attention_proxy_gap",
    )

    terminal = codec["terminal_accounting"]
    memory_labels = ["L5C64\ncurrent uint8", "compact30\ncurrent uint8", "compact30\nint32 packed"]
    memory_values = [
        terminal["uniform_l5_current_index_plus_codebook_mib"],
        terminal["compact30_current_uint8_index_mib"] + terminal["compact30_int4_codebook_mib"],
        terminal["compact30_int32_index_plus_codebook_mib"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(memory_labels, memory_values, color=["#4C78A8", "#E15759", "#59A14F"], edgecolor="#202020")
    ax.bar_label(bars, fmt="%.1f MiB", padding=3)
    ax.set_ylabel("Historical indices + resident K/V codebooks (MiB)")
    ax.set_title("Variable-bit packing is required for compact30's physical-memory win")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, root, "04_codec_memory_accounting")

    canonical_table = pd.read_csv(inputs["canonical_table"])
    canonical_256 = canonical_table[
        (canonical_table["total_codebook_count"] == 256)
        & canonical_table["transform"].isin(["raw", "center"])
    ].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    width = 0.35
    for offset, transform, color in [(-width/2, "raw", "#BAB0AC"), (width/2, "center", "#4C78A8")]:
        part = canonical_256.set_index(["transform", "K_or_V"])
        vals = [float(part.loc[(transform, kv), "mean_distortion_to_local_ratio"]) for kv in ("K", "V")]
        ax.bar(x + offset, vals, width, label=transform, color=color, edgecolor="#202020")
    ax.axhline(1.10, color="#E15759", linestyle="--", label="acceptance gate 1.10")
    ax.set_xticks(x, ["K", "V"])
    ax.set_ylabel("Shared / local distortion")
    ax.set_title("Centering helps, but canonical sharing remains outside the gate")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, root, "05_canonical_sharing_negative")

    summary = {
        "schema_version": "wan5b_p1_final_research_summary.v1",
        "completed": p0.utc_now(), "config_hash": cfg_hash,
        "cache_scope": {
            "axis": "T=44 video-latent time axis; not diffusion denoising timestep",
            "mechanism": "video self-attention attn1 custom ChunkedKVCache",
            "stored": "historical K and V; Q dump is diagnostic only",
            "hook": "K post-QK-norm pre-RoPE/PRoPE; V pre-PRoPE",
            "current_segment": "BF16 and bit-exactly untouched in exact replay",
        },
        "recommended_research_policy": {
            "captured_layers": {"0": "uniform_l5_c64", "10": "compact30", "20": "compact30", "29": "uniform_l5_c64"},
            "index_codec": "30-bit payload in uint32/int32 container",
            "codebooks": "local layer-head actual INT4 centroid payload + per-centroid FP16 scale",
            "status": "validated for four representative layers at one late-cache call; not yet a 30-layer runtime policy",
        },
        "primary_holdout": hybrid_primary,
        "compact_holdout": compact_holdout,
        "hypothesis_outcomes": hypotheses,
        "failures": failures,
        "hard_limits": [
            "No decoded-video quality metric was run in P1.",
            "Exact attention evidence covers eight prompts total, four representative layers, and one late-cache call per layer.",
            "The int32 codec is an eager-PyTorch prototype and is not integrated into the runtime cache/attention kernel.",
            "The four-layer hybrid policy cannot be extrapolated to all 30 layers without additional all-layer evidence.",
        ],
        "source_records": source_records,
    }
    p0.atomic_json(root / "summary.json", summary)

    decision = [
        "# P1 research decision", "",
        "## Decision", "",
        "For the four representative layers with exact holdout evidence, use local layer-head RVQ with a frozen layer-selective schedule: L5C64 at layers 0 and 29, compact30 `[64,32,16,8,8,8,8,8]` at layers 10 and 20. Store actual INT4 centroids with per-centroid FP16 scales and pack the 30 index bits into one int32 container.", "",
        f"On previously unused prompts 4-7, this policy lowers joint K+V exact attention-output relative L2 by **{100*hybrid_primary['mean_prompt_relative_improvement']:.3f}%** vs uniform L5C64 (95% prompt-bootstrap CI **[{100*hybrid_primary['bootstrap_95_ci_low']:.3f}%, {100*hybrid_primary['bootstrap_95_ci_high']:.3f}%]**, 4/4 wins).", "",
        "## Rejected paths", "",
        "- Do not use 10xC8 as the winner: it improves held-out reconstruction by 1.746% but worsens exact joint attention error by 3.853% (4/4 prompts lose).", "",
        "- Do not use <=256 canonical shared codebooks: no transform passes the 1.10 distortion/local gate for both K and V.", "",
        "## Required next gate", "",
        "Capture all 30 layers (or a preregistered denser layer grid) on new prompts, freeze a complete layer policy, integrate the int32 codec into the cache path, then run end-to-end decoded-video quality and latency. Until that is done, this is a research policy—not a production deployment claim.",
    ]
    write_text_atomic(root / "research_decision.md", "\n".join(decision) + "\n")

    report = [
        "# Wan2.2 5B historical KV-cache compression — P1 report", "",
        f"Config hash: `{cfg_hash}`  ",
        "Statistical unit for inferential claims: prompt.", "",
        "## Cache identity and axis", "",
        "These are true BF16 historical K/V from video self-attention `attn1` and the custom `ChunkedKVCache`. `T=44` is the video-latent time axis (11 source chunks), not the diffusion denoising-timestep axis. K is captured after QK norm and before RoPE/PRoPE; V is before PRoPE. Q exists only in diagnostics. Exact replay quantizes historical K/V while leaving the current segment bit-exact BF16.", "",
        "## Main outcomes", "",
    ]
    report.extend(markdown_table(hypotheses, [("id", "Hypothesis"), ("type", "Type"), ("status", "Verdict"), ("result", "Result"), ("decision", "Action")]))
    report.extend([
        "", "## Final holdout", "",
        f"The layer map was selected on exact prompts 0-3 and frozen before prompts 4-7 were generated. On prompts 4-7, the hybrid improves joint K+V exact attention error by **{100*hybrid_primary['mean_prompt_relative_improvement']:.3f}%** vs L5C64 (95% CI [{100*hybrid_primary['bootstrap_95_ci_low']:.3f}%, {100*hybrid_primary['bootstrap_95_ci_high']:.3f}%], 4/4 wins), and by **{100*comparison(hybrid['comparisons'], 'layer_hybrid', 'compact30', 'K_and_V')['mean_prompt_relative_improvement']:.3f}%** vs all-captured-layer compact30 (4/4 wins).", "",
        "Layerwise compact30-vs-L5 effects on holdout prompts replicate the development pattern: layer 0 −5.567%, layer 10 +3.291%, layer 20 +2.000%, layer 29 −1.548%. This supports layer heterogeneity within the captured scope.", "",
        "## Reconstruction proxy failure", "",
        "10xC8 is the clearest negative result. It won prompt-separated reconstruction confirmation, yet lost exact replay. The post-hoc v2 decomposition shows why: its K uniform MSE is slightly lower but reference-normalized q-projected K error is higher; its V uniform MSE is 1.305% lower but cross-token error coherence is 16.682% higher, producing 6.668% higher V-only output-error energy. MSE-only schedule selection is therefore unsafe.", "",
        "## Memory and codec", "",
        f"The codec packs `[6,5,4,3,3,3,3,3]` into a 32-bit container with exact round trip. On `[24,35200]`, pack/unpack are {codec['pack_ms']:.3f}/{codec['unpack_ms']:.3f} ms. Terminal historical index+codebook accounting is {terminal['uniform_l5_current_index_plus_codebook_mib']:.3f} MiB for current L5C64 uint8 storage versus {terminal['compact30_int32_index_plus_codebook_mib']:.3f} MiB for compact30 int32, a {100*terminal['compact30_int32_reduction_vs_uniform_l5_current']:.3f}% reduction. This excludes current/retained BF16, workspace, and container overhead beyond the index tensor.", "",
        "## Failure analysis and corrections", "",
    ])
    report.extend(markdown_table(failures, [("event", "Event"), ("status", "Status"), ("cause", "Cause"), ("resolution", "Resolution")]))
    report.extend([
        "", "## Hard limits", "",
        "- No decoded-video quality metrics were run; attention-output improvements are not a VBench/FVD claim.", "",
        "- Exact replay covers prompts 0-7, layers 0/10/20/29, and one late-cache call per layer.", "",
        "- The hybrid mapping covers only those four layers; no complete 30-layer runtime policy is claimed.", "",
        "- The codec benchmark excludes RVQ assignment, allocation cleanup, transfer, and attention; runtime integration is pending.", "",
        "## Artifacts", "",
        "- `summary.json`: machine-readable synthesis and source hashes", "",
        "- `research_decision.md`: concise decision and next gate", "",
        "- `tables/`: hypothesis, key-result, and failure tables", "",
        "- `plots/`: PNG/PDF figures", "",
        "- `run_info.json`: reproducibility metadata", "",
    ])
    write_text_atomic(root / "report.md", "\n".join(report) + "\n")

    artifact_files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "run_info.json"}
    )
    p0.atomic_json(
        root / "artifact_manifest.json",
        {
            "schema_version": "wan5b_p1_final_artifact_manifest.v1",
            "files": [
                {"path": str(path.relative_to(root)), "sha256": p0.sha256_file(path), "bytes": path.stat().st_size}
                for path in artifact_files
            ],
        },
    )
    compact_info = load_json(inputs["compact_summary"].parents[0] / "run_info.json")
    hybrid_info = load_json(inputs["hybrid_holdout_summary"].parents[0] / "run_info.json")
    prompt_list = list(dict.fromkeys(compact_info["prompt_list"] + hybrid_info["prompt_list"]))
    registries = [p0.resolve_path(value) for value in cfg["hypothesis_registries"]]
    p0.atomic_json(
        root / "run_info.json",
        {
            "git_commit": p0.git_commit(), "config_path": str(cfg_path), "config_hash": cfg_hash,
            "codebook_path": {
                "compact": str(inputs["compact_codebook_manifest"].resolve()),
                "exact_dev": str(inputs["exact_codebook_manifest"].resolve()),
                "hybrid_holdout": str(inputs["hybrid_exact_codebook_manifest"].resolve()),
            },
            "codebook_hash": {
                "compact": source_records["compact_codebook_manifest"]["sha256"],
                "exact_dev": source_records["exact_codebook_manifest"]["sha256"],
                "hybrid_holdout": source_records["hybrid_exact_codebook_manifest"]["sha256"],
            },
            "hostname": p0.socket.gethostname(), "gpu_type": p0.gpu_name(),
            "seed": int(cfg["seed"]), "prompt_list": prompt_list, "timestamp": p0.utc_now(),
            "model_id": cfg["model_id"], "source_records": source_records,
            "hypothesis_registries": [
                {"path": str(path.resolve()), "sha256": p0.sha256_file(path)} for path in registries
            ],
            "known_failed_artifacts": cfg["known_failed_artifacts"],
        },
    )
    print(json.dumps({"output": str(root), "primary": hybrid_primary, "artifacts": len(artifact_files)}, indent=2))


if __name__ == "__main__":
    main()
