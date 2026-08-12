#!/usr/bin/env python3
"""Publish protocol-safe views beside the Structured master table.

The original wide master is retained for provenance.  These addendum tables
prevent visually adjacent cells from suggesting that full720, strict-v4, and
960-unit critical-screen reconstruction errors are directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_truncatable_metric_study_20260723"
)
FORMAL_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "structured_residual_video_benchmark_v1_20260723"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "structured_metric_master_table_final_20260723"
)
VBench10 = (
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
)
INDEPENDENT7 = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)
FORMAL_METHODS = (
    "direct_quantvideo_int2",
    "direct_quantvideo_int4",
    "l8c64_rvq_residual_qv_int2",
    "l8c64_rvq_residual_qv_int4",
    "l8c64_rvq_residual_structured_gaussian_h1r1_b2",
    "l8c64_rvq_residual_structured_gaussian_h1r1_b4",
)
OLD_SCHEMES = (
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)
CRITICAL_METHODS = (
    "l8_plus_quantvideo_residual_int2",
    "structured_gaussian_h1r1_b2",
    "l8_plus_quantvideo_residual_int4",
    "structured_gaussian_h1r1_b4",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--formal-summary", type=Path, default=FORMAL_ROOT / "summary.json"
    )
    parser.add_argument(
        "--prior-master",
        type=Path,
        default=STUDY_ROOT / "existing_methods_master.csv",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=STUDY_ROOT / "diagnostic_comparison.csv",
    )
    args = parser.parse_args()
    for name in ("output_root", "formal_summary", "prior_master", "diagnostic"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def csv_bytes(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def formal_video_rows(summary_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    lookup = {row["setting"]: row for row in payload["metrics"]}
    if set(FORMAL_METHODS) - set(lookup):
        raise ValueError("Formal summary does not contain all six methods")
    fields = [
        "protocol_id",
        "method",
        "bits",
        "videos",
        "pixel_MSE",
        "PSNR",
        "SSIM",
        "LPIPS_normalized_v2",
        *VBench10,
        "vbench_mean_10d",
        "vbench_mean_7_independent_variable",
        "quant_time_s_per_video",
        "denoise_time_s_per_video",
        "joint_payload_MiB",
        "joint_compression_ratio_vs_BF16",
        "residual_payload_MiB",
        "residual_compression_ratio_vs_BF16",
        "payload_scope",
        "source",
    ]
    rows = []
    for method in FORMAL_METHODS:
        source = lookup[method]
        if int(source["videos"]) != 10:
            raise ValueError(f"{method}: expected ten formal videos")
        bits = 2 if method.endswith("2") else 4
        rows.append(
            {
                "protocol_id": (
                    "formal_closed_loop_10prompt_same_seed_frame0_excluded_v1"
                ),
                "method": method,
                "bits": bits,
                "videos": source["videos"],
                "pixel_MSE": source["mse"],
                "PSNR": source["psnr"],
                "SSIM": source["ssim"],
                "LPIPS_normalized_v2": source["lpips"],
                **{field: source[field] for field in VBench10},
                "vbench_mean_10d": sum(
                    float(source[field]) for field in VBench10
                )
                / len(VBench10),
                "vbench_mean_7_independent_variable": sum(
                    float(source[field]) for field in INDEPENDENT7
                )
                / len(INDEPENDENT7),
                "quant_time_s_per_video": source["quant_mean_s"],
                "denoise_time_s_per_video": source["denoise_mean_s"],
                "joint_payload_MiB": source["kv_cache_mem_mib"],
                "joint_compression_ratio_vs_BF16": source[
                    "end_to_end_compression_vs_bf16"
                ],
                "residual_payload_MiB": (
                    source["residual_payload_mib"]
                    if source["residual_payload_mib"] is not None
                    else "N/A"
                ),
                "residual_compression_ratio_vs_BF16": (
                    source["residual_compression_vs_bf16"]
                    if source["residual_compression_vs_bf16"] is not None
                    else "N/A"
                ),
                "payload_scope": source["payload_scope"],
                "source": str(summary_path),
            }
        )
    return rows, fields


def strict_v4_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    source_rows = read_csv(path)
    wanted = {
        f"{scheme}_b{bits}" for scheme in OLD_SCHEMES for bits in range(1, 5)
    }
    lookup = {row["method"]: row for row in source_rows}
    if wanted - set(lookup):
        raise ValueError(f"Missing strict-v4 rows: {sorted(wanted - set(lookup))}")
    copied_fields = [
        "method",
        "label",
        "method_family",
        "residual_codec",
        "prefix_bits",
        "offline_units",
        "offline_prompts",
        "video_prompts",
        "physical_residual_or_codec_bpp",
        "residual_or_codec_ratio_vs_BF16",
        "joint_ratio_vs_BF16_primary",
        "joint_payload_MiB_primary",
        "K_MSE",
        "V_MSE",
        "QK_logits_MSE",
        "QKV_output_MSE",
        "QKV_output_cosine",
        "exact_K_only_output_MSE_2half_mean",
        "exact_V_only_output_MSE_2half_mean",
        "exact_KV_output_MSE_2half_mean",
        "pixel_MSE",
        "PSNR",
        "SSIM",
        "LPIPS_normalized_v2",
        *VBench10,
        "vbench_mean_10d",
        "vbench_mean_7_independent_variable",
        "encode_original_s_per_video",
        "decode_original_s_per_video",
        "protocol_notes",
    ]
    fields = ["protocol_id", *copied_fields, "source"]
    rows = []
    for scheme in OLD_SCHEMES:
        for bits in range(1, 5):
            source = lookup[f"{scheme}_b{bits}"]
            if source["source_protocol_group"] != "strict-v4":
                raise ValueError(f"{source['method']}: not strict-v4")
            rows.append(
                {
                    "protocol_id": "strict_v4_7200_units_10prompt_v1",
                    **{field: source[field] for field in copied_fields},
                    "source": str(path),
                }
            )
    return rows, fields


def critical_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    source_rows = read_csv(path)
    lookup = {row["method"]: row for row in source_rows}
    if set(CRITICAL_METHODS) - set(lookup):
        raise ValueError("Missing critical comparison rows")
    copied_fields = [
        "method",
        "label",
        "method_family",
        "nominal_prefix_or_scalar_bits",
        "physical_residual_or_codec_bpp",
        "residual_or_codec_ratio_vs_BF16",
        "joint_ratio_vs_BF16_full_video_accounting",
        "critical_protocol",
        "critical_units",
        "critical_K_MSE",
        "critical_V_MSE",
        "critical_QK_MSE",
        "critical_QK_abs_p999",
        "critical_attention_KL",
        "critical_QKV_output_MSE",
        "critical_QKV_output_cosine",
        "critical_QKV_output_median_unit",
        "critical_QKV_output_p99_unit",
        "critical_QKV_output_max_unit",
        "critical_QKV_output_top1pct_error_mass_fraction",
        "critical_QKV_prompt_wins_of_10",
        "critical_QKV_unit_win_rate",
        "source_notes",
    ]
    fields = ["protocol_id", "comparison_role", *copied_fields, "source"]
    rows = []
    for method in CRITICAL_METHODS:
        source = lookup[method]
        rows.append(
            {
                "protocol_id": (
                    "structured_critical_screen_10prompt_4layer_24head_v1"
                ),
                "comparison_role": (
                    "candidate" if method.startswith("structured_") else "baseline"
                ),
                **{field: source[field] for field in copied_fields},
                "source": str(path),
            }
        )
    for bits in (2, 4):
        selected = [
            row
            for row in rows
            if int(row["nominal_prefix_or_scalar_bits"]) == bits
        ]
        if len(selected) != 2:
            raise ValueError(f"B{bits}: expected one baseline and candidate")
        rates = [
            float(row["physical_residual_or_codec_bpp"]) for row in selected
        ]
        if max(rates) - min(rates) > 1e-6:
            raise ValueError(f"B{bits}: critical comparison is not equal-rate")
    return rows, fields


def markdown(
    formal_rows: Sequence[Mapping[str, Any]],
    critical_rows_: Sequence[Mapping[str, Any]],
) -> str:
    formal_lookup = {row["method"]: row for row in formal_rows}
    critical_lookup = {row["method"]: row for row in critical_rows_}

    def improvement(candidate: float, baseline: float) -> float:
        return (baseline - candidate) / baseline * 100.0

    return "\n".join(
        [
            "# Protocol-safe views addendum",
            "",
            "原 `master_comparison.csv` 保留作为宽表证据索引，但其紧凑展示把 "
            "full720、strict-v4 与 960-unit critical-screen 的 K/V/QK/"
            "attention proxy 放在相邻同名列中。即使原 README 有警告，这种"
            "视觉并列仍可能被误读为直接横比；因此误差结论应优先使用本页的"
            "协议隔离视图。",
            "",
            "## A. Formal closed-loop video（六方法）",
            "",
            "文件：`formal_video_six_methods_protocol_safe.csv`。这里不放任何"
            " K/V/QK/attention proxy，只保留同一 10-prompt、same-seed、"
            "frame-0 excluded Similarity/VBench/runtime/rate。",
            "",
            "| 方法 | Pixel MSE | PSNR | SSIM | LPIPS-v2 | VBench-10 |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                "| {method} | {pixel:.6f} | {psnr:.4f} | {ssim:.6f} | "
                "{lpips:.6f} | {vbench:.6f} |".format(
                    method=row["method"],
                    pixel=float(row["pixel_MSE"]),
                    psnr=float(row["PSNR"]),
                    ssim=float(row["SSIM"]),
                    lpips=float(row["LPIPS_normalized_v2"]),
                    vbench=float(row["vbench_mean_10d"]),
                )
                for row in formal_rows
            ],
            "",
            "## B. strict-v4 四种旧 codec",
            "",
            "文件：`strict_v4_four_codecs_protocol_safe.csv`。仅这 16 行共享 "
            "`strict_v4_7200_units_10prompt_v1`，可在 B1–B4 内比较 proxy、"
            "exact replay、Similarity、VBench 与 runtime。",
            "",
            "## C. critical-screen Structured vs RVQ+QuantVideo",
            "",
            "文件：`critical_screen_structured_vs_rvq_qv_protocol_safe.csv`。"
            "四行共享同一 prompt/layer/head 采样与等 residual payload，是"
            " Structured B2/B4 proxy 改善的正确对照。",
            "",
            "| B | K MSE改善 | V MSE改善 | QK MSE改善 | Attn-out MSE改善 |",
            "|---:|---:|---:|---:|---:|",
            *[
                (
                    "| {bits} | {k:+.2f}% | {v:+.2f}% | {qk:+.2f}% | "
                    "{out:+.2f}% |"
                ).format(
                    bits=bits,
                    k=improvement(
                        float(
                            critical_lookup[
                                f"structured_gaussian_h1r1_b{bits}"
                            ]["critical_K_MSE"]
                        ),
                        float(
                            critical_lookup[
                                f"l8_plus_quantvideo_residual_int{bits}"
                            ]["critical_K_MSE"]
                        ),
                    ),
                    v=improvement(
                        float(
                            critical_lookup[
                                f"structured_gaussian_h1r1_b{bits}"
                            ]["critical_V_MSE"]
                        ),
                        float(
                            critical_lookup[
                                f"l8_plus_quantvideo_residual_int{bits}"
                            ]["critical_V_MSE"]
                        ),
                    ),
                    qk=improvement(
                        float(
                            critical_lookup[
                                f"structured_gaussian_h1r1_b{bits}"
                            ]["critical_QK_MSE"]
                        ),
                        float(
                            critical_lookup[
                                f"l8_plus_quantvideo_residual_int{bits}"
                            ]["critical_QK_MSE"]
                        ),
                    ),
                    out=improvement(
                        float(
                            critical_lookup[
                                f"structured_gaussian_h1r1_b{bits}"
                            ]["critical_QKV_output_MSE"]
                        ),
                        float(
                            critical_lookup[
                                f"l8_plus_quantvideo_residual_int{bits}"
                            ]["critical_QKV_output_MSE"]
                        ),
                    ),
                )
                for bits in (2, 4)
            ],
            "",
            "关键限制：C 的 proxy 改善不能替代 A 的 closed-loop 视频结果。"
            "例如 Structured B2 的 critical attention-output MSE 显著更低，"
            "但 formal Pixel MSE 并未优于等码率 L8+QV；这正是必须隔离协议"
            "展示的原因。",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    if not args.output_root.exists():
        raise FileNotFoundError(args.output_root)
    output_names = (
        "formal_video_six_methods_protocol_safe.csv",
        "strict_v4_four_codecs_protocol_safe.csv",
        "critical_screen_structured_vs_rvq_qv_protocol_safe.csv",
        "protocol_safe_views_zh.md",
        "protocol_safe_views_config.json",
        "protocol_safe_views_evidence_manifest.json",
        "protocol_safe_views_run_info.json",
    )
    collisions = [
        str(args.output_root / name)
        for name in output_names
        if (args.output_root / name).exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite: {collisions}")

    formal_rows_, formal_fields = formal_video_rows(args.formal_summary)
    strict_rows_, strict_fields = strict_v4_rows(args.prior_master)
    critical_rows_, critical_fields = critical_rows(args.diagnostic)
    source_paths = {
        "formal_summary": args.formal_summary,
        "formal_config": args.formal_summary.parent / "config.json",
        "formal_run_info": args.formal_summary.parent / "run_info.json",
        "prior_master": args.prior_master,
        "prior_run_info": args.prior_master.parent / "run_info.json",
        "critical_diagnostic": args.diagnostic,
        "critical_run_info": args.diagnostic.parent / "diagnostic_run_info.json",
        "builder_script": Path(__file__).resolve(),
        "original_master": args.output_root / "master_comparison.csv",
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing provenance: {missing}")
    config = {
        "schema_version": "structured_protocol_safe_views.config.v1",
        "operation": "read-only CPU aggregation; no metric recomputation",
        "output_root": str(args.output_root),
        "views": {
            "formal_video": {
                "protocol_id": (
                    "formal_closed_loop_10prompt_same_seed_frame0_excluded_v1"
                ),
                "rows": len(formal_rows_),
                "forbidden_fields": ["K_MSE", "V_MSE", "QK_MSE", "attention_MSE"],
            },
            "strict_v4": {
                "protocol_id": "strict_v4_7200_units_10prompt_v1",
                "rows": len(strict_rows_),
            },
            "critical_screen": {
                "protocol_id": (
                    "structured_critical_screen_10prompt_4layer_24head_v1"
                ),
                "rows": len(critical_rows_),
                "same_payload_checked": True,
            },
        },
    }
    config_payload = json_bytes(config)
    output_payloads = {
        "formal_video_six_methods_protocol_safe.csv": csv_bytes(
            formal_rows_, formal_fields
        ),
        "strict_v4_four_codecs_protocol_safe.csv": csv_bytes(
            strict_rows_, strict_fields
        ),
        "critical_screen_structured_vs_rvq_qv_protocol_safe.csv": csv_bytes(
            critical_rows_, critical_fields
        ),
        "protocol_safe_views_zh.md": (
            markdown(formal_rows_, critical_rows_) + "\n"
        ).encode("utf-8"),
        "protocol_safe_views_config.json": config_payload,
    }
    evidence = {
        "schema_version": "structured_protocol_safe_views.evidence.v1",
        "sources": [
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in source_paths.items()
        ],
    }
    output_payloads["protocol_safe_views_evidence_manifest.json"] = json_bytes(
        evidence
    )
    formal_run = json.loads(
        (args.formal_summary.parent / "run_info.json").read_text(
            encoding="utf-8"
        )
    )
    run_info = {
        "schema_version": "structured_protocol_safe_views.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(
            args.output_root / "protocol_safe_views_config.json"
        ),
        "config_hash": hashlib.sha256(config_payload).hexdigest(),
        "codebook_path": formal_run["codebook_path"],
        "codebook_hash": formal_run["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": formal_run["seed"],
        "prompt_list": formal_run["prompt_list"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": "read-only protocol-safe view aggregation",
        "output_hashes": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in output_payloads.items()
        },
    }
    output_payloads["protocol_safe_views_run_info.json"] = json_bytes(run_info)
    for name in output_names:
        create_only(args.output_root / name, output_payloads[name])
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "formal_rows": len(formal_rows_),
                "strict_v4_rows": len(strict_rows_),
                "critical_rows": len(critical_rows_),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
