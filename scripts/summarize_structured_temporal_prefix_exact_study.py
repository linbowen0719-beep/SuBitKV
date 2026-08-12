#!/usr/bin/env python3
"""Build the create-only Chinese summary for the temporal-prefix screens."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import screen_structured_temporal_prefix_exact_attention as screen


REPO = Path(__file__).resolve().parents[1]
DATA = screen.DEFAULT_DATA
DEFAULT_PARENT = DATA / "structured_temporal_prefix_exact_screen_v1_20260723"
DEFAULT_SPLIT = (
    DATA / "structured_temporal_kv_split_exact_followup_v1_20260723"
)
DEFAULT_OUTPUT = (
    REPO
    / "results"
    / "analysis"
    / "structured_temporal_prefix_exact_study_v1_20260723"
)

QV = "rvq_plus_quantvideo_spanwise_int2"
UNIFORM = "rvq_plus_structured_uniform_b2_spanwise"
JOINT = "rvq_plus_structured_temporal_b1_compatible"
STRICT = "rvq_plus_structured_temporal_strict_b0"
V_TEMPORAL = "rvq_plus_structured_K_uniform_V_temporal"
K_TEMPORAL = "rvq_plus_structured_K_temporal_V_uniform"
LABELS = {
    QV: "RVQ+QuantVideo INT2（span-wise）",
    UNIFORM: "Structured uniform K/V B2",
    JOINT: "Structured K/V [4,1,1,1,3]",
    STRICT: "Structured K/V [4,0,0,2,4]",
    V_TEMPORAL: "K uniform / V [4,1,1,1,3]",
    K_TEMPORAL: "K [4,1,1,1,3] / V uniform",
}
METRICS = (
    "historical_K_mse",
    "historical_V_mse",
    "attention_logit_mse",
    "attention_kl_reference_to_candidate",
    "attention_output_mse",
    "historical_signed_age_center_shift_tokens",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260723)
    args = parser.parse_args()
    for name in ("parent_root", "split_root", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return screen.exact.replay.csv_bytes(rows)


def write_bytes(path: Path, payload: bytes) -> None:
    screen.exact.replay.create_only_bytes(path, payload)


def write_json(path: Path, value: Any) -> None:
    screen.exact.replay.create_only_json(path, value)


def value(row: Mapping[str, Any], metric: str) -> float:
    result = float(row[metric])
    if metric == "historical_signed_age_center_shift_tokens":
        result = abs(result)
    return result


def method_rows(
    parent: Sequence[Mapping[str, Any]],
    split: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    result = {
        method: [
            row for row in parent if str(row["method"]) == method
        ]
        for method in (QV, UNIFORM, JOINT, STRICT)
    }
    result.update(
        {
            method: [
                row for row in split if str(row["method"]) == method
            ]
            for method in (V_TEMPORAL, K_TEMPORAL)
        }
    )
    for method, rows in result.items():
        if len(rows) != 10 * 4 * 24 * 2 * 3:
            raise RuntimeError(f"{method} has {len(rows)} raw rows")
    # Both repeated baselines must be numerically identical, even though their
    # config hashes differ.
    for method in (QV, UNIFORM):
        parent_values = sorted(
            (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                row["attention_half"],
                row["intervention"],
                *(float(row[metric]) for metric in METRICS),
            )
            for row in parent
            if row["method"] == method
        )
        split_values = sorted(
            (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                row["attention_half"],
                row["intervention"],
                *(float(row[metric]) for metric in METRICS),
            )
            for row in split
            if row["method"] == method
        )
        if parent_values != split_values:
            raise RuntimeError(f"Repeated baseline differs: {method}")
    return result


def tail(values: np.ndarray, q: float) -> float:
    return screen.empirical_cvar(values, q)


def build_method_summary(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for method, rows in rows_by_method.items():
        selected = [row for row in rows if row["intervention"] == "K_and_V"]
        item: dict[str, Any] = {
            "method": method,
            "label": LABELS[method],
            "prompts": 10,
            "layers": 4,
            "heads": 24,
            "attention_halves": 2,
            "raw_head_half_rows": len(selected),
        }
        for metric in METRICS:
            values = np.asarray(
                [value(row, metric) for row in selected], dtype=np.float64
            )
            name = (
                "absolute_age_shift"
                if metric == "historical_signed_age_center_shift_tokens"
                else metric
            )
            item[f"{name}_mean"] = float(values.mean())
            item[f"{name}_p99"] = float(np.quantile(values, 0.99))
            item[f"{name}_p999"] = float(np.quantile(values, 0.999))
            item[f"{name}_cvar99"] = tail(values, 0.99)
            item[f"{name}_cvar999"] = tail(values, 0.999)
        output.append(item)
    return output


def prompt_values(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    intervention: str,
) -> np.ndarray:
    output = []
    for prompt in range(10):
        selected = [
            row
            for row in rows
            if (
                int(row["prompt_id"]) == prompt
                and row["intervention"] == intervention
            )
        ]
        if len(selected) != 4 * 24 * 2:
            raise RuntimeError(
                f"prompt {prompt}/{intervention} has {len(selected)} rows"
            )
        output.append(np.mean([value(row, metric) for row in selected]))
    return np.asarray(output, dtype=np.float64)


def paired_bootstrap(
    *,
    candidate: np.ndarray,
    baseline: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    improvement = 100.0 * (
        float(baseline.mean()) - float(candidate.mean())
    ) / max(float(baseline.mean()), 1e-30)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(samples, len(candidate)))
    candidate_boot = candidate[indices].mean(axis=1)
    baseline_boot = baseline[indices].mean(axis=1)
    bootstrap = 100.0 * (
        baseline_boot - candidate_boot
    ) / np.maximum(baseline_boot, 1e-30)
    return (
        improvement,
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    )


def build_bootstrap(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    samples: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    output = []
    candidates = (JOINT, STRICT, V_TEMPORAL, K_TEMPORAL)
    for candidate_method in candidates:
        for baseline_method in (UNIFORM, QV):
            for intervention in ("K_only", "V_only", "K_and_V"):
                for metric in METRICS:
                    candidate = prompt_values(
                        rows_by_method[candidate_method],
                        metric,
                        intervention,
                    )
                    baseline = prompt_values(
                        rows_by_method[baseline_method],
                        metric,
                        intervention,
                    )
                    seed_payload = (
                        f"{base_seed}|{candidate_method}|{baseline_method}|"
                        f"{intervention}|{metric}"
                    ).encode()
                    seed = int(
                        hashlib.sha256(seed_payload).hexdigest()[:16], 16
                    ) % (2**32)
                    improvement, low, high = paired_bootstrap(
                        candidate=candidate,
                        baseline=baseline,
                        samples=samples,
                        seed=seed,
                    )
                    output.append(
                        {
                            "candidate": candidate_method,
                            "candidate_label": LABELS[candidate_method],
                            "baseline": baseline_method,
                            "baseline_label": LABELS[baseline_method],
                            "intervention": intervention,
                            "metric": metric,
                            "candidate_prompt_macro_mean": float(
                                candidate.mean()
                            ),
                            "baseline_prompt_macro_mean": float(
                                baseline.mean()
                            ),
                            "relative_improvement_pct": improvement,
                            "bootstrap_ci95_low_pct": low,
                            "bootstrap_ci95_high_pct": high,
                            "prompt_wins": int((candidate < baseline).sum()),
                            "prompt_ties": int((candidate == baseline).sum()),
                            "prompt_losses": int((candidate > baseline).sum()),
                            "n_prompts": len(candidate),
                            "bootstrap_samples": samples,
                            "bootstrap_seed": seed,
                            "statistical_unit": "prompt",
                            "adaptive_followup": candidate_method
                            in {V_TEMPORAL, K_TEMPORAL},
                        }
                    )
    return output


def build_age_summary(
    parent_root: Path,
    split_root: Path,
) -> list[dict[str, Any]]:
    sources = {
        JOINT: parent_root,
        STRICT: parent_root,
        V_TEMPORAL: split_root,
        K_TEMPORAL: split_root,
    }
    output = []
    for method, root in sources.items():
        rows = read_csv(
            root / "tables" / "age_bin_paired_comparisons.csv"
        )
        for span_index in range(5):
            selected = next(
                row
                for row in rows
                if (
                    row["candidate"] == method
                    and row["baseline"] == UNIFORM
                    and row["attention_half"] == "both"
                    and row["intervention"] == "K_and_V"
                    and row["metric"]
                    == "attention_output_contribution_mse"
                    and int(row["span_index"]) == span_index
                )
            )
            output.append(
                {
                    "method": method,
                    "label": LABELS[method],
                    "span_index": span_index,
                    "age_order": "0=oldest anchor; 4=newest terminal",
                    "mean_relative_improvement_pct": float(
                        selected["relative_improvement_pct"]
                    ),
                    "p99_ratio_vs_uniform": float(selected["p99_ratio"]),
                    "p999_ratio_vs_uniform": float(selected["p999_ratio"]),
                    "cvar99_ratio_vs_uniform": float(
                        selected["cvar99_ratio"]
                    ),
                    "cvar999_ratio_vs_uniform": float(
                        selected["cvar999_ratio"]
                    ),
                    "contribution_mse_is_additive": False,
                    "adaptive_followup": method
                    in {V_TEMPORAL, K_TEMPORAL},
                }
            )
    return output


def build_payload_runtime(
    rows_by_method: Mapping[str, Sequence[Mapping[str, Any]]],
    parent_root: Path,
    split_root: Path,
) -> list[dict[str, Any]]:
    payload_source = {}
    for root in (parent_root, split_root):
        for row in read_csv(root / "tables" / "full_model_payload_audit.csv"):
            payload_source.setdefault(row["method"], row)
    output = []
    bf16_mib = (
        30 * 2 * 1 * 24 * 35_200 * 128 * 2 / 2**20
    )
    for method, rows in rows_by_method.items():
        # One row per task avoids the 144-fold half/intervention/head repeat.
        selected = [
            row
            for row in rows
            if (
                row["attention_half"] == "standard_rope"
                and row["intervention"] == "K_and_V"
                and int(row["head"]) == 0
            )
        ]
        payload = payload_source[method]
        joint_mib = float(
            payload[
                "full_30_layer_joint_MiB_excluding_shared_codebooks"
            ]
        )
        output.append(
            {
                "method": method,
                "label": LABELS[method],
                "schedule_K": selected[0].get(
                    "temporal_schedule_K",
                    selected[0].get("temporal_schedule"),
                ),
                "schedule_V": selected[0].get(
                    "temporal_schedule_V",
                    selected[0].get("temporal_schedule"),
                ),
                "full_30_layer_residual_MiB": float(
                    payload["full_30_layer_residual_MiB"]
                ),
                "full_30_layer_joint_MiB_excluding_shared_codebooks": (
                    joint_mib
                ),
                "BF16_KV_baseline_MiB": bf16_mib,
                "joint_compression_ratio_vs_BF16_excluding_shared_codebooks": (
                    bf16_mib / joint_mib
                ),
                "exact_equal_uniform_B2": (
                    payload["exact_equal_uniform_B2"] == "True"
                ),
                "mean_spanwise_residual_encode_K_plus_V_seconds_per_layer": (
                    float(
                        np.mean(
                            [
                                float(row["residual_encode_K_seconds"])
                                + float(row["residual_encode_V_seconds"])
                                for row in selected
                            ]
                        )
                    )
                ),
                "mean_spanwise_residual_decode_K_plus_V_seconds_per_layer": (
                    float(
                        np.mean(
                            [
                                float(row["residual_decode_K_seconds"])
                                + float(row["residual_decode_V_seconds"])
                                for row in selected
                            ]
                        )
                    )
                ),
                "runtime_interval": (
                    "sum of five synchronized span calls for one layer, K+V; "
                    "structured encode is the shared B4 progressive encode"
                ),
                "shared_codebooks_included": False,
            }
        )
    return output


def find_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    baseline: str,
    metric: str,
    intervention: str = "K_and_V",
) -> Mapping[str, Any]:
    return next(
        row
        for row in rows
        if (
            row["candidate"] == candidate
            and row["baseline"] == baseline
            and row["metric"] == metric
            and row["intervention"] == intervention
        )
    )


def report_text(
    summary: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    age: Sequence[Mapping[str, Any]],
    payload: Sequence[Mapping[str, Any]],
) -> str:
    by_method = {row["method"]: row for row in summary}
    lines = [
        "# Wan2.2-5B：span-wise temporal prefix exact-attention 研究结果",
        "",
        "## 结论",
        "",
        "1. 原始确认性 screen 没有候选通过预注册 gate。K/V 同时使用 "
        "`[4,1,1,1,3]` 虽把 mean attention-output MSE 降低 "
        "`71.92%`，但 QK-logit P99/P999 变为 uniform B2 的 "
        "`3.72×/4.00×`；strict-B0 的 mean output 反而退化 `4.91%`，"
        "且出现严重 tail/CVaR。",
        "2. 查看上述 held-out 结果后做的 K/V-split 属于 adaptive "
        "exploration。保持 K uniform、只给 V 使用 `[4,1,1,1,3]` 时，"
        "K MSE、QK、KL、age shift 与 uniform 逐项完全相同，同时 mean "
        "output MSE 改善 `53.96%`，10/10 prompt 获胜，prompt bootstrap "
        "95% CI 为 `[48.78%,58.83%]`。",
        "3. 该 V-only temporal 候选仍未过严格 gate：output P999 是 "
        "`1.014×`（恶化 1.41%），尽管 P99=`0.325×`、CVaR99=`0.701×`、"
        "CVaR999=`0.998×`。因此它是很强的机制信号，不是可直接宣称的"
        "确认性胜利；必须换新 prompt/seed。",
        "",
        "## 主表（K_and_V，standard/camera 两分支合并）",
        "",
        "| 方法 | K MSE | V MSE | QK MSE | KL | output MSE | output P99 | output P999 | output CVaR99 | |age shift| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (QV, UNIFORM, JOINT, STRICT, V_TEMPORAL, K_TEMPORAL):
        row = by_method[method]
        lines.append(
            f"| {LABELS[method]} | "
            f"{row['historical_K_mse_mean']:.6g} | "
            f"{row['historical_V_mse_mean']:.6g} | "
            f"{row['attention_logit_mse_mean']:.6g} | "
            f"{row['attention_kl_reference_to_candidate_mean']:.6g} | "
            f"{row['attention_output_mse_mean']:.6g} | "
            f"{row['attention_output_mse_p99']:.6g} | "
            f"{row['attention_output_mse_p999']:.6g} | "
            f"{row['attention_output_mse_cvar99']:.6g} | "
            f"{row['absolute_age_shift_mean']:.6g} |"
        )
    v_uniform = find_bootstrap(
        bootstrap, V_TEMPORAL, UNIFORM, "attention_output_mse"
    )
    v_qv = find_bootstrap(
        bootstrap, V_TEMPORAL, QV, "attention_output_mse"
    )
    k_uniform = find_bootstrap(
        bootstrap, K_TEMPORAL, UNIFORM, "attention_output_mse"
    )
    lines += [
        "",
        "## Prompt-paired bootstrap",
        "",
        "| candidate vs baseline | output改善 | 95% CI | wins/ties/losses |",
        "|---|---:|---:|---:|",
        (
            f"| K uniform / V temporal vs uniform | "
            f"{v_uniform['relative_improvement_pct']:.3f}% | "
            f"[{v_uniform['bootstrap_ci95_low_pct']:.3f}%,"
            f"{v_uniform['bootstrap_ci95_high_pct']:.3f}%] | "
            f"{v_uniform['prompt_wins']}/{v_uniform['prompt_ties']}/"
            f"{v_uniform['prompt_losses']} |"
        ),
        (
            f"| K uniform / V temporal vs QV2 | "
            f"{v_qv['relative_improvement_pct']:.3f}% | "
            f"[{v_qv['bootstrap_ci95_low_pct']:.3f}%,"
            f"{v_qv['bootstrap_ci95_high_pct']:.3f}%] | "
            f"{v_qv['prompt_wins']}/{v_qv['prompt_ties']}/"
            f"{v_qv['prompt_losses']} |"
        ),
        (
            f"| K temporal / V uniform vs uniform | "
            f"{k_uniform['relative_improvement_pct']:.3f}% | "
            f"[{k_uniform['bootstrap_ci95_low_pct']:.3f}%,"
            f"{k_uniform['bootstrap_ci95_high_pct']:.3f}%] | "
            f"{k_uniform['prompt_wins']}/{k_uniform['prompt_ties']}/"
            f"{k_uniform['prompt_losses']} |"
        ),
        "",
        "Bootstrap 单位是 prompt；每个 prompt 内先平均 4 layers×24 heads×"
        "2 attention halves。完整 K-only/V-only/KV、K/V/QK/KL/age 表见 "
        "`paired_prompt_bootstrap.csv`。",
        "",
        "## K/V 机制分解",
        "",
    ]
    for method in (V_TEMPORAL, K_TEMPORAL):
        parts = []
        for intervention in ("K_only", "V_only", "K_and_V"):
            row = find_bootstrap(
                bootstrap,
                method,
                UNIFORM,
                "attention_output_mse",
                intervention,
            )
            parts.append(
                f"{intervention}={row['relative_improvement_pct']:.2f}%"
            )
        lines.append(f"- {LABELS[method]}：{', '.join(parts)}。")
    lines += [
        "",
        "K-uniform/V-temporal 的 K-only 为严格 0% 差异，而 V-only 改善 "
        "`70.05%`；反向候选的 V-only 为严格 0%，K-only 改善 `79.28%`，"
        "但其 QK tail 同时被破坏。这说明主要可部署方向应是 V temporal，"
        "而不是继续把 K/V 绑在同一 schedule。",
        "",
        "## Age-bin 解释",
        "",
        "span 0 是最老 anchor，span 4 是最新 terminal span。"
        "`output contribution MSE` 是每个 span 的加性输出向量误差；"
        "其 MSE 本身不能跨 span 相加，因为存在交叉项。",
        "",
        "| candidate | span | mean改善 | P999 ratio | CVaR99 ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in (V_TEMPORAL, K_TEMPORAL):
        for row in age:
            if row["method"] == method:
                lines.append(
                    f"| {LABELS[method]} | {row['span_index']} | "
                    f"{row['mean_relative_improvement_pct']:.2f}% | "
                    f"{row['p999_ratio_vs_uniform']:.3f}× | "
                    f"{row['cvar99_ratio_vs_uniform']:.3f}× |"
                )
    payload_row = next(row for row in payload if row["method"] == V_TEMPORAL)
    lines += [
        "",
        "V temporal 把误差从高-utility 的 anchor/newest span 移向 middle："
        "span 0/4 mean 改善约 50.13%/54.60%，但 span 1–3 contribution "
        "mean 明显恶化。全局 attention 权重使总 output 仍改善；这也解释了"
        "为什么全局 V reconstruction MSE 会变差而 attention output 更好。",
        "",
        "## Payload 与 runtime 口径",
        "",
        f"所有方法在完整 30 层、K/V terminal cache 上 residual payload 都是 "
        f"`{payload_row['full_30_layer_residual_MiB']:.6f} MiB`；加 L8C64 "
        f"indices、但不含 shared codebooks 时为 "
        f"`{payload_row['full_30_layer_joint_MiB_excluding_shared_codebooks']:.6f} MiB`，"
        f"相对同形状 BF16 为 "
        f"`{payload_row['joint_compression_ratio_vs_BF16_excluding_shared_codebooks']:.6f}×`。"
        "所有候选逐字节等于 uniform B2。",
        "",
        "runtime 是一个 layer 的五个 7040-token span、K+V 同步调用总和；"
        "structured encode 计共享的 B4 progressive encode。它是当前 eager "
        "实现/并发机器状态的测量，不应外推为 fused kernel 吞吐。",
        "",
        "## 证据边界",
        "",
        "- exact replay 使用捕获的完整 35200-token key/value 轴并分成 "
        "5×7040 span；它是 terminal post-commit counterfactual。原 capture "
        "在线 attention 时最后 3520 token 尚属 current，因此不能把本结果"
        "写成在线因果路径已完全一致。",
        "- 第一轮 schedule 在看 test 结果前冻结，是确认性 screen；K/V split "
        "是在看过同一 10-test aggregate 后提出，只能算 adaptive mechanism "
        "check。没有独立未见 exact captures，因此下一步必须换 prompt 或"
        "生成 seed，再决定是否跑 closed-loop Similarity/VBench。",
        "- 任何 mean 优势都不能覆盖 P999/CVaR 风险；当前严格 gate 结果仍是 "
        "`any_pass=false`。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"create-only output exists: {args.output}")
    args.output.mkdir(parents=True)
    source_files = [
        args.parent_root / "config.json",
        args.parent_root / "tables" / "exact_attention_metrics.csv",
        args.parent_root / "tables" / "age_bin_paired_comparisons.csv",
        args.parent_root / "tables" / "full_model_payload_audit.csv",
        args.parent_root / "tables" / "promotion_gate.json",
        args.split_root / "config.json",
        args.split_root / "tables" / "exact_attention_metrics.csv",
        args.split_root / "tables" / "age_bin_paired_comparisons.csv",
        args.split_root / "tables" / "full_model_payload_audit.csv",
        args.split_root / "tables" / "promotion_gate.json",
    ]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    parent = read_csv(source_files[1])
    split = read_csv(source_files[6])
    rows_by_method = method_rows(parent, split)
    summary = build_method_summary(rows_by_method)
    bootstrap = build_bootstrap(
        rows_by_method,
        samples=args.bootstrap_samples,
        base_seed=args.bootstrap_seed,
    )
    age = build_age_summary(args.parent_root, args.split_root)
    payload = build_payload_runtime(
        rows_by_method, args.parent_root, args.split_root
    )
    config = {
        "schema_version": "structured_temporal_prefix_exact_study.config.v1",
        "parent_root": str(args.parent_root),
        "split_root": str(args.split_root),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "statistical_unit": "prompt",
        "method_labels": LABELS,
        "adaptive_methods": [V_TEMPORAL, K_TEMPORAL],
        "confirmatory_claim_allowed_for_adaptive_methods": False,
    }
    outputs = {
        "config": args.output / "config.json",
        "summary": args.output / "method_tail_summary.csv",
        "bootstrap": args.output / "paired_prompt_bootstrap.csv",
        "age": args.output / "age_bin_summary.csv",
        "payload": args.output / "payload_runtime.csv",
        "report": args.output / "README.md",
    }
    write_json(outputs["config"], config)
    write_bytes(outputs["summary"], csv_bytes(summary))
    write_bytes(outputs["bootstrap"], csv_bytes(bootstrap))
    write_bytes(outputs["age"], csv_bytes(age))
    write_bytes(outputs["payload"], csv_bytes(payload))
    write_bytes(
        outputs["report"],
        report_text(summary, bootstrap, age, payload).encode("utf-8"),
    )
    manifest = {
        "schema_version": "structured_temporal_prefix_exact_study.evidence.v1",
        "source_code_sha256": {
            str(REPO / "scripts/screen_structured_temporal_prefix_exact_attention.py"): sha256(
                REPO
                / "scripts/screen_structured_temporal_prefix_exact_attention.py"
            ),
            str(REPO / "scripts/screen_structured_temporal_kv_split_exact_followup.py"): sha256(
                REPO
                / "scripts/screen_structured_temporal_kv_split_exact_followup.py"
            ),
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(REPO / "tests/test_structured_temporal_prefix_exact_screen.py"): sha256(
                REPO
                / "tests/test_structured_temporal_prefix_exact_screen.py"
            ),
        },
        "source_sha256": {str(path): sha256(path) for path in source_files},
        "output_sha256": {
            str(path): sha256(path) for path in outputs.values()
        },
        "metric_accounting_changed": False,
        "new_GPU_evaluation": False,
        "bootstrap_is_post_aggregate_uncertainty_analysis": True,
        "README_status_updates": {
            str(args.parent_root / "status_update.json"): sha256(
                args.parent_root / "status_update.json"
            ),
            str(args.split_root / "status_update.json"): sha256(
                args.split_root / "status_update.json"
            ),
        },
    }
    manifest_path = args.output / "evidence_manifest.json"
    write_json(manifest_path, manifest)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    prompt_list = json.loads(
        (args.parent_root / "run_info" / "aggregate.json").read_text(
            encoding="utf-8"
        )
    )["prompt_list"]
    run_info = {
        "schema_version": "structured_temporal_prefix_exact_study.run_info.v1",
        "git_commit": commit,
        "config_path": str(outputs["config"]),
        "config_hash": sha256(outputs["config"]),
        "codebook_path": {
            "K": str(DATA / "nonorm_k_l8c64_codebooks"),
            "V": str(DATA / "nonorm_v_l8c64_codebooks"),
            "structured_params": str(screen.DEFAULT_PARAMS),
        },
        "codebook_hash": {
            **screen.exact.replay.KNOWN_CODEBOOK_HASHES,
            "structured_params": sha256(screen.DEFAULT_PARAMS),
        },
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": args.bootstrap_seed,
        "prompt_list": prompt_list,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution": "CPU-only deterministic aggregation and prompt bootstrap",
        "output_sha256": {
            **{str(path): sha256(path) for path in outputs.values()},
            str(manifest_path): sha256(manifest_path),
        },
    }
    write_json(args.output / "run_info.json", run_info)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "methods": len(summary),
                "bootstrap_rows": len(bootstrap),
                "age_rows": len(age),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
