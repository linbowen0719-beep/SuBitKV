#!/usr/bin/env python3
"""Create-only summary for the exact tensor-selective residual hybrid screen."""

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
from typing import Any, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "rvq_plus_quantvideo",
    "rvq_plus_structured_gaussian_h1r1",
    "rvq_plus_structured_k_quantvideo_v",
    "rvq_plus_quantvideo_k_structured_v",
)
PRIMARY = "rvq_plus_structured_k_quantvideo_v"
REVERSE = "rvq_plus_quantvideo_k_structured_v"
QV = "rvq_plus_quantvideo"
STRUCTURED = "rvq_plus_structured_gaussian_h1r1"
METRICS = {
    "K_MSE": "historical_K_mse",
    "V_MSE": "historical_V_mse",
    "QK_logit_MSE": "attention_logit_mse",
    "attention_KL": "attention_kl_reference_to_candidate",
    "attention_output_MSE": "attention_output_mse",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    fields = list(rows[0])
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected_rows(
    rows: Iterable[dict[str, str]],
    method: str,
    metric_name: str,
) -> list[dict[str, str]]:
    result = []
    for row in rows:
        if row["method"] != method or int(row["bits"]) != 2:
            continue
        if row["intervention"] != "K_and_V":
            continue
        if metric_name in {"K_MSE", "V_MSE"} and row["attention_half"] != "standard_rope":
            continue
        result.append(row)
    return result


def cvar(values: np.ndarray, probability: float) -> tuple[float, float]:
    threshold = float(np.quantile(values, probability))
    tail = values[values >= threshold]
    return threshold, float(tail.mean())


def prompt_means(rows: list[dict[str, str]], field: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prompt_id"])].append(float(row[field]))
    return {key: float(np.mean(value)) for key, value in grouped.items()}


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    args.experiment_root = args.experiment_root.resolve()
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.experiment_root / "analysis"
    )
    return args


def main() -> None:
    args = parse_args()
    raw_path = args.experiment_root / "tables" / "exact_attention_metrics.csv"
    config_path = args.experiment_root / "config.json"
    prereg_path = args.experiment_root / "preregistration.json"
    for path in (raw_path, config_path, prereg_path):
        if not path.exists():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    canonical_config = (
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    config_hash = hashlib.sha256(canonical_config).hexdigest()
    if prereg["config_hash"] != config_hash:
        raise RuntimeError("Preregistration/config hash mismatch")
    rows = read_csv(raw_path)
    observed_methods = {row["method"] for row in rows}
    missing = set(METHODS) - observed_methods
    if missing:
        raise RuntimeError(f"Missing methods: {sorted(missing)}")
    expected_raw_rows = 10 * 4 * 5 * 2 * 3 * 24
    if len(rows) != expected_raw_rows:
        raise RuntimeError(
            f"Raw row coverage mismatch: {len(rows)} != {expected_raw_rows}"
        )

    summary_rows: list[dict[str, Any]] = []
    value_lookup: dict[tuple[str, str], float] = {}
    statistics_lookup: dict[tuple[str, str], dict[str, float]] = {}
    prompt_lookup: dict[tuple[str, str], dict[int, float]] = {}
    for method in METHODS:
        for metric_name, field in METRICS.items():
            subset = selected_rows(rows, method, metric_name)
            values = np.asarray([float(row[field]) for row in subset], dtype=np.float64)
            p99, cvar99 = cvar(values, 0.99)
            mean = float(values.mean())
            value_lookup[(method, metric_name)] = mean
            statistics_lookup[(method, metric_name)] = {
                "mean": mean,
                "p99": p99,
                "cvar99": cvar99,
            }
            prompt_lookup[(method, metric_name)] = prompt_means(subset, field)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric_name,
                    "mean": mean,
                    "p99": p99,
                    "cvar99": cvar99,
                    "units": len(values),
                    "unit_axis": (
                        "prompt*layer*head"
                        if metric_name in {"K_MSE", "V_MSE"}
                        else "prompt*layer*head*attention_half"
                    ),
                }
            )

    comparison_rows: list[dict[str, Any]] = []
    for candidate in (PRIMARY, REVERSE):
        for baseline in (QV, STRUCTURED):
            for metric_name in METRICS:
                base = value_lookup[(baseline, metric_name)]
                cand = value_lookup[(candidate, metric_name)]
                base_statistics = statistics_lookup[(baseline, metric_name)]
                candidate_statistics = statistics_lookup[(candidate, metric_name)]
                base_prompts = prompt_lookup[(baseline, metric_name)]
                cand_prompts = prompt_lookup[(candidate, metric_name)]
                common = sorted(set(base_prompts) & set(cand_prompts))
                wins = sum(cand_prompts[prompt] < base_prompts[prompt] for prompt in common)
                ties = sum(cand_prompts[prompt] == base_prompts[prompt] for prompt in common)
                comparison_rows.append(
                    {
                        "candidate": candidate,
                        "baseline": baseline,
                        "metric": metric_name,
                        "baseline_mean": base,
                        "candidate_mean": cand,
                        "improvement_percent": improvement(base, cand),
                        "baseline_p99": base_statistics["p99"],
                        "candidate_p99": candidate_statistics["p99"],
                        "p99_improvement_percent": improvement(
                            base_statistics["p99"],
                            candidate_statistics["p99"],
                        ),
                        "baseline_cvar99": base_statistics["cvar99"],
                        "candidate_cvar99": candidate_statistics["cvar99"],
                        "cvar99_improvement_percent": improvement(
                            base_statistics["cvar99"],
                            candidate_statistics["cvar99"],
                        ),
                        "prompt_wins": wins,
                        "prompt_ties": ties,
                        "prompts": len(common),
                    }
                )

    payload_by_method: dict[str, list[float]] = defaultdict(list)
    payload_by_identity: dict[tuple[int, int, int, str], float] = {}
    for method in METHODS:
        subset = selected_rows(rows, method, "K_MSE")
        for row in subset:
            payload = float(row["residual_payload_K_MiB"]) + float(
                row["residual_payload_V_MiB"]
            )
            identity = (
                int(row["prompt_id"]),
                int(row["layer"]),
                int(row["head"]),
                method,
            )
            payload_by_identity[identity] = payload
            payload_by_method[method].append(payload)
    payload_summary = {
        method: {
            "mean_residual_KV_MiB": float(np.mean(values)),
            "min_residual_KV_MiB": float(np.min(values)),
            "max_residual_KV_MiB": float(np.max(values)),
        }
        for method, values in payload_by_method.items()
    }
    runtime_rows: list[dict[str, Any]] = []
    for method in METHODS:
        subset = selected_rows(rows, method, "K_MSE")
        runtime_rows.append(
            {
                "method": method,
                "residual_encode_K_seconds_mean": float(
                    np.mean([float(row["residual_encode_K_seconds"]) for row in subset])
                ),
                "residual_encode_V_seconds_mean": float(
                    np.mean([float(row["residual_encode_V_seconds"]) for row in subset])
                ),
                "residual_encode_KV_seconds_mean": float(
                    np.mean(
                        [
                            float(row["residual_encode_K_seconds"])
                            + float(row["residual_encode_V_seconds"])
                            for row in subset
                        ]
                    )
                ),
                "residual_decode_K_seconds_mean": float(
                    np.mean([float(row["residual_decode_K_seconds"]) for row in subset])
                ),
                "residual_decode_V_seconds_mean": float(
                    np.mean([float(row["residual_decode_V_seconds"]) for row in subset])
                ),
                "residual_decode_KV_seconds_mean": float(
                    np.mean(
                        [
                            float(row["residual_decode_K_seconds"])
                            + float(row["residual_decode_V_seconds"])
                            for row in subset
                        ]
                    )
                ),
                "total_payload_KV_MiB_mean": float(
                    np.mean(
                        [
                            float(row["total_payload_K_MiB"])
                            + float(row["total_payload_V_MiB"])
                            for row in subset
                        ]
                    )
                ),
                "joint_compression_ratio_vs_BF16_mean": float(
                    np.mean(
                        [
                            (
                                float(row["source_BF16_K_MiB"])
                                + float(row["source_BF16_V_MiB"])
                            )
                            / (
                                float(row["total_payload_K_MiB"])
                                + float(row["total_payload_V_MiB"])
                            )
                            for row in subset
                        ]
                    )
                ),
            }
        )
    max_payload_delta = 0.0
    for prompt in range(10):
        for layer in (0, 10, 20, 29):
            for head in range(24):
                values = [
                    payload_by_identity[(prompt, layer, head, method)]
                    for method in METHODS
                ]
                max_payload_delta = max(max_payload_delta, max(values) - min(values))
    payload_equal = max_payload_delta == 0.0
    if not payload_equal:
        raise RuntimeError(f"B2 payload mismatch: {max_payload_delta} MiB")

    gate_row = next(
        row
        for row in comparison_rows
        if row["candidate"] == PRIMARY
        and row["baseline"] == QV
        and row["metric"] == "attention_output_MSE"
    )
    gate_pass = float(gate_row["improvement_percent"]) >= 10.0
    structured_gate_row = next(
        row
        for row in comparison_rows
        if row["candidate"] == PRIMARY
        and row["baseline"] == STRUCTURED
        and row["metric"] == "attention_output_MSE"
    )
    validations = []
    for path in sorted(args.experiment_root.rglob("validation.json")):
        validation = json.loads(path.read_text(encoding="utf-8"))
        current_errors = [
            float(value)
            for value in validation["current_segment_max_abs"].values()
        ]
        validations.append(
            {
                "path": str(path),
                "metric_audit_passed": bool(validation["metric_audit_passed"]),
                "coverage_complete": bool(validation["coverage"]["complete"]),
                "current_segment_max_abs": max(current_errors),
            }
        )
    validation_pass = (
        len(validations) == 40
        and all(
            row["metric_audit_passed"]
            and row["coverage_complete"]
            and row["current_segment_max_abs"] == 0.0
            for row in validations
        )
    )
    if not validation_pass:
        raise RuntimeError("Per-unit exact-attention validation failed")

    raw_lookup = {
        (
            row["method"],
            row["prompt_id"],
            row["layer"],
            row["attention_half"],
            row["intervention"],
            row["head"],
        ): row
        for row in rows
    }
    identity_checks = (
        (
            PRIMARY,
            STRUCTURED,
            ("historical_K_mse", "attention_logit_mse", "attention_kl_reference_to_candidate"),
        ),
        (PRIMARY, QV, ("historical_V_mse",)),
        (
            REVERSE,
            QV,
            ("historical_K_mse", "attention_logit_mse", "attention_kl_reference_to_candidate"),
        ),
        (REVERSE, STRUCTURED, ("historical_V_mse",)),
    )
    identity_audit = []
    for candidate, reference, fields in identity_checks:
        max_abs = 0.0
        for key, row in raw_lookup.items():
            if key[0] != candidate:
                continue
            other = raw_lookup[(reference, *key[1:])]
            for field in fields:
                max_abs = max(
                    max_abs,
                    abs(float(row[field]) - float(other[field])),
                )
        identity_audit.append(
            {
                "candidate": candidate,
                "reference": reference,
                "fields": list(fields),
                "max_abs": max_abs,
                "passed": max_abs == 0.0,
            }
        )
    if not all(row["passed"] for row in identity_audit):
        raise RuntimeError("Tensor hybrid source-identity audit failed")
    result = {
        "schema_version": "tensor_selective_residual_hybrid.summary.v1",
        "config_hash": config_hash,
        "preregistration_hash": sha256(prereg_path),
        "raw_metrics_hash": sha256(raw_path),
        "formal_units": {
            "prompts": 10,
            "layers": 4,
            "heads": 24,
            "attention_halves": 2,
        },
        "payload": {
            "baseline": "BF16 residual payload; MiB=2^20 bytes",
            "per_method": payload_summary,
            "max_paired_delta_MiB": max_payload_delta,
            "exact_equal": payload_equal,
        },
        "primary_gate": {
            "metric": "joint K_and_V attention_output_MSE mean",
            "baseline": QV,
            "threshold_improvement_percent": 10.0,
            "observed_improvement_percent": gate_row["improvement_percent"],
            "passed": gate_pass,
            "video_or_vbench_authorized": False,
        },
        "comparison_to_uniform_structured_B2": {
            "attention_output_MSE_improvement_percent": (
                structured_gate_row["improvement_percent"]
            ),
            "outperforms": (
                float(structured_gate_row["improvement_percent"]) > 0.0
            ),
        },
        "reverse_ablation_non_promotable": True,
    }
    self_audit = {
        "schema_version": "tensor_selective_residual_hybrid.self_audit.v1",
        "raw_rows": len(rows),
        "expected_raw_rows": expected_raw_rows,
        "unit_validations": validations,
        "all_unit_validations_passed": validation_pass,
        "tensor_source_identity": identity_audit,
        "payload_exact_equal": payload_equal,
        "payload_max_paired_delta_MiB": max_payload_delta,
        "config_hash_matches_preregistration": True,
        "passed": True,
    }

    def display(method: str) -> str:
        return {
            QV: "RVQ+QV2 (K/V)",
            STRUCTURED: "RVQ+Structured B2 (K/V)",
            PRIMARY: "Hybrid: K=Structured, V=QV2",
            REVERSE: "Reverse: K=QV2, V=Structured",
        }[method]

    table_lines = [
        "| 方法 | K MSE | V MSE | QK MSE | KL | Output MSE | residual K+V MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        table_lines.append(
            "| "
            + " | ".join(
                [
                    display(method),
                    f"{value_lookup[(method, 'K_MSE')]:.8g}",
                    f"{value_lookup[(method, 'V_MSE')]:.8g}",
                    f"{value_lookup[(method, 'QK_logit_MSE')]:.8g}",
                    f"{value_lookup[(method, 'attention_KL')]:.8g}",
                    f"{value_lookup[(method, 'attention_output_MSE')]:.8g}",
                    f"{payload_summary[method]['mean_residual_KV_MiB']:.6f}",
                ]
            )
            + " |"
        )
    primary_vs_qv = {
        row["metric"]: row for row in comparison_rows
        if row["candidate"] == PRIMARY and row["baseline"] == QV
    }
    reverse_vs_qv = {
        row["metric"]: row for row in comparison_rows
        if row["candidate"] == REVERSE and row["baseline"] == QV
    }
    runtime_lines = [
        "| 方法 | encode K/V s | decode K/V s | total payload MiB | joint × vs BF16 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in runtime_rows:
        runtime_lines.append(
            f"| {display(str(row['method']))} | "
            f"{row['residual_encode_KV_seconds_mean']:.6f} | "
            f"{row['residual_decode_KV_seconds_mean']:.6f} | "
            f"{row['total_payload_KV_MiB_mean']:.6f} | "
            f"{row['joint_compression_ratio_vs_BF16_mean']:.6f} |"
        )
    readme = f"""# K/V tensor-selective residual hybrid exact-attention screen

结论：预注册主候选 `K=Structured B2, V=QuantVideo INT2` 的联合
attention-output MSE 相对 `K/V 都用 QuantVideo INT2` 改善
`{primary_vs_qv['attention_output_MSE']['improvement_percent']:.3f}%`，
{'达到' if gate_pass else '未达到'}预注册的 `10%` 晋级线。因此本实验
**不运行视频或 VBench**。但它相对更强的 uniform Structured B2
退化 `{-float(structured_gate_row['improvement_percent']):.3f}%`，所以不能
宣称超过当前最佳等载荷 codec。反向组合只作为机制消融，不具备晋级资格。

## Formal mean

{chr(10).join(table_lines)}

所有 B2 方法的 paired residual payload 完全一致；最大差为
`{max_payload_delta:.9f} MiB`。基线是 BF16 residual，MiB 按 `2^20`
bytes。RVQ base 在所有方法间完全相同，未计入上表 residual payload。

## Runtime 与联合载荷

{chr(10).join(runtime_lines)}

时间是 exact screen 中对一次完整 K 或 V tensor 的同步 residual codec
encode/decode 区间，不包含 RVQ base、attention forward 或 I/O。Structured
encode 是 B4 一次编码再物理截断到 B2；因此它与直接 QV2 的 encode 区间
语义不同，不能当作生产吞吐。

## 相对 QV2 的主候选变化

| 指标 | mean 改善 | prompt wins |
|---|---:|---:|
""" + "\n".join(
        f"| {metric} | {row['improvement_percent']:.3f}% | "
        f"{row['prompt_wins']}/{row['prompts']} (ties {row['prompt_ties']}) |"
        for metric, row in primary_vs_qv.items()
    ) + f"""

## 相对 uniform Structured B2

Primary 的 K/QK/KL 与 uniform Structured 完全相同，但 V 保留 QV2，
导致 V MSE 退化
`{-next(row for row in comparison_rows if row['candidate'] == PRIMARY and row['baseline'] == STRUCTURED and row['metric'] == 'V_MSE')['improvement_percent']:.3f}%`，
联合 output MSE 退化
`{-float(structured_gate_row['improvement_percent']):.3f}%`。因此“保留
QV-V 作为稳健 anchor”只在相对 QV2 的局部目标成立，并未赢过 uniform
Structured teacher-forced exact-attention 基线。

## 相对 QV2 的 mean / P99 / CVaR99 改善

| 候选 | 指标 | mean | P99 | CVaR99 |
|---|---|---:|---:|---:|
""" + "\n".join(
        f"| {'Primary' if row['candidate'] == PRIMARY else 'Reverse'} | "
        f"{row['metric']} | {row['improvement_percent']:.3f}% | "
        f"{row['p99_improvement_percent']:.3f}% | "
        f"{row['cvar99_improvement_percent']:.3f}% |"
        for row in comparison_rows
        if row["baseline"] == QV
    ) + f"""

## Reverse 机制消融

反向组合 `K=QV2, V=Structured B2` 相对 QV2 的 output MSE 改善
`{reverse_vs_qv['attention_output_MSE']['improvement_percent']:.3f}%`
（prompt wins `{reverse_vs_qv['attention_output_MSE']['prompt_wins']}/10`）。
它说明在该 teacher-forced exact-attention 口径中，V codec 对 output
误差更直接；但它是预注册的 non-promotable ablation，不能事后替换主候选。

## 统计口径

- K/V：`10 prompts × 4 layers × 24 heads = 960` 单位。
- QK/KL/output：再包含 standard-RoPE 与 camera-ProPE 两半，共 `1920` 单位。
- `P99` 是单位分布的 99% 分位，`CVaR99` 是大于等于 P99 的尾部均值；
  完整数值见 `metric_summary.csv`。
- prompt wins 先在每个 prompt 内对 layer/head/half 做 macro mean，再逐 prompt
  比较，lower-is-better。
- attention 指标来自同时替换 K 和 V 后的联合 forward，不是 K/V-only 代理相加。
- formal config 与 preregistration 在任何 formal worker 输出前冻结。
"""

    run_info = {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": config["codebook_path"],
        "codebook_hash": config["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "seed": config["seed"],
        "prompt_list": sorted(
            {
                str(capture["prompt"])
                for capture in config["captures"]
            }
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_raw_metrics": str(raw_path),
        "source_raw_metrics_hash": sha256(raw_path),
        "preregistration_path": str(prereg_path),
        "preregistration_hash": sha256(prereg_path),
    }
    outputs = {
        args.output_dir / "metric_summary.csv": csv_bytes(summary_rows),
        args.output_dir / "pairwise_comparison.csv": csv_bytes(comparison_rows),
        args.output_dir / "runtime_payload.csv": csv_bytes(runtime_rows),
        args.output_dir / "summary.json": json_bytes(result),
        args.output_dir / "self_audit.json": json_bytes(self_audit),
        args.output_dir / "run_info.json": json_bytes(run_info),
        args.output_dir / "README_zh.md": readme.encode(),
    }
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            f"Refusing partial/complete overwrite in {args.output_dir}"
        )
    for path, payload in outputs.items():
        atomic_create(path, payload)
    print(json.dumps(result["primary_gate"], ensure_ascii=False))


if __name__ == "__main__":
    main()
