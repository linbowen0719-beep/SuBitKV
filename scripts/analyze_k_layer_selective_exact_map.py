#!/usr/bin/env python3
"""Summarize the frozen prompt-9 all-layer exact tensor-hybrid replay.

The three layer sets are read from the preregistered development design.  A
layer-selective candidate is synthesized without re-encoding: rows from the
K=Structured/V=QV method are selected on registered layers and rows from the
K/V=QV method elsewhere.  This is valid only for the teacher-forced,
per-layer exact-replay metrics and is not a closed-loop video result.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXACT_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test"
    "/tensor_hybrid_exact_all_layers_dev_p9_v1_20260724"
)
DEFAULT_DESIGN_ROOT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_structured_dev_design_20260724"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "k_layer_selective_exact_p9_map_20260724"
)
QV = "rvq_plus_quantvideo"
HYBRID = "rvq_plus_structured_k_quantvideo_v"
FIELDS = {
    "K_MSE": "historical_K_mse",
    "V_MSE": "historical_V_mse",
    "QK_logit_MSE": "attention_logit_mse",
    "attention_KL": "attention_kl_reference_to_candidate",
    "attention_output_MSE": "attention_output_mse",
}
EXPECTED_CANDIDATES = {
    "A_mid10": list(range(8, 16)),
    "B_mid20": list(range(16, 24)),
    "C_mid10_mid20": list(range(8, 24)),
}
CANDIDATE_LABELS = {
    "A_mid10": "A",
    "B_mid20": "B",
    "C_mid10_mid20": "C",
}
PAYLOAD_FIELDS = (
    "residual_payload_K_bytes",
    "residual_payload_V_bytes",
    "total_payload_K_bytes",
    "total_payload_V_bytes",
    "residual_physical_bits_per_scalar_K",
    "residual_physical_bits_per_scalar_V",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def parse_layers(value: Any) -> list[int]:
    if isinstance(value, list):
        return sorted({int(item) for item in value})
    result: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(token))
    return sorted(result)


def load_candidate_registry(
    design_root: Path,
) -> tuple[dict[str, list[int]], Path, Path]:
    registry_path = design_root / "candidate_registry.csv"
    protocol_path = design_root / "protocol.json"
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"candidate registry CSV absent from {design_root}"
        )
    if not protocol_path.is_file():
        raise FileNotFoundError(f"frozen protocol absent from {design_root}")
    with registry_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, list[int]] = {}
    for row in rows:
        candidate_id = (
            row.get("candidate_id") or row.get("id") or row.get("candidate")
        )
        layer_value = (
            row.get("structured_K_layers")
            or row.get("structured_k_layers")
            or row.get("layers")
            or row.get("layer_spec")
        )
        if not candidate_id or layer_value is None:
            raise ValueError(f"unrecognized candidate registry row: {row}")
        candidate_id = str(candidate_id)
        if candidate_id in result:
            raise ValueError(f"duplicate candidate id: {candidate_id}")
        result[candidate_id] = parse_layers(layer_value)
    if result != EXPECTED_CANDIDATES:
        raise ValueError(f"frozen candidate sets changed: {result}")

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_candidates: dict[str, list[int]] = {}
    for candidate in protocol.get("candidates", []):
        candidate_id = candidate.get("id")
        layer_value = candidate.get("structured_K_layers")
        if not candidate_id or layer_value is None:
            raise ValueError(f"unrecognized protocol candidate: {candidate}")
        protocol_candidates[str(candidate_id)] = parse_layers(layer_value)
    if protocol_candidates != EXPECTED_CANDIDATES:
        raise ValueError(
            f"protocol candidate sets changed: {protocol_candidates}"
        )
    if protocol.get("status") != "DESIGN_ONLY_NO_CANDIDATE_RUNS":
        raise ValueError(
            f"unexpected frozen design status: {protocol.get('status')}"
        )
    return result, registry_path, protocol_path


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        raise ValueError("empty metric set")
    return float(array.mean())


def tail(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    threshold = float(np.quantile(array, 0.99))
    return threshold, float(array[array >= threshold].mean())


def improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-30)


def selected_rows(
    rows: list[dict[str, str]],
    *,
    method: str,
    metric: str,
    layers: set[int] | None = None,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        if (
            row["method"] != method
            or row["intervention"] != "K_and_V"
            or int(row["bits"]) != 2
        ):
            continue
        if layers is not None and int(row["layer"]) not in layers:
            continue
        if metric in {"K_MSE", "V_MSE"} and row["attention_half"] != "standard_rope":
            continue
        output.append(row)
    return output


def validate_exact_input(
    config: dict[str, Any],
    rows: list[dict[str, str]],
    *,
    config_hash: str,
) -> dict[str, int]:
    if config.get("prompts") != [9]:
        raise ValueError("exact map must contain prompt 9 only")
    if config.get("layers") != list(range(30)):
        raise ValueError("exact map must contain all 30 layers")
    if config.get("tensor_hybrid_screen") is not True:
        raise ValueError("tensor-hybrid exact screen was not enabled")
    if config.get("seed") != 42:
        raise ValueError(f"unexpected codec seed: {config.get('seed')}")

    captures = config.get("captures", [])
    capture_keys = sorted(
        (
            int(capture["prompt_id"]),
            int(capture["layer"]),
            int(capture["task_index"]),
        )
        for capture in captures
    )
    expected_capture_keys = [(9, layer, layer) for layer in range(30)]
    if capture_keys != expected_capture_keys:
        raise ValueError("capture manifest is not prompt-9/all-30-layer complete")

    method_points = {
        (point.get("method"), int(point.get("bits", -1)))
        for point in config.get("method_points", [])
    }
    for required in ((QV, 2), (HYBRID, 2)):
        if required not in method_points:
            raise ValueError(f"required method point missing: {required}")

    if len(rows) != 21600:
        raise ValueError(f"raw row count {len(rows)} != 21600")
    if {row["config_hash"] for row in rows} != {config_hash}:
        raise ValueError("raw rows do not all reference the exact config hash")
    if {int(row["prompt_id"]) for row in rows} != {9}:
        raise ValueError("raw rows contain prompts other than prompt 9")
    if sorted({int(row["layer"]) for row in rows}) != list(range(30)):
        raise ValueError("raw rows do not cover all 30 layers")
    if sorted({int(row["head"]) for row in rows}) != list(range(24)):
        raise ValueError("raw rows do not cover all 24 heads")
    if {row["attention_half"] for row in rows} != {
        "standard_rope",
        "camera_prope",
    }:
        raise ValueError("raw rows do not cover both attention halves")
    if {row["hook_stage"] for row in rows} != {
        "pre_positional_historical_only"
    }:
        raise ValueError("unexpected attention hook stage")

    counts: dict[str, int] = {}
    for method in (QV, HYBRID):
        for metric, field in FIELDS.items():
            subset = selected_rows(rows, method=method, metric=metric)
            expected_count = 720 if metric in {"K_MSE", "V_MSE"} else 1440
            if len(subset) != expected_count:
                raise ValueError(
                    f"{method}/{metric} has {len(subset)} rows; "
                    f"expected {expected_count}"
                )
            values = np.asarray(
                [float(row[field]) for row in subset], dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"non-finite values in {method}/{metric}")
            counts[f"{method}:{metric}"] = len(subset)
    return counts


def audit_equal_payload(rows: list[dict[str, str]]) -> dict[str, Any]:
    indexed: dict[str, dict[tuple[int, int, str], dict[str, str]]] = {}
    for method in (QV, HYBRID):
        subset = selected_rows(
            rows, method=method, metric="attention_output_MSE"
        )
        method_index: dict[tuple[int, int, str], dict[str, str]] = {}
        for row in subset:
            key = (
                int(row["layer"]),
                int(row["head"]),
                row["attention_half"],
            )
            if key in method_index:
                raise ValueError(f"duplicate payload comparison key: {key}")
            method_index[key] = row
        indexed[method] = method_index
    if indexed[QV].keys() != indexed[HYBRID].keys():
        raise ValueError("payload methods do not have identical unit coverage")
    for key, qv_row in indexed[QV].items():
        hybrid_row = indexed[HYBRID][key]
        for field in PAYLOAD_FIELDS:
            if float(qv_row[field]) != float(hybrid_row[field]):
                raise ValueError(
                    f"payload mismatch at {key}/{field}: "
                    f"{qv_row[field]} versus {hybrid_row[field]}"
                )

    representative = next(iter(indexed[QV].values()))
    integer_fields = PAYLOAD_FIELDS[:4]
    for method, method_index in indexed.items():
        for field in integer_fields:
            values = {int(row[field]) for row in method_index.values()}
            if len(values) != 1:
                raise ValueError(f"{method}/{field} varies across units")
    source_k_bytes = {int(row["source_BF16_K_bytes"]) for row in indexed[QV].values()}
    source_v_bytes = {int(row["source_BF16_V_bytes"]) for row in indexed[QV].values()}
    if len(source_k_bytes) != 1 or len(source_v_bytes) != 1:
        raise ValueError("source BF16 payload varies across units")

    residual_k_bytes = int(representative["residual_payload_K_bytes"])
    residual_v_bytes = int(representative["residual_payload_V_bytes"])
    total_k_bytes = int(representative["total_payload_K_bytes"])
    total_v_bytes = int(representative["total_payload_V_bytes"])
    source_k = next(iter(source_k_bytes))
    source_v = next(iter(source_v_bytes))
    mib = float(1 << 20)
    return {
        "byte_exact_equal_between_qv2_and_hybrid": True,
        "comparison_unit_count": len(indexed[QV]),
        "baseline": "historical BF16 K+V, 16 bits/scalar",
        "MiB_definition": "2^20 bytes",
        "residual_payload_K_bytes": residual_k_bytes,
        "residual_payload_V_bytes": residual_v_bytes,
        "residual_payload_K_MiB": residual_k_bytes / mib,
        "residual_payload_V_MiB": residual_v_bytes / mib,
        "joint_residual_payload_MiB": (
            residual_k_bytes + residual_v_bytes
        )
        / mib,
        "total_payload_K_bytes": total_k_bytes,
        "total_payload_V_bytes": total_v_bytes,
        "joint_total_payload_MiB": (total_k_bytes + total_v_bytes) / mib,
        "source_BF16_K_MiB": source_k / mib,
        "source_BF16_V_MiB": source_v / mib,
        "joint_compression_ratio_vs_historical_BF16": (
            source_k + source_v
        )
        / (total_k_bytes + total_v_bytes),
        "residual_physical_bits_per_scalar_K": float(
            representative["residual_physical_bits_per_scalar_K"]
        ),
        "residual_physical_bits_per_scalar_V": float(
            representative["residual_physical_bits_per_scalar_V"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--design-root", type=Path, default=DEFAULT_DESIGN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.exact_root = args.exact_root.resolve()
    args.design_root = args.design_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"create-only output exists: {args.output_dir}")

    raw_path = args.exact_root / "tables" / "exact_attention_metrics.csv"
    config_path = args.exact_root / "config.json"
    aggregate_path = args.exact_root / "run_info" / "aggregate.json"
    if (
        not raw_path.is_file()
        or not config_path.is_file()
        or not aggregate_path.is_file()
    ):
        raise FileNotFoundError(args.exact_root)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    aggregate_run = json.loads(aggregate_path.read_text(encoding="utf-8"))
    with raw_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    config_hash = sha256_file(config_path)
    raw_hash = sha256_file(raw_path)
    sample_counts = validate_exact_input(
        config, rows, config_hash=config_hash
    )
    if aggregate_run.get("config_hash") != config_hash:
        raise ValueError("aggregate run_info config hash mismatch")
    if aggregate_run.get("codebook_hash", {}).get("K") != config[
        "codebook_hash"
    ]["K"] or aggregate_run.get("codebook_hash", {}).get("V") != config[
        "codebook_hash"
    ]["V"]:
        raise ValueError("aggregate run_info RVQ codebook hash mismatch")
    if aggregate_run.get("codebook_hash", {}).get(
        "structured_params"
    ) != config.get("params_hash"):
        raise ValueError("aggregate run_info structured parameter hash mismatch")
    payload_audit = audit_equal_payload(rows)

    candidates, registry_path, protocol_path = load_candidate_registry(
        args.design_root
    )
    values_by_layer: dict[int, dict[tuple[str, str], float]] = {}
    for layer in range(30):
        values: dict[tuple[str, str], float] = {}
        for method in (QV, HYBRID):
            for metric, field in FIELDS.items():
                subset = selected_rows(
                    rows, method=method, metric=metric, layers={layer}
                )
                expected_count = (
                    24 if metric in {"K_MSE", "V_MSE"} else 48
                )
                if len(subset) != expected_count:
                    raise ValueError(
                        f"layer {layer}/{method}/{metric} has "
                        f"{len(subset)} rows; expected {expected_count}"
                    )
                values[(method, metric)] = mean(
                    float(row[field]) for row in subset
                )
        values_by_layer[layer] = values

    qv_output_sum = sum(
        values_by_layer[layer][(QV, "attention_output_MSE")]
        for layer in range(30)
    )
    layer_rows: list[dict[str, Any]] = []
    for layer in range(30):
        values = values_by_layer[layer]
        qv_output = values[(QV, "attention_output_MSE")]
        hybrid_output = values[(HYBRID, "attention_output_MSE")]
        layer_rows.append(
            {
                "layer": layer,
                "qv_K_mse": values[(QV, "K_MSE")],
                "hybrid_K_mse": values[(HYBRID, "K_MSE")],
                "K_mse_improvement_percent": improvement(
                    values[(QV, "K_MSE")], values[(HYBRID, "K_MSE")]
                ),
                "qv_V_mse": values[(QV, "V_MSE")],
                "hybrid_V_mse": values[(HYBRID, "V_MSE")],
                "V_mse_improvement_percent": improvement(
                    values[(QV, "V_MSE")], values[(HYBRID, "V_MSE")]
                ),
                "qv_QK_logit_mse": values[(QV, "QK_logit_MSE")],
                "hybrid_QK_logit_mse": values[
                    (HYBRID, "QK_logit_MSE")
                ],
                "QK_mse_improvement_percent": improvement(
                    values[(QV, "QK_logit_MSE")],
                    values[(HYBRID, "QK_logit_MSE")],
                ),
                "qv_attention_KL": values[(QV, "attention_KL")],
                "hybrid_attention_KL": values[(HYBRID, "attention_KL")],
                "KL_improvement_percent": improvement(
                    values[(QV, "attention_KL")],
                    values[(HYBRID, "attention_KL")],
                ),
                "qv_output_mse": qv_output,
                "hybrid_output_mse": hybrid_output,
                "output_improvement_percent": improvement(
                    qv_output, hybrid_output
                ),
                "qv_output_mse_share_percent": 100.0
                * qv_output
                / qv_output_sum,
            }
        )

    qv_metric_rows = {
        metric: selected_rows(rows, method=QV, metric=metric)
        for metric in FIELDS
    }
    synthesis_rows: list[dict[str, Any]] = []
    for candidate_id, selected in candidates.items():
        selected_set = set(selected)
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "candidate_label": CANDIDATE_LABELS[candidate_id],
            "structured_K_layers": ",".join(str(item) for item in selected),
            "structured_layer_count": len(selected),
        }
        for metric, field in FIELDS.items():
            synthesized: list[float] = []
            for layer in range(30):
                method = HYBRID if layer in selected_set else QV
                synthesized.extend(
                    float(row[field])
                    for row in selected_rows(
                        rows, method=method, metric=metric, layers={layer}
                    )
                )
            reference = [float(row[field]) for row in qv_metric_rows[metric]]
            reference_mean = mean(reference)
            candidate_mean = mean(synthesized)
            reference_p99, reference_cvar99 = tail(reference)
            candidate_p99, candidate_cvar99 = tail(synthesized)
            prefix = metric.lower()
            record[f"{prefix}_sample_count"] = len(reference)
            record[f"{prefix}_qv_mean"] = reference_mean
            record[f"{prefix}_candidate_mean"] = candidate_mean
            record[f"{prefix}_mean_improvement_percent"] = improvement(
                reference_mean, candidate_mean
            )
            record[f"{prefix}_qv_p99"] = reference_p99
            record[f"{prefix}_candidate_p99"] = candidate_p99
            record[f"{prefix}_p99_improvement_percent"] = improvement(
                reference_p99, candidate_p99
            )
            record[f"{prefix}_qv_cvar99"] = reference_cvar99
            record[f"{prefix}_candidate_cvar99"] = candidate_cvar99
            record[f"{prefix}_cvar99_improvement_percent"] = improvement(
                reference_cvar99, candidate_cvar99
            )
        selected_reference_output = [
            float(row[FIELDS["attention_output_MSE"]])
            for row in selected_rows(
                rows,
                method=QV,
                metric="attention_output_MSE",
                layers=selected_set,
            )
        ]
        selected_hybrid_output = [
            float(row[FIELDS["attention_output_MSE"]])
            for row in selected_rows(
                rows,
                method=HYBRID,
                metric="attention_output_MSE",
                layers=selected_set,
            )
        ]
        record["selected_layer_output_sample_count"] = len(
            selected_reference_output
        )
        record["selected_layer_output_qv_mean"] = mean(
            selected_reference_output
        )
        record["selected_layer_output_hybrid_mean"] = mean(
            selected_hybrid_output
        )
        record[
            "selected_layer_output_mean_improvement_percent"
        ] = improvement(
            record["selected_layer_output_qv_mean"],
            record["selected_layer_output_hybrid_mean"],
        )
        record["equal_B2_payload"] = payload_audit[
            "byte_exact_equal_between_qv2_and_hybrid"
        ]
        synthesis_rows.append(record)

    full_hybrid_reference: dict[str, Any] = {}
    for metric, field in FIELDS.items():
        reference = [
            float(row[field])
            for row in selected_rows(rows, method=QV, metric=metric)
        ]
        hybrid = [
            float(row[field])
            for row in selected_rows(rows, method=HYBRID, metric=metric)
        ]
        full_hybrid_reference[metric] = {
            "sample_count": len(reference),
            "qv_mean": mean(reference),
            "full_hybrid_mean": mean(hybrid),
            "mean_improvement_percent": improvement(
                mean(reference), mean(hybrid)
            ),
        }

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    frozen_at = datetime.fromisoformat(
        protocol["frozen_at"].replace("Z", "+00:00")
    )
    exact_run_at = datetime.fromisoformat(aggregate_run["timestamp"])
    if frozen_at >= exact_run_at:
        raise ValueError(
            "frozen design timestamp is not earlier than exact run timestamp"
        )
    summary = {
        "schema_version": "k_layer_selective_exact_p9_map.summary.v1",
        "scope": {
            "prompt_ids": [9],
            "layers": list(range(30)),
            "heads": 24,
            "attention_halves": 2,
            "teacher_forced": True,
            "closed_loop_video": False,
        },
        "raw_rows": len(rows),
        "input_evidence": {
            "exact_config_path": str(config_path),
            "exact_config_sha256": config_hash,
            "raw_metrics_path": str(raw_path),
            "raw_metrics_sha256": raw_hash,
            "aggregate_run_info_path": str(aggregate_path),
            "aggregate_run_info_sha256": sha256_file(aggregate_path),
            "frozen_protocol_path": str(protocol_path),
            "frozen_protocol_sha256": sha256_file(protocol_path),
            "candidate_registry_path": str(registry_path),
            "candidate_registry_sha256": sha256_file(registry_path),
            "design_frozen_at": protocol["frozen_at"],
            "exact_run_timestamp": aggregate_run["timestamp"],
        },
        "aggregation": {
            "row_filter": "bits=2, intervention=K_and_V",
            "K_and_V_reconstruction_MSE": (
                "24 heads; standard_rope rows only because reconstruction "
                "values are byte-identical duplicates across the two halves"
            ),
            "attention_metrics": (
                "24 heads x both attention halves; unit-level arithmetic mean"
            ),
            "tail": (
                "numpy linear P99 threshold; CVaR99 is the arithmetic mean "
                "of observed units at or above that threshold"
            ),
            "improvement_percent": "100*(QV2-candidate)/QV2",
            "sample_counts": sample_counts,
        },
        "payload_audit": payload_audit,
        "candidate_synthesis": synthesis_rows,
        "full_K_hybrid_teacher_forced_reference": full_hybrid_reference,
        "limitations": [
            "Prompt 9 is a development prompt and cannot establish generalization.",
            "Layer synthesis is exact for independently scored teacher-forced layers, not for a 30-layer closed-loop trajectory.",
            "The three candidate boundaries were frozen before this exact all-layer run was created.",
            "All methods use equal B2 residual payload; RVQ base is identical.",
            "This analysis is diagnostic only and cannot trigger held-out or VBench evaluation.",
        ],
    }

    table = [
        "| Candidate | Structured K layers | Selected-band output | All-30 output | P99 | CVaR99 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in synthesis_rows:
        table.append(
            f"| {row['candidate_id']} ({row['candidate_label']}) | "
            f"{row['structured_K_layers']} | "
            f"{row['selected_layer_output_mean_improvement_percent']:.3f}% | "
            f"{row['attention_output_mse_mean_improvement_percent']:.3f}% | "
            f"{row['attention_output_mse_p99_improvement_percent']:.3f}% | "
            f"{row['attention_output_mse_cvar99_improvement_percent']:.3f}% |"
        )
    full_output_improvement = full_hybrid_reference[
        "attention_output_MSE"
    ]["mean_improvement_percent"]
    readme = f"""# K layer-selective exact map（prompt 9）

这是冻结候选后的 30 层 teacher-forced exact replay，不是视频质量结果。
每个候选只把注册层的 K residual 换成 Structured B2；其他 K 和全部 V
仍为 QuantVideo INT2。逐行审计确认两种 B2 codec 的 physical payload 完全相同：
residual K+V 为 `{payload_audit['joint_residual_payload_MiB']:.6f} MiB`，
RVQ base + residual K+V 为 `{payload_audit['joint_total_payload_MiB']:.6f} MiB`，
相对历史 BF16 K+V 为 `{payload_audit['joint_compression_ratio_vs_historical_BF16']:.6f}x`。

## 合成结果

{chr(10).join(table)}

Prompt 9 的逐层改善很不均匀：弱层包括 L0/L29，而中间层同时具有较大的
QV2 output-error mass 与 Structured 改善。这个结果支持先验的中层选择，
但不能预测 closed-loop rollout。作为参照，30 层 K 全部替换时，本次
teacher-forced output MSE 改善为 `{full_output_improvement:.3f}%`；这同样不是
闭环视频质量。

统计口径为 `bits=2 + K_and_V intervention`。K/V 重建 MSE 对两个 attention
half 是完全重复值，因此只计 `standard_rope` 的 24 个 head；QK、KL 和
attention-output 同时计 24 个 head 与两个 half。P99 使用 NumPy 线性分位点，
CVaR99 是所有不低于该阈值的实测 unit 均值。

三组仍必须完整跑完 p0/4/9 视频门槛，且不能根据本表新增或移动层边界。
本表不能用于触发 held-out 或 VBench。
"""

    prompt_list = sorted(
        {str(capture["prompt"]) for capture in config["captures"]}
    )
    script_path = Path(__file__).resolve()
    run_info = {
        "schema_version": "k_layer_selective_exact_p9_map.run_info.v1",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "codebook_path": config["codebook_path"],
        "codebook_hash": config["codebook_hash"],
        "structured_params_path": config["params_path"],
        "structured_params_hash": config["params_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": config["seed"],
        "prompt_list": prompt_list,
        "prompt_ids": config["prompts"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_raw_metrics": str(raw_path),
        "source_raw_metrics_hash": raw_hash,
        "design_root": str(args.design_root),
        "design_protocol_path": str(protocol_path),
        "design_protocol_hash": sha256_file(protocol_path),
        "candidate_registry_path": str(registry_path),
        "candidate_registry_hash": sha256_file(registry_path),
        "analysis_script_path": str(script_path),
        "analysis_script_hash": sha256_file(script_path),
        "execution_device": "CPU only; no GPU work",
    }
    output_bytes = {
        "layer_metrics.csv": csv_bytes(layer_rows),
        "candidate_synthesis.csv": csv_bytes(synthesis_rows),
        "summary.json": json_bytes(summary),
        "run_info.json": json_bytes(run_info),
        "README_zh.md": readme.encode("utf-8"),
    }
    evidence_manifest = {
        "schema_version": "k_layer_selective_exact_p9_map.evidence.v1",
        "input_artifacts": {
            str(config_path): config_hash,
            str(raw_path): raw_hash,
            str(aggregate_path): sha256_file(aggregate_path),
            str(protocol_path): sha256_file(protocol_path),
            str(registry_path): sha256_file(registry_path),
            str(script_path): sha256_file(script_path),
        },
        "output_artifacts": {
            name: sha256_bytes(data) for name, data in output_bytes.items()
        },
        "audit": {
            "raw_row_count": len(rows),
            "sample_counts": sample_counts,
            "payload_byte_exact_equal": payload_audit[
                "byte_exact_equal_between_qv2_and_hybrid"
            ],
            "design_frozen_before_exact_run": frozen_at < exact_run_at,
            "new_video_or_gpu_work": False,
            "heldout_or_vbench_read": False,
        },
    }
    output_bytes["evidence_manifest.json"] = json_bytes(evidence_manifest)
    for name, data in output_bytes.items():
        atomic_create(args.output_dir / name, data)


if __name__ == "__main__":
    main()
