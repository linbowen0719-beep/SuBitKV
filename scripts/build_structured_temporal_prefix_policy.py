#!/usr/bin/env python3
"""Build a create-only CPU audit of temporal structured-suffix policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_videogen.temporal_prefix_policy import (
    InfeasiblePrefixPolicyError,
    StructuredPayloadSpec,
    audit_schedule_history,
    payload_audit_rows,
    plan_temporal_prefix_policy,
    structured_payload_components,
    transition_rows,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO / "results/analysis/structured_temporal_prefix_policy_20260723"
)
FORMAL_REFERENCE = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "structured_residual_video_benchmark_v1_20260723/videos/"
    "l8c64_rvq_residual_structured_gaussian_h1r1_b2/metrics_prompt_0.csv"
)
CODEC_PARAMS = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "truncatable_residual_strict_final_20260723/training/codec_params.pt"
)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _gpu_type() -> str:
    command = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    names = sorted({line.strip() for line in command.stdout.splitlines() if line.strip()})
    return ", ".join(names) if names else "not available (CPU-only planner)"


def _formal_crosscheck(expected: Mapping[str, Any]) -> dict[str, Any]:
    if not FORMAL_REFERENCE.exists():
        return {
            "available": False,
            "path": str(FORMAL_REFERENCE),
            "reason": "reference CSV does not exist",
        }
    with FORMAL_REFERENCE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one formal-reference row, found {len(rows)}")
    row = rows[0]
    comparisons = {
        "structured_residual_MiB": {
            "calculated": float(expected["structured_residual_MiB"]),
            "formal_field": "residual_kv_payload_mib",
            "formal": float(row["residual_kv_payload_mib"]),
        },
        "joint_l8c64_plus_structured_MiB": {
            "calculated": float(expected["joint_l8c64_plus_structured_MiB"]),
            "formal_field": "kv_cache_mem_mb",
            "formal": float(row["kv_cache_mem_mb"]),
        },
        "structured_ratio_vs_bf16": {
            "calculated": float(expected["structured_ratio_vs_bf16"]),
            "formal_field": "residual_compression_ratio_vs_bf16",
            "formal": float(row["residual_compression_ratio_vs_bf16"]),
        },
        "joint_ratio_vs_bf16": {
            "calculated": float(expected["joint_ratio_vs_bf16"]),
            "formal_field": "compression_ratio_including_shared_codebook",
            "formal": float(row["compression_ratio_including_shared_codebook"]),
        },
    }
    for comparison in comparisons.values():
        comparison["absolute_difference"] = abs(
            comparison["calculated"] - comparison["formal"]
        )
        comparison["matches_report_rounding"] = (
            comparison["absolute_difference"] <= 1e-6
        )
    return {
        "available": True,
        "path": str(FORMAL_REFERENCE),
        "sha256": _sha256(FORMAL_REFERENCE),
        "prompt_id": int(row["prompt_id"]),
        "comparisons": comparisons,
        "all_match_report_rounding": all(
            item["matches_report_rounding"] for item in comparisons.values()
        ),
    }


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _report(
    *,
    strict_rows: list[dict[str, Any]],
    relaxed_rows: list[dict[str, Any]],
    strict_audit: Mapping[str, Any],
    relaxed_audit: Mapping[str, Any],
    relaxed_strict_audit: Mapping[str, Any],
    infeasible_b1_message: str,
    crosscheck: Mapping[str, Any],
    components: Mapping[str, int],
) -> str:
    lines = [
        "# Structured residual temporal prefix policy：单调性与物理 payload 审计",
        "",
        "## 结论",
        "",
        "在最终 `5 spans × uniform-B2 = 10 planes` 的固定 suffix-plane "
        "cap 下，本次预注册两个候选：",
        "",
        "1. **严格 fresh-B4 / 允许 B0**：每个新 span 都以 B4 物理提交，"
        "anchor 始终 B4，旧 span 只删 refinement plane。B0 表示只丢该 span "
        "的全部 scalar suffix planes，RVQ base、C256 predictor、uint8 ID 和 "
        "G128 FP16 scale 仍保留。",
        "2. **现有 B1–B4 codec 兼容 / 放松最后 fresh**：所有 middle span "
        "稳定降到 B1；第 5 个新 span 因硬 cap 只能以 B3 提交。它对已有 "
        "span 仍严格只降不升，但不满足“每个 fresh 都 B4”。",
        "",
        "| n | 严格 fresh-B4（B0 设计） | planes | B1-compatible relaxed | planes |",
        "|---:|---|---:|---|---:|",
    ]
    for strict, relaxed in zip(strict_rows, relaxed_rows):
        lines.append(
            f"| {strict['span_count']} | `{strict['schedule']}` | "
            f"{strict['plane_units']} | `{relaxed['schedule']}` | "
            f"{relaxed['plane_units']} |"
        )
    lines += [
        "",
        f"严格候选完整 invariant audit：`{strict_audit['valid']}`；"
        f"relaxed 候选按其声明约束 audit：`{relaxed_audit['valid']}`。"
        "relaxed 候选若改用严格 fresh-B4 规则审计则失败，且失败点是第 5 "
        f"个 fresh span：`{relaxed_strict_audit['violations'][-1]}`。",
        "",
        "## 为什么 `[4,1,1,1,3]` 不是严格 fresh-B4 解",
        "",
        "若每个 span 至少保留 B1，同时要求 anchor B4 和第 5 个 fresh span "
        "B4，则终态最低 plane 数为",
        "",
        "\\[",
        "4_{anchor}+4_{fresh}+3\\times1_{middle}=11>10.",
        "\\]",
        "",
        f"规划器的原始不可行诊断为：`{infeasible_b1_message}`。"
        "`[4,1,1,1,3]` 本身对旧 span 是单调的且总量等于 10，但新 span "
        "从未在受 cap 约束的 committed state 中保有 B4；若先把 B4 全部存下"
        "再删到 B3，瞬时 stored planes 会达到 11。因此它只能作为明确放松 "
        "fresh 条件的第二候选，不能冒充严格解。",
        "",
        "## 完整物理 payload",
        "",
        "口径与正在使用的 Wan2.2-5B runner 一致：每个量化 span 的实际状态"
        "形状为 `[1,24,7040,128]`，30 层、K/V 分开，并以 "
        "`logical_streams=2` 计入 RoPE/PRoPE 两条逻辑流。原始单流维数仍是 "
        "128；这里的 `×2` 只是正式 cache memory accounting。",
        "",
        "每个 span 聚合全部 30 层、K/V 和两条逻辑流后：",
        "",
        "| 固定/可截断组成 | MiB |",
        "|---|---:|",
        f"| C256 BF16 predictor centroids | "
        f"{components['predictor_centroid_bytes_per_span'] / 2**20:.6f} |",
        f"| uint8 cluster IDs | "
        f"{components['cluster_id_bytes_per_span'] / 2**20:.6f} |",
        f"| G128 FP16 scales | "
        f"{components['scale_bytes_per_span'] / 2**20:.6f} |",
        f"| 一个 suffix plane | "
        f"{components['one_suffix_plane_bytes_per_span'] / 2**20:.6f} |",
        f"| L8C64 6-bit RVQ indices | "
        f"{components['rvq_index_bytes_per_span'] / 2**20:.6f} |",
        f"| shared L8C64 INT4 codebooks（全模型仅一次） | "
        f"{components['shared_rvq_codebook_bytes'] / 2**20:.6f} |",
        "",
        "下面的 structured residual 已完整包含 predictor centroid、ID、scale "
        "和实际保留 planes；joint 再加 L8C64 6-bit indices 与全模型一次性 "
        "INT4 codebook。MiB 为 `bytes / 2^20`，压缩率基线为相同 K/V 的 BF16。",
        "",
        "| candidate | n | schedule | structured residual MiB | joint MiB | "
        "BF16 MiB | joint ratio |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for candidate, rows in (
        ("strict_B0", strict_rows),
        ("relaxed_B1", relaxed_rows),
    ):
        for row in rows:
            lines.append(
                f"| {candidate} | {row['span_count']} | `{row['schedule']}` | "
                f"{_fmt(row['structured_residual_MiB'])} | "
                f"{_fmt(row['joint_l8c64_plus_structured_MiB'])} | "
                f"{_fmt(row['bf16_kv_baseline_MiB'])} | "
                f"{_fmt(row['joint_ratio_vs_bf16'], 6)}× |"
            )
    final = strict_rows[-1]
    lines += [
        "",
        "两个候选终态都恰好 10 planes，所以终态 payload 与 uniform-B2 "
        "逐字节相同：structured residual "
        f"`{final['structured_residual_MiB']:.6f} MiB`，"
        "加 L8C64 RVQ indices 和 shared INT4 codebooks 后为 "
        f"`{final['joint_l8c64_plus_structured_MiB']:.6f} MiB`，"
        f"相对 BF16 为 `{final['joint_ratio_vs_bf16']:.6f}×`。"
        "早期状态允许使用最终 cap 的闲置 planes，因此相对“当前 n 个 span "
        "均匀 B2”会更大，但始终不超过最终 5-span capacity。",
        "",
        "正式 prompt-0 accounting 的只读交叉检查："
        f"`all_match_report_rounding={crosscheck.get('all_match_report_rounding')}`。"
        "详细逐字段差异见 `feasibility_audit.json`。",
        "",
        "## 实现边界与下一步",
        "",
        "- 本轮只新增纯 Python planner、审计与 CPU smoke；没有修改 "
        "`compress.py`、`uncompress.py`、pipeline、`run_small_eval.py` 或 "
        "`structured_truncatable.py`，也没有占用 GPU。",
        "- 当前 structured decoder 只接受 B1–B4。严格候选中的 B0 是待实现"
        "设计语义，接入前必须新增 predictor-only/B0 decode 和 smoke test；"
        "本轮禁止接入 core。",
        "- 本结果只证明可实现性、单调性和物理预算，不包含质量收益。两个候选"
        "已预注册，未来应先做 exact attention/短 closed-loop gate，再决定"
        "是否跑完整 Similarity/VBench。",
        "- 全局 Gaussian tree 参数不随 cache/span 复制，和现有正式口径一样"
        "不计入 per-cache payload；其来源文件和 hash 记录在 manifest。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"create-only result directory already exists; choose a new path: {output}"
        )
    output.mkdir(parents=True)

    config = {
        "schema_version": "structured_temporal_prefix_policy.config.v1",
        "policy": {
            "max_spans": 5,
            "target_uniform_bits": 2,
            "max_bits": 4,
            "anchor_bits": 4,
            "strict_min_suffix_bits": 0,
            "relaxed_min_suffix_bits": 1,
            "plane_cap": 10,
        },
        "payload": {
            "batch": 1,
            "heads": 24,
            "tokens": 7040,
            "dim": 128,
            "layers": 30,
            "tensor_families": 2,
            "logical_streams": 2,
            "predictor_clusters": 256,
            "predictor_value_bits": 16,
            "cluster_id_bits": 8,
            "group_size": 128,
            "scale_bits": 16,
            "bf16_bits": 16,
            "rvq_levels": 8,
            "rvq_codebook_size": 64,
            "rvq_codebook_value_bits": 4,
            "rvq_codebook_scale_bits": 16,
            "logical_head_dim": 256,
        },
        "formal_reference": str(FORMAL_REFERENCE),
        "codec_params": str(CODEC_PARAMS),
        "randomness": "none",
    }
    config_path = output / "config.json"
    _write_json(config_path, config)

    strict = plan_temporal_prefix_policy(
        max_spans=5,
        target_uniform_bits=2,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=0,
        require_fresh_max=True,
    )
    relaxed = plan_temporal_prefix_policy(
        max_spans=5,
        target_uniform_bits=2,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=1,
        require_fresh_max=False,
    )
    strict_audit = audit_schedule_history(
        strict.schedules,
        plane_cap=10,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=0,
        require_fresh_max=True,
    )
    relaxed_audit = audit_schedule_history(
        relaxed.schedules,
        plane_cap=10,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=1,
        require_fresh_max=False,
    )
    relaxed_strict_audit = audit_schedule_history(
        relaxed.schedules,
        plane_cap=10,
        max_bits=4,
        anchor_bits=4,
        min_suffix_bits=1,
        require_fresh_max=True,
    )
    try:
        plan_temporal_prefix_policy(
            max_spans=5,
            target_uniform_bits=2,
            max_bits=4,
            anchor_bits=4,
            min_suffix_bits=1,
            require_fresh_max=True,
        )
    except InfeasiblePrefixPolicyError as error:
        infeasible_b1_message = str(error)
    else:
        raise AssertionError("strict B1 policy unexpectedly became feasible")

    spec = StructuredPayloadSpec(**config["payload"])
    components = structured_payload_components(spec)
    strict_payload = [
        {"candidate": "strict_fresh_b4_allow_b0", **row}
        for row in payload_audit_rows(strict, spec)
    ]
    relaxed_payload = [
        {"candidate": "relaxed_b1_compatible_last_fresh_b3", **row}
        for row in payload_audit_rows(relaxed, spec)
    ]
    crosscheck = _formal_crosscheck(strict_payload[-1])

    schedule_rows: list[dict[str, Any]] = []
    for name, plan in (
        ("strict_fresh_b4_allow_b0", strict),
        ("relaxed_b1_compatible_last_fresh_b3", relaxed),
    ):
        for span_count, schedule in enumerate(plan.schedules, start=1):
            schedule_rows.append(
                {
                    "candidate": name,
                    "span_count": span_count,
                    "schedule": "[" + ",".join(map(str, schedule)) + "]",
                    "plane_units": sum(schedule),
                    "plane_cap": plan.plane_cap,
                    "cap_slack": plan.plane_cap - sum(schedule),
                    "fresh_bits": schedule[-1],
                    "anchor_bits": schedule[0],
                    "minimum_stored_bits": min(schedule),
                    "old_spans_never_upgrade": True,
                }
            )
    transition_output: list[dict[str, Any]] = []
    for name, plan in (
        ("strict_fresh_b4_allow_b0", strict),
        ("relaxed_b1_compatible_last_fresh_b3", relaxed),
    ):
        transition_output.extend(
            {"candidate": name, **row} for row in transition_rows(plan)
        )

    _write_csv(output / "policy_schedules.csv", schedule_rows)
    _write_csv(output / "transitions.csv", transition_output)
    _write_csv(output / "payload_audit.csv", [*strict_payload, *relaxed_payload])
    feasibility = {
        "schema_version": "structured_temporal_prefix_policy.feasibility.v1",
        "strict_fresh_b4_allow_b0": strict_audit,
        "relaxed_b1_compatible": relaxed_audit,
        "relaxed_candidate_under_strict_fresh_rule": relaxed_strict_audit,
        "strict_b1_infeasibility": {
            "feasible": False,
            "lower_bound_equation": "4(anchor)+4(fresh)+3*1(middle)=11>10",
            "planner_error": infeasible_b1_message,
        },
        "payload_components_bytes": components,
        "formal_uniform_b2_crosscheck": crosscheck,
    }
    _write_json(output / "feasibility_audit.json", feasibility)

    report = _report(
        strict_rows=strict_payload,
        relaxed_rows=relaxed_payload,
        strict_audit=strict_audit,
        relaxed_audit=relaxed_audit,
        relaxed_strict_audit=relaxed_strict_audit,
        infeasible_b1_message=infeasible_b1_message,
        crosscheck=crosscheck,
        components=components,
    )
    (output / "README.md").write_text(report, encoding="utf-8")

    source_paths = [
        REPO / "quant_videogen/temporal_prefix_policy.py",
        REPO / "tests/test_temporal_prefix_policy.py",
        Path(__file__).resolve(),
        config_path,
        output / "policy_schedules.csv",
        output / "transitions.csv",
        output / "payload_audit.csv",
        output / "feasibility_audit.json",
        output / "README.md",
    ]
    if FORMAL_REFERENCE.exists():
        source_paths.append(FORMAL_REFERENCE)
    if CODEC_PARAMS.exists():
        source_paths.append(CODEC_PARAMS)
    manifest = {
        "schema_version": "structured_temporal_prefix_policy.evidence.v1",
        "accounting_scope": (
            "all 30 layers, K/V, two logical RoPE/PRoPE streams; structured "
            "predictor centroids + uint8 IDs + G128 FP16 scales + retained "
            "suffix planes; joint adds L8C64 6-bit indices and shared INT4 "
            "codebooks"
        ),
        "bf16_baseline": "same logical K/V values at 16 bits/value",
        "global_gaussian_params_in_cache_payload": False,
        "no_gpu_compute": True,
        "pipeline_modified": False,
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
    }
    _write_json(output / "evidence_manifest.json", manifest)

    recorded_outputs = [
        config_path,
        output / "policy_schedules.csv",
        output / "transitions.csv",
        output / "payload_audit.csv",
        output / "feasibility_audit.json",
        output / "README.md",
        output / "evidence_manifest.json",
    ]
    run_info = {
        "schema_version": "structured_temporal_prefix_policy.run_info.v1",
        "git_commit": _git_commit(),
        "config_path": str(config_path),
        "config_hash": _sha256(config_path),
        "codebook_path": None,
        "codebook_hash": None,
        "hostname": socket.gethostname(),
        "gpu_type": _gpu_type(),
        "seed": 0,
        "prompt_list": [],
        "timestamp": datetime.now().astimezone().isoformat(),
        "execution": "deterministic CPU-only analytical planner; no RNG used",
        "output_sha256": {str(path): _sha256(path) for path in recorded_outputs},
    }
    _write_json(output / "run_info.json", run_info)

    print(json.dumps({
        "output": str(output),
        "strict_final": strict.schedules[-1],
        "relaxed_final": relaxed.schedules[-1],
        "strict_planes": [sum(item) for item in strict.schedules],
        "relaxed_planes": [sum(item) for item in relaxed.schedules],
        "formal_crosscheck": crosscheck.get("all_match_report_rounding"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
