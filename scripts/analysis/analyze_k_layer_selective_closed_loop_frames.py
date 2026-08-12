#!/usr/bin/env python3
"""Create-only multi-candidate frame diagnostics for the K-layer dev screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import time
from typing import Any, Iterable, Sequence

import torch


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[2]
BASE_SCRIPT = SCRIPT.with_name("analyze_closed_loop_frame_diagnostics.py")
SPEC = importlib.util.spec_from_file_location("closed_loop_frame_base", BASE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

CANDIDATES = ("A_mid10", "B_mid20", "C_mid10_mid20")
COMPARATORS = ("rvq_quantvideo_int2", "uniform_structured_b2")
PROMPTS = (0, 4, 9)
EXPECTED_METHODS = set(CANDIDATES) | set(COMPARATORS)
MANIFEST_SCHEMA = (
    "k_layer_selective_closed_loop_frame_diagnostics.input_manifest.v1"
)
SUMMARY_SCHEMA = "k_layer_selective_closed_loop_frame_diagnostics.summary.v1"
RUN_INFO_SCHEMA = (
    "k_layer_selective_closed_loop_frame_diagnostics.run_info.v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-label", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def validate_artifact(record: dict[str, Any], *, label: str) -> Path:
    path = Path(str(record.get("path", ""))).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != record.get("sha256"):
        raise ValueError(f"{label}: hash mismatch: {path}")
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"{label}: byte-size mismatch: {path}")
    return path


def _validate_formal_gate(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    gate = manifest.get("formal_gate")
    if not isinstance(gate, dict):
        raise ValueError("formal_gate missing")
    summary_path = validate_artifact(gate["summary"], label="gate summary")
    evidence_path = validate_artifact(
        gate["evidence_manifest"], label="gate evidence"
    )
    run_info_path = validate_artifact(gate["run_info"], label="gate run_info")
    summary = load_json(summary_path)
    evidence = load_json(evidence_path)
    run_info = load_json(run_info_path)
    if (
        summary.get("decision") != "STOP_NO_HELDOUT_NO_VBENCH"
        or summary.get("winner") is not None
        or gate.get("decision") != summary["decision"]
        or gate.get("winner") is not None
    ):
        raise ValueError("formal gate is not the frozen negative development result")
    if summary.get("runtime_audit_count") != 9 or summary.get("similarity_count") != 9:
        raise ValueError("formal gate does not cover all nine development units")
    for name, path in (
        ("summary.json", summary_path),
        ("evidence_manifest.json", evidence_path),
    ):
        if run_info.get("artifacts", {}).get(name) != sha256_file(path):
            raise ValueError(f"gate run_info hash mismatch for {name}")
    return summary, evidence, run_info


def validate_inputs(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong schema")
    source_protocol = validate_artifact(
        manifest["source_protocol_manifest"], label="source protocol manifest"
    )
    source_payload = load_json(source_protocol)
    if manifest.get("protocol") != source_payload.get("protocol"):
        raise ValueError("frame/chunk protocol differs from prior tensor diagnostics")
    protocol = manifest["protocol"]
    expected_protocol = {
        "expected_decoded_frames": 173,
        "skip_decoded_frame0": True,
        "expected_compared_frames": 172,
        "expected_width": 1280,
        "expected_height": 704,
        "expected_fps": 16.0,
        "generation_chunks": 11,
        "latent_frames_per_generation_chunk": 4,
        "vae_temporal_stride_after_first_latent": 4,
        "phase_chunks": {
            "early": [0, 1, 2, 3],
            "mid": [4, 5, 6, 7],
            "late": [8, 9, 10],
        },
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            raise ValueError(f"protocol.{field} changed")

    methods = manifest.get("methods")
    if not isinstance(methods, list):
        raise ValueError("methods missing")
    by_method = {str(item["method_id"]): item for item in methods}
    if set(by_method) != EXPECTED_METHODS or len(methods) != len(EXPECTED_METHODS):
        raise ValueError(f"method set mismatch: {sorted(by_method)}")
    if any(by_method[item].get("role") != "candidate" for item in CANDIDATES):
        raise ValueError("candidate roles changed")
    if any(by_method[item].get("role") != "comparator" for item in COMPARATORS):
        raise ValueError("comparator roles changed")

    gate_summary, gate_evidence, gate_run_info = _validate_formal_gate(manifest)
    runtime_evidence = {
        (str(item["candidate_id"]), int(item["prompt_id"])): item
        for item in gate_evidence["runtime"]
    }
    similarity_evidence = {
        (str(item["candidate_id"]), int(item["prompt_id"])): item
        for item in gate_evidence["candidate_similarity"]
    }
    expected_keys = {
        (candidate, prompt) for candidate in CANDIDATES for prompt in PROMPTS
    }
    if set(runtime_evidence) != expected_keys or set(similarity_evidence) != expected_keys:
        raise ValueError("gate evidence grid mismatch")

    prompts = manifest.get("prompts")
    if not isinstance(prompts, list) or [
        int(item["prompt_id"]) for item in prompts
    ] != list(PROMPTS):
        raise ValueError("development prompt set/order changed")
    input_artifacts = []
    for prompt in prompts:
        prompt_id = int(prompt["prompt_id"])
        reference_path = validate_artifact(
            prompt["reference"], label=f"reference p{prompt_id}"
        )
        input_artifacts.append(
            {
                "kind": "reference",
                "prompt_id": prompt_id,
                "method_id": "bf16",
                "path": str(reference_path),
                "sha256": sha256_file(reference_path),
            }
        )
        tests = prompt.get("tests")
        if not isinstance(tests, dict) or set(tests) != EXPECTED_METHODS:
            raise ValueError(f"prompt {prompt_id}: test method grid mismatch")
        for method_id, record in tests.items():
            video_path = validate_artifact(
                record, label=f"video {method_id} p{prompt_id}"
            )
            input_artifacts.append(
                {
                    "kind": "test_video",
                    "prompt_id": prompt_id,
                    "method_id": method_id,
                    "path": str(video_path),
                    "sha256": sha256_file(video_path),
                }
            )
            if method_id not in CANDIDATES:
                continue
            runtime_path = validate_artifact(
                record["runtime_run_info"],
                label=f"runtime {method_id} p{prompt_id}",
            )
            runtime = load_json(runtime_path)
            evidence = runtime_evidence[(method_id, prompt_id)]
            if (
                str(runtime_path) != str(Path(evidence["run_info_path"]).resolve())
                or sha256_file(runtime_path) != evidence["run_info_sha256"]
                or runtime.get("candidate_id") != method_id
                or int(runtime["prompt_list"][0]["prompt_id"]) != prompt_id
                or Path(runtime["video_path"]["path"]).resolve() != video_path
                or runtime["video_path"]["sha256"] != record["sha256"]
            ):
                raise ValueError(
                    f"candidate runtime/evaluation/video mismatch: {method_id} p{prompt_id}"
                )
            runtime_config = Path(runtime["config_path"]).resolve()
            if sha256_file(runtime_config) != runtime["config_hash"]:
                raise ValueError(f"{runtime_config}: runtime config hash mismatch")
            similarity_path = validate_artifact(
                record["similarity_source"],
                label=f"Similarity {method_id} p{prompt_id}",
            )
            similarity = similarity_evidence[(method_id, prompt_id)]
            if (
                str(similarity_path) != str(Path(similarity["json_path"]).resolve())
                or sha256_file(similarity_path) != similarity["json_sha256"]
            ):
                raise ValueError(
                    f"candidate Similarity/evaluation mismatch: {method_id} p{prompt_id}"
                )
            input_artifacts.extend(
                [
                    {
                        "kind": "runtime_run_info",
                        "prompt_id": prompt_id,
                        "method_id": method_id,
                        "path": str(runtime_path),
                        "sha256": sha256_file(runtime_path),
                    },
                    {
                        "kind": "similarity_json",
                        "prompt_id": prompt_id,
                        "method_id": method_id,
                        "path": str(similarity_path),
                        "sha256": sha256_file(similarity_path),
                    },
                ]
            )

    checks = manifest.get("archived_metric_checks")
    if not isinstance(checks, list) or len(checks) != 15:
        raise ValueError("expected 15 archived metric checks")
    check_keys = {(str(row["method_id"]), int(row["prompt_id"])) for row in checks}
    if check_keys != {
        (method, prompt) for method in EXPECTED_METHODS for prompt in PROMPTS
    }:
        raise ValueError("archived metric check grid mismatch")
    for check in checks:
        source = Path(str(check["source_path"])).resolve()
        if sha256_file(source) != check["source_sha256"]:
            raise ValueError(f"{source}: archived metric source hash mismatch")

    guards = manifest.get("scope_guards", {})
    if (
        guards.get("development_prompts_only") != list(PROMPTS)
        or guards.get("heldout_prompts_read") is not False
        or guards.get("vbench_run") is not False
        or guards.get("new_video_generation") is not False
    ):
        raise ValueError("scope guards changed")
    return {
        "manifest": manifest,
        "methods": by_method,
        "formal_gate_summary": gate_summary,
        "formal_gate_evidence": gate_evidence,
        "formal_gate_run_info": gate_run_info,
        "input_artifacts": input_artifacts,
    }


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise ValueError("cannot average empty values")
    return sum(items) / len(items)


def pairwise_all(
    frame_rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_frames = []
    all_segments = []
    all_tails = []
    all_sets: dict[str, Any] = {}
    for candidate in CANDIDATES:
        frames, segments, tails, frame_sets = base.paired_outputs(
            frame_rows,
            candidate_method=candidate,
            comparator_methods=COMPARATORS,
        )
        all_frames.extend(frames)
        all_segments.extend(segments)
        all_tails.extend(tails)
        duplicate = set(all_sets) & set(frame_sets)
        if duplicate:
            raise ValueError(f"duplicate worst-frame keys: {duplicate}")
        all_sets.update(frame_sets)
    return all_frames, all_segments, all_tails, all_sets


def macro_segments(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["candidate_method"]),
            str(row["comparator_method"]),
            str(row["segment_type"]),
            str(row["segment_id"]),
        )
        groups.setdefault(key, []).append(row)
    output = []
    for key, items in groups.items():
        if len(items) != 3 or {int(item["prompt_id"]) for item in items} != set(
            PROMPTS
        ):
            raise ValueError(f"incomplete prompt-macro segment: {key}")
        candidate_mse = _mean(float(item["candidate_mse_mean"]) for item in items)
        comparator_mse = _mean(float(item["comparator_mse_mean"]) for item in items)
        output.append(
            {
                "candidate_method": key[0],
                "comparator_method": key[1],
                "segment_type": key[2],
                "segment_id": key[3],
                "prompts": 3,
                "frames_per_prompt": int(items[0]["frames"]),
                "candidate_prompt_macro_mse": candidate_mse,
                "comparator_prompt_macro_mse": comparator_mse,
                "candidate_mse_improvement_pct": base.improvement_pct(
                    candidate_mse, comparator_mse
                ),
                "mean_prompt_mse_improvement_pct": _mean(
                    float(item["candidate_mse_improvement_pct"]) for item in items
                ),
                "prompt_mse_wins": sum(
                    float(item["candidate_mse_improvement_pct"]) > 0 for item in items
                ),
                "mean_frame_mse_win_rate": _mean(
                    float(item["candidate_mse_win_rate"]) for item in items
                ),
                "candidate_psnr_minus_comparator_db_prompt_macro": _mean(
                    float(item["candidate_psnr_minus_comparator_db_mean"])
                    for item in items
                ),
                "candidate_ssim_minus_comparator_prompt_macro": _mean(
                    float(item["candidate_ssim_minus_comparator_mean"])
                    for item in items
                ),
                "worst_prompt_mse_improvement_pct": min(
                    float(item["candidate_mse_improvement_pct"]) for item in items
                ),
            }
        )
    segment_type_order = {"overall": 0, "phase": 1, "generation_chunk": 2}
    phase_order = {"all": -1, "early": 0, "mid": 1, "late": 2}
    return sorted(
        output,
        key=lambda row: (
            CANDIDATES.index(row["candidate_method"]),
            COMPARATORS.index(row["comparator_method"]),
            segment_type_order[row["segment_type"]],
            phase_order.get(
                row["segment_id"],
                int(row["segment_id"])
                if str(row["segment_id"]).isdigit()
                else 99,
            ),
        ),
    )


def macro_tails(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (str(row["candidate_method"]), str(row["comparator_method"])), []
        ).append(row)
    output = []
    for (candidate, comparator), items in groups.items():
        if len(items) != 3:
            raise ValueError(f"incomplete macro tail: {candidate} vs {comparator}")
        record: dict[str, Any] = {
            "candidate_method": candidate,
            "comparator_method": comparator,
            "prompts": 3,
        }
        for label, candidate_field, comparator_field in (
            ("p95", "candidate_mse_p95", "comparator_mse_p95"),
            ("p99", "candidate_mse_p99", "comparator_mse_p99"),
            ("cvar95", "candidate_mse_cvar95", "comparator_mse_cvar95"),
            ("max", "candidate_mse_max", "comparator_mse_max"),
        ):
            candidate_value = _mean(float(item[candidate_field]) for item in items)
            comparator_value = _mean(float(item[comparator_field]) for item in items)
            record[f"candidate_prompt_macro_mse_{label}"] = candidate_value
            record[f"comparator_prompt_macro_mse_{label}"] = comparator_value
            record[f"candidate_{label}_improvement_pct"] = base.improvement_pct(
                candidate_value, comparator_value
            )
            record[f"prompt_{label}_wins"] = sum(
                float(item[candidate_field]) < float(item[comparator_field])
                for item in items
            )
        record["mean_cvar95_frame_set_jaccard"] = _mean(
            float(item["cvar95_frame_set_jaccard"]) for item in items
        )
        output.append(record)
    return sorted(
        output,
        key=lambda row: (
            CANDIDATES.index(row["candidate_method"]),
            COMPARATORS.index(row["comparator_method"]),
        ),
    )


def prompt_wins(segment_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_method": row["candidate_method"],
            "comparator_method": row["comparator_method"],
            "prompt_id": row["prompt_id"],
            "candidate_mse_improvement_pct": row["candidate_mse_improvement_pct"],
            "candidate_mse_win": float(row["candidate_mse_improvement_pct"]) > 0,
            "candidate_mse_win_rate_over_frames": row["candidate_mse_win_rate"],
            "candidate_psnr_minus_comparator_db_mean": row[
                "candidate_psnr_minus_comparator_db_mean"
            ],
            "candidate_psnr_win": (
                float(row["candidate_psnr_minus_comparator_db_mean"]) > 0
            ),
            "candidate_ssim_minus_comparator_mean": row[
                "candidate_ssim_minus_comparator_mean"
            ],
            "candidate_ssim_win": (
                float(row["candidate_ssim_minus_comparator_mean"]) > 0
            ),
        }
        for row in segment_rows
        if row["segment_type"] == "overall"
    ]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    base.write_csv_create_only(path, rows, list(rows[0]))


def _readme(
    *,
    gate_summary: dict[str, Any],
    macro: Sequence[dict[str, Any]],
    tails: Sequence[dict[str, Any]],
    wins: Sequence[dict[str, Any]],
) -> str:
    overall = {
        (row["candidate_method"], row["comparator_method"]): row
        for row in macro
        if row["segment_type"] == "overall"
    }
    phases = {
        (
            row["candidate_method"],
            row["comparator_method"],
            row["segment_id"],
        ): row
        for row in macro
        if row["segment_type"] == "phase"
    }
    tail_lookup = {
        (row["candidate_method"], row["comparator_method"]): row for row in tails
    }
    gate_rank = {row["candidate_id"]: row for row in gate_summary["ranking"]}

    gate_lines = []
    overall_lines = []
    phase_lines = []
    tail_lines = []
    conclusion_lines = []
    for candidate in CANDIDATES:
        gate = gate_rank[candidate]
        gate_lines.append(
            f"| {candidate} | "
            f"{gate['improvement_vs_rvq_quantvideo_int2_percent']:.3f}% | "
            f"{gate['improvement_vs_uniform_structured_b2_percent']:.3f}% |"
        )
        qv = overall[(candidate, "rvq_quantvideo_int2")]
        uniform = overall[(candidate, "uniform_structured_b2")]
        overall_lines.append(
            f"| {candidate} | {qv['candidate_mse_improvement_pct']:.3f}% "
            f"({qv['prompt_mse_wins']}/3) | "
            f"{uniform['candidate_mse_improvement_pct']:.3f}% "
            f"({uniform['prompt_mse_wins']}/3) | "
            f"{qv['mean_frame_mse_win_rate']:.3f} |"
        )
        for comparator in COMPARATORS:
            phase_lines.append(
                f"| {candidate} | {comparator} | "
                f"{phases[(candidate, comparator, 'early')]['candidate_mse_improvement_pct']:.3f}% | "
                f"{phases[(candidate, comparator, 'mid')]['candidate_mse_improvement_pct']:.3f}% | "
                f"{phases[(candidate, comparator, 'late')]['candidate_mse_improvement_pct']:.3f}% |"
            )
            tail = tail_lookup[(candidate, comparator)]
            tail_lines.append(
                f"| {candidate} | {comparator} | "
                f"{tail['candidate_p95_improvement_pct']:.3f}% | "
                f"{tail['candidate_cvar95_improvement_pct']:.3f}% | "
                f"{tail['prompt_cvar95_wins']}/3 |"
            )
        qv_phase = [
            phases[(candidate, "rvq_quantvideo_int2", phase)][
                "candidate_mse_improvement_pct"
            ]
            for phase in ("early", "mid", "late")
        ]
        conclusion_lines.append(
            f"- `{candidate}` 相对 QV2 的 early/mid/late pixel-MSE 改善依次为 "
            f"`{qv_phase[0]:.3f}% / {qv_phase[1]:.3f}% / {qv_phase[2]:.3f}%`；"
            f"整体 prompt 胜数为 `{qv['prompt_mse_wins']}/3`，CVaR95 胜数为 "
            f"`{tail_lookup[(candidate, 'rvq_quantvideo_int2')]['prompt_cvar95_wins']}/3`。"
        )

    prompt_lines = "\n".join(
        "| {candidate_method} | {comparator_method} | {prompt_id} | "
        "{candidate_mse_improvement_pct:.3f}% | {candidate_psnr_minus_comparator_db_mean:.4f} | "
        "{candidate_ssim_minus_comparator_mean:.6f} |".format(**row)
        for row in wins
    )
    return f"""# K layer-selective 闭环逐帧诊断（development p0/4/9）

## 结论

正式 development gate 已冻结为 **`STOP_NO_HELDOUT_NO_VBENCH`**。本诊断没有
重新选择候选，也没有读取 prompt 1/5/8 或运行 VBench；它只解释为什么 A/B/C
没有超过双基线。

normalized-v2 三损失 gate：

| 候选 | vs RVQ+QV2 composite | vs uniform B2 composite |
|---|---:|---:|
{chr(10).join(gate_lines)}

逐帧 RGB pixel-MSE 的 prompt-macro 结果：

| 候选 | vs QV2 改善（prompt胜数） | vs uniform 改善（prompt胜数） | vs QV2 平均逐帧胜率 |
|---|---:|---:|---:|
{chr(10).join(overall_lines)}

这里的 pixel MSE 与正式 Similarity MSE 完全同口径，但不能替代 LPIPS，也不能
重新定义正式 composite。三个候选在 gate 中全部为负；逐帧表用于区分这种失败
是贯穿 rollout、集中在某一 phase，还是由少数尾部帧主导。

## 时间阶段

| 候选 | 对照 | early MSE改善 | mid MSE改善 | late MSE改善 |
|---|---|---:|---:|---:|
{chr(10).join(phase_lines)}

{chr(10).join(conclusion_lines)}

## 尾部风险

P95 是每个 prompt 的 frame-MSE 95 分位数后再作 prompt macro；CVaR95 是每个
prompt 中不小于其 P95 的帧 MSE 均值后再作 macro。

| 候选 | 对照 | P95改善 | CVaR95改善 | CVaR95 prompt胜数 |
|---|---|---:|---:|---:|
{chr(10).join(tail_lines)}

## Prompt 胜负

| 候选 | 对照 | prompt | MSE改善 | PSNR差(dB) | SSIM差 |
|---|---|---:|---:|---:|---:|
{prompt_lines}

## 口径与限制

- 解码 RGB `[0,1]`；跳过 decoded frame 0；每条视频比较 172/173 帧。
- chunk 映射与先前 tensor diagnostics 完全一致：chunk 0 对应 frame 0–12，
  以后每个 chunk 对应 16 个 decoded frames；early/mid/late 为 chunks
  `0–3 / 4–7 / 8–10`。
- MSE 为 RGB 全像素均值；PSNR 是逐帧 dB 后算术平均；SSIM 使用 11×11
  `avg_pool2d`，与正式 Similarity evaluator 相同。
- 全部视频都是 `CLI50, effective 4-step few-step`，不是真正 50-step。
- 本分析不计算 LPIPS、KV/attention error 或 VBench，也不作因果证明。
"""


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"create-only output exists: {output_dir}")
    audit = validate_inputs(manifest_path)
    manifest = audit["manifest"]
    methods = audit["methods"]

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    started = time.perf_counter()
    frame_rows = []
    video_artifacts = []
    for prompt in manifest["prompts"]:
        prompt_id = int(prompt["prompt_id"])
        reference_path = Path(prompt["reference"]["path"]).resolve()
        for method_id in (*CANDIDATES, *COMPARATORS):
            test_path = Path(prompt["tests"][method_id]["path"]).resolve()
            print(
                f"[compare] p{prompt_id} {method_id} on {device}", flush=True
            )
            rows, metadata = base.compare_pair(
                prompt_id=prompt_id,
                prompt=prompt["prompt"],
                method_id=method_id,
                method_label=methods[method_id]["label"],
                reference_path=reference_path,
                test_path=test_path,
                protocol=manifest["protocol"],
                device=device,
            )
            frame_rows.extend(rows)
            video_artifacts.append(
                {
                    "prompt_id": prompt_id,
                    "method_id": method_id,
                    "path": str(test_path),
                    "sha256": sha256_file(test_path),
                    **metadata,
                }
            )
            print(
                f"[done] p{prompt_id} {method_id}: {metadata['evaluation_seconds']:.2f}s",
                flush=True,
            )

    method_summary = base.method_summaries(frame_rows)
    chunk_summary = base.grouped_summaries(frame_rows, "generation_chunk")
    phase_summary = base.grouped_summaries(frame_rows, "phase")
    base.add_error_mass_fraction(chunk_summary, method_summary)
    base.add_error_mass_fraction(phase_summary, method_summary)
    pair_frames, pair_segments, pair_tails, worst_sets = pairwise_all(frame_rows)
    macro_segment_rows = macro_segments(pair_segments)
    macro_tail_rows = macro_tails(pair_tails)
    prompt_win_rows = prompt_wins(pair_segments)
    validations = base.validate_archived_metrics(
        method_summary, manifest["archived_metric_checks"], set(PROMPTS)
    )
    if len(validations) != 15 or not all(row["passed"] for row in validations):
        raise ValueError("archived metric validation grid failed")

    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "per_frame_metrics.csv": frame_rows,
        "method_summary.csv": method_summary,
        "generation_chunk_summary.csv": chunk_summary,
        "phase_summary.csv": phase_summary,
        "candidate_pairwise_per_frame.csv": pair_frames,
        "candidate_pairwise_segments.csv": pair_segments,
        "candidate_pairwise_tails.csv": pair_tails,
        "candidate_pairwise_macro_segments.csv": macro_segment_rows,
        "candidate_pairwise_macro_tails.csv": macro_tail_rows,
        "prompt_win_summary.csv": prompt_win_rows,
    }
    for name, rows in paths.items():
        _write_csv(output_dir / name, rows)

    config = {
        "schema_version": (
            "k_layer_selective_closed_loop_frame_diagnostics.config.v1"
        ),
        "run_label": args.run_label,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "device_requested": args.device,
        "device_used": str(device),
        "prompt_ids": list(PROMPTS),
        "candidate_methods": list(CANDIDATES),
        "comparator_methods": list(COMPARATORS),
        "protocol": manifest["protocol"],
        "metric_implementation": {
            "decoded_color_space": "RGB",
            "decoded_range": [0.0, 1.0],
            "frame_0_included": False,
            "mse": "torch F.mse_loss reduction=mean over RGB pixels",
            "psnr": "10*log10(1/max(frame_MSE,1e-12)); arithmetic mean of per-frame dB",
            "ssim": "torch avg_pool2d 11x11 RGB; same as compare_video_quality.py",
            "p95": "numpy linear empirical frame-MSE quantile within each prompt/method",
            "cvar95": "mean of frame MSE >= the within-prompt p95",
            "prompt_macro": "arithmetic mean over the three prompt-level statistics",
        },
    }
    config_path = output_dir / "experiment_config.json"
    base.write_json_create_only(config_path, config)

    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "PASS",
        "formal_gate_decision": audit["formal_gate_summary"]["decision"],
        "formal_gate_winner": audit["formal_gate_summary"]["winner"],
        "prompt_ids": list(PROMPTS),
        "candidate_methods": list(CANDIDATES),
        "comparator_methods": list(COMPARATORS),
        "protocol": manifest["protocol"],
        "archived_metric_validations": validations,
        "video_artifacts": video_artifacts,
        "candidate_pairwise_macro_overall": [
            row for row in macro_segment_rows if row["segment_type"] == "overall"
        ],
        "candidate_pairwise_macro_phases": [
            row for row in macro_segment_rows if row["segment_type"] == "phase"
        ],
        "candidate_pairwise_macro_tails": macro_tail_rows,
        "prompt_wins": prompt_win_rows,
        "worst_frame_sets": worst_sets,
        "row_counts": {name: len(rows) for name, rows in paths.items()},
        "evaluation_seconds": time.perf_counter() - started,
        "scope": {
            "new_video_generation": False,
            "heldout_prompts_read": False,
            "vbench_run": False,
            "lpips_recomputed": False,
        },
        "limitations": [
            "Decoded-video diagnosis does not identify a unique KV or attention cause.",
            "Formal advancement remains governed by the frozen normalized-v2 three-loss composite, not pixel MSE alone.",
            "All runs are CLI50 effective-four few-step, not true 50-step.",
        ],
    }
    summary_path = output_dir / "summary.json"
    base.write_json_create_only(summary_path, summary)

    readme_path = output_dir / "README_zh.md"
    with readme_path.open("x", encoding="utf-8") as handle:
        handle.write(
            _readme(
                gate_summary=audit["formal_gate_summary"],
                macro=macro_segment_rows,
                tails=macro_tail_rows,
                wins=prompt_win_rows,
            )
        )

    output_names = [
        *paths,
        "experiment_config.json",
        "summary.json",
        "README_zh.md",
    ]
    evidence = {
        "schema_version": (
            "k_layer_selective_closed_loop_frame_diagnostics.evidence_manifest.v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "formal_gate": manifest["formal_gate"],
        "candidate_runtime_and_video_inputs": audit["input_artifacts"],
        "source_analysis": {
            "path": str(SCRIPT),
            "sha256": sha256_file(SCRIPT),
            "reused_base_script": str(BASE_SCRIPT),
            "reused_base_script_sha256": sha256_file(BASE_SCRIPT),
        },
        "output_artifacts": {
            name: {
                "path": str((output_dir / name).resolve()),
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
            }
            for name in output_names
        },
        "accounting": {
            "MiB": "2**20 bytes (inherited codec accounting; no payload recomputation)",
            "frame_mapping": manifest["protocol"]["decoded_chunk_mapping"],
            "frame_0_excluded": True,
            "metrics": "RGB MSE, per-frame PSNR dB, 11x11 RGB SSIM",
            "new_video_generation": False,
            "heldout_read": False,
            "vbench_run": False,
        },
    }
    evidence_path = output_dir / "evidence_manifest.json"
    base.write_json_create_only(evidence_path, evidence)

    run_info = {
        "schema_version": RUN_INFO_SCHEMA,
        "git_commit": git_commit(),
        "config_path": str(config_path.resolve()),
        "config_hash": sha256_file(config_path),
        "codebook_path": manifest["codebook_path"],
        "codebook_hash": manifest["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "seed": 42,
        "prompt_list": [
            {"prompt_id": item["prompt_id"], "prompt": item["prompt"]}
            for item in manifest["prompts"]
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_label": args.run_label,
        "source_script": str(SCRIPT),
        "source_script_sha256": sha256_file(SCRIPT),
        "reused_base_script": str(BASE_SCRIPT),
        "reused_base_script_sha256": sha256_file(BASE_SCRIPT),
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "evidence_manifest_path": str(evidence_path.resolve()),
        "evidence_manifest_sha256": sha256_file(evidence_path),
        "output_artifacts": {
            name: sha256_file(output_dir / name) for name in output_names
        },
        "new_video_generation": False,
        "heldout_prompts_read": False,
        "vbench_run": False,
        "lpips_recomputed": False,
    }
    base.write_json_create_only(output_dir / "run_info.json", run_info)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(output_dir),
                "frame_rows": len(frame_rows),
                "pairwise_frame_rows": len(pair_frames),
                "evaluation_seconds": summary["evaluation_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
