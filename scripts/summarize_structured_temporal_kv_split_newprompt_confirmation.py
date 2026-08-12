#!/usr/bin/env python3
"""Build the auditable Chinese summary for the fixed new-prompt confirmation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import socket
import subprocess
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = (
    REPO
    / "results/analysis/"
    "structured_temporal_kv_split_newprompt_confirmation_v1_20260723"
)
SOURCE_CAPTURE_ROOT = (
    REPO
    / "results/analysis/"
    "wan5b_rvq_mechanisms_p1_posthoc_map_confirmation_captures"
)
PREREGISTRATION = (
    REPO
    / "experiments/"
    "structured_temporal_kv_split_newprompt_confirmation_v1_20260723_"
    "preregistration.json"
)
BASE_SCRIPT = REPO / "scripts/screen_structured_temporal_prefix_exact_attention.py"
WRAPPER_SCRIPT = (
    REPO / "scripts/screen_structured_temporal_kv_split_newprompt_confirmation.py"
)

QV = "rvq_plus_quantvideo_spanwise_int2"
UNIFORM = "rvq_plus_structured_uniform_b2_spanwise"
PRIMARY = "rvq_plus_structured_K_uniform_V_41113"
SECONDARY = "rvq_plus_structured_K_uniform_V_31114"
METHODS = (QV, UNIFORM, PRIMARY, SECONDARY)
LABELS = {
    QV: "RVQ+QuantVideo INT2（span-wise）",
    UNIFORM: "Structured uniform K/V=22222",
    PRIMARY: "K=22222，V=41113（primary）",
    SECONDARY: "K=22222，V=31114（secondary）",
}
METRICS = (
    "historical_K_mse",
    "historical_V_mse",
    "attention_logit_mse",
    "attention_kl_reference_to_candidate",
    "attention_output_mse",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = sorted({str(key) for row in rows for key in row})
    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def ensure_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace different output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def empirical_cvar(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    count = max(1, int(math.ceil((1.0 - quantile) * len(array))))
    return float(np.partition(array, len(array) - count)[-count:].mean())


def selected_rows(
    rows: Sequence[Mapping[str, str]],
    method: str,
    half: str = "both",
) -> list[Mapping[str, str]]:
    return [
        row
        for row in rows
        if (
            row["method"] == method
            and row["intervention"] == "K_and_V"
            and (half == "both" or row["attention_half"] == half)
        )
    ]


def metric_mean(
    rows: Sequence[Mapping[str, str]], method: str, metric: str, half: str = "both"
) -> float:
    values = [float(row[metric]) for row in selected_rows(rows, method, half)]
    return float(np.mean(values))


def build_method_summary(
    raw: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    method_configs = {
        str(item["method"]): item for item in config["methods"]
    }
    baseline_output = metric_mean(raw, UNIFORM, "attention_output_mse")
    qv_output = metric_mean(raw, QV, "attention_output_mse")
    output = []
    for method in METHODS:
        rows = selected_rows(raw, method)
        by_task = {}
        for row in rows:
            if row["attention_half"] == "standard_rope" and int(row["head"]) == 0:
                by_task[int(row["task_index"])] = row
        total_encode = [
            float(row["total_encode_K_seconds"])
            + float(row["total_encode_V_seconds"])
            for row in by_task.values()
        ]
        total_decode = [
            float(row["total_decode_K_seconds"])
            + float(row["total_decode_V_seconds"])
            for row in by_task.values()
        ]
        item: dict[str, Any] = {
            "method": method,
            "label_zh": LABELS[method],
            "schedule_K": json.dumps(
                method_configs[method]["schedule_K"], separators=(",", ":")
            ),
            "schedule_V": json.dumps(
                method_configs[method]["schedule_V"], separators=(",", ":")
            ),
            "prompts": len({int(row["prompt_id"]) for row in rows}),
            "layers": len({int(row["layer"]) for row in rows}),
            "raw_head_half_units": len(rows),
            "median_encode_seconds_per_prompt_layer": float(
                np.median(total_encode)
            ),
            "median_decode_seconds_per_prompt_layer": float(
                np.median(total_decode)
            ),
            "K_plus_V_payload_MiB_per_layer": (
                float(rows[0]["total_payload_K_MiB"])
                + float(rows[0]["total_payload_V_MiB"])
            ),
            "joint_compression_ratio_vs_BF16": float(
                rows[0]["joint_compression_ratio_K_vs_BF16"]
            ),
            "full_30_layer_joint_MiB_excluding_shared_codebooks": (
                30
                * (
                    float(rows[0]["total_payload_K_MiB"])
                    + float(rows[0]["total_payload_V_MiB"])
                )
            ),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_p99"] = float(np.quantile(values, 0.99))
            item[f"{metric}_p999"] = float(np.quantile(values, 0.999))
            item[f"{metric}_cvar99"] = empirical_cvar(values, 0.99)
        output_mean = item["attention_output_mse_mean"]
        item["output_improvement_vs_uniform_pct"] = (
            100.0 * (baseline_output - output_mean) / baseline_output
        )
        item["output_improvement_vs_quantvideo_pct"] = (
            100.0 * (qv_output - output_mean) / qv_output
        )
        output.append(item)
    return output


def prompt_bootstrap(
    raw: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_rows = []
    summaries = []
    for candidate in (PRIMARY, SECONDARY):
        for half in ("standard_rope", "camera_prope", "both"):
            paired = []
            for prompt_id in range(4):
                baseline_values = [
                    float(row["attention_output_mse"])
                    for row in selected_rows(raw, UNIFORM, half)
                    if int(row["prompt_id"]) == prompt_id
                ]
                candidate_values = [
                    float(row["attention_output_mse"])
                    for row in selected_rows(raw, candidate, half)
                    if int(row["prompt_id"]) == prompt_id
                ]
                baseline = float(np.mean(baseline_values))
                value = float(np.mean(candidate_values))
                improvement = 100.0 * (baseline - value) / baseline
                paired.append((baseline, value))
                prompt_rows.append(
                    {
                        "candidate": candidate,
                        "attention_half": half,
                        "prompt_id": prompt_id,
                        "baseline_output_mse": baseline,
                        "candidate_output_mse": value,
                        "relative_improvement_pct": improvement,
                        "win": value < baseline,
                    }
                )
            array = np.asarray(paired, dtype=np.float64)
            rng = np.random.default_rng(
                20260723
                + (0 if candidate == PRIMARY else 100)
                + {"standard_rope": 0, "camera_prope": 10, "both": 20}[half]
            )
            indices = rng.integers(0, len(array), size=(10_000, len(array)))
            baseline_boot = array[indices, 0].mean(axis=1)
            candidate_boot = array[indices, 1].mean(axis=1)
            improvement_boot = (
                100.0
                * (baseline_boot - candidate_boot)
                / np.maximum(baseline_boot, 1e-30)
            )
            overall = (
                100.0
                * (float(array[:, 0].mean()) - float(array[:, 1].mean()))
                / float(array[:, 0].mean())
            )
            summaries.append(
                {
                    "candidate": candidate,
                    "attention_half": half,
                    "prompts": len(array),
                    "prompt_wins": int(np.sum(array[:, 1] < array[:, 0])),
                    "relative_improvement_pct": overall,
                    "prompt_bootstrap_95ci_low_pct": float(
                        np.quantile(improvement_boot, 0.025)
                    ),
                    "prompt_bootstrap_95ci_high_pct": float(
                        np.quantile(improvement_boot, 0.975)
                    ),
                    "bootstrap_samples": 10_000,
                    "bootstrap_seed_policy": (
                        "20260723 + candidate_offset + half_offset"
                    ),
                }
            )
    return prompt_rows, summaries


def span_tradeoff(
    span_rows: Sequence[Mapping[str, str]],
    age_comparisons: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    output = []
    for candidate in (PRIMARY, SECONDARY):
        for span_index in range(5):
            baseline = [
                float(row["mse"])
                for row in span_rows
                if (
                    row["method"] == UNIFORM
                    and row["tensor"] == "V"
                    and int(row["span_index"]) == span_index
                )
            ]
            values = [
                float(row["mse"])
                for row in span_rows
                if (
                    row["method"] == candidate
                    and row["tensor"] == "V"
                    and int(row["span_index"]) == span_index
                )
            ]
            age = next(
                row
                for row in age_comparisons
                if (
                    row["candidate"] == candidate
                    and row["baseline"] == UNIFORM
                    and row["attention_half"] == "both"
                    and row["intervention"] == "K_and_V"
                    and row["metric"] == "attention_output_contribution_mse"
                    and int(row["span_index"]) == span_index
                )
            )
            base_mean = float(np.mean(baseline))
            candidate_mean = float(np.mean(values))
            output.append(
                {
                    "candidate": candidate,
                    "span_index": span_index,
                    "age_order": "0=oldest_anchor,4=newest_terminal",
                    "suffix_depth": int(
                        next(
                            row["suffix_depth"]
                            for row in span_rows
                            if (
                                row["method"] == candidate
                                and row["tensor"] == "V"
                                and int(row["span_index"]) == span_index
                            )
                        )
                    ),
                    "uniform_V_mse": base_mean,
                    "candidate_V_mse": candidate_mean,
                    "V_mse_improvement_pct": (
                        100.0 * (base_mean - candidate_mean) / base_mean
                    ),
                    "output_contribution_improvement_pct": float(
                        age["relative_improvement_pct"]
                    ),
                    "output_contribution_p999_ratio": float(age["p999_ratio"]),
                    "output_contribution_cvar99_ratio": float(
                        age["cvar99_ratio"]
                    ),
                }
            )
    return output


def compatibility_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for task in config["captures"]:
        validation = (
            ROOT
            / "units"
            / f"prompt_{int(task['prompt_id']):02d}"
            / f"layer_{int(task['layer']):02d}"
            / "validation.json"
        )
        value = json.loads(validation.read_text(encoding="utf-8"))
        output.append(
            {
                "prompt_id": int(task["prompt_id"]),
                "layer": int(task["layer"]),
                "capture_sha256": task["capture_sha256"],
                "capture_hash_verified": value["capture_hash_verified"],
                "history_tokens": value["capture_in_call_history_tokens"],
                "current_tokens": value["current_tokens"],
                "terminal_tokens": value["terminal_tokens_reconstructed"],
                "span_ranges": json.dumps(
                    value["terminal_span_ranges"], separators=(",", ":")
                ),
                "terminal_is_5x7040": (
                    value["terminal_span_ranges"]
                    == [
                        [0, 7040],
                        [7040, 14080],
                        [14080, 21120],
                        [21120, 28160],
                        [28160, 35200],
                    ]
                ),
                "exact_transform_replay": value["exact_transform_replay"],
                "metric_audit_passed": value["metric_audit_passed"],
                "all_payloads_byte_equal": value[
                    "all_residual_payloads_byte_equal"
                ],
                "terminal_post_commit_counterfactual": value[
                    "terminal_post_commit_counterfactual"
                ],
            }
        )
    return output


def table_markdown(
    summaries: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "| 方法 | K/V schedule | K MSE | V MSE | QK MSE | KL | Output MSE | Output vs uniform | Output vs QV2 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {label} | {k}/{v} | {km:.6g} | {vm:.6g} | {lm:.6g} | "
            "{kl:.6g} | {om:.6g} | {iu:+.2f}% | {iq:+.2f}% |".format(
                label=row["label_zh"],
                k=row["schedule_K"],
                v=row["schedule_V"],
                km=row["historical_K_mse_mean"],
                vm=row["historical_V_mse_mean"],
                lm=row["attention_logit_mse_mean"],
                kl=row["attention_kl_reference_to_candidate_mean"],
                om=row["attention_output_mse_mean"],
                iu=row["output_improvement_vs_uniform_pct"],
                iq=row["output_improvement_vs_quantvideo_pct"],
            )
        )
    return "\n".join(lines)


def build_readme(
    summaries: Sequence[Mapping[str, Any]],
    bootstrap: Sequence[Mapping[str, Any]],
    compatibility: Sequence[Mapping[str, Any]],
) -> str:
    by_key = {
        (row["candidate"], row["attention_half"]): row for row in bootstrap
    }
    p_all = by_key[(PRIMARY, "both")]
    s_all = by_key[(SECONDARY, "both")]
    p_std = by_key[(PRIMARY, "standard_rope")]
    p_cam = by_key[(PRIMARY, "camera_prope")]
    s_std = by_key[(SECONDARY, "standard_rope")]
    s_cam = by_key[(SECONDARY, "camera_prope")]
    return f"""# K/V 分离时间前缀：4-prompt 固定假设复核

## 结论

固定假设在这组 4 prompts × 5 layers 的 exact-attention replay 上成立：

- primary `K=22222, V=41113` 相对 uniform B2 的 joint output MSE 改善
  **{p_all['relative_improvement_pct']:.2f}%**，prompt-bootstrap 95% CI
  **[{p_all['prompt_bootstrap_95ci_low_pct']:.2f}%,
  {p_all['prompt_bootstrap_95ci_high_pct']:.2f}%]**，4/4 prompts 获胜；
- 预先冻结的 secondary `K=22222, V=31114` 改善
  **{s_all['relative_improvement_pct']:.2f}%**，95% CI
  **[{s_all['prompt_bootstrap_95ci_low_pct']:.2f}%,
  {s_all['prompt_bootstrap_95ci_high_pct']:.2f}%]**，4/4 prompts 获胜；
- 两个候选都通过冻结 gate 的全部 mean、P99、P999、CVaR 和 newest-span
  约束；物理 payload 与 uniform B2、span-wise QuantVideo INT2
  **逐字节相同**。

这不是 decoded-video / VBench 结论。下一步只能晋级到独立 prompt/seed 的短
closed-loop；通过后再跑视频 Similarity 与 VBench。

## 主表

口径：两种 attention half 合并、`K_and_V` intervention，4 prompts ×
5 layers × 24 heads × 2 halves = 960 raw units/method。MSE 越低越好。

{table_markdown(summaries)}

额外分支检查：

| 候选 | standard 改善 | camera 改善 | joint 改善 |
|---|---:|---:|---:|
| V=41113 | {p_std['relative_improvement_pct']:.2f}% | {p_cam['relative_improvement_pct']:.2f}% | {p_all['relative_improvement_pct']:.2f}% |
| V=31114 | {s_std['relative_improvement_pct']:.2f}% | {s_cam['relative_improvement_pct']:.2f}% | {s_all['relative_improvement_pct']:.2f}% |

## 为什么 V MSE 更差，attention output 反而更好

这里给出了一个干净的反例。`V=41113/31114` 的全局 V MSE 相对 uniform
分别恶化约 72.52%/70.41%，但 output MSE 却改善 58.07%/75.10%。
原因不是数值矛盾，而是目标函数不同：

`global V MSE` 对全部 token/channel 等权；attention output 则是
`A·V`，误差由 attention mass、query、位置变换和跨 token 误差相关性加权。
把 plane 从三个低 utility 中间 span 移到 oldest anchor 和 newest span，
虽然牺牲了大量等权元素，却显著降低了真正进入输出的误差。`span_tradeoff.csv`
显示中间 span 的 contribution 确实恶化，但首尾收益更大。

K 对两个候选始终是 uniform 22222，所以 K reconstruction、QK/logit、KL
和 attention-age shift 都与 uniform **bit-identical**。这把收益因果地定位到
V 的时间预算分配，而不是 K 或随机 seed。

## Payload

- 每个 tensor/layer：RVQ 6.4453125 MiB + residual 35.6982422 MiB；
- K+V/layer：84.2871094 MiB；
- 相对同一份 BF16 K/V：4.89399×；
- 外推 30 层：2528.613281 MiB，排除共享 RVQ/固定 Gaussian codebooks；
- 每个候选 K、V 均各 10 个 suffix-plane units；所有 20 units 的 validation
  都确认 payload byte-identical。

## Capture 兼容性与统计独立性

实际 capture 为 BF16 `[1,24,35200,128]`，metadata 为 history=31680、
current=3520；terminal full-key 轴正好切成
`[0:7040]...[28160:35200]` 五个物理 span。共
{len(compatibility)}/20 units 通过 capture hash、exact positional replay、
metric audit 和 payload audit。

必须严格说明：这四个 prompts **没有参与当前 temporal schedule 的形成**，
所以是 novel-to-this-hypothesis；但它们在 2026-07-20 已用于另一项
P1-F1 post-hoc RVQ layer-map exact-attention 确认，因此不是全局
never-seen。`preregistration.json` 在本轮任何 temporal codec 指标产生前冻结；
primary、secondary 和“不得再选第三候选”均已写死。

## Scope 限制

- replay 重建全部 35200 terminal tokens，包括原 call 的 3520 current segment；
  这是 terminal post-commit counterfactual，不是在线 runtime 已经压缩 current
  segment 的声明；
- 只覆盖层 0/1/23/25/29 的单个 late call；
- exact SDPA-core standard RoPE/camera PRoPE，不等同完整 30 层 closed-loop；
- 4 prompts 的 bootstrap CI 只衡量 prompt 采样，不消除它们的历史使用。

## 可复核文件

- `confirmation_method_summary.csv`：主表全部数值、tails、payload、观测 runtime；
- `prompt_bootstrap.csv`、`prompt_level_output.csv`：prompt 独立统计；
- `span_tradeoff.csv`：各年龄 span 的 V MSE / output contribution；
- `capture_compatibility.csv`：20 个 unit 的兼容性与审计；
- `tables/`：原始 11520 行 metrics、P99/P999/CVaR、age-bin、payload 和 gate；
- `evidence_manifest.json`：输入、代码、表格 SHA256。

注意：自动生成的 `README.md` 是共享 screen 的旧 boilerplate，其中“10 prompts、
0/10/20/29 layers”不适用于本 wrapper；本文件和 `config.json` 才是权威口径。
"""


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    raw = read_csv(ROOT / "tables/exact_attention_metrics.csv")
    spans = read_csv(ROOT / "tables/span_metrics.csv")
    age = read_csv(ROOT / "tables/age_bin_paired_comparisons.csv")
    gate = json.loads(
        (ROOT / "tables/promotion_gate.json").read_text(encoding="utf-8")
    )
    if not gate["preregistered"] or not gate["any_pass"]:
        raise RuntimeError("Expected a completed, passing preregistered gate")
    summaries = build_method_summary(raw, config)
    prompt_rows, bootstrap = prompt_bootstrap(raw)
    span_rows = span_tradeoff(spans, age)
    compatibility = compatibility_rows(config)
    if len(compatibility) != 20 or not all(
        row["capture_hash_verified"]
        and row["terminal_is_5x7040"]
        and row["exact_transform_replay"]
        and row["metric_audit_passed"]
        and row["all_payloads_byte_equal"]
        for row in compatibility
    ):
        raise RuntimeError("Capture compatibility audit failed")

    generated = {
        ROOT / "confirmation_method_summary.csv": csv_bytes(summaries),
        ROOT / "prompt_level_output.csv": csv_bytes(prompt_rows),
        ROOT / "prompt_bootstrap.csv": csv_bytes(bootstrap),
        ROOT / "span_tradeoff.csv": csv_bytes(span_rows),
        ROOT / "capture_compatibility.csv": csv_bytes(compatibility),
        ROOT / "README_zh.md": (
            build_readme(summaries, bootstrap, compatibility) + "\n"
        ).encode("utf-8"),
    }
    for path, payload in generated.items():
        ensure_identical(path, payload)

    input_paths = [
        PREREGISTRATION,
        SOURCE_CAPTURE_ROOT / "capture_manifest.json",
        SOURCE_CAPTURE_ROOT / "run_info.json",
        ROOT / "config.json",
        ROOT / "README.md",
        ROOT / "run_info/aggregate.json",
        ROOT / "run_info/worker_shard_000_of_001.json",
        ROOT / "tables/exact_attention_metrics.csv",
        ROOT / "tables/span_metrics.csv",
        ROOT / "tables/age_bin_metrics.csv",
        ROOT / "tables/tail_cvar_summary.csv",
        ROOT / "tables/age_bin_tail_cvar_summary.csv",
        ROOT / "tables/paired_comparisons.csv",
        ROOT / "tables/age_bin_paired_comparisons.csv",
        ROOT / "tables/full_model_payload_audit.csv",
        ROOT / "tables/promotion_gate.json",
        BASE_SCRIPT,
        WRAPPER_SCRIPT,
        Path(__file__).resolve(),
        *generated.keys(),
    ]
    evidence = {
        "schema_version": (
            "structured_temporal_kv_split_newprompt_confirmation.evidence.v1"
        ),
        "config_hash": sha256_file(ROOT / "config.json"),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip(),
        "files": {str(path): sha256_file(path) for path in input_paths},
        "capture_files": [
            {
                "path": task["capture_path"],
                "sha256": task["capture_sha256"],
                "verified_during_worker": True,
            }
            for task in config["captures"]
        ],
        "validation": {
            "units": len(compatibility),
            "all_capture_hashes_verified": True,
            "all_terminal_axes_5x7040": True,
            "all_exact_transform_replays_passed": True,
            "all_metric_audits_passed": True,
            "all_payloads_byte_equal": True,
            "gate_preregistered": gate["preregistered"],
            "gate_any_pass": gate["any_pass"],
            "globally_blind_prompts": False,
            "novel_to_temporal_prefix_hypothesis": True,
        },
    }
    evidence_path = ROOT / "evidence_manifest.json"
    ensure_identical(evidence_path, json_bytes(evidence))

    aggregate_info = json.loads(
        (ROOT / "run_info/aggregate.json").read_text(encoding="utf-8")
    )
    summary_info = {
        "schema_version": (
            "structured_temporal_kv_split_newprompt_confirmation.summary_run.v1"
        ),
        "git_commit": aggregate_info["git_commit"],
        "config_path": aggregate_info["config_path"],
        "config_hash": aggregate_info["config_hash"],
        "codebook_path": aggregate_info["codebook_path"],
        "codebook_hash": aggregate_info["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": "NVIDIA RTX 6000 Ada Generation (physical GPU 6)",
        "seed": aggregate_info["seed"],
        "prompt_list": aggregate_info["prompt_list"],
        "timestamp": aggregate_info["timestamp"],
        "status": "complete",
        "units": len(compatibility),
        "methods": list(METHODS),
        "evidence_manifest": str(evidence_path),
        "evidence_manifest_hash": sha256_file(evidence_path),
    }
    ensure_identical(ROOT / "run_info/summary.json", json_bytes(summary_info))
    print(
        json.dumps(
            {
                "output": str(ROOT),
                "units": len(compatibility),
                "primary_joint_improvement_pct": next(
                    row["relative_improvement_pct"]
                    for row in bootstrap
                    if row["candidate"] == PRIMARY
                    and row["attention_half"] == "both"
                ),
                "secondary_joint_improvement_pct": next(
                    row["relative_improvement_pct"]
                    for row in bootstrap
                    if row["candidate"] == SECONDARY
                    and row["attention_half"] == "both"
                ),
                "gate_any_pass": gate["any_pass"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
