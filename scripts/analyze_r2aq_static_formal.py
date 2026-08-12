#!/usr/bin/env python3
"""Build a reproducible, CPU-only analysis of the formal R²AQ static run.

The static benchmark is an offline, sampled, pre-RoPE attention proxy.  This
script deliberately does not reinterpret it as exact attention replay or
closed-loop video quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUN_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "r2aq_static_formal_train20_20260724_root_v1"
)
DEFAULT_PARAMS_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "r2aq_shared_params_20260724_root_v1"
)
DEFAULT_OUTPUT = Path(
    "results/analysis/r2aq_v1_static_formal_20260724"
)
SEED = 20260724
BOOTSTRAP_SAMPLES = 10_000
MIB = 2**20

LOWER_IS_BETTER = {
    "K_MSE",
    "V_MSE",
    "QK_logits_MSE",
    "attention_KL",
    "attention_probability_MSE",
    "QKV_output_MSE",
    "QKV_output_squared_error_P99",
    "QKV_output_squared_error_P999",
    "QKV_output_squared_error_CVaR99",
}
HIGHER_IS_BETTER = {"attention_top1_agreement"}
PAIR_METRICS = sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER)

QUALITY_COLUMNS = [
    "method",
    "ablation",
    "bits",
    "prompts",
    "units",
    "K_MSE",
    "V_MSE",
    "QK_logits_MSE",
    "attention_KL",
    "attention_probability_MSE",
    "attention_top1_agreement",
    "QKV_output_MSE",
    "QKV_output_squared_error_P99",
    "QKV_output_squared_error_P999",
    "QKV_output_squared_error_CVaR99",
    "attention_scope",
]

RUNTIME_COLUMNS = [
    "method",
    "ablation",
    "bits",
    "encode_ms_K",
    "encode_ms_V",
    "decode_ms_K",
    "decode_ms_V",
    "encode_decode_ms_KV",
]

PAYLOAD_COMPONENTS = [
    "centroid_payload_bytes",
    "label_payload_bytes",
    "scale_payload_bytes",
    "plane_payload_bytes",
    "selector_payload_bytes",
    "tail_value_payload_bytes",
    "tail_scale_payload_bytes",
    "tail_id_payload_bytes",
    "other_payload_bytes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def read_rows(run_root: Path) -> pd.DataFrame:
    paths = sorted((run_root / "rows").glob("shard_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no shard rows under {run_root / 'rows'}")
    rows = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    key = ["method", "prompt_id", "layer", "head"]
    if rows.duplicated(key).any():
        raise RuntimeError("duplicate method/prompt/layer/head rows")
    counts = rows.groupby("method").size()
    if counts.nunique() != 1:
        raise RuntimeError(f"method coverage differs: {counts.to_dict()}")
    return rows


def method_ablation(method: str) -> str:
    if method == "l8c64_int4_base":
        return "rvq_base"
    if method.startswith("rvq_qv_"):
        return "quantvideo_reference"
    if method.startswith("r2aq_"):
        body = method[len("r2aq_") :]
        return body.rsplit("_b", 1)[0]
    return "unknown"


def bootstrap_interval(
    values: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bootstrap expects a non-empty vector")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    estimates = values[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def build_quality(prompt: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, group in prompt.groupby("method", sort=True):
        record: dict[str, Any] = {
            "method": method,
            "ablation": method_ablation(method),
            "bits": int(group["bits"].iloc[0]),
            "prompts": int(group["prompt_id"].nunique()),
            "units": int(group["units"].sum()),
            "attention_scope": (
                "offline_sampled_pre_rope_attention_without_positional_replay"
            ),
        }
        for metric in PAIR_METRICS:
            record[metric] = float(group[metric].mean())
        records.append(record)
    return pd.DataFrame(records)[QUALITY_COLUMNS]


def build_paired(
    prompt: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    summaries: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    for bits in (2, 4):
        baseline_method = f"rvq_qv_b{bits}"
        baseline = (
            prompt[prompt["method"] == baseline_method]
            .set_index("prompt_id")
            .sort_index()
        )
        candidates = sorted(
            method
            for method in prompt["method"].unique()
            if method.startswith("r2aq_") and method.endswith(f"_b{bits}")
        )
        for method in candidates:
            candidate = (
                prompt[prompt["method"] == method]
                .set_index("prompt_id")
                .sort_index()
            )
            if not candidate.index.equals(baseline.index):
                raise RuntimeError(
                    f"paired prompt coverage differs: {method} vs {baseline_method}"
                )
            for metric in PAIR_METRICS:
                base = baseline[metric].to_numpy(dtype=np.float64)
                cand = candidate[metric].to_numpy(dtype=np.float64)
                if np.any(base == 0):
                    raise RuntimeError(
                        f"zero baseline prevents relative comparison: "
                        f"{baseline_method}/{metric}"
                    )
                if metric in LOWER_IS_BETTER:
                    relative = (base - cand) / base
                    absolute = base - cand
                    wins = cand < base
                else:
                    relative = (cand - base) / base
                    absolute = cand - base
                    wins = cand > base
                ci_low, ci_high = bootstrap_interval(
                    relative, rng, BOOTSTRAP_SAMPLES
                )
                summaries.append(
                    {
                        "method": method,
                        "ablation": method_ablation(method),
                        "bits": bits,
                        "baseline": baseline_method,
                        "metric": metric,
                        "direction": (
                            "lower_is_better"
                            if metric in LOWER_IS_BETTER
                            else "higher_is_better"
                        ),
                        "prompts": int(relative.size),
                        "baseline_prompt_macro_mean": float(base.mean()),
                        "candidate_prompt_macro_mean": float(cand.mean()),
                        "paired_absolute_improvement_mean": float(
                            absolute.mean()
                        ),
                        "paired_relative_improvement_mean": float(
                            relative.mean()
                        ),
                        "paired_relative_improvement_CI_low": ci_low,
                        "paired_relative_improvement_CI_high": ci_high,
                        "prompt_win_rate": float(wins.mean()),
                        "bootstrap_samples": BOOTSTRAP_SAMPLES,
                        "bootstrap_seed": SEED,
                    }
                )
                for prompt_id, b, c, a, r, win in zip(
                    baseline.index,
                    base,
                    cand,
                    absolute,
                    relative,
                    wins,
                    strict=True,
                ):
                    prompt_records.append(
                        {
                            "method": method,
                            "ablation": method_ablation(method),
                            "bits": bits,
                            "baseline": baseline_method,
                            "metric": metric,
                            "prompt_id": int(prompt_id),
                            "baseline_value": float(b),
                            "candidate_value": float(c),
                            "absolute_improvement": float(a),
                            "relative_improvement": float(r),
                            "win": bool(win),
                        }
                    )
    return pd.DataFrame(summaries), pd.DataFrame(prompt_records)


def build_runtime(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, group in rows.groupby("method", sort=True):
        if method.startswith("r2aq_"):
            timed_mask = group[
                ["encode_ms_K", "encode_ms_V", "decode_ms_K", "decode_ms_V"]
            ].notna().all(axis=1)
        elif method.startswith("rvq_qv_"):
            timed_mask = group["encode_decode_ms_KV"].notna()
        else:
            timed_mask = pd.Series(False, index=group.index)
        record: dict[str, Any] = {
            "method": method,
            "ablation": method_ablation(method),
            "bits": int(group["bits"].iloc[0]),
            "timed_units": int(timed_mask.sum()),
        }
        for column in RUNTIME_COLUMNS[3:]:
            values = group[column].dropna()
            record[column] = (
                float(values.mean()) if not values.empty else np.nan
            )
        if method.startswith("r2aq_"):
            encode = group["encode_ms_K"] + group["encode_ms_V"]
            decode = group["decode_ms_K"] + group["decode_ms_V"]
            roundtrip = encode + decode
            record["timing_interval"] = (
                "R²AQ residual encode of full B4 K and V states; decode at "
                "reported prefix depth; CUDA synchronized around each call"
            )
        elif method.startswith("rvq_qv_"):
            encode = pd.Series(dtype=np.float64)
            decode = pd.Series(dtype=np.float64)
            roundtrip = group["encode_decode_ms_KV"].dropna()
            record["timing_interval"] = (
                "QuantVideo residual K+V encode and decode roundtrip; CUDA "
                "synchronized around the combined interval"
            )
        else:
            encode = pd.Series(dtype=np.float64)
            decode = pd.Series(dtype=np.float64)
            roundtrip = pd.Series(dtype=np.float64)
            record["timing_interval"] = "not timed"
        for name, values in (
            ("encode_total_ms_KV", encode),
            ("decode_total_ms_KV", decode),
            ("roundtrip_total_ms_KV", roundtrip),
        ):
            values = values.dropna()
            record[name] = float(values.mean()) if len(values) else np.nan
            record[f"{name}_median"] = (
                float(values.median()) if len(values) else np.nan
            )
            record[f"{name}_P95"] = (
                float(values.quantile(0.95)) if len(values) else np.nan
            )
            record[f"{name}_max"] = (
                float(values.max()) if len(values) else np.nan
            )
        records.append(record)
    columns = [
        "method",
        "ablation",
        "bits",
        "timed_units",
        "encode_ms_K",
        "encode_ms_V",
        "encode_total_ms_KV",
        "encode_total_ms_KV_median",
        "encode_total_ms_KV_P95",
        "encode_total_ms_KV_max",
        "decode_ms_K",
        "decode_ms_V",
        "decode_total_ms_KV",
        "decode_total_ms_KV_median",
        "decode_total_ms_KV_P95",
        "decode_total_ms_KV_max",
        "encode_decode_ms_KV",
        "roundtrip_total_ms_KV",
        "roundtrip_total_ms_KV_median",
        "roundtrip_total_ms_KV_P95",
        "roundtrip_total_ms_KV_max",
        "timing_interval",
    ]
    return pd.DataFrame(records)[columns]


def shared_tables(method: str, reported_bytes: int) -> tuple[str, int]:
    if reported_bytes == 0:
        return "", 0
    if reported_bytes == 248:
        return "r2aq_tree_K_v1;r2aq_tree_V_v1", 248
    if reported_bytes == 372:
        return (
            "r2aq_thresholds_K_v1;r2aq_tree_K_v1;"
            "r2aq_thresholds_V_v1;r2aq_tree_V_v1",
            372,
        )
    if reported_bytes == 1448:
        return (
            "r2aq_tree_K_v1;r2aq_tree_V_v1;"
            "r2aq_utility_layer_KV_age_channel_v1",
            1448,
        )
    if reported_bytes == 1572:
        return (
            "r2aq_thresholds_K_v1;r2aq_tree_K_v1;"
            "r2aq_thresholds_V_v1;r2aq_tree_V_v1;"
            "r2aq_utility_layer_KV_age_channel_v1",
            1572,
        )
    raise RuntimeError(
        f"unrecognized shared table accounting for {method}: {reported_bytes}"
    )


def build_payload(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, group in rows.groupby("method", sort=True):
        reported_values = group["shared_table_bytes"].astype(int).tolist()
        nonzero = [value for value in reported_values if value]
        if not nonzero:
            reported_global = 0
        elif len(set(reported_values)) == 1:
            # Backward-compatible reader for the earlier per-row report whose
            # analyzer deduplicated identical global tables post hoc.
            reported_global = nonzero[0]
        elif len(nonzero) == 1:
            # Strict-rate v2: one deterministic benchmark owner charges and
            # reserves the global table bytes; every other unit reports zero.
            reported_global = nonzero[0]
        else:
            raise RuntimeError(
                f"shared table ownership is not unique for {method}: "
                f"{reported_values}"
            )
        table_ids, shared_global = shared_tables(method, reported_global)
        source = int(group["source_BF16_bytes"].sum())
        residual_local = int(group["residual_payload_bytes"].sum())
        rvq_indices = int(group["rvq_index_payload_bytes"].sum())
        rvq_codebooks = int(group["rvq_codebook_payload_bytes"].sum())
        rvq = rvq_indices + rvq_codebooks
        residual_with_shared = residual_local + shared_global
        joint = rvq + residual_with_shared
        record: dict[str, Any] = {
            "method": method,
            "ablation": method_ablation(method),
            "bits": int(group["bits"].iloc[0]),
            "units": int(group.shape[0]),
            "source_BF16_bytes": source,
            "source_BF16_MiB": source / MIB,
            "residual_local_bytes": residual_local,
            "residual_local_MiB": residual_local / MIB,
            "shared_table_ids": table_ids,
            "shared_table_bytes_global_once": shared_global,
            "residual_including_shared_bytes": residual_with_shared,
            "residual_including_shared_MiB": residual_with_shared / MIB,
            "rvq_index_bytes": rvq_indices,
            "rvq_codebook_bytes": rvq_codebooks,
            "rvq_bytes": rvq,
            "joint_payload_bytes": joint,
            "joint_payload_MiB": joint / MIB,
            "joint_compression_ratio_vs_sampled_KV_BF16": source / joint,
            "shared_accounting": (
                "distinct table IDs charged once over the complete "
                "40-unit benchmark"
            ),
        }
        for component in PAYLOAD_COMPONENTS:
            record[component.replace("_payload", "_local")] = int(
                group[component].sum()
            )
        records.append(record)
    return pd.DataFrame(records)


def build_predictor_diagnostics(prompt: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "K_codec_bf16_predictor_suffix_energy",
        "K_codec_int4_predictor_suffix_energy",
        "K_codec_predictor_suffix_energy_relative_increase",
        "K_codec_centroid_quantization_mse",
        "V_codec_bf16_predictor_suffix_energy",
        "V_codec_int4_predictor_suffix_energy",
        "V_codec_predictor_suffix_energy_relative_increase",
        "V_codec_centroid_quantization_mse",
        "K_codec_realized_selection_fraction",
        "V_codec_realized_selection_fraction",
        "K_codec_reclaimed_bytes",
        "V_codec_reclaimed_bytes",
    ]
    records: list[dict[str, Any]] = []
    for method, group in prompt.groupby("method", sort=True):
        if not method.startswith("r2aq_"):
            continue
        record: dict[str, Any] = {
            "method": method,
            "ablation": method_ablation(method),
            "bits": int(group["bits"].iloc[0]),
        }
        for column in columns:
            record[column] = float(group[column].mean())
        records.append(record)
    return pd.DataFrame(records)


def markdown_table(
    frame: pd.DataFrame, columns: list[str], digits: int = 6
) -> str:
    view = frame[columns].copy()
    for column in columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else f"{float(value):.{digits}g}"
                )
            )
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body])


def build_readme(
    quality: pd.DataFrame,
    paired: pd.DataFrame,
    payload: pd.DataFrame,
    runtime: pd.DataFrame,
) -> str:
    selected = quality[
        quality["method"].isin(
            [
                "rvq_qv_b2",
                "r2aq_uniform_bonus_b2",
                "r2aq_scale_waterfill_b2",
                "r2aq_utility_waterfill_b2",
                "r2aq_full_tree_b2",
                "rvq_qv_b4",
                "r2aq_uniform_bonus_b4",
                "r2aq_scale_waterfill_b4",
                "r2aq_utility_waterfill_b4",
                "r2aq_full_tree_b4",
            ]
        )
    ].copy()
    selected = selected.merge(
        payload[
            [
                "method",
                "joint_payload_MiB",
                "joint_compression_ratio_vs_sampled_KV_BF16",
            ]
        ],
        on="method",
        how="left",
    )
    output_pair = paired[paired["metric"] == "QKV_output_MSE"].copy()
    pair_view = output_pair[
        [
            "method",
            "paired_relative_improvement_mean",
            "paired_relative_improvement_CI_low",
            "paired_relative_improvement_CI_high",
            "prompt_win_rate",
        ]
    ]
    runtime_view = runtime[
        runtime["method"].isin(
            [
                "rvq_qv_b2",
                "r2aq_full_tree_b2",
                "rvq_qv_b4",
                "r2aq_utility_waterfill_b4",
                "r2aq_full_tree_b4",
            ]
        )
    ]
    return f"""# R²AQ formal static 分析（2026-07-24）

## 口径

这是对已有 20 个训练 prompt、每个 prompt 2 个固定 layer/head 单元的
CPU 后处理；没有启动新 GPU 实验。attention 指标是
`offline_sampled_pre_rope_attention_without_positional_replay`：它既不是带
RoPE/PRoPE 的精确 attention replay，也不是闭环视频 Similarity 或 VBench，
不得据此宣称视频质量已经改善。

所有统计先在每个 prompt 内对两个单元做 macro average，再在 20 个 prompt
上配对。95% CI 使用固定 seed `{SEED}` 的 `{BOOTSTRAP_SAMPLES}` 次 prompt
bootstrap。`top-k` 没有被原实验记录；这里只原样报告已有的
`attention_top1_agreement`，没有补算或冒充 top-k。

## 主要绝对结果

{markdown_table(selected, [
    "method", "K_MSE", "V_MSE", "QK_logits_MSE", "attention_KL",
    "attention_top1_agreement", "QKV_output_MSE",
    "QKV_output_squared_error_P99",
    "QKV_output_squared_error_P999",
    "QKV_output_squared_error_CVaR99", "joint_payload_MiB",
    "joint_compression_ratio_vs_sampled_KV_BF16"
])}

## 与同位宽 RVQ+QuantVideo 的逐 prompt 配对结果

下表只比较有同位宽 QV reference 的 B2/B4。正值表示改善。

{markdown_table(pair_view, [
    "method", "paired_relative_improvement_mean",
    "paired_relative_improvement_CI_low",
    "paired_relative_improvement_CI_high", "prompt_win_rate"
])}

## 码率记账

`payload.csv` 从 40 个原始 unit 行重新求和。online predictor、IDs、scale、
selector、bitplanes 和每 unit 的 L8C64 INT4 RVQ payload 均计入；共享表按
逻辑 table ID 去重后，在完整 benchmark 由唯一 owner 计一次：两张 K/V
tree reconstruction 为 248 B，两张 encoder threshold 表为 124 B，
utility 表为 1200 B。因此 full-tree/utility 方法全局共享开销为
1572 B；owner 的 local refinement 同时扣除这 1572 B，保持总物理预算
严格相等。压缩率基线是本次抽样 K+V 的
BF16 物理字节数，MiB 定义为 `2**20` bytes。

## Runtime

{markdown_table(runtime_view, [
    "method", "roundtrip_total_ms_KV", "roundtrip_total_ms_KV_median",
    "roundtrip_total_ms_KV_P95", "roundtrip_total_ms_KV_max"
])}

R²AQ encode 是一次生成 full-B4 流的 K/V 编码，decode 是对应截断深度的
K/V 解码；QV 只记录了 K+V encode+decode 合并区间，因而两者不能拆成完全
相同的阶段。时间均为 benchmark 中 CUDA synchronize 包围的单个抽样 unit
算术平均，未包含 RVQ base 重建和 attention proxy。QV B2 在每个 shard
首次调用包含明显的 Triton/JIT 冷启动（最大值约 7.5 s），因此其 mean
不能当作稳态吞吐；`median` 更接近本次采样的稳态单元时间。该静态计时
没有预热和重复测量，只作为诊断，不用于 decode `<=10%` 的最终验收。

## 文件

- `quality_attention.csv`：23 个方法的 K/V、QK、KL、top1 和 output tails。
- `paired_relative_improvement.csv`：B2/B4 的 5 个 ablation，逐指标配对
  relative improvement、95% CI 与胜率。
- `paired_prompt_values.csv`：上述统计的逐 prompt 原始配对值。
- `payload.csv`：完整 40-unit benchmark 的全局去重物理 payload。
- `runtime.csv`：明确计时区间的编解码时间。
- `predictor_diagnostics.csv`：INT4 centroid 对 suffix energy 的影响。
- `evidence_manifest.json` 与 `run_info.json`：来源、hash 和复现口径。
"""


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, float_format="%.12g")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--params-root", type=Path, default=DEFAULT_PARAMS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-owned-output",
        action="store_true",
        help=(
            "refresh this script's own in-progress output only; never use on "
            "an experiment directory"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output
    if not output.is_absolute():
        output = repo / output
    if output.exists():
        owned_config = output / "config.json"
        owned = False
        if owned_config.is_file():
            try:
                owned = (
                    json.loads(owned_config.read_text())["schema_version"]
                    == "r2aq_static_formal_analysis_config.v1"
                )
            except (KeyError, json.JSONDecodeError):
                owned = False
        if not args.refresh_owned_output or not owned:
            raise FileExistsError(
                f"refusing to overwrite existing output: {output}"
            )
    else:
        output.mkdir(parents=True)

    source_paths = [
        args.run_root / "rows" / "shard_000.csv",
        args.run_root / "rows" / "shard_001.csv",
        args.run_root / "aggregate" / "prompt_macro.csv",
        args.run_root / "aggregate" / "summary.csv",
        args.run_root / "aggregate" / "summary.json",
        args.run_root / "run_info" / "aggregate.json",
        args.run_root / "run_info" / "aggregate_config.json",
        args.params_root / "params.pt",
        args.params_root / "params_summary.json",
        args.params_root / "utility_weights.json",
        args.params_root / "manifest.json",
    ]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing sources: {missing}")

    config = {
        "schema_version": "r2aq_static_formal_analysis_config.v1",
        "run_root": str(args.run_root),
        "params_root": str(args.params_root),
        "output": str(output),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": SEED,
        "pairing_unit": "prompt macro after averaging two layer/head units",
        "comparators": {"B2": "rvq_qv_b2", "B4": "rvq_qv_b4"},
        "attention_scope": (
            "offline_sampled_pre_rope_attention_without_positional_replay"
        ),
        "shared_table_accounting": (
            "one deterministic benchmark owner charges and reserves each "
            "logical table ID globally over all 40 benchmark units"
        ),
        "memory_unit": "MiB=2**20 bytes",
        "source_baseline": "sampled K+V BF16, 16 bits/value",
    }
    config_path = output / "config.json"
    config_path.write_bytes(stable_json_bytes(config))

    rows = read_rows(args.run_root)
    prompt = pd.read_csv(args.run_root / "aggregate" / "prompt_macro.csv")
    quality = build_quality(prompt)
    paired, paired_prompt = build_paired(prompt)
    runtime = build_runtime(rows)
    payload = build_payload(rows)
    diagnostics = build_predictor_diagnostics(prompt)

    outputs = {
        "quality_attention": output / "quality_attention.csv",
        "paired_relative_improvement": (
            output / "paired_relative_improvement.csv"
        ),
        "paired_prompt_values": output / "paired_prompt_values.csv",
        "runtime": output / "runtime.csv",
        "payload": output / "payload.csv",
        "predictor_diagnostics": output / "predictor_diagnostics.csv",
    }
    write_csv(quality, outputs["quality_attention"])
    write_csv(paired, outputs["paired_relative_improvement"])
    write_csv(paired_prompt, outputs["paired_prompt_values"])
    write_csv(runtime, outputs["runtime"])
    write_csv(payload, outputs["payload"])
    write_csv(diagnostics, outputs["predictor_diagnostics"])
    readme_path = output / "README.md"
    readme_path.write_text(
        build_readme(quality, paired, payload, runtime), encoding="utf-8"
    )

    original_run_info = json.loads(
        (args.run_root / "run_info" / "run_shard_000.json").read_text()
    )
    run_info = {
        "schema_version": "r2aq_static_formal_analysis_run_info.v1",
        "git_commit": git_commit(repo),
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": original_run_info["codebook_path"],
        "codebook_hash": original_run_info["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": "not used; CPU-only analysis",
        "seed": SEED,
        "prompt_list": original_run_info["prompt_list"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "source_experiment": str(args.run_root),
        "training_parameters": str(args.params_root),
    }
    run_info_path = output / "run_info.json"
    run_info_path.write_bytes(stable_json_bytes(run_info))

    artifact_paths = [
        config_path,
        *outputs.values(),
        readme_path,
        run_info_path,
    ]
    manifest = {
        "schema_version": "r2aq_static_formal_evidence_manifest.v1",
        "sources": {
            str(path): sha256_file(path) for path in source_paths
        },
        "artifacts": {
            path.name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        },
        "implementation": {
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve())
        },
        "accounting_notes": {
            "shared_tables": (
                "K/V tree table IDs and utility table ID are each charged "
                "once globally per evaluated method, not once per unit"
            ),
            "paired_statistics": (
                "20-prompt paired macro; 10000 bootstrap resamples; seed "
                f"{SEED}"
            ),
            "top_k": (
                "source run contains top1 agreement only; no top-k metric "
                "was synthesized"
            ),
            "scope": (
                "offline sampled pre-RoPE proxy, not exact attention, "
                "Similarity, closed-loop video, or VBench"
            ),
        },
    }
    (output / "evidence_manifest.json").write_bytes(
        stable_json_bytes(manifest)
    )
    print(output)


if __name__ == "__main__":
    main()
