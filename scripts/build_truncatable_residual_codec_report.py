#!/usr/bin/env python3
"""Build the auditable report for the four truncatable RVQ-residual codecs.

The benchmark directories are treated as immutable evidence.  This script only
reads them and creates derived CSV/JSON/Markdown artifacts.  A final report is
created only when training, residual-codec runtime, offline reconstruction,
exact-attention replay, Similarity, and all ten supported VBench dimensions are
complete and internally consistent.

Use ``--allow-partial`` to emit explicitly suffixed draft artifacts.  Drafts do
not occupy any final artifact path.  All writes are create-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import socket
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test/"
    "truncatable_residual_benchmark_20260722"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "analysis" / "truncatable_residual_codec_20260722"
)
DEFAULT_REPORT = (
    REPO_ROOT / "docs" / "wan2_2_5b_truncatable_residual_codec_benchmark_zh.md"
)
IMPLEMENTATION_SNAPSHOT_PATHS = (
    REPO_ROOT / "quant_videogen" / "truncatable_residual.py",
    REPO_ROOT / "quant_videogen" / "rvq.py",
    REPO_ROOT / "quant_videogen" / "kernels" / "triton" / "rvq.py",
    REPO_ROOT / "quant_videogen" / "compress.py",
    REPO_ROOT / "quant_videogen" / "uncompress.py",
    REPO_ROOT / "scripts" / "run_fixed_budget_rvq_screen.py",
    REPO_ROOT / "scripts" / "eval" / "run_small_eval.py",
    REPO_ROOT / "scripts" / "run_truncatable_residual_codec_bench.py",
    REPO_ROOT / "scripts" / "evaluate_truncatable_residual_exact_attention.py",
    REPO_ROOT / "scripts" / "run_truncatable_residual_video_bench.py",
    REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py",
    REPO_ROOT / "scripts" / "collect_wan5b_p1_multi_prompt_captures.py",
    REPO_ROOT
    / "experiments"
    / "HY-WorldPlay"
    / "wan"
    / "models"
    / "dits"
    / "arwan_w_action_w_mem_relative_rope.py",
)
IMPLEMENTATION_SNAPSHOT_SCOPE = (
    "current audited source state, including every implementation source "
    "category recorded by strict training/run_info.json implementation_hash "
    "plus the exact-attention and video evaluation entry points; source-run "
    "implementation_hash records remain authoritative for historical executed "
    "code"
)

SCHEMES = (
    "nested_affine",
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)
SCHEME_ZH = {
    "rvq_base": "纯 L8C64 INT4 RVQ",
    "nested_affine": "嵌套仿射 bitplane",
    "gaussian_tree": "多码率 Gaussian tree",
    "additive_binary": "加性二值残差",
    "hierarchical_vq": "分层低维 VQ",
}
BITS = (1, 2, 3, 4)
ATTENTION_HALVES = ("standard_rope", "camera_prope")
INTERVENTIONS = ("K_only", "V_only", "K_and_V")
SUPPORTED_VBENCH = (
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
UNSUPPORTED_VBENCH = (
    "appearance_style",
    "color",
    "multiple_objects",
    "object_class",
    "scene",
    "spatial_relationship",
)
VBENCH_PER_PROMPT_FIXED_SCALE = {
    dimension: (0.01 if dimension == "imaging_quality" else 1.0)
    for dimension in SUPPORTED_VBENCH
}
SIMILARITY_METRICS = ("psnr", "ssim", "lpips")
VIDEO_EQUAL_PAYLOAD_METRICS = (
    "psnr",
    "ssim",
    "lpips",
    "mse",
    *SUPPORTED_VBENCH,
)
EQUAL_PAYLOAD_SCHEMES = (
    "gaussian_tree",
    "additive_binary",
    "hierarchical_vq",
)
LOWER_IS_BETTER_VIDEO_METRICS = {"lpips", "mse"}
RVQ_RUNTIME_ACCELERATION_CACHE_MIB = (
    2  # K and V codebook sets
    * 30  # transformer layers
    * 24  # attention heads
    * 8  # RVQ levels
    * 64  # centroids per level
    * (128 * 2 + 4)  # BF16 centroid plus FP32 squared norm
    / (1 << 20)
)
STRICT_SIMILARITY_DIR_NAME = "similarity_lpips_normalized_v2"
STRICT_SIMILARITY_PROTOCOL_ID = "frame_aligned_psnr_ssim_lpips_normalized_v2"
LEGACY_SIMILARITY_DIR_NAME = "similarity"
SIMILARITY_EVALUATOR = REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"
VBENCH_INSTALL_ROOT = Path("/home/bowen/anaconda3/lib/python3.11/site-packages/vbench")
VBENCH_CURRENT_IMPLEMENTATION_PATHS = (
    VBENCH_INSTALL_ROOT / "__init__.py",
    *(VBENCH_INSTALL_ROOT / f"{dimension}.py" for dimension in SUPPORTED_VBENCH),
)
LEGACY_SIMILARITY_INVALIDATION_REASON = (
    "legacy evaluator passed [0, 1] tensors to lpips 0.1.4 without "
    "normalize=True; strict v2 artifacts replace it for all report metrics"
)
PAIRED_METRICS = (*SIMILARITY_METRICS, *SUPPORTED_VBENCH)
PAIRED_CONTRASTS = (
    "residual_prefix_vs_pure_l8",
    "incremental_refinement_plane",
    "same_prefix_count_vs_nested_affine_not_rate_matched",
)

TRAIN_METRIC_FIELDS = {
    "scheme",
    "hadamard",
    "tensor",
    "bits",
    "residual_MSE",
    "relative_residual_MSE",
    "payload_bytes",
    "compression_ratio_vs_BF16_residual",
}
RUNTIME_FIELDS = {
    "scheme",
    "bits",
    "hadamard",
    "encode_B4_ms",
    "truncate_to_prefix_ms",
    "decode_prefix_ms",
    "elements_KV",
    "payload_bytes_KV",
    "shared_parameter_bytes",
    "compression_ratio_vs_BF16_residual",
    "single_cache_ratio_including_shared_parameters",
    "shape_per_tensor",
    "measurement_interval",
    "trials",
    "warmup",
}
OFFLINE_FIELDS = {
    "method",
    "bits",
    "prompts",
    "coverage_units",
    "expected_units",
    "K_MSE",
    "V_MSE",
    "K_relative_MSE",
    "V_relative_MSE",
    "QK_logits_MSE",
    "attention_probability_MSE",
    "attention_KL",
    "QKV_output_MSE",
    "QKV_output_cosine",
    "residual_compression_ratio_vs_BF16",
    "residual_payload_MiB",
    "shared_parameter_MiB_charged_once",
    "single_unit_ratio_including_shared_parameters",
    "evaluated_payload_MiB",
    "dataset_ratio_including_shared_parameters_once",
}
EXACT_FIELDS = {
    "scheme",
    "bits",
    "attention_half",
    "intervention",
    "n_prompts",
    "historical_K_mse_mean",
    "historical_V_mse_mean",
    "attention_kl_reference_to_candidate_mean",
    "attention_output_mse_mean",
}
VIDEO_SCALAR_FIELDS = {
    "setting",
    "scheme",
    "prefix_bits",
    "hadamard",
    "psnr",
    "ssim",
    "lpips",
    *SUPPORTED_VBENCH,
}
VIDEO_RUNTIME_FIELDS = {
    "scheme",
    "prefix_bits",
    "hadamard",
    "prompt_id",
    "prompt",
    "seed",
    "video_path",
    "kv_payload_mib",
    "shared_codebook_mib",
    "bf16_reference_mib_inferred",
    "joint_compression_including_rvq_and_codec_params",
    "online_kv_encode_time_s",
    "online_kv_decode_time_s",
    "online_kv_decode_calls",
    "e2e_time_s",
}
SIMILARITY_PROMPT_FIELDS = {
    "setting",
    "scheme",
    "prefix_bits",
    "hadamard",
    "prompt_id",
    "prompt",
    "reference_video",
    "test_video",
    "psnr",
    "ssim",
    "lpips",
    "mse",
}
VBENCH_PROMPT_FIELDS = {
    "setting",
    "scheme",
    "prefix_bits",
    "hadamard",
    "prompt_id",
    "dimension",
    "value",
}
PAIRED_FIELDS = {
    "candidate",
    "baseline",
    "contrast",
    "metric",
    "prompts",
    "candidate_mean",
    "baseline_mean",
    "mean_paired_improvement",
    "bootstrap_95_ci_low",
    "bootstrap_95_ci_high",
    "prompt_win_rate",
}


class AuditError(RuntimeError):
    """Raised when final report evidence is incomplete or invalid."""


@dataclass
class Section:
    name: str
    complete: bool = False
    rows: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.errors

    def status(self) -> str:
        if self.complete and self.valid:
            return "complete"
        if self.errors:
            return "invalid"
        return "missing"


STRICT_V2_IMPLEMENTATION_AUDIT_NOTES = (
    "- Gaussian tree 以逐节点 binary k-means 仅作初始化，随后使用 "
    "exact-boundary coordinate descent 直接最小化 "
    "`lambda=[0.1,1.0,0.2,1.0]` 的多 prefix 联合加权 MSE；每轮还更新 "
    "prefix centroid，且只接受联合目标不增的步骤，FP16 打包变差时回退到"
    "初始打包参数。其限制是阈值只枚举观测样本之间、两侧 child 均非空的"
    "有限边界，不含外侧 `±infinity` 端点；因此仍是局部坐标下降，不是全局"
    "最优保证。",
    "- Additive binary 使用 fixed-sign all-prefix weighted least-squares，"
    "随后重新分配 sign 并作单调 line search；约束 alpha 始终为严格递减"
    "正数，FP16 打包若劣于初始化同样回退。该交替优化仍依赖初始化和有限"
    "迭代，不能视为联合离散问题的全局最优解。",
    "- Gaussian、Additive 与 HVQ 的训练归一化先把每个 G64 block 的 RMS "
    "实际量化为 FP16，再用反量化 scale 归一化，与物理 codec 路径一致。"
    "Direct/Hadamard 各自拟合时使用相同 residual fit 行索引，变换选择也"
    "共享同一个、与拟合集不重叠的 held-out 集，因此不存在旧报告所述的"
    "抽样行混杂。",
    "- residual codec microbenchmark 与在线视频 codec 都是 eager PyTorch "
    "reference 路径，尚无 fused/Triton codec kernel；时间只能描述当前"
    "实现复杂度，不能当作优化后部署吞吐。",
    "- strict-final 与早期 v1 同时改变了 runtime-BF16 RVQ assignment、"
    "联合前缀拟合、物理 FP16 RMS 归一化和重新训练后的参数。没有单因素"
    "控制重跑时，strict-v1 数值差异不能归因于其中某一个改动。",
)


def implementation_audit_notes(training: Section) -> tuple[str, ...]:
    run_info = training.metadata.get("run_info", {})
    protocol = (
        run_info.get("experiment_protocol_version")
        if isinstance(run_info, Mapping)
        else None
    )
    if protocol == "truncatable_residual_strict_v2":
        return STRICT_V2_IMPLEMENTATION_AUDIT_NOTES
    return (
        "- 当前 training artifact 未声明 "
        "`truncatable_residual_strict_v2` 协议，因此不能据此推断其 Gaussian/"
        "Additive 拟合目标、FP16 RMS 归一化或 Direct/Hadamard 抽样是否采用"
        " strict-final 实现；相关结果只能按 artifact 自身 provenance 解读。",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--exact-attention-dir",
        type=Path,
        default=None,
        help=(
            "Exact-attention evidence directory. When omitted, prefer "
            "<benchmark-root>/exact_attention_v2_base when it exists, then "
            "fall back to <benchmark-root>/exact_attention."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Emit an explicitly marked draft from available valid sections. "
            "Without this flag, any missing section prevents all writes."
        ),
    )
    parser.add_argument(
        "--partial-tag",
        default="partial",
        help="Suffix used by draft artifacts; ignored for a final report.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Validate and print the section matrix without writing artifacts.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_directory_fingerprint(directory: Path) -> dict[str, Any]:
    """Hash an immutable directory tree by relative path and file content."""

    root = directory.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evidence directory does not exist: {root}")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError(f"Evidence directory is empty: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        content_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
    return {
        "algorithm": "sha256(relative_path\\0size\\0sha256(file)\\n)",
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_bytes,
    }


def resolve_exact_attention_dir(
    benchmark_root: Path,
    explicit: Path | None = None,
) -> tuple[Path, str]:
    if explicit is not None:
        return explicit.resolve(), "explicit_cli"
    preferred = (benchmark_root / "exact_attention_v2_base").resolve()
    if preferred.is_dir():
        return preferred, "auto_prefer_v2_base"
    return (
        (benchmark_root / "exact_attention").resolve(),
        "auto_fallback_legacy",
    )


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b""
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_fields(
    rows: Sequence[Mapping[str, Any]], fields: set[str], label: str
) -> None:
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = sorted(fields - set(rows[0]))
    if missing:
        raise ValueError(f"{label} lacks fields: {missing}")


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} is non-finite: {result}")
    return result


def optional_float(value: Any) -> float | None:
    if value in (None, "", "None", "null", "nan", "NaN"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def int_value(value: Any, label: str) -> int:
    result = finite_float(value, label)
    rounded = round(result)
    if abs(result - rounded) > 1e-9:
        raise ValueError(f"{label} is not integral: {result}")
    return int(rounded)


def validate_offline_shard_grid(
    shards: Any,
    configured_num_shards: Any,
) -> int:
    """Validate aggregate shard count and IDs against its recorded config."""

    expected_count = int_value(
        configured_num_shards,
        "offline configured num_shards",
    )
    if expected_count <= 0:
        raise ValueError("offline configured num_shards must be positive")
    if not isinstance(shards, list) or len(shards) != expected_count:
        raise ValueError(
            "offline summary input shard count differs from configured "
            f"num_shards={expected_count}"
        )
    if any(not isinstance(item, Mapping) for item in shards):
        raise ValueError("offline input shard record is not an object")
    shard_ids = {
        int_value(item.get("shard_index"), "offline shard index") for item in shards
    }
    expected_ids = set(range(expected_count))
    if shard_ids != expected_ids:
        raise ValueError(
            "offline input shard IDs differ from configured range "
            f"0..{expected_count - 1}"
        )
    return expected_count


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"Not a boolean value: {value!r}")


def close(a: float, b: float, *, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))


def paired_bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    if not values:
        raise ValueError("paired bootstrap requires non-empty values")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(int(seed))
    means = generator.choice(
        array,
        size=(samples, len(array)),
        replace=True,
    ).mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_ratio_of_means_bootstrap_ci(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Bootstrap ``1 - mean(candidate) / mean(baseline)`` by paired prompt."""

    candidate_array = np.asarray(candidate, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if (
        candidate_array.ndim != 1
        or baseline_array.ndim != 1
        or len(candidate_array) == 0
        or len(candidate_array) != len(baseline_array)
    ):
        raise ValueError("paired ratio bootstrap requires equal non-empty vectors")
    if (
        not np.all(np.isfinite(candidate_array))
        or not np.all(np.isfinite(baseline_array))
        or np.any(candidate_array < 0)
        or np.any(baseline_array <= 0)
    ):
        raise ValueError(
            "paired ratio bootstrap requires finite non-negative candidate "
            "and strictly positive baseline values"
        )
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0,
        len(candidate_array),
        size=(samples, len(candidate_array)),
    )
    candidate_means = candidate_array[indices].mean(axis=1)
    baseline_means = baseline_array[indices].mean(axis=1)
    if np.any(baseline_means <= 0):
        raise ValueError("paired ratio bootstrap sampled a zero baseline mean")
    reductions = 1.0 - candidate_means / baseline_means
    return (
        float(np.quantile(reductions, 0.025)),
        float(np.quantile(reductions, 0.975)),
    )


def paired_relative_improvement_bootstrap_ci(
    winner: Sequence[float],
    runner_up: Sequence[float],
    *,
    higher_is_better: bool,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float] | None:
    """Bootstrap a paired winner/runner ratio with a positive-better sign."""

    winner_array = np.asarray(winner, dtype=np.float64)
    runner_array = np.asarray(runner_up, dtype=np.float64)
    if (
        winner_array.ndim != 1
        or runner_array.ndim != 1
        or len(winner_array) == 0
        or len(winner_array) != len(runner_array)
        or not np.all(np.isfinite(winner_array))
        or not np.all(np.isfinite(runner_array))
        or np.any(winner_array < 0)
        or np.any(runner_array < 0)
    ):
        raise ValueError(
            "paired relative bootstrap requires equal finite non-negative vectors"
        )
    if float(runner_array.mean()) <= 0:
        return None
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(
        0,
        len(winner_array),
        size=(samples, len(winner_array)),
    )
    winner_means = winner_array[indices].mean(axis=1)
    runner_means = runner_array[indices].mean(axis=1)
    if np.any(runner_means <= 0):
        raise ValueError("paired relative bootstrap sampled a zero runner mean")
    if higher_is_better:
        improvements = winner_means / runner_means - 1.0
    else:
        improvements = 1.0 - winner_means / runner_means
    return (
        float(np.quantile(improvements, 0.025)),
        float(np.quantile(improvements, 0.975)),
    )


def video_equal_payload_comparisons(
    similarity_rows: Sequence[Mapping[str, Any]],
    vbench_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare Gaussian/Additive/HVQ at their exactly matched physical rates."""

    prompt_values: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
    for row in similarity_rows:
        scheme = str(row.get("scheme", ""))
        if scheme not in EQUAL_PAYLOAD_SCHEMES:
            continue
        bits = int_value(row["prefix_bits"], "Similarity equal-rate bits")
        prompt_id = int_value(row["prompt_id"], "Similarity equal-rate prompt")
        for metric in ("psnr", "ssim", "lpips", "mse"):
            key = (scheme, bits, metric)
            if prompt_id in prompt_values[key]:
                raise ValueError(f"duplicate equal-rate Similarity row: {key}")
            prompt_values[key][prompt_id] = finite_float(
                row[metric], f"equal-rate Similarity {metric}"
            )
    for row in vbench_rows:
        scheme = str(row.get("scheme", ""))
        if scheme not in EQUAL_PAYLOAD_SCHEMES:
            continue
        bits = int_value(row["prefix_bits"], "VBench equal-rate bits")
        prompt_id = int_value(row["prompt_id"], "VBench equal-rate prompt")
        metric = str(row["dimension"])
        key = (scheme, bits, metric)
        if prompt_id in prompt_values[key]:
            raise ValueError(f"duplicate equal-rate VBench row: {key}")
        prompt_values[key][prompt_id] = finite_float(
            row["value"], f"equal-rate VBench {metric}"
        )

    scheme_order = {scheme: index for index, scheme in enumerate(EQUAL_PAYLOAD_SCHEMES)}
    output: list[dict[str, Any]] = []
    for bits in BITS:
        for metric_index, metric in enumerate(VIDEO_EQUAL_PAYLOAD_METRICS):
            values_by_scheme: dict[str, list[float]] = {}
            for scheme in EQUAL_PAYLOAD_SCHEMES:
                values = prompt_values.get((scheme, bits, metric), {})
                if set(values) != set(range(10)):
                    raise ValueError(
                        f"equal-rate prompt grid incomplete for "
                        f"{scheme}/b{bits}/{metric}"
                    )
                values_by_scheme[scheme] = [
                    values[prompt_id] for prompt_id in range(10)
                ]
            means = {
                scheme: sum(values) / len(values)
                for scheme, values in values_by_scheme.items()
            }
            higher_is_better = metric not in LOWER_IS_BETTER_VIDEO_METRICS
            ordered = sorted(
                EQUAL_PAYLOAD_SCHEMES,
                key=lambda scheme: (
                    -means[scheme] if higher_is_better else means[scheme],
                    scheme_order[scheme],
                ),
            )
            winner, runner_up = ordered[:2]
            winner_values = values_by_scheme[winner]
            runner_values = values_by_scheme[runner_up]
            if higher_is_better:
                differences = [
                    candidate - baseline
                    for candidate, baseline in zip(
                        winner_values, runner_values, strict=True
                    )
                ]
            else:
                differences = [
                    baseline - candidate
                    for candidate, baseline in zip(
                        winner_values, runner_values, strict=True
                    )
                ]
            seed = int(
                hashlib.sha256(
                    f"video-equal-rate:{bits}:{metric}".encode("utf-8")
                ).hexdigest()[:16],
                16,
            )
            absolute_low, absolute_high = paired_bootstrap_ci(
                differences,
                seed=seed,
            )
            relative_ci = paired_relative_improvement_bootstrap_ci(
                winner_values,
                runner_values,
                higher_is_better=higher_is_better,
                seed=seed,
            )
            runner_mean = means[runner_up]
            relative = None
            relative_low = None
            relative_high = None
            if runner_mean > 0:
                if higher_is_better:
                    relative = means[winner] / runner_mean - 1.0
                else:
                    relative = 1.0 - means[winner] / runner_mean
                if relative_ci is None:
                    raise ValueError("missing finite equal-rate relative CI")
                relative_low, relative_high = relative_ci
            output.append(
                {
                    "bits": bits,
                    "physical_bits_per_value": bits + 0.25,
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "winner": winner,
                    "runner_up": runner_up,
                    "winner_mean": means[winner],
                    "runner_up_mean": runner_mean,
                    "mean_paired_improvement": sum(differences) / len(differences),
                    "paired_improvement_ci95_low": absolute_low,
                    "paired_improvement_ci95_high": absolute_high,
                    "relative_improvement_pct": (
                        None if relative is None else 100.0 * relative
                    ),
                    "relative_improvement_ci95_low_pct": (
                        None if relative_low is None else 100.0 * relative_low
                    ),
                    "relative_improvement_ci95_high_pct": (
                        None if relative_high is None else 100.0 * relative_high
                    ),
                    "prompt_win_rate": sum(value > 0 for value in differences)
                    / len(differences),
                    "statistically_resolved_95": (
                        relative_low is not None and relative_low > 0
                    ),
                    "meets_10pct_relative_improvement_threshold_95": (
                        relative_low is not None and relative_low > 0.10
                    ),
                    "comparison_scope": (
                        "exploratory post-hoc winner-vs-runner paired prompts; "
                        "n=10; no multiplicity or selection correction"
                    ),
                    "metric_order": metric_index,
                }
            )
    return output


def vbench_signal_audit(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Identify constant and exactly duplicated per-prompt VBench signals."""

    by_dimension: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    for row in rows:
        dimension = str(row["dimension"])
        key = (
            str(row["setting"]),
            int_value(row["prompt_id"], "VBench signal prompt"),
        )
        if key in by_dimension[dimension]:
            raise ValueError(f"duplicate VBench signal row: {dimension}/{key}")
        by_dimension[dimension][key] = finite_float(
            row["value"], f"VBench signal {dimension}"
        )
    expected_keys = next(iter(by_dimension.values())).keys()
    output: list[dict[str, Any]] = []
    previous: dict[str, list[float]] = {}
    for dimension in SUPPORTED_VBENCH:
        values_by_key = by_dimension.get(dimension, {})
        if values_by_key.keys() != expected_keys:
            raise ValueError(f"VBench signal coverage differs for {dimension}")
        values = [values_by_key[key] for key in sorted(expected_keys)]
        duplicate_of = None
        for prior_dimension, prior_values in previous.items():
            if values == prior_values:
                duplicate_of = prior_dimension
                break
        minimum = min(values)
        maximum = max(values)
        constant = minimum == maximum
        output.append(
            {
                "dimension": dimension,
                "rows": len(values),
                "minimum": minimum,
                "maximum": maximum,
                "constant": constant,
                "duplicate_of": duplicate_of,
                "per_prompt_scale_applied": VBENCH_PER_PROMPT_FIXED_SCALE[dimension],
                "counts_as_unique_varying_signal": (
                    not constant and duplicate_of is None
                ),
            }
        )
        previous[dimension] = values
    return output


def exact_equal_payload_comparisons(
    summary_rows: Sequence[Mapping[str, Any]],
    prompt_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build paired winner-vs-runner-up exact-output comparisons by prompt."""

    schemes = ("gaussian_tree", "additive_binary", "hierarchical_vq")
    prompt_lookup: dict[tuple[str, int, str, int], float] = {}
    for row in prompt_rows:
        if row.get("intervention") != "K_and_V":
            continue
        key = (
            str(row["scheme"]),
            int_value(row["bits"], "exact prompt bits"),
            str(row["attention_half"]),
            int_value(row["prompt_id"], "exact prompt_id"),
        )
        if key in prompt_lookup:
            raise ValueError(f"duplicate exact prompt macro key: {key}")
        prompt_lookup[key] = finite_float(
            row["attention_output_mse"],
            "exact prompt attention_output_mse",
        )
    result: list[dict[str, Any]] = []
    for bits in BITS:
        for half in ATTENTION_HALVES:
            candidates = [
                row
                for row in summary_rows
                if row["scheme"] in schemes
                and int_value(row["bits"], "exact summary bits") == bits
                and row["attention_half"] == half
                and row["intervention"] == "K_and_V"
            ]
            if len(candidates) != len(schemes):
                raise ValueError(
                    f"exact equal-payload candidates incomplete for b{bits}/{half}"
                )
            ordered = sorted(
                candidates,
                key=lambda row: finite_float(
                    row["attention_output_mse_mean"],
                    "exact attention_output_mse_mean",
                ),
            )
            winner, runner_up = ordered[:2]
            winner_prompt = [
                prompt_lookup[
                    (
                        str(winner["scheme"]),
                        bits,
                        half,
                        prompt_id,
                    )
                ]
                for prompt_id in range(10)
            ]
            runner_prompt = [
                prompt_lookup[
                    (
                        str(runner_up["scheme"]),
                        bits,
                        half,
                        prompt_id,
                    )
                ]
                for prompt_id in range(10)
            ]
            differences = [
                baseline - candidate
                for candidate, baseline in zip(
                    winner_prompt,
                    runner_prompt,
                    strict=True,
                )
            ]
            seed = int(
                hashlib.sha256(f"exact:{bits}:{half}".encode("utf-8")).hexdigest()[:16],
                16,
            )
            low, high = paired_bootstrap_ci(differences, seed=seed)
            winner_mean = sum(winner_prompt) / len(winner_prompt)
            runner_mean = sum(runner_prompt) / len(runner_prompt)
            if runner_mean <= 0:
                raise ValueError("exact runner-up mean must be positive")
            relative_reduction = 1.0 - winner_mean / runner_mean
            relative_low, relative_high = paired_ratio_of_means_bootstrap_ci(
                winner_prompt,
                runner_prompt,
                seed=seed,
            )
            result.append(
                {
                    "bits": bits,
                    "physical_bits_per_value": bits + 0.25,
                    "attention_half": half,
                    "winner": winner["scheme"],
                    "runner_up": runner_up["scheme"],
                    "winner_mean": winner_mean,
                    "runner_up_mean": runner_mean,
                    "paired_output_mse_reduction": sum(differences) / len(differences),
                    "paired_output_mse_reduction_ci95_low": low,
                    "paired_output_mse_reduction_ci95_high": high,
                    "ratio_of_means_relative_output_mse_reduction_pct": (
                        100.0 * relative_reduction
                    ),
                    "ratio_of_means_relative_output_mse_reduction_ci95_low_pct": (
                        100.0 * relative_low
                    ),
                    "ratio_of_means_relative_output_mse_reduction_ci95_high_pct": (
                        100.0 * relative_high
                    ),
                    "prompt_win_rate": sum(value > 0 for value in differences)
                    / len(differences),
                    "statistically_resolved_95": relative_low > 0,
                    "meets_10pct_relative_improvement_threshold_95": (
                        relative_low > 0.10
                    ),
                    "comparison_scope": (
                        "exploratory paired-prompt ratio-of-means; n=10; "
                        "one of 8 bit-by-half comparisons; no multiplicity "
                        "or post-selection correction"
                    ),
                }
            )
    return result


def evidence_entry(path: Path, section: str, role: str) -> dict[str, Any]:
    return {
        "section": section,
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def validate_recorded_file(
    info: Mapping[str, Any],
    *,
    path_key: str,
    hash_key: str,
    label: str,
    expected_path: Path | None = None,
) -> Path:
    """Validate a run_info file reference, not only its recorded digest."""

    raw_path = info.get(path_key)
    raw_hash = info.get(hash_key)
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} lacks a non-empty {path_key}")
    if not is_sha256(raw_hash):
        raise ValueError(f"{label} lacks a SHA256-shaped {hash_key}")
    path = Path(raw_path).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ValueError(
            f"{label} {path_key} mismatch: {path} != {expected_path.resolve()}"
        )
    if not path.is_file():
        raise ValueError(f"{label} recorded file is absent: {path}")
    actual = sha256_file(path)
    if raw_hash != actual:
        raise ValueError(f"{label} {hash_key} mismatch: {raw_hash} != {actual}")
    return path


def validate_codebook_record(info: Mapping[str, Any], *, label: str) -> None:
    """Require resolvable codebook directories and well-formed recorded hashes.

    Historical producers use two explicitly different codebook digest
    semantics (directory-byte hashes and registered logical dataset hashes), so
    this report must not silently compare those values as if they were the same
    hash.  It can still prove that every recorded path exists and that the path
    and hash key sets are aligned.
    """

    paths = info.get("codebook_path")
    hashes = info.get("codebook_hash")
    if isinstance(paths, Mapping) and isinstance(hashes, Mapping):
        if set(paths) != set(hashes) or not paths:
            raise ValueError(f"{label} codebook path/hash keys differ")
        pairs = [(str(paths[key]), hashes[key]) for key in paths]
    elif isinstance(paths, list) and isinstance(hashes, Mapping):
        normalized_paths = {str(Path(str(value)).resolve()) for value in paths}
        normalized_hashes = {
            str(Path(str(key)).resolve()): value for key, value in hashes.items()
        }
        if normalized_paths != set(normalized_hashes) or not paths:
            raise ValueError(f"{label} codebook path/hash keys differ")
        pairs = [
            (
                str(value),
                normalized_hashes[str(Path(str(value)).resolve())],
            )
            for value in paths
        ]
    else:
        raise ValueError(f"{label} codebook path/hash records are malformed")
    for raw_path, raw_hash in pairs:
        path = Path(raw_path).resolve()
        if not path.is_dir():
            raise ValueError(f"{label} codebook directory is absent: {path}")
        if not is_sha256(raw_hash):
            raise ValueError(f"{label} codebook hash is not SHA256-shaped")


def load_required(
    section: Section,
    path: Path,
    role: str,
    loader: Any,
) -> Any | None:
    if not path.is_file():
        section.missing.append(str(path))
        return None
    try:
        value = loader(path)
        section.evidence.append(evidence_entry(path, section.name, role))
        return value
    except Exception as error:
        section.errors.append(f"{path}: {type(error).__name__}: {error}")
        return None


def validate_run_info(
    info: Mapping[str, Any],
    config_path: Path,
    *,
    params_path: Path | None = None,
) -> None:
    required = {
        "git_commit",
        "config_path",
        "config_hash",
        "codebook_path",
        "codebook_hash",
        "hostname",
        "gpu_type",
        "seed",
        "prompt_list",
        "timestamp",
    }
    missing = sorted(required - set(info))
    if missing:
        raise ValueError(f"run_info lacks AGENTS fields: {missing}")
    validate_recorded_file(
        info,
        path_key="config_path",
        hash_key="config_hash",
        label="run_info config",
        expected_path=config_path,
    )
    validate_codebook_record(info, label="run_info")
    if params_path is not None:
        validate_recorded_file(
            info,
            path_key="codec_params_path",
            hash_key="codec_params_hash",
            label="run_info codec params",
            expected_path=params_path,
        )


def validate_training_links(
    info: Mapping[str, Any],
    training: Section,
    *,
    require_implementation_match: bool,
) -> None:
    """Tie a downstream phase to the audited training artifacts."""

    if not training.complete:
        return
    validate_recorded_file(
        info,
        path_key="selection_path",
        hash_key="selection_hash",
        label="downstream selection",
        expected_path=Path(training.metadata["selection_path"]),
    )
    manifest_path = training.metadata.get("manifest_path")
    if "codec_params_manifest_path" in info or "codec_params_manifest_hash" in info:
        if not manifest_path:
            raise ValueError("training manifest metadata is unavailable")
        validate_recorded_file(
            info,
            path_key="codec_params_manifest_path",
            hash_key="codec_params_manifest_hash",
            label="downstream codec manifest",
            expected_path=Path(str(manifest_path)),
        )
    training_info = training.metadata.get("run_info", {})
    if require_implementation_match:
        expected = (
            training_info.get("implementation_hash")
            if isinstance(training_info, Mapping)
            else None
        )
        observed = info.get("implementation_hash")
        if not isinstance(expected, Mapping) or observed != expected:
            raise ValueError("downstream implementation_hash differs from training")
    expected_protocol = (
        training_info.get("experiment_protocol_version")
        if isinstance(training_info, Mapping)
        else None
    )
    observed_protocol = info.get("experiment_protocol_version")
    if (
        expected_protocol is not None
        and observed_protocol is not None
        and observed_protocol != expected_protocol
    ):
        raise ValueError("downstream experiment protocol differs from training")


def audit_training(root: Path) -> Section:
    section = Section("training")
    directory = root / "training"
    paths = {
        "params": directory / "codec_params.pt",
        "manifest": directory / "codec_params_manifest.json",
        "selection": directory / "selection.json",
        "metrics": directory / "train_metrics.csv",
        "config": directory / "config.json",
        "run_info": directory / "run_info.json",
    }
    params = load_required(section, paths["params"], "codec parameters", lambda p: p)
    manifest = load_required(
        section, paths["manifest"], "parameter manifest", read_json
    )
    selection = load_required(
        section, paths["selection"], "Hadamard selection", read_json
    )
    metrics = load_required(section, paths["metrics"], "training metrics", read_csv)
    config = load_required(section, paths["config"], "training config", read_json)
    run_info = load_required(
        section, paths["run_info"], "training provenance", read_json
    )
    if section.missing or section.errors:
        return section
    try:
        assert params is not None
        assert isinstance(manifest, Mapping)
        assert isinstance(selection, Mapping)
        assert isinstance(metrics, list)
        assert isinstance(config, Mapping)
        assert isinstance(run_info, Mapping)
        if manifest.get("schema_version") != "truncatable_residual_params_manifest.v1":
            raise ValueError("unexpected codec parameter manifest schema")
        params_hash = sha256_file(params)
        if manifest.get("codec_params_sha256") != params_hash:
            raise ValueError("codec_params.pt does not match its manifest hash")
        manifest_params_path = Path(str(manifest.get("codec_params_path"))).resolve()
        if manifest_params_path != params.resolve():
            raise ValueError("codec parameter manifest points to another params file")
        selected = selection.get("selected_hadamard")
        if not isinstance(selected, Mapping) or set(selected) != set(SCHEMES):
            raise ValueError("selection does not contain exactly the four schemes")
        require_fields(metrics, TRAIN_METRIC_FIELDS, "train_metrics.csv")
        keys = {
            (
                row["scheme"],
                bool_value(row["hadamard"]),
                row["tensor"],
                int_value(row["bits"], "training bits"),
            )
            for row in metrics
        }
        expected = {
            (scheme, hadamard, tensor, bits)
            for scheme in SCHEMES
            for hadamard in (False, True)
            for tensor in ("K", "V")
            for bits in BITS
        }
        if keys != expected or len(metrics) != len(expected):
            raise ValueError(
                f"training coverage is {len(keys)}/{len(expected)} unique rows"
            )
        prefix_weights = {1: 0.1, 2: 1.0, 3: 0.2, 4: 1.0}
        for scheme in SCHEMES:
            scores: dict[bool, float] = {}
            for hadamard in (False, True):
                candidates = [
                    row
                    for row in metrics
                    if row["scheme"] == scheme
                    and bool_value(row["hadamard"]) == hadamard
                ]
                scores[hadamard] = sum(
                    prefix_weights[int_value(row["bits"], "training bits")]
                    * finite_float(
                        row["relative_residual_MSE"],
                        "training relative residual MSE",
                    )
                    for row in candidates
                ) / (2.0 * sum(prefix_weights.values()))
            expected_hadamard = scores[True] < scores[False]
            if bool_value(selected[scheme]) != expected_hadamard:
                raise ValueError(
                    f"{scheme}: stored Hadamard selection differs from "
                    "train-only held-out objective"
                )
        param_bytes = manifest.get("selected_parameter_bytes_by_scheme")
        if not isinstance(param_bytes, Mapping) or set(param_bytes) != set(SCHEMES):
            raise ValueError("manifest lacks per-scheme selected parameter bytes")
        per_scheme = {key: int(param_bytes[key]) for key in SCHEMES}
        selected_total = int(manifest["selected_parameter_bytes_total"])
        if sum(per_scheme.values()) != selected_total:
            raise ValueError("per-scheme parameter bytes do not sum to selected total")
        if int(selection["parameter_bytes_selected"]) != selected_total:
            raise ValueError("selection and manifest selected parameter bytes differ")
        validate_run_info(run_info, paths["config"], params_path=params)
        validate_recorded_file(
            run_info,
            path_key="selection_path",
            hash_key="selection_hash",
            label="training selection",
            expected_path=paths["selection"],
        )
        validate_recorded_file(
            run_info,
            path_key="codec_params_manifest_path",
            hash_key="codec_params_manifest_hash",
            label="training codec manifest",
            expected_path=paths["manifest"],
        )
        if run_info.get("codec_params_hash") != params_hash:
            raise ValueError("training run_info codec parameter hash mismatch")
        prompt_list = run_info.get("prompt_list")
        prompt_ids = run_info.get("prompt_ids")
        if (
            not isinstance(prompt_list, list)
            or len(prompt_list) != 20
            or list(prompt_ids or []) != list(range(20))
        ):
            raise ValueError("training run_info is not the fixed 20-prompt cohort")
        if config.get("train_prompts") != "0:20":
            raise ValueError("training config does not select prompts 0:20")
        if set(config.get("schemes", ())) != set(SCHEMES):
            raise ValueError("training config does not cover all four schemes")
        protocol_values = {
            value
            for value in (
                config.get("experiment_protocol_version"),
                run_info.get("experiment_protocol_version"),
                manifest.get("training_protocol_version"),
                selection.get("training_protocol_version"),
            )
            if value is not None
        }
        if len(protocol_values) != 1:
            raise ValueError("training protocol versions are inconsistent")
        for row in metrics:
            for name in (
                "residual_MSE",
                "relative_residual_MSE",
                "payload_bytes",
                "compression_ratio_vs_BF16_residual",
            ):
                finite_float(row[name], f"training {name}")
            bits = int_value(row["bits"], "training bits")
            overhead = 1.0 if row["scheme"] == "nested_affine" else 0.25
            if not close(
                finite_float(
                    row["compression_ratio_vs_BF16_residual"],
                    "training residual ratio",
                ),
                16.0 / (bits + overhead),
            ):
                raise ValueError(
                    f"{row['scheme']} b{bits}: training payload ratio differs"
                )
            row["selected_transform"] = bool_value(row["hadamard"]) == bool_value(
                selected[row["scheme"]]
            )
        scheme_order = {value: index for index, value in enumerate(SCHEMES)}
        tensor_order = {"K": 0, "V": 1}
        section.rows = sorted(
            (dict(row) for row in metrics),
            key=lambda row: (
                scheme_order[row["scheme"]],
                bool_value(row["hadamard"]),
                tensor_order[row["tensor"]],
                int_value(row["bits"], "training bits"),
            ),
        )
        section.metadata = {
            "params_path": str(params.resolve()),
            "params_file_bytes": params.stat().st_size,
            "params_hash": params_hash,
            "manifest_path": str(paths["manifest"].resolve()),
            "manifest_hash": sha256_file(paths["manifest"]),
            "selection_path": str(paths["selection"].resolve()),
            "selection_hash": sha256_file(paths["selection"]),
            "selected_hadamard": {key: bool_value(selected[key]) for key in SCHEMES},
            "parameter_bytes_by_scheme": per_scheme,
            "selected_parameter_bytes_total": selected_total,
            "all_ablation_parameter_bytes": int(
                manifest["ablation_candidate_parameter_bytes_total"]
            ),
            "selection_basis": selection.get("selection_basis"),
            "rvq_assignment": (
                config.get("rvq_assignment") or run_info.get("rvq_assignment")
            ),
            "run_info": dict(run_info),
        }
        section.complete = True
    except Exception as error:
        section.errors.append(f"{type(error).__name__}: {error}")
        section.rows = []
    return section


def audit_runtime(root: Path, training: Section) -> Section:
    section = Section("runtime")
    paths = {
        "csv": root / "runtime" / "runtime.csv",
        "json": root / "runtime" / "runtime.json",
        "config": root / "run_info" / "benchmark_config.json",
        "run_info": root / "run_info" / "benchmark.json",
    }
    rows = load_required(section, paths["csv"], "codec runtime table", read_csv)
    rows_json = load_required(section, paths["json"], "codec runtime JSON", read_json)
    config = load_required(section, paths["config"], "runtime config", read_json)
    info = load_required(section, paths["run_info"], "runtime provenance", read_json)
    if section.missing or section.errors:
        return section
    try:
        assert isinstance(rows, list)
        assert isinstance(rows_json, list)
        assert isinstance(config, Mapping)
        assert isinstance(info, Mapping)
        require_fields(rows, RUNTIME_FIELDS, "runtime.csv")
        if len(rows_json) != len(rows):
            raise ValueError("runtime CSV/JSON row counts differ")
        keys = {(row["scheme"], int_value(row["bits"], "runtime bits")) for row in rows}
        expected = {(scheme, bits) for scheme in SCHEMES for bits in BITS}
        if keys != expected or len(rows) != len(expected):
            raise ValueError(
                f"runtime coverage is {len(keys)}/{len(expected)} unique rows"
            )
        json_by_key = {
            (
                str(row["scheme"]),
                int_value(row["bits"], "runtime JSON bits"),
            ): row
            for row in rows_json
            if isinstance(row, Mapping)
        }
        csv_by_key = {
            (row["scheme"], int_value(row["bits"], "runtime CSV bits")): row
            for row in rows
        }
        if set(json_by_key) != set(csv_by_key) or len(json_by_key) != len(rows_json):
            raise ValueError("runtime CSV/JSON unique keys differ")
        for key, csv_row in csv_by_key.items():
            json_row = json_by_key[key]
            for field_name in RUNTIME_FIELDS - {
                "scheme",
                "bits",
                "hadamard",
                "shape_per_tensor",
                "measurement_interval",
            }:
                if not close(
                    finite_float(csv_row[field_name], f"runtime CSV {field_name}"),
                    finite_float(json_row[field_name], f"runtime JSON {field_name}"),
                ):
                    raise ValueError(f"runtime CSV/JSON differ for {key}/{field_name}")
            if bool_value(csv_row["hadamard"]) != bool_value(json_row["hadamard"]):
                raise ValueError(f"runtime CSV/JSON differ for {key}/hadamard")
            for field_name in ("shape_per_tensor", "measurement_interval"):
                if str(csv_row[field_name]) != str(json_row[field_name]):
                    raise ValueError(f"runtime CSV/JSON differ for {key}/{field_name}")
        params_path = (
            Path(training.metadata["params_path"]) if training.complete else None
        )
        validate_run_info(info, paths["config"], params_path=params_path)
        validate_training_links(
            info,
            training,
            require_implementation_match=True,
        )
        if set(config.get("schemes", ())) != set(SCHEMES):
            raise ValueError("runtime config does not cover all four schemes")
        parameter_bytes = (
            training.metadata.get("parameter_bytes_by_scheme", {})
            if training.complete
            else {}
        )
        selected_hadamard = (
            training.metadata.get("selected_hadamard", {}) if training.complete else {}
        )
        output = []
        encodes: dict[str, set[float]] = defaultdict(set)
        shapes: set[str] = set()
        element_counts: set[int] = set()
        intervals: set[str] = set()
        for raw in rows:
            row = dict(raw)
            scheme = row["scheme"]
            bits = int_value(row["bits"], "runtime bits")
            elements = int_value(row["elements_KV"], "elements_KV")
            element_counts.add(elements)
            shapes.add(str(row["shape_per_tensor"]))
            intervals.add(str(row["measurement_interval"]))
            hadamard = bool_value(row["hadamard"])
            if selected_hadamard and hadamard != bool_value(selected_hadamard[scheme]):
                raise ValueError(
                    f"{scheme} b{bits}: runtime Hadamard choice differs "
                    "from training selection"
                )
            payload = int_value(row["payload_bytes_KV"], "payload_bytes_KV")
            source = elements * 2
            ratio = finite_float(
                row["compression_ratio_vs_BF16_residual"],
                "runtime compression ratio",
            )
            if not close(ratio, source / payload):
                raise ValueError(f"{scheme} b{bits}: residual ratio is inconsistent")
            shared = int_value(row["shared_parameter_bytes"], "shared_parameter_bytes")
            if parameter_bytes and shared != int(parameter_bytes[scheme]):
                raise ValueError(
                    f"{scheme}: runtime shared parameter bytes {shared} "
                    f"!= manifest {parameter_bytes[scheme]}"
                )
            ratio_shared = finite_float(
                row["single_cache_ratio_including_shared_parameters"],
                "runtime single-cache ratio",
            )
            if not close(ratio_shared, source / (payload + shared)):
                raise ValueError(
                    f"{scheme} b{bits}: shared-parameter ratio is inconsistent"
                )
            encode = finite_float(row["encode_B4_ms"], "encode_B4_ms")
            encodes[scheme].add(round(encode, 9))
            for name in ("truncate_to_prefix_ms", "decode_prefix_ms"):
                finite_float(row[name], name)
            if (
                int_value(row["trials"], "runtime trials") != 20
                or int_value(row["warmup"], "runtime warmup") != 5
            ):
                raise ValueError("runtime benchmark is not 5-warmup/20-trial")
            effective_bits = payload * 8.0 / elements
            expected_overhead = 1.0 if scheme == "nested_affine" else 0.25
            if not close(effective_bits, bits + expected_overhead):
                raise ValueError(
                    f"{scheme} b{bits}: {effective_bits:.6f} physical bits/value "
                    f"does not match b+{expected_overhead}"
                )
            row.update(
                {
                    "physical_effective_bits_per_value": effective_bits,
                    "local_metadata_bits_per_value": effective_bits - bits,
                    "BF16_residual_source_bytes_KV": source,
                    "rate_baseline": "BF16 residual; physical head_dim=128",
                    "shared_parameter_scope": "charged once per cache",
                }
            )
            output.append(row)
        if any(len(values) != 1 for values in encodes.values()):
            raise ValueError(
                "encode_B4_ms must be identical across prefixes of one scheme"
            )
        if len(shapes) != 1 or len(element_counts) != 1:
            raise ValueError("runtime rows do not share one benchmark tensor shape")
        if intervals != {"residual codec only; RVQ reconstruction excluded"}:
            raise ValueError("runtime measurement interval differs across rows")
        scheme_order = {value: index for index, value in enumerate(SCHEMES)}
        section.rows = sorted(
            output,
            key=lambda row: (
                scheme_order[row["scheme"]],
                int_value(row["bits"], "runtime bits"),
            ),
        )
        section.metadata = {
            "run_info": dict(info),
            "shape_per_tensor": rows[0]["shape_per_tensor"],
            "elements_KV": int_value(rows[0]["elements_KV"], "elements_KV"),
            "measurement_interval": rows[0]["measurement_interval"],
        }
        section.complete = True
    except Exception as error:
        section.errors.append(f"{type(error).__name__}: {error}")
        section.rows = []
    return section


def audit_offline(root: Path, training: Section) -> Section:
    section = Section("offline")
    paths = {
        "summary_csv": root / "aggregate" / "summary.csv",
        "summary_json": root / "aggregate" / "summary.json",
        "prompt_macro": root / "aggregate" / "prompt_macro.csv",
        "config": root / "run_info" / "aggregate_config.json",
        "run_info": root / "run_info" / "aggregate.json",
    }
    rows = load_required(
        section, paths["summary_csv"], "offline prompt-bootstrap summary", read_csv
    )
    summary = load_required(
        section, paths["summary_json"], "offline summary metadata", read_json
    )
    prompt_rows = load_required(
        section, paths["prompt_macro"], "offline prompt macro", read_csv
    )
    config = load_required(section, paths["config"], "offline config", read_json)
    info = load_required(section, paths["run_info"], "offline provenance", read_json)
    if section.missing or section.errors:
        return section
    try:
        assert isinstance(rows, list)
        assert isinstance(summary, Mapping)
        assert isinstance(prompt_rows, list)
        assert isinstance(config, Mapping)
        assert isinstance(info, Mapping)
        require_fields(rows, OFFLINE_FIELDS, "aggregate/summary.csv")
        require_fields(
            prompt_rows,
            {
                "method",
                "bits",
                "prompt_id",
                "units",
                "K_MSE",
                "V_MSE",
                "K_relative_MSE",
                "V_relative_MSE",
                "QK_logits_MSE",
                "attention_probability_MSE",
                "attention_KL",
                "QKV_output_MSE",
                "QKV_output_cosine",
            },
            "aggregate/prompt_macro.csv",
        )
        expected_methods = {"l8c64_int4_base"} | {
            f"{scheme}_b{bits}" for scheme in SCHEMES for bits in BITS
        }
        methods = {row["method"] for row in rows}
        if methods != expected_methods or len(rows) != len(expected_methods):
            raise ValueError(
                f"offline method coverage is {len(methods)}/{len(expected_methods)}"
            )
        expected_bits_by_method = {
            "l8c64_int4_base": 0,
            **{f"{scheme}_b{bits}": bits for scheme in SCHEMES for bits in BITS},
        }
        for row in rows:
            if (
                int_value(row["bits"], "offline bits")
                != expected_bits_by_method[row["method"]]
            ):
                raise ValueError("offline summary method/bits mapping differs")
        coverage = summary.get("coverage")
        if not isinstance(coverage, Mapping) or not coverage.get("complete"):
            raise ValueError("offline summary does not claim complete coverage")
        observed = int(coverage.get("observed_units", -1))
        expected_units = int(coverage.get("expected_units", -1))
        if observed != 7200 or expected_units != 7200:
            raise ValueError(
                f"offline coverage is {observed}/{expected_units}, expected 7200/7200"
            )
        if len(prompt_rows) != 17 * 10:
            raise ValueError(
                f"offline prompt macro has {len(prompt_rows)} rows, expected 170"
            )
        prompt_keys = {
            (
                row["method"],
                int_value(row["prompt_id"], "offline prompt ID"),
            )
            for row in prompt_rows
        }
        expected_prompt_keys = {
            (method, prompt_id)
            for method in expected_methods
            for prompt_id in range(10)
        }
        if prompt_keys != expected_prompt_keys or len(prompt_keys) != len(prompt_rows):
            raise ValueError("offline prompt macro unique-key coverage differs")
        for row in prompt_rows:
            if (
                int_value(row["bits"], "offline prompt bits")
                != expected_bits_by_method[row["method"]]
            ):
                raise ValueError("offline prompt method/bits mapping differs")
        summary_json_rows = summary.get("summary")
        if not isinstance(summary_json_rows, list) or len(summary_json_rows) != 17:
            raise ValueError("offline summary JSON does not contain 17 rows")
        csv_by_method = {row["method"]: row for row in rows}
        json_by_method = {row["method"]: row for row in summary_json_rows}
        if set(csv_by_method) != set(json_by_method):
            raise ValueError("offline summary CSV/JSON method sets differ")
        for method in csv_by_method:
            if (
                int_value(
                    json_by_method[method]["bits"],
                    "offline JSON bits",
                )
                != expected_bits_by_method[method]
            ):
                raise ValueError("offline summary JSON method/bits mapping differs")
            for metric in ("K_MSE", "V_MSE", "attention_KL", "QKV_output_MSE"):
                if not close(
                    finite_float(csv_by_method[method][metric], metric),
                    finite_float(json_by_method[method][metric], metric),
                ):
                    raise ValueError(
                        f"offline summary CSV/JSON differ for {method}/{metric}"
                    )
        shards = summary.get("input_shards")
        validate_offline_shard_grid(shards, config.get("num_shards"))
        if config.get("input_shards") != shards:
            raise ValueError("offline aggregate config/summary shard records differ")
        if any(item.get("provenance") != "validated" for item in shards):
            raise ValueError("offline input shard provenance is not fully validated")
        for item in shards:
            shard_index = int_value(item["shard_index"], "offline shard index")
            for path_key, hash_key, role in (
                ("csv_path", "csv_sha256", "offline raw shard"),
                ("config_path", "config_sha256", "offline shard config"),
                ("run_info_path", "run_info_sha256", "offline shard provenance"),
            ):
                path = validate_recorded_file(
                    item,
                    path_key=path_key,
                    hash_key=hash_key,
                    label=f"offline shard {shard_index} {role}",
                )
                section.evidence.append(
                    evidence_entry(
                        path,
                        section.name,
                        f"{role} {shard_index}",
                    )
                )
        prompt_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in prompt_rows:
            prompt_groups[str(row["method"])].append(row)
            if int_value(row["units"], "offline prompt units") != 720:
                raise ValueError("offline prompt macro unit count is not 720")
        mean_metrics = (
            "K_MSE",
            "V_MSE",
            "K_relative_MSE",
            "V_relative_MSE",
            "QK_logits_MSE",
            "attention_probability_MSE",
            "attention_KL",
            "QKV_output_MSE",
            "QKV_output_cosine",
        )
        for method, summary_row in csv_by_method.items():
            prompts_for_method = prompt_groups[method]
            if len(prompts_for_method) != 10:
                raise ValueError(f"offline prompt group incomplete for {method}")
            for metric in mean_metrics:
                prompt_mean = (
                    sum(
                        finite_float(row[metric], f"offline prompt {metric}")
                        for row in prompts_for_method
                    )
                    / 10.0
                )
                if not close(
                    finite_float(summary_row[metric], f"offline {metric}"),
                    prompt_mean,
                    rtol=1e-7,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"offline summary mean differs from prompt macro for "
                        f"{method}/{metric}"
                    )
        for row in rows:
            if int_value(row["prompts"], "offline prompts") != 10:
                raise ValueError(f"{row['method']}: offline prompts != 10")
            if int_value(row["coverage_units"], "offline coverage") != 7200:
                raise ValueError(f"{row['method']}: offline coverage != 7200")
            for name in (
                "K_MSE",
                "V_MSE",
                "K_relative_MSE",
                "V_relative_MSE",
                "QK_logits_MSE",
                "attention_probability_MSE",
                "attention_KL",
                "QKV_output_MSE",
                "QKV_output_cosine",
            ):
                finite_float(row[name], f"offline {name}")
            if row["method"] != "l8c64_int4_base":
                bits = int_value(row["bits"], "offline rate bits")
                scheme = row["method"].rsplit("_b", 1)[0]
                overhead = 1.0 if scheme == "nested_affine" else 0.25
                expected_residual_ratio = 16.0 / (bits + overhead)
                residual_ratio = finite_float(
                    row["residual_compression_ratio_vs_BF16"],
                    "offline residual ratio",
                )
                if not close(residual_ratio, expected_residual_ratio):
                    raise ValueError(
                        f"{row['method']}: offline physical residual ratio differs"
                    )
                unit_payload = finite_float(
                    row["residual_payload_MiB"],
                    "offline unit residual payload",
                )
                evaluated_payload = finite_float(
                    row["evaluated_payload_MiB"],
                    "offline evaluated payload",
                )
                if not close(
                    evaluated_payload,
                    unit_payload * 7200,
                    rtol=1e-7,
                    atol=1e-9,
                ):
                    raise ValueError(
                        f"{row['method']}: evaluated payload is not 7200 units"
                    )
                shared = finite_float(
                    row["shared_parameter_MiB_charged_once"],
                    "offline shared parameters",
                )
                if training.complete:
                    expected_shared = int(
                        training.metadata["parameter_bytes_by_scheme"][scheme]
                    ) / (1 << 20)
                    if not close(shared, expected_shared):
                        raise ValueError(
                            f"{row['method']}: offline shared parameter charge differs"
                        )
                unit_source = unit_payload * residual_ratio
                single_ratio = finite_float(
                    row["single_unit_ratio_including_shared_parameters"],
                    "offline single-unit ratio",
                )
                if not close(
                    single_ratio,
                    unit_source / (unit_payload + shared),
                    rtol=1e-7,
                ):
                    raise ValueError(
                        f"{row['method']}: single-unit ratio is inconsistent"
                    )
                dataset_ratio = finite_float(
                    row["dataset_ratio_including_shared_parameters_once"],
                    "offline dataset ratio",
                )
                if not close(
                    dataset_ratio,
                    unit_source * 7200 / (evaluated_payload + shared),
                    rtol=1e-7,
                ):
                    raise ValueError(f"{row['method']}: dataset ratio is inconsistent")
        params_path = (
            Path(training.metadata["params_path"]) if training.complete else None
        )
        validate_run_info(info, paths["config"], params_path=params_path)
        validate_training_links(
            info,
            training,
            require_implementation_match=True,
        )
        if list(info.get("prompt_ids", [])) != list(range(10)):
            raise ValueError("offline run_info prompt IDs are not 0..9")
        if (
            not isinstance(info.get("prompt_list"), list)
            or len(info["prompt_list"]) != 10
        ):
            raise ValueError("offline run_info prompt cohort is not 10 prompts")
        normalized_rows = []
        for raw in rows:
            row = dict(raw)
            if row["method"] == "l8c64_int4_base":
                row["scheme"] = "rvq_base"
                row["scheme_zh"] = "L8C64 INT4 RVQ base"
            else:
                scheme = row["method"].rsplit("_b", 1)[0]
                row["scheme"] = scheme
                row["scheme_zh"] = SCHEME_ZH[scheme]
            normalized_rows.append(row)
        scheme_order = {
            "rvq_base": -1,
            **{value: index for index, value in enumerate(SCHEMES)},
        }
        section.rows = sorted(
            normalized_rows,
            key=lambda row: (
                scheme_order[row["scheme"]],
                int_value(row["bits"], "offline bits"),
            ),
        )
        section.metadata = {
            "coverage": dict(coverage),
            "aggregation": summary.get("aggregation"),
            "attention_scope": summary.get("attention_scope"),
            "run_info": dict(info),
        }
        section.complete = True
    except Exception as error:
        section.errors.append(f"{type(error).__name__}: {error}")
        section.rows = []
    return section


def validate_exact_scope_config(config: Mapping[str, Any]) -> None:
    """Validate every exact-replay property stated as fact in the report."""

    accounting = config.get("attention_accounting")
    if not isinstance(accounting, Mapping):
        raise ValueError("exact config lacks attention_accounting")
    expected = {
        "hook_stage": "pre_positional_historical_only",
        "current_segment": "unchanged BF16 from the same attention call",
        "processor_semantics": (
            "two non-causal SDPA halves: standard RoPE and camera PRoPE"
        ),
        "camera_output": (
            "includes exact camera apply_fn_o-equivalent matrix; learned "
            "to_out layers unavailable"
        ),
    }
    for key, expected_value in expected.items():
        if accounting.get(key) != expected_value:
            raise ValueError(f"exact attention scope {key} differs from report claim")
    if set(accounting.get("interventions", ())) != set(INTERVENTIONS):
        raise ValueError("exact attention intervention grid differs")
    rvq = config.get("rvq")
    if not isinstance(rvq, Mapping):
        raise ValueError("exact config lacks RVQ protocol")
    expected_rvq = {
        "levels": 8,
        "codebook_size": 64,
        "dimension": 128,
        "heads": 24,
        "runtime_codebook_storage": "int4",
        "scale_storage_dtype": "uint4",
    }
    for key, expected_value in expected_rvq.items():
        if rvq.get(key) != expected_value:
            raise ValueError(f"exact RVQ {key} differs from report protocol")
    if config.get("verify_capture_hashes") is not True:
        raise ValueError("exact run did not require capture hash verification")


def audit_exact(
    root: Path,
    training: Section,
    *,
    exact_attention_dir: Path | None = None,
    selection_policy: str | None = None,
) -> Section:
    section = Section("exact_attention")
    if exact_attention_dir is None:
        directory, resolved_policy = resolve_exact_attention_dir(root)
    else:
        directory = exact_attention_dir.resolve()
        resolved_policy = selection_policy or "explicit_caller"
    paths = {
        "summary_csv": directory / "tables" / "prompt_bootstrap_summary.csv",
        "summary_json": directory / "tables" / "prompt_bootstrap_summary.json",
        "prompt_macro": directory / "tables" / "prompt_macro_metrics.csv",
        "raw_metrics": directory / "tables" / "exact_attention_metrics.csv",
        "config": directory / "config.json",
        "run_info": directory / "run_info" / "aggregate.json",
    }
    rows = load_required(
        section, paths["summary_csv"], "exact-attention summary", read_csv
    )
    summary = load_required(
        section, paths["summary_json"], "exact-attention summary metadata", read_json
    )
    prompt_rows = load_required(
        section, paths["prompt_macro"], "exact-attention prompt macro", read_csv
    )
    # Hash the raw table for evidence but do not parse it into memory.
    raw_marker = load_required(
        section, paths["raw_metrics"], "exact-attention raw metrics", lambda p: p
    )
    config = load_required(
        section, paths["config"], "exact-attention config", read_json
    )
    info = load_required(
        section, paths["run_info"], "exact-attention provenance", read_json
    )
    if section.missing or section.errors:
        return section
    try:
        assert raw_marker is not None
        assert isinstance(rows, list)
        assert isinstance(summary, Mapping)
        assert isinstance(prompt_rows, list)
        assert isinstance(config, Mapping)
        assert isinstance(info, Mapping)
        validate_exact_scope_config(config)
        require_fields(rows, EXACT_FIELDS, "exact attention summary")
        configured_schemes = tuple(config.get("schemes", ()))
        configured_bits = tuple(
            int_value(value, "exact configured bits")
            for value in config.get("bits", ())
        )
        if set(configured_schemes) != set(SCHEMES):
            raise ValueError("exact config does not cover all four schemes")
        if set(configured_bits) != set(BITS):
            raise ValueError("exact config does not cover prefix bits 1..4")
        summary_schema = summary.get("schema_version")
        if summary_schema == (
            "truncatable_residual_exact_attention.prompt_bootstrap_summary.v2"
        ):
            method_points = {("rvq_base", 0)} | {
                (scheme, bits)
                for scheme in configured_schemes
                for bits in configured_bits
            }
        elif summary_schema == (
            "truncatable_residual_exact_attention.prompt_bootstrap_summary.v1"
        ):
            method_points = {
                (scheme, bits)
                for scheme in configured_schemes
                for bits in configured_bits
            }
        else:
            raise ValueError("unexpected exact-attention summary schema")
        expected = {
            (scheme, bits, half, intervention)
            for scheme, bits in method_points
            for half in ATTENTION_HALVES
            for intervention in INTERVENTIONS
        }
        keys = {
            (
                row["scheme"],
                int_value(row["bits"], "exact bits"),
                row["attention_half"],
                row["intervention"],
            )
            for row in rows
        }
        if keys != expected or len(rows) != len(expected):
            raise ValueError(
                f"exact summary coverage is {len(keys)}/{len(expected)} rows"
            )
        if len(prompt_rows) != len(expected) * 10:
            raise ValueError(
                f"exact prompt macro has {len(prompt_rows)} rows, "
                f"expected {len(expected) * 10}"
            )
        prompt_keys = {
            (
                row["scheme"],
                int_value(row["bits"], "exact prompt bits"),
                row["attention_half"],
                row["intervention"],
                int_value(row["prompt_id"], "exact prompt ID"),
            )
            for row in prompt_rows
        }
        expected_prompt_keys = {
            (scheme, bits, half, intervention, prompt_id)
            for scheme, bits, half, intervention in expected
            for prompt_id in range(10)
        }
        if prompt_keys != expected_prompt_keys or len(prompt_rows) != len(prompt_keys):
            raise ValueError("exact prompt macro unique-key coverage differs")
        for row in prompt_rows:
            for metric in (
                "historical_K_mse",
                "historical_V_mse",
                "attention_kl_reference_to_candidate",
                "attention_output_mse",
            ):
                finite_float(row[metric], f"exact prompt {metric}")
        for row in rows:
            if int_value(row["n_prompts"], "exact n_prompts") != 10:
                raise ValueError("exact attention aggregation is not 10-prompt")
            for name in (
                "historical_K_mse_mean",
                "historical_V_mse_mean",
                "attention_kl_reference_to_candidate_mean",
                "attention_output_mse_mean",
            ):
                finite_float(row[name], f"exact {name}")
        summary_json_rows = summary.get("rows")
        if not isinstance(summary_json_rows, list) or len(summary_json_rows) != len(
            expected
        ):
            raise ValueError("exact-attention summary JSON row coverage differs")
        csv_by_key = {
            (
                row["scheme"],
                int_value(row["bits"], "exact CSV bits"),
                row["attention_half"],
                row["intervention"],
            ): row
            for row in rows
        }
        json_by_key = {
            (
                row["scheme"],
                int_value(row["bits"], "exact JSON bits"),
                row["attention_half"],
                row["intervention"],
            ): row
            for row in summary_json_rows
        }
        if set(csv_by_key) != set(json_by_key):
            raise ValueError("exact-attention summary CSV/JSON keys differ")
        for key in csv_by_key:
            for metric in (
                "historical_K_mse_mean",
                "historical_V_mse_mean",
                "attention_kl_reference_to_candidate_mean",
                "attention_output_mse_mean",
            ):
                if not close(
                    finite_float(csv_by_key[key][metric], metric),
                    finite_float(json_by_key[key][metric], metric),
                ):
                    raise ValueError(
                        f"exact summary CSV/JSON differ for {key}/{metric}"
                    )
        prompt_by_key: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        for row in prompt_rows:
            prompt_by_key[
                (
                    str(row["scheme"]),
                    int_value(row["bits"], "exact prompt bits"),
                    str(row["attention_half"]),
                    str(row["intervention"]),
                )
            ].append(row)
        mean_sources = (
            ("historical_K_mse_mean", "historical_K_mse"),
            ("historical_V_mse_mean", "historical_V_mse"),
            (
                "attention_kl_reference_to_candidate_mean",
                "attention_kl_reference_to_candidate",
            ),
            ("attention_output_mse_mean", "attention_output_mse"),
        )
        for key, csv_row in csv_by_key.items():
            prompts_for_key = prompt_by_key[key]
            if len(prompts_for_key) != 10:
                raise ValueError(f"exact prompt group is incomplete for {key}")
            for summary_metric, prompt_metric in mean_sources:
                prompt_mean = sum(
                    finite_float(row[prompt_metric], f"exact {prompt_metric}")
                    for row in prompts_for_key
                ) / len(prompts_for_key)
                if not close(
                    finite_float(csv_row[summary_metric], summary_metric),
                    prompt_mean,
                    rtol=1e-7,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"exact summary mean differs from prompt macro for "
                        f"{key}/{summary_metric}"
                    )
            dynamic_mean_fields = [
                field_name
                for field_name in csv_row
                if field_name.endswith("_mean")
                and field_name.removesuffix("_mean") in prompts_for_key[0]
            ]
            for summary_metric in dynamic_mean_fields:
                prompt_metric = summary_metric.removesuffix("_mean")
                prompt_values = [
                    optional_float(row.get(prompt_metric)) for row in prompts_for_key
                ]
                summary_value = optional_float(csv_row.get(summary_metric))
                if all(value is None for value in prompt_values):
                    if summary_value is not None:
                        raise ValueError(
                            f"exact optional mean should be absent for "
                            f"{key}/{summary_metric}"
                        )
                    continue
                if any(value is None for value in prompt_values):
                    raise ValueError(
                        f"exact prompt metric is partially absent for "
                        f"{key}/{prompt_metric}"
                    )
                assert summary_value is not None
                prompt_mean = sum(
                    value for value in prompt_values if value is not None
                ) / len(prompt_values)
                if not close(
                    summary_value,
                    prompt_mean,
                    rtol=1e-7,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"exact dynamic summary mean differs from prompt macro "
                        f"for {key}/{summary_metric}"
                    )
            if (
                int_value(csv_row["bootstrap_samples"], "exact bootstrap samples")
                != 10_000
                or int_value(csv_row["bootstrap_seed"], "exact bootstrap seed")
                != 20_260_722
                or csv_row["statistical_unit"] != "prompt"
            ):
                raise ValueError("exact bootstrap/statistical-unit protocol differs")
        if summary.get("config_hash") != sha256_file(paths["config"]):
            raise ValueError("exact-attention summary config hash mismatch")
        params_path = (
            Path(training.metadata["params_path"]) if training.complete else None
        )
        validate_run_info(info, paths["config"], params_path=params_path)
        validate_training_links(
            info,
            training,
            require_implementation_match=False,
        )
        if training.complete:
            if config.get("codec_params_hash") != training.metadata["params_hash"]:
                raise ValueError("exact config codec parameter hash mismatch")
            if config.get("selection_hash") != training.metadata["selection_hash"]:
                raise ValueError("exact config selection hash mismatch")
            if (
                config.get("selected_hadamard")
                != training.metadata["selected_hadamard"]
            ):
                raise ValueError("exact config Hadamard selection mismatch")
        prompt_ids = tuple(
            int_value(value, "exact configured prompt")
            for value in config.get("prompts", ())
        )
        layers = tuple(
            int_value(value, "exact configured layer")
            for value in config.get("layers", ())
        )
        if sorted(prompt_ids) != list(range(10)):
            raise ValueError("exact attention config prompts are not 0..9")
        if sorted(layers) != [0, 10, 20, 29]:
            raise ValueError("exact attention did not use layers 0,10,20,29")
        expected_units = len(prompt_ids) * len(layers)
        captures = config.get("captures")
        if not isinstance(captures, list) or len(captures) != expected_units:
            raise ValueError(
                "exact config capture count differs from prompt/layer grid"
            )
        capture_keys = {
            (
                int_value(item["prompt_id"], "exact capture prompt"),
                int_value(item["layer"], "exact capture layer"),
            )
            for item in captures
            if isinstance(item, Mapping)
        }
        expected_capture_keys = {
            (prompt_id, layer) for prompt_id in prompt_ids for layer in layers
        }
        if capture_keys != expected_capture_keys or len(capture_keys) != len(captures):
            raise ValueError("exact capture prompt/layer grid differs")
        source_info_cache: dict[Path, Mapping[str, Any]] = {}
        source_evidence_paths: set[Path] = set()
        for item in captures:
            if not isinstance(item, Mapping):
                raise ValueError("exact capture record is not an object")
            prompt_id = int_value(item["prompt_id"], "exact capture prompt")
            layer = int_value(item["layer"], "exact capture layer")
            capture_path = Path(str(item["capture_path"])).resolve()
            if not capture_path.is_file():
                raise ValueError(f"exact source capture is absent: {capture_path}")
            source_info_path = validate_recorded_file(
                item,
                path_key="source_run_info",
                hash_key="source_run_info_hash",
                label=f"exact capture p{prompt_id}/l{layer} source run_info",
            )
            if source_info_path not in source_evidence_paths:
                section.evidence.append(
                    evidence_entry(
                        source_info_path,
                        section.name,
                        f"exact capture source provenance prompt {prompt_id}",
                    )
                )
                source_evidence_paths.add(source_info_path)
            source_info = source_info_cache.get(source_info_path)
            if source_info is None:
                loaded = read_json(source_info_path)
                if not isinstance(loaded, Mapping):
                    raise ValueError("exact capture source run_info is malformed")
                source_info = loaded
                source_info_cache[source_info_path] = source_info
            if (
                source_info.get("source_capture_is_bf16") is not True
                or source_info.get("kv_cache_mode") != "pre_kv_reconstruct"
                or int_value(
                    source_info.get("prompt_id"),
                    "exact source prompt ID",
                )
                != prompt_id
            ):
                raise ValueError("exact capture source BF16/prompt provenance differs")
            source_capture_rows = source_info.get("capture_files")
            if not isinstance(source_capture_rows, list):
                raise ValueError("exact source run_info lacks capture_files")
            matching = [
                row
                for row in source_capture_rows
                if isinstance(row, Mapping)
                and Path(str(row.get("path"))).resolve() == capture_path
            ]
            if len(matching) != 1:
                raise ValueError(
                    "exact source run_info does not uniquely register capture"
                )
            source_capture = matching[0]
            expected_capture_hash = str(item.get("capture_sha256"))
            if (
                source_capture.get("sha256") != expected_capture_hash
                or not is_sha256(expected_capture_hash)
                or int_value(
                    source_capture.get("layer"),
                    "exact source capture layer",
                )
                != layer
                or int_value(
                    source_capture.get("history_tokens"),
                    "exact source history tokens",
                )
                != 31_680
                or int_value(
                    source_capture.get("current_tokens"),
                    "exact source current tokens",
                )
                != 3_520
                or source_capture.get("alignment") != "exact_same_call_same_indices"
            ):
                raise ValueError(
                    "exact source capture metadata differs from replay scope"
                )
            recorded_bytes = int_value(
                source_capture.get("bytes"),
                "exact source capture bytes",
            )
            if capture_path.stat().st_size != recorded_bytes:
                raise ValueError("exact source capture size changed after replay")
        heads = int_value(
            config.get("rvq", {}).get("heads", 24),
            "exact configured heads",
        )
        expected_raw_rows = expected_units * heads * len(expected)
        expected_prompt_rows = len(expected) * len(prompt_ids)
        if int(info.get("summary_rows", -1)) != len(expected):
            raise ValueError("exact run_info summary row count mismatch")
        if int(info.get("units", -1)) != expected_units:
            raise ValueError("exact run_info unit count differs from config grid")
        if int(info.get("raw_rows", -1)) != expected_raw_rows:
            raise ValueError(
                "exact run_info raw row count differs from the dynamic "
                f"Cartesian grid ({expected_raw_rows})"
            )
        if int(info.get("prompt_macro_rows", -1)) != expected_prompt_rows:
            raise ValueError(
                "exact run_info prompt macro row count differs from the "
                f"dynamic grid ({expected_prompt_rows})"
            )
        if list(info.get("prompt_ids", [])) != list(prompt_ids):
            raise ValueError("exact run_info prompt IDs differ from config")
        info_prompts = info.get("prompt_list")
        if not isinstance(info_prompts, list) or len(info_prompts) != 10:
            raise ValueError("exact run_info prompt cohort is not 10 prompts")
        macro_prompt_by_id: dict[int, str] = {}
        for row in prompt_rows:
            prompt_id = int_value(row["prompt_id"], "exact prompt ID")
            prompt = str(row["prompt"])
            previous = macro_prompt_by_id.setdefault(prompt_id, prompt)
            if previous != prompt:
                raise ValueError("exact prompt text differs within prompt ID")
        if [macro_prompt_by_id[prompt_id] for prompt_id in prompt_ids] != [
            str(value) for value in info_prompts
        ]:
            raise ValueError("exact prompt macro text differs from run_info")
        scheme_order = {
            "rvq_base": -1,
            **{value: index for index, value in enumerate(SCHEMES)},
        }
        half_order = {value: index for index, value in enumerate(ATTENTION_HALVES)}
        intervention_order = {value: index for index, value in enumerate(INTERVENTIONS)}
        section.rows = sorted(
            (dict(row) for row in rows),
            key=lambda row: (
                scheme_order[row["scheme"]],
                int_value(row["bits"], "exact bits"),
                half_order[row["attention_half"]],
                intervention_order[row["intervention"]],
            ),
        )
        section.metadata = {
            "source_directory": str(directory),
            "source_directory_selection_policy": resolved_policy,
            "source_directory_fingerprint": stable_directory_fingerprint(directory),
            "layers": list(config.get("layers", [])),
            "prompts": list(config.get("prompts", [])),
            "hook_stage": config.get("attention_accounting", {}).get("hook_stage"),
            "current_segment": config.get("attention_accounting", {}).get(
                "current_segment"
            ),
            "camera_output_scope": config.get("attention_accounting", {}).get(
                "camera_output"
            ),
            "processor_semantics": config.get("attention_accounting", {}).get(
                "processor_semantics"
            ),
            "pure_rvq_base_present": ("rvq_base", 0) in method_points,
            "dynamic_expected_counts": {
                "units": expected_units,
                "heads": heads,
                "method_points": len(method_points),
                "summary_rows": len(expected),
                "prompt_macro_rows": expected_prompt_rows,
                "raw_rows": expected_raw_rows,
            },
            "prompt_rows": [dict(row) for row in prompt_rows],
            "equal_payload_paired": exact_equal_payload_comparisons(rows, prompt_rows),
            "run_info": dict(info),
        }
        section.complete = True
    except Exception as error:
        section.errors.append(f"{type(error).__name__}: {error}")
        section.rows = []
    return section


def latest_video_summary(root: Path) -> tuple[Path | None, int | None]:
    candidates = []
    for path in root.glob("summary*.json"):
        if not path.is_file():
            continue
        try:
            payload = read_json(path)
            version = int(payload.get("summary_version", -1))
        except Exception:
            continue
        if payload.get("schema_version") == (
            "truncatable_residual_video_benchmark.summary.v1"
        ):
            candidates.append((version, path))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])[1], max(
        item[0] for item in candidates
    )


def candidate_setting(scheme: str, bits: int) -> str:
    return f"l8c64_rvq_residual_{scheme}_b{bits}"


def validate_vbench_source_bundle(
    record: Mapping[str, Any],
    *,
    expected_setting: str,
    expected_dimension: str,
    expected_videos: Mapping[int, Path],
    expected_prompts: Sequence[str],
    expected_values: Mapping[int, float],
) -> tuple[Path, Path]:
    """Revalidate a VBench result and its prompt/video binding from source."""

    label = f"VBench source {expected_setting}/{expected_dimension}"
    if (
        record.get("setting") != expected_setting
        or record.get("dimension") != expected_dimension
    ):
        raise ValueError(f"{label} setting/dimension metadata differs")
    result_path = validate_recorded_file(
        record,
        path_key="eval_results_path",
        hash_key="eval_results_sha256",
        label=f"{label} eval result",
    )
    suffix = "_eval_results.json"
    if not result_path.name.endswith(suffix):
        raise ValueError(f"{label} has an unexpected eval-result filename")
    expected_full_info = result_path.with_name(
        result_path.name[: -len(suffix)] + "_full_info.json"
    )
    full_info_path = validate_recorded_file(
        record,
        path_key="full_info_path",
        hash_key="full_info_sha256",
        label=f"{label} full-info",
        expected_path=expected_full_info,
    )
    if set(expected_videos) != set(range(10)):
        raise ValueError(f"{label} expected-video mapping is incomplete")
    if len(expected_prompts) != 10 or set(expected_values) != set(range(10)):
        raise ValueError(f"{label} prompt/value mapping is incomplete")

    result = read_json(result_path)
    if not isinstance(result, Mapping):
        raise ValueError(f"{label} eval result is not an object")
    dimension_payload = result.get(expected_dimension)
    if (
        not isinstance(dimension_payload, list)
        or len(dimension_payload) < 2
        or not isinstance(dimension_payload[1], list)
        or len(dimension_payload[1]) != 10
    ):
        raise ValueError(f"{label} eval result has malformed prompt details")
    observed_result_ids: set[int] = set()
    scale = VBENCH_PER_PROMPT_FIXED_SCALE[expected_dimension]
    for detail in dimension_payload[1]:
        if not isinstance(detail, Mapping):
            raise ValueError(f"{label} eval result detail is malformed")
        video = Path(str(detail.get("video_path", ""))).resolve()
        try:
            prompt_id = int(video.stem.split("-", 1)[0])
        except ValueError as error:
            raise ValueError(f"{label} result has invalid video path") from error
        if (
            prompt_id not in range(10)
            or prompt_id in observed_result_ids
            or video != expected_videos[prompt_id].resolve()
        ):
            raise ValueError(f"{label} result prompt/video binding differs")
        value = (
            finite_float(
                detail.get("video_results"),
                f"{label} prompt {prompt_id} value",
            )
            * scale
        )
        if not close(
            value,
            finite_float(
                expected_values[prompt_id],
                f"{label} expected prompt {prompt_id} value",
            ),
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(f"{label} result value differs from prompt table")
        observed_result_ids.add(prompt_id)
    aggregate = finite_float(dimension_payload[0], f"{label} aggregate")
    expected_aggregate = (
        sum(
            finite_float(value, f"{label} expected value")
            for value in expected_values.values()
        )
        / 10.0
    )
    if not close(aggregate, expected_aggregate, rtol=1e-7, atol=1e-10):
        raise ValueError(f"{label} aggregate differs from prompt table")

    full_info = read_json(full_info_path)
    if not isinstance(full_info, list) or len(full_info) != 10:
        raise ValueError(f"{label} full-info does not contain ten prompts")
    observed_full_info_ids: set[int] = set()
    for entry in full_info:
        if not isinstance(entry, Mapping) or entry.get("dimension") != [
            expected_dimension
        ]:
            raise ValueError(f"{label} full-info dimension differs")
        videos = entry.get("video_list")
        if not isinstance(videos, list) or len(videos) != 1:
            raise ValueError(f"{label} full-info video list is malformed")
        video = Path(str(videos[0])).resolve()
        try:
            prompt_id = int(video.stem.split("-", 1)[0])
        except ValueError as error:
            raise ValueError(f"{label} full-info has invalid video path") from error
        if (
            prompt_id not in range(10)
            or prompt_id in observed_full_info_ids
            or video != expected_videos[prompt_id].resolve()
            or entry.get("prompt_en") != expected_prompts[prompt_id]
        ):
            raise ValueError(f"{label} full-info prompt/video binding differs")
        observed_full_info_ids.add(prompt_id)
    if observed_result_ids != set(range(10)) or observed_full_info_ids != set(
        range(10)
    ):
        raise ValueError(f"{label} source prompt IDs are incomplete")
    return result_path, full_info_path


def validate_video_setting_metadata(
    row: Mapping[str, Any],
    *,
    label: str,
    selected_hadamard: Mapping[str, Any],
) -> None:
    setting = str(row.get("setting"))
    scheme = str(row.get("scheme"))
    if setting == "bf16":
        if (
            scheme != "none"
            or optional_float(row.get("prefix_bits")) is not None
            or bool_value(row.get("hadamard"))
        ):
            raise ValueError(f"{label}: malformed BF16 setting metadata")
        return
    if setting == "l8c64_int4":
        if (
            scheme != "none"
            or int_value(row.get("prefix_bits"), f"{label} prefix bits") != 0
            or bool_value(row.get("hadamard"))
        ):
            raise ValueError(f"{label}: malformed L8 setting metadata")
        return
    if scheme not in SCHEMES:
        raise ValueError(f"{label}: unknown candidate scheme")
    bits = int_value(row.get("prefix_bits"), f"{label} prefix bits")
    if setting != candidate_setting(scheme, bits):
        raise ValueError(f"{label}: candidate setting/scheme/bits differ")
    if scheme not in selected_hadamard or bool_value(row.get("hadamard")) != bool_value(
        selected_hadamard[scheme]
    ):
        raise ValueError(f"{label}: candidate Hadamard metadata differs")


def expected_paired_specs() -> list[tuple[str, str, str, str]]:
    candidates = {
        (scheme, bits): candidate_setting(scheme, bits)
        for scheme in SCHEMES
        for bits in BITS
    }
    pairs: list[tuple[str, str, str]] = [
        (
            candidates[(scheme, bits)],
            "l8c64_int4",
            "residual_prefix_vs_pure_l8",
        )
        for scheme in SCHEMES
        for bits in BITS
    ]
    pairs.extend(
        (
            candidates[(scheme, bits)],
            candidates[(scheme, bits - 1)],
            "incremental_refinement_plane",
        )
        for scheme in SCHEMES
        for bits in (2, 3, 4)
    )
    pairs.extend(
        (
            candidates[(scheme, bits)],
            candidates[("nested_affine", bits)],
            "same_prefix_count_vs_nested_affine_not_rate_matched",
        )
        for bits in BITS
        for scheme in SCHEMES
        if scheme != "nested_affine"
    )
    return [
        (candidate, baseline, contrast, metric)
        for candidate, baseline, contrast in pairs
        for metric in PAIRED_METRICS
    ]


def expected_paired_keys() -> set[tuple[str, str, str, str]]:
    return set(expected_paired_specs())


def validate_candidate_config_audit(
    summary: Mapping[str, Any],
    *,
    expected_params_path: Path | None,
    expected_prompts: Sequence[str],
) -> list[dict[str, Any]]:
    audit = summary.get("candidate_task_config_audit")
    if not isinstance(audit, Mapping) or audit.get("status") != "passed":
        raise ValueError("video summary lacks passed per-task config audit")
    evidence = audit.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("video per-task config evidence is absent")
    expected = {
        (scheme, bits, prompt_id)
        for scheme in SCHEMES
        for bits in BITS
        for prompt_id in range(10)
    }
    keys = {
        (
            str(item["scheme"]),
            int_value(item["prefix_bits"], "config audit bits"),
            int_value(item["prompt_id"], "config audit prompt"),
        )
        for item in evidence
        if isinstance(item, Mapping)
    }
    if (
        len(evidence) != len(expected)
        or keys != expected
        or int(audit.get("validated_tasks", -1)) != len(expected)
        or int(audit.get("expected_tasks", -1)) != len(expected)
    ):
        raise ValueError("video per-task config audit coverage is not 160/160")
    normalized = []
    selected_hadamard = summary.get("selected_hadamard")
    if not isinstance(selected_hadamard, Mapping):
        raise ValueError("video summary lacks selected Hadamard mapping")
    for raw in evidence:
        if not isinstance(raw, Mapping) or raw.get("validation") != "passed":
            raise ValueError("video per-task config evidence contains failure")
        item = dict(raw)
        scheme = str(item["scheme"])
        bits = int_value(item["prefix_bits"], "config audit bits")
        prompt_id = int_value(item["prompt_id"], "config audit prompt")
        expected_setting = candidate_setting(scheme, bits)
        if item.get("setting") != expected_setting:
            raise ValueError("video config evidence setting is inconsistent")
        expected_hadamard = bool_value(selected_hadamard[scheme])
        if bool_value(item.get("hadamard")) != expected_hadamard:
            raise ValueError("video config evidence Hadamard choice differs")
        for path_key, hash_key in (
            ("metrics_path", "metrics_sha256"),
            ("config_path", "config_sha256"),
        ):
            path = Path(str(item[path_key]))
            if not path.is_file() or sha256_file(path) != item[hash_key]:
                raise ValueError(f"video config evidence changed after audit: {path}")
        config_path = Path(str(item["config_path"])).resolve()
        metrics_path = Path(str(item["metrics_path"])).resolve()
        config_payload = read_json(config_path)
        args = config_payload.get("args")
        if not isinstance(args, Mapping):
            raise ValueError(f"video task config lacks resolved args: {config_path}")
        expected_config_name = config_path.name
        expected_metrics_name = (
            "metrics" + expected_config_name.removeprefix("config")
        ).removesuffix(".json") + ".csv"
        required_args: dict[str, Any] = {
            "prompt_start": prompt_id,
            "num_prompts": 1,
            "output_dir": str(config_path.parent),
            "metrics_name": expected_metrics_name,
            "config_name": expected_config_name,
            "seed": 42,
            "setting": "kvrvq-i4",
            "num_frames": 173,
            "num_chunk": 11,
            "num_inference_steps": 50,
            "rvq_residual_quantize": True,
            "rvq_residual_codec": "truncatable",
            "truncatable_residual_scheme": scheme,
            "truncatable_residual_bits": bits,
            "truncatable_residual_hadamard": expected_hadamard,
            "truncatable_residual_seed": 20260722,
            "k_rvq_compute_dtype": "bf16",
            "v_rvq_compute_dtype": "bf16",
            "rvq_codebook_storage_dtype": "int4",
            "rvq_scale_storage_dtype": "uint4",
            "rvq_num_levels": 8,
            "kv_logical_head_dim": 256,
            "kv_quant_factor": 2,
        }
        if expected_params_path is not None:
            required_args["truncatable_residual_params_path"] = str(
                expected_params_path.resolve()
            )
        for key, expected_value in required_args.items():
            observed = args.get(key)
            if key in {
                "output_dir",
                "truncatable_residual_params_path",
            }:
                matches = (
                    Path(str(observed)).resolve() == Path(str(expected_value)).resolve()
                )
            else:
                matches = observed == expected_value
            if not matches:
                raise ValueError(
                    f"video task config {config_path} has {key}={observed!r}, "
                    f"expected {expected_value!r}"
                )
        if metrics_path.name != expected_metrics_name:
            raise ValueError("video task metrics/config attempt suffix differs")
        attempt_rows = read_csv(metrics_path)
        if len(attempt_rows) != 1:
            raise ValueError("video task metrics attempt is not one prompt")
        attempt = attempt_rows[0]
        if (
            int_value(attempt["prompt_id"], "attempt prompt") != prompt_id
            or int_value(attempt["seed"], "attempt seed") != 42
            or attempt["setting"] != "kvrvq-i4"
            or attempt["prompt"] != expected_prompts[prompt_id]
        ):
            raise ValueError("video task metrics/config prompt mapping differs")
        recorded_video = Path(str(attempt["video_path"])).resolve()
        if (
            recorded_video != config_path.parent / f"{prompt_id}-42.mp4"
            or not recorded_video.is_file()
        ):
            raise ValueError("video task metrics points to an invalid video")
        item["video_path"] = str(recorded_video)
        item["video_bytes"] = recorded_video.stat().st_size
        item["video_sha256"] = sha256_file(recorded_video)
        normalized.append(item)
    return normalized


def validate_normalized_similarity_payload_protocol(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Mapping[str, Any], Path]:
    """Require the standard LPIPS input contract and its executed evaluator."""

    if payload.get("schema_version") != "frame_aligned_video_similarity.v2":
        raise ValueError(f"{label} is not a normalized Similarity v2 payload")
    protocol = payload.get("metric_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"{label} lacks metric_protocol")
    expected_protocol_fields = {
        "protocol_id": STRICT_SIMILARITY_PROTOCOL_ID,
        "decoded_color_space": "RGB",
        "decoded_float_input_range": [0.0, 1.0],
        "frame_0_included": False,
        "per_video_aggregation": ("arithmetic mean over aligned decoded frames"),
        "dataset_aggregation": ("arithmetic mean over prompt-level metrics"),
        "psnr": {
            "data_range": 1.0,
            "aggregation": "mean of per-frame dB values",
        },
        "ssim": {
            "implementation": "torch_avg_pool_11x11_rgb",
            "data_range": 1.0,
        },
    }
    for field_name, expected in expected_protocol_fields.items():
        if protocol.get(field_name) != expected:
            raise ValueError(
                f"{label} normalized protocol field {field_name!r} differs"
            )
    lpips_protocol = protocol.get("lpips")
    expected_lpips = {
        "package": "lpips",
        "package_version": "0.1.4",
        "network": "vgg",
        "model_version": "0.1",
        "input_range_before_lpips": [0.0, 1.0],
        "normalize_input": True,
        "effective_input_range": [-1.0, 1.0],
        "input_contract": "standard_lpips_minus_one_to_one",
    }
    if not isinstance(lpips_protocol, Mapping):
        raise ValueError(f"{label} lacks LPIPS protocol evidence")
    if dict(lpips_protocol) != expected_lpips:
        raise ValueError(f"{label} is not standard normalized LPIPS 0.1.4/VGG")
    evaluator = payload.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError(f"{label} lacks evaluator evidence")
    evaluator_path = validate_recorded_file(
        evaluator,
        path_key="path",
        hash_key="sha256",
        label=f"{label} evaluator",
        expected_path=SIMILARITY_EVALUATOR,
    )
    return protocol, evaluator_path


def validate_similarity_summary_evidence_chain(
    summary: Mapping[str, Any],
    *,
    root: Path,
    config_path: Path,
    params_path: Path | None,
    root_run_info: Mapping[str, Any],
    expected_settings: set[str],
    expected_prompts: Sequence[str],
    expected_references: Mapping[int, Path],
) -> dict[str, Any]:
    """Validate summary -> completion -> run_info -> 17 source bundles."""

    artifact_directory = (root / STRICT_SIMILARITY_DIR_NAME).resolve()
    protocol = summary.get("similarity_metric_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("video summary lacks normalized similarity_metric_protocol")
    expected_protocol_keys = {
        "protocol_id",
        "artifact_directory",
        "lpips_input_range_before_normalization",
        "lpips_normalize_input",
        "lpips_effective_input_range",
        "frame_0_included",
        "settings",
        "prompts_per_setting",
        "evaluator_path",
        "evaluator_sha256",
        "run_info_path",
        "run_info_sha256",
        "completion_manifest_path",
        "completion_manifest_sha256",
    }
    if set(protocol) != expected_protocol_keys:
        raise ValueError("video summary normalized Similarity protocol fields differ")
    expected_scalars = {
        "protocol_id": STRICT_SIMILARITY_PROTOCOL_ID,
        "artifact_directory": str(artifact_directory),
        "lpips_input_range_before_normalization": [0.0, 1.0],
        "lpips_normalize_input": True,
        "lpips_effective_input_range": [-1.0, 1.0],
        "frame_0_included": False,
        "settings": len(expected_settings),
        "prompts_per_setting": 10,
    }
    for field_name, expected in expected_scalars.items():
        if protocol.get(field_name) != expected:
            raise ValueError(
                f"video summary Similarity protocol field {field_name!r} differs"
            )
    if not artifact_directory.is_dir():
        raise ValueError(
            f"normalized Similarity directory is absent: {artifact_directory}"
        )
    evaluator_path = validate_recorded_file(
        protocol,
        path_key="evaluator_path",
        hash_key="evaluator_sha256",
        label="video summary Similarity evaluator",
        expected_path=SIMILARITY_EVALUATOR,
    )
    run_info_path = validate_recorded_file(
        protocol,
        path_key="run_info_path",
        hash_key="run_info_sha256",
        label="video summary Similarity run_info",
        expected_path=artifact_directory / "run_info.json",
    )
    completion_path = validate_recorded_file(
        protocol,
        path_key="completion_manifest_path",
        hash_key="completion_manifest_sha256",
        label="video summary Similarity completion manifest",
        expected_path=artifact_directory / "completion_manifest.json",
    )

    similarity_info = read_json(run_info_path)
    if not isinstance(similarity_info, Mapping):
        raise ValueError("normalized Similarity run_info is not an object")
    if similarity_info.get("schema_version") != (
        "normalized_lpips_similarity.run_info.v2"
    ):
        raise ValueError("normalized Similarity run_info schema differs")
    validate_run_info(
        similarity_info,
        config_path,
        params_path=params_path,
    )
    for field_name in (
        "git_commit",
        "config_path",
        "config_hash",
        "codebook_path",
        "codebook_hash",
        "codec_params_path",
        "codec_params_hash",
        "hostname",
        "gpu_type",
        "seed",
    ):
        if similarity_info.get(field_name) != root_run_info.get(field_name):
            raise ValueError(
                f"normalized Similarity run_info field {field_name!r} "
                "differs from the video run"
            )
    if int_value(
        similarity_info.get("seed"), "Similarity seed"
    ) != 42 or similarity_info.get("prompt_list") != list(expected_prompts):
        raise ValueError("normalized Similarity run_info prompt/seed cohort differs")
    expected_run_protocol = {
        "protocol_id": STRICT_SIMILARITY_PROTOCOL_ID,
        "lpips_normalize_input": True,
        "input_range_before_lpips": [0.0, 1.0],
        "effective_input_range": [-1.0, 1.0],
        "frame_0_included": False,
        "evaluator_path": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
    }
    if similarity_info.get("metric_protocol") != expected_run_protocol:
        raise ValueError("normalized Similarity run_info metric protocol differs")
    reference_records = similarity_info.get("reference_videos")
    if not isinstance(reference_records, list) or len(reference_records) != 10:
        raise ValueError(
            "normalized Similarity run_info reference evidence is incomplete"
        )
    observed_reference_ids: set[int] = set()
    for raw in reference_records:
        if not isinstance(raw, Mapping):
            raise ValueError("normalized Similarity reference record is malformed")
        reference = validate_recorded_file(
            raw,
            path_key="path",
            hash_key="sha256",
            label="normalized Similarity BF16 reference",
        )
        try:
            prompt_id = int(reference.stem.rsplit("-", 1)[0])
        except ValueError as error:
            raise ValueError(
                "normalized Similarity reference lacks prompt ID"
            ) from error
        if (
            prompt_id not in range(10)
            or prompt_id in observed_reference_ids
            or reference != expected_references[prompt_id].resolve()
            or int_value(
                raw.get("bytes"),
                "normalized Similarity reference bytes",
            )
            != reference.stat().st_size
        ):
            raise ValueError("normalized Similarity reference path/hash/bytes differ")
        observed_reference_ids.add(prompt_id)
    if observed_reference_ids != set(range(10)):
        raise ValueError("normalized Similarity reference prompt IDs are incomplete")
    run_legacy = similarity_info.get("legacy_similarity")
    expected_legacy_directory = (root / LEGACY_SIMILARITY_DIR_NAME).resolve()
    if run_legacy != {
        "path": str(expected_legacy_directory),
        "status": "preserved_invalid_for_standard_lpips",
    }:
        raise ValueError("normalized Similarity run_info legacy invalidation differs")

    invalidation = summary.get("legacy_similarity_invalidation")
    if not isinstance(invalidation, Mapping):
        raise ValueError("video summary lacks legacy_similarity_invalidation")
    expected_invalidation_keys = {
        "status",
        "reason",
        "preserved_not_overwritten",
        "candidate_directory",
        "reused_l8_csv",
        "reused_l8_csv_sha256",
    }
    if set(invalidation) != expected_invalidation_keys:
        raise ValueError("video summary legacy Similarity invalidation fields differ")
    if (
        invalidation.get("status") != "invalid_for_standard_lpips"
        or invalidation.get("reason") != LEGACY_SIMILARITY_INVALIDATION_REASON
        or invalidation.get("preserved_not_overwritten") is not True
        or Path(str(invalidation.get("candidate_directory"))).resolve()
        != expected_legacy_directory
        or not expected_legacy_directory.is_dir()
    ):
        raise ValueError("video summary does not strictly invalidate legacy LPIPS")
    reused_baselines = root_run_info.get("reused_baselines")
    if not isinstance(reused_baselines, Mapping):
        raise ValueError(
            "video run_info lacks the reused baseline registry needed "
            "to bind legacy L8 Similarity"
        )
    raw_legacy_l8_csv = reused_baselines.get("l8c64_int4_similarity")
    if not isinstance(raw_legacy_l8_csv, str) or not raw_legacy_l8_csv:
        raise ValueError(
            "video run_info lacks the authoritative legacy L8 Similarity CSV path"
        )
    expected_legacy_l8_csv = Path(raw_legacy_l8_csv).resolve()
    legacy_l8_csv = validate_recorded_file(
        invalidation,
        path_key="reused_l8_csv",
        hash_key="reused_l8_csv_sha256",
        label="legacy L8 Similarity control",
        expected_path=expected_legacy_l8_csv,
    )

    completion = read_json(completion_path)
    if not isinstance(completion, Mapping):
        raise ValueError("normalized Similarity completion manifest is not an object")
    expected_completion_keys = {
        "schema_version",
        "metric_protocol_id",
        "settings",
        "prompts_per_setting",
        "run_info_path",
        "run_info_sha256",
        "artifacts",
    }
    if set(completion) != expected_completion_keys:
        raise ValueError("normalized Similarity completion manifest fields differ")
    if (
        completion.get("schema_version") != "normalized_lpips_similarity.completion.v2"
        or completion.get("metric_protocol_id") != STRICT_SIMILARITY_PROTOCOL_ID
        or int_value(completion.get("settings"), "Similarity settings")
        != len(expected_settings)
        or int_value(
            completion.get("prompts_per_setting"),
            "Similarity prompts per setting",
        )
        != 10
        or Path(str(completion.get("run_info_path"))).resolve() != run_info_path
        or completion.get("run_info_sha256") != sha256_file(run_info_path)
    ):
        raise ValueError("normalized Similarity completion metadata differs")
    completion_sources = completion.get("artifacts")
    summary_sources = summary.get("similarity_source_evidence")
    if not isinstance(completion_sources, list) or not isinstance(
        summary_sources, list
    ):
        raise ValueError("normalized Similarity source evidence is absent")

    def sources_by_setting(
        values: Sequence[Any],
        *,
        label: str,
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for raw in values:
            if not isinstance(raw, Mapping):
                raise ValueError(f"{label} source record is malformed")
            setting = str(raw.get("setting"))
            if setting in result:
                raise ValueError(f"{label} has duplicate setting {setting}")
            result[setting] = raw
        if set(result) != expected_settings or len(values) != len(expected_settings):
            raise ValueError(
                f"{label} coverage is not "
                f"{len(expected_settings)}/{len(expected_settings)}"
            )
        return result

    completion_by_setting = sources_by_setting(
        completion_sources,
        label="Similarity completion manifest",
    )
    summary_by_setting = sources_by_setting(
        summary_sources,
        label="video summary Similarity evidence",
    )
    for setting in expected_settings:
        if dict(completion_by_setting[setting]) != dict(summary_by_setting[setting]):
            raise ValueError(
                f"Similarity summary/completion evidence differs for {setting}"
            )
    return {
        "artifact_directory": artifact_directory,
        "protocol": dict(protocol),
        "run_info_path": run_info_path,
        "run_info": dict(similarity_info),
        "completion_path": completion_path,
        "completion": dict(completion),
        "source_by_setting": summary_by_setting,
        "legacy_invalidation": dict(invalidation),
        "legacy_directory": expected_legacy_directory,
        "legacy_l8_csv": legacy_l8_csv,
    }


def validate_normalized_similarity_source_bundle(
    source: Mapping[str, Any],
    *,
    expected_setting: str,
    artifact_directory: Path,
    expected_test_directory: Path,
    expected_test_videos: Mapping[int, Path],
    expected_references: Mapping[int, Path],
    expected_prompts: Sequence[str],
    summary_rows: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Revalidate one normalized JSON/CSV pair and its summary-table rows."""

    expected_source_keys = {
        "setting",
        "json_path",
        "json_sha256",
        "csv_path",
        "csv_sha256",
        "source",
        "metric_protocol_id",
        "lpips_protocol",
        "evaluator",
    }
    if set(source) != expected_source_keys:
        raise ValueError(f"Similarity source {expected_setting} evidence fields differ")
    if (
        source.get("setting") != expected_setting
        or source.get("source") != "strict_same_seed_BF16_comparison"
        or source.get("metric_protocol_id") != STRICT_SIMILARITY_PROTOCOL_ID
    ):
        raise ValueError(f"Similarity source {expected_setting} metadata differs")
    json_path = validate_recorded_file(
        source,
        path_key="json_path",
        hash_key="json_sha256",
        label=f"normalized Similarity JSON {expected_setting}",
    )
    csv_path = validate_recorded_file(
        source,
        path_key="csv_path",
        hash_key="csv_sha256",
        label=f"normalized Similarity CSV {expected_setting}",
    )
    allowed_stem = re.fullmatch(
        re.escape(expected_setting) + r"(?:_retry[1-9][0-9]*)?",
        json_path.stem,
    )
    if (
        json_path.parent != artifact_directory.resolve()
        or csv_path.parent != artifact_directory.resolve()
        or json_path.stem != csv_path.stem
        or allowed_stem is None
    ):
        raise ValueError(f"Similarity source {expected_setting} path binding differs")
    payload = read_json(json_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Similarity source {expected_setting} JSON is not an object")
    protocol, _evaluator_path = validate_normalized_similarity_payload_protocol(
        payload,
        label=f"Similarity source {expected_setting}",
    )
    if source.get("lpips_protocol") != protocol.get("lpips") or source.get(
        "evaluator"
    ) != payload.get("evaluator"):
        raise ValueError(
            f"Similarity source {expected_setting} protocol evidence differs"
        )
    if (
        payload.get("reference") != "bf16"
        or payload.get("test") != expected_test_directory.name
        or int_value(
            payload.get("num_videos"),
            f"Similarity {expected_setting} num_videos",
        )
        != 10
    ):
        raise ValueError(
            f"Similarity source {expected_setting} dataset binding differs"
        )
    human_label = payload.get("setting")
    if not isinstance(human_label, str) or not human_label:
        raise ValueError(
            f"Similarity source {expected_setting} lacks its display label"
        )
    rows = read_csv(csv_path)
    videos = payload.get("videos")
    if len(rows) != 10 or not isinstance(videos, list) or len(videos) != 10:
        raise ValueError(f"Similarity source {expected_setting} is not 10 prompts")
    csv_by_prompt: dict[int, Mapping[str, Any]] = {}
    json_by_prompt: dict[int, Mapping[str, Any]] = {}
    for raw in rows:
        prompt_id = int_value(
            raw.get("prompt_id"),
            f"Similarity {expected_setting} CSV prompt",
        )
        if prompt_id in csv_by_prompt:
            raise ValueError(
                f"Similarity source {expected_setting} has duplicate CSV prompt"
            )
        csv_by_prompt[prompt_id] = raw
    for raw in videos:
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Similarity source {expected_setting} video row is malformed"
            )
        prompt_id = int_value(
            raw.get("prompt_id"),
            f"Similarity {expected_setting} JSON prompt",
        )
        if prompt_id in json_by_prompt:
            raise ValueError(
                f"Similarity source {expected_setting} has duplicate JSON prompt"
            )
        json_by_prompt[prompt_id] = raw
    if (
        set(csv_by_prompt) != set(range(10))
        or set(json_by_prompt) != set(range(10))
        or set(summary_rows) != set(range(10))
    ):
        raise ValueError(
            f"Similarity source {expected_setting} prompt IDs are incomplete"
        )
    for prompt_id in range(10):
        csv_row = csv_by_prompt[prompt_id]
        json_row = json_by_prompt[prompt_id]
        summary_row = summary_rows[prompt_id]
        if (
            csv_row.get("prompt") != expected_prompts[prompt_id]
            or json_row.get("prompt") != expected_prompts[prompt_id]
            or csv_row.get("setting") != human_label
            or json_row.get("setting") != human_label
        ):
            raise ValueError(
                f"Similarity source {expected_setting}/{prompt_id} "
                "prompt/display label differs"
            )
        expected_reference = expected_references[prompt_id].resolve()
        expected_test = expected_test_videos[prompt_id].resolve()
        for row_label, row in (
            ("CSV", csv_row),
            ("JSON", json_row),
            ("summary", summary_row),
        ):
            if (
                Path(str(row.get("reference_video"))).resolve() != expected_reference
                or Path(str(row.get("test_video"))).resolve() != expected_test
                or int_value(
                    row.get("frames"),
                    f"Similarity {row_label} frames",
                )
                != 173
                or int_value(
                    row.get("compared_frames"),
                    f"Similarity {row_label} compared frames",
                )
                != 172
            ):
                raise ValueError(
                    f"Similarity source {expected_setting}/{prompt_id} "
                    f"{row_label} video/frame binding differs"
                )
        if summary_row.get("prompt") != expected_prompts[prompt_id]:
            raise ValueError(
                f"Similarity summary prompt differs at {expected_setting}/{prompt_id}"
            )
        for metric in ("psnr", "ssim", "lpips", "mse"):
            csv_value = finite_float(
                csv_row.get(metric),
                f"Similarity {expected_setting} CSV {metric}",
            )
            if csv_value < 0:
                raise ValueError(f"Similarity {expected_setting} has negative {metric}")
            for row_label, row in (
                ("JSON", json_row),
                ("summary", summary_row),
            ):
                value = finite_float(
                    row.get(metric),
                    f"Similarity {expected_setting} {row_label} {metric}",
                )
                if not close(
                    value,
                    csv_value,
                    rtol=1e-9,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"Similarity {expected_setting}/{prompt_id} "
                        f"{metric} differs across CSV/{row_label}"
                    )
    for metric in SIMILARITY_METRICS:
        mean = (
            sum(
                finite_float(row[metric], f"Similarity source {metric}") for row in rows
            )
            / 10.0
        )
        reported = finite_float(
            payload.get(f"{metric}_mean"),
            f"Similarity source {expected_setting} {metric}_mean",
        )
        if not close(mean, reported, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"Similarity source {expected_setting} {metric} mean differs"
            )
    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "rows": rows,
    }


def validate_legacy_similarity_controls(
    *,
    legacy_directory: Path,
    legacy_l8_csv: Path,
    strict_rows_by_setting: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Prove PSNR/SSIM/MSE invariance without accepting legacy LPIPS."""

    checked_paths: list[Path] = []
    compared_rows = 0
    changed_lpips_rows = 0
    max_psnr_delta = 0.0
    max_ssim_delta = 0.0
    max_mse_delta = 0.0
    for setting, strict_rows in strict_rows_by_setting.items():
        legacy_path = (
            legacy_l8_csv
            if setting == "l8c64_int4"
            else legacy_directory / f"{setting}.csv"
        ).resolve()
        if not legacy_path.is_file():
            raise ValueError(f"legacy Similarity control is absent: {legacy_path}")
        legacy_rows = read_csv(legacy_path)
        legacy_by_prompt = {
            int_value(
                row.get("prompt_id"),
                f"legacy Similarity {setting} prompt",
            ): row
            for row in legacy_rows
        }
        strict_by_prompt = {
            int_value(
                row.get("prompt_id"),
                f"strict Similarity {setting} prompt",
            ): row
            for row in strict_rows
        }
        if (
            len(legacy_rows) != 10
            or len(legacy_by_prompt) != 10
            or set(legacy_by_prompt) != set(range(10))
            or set(strict_by_prompt) != set(range(10))
        ):
            raise ValueError(
                f"legacy Similarity control coverage differs for {setting}"
            )
        for prompt_id in range(10):
            old = legacy_by_prompt[prompt_id]
            strict = strict_by_prompt[prompt_id]
            for field_name in (
                "prompt",
                "reference_video",
                "test_video",
                "frames",
                "compared_frames",
            ):
                if str(old.get(field_name)) != str(strict.get(field_name)):
                    raise ValueError(
                        f"legacy/strict Similarity binding differs for "
                        f"{setting}/{prompt_id}/{field_name}"
                    )
            deltas = {
                metric: abs(
                    finite_float(
                        old.get(metric),
                        f"legacy Similarity {metric}",
                    )
                    - finite_float(
                        strict.get(metric),
                        f"strict Similarity {metric}",
                    )
                )
                for metric in ("psnr", "ssim", "mse")
            }
            max_psnr_delta = max(max_psnr_delta, deltas["psnr"])
            max_ssim_delta = max(max_ssim_delta, deltas["ssim"])
            max_mse_delta = max(max_mse_delta, deltas["mse"])
            if any(delta > 1e-12 for delta in deltas.values()):
                raise ValueError(
                    f"legacy normalization unexpectedly changed "
                    f"PSNR/SSIM/MSE for {setting}/{prompt_id}"
                )
            if not close(
                finite_float(old.get("lpips"), "legacy LPIPS"),
                finite_float(strict.get("lpips"), "normalized LPIPS"),
                rtol=1e-12,
                atol=1e-12,
            ):
                changed_lpips_rows += 1
            compared_rows += 1
        checked_paths.append(legacy_path)
    if compared_rows != 170 or changed_lpips_rows == 0:
        raise ValueError(
            "legacy/normalized Similarity control did not cover 170 rows "
            "with a changed LPIPS result"
        )
    return {
        "rows": compared_rows,
        "legacy_lpips_values_changed_rows": changed_lpips_rows,
        "psnr_max_abs_delta": max_psnr_delta,
        "ssim_max_abs_delta": max_ssim_delta,
        "mse_max_abs_delta": max_mse_delta,
        "legacy_csv_paths": checked_paths,
    }


def audit_video(root: Path, training: Section) -> Section:
    section = Section("video_similarity_vbench")
    summary_path, version = latest_video_summary(root)
    if summary_path is None or version is None:
        section.missing.append(str(root / "summary.json"))
        return section
    suffix = "" if version == 1 else f"_v{version}"
    paths = {
        "summary": summary_path,
        "metrics": root / "tables" / f"metrics{suffix}.csv",
        "runtime": root / "tables" / f"runtime_per_prompt{suffix}.csv",
        "similarity": root / "tables" / f"similarity_per_prompt{suffix}.csv",
        "vbench": root / "tables" / f"vbench_per_prompt{suffix}.csv",
        "paired": root / "tables" / f"paired_comparisons{suffix}.csv",
        "coverage": root / "vbench" / "custom_input_coverage.json",
        "config": root / "config.json",
        "run_info": root / "run_info.json",
    }
    summary = load_required(section, paths["summary"], "video summary", read_json)
    metrics = load_required(section, paths["metrics"], "video scalar metrics", read_csv)
    runtime = load_required(section, paths["runtime"], "video runtime", read_csv)
    similarity = load_required(
        section, paths["similarity"], "Similarity prompt metrics", read_csv
    )
    vbench = load_required(section, paths["vbench"], "VBench prompt metrics", read_csv)
    paired = load_required(
        section, paths["paired"], "paired quality comparisons", read_csv
    )
    coverage = load_required(
        section, paths["coverage"], "VBench custom-input coverage", read_json
    )
    config = load_required(section, paths["config"], "video config", read_json)
    info = load_required(section, paths["run_info"], "video provenance", read_json)
    if section.missing or section.errors:
        return section
    try:
        assert isinstance(summary, Mapping)
        assert isinstance(metrics, list)
        assert isinstance(runtime, list)
        assert isinstance(similarity, list)
        assert isinstance(vbench, list)
        assert isinstance(paired, list)
        assert isinstance(coverage, Mapping)
        assert isinstance(config, Mapping)
        assert isinstance(info, Mapping)
        require_fields(metrics, VIDEO_SCALAR_FIELDS, "video metrics")
        require_fields(runtime, VIDEO_RUNTIME_FIELDS, "video runtime")
        require_fields(
            similarity,
            SIMILARITY_PROMPT_FIELDS,
            "Similarity prompt metrics",
        )
        require_fields(
            vbench,
            VBENCH_PROMPT_FIELDS,
            "VBench prompt metrics",
        )
        require_fields(paired, PAIRED_FIELDS, "paired comparisons")
        expected_settings = {"bf16", "l8c64_int4"} | {
            candidate_setting(scheme, bits) for scheme in SCHEMES for bits in BITS
        }
        selected_hadamard = summary.get("selected_hadamard")
        if not isinstance(selected_hadamard, Mapping) or set(selected_hadamard) != set(
            SCHEMES
        ):
            raise ValueError("video summary Hadamard mapping is incomplete")
        settings = {row["setting"] for row in metrics}
        if settings != expected_settings or len(metrics) != len(expected_settings):
            raise ValueError(
                f"video scalar coverage is {len(settings)}/{len(expected_settings)}"
            )
        for row in metrics:
            validate_video_setting_metadata(
                row,
                label="video scalar",
                selected_hadamard=selected_hadamard,
            )
        candidate_keys = {
            (row["scheme"], int_value(row["prefix_bits"], "video bits"))
            for row in metrics
            if row["scheme"] in SCHEMES
        }
        expected_candidates = {(scheme, bits) for scheme in SCHEMES for bits in BITS}
        if candidate_keys != expected_candidates:
            raise ValueError("video candidates do not cover four schemes x four bits")
        if len(runtime) != 16 * 10:
            raise ValueError(f"video runtime has {len(runtime)} rows, expected 160")
        runtime_keys = {
            (
                row["scheme"],
                int_value(row["prefix_bits"], "video runtime bits"),
                int_value(row["prompt_id"], "video runtime prompt_id"),
            )
            for row in runtime
        }
        expected_runtime_keys = {
            (scheme, bits, prompt_id)
            for scheme in SCHEMES
            for bits in BITS
            for prompt_id in range(10)
        }
        if runtime_keys != expected_runtime_keys:
            raise ValueError("video runtime does not cover every candidate/prompt")
        if len(runtime_keys) != len(runtime):
            raise ValueError("video runtime contains duplicate candidate/prompt rows")
        for row in runtime:
            scheme = str(row["scheme"])
            bits = int_value(row["prefix_bits"], "video runtime bits")
            if (
                scheme not in SCHEMES
                or bits not in BITS
                or bool_value(row["hadamard"]) != bool_value(selected_hadamard[scheme])
            ):
                raise ValueError("video runtime candidate metadata differs")
            if int_value(row["seed"], "video runtime seed") != 42:
                raise ValueError("video runtime contains a non-42 seed")
            for field in (
                "kv_payload_mib",
                "shared_codebook_mib",
                "bf16_reference_mib_inferred",
                "joint_compression_including_rvq_and_codec_params",
                "online_kv_encode_time_s",
                "online_kv_decode_time_s",
                "online_kv_decode_calls",
                "e2e_time_s",
            ):
                value = finite_float(row[field], f"video runtime {field}")
                if value < 0:
                    raise ValueError(f"video runtime has negative {field}")
        if len(similarity) != 17 * 10:
            raise ValueError(
                f"Similarity prompt table has {len(similarity)} rows, expected 170"
            )
        candidate_settings = {
            candidate_setting(scheme, bits) for scheme in SCHEMES for bits in BITS
        }
        expected_similarity_keys = {
            (setting, prompt_id)
            for setting in {"l8c64_int4", *candidate_settings}
            for prompt_id in range(10)
        }
        similarity_keys = {
            (
                row["setting"],
                int_value(row["prompt_id"], "Similarity prompt_id"),
            )
            for row in similarity
        }
        if similarity_keys != expected_similarity_keys or len(similarity_keys) != len(
            similarity
        ):
            raise ValueError("Similarity unique setting/prompt coverage differs")
        for row in similarity:
            validate_video_setting_metadata(
                row,
                label="Similarity prompt metric",
                selected_hadamard=selected_hadamard,
            )
            for field in ("psnr", "ssim", "lpips", "mse"):
                value = finite_float(row[field], f"Similarity {field}")
                if value < 0:
                    raise ValueError(f"Similarity has negative {field}")
        if len(vbench) != 18 * 10 * len(SUPPORTED_VBENCH):
            raise ValueError(
                f"VBench prompt table has {len(vbench)} rows, expected "
                f"{18 * 10 * len(SUPPORTED_VBENCH)}"
            )
        expected_vbench_keys = {
            (setting, prompt_id, dimension)
            for setting in expected_settings
            for prompt_id in range(10)
            for dimension in SUPPORTED_VBENCH
        }
        vbench_keys = {
            (
                row["setting"],
                int_value(row["prompt_id"], "VBench prompt_id"),
                row["dimension"],
            )
            for row in vbench
        }
        if vbench_keys != expected_vbench_keys or len(vbench_keys) != len(vbench):
            raise ValueError("VBench unique setting/prompt/dimension coverage differs")
        for row in vbench:
            validate_video_setting_metadata(
                row,
                label="VBench prompt metric",
                selected_hadamard=selected_hadamard,
            )
            finite_float(row["value"], "VBench prompt value")
        expected_pair_keys = expected_paired_keys()
        pair_keys = {
            (
                row["candidate"],
                row["baseline"],
                row["contrast"],
                row["metric"],
            )
            for row in paired
        }
        if pair_keys != expected_pair_keys or len(pair_keys) != len(paired):
            raise ValueError(
                f"paired comparison coverage is "
                f"{len(pair_keys)}/{len(expected_pair_keys)} unique rows"
            )
        evaluated = tuple(coverage.get("evaluated_custom_input_dimensions", []))
        unavailable = tuple(
            coverage.get("unavailable_in_installed_vbench_custom_input", [])
        )
        if set(evaluated) != set(SUPPORTED_VBENCH):
            raise ValueError("VBench supported dimension coverage differs")
        if set(unavailable) != set(UNSUPPORTED_VBENCH):
            raise ValueError("VBench unsupported dimension record differs")
        if coverage.get("missing_supported_results"):
            raise ValueError("VBench coverage reports missing supported results")
        scale_record = summary.get("vbench_custom_input_coverage", {}).get(
            "per_prompt_fixed_unit_scale"
        )
        if scale_record != VBENCH_PER_PROMPT_FIXED_SCALE:
            raise ValueError(
                "video summary lacks the fixed VBench per-prompt unit scale"
            )
        if summary.get("config_hash") != sha256_file(paths["config"]):
            raise ValueError("video summary config hash mismatch")
        params_path = (
            Path(training.metadata["params_path"]) if training.complete else None
        )
        validate_run_info(info, paths["config"], params_path=params_path)
        validate_training_links(
            info,
            training,
            require_implementation_match=False,
        )
        if int_value(config.get("prompts"), "video configured prompts") != 10:
            raise ValueError("video config is not the fixed 10-prompt benchmark")
        residual_config = config.get("residual")
        if (
            not isinstance(residual_config, Mapping)
            or set(residual_config.get("schemes", ())) != set(SCHEMES)
            or set(residual_config.get("prefix_bits", ())) != set(BITS)
            or residual_config.get("single_encode_then_prefix_decode_required")
            is not True
        ):
            raise ValueError("video config residual truncation protocol differs")
        prompt_payload = info.get("prompt_list")
        if not isinstance(prompt_payload, Mapping):
            raise ValueError("video run_info prompt_list is not split by cohort")
        test_prompts = prompt_payload.get("test")
        if not isinstance(test_prompts, list) or len(test_prompts) != 10:
            raise ValueError("video run_info test prompt cohort is not 10 prompts")
        test_prompts = [str(value) for value in test_prompts]
        runtime_video_by_key: dict[tuple[str, int], Path] = {}
        for row in runtime:
            scheme = str(row["scheme"])
            bits = int_value(row["prefix_bits"], "video runtime bits")
            prompt_id = int_value(row["prompt_id"], "video runtime prompt")
            setting = candidate_setting(scheme, bits)
            if row["prompt"] != test_prompts[prompt_id]:
                raise ValueError(
                    f"video runtime prompt mapping differs at {setting}/{prompt_id}"
                )
            video_path = Path(str(row["video_path"])).resolve()
            if video_path.name != f"{prompt_id}-42.mp4" or not video_path.is_file():
                raise ValueError(
                    f"video runtime path is invalid at {setting}/{prompt_id}"
                )
            runtime_video_by_key[(setting, prompt_id)] = video_path
        reference_records = info.get("reference_videos")
        if not isinstance(reference_records, list) or len(reference_records) != 10:
            raise ValueError("video run_info reference video evidence is incomplete")
        reference_by_prompt: dict[int, Path] = {}
        for raw_reference in reference_records:
            if not isinstance(raw_reference, Mapping):
                raise ValueError("video reference record is malformed")
            reference = validate_recorded_file(
                raw_reference,
                path_key="path",
                hash_key="sha256",
                label="video BF16 reference",
            )
            try:
                prompt_id = int(reference.stem.rsplit("-", 1)[0])
            except ValueError as error:
                raise ValueError(
                    f"video reference filename lacks prompt ID: {reference}"
                ) from error
            if (
                reference.name != f"{prompt_id}-42.mp4"
                or prompt_id not in range(10)
                or prompt_id in reference_by_prompt
                or int_value(
                    raw_reference.get("bytes"),
                    "video reference bytes",
                )
                != reference.stat().st_size
            ):
                raise ValueError("video BF16 reference prompt/hash record differs")
            reference_by_prompt[prompt_id] = reference
            section.evidence.append(
                evidence_entry(
                    reference,
                    section.name,
                    f"BF16 reference video prompt {prompt_id}",
                )
            )
        if set(reference_by_prompt) != set(range(10)):
            raise ValueError("video BF16 reference prompt IDs are incomplete")
        reused_baselines = info.get("reused_baselines")
        if not isinstance(reused_baselines, Mapping):
            raise ValueError("video run_info lacks reused baseline registry")
        l8_video_dir = Path(str(reused_baselines.get("l8c64_int4_videos"))).resolve()
        if not l8_video_dir.is_dir():
            raise ValueError("registered L8 baseline video directory is absent")
        l8_evidence_paths: set[Path] = set()
        for row in similarity:
            setting = str(row["setting"])
            prompt_id = int_value(row["prompt_id"], "Similarity prompt")
            if row["prompt"] != test_prompts[prompt_id]:
                raise ValueError(
                    f"Similarity prompt mapping differs at {setting}/{prompt_id}"
                )
            reference = Path(str(row["reference_video"])).resolve()
            test_video = Path(str(row["test_video"])).resolve()
            if reference != reference_by_prompt[prompt_id]:
                raise ValueError(
                    f"Similarity reference path/prompt differs at {setting}/{prompt_id}"
                )
            if setting in candidate_settings:
                if test_video != runtime_video_by_key[(setting, prompt_id)]:
                    raise ValueError(
                        f"Similarity test path differs from runtime evidence at "
                        f"{setting}/{prompt_id}"
                    )
            else:
                expected_l8 = (l8_video_dir / f"{prompt_id}-42.mp4").resolve()
                if test_video != expected_l8 or not test_video.is_file():
                    raise ValueError(
                        f"Similarity L8 baseline path differs at {setting}/{prompt_id}"
                    )
                if test_video not in l8_evidence_paths:
                    section.evidence.append(
                        evidence_entry(
                            test_video,
                            section.name,
                            f"L8 baseline video prompt {prompt_id}",
                        )
                    )
                    l8_evidence_paths.add(test_video)
        if training.complete:
            if summary.get("codec_params_hash") != training.metadata["params_hash"]:
                raise ValueError("video summary codec parameter hash mismatch")
            selected = training.metadata["selected_hadamard"]
            if summary.get("selected_hadamard") != selected:
                raise ValueError("video summary Hadamard selection mismatch")
            full_params_mib = training.metadata["params_file_bytes"] / (1 << 20)
            for row in runtime:
                payload = finite_float(row["kv_payload_mib"], "online KV payload")
                codebook = finite_float(
                    row["shared_codebook_mib"], "online RVQ codebook"
                )
                bf16 = finite_float(
                    row["bf16_reference_mib_inferred"],
                    "online BF16 reference",
                )
                reported = finite_float(
                    row["joint_compression_including_rvq_and_codec_params"],
                    "online full-file joint ratio",
                )
                recomputed = bf16 / (payload + codebook + full_params_mib)
                if not close(reported, recomputed, rtol=2e-6):
                    raise ValueError(
                        "video full-params joint ratio does not match the "
                        "per-prompt payload accounting"
                    )
        config_evidence = validate_candidate_config_audit(
            summary,
            expected_params_path=(
                Path(training.metadata["params_path"]) if training.complete else None
            ),
            expected_prompts=test_prompts,
        )
        for item in config_evidence:
            section.evidence.append(
                {
                    "section": section.name,
                    "role": (
                        "candidate video "
                        f"{item['scheme']} b{item['prefix_bits']} "
                        f"prompt {item['prompt_id']}"
                    ),
                    "path": item["video_path"],
                    "bytes": item["video_bytes"],
                    "sha256": item["video_sha256"],
                }
            )
        expected_videos_by_setting: dict[str, dict[int, Path]] = {
            "bf16": dict(reference_by_prompt),
            "l8c64_int4": {
                prompt_id: l8_video_dir / f"{prompt_id}-42.mp4"
                for prompt_id in range(10)
            },
        }
        expected_test_directories = {
            "l8c64_int4": l8_video_dir,
        }
        for setting in candidate_settings:
            expected_videos_by_setting[setting] = {
                prompt_id: runtime_video_by_key[(setting, prompt_id)]
                for prompt_id in range(10)
            }
            expected_test_directories[setting] = expected_videos_by_setting[setting][
                0
            ].parent
        similarity_rows_by_setting: dict[str, dict[int, Mapping[str, Any]]] = (
            defaultdict(dict)
        )
        for row in similarity:
            setting = str(row["setting"])
            prompt_id = int_value(row["prompt_id"], "Similarity summary prompt")
            if prompt_id in similarity_rows_by_setting[setting]:
                raise ValueError(
                    f"duplicate Similarity summary row for {setting}/{prompt_id}"
                )
            similarity_rows_by_setting[setting][prompt_id] = row
        expected_similarity_settings = {
            "l8c64_int4",
            *candidate_settings,
        }
        similarity_chain = validate_similarity_summary_evidence_chain(
            summary,
            root=root,
            config_path=paths["config"],
            params_path=params_path,
            root_run_info=info,
            expected_settings=expected_similarity_settings,
            expected_prompts=test_prompts,
            expected_references=reference_by_prompt,
        )
        section.evidence.extend(
            [
                evidence_entry(
                    Path(similarity_chain["run_info_path"]),
                    section.name,
                    "normalized LPIPS Similarity run_info",
                ),
                evidence_entry(
                    Path(similarity_chain["completion_path"]),
                    section.name,
                    "normalized LPIPS Similarity completion manifest",
                ),
                evidence_entry(
                    SIMILARITY_EVALUATOR,
                    section.name,
                    "normalized LPIPS Similarity evaluator",
                ),
            ]
        )
        strict_source_rows: dict[str, Sequence[Mapping[str, Any]]] = {}
        for setting in sorted(expected_similarity_settings):
            source_bundle = validate_normalized_similarity_source_bundle(
                similarity_chain["source_by_setting"][setting],
                expected_setting=setting,
                artifact_directory=similarity_chain["artifact_directory"],
                expected_test_directory=expected_test_directories[setting],
                expected_test_videos=expected_videos_by_setting[setting],
                expected_references=reference_by_prompt,
                expected_prompts=test_prompts,
                summary_rows=similarity_rows_by_setting[setting],
            )
            strict_source_rows[setting] = source_bundle["rows"]
            section.evidence.append(
                evidence_entry(
                    source_bundle["json_path"],
                    section.name,
                    f"normalized Similarity JSON {setting}",
                )
            )
            section.evidence.append(
                evidence_entry(
                    source_bundle["csv_path"],
                    section.name,
                    f"normalized Similarity CSV {setting}",
                )
            )
        legacy_control = validate_legacy_similarity_controls(
            legacy_directory=similarity_chain["legacy_directory"],
            legacy_l8_csv=similarity_chain["legacy_l8_csv"],
            strict_rows_by_setting=strict_source_rows,
        )
        for legacy_path in legacy_control["legacy_csv_paths"]:
            section.evidence.append(
                evidence_entry(
                    legacy_path,
                    section.name,
                    "legacy Similarity PSNR/SSIM/MSE control only",
                )
            )
        metrics_by_setting = {row["setting"]: row for row in metrics}
        similarity_values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for row in similarity:
            setting = row["setting"]
            prompt_id = int_value(row["prompt_id"], "Similarity prompt")
            for metric in SIMILARITY_METRICS:
                similarity_values[(setting, metric)][prompt_id] = finite_float(
                    row[metric], f"Similarity {metric}"
                )
        vbench_values: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for row in vbench:
            vbench_values[(row["setting"], row["dimension"])][
                int_value(row["prompt_id"], "VBench prompt")
            ] = finite_float(row["value"], "VBench prompt value")
        vbench_sources = summary.get("vbench_source_evidence")
        if not isinstance(vbench_sources, list):
            raise ValueError("video summary lacks VBench source evidence")
        expected_vbench_source_keys = {
            (setting, dimension)
            for setting in expected_settings
            for dimension in SUPPORTED_VBENCH
        }
        vbench_source_by_key = {
            (str(item["setting"]), str(item["dimension"])): item
            for item in vbench_sources
            if isinstance(item, Mapping)
        }
        if (
            len(vbench_sources) != len(expected_vbench_source_keys)
            or set(vbench_source_by_key) != expected_vbench_source_keys
        ):
            raise ValueError(
                "VBench source evidence coverage is not "
                f"{len(expected_vbench_source_keys)}/"
                f"{len(expected_vbench_source_keys)} unique bundles"
            )
        for key, source in vbench_source_by_key.items():
            setting, dimension = key
            result_path, full_info_path = validate_vbench_source_bundle(
                source,
                expected_setting=setting,
                expected_dimension=dimension,
                expected_videos=expected_videos_by_setting[setting],
                expected_prompts=test_prompts,
                expected_values=vbench_values[key],
            )
            section.evidence.append(
                evidence_entry(
                    result_path,
                    section.name,
                    f"VBench eval result {setting}/{dimension}",
                )
            )
            section.evidence.append(
                evidence_entry(
                    full_info_path,
                    section.name,
                    f"VBench full-info {setting}/{dimension}",
                )
            )
        for setting in {"l8c64_int4", *candidate_settings}:
            scalar = metrics_by_setting[setting]
            for metric in SIMILARITY_METRICS:
                mean = sum(similarity_values[(setting, metric)].values()) / 10.0
                if not close(
                    finite_float(scalar[metric], f"video scalar {metric}"),
                    mean,
                    rtol=1e-7,
                    atol=1e-10,
                ):
                    raise ValueError(f"video scalar {setting}/{metric} != prompt mean")
        for setting in expected_settings:
            scalar = metrics_by_setting[setting]
            for dimension in SUPPORTED_VBENCH:
                mean = sum(vbench_values[(setting, dimension)].values()) / 10.0
                if not close(
                    finite_float(
                        scalar[dimension],
                        f"video scalar {dimension}",
                    ),
                    mean,
                    rtol=1e-7,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"video scalar {setting}/{dimension} != prompt mean"
                    )
        summary_metrics = summary.get("metrics")
        if not isinstance(summary_metrics, list):
            raise ValueError("video summary JSON lacks scalar metrics")
        json_by_setting = {
            str(row["setting"]): row
            for row in summary_metrics
            if isinstance(row, Mapping)
        }
        if set(json_by_setting) != expected_settings:
            raise ValueError("video summary JSON setting coverage differs")
        for setting, csv_row in metrics_by_setting.items():
            json_row = json_by_setting[setting]
            for metric in (
                "psnr",
                "ssim",
                "lpips",
                *SUPPORTED_VBENCH,
            ):
                csv_value = optional_float(csv_row.get(metric))
                json_value = optional_float(json_row.get(metric))
                if csv_value is None and json_value is None:
                    continue
                if (
                    csv_value is None
                    or json_value is None
                    or not close(csv_value, json_value)
                ):
                    raise ValueError(f"video CSV/JSON differ for {setting}/{metric}")
        summary_paired = summary.get("paired_comparisons")
        if not isinstance(summary_paired, list):
            raise ValueError("video summary JSON lacks paired comparisons")
        summary_paired_by_key = {
            (
                str(row["candidate"]),
                str(row["baseline"]),
                str(row["contrast"]),
                str(row["metric"]),
            ): row
            for row in summary_paired
            if isinstance(row, Mapping)
        }
        if set(summary_paired_by_key) != expected_pair_keys or len(
            summary_paired_by_key
        ) != len(summary_paired):
            raise ValueError("video summary JSON paired coverage differs")
        all_prompt_values = {
            **similarity_values,
            **vbench_values,
        }
        canonical_pair_index = {
            key: index for index, key in enumerate(expected_paired_specs())
        }
        for row in paired:
            metric = row["metric"]
            candidate = row["candidate"]
            baseline = row["baseline"]
            pair_key = (
                candidate,
                baseline,
                row["contrast"],
                metric,
            )
            candidate_prompt = all_prompt_values[(candidate, metric)]
            baseline_prompt = all_prompt_values[(baseline, metric)]
            if set(candidate_prompt) != set(range(10)) or set(baseline_prompt) != set(
                range(10)
            ):
                raise ValueError("paired comparison prompt values are incomplete")
            direction = -1.0 if metric == "lpips" else 1.0
            differences = [
                direction * (candidate_prompt[prompt_id] - baseline_prompt[prompt_id])
                for prompt_id in range(10)
            ]
            expected_candidate_mean = sum(candidate_prompt.values()) / 10.0
            expected_baseline_mean = sum(baseline_prompt.values()) / 10.0
            expected_improvement = sum(differences) / 10.0
            expected_win_rate = sum(value > 0 for value in differences) / 10.0
            checks = (
                ("candidate_mean", expected_candidate_mean),
                ("baseline_mean", expected_baseline_mean),
                ("mean_paired_improvement", expected_improvement),
                ("prompt_win_rate", expected_win_rate),
            )
            for field, expected_value in checks:
                if not close(
                    finite_float(row[field], f"paired {field}"),
                    expected_value,
                    rtol=1e-7,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"paired {candidate}/{baseline}/{metric} "
                        f"has inconsistent {field}"
                    )
            low = finite_float(row["bootstrap_95_ci_low"], "paired CI low")
            high = finite_float(row["bootstrap_95_ci_high"], "paired CI high")
            if low > high:
                raise ValueError("paired comparison has reversed CI")
            expected_low, expected_high = paired_bootstrap_ci(
                differences,
                seed=42 + 1009 * (canonical_pair_index[pair_key] + 1),
            )
            if not close(low, expected_low, rtol=1e-7, atol=1e-12) or not close(
                high,
                expected_high,
                rtol=1e-7,
                atol=1e-12,
            ):
                raise ValueError(f"paired bootstrap CI is inconsistent for {pair_key}")
            json_row = summary_paired_by_key[pair_key]
            for field in (
                "candidate_mean",
                "baseline_mean",
                "mean_paired_improvement",
                "bootstrap_95_ci_low",
                "bootstrap_95_ci_high",
                "prompt_win_rate",
            ):
                if not close(
                    finite_float(row[field], f"paired CSV {field}"),
                    finite_float(json_row[field], f"paired JSON {field}"),
                    rtol=1e-7,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"video paired CSV/JSON differ for {pair_key}/{field}"
                    )
        # Preserve all reported metrics and add a clearly descriptive mean.
        output = []
        for raw in metrics:
            row = dict(raw)
            values = [
                finite_float(row[name], f"VBench {name}") for name in SUPPORTED_VBENCH
            ]
            row["vbench_10d_unweighted_descriptive_mean"] = sum(values) / len(values)
            row["vbench_mean_note"] = (
                "descriptive unweighted mean only; not an official VBench score"
            )
            output.append(row)
        setting_order = {"bf16": -2, "l8c64_int4": -1}
        scheme_order = {value: index for index, value in enumerate(SCHEMES)}
        equal_payload_rows = video_equal_payload_comparisons(
            similarity,
            vbench,
        )
        signal_audit_rows = vbench_signal_audit(vbench)
        section.rows = sorted(
            output,
            key=lambda row: (
                setting_order.get(
                    row["setting"],
                    scheme_order.get(row["scheme"], len(SCHEMES)),
                ),
                int_value(row["prefix_bits"], "video bits")
                if row["scheme"] in SCHEMES
                else -1,
            ),
        )
        section.metadata = {
            "summary_version": version,
            "summary_path": str(summary_path.resolve()),
            "runtime_rows": runtime,
            "similarity_rows": similarity,
            "vbench_rows": vbench,
            "paired_rows": paired,
            "equal_payload_rows": equal_payload_rows,
            "vbench_signal_audit_rows": signal_audit_rows,
            "coverage": dict(coverage),
            "config_evidence": config_evidence,
            "run_info": dict(info),
            "similarity_protocol": similarity_chain["protocol"],
            "similarity_run_info": similarity_chain["run_info"],
            "similarity_completion": similarity_chain["completion"],
            "similarity_source_evidence": [
                dict(similarity_chain["source_by_setting"][setting])
                for setting in sorted(expected_similarity_settings)
            ],
            "legacy_similarity_invalidation": similarity_chain["legacy_invalidation"],
            "legacy_similarity_control": {
                key: value
                for key, value in legacy_control.items()
                if key != "legacy_csv_paths"
            },
        }
        section.complete = True
    except Exception as error:
        section.errors.append(f"{type(error).__name__}: {error}")
        section.rows = []
    return section


def selected_training_rows(training: Section) -> list[dict[str, Any]]:
    if not training.complete:
        return []
    return [row for row in training.rows if bool(row.get("selected_transform"))]


def build_parameter_rows(training: Section) -> list[dict[str, Any]]:
    if not training.complete:
        return []
    per_scheme = training.metadata["parameter_bytes_by_scheme"]
    rows = []
    for scheme in SCHEMES:
        value = int(per_scheme[scheme])
        rows.append(
            {
                "scheme": scheme,
                "scheme_zh": SCHEME_ZH[scheme],
                "selected_hadamard": training.metadata["selected_hadamard"][scheme],
                "selected_parameter_bytes": value,
                "selected_parameter_MiB": value / (1 << 20),
                "scope": "globally shared; charge once per deployed scheme",
            }
        )
    rows.extend(
        [
            {
                "scheme": "__all_selected__",
                "scheme_zh": "四种方案的 selected 参数总和",
                "selected_hadamard": None,
                "selected_parameter_bytes": training.metadata[
                    "selected_parameter_bytes_total"
                ],
                "selected_parameter_MiB": training.metadata[
                    "selected_parameter_bytes_total"
                ]
                / (1 << 20),
                "scope": "benchmark artifact; not required by one-scheme deployment",
            },
            {
                "scheme": "__params_file__",
                "scheme_zh": "codec_params.pt 文件",
                "selected_hadamard": None,
                "selected_parameter_bytes": training.metadata["params_file_bytes"],
                "selected_parameter_MiB": training.metadata["params_file_bytes"]
                / (1 << 20),
                "scope": (
                    "conservative archival upper bound; includes selected and "
                    "ablation candidates plus serialization"
                ),
            },
        ]
    )
    return rows


def row_by_scheme_bit(
    rows: Sequence[Mapping[str, Any]],
    scheme_field: str,
    bits_field: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    result = {}
    for row in rows:
        scheme = str(row.get(scheme_field, ""))
        if scheme not in SCHEMES:
            continue
        bits = int_value(row[bits_field], bits_field)
        result[(scheme, bits)] = row
    return result


def build_rate_rows(
    training: Section,
    runtime: Section,
    offline: Section,
    video: Section,
) -> list[dict[str, Any]]:
    runtime_by = (
        row_by_scheme_bit(runtime.rows, "scheme", "bits") if runtime.complete else {}
    )
    offline_by: dict[tuple[str, int], Mapping[str, Any]] = {}
    if offline.complete:
        for row in offline.rows:
            if row["method"] == "l8c64_int4_base":
                continue
            scheme = (
                row["scheme"] if row.get("scheme") else row["method"].rsplit("_b", 1)[0]
            )
            offline_by[(scheme, int_value(row["bits"], "offline bits"))] = row
    video_by = (
        row_by_scheme_bit(video.rows, "scheme", "prefix_bits") if video.complete else {}
    )
    physical_codebook_mib = None
    if video.complete:
        l8_rows = [row for row in video.rows if row.get("setting") == "l8c64_int4"]
        if len(l8_rows) != 1:
            raise ValueError("physical L8 codebook accounting row is not unique")
        physical_codebook_mib = optional_float(
            l8_rows[0].get("shared_codebook_mib_mean")
        )
        if physical_codebook_mib is None:
            raise ValueError("physical L8 codebook MiB is unavailable")
    online_samples: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    if video.complete:
        for row in video.metadata["runtime_rows"]:
            scheme = str(row.get("scheme", ""))
            if scheme in SCHEMES:
                online_samples[
                    (scheme, int_value(row["prefix_bits"], "video runtime bits"))
                ].append(row)
    params_by = (
        training.metadata.get("parameter_bytes_by_scheme", {})
        if training.complete
        else {}
    )
    selected_total = (
        int(training.metadata["selected_parameter_bytes_total"])
        if training.complete
        else None
    )
    params_file = (
        int(training.metadata["params_file_bytes"]) if training.complete else None
    )
    rows = []
    for scheme in SCHEMES:
        for bits in BITS:
            runtime_row = runtime_by.get((scheme, bits), {})
            offline_row = offline_by.get((scheme, bits), {})
            video_row = video_by.get((scheme, bits), {})
            samples = online_samples.get((scheme, bits), [])

            def sample_mean(name: str) -> float | None:
                values = [
                    value
                    for value in (optional_float(row.get(name)) for row in samples)
                    if value is not None
                ]
                return sum(values) / len(values) if values else None

            kv_payload = sample_mean("kv_payload_mib")
            if kv_payload is None:
                kv_payload = optional_float(video_row.get("kv_payload_mib_mean"))
            rvq_codebook = sample_mean("shared_codebook_mib")
            if rvq_codebook is None:
                rvq_codebook = optional_float(video_row.get("shared_codebook_mib_mean"))
            bf16_reference = sample_mean("bf16_reference_mib_inferred")
            if bf16_reference is None:
                bf16_reference = optional_float(
                    video_row.get("bf16_reference_mib_inferred_mean")
                )
            no_params_values = [
                value
                for value in (
                    optional_float(
                        row.get("compression_ratio_including_shared_codebook")
                    )
                    for row in samples
                )
                if value is not None
            ]
            no_params_ratio = (
                sum(no_params_values) / len(no_params_values)
                if no_params_values
                else optional_float(
                    video_row.get("compression_ratio_including_shared_codebook_mean")
                )
            )
            if (
                bf16_reference is None
                and kv_payload is not None
                and rvq_codebook is not None
                and no_params_ratio is not None
            ):
                bf16_reference = no_params_ratio * (kv_payload + rvq_codebook)

            def joint_ratio(parameter_bytes: int | None) -> float | None:
                if parameter_bytes is None:
                    return None
                per_prompt = []
                for sample in samples:
                    sample_bf16 = optional_float(
                        sample.get("bf16_reference_mib_inferred")
                    )
                    sample_payload = optional_float(sample.get("kv_payload_mib"))
                    sample_codebook = optional_float(sample.get("shared_codebook_mib"))
                    if (
                        sample_bf16 is not None
                        and sample_payload is not None
                        and sample_codebook is not None
                    ):
                        per_prompt.append(
                            sample_bf16
                            / (
                                sample_payload
                                + sample_codebook
                                + parameter_bytes / (1 << 20)
                            )
                        )
                if per_prompt:
                    return sum(per_prompt) / len(per_prompt)
                if bf16_reference is None or kv_payload is None or rvq_codebook is None:
                    return None
                return bf16_reference / (
                    kv_payload + rvq_codebook + parameter_bytes / (1 << 20)
                )

            def physical_joint_ratio(
                parameter_bytes: int | None,
                *,
                include_runtime_acceleration_cache: bool = False,
            ) -> float | None:
                """Use physical K128/V128 payload and canonical/runtime overhead."""

                if parameter_bytes is None or physical_codebook_mib is None:
                    return None
                runtime_overhead = (
                    RVQ_RUNTIME_ACCELERATION_CACHE_MIB
                    if include_runtime_acceleration_cache
                    else 0.0
                )
                per_prompt = []
                for sample in samples:
                    sample_bf16 = optional_float(
                        sample.get("bf16_reference_mib_inferred")
                    )
                    sample_payload = optional_float(sample.get("kv_payload_mib"))
                    if sample_bf16 is not None and sample_payload is not None:
                        per_prompt.append(
                            (sample_bf16 / 2.0)
                            / (
                                sample_payload / 2.0
                                + physical_codebook_mib
                                + runtime_overhead
                                + parameter_bytes / (1 << 20)
                            )
                        )
                if per_prompt:
                    return sum(per_prompt) / len(per_prompt)
                if bf16_reference is None or kv_payload is None:
                    return None
                return (bf16_reference / 2.0) / (
                    kv_payload / 2.0
                    + physical_codebook_mib
                    + runtime_overhead
                    + parameter_bytes / (1 << 20)
                )

            rows.append(
                {
                    "scheme": scheme,
                    "scheme_zh": SCHEME_ZH[scheme],
                    "bits": bits,
                    "physical_effective_bits_per_residual_value": optional_float(
                        runtime_row.get("physical_effective_bits_per_value")
                    ),
                    "local_metadata_bits_per_residual_value": optional_float(
                        runtime_row.get("local_metadata_bits_per_value")
                    ),
                    "residual_only_ratio_vs_BF16_residual": optional_float(
                        runtime_row.get("compression_ratio_vs_BF16_residual")
                    ),
                    "single_sampled_cache_residual_ratio_including_scheme_params": (
                        optional_float(
                            runtime_row.get(
                                "single_cache_ratio_including_shared_parameters"
                            )
                        )
                    ),
                    "offline_evaluated_dataset_residual_ratio_including_scheme_params_once": (
                        optional_float(
                            offline_row.get(
                                "dataset_ratio_including_shared_parameters_once"
                            )
                        )
                    ),
                    "online_rvq_plus_residual_payload_MiB": kv_payload,
                    "online_shared_rvq_codebook_MiB": rvq_codebook,
                    "online_BF16_reference_MiB_inferred": bf16_reference,
                    "online_physical_preposition_rvq_plus_residual_payload_MiB": (
                        None if kv_payload is None else kv_payload / 2.0
                    ),
                    "online_physical_preposition_shared_rvq_codebook_MiB": (
                        physical_codebook_mib
                    ),
                    "online_runtime_rvq_acceleration_cache_MiB": (
                        RVQ_RUNTIME_ACCELERATION_CACHE_MIB
                    ),
                    "online_physical_preposition_BF16_reference_MiB": (
                        None if bf16_reference is None else bf16_reference / 2.0
                    ),
                    "online_joint_ratio_without_codec_params": no_params_ratio,
                    "online_joint_ratio_including_scheme_params_once": joint_ratio(
                        int(params_by[scheme]) if scheme in params_by else None
                    ),
                    "online_joint_ratio_including_all_selected_params_once": joint_ratio(
                        selected_total
                    ),
                    "online_joint_ratio_including_full_params_file_conservative_lower_bound": (
                        joint_ratio(params_file)
                    ),
                    "online_physical_preposition_joint_ratio_including_scheme_params_once": (
                        physical_joint_ratio(
                            int(params_by[scheme]) if scheme in params_by else None
                        )
                    ),
                    "online_physical_preposition_joint_ratio_including_full_params_file_conservative_lower_bound": (
                        physical_joint_ratio(params_file)
                    ),
                    "online_gpu_resident_estimate_joint_ratio_including_scheme_params_once": (
                        physical_joint_ratio(
                            int(params_by[scheme]) if scheme in params_by else None,
                            include_runtime_acceleration_cache=True,
                        )
                    ),
                    "video_runner_reported_full_file_joint_ratio": optional_float(
                        video_row.get(
                            "joint_compression_including_rvq_and_codec_params"
                        )
                    ),
                    "scheme_parameter_bytes": (
                        int(params_by[scheme]) if scheme in params_by else None
                    ),
                    "rate_scope_note": (
                        "offline/runtime residual ratio uses physical dim=128 and "
                        "excludes RVQ; online runner fields use a two-position-stream "
                        "equivalent dim=256 accounting; physical pre-position "
                        "canonical fields use K dim=128 and V dim=128; GPU-resident "
                        "no-offload estimate additionally charges BF16/FP32 runtime "
                        "codebook caches; formal video runs use generation KV CPU offload"
                    ),
                }
            )
    return rows


def fmt(value: Any, digits: int = 6) -> str:
    number = optional_float(value)
    if number is None:
        return "—"
    if number == 0:
        return "0"
    if abs(number) >= 1000 or abs(number) < 1e-4:
        return f"{number:.3e}"
    rendered = f"{number:.{digits}f}"
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def render_report(
    sections: Mapping[str, Section],
    *,
    final: bool,
    source_root: Path,
    rate_rows: Sequence[Mapping[str, Any]],
    parameter_rows: Sequence[Mapping[str, Any]],
) -> str:
    training = sections["training"]
    runtime = sections["runtime"]
    offline = sections["offline"]
    exact = sections["exact_attention"]
    video = sections["video_similarity_vbench"]
    title_suffix = "" if final else "（部分结果草稿，禁止作为最终结论引用）"
    lines = [
        f"# Wan2.2-5B L8C64 INT4 RVQ 可截断残差压缩实验报告{title_suffix}",
        "",
        f"- 数据源：`{source_root}`",
        "- 比较对象：四种 residual codec，均在 L8C64 INT4 RVQ 基础上编码 "
        "`r = BF16(x) - BF16(RVQ_INT4(x))`。",
        "- 截断协议：只编码一次 B4 状态，再物理截断到 1/2/3/4 个前缀 plane；"
        "不是为每个 bit 预算重新编码。",
        "- 本文所有 MiB 均为 `2^20 bytes`。",
        "",
        "## 完整性审计",
        "",
        markdown_table(
            ("阶段", "状态", "有效行数", "缺失", "错误"),
            [
                (
                    name,
                    section.status(),
                    len(section.rows),
                    "<br>".join(section.missing) if section.missing else "—",
                    "<br>".join(section.errors) if section.errors else "—",
                )
                for name, section in sections.items()
            ],
        ),
        "",
    ]
    if not final:
        lines.extend(
            [
                "> 当前证据未齐。下列表格只呈现已经通过 schema、覆盖率和哈希"
                "检查的阶段；缺失格不做任何插值或估计。",
                "",
            ]
        )

    lines.extend(
        [
            "## 压缩率口径审计",
            "",
            "必须区分以下口径，不能把它们写成同一个“压缩倍率”：",
            "",
            "1. `residual-only`：BF16 residual 字节数 / residual payload；"
            "包含 bitplane 与每个 block 的 FP16 局部统计，但排除 RVQ 与共享参数。",
            "2. `residual + scheme params`：在一个缓存或整套评估数据上只加一次"
            "当前方案的共享参数。",
            "3. `physical-dimension canonical joint`：pre-position 历史缓存中 "
            "K 的 head dim=128、V 的 head dim=128；分子和 payload 只按一份 "
            "K/V 计数，并包含 canonical packed INT4 RVQ codebook 与 codec "
            "params。这是压缩表示大小，不等同于 GPU resident memory。",
            "4. `hypothetical no-offload GPU-resident estimate`：在 canonical "
            "joint 上再计入默认开启的 `HYWORLDPLAY_RVQ_RUNTIME_CODEBOOK_CACHE`，"
            "即 BF16 centroid 与 FP32 squared-norm 派生表；不含模型权重、"
            "activation 或 allocator reserve。正式视频 run 使用 generation "
            "KV CPU offload，所以这不是该 run 的瞬时 GPU 实测值。",
            "5. `runner dual-stream equivalent`：在线 runner 为了与旧的 "
            "RoPE/PRoPE 双流 final-cache 口径对齐，显式使用 "
            "`kv_logical_head_dim=256` 并将 packed payload 乘 2。这是容量等价"
            "记账，不是本次 `pre_kv_reconstruct` 实际驻留 MiB。",
            "",
            "嵌套仿射使用 G32 且每组保存 FP16 min/max，局部元数据恰为 "
            "`1 bit/value`，所以物理码率为 `b+1`；另三种方案使用 G64 FP16 RMS，"
            "局部元数据为 `0.25 bit/value`，物理码率为 `b+0.25`。固定 seed 的 "
            "Hadamard 符号/旋转不存 payload。",
            "",
            "**公平预算规则：**这里的 `b` 是可截断前缀 plane 数，不是完整"
            "物理 bit/value。因此 affine-b1（2 bit/value）与其余方案-b1"
            "（1.25 bit/value）不是同预算点，不能用“同 b 胜负”宣称同压缩率"
            "优势。跨方案比较必须按下表的物理 bit/value 或实际 payload bytes "
            "配对；离散前缀没有完全相同的点时，应报告相邻 rate-quality 曲线，"
            "而不是静默挑一个近邻。共享参数也必须按目标缓存/数据集只计一次。",
            "",
        ]
    )
    if rate_rows:
        lines.extend(
            [
                markdown_table(
                    (
                        "方案",
                        "前缀 planes b",
                        "物理 bit/value",
                        "residual-only ×",
                        "单缓存+方案参数 ×",
                        "物理维度 canonical joint+方案参数 ×",
                        "no-offload GPU-resident估计+方案参数 ×",
                        "runner 双流等价 joint+方案参数 ×",
                        "canonical joint+完整参数文件 ×（保守下界）",
                    ),
                    [
                        (
                            SCHEME_ZH[row["scheme"]],
                            row["bits"],
                            fmt(
                                row["physical_effective_bits_per_residual_value"],
                                4,
                            ),
                            fmt(row["residual_only_ratio_vs_BF16_residual"], 4),
                            fmt(
                                row[
                                    "single_sampled_cache_residual_ratio_including_scheme_params"
                                ],
                                4,
                            ),
                            fmt(
                                row[
                                    "online_physical_preposition_joint_ratio_including_scheme_params_once"
                                ],
                                4,
                            ),
                            fmt(
                                row[
                                    "online_gpu_resident_estimate_joint_ratio_including_scheme_params_once"
                                ],
                                4,
                            ),
                            fmt(
                                row["online_joint_ratio_including_scheme_params_once"],
                                4,
                            ),
                            fmt(
                                row[
                                    "online_physical_preposition_joint_ratio_including_full_params_file_conservative_lower_bound"
                                ],
                                4,
                            ),
                        )
                        for row in rate_rows
                    ],
                ),
                "",
                "本表物理维度口径不考虑 RoPE/PRoPE 造成的双流等价翻倍：单个 K "
                "向量是 128 维，单个 V 向量也是 128 维，K+V 合计 256 个 "
                "BF16 value。本次 173-frame 视频对应 44 个 latent frame；"
                "结束时 current chunk 占 4 个，历史缓存实际为 10 个 chunk × "
                "每 chunk 4 个 = 40 个 latent history frame。配置中的 "
                "`memory_frames=80` 是 latent-frame 选择预算/上限，本次短视频"
                "没有填满该上限。"
                "runner 的 BF16 等价基线为 "
                f"{fmt(rate_rows[0]['online_BF16_reference_MiB_inferred'], 3)} "
                "MiB，换回实际 pre-position K/V 后为 "
                f"{fmt(rate_rows[0]['online_physical_preposition_BF16_reference_MiB'], 3)} "
                "MiB；runner 的 codebook 等价记账为 "
                f"{fmt(rate_rows[0]['online_shared_rvq_codebook_MiB'], 3)} MiB，"
                "128 维 K/V codebook 的 canonical packed INT4 存储合计为 "
                f"{fmt(rate_rows[0]['online_physical_preposition_shared_rvq_codebook_MiB'], 3)} "
                "MiB。默认 runtime acceleration cache 另外占 "
                f"{fmt(rate_rows[0]['online_runtime_rvq_acceleration_cache_MiB'], 3)} "
                "MiB（BF16 centroids 180 MiB + FP32 squared norms 2.8125 MiB）。"
                "正式 run 设置了 `HYWORLDPLAY_GENERATION_KV_CACHE_CPU_OFFLOAD=1`；"
                "表中 no-offload GPU-resident estimate 只回答“若整套 cache "
                "常驻 GPU”的容量，不是正式 run 的瞬时实测。canonical "
                "compression rate、runner 双流等价 rate 与该估计必须分栏，"
                "不能混称。",
                "",
            ]
        )
    if parameter_rows:
        lines.extend(
            [
                "共享参数开销：",
                "",
                markdown_table(
                    ("对象", "bytes", "MiB", "使用口径"),
                    [
                        (
                            row["scheme_zh"],
                            row["selected_parameter_bytes"],
                            fmt(row["selected_parameter_MiB"], 6),
                            row["scope"],
                        )
                        for row in parameter_rows
                    ],
                ),
                "",
                "`codec_params.pt` 同时保存 selected 参数、direct/Hadamard "
                "ablation candidates 与序列化元数据。将整个文件加到每个候选方案"
                "只应视为保守下界；单方案部署应使用 manifest 的 per-scheme bytes。",
                "",
                "另外，在线 runner 中字段 "
                "`residual_compression_ratio_vs_bf16` 实际计算的是“完整 BF16 "
                "历史缓存字节 / residual payload”，不是“BF16 residual / "
                "residual payload”。因此本报告不把该字段当 residual-only 倍率。",
                "",
            ]
        )

    lines.extend(["## 训练集 held-out 前缀重建", ""])
    selected_rows = selected_training_rows(training)
    if selected_rows:
        by = {
            (row["scheme"], row["tensor"], int_value(row["bits"], "bits")): row
            for row in selected_rows
        }
        prefix_weights = {1: 0.1, 2: 1.0, 3: 0.2, 4: 1.0}

        def transform_score(scheme: str, hadamard: bool) -> float:
            candidates = [
                row
                for row in training.rows
                if row["scheme"] == scheme and bool_value(row["hadamard"]) == hadamard
            ]
            return sum(
                prefix_weights[int_value(row["bits"], "bits")]
                * finite_float(
                    row["relative_residual_MSE"],
                    "relative_residual_MSE",
                )
                for row in candidates
            ) / (2.0 * sum(prefix_weights.values()))

        lines.extend(
            [
                markdown_table(
                    ("方案", "b", "K 相对 residual MSE", "V 相对 residual MSE"),
                    [
                        (
                            SCHEME_ZH[scheme],
                            bits,
                            fmt(by[(scheme, "K", bits)]["relative_residual_MSE"]),
                            fmt(by[(scheme, "V", bits)]["relative_residual_MSE"]),
                        )
                        for scheme in SCHEMES
                        for bits in BITS
                    ],
                ),
                "",
                "Hadamard 选择只使用 20 个训练 prompt 内部与拟合样本不重叠的 "
                "held-out 子集；未使用 10 个测试 prompt、Similarity 或 VBench。",
                "",
                "Hadamard held-out 加权目标消融（越低越好，"
                "`lambda=[0.1,1.0,0.2,1.0]`）：",
                "",
                markdown_table(
                    ("方案", "Direct", "Hadamard", "相对改善"),
                    [
                        (
                            SCHEME_ZH[scheme],
                            fmt(transform_score(scheme, False), 6),
                            fmt(transform_score(scheme, True), 6),
                            fmt(
                                100.0
                                * (
                                    transform_score(scheme, False)
                                    - transform_score(scheme, True)
                                )
                                / transform_score(scheme, False),
                                3,
                            )
                            + "%",
                        )
                        for scheme in SCHEMES
                    ],
                ),
                "",
                "四种方案都选择 Hadamard，但改善仅约 1–3%；而且没有在 test/"
                "exact/video 上跑 no-Hadamard 对照，所以它是小幅训练集预处理收益，"
                "不能表述为 10% 级创新。",
                "",
            ]
        )
    else:
        lines.extend(["未完成。", ""])

    lines.extend(["## Residual codec 速度", ""])
    if runtime.complete:
        runtime_by = row_by_scheme_bit(runtime.rows, "scheme", "bits")
        lines.extend(
            [
                markdown_table(
                    (
                        "方案",
                        "B4 encode ms",
                        "b1 decode ms",
                        "b2 decode ms",
                        "b3 decode ms",
                        "b4 decode ms",
                    ),
                    [
                        (
                            SCHEME_ZH[scheme],
                            fmt(runtime_by[(scheme, 1)]["encode_B4_ms"], 3),
                            *(
                                fmt(
                                    runtime_by[(scheme, bits)]["decode_prefix_ms"],
                                    3,
                                )
                                for bits in BITS
                            ),
                        )
                        for scheme in SCHEMES
                    ],
                ),
                "",
                "测量区间仅含 residual codec：B4 encode、前缀截断、前缀 decode；"
                "不含 L8C64 RVQ 重建。形状为每个 K/V "
                f"`{runtime.metadata.get('shape_per_tensor')}`，CUDA event 中位数。"
                "这是当前 eager PyTorch reference implementation 的端到端算子"
                "时间，不是 fused/Triton kernel 的吞吐上限；方案间速度差只能解释"
                "当前实现复杂度。",
                "",
            ]
        )
    else:
        lines.extend(["未完成。", ""])

    lines.extend(["## 离线 KV 重建与 sampled-attention proxy", ""])
    if offline.complete:
        lines.extend(
            [
                markdown_table(
                    (
                        "方法",
                        "K MSE",
                        "V MSE",
                        "QK logits MSE",
                        "attention KL",
                        "output MSE",
                        "output cosine",
                    ),
                    [
                        (
                            row["method"],
                            fmt(row["K_MSE"]),
                            fmt(row["V_MSE"]),
                            fmt(row["QK_logits_MSE"]),
                            fmt(row["attention_KL"]),
                            fmt(row["QKV_output_MSE"]),
                            fmt(row["QKV_output_cosine"]),
                        )
                        for row in offline.rows
                    ],
                ),
                "",
                "统计覆盖固定 10 test prompts × 30 layers × 24 heads = 7200 "
                "单元，先在 layer/head 内计算，再做 prompt macro 与 prompt bootstrap。"
                "这只是无 positional replay 的 sampled proxy，不是在线 attention。",
                "",
            ]
        )
    else:
        lines.extend(["未完成。", ""])

    lines.extend(["## 精确 positional / attention replay", ""])
    if exact.complete:
        exact_rows = [row for row in exact.rows if row["intervention"] == "K_and_V"]
        paired_exact = exact.metadata["equal_payload_paired"]
        resolved_exact = [
            row for row in paired_exact if row["statistically_resolved_95"]
        ]
        ten_percent_exact = [
            row
            for row in paired_exact
            if row["meets_10pct_relative_improvement_threshold_95"]
        ]
        exact_fingerprint = exact.metadata["source_directory_fingerprint"]
        lines.extend(
            [
                "证据目录：`"
                + exact.metadata["source_directory"]
                + "`；稳定目录 SHA256：`"
                + exact_fingerprint["sha256"]
                + "`（"
                + str(exact_fingerprint["files"])
                + " files，"
                + str(exact_fingerprint["bytes"])
                + " bytes）。",
                "",
                markdown_table(
                    (
                        "方案",
                        "b",
                        "attention half",
                        "K MSE",
                        "V MSE",
                        "attention KL",
                        "output MSE",
                    ),
                    [
                        (
                            SCHEME_ZH[row["scheme"]],
                            row["bits"],
                            row["attention_half"],
                            fmt(row["historical_K_mse_mean"]),
                            fmt(row["historical_V_mse_mean"]),
                            fmt(row["attention_kl_reference_to_candidate_mean"]),
                            fmt(row["attention_output_mse_mean"]),
                        )
                        for row in exact_rows
                    ],
                ),
                "",
                "作用域必须限定为：每个 test prompt × layer 的一个 late "
                "attention-call 快照，历史段 31,680 tokens，当前 BF16 段 "
                "3,520 tokens；层为 0/10/20/29。缓存机制是 video "
                "self-attention `attn1` 的自定义 `ChunkedKVCache`，不是 "
                "cross-attention 或 Q cache；底层历史按视频 latent/chunk 累积，"
                "不是扩散去噪 timestep。这里的表只取一个固定 late-call 快照，"
                "没有沿 latent/chunk 年龄扫描，所以表的横轴也不是时间轴。replay "
                "只替换历史 self-attention KV，保留同一 call 的当前 BF16 "
                "segment，并分别复现 standard "
                "RoPE 与 camera PRoPE 两个非因果 SDPA half。camera half 包含"
                "精确的 apply_fn_o 等价矩阵，但原 capture 不含 learned `to_out`，"
                "因此结果只代表 SDPA-core/camera-output scope。",
                "",
                "K-only、V-only、K+V 三种 intervention 的完整表见 CSV。"
                + (
                    " 新版 exact grid 还显式包含纯 L8C64 INT4 RVQ base。"
                    if exact.metadata.get("pure_rvq_base_present")
                    else " 当前旧版 exact grid 未包含纯 RVQ base；不能从本表"
                    "补写该对照。"
                ),
                "",
                "### 同 residual payload 的胜者",
                "",
                "下表只比较 Gaussian / Additive / Hierarchical-VQ；三者的 "
                "bitplane 与 G64-RMS payload 完全相同。HVQ 的约 2 MiB 共享参数"
                "仍需另计。点估计胜者只有在逐 prompt 配对 bootstrap 的 95% CI "
                "下界大于 0 时才称为“分出胜负”。相对降幅按每次配对 prompt "
                "重采样后的 `1-mean(winner)/mean(runner-up)` 计算；绝对 MSE 差"
                "的 CI 仅作量纲内辅助，不能冒充百分比证据。",
                "",
                markdown_table(
                    (
                        "物理 bit/value",
                        "attention half",
                        "点估计胜者",
                        "runner-up",
                        "相对 output MSE 降幅",
                        "相对 95% CI",
                        "prompt 胜率",
                        "判定",
                        "10% 门槛",
                    ),
                    [
                        (
                            fmt(row["physical_bits_per_value"], 2),
                            row["attention_half"],
                            SCHEME_ZH[row["winner"]],
                            SCHEME_ZH[row["runner_up"]],
                            fmt(
                                row["ratio_of_means_relative_output_mse_reduction_pct"],
                                2,
                            )
                            + "%",
                            "["
                            + fmt(
                                row[
                                    "ratio_of_means_relative_output_mse_reduction_ci95_low_pct"
                                ],
                                2,
                            )
                            + "%, "
                            + fmt(
                                row[
                                    "ratio_of_means_relative_output_mse_reduction_ci95_high_pct"
                                ],
                                2,
                            )
                            + "%]",
                            fmt(100.0 * row["prompt_win_rate"], 1) + "%",
                            (
                                "支持点估计胜者"
                                if row["statistically_resolved_95"]
                                else "未分出胜负（95% CI 跨 0）"
                            ),
                            (
                                "CI 下界 >10%"
                                if row["meets_10pct_relative_improvement_threshold_95"]
                                else "不满足"
                            ),
                        )
                        for row in paired_exact
                    ],
                ),
                "",
                f"共 {len(paired_exact)} 个 bit×half 配对，其中 "
                f"{len(resolved_exact)} 个的 95% CI 完全大于 0；其余 "
                f"{len(paired_exact) - len(resolved_exact)} 个按本报告判定规则"
                "记为未分出胜负。报告不再根据 aggregate 最小值硬编码默认方案。",
                "",
                f"按用户的 10% 门槛，只有相对降幅 bootstrap 95% CI 下界"
                f"严格大于 10% 才计入；当前满足的探索性格点为 "
                f"{len(ten_percent_exact)}/{len(paired_exact)}。这里共有 8 个 "
                "bit×attention-half 比较、每格 n=10 prompts，未做多重比较校正；"
                "而且 winner/runner-up 也在同一批 10 prompts 上事后选择，所报 "
                "CI 是选定 pair 的条件 CI，未校正选择偏差。因此即使过门槛也只"
                "能记为待独立复现实验的探索性证据，不能直接写成已确认创新。",
                "",
                "绝对配对差及其 CI、相对 ratio-of-means 点估计/CI 均保留在 "
                "`exact_equal_payload_comparisons.csv`。",
                "",
                "两个 attention half 的输出变换不同，绝对 output MSE 必须分开"
                "报告，不能把 standard 与 camera 的数值直接平均成一个指标。",
                "",
            ]
        )
    else:
        lines.extend(["未完成。", ""])

    lines.extend(["## Similarity 与 VBench", ""])
    if video.complete:
        paired_rows = video.metadata["paired_rows"]
        equal_payload_rows = video.metadata["equal_payload_rows"]
        vbench_signal_rows = video.metadata["vbench_signal_audit_rows"]
        similarity_protocol = video.metadata["similarity_protocol"]
        legacy_control = video.metadata["legacy_similarity_control"]
        base_pairs = {
            (str(row["candidate"]), str(row["metric"])): row
            for row in paired_rows
            if row["contrast"] == "residual_prefix_vs_pure_l8"
        }

        def paired_cell(row: Mapping[str, Any]) -> str:
            low = finite_float(row["bootstrap_95_ci_low"], "paired low")
            high = finite_float(row["bootstrap_95_ci_high"], "paired high")
            if low > 0:
                verdict = "正"
            elif high < 0:
                verdict = "负"
            else:
                verdict = "未定"
            return (
                fmt(row["mean_paired_improvement"], 5)
                + " ["
                + fmt(low, 5)
                + ","
                + fmt(high, 5)
                + "]; "
                + fmt(
                    100.0 * finite_float(row["prompt_win_rate"], "paired win rate"),
                    0,
                )
                + "%; "
                + verdict
            )

        def equal_payload_improvement_cell(row: Mapping[str, Any]) -> str:
            if row["metric"] == "psnr":
                return (
                    fmt(row["mean_paired_improvement"], 3)
                    + " dB ["
                    + fmt(row["paired_improvement_ci95_low"], 3)
                    + ","
                    + fmt(row["paired_improvement_ci95_high"], 3)
                    + "]"
                )
            return (
                fmt(row["relative_improvement_pct"], 3)
                + "% ["
                + fmt(row["relative_improvement_ci95_low_pct"], 3)
                + "%,"
                + fmt(row["relative_improvement_ci95_high_pct"], 3)
                + "%]"
            )

        candidate_metric_rows = []
        vbench_resolution_rows = []
        jointly_positive_similarity: list[str] = []
        for scheme in SCHEMES:
            for bits in BITS:
                setting = candidate_setting(scheme, bits)
                similarity_pairs = [
                    base_pairs[(setting, metric)] for metric in SIMILARITY_METRICS
                ]
                if all(
                    finite_float(row["bootstrap_95_ci_low"], "paired low") > 0
                    for row in similarity_pairs
                ):
                    jointly_positive_similarity.append(setting)
                candidate_metric_rows.append(
                    (
                        SCHEME_ZH[scheme],
                        bits,
                        *(
                            paired_cell(base_pairs[(setting, metric)])
                            for metric in SIMILARITY_METRICS
                        ),
                    )
                )
                dimension_pairs = [
                    base_pairs[(setting, metric)] for metric in SUPPORTED_VBENCH
                ]
                positive = sum(
                    finite_float(row["bootstrap_95_ci_low"], "paired low") > 0
                    for row in dimension_pairs
                )
                negative = sum(
                    finite_float(row["bootstrap_95_ci_high"], "paired high") < 0
                    for row in dimension_pairs
                )
                vbench_resolution_rows.append(
                    (
                        SCHEME_ZH[scheme],
                        bits,
                        positive,
                        negative,
                        len(SUPPORTED_VBENCH) - positive - negative,
                    )
                )
        lines.extend(
            [
                markdown_table(
                    (
                        "setting",
                        "PSNR",
                        "SSIM",
                        "LPIPS",
                        "VBench 10列审计均值（非总分）",
                    ),
                    [
                        (
                            row["setting"],
                            fmt(row["psnr"], 4),
                            fmt(row["ssim"], 6),
                            fmt(row["lpips"], 6),
                            fmt(
                                row["vbench_10d_unweighted_descriptive_mean"],
                                6,
                            ),
                        )
                        for row in video.rows
                    ],
                ),
                "",
                "VBench 10 个 custom-input 维度明细：",
                "",
                markdown_table(
                    ("setting", *SUPPORTED_VBENCH),
                    [
                        (
                            row["setting"],
                            *(fmt(row[dimension], 4) for dimension in SUPPORTED_VBENCH),
                        )
                        for row in video.rows
                    ],
                ),
                "",
                "Similarity 使用同 prompt/seed 的 BF16 视频，帧对齐并排除 "
                "frame 0。本报告实际读取的是 "
                f"`{similarity_protocol['protocol_id']}`（目录 "
                f"`{similarity_protocol['artifact_directory']}`）：LPIPS "
                "明确以 `normalize=True` 将解码后的 [0,1] RGB 转到标准 "
                "[-1,1] 输入域。",
                "",
                "旧 `similarity/` 目录的 LPIPS 因把 [0,1] 张量直接传给 "
                "LPIPS 0.1.4 而无效，旧 LPIPS 未进入任何报告表或统计。该 "
                "normalize 开关不参与 PSNR、SSIM、MSE 计算；严格审计还逐行"
                f"复核了 {legacy_control['rows']} 个旧/v2 对照，PSNR、SSIM、"
                "MSE 的最大绝对差分别为 "
                f"{legacy_control['psnr_max_abs_delta']:.3e}、"
                f"{legacy_control['ssim_max_abs_delta']:.3e}、"
                f"{legacy_control['mse_max_abs_delta']:.3e}，因此这三项不受"
                "此次修复影响。为保持单一来源，表中的 PSNR/SSIM 也统一从 "
                "normalized-v2 bundle 读取。",
                "",
                "VBench 报告 custom-input 可用的 10 个维度；object_class、"
                "multiple_objects、scene、appearance_style、color、"
                "spatial_relationship 在当前安装的 custom-input 模式不可用，"
                "不得记为 0。`imaging_quality` 的逐 prompt 原始输出按固定 "
                "0.01 缩放，其余维度按 1.0；上表 10 列算术平均仅作为数据完整性"
                "检查，不是 VBench 官方综合分数。",
                "",
                "VBench 信号独立性审计：",
                "",
                markdown_table(
                    (
                        "维度",
                        "min",
                        "max",
                        "常数",
                        "与前维度逐项重复",
                        "输入缩放",
                        "独立且可变",
                    ),
                    [
                        (
                            row["dimension"],
                            fmt(row["minimum"], 6),
                            fmt(row["maximum"], 6),
                            row["constant"],
                            row["duplicate_of"] or "—",
                            fmt(row["per_prompt_scale_applied"], 2),
                            row["counts_as_unique_varying_signal"],
                        )
                        for row in vbench_signal_rows
                    ],
                ),
                "",
                f"10 列中只有 "
                f"{sum(bool(row['counts_as_unique_varying_signal']) for row in vbench_signal_rows)} "
                "个独立且会变化的信号：`dynamic_degree` 在所有 setting/prompt "
                "恒为 1，`human_action` 恒为 0；`overall_consistency` 与 "
                "`temporal_style` 在全部 180 个 setting×prompt 上逐项完全相同，"
                "当前安装的两个实现也执行同一种 ViCLIP/prompt 相似度流程。"
                "因此不能双计这两列，也不能把两个常数维度当区分证据。",
                "",
                "BF16 与纯 L8 基线复用 2026-07-22 的同 prompt/seed 权威视频和 "
                "VBench bundle；16 个候选来自 2026-07-23。本报告重新校验了"
                "逐 prompt 文本、视频路径、帧数与文件 hash，未把不同 prompt "
                "集合混在一起。",
                "",
                "原始 VBench run 没有记录执行时 Python package hash；"
                "`evidence_manifest.json` 额外保存了当前已安装 10 个维度模块"
                "的事后 SHA256 snapshot，但它不能密码学证明运行后模块从未变化。"
                "这是 VBench provenance 的剩余限制。",
                "",
                "### 同物理码率 codec 间的逐 prompt 比较",
                "",
                "Gaussian、Additive、HVQ 在每个 b 都同为 `b+0.25` "
                "bit/residual-value；Affine 为 `b+1`，没有完全同码率对手，"
                "所以不进入本表。winner/runner 在同一批 10 prompts 上事后"
                "选择，CI 未校正选择偏差或多重比较，只能作为探索性结果。",
                "",
                markdown_table(
                    (
                        "bpp",
                        "指标",
                        "winner",
                        "runner",
                        "winner mean",
                        "runner mean",
                        "改善 [95% CI]",
                        "prompt胜率",
                        "CI下界>10%",
                    ),
                    [
                        (
                            fmt(row["physical_bits_per_value"], 2),
                            str(row["metric"]).upper(),
                            SCHEME_ZH[str(row["winner"])],
                            SCHEME_ZH[str(row["runner_up"])],
                            fmt(row["winner_mean"], 6),
                            fmt(row["runner_up_mean"], 6),
                            equal_payload_improvement_cell(row),
                            fmt(
                                100.0
                                * finite_float(
                                    row["prompt_win_rate"],
                                    "equal-rate win rate",
                                ),
                                0,
                            )
                            + "%",
                            (
                                "不适用（dB）"
                                if row["metric"] == "psnr"
                                else row[
                                    "meets_10pct_relative_improvement_threshold_95"
                                ]
                            ),
                        )
                        for row in equal_payload_rows
                        if row["metric"] in {"psnr", "ssim", "lpips", "mse"}
                    ],
                ),
                "",
                (
                    "PSNR 是对数量纲 dB，本表对 PSNR 报告绝对 ΔdB，不把 dB "
                    "数值之比解释为线性质量百分比，也不对它套用 10% 门槛。"
                    "SSIM/LPIPS 主指标中，同码率没有任何一格达到“相对改善 "
                    "95% CI 下界 >10%”。唯一过线的是 1.25 bpp 的逐像素 MSE："
                    + "；".join(
                        SCHEME_ZH[str(row["winner"])]
                        + " 相对 "
                        + SCHEME_ZH[str(row["runner_up"])]
                        + " 改善 "
                        + fmt(row["relative_improvement_pct"], 3)
                        + "% ["
                        + fmt(
                            row["relative_improvement_ci95_low_pct"],
                            3,
                        )
                        + "%,"
                        + fmt(
                            row["relative_improvement_ci95_high_pct"],
                            3,
                        )
                        + "%]"
                        for row in equal_payload_rows
                        if row["metric"] == "mse"
                        and row["meets_10pct_relative_improvement_threshold_95"]
                    )
                    + "。像素 MSE 不是 LPIPS，不能把它改写为感知质量过线。"
                ),
                "",
                "同码率 VBench 的 winner-vs-runner 摘要：",
                "",
                markdown_table(
                    (
                        "bpp",
                        "最大点估计维度",
                        "最大相对差",
                        "该项95% CI",
                        "raw 10维中CI>0",
                        "raw 10维中CI下界>10%",
                    ),
                    [
                        (
                            fmt(bits + 0.25, 2),
                            max(
                                (
                                    row
                                    for row in equal_payload_rows
                                    if row["bits"] == bits
                                    and row["metric"] in SUPPORTED_VBENCH
                                    and row["relative_improvement_pct"] is not None
                                ),
                                key=lambda row: finite_float(
                                    row["relative_improvement_pct"],
                                    "VBench relative point",
                                ),
                            )["metric"],
                            fmt(
                                max(
                                    finite_float(
                                        row["relative_improvement_pct"],
                                        "VBench relative point",
                                    )
                                    for row in equal_payload_rows
                                    if row["bits"] == bits
                                    and row["metric"] in SUPPORTED_VBENCH
                                    and row["relative_improvement_pct"] is not None
                                ),
                                3,
                            )
                            + "%",
                            (
                                lambda best: (
                                    "["
                                    + fmt(
                                        best["relative_improvement_ci95_low_pct"],
                                        3,
                                    )
                                    + "%,"
                                    + fmt(
                                        best["relative_improvement_ci95_high_pct"],
                                        3,
                                    )
                                    + "%]"
                                )
                            )(
                                max(
                                    (
                                        row
                                        for row in equal_payload_rows
                                        if row["bits"] == bits
                                        and row["metric"] in SUPPORTED_VBENCH
                                        and row["relative_improvement_pct"] is not None
                                    ),
                                    key=lambda row: finite_float(
                                        row["relative_improvement_pct"],
                                        "VBench relative point",
                                    ),
                                )
                            ),
                            sum(
                                bool(row["statistically_resolved_95"])
                                for row in equal_payload_rows
                                if row["bits"] == bits
                                and row["metric"] in SUPPORTED_VBENCH
                            ),
                            sum(
                                bool(
                                    row["meets_10pct_relative_improvement_threshold_95"]
                                )
                                for row in equal_payload_rows
                                if row["bits"] == bits
                                and row["metric"] in SUPPORTED_VBENCH
                            ),
                        )
                        for bits in BITS
                    ],
                ),
                "",
                f"raw 计数合计："
                f"{sum(bool(row['statistically_resolved_95']) for row in equal_payload_rows if row['metric'] in SUPPORTED_VBENCH)}/"
                f"{len(BITS) * len(SUPPORTED_VBENCH)} 个同码率 VBench 比较的未校正 "
                "95% CI >0；"
                f"{sum(bool(row['meets_10pct_relative_improvement_threshold_95']) for row in equal_payload_rows if row['metric'] in SUPPORTED_VBENCH)}/"
                f"{len(BITS) * len(SUPPORTED_VBENCH)} 个达到 10% 门槛。由于存在"
                "重复/常数维度且做了 winner 选择与 40 次扫描，这些计数不能当作"
                "确认性显著性结论；结论只取“没有 VBench 10% 证据”。完整明细见 "
                "`video_equal_payload_comparisons.csv`。",
                "",
                "### 相对纯 L8 的逐 prompt 配对证据",
                "",
                "每格依次为“方向统一后的平均改善 [95% bootstrap CI]；prompt "
                "胜率；判定”。PSNR/SSIM 为 candidate−base，LPIPS 为 "
                "base−candidate，所以正值一律表示 residual prefix 更好；CI "
                "跨 0 必须记为“未定”。",
                "",
                markdown_table(
                    ("方案", "b", "PSNR", "SSIM", "LPIPS"),
                    candidate_metric_rows,
                ),
                "",
                "VBench 原始 10 列的统计判定数量（含重复/常数列，仅作审计）：",
                "",
                markdown_table(
                    ("方案", "b", "显著正向维度", "显著负向维度", "未定维度"),
                    vbench_resolution_rows,
                ),
                "",
                "上述 16×10 次相对 L8 比较同样没有做多重比较校正，n=10；"
                "显著维度数量只作探索性描述，不能读成确认性 VBench 总分。",
                "",
                (
                    "Similarity 三项 CI 同时为正的候选由数据动态筛得："
                    + "、".join(jointly_positive_similarity)
                    + "。这只说明相对纯 L8 的增量质量证据，不构成候选 codec "
                    "之间的同码率排名。"
                    if jointly_positive_similarity
                    else "没有任何候选在 PSNR、SSIM、LPIPS 三项上都获得 "
                    "95% CI 同时为正的证据，因此当前不能宣称 residual codec "
                    "相对纯 L8 有稳健的三指标收益。"
                ),
                "",
                "`same_prefix_count_vs_nested_affine_not_rate_matched` 只比较相同"
                "前缀 plane 数；由于 affine 多 0.75 bit/value，它不是同物理码率"
                "对照。incremental refinement 与该对照的完整 paired CI/胜率"
                "保留在 `paired_comparisons.csv`。",
                "",
                "在线视频运行时间（每个 173-frame 视频的 10-prompt 均值）：",
                "",
                markdown_table(
                    (
                        "方案",
                        "b",
                        "KV encode s",
                        "KV decode s",
                        "decode μs/call",
                        "端到端 s",
                    ),
                    [
                        (
                            SCHEME_ZH[row["scheme"]],
                            row["prefix_bits"],
                            fmt(row.get("online_kv_encode_time_s_mean"), 4),
                            fmt(row.get("online_kv_decode_time_s_mean"), 4),
                            fmt(
                                row.get("online_kv_decode_us_per_call_mean"),
                                3,
                            ),
                            fmt(row.get("e2e_time_s_mean"), 3),
                        )
                        for row in video.rows
                        if row["scheme"] in SCHEMES
                    ],
                ),
                "",
                "这里的 encode 是在线 L8 RVQ encode 加 residual encode；decode "
                "是所有去噪调用中压缩 span 的 K/V 反量化、residual decode 与"
                "相加/contiguous 总和；不含 `ChunkedKVCache.read` 外层拼接、"
                "后续 RoPE/PRoPE 重建或 attention。encode timer 包含 cache "
                "read、30 层压缩与 store。它们与前面的 residual-codec-only "
                "CUDA microbenchmark 不是同一测量区间，不能直接相减。"
                "当前在线 codec 路径同样是 eager/reference 实现，未做 fused "
                "kernel 优化；绝对时间不能当作优化后部署速度。",
                "",
            ]
        )
    else:
        lines.extend(["未完成。", ""])

    implementation_locked = [
        name
        for name, section in (
            ("training", training),
            ("runtime", runtime),
            ("offline", offline),
            ("exact", exact),
            ("video", video),
        )
        if isinstance(
            section.metadata.get("run_info", {}).get("implementation_hash"),
            Mapping,
        )
    ]
    implementation_unlocked = [
        name
        for name, section in (
            ("training", training),
            ("runtime", runtime),
            ("offline", offline),
            ("exact", exact),
            ("video", video),
        )
        if section.complete and name not in implementation_locked
    ]
    assignment_record = training.metadata.get("rvq_assignment")
    if isinstance(assignment_record, Mapping):
        assignment_note = (
            "- 训练 artifact 声明的 RVQ assignment mode 为 `"
            + str(assignment_record.get("mode", "unknown"))
            + "`；该字段已作为来源证据保留。"
        )
    else:
        assignment_note = (
            "- 当前训练 artifact 未记录 RVQ assignment/累加精度协议。历史路径"
            "可能使用 FP32 assignment proxy，而 exact/video 使用 BF16 runtime "
            "assignment；因此在 strict runtime-BF16 重跑完成前，训练/离线与"
            "在线结果之间存在协议不一致风险。"
        )
    lines.extend(
        [
            "## 实现审计与实验限制",
            "",
            *implementation_audit_notes(training),
            "- HVQ 内部有 16 个 1-bit/4D plane；本轮公共 API 与实验只验证 "
            "`T=4/8/12/16`，即整数 1/2/3/4 bit/value。尚未验证任意 "
            "0.25-bit 截断或 `T=6`。",
            "- exact replay 的运行配置强制 `verify_capture_hashes=true`；报告"
            "阶段重新核对 10 个 capture-source run_info 的 SHA256、其中登记的"
            "每个 capture SHA256/token/scope 元数据和当前文件大小，但不再次读取"
            "约数十 GiB capture 全内容做第二遍 SHA256。exact 输出目录本身另有"
            "稳定目录 fingerprint。",
            assignment_note,
            "- 来源 run_info 中含 `implementation_hash` 的已完成阶段："
            + ("、".join(implementation_locked) if implementation_locked else "无")
            + "；不含该字段的已完成阶段："
            + ("、".join(implementation_unlocked) if implementation_unlocked else "无")
            + "。报告 manifest 的 `current_implementation_snapshot` 只是生成"
            "报告时的源码快照，不能反向证明未锁定阶段运行时使用的历史源码。",
            "",
            "## 结论使用边界",
            "",
            "- 离线 sampled-attention proxy 与精确 positional replay 必须分栏"
            "报告，不能把前者称为在线 attention 误差。",
            "- 视频日志里的旧 RVQ diagnostics 位于 residual correction 之前，"
            "不能代表四个候选 residual codec 的 attention 误差。",
            "- 所有最终质量结论应同时查看 10-prompt paired comparison、"
            "prompt bootstrap CI 和逐 prompt 胜率，不能只看 aggregate 单点。",
            "- `hierarchical_vq` 的约 2 MiB 共享表在小缓存口径下不可忽略；"
            "在完整视频缓存上应只计一次。",
            "",
        ]
    )
    return "\n".join(lines)


def safe_tag(value: str) -> str:
    if not value or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for ch in value
    ):
        raise ValueError("--partial-tag must contain only letters, digits, _ or -")
    return value


def artifact_name(base: str, final: bool, tag: str) -> str:
    if final:
        return base
    stem, suffix = os.path.splitext(base)
    return f"{stem}_{tag}{suffix}"


def create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def validate_create_only_targets(targets: Sequence[Path]) -> None:
    """Reject path collisions and pre-existing outputs before any write."""

    resolved = [path.resolve() for path in targets]
    if len(set(resolved)) != len(resolved):
        duplicates = sorted(
            str(path) for path in set(resolved) if resolved.count(path) > 1
        )
        raise FileExistsError(
            "Derived artifact targets collide: " + ", ".join(duplicates)
        )
    existing = [str(path) for path in resolved if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing derived artifact(s): " + ", ".join(existing)
        )


def exact_attention_source_record(
    section: Section,
    *,
    directory: Path,
    selection_policy: str,
) -> dict[str, Any]:
    fingerprint = section.metadata.get("source_directory_fingerprint")
    config_evidence = next(
        (
            item
            for item in section.evidence
            if item.get("role") == "exact-attention config"
        ),
        None,
    )
    return {
        "directory": str(directory.resolve()),
        "selection_policy": selection_policy,
        "directory_fingerprint": fingerprint,
        "config_path": (
            config_evidence.get("path")
            if isinstance(config_evidence, Mapping)
            else None
        ),
        "config_sha256": (
            config_evidence.get("sha256")
            if isinstance(config_evidence, Mapping)
            else None
        ),
    }


def main() -> int:
    args = parse_args()
    root = args.benchmark_root.resolve()
    exact_directory, exact_selection_policy = resolve_exact_attention_dir(
        root, args.exact_attention_dir
    )
    tag = safe_tag(args.partial_tag)
    training = audit_training(root)
    runtime = audit_runtime(root, training)
    offline = audit_offline(root, training)
    exact = audit_exact(
        root,
        training,
        exact_attention_dir=exact_directory,
        selection_policy=exact_selection_policy,
    )
    video = audit_video(root, training)
    sections = {
        section.name: section for section in (training, runtime, offline, exact, video)
    }
    all_complete = all(
        section.complete and section.valid for section in sections.values()
    )
    status_payload = {
        name: {
            "status": section.status(),
            "rows": len(section.rows),
            "missing": section.missing,
            "errors": section.errors,
            "warnings": section.warnings,
        }
        for name, section in sections.items()
    }
    status_payload["exact_attention"]["source_directory"] = str(exact_directory)
    status_payload["exact_attention"]["source_directory_selection_policy"] = (
        exact_selection_policy
    )
    status_payload["exact_attention"]["source_directory_fingerprint"] = (
        exact.metadata.get("source_directory_fingerprint")
    )
    print(json.dumps(status_payload, indent=2, ensure_ascii=False))
    if args.audit_only:
        return 0 if all_complete else 2
    if not all_complete and not args.allow_partial:
        raise AuditError(
            "Final report evidence is incomplete or invalid. No artifact was "
            "written. Pass --allow-partial only for an explicitly marked draft."
        )
    final = all_complete
    if args.allow_partial and all_complete:
        # Complete evidence should produce final paths even if the permissive flag
        # was left on; there is no reason to label a complete report as partial.
        final = True

    parameter_rows = build_parameter_rows(training)
    rate_rows = build_rate_rows(training, runtime, offline, video)
    selected_train = selected_training_rows(training)
    exact_source = exact_attention_source_record(
        exact,
        directory=exact_directory,
        selection_policy=exact_selection_policy,
    )
    vbench_implementation_snapshot = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in VBENCH_CURRENT_IMPLEMENTATION_PATHS
        if path.is_file()
    ]
    outputs: dict[str, bytes] = {}

    def add_csv(name: str, rows: Sequence[Mapping[str, Any]]) -> None:
        if rows:
            outputs[artifact_name(name, final, tag)] = csv_bytes(rows)

    add_csv("training_quality.csv", selected_train)
    add_csv("parameter_overhead.csv", parameter_rows)
    add_csv("codec_runtime.csv", runtime.rows)
    add_csv("offline_kv_attention.csv", offline.rows)
    add_csv("exact_attention.csv", exact.rows)
    if exact.complete:
        add_csv(
            "exact_equal_payload_comparisons.csv",
            exact.metadata["equal_payload_paired"],
        )
    add_csv("video_quality.csv", video.rows)
    if video.complete:
        add_csv("runtime_per_prompt.csv", video.metadata["runtime_rows"])
        add_csv("similarity_per_prompt.csv", video.metadata["similarity_rows"])
        add_csv("vbench_per_prompt.csv", video.metadata["vbench_rows"])
        add_csv("paired_comparisons.csv", video.metadata["paired_rows"])
        add_csv(
            "video_equal_payload_comparisons.csv",
            video.metadata["equal_payload_rows"],
        )
        add_csv(
            "vbench_signal_audit.csv",
            video.metadata["vbench_signal_audit_rows"],
        )
    add_csv("compression_accounting.csv", rate_rows)

    config = {
        "schema_version": "truncatable_residual_codec_report.config.v1",
        "benchmark_root": str(root),
        "exact_attention_source": exact_source,
        "output_dir": str(args.output_dir.resolve()),
        "report_path": str(args.report_path.resolve()),
        "report_status": "final" if final else "partial",
        "partial_tag": None if final else tag,
        "required_schemes": list(SCHEMES),
        "required_bits": list(BITS),
        "required_test_prompts": 10,
        "required_offline_units": 7200,
        "required_exact_layers": [0, 10, 20, 29],
        "required_vbench_dimensions": list(SUPPORTED_VBENCH),
        "vbench_current_installed_implementation_snapshot": (
            vbench_implementation_snapshot
        ),
        "vbench_snapshot_scope": (
            "post-hoc current installed modules; source VBench runs did not "
            "record execution-time package hashes"
        ),
        "rate_accounting": {
            "memory_unit": "MiB = 2**20 bytes",
            "residual_only_baseline": (
                "physical-head-dim=128 BF16 residual, 16 bits/value"
            ),
            "online_joint_baseline": (
                "physical-dimension canonical compressed representation: K "
                "head_dim=128 and V head_dim=128; hypothetical no-offload "
                "GPU-resident estimate also charges runtime acceleration "
                "codebook caches; the formal run uses generation KV CPU "
                "offload; runner retains a separately labelled "
                "RoPE/PRoPE dual-stream-equivalent dim=256 accounting"
            ),
            "measured_final_history": (
                "10 historical chunks * 4 latent frames/chunk = 40 latent "
                "history frames; pipeline memory_frames=80 is a latent-frame "
                "selection budget/upper bound not filled by this short video"
            ),
            "residual_payload_includes": (
                "packed prefix planes and per-block FP16 scale or min/max"
            ),
            "residual_payload_excludes": (
                "RVQ indices, RVQ codebook, globally shared codec parameters"
            ),
            "shared_parameter_variants": (
                "per-scheme selected tensors; all-selected tensors; full "
                "codec_params.pt archival upper bound"
            ),
        },
    }
    config_name = artifact_name("report_config.json", final, tag)
    outputs[config_name] = json_bytes(config)
    config_hash = sha256_bytes(outputs[config_name])

    report = render_report(
        sections,
        final=final,
        source_root=root,
        rate_rows=rate_rows,
        parameter_rows=parameter_rows,
    )
    report_path = args.report_path.resolve()
    if not final:
        report_path = report_path.with_name(
            f"{report_path.stem}_{tag}{report_path.suffix}"
        )

    evidence = sorted(
        (item for section in sections.values() for item in section.evidence),
        key=lambda item: (item["section"], item["path"]),
    )
    derived_preview = {
        str((args.output_dir / name).resolve()): {
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
        }
        for name, payload in outputs.items()
    }
    derived_preview[str(report_path)] = {
        "bytes": len(report.encode("utf-8")),
        "sha256": sha256_bytes(report.encode("utf-8")),
    }
    manifest = {
        "schema_version": "truncatable_residual_codec_report.evidence_manifest.v1",
        "status": "final" if final else "partial",
        "generated_at": utc_now(),
        "benchmark_root": str(root),
        "exact_attention_source": exact_source,
        "generator_path": str(Path(__file__).resolve()),
        "generator_git_commit": git_commit(),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "current_implementation_snapshot": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in IMPLEMENTATION_SNAPSHOT_PATHS
        ],
        "vbench_current_installed_implementation_snapshot": (
            vbench_implementation_snapshot
        ),
        "vbench_snapshot_scope": (
            "post-hoc current installed modules; not an execution-time hash"
        ),
        "implementation_snapshot_scope": IMPLEMENTATION_SNAPSHOT_SCOPE,
        "section_status": status_payload,
        "source_artifacts": evidence,
        "derived_artifacts": derived_preview,
        "accounting_audit": {
            "nested_affine_local_metadata_bits_per_value": 1.0,
            "other_schemes_local_metadata_bits_per_value": 0.25,
            "fixed_hadamard_payload_bytes": 0,
            "codec_params_file_is_conservative_upper_bound": True,
            "physical_online_cache_head_dimensions": {
                "K": 128,
                "V": 128,
                "combined_BF16_values_per_token_head": 256,
            },
            "canonical_packed_rvq_codebook_MiB": 46.40625,
            "runtime_rvq_acceleration_cache": {
                "enabled_by_default": True,
                "BF16_centroids_MiB": 180.0,
                "FP32_squared_norms_MiB": 2.8125,
                "total_MiB": RVQ_RUNTIME_ACCELERATION_CACHE_MIB,
                "canonical_rate_excludes_this_cache": True,
                "hypothetical_no_offload_gpu_resident_estimate_includes_this_cache": (
                    True
                ),
                "formal_video_run_generation_kv_cpu_offload": True,
            },
            "measured_final_physical_history": {
                "chunks": 10,
                "latent_frames_per_chunk": 4,
                "latent_history_frames": 40,
                "pipeline_memory_frames_config": 80,
                "pipeline_memory_frames_semantics": (
                    "latent-frame selection budget/upper bound; not filled by "
                    "this 44-latent-frame video"
                ),
            },
            "runner_dual_stream_equivalent_accounting": {
                "logical_head_dim": 256,
                "packed_payload_multiplier": 2,
                "is_physical_resident_memory": False,
            },
            "online_residual_compression_field_warning": (
                "run_small_eval residual_compression_ratio_vs_bf16 uses full "
                "BF16 cache bytes as numerator; it is not residual-only rate"
            ),
        },
    }
    manifest_name = artifact_name("evidence_manifest.json", final, tag)
    outputs[manifest_name] = json_bytes(manifest)
    prompt_list: list[str] = []
    source_info: Mapping[str, Any] = {}
    if video.complete:
        source_info = video.metadata["run_info"]
        raw_prompts = source_info.get("prompt_list", {})
        if isinstance(raw_prompts, Mapping):
            prompt_list = [str(value) for value in raw_prompts.get("test", [])]
        elif isinstance(raw_prompts, list):
            prompt_list = [str(value) for value in raw_prompts]
    elif training.complete:
        source_info = training.metadata["run_info"]
        raw_prompts = source_info.get("prompt_list", [])
        if isinstance(raw_prompts, list):
            prompt_list = [str(value) for value in raw_prompts]
    run_info = {
        "schema_version": "truncatable_residual_codec_report.run_info.v1",
        "git_commit": git_commit(),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str((args.output_dir / config_name).resolve()),
        "config_hash": config_hash,
        "codebook_path": source_info.get("codebook_path"),
        "codebook_hash": source_info.get("codebook_hash"),
        "codec_params_path": (
            training.metadata.get("params_path") if training.complete else None
        ),
        "codec_params_hash": (
            training.metadata.get("params_hash") if training.complete else None
        ),
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": 42,
        "prompt_list": prompt_list,
        "timestamp": utc_now(),
        "status": "final" if final else "partial",
        "source_git_commits": sorted(
            {
                str(section.metadata.get("run_info", {}).get("git_commit"))
                for section in sections.values()
                if section.metadata.get("run_info", {}).get("git_commit")
            }
        ),
        "derived_artifact_hashes": {
            path: metadata["sha256"] for path, metadata in derived_preview.items()
        },
    }
    run_info_name = artifact_name("run_info.json", final, tag)
    outputs[run_info_name] = json_bytes(run_info)

    targets = [args.output_dir / name for name in outputs] + [report_path]
    validate_create_only_targets(targets)
    # All validation and overwrite checks complete before the first write.
    for name, payload in outputs.items():
        create_only(args.output_dir / name, payload)
    create_only(report_path, report.encode("utf-8"))
    print(
        json.dumps(
            {
                "status": "final" if final else "partial",
                "output_dir": str(args.output_dir.resolve()),
                "report": str(report_path),
                "artifacts": sorted(outputs),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, FileExistsError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
