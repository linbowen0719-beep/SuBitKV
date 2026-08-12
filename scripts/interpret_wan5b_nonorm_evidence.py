#!/usr/bin/env python3
"""Interpret raw Wan 5B no-norm evidence using preregistered rules only.

This is a read-only post-processing stage for
``run_wan5b_nonorm_evidence.py`` outputs.  It never edits raw evidence.  Every
scientific observation is derived from an explicit prediction/decision rule in
``wan5b_nonorm_attention_report_hypotheses.json``; unavailable variants remain
inconclusive instead of being imputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HYPOTHESES = (
    REPO_ROOT / "experiments/long_video/wan5b_nonorm_attention_report_hypotheses.json"
)
SCHEMA = "wan5b_report_evidence.v1"
EXPERIMENT_IDS = {
    "H1": "h1_real_post_rope_attention_timing",
    "H2": "h2_fixed_progressive_control",
    "H3": "h3_radial_angular_decomposition",
    "H4": "h4_real_post_rope_codebook_routing",
    "H5": "h5_real_post_rope_temporal_masks",
    "H6": "h6_paired_pre_post_rope_routing",
    "H7": "h7_matched_rate_offline_gate",
}
RUN_INFO_REQUIRED = (
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
)


class RuleBook:
    """Small, strict accessor for numeric thresholds in registered rule text."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        hypotheses = {item["id"]: item for item in manifest.get("hypotheses", [])}
        self.hypotheses = hypotheses
        self.rules = {
            (hypothesis_id, rule["id"]): str(rule["if"])
            for hypothesis_id, hypothesis in hypotheses.items()
            for rule in hypothesis.get("decision_rules", [])
        }

    def condition(self, hypothesis_id: str, rule_id: str) -> str:
        try:
            return self.rules[(hypothesis_id, rule_id)]
        except KeyError as exc:
            raise ValueError(f"Missing registered rule {hypothesis_id}/{rule_id}") from exc

    def comparison(self, hypothesis_id: str, rule_id: str, operator: str, *, occurrence: int = 0) -> float:
        text = self.condition(hypothesis_id, rule_id)
        pattern = {
            ">=": r">=\s*([0-9]+(?:\.[0-9]+)?)",
            "<=": r"<=\s*([0-9]+(?:\.[0-9]+)?)",
            ">": r"(?<![<>=])>\s*([0-9]+(?:\.[0-9]+)?)",
            "<": r"(?<![<>=])<\s*([0-9]+(?:\.[0-9]+)?)",
        }[operator]
        matches = re.findall(pattern, text)
        if occurrence >= len(matches):
            raise ValueError(
                f"Rule {hypothesis_id}/{rule_id} lacks comparison {operator} occurrence {occurrence}: {text}"
            )
        return float(matches[occurrence])

    def percent(self, hypothesis_id: str, rule_id: str) -> float:
        text = self.condition(hypothesis_id, rule_id)
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*percent", text, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"Rule {hypothesis_id}/{rule_id} lacks a percent threshold: {text}")
        return float(match.group(1)) / 100.0

    def selection_fraction(self, hypothesis_id: str, rule_id: str) -> float:
        text = self.condition(hypothesis_id, rule_id)
        match = re.search(r"selection_fraction\s+([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            raise ValueError(f"Rule {hypothesis_id}/{rule_id} lacks selection_fraction: {text}")
        return float(match.group(1))

    def controlled_fractions(self, hypothesis_id: str) -> tuple[float, ...]:
        controlled = self.hypotheses[hypothesis_id].get("controlled_experiment", {})
        raw = controlled.get("selection_fractions", [])
        return tuple(float(item) for item in raw)

    def validity_gate(self, hypothesis_id: str) -> dict[str, Any]:
        controlled = self.hypotheses[hypothesis_id].get("controlled_experiment", {})
        gate = controlled.get("validity_gate")
        if not isinstance(gate, dict):
            raise ValueError(f"Missing registered validity gate for {hypothesis_id}")
        metric = gate.get("metric")
        try:
            maximum = float(gate["max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Registered validity gate for {hypothesis_id} lacks a finite max"
            ) from exc
        if not isinstance(metric, str) or not metric or not math.isfinite(maximum):
            raise ValueError(f"Invalid registered validity gate for {hypothesis_id}: {gate}")
        return {**gate, "max": maximum}

    def minimum_k_sizes(self, hypothesis_id: str, rule_id: str) -> int:
        text = self.condition(hypothesis_id, rule_id)
        match = re.search(
            r"at least\s+([0-9]+|one|two|three|four|five|six|seven|eight|nine|ten)\s+K sizes",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                f"Rule {hypothesis_id}/{rule_id} lacks an 'at least N K sizes' condition: {text}"
            )
        token = match.group(1).lower()
        if token.isdigit():
            return int(token)
        return {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
        }[token]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="+",
        help="One raw offline run, or multiple H1 runs with --combine-h1-sweep.",
    )
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_HYPOTHESES)
    parser.add_argument(
        "--combine-h1-sweep",
        action="store_true",
        help="Pool only H1 tracks from multiple immutable raw runs into one canonical H1 record.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Must not already exist. Defaults to a timestamped directory below run_dir/interpreted_canonical.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_json_values(values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = json.dumps(value, sort_keys=True, ensure_ascii=False)
        if marker not in seen:
            seen.add(marker)
            unique.append(value)
    return unique


def collapse_json_values(values: Iterable[Any]) -> Any:
    unique = unique_json_values(values)
    if not unique:
        return None
    return unique[0] if len(unique) == 1 else unique


def get(mapping: Any, *keys: str, default: Any = None) -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def median(values: Iterable[float | None]) -> float | None:
    ordered = sorted(value for item in values if (value := finite_number(item)) is not None)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def all_true(values: Sequence[bool | None]) -> bool | None:
    known = [value for value in values if value is not None]
    if not known:
        return None
    if len(known) != len(values):
        return None
    return all(known)


def observation(prediction_id: str, status: str, explanation: str) -> dict[str, str]:
    if status not in {"supported", "contradicted", "inconclusive", "not_tested"}:
        raise ValueError(f"Invalid observation status: {status}")
    return {"prediction_id": prediction_id, "status": status, "explanation": explanation}


def aggregate_verdict(observations: Sequence[dict[str, str]]) -> tuple[str, str]:
    statuses = [item["status"] for item in observations]
    supported = statuses.count("supported")
    contradicted = statuses.count("contradicted")
    unresolved = statuses.count("inconclusive") + statuses.count("not_tested")
    if contradicted and not supported:
        return "contradicted", "至少一条预注册预测被直接反驳，且没有支持项抵消。"
    if supported and contradicted:
        return "partially_supported", "预注册预测同时出现支持和反驳结果。"
    if supported and unresolved:
        return "partially_supported", "已有支持结果，但部分预测仍缺少可判定证据。"
    if supported:
        return "supported", "所有可用的预注册预测均得到支持。"
    return "inconclusive", "没有足够的预注册指标形成支持或反驳判定。"


def timing_ms(payload: Any) -> float | None:
    return finite_number(get(payload, "median_ms"))


def raw_sections(tracks: Sequence[dict[str, Any]], hypothesis_id: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows = []
    for track in tracks:
        section = track.get(hypothesis_id)
        if isinstance(section, dict) and section.get("status") != "unavailable":
            rows.append((track, section))
    return rows


def interpret_h1(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    # Unlike most sections, unavailable/legacy H1 rows are scientifically
    # meaningful here: one guarded row makes the requested K/layer sweep
    # incomplete.  Do not drop them via raw_sections().
    sections = [
        (track, section)
        for track in tracks
        if isinstance((section := track.get("H1")), dict)
    ]
    rows: list[dict[str, Any]] = []
    decode_fractions: list[float] = []
    speedup_rows: list[dict[str, float | int]] = []
    row_validity: list[dict[str, Any]] = []
    validity_gate = rules.validity_gate("H1")
    validity_max = float(validity_gate["max"])
    for track, section in sections:
        timing = section.get("timing", {})
        key_shape = get(track, "shape", "key", default=[])
        k_tokens = key_shape[2] if isinstance(key_shape, list) and len(key_shape) > 2 else None
        layer = track.get("layer")
        dense_qk = timing_ms(timing.get("dense_post_rope_qk"))
        dense_sdpa = timing_ms(timing.get("dense_post_rope_sdpa"))
        exact_ms = timing_ms(timing.get("exact_compressed_streaming_full_attention"))
        standalone_ms = timing_ms(timing.get("standalone_compressed_qk"))
        full_speedup = (
            dense_sdpa / exact_ms
            if dense_sdpa is not None and exact_ms is not None and exact_ms > 0
            else None
        )
        qk_speedup = (
            dense_qk / standalone_ms
            if dense_qk is not None and standalone_ms is not None and standalone_ms > 0
            else None
        )
        exact = section.get("nonmaterializing_exact_compressed", {})
        standalone = section.get("standalone_compressed_qk", {})
        guard = section.get("exact_reference_guard", {})
        equivalence_rel_l2 = finite_number(
            get(section, "accuracy", "streaming_vs_explicit_decode_rope_sdpa", "relative_l2")
        )
        equivalence_flag = exact.get("reference_equivalence_pass") is True
        equivalence_gate_pass = (
            equivalence_rel_l2 is not None
            and equivalence_rel_l2 <= validity_max
            and equivalence_flag
        )
        exact_measured = (
            exact.get("status") == "measured"
            and exact_ms is not None
            and exact_ms > 0
            and guard.get("streaming_full_attention_called") is not False
        )
        standalone_measured = (
            standalone.get("status") == "measured"
            and standalone_ms is not None
            and standalone_ms > 0
            and guard.get("standalone_qk_called") is True
        )
        baselines_measured = (
            dense_qk is not None
            and dense_qk > 0
            and dense_sdpa is not None
            and dense_sdpa > 0
        )
        row_complete = (
            baselines_measured
            and exact_measured
            and standalone_measured
            and equivalence_gate_pass
        )
        row_validity.append(
            {
                "layer": layer,
                "k_tokens": k_tokens,
                "complete": row_complete,
                "exact_status": exact.get("status"),
                "standalone_status": standalone.get("status"),
                "standalone_qk_called": guard.get("standalone_qk_called"),
                "dense_qk_measured": dense_qk is not None and dense_qk > 0,
                "dense_sdpa_measured": dense_sdpa is not None and dense_sdpa > 0,
                "streaming_relative_l2": equivalence_rel_l2,
                "streaming_gate_max": validity_max,
                "reference_equivalence_pass": equivalence_flag,
            }
        )
        if None not in (k_tokens, full_speedup, qk_speedup):
            speedup_rows.append(
                {
                    "layer": int(layer) if layer is not None else -1,
                    "k_tokens": int(k_tokens),
                    "full_attention_speedup_vs_bf16": float(full_speedup),
                    "standalone_qk_speedup_vs_bf16": float(qk_speedup),
                }
            )
        variants = {
            "bf16_qk": timing.get("dense_post_rope_qk"),
            "bf16_sdpa": timing.get("dense_post_rope_sdpa"),
            "decode_pre_k": timing.get("decode_pre_k"),
            "decode_pre_v": timing.get("decode_pre_v"),
            "explicit_decode_rope_qk": timing.get("explicit_decode_rope_qk"),
            "decode_rope_sdpa": timing.get("decode_plus_real_rope_plus_sdpa"),
            "standalone_compressed_qk": timing.get("standalone_compressed_qk"),
            "exact_compressed_streaming_full_attention": timing.get(
                "exact_compressed_streaming_full_attention"
            ),
        }
        for variant, stats in variants.items():
            if isinstance(stats, dict):
                speedup = None
                if variant == "standalone_compressed_qk":
                    speedup = qk_speedup
                elif variant == "exact_compressed_streaming_full_attention":
                    speedup = full_speedup
                rows.append(
                    {
                        "layer": layer,
                        "variant": variant,
                        "k_tokens": k_tokens,
                        "total_latency_ms": timing_ms(stats),
                        "p10_ms": finite_number(stats.get("p10_ms")),
                        "p90_ms": finite_number(stats.get("p90_ms")),
                        "speedup_vs_bf16": speedup,
                        "streaming_reference_relative_l2": (
                            equivalence_rel_l2
                            if variant == "exact_compressed_streaming_full_attention"
                            else None
                        ),
                    }
                )
        dense = dense_sdpa
        restored = timing_ms(timing.get("decode_plus_real_rope_plus_sdpa"))
        dk = timing_ms(timing.get("decode_pre_k"))
        dv = timing_ms(timing.get("decode_pre_v"))
        if None not in (dense, restored, dk, dv) and restored > dense:
            decode_fractions.append((dk + dv) / (restored - dense))

    all_rows_complete = bool(row_validity) and all(item["complete"] for item in row_validity)
    full_threshold = rules.comparison("H1", "H1-R1", "<")
    minimum_k_sizes = rules.minimum_k_sizes("H1", "H1-R1")
    tested_k_sizes = sorted({int(item["k_tokens"]) for item in speedup_rows})
    p1_available = all_rows_complete and len(tested_k_sizes) >= minimum_k_sizes
    p1_supported = p1_available and all(
        item["full_attention_speedup_vs_bf16"] < full_threshold for item in speedup_rows
    )

    standalone_threshold = rules.comparison("H1", "H1-R2", ">")
    long_k_threshold = rules.comparison("H1", "H1-R2", ">=")
    r2_full_threshold = rules.comparison("H1", "H1-R2", "<")
    p2_supported = all_rows_complete and any(
        item["k_tokens"] >= long_k_threshold
        and item["standalone_qk_speedup_vs_bf16"] > standalone_threshold
        and item["full_attention_speedup_vs_bf16"] < r2_full_threshold
        for item in speedup_rows
    )
    r4_threshold = rules.comparison("H1", "H1-R4", "<=")
    r4_no_break_even = all_rows_complete and bool(speedup_rows) and all(
        item["standalone_qk_speedup_vs_bf16"] <= r4_threshold for item in speedup_rows
    )

    p3_value = median(decode_fractions)
    decode_threshold = rules.comparison("H1", "H1-R3", ">=")
    p3 = (
        observation(
            "H1-P3",
            "supported" if p3_value is not None and p3_value >= decode_threshold else "inconclusive",
            f"decode_fraction_of_extra_latency median={p3_value!r}; H1-R3 threshold is >={decode_threshold}.",
        )
        if p3_value is not None
        else observation("H1-P3", "not_tested", "缺少可计算 decode fraction 的完整 timing。")
    )
    observations = [
        observation(
            "H1-P1",
            "supported" if p1_supported else ("inconclusive" if sections else "not_tested"),
            (
                f"H1-R1: {len(tested_k_sizes)} distinct K sizes tested (requires "
                f">={minimum_k_sizes}); exact/BF16 speedup threshold is <{full_threshold}."
            ),
        ),
        observation(
            "H1-P2",
            "supported" if p2_supported else ("inconclusive" if sections else "not_tested"),
            (
                "H1-R2 long-K break-even observed."
                if p2_supported
                else (
                    f"H1-R4: all measured standalone-QK speedups are <={r4_threshold}; "
                    "current shapes do not cross break-even."
                    if r4_no_break_even
                    else "H1-R2 is not satisfied or the exact/standalone sweep is incomplete."
                )
            ),
        ),
        p3,
    ]
    if not sections:
        status = "missing"
        verdict_status = "inconclusive"
        verdict_explanation = "缺少 H1 raw evidence。"
    elif not all_rows_complete:
        status = "partial"
        verdict_status = "inconclusive"
        invalid_rows = [item for item in row_validity if not item["complete"]]
        verdict_explanation = (
            "至少一个请求的 H1 layer/K row 缺少 exact/standalone 实测、被 element guard 跳过，"
            f"或 streaming-vs-explicit relative L2 未通过预注册 gate <={validity_max}; "
            f"invalid_rows={invalid_rows}."
        )
    else:
        status = "complete"
        verdict_status, verdict_explanation = aggregate_verdict(observations)

    applied_rule_ids: list[str] = []
    if p1_available:
        applied_rule_ids.append("H1-R1")
    if all_rows_complete and speedup_rows:
        applied_rule_ids.append("H1-R2")
        if r4_no_break_even:
            applied_rule_ids.append("H1-R4")
    if p3_value is not None:
        applied_rule_ids.append("H1-R3")
    return {
        "status": status,
        "tables": [
            {"title": "H1 timing", "columns": list(rows[0]) if rows else [], "rows": rows},
            {
                "title": "H1 exact-reference validity",
                "columns": list(row_validity[0]) if row_validity else [],
                "rows": row_validity,
            },
        ],
        "observations": observations,
        "verdict": {"status": verdict_status, "explanation": verdict_explanation},
        "applied_rule_ids": applied_rule_ids,
    }


def interpret_h2(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    sections = raw_sections(tracks, "H2")
    rows: list[dict[str, Any]] = []
    metadata_checks: list[bool] = []
    runtime_checks: list[bool] = []
    weak_proxy: list[bool] = []
    strong_proxy: list[bool] = []
    dominance: list[bool] = []
    runtime_references: list[str] = []
    for track, section in sections:
        storage = section.get("storage", {})
        timing = section.get("timing", {})
        post = section.get("post_rope_attention", {})

        def total_storage(variant: str, field: str) -> float | None:
            values = [
                finite_number(get(storage, f"{variant}_k", field)),
                finite_number(get(storage, f"{variant}_v", field)),
            ]
            return sum(values) if all(item is not None for item in values) else None

        fixed8_bytes = total_storage("fixed_l8", "physical_tensor_payload_bytes")
        fixed6_bytes = total_storage("fixed_l6", "physical_tensor_payload_bytes")
        progressive_bytes = total_storage("token_progressive", "physical_tensor_payload_bytes")
        coarse_bytes = total_storage("coarse_chunk", "physical_uint8_index_bytes")
        coarse_schedule_bytes = total_storage("coarse_chunk", "coarse_schedule_bytes")

        def paired_decode(prefix: str) -> float | None:
            values = (
                timing_ms(timing.get(f"{prefix}_decode_k")),
                timing_ms(timing.get(f"{prefix}_decode_v")),
            )
            return sum(values) if all(value is not None for value in values) else None

        fixed8_decode = paired_decode("fixed_l8")
        fixed6_decode = paired_decode("fixed_l6")
        progressive_decode = paired_decode("token_progressive")
        coarse_decode = paired_decode("coarse_chunk")
        progressive_depth = median(
            (
                get(section, "observed_average_depth", "k"),
                get(section, "observed_average_depth", "v"),
            )
        )
        coarse_depth = median(
            (
                get(storage, "coarse_chunk_k", "average_depth"),
                get(storage, "coarse_chunk_v", "average_depth"),
            )
        )
        fixed8_pv = finite_number(get(post, "fixed_l8", "pv", "relative_l2"))
        fixed_pv = finite_number(get(post, "fixed_l6", "pv", "relative_l2"))
        progressive_pv = finite_number(get(post, "token_progressive", "pv", "relative_l2"))
        coarse_pv = finite_number(get(post, "coarse_chunk", "pv", "relative_l2"))
        for variant, payload, latency, pv, depth, separate_schedule in (
            ("fixed_l8", fixed8_bytes, fixed8_decode, fixed8_pv, 8.0, 0.0),
            ("fixed_l6", fixed6_bytes, fixed6_decode, fixed_pv, 6.0, 0.0),
            (
                "token_progressive",
                progressive_bytes,
                progressive_decode,
                progressive_pv,
                progressive_depth,
                0.0,
            ),
            (
                "coarse_chunk",
                coarse_bytes,
                coarse_decode,
                coarse_pv,
                coarse_depth,
                coarse_schedule_bytes,
            ),
        ):
            rows.append(
                {
                    "layer": track.get("layer"),
                    "variant": variant,
                    "payload_bytes": payload,
                    "decode_latency_ms": latency,
                    "pv_relative_l2": pv,
                    "qk_attention_kl": finite_number(get(post, variant, "qk_kl")),
                    "average_depth": depth,
                    "separate_schedule_metadata_bytes": separate_schedule,
                    "payload_bytes_including_separate_schedule": (
                        payload + separate_schedule
                        if payload is not None and separate_schedule is not None
                        else None
                    ),
                }
            )
        if fixed8_bytes is not None and progressive_bytes is not None:
            metadata_checks.append(progressive_bytes >= fixed8_bytes)
        # The newly measured coarse bucket is the intended matched-average-depth
        # regular reference.  Legacy evidence has no coarse row, so retain the
        # registered fixed-L6 fallback instead of imputing a coarse timing.
        matched_decode = None
        reference = None
        if (
            coarse_decode is not None
            and coarse_decode > 0
            and coarse_depth is not None
            and progressive_depth is not None
        ):
            matched_decode = coarse_decode
            reference = "coarse_chunk_matched_average_depth"
        elif fixed6_decode is not None and fixed6_decode > 0:
            matched_decode = fixed6_decode
            reference = "legacy_fixed_l6_fallback"
        if (
            matched_decode is not None
            and progressive_decode is not None
            and progressive_decode > 0
        ):
            runtime_checks.append(
                progressive_decode / matched_decode
                >= rules.comparison("H2", "H2-R2", ">=")
            )
            runtime_references.append(str(reference))
        correlations = section.get("sensitivity_correlation_l8", {})
        corr_values = [
            finite_number(correlations.get("k_residual_rms_vs_qk_sensitivity_spearman")),
            finite_number(correlations.get("v_residual_rms_vs_pv_sensitivity_spearman")),
        ]
        if all(item is not None for item in corr_values):
            weak_proxy.append(
                all(abs(item) < rules.comparison("H2", "H2-R3", "<") for item in corr_values)
            )
            strong_proxy.append(
                any(abs(item) >= rules.comparison("H2", "H2-R4", ">=") for item in corr_values)
            )
        if (
            None not in (
                fixed6_bytes,
                progressive_bytes,
                fixed_pv,
                progressive_pv,
                fixed6_decode,
                progressive_decode,
            )
            and fixed6_decode > 0
            and progressive_decode > 0
        ):
            dominance.append(
                fixed6_bytes <= progressive_bytes
                and fixed_pv <= progressive_pv
                and fixed6_decode <= progressive_decode
            )

    def direct_obs(pid: str, checks: list[bool], explanation: str) -> dict[str, str]:
        if not checks:
            return observation(pid, "not_tested", f"缺少受控数据；{explanation}")
        return observation(pid, "supported" if all(checks) else "inconclusive", explanation)

    if strong_proxy and any(strong_proxy):
        p3 = observation(
            "H2-P3",
            "contradicted",
            "至少一个 K/PV Spearman |rho|>=0.5，触发 H2-R4，反驳弱 sensitivity proxy 预测。",
        )
    elif weak_proxy and all(weak_proxy):
        p3 = observation(
            "H2-P3", "supported", "所有成对 K/PV Spearman |rho|<0.3，触发 H2-R3。"
        )
    else:
        p3 = observation("H2-P3", "inconclusive", "Spearman 未统一落入 H2-R3 或 H2-R4 区间。")
    observations = [
        direct_obs("H2-P1", metadata_checks, "按实测 physical tensor bytes 应用 H2-R1。"),
        direct_obs(
            "H2-P2",
            runtime_checks,
            (
                "按 token-progressive/matched-reference decode ratio"
                f">={rules.comparison('H2', 'H2-R2', '>=')} 应用 H2-R2；"
                f"references={runtime_references}."
            ),
        ),
        p3,
        direct_obs("H2-P4", dominance, "fixed L6 必须同时不增 rate、PV error 与 decode latency 才算支配。"),
    ]
    verdict, explanation = aggregate_verdict(observations)
    return {
        "status": "complete" if sections else "missing",
        "tables": [{"title": "H2 fixed/progressive control", "columns": list(rows[0]) if rows else [], "rows": rows}],
        "observations": observations,
        "verdict": {"status": verdict, "explanation": explanation},
        "applied_rule_ids": ["H2-R1", "H2-R2", "H2-R3", "H2-R4"],
    }


def interpret_h3(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    sections = raw_sections(tracks, "H3")
    rows: list[dict[str, Any]] = []
    scalar_ratios: dict[int, float] = {}
    directional_gains: list[float] = []
    surrogate_mismatches: list[bool] = []
    for track, section in sections:
        methods = section.get("methods", {})
        base = methods.get("uncalibrated", {})
        base_qk = finite_number(base.get("qk_mse"))
        base_pv = finite_number(get(base, "pv", "relative_l2"))
        for name, method in methods.items():
            ratio = finite_number(method.get("qk_error_recoverable_fraction_vs_uncalibrated"))
            rows.append(
                {
                    "layer": track.get("layer"),
                    "method": name,
                    "radial_recoverable_ratio": ratio if name == "block_scalar" else None,
                    "qk_error_reduction_fraction": ratio,
                    "pv_relative_l2": finite_number(get(method, "pv", "relative_l2")),
                }
            )
        scalar = finite_number(get(methods, "block_scalar", "qk_error_recoverable_fraction_vs_uncalibrated"))
        diagonal = finite_number(
            get(methods, "per_channel_diagonal", "qk_error_recoverable_fraction_vs_uncalibrated")
        )
        if scalar is not None:
            scalar_ratios[int(track.get("layer"))] = scalar
        if scalar is not None and diagonal is not None and abs(scalar) > 1e-12:
            directional_gains.append((diagonal - scalar) / abs(scalar))
        mismatch = False
        tested = False
        for name in ("block_scalar", "per_channel_diagonal", "q_cov_full_linear_upper_bound"):
            method = methods.get(name, {})
            qk = finite_number(method.get("qk_mse"))
            pv = finite_number(get(method, "pv", "relative_l2"))
            if None not in (qk, pv, base_qk, base_pv):
                tested = True
                mismatch |= qk < base_qk and pv > base_pv
        if tested:
            surrogate_mismatches.append(mismatch)

    required_layers = {0, 10, 20, 29}
    low_threshold = rules.comparison("H3", "H3-R1", "<=")
    high_threshold = rules.comparison("H3", "H3-R2", ">=")
    directional_threshold = rules.percent("H3", "H3-R3")
    low_count = sum(value <= low_threshold for value in scalar_ratios.values())
    high_count = sum(value >= high_threshold for value in scalar_ratios.values())
    if required_layers.issubset(scalar_ratios) and low_count >= 3:
        p2 = observation("H3-P2", "supported", "四层齐全且至少三层 radial ratio<=0.2，触发 H3-R1。")
    elif required_layers.issubset(scalar_ratios) and high_count >= 3:
        p2 = observation("H3-P2", "contradicted", "四层齐全且至少三层 radial ratio>=0.5，触发 H3-R2。")
    else:
        p2 = observation(
            "H3-P2",
            "not_tested" if not required_layers.issubset(scalar_ratios) else "inconclusive",
            f"observed layers={sorted(scalar_ratios)}；H3-R1/R2 要求四层中至少三层达到阈值。",
        )
    directional = median(directional_gains)
    p3 = (
        observation(
            "H3-P3",
            "supported" if directional is not None and directional >= directional_threshold else "inconclusive",
            f"per-channel 相对 scalar 的 QK reduction median gain={directional!r}；H3-R3 threshold={directional_threshold}。",
        )
        if directional is not None
        else observation("H3-P3", "not_tested", "缺少 scalar/per-channel 成对 reduction。")
    )
    p4 = (
        observation(
            "H3-P4",
            "supported" if any(surrogate_mismatches) else "inconclusive",
            "检查是否存在 QK MSE 改善而 PV relative L2 恶化，按 H3-R4 判定。",
        )
        if surrogate_mismatches
        else observation("H3-P4", "not_tested", "缺少 QK/PV 成对结果。")
    )
    observations = [
        observation("H3-P1", "inconclusive", "manifest 未为 alpha 接近 1 注册数值阈值，解释器不自定义阈值。"),
        p2,
        p3,
        p4,
    ]
    verdict, explanation = aggregate_verdict(observations)
    complete = required_layers.issubset({int(track.get("layer")) for track, _ in sections})
    return {
        "status": "complete" if complete else ("partial" if sections else "missing"),
        "tables": [{"title": "H3 radial/directional controls", "columns": list(rows[0]) if rows else [], "rows": rows}],
        "observations": observations,
        "verdict": {"status": verdict if complete else "inconclusive", "explanation": explanation},
        "applied_rule_ids": ["H3-R1", "H3-R2", "H3-R3", "H3-R4"],
    }


def fraction_item(mapping: Any, fraction: float) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    for key, value in mapping.items():
        number = finite_number(key)
        if number is not None and abs(number - fraction) < 1e-6 and isinstance(value, dict):
            return value
    return None


def interpret_h4(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    sections = raw_sections(tracks, "H4")
    rows: list[dict[str, Any]] = []
    group_fail: list[bool] = []
    recall_checks: list[bool] = []
    monotonic_checks: list[bool] = []
    viable_points = 0
    high_recall_slow = 0
    fractions = rules.controlled_fractions("H4")
    group_threshold = rules.comparison("H4", "H4-R1", ">=")
    recall_threshold = rules.comparison("H4", "H4-R2", ">=")
    viable_speedup = rules.comparison("H4", "H4-R3", ">")
    viable_error = rules.comparison("H4", "H4-R3", "<=")
    target_fraction = rules.selection_fraction("H4", "H4-R2")
    for track, section in sections:
        group_error = finite_number(get(section, "group_mean_output", "relative_l2"))
        if group_error is not None:
            group_fail.append(group_error >= group_threshold)
            rows.append(
                {
                    "layer": track.get("layer"),
                    "variant": "group_mean",
                    "selection_fraction": 1.0,
                    "attention_output_relative_l2": group_error,
                    "top_page_recall": None,
                    "complete_path_speedup_vs_bf16": None,
                }
            )
        fraction_errors = []
        for fraction in fractions:
            item = fraction_item(section.get("fractions"), fraction)
            if item is None:
                continue
            rel = finite_number(get(item, "selected_output", "relative_l2"))
            recall = finite_number(item.get("oracle_page_recall"))
            dense = timing_ms(item.get("dense_sdpa_timing"))
            sparse = timing_ms(item.get("route_selected_index_decode_rope_attention_timing"))
            speedup = dense / sparse if None not in (dense, sparse) and sparse > 0 else None
            rows.append(
                {
                    "layer": track.get("layer"),
                    "variant": "rvq_routed_selected_exact",
                    "selection_fraction": fraction,
                    "attention_output_relative_l2": rel,
                    "top_page_recall": recall,
                    "complete_path_speedup_vs_bf16": speedup,
                }
            )
            tail_rel = finite_number(get(item, "grouped_tail_output", "relative_l2"))
            rows.append(
                {
                    "layer": track.get("layer"),
                    "variant": "rvq_routed_sparse_with_grouped_tail",
                    "selection_fraction": fraction,
                    "attention_output_relative_l2": tail_rel,
                    "top_page_recall": recall,
                    "complete_path_speedup_vs_bf16": None,
                }
            )
            if rel is not None:
                fraction_errors.append((fraction, rel))
            if abs(fraction - target_fraction) < 1e-6 and recall is not None:
                recall_checks.append(recall >= recall_threshold)
                if recall >= recall_threshold and speedup is not None and speedup <= viable_speedup:
                    high_recall_slow += 1
            if None not in (speedup, rel) and speedup > viable_speedup and rel <= viable_error:
                viable_points += 1
        if len(fraction_errors) == len(fractions):
            ordered = [value for _, value in sorted(fraction_errors)]
            monotonic_checks.append(ordered[0] >= ordered[1] >= ordered[2])

    observations = [
        observation(
            "H4-P1",
            "supported" if group_fail and all(group_fail) else ("inconclusive" if group_fail else "not_tested"),
            "按 H4-R1 检查每层 group-mean output relative L2>=0.5。",
        ),
        observation(
            "H4-P2",
            "supported" if recall_checks and all(recall_checks) else ("inconclusive" if recall_checks else "not_tested"),
            "按 H4-R2 检查 selection_fraction=0.2 时 top-page recall>=0.8。",
        ),
        observation(
            "H4-P3",
            "supported" if monotonic_checks and all(monotonic_checks) else ("inconclusive" if monotonic_checks else "not_tested"),
            "按预注册 fractions 检查 sparse output error 是否单调不增。",
        ),
        observation(
            "H4-P4",
            "not_tested",
            "当前 raw run 没有同层多 K 长度扫描，不能判定速度随 K 增长；不从单一 K 外推。",
        ),
    ]
    verdict, explanation = aggregate_verdict(observations)
    if viable_points:
        explanation += f" 另有 {viable_points} 个点满足 H4-R3 的 speedup>1 且 relL2<=0.15。"
    elif high_recall_slow:
        explanation += f" {high_recall_slow} 个点触发 H4-R4：representation recall 高但 runtime 尚不可行。"
    layers = {int(track.get("layer")) for track, _ in sections}
    complete = {10, 20}.issubset(layers)
    return {
        "status": "complete" if complete else ("partial" if sections else "missing"),
        "tables": [{"title": "H4 codebook routing", "columns": list(rows[0]) if rows else [], "rows": rows}],
        "observations": observations,
        "verdict": {"status": verdict if complete else "inconclusive", "explanation": explanation},
        "applied_rule_ids": ["H4-R1", "H4-R2", "H4-R3", "H4-R4"],
    }


def interpret_h5(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    sections = raw_sections(tracks, "H5")
    rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    profile_summary: list[dict[str, Any]] = []
    age_mass_spearman: list[tuple[Any, float | None]] = []
    recent_vs_uniform: list[bool] = []
    anchor: list[bool] = []
    content: list[bool] = []
    fractions = rules.controlled_fractions("H5")
    for track, section in sections:
        profile = section.get("temporal_attention_profile", {})
        rho = finite_number(profile.get("spearman_age_vs_attention_mass"))
        adjacent_decay = finite_number(
            profile.get("adjacent_decay_fraction_newest_to_oldest")
        )
        chunk_tokens = finite_number(profile.get("chunk_tokens"))
        age_mass_spearman.append((track.get("layer"), rho))
        if isinstance(profile, dict) and profile:
            profile_summary.append(
                {
                    "layer": track.get("layer"),
                    "chunk_tokens": chunk_tokens,
                    "num_chunks": finite_number(profile.get("num_chunks")),
                    "spearman_age_vs_attention_mass": rho,
                    "adjacent_decay_fraction_newest_to_oldest": adjacent_decay,
                }
            )
        raw_profile_rows = profile.get("rows") if isinstance(profile, dict) else None
        if isinstance(raw_profile_rows, list):
            for profile_row in raw_profile_rows:
                if not isinstance(profile_row, dict):
                    continue
                profile_rows.append(
                    {
                        "layer": track.get("layer"),
                        "chunk_tokens": chunk_tokens,
                        "chunk_index_oldest_to_newest": finite_number(
                            profile_row.get("chunk_index_oldest_to_newest")
                        ),
                        "age_chunks": finite_number(profile_row.get("age_chunks")),
                        "attention_mass": finite_number(profile_row.get("attention_mass")),
                    }
                )
        for fraction in fractions:
            item = fraction_item(section.get("fractions"), fraction)
            if item is None:
                continue
            values: dict[str, float | None] = {}
            for variant in (
                "recent_only",
                "first_plus_recent",
                "uniform_temporal",
                "content_routing",
                "oracle_top_attention",
            ):
                rel = finite_number(get(item, variant, "output", "relative_l2"))
                values[variant] = rel
                rows.append(
                    {
                        "layer": track.get("layer"),
                        "variant": variant,
                        "selection_fraction": fraction,
                        "pv_relative_l2": rel,
                        "captured_attention_mass": finite_number(
                            get(item, variant, "captured_attention_mass")
                        ),
                    }
                )
            if None not in (values["recent_only"], values["uniform_temporal"]):
                recent_vs_uniform.append(values["recent_only"] < values["uniform_temporal"])
            if None not in (values["first_plus_recent"], values["recent_only"]):
                anchor.append(values["first_plus_recent"] < values["recent_only"])
            if None not in (values["content_routing"], values["first_plus_recent"]):
                content.append(values["content_routing"] < values["first_plus_recent"])

    def comparison(pid: str, checks: list[bool], text: str) -> dict[str, str]:
        if not checks:
            return observation(pid, "not_tested", f"缺少 matched-budget rows；{text}")
        if all(checks):
            return observation(pid, "supported", text)
        if not any(checks):
            return observation(pid, "contradicted", text)
        return observation(pid, "inconclusive", f"不同 layer/fraction 方向不一致；{text}")

    finite_rhos = [rho for _, rho in age_mass_spearman if rho is not None]
    rho_by_layer = {str(layer): rho for layer, rho in age_mass_spearman}
    profiles_complete = bool(age_mass_spearman) and len(finite_rhos) == len(age_mass_spearman)
    if profiles_complete and all(rho < 0 for rho in finite_rhos):
        p1 = observation(
            "H5-P1",
            "supported",
            f"各层 age-vs-attention-mass Spearman 均为负：{rho_by_layer}；仅按方向判定。",
        )
    elif profiles_complete and all(rho > 0 for rho in finite_rhos):
        p1 = observation(
            "H5-P1",
            "contradicted",
            f"各层 age-vs-attention-mass Spearman 均为正：{rho_by_layer}；仅按方向判定。",
        )
    elif finite_rhos:
        p1 = observation(
            "H5-P1",
            "inconclusive",
            (
                "age-vs-attention-mass Spearman 缺层、出现零值或 layer 间符号不一致："
                f"{rho_by_layer}；不设置或补造幅度阈值。"
            ),
        )
    else:
        p1 = observation(
            "H5-P1", "not_tested", "raw evidence 没有时间距离与 attention mass 的配对曲线。"
        )

    observations = [
        p1,
        comparison("H5-P2", recent_vs_uniform, "应用 H5-R1：recent PV relL2 < uniform。"),
        comparison("H5-P3", anchor, "应用 H5-R2：first+recent PV relL2 < recent-only。"),
        comparison("H5-P4", content, "应用 H5-R3：content routing PV relL2 < first+recent。"),
    ]
    verdict, explanation = aggregate_verdict(observations)
    layers = {int(track.get("layer")) for track, _ in sections}
    complete = {10, 20}.issubset(layers)
    return {
        "status": "complete" if complete else ("partial" if sections else "missing"),
        "tables": [
            {"title": "H5 temporal masks", "columns": list(rows[0]) if rows else [], "rows": rows},
            {
                "title": "H5 temporal attention profile summary",
                "columns": list(profile_summary[0]) if profile_summary else [],
                "rows": profile_summary,
            },
            {
                "title": "H5 temporal attention profile rows",
                "columns": list(profile_rows[0]) if profile_rows else [],
                "rows": profile_rows,
            },
        ],
        "observations": observations,
        "verdict": {"status": verdict if complete else "inconclusive", "explanation": explanation},
        "applied_rule_ids": ["H5-R1", "H5-R2", "H5-R3", "H5-R4"],
    }


def interpret_h6(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    sections = raw_sections(tracks, "H6")
    rows: list[dict[str, Any]] = []
    gaps: list[float] = []
    recovered: list[float] = []
    monotonic_rope: list[bool] = []
    acceleration_gates: list[bool] = []
    half_gaps: dict[str, list[float]] = {"standard_rope": [], "camera_prope": []}
    half_recovered: dict[str, list[float]] = {"standard_rope": [], "camera_prope": []}
    replay_rows: list[dict[str, Any]] = []
    replay_complete: list[bool] = []
    target_fraction = rules.selection_fraction("H6", "H6-R3")
    gap_threshold = rules.comparison("H6", "H6-R1", ">=")
    recovery_threshold = rules.percent("H6", "H6-R2")
    viable_error = rules.comparison("H6", "H6-R3", "<=")
    viable_speedup = rules.comparison("H6", "H6-R3", ">")
    for track, section in sections:
        layer_replays: list[bool] = []
        for half_name in ("standard_rope", "camera_prope"):
            half = section.get(half_name, {})
            replay = half.get("replay", {}) if isinstance(half, dict) else {}
            exact_replay = (
                isinstance(half, dict)
                and half.get("status") == "available"
                and replay.get("status") == "available"
                and replay.get("exact_replay") is True
            )
            layer_replays.append(exact_replay)
            replay_rows.append(
                {
                    "layer": track.get("layer"),
                    "rope_half": half_name,
                    "half_status": half.get("status") if isinstance(half, dict) else None,
                    "replay_status": replay.get("status") if isinstance(replay, dict) else None,
                    "exact_replay": replay.get("exact_replay") if isinstance(replay, dict) else None,
                    "query_replay_max_abs": finite_number(replay.get("query_replay_max_abs")) if isinstance(replay, dict) else None,
                    "key_replay_max_abs": finite_number(replay.get("key_replay_max_abs")) if isinstance(replay, dict) else None,
                    "value_replay_max_abs": finite_number(replay.get("value_replay_max_abs")) if isinstance(replay, dict) else None,
                }
            )
            if not isinstance(half, dict) or half.get("status") != "available":
                continue
            fraction_latencies: dict[str, list[tuple[float, float]]] = {}
            raw_fractions = sorted(
                value
                for key in get(half, "fractions", default={})
                if (value := finite_number(key)) is not None
            )
            dense = timing_ms(half.get("dense_bf16_position_attention_timing"))
            if dense is None:
                dense = timing_ms(half.get("dense_sdpa_timing"))
            full_decode = timing_ms(half.get("full_decode_position_attention_timing"))
            for fraction in raw_fractions:
                item = fraction_item(half.get("fractions"), fraction)
                if item is None:
                    continue
                for variant in (
                    "naive_pre_rope",
                    "position_aware_page_center",
                    "post_rope_upper_bound",
                ):
                    method = item.get(variant, {})
                    recall = finite_number(method.get("oracle_page_recall"))
                    rel = finite_number(get(method, "output", "relative_l2"))
                    total = timing_ms(method.get("route_selected_decode_position_attention_timing"))
                    if total is None:
                        total = timing_ms(method.get("route_selected_decode_rope_attention_timing"))
                    speedup = dense / total if None not in (dense, total) and total > 0 else None
                    position_ms = timing_ms(method.get("selected_only_position_transform_timing"))
                    if position_ms is None:
                        position_ms = timing_ms(method.get("selected_only_saved_rope_timing"))
                    rows.append(
                        {
                            "layer": track.get("layer"),
                            "rope_half": half_name,
                            "variant": variant,
                            "selection_fraction": fraction,
                            "attention_output_relative_l2": rel,
                            "top_page_recall": recall,
                            "complete_path_speedup_vs_bf16": speedup,
                            "position_transform_latency_ms": position_ms,
                            "dense_bf16_latency_ms": dense,
                            "full_decode_position_attention_latency_ms": full_decode,
                            "complete_selected_path_latency_ms": total,
                        }
                    )
                    if position_ms is not None:
                        fraction_latencies.setdefault(variant, []).append((fraction, position_ms))
                    if abs(fraction - target_fraction) < 1e-6 and variant == "position_aware_page_center" and None not in (rel, speedup):
                        acceleration_gates.append(rel <= viable_error and speedup > viable_speedup)
                if abs(fraction - target_fraction) < 1e-6:
                    naive = finite_number(get(item, "naive_pre_rope", "oracle_page_recall"))
                    pos = finite_number(get(item, "position_aware_page_center", "oracle_page_recall"))
                    post = finite_number(get(item, "post_rope_upper_bound", "oracle_page_recall"))
                    if None not in (naive, pos, post):
                        gap = post - naive
                        gaps.append(gap)
                        half_gaps[half_name].append(gap)
                        if gap > 0:
                            recovery = (pos - naive) / gap
                            recovered.append(recovery)
                            half_recovered[half_name].append(recovery)
            for values in fraction_latencies.values():
                if len(values) >= 2:
                    ordered = [value for _, value in sorted(values)]
                    monotonic_rope.append(
                        all(left <= right for left, right in zip(ordered, ordered[1:]))
                    )
        replay_complete.append(all(layer_replays))

    gap_median = median(gaps)
    recovery_median = median(recovered)
    gap_by_half = {name: median(values) for name, values in half_gaps.items()}
    recovery_by_half = {name: median(values) for name, values in half_recovered.items()}
    observations = [
        observation(
            "H6-P1",
            "supported" if gap_median is not None and gap_median >= gap_threshold else "inconclusive",
            f"post-naive recall gap median={gap_median!r}, by_half={gap_by_half}; H6-R1 threshold={gap_threshold}。",
        ) if gap_median is not None else observation("H6-P1", "not_tested", "缺少 20% naive/post recall。"),
        observation(
            "H6-P2",
            "supported" if recovery_median is not None and recovery_median >= recovery_threshold else "inconclusive",
            f"position-aware recovered gap median={recovery_median!r}, by_half={recovery_by_half}; H6-R2 threshold={recovery_threshold}。",
        ) if recovery_median is not None else observation("H6-P2", "not_tested", "post-naive gap非正或缺少 recall。"),
        observation(
            "H6-P3",
            "supported" if monotonic_rope and all(monotonic_rope) else ("inconclusive" if monotonic_rope else "not_tested"),
            "按 5%/10%/20% 检查 selected-only RoPE latency 是否随 selection fraction 单调增加。",
        ),
        observation(
            "H6-P4",
            "supported" if acceleration_gates and all(acceleration_gates) else ("contradicted" if acceleration_gates else "not_tested"),
            "按 H6-R3/R4：20% position-aware relL2<=0.15 且 full-path speedup>1；否则尚不可行。",
        ),
    ]
    verdict, explanation = aggregate_verdict(observations)
    complete = bool(sections) and len(replay_complete) == len(sections) and all(replay_complete)
    if not complete:
        explanation = (
            "H6 仅在每个 paired track 的 standard-RoPE 与 camera-PRoPE 都有 exact replay "
            "且对应路由表存在时才完整；当前 replay gate 未全部满足。"
        )
    return {
        "status": "complete" if complete else ("partial" if sections else "missing"),
        "tables": [
            {"title": "H6 paired pre/post positional routing", "columns": list(rows[0]) if rows else [], "rows": rows},
            {"title": "H6 positional replay gates", "columns": list(replay_rows[0]) if replay_rows else [], "rows": replay_rows},
        ],
        "observations": observations,
        "verdict": {"status": verdict if complete else "inconclusive", "explanation": explanation},
        "applied_rule_ids": ["H6-R1", "H6-R2", "H6-R3", "H6-R4"],
    }


def interpret_h7(tracks: Sequence[dict[str, Any]], rules: RuleBook) -> dict[str, Any]:
    sections = raw_sections(tracks, "H7")
    rows: list[dict[str, Any]] = []
    quality_gates: list[bool] = []
    runtime_gates: list[bool] = []
    matched_rates: list[bool] = []
    for track, section in sections:
        uniform_rel = finite_number(get(section, "post_rope_attention_output", "uniform_l6", "relative_l2"))
        tiered_rel = finite_number(get(section, "post_rope_attention_output", "tiered", "relative_l2"))
        uniform_ms = timing_ms(get(section, "decode_timing", "uniform_l6"))
        tiered_ms = timing_ms(get(section, "decode_timing", "tiered"))
        uniform_bytes = finite_number(
            get(section, "storage_per_tensor", "uniform_l6", "physical_uint8_index_bytes")
        )
        tiered_bytes = finite_number(
            get(section, "storage_per_tensor", "tiered", "physical_uint8_index_bytes")
        )
        for variant, rel, latency, payload in (
            ("uniform_l6", uniform_rel, uniform_ms, uniform_bytes),
            ("temporal_tiered", tiered_rel, tiered_ms, tiered_bytes),
        ):
            rows.append(
                {
                    "layer": track.get("layer"),
                    "variant": variant,
                    "pv_relative_l2": rel,
                    "payload_bytes": payload,
                    "decode_latency_ms": latency,
                }
            )
        if None not in (uniform_rel, tiered_rel):
            quality_gates.append(tiered_rel <= uniform_rel)
        if None not in (uniform_ms, tiered_ms) and uniform_ms > 0:
            runtime_gates.append(
                tiered_ms / uniform_ms <= rules.comparison("H7", "H7-R2", "<=")
            )
        if None not in (uniform_bytes, tiered_bytes):
            matched_rates.append(tiered_bytes == uniform_bytes)

    quality_available = bool(quality_gates) and bool(matched_rates)
    runtime_available = bool(runtime_gates)
    quality_pass = quality_available and all(quality_gates) and all(matched_rates)
    runtime_pass = runtime_available and all(runtime_gates)
    observations = [
        observation(
            "H7-P1",
            "supported" if quality_pass else ("contradicted" if quality_available else "not_tested"),
            "H7-R1：physical uint8 index bytes相等，且 tiered PV relL2<=uniform L6。",
        ),
        observation(
            "H7-P2",
            "inconclusive",
            "该预测明确允许 raw MSE 任意方向，不把 MSE 方向自由解释为支持或反驳。",
        ),
        observation(
            "H7-P3",
            "supported" if runtime_pass else ("contradicted" if runtime_available else "not_tested"),
            "H7-R2：tiered/uniform L6 decode latency ratio<=1.10；与 token progressive 的独立对照不由本 gate 代替。",
        ),
    ]
    both = quality_pass and runtime_pass
    if not sections:
        status = "missing"
        verdict = "inconclusive"
        explanation = "缺少 H7 offline evidence。"
    elif not quality_available or not runtime_available:
        status = "partial"
        verdict = "inconclusive"
        explanation = "H7 quality/rate 或 runtime gate 字段不完整；按 missing policy 不做负结果插补。"
    elif both:
        status = "complete"
        verdict = "supported"
        explanation = "H7-R1 与 H7-R2 同时通过，触发 H7-R3：可运行三 prompt video smoke。"
    else:
        status = "complete"
        verdict = "contradicted"
        explanation = "H7-R1 或 H7-R2 未通过，触发 H7-R4：在 negative offline result 后停止。"
    return {
        "status": status,
        "tables": [{"title": "H7 matched-rate offline gate", "columns": list(rows[0]) if rows else [], "rows": rows}],
        "observations": observations,
        "verdict": {"status": verdict, "explanation": explanation},
        "applied_rule_ids": ["H7-R1", "H7-R2", "H7-R3" if both else "H7-R4"],
        "video_smoke_authorized": both,
    }


INTERPRETERS: dict[str, Callable[[Sequence[dict[str, Any]], RuleBook], dict[str, Any]]] = {
    "H1": interpret_h1,
    "H2": interpret_h2,
    "H3": interpret_h3,
    "H4": interpret_h4,
    "H5": interpret_h5,
    "H6": interpret_h6,
    "H7": interpret_h7,
}


def canonical_scope(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": baseline["model_id"],
        "implementation": baseline["implementation"],
        "input_norm_mode": baseline["input_norm_mode"],
        "rvq_levels": baseline["rvq_levels"],
        "rvq_centroids": baseline["rvq_centroids"],
        "codebook_storage_dtype": baseline["codebook_storage_dtype"],
        "stores_token_scale": baseline["scale_storage"]["stored_in_no_norm_payload"],
    }


def create_output_dir(run_dir: Path, requested: Path | None) -> Path:
    if requested is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        requested = run_dir / "interpreted_canonical" / stamp
    if requested.exists():
        raise FileExistsError(f"Refusing to overwrite interpreted evidence: {requested}")
    requested.mkdir(parents=True, exist_ok=False)
    return requested


def create_h1_sweep_output_dir(
    run_dirs: Sequence[Path], requested: Path | None
) -> Path:
    if requested is None:
        common_parent = Path(os.path.commonpath([str(path.parent) for path in run_dirs]))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        requested = common_parent / "combined_h1_sweep" / stamp
    if requested.exists():
        raise FileExistsError(f"Refusing to overwrite interpreted evidence: {requested}")
    requested.mkdir(parents=True, exist_ok=False)
    return requested


def interpret_h1_sweep(
    run_dirs: Sequence[Path],
    hypotheses_path: Path,
    output_dir: Path | None = None,
) -> Path:
    """Pool immutable per-K H1 runs into one provenance-complete H1 record."""

    resolved_dirs = [path.resolve() for path in run_dirs]
    if len(resolved_dirs) < 2:
        raise ValueError("H1 sweep combination requires at least two run directories")
    if len(set(resolved_dirs)) != len(resolved_dirs):
        raise ValueError("H1 sweep combination received duplicate run directories")

    manifest = read_json(hypotheses_path)
    rules = RuleBook(manifest)
    pooled_tracks: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    observed_pairs: set[tuple[int, int]] = set()
    prompt_items: list[Any] = []

    for source_index, run_dir in enumerate(resolved_dirs):
        raw_path = run_dir / "evidence.json"
        run_info_path = run_dir / "run_info.json"
        if not raw_path.exists() or not run_info_path.exists():
            raise FileNotFoundError(
                f"H1 sweep source must contain evidence.json and run_info.json: {run_dir}"
            )
        raw = read_json(raw_path)
        run_info = read_json(run_info_path)
        if raw.get("baseline") != manifest["baseline"]["baseline_id"]:
            raise ValueError(f"H1 sweep source baseline mismatch: {raw_path}")
        if not isinstance(run_info, dict):
            raise ValueError(f"H1 sweep source run_info is not an object: {run_info_path}")
        missing_run_info = [key for key in RUN_INFO_REQUIRED if key not in run_info]
        if missing_run_info:
            raise ValueError(
                f"H1 sweep source run_info lacks {missing_run_info}: {run_info_path}"
            )
        tracks = raw.get("tracks")
        if not isinstance(tracks, list):
            raise ValueError(f"H1 sweep source tracks is not a list: {raw_path}")
        h1_tracks = [track for track in tracks if isinstance(track.get("H1"), dict)]
        if not h1_tracks:
            raise ValueError(f"H1 sweep source contains no H1 tracks: {raw_path}")
        for track in h1_tracks:
            key_shape = get(track, "shape", "key", default=[])
            if not isinstance(key_shape, list) or len(key_shape) <= 2:
                raise ValueError(f"H1 sweep track lacks key length: {raw_path}")
            layer = int(track["layer"])
            k_tokens = int(key_shape[2])
            pair = (layer, k_tokens)
            if pair in observed_pairs:
                raise ValueError(
                    f"Duplicate H1 layer/K row across sweep sources: layer={layer}, K={k_tokens}"
                )
            observed_pairs.add(pair)
            pooled_tracks.append(track)

        prompts = run_info.get("prompt_list")
        prompt_items.extend(prompts if isinstance(prompts, list) else [prompts])
        source_runs.append(
            {
                "source_index": source_index,
                "run_dir": str(run_dir),
                "evidence_path": str(raw_path),
                "evidence_sha256": file_hash(raw_path),
                "run_info_path": str(run_info_path),
                "run_info_sha256": file_hash(run_info_path),
                "git_commit": run_info["git_commit"],
                "config_path": run_info["config_path"],
                "config_hash": run_info["config_hash"],
                "codebook_path": run_info["codebook_path"],
                "codebook_hash": run_info["codebook_hash"],
                "hostname": run_info["hostname"],
                "gpu_type": run_info["gpu_type"],
                "seed": run_info["seed"],
                "prompt_list": run_info["prompt_list"],
                "timestamp": run_info["timestamp"],
                "h1_track_count": len(h1_tracks),
            }
        )

    interpreted = interpret_h1(pooled_tracks, rules)
    distinct_k_sizes = sorted({pair[1] for pair in observed_pairs})
    controlled_layers = {
        int(layer)
        for layer in get(
            rules.hypotheses,
            "H1",
            "controlled_experiment",
            "layers",
            default=[],
        )
    }
    layer_coverage = {
        str(k_tokens): sorted(layer for layer, key_length in observed_pairs if key_length == k_tokens)
        for k_tokens in distinct_k_sizes
    }
    incomplete_layer_coverage = {
        k_tokens: layers
        for k_tokens, layers in layer_coverage.items()
        if controlled_layers and set(layers) != controlled_layers
    }
    if incomplete_layer_coverage:
        interpreted["status"] = "partial"
        interpreted["verdict"] = {
            "status": "inconclusive",
            "explanation": (
                "H1 sweep does not cover every preregistered layer at each supplied K; "
                f"coverage={layer_coverage}."
            ),
        }

    generated_at = datetime.now(timezone.utc).isoformat()
    aggregate_run_info = {
        "schema_version": "wan5b_h1_sweep_aggregate_run_info.v1",
        "git_commit": collapse_json_values(item["git_commit"] for item in source_runs),
        "config_path": collapse_json_values(item["config_path"] for item in source_runs),
        "config_hash": collapse_json_values(item["config_hash"] for item in source_runs),
        "codebook_path": collapse_json_values(item["codebook_path"] for item in source_runs),
        "codebook_hash": collapse_json_values(item["codebook_hash"] for item in source_runs),
        "hostname": collapse_json_values(item["hostname"] for item in source_runs),
        "gpu_type": collapse_json_values(item["gpu_type"] for item in source_runs),
        "seed": collapse_json_values(item["seed"] for item in source_runs),
        "prompt_list": unique_json_values(prompt_items),
        "timestamp": generated_at,
        "aggregation_kind": "immutable_h1_k_sweep",
        "hypotheses_path": str(hypotheses_path.resolve()),
        "hypotheses_sha256": file_hash(hypotheses_path),
        "source_runs": source_runs,
        "pooled_h1_track_count": len(pooled_tracks),
        "distinct_k_sizes": distinct_k_sizes,
        "layer_coverage": layer_coverage,
    }

    output = create_h1_sweep_output_dir(resolved_dirs, output_dir)
    aggregate_run_info_path = output / "aggregate_run_info.json"
    write_json(aggregate_run_info_path, aggregate_run_info)
    h1_path = output / "h1.json"
    record = {
        "schema_version": SCHEMA,
        "hypothesis_id": "H1",
        "experiment_id": EXPERIMENT_IDS["H1"],
        "baseline_id": manifest["baseline"]["baseline_id"],
        "status": interpreted["status"],
        "scope": canonical_scope(manifest["baseline"]),
        "run_info_path": os.path.relpath(aggregate_run_info_path, output),
        "tables": interpreted["tables"],
        "observations": interpreted["observations"],
        "verdict": interpreted["verdict"],
        "artifacts": [
            *[item["evidence_path"] for item in source_runs],
            *[item["run_info_path"] for item in source_runs],
            str(hypotheses_path.resolve()),
        ],
        "interpretation": {
            "policy": "manifest_rules_only_no_imputation",
            "aggregation_kind": "immutable_h1_k_sweep",
            "source_evidence_sha256": [
                {
                    "path": item["evidence_path"],
                    "sha256": item["evidence_sha256"],
                }
                for item in source_runs
            ],
            "source_run_info_sha256": [
                {
                    "path": item["run_info_path"],
                    "sha256": item["run_info_sha256"],
                }
                for item in source_runs
            ],
            "hypotheses_sha256": file_hash(hypotheses_path),
            "applied_rule_ids": interpreted.get("applied_rule_ids", []),
            "source_run_count": len(source_runs),
            "pooled_h1_track_count": len(pooled_tracks),
            "distinct_k_sizes": distinct_k_sizes,
            "layer_coverage": layer_coverage,
            "timestamp": generated_at,
        },
    }
    write_json(h1_path, record)
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "aggregation_kind": "immutable_h1_k_sweep",
            "hypotheses_path": str(hypotheses_path.resolve()),
            "hypotheses_sha256": file_hash(hypotheses_path),
            "aggregate_run_info_path": str(aggregate_run_info_path),
            "aggregate_run_info_sha256": file_hash(aggregate_run_info_path),
            "source_runs": source_runs,
            "records": [str(h1_path)],
        },
    )
    return output


def interpret_run(run_dir: Path, hypotheses_path: Path, output_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    raw_path = run_dir / "evidence.json"
    run_info_path = run_dir / "run_info.json"
    if not raw_path.exists() or not run_info_path.exists():
        raise FileNotFoundError("run_dir must contain evidence.json and run_info.json")
    manifest = read_json(hypotheses_path)
    raw = read_json(raw_path)
    if raw.get("baseline") != manifest["baseline"]["baseline_id"]:
        raise ValueError("Raw evidence baseline does not match hypotheses manifest")
    tracks = raw.get("tracks")
    if not isinstance(tracks, list):
        raise ValueError("Raw evidence tracks must be a list")
    registered = {item["id"]: item for item in manifest.get("hypotheses", [])}
    if set(registered) != set(INTERPRETERS):
        raise ValueError("Interpreter and hypotheses manifest H1-H7 sets differ")
    rules = RuleBook(manifest)

    output = create_output_dir(run_dir, output_dir)
    records: list[Path] = []
    relative_run_info = os.path.relpath(run_info_path, output)
    for hypothesis_id, interpreter in INTERPRETERS.items():
        interpreted = interpreter(tracks, rules)
        record = {
            "schema_version": SCHEMA,
            "hypothesis_id": hypothesis_id,
            "experiment_id": EXPERIMENT_IDS[hypothesis_id],
            "baseline_id": manifest["baseline"]["baseline_id"],
            "status": interpreted["status"],
            "scope": canonical_scope(manifest["baseline"]),
            "run_info_path": relative_run_info,
            "tables": interpreted["tables"],
            "observations": interpreted["observations"],
            "verdict": interpreted["verdict"],
            "artifacts": [str(raw_path), str(hypotheses_path.resolve())],
            "interpretation": {
                "policy": "manifest_rules_only_no_imputation",
                "source_evidence_sha256": file_hash(raw_path),
                "hypotheses_sha256": file_hash(hypotheses_path),
                "applied_rule_ids": interpreted.get("applied_rule_ids", []),
                "video_smoke_authorized": interpreted.get("video_smoke_authorized"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        path = output / f"{hypothesis_id.lower()}.json"
        write_json(path, record)
        records.append(path)
    write_json(
        output / "manifest.json",
        {
            "schema_version": SCHEMA,
            "source_run_dir": str(run_dir),
            "raw_evidence_path": str(raw_path),
            "raw_evidence_sha256": file_hash(raw_path),
            "hypotheses_path": str(hypotheses_path.resolve()),
            "hypotheses_sha256": file_hash(hypotheses_path),
            "records": [str(path) for path in records],
        },
    )
    return output


def main() -> None:
    args = parse_args()
    if args.combine_h1_sweep:
        output = interpret_h1_sweep(args.run_dir, args.hypotheses, args.output_dir)
    else:
        if len(args.run_dir) != 1:
            raise SystemExit(
                "Multiple run directories require the explicit --combine-h1-sweep flag"
            )
        output = interpret_run(args.run_dir[0], args.hypotheses, args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
