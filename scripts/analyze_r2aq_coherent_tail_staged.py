#!/usr/bin/env python3
"""Create-only paired analysis for the prefix-safe coherent-tail screen."""

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
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parent.parent
DEFAULT_DATA = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test"
)
DEFAULT_STAGED = (
    DEFAULT_DATA
    / "r2aq_coherent_tail_g128_f100_staged_exact_3p4l_20260724_b4diag_v1"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "analysis"
    / "r2aq_coherent_tail_g128_staged_20260724_b4diag_v1"
)
TAIL_METRICS = (
    "attention_output_squared_error_mean",
    "attention_output_squared_error_P99",
    "attention_output_squared_error_P999",
    "attention_output_squared_error_CVaR99",
)
LOWER_METRICS = (
    *TAIL_METRICS,
    "attention_logit_mse",
    "attention_kl_reference_to_candidate",
    "historical_K_mse",
    "historical_V_mse",
)
METHOD = "r2aq_coherent_tail_g128"
REFERENCE = "rvq_plus_quantvideo"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--staged-root", type=Path, default=DEFAULT_STAGED)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260724)
    args = parser.parse_args(argv)
    args.staged_root = args.staged_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        parser.error(f"create-only output exists: {args.output_root}")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    for relative in (
        "config.json",
        "tables/exact_attention_metrics.csv",
        "tables/prompt_macro_metrics.csv",
    ):
        if not (args.staged_root / relative).is_file():
            parser.error(f"missing staged input: {relative}")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = sorted({key for row in rows for key in row})
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def paired_bootstrap(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("paired bootstrap arrays must be aligned vectors")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(candidate), size=(samples, len(candidate)))
    c = candidate[indices].mean(1)
    r = reference[indices].mean(1)
    improvement = 1.0 - c / r
    return (
        float(np.mean(improvement)),
        float(np.quantile(improvement, 0.025)),
        float(np.quantile(improvement, 0.975)),
    )


def prompt_bootstrap_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    lookup = {
        (
            int(row["prompt_id"]),
            row["method"],
            int(row["bits"]),
            row["attention_half"],
            row["intervention"],
        ): row
        for row in rows
    }
    prompts = sorted({int(row["prompt_id"]) for row in rows})
    result: list[dict[str, Any]] = []
    for half in ("standard_rope", "camera_prope"):
        for intervention in ("K_only", "V_only", "K_and_V"):
            for metric_index, metric in enumerate(LOWER_METRICS):
                candidate = np.asarray(
                    [
                        float(
                            lookup[
                                (prompt, METHOD, 4, half, intervention)
                            ][metric]
                        )
                        for prompt in prompts
                    ]
                )
                reference = np.asarray(
                    [
                        float(
                            lookup[
                                (prompt, REFERENCE, 4, half, intervention)
                            ][metric]
                        )
                        for prompt in prompts
                    ]
                )
                if not bool(np.any(reference)):
                    continue
                _boot_mean, low, high = paired_bootstrap(
                    candidate,
                    reference,
                    samples=samples,
                    seed=seed + metric_index,
                )
                ratio = float(candidate.mean() / reference.mean())
                result.append(
                    {
                        "attention_half": half,
                        "intervention": intervention,
                        "metric": metric,
                        "n_prompts": len(prompts),
                        "candidate_mean": float(candidate.mean()),
                        "qv_mean": float(reference.mean()),
                        "candidate_over_qv": ratio,
                        "improvement_fraction": 1.0 - ratio,
                        "paired_bootstrap_improvement_ci95_low": low,
                        "paired_bootstrap_improvement_ci95_high": high,
                        "prompt_win_rate": float(
                            np.mean(candidate < reference)
                        ),
                        "prompt_ids": ",".join(map(str, prompts)),
                        "per_prompt_candidate_over_qv": ",".join(
                            f"{value:.9g}"
                            for value in candidate / reference
                        ),
                    }
                )
    return result


def unit_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int, int, str, int, str, str], list[Mapping[str, str]]
    ] = defaultdict(list)
    for row in rows:
        key = (
            int(row["prompt_id"]),
            int(row["layer"]),
            row["method"],
            int(row["bits"]),
            row["attention_half"],
            row["intervention"],
        )
        grouped[key].append(row)
    macro: dict[tuple[int, int, str, int, str, str], dict[str, float]] = {}
    for key, values in grouped.items():
        macro[key] = {
            metric: float(np.mean([float(row[metric]) for row in values]))
            for metric in TAIL_METRICS
        }
    result: list[dict[str, Any]] = []
    units = sorted({(key[0], key[1]) for key in grouped})
    for prompt, layer in units:
        for half in ("standard_rope", "camera_prope"):
            for intervention in ("K_only", "V_only", "K_and_V"):
                candidate = macro[
                    (prompt, layer, METHOD, 4, half, intervention)
                ]
                reference = macro[
                    (prompt, layer, REFERENCE, 4, half, intervention)
                ]
                row: dict[str, Any] = {
                    "prompt_id": prompt,
                    "layer": layer,
                    "attention_half": half,
                    "intervention": intervention,
                }
                for metric in TAIL_METRICS:
                    row[f"{metric}_candidate"] = candidate[metric]
                    row[f"{metric}_qv"] = reference[metric]
                    row[f"{metric}_candidate_over_qv"] = (
                        candidate[metric] / reference[metric]
                    )
                result.append(row)
    return result


def payload_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    unique: dict[tuple[int, int, int], Mapping[str, str]] = {}
    for row in rows:
        if row["method"] != METHOD or int(row["bits"]) not in (2, 4):
            continue
        key = (
            int(row["prompt_id"]),
            int(row["layer"]),
            int(row["bits"]),
        )
        unique.setdefault(key, row)
    result: list[dict[str, Any]] = []
    for (prompt, layer, bits), row in sorted(unique.items()):
        shared_k = int(float(row["r2aq_shared_payload_K_bytes"]))
        shared_v = int(float(row["r2aq_shared_payload_V_bytes"]))
        delta_k = int(float(row["residual_payload_delta_K_bytes_vs_QV"]))
        delta_v = int(float(row["residual_payload_delta_V_bytes_vs_QV"]))
        item: dict[str, Any] = {
            "prompt_id": prompt,
            "layer": layer,
            "bits": bits,
            "strict_rate_match_vs_QV": row["strict_rate_match_vs_QV"],
            "delta_K_bytes_including_shared": delta_k,
            "delta_V_bytes_including_shared": delta_v,
            "shared_K_bytes": shared_k,
            "shared_V_bytes": shared_v,
            "local_delta_K_bytes_vs_QV": delta_k - shared_k,
            "local_delta_V_bytes_vs_QV": delta_v - shared_v,
        }
        if bits == 4:
            spans = int(row["physical_quant_span_count"])
            encoded = int(row["encoded_history_tokens"])
            total_groups = encoded * 24 * 128 // 64
            selected = round(
                total_groups * float(row["selected_fraction_V"])
            )
            tail_count = (
                int(row["r2aq_tail_id_payload_bytes_V"]) // 2
            )
            item.update(
                {
                    "physical_span_count": spans,
                    "selected_P5_groups_before_tail": selected,
                    "tail_G128_group_count": tail_count,
                    "terminal_P5_group_count": (
                        selected - tail_count * 67 // 8
                    ),
                    "tail_value_bytes": int(
                        row["r2aq_tail_value_payload_bytes_V"]
                    ),
                    "tail_scale_bytes": int(
                        row["r2aq_tail_scale_payload_bytes_V"]
                    ),
                    "tail_id_bytes": int(
                        row["r2aq_tail_id_payload_bytes_V"]
                    ),
                }
            )
        result.append(item)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raw_path = args.staged_root / "tables" / "exact_attention_metrics.csv"
    macro_path = args.staged_root / "tables" / "prompt_macro_metrics.csv"
    config_path = args.staged_root / "config.json"
    raw = read_csv(raw_path)
    macros = read_csv(macro_path)
    paired = prompt_bootstrap_rows(
        macros, samples=args.bootstrap_samples, seed=args.seed
    )
    units = unit_rows(raw)
    payload = payload_rows(raw)
    target = {
        (
            row["attention_half"],
            row["intervention"],
            row["metric"],
        ): row
        for row in paired
    }
    focus = {
        f"{half}.{intervention}.{metric}": target[
            (half, intervention, metric)
        ]
        for half in ("standard_rope", "camera_prope")
        for intervention in ("K_only", "V_only", "K_and_V")
        for metric in (
            "attention_output_squared_error_mean",
            "attention_output_squared_error_CVaR99",
        )
    }
    summary = {
        "schema_version": "r2aq_coherent_tail_staged.summary.v1",
        "method": METHOD,
        "reference": REFERENCE,
        "statistical_unit": "prompt",
        "n_prompts": 3,
        "layers": [0, 10, 20, 29],
        "heads_per_layer": 24,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "descriptive_only_small_n": True,
        "local_payload_exact_all_B2_B4": all(
            row["local_delta_K_bytes_vs_QV"] == 0
            and row["local_delta_V_bytes_vs_QV"] == 0
            for row in payload
        ),
        "focus": focus,
    }
    config = {
        "schema_version": "r2aq_coherent_tail_staged.analysis.config.v1",
        "staged_root": str(args.staged_root),
        "staged_config_hash": sha256_file(config_path),
        "raw_metrics_hash": sha256_file(raw_path),
        "prompt_macro_hash": sha256_file(macro_path),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    config_payload = json_bytes(config)
    config_hash = hashlib.sha256(config_payload).hexdigest()
    outputs = {
        "config.json": config_payload,
        "paired_prompt_bootstrap.csv": csv_bytes(paired),
        "paired_unit_metrics.csv": csv_bytes(units),
        "payload_audit.csv": csv_bytes(payload),
        "summary.json": json_bytes(summary),
    }
    args.output_root.mkdir(parents=True)
    for name, value in outputs.items():
        create_only(args.output_root / name, value)
    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    run_info = {
        "schema_version": "r2aq_coherent_tail_staged.analysis.run_info.v1",
        "git_commit": git_commit(),
        "config_path": str(args.output_root / "config.json"),
        "config_hash": config_hash,
        "codebook_path": source_config["codebook_path"],
        "codebook_hash": source_config["codebook_hash"],
        "hostname": socket.gethostname(),
        "gpu_type": None,
        "seed": args.seed,
        "prompt_list": [
            source_config["captures"][index]["prompt"]
            for index in (0, 4, 8)
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_hashes": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in outputs.items()
        },
        "source_hashes": {
            str(SCRIPT): sha256_file(SCRIPT),
            str(raw_path): sha256_file(raw_path),
            str(macro_path): sha256_file(macro_path),
        },
    }
    create_only(args.output_root / "run_info.json", json_bytes(run_info))
    print(
        f"[done] paired={len(paired)} units={len(units)} "
        f"payload={len(payload)} output={args.output_root}"
    )


if __name__ == "__main__":
    main()
