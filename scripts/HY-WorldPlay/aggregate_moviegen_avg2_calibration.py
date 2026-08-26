#!/usr/bin/env python3
"""Aggregate a selected HY training prefix into K/V avg2 distortion profiles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quant_videogen.hyworldplay8b_cache_codec import AVG2_RESIDUAL_BITS  # noqa: E402
from quant_videogen.hyworldplay_attention_calibration import CALIBRATION_SCHEMA as EVENT_SCHEMA  # noqa: E402
from build_moviegen_avg2_profile import CALIBRATION_SCHEMA  # noqa: E402


DEFAULT_PROMPTS = 20
LAYERS = 54
HISTORY_STEPS = 9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_create(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def monotone(values: dict[int, float]) -> dict[int, float]:
    result: dict[int, float] = {}
    lower_bound = 0.0
    for bits in AVG2_RESIDUAL_BITS:
        lower_bound = max(lower_bound, float(values[bits]))
        result[bits] = lower_bound
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--codebook-root", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=DEFAULT_PROMPTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    all_events: list[dict[str, Any]] = []
    sources = []
    if not 1 <= int(args.prompts) <= 20:
        raise ValueError("--prompts must be in [1,20]")
    for prompt in range(int(args.prompts)):
        path = args.runs_dir / f"prompt_{prompt:02d}" / "generation_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        events = metrics.get("kv_avg2_calibration_events", [])
        cells = {(int(row["layer"]), int(row["history_spans"])) for row in events}
        expected = {(layer, spans) for layer in range(LAYERS) for spans in range(1, HISTORY_STEPS + 1)}
        if cells != expected or len(events) != len(expected):
            raise ValueError(
                f"prompt {prompt} calibration covers {len(cells)} cells, expected {len(expected)}"
            )
        if any(row.get("schema_version") != EVENT_SCHEMA for row in events):
            raise ValueError(f"prompt {prompt} contains a non-v2 calibration event")
        all_events.extend(events)
        sources.append({"prompt": prompt, "path": str(path.resolve()), "sha256": sha256_file(path)})
    tensor_profiles: dict[str, Any] = {}
    for name in ("K", "V"):
        raw_distortions = {
            bits: sum(float(row["distortion"][name][str(bits)]) for row in all_events) / len(all_events)
            for bits in AVG2_RESIDUAL_BITS
        }
        age_sums = [0.0] * 10
        age_counts = [0] * 10
        for row in all_events:
            for age, mass in enumerate(row["age_attention_mass"]):
                age_sums[age] += float(mass)
                age_counts[age] += 1
        observed_age = [
            age_sums[age] / age_counts[age] if age_counts[age] else 0.0
            for age in range(9)
        ]
        # The terminal audit introduces age nine without denoising.  Use the
        # oldest observed train-only marginal, never a test-derived value.
        observed_age.append(observed_age[-1])
        age_weights = [max(value, 1e-9) for value in observed_age]
        raw_components: dict[str, dict[str, float]] = {}
        component_names = (
            ("attention_kl", "normalized_output_mse")
            if name == "K"
            else ("normalized_output_mse",)
        )
        for component in component_names:
            raw_components[component] = {
                str(bits): sum(
                    float(row["raw_metrics"][name][str(bits)][component])
                    for row in all_events
                )
                / len(all_events)
                for bits in AVG2_RESIDUAL_BITS
            }
        fitted = monotone(raw_distortions)
        tensor_profiles[name] = {
            "distortions": {str(bits): fitted[bits] for bits in AVG2_RESIDUAL_BITS},
            "raw_distortions": {str(bits): raw_distortions[bits] for bits in AVG2_RESIDUAL_BITS},
            "raw_components": raw_components,
            "age_weights": age_weights,
            "age_weights_raw": observed_age,
            "distortion_objective": (
                "attention_KL_plus_normalized_output_MSE"
                if name == "K"
                else "normalized_output_MSE"
            ),
            "monotonicization": "cumulative maximum from B8 through B1",
        }
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": CALIBRATION_SCHEMA,
        "tensor_profiles": tensor_profiles,
        "training": {
            "split": "moviegen_train20" if int(args.prompts) == 20 else f"moviegen_train{int(args.prompts)}_fast_validation",
            "prompts": int(args.prompts),
            "layers": LAYERS,
            "history_steps": HISTORY_STEPS,
            "events": len(all_events),
            "test_prompts_read": False,
            "query_streams": ["rope", "prope"],
            "includes_real_text_kv": True,
        },
        "sources": sources,
    }
    calibration = output / "attention_calibration.json"
    write_create(calibration, payload)
    config = {
        "schema_version": "hyworldplay_rvq_attention_calibration.aggregate_config.v2",
        "runs_dir": str(args.runs_dir.resolve()),
        "codebook_root": str(args.codebook_root.resolve()),
        "prompts": int(args.prompts),
    }
    write_create(output / "config.json", config)
    write_create(
        output / "run_info.json",
        {
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_state": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain=v1"], text=True
            ).splitlines(),
            "config_path": str((output / "config.json").resolve()),
            "config_hash": sha256_file(output / "config.json"),
            "codebook_path": str(args.codebook_root.resolve()),
            "codebook_hash": sha256_file(args.codebook_root / "manifest.json"),
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": 42,
            "prompt_list": [
                line.strip()
                for line in (REPO_ROOT / "assets/moviegen_train20.txt")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ][: int(args.prompts)],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibration_hash": sha256_file(calibration),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
