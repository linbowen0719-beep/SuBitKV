#!/usr/bin/env python3
"""Build nonlinear D3/D4/D5 layer-transition slopes from formal probes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "wan5b.layer_transition_table.v1"
FIELDS = (
    "split",
    "prompt_id",
    "layer",
    "tensor",
    "attention_half",
    "metric",
    "transition",
    "local_error",
    "terminal_error",
    "amplification",
    "raw_terminal_delta",
    "terminal_delta_clamped",
    "probe_path",
    "probe_hash",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-roots", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args(argv)
    args.probe_roots = tuple(
        Path(item).expanduser().resolve()
        for item in args.probe_roots.split(",")
        if item
    )
    args.output_root = args.output_root.expanduser().resolve()
    if len(args.probe_roots) != args.prompt_count:
        parser.error("probe root count must equal prompt-count")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_probe(root: Path) -> tuple[Mapping[str, Any], Path, str]:
    info = json.loads((root / "run_info.json").read_text(encoding="utf-8"))
    if (
        info.get("schema_version")
        != "wan5b.layer_propagation_collection.v1.run_info"
        or info.get("status") != "complete"
    ):
        raise ValueError(f"incomplete propagation probe: {root}")
    path = Path(info["probe_path"]).resolve()
    digest = sha256_file(path)
    if digest != info["probe_hash"]:
        raise RuntimeError(f"probe hash mismatch: {path}")
    probe = json.loads(path.read_text(encoding="utf-8"))
    if probe.get("probe_mode") != "codec_D3_D4_D5_operating_points":
        raise ValueError("transition table requires operating-point probes")
    return probe, path, digest


def _point_index(
    record: Mapping[str, Any],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    points = {
        (str(point["tensor"]), int(point["depth"])): point
        for point in record.get("operating_points", [])
    }
    expected = {
        (tensor, depth)
        for tensor in ("K", "V")
        for depth in (3, 4, 5)
    }
    if set(points) != expected:
        raise ValueError("operating-point layer record is incomplete")
    return points


def build_rows(
    roots: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, str], list[int]]:
    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    prompts: list[int] = []
    for root in roots:
        probe, path, digest = load_probe(root)
        prompt = int(probe["prompt_id"])
        prompts.append(prompt)
        sources[str(path)] = digest
        records = probe.get("records")
        if (
            not isinstance(records, list)
            or [int(item["layer"]) for item in records] != list(range(30))
        ):
            raise ValueError("formal transition probe must contain layers 0..29")
        for record in records:
            layer = int(record["layer"])
            points = _point_index(record)
            for tensor in ("K", "V"):
                for metric, statistic, local_source in (
                    ("output_mean", "mean", "realized"),
                    ("output_cvar99", "cvar99", "target"),
                ):
                    local: dict[int, float] = {}
                    terminal: dict[int, float] = {}
                    for depth in (3, 4, 5):
                        point = points[(tensor, depth)]
                        local[depth] = (
                            float(
                                point["realized_local_attention_output"][
                                    statistic
                                ]
                            )
                            if local_source == "realized"
                            else float(point["target_local_cvar99"])
                        )
                        terminal[depth] = float(
                            point[
                                "terminal_projected_noise_floor_subtracted"
                            ][statistic]
                        )
                    transitions = {
                        "base": (local[4], terminal[4], terminal[4]),
                        "high": (
                            local[4] - local[5],
                            terminal[4] - terminal[5],
                            terminal[4],
                        ),
                        "low": (
                            local[3] - local[4],
                            terminal[3] - terminal[4],
                            terminal[4],
                        ),
                    }
                    for transition, (
                        local_delta,
                        raw_terminal_delta,
                        terminal_scale,
                    ) in transitions.items():
                        if not math.isfinite(local_delta) or local_delta <= 0:
                            raise ValueError(
                                "local D3/D4/D5 errors are not monotone"
                            )
                        floor = max(abs(terminal_scale) * 1e-6, 1e-30)
                        effective_terminal = max(raw_terminal_delta, floor)
                        for half in ("standard", "camera"):
                            rows.append(
                                {
                                    "split": "train",
                                    "prompt_id": prompt,
                                    "layer": layer,
                                    "tensor": tensor,
                                    "attention_half": half,
                                    "metric": metric,
                                    "transition": transition,
                                    "local_error": local_delta,
                                    "terminal_error": effective_terminal,
                                    "amplification": (
                                        effective_terminal / local_delta
                                    ),
                                    "raw_terminal_delta": raw_terminal_delta,
                                    "terminal_delta_clamped": bool(
                                        raw_terminal_delta < floor
                                    ),
                                    "probe_path": str(path),
                                    "probe_hash": digest,
                                }
                            )
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate propagation prompt IDs")
    return rows, sources, sorted(prompts)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows, sources, prompts = build_rows(args.probe_roots)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    table = args.output_root / "propagation.csv"
    with table.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "schema_version": SCHEMA,
        "probe_roots": [str(item) for item in args.probe_roots],
        "prompt_ids": prompts,
        "prompt_count": args.prompt_count,
        "layers": list(range(30)),
        "transitions": ["base", "high", "low"],
        "high": "(T4-T5)/(L4-L5)",
        "low": "(T3-T4)/(L3-L4)",
        "base": "T4/L4",
        "negative_or_zero_terminal_delta": (
            "floored to 1e-6 of the prompt-layer B4 terminal error; raw value "
            "and clamp flag are retained"
        ),
        "output_mean_local_denominator": "realized Rademacher local MSE",
        "output_cvar99_local_denominator": (
            "train exact-attention target CVaR99; perturbation amplitude remains "
            "calibrated by target mean"
        ),
        "seed": args.seed,
    }
    config_path = args.output_root / "config.json"
    write_json(config_path, config)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip() or "unknown"
    write_json(
        args.output_root / "run_info.json",
        {
            "schema_version": f"{SCHEMA}.run_info",
            "git_commit": commit,
            "config_path": str(config_path),
            "config_hash": sha256_file(config_path),
            "codebook_path": None,
            "codebook_hash": None,
            "hostname": socket.gethostname(),
            "gpu_type": None,
            "seed": args.seed,
            "prompt_list": prompts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
    )
    write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "files": {
                "propagation.csv": sha256_file(table),
                "config.json": sha256_file(config_path),
                "run_info.json": sha256_file(args.output_root / "run_info.json"),
            },
            "sources": sources,
        },
    )


if __name__ == "__main__":
    main()
