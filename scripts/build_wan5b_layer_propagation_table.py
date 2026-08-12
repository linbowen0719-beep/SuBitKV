#!/usr/bin/env python3
"""Expand audited Wan5B propagation probes into the routing fitter schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "wan5b.layer_propagation_table.v1"
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
    "probe_epsilon",
    "probe_seed",
    "probe_path",
    "probe_hash",
    "coefficient_scope",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-roots",
        required=True,
        help="Comma-separated collection artifact roots.",
    )
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
    run_info_path = root / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    if (
        run_info.get("schema_version")
        != "wan5b.layer_propagation_collection.v1.run_info"
        or run_info.get("status") != "complete"
    ):
        raise ValueError(f"incomplete propagation collection: {root}")
    path = Path(run_info["probe_path"]).expanduser().resolve()
    observed = sha256_file(path)
    if observed != run_info.get("probe_hash"):
        raise RuntimeError(f"propagation probe hash mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "wan5b.layer_propagation_probe.v1":
        raise ValueError(f"unexpected probe schema: {path}")
    return value, path, observed


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
            raise ValueError("every formal probe must contain layers 0..29")
        for item in records:
            local = item["local_attention_output"]
            terminal = item["terminal_projected_noise_floor_subtracted"]
            if (
                float(item["control_fraction_mean"]) > 0.05
                or float(item["control_fraction_cvar99"]) > 0.05
            ):
                raise ValueError(
                    "zero-perturbation replay noise exceeds 5% of signal"
                )
            for tensor in ("K", "V"):
                for half in ("standard", "camera"):
                    for metric, statistic in (
                        ("output_mean", "mean"),
                        ("output_cvar99", "cvar99"),
                    ):
                        local_error = float(local[statistic])
                        terminal_error = float(terminal[statistic])
                        amplification = terminal_error / max(
                            local_error, 1e-30
                        )
                        for transition in ("base", "high", "low"):
                            rows.append(
                                {
                                    "split": "train",
                                    "prompt_id": prompt,
                                    "layer": int(item["layer"]),
                                    "tensor": tensor,
                                    "attention_half": half,
                                    "metric": metric,
                                    "transition": transition,
                                    "local_error": local_error,
                                    "terminal_error": terminal_error,
                                    "amplification": amplification,
                                    "probe_epsilon": float(
                                        probe[
                                            "epsilon_relative_to_hidden_rms"
                                        ]
                                    ),
                                    "probe_seed": int(probe["seed"]),
                                    "probe_path": str(path),
                                    "probe_hash": digest,
                                    "coefficient_scope": (
                                        "linear layer-only coefficient duplicated "
                                        "for base/high/low transitions"
                                    ),
                                }
                            )
    if len(prompts) != len(set(prompts)):
        raise ValueError("probe roots contain duplicate train prompt IDs")
    return rows, sources, sorted(prompts)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    rows, sources, prompts = build_rows(args.probe_roots)
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    args.output_root.mkdir(parents=True)
    table_path = args.output_root / "propagation.csv"
    with table_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "schema_version": SCHEMA,
        "probe_roots": [str(item) for item in args.probe_roots],
        "probe_sources": sources,
        "prompt_count": args.prompt_count,
        "prompt_ids": prompts,
        "layers": list(range(30)),
        "coefficient_scope": "layer_only",
        "expansion": {
            "output_mean": "terminal projected mean / local perturbation mean",
            "output_cvar99": (
                "terminal projected CVaR99 / local perturbation CVaR99"
            ),
            "K_V": "shared downstream coefficient",
            "standard_camera": "shared downstream coefficient",
        },
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
    run_info = {
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
    }
    write_json(args.output_root / "run_info.json", run_info)
    write_json(
        args.output_root / "evidence_manifest.json",
        {
            "schema_version": f"{SCHEMA}.evidence",
            "files": {
                "propagation.csv": sha256_file(table_path),
                "config.json": sha256_file(config_path),
                "run_info.json": sha256_file(args.output_root / "run_info.json"),
            },
            "sources": sources,
        },
    )


if __name__ == "__main__":
    main()
