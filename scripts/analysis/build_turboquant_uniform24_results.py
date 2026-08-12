#!/usr/bin/env python3
"""Aggregate strict TurboQuant B2/B4 cross-model reruns (create-only)."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
METHODS = (
    "turboquant_mse_k2_v2",
    "turboquant_mse_k4_v4",
    "turboquant_prod_k2_v2",
    "turboquant_prod_k4_v4",
)
LABELS = {
    "turboquant_mse_k2_v2": "TurboQuant KV-MSE K2/V2",
    "turboquant_mse_k4_v4": "TurboQuant KV-MSE K4/V4",
    "turboquant_prod_k2_v2": "TurboQuant K-prod/V-MSE K2/V2",
    "turboquant_prod_k4_v4": "TurboQuant K-prod/V-MSE K4/V4",
}
DIMS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality",
    "imaging_quality", "human_action", "temporal_style",
    "overall_consistency",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def vbench(root: Path, method: str) -> dict[str, float]:
    scores = {}
    for dimension in DIMS:
        paths = sorted((root / method / dimension).glob("*_eval_results.json"))
        if not paths:
            raise FileNotFoundError(f"missing VBench result: {method}/{dimension}")
        payload = json.loads(paths[-1].read_text(encoding="utf-8"))
        value = payload[dimension]
        scores[dimension] = float(value[0] if isinstance(value, list) else value)
    return scores


def wan_rows(root: Path) -> tuple[list[dict], list[dict]]:
    summary = pd.read_csv(root / "summary.csv").set_index("method")
    aggregate, per_prompt = [], []
    for method in METHODS:
        row = summary.loc[method]
        scores = {dim: float(row[f"vbench_{dim}"]) for dim in DIMS}
        aggregate.append({
            "model": "Wan2.2 5B", "method": method,
            "method_label": LABELS[method], "nominal_bits": 2 if "k2_v2" in method else 4,
            "psnr": float(row.postcompression_psnr),
            "ssim": float(row.postcompression_ssim),
            "lpips": float(row.postcompression_lpips),
            "vbench10": float(row.vbench10_mean),
            "terminal_kv_mib": float(row.kv_cache_mem_mb) / 2.0,
            "codec_only_ratio": float(row.compression_ratio_vs_bf16) / 2.0,
            "encode_s": float(row.quant_encode_mean_s),
            "decode_s": float(row.dequant_decode_mean_s),
            "e2e_s": float(row.e2e_mean_s),
            "peak_gpu_mib": float(row.peak_gpu_mem_mb),
            "shared_fixed_mib": 0.125 if "_prod_" in method else 0.0625,
            **{f"vbench_{key}": value for key, value in scores.items()},
        })
        metrics = list(csv.DictReader(
            (root / "videos" / method / "metrics.csv").open(encoding="utf-8")
        ))
        quality = json.loads((root / "similarity_postcompression" / f"{method}.json").read_text())
        by_prompt = {int(item["prompt_id"]): item for item in quality["videos"]}
        for item in metrics:
            prompt = int(item["prompt_id"])
            q = by_prompt[prompt]
            per_prompt.append({
                "model": "Wan2.2 5B", "method": method, "prompt": prompt,
                "psnr": q["psnr"], "ssim": q["ssim"], "lpips": q["lpips"],
                "terminal_kv_mib": float(item["kv_cache_mem_mb"]) / 2.0,
                "codec_only_ratio": float(item["compression_ratio"]) / 2.0,
                "encode_s": item["quant_time_s"], "decode_s": item["dequant_time_s"],
                "e2e_s": item["e2e_time_s"],
            })
    return aggregate, per_prompt


def self_rows(run_root: Path, vbench_root: Path) -> tuple[list[dict], list[dict]]:
    aggregate, per_prompt = [], []
    for method in METHODS:
        scores = vbench(vbench_root, method)
        method_rows = []
        for prompt in range(10):
            run = run_root / method / f"prompt_{prompt:04d}_seed_0"
            metrics = json.loads((run / "metrics.json").read_text())
            quality = json.loads((run / "quality_cross_model.json").read_text())
            item = {
                "model": "Self-Forcing", "method": method, "prompt": prompt,
                "psnr": quality["psnr"], "ssim": quality["ssim"],
                "lpips": quality["lpips"],
                "terminal_kv_mib": metrics["kv_payload_mib"],
                "codec_only_ratio": metrics["compression_ratio_vs_bf16"],
                "encode_s": metrics["quant_encode_seconds"],
                "decode_s": metrics["quant_decode_seconds"],
                "e2e_s": metrics["e2e_seconds"],
            }
            method_rows.append(item)
            per_prompt.append(item)
        numeric = pd.DataFrame(method_rows).mean(numeric_only=True).to_dict()
        aggregate.append({
            "model": "Self-Forcing", "method": method,
            "method_label": LABELS[method], "nominal_bits": 2 if "k2_v2" in method else 4,
            "psnr": numeric["psnr"], "ssim": numeric["ssim"],
            "lpips": numeric["lpips"], "vbench10": float(np.mean(list(scores.values()))),
            "terminal_kv_mib": numeric["terminal_kv_mib"],
            "codec_only_ratio": numeric["codec_only_ratio"],
            "encode_s": numeric["encode_s"], "decode_s": numeric["decode_s"],
            "e2e_s": numeric["e2e_s"],
            "peak_gpu_mib": float(max(
                json.loads((run_root / method / f"prompt_{p:04d}_seed_0/metrics.json").read_text())["peak_allocated_mib"]
                for p in range(10)
            )),
            "shared_fixed_mib": 0.125 if "_prod_" in method else 0.0625,
            **{f"vbench_{key}": value for key, value in scores.items()},
        })
    return aggregate, per_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "aggregate"
    output.mkdir(parents=True, exist_ok=False)
    wan, wan_per = wan_rows(root / "wan2_2_5b")
    self_rows_main, self_per = self_rows(
        root / "self_forcing/formal10_v4", root / "self_forcing/vbench10"
    )
    main_rows = wan + self_rows_main
    write_csv(output / "closed_loop_main.csv", main_rows)
    write_csv(output / "per_prompt.csv", wan_per + self_per)
    exact = pd.concat([
        pd.read_csv(root / "wan2_2_5b/exact/exact_kv_attention.csv"),
        pd.read_csv(root / "self_forcing/formal10_v4/exact_kv_attention.csv"),
    ], ignore_index=True)
    exact.to_csv(output / "exact_kv_attention.csv", index=False)
    inputs = sorted(path for path in root.rglob("*") if path.is_file() and output not in path.parents)
    manifest = {str(path): sha256(path) for path in inputs}
    (output / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    (output / "run_info.json").write_text(json.dumps({
        "git_commit": commit, "config_path": str(root / "wan2_2_5b/config.json"),
        "config_hash": sha256(root / "wan2_2_5b/config.json"),
        "codebook_path": None, "codebook_hash": None,
        "hostname": socket.gethostname(), "gpu_type": "RTX 6000 Ada",
        "seed": {"Wan2.2 5B": 42, "Self-Forcing": 0},
        "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "methods": list(METHODS),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
