#!/usr/bin/env python3
"""Validate full-protocol outputs and create the paper main-table CSV/LaTeX."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
from statistics import fmean
from typing import Any

from paper_main_common import git_commit, load_protocol, sha256_file


METHOD_META = {
    "bf16": ("BF16", 16),
    "quantvideo_prerope_int4": ("QuantVideo", 4),
    "kivi_all_int4": ("KIVI", 4),
    "turboquant_prod_k4_v4": ("TurboQuant", 4),
    "adaptive_temporal_int8_avg4": ("Ours", 4),
    "adaptive_temporal_avg4": ("Ours", 4),
    "adaptive_temporal_int8_avg3": ("Ours", 3),
    "adaptive_temporal_avg3": ("Ours", 3),
    "quantvideo_prerope_int2": ("QuantVideo", 2),
    "kivi_all_int2": ("KIVI", 2),
    "turboquant_prod_k2_v2": ("TurboQuant", 2),
    "adaptive_temporal_int8_avg2": ("Ours", 2),
    "adaptive_temporal_avg2": ("Ours", 2),
    "adaptive_temporal_int8_avg1": ("Ours", 1),
    "adaptive_temporal_avg1": ("Ours", 1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path)
    parser.add_argument("--self-root", type=Path)
    parser.add_argument("--self-vbench-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def vbench_score(root: Path, method: str, dimension: str) -> float:
    files = sorted((root / method / dimension).glob("*_eval_results.json"))
    if len(files) != 1:
        raise RuntimeError(
            f"expected exactly one VBench result for {method}/{dimension}, found {len(files)}"
        )
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    value = payload[dimension]
    return float(value[0] if isinstance(value, list) else value)


def table_ratio(method: str, raw: float) -> float:
    # Wan's direct pre-position codecs store D128 but the legacy runner reports
    # a D256 logical baseline. The adaptive representation is already physical.
    if method.startswith(("quantvideo_", "kivi_", "turboquant_")):
        return raw / 2.0
    return raw


def collect_wan(
    root: Path, methods: list[str], dimensions: list[str], prompt_count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = {row["method"]: row for row in read_csv(root / "summary.csv")}
    results: list[dict[str, Any]] = []
    per_prompt: list[dict[str, Any]] = []
    for method in methods:
        if method not in summary:
            raise RuntimeError(f"Wan summary is missing {method}")
        item = summary[method]
        label, bits = METHOD_META[method]
        results.append({
            "model": "Wan2.2 5B", "method": label, "method_id": method,
            "bits": bits,
            "compression_ratio": table_ratio(method, float(item["compression_ratio_vs_bf16"])),
            "psnr": None if method == "bf16" else float(item["postcompression_psnr"]),
            "ssim": None if method == "bf16" else float(item["postcompression_ssim"]),
            "lpips": None if method == "bf16" else float(item["postcompression_lpips"]),
            "vbench10": float(item["vbench10_mean"]),
            "prompts": int(item["prompts"]),
            "e2e_seconds": float(item["e2e_mean_s"]),
            "encode_seconds": float(item["quant_encode_mean_s"]) if item["quant_encode_mean_s"] else None,
            "decode_seconds": float(item["dequant_decode_mean_s"]) if item["dequant_decode_mean_s"] else None,
            "peak_gpu_mib": float(item["peak_gpu_mem_mb"]),
            "kv_payload_mib": float(item["kv_cache_mem_mb"]),
        })
        quality_path = root / "similarity_postcompression" / f"{method}.csv"
        quality = {int(row["prompt_id"]): row for row in read_csv(quality_path)}
        runtime = {
            int(row["prompt_id"]): row
            for row in read_csv(root / "videos" / method / "metrics.csv")
        }
        if set(runtime) != set(range(prompt_count)) or set(quality) != set(range(prompt_count)):
            raise RuntimeError(f"Wan {method}: expected prompt ids 0..{prompt_count - 1}")
        for prompt in range(prompt_count):
            per_prompt.append({
                "model": "Wan2.2 5B", "method_id": method, "prompt_id": prompt,
                "compression_ratio": table_ratio(method, float(runtime[prompt]["compression_ratio"])),
                "psnr": None if method == "bf16" else float(quality[prompt]["psnr"]),
                "ssim": None if method == "bf16" else float(quality[prompt]["ssim"]),
                "lpips": None if method == "bf16" else float(quality[prompt]["lpips"]),
                "e2e_seconds": float(runtime[prompt]["e2e_time_s"]),
                "encode_seconds": float(runtime[prompt]["quant_time_s"]) if runtime[prompt]["quant_time_s"] else None,
                "decode_seconds": float(runtime[prompt]["dequant_time_s"]) if runtime[prompt]["dequant_time_s"] else None,
                "peak_gpu_mib": float(runtime[prompt]["peak_gpu_mem_mb"]),
                "kv_payload_mib": float(runtime[prompt]["kv_cache_mem_mb"]),
            })
    return results, per_prompt


def collect_self(
    root: Path, vbench_root: Path, methods: list[str], dimensions: list[str],
    prompt_count: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    per_prompt: list[dict[str, Any]] = []
    for method in methods:
        ratios: list[float] = []
        quality_values: list[dict[str, float]] = []
        runtime_values: list[dict[str, Any]] = []
        for prompt in range(prompt_count):
            run = root / method / f"prompt_{prompt:04d}_seed_{seed}"
            if not (run / "completed.json").is_file():
                raise FileNotFoundError(f"incomplete Self-Forcing run: {run}")
            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            ratio = float(metrics.get(
                "compression_ratio_vs_bf16_including_shared_codebook",
                metrics.get("compression_ratio_vs_bf16", 1.0),
            ))
            ratios.append(ratio)
            runtime_values.append(metrics)
            quality: dict[str, float] = {}
            if method != "bf16":
                quality = json.loads(
                    (run / "quality_cross_model.json").read_text(encoding="utf-8")
                )
                quality_values.append(quality)
            per_prompt.append({
                "model": "Self-Forcing", "method_id": method, "prompt_id": prompt,
                "compression_ratio": ratio,
                "psnr": quality.get("psnr"), "ssim": quality.get("ssim"),
                "lpips": quality.get("lpips"),
                "e2e_seconds": float(metrics["e2e_seconds"]),
                "encode_seconds": float(metrics["quant_encode_seconds"]),
                "decode_seconds": float(metrics["quant_decode_seconds"]),
                "peak_gpu_mib": float(metrics["peak_nvidia_mib"]),
                "kv_payload_mib": float(metrics["kv_payload_mib"]) + float(metrics.get("shared_codebook_mib", 0.0)),
            })
        label, bits = METHOD_META[method]
        scores = [vbench_score(vbench_root, method, dim) for dim in dimensions]
        results.append({
            "model": "Self-Forcing", "method": label, "method_id": method,
            "bits": bits, "compression_ratio": fmean(ratios),
            "psnr": None if method == "bf16" else fmean(v["psnr"] for v in quality_values),
            "ssim": None if method == "bf16" else fmean(v["ssim"] for v in quality_values),
            "lpips": None if method == "bf16" else fmean(v["lpips"] for v in quality_values),
            "vbench10": fmean(scores), "prompts": prompt_count,
            "e2e_seconds": fmean(float(v["e2e_seconds"]) for v in runtime_values),
            "encode_seconds": fmean(float(v["quant_encode_seconds"]) for v in runtime_values),
            "decode_seconds": fmean(float(v["quant_decode_seconds"]) for v in runtime_values),
            "peak_gpu_mib": max(float(v["peak_nvidia_mib"]) for v in runtime_values),
            "kv_payload_mib": fmean(
                float(v["kv_payload_mib"]) + float(v.get("shared_codebook_mib", 0.0))
                for v in runtime_values
            ),
        })
    return results, per_prompt


def write_csv_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def latex_value(value: float | None, digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{llcccccc}", r"\toprule",
        r"Model & Method & Bits $\downarrow$ & Comp. $\uparrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & LPIPS $\downarrow$ & VBench10 $\uparrow$ \\",
        r"\midrule",
    ]
    for model in ("Wan2.2 5B", "Self-Forcing"):
        selected = [row for row in rows if row["model"] == model]
        for index, row in enumerate(selected):
            model_cell = rf"\multirow{{{len(selected)}}}{{*}}{{{model}}}" if index == 0 else ""
            method = rf"\textbf{{{row['method']}}}" if row["method"] == "Ours" else row["method"]
            lines.append(
                f"{model_cell} & {method} & {row['bits']} & {row['compression_ratio']:.2f}$\\times$ & "
                f"{latex_value(row['psnr'], 2)} & {latex_value(row['ssim'], 3)} & "
                f"{latex_value(row['lpips'], 3)} & {row['vbench10']:.4f} \\\\"
            )
        if model == "Wan2.2 5B":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = load_protocol(args.config.resolve())
    prompt_count = int(config["protocol"]["prompt_count"])
    dimensions = list(config["protocol"]["vbench_dimensions"])
    wan_root = (args.wan_root or Path(config["models"]["wan2_2_5b"]["output_root"])).resolve()
    self_root = (args.self_root or Path(config["models"]["self_forcing"]["output_root"])).resolve()
    self_vbench = (args.self_vbench_root or self_root / "vbench10").resolve()
    output = (args.output_root or Path(config["output"]["aggregate_root"])).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite aggregate output: {output}")
    wan_rows, wan_prompt = collect_wan(
        wan_root, list(config["methods"]["wan2_2_5b"]), dimensions, prompt_count
    )
    self_rows, self_prompt = collect_self(
        self_root, self_vbench, list(config["methods"]["self_forcing"]),
        dimensions, prompt_count, int(config["models"]["self_forcing"]["seed"]),
    )
    rows = wan_rows + self_rows
    output.mkdir(parents=True)
    results_path = output / "main_results.csv"
    per_prompt_path = output / "per_prompt.csv"
    tex_path = output / "main_results.tex"
    write_csv_create_only(results_path, rows)
    write_csv_create_only(per_prompt_path, wan_prompt + self_prompt)
    write_latex(tex_path, rows)
    run_info = {
        "git_commit": git_commit(), "config_path": str(args.config.resolve()),
        "config_hash": sha256_file(args.config.resolve()),
        "codebook_path": {
            model: {
                "k": config["bundle"][model]["k_codebooks"],
                "v": config["bundle"][model]["v_codebooks"],
            }
            for model in ("wan2_2_5b", "self_forcing")
        },
        "codebook_hash": {
            key: value for key, value in config["expected_hashes"].items()
            if "codebooks" in key
        },
        "hostname": socket.gethostname(), "gpu_type": None,
        "seed": {
            "wan2_2_5b": config["models"]["wan2_2_5b"]["seed"],
            "self_forcing": config["models"]["self_forcing"]["seed"],
        },
        "prompt_list": Path(config["protocol"]["prompt_file"]).read_text(encoding="utf-8").splitlines(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "wan_summary.csv": sha256_file(wan_root / "summary.csv"),
            "main_results.csv": sha256_file(results_path),
            "per_prompt.csv": sha256_file(per_prompt_path),
            "main_results.tex": sha256_file(tex_path),
        },
        "metric_accounting": {
            "similarity": "same prompt/seed BF16; post-uncompressed-prefix frames",
            "compression_ratio": "complete physical representation including shared codebooks",
            "vbench10": dimensions,
        },
    }
    with (output / "run_info.json").open("x", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
