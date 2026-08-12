#!/usr/bin/env python3
"""Aggregate the Self-Forcing model-swap replication and write its report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess


REPO = Path(__file__).resolve().parents[2]
METHODS = (
    "bf16", "direct_qv_int2", "direct_qv_int4",
    "adaptive_uniform_b8", "adaptive_uniform_b4", "adaptive_uniform_b3",
    "adaptive_uniform_b2", "adaptive_uniform_b1",
    "adaptive_temporal_avg4", "adaptive_temporal_avg3",
    "adaptive_temporal_avg2", "adaptive_temporal_avg1",
)
LABELS = {
    "bf16": "BF16", "direct_qv_int2": "Direct QuantVideo INT2",
    "direct_qv_int4": "Direct QuantVideo INT4",
    **{f"adaptive_uniform_b{b}": f"Adaptive RVQ + Uniform B{b}" for b in (8, 4, 3, 2, 1)},
    **{f"adaptive_temporal_avg{b}": f"Adaptive RVQ + Time-adaptive avg{b}" for b in (4, 3, 2, 1)},
}
BITS = {"bf16": "—", "direct_qv_int2": "—", "direct_qv_int4": "—",
        **{f"adaptive_uniform_b{b}": str(b) for b in (8, 4, 3, 2, 1)},
        **{f"adaptive_temporal_avg{b}": str(b) for b in (4, 3, 2, 1)}}
VBENCH_DIMS = (
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "human_action", "temporal_style", "overall_consistency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--doc", type=Path, required=True)
    parser.add_argument("--prepare-vbench", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dir(root: Path, method: str, prompt: int) -> Path:
    return root / method / f"prompt_{prompt:04d}_seed_0"


def prepare_vbench(args: argparse.Namespace) -> None:
    for method in METHODS:
        inputs = args.output_root / "vbench/inputs" / method
        inputs.mkdir(parents=True, exist_ok=True)
        prompt_map = {}
        for prompt in range(10):
            directory = run_dir(args.output_root, method, prompt)
            source = (directory / "0-0_ema.mp4").resolve()
            if not source.exists():
                raise FileNotFoundError(source)
            link = inputs / f"prompt_{prompt:04d}.mp4"
            if not link.exists():
                link.symlink_to(source)
            info = json.loads((directory / "run_info.json").read_text(encoding="utf-8"))
            prompt_map[link.name] = info["prompt_list"][0]
        atomic_json(args.output_root / f"vbench/prompts_{method}.json", prompt_map)


def vbench_score(path: Path, dimension: str) -> float:
    files = sorted(path.glob("*_eval_results.json"))
    if not files:
        raise FileNotFoundError(f"missing VBench result in {path}")
    stack = [json.loads(files[-1].read_text(encoding="utf-8"))]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            value = item.get(dimension)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                return float(value[0])
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    raise ValueError(f"cannot find {dimension} in {files[-1]}")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    if args.prepare_vbench:
        prepare_vbench(args)
        return
    args.result_root.mkdir(parents=True, exist_ok=False)
    rows = []
    prefix_rows = []
    for method in METHODS:
        for prompt in range(10):
            directory = run_dir(args.output_root, method, prompt)
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            gate = metrics["uncompressed_prefix_gate"]
            reference = json.loads(
                (run_dir(args.output_root, "bf16", prompt) / "metrics.json").read_text(encoding="utf-8")
            )["uncompressed_prefix_gate"]
            passed = gate == reference
            prefix_rows.append({"method": method, "prompt": prompt, "passed": passed,
                                "latent_hash": gate["latent_bf16_sha256"],
                                "rgb_hash": gate["raw_rgb_u8_sha256"]})
            if not passed:
                raise RuntimeError(f"uncompressed-prefix gate failed: {method} prompt {prompt}")
            quality = metrics.get("quality_vs_bf16") or {
                "mse": 0.0, "psnr": 120.0, "ssim": 1.0, "lpips": 0.0,
            }
            rows.append({
                "method": method, "prompt": prompt, "mse": quality["mse"],
                "psnr": quality["psnr"], "ssim": quality["ssim"], "lpips": quality["lpips"],
                "kv_peak_mib": metrics["kv_peak_payload_mib"],
                "terminal_ratio": metrics["compression_ratio_vs_bf16_including_shared_codebook"],
                "peak_ratio": metrics["bf16_kv_bytes"] / (metrics["kv_peak_payload_mib"] * 1024**2 + metrics["shared_codebook_bytes"]),
                "encode_s": metrics["quant_encode_seconds"], "decode_s": metrics["quant_decode_seconds"],
            })
    summaries = []
    vbench_rows = []
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method]
        dimensions = {}
        for dimension in VBENCH_DIMS:
            score = vbench_score(args.output_root / f"vbench/{method}/{dimension}", dimension)
            dimensions[dimension] = score
            vbench_rows.append({"method": method, "dimension": dimension, "score": score})
        summaries.append({
            "method": method, "label": LABELS[method], "residual_bit": BITS[method],
            **{key: mean([float(r[key]) for r in selected]) for key in (
                "mse", "psnr", "ssim", "lpips", "kv_peak_mib", "terminal_ratio",
                "peak_ratio", "encode_s", "decode_s")},
            "vbench10": mean(list(dimensions.values())),
        })
    for name, table in (("per_prompt.csv", rows), ("prefix_gate.csv", prefix_rows),
                        ("vbench.csv", vbench_rows), ("summary.csv", summaries)):
        with (args.result_root / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    table_lines = [
        "| 方法 | residual bit | PSNR↑ | SSIM↑ | LPIPS↓ | VBench-10↑ | KV peak MiB | 终态压缩率 | 峰值压缩率 | encode s↓ | decode s↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summaries:
        table_lines.append(
            f"| {r['label']} | {r['residual_bit']} | {r['psnr']:.4f} | {r['ssim']:.6f} | "
            f"{r['lpips']:.6f} | {r['vbench10']:.6f} | {r['kv_peak_mib']:.2f} | "
            f"{r['terminal_ratio']:.4f}× | {r['peak_ratio']:.4f}× | "
            f"{r['encode_s']:.3f} | {r['decode_s']:.3f} |"
        )
    text = f"""# Self-Forcing：Adaptive RVQ + 渐进时间自适应压缩模型替换复测

## 结论摘要

本报告将 `docs/wan2_2_5b_progressive_int8_temporal_final_zh.md` 第一张主表的12种方法原样移植到 Self-Forcing Wan2.1-T2V-1.3B。测试为10个 held-out MovieGen prompts、seed 0、30 latent frames，即10次三帧chunk admission；缓存位置统一为pre-RoPE、D128。

Similarity排除压缩发生前的首个chunk所解码的前9帧，在余下108帧上计算。全部120次运行同时通过首3 latent和写MP4前前9帧uint8逐值哈希门槛。压缩率以Self-Forcing BF16 KV为基线，包含Adaptive RVQ、INT4 codebook、scale、routing和residual payload。runtime为实际完整视频生命周期累计值，不是30层线性估计。

{chr(10).join(table_lines)}

## 协议与复现口径

- 模型：Self-Forcing Wan2.1-T2V-1.3B DMD，30层、12 heads、D128。
- RVQ：20 train prompts拟合的Self-Forcing专属无归一化L16C64 codebook；每个chunk出生时按连续两级低于6.29%的规则在L4–L16停止，之后RVQ状态冻结。
- K residual：frame-mean predictor、H128、G128 Gaussian tree、train-only静态layer×head routing。
- V residual：G128 progressive sign-adaptive absmax；B4叶节点不变，B1/B2/B3为物理前缀，B8在B4 cell内细分。
- 时间调度：沿用原主表冻结的train-only 16/8/4/3/2/1率失真profile，终态名义平均分别为4/3/2/1 bit。
- VBench-10维度与原主表完全一致。

## 数据文件

- `summary.csv`：主表聚合。
- `per_prompt.csv`：120次逐prompt质量、memory和runtime。
- `prefix_gate.csv`：首个未压缩chunk硬门槛hash。
- `vbench.csv`：12种方法×10维VBench原始分数。
"""
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    if args.doc.exists():
        raise FileExistsError(args.doc)
    args.doc.write_text(text, encoding="utf-8")
    config = {
        "output_root": str(args.output_root), "methods": METHODS,
        "prompts": list(range(10)), "vbench_dimensions": VBENCH_DIMS,
        "similarity_frames": list(range(9, 117)), "seed": 0,
    }
    atomic_json(args.result_root / "config.json", config)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    atomic_json(args.result_root / "run_info.json", {
        "git_commit": commit, "config_path": str(args.result_root / "config.json"),
        "config_hash": file_hash(args.result_root / "config.json"),
        "codebook_path": str(args.output_root / "calibration_inputs.json"),
        "codebook_hash": file_hash(args.output_root / "calibration_inputs.json"),
        "hostname": socket.gethostname(), "gpu_type": "NVIDIA RTX 6000 Ada Generation",
        "seed": 0, "prompt_list": list(range(10)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
