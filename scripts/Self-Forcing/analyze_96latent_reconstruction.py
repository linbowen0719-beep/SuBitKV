#!/usr/bin/env python3
"""Compare Self-Forcing KV reconstruction error and RVQ train/inference loss."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import statistics
import subprocess

import torch

from subitkv.rvq import rvq_dequantize_tensor_cache, rvq_encode_tensor_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("qvg_int2", "qvg_int4", "rvq_l8c64_int4")
TENSORS = ("K", "V")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--rvq-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "p10": percentile(0.10),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "min": ordered[0],
        "max": ordered[-1],
    }


def aggregate_audit_rows(rows: list[dict]) -> dict:
    elements = sum(int(row["element_count"]) for row in rows)
    vectors = sum(int(row["vector_count"]) for row in rows)
    error_sq = sum(float(row["squared_error_sum"]) for row in rows)
    source_sq = sum(float(row["source_squared_sum"]) for row in rows)
    vector_relative = sum(float(row["vector_relative_squared_error_sum"]) for row in rows)
    relative_squared = error_sq / source_sq
    return {
        "raw_mse": error_sq / elements,
        "relative_squared_error": relative_squared,
        "relative_l2": relative_squared**0.5,
        "mean_vector_relative_squared_error": vector_relative / vectors,
        "element_count": elements,
        "vector_count": vectors,
    }


def load_inference_audits(root: Path) -> tuple[list[dict], list[dict], list[dict]]:
    per_prompt = []
    per_layer = []
    aggregate = []
    for method in METHODS:
        method_rows = {tensor: [] for tensor in TENSORS}
        for prompt in range(10):
            path = root / method / f"prompt_{prompt:04d}_seed_0" / "metrics.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            audit = payload["kv_reconstruction_audit"]
            if len(audit["rows"]) != 1920:
                raise ValueError(f"Expected 1920 audit rows in {path}")
            for tensor in TENSORS:
                rows = [row for row in audit["rows"] if row["tensor"] == tensor]
                metrics = aggregate_audit_rows(rows)
                per_prompt.append({"method": method, "prompt": prompt, "tensor": tensor, **metrics})
                method_rows[tensor].extend(rows)
                for layer in range(30):
                    layer_rows = [row for row in rows if int(row["layer"]) == layer]
                    per_layer.append({
                        "method": method,
                        "prompt": prompt,
                        "tensor": tensor,
                        "layer": layer,
                        **aggregate_audit_rows(layer_rows),
                    })
        for tensor in TENSORS:
            aggregate.append({
                "method": method,
                "split": "actual_compressed_ar_inference",
                "tensor": tensor,
                **aggregate_audit_rows(method_rows[tensor]),
            })
    return per_prompt, per_layer, aggregate


def load_video_quality(root: Path) -> list[dict]:
    rows = []
    for method in METHODS:
        prompt_rows = []
        for prompt in range(10):
            path = root / method / f"prompt_{prompt:04d}_seed_0" / "metrics.json"
            metrics = json.loads(path.read_text(encoding="utf-8"))["quality_vs_bf16"]
            prompt_rows.append(metrics)
        rows.append({
            "method": method,
            "psnr_mean": statistics.fmean(row["psnr"] for row in prompt_rows),
            "ssim_mean": statistics.fmean(row["ssim"] for row in prompt_rows),
            "lpips_mean": statistics.fmean(row["lpips"] for row in prompt_rows),
        })
    return rows


def codebook_training_objective(rvq_root: Path) -> tuple[list[dict], list[dict]]:
    detailed = []
    aggregate = []
    for tensor in TENSORS:
        values = []
        directory = rvq_root / f"{tensor.lower()}_codebooks"
        for layer in range(30):
            for head in range(12):
                payload = torch.load(
                    directory / f"layer_{layer}_head_{head}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                stage_losses = payload["config"]["residual_mse"]
                if len(stage_losses) != 8:
                    raise ValueError(f"Expected eight stages for {tensor} layer={layer} head={head}")
                final_loss = float(stage_losses[-1])
                values.append(final_loss)
                detailed.append({
                    "tensor": tensor,
                    "layer": layer,
                    "head": head,
                    "stage8_normalized_mse": final_loss,
                })
        aggregate.append({
            "method": "rvq_l8c64_fp32_training_objective",
            "split": "training_codebook_fit_50000_vectors_per_layer_head",
            "tensor": tensor,
            **{f"loss_{key}": value for key, value in summarize(values).items()},
            "layer_head_count": len(values),
        })
    return detailed, aggregate


@torch.no_grad()
def replay_rvq_training_dump(rvq_root: Path, device: torch.device) -> tuple[list[dict], list[dict]]:
    detailed = []
    all_rows = {tensor: [] for tensor in TENSORS}
    for prompt in range(10):
        for layer in range(30):
            dump = torch.load(
                rvq_root / "kv_dump" / "train" / str(prompt) / f"layer_{layer}.pt",
                map_location="cpu",
                weights_only=False,
            )
            for tensor in TENSORS:
                reference = dump[tensor].to(device)
                packed = rvq_encode_tensor_cache(
                    reference,
                    layer_idx=layer,
                    codebook_dir=rvq_root / f"{tensor.lower()}_codebooks",
                    batch_size=2048,
                    compute_dtype="bf16",
                    codebook_storage_dtype="int4",
                    scale_storage_dtype="uint4",
                )
                reconstructed = rvq_dequantize_tensor_cache(packed, reference.dtype)
                reference_float = reference.float()
                error_sq = (reference_float - reconstructed.float()).square()
                source_sq = reference_float.square()
                vector_error = error_sq.sum(dim=-1)
                vector_source = source_sq.sum(dim=-1).clamp_min(1e-20)
                row = {
                    "prompt": prompt,
                    "layer": layer,
                    "tensor": tensor,
                    "element_count": int(reference.numel()),
                    "vector_count": int(reference.numel() // reference.shape[-1]),
                    "squared_error_sum": float(error_sq.sum().item()),
                    "source_squared_sum": float(source_sq.sum().item()),
                    "vector_relative_squared_error_sum": float((vector_error / vector_source).sum().item()),
                }
                detailed.append({
                    "prompt": prompt,
                    "layer": layer,
                    "tensor": tensor,
                    **aggregate_audit_rows([row]),
                })
                all_rows[tensor].append(row)
                del reference, packed, reconstructed, reference_float, error_sq, source_sq
                del vector_error, vector_source
            del dump
    aggregate = [{
        "method": "rvq_l8c64_int4",
        "split": "training_dump_deployed_codec",
        "tensor": tensor,
        **aggregate_audit_rows(all_rows[tensor]),
    } for tensor in TENSORS]
    return detailed, aggregate


def report_markdown(
    audit_aggregate: list[dict],
    train_objective: list[dict],
    train_replay: list[dict],
    quality: list[dict],
) -> str:
    lookup = {(row["method"], row["tensor"]): row for row in audit_aggregate}
    train_fit = {row["tensor"]: row for row in train_objective}
    train_codec = {row["tensor"]: row for row in train_replay}
    quality_lookup = {row["method"]: row for row in quality}
    lines = [
        "# Self-Forcing 96-latent KV压缩误差",
        "",
        "## 实际AR推理重建误差",
        "",
        "以下误差来自10个MovieGen test prompts的真实压缩轨迹。每个视频32个chunk，每chunk 3 latent frames；K/V分开统计。",
        "",
        "| 方法 | Tensor | Raw MSE | Relative squared error | Relative L2 | Mean vector normalized MSE | PSNR | SSIM | LPIPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        q = quality_lookup[method]
        for tensor in TENSORS:
            row = lookup[(method, tensor)]
            lines.append(
                f"| {method} | {tensor} | {row['raw_mse']:.6g} | "
                f"{row['relative_squared_error']:.6g} | {row['relative_l2']:.6g} | "
                f"{row['mean_vector_relative_squared_error']:.6g} | "
                f"{q['psnr_mean']:.4f} | {q['ssim_mean']:.4f} | {q['lpips_mean']:.4f} |"
            )
    lines.extend([
        "",
        "## RVQ训练集与实际推理",
        "",
        "训练objective是FP32 codebook完成第8个stage后的per-vector RMS-normalized residual MSE。部署态训练loss使用相同训练dump，但经过实际INT4 codebook value与UINT4 scale codec。",
        "",
        "| Tensor | FP32训练objective | INT4训练dump loss | INT4实际推理loss | 推理/训练dump |",
        "|---|---:|---:|---:|---:|",
    ])
    for tensor in TENSORS:
        inference = lookup[("rvq_l8c64_int4", tensor)]["mean_vector_relative_squared_error"]
        deployed_train = train_codec[tensor]["mean_vector_relative_squared_error"]
        lines.append(
            f"| {tensor} | {train_fit[tensor]['loss_mean']:.6g} | {deployed_train:.6g} | "
            f"{inference:.6g} | {inference / deployed_train:.3f}x |"
        )
    qvg_int2_k = lookup[("qvg_int2", "K")]["mean_vector_relative_squared_error"]
    qvg_int2_v = lookup[("qvg_int2", "V")]["mean_vector_relative_squared_error"]
    qvg_int4_k = lookup[("qvg_int4", "K")]["mean_vector_relative_squared_error"]
    qvg_int4_v = lookup[("qvg_int4", "V")]["mean_vector_relative_squared_error"]
    rvq_k = lookup[("rvq_l8c64_int4", "K")]["mean_vector_relative_squared_error"]
    rvq_v = lookup[("rvq_l8c64_int4", "V")]["mean_vector_relative_squared_error"]
    lines.extend([
        "",
        "## 结论",
        "",
        f"- QVG INT4相对QVG INT2将K/V normalized MSE分别降低{qvg_int2_k / qvg_int4_k:.1f}x和{qvg_int2_v / qvg_int4_v:.1f}x。",
        f"- RVQ L8C64 INT4的K/V normalized MSE分别是QVG INT4的{rvq_k / qvg_int4_k:.1f}x和{rvq_v / qvg_int4_v:.1f}x，也是QVG INT2的{rvq_k / qvg_int2_k:.2f}x和{rvq_v / qvg_int2_v:.2f}x。",
        f"- INT4部署存储使RVQ训练集K/V loss相对FP32训练objective增加{(train_codec['K']['mean_vector_relative_squared_error'] / train_fit['K']['loss_mean'] - 1) * 100:.1f}%和{(train_codec['V']['mean_vector_relative_squared_error'] / train_fit['V']['loss_mean'] - 1) * 100:.1f}%。",
        f"- 从部署态训练dump到实际AR推理，K/V loss再增加{(rvq_k / train_codec['K']['mean_vector_relative_squared_error'] - 1) * 100:.1f}%和{(rvq_v / train_codec['V']['mean_vector_relative_squared_error'] - 1) * 100:.1f}%，存在明确的train-to-inference gap。",
        "",
        "## 口径说明",
        "",
        "- Raw MSE按所有元素加权，relative squared error为全局误差能量/输入能量。",
        "- Mean vector normalized MSE先对每个128维KV向量计算误差能量/输入能量，再对向量平均；它与RVQ的per-vector RMS训练objective同口径。",
        "- 三种方法的实际推理误差来自各自AR轨迹，因此同时包含压缩器误差和压缩导致的后续轨迹漂移；不是在同一静态tensor上的离线比较。",
        "- 视频PSNR/SSIM/LPIPS沿用原10条正式结果，没有改变帧聚合方式。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_prompt, per_layer, audit_aggregate = load_inference_audits(args.inference_root)
    quality = load_video_quality(args.formal_root)
    train_detail, train_objective = codebook_training_objective(args.rvq_root)
    train_replay_detail, train_replay = replay_rvq_training_dump(args.rvq_root, torch.device(args.device))

    write_csv(args.output_dir / "inference_error_per_prompt.csv", per_prompt)
    write_csv(args.output_dir / "inference_error_per_layer_prompt.csv", per_layer)
    write_csv(args.output_dir / "inference_error_aggregate.csv", audit_aggregate)
    write_csv(args.output_dir / "video_quality_aggregate.csv", quality)
    write_csv(args.output_dir / "rvq_training_objective_per_layer_head.csv", train_detail)
    write_csv(args.output_dir / "rvq_training_replay_per_layer_prompt.csv", train_replay_detail)
    inference_rvq = {
        row["tensor"]: row for row in audit_aggregate
        if row["method"] == "rvq_l8c64_int4"
    }
    objective_rvq = {row["tensor"]: row for row in train_objective}
    replay_rvq = {row["tensor"]: row for row in train_replay}
    comparison_rows = []
    for tensor in TENSORS:
        comparison_rows.extend([
            {
                "source": "fp32_codebook_training_objective",
                "tensor": tensor,
                "mean_vector_normalized_mse": objective_rvq[tensor]["loss_mean"],
                "raw_mse": "",
                "global_relative_squared_error": "",
            },
            {
                "source": "training_dump_deployed_int4_codec",
                "tensor": tensor,
                "mean_vector_normalized_mse": replay_rvq[tensor]["mean_vector_relative_squared_error"],
                "raw_mse": replay_rvq[tensor]["raw_mse"],
                "global_relative_squared_error": replay_rvq[tensor]["relative_squared_error"],
            },
            {
                "source": "actual_compressed_ar_inference_int4_codec",
                "tensor": tensor,
                "mean_vector_normalized_mse": inference_rvq[tensor]["mean_vector_relative_squared_error"],
                "raw_mse": inference_rvq[tensor]["raw_mse"],
                "global_relative_squared_error": inference_rvq[tensor]["relative_squared_error"],
            },
        ])
    write_csv(args.output_dir / "rvq_training_vs_inference.csv", comparison_rows)
    (args.output_dir / "report.md").write_text(
        report_markdown(audit_aggregate, train_objective, train_replay, quality),
        encoding="utf-8",
    )
    config = {
        "inference_root": str(args.inference_root.resolve()),
        "formal_root": str(args.formal_root.resolve()),
        "rvq_root": str(args.rvq_root.resolve()),
        "methods": METHODS,
        "prompts": 10,
        "latent_frames": 96,
        "chunks": 32,
        "latent_frames_per_chunk": 3,
        "layers": 30,
        "heads": 12,
    }
    atomic_json(args.output_dir / "analysis_config.json", config)
    config_hash = sha256_file(args.output_dir / "analysis_config.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    gpu_type = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()[0].strip()
    atomic_json(args.output_dir / "run_info.json", {
        "git_commit": commit,
        "config_path": str((args.output_dir / "analysis_config.json").resolve()),
        "config_hash": config_hash,
        "codebook_path": str(args.rvq_root.resolve()),
        "codebook_hash": {
            tensor: sha256_file(args.rvq_root / f"{tensor.lower()}_codebooks_manifest.json")
            for tensor in TENSORS
        },
        "hostname": socket.gethostname(),
        "gpu_type": gpu_type,
        "seed": 0,
        "prompt_list": [
            (args.formal_root / "bf16" / f"prompt_{prompt:04d}_seed_0" / "prompt.txt")
            .read_text(encoding="utf-8").strip()
            for prompt in range(10)
        ],
        "prompt_source": "/data/bowen/t2v_kv_bench_moviegen_20train_10test/prompts_test.txt",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_hash": sha256_file(Path(__file__)),
        "inference_instrumentation": str(
            (REPO_ROOT / "experiments/Self-Forcing/pipeline/causal_inference.py").resolve()
        ),
        "inference_instrumentation_hash": sha256_file(
            REPO_ROOT / "experiments/Self-Forcing/pipeline/causal_inference.py"
        ),
    })


if __name__ == "__main__":
    main()
