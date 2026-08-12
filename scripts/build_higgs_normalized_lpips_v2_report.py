#!/usr/bin/env python
"""Build the auditable normalized-LPIPS-v2 report for existing HIGGS videos."""

from __future__ import annotations

import csv
import hashlib
import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    "/data/bowen/t2v_kv_bench_moviegen_20train_10test"
)
SOURCE_ROOT = DATA_ROOT / "higgs_residual_video_benchmark_20260722"
REFERENCE_DIR = DATA_ROOT / "videos" / "bf16"
OUTPUT_DIR = (
    REPO_ROOT / "results" / "analysis"
    / "higgs_normalized_lpips_v2_20260723"
)
EVALUATOR = REPO_ROOT / "scripts" / "eval" / "compare_video_quality.py"

SETTINGS = (
    "l8c64_rvq_residual_higgs_g32r1_int2",
    "l16c64_rvq_residual_higgs_g32r1_int2",
)
RVQ_DEPTH = {
    "l8c64_rvq_residual_higgs_g32r1_int2": 8,
    "l16c64_rvq_residual_higgs_g32r1_int2": 16,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def output_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(OUTPUT_DIR)): sha256_file(path)
        for path in sorted(paths)
    }


def source_files() -> list[Path]:
    files = [
        Path(__file__).resolve(),
        EVALUATOR,
        SOURCE_ROOT / "config.json",
        SOURCE_ROOT / "run_info.json",
        SOURCE_ROOT / "tables" / "metrics.csv",
    ]
    for setting in SETTINGS:
        setting_dir = SOURCE_ROOT / "videos" / setting
        files.append(setting_dir / "metrics.csv")
        files.extend(sorted(setting_dir.glob("*.mp4")))
    files.extend(sorted(REFERENCE_DIR.glob("*.mp4")))
    return sorted(files)


def video_set_hash(paths: list[Path]) -> str:
    lines = [
        f"{sha256_file(path)}  {path}\n"
        for path in sorted(paths)
    ]
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def main() -> None:
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected pre-created output directory: {OUTPUT_DIR}"
        )

    source_run_info = json.loads(
        (SOURCE_ROOT / "run_info.json").read_text(encoding="utf-8")
    )
    with (SOURCE_ROOT / "tables" / "metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        old_rows = {
            row["setting"]: row for row in csv.DictReader(handle)
        }

    combined_rows: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}
    result_json_paths: list[Path] = []
    result_csv_paths: list[Path] = []
    for setting in SETTINGS:
        result_json_path = OUTPUT_DIR / f"{setting}.json"
        result_csv_path = OUTPUT_DIR / f"{setting}.csv"
        result_json_paths.append(result_json_path)
        result_csv_paths.append(result_csv_path)
        result = json.loads(result_json_path.read_text(encoding="utf-8"))
        old = old_rows[setting]

        videos = result["videos"]
        prompt_ids = [int(item["prompt_id"]) for item in videos]
        frames = {int(item["frames"]) for item in videos}
        compared_frames = {
            int(item["compared_frames"]) for item in videos
        }
        psnr_abs_diff = abs(
            float(result["psnr_mean"]) - float(old["psnr"])
        )
        ssim_abs_diff = abs(
            float(result["ssim_mean"]) - float(old["ssim"])
        )
        if result["metric_protocol"]["protocol_id"] != (
            "frame_aligned_psnr_ssim_lpips_normalized_v2"
        ):
            raise AssertionError(f"{setting}: wrong protocol")
        if result["metric_protocol"]["frame_0_included"]:
            raise AssertionError(f"{setting}: frame 0 unexpectedly included")
        if prompt_ids != list(range(10)):
            raise AssertionError(f"{setting}: unexpected prompt IDs")
        if frames != {173} or compared_frames != {172}:
            raise AssertionError(
                f"{setting}: unexpected frame accounting"
            )
        if psnr_abs_diff > 1e-12 or ssim_abs_diff > 1e-12:
            raise AssertionError(
                f"{setting}: PSNR/SSIM changed under LPIPS-only protocol fix"
            )

        mse_mean = sum(float(item["mse"]) for item in videos) / len(videos)
        legacy_lpips = float(old["lpips"])
        normalized_lpips = float(result["lpips_mean"])
        combined_rows.append(
            {
                "setting": setting,
                "rvq_depth": RVQ_DEPTH[setting],
                "residual_codec": "HIGGS-lite G32 R1 INT2",
                "num_videos": int(result["num_videos"]),
                "frames_per_video": 173,
                "compared_frames_per_video": 172,
                "frame_0_included": False,
                "video_mse_mean": mse_mean,
                "psnr_db_mean": float(result["psnr_mean"]),
                "ssim_mean": float(result["ssim_mean"]),
                "lpips_normalized_v2_mean": normalized_lpips,
                "lpips_legacy_unnormalized_v1_mean": legacy_lpips,
                "normalized_vs_legacy_lpips_change_pct": (
                    (normalized_lpips / legacy_lpips - 1.0) * 100.0
                ),
                "psnr_old_abs_diff": psnr_abs_diff,
                "ssim_old_abs_diff": ssim_abs_diff,
                "protocol_id": result["metric_protocol"]["protocol_id"],
            }
        )
        checks[setting] = {
            "prompt_ids_exactly_0_through_9": True,
            "all_source_and_reference_frames": 173,
            "all_compared_frames": 172,
            "frame_0_excluded": True,
            "psnr_old_abs_diff": psnr_abs_diff,
            "ssim_old_abs_diff": ssim_abs_diff,
            "psnr_ssim_match_tolerance": 1e-12,
            "psnr_ssim_match_pass": True,
        }

    combined_path = OUTPUT_DIR / "combined.csv"
    with combined_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(combined_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(combined_rows)

    commands = []
    for gpu, setting in zip((6, 7), SETTINGS):
        commands.append(
            " ".join(
                [
                    f"CUDA_VISIBLE_DEVICES={gpu}",
                    "/home/bowen/anaconda3/envs/hw8/bin/python",
                    "scripts/eval/compare_video_quality.py",
                    f"--reference-dir {REFERENCE_DIR}",
                    f"--test-dir {SOURCE_ROOT / 'videos' / setting}",
                    (
                        "--metrics-csv "
                        f"{SOURCE_ROOT / 'videos' / setting / 'metrics.csv'}"
                    ),
                    f"--setting {setting}",
                    f"--output-json {OUTPUT_DIR / (setting + '.json')}",
                    f"--output-csv {OUTPUT_DIR / (setting + '.csv')}",
                    "--device cuda:0",
                    "--lpips-normalize-input",
                ]
            )
        )

    config = {
        "purpose": (
            "Read-only recomputation of normalized LPIPS-v2 for two already "
            "generated RVQ+HIGGS formal-video settings."
        ),
        "operation": (
            "Similarity evaluation only; no video generation and no codec "
            "encode/decode rerun."
        ),
        "methods": list(SETTINGS),
        "reference_dir": str(REFERENCE_DIR),
        "test_dirs": {
            setting: str(SOURCE_ROOT / "videos" / setting)
            for setting in SETTINGS
        },
        "protocol": {
            "id": "frame_aligned_psnr_ssim_lpips_normalized_v2",
            "seed": 42,
            "videos_per_method": 10,
            "frames_per_video": 173,
            "exclude_frame_0": True,
            "compared_frames_per_video": 172,
            "decoded_rgb_input_range": [0.0, 1.0],
            "lpips_network": "vgg",
            "lpips_normalize_input": True,
            "lpips_effective_input_range": [-1.0, 1.0],
            "dataset_aggregation": (
                "arithmetic mean over ten prompt-level means"
            ),
        },
        "evaluator": str(EVALUATOR),
        "commands": commands,
        "legacy_comparator": str(
            SOURCE_ROOT / "tables" / "metrics.csv"
        ),
    }
    config_path = OUTPUT_DIR / "config.json"
    write_json(config_path, config)

    row_by_depth = {
        int(row["rvq_depth"]): row for row in combined_rows
    }
    l8 = row_by_depth[8]
    l16 = row_by_depth[16]
    readme = f"""# HIGGS formal video：normalized LPIPS-v2 补算

## 结论

本目录只读取已生成的 20 个 HIGGS 视频，并以同 seed 的 BF16
视频为 reference 重算 Similarity；没有重新生成视频，也没有重跑
codec。评测严格使用 `--lpips-normalize-input`，frame 0 排除，因此每个
视频比较 172/173 帧。

| 方法 | 视频 MSE ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS normalized-v2 ↓ |
|---|---:|---:|---:|---:|
| L8C64 RVQ + HIGGS G32-R1 INT2 | {l8['video_mse_mean']:.9f} | {l8['psnr_db_mean']:.6f} | {l8['ssim_mean']:.6f} | {l8['lpips_normalized_v2_mean']:.6f} |
| L16C64 RVQ + HIGGS G32-R1 INT2 | {l16['video_mse_mean']:.9f} | {l16['psnr_db_mean']:.6f} | {l16['ssim_mean']:.6f} | {l16['lpips_normalized_v2_mean']:.6f} |

旧表的 LPIPS 是未归一化 legacy-v1，不能与 normalized-v2 横向混用：

| 方法 | legacy-v1 | normalized-v2 | 数值变化 |
|---|---:|---:|---:|
| L8C64 + HIGGS | {l8['lpips_legacy_unnormalized_v1_mean']:.6f} | {l8['lpips_normalized_v2_mean']:.6f} | {l8['normalized_vs_legacy_lpips_change_pct']:+.3f}% |
| L16C64 + HIGGS | {l16['lpips_legacy_unnormalized_v1_mean']:.6f} | {l16['lpips_normalized_v2_mean']:.6f} | {l16['normalized_vs_legacy_lpips_change_pct']:+.3f}% |

PSNR/SSIM 与旧评测逐浮点一致（绝对差均为 0），说明本次唯一的指标
口径变化是 LPIPS 输入域从错误/legacy 的 `[0,1]` 直接输入，修正为
LPIPS 标准的 `[-1,1]` 有效输入域。

## 统计口径

- 10 个固定 test prompts，seed 42；每个 prompt 一个 173 帧视频。
- reference/test 都按 RGB 解码并逐帧对齐。
- frame 0 排除；每个视频对剩余 172 帧取算术平均。
- dataset mean 是 10 个 prompt-level mean 的算术平均。
- PSNR 是逐帧 dB 的平均，不是由 dataset MSE 反算。
- LPIPS 使用 VGG、model version 0.1、`normalize=True`。
- 这是 Similarity 口径修正；VBench、payload、运行时间均沿用旧 formal
  run，本目录没有重算它们。

## 文件

- `l8c64_rvq_residual_higgs_g32r1_int2.json/.csv`：L8 逐 prompt 结果。
- `l16c64_rvq_residual_higgs_g32r1_int2.json/.csv`：L16 逐 prompt 结果。
- `combined.csv`：两种设置的汇总与 legacy 口径审计。
- `config.json`：命令和冻结协议。
- `run_info.json`、`evidence_manifest.json`：环境、来源、哈希与一致性检查。
"""
    readme_path = OUTPUT_DIR / "README_zh.md"
    readme_path.write_text(readme, encoding="utf-8")

    sources = source_files()
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")
    source_hashes = {
        str(path): sha256_file(path) for path in sources
    }
    source_set_hashes = {
        "bf16_reference_10_videos": video_set_hash(
            sorted(REFERENCE_DIR.glob("*.mp4"))
        ),
        "l8_higgs_10_videos": video_set_hash(
            sorted(
                (
                    SOURCE_ROOT / "videos"
                    / SETTINGS[0]
                ).glob("*.mp4")
            )
        ),
        "l16_higgs_10_videos": video_set_hash(
            sorted(
                (
                    SOURCE_ROOT / "videos"
                    / SETTINGS[1]
                ).glob("*.mp4")
            )
        ),
    }

    ordinary_outputs = (
        result_json_paths
        + result_csv_paths
        + [combined_path, config_path, readme_path]
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    run_info = {
        "git_commit": git_commit,
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "codebook_path": None,
        "codebook_hash": None,
        "codebook_note": (
            "Not applicable to this read-only video metric pass; no codec "
            "or codebook was loaded."
        ),
        "hostname": socket.gethostname(),
        "gpu_type": "NVIDIA RTX 6000 Ada Generation",
        "gpu_indices_physical": [6, 7],
        "seed": 42,
        "prompt_list": source_run_info["prompt_list"]["test"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": "/home/bowen/anaconda3/envs/hw8",
        "operation": config["operation"],
        "metric_protocol": config["protocol"],
        "source_hashes": source_hashes,
        "source_set_hashes": source_set_hashes,
        "output_hashes_excluding_run_info_and_manifest": output_hashes(
            ordinary_outputs
        ),
        "hash_scope_note": (
            "evidence_manifest hashes every output except itself; a manifest "
            "cannot contain its own stable cryptographic hash."
        ),
    }
    run_info_path = OUTPUT_DIR / "run_info.json"
    write_json(run_info_path, run_info)

    evidence = {
        "schema_version": "higgs_normalized_lpips_v2_evidence.v1",
        "protocol_id": config["protocol"]["id"],
        "source_hashes": source_hashes,
        "source_set_hashes": source_set_hashes,
        "output_hashes_excluding_manifest_itself": output_hashes(
            ordinary_outputs + [run_info_path]
        ),
        "checks": checks,
        "manifest_self_hash": None,
        "manifest_self_hash_note": (
            "Intentionally null: embedding a file's own cryptographic hash "
            "would be self-referential."
        ),
    }
    write_json(OUTPUT_DIR / "evidence_manifest.json", evidence)

    print(json.dumps({
        "output_dir": str(OUTPUT_DIR),
        "combined": combined_rows,
        "checks": checks,
    }, indent=2))


if __name__ == "__main__":
    main()
